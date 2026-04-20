"""
GRPO Plugin: FinAgent 工具调用环境 + Reward 函数

用于 TRL GRPOTrainer 的 environment_factory 和 reward_funcs。

V4.1 Reward（基于 V3 训练失败的根因分析 + Review 修正）：
  - 从"LLM-as-Judge 为主"转向"多维度细粒度规则 reward 为主 + 可选轻量 LLM 补充"
  - 4 个规则维度：tool_coverage + query_quality + calc_behavior + strategy_match
  - 可选第 5 维度：LLM 二元判断（合理/不合理）作为 anti-hacking 安全网
  - 继承 V3 的硬约束：格式检查 + 重复 query 惩罚 + DAPO overlong penalty

设计文档：docs/FinAgent_Reward_V4.1_Design.md
知识库：reward_knowledge_base.py

依赖：
  - hybrid_search.py（FinAgentRetriever）
  - tools.py（FinAgentTools, TOOLS_NATIVE）
  - reward_knowledge_base.py（指标映射、工具需求配置等）
  - qwen3-max API（维度 5 LLM 二元判断，可选）
"""

import re
import os
import logging
import threading
from typing import Optional, List, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ============ 配置 ============

# LLM 二元判断配置（维度 5，可选）
USE_LLM = os.environ.get("REWARD_USE_LLM", "0") == "1"
LLM_MODEL = os.environ.get("REWARD_LLM_MODEL", "qwen3-max")
LLM_API_KEY = os.environ.get("REWARD_LLM_API_KEY",
                             os.environ.get("COMPLETENESS_API_KEY", ""))
LLM_BASE_URL = os.environ.get(
    "REWARD_LLM_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
LLM_MAX_WORKERS = int(os.environ.get("REWARD_LLM_MAX_WORKERS", "8"))

# DAPO overlong penalty 参数
OVERLONG_L_MAX = 8192
OVERLONG_L_CLIP = 12288
OVERLONG_C = 0.5

# 格式不合规惩罚
FORMAT_INVALID_PENALTY = -1.0

# 重复 query 惩罚阈值（V4.1 从 V3 的 0.7 放松到 0.85）
REPEAT_QUERY_THRESHOLD = 0.85

# Reward 权重（根据 USE_LLM 切换）
# V4.1.2: calc_behavior 从 0.20 降到 0.10（95% 问题零 variance，留给 SFT 阶段优化）
#          腾出的权重分给有 variance 的维度
if USE_LLM:
    W_TOOL_COVERAGE = 0.32
    W_QUERY_QUALITY = 0.23
    W_CALC_BEHAVIOR = 0.10
    W_STRATEGY_MATCH = 0.15
    W_LLM_JUDGE = 0.20
else:
    W_TOOL_COVERAGE = 0.37
    W_QUERY_QUALITY = 0.28
    W_CALC_BEHAVIOR = 0.10
    W_STRATEGY_MATCH = 0.25
    W_LLM_JUDGE = 0.0

VALID_TOOLS = {"search_financial", "search_report", "search_industry", "calculate"}

# LLM client（延迟初始化）
_llm_client = None
_llm_client_lock = threading.Lock()


# ============ FinAgentEnv: TRL environment_factory ============

class FinAgentEnv:
    """
    ReAct Agent 的工具调用环境。

    TRL GRPOTrainer 的 environment_factory 会为每条轨迹创建一个独立实例。
    环境类的所有 public 方法（非 _ 开头、非 reset）自动被发现为工具。
    """

    _shared_retriever = None
    _shared_tools = None
    _init_lock = threading.Lock()

    @classmethod
    def _init_shared_resources(cls):
        """延迟初始化共享的检索器和工具实例（线程安全）"""
        if cls._shared_retriever is None:
            with cls._init_lock:
                if cls._shared_retriever is None:  # double-check
                    from hybrid_search import FinAgentRetriever
                    from tools import FinAgentTools
                    cls._shared_retriever = FinAgentRetriever()
                    cls._shared_tools = FinAgentTools(cls._shared_retriever)
                    logger.info("FinAgentEnv: 共享检索器已初始化")

    def __init__(self, **kwargs):
        """接受 **kwargs 兼容 TRL 可能传入的额外参数"""
        self._init_shared_resources()
        self.tool_steps = []       # [{tool, query}, ...]（不记录 observation）
        self.has_calculate = False
        self.calc_results = []     # calculate 返回的结果文本

    def reset(self, **kwargs) -> Optional[str]:
        """每个 episode 开始时调用"""
        self.tool_steps = []
        self.has_calculate = False
        self.calc_results = []
        return None

    def search_financial(self, query: str) -> str:
        """
        搜索公司财务数据（ROE、净利率、资产负债率等）。

        Args:
            query: 检索关键词，如"贵州茅台 ROE 2024"

        Returns:
            检索结果文本
        """
        self.tool_steps.append({"tool": "search_financial", "query": query})
        try:
            result, retrieved = self._shared_tools.call("search_financial", {"query": query})
        except Exception as e:
            logger.warning(f"search_financial 调用失败: {e}")
            return f"搜索失败：{str(e)}"
        # 局部变量,无共享状态 race(8 并发 rollout 不会互相覆盖)
        self.tool_steps[-1]['retrieved'] = retrieved
        return result

    def search_report(self, query: str) -> str:
        """
        搜索券商研报（评级、目标价、EPS预测、深度分析）。

        Args:
            query: 检索关键词，如"贵州茅台 最新研报 2025"

        Returns:
            检索结果文本
        """
        self.tool_steps.append({"tool": "search_report", "query": query})
        try:
            result, retrieved = self._shared_tools.call("search_report", {"query": query})
        except Exception as e:
            logger.warning(f"search_report 调用失败: {e}")
            return f"搜索失败：{str(e)}"
        self.tool_steps[-1]['retrieved'] = retrieved
        return result

    def search_industry(self, query: str) -> str:
        """
        搜索行业对比数据（同行业公司指标排名和均值）。

        Args:
            query: 检索关键词，如"白酒行业 ROE 盈利能力"

        Returns:
            检索结果文本
        """
        self.tool_steps.append({"tool": "search_industry", "query": query})
        try:
            result, retrieved = self._shared_tools.call("search_industry", {"query": query})
        except Exception as e:
            logger.warning(f"search_industry 调用失败: {e}")
            return f"搜索失败：{str(e)}"
        self.tool_steps[-1]['retrieved'] = retrieved
        return result

    def calculate(self, expression: str) -> str:
        """
        计算数学表达式。所有数值计算（同比增长率、杜邦分析等）必须使用此工具。

        Args:
            expression: 纯数学表达式，如"(1505 - 1294) / 1294 * 100"

        Returns:
            计算结果
        """
        self.tool_steps.append({"tool": "calculate", "query": expression})
        self.has_calculate = True
        try:
            result, _ = self._shared_tools.call("calculate", {"expression": expression})
            self.calc_results.append(result)
            return result
        except Exception as e:
            logger.warning(f"calculate 调用失败: {e}")
            error_result = f"计算失败：{str(e)}"
            self.calc_results.append(error_result)
            return error_result


# ============ V4.1 统一 Reward 函数 ============

def finagent_reward(completions, environments=None, **kwargs) -> list:
    """
    FinAgent GRPO V4.1 统一 reward 函数。

    4 个规则维度 + 可选 LLM 二元判断：
      0.30/0.35 × tool_coverage    — 工具覆盖完整性
      0.20/0.25 × query_quality    — query 质量
      0.15/0.20 × calc_behavior    — 计算行为合理性
      0.15/0.20 × strategy_match   — 搜索策略匹配度
      0.20/0.00 × llm_judge        — LLM 二元判断（可选）

    + 格式检查（前置 -1.0）+ 重复 query 惩罚 + DAPO overlong penalty

    Args:
        completions: list，每条轨迹的完整生成（可能是 str 或 list[dict]）
        environments: list[FinAgentEnv]
        **kwargs: 包含 question, type 等数据集字段
    """
    from reward_knowledge_base import (
        TOOL_REQUIREMENTS, check_needs_calc, extract_dimensions,
        is_comprehensive, compute_tool_coverage, compute_single_query_score,
        detect_mental_calc, count_unique_companies,
        extract_metrics_from_queries, get_dimensions_from_metrics,
        compute_metric_dimension_match, apply_anti_hacking_penalty,
        is_reject_response,
    )

    if environments is None:
        environments = kwargs.get("environments", kwargs.get("envs", []))

    # TRL environment_factory 模式下 completions 可能是 list[list[dict]]
    completions = [_completion_to_str(c) for c in completions]

    rewards = []
    llm_tasks = {}  # idx -> (question, question_type, tool_steps)

    for idx, (env, completion) in enumerate(zip(environments, completions)):
        question, question_type = _extract_question_and_type(idx, kwargs)

        # ---- Step 0: 前置硬约束 ----
        if not _is_format_valid(env, completion, question_type):
            rewards.append(FORMAT_INVALID_PENALTY)
            continue

        # ---- Step 0.5: reject 类型特殊处理 ----
        if question_type == "reject":
            answer = _extract_final_answer(completion)
            reward = _compute_reject_reward(env, answer)
            rewards.append(reward)
            continue

        # 占位，后面填真实值
        rewards.append(None)

        # 收集 LLM 任务（如果启用）
        if USE_LLM:
            llm_tasks[idx] = (question, question_type, env.tool_steps)

    # ---- LLM 二元判断（并行，仅当 USE_LLM=1 时）----
    llm_scores = {}
    if USE_LLM and llm_tasks:
        llm_scores = _batch_llm_binary_judge(llm_tasks)

    # ---- 计算每条轨迹的 V4.1 reward ----
    for idx, (env, completion) in enumerate(zip(environments, completions)):
        if rewards[idx] is not None:
            continue  # 已经处理过（格式不合规 或 reject）

        question, question_type = _extract_question_and_type(idx, kwargs)
        answer = _extract_final_answer(completion)
        queries = [s["query"] for s in env.tool_steps]
        needs_calc = check_needs_calc(question, answer, question_type)
        dimensions = extract_dimensions(question)
        type_config = TOOL_REQUIREMENTS.get(question_type,
                                            TOOL_REQUIREMENTS.get("single_company_medium"))

        # ---- 维度 1: tool_coverage (0.30/0.35) ----
        tool_cov = compute_tool_coverage(type_config, env.tool_steps, queries, needs_calc)

        # ---- 维度 2: query_quality (0.20/0.25) ----
        query_qual = _compute_query_quality(env.tool_steps, dimensions)

        # ---- 维度 3: calc_behavior (0.15/0.20) ----
        calc_beh = _compute_calc_behavior(env, answer, needs_calc)

        # ---- 维度 4: strategy_match (0.15/0.20) ----
        strat = _compute_strategy_match(question_type, question, env.tool_steps, queries)

        # ---- 加权组合 ----
        base_reward = (
            W_TOOL_COVERAGE * tool_cov +
            W_QUERY_QUALITY * query_qual +
            W_CALC_BEHAVIOR * calc_beh +
            W_STRATEGY_MATCH * strat
        )

        # LLM 二元判断（可选）
        if USE_LLM:
            llm_score = llm_scores.get(idx, 0.5)
            base_reward += W_LLM_JUDGE * llm_score

        # ---- 硬约束扣分 ----
        penalty = _call_quality_penalty(env)
        base_reward = max(base_reward + penalty, 0.0)

        # ---- DAPO overlong penalty ----
        length = _estimate_token_count(completion)
        base_reward = _apply_overlong_penalty(base_reward, length)

        rewards[idx] = base_reward

        # 收集各维度分数用于 logging
        with _metrics_lock:
            _metrics["tool_coverage_scores"].append(tool_cov)
            _metrics["query_quality_scores"].append(query_qual)
            _metrics["calc_behavior_scores"].append(calc_beh)
            _metrics["strategy_match_scores"].append(strat)
            if USE_LLM:
                _metrics["llm_judge_scores"].append(llm_scores.get(idx, 0.5))

    # 自定义指标 logging
    _log_custom_metrics(environments, completions, rewards, kwargs)

    return rewards


# ============ 维度 2: query_quality ============

def _compute_query_quality(tool_steps: list, target_dims: Set[str]) -> float:
    """
    计算轨迹整体的 query 质量得分。

    对每条 non-calculate query 独立打分，取平均。
    包含 anti-hacking 指标-维度匹配检查。
    """
    from reward_knowledge_base import (
        compute_single_query_score,
        compute_metric_dimension_match,
        apply_anti_hacking_penalty,
    )

    scores = []
    for step in tool_steps:
        tool_name = step.get("tool", "")
        query = step.get("query", "")

        if tool_name == "calculate":
            continue  # calculate 的 query 由 calc_behavior 维度评

        # 基础质量评分
        score = compute_single_query_score(query, tool_name)

        # anti-hacking 指标匹配检查
        if target_dims:
            match_rate = compute_metric_dimension_match(query, target_dims)
            score = apply_anti_hacking_penalty(score, match_rate)

        scores.append(score)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# ============ 维度 3: calc_behavior ============

def _compute_calc_behavior(env, answer: str, needs_calc: bool) -> float:
    """
    V4.1 计算行为合理性评分（6 种情况）。

    needs_calc 由双路径判定：question 关键词 OR answer 心算检测。
    """
    used_calc = env.has_calculate
    calc_queries = [s["query"] for s in env.tool_steps if s["tool"] == "calculate"]
    calc_results = getattr(env, "calc_results", []) or []
    successful_results = [r for r in calc_results if not _is_calc_error_result(r)]

    if needs_calc and used_calc:
        # 需要且用了：只有真正拿到有效结果才给正向 reward
        if not successful_results:
            return 0.0

        # 需要且用了 → 0.50-1.00（看表达式质量）
        base = 0.50
        for calc_query in calc_queries:
            # 表达式包含数字和运算符
            if re.search(r'\d+\.?\d*\s*[-+*/×÷]\s*\d+', calc_query):
                base += 0.25
                break
        # 有效计算结果被答案引用，说明模型真的用到了 calculate
        if _answer_reuses_calc_result(answer, successful_results):
            base += 0.25
        return min(base, 1.0)

    elif needs_calc and not used_calc:
        # 需要但没用 → 0.00
        return 0.0

    elif not needs_calc and used_calc:
        # 不需要但用了：失败调用不应给正分；成功调用但答案未用到则轻微扣分
        if not successful_results:
            return 0.0
        has_source = _answer_reuses_calc_result(answer, successful_results)
        return 0.50 if has_source else 0.45

    else:
        # 不需要且没用 → 0.50（中性值）
        return 0.50


def _is_calc_error_result(result: str) -> bool:
    """判断 calculate 输出是否为错误结果。"""
    text = str(result or "").strip()
    if not text:
        return True
    return text.startswith("[计算错误]") or text.startswith("计算失败")


def _answer_reuses_calc_result(answer: str, calc_results: List[str]) -> bool:
    """检查答案是否引用了有效 calculate 结果中的数字。"""
    if not answer:
        return False
    for cr in calc_results:
        numbers = re.findall(r'[\d.]+', cr)
        for num in numbers:
            if len(num) >= 3 and re.search(r'(?<!\d)' + re.escape(num) + r'(?!\d)', answer):
                return True
    return False


# ============ 维度 4: strategy_match ============

def _compute_strategy_match(question_type: str, question: str,
                            tool_steps: list, queries: List[str]) -> float:
    """
    搜索策略匹配度评分（按 question_type 分别处理）。
    """
    from reward_knowledge_base import (
        extract_dimensions, is_comprehensive,
        extract_metrics_from_queries, get_dimensions_from_metrics,
        count_unique_companies, query_has_industry_name,
        query_has_company_name, extract_company_name,
    )

    tools_used = [s["tool"] for s in tool_steps]

    if question_type == "company_comparison":
        return _strategy_comparison(tool_steps, queries)
    elif question_type == "risk_analysis":
        return _strategy_risk(queries, question)
    elif question_type == "single_company_medium":
        return _strategy_medium(question, queries)
    elif question_type == "industry_analysis":
        return _strategy_industry(tools_used)
    elif question_type == "financial_query":
        return _strategy_financial_query(queries)
    elif question_type == "single_company_simple":
        return _strategy_simple(queries)
    elif question_type == "reject":
        return 1.0  # reject 已在上层处理
    else:
        return 0.5  # 未知类型给中间值


def _strategy_comparison(tool_steps: list, queries: List[str]) -> float:
    """company_comparison：两家公司搜索的指标是否一致"""
    from reward_knowledge_base import (
        count_unique_companies, extract_metrics_from_queries,
        extract_company_name,
    )

    companies = count_unique_companies(tool_steps)
    if len(companies) < 2:
        return 0.1  # 只搜了一家

    # 按公司分组 query 中的指标
    company_list = list(companies)
    metrics_per_company = {c: set() for c in company_list}

    for step in tool_steps:
        if step["tool"] in ("search_financial", "search_report"):
            query = step["query"]
            company = extract_company_name(query)
            if company and company in metrics_per_company:
                query_metrics = extract_metrics_from_queries([query])
                metrics_per_company[company].update(query_metrics)

    # 计算指标重叠度
    metric_sets = [v for v in metrics_per_company.values() if v]
    if len(metric_sets) < 2:
        return 0.3

    s_a, s_b = metric_sets[0], metric_sets[1]
    max_size = max(len(s_a), len(s_b))
    if max_size == 0:
        return 0.3

    overlap = len(s_a & s_b) / max_size

    if overlap >= 0.8:
        return 1.0
    elif overlap >= 0.5:
        return 0.5 + overlap * 0.5
    else:
        return 0.3


def _strategy_risk(queries: List[str], question: str = "") -> float:
    """
    risk_analysis：搜索维度是否匹配问题需求。

    V4.1.2 修正：不再一刀切要求双维度。
    - 如果问题明确只问偿债或只问盈利 → 覆盖对应维度即满分
    - 如果问题是通用风险（未指定维度）→ 仍要求双维度
    - 保留"至少覆盖一个维度"的底线
    """
    from reward_knowledge_base import (
        extract_metrics_from_queries, get_dimensions_from_metrics,
        extract_dimensions,
    )
    all_metrics = extract_metrics_from_queries(queries)
    covered_dims = get_dimensions_from_metrics(all_metrics)

    has_leverage = "偿债/杠杆" in covered_dims
    has_profit = "盈利能力" in covered_dims

    # 从问题中提取目标维度
    target_dims = extract_dimensions(question) if question else set()

    if target_dims:
        # 问题指定了维度 → 按覆盖比例打分
        if target_dims.issubset(covered_dims):
            return 1.0
        elif target_dims & covered_dims:
            return 0.6
        else:
            return 0.1
    else:
        # 通用风险问题（未指定维度）→ 默认要求偿债+盈利双维度
        if has_leverage and has_profit:
            return 1.0
        elif has_leverage or has_profit:
            return 0.5
        else:
            return 0.0


def _strategy_medium(question: str, queries: List[str]) -> float:
    """single_company_medium：定向 vs 全面两套逻辑"""
    from reward_knowledge_base import (
        extract_dimensions, is_comprehensive,
        extract_metrics_from_queries, get_dimensions_from_metrics,
    )

    target_dims = extract_dimensions(question)
    all_metrics = extract_metrics_from_queries(queries)
    covered_dims = get_dimensions_from_metrics(all_metrics)

    if is_comprehensive(question):
        # 全面问题：按维度数计分
        n = len(covered_dims)
        if n >= 3:
            return 1.0
        elif n == 2:
            return 0.7
        elif n == 1:
            return 0.4
        else:
            return 0.1
    else:
        # 定向问题：检查是否覆盖了目标维度
        if not target_dims:
            return 0.5  # 未识别到目标维度，给中间值
        if target_dims.issubset(covered_dims):
            return 1.0
        elif target_dims & covered_dims:
            return 0.6
        else:
            return 0.2


def _strategy_industry(tools_used: list) -> float:
    """industry_analysis：是否从行业和个股两个层面搜索"""
    has_industry = "search_industry" in tools_used
    has_financial = "search_financial" in tools_used

    if has_industry and has_financial:
        return 1.0
    elif has_industry:
        return 0.6
    elif has_financial:
        return 0.3
    else:
        return 0.1


def _strategy_financial_query(queries: List[str]) -> float:
    """financial_query：query 包含公司名+指标名"""
    from reward_knowledge_base import query_has_company_name, query_has_metric
    best = 0.1
    for q in queries:
        has_company = query_has_company_name(q)
        has_metric = query_has_metric(q)
        if has_company and has_metric:
            return 0.8  # 最高分，early exit
        elif has_company or has_metric:
            best = max(best, 0.4)
    return best


def _strategy_simple(queries: List[str]) -> float:
    """single_company_simple：query 包含公司名+关键词"""
    from reward_knowledge_base import extract_company_name
    report_keywords = ["评级", "目标价", "前景", "投资", "研报", "推荐", "买入", "增持"]
    best = 0.1
    for q in queries:
        has_company = bool(extract_company_name(q))
        has_keyword = any(kw in q for kw in report_keywords)
        if has_company and has_keyword:
            return 0.8  # 最高分，early exit
        elif has_company:
            best = max(best, 0.4)
    return best


# ============ Reject 特殊处理 ============

def _compute_reject_reward(env, answer: str) -> float:
    """reject 类型：少搜是对的"""
    from reward_knowledge_base import is_reject_response

    n_calls = len(env.tool_steps)
    rejected = is_reject_response(answer)

    if rejected:
        if n_calls <= 1:
            return 1.0
        elif n_calls == 2:
            return 0.75
        elif n_calls == 3:
            return 0.50
        else:
            return 0.25
    else:
        return 0.0  # 没拒绝


# ============ LLM 二元判断（维度 5，可选）============

LLM_BINARY_PROMPT = """你是一个金融分析 Agent 工具调用策略的评审。

## Agent 的能力范围
Agent **仅有**以下 4 个工具，没有其他信息来源：
1. **search_financial(query)**：搜索公司财务数据（ROE、净利率、资产负债率等），每次返回 top-3 条
2. **search_report(query)**：搜索券商研报（目标价、评级、EPS 预测），每次返回 top-3 条
3. **search_industry(query)**：搜索行业对比数据（同行业公司指标均值和排名）
4. **calculate(expression)**：计算数学表达式

数据库覆盖：沪深300公司的财务指标 + 券商研报 + 30个行业对比数据。
**不包含**：现金流明细、费用率、股价K线、管理层信息、政策原文。

## 评判标准
请判断 Agent 的搜索策略在**上述工具能力范围内**是否合理：
- **合理**：在有限的工具内，做出了基本正确的工具选择和搜索顺序。不要求完美，只要主要信息覆盖到即可。为获取不同维度的数据而分多次搜索是合理的，但用相似的 query 重复搜索同一类信息算冗余。
- **不合理**：明显的工具选择错误（如该搜财务数据却只搜了研报），或完全遗漏了关键信息维度（如对比两家公司只搜了一家）。

## 待评审的问题和工具调用

问题：{question}
问题类型：{question_type}

Agent 的工具调用序列：
{formatted_steps}

只回答"合理"或"不合理"，然后用一句话说明理由。

格式：
判断：合理/不合理
理由：XXX"""


def _batch_llm_binary_judge(llm_tasks: dict) -> dict:
    """并行调用 LLM 二元判断"""
    scores = {}

    def _call_single(idx, question, question_type, tool_steps):
        if not tool_steps:
            return idx, 0.0

        formatted_steps = "\n".join(
            f"  Step {i+1}: {s['tool']}(query=\"{s['query']}\")"
            for i, s in enumerate(tool_steps)
        )
        prompt = LLM_BINARY_PROMPT.format(
            question=question,
            question_type=question_type,
            formatted_steps=formatted_steps,
        )

        try:
            global _llm_client
            if _llm_client is None:
                with _llm_client_lock:
                    if _llm_client is None:
                        from openai import OpenAI
                        _llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

            response = _llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1,
                extra_body={"enable_thinking": False},
            )
            result = response.choices[0].message.content.strip()
            # Strip thinking tags
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

            # 解析"合理"/"不合理"
            if "不合理" in result:
                return idx, 0.0
            elif "合理" in result:
                return idx, 1.0
            else:
                return idx, 0.5  # 无法解析
        except Exception as e:
            logger.error(f"LLM binary judge 失败 (idx={idx}): {e}")
            return idx, 0.5

    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as executor:
        futures = []
        for idx, (question, question_type, tool_steps) in llm_tasks.items():
            futures.append(executor.submit(_call_single, idx, question, question_type, tool_steps))

        for future in as_completed(futures):
            try:
                idx, score = future.result()
                scores[idx] = score
            except Exception as e:
                logger.error(f"LLM future 异常: {e}")

    return scores


# ============ 硬约束（继承自 V3）============

def _is_format_valid(env, completion, question_type: str) -> bool:
    """
    V4.1 格式检查（两层）：
    1. completion 非空
    2. environment_factory 下有内容但 tool_steps 为空且非 reject → JSON 解析失败
    """
    comp_str = str(completion) if completion else ""
    if not comp_str.strip():
        return False
    if not env.tool_steps and question_type != "reject" and len(comp_str) > 100:
        return False
    return True


def _call_quality_penalty(env) -> float:
    """硬约束扣分：重复 query + 无效工具"""
    penalty = 0.0

    # 重复调用惩罚（V4.1 阈值放松到 0.85）
    queries = [s["query"] for s in env.tool_steps]
    for i, q in enumerate(queries):
        for prev_q in queries[:i]:
            if _keyword_overlap(q, prev_q) > REPEAT_QUERY_THRESHOLD:
                penalty -= 0.3
                break

    # 无效工具惩罚
    for s in env.tool_steps:
        if s["tool"] not in VALID_TOOLS:
            penalty -= 0.5

    return max(penalty, -1.0)


def _apply_overlong_penalty(base_reward: float, length: int) -> float:
    """DAPO 渐进式 overlong penalty"""
    if length <= OVERLONG_L_MAX:
        return base_reward
    elif length <= OVERLONG_L_CLIP:
        penalty = (length - OVERLONG_L_MAX) / (OVERLONG_L_CLIP - OVERLONG_L_MAX) * OVERLONG_C
        return base_reward - penalty
    else:
        return base_reward - OVERLONG_C


# ============ 辅助函数 ============

def _completion_to_str(completion) -> str:
    """将 completion 转为字符串（兼容 list[dict] 格式）"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for msg in completion:
            if isinstance(msg, dict):
                parts.append(msg.get("content", ""))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    return str(completion)


def _extract_final_answer(completion: str) -> str:
    """从 completion 中提取最终答案"""
    parts = completion.split("<tool_call>")
    last_part = parts[-1] if parts else completion

    for tag in ["</tool_call>", "<|im_end|>", "<|im_start|>"]:
        last_part = last_part.replace(tag, "")

    if "<tool_response>" in last_part:
        parts2 = last_part.split("</tool_response>")
        last_part = parts2[-1] if len(parts2) > 1 else last_part

    return last_part.strip()


def _keyword_overlap(query1: str, query2: str) -> float:
    """计算两个 query 的关键词重叠率（2-gram 级别）"""
    words1 = set(query1[i:i+2] for i in range(len(query1)-1)) if query1 else set()
    words2 = set(query2[i:i+2] for i in range(len(query2)-1)) if query2 else set()
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    return len(intersection) / min(len(words1), len(words2))


def _estimate_token_count(text: str) -> int:
    """粗估 token 数量（中文 1 字 ≈ 1.5 token）"""
    return int(len(str(text)) * 1.5)


def _extract_question_and_type(idx: int, kwargs: dict) -> tuple:
    """从 kwargs 中提取第 idx 条的问题和类型"""
    question = ""
    question_type = ""

    if "question" in kwargs:
        q_list = kwargs["question"]
        if isinstance(q_list, list) and idx < len(q_list):
            question = q_list[idx]
        elif isinstance(q_list, str):
            question = q_list

    if "type" in kwargs:
        t_list = kwargs["type"]
        if isinstance(t_list, list) and idx < len(t_list):
            question_type = t_list[idx]
        elif isinstance(t_list, str):
            question_type = t_list

    return question, question_type


# ============ 自定义监控指标 ============

_metrics = {
    "calculate_rate": [],
    "mental_math_rate": [],
    "tool_call_count": [],
    "tool_coverage_scores": [],
    "query_quality_scores": [],
    "calc_behavior_scores": [],
    "strategy_match_scores": [],
    "llm_judge_scores": [],
    "llm_judge_failures": 0,
}
_metrics_lock = threading.Lock()


def _log_custom_metrics(environments, completions, rewards, kwargs):
    """记录自定义监控指标"""
    from reward_knowledge_base import detect_mental_calc

    with _metrics_lock:
        for idx, env in enumerate(environments):
            if rewards[idx] == FORMAT_INVALID_PENALTY:
                continue

            answer = _extract_final_answer(_completion_to_str(completions[idx]))
            _metrics["calculate_rate"].append(1.0 if env.has_calculate else 0.0)
            _metrics["mental_math_rate"].append(1.0 if detect_mental_calc(answer) else 0.0)
            _metrics["tool_call_count"].append(len(env.tool_steps))


def get_and_reset_metrics() -> dict:
    """获取并重置自定义指标（由训练脚本定期调用）"""
    with _metrics_lock:
        result = {}
        for key in [
            "calculate_rate", "mental_math_rate", "tool_call_count",
            "tool_coverage_scores", "query_quality_scores",
            "calc_behavior_scores", "strategy_match_scores",
            "llm_judge_scores",
        ]:
            values = _metrics[key]
            if values:
                result[key] = sum(values) / len(values)
            else:
                result[key] = 0.0
            _metrics[key] = []

        result["llm_judge_failures"] = _metrics["llm_judge_failures"]
        _metrics["llm_judge_failures"] = 0
        return result


# ============ 导出 ============

# GRPOTrainer 使用方式：
#
# from grpo_plugin import FinAgentEnv, finagent_reward
#
# trainer = GRPOTrainer(
#     model=model,
#     train_dataset=dataset,
#     reward_funcs=[finagent_reward],
#     reward_weights=[1.0],
#     environment_factory=FinAgentEnv,
#     args=grpo_config,
# )
