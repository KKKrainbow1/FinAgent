"""
GRPO Plugin: FinAgent 工具调用环境 + Reward 函数（V2 重写）

用于 TRL GRPOTrainer 的 environment_factory 和 reward_funcs。
基于 GRPO Reward Design V5（Planner-focused）。

V2 重写要点（基于 unsloth-buddy review）：
  1. 合并为单个 reward 函数，格式检查/overlong penalty 不再被权重稀释
  2. completeness LLM 调用改为并行（ThreadPoolExecutor）
  3. 工具调用加 try-except 防训练中断
  4. 共享资源加线程锁
  5. 分数解析更鲁棒
  6. FinAgentEnv.__init__ 接受 **kwargs 兼容 TRL

设计原则：
  - Reward 评价工具调用策略（Planner），不评价答案质量（Summarizer）
  - completeness（LLM 判断工具调用序列完整性）+ calc_behavior（规则检测 calculate 使用）
  - 参考论文：RLTR（EMNLP 2025）、ToolRL（NeurIPS 2025）、DAPO

依赖：
  - hybrid_search.py（FinAgentRetriever）
  - tools.py（FinAgentTools, TOOLS_NATIVE）
  - qwen3-max API（completeness 评估）
"""

import re
import os
import logging
import threading
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ============ 配置 ============

# completeness LLM 配置
COMPLETENESS_MODEL = os.environ.get("COMPLETENESS_MODEL", "qwen3-max")
COMPLETENESS_API_KEY = os.environ.get("COMPLETENESS_API_KEY", "")
COMPLETENESS_BASE_URL = os.environ.get(
    "COMPLETENESS_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
COMPLETENESS_MAX_WORKERS = int(os.environ.get("COMPLETENESS_MAX_WORKERS", "8"))

# DAPO overlong penalty 参数（基于 sft_v3_r32 推理统计）
OVERLONG_L_MAX = 8192      # P90 ≈ 8800 token，正常轨迹不惩罚
OVERLONG_L_CLIP = 12288    # 最长 ≈ 11800 token + 余量
OVERLONG_C = 0.5           # 最大惩罚幅度

# 格式不合规惩罚（Phase 2b 验证：>98% 合规用 -1.0，<95% 改为 -0.5）
FORMAT_INVALID_PENALTY = -1.0

# 心算检测正则（检查 Answer，不检查 Thought）
MENTAL_MATH_PATTERNS = [
    re.compile(r'[\d.]+%?\s*[×*]\s*[\d.]+%?\s*[×*]\s*[\d.]+'),    # 杜邦三因子乘法
    re.compile(r'1\s*/\s*[\d.]+%?\s*[=≈]\s*[\d.]+'),               # 权益乘数 1/X%≈Y
    re.compile(r'[\d.]+\s*[-−]\s*[\d.]+\s*[=≈]\s*[\d.]+'),         # 差值 A-B=C
    re.compile(r'[\d.]+\s*[/÷]\s*[\d.]+\s*[=≈]\s*[\d.]+'),         # 除法 A/B=C
    re.compile(r'[\d.]+\s*[+]\s*[\d.]+\s*[=≈]\s*[\d.]+'),          # 加法 A+B=C
]

# V2 Checklist：按问题类型的评分标准（替代 V1 的 TYPE_EXPECTATIONS）
TYPE_CHECKLISTS = {
    "financial_query": """该类型（财务数据查询）评分标准：
- 基础分 3 分：调用了 search_financial 查询目标指标
- +1 分：query 中包含了具体指标名称（如 ROE、净利率）和年份
- +1 分：query 中包含了公司全称（而非简称或模糊描述）
- 封顶 5 分
注意：这是简单查询，1 次 search_financial 即可满足。不应因为没有调用 search_report 或 search_industry 而扣分。""",

    "single_company_simple": """该类型（单公司简单问题）评分标准：
- 基础分 3 分：调用了 search_report 查询研报观点
- +1 分：query 精准（包含公司名 + 问题相关关键词）
- +1 分：同时搜索了 search_financial 补充数据支撑
- 封顶 5 分
注意：这是简单问题，1-2 次工具调用即可。不应因为没有调用 calculate 或 search_industry 而扣分。""",

    "single_company_medium": """该类型（单公司中等分析）评分标准：
- 基础分 1 分：至少调用了一个工具
- +1 分：调用了 search_financial，query 包含与问题相关的具体指标类别（盈利能力 → ROE/净利率/毛利率；偿债能力 → 资产负债率/流动比率；全面评估 → 至少覆盖两个维度）
- +1 分：调用了 search_report 获取券商研报观点，或调用了 search_industry 获取行业对比数据（单公司指标与行业均值对比才有分析意义）
- +1 分：如果问题涉及杜邦分析（ROE 拆解为净利率×周转率×权益乘数）、增长率计算或数值对比，调用了 calculate（如果问题不涉及计算，此项自动满足）
- +1 分：query 精准（包含具体指标名称、年份、公司全称，而非模糊描述）
- 封顶 5 分
注意：不是所有 medium 问题都需要所有工具。请根据具体问题判断哪些条件适用。""",

    "company_comparison": """该类型（公司对比）评分标准：
- 基础分 1 分：至少调用了一个工具
- +1 分：搜索了第一家公司的相关数据（search_financial 或 search_report）
- +1 分：搜索了第二家公司的相关数据（search_financial 或 search_report）
- +1 分：如果问题涉及数值差异对比，调用了 calculate；或调用了 search_industry 获取行业基准作为对比参照（满足其一即可）
- +1 分：query 精准且两家公司搜索的指标一致（便于横向对比）
- 封顶 5 分
关键：对比类问题必须搜索所有涉及的公司。只搜了一家公司最高 3 分。""",

    "risk_analysis": """该类型（风险分析）评分标准：
- 基础分 1 分：至少调用了一个工具
- +1 分：调用了 search_financial 搜索了杠杆/偿债指标（资产负债率、流动比率、速动比率）或盈利指标（ROE、净利率）
- +1 分：调用了 search_report 获取研报中的风险分析或信用评级观点
- +1 分：调用了 search_industry 获取行业对比数据，以便判断公司指标是否偏离行业正常水平
- +1 分：query 精准，偿债搜索包含杠杆指标名称，盈利搜索包含利润指标名称（而非泛泛的"风险"或"财务"）
- 封顶 5 分""",

    "industry_analysis": """该类型（行业分析）评分标准：
- 基础分 1 分：至少调用了一个工具
- +1 分：调用了 search_industry 搜索行业对比数据（行业全景：多家公司的指标排名和均值）
- +1 分：调用了 search_financial 搜索了具体代表性个股的财务数据（补充行业数据中缺失的细节）
- +1 分：调用了 search_report 获取行业相关研报观点或行业前景分析
- +1 分：query 覆盖了问题涉及的核心行业名称和具体指标（如 ROE、净利率、毛利率，而非泛泛的"盈利"）
- 封顶 5 分""",

    "reject": """该类型（应拒绝的问题）评分标准：
- 5 分：识别出问题超出数据库能力范围，工具调用 ≤ 1 次后即拒绝
- 4 分：进行了 1-2 次工具调用后识别出无法回答，合理拒绝
- 3 分：进行了 3 次以上工具调用才拒绝，浪费了检索资源
- 2 分：没有拒绝，强行回答了一个超出范围的问题
- 1 分：大量无效工具调用，且强行回答
注意：对于 reject 类问题，"少搜"是对的，"多搜"反而扣分。这与其他类型相反。""",
}

# V2 Few-shot Examples（嵌入 prompt）
FEW_SHOT_EXAMPLES = {
    "single_company_medium": """【示例 A】5 分
问题：全面评估贵州茅台的盈利能力
工具调用序列：
  Step 1: search_financial(query="贵州茅台 ROE 净利率 总资产周转率 权益乘数 2024")
  Step 2: search_industry(query="白酒行业 ROE 净利率 对比")
  Step 3: search_report(query="贵州茅台 盈利能力 投资评级")
  Step 4: calculate(query="29.9 / 100 * 0.61 * 5.48")
理由：搜索了财务数据含杜邦三因子（+1），行业对比提供分析上下文（+1），涉及杜邦拆解调用了 calculate 验证 ROE = 净利率 × 周转率 × 权益乘数（+1），query 精准包含具体指标和年份（+1），基础分 1，合计 5 分。

【示例 B】2 分
问题：分析格力电器的盈利能力
工具调用序列：
  Step 1: search_financial(query="格力电器")
理由：搜索了财务数据（+1），但 query 无具体指标名和年份（+0）、没搜研报或行业对比（+0）、涉及盈利分析但没调 calculate（+0），基础分 1，合计 2 分。""",

    "company_comparison": """【示例 A】5 分
问题：宁德时代和比亚迪的盈利能力谁更强？
工具调用序列：
  Step 1: search_financial(query="宁德时代 ROE 净利率 毛利率 2024")
  Step 2: search_financial(query="比亚迪 ROE 净利率 毛利率 2024")
  Step 3: search_industry(query="新能源行业 ROE 净利率 对比")
  Step 4: calculate(query="20.5 - 15.3")
理由：两家公司都搜了且指标一致（+1, +1），调用了 calculate 算差值并有行业基准参照（+1），query 精准且指标一致（+1），基础分 1，合计 5 分。

【示例 B】2 分
问题：宁德时代和比亚迪的盈利能力谁更强？
工具调用序列：
  Step 1: search_financial(query="宁德时代 ROE 2024")
理由：只搜了宁德时代（+1），没搜比亚迪（+0），无法对比。基础分 1，合计 2 分。""",

    "financial_query": """【示例 A】5 分
问题：贵州茅台的 ROE 是多少？
工具调用序列：
  Step 1: search_financial(query="贵州茅台 ROE 2024")
理由：调用了 search_financial（基础 3 分），query 包含具体指标和年份（+1），包含公司全称（+1），合计 5 分。

【示例 B】3 分
问题：贵州茅台的 ROE 是多少？
工具调用序列：
  Step 1: search_financial(query="茅台 盈利")
理由：调用了 search_financial（基础 3 分），但 query 用了简称且没指定 ROE 和年份（+0, +0），合计 3 分。""",

    "industry_analysis": """【示例 A】5 分
问题：光伏行业主要公司的盈利能力如何？
工具调用序列：
  Step 1: search_industry(query="光伏行业 ROE 净利率 毛利率 对比")
  Step 2: search_financial(query="隆基绿能 ROE 净利率 毛利率 2024")
  Step 3: search_financial(query="通威股份 ROE 净利率 2024")
  Step 4: search_report(query="光伏行业 盈利能力 前景 2024")
理由：调用了 search_industry 获取行业全景（+1），search_financial 搜了多家代表性个股补充细节（+1），search_report 获取行业研报（+1），query 覆盖了核心行业名称和具体指标名（+1），基础分 1，合计 5 分。

【示例 B】2 分
问题：白酒行业的盈利能力排名如何？
工具调用序列：
  Step 1: search_financial(query="贵州茅台 ROE 2024")
理由：只搜了单个公司的财务数据（+1），没有调用 search_industry 获取行业全景（+0）、没搜研报（+0）、query 只覆盖了一家公司（+0），基础分 1，合计 2 分。""",

    "risk_analysis": """【示例 A】5 分
问题：分析中国平安的财务风险
工具调用序列：
  Step 1: search_financial(query="中国平安 资产负债率 流动比率 速动比率 2024")
  Step 2: search_financial(query="中国平安 ROE 净利率 2024")
  Step 3: search_report(query="中国平安 风险分析 信用评级")
  Step 4: search_industry(query="保险行业 资产负债率 ROE 对比")
理由：分层搜索偿债指标和盈利指标（+1），研报风险分析和信用评级（+1），行业对比判断指标是否偏离正常水平（+1），query 精准含杠杆指标名和利润指标名（+1），基础分 1，合计 5 分。

【示例 B】2 分
问题：评估万科的债务风险
工具调用序列：
  Step 1: search_financial(query="万科")
理由：搜索了财务数据（+1），但 query 太模糊无杠杆指标名（+0）、没搜研报风险分析（+0）、没搜行业对比（+0），基础分 1，合计 2 分。""",

    "reject": """【示例 A】5 分
问题：分析特斯拉 2024 年在中国市场的表现
工具调用序列：
  Step 1: search_report(query="特斯拉 中国市场")
  → Agent 识别出检索结果不相关，拒绝回答
理由：1 次工具调用后识别出超出范围并拒绝，符合预期。5 分。

【示例 B】2 分
问题：分析苹果公司的财务状况
工具调用序列：
  Step 1: search_financial(query="苹果 财务")
  Step 2: search_report(query="苹果 评级")
  Step 3: search_industry(query="消费电子 对比")
  → Agent 用搜到的不相关数据强行生成了答案
理由：大量无效搜索且没有拒绝，强行回答了超出范围的问题。2 分。""",
}

# 默认 few-shot（用于没有专属示例的类型）
DEFAULT_FEW_SHOT = """【示例 A】高分轨迹
- 搜索覆盖了问题涉及的核心信息维度
- query 精准，包含公司全称、具体指标、年份
- 需要计算时调用了 calculate

【示例 B】低分轨迹
- 只搜了部分信息，关键维度缺失
- query 模糊，缺少具体指标或年份
- 需要计算但没调用 calculate"""

# Completeness Prompt 模板（V2）
COMPLETENESS_PROMPT_TEMPLATE = """你是一位金融分析 Agent 的工具调用评审专家。你需要评估：对于给定的问题，Agent 的工具调用序列是否收集了足够的信息来回答问题。

你只评价"搜了什么"，不评价"搜到的内容好不好"或"最终答案写得好不好"。

## 任务信息

问题：{question}
问题类型：{question_type}

## Agent 的工具调用序列

{formatted_steps}

## 可用工具

| 工具 | 功能 | 示例 query |
|------|------|----------|
| search_financial | 搜索公司财务数据（ROE、净利率、资产负债率、周转率等） | "贵州茅台 ROE 净利率 2024" |
| search_report | 搜索券商研报（评级、目标价、EPS预测、深度分析） | "宁德时代 投资评级 目标价" |
| search_industry | 搜索行业对比数据（同行业公司指标排名和均值） | "白酒行业 ROE 净利率 对比" |
| calculate | 数学计算（杜邦拆解、增长率计算、差值对比等） | "36.99 / 100 * 0.6 * 2.1" |

## 金融分析参考框架

好的工具调用策略通常遵循以下原则：
- 盈利能力分析应搜索 ROE、净利率、毛利率等指标，涉及 ROE 拆解时应调用 calculate（杜邦分析：ROE = 净利率 × 总资产周转率 × 权益乘数）
- 风险/偿债分析应搜索资产负债率、流动比率等杠杆指标，与盈利指标分开搜索更精准
- 单公司指标应与行业均值对比才有分析意义（search_industry）
- 对比分析应为两家公司搜索一致的指标，便于横向对比
- 数值对比、增长率计算（如同比增长率 = (本期-上期)/上期）、比率计算应使用 calculate，不应心算

## 该类型问题的评分标准

{type_checklist}

## 评分示例

{few_shot_examples}

## 输出要求

逐项检查上述评分标准，统计满足的条件数，然后输出：
1. 分数（1-5）
2. 一句话理由（说明满足了哪些条件、缺少了什么）

格式：
分数：X
理由：XXX"""

VALID_TOOLS = {"search_financial", "search_report", "search_industry", "calculate"}

# LLM client（延迟初始化，模块级共享）
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
                    cls._shared_retriever.load_index()
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
            return self._shared_tools.call("search_financial", {"query": query})
        except Exception as e:
            logger.warning(f"search_financial 调用失败: {e}")
            return f"搜索失败：{str(e)}"

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
            return self._shared_tools.call("search_report", {"query": query})
        except Exception as e:
            logger.warning(f"search_report 调用失败: {e}")
            return f"搜索失败：{str(e)}"

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
            return self._shared_tools.call("search_industry", {"query": query})
        except Exception as e:
            logger.warning(f"search_industry 调用失败: {e}")
            return f"搜索失败：{str(e)}"

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
            result = self._shared_tools.call("calculate", {"expression": expression})
            self.calc_results.append(result)
            return result
        except Exception as e:
            logger.warning(f"calculate 调用失败: {e}")
            return f"计算失败：{str(e)}"


# ============ 统一 Reward 函数 ============

def finagent_reward(completions, environments=None, **kwargs) -> list[float]:
    """
    FinAgent GRPO 统一 reward 函数。

    合并所有 reward 逻辑为一个函数，确保：
    - 格式不合规直接返回 -1.0（不被权重稀释）
    - DAPO overlong penalty 直接从总分扣除
    - completeness 和 calc_behavior 按 0.6:0.3 加权

    用法：
        trainer = GRPOTrainer(
            reward_funcs=[finagent_reward],
            reward_weights=[1.0],
            environment_factory=FinAgentEnv,
            ...
        )

    Args:
        completions: list[str]，每条轨迹的完整生成文本
        environments: list[FinAgentEnv]，TRL 通过 kwargs 传入的环境实例列表
        **kwargs: 包含数据集字段（question, type 等）

    Returns:
        list[float]，每条轨迹的 reward
    """
    if environments is None:
        environments = kwargs.get("environments", kwargs.get("envs", []))

    rewards = []
    # 收集需要调 LLM 的任务
    llm_tasks = {}  # idx -> (question, question_type, tool_steps)

    for idx, (env, completion) in enumerate(zip(environments, completions)):
        # ---- Step 1: 格式合规前置检查 ----
        if not _is_format_valid(completion):
            rewards.append(FORMAT_INVALID_PENALTY)
            continue

        # 先占位，后面填入真实值
        rewards.append(None)

        # 收集 LLM 任务
        question, question_type = _extract_question_and_type(idx, kwargs)
        llm_tasks[idx] = (question, question_type, env.tool_steps)

    # ---- Step 2: 并行调用 completeness LLM ----
    completeness_scores = {}
    with ThreadPoolExecutor(max_workers=COMPLETENESS_MAX_WORKERS) as executor:
        futures = {}
        for idx, (question, question_type, tool_steps) in llm_tasks.items():
            if not tool_steps:
                completeness_scores[idx] = 0.0
                continue
            future = executor.submit(
                _call_completeness_llm, question, question_type, tool_steps
            )
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                completeness_scores[idx] = future.result()
            except Exception as e:
                logger.error(f"completeness LLM 异常 (idx={idx}): {e}")
                completeness_scores[idx] = 0.5
                with _metrics_lock:
                    _metrics["completeness_llm_failures"] += 1

    # ---- Step 3: 计算每条轨迹的最终 reward ----
    for idx in llm_tasks:
        env = environments[idx]
        completion = completions[idx]
        answer = _extract_final_answer(completion)

        # completeness（权重 0.60）
        comp = completeness_scores.get(idx, 0.5)

        # calc_behavior（权重 0.30）
        calc = _calc_behavior(env, answer)

        # 加权合并（2:1 比例，和为 1.0）
        base_reward = 0.667 * comp + 0.333 * calc

        # 硬约束扣分：重复调用 + 无效工具
        penalty = _call_quality_penalty(env)
        base_reward = max(base_reward + penalty, 0.0)  # clip 到 >= 0

        # DAPO overlong penalty（直接从总分扣，不被权重稀释）
        length = _estimate_token_count(completion)
        base_reward = _apply_overlong_penalty(base_reward, length)

        rewards[idx] = base_reward

    # 自定义指标 logging
    _log_custom_metrics(environments, completions, rewards, completeness_scores, kwargs)

    return rewards


# ============ Reward 子函数 ============

def _calc_behavior(env, answer: str) -> float:
    """
    calculate 行为检测（4 档分级）

    - 1.0：调了 calculate 且结果被答案引用
    - 0.7：调了 calculate 但结果没被引用
    - 0.5：没调也没心算（不需要计算的问题）
    - 0.0：没调但有心算模式
    """
    if env.has_calculate:
        cited = any(_result_in_answer(r, answer) for r in env.calc_results)
        return 1.0 if cited else 0.7
    else:
        return 0.0 if _has_mental_math(answer) else 0.5


def _call_quality_penalty(env) -> float:
    """硬约束扣分：重复调用 + 无效工具"""
    penalty = 0.0

    # 重复调用惩罚
    queries = [s["query"] for s in env.tool_steps]
    for i, q in enumerate(queries):
        for prev_q in queries[:i]:
            if _keyword_overlap(q, prev_q) > 0.7:
                penalty -= 0.3
                break

    # 无效工具惩罚
    for s in env.tool_steps:
        if s["tool"] not in VALID_TOOLS:
            penalty -= 0.5

    return max(penalty, -1.0)  # clip 到 >= -1.0


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

def _extract_final_answer(completion: str) -> str:
    """
    从 completion 中提取最终答案。

    注意：TRL environment_factory 模式下 completion 的具体格式
    需要在 dry-run 时验证。当前实现基于 Qwen2.5 chat_template 的假设。
    """
    parts = completion.split("<tool_call>")
    last_part = parts[-1] if parts else completion

    for tag in ["</tool_call>", "<|im_end|>", "<|im_start|>"]:
        last_part = last_part.replace(tag, "")

    if "<tool_response>" in last_part:
        parts2 = last_part.split("</tool_response>")
        last_part = parts2[-1] if len(parts2) > 1 else last_part

    return last_part.strip()


def _has_mental_math(answer: str) -> bool:
    """检查答案中是否包含心算模式"""
    return any(p.search(answer) for p in MENTAL_MATH_PATTERNS)


def _result_in_answer(calc_result: str, answer: str) -> bool:
    """检查 calculate 返回的结果数字是否出现在答案中（完整匹配）"""
    numbers = re.findall(r'[\d.]+', calc_result)
    for num in numbers:
        if len(num) < 3:
            continue  # 跳过太短的数字
        if re.search(r'(?<!\d)' + re.escape(num) + r'(?!\d)', answer):
            return True
    return False


def _is_format_valid(completion: str) -> bool:
    """检查输出格式是否合规"""
    if not completion or not completion.strip():
        return False
    if "<tool_call>" in completion:
        if completion.count("<tool_call>") != completion.count("</tool_call>"):
            return False
    return True


def _keyword_overlap(query1: str, query2: str) -> float:
    """计算两个 query 的关键词重叠率（2-gram 级别）"""
    words1 = set(query1.split()) if ' ' in query1 else set(
        query1[i:i+2] for i in range(len(query1)-1)
    )
    words2 = set(query2.split()) if ' ' in query2 else set(
        query2[i:i+2] for i in range(len(query2)-1)
    )
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    return len(intersection) / min(len(words1), len(words2))


def _estimate_token_count(text: str) -> int:
    """粗估 token 数量（中文 1 字 ≈ 1.5 token）"""
    return int(len(text) * 1.5)


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

    if not question:
        logger.warning(f"idx={idx}: question 为空，completeness 评估可能不准确")

    return question, question_type


def _call_completeness_llm(question: str, question_type: str, tool_steps: list) -> float:
    """
    调用 LLM 评估工具调用完整性，返回 0-1 的归一化分数。
    V2：checklist 化评分 + few-shot examples + 按类型拆分标准。
    只传 tool name + query，不传 observation（Planner-focused）。
    """
    # 格式化工具调用序列
    if not tool_steps:
        formatted_steps = "  （无工具调用）"
    else:
        formatted_steps = "\n".join(
            f"  Step {i+1}: {s['tool']}(query=\"{s['query']}\")"
            for i, s in enumerate(tool_steps)
        )

    # 获取该类型的 checklist 和 few-shot
    type_checklist = TYPE_CHECKLISTS.get(question_type, TYPE_CHECKLISTS["single_company_medium"])
    few_shot = FEW_SHOT_EXAMPLES.get(question_type, DEFAULT_FEW_SHOT)

    prompt = COMPLETENESS_PROMPT_TEMPLATE.format(
        question=question,
        question_type=question_type,
        formatted_steps=formatted_steps,
        type_checklist=type_checklist,
        few_shot_examples=few_shot,
    )

    try:
        global _llm_client
        if _llm_client is None:
            with _llm_client_lock:
                if _llm_client is None:
                    from openai import OpenAI
                    _llm_client = OpenAI(
                        api_key=COMPLETENESS_API_KEY,
                        base_url=COMPLETENESS_BASE_URL,
                    )
        response = _llm_client.chat.completions.create(
            model=COMPLETENESS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
        )
        result = response.choices[0].message.content.strip()
        return _parse_completeness_score(result)

    except Exception as e:
        logger.error(f"completeness LLM 调用失败: {e}")
        return 0.5


def _parse_completeness_score(result: str) -> float:
    """
    从 LLM 输出中解析 1-5 分，归一化到 0-1。

    V2 输出格式："分数：X\n理由：XXX"
    """
    # 优先：匹配 V2 格式 "分数：X" 或 "分数:X"
    match = re.search(r'分数\s*[:：]\s*([1-5])', result)
    if match:
        return (int(match.group(1)) - 1) / 4

    # 其次：匹配 "X分" 模式
    match = re.search(r'([1-5])\s*分', result)
    if match:
        return (int(match.group(1)) - 1) / 4

    # 再次：匹配 "评分X" / "给X" / "打X" 模式
    match = re.search(r'(?:评分|给|打|得)\s*[:：]?\s*([1-5])', result)
    if match:
        return (int(match.group(1)) - 1) / 4

    # 兜底：取第一行首个 1-5 数字
    first_line = result.split('\n')[0]
    match = re.search(r'([1-5])', first_line)
    if match:
        return (int(match.group(1)) - 1) / 4

    logger.warning(f"completeness LLM 无法解析分数: {result}")
    return 0.5


# ============ 自定义监控指标 ============

# 用模块级变量累积指标，由外部定期读取和清零
_metrics = {
    "calculate_rate": [],
    "mental_math_rate": [],
    "completeness_scores": [],
    "tool_call_count": [],
    "completeness_llm_failures": 0,
}
_metrics_lock = threading.Lock()


def _log_custom_metrics(environments, completions, rewards, completeness_scores, kwargs):
    """记录自定义监控指标"""
    with _metrics_lock:
        for idx, env in enumerate(environments):
            if rewards[idx] == FORMAT_INVALID_PENALTY:
                continue  # 格式不合规的不计入统计

            answer = _extract_final_answer(completions[idx])
            _metrics["calculate_rate"].append(1.0 if env.has_calculate else 0.0)
            _metrics["mental_math_rate"].append(1.0 if _has_mental_math(answer) else 0.0)
            _metrics["tool_call_count"].append(len(env.tool_steps))

            if idx in completeness_scores:
                _metrics["completeness_scores"].append(completeness_scores[idx])


def get_and_reset_metrics() -> dict:
    """获取并重置自定义指标（由训练脚本定期调用）"""
    with _metrics_lock:
        result = {}
        for key in ["calculate_rate", "mental_math_rate", "completeness_scores", "tool_call_count"]:
            values = _metrics[key]
            if values:
                result[key] = sum(values) / len(values)
            else:
                result[key] = 0.0
            _metrics[key] = []

        result["completeness_llm_failures"] = _metrics["completeness_llm_failures"]
        _metrics["completeness_llm_failures"] = 0
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
