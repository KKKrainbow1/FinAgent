"""
FinAgent SFT 数据生成脚本 V2（ReAct Loop 版）

改进点（对比 V1）：
  - 问题 1（47% Thought 连贯性断裂）：V1 一次性生成所有 Thought 后再填 Observation，
    导致中间 Thought 写"已获取数据"的空洞句。V2 改为逐步 ReAct Loop，每步 Thought
    在看到上一步真实 Observation 后生成，天然引用具体数字。
  - 问题 2（27% 时间一致性）：V2 的 STEP_PROMPT 和 ANSWER_PROMPT 均包含时间一致性约束，
    模型在每步都能看到真实返回的年份数据，可主动选择一致的数据期。
  - Question 来源：从 sft_data_final_v3.jsonl 去重清洗 + supplementary_questions.jsonl 补充，
    不再由 LLM 生成 question。

运行方式：
    # 先测试 5 条
    python 10_generate_sft_data.py --test 5

    # 生成全部
    python 10_generate_sft_data.py

    # 只生成某个类型
    python 10_generate_sft_data.py --type financial_query

    # 从断点续传
    python 10_generate_sft_data.py --resume

环境变量（运行前设置）：
    export OPENAI_API_KEY="你的百炼API Key"
    export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
"""

import json
import os
import re
import time
import random
import argparse
import logging
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from hybrid_search import FinAgentRetriever
from tools import FinAgentTools

# ============ 配置 ============

OUTPUT_DIR = "./data/sft"
SEED_DATA_PATH = "./sft_seed_data_v3.jsonc"
QUESTIONS_PATH = "./data/sft/all_questions_v2.jsonl"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint_v2.json")
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sft_data_v2.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, f"gen_v2_{datetime.now():%Y%m%d_%H%M%S}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============ 步数控制 ============

MIN_STEPS = {
    "financial_query": 1,
    "single_company_simple": 2,
    "single_company_medium": 3,
    "company_comparison": 3,
    "industry_analysis": 3,
    "risk_analysis": 3,
    "reject": 1,
}

MAX_STEPS = {
    "financial_query": 3,
    "single_company_simple": 3,
    "single_company_medium": 4,
    "company_comparison": 5,
    "industry_analysis": 5,
    "risk_analysis": 4,
    "reject": 2,
}

MODEL = "qwen3-max"
VALID_TOOLS = {"search_report", "search_financial", "calculate", "finish"}
MAX_RETRY = 3


# ============ Question 加载 ============

def load_questions() -> list:
    """
    从预处理好的 all_questions_v2.jsonl 加载所有 question。
    该文件已完成去重、非A股 comparison 转 reject、补充 question 合并。
    返回: [{"question": str, "type": str}, ...]
    """
    questions = []
    with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            questions.append({"question": d["question"], "type": d["type"]})

    logger.info(f"加载 question: {len(questions)} 条")
    from collections import Counter
    dist = Counter(q["type"] for q in questions)
    for t, c in dist.most_common():
        logger.info(f"  {t}: {c}")

    return questions


# ============ 预检索（复用 V1） ============

def pre_retrieve(question: str, retriever: FinAgentRetriever) -> str:
    """预检索：探测检索系统能返回什么数据"""
    # 从 question 中提取关键词做探测查询
    probe_queries = [question]

    available_info = []
    for query in probe_queries:
        try:
            results_f = retriever.search_financial(query, top_k=2)
            for r in results_f:
                source = r["metadata"].get("source_type", "unknown")
                text_preview = r["text"][:150]
                available_info.append(f"- [{source}] {text_preview}...")

            results_r = retriever.search_report(query, top_k=2)
            for r in results_r:
                source = r["metadata"].get("source_type", "unknown")
                text_preview = r["text"][:150]
                available_info.append(f"- [{source}] {text_preview}...")
        except Exception as e:
            logger.warning(f"预检索失败: {e}")

    if not available_info:
        return "[预检索结果为空]"

    available_info = list(set(available_info))[:8]
    return "\n".join(available_info)


# ============ Prompt 定义 ============

STEP_PROMPT = """你是"金融翻译官"，一个专业的A股上市公司分析助手。
你正在逐步分析用户的问题，每次只输出一步（Thought + Action + Action Input）。

## 数据库覆盖范围（⭐生成 query 时必须参考）
- **财务数据**：2022H1 ~ 2024年报（共6期，300家A股公司）。没有2025年及以后的财务数据。
- **券商研报**：2017年 ~ 2026年3月（约39,000篇元数据 + 64,000篇PDF正文chunk）
  - 注意：数据库中存在 2017-2021 年的老研报，评级和目标价已严重过时
  - 除非用户明确询问历史数据，否则应优先检索最近1-2年的研报
- **时效性原则**：
  - 用户问"XX怎么样"/"XX盈利能力" → 检索最新数据（query 中加 "2024"）
  - 用户问"XX近几年趋势"/"XX变化" → 不限定年份，让检索返回多期数据
  - 用户问"XX目标价"/"XX评级" → query 中加 "2025 2026" 以获取最新研报

## 可用工具
| 工具名 | 功能 | 输入格式 |
|--------|------|----------|
| search_report | 检索券商研报 | query字符串 |
| search_financial | 检索财务数据 | query字符串 |
| calculate | 数学计算 | 纯数学表达式（如 36.99 - 24.53） |
| finish | 输出最终答案 | 分析报告文本 |

## 工具使用原则
- search_financial 的 query 中只写一家公司名，不要同时查多家
- 所有数值计算必须通过 calculate 工具完成，不要在 Thought 中心算
- 确认信息已足够后才使用 finish

## Thought 写作要求（⭐核心）
- **第一步**：说明分析框架和需要获取什么数据
- **后续步骤（最重要）**：必须引用上一步 Observation 中的**至少2个具体数字**，
  说明这些数字意味着什么，然后解释为什么需要执行下一步
  - 好的例子："Observation 显示阳光电源2024年 ROE 为29.90%（好，>15%），毛利率29.42%，
    但营收增速仅7.76%（较2023年79.47%大幅放缓）。盈利能力强但增长乏力，
    需进一步检索券商研报了解市场对其未来增长的预期。"
  - 坏的例子："已获取数据，接下来查研报。"（❌ 没有引用任何数字）
- **finish 步的 Thought**：综合前面所有数据给出明确判断，不要用假设句式

## 时间一致性要求（⭐重要）
- 如果 Observation 返回了多个年份的数据，在 Thought 中明确选择使用哪一期
- 杜邦分析的三个指标（净利率、总资产周转率、权益乘数）必须来自同一期数据
- 不要在同一段分析中混用年报和半年报数据
- 优先使用最新的年报数据（2024年报 > 2023年报 > 半年报）

## 预检索结果（数据库实际能返回的信息样例）
{pre_retrieved_info}

## 用户问题
{question}

## 问题类型
{question_type}

## reject 类型特殊说明
如果问题类型是 reject，说明用户问的是数据库覆盖范围外的内容（如美股、港股、加密货币、期货、非金融问题等）。
处理方式：**第一步必须先用 search_report 或 search_financial 尝试检索**，确认数据库中没有相关数据后，
第二步再 finish 输出礼貌的拒绝回答（说明我们的数据库只覆盖A股沪深300公司，建议用户调整问题）。
不要跳过检索直接拒绝。

## 已有的分析过程
{history}

## 步数信息
当前是第 {step_num} 步（共需 {min_steps}~{max_steps} 步，其中最后一步必须是 finish）。
{step_hint}

请输出当前这一步：
```json
{{"thought": "...", "action": "...", "action_input": "..."}}
```
只输出 JSON，不要输出其他内容。"""


CALC_FILL_PROMPT = """根据以下已获取的真实数据，为 calculate 步骤生成正确的纯数学表达式。

用户问题：{question}

已获取的数据：
{observations}

calculate 步骤的 Thought：{calc_thought}

要求：
1. 输出一个纯数学表达式，只包含数字和运算符（+、-、*、/、括号）
2. 表达式中的数字必须来自上面的数据，不能编造
3. 只输出表达式本身，不要输出其他文字
4. 如果数据中找不到需要的数字，输出 SKIP

示例：
- Thought说要算ROE差值，数据中A公司ROE=36.99%，B公司ROE=24.53% → 输出：36.99 - 24.53
- Thought说要算权益乘数，数据中资产负债率=42.70% → 输出：1 / (1 - 42.70 / 100)
- Thought说要算杜邦ROE，数据中净利率=14.47%，周转率=0.49，资产负债率=42.70% → 输出：14.47 / 100 * 0.49 * (1 / (1 - 42.70 / 100))
"""


ANSWER_PROMPT = """你是金融分析师。根据以下检索到的真实数据，生成最终分析报告。

用户问题：{question}

已获取的数据（来自真实检索系统）：
{observations}

## 写作要求
1. 所有数字必须来自上述数据，不能编造
2. 每个关键指标应附带判断（好/中/差）和参考标准
3. 引用德勤财务分析框架：盈利能力、偿债能力、营运能力三维度
4. 涉及 ROE 时，可用杜邦分析拆解（ROE = 净利率 × 周转率 × 权益乘数）
5. 偿债指标引用判断标准（流动比率 2:1 正常，资产负债率 40-60% 适宜）
6. 如果数据不足，明确说明"数据有限"，不要强行分析
7. 控制在 300-500 字
8. 最后要有风险提示

## 时间一致性要求（⭐重要）
- 杜邦分析的三个指标必须来自同一期数据（同一年份 + 同一报告类型）
- 如果数据中有多个年份，选择最新的年报数据进行主要分析
- 不要在同一段分析中混用年报和半年报的数据
- 明确标注数据来源期别，如"根据2024年年报数据"

## 种子示例的答案风格（参考结构和深度）
{answer_example}
"""


# ============ 种子数据 ============

def load_seed_data(seed_path: str) -> dict:
    """加载种子数据，按类型分组"""
    with open(seed_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(r'//.*?\n', '\n', content)
    seeds = json.loads(content)

    grouped = {}
    for seed in seeds:
        t = seed["type"]
        if t not in grouped:
            grouped[t] = []
        grouped[t].append(seed)
    return grouped


def get_answer_example(seed_grouped: dict, qtype: str) -> str:
    """获取种子数据中的答案作为风格参考"""
    seeds = seed_grouped.get(qtype, [])
    for seed in seeds:
        for step in seed.get("steps", []):
            if step.get("action") == "finish" and step["action_input"] != "PLACEHOLDER":
                return step["action_input"]
    return ""


# ============ ReAct Loop 核心 ============

def format_history(history: list) -> str:
    """将已有步骤格式化为文本，供 STEP_PROMPT 使用"""
    if not history:
        return "（这是第一步，尚无历史记录）"

    parts = []
    for i, step in enumerate(history):
        parts.append(f"Step {i+1}:")
        parts.append(f"  Thought: {step['thought']}")
        parts.append(f"  Action: {step['action']}")
        parts.append(f"  Action Input: {step['action_input']}")
        if step.get("observation"):
            parts.append(f"  Observation: {step['observation']}")
    return "\n".join(parts)


def generate_step_hint(step_num: int, min_steps: int, max_steps: int) -> str:
    """生成步数提示"""
    remaining_min = max(0, min_steps - step_num)
    remaining_max = max_steps - step_num

    if remaining_min > 1:
        return f"还需要至少 {remaining_min} 步（含 finish），请继续检索数据，不要提前 finish。"
    elif remaining_max <= 1:
        return "这是最后一步，请使用 finish 输出最终答案。"
    else:
        return "如果信息已足够，可以使用 finish；否则继续检索。"


def call_llm_for_step(client: OpenAI, question: str, question_type: str,
                      history: list, step_num: int, min_steps: int,
                      max_steps: int, pre_info: str) -> dict:
    """调用 LLM 生成当前步的 Thought + Action + Action Input"""

    prompt = STEP_PROMPT.format(
        pre_retrieved_info=pre_info,
        question=question,
        question_type=question_type,
        history=format_history(history),
        step_num=step_num + 1,  # 显示为 1-indexed
        min_steps=min_steps,
        max_steps=max_steps,
        step_hint=generate_step_hint(step_num, min_steps, max_steps),
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=800,
        extra_body={"enable_thinking": False},
    )

    result = json.loads(response.choices[0].message.content)
    return result


def fill_calculate(client: OpenAI, question: str, history: list, thought: str) -> tuple:
    """
    Calculate 专用路径：用 CALC_FILL_PROMPT 生成表达式 + 正则校验 + eval。
    返回 (action_input, observation) 或 None 表示失败。
    """
    obs_parts = []
    for i, step in enumerate(history):
        if step.get("observation"):
            obs_parts.append(f"[Step {i+1} - {step['action']}] {step['observation']}")
    obs_text = "\n".join(obs_parts)

    prompt = CALC_FILL_PROMPT.format(
        question=question,
        observations=obs_text,
        calc_thought=thought,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
        extra_body={"enable_thinking": False},
    )
    expr = response.choices[0].message.content.strip()

    if expr == "SKIP" or not expr:
        return None

    # 清理 markdown 标记
    expr = expr.replace("`", "").strip()

    # 正则校验：只允许数字和运算符
    if not re.match(r'^[\d\s\.\+\-\*/\(\)]+$', expr):
        logger.warning(f"calculate 表达式格式异常: {expr}")
        return None

    try:
        result = eval(expr)
        return expr, f"计算结果: {result}"
    except Exception:
        logger.warning(f"calculate 执行失败: {expr}")
        return None


def generate_answer(client: OpenAI, question: str, history: list,
                    seed_grouped: dict, qtype: str) -> str:
    """生成 finish 步的最终答案"""
    obs_parts = []
    for i, step in enumerate(history):
        if step.get("observation"):
            obs_parts.append(f"[Step {i+1}] {step['observation']}")
    obs_text = "\n".join(obs_parts)

    answer_example = get_answer_example(seed_grouped, qtype)

    prompt = ANSWER_PROMPT.format(
        question=question,
        observations=obs_text,
        answer_example=answer_example if answer_example else "（无示例，请按写作要求生成）",
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500,
        extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content


def react_loop(client: OpenAI, tools: FinAgentTools, question: str,
               question_type: str, pre_info: str,
               seed_grouped: dict) -> dict:
    """
    核心 ReAct Loop。逐步生成 Thought/Action，执行 Action 获取 Observation。
    返回完成的 plan dict，或 None 表示失败。
    """
    min_steps = MIN_STEPS.get(question_type, 2)
    max_steps = MAX_STEPS.get(question_type, 4)

    history = []
    retrieval_quality = True

    for step_num in range(max_steps):
        # 1. 调用 LLM 生成当前步
        retry_count = 0
        step_result = None

        while retry_count < MAX_RETRY:
            try:
                step_result = call_llm_for_step(
                    client, question, question_type, history,
                    step_num, min_steps, max_steps, pre_info
                )
            except Exception as e:
                logger.warning(f"LLM 调用失败 (retry {retry_count}): {e}")
                retry_count += 1
                time.sleep(1)
                continue

            thought = step_result.get("thought", "")
            action = step_result.get("action", "")
            action_input = step_result.get("action_input", "")

            # 校验 action 合法性
            if action not in VALID_TOOLS:
                logger.warning(f"非法工具 '{action}'，重试")
                retry_count += 1
                continue

            # 防护：首步禁止直接 finish
            if step_num == 0 and action == "finish":
                logger.warning("首步试图 finish，重试")
                retry_count += 1
                continue

            # 防护：未达最少步数禁止 finish（finish 算一步，所以用 step_num + 1）
            if (step_num + 1) < min_steps and action == "finish":
                logger.warning(f"步数不足 ({step_num + 1} < {min_steps})，禁止 finish，重试")
                retry_count += 1
                continue

            break  # 校验通过

        if step_result is None or retry_count >= MAX_RETRY:
            logger.error(f"步骤生成失败，已重试 {MAX_RETRY} 次")
            return None

        thought = step_result.get("thought", "")
        action = step_result.get("action", "")
        action_input = step_result.get("action_input", "")

        # 2. 分支执行
        if action == "finish":
            # 生成最终答案
            try:
                answer = generate_answer(client, question, history,
                                         seed_grouped, question_type)
            except Exception as e:
                logger.error(f"答案生成失败: {e}")
                return None

            history.append({
                "thought": thought,
                "action": "finish",
                "action_input": answer,
            })
            break

        elif action == "calculate":
            # 专用路径：CALC_FILL_PROMPT + 正则 + eval
            calc_result = fill_calculate(client, question, history, thought)

            if calc_result is None:
                # calculate 失败，跳过这步，让模型在下一步直接 finish
                logger.warning(f"calculate 失败，跳过")
                # 不 append 到 history，让循环继续
                # 但减少 max_steps 避免死循环
                continue

            expr, observation = calc_result
            history.append({
                "thought": thought,
                "action": "calculate",
                "action_input": expr,
                "observation": observation,
            })

        else:  # search_financial / search_report
            try:
                observation = tools.call(action, action_input)
            except Exception as e:
                observation = f"[工具调用失败] {e}"
                logger.warning(f"工具调用失败 [{action}({action_input})]: {e}")

            # 检查检索质量
            if "未找到" in observation or len(observation) < 50:
                retrieval_quality = False
                logger.warning(f"检索质量低: {observation[:100]}")

            history.append({
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation,
            })

        # API 限频
        time.sleep(0.3)

    # 确保最后一步是 finish
    if not history or history[-1].get("action") != "finish":
        logger.warning("循环结束但未 finish，强制生成答案")
        try:
            answer = generate_answer(client, question, history,
                                     seed_grouped, question_type)
            history.append({
                "thought": "综合以上分析数据，生成最终报告。",
                "action": "finish",
                "action_input": answer,
            })
        except Exception as e:
            logger.error(f"强制生成答案失败: {e}")
            return None

    return {
        "question": question,
        "type": question_type,
        "steps": history,
        "retrieval_quality": retrieval_quality,
    }


# ============ 七层验证（复用 V1） ============

def validate_sample(sample: dict) -> tuple:
    """七层自动验证，返回 (通过?, 错误列表)"""
    errors = []
    steps = sample.get("steps", [])

    # 1. 格式完整性
    if not sample.get("question"):
        errors.append("FORMAT: 缺少 question")
    if not steps:
        errors.append("FORMAT: 缺少 steps")
    if not any(s.get("action") == "finish" for s in steps):
        errors.append("FORMAT: 缺少 finish 步骤")

    # 2. 工具调用合法性
    for i, step in enumerate(steps):
        action = step.get("action", "")
        if action not in VALID_TOOLS:
            errors.append(f"TOOL: 第{i+1}步使用非法工具 '{action}'")

    # 3. Observation 真实性
    for i, step in enumerate(steps):
        if step.get("action") != "finish" and not step.get("observation"):
            errors.append(f"OBS: 第{i+1}步缺少 observation")

    # 4. 答案-证据一致性
    qtype = sample.get("type", "")
    if qtype != "reject":
        finish_step = next((s for s in steps if s.get("action") == "finish"), None)
        if finish_step:
            answer = finish_step.get("action_input", "")
            if answer and answer != "PLACEHOLDER":
                answer_nums = set(re.findall(r'\d+\.?\d*', answer))
                obs_text = " ".join(
                    s.get("observation", "") for s in steps
                    if s.get("observation") and s.get("action") != "finish"
                )
                obs_nums = set(re.findall(r'\d+\.?\d*', obs_text))
                if answer_nums:
                    coverage = len(answer_nums & obs_nums) / len(answer_nums)
                    if coverage < 0.5:
                        errors.append(f"CONSISTENCY: 答案数字溯源率仅 {coverage:.0%}")

    # 5. 步数合理性
    num_steps = len(steps)
    if num_steps < 1:
        errors.append(f"STEPS: 步数过少({num_steps})")
    if num_steps > 6:
        errors.append(f"STEPS: 步数过多({num_steps})")

    # 6. 检索质量
    if not sample.get("retrieval_quality", True):
        errors.append("RETRIEVAL: 检索返回结果相关性低")

    # 7. calculate 执行验证
    for i, step in enumerate(steps):
        if step.get("action") != "calculate":
            continue
        expr = step.get("action_input", "")
        calc_nums = set(re.findall(r'\d+\.?\d*', expr))
        if not calc_nums:
            errors.append(f"CALC: 第{i+1}步 calculate 表达式无数字")
            continue
        prior_obs = " ".join(
            s.get("observation", "") for s in steps[:i]
            if s.get("observation")
        )
        prior_nums = set(re.findall(r'\d+\.?\d*', prior_obs))
        unmatched = calc_nums - prior_nums
        # 允许公式常数：100（百分比转小数）、2、1 等
        CALC_CONSTANTS = {"100", "1000", "10000"}
        significant_unmatched = {n for n in unmatched if float(n) > 10 and n not in CALC_CONSTANTS}
        if significant_unmatched:
            errors.append(f"CALC: 第{i+1}步数字 {significant_unmatched} 未在前序 Observation 中出现")

    return len(errors) == 0, errors


# ============ 格式转换 ============

def format_as_sft_sample(plan: dict) -> dict:
    """将完成的 plan 转换为 SFT 训练格式"""
    trajectory_parts = []
    for step in plan["steps"]:
        trajectory_parts.append(f"Thought: {step['thought']}")
        trajectory_parts.append(f"Action: {step['action']}")
        trajectory_parts.append(f"Action Input: {step['action_input']}")
        if step.get("observation"):
            trajectory_parts.append(f"Observation: {step['observation']}")

    trajectory_text = "\n".join(trajectory_parts)

    return {
        "question": plan["question"],
        "type": plan["type"],
        "steps": plan["steps"],
        "trajectory_text": trajectory_text,
        "num_steps": len(plan["steps"]),
        "tools_used": [s["action"] for s in plan["steps"]],
    }


# ============ 断点续传 ============

def save_checkpoint(results: list, stats: dict):
    with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"results": results, "stats": stats}, f, ensure_ascii=False, indent=2)
    logger.info(f"[checkpoint] 已保存 {len(results)} 条数据")


def load_checkpoint() -> tuple:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"[断点续传] 已有 {len(data['results'])} 条数据")
        return data["results"], data["stats"]
    return [], {}


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="FinAgent SFT 数据生成 V2（ReAct Loop）")
    parser.add_argument("--test", type=int, default=0, help="测试模式，只生成N条")
    parser.add_argument("--type", type=str, default="", help="只生成指定类型")
    parser.add_argument("--resume", action="store_true", help="从断点续传")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("FinAgent SFT 数据生成 V2（ReAct Loop）开始")
    logger.info(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info("=" * 60)

    # 初始化
    client = OpenAI()
    logger.info("[1/4] 初始化 OpenAI 客户端完成")

    retriever = FinAgentRetriever()
    retriever.load_index()
    tools = FinAgentTools(retriever)
    logger.info("[2/4] 加载检索索引完成")

    seed_grouped = load_seed_data(SEED_DATA_PATH)
    logger.info(f"[3/4] 加载种子数据: {', '.join(f'{k}={len(v)}' for k, v in seed_grouped.items())}")

    # 加载 question
    questions = load_questions()
    if args.type:
        questions = [q for q in questions if q["type"] == args.type]
    if args.test > 0:
        questions = questions[:args.test]
    logger.info(f"[4/4] 待生成: {len(questions)} 条")

    # 断点续传
    results = []
    stats = {"total": 0, "success": 0, "failed_react": 0,
             "failed_validation": 0, "low_retrieval": 0}
    if args.resume:
        results, stats = load_checkpoint()

    start_idx = len(results)

    # 随机打乱（但保持可复现）
    random.shuffle(questions)

    # 主循环
    for idx, task in enumerate(questions[start_idx:], start=start_idx):
        question = task["question"]
        qtype = task["type"]
        stats["total"] += 1

        if idx % 10 == 0:
            logger.info(f"--- 进度: {idx}/{len(questions)} | 成功: {stats['success']} | "
                        f"失败: {stats['failed_react'] + stats['failed_validation']} ---")

        try:
            # Phase 0: 预检索
            pre_info = pre_retrieve(question, retriever)

            # Phase 1: ReAct Loop
            plan = react_loop(
                client, tools, question, qtype, pre_info, seed_grouped
            )

            if not plan:
                stats["failed_react"] += 1
                logger.warning(f"[{idx}] ReAct Loop 失败: {question[:50]}")
                continue

            # 检索质量检查
            if not plan.get("retrieval_quality", True) and qtype != "reject":
                stats["low_retrieval"] += 1
                logger.warning(f"[{idx}] 检索质量低: {question[:50]}")
                # 仍然保存，但标记

            # Phase 2: 验证
            passed, validation_errors = validate_sample(plan)
            if not passed:
                stats["failed_validation"] += 1
                logger.warning(f"[{idx}] 验证失败: {validation_errors}")
                plan["validation_errors"] = validation_errors
                plan["validation_passed"] = False
            else:
                plan["validation_passed"] = True
                plan["validation_errors"] = []

            # 格式化并保存
            sft_sample = format_as_sft_sample(plan)
            sft_sample["validation_passed"] = plan["validation_passed"]
            sft_sample["validation_errors"] = plan.get("validation_errors", [])

            results.append(sft_sample)
            stats["success"] += 1

            # 每 20 条保存一次断点
            if len(results) % 20 == 0:
                save_checkpoint(results, stats)

            # API 限频
            time.sleep(0.3)

        except Exception as e:
            logger.error(f"[{idx}] 未预期错误: {e}")
            stats["failed_react"] += 1
            continue

    # ============ 最终保存 ============

    with open(FINAL_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for sample in results:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # 统计报告
    type_dist = {}
    tool_dist = {}
    step_dist = {}
    for sample in results:
        t = sample.get("type", "unknown")
        type_dist[t] = type_dist.get(t, 0) + 1
        for tool in sample.get("tools_used", []):
            tool_dist[tool] = tool_dist.get(tool, 0) + 1
        n = sample.get("num_steps", 0)
        step_dist[str(n)] = step_dist.get(str(n), 0) + 1

    validation_passed = sum(1 for s in results if s.get("validation_passed", False))

    final_stats = {
        **stats,
        "type_distribution": type_dist,
        "tool_distribution": tool_dist,
        "step_distribution": step_dist,
        "validation_pass_rate": f"{validation_passed}/{len(results)} ({validation_passed/max(len(results),1):.1%})",
        "timestamp": datetime.now().isoformat(),
    }

    stats_path = os.path.join(OUTPUT_DIR, "generation_stats_v2.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("SFT 数据生成 V2 完成")
    logger.info(f"总数: {len(results)} 条")
    logger.info(f"验证通过: {validation_passed} ({validation_passed/max(len(results),1):.1%})")
    logger.info(f"类型分布: {type_dist}")
    logger.info(f"工具分布: {tool_dist}")
    logger.info(f"步数分布: {step_dist}")
    logger.info(f"输出路径: {FINAL_OUTPUT_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
