"""
FinAgent SFT 数据生成脚本（Final - V1 生成引擎 + V2 输出格式）

设计思路：
  - 生成引擎（V1）：STEP_PROMPT + response_format: json_object + CALC_FILL_PROMPT + ANSWER_PROMPT
    稳定、高质量、成功率 ~95%，已在 801 条数据上验证
  - 输出格式（V2）：OpenAI messages 格式，对齐 Qwen2.5 chat_template
    训练时直接 apply_chat_template(tools=TOOLS_NATIVE) 渲染

  为什么不用纯 V2（原生 tools 参数）？
    Qwen3-Max 用原生 tool calling 时太"聪明"，总想跳过工具直接回答，
    成功率仅 60%。V1 的 response_format: json_object 强制模型输出结构化字段，
    没有"不调工具"的选项，稳定性远高于原生 tool calling。

  为什么不用纯 V1（纯文本轨迹）？
    V1 输出的纯文本 ReAct 格式（Thought:/Action:/Action Input:）不是 Qwen2.5
    预训练时见过的格式，SFT 时模型需要从头学。V2 的 messages 格式对齐 Qwen2.5
    原生 tool calling（<tool_call>/<tool_response>），复用预训练先验。

运行方式：
    python 10_generate_sft_data.py --test 5
    python 10_generate_sft_data.py
    python 10_generate_sft_data.py --type financial_query
    python 10_generate_sft_data.py --resume

环境变量：
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
from tools import FinAgentTools, TOOLS_NATIVE

# ============ 配置 ============

OUTPUT_DIR = "./data/sft"
SEED_DATA_PATH = "./sft_seed_data_v3.jsonc"
QUESTIONS_PATH = "./data/sft/all_questions_v2.jsonl"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint_v3_native.json")
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sft_data_v3_native.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, f"gen_final_{datetime.now():%Y%m%d_%H%M%S}.log")),
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
VALID_TOOLS = {"search_report", "search_financial", "search_industry", "calculate", "finish"}
MAX_RETRY = 3


# ============ V1 生成模板（已验证高质量） ============

STEP_PROMPT = """你是"金融翻译官"，一个专业的A股上市公司分析助手。
你正在逐步分析用户的问题，每次只输出一步（Thought + Action + Action Input）。

## 数据库覆盖范围（⭐生成 query 时必须参考）
- **财务数据**：2022H1 ~ 2025年报（300家A股公司，所有公司有2025H1，约半数有2025年报，其余最新年报为2024年）
- **行业对比数据**：按行业聚合的多家公司核心指标对比表和行业均值（沪深300成分股），用 search_industry 检索
- **券商研报**：2017年 ~ 2026年3月（约39,000篇元数据 + 64,000篇PDF正文chunk）
  - 注意：数据库中存在 2017-2021 年的老研报，评级和目标价已严重过时
  - 除非用户明确询问历史数据，否则应优先检索最近1-2年的研报
- **时效性原则**：
  - 用户问"XX怎么样"/"XX盈利能力" → 检索最新数据（query 中加 "2025" 或 "2024"）
  - 用户问"XX近几年趋势"/"XX变化" → 不限定年份，让检索返回多期数据
  - 用户问"XX目标价"/"XX评级" → query 中加 "2025 2026" 以获取最新研报

## 可用工具
| 工具名 | 功能 | 输入格式 |
|--------|------|----------|
| search_report | 检索券商研报 | query字符串 |
| search_financial | 检索单个公司财务数据 | query字符串 |
| search_industry | 检索行业对比数据（多家公司指标对比表+行业均值） | query字符串 |
| calculate | 数学计算 | 纯数学表达式（如 36.99 - 24.53） |
| finish | 输出最终答案 | 分析报告文本 |

## 工具使用原则
- search_financial 的 query 中只写一家公司名，不要同时查多家
- 涉及行业分析、行业对比时，优先使用 search_industry 获取行业对比数据，而非逐个公司用 search_financial
- search_industry 一次返回同行业多家公司的核心指标和行业均值
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
- 优先使用最新的年报数据（2025年报 > 2024年报 > 2023年报 > 半年报），半年报可作为补充参考但不能替代年报

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
- Thought说要算同比增长率，数据中2024年营收=1505亿，2023年营收=1294亿 → 输出：(1505 - 1294) / 1294 * 100
- Thought说要验算杜邦ROE，数据中净利率=14.47%，周转率=0.49，权益乘数=1.75 → 输出：14.47 / 100 * 0.49 * 1.75 * 100
"""


ANSWER_PROMPT = """你是金融分析师。根据以下检索到的真实数据，生成最终分析报告。

用户问题：{question}

问题类型：{question_type}

已获取的数据（来自真实检索系统）：
{observations}

## 核心原则：回答用户的问题，不要套模板
最终报告应围绕用户的问题展开，只分析与问题相关的内容。

## 按问题类型调整回答深度
- **financial_query**（如"ROE是多少""流动比率多少"）：先直接回答数字，再适度展开相关分析，不要做全维度分析。控制在 150-300 字。
- **single_company_simple**（如"目标价多少""研报观点"）：直接回答核心信息，可补充少量相关背景。控制在 200-350 字。
- **single_company_medium**（如"全面评估财务状况""盈利能力如何"）：按德勤框架选择相关维度深入分析。控制在 400-600 字。
- **company_comparison**：用表格对比核心指标，重点分析差异原因。控制在 400-600 字。
- **risk_analysis**：重点分析偿债能力和风险因素。控制在 400-600 字。
- **industry_analysis**：基于 search_industry 返回的行业对比数据，分析行业整体趋势和公司排名。控制在 400-600 字。
- **reject**：简短说明数据库不覆盖该内容，建议调整问题。控制在 100-200 字。

## 分析方法（德勤财务分析方法论）

### 财务比率分析（根据问题选择相关维度，不需要每次都覆盖全部三个）
1. **盈利能力**：销售毛利率、销售净利率、ROA、ROE、EPS。ROE 应在同行业内对比，不能用绝对阈值判定。
2. **偿债能力**：流动比率（2:1 左右为宜，过高说明资金闲置）、速动比率（1:1 左右）、现金比率（≥20%）、资产负债率（非金融 40-60% 适宜；银行/保险/证券 85-95% 为正常水平）、利息支付倍数（≥1）。
3. **营运能力**：总资产周转率、存货周转率、应收账款周转率。周转率因行业而异。

### 杜邦分析法（适用于深度分析 ROE 驱动因素，不要在简单查询中使用）
- ROE = 销售净利率 × 总资产周转率 × 权益乘数
- 三个因子必须来自同一期数据

## 写作要求
1. 所有数字必须来自上述数据，不能编造
2. 与问题相关的关键指标附带判断和行业对比参考，不相关的不要展开
3. 如果某个维度的数据不足，不要提及该维度（不要写"数据有限，无法评估XX"）
4. 风险提示需具体（包含数字或事件），简单查询（financial_query、single_company_simple）不需要强制添加风险提示
5. 金融行业（银行/保险/证券）无存货周转率、毛利率等指标，应解释行业特性而非说"数据缺失"

## 时间一致性要求
- 杜邦分析的三个指标必须来自同一期数据（同一年份 + 同一报告类型）
- 如果数据中有多个年份，选择最新的年报数据进行主要分析
- 不要在同一段分析中混用年报和半年报的数据
- 明确标注数据来源期别，如"根据2024年年报数据"

## 种子示例的答案风格（参考结构和深度）
{answer_example}
"""


# ============ Question 加载 ============

def load_questions() -> list:
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


# ============ 预检索 ============

def pre_retrieve(question: str, retriever: FinAgentRetriever,
                 question_type: str = "") -> str:
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
            # 行业分析类问题额外预检索行业数据
            if question_type in ("industry_analysis", "company_comparison"):
                results_i = retriever.search_industry(query, top_k=1)
                for r in results_i:
                    industry = r["metadata"].get("industry", "未知")
                    text_preview = r["text"][:300]
                    available_info.append(f"- [industry | {industry}] {text_preview}...")
        except Exception as e:
            logger.warning(f"预检索失败: {e}")
    if not available_info:
        return "[预检索结果为空]"
    available_info = list(set(available_info))[:8]
    return "\n".join(available_info)


# ============ 种子数据 ============

def load_seed_data(seed_path: str) -> dict:
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
    seeds = seed_grouped.get(qtype, [])
    for seed in seeds:
        for step in seed.get("steps", []):
            if step.get("action") == "finish" and step["action_input"] != "PLACEHOLDER":
                return step["action_input"]
    return ""


# ============ V1 生成引擎 ============

def format_history(history: list) -> str:
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
    """调用 LLM 生成当前步的 Thought + Action + Action Input（V1 方式）"""
    prompt = STEP_PROMPT.format(
        pre_retrieved_info=pre_info,
        question=question,
        question_type=question_type,
        history=format_history(history),
        step_num=step_num + 1,
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
    """Calculate 专用路径：CALC_FILL_PROMPT + 正则校验 + eval"""
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

    expr = expr.replace("`", "").strip()
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
    """生成 finish 步的最终答案（V1 ANSWER_PROMPT）"""
    obs_parts = []
    for i, step in enumerate(history):
        if step.get("observation"):
            obs_parts.append(f"[Step {i+1}] {step['observation']}")
    obs_text = "\n".join(obs_parts)

    answer_example = get_answer_example(seed_grouped, qtype)

    prompt = ANSWER_PROMPT.format(
        question=question,
        question_type=qtype,
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
               question_type: str, pre_info: str, seed_grouped: dict) -> dict:
    """V1 生成引擎：逐步生成 Thought/Action，执行 Action 获取 Observation"""
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

            if action not in VALID_TOOLS:
                logger.warning(f"非法工具 '{action}'，重试")
                retry_count += 1
                continue

            if step_num == 0 and action == "finish":
                logger.warning("首步试图 finish，重试")
                retry_count += 1
                continue

            if (step_num + 1) < min_steps and action == "finish":
                logger.warning(f"步数不足 ({step_num + 1} < {min_steps})，禁止 finish，重试")
                retry_count += 1
                continue

            break

        if step_result is None or retry_count >= MAX_RETRY:
            logger.error(f"步骤生成失败，已重试 {MAX_RETRY} 次")
            return None

        thought = step_result.get("thought", "")
        action = step_result.get("action", "")
        action_input = step_result.get("action_input", "")

        # 2. 分支执行
        if action == "finish":
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
            calc_result = fill_calculate(client, question, history, thought)
            if calc_result is None:
                logger.warning(f"calculate 失败，跳过")
                continue
            expr, observation = calc_result
            history.append({
                "thought": thought,
                "action": "calculate",
                "action_input": expr,
                "observation": observation,
            })

        else:  # search_financial / search_report / search_industry
            try:
                observation = tools.call(action, action_input)
            except Exception as e:
                observation = f"[工具调用失败] {e}"
                logger.warning(f"工具调用失败 [{action}({action_input})]: {e}")

            if "未找到" in observation or len(observation) < 50:
                retrieval_quality = False
                logger.warning(f"检索质量低: {observation[:100]}")

            history.append({
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation,
            })

        logger.info(f"  Step {step_num+1}: {action}({str(action_input)[:60]})")
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


# ============ V2 格式转换层 ============

def format_as_sft_sample(plan: dict) -> dict:
    """
    将 V1 生成的 plan 转换为 V2 messages 格式

    V1 步骤格式：{"thought", "action", "action_input", "observation"}
    V2 messages 格式：assistant(content+tool_calls) + tool(observation)

    关键转换：
    - search/calculate 步 → assistant(content=thought, tool_calls) + tool(observation)
    - finish 步 → assistant(content=thought + \n\n + 报告)
    - action_input 字符串 → JSON arguments
    """
    from prompts import SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": plan["question"]},
    ]

    steps = plan["steps"]
    call_counter = 0
    tools_used = []

    for step in steps:
        action = step["action"]
        thought = step["thought"]
        action_input = step["action_input"]

        if action == "finish":
            # 最终回答：thought + 报告合并为 content
            final_content = f"{thought}\n\n{action_input}" if thought else action_input
            messages.append({
                "role": "assistant",
                "content": final_content,
            })

        else:
            # 工具调用步：转换为 tool_calls 格式
            tool_call_id = f"call_{call_counter}"
            call_counter += 1
            tools_used.append(action)

            # action_input 转为 JSON arguments
            if action in ("search_report", "search_financial", "search_industry"):
                arguments = json.dumps({"query": action_input}, ensure_ascii=False)
            elif action == "calculate":
                arguments = json.dumps({"expression": action_input}, ensure_ascii=False)
            else:
                arguments = json.dumps({"input": action_input}, ensure_ascii=False)

            # assistant 消息：content=Thought + tool_calls=Action
            assistant_msg = {
                "role": "assistant",
                "content": thought,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": action,
                        "arguments": arguments,
                    }
                }]
            }
            messages.append(assistant_msg)

            # tool 消息：Observation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": step.get("observation", ""),
            })

    return {
        "question": plan["question"],
        "type": plan["type"],
        "messages": messages,
        "num_tool_steps": len(tools_used),
        "tools_used": tools_used,
    }


# ============ 验证 ============

def validate_sample(sample: dict) -> tuple:
    errors = []
    steps = sample.get("steps", [])

    if not sample.get("question"):
        errors.append("FORMAT: 缺少 question")
    if not steps:
        errors.append("FORMAT: 缺少 steps")
    if not any(s.get("action") == "finish" for s in steps):
        errors.append("FORMAT: 缺少 finish 步骤")

    for i, step in enumerate(steps):
        action = step.get("action", "")
        if action not in VALID_TOOLS:
            errors.append(f"TOOL: 第{i+1}步使用非法工具 '{action}'")

    for i, step in enumerate(steps):
        if step.get("action") != "finish" and not step.get("observation"):
            errors.append(f"OBS: 第{i+1}步缺少 observation")

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

    num_steps = len(steps)
    if num_steps < 1:
        errors.append(f"STEPS: 步数过少({num_steps})")
    if num_steps > 6:
        errors.append(f"STEPS: 步数过多({num_steps})")

    if not sample.get("retrieval_quality", True):
        errors.append("RETRIEVAL: 检索返回结果相关性低")

    for i, step in enumerate(steps):
        if step.get("action") != "calculate":
            continue
        expr = step.get("action_input", "")
        calc_nums = set(re.findall(r'\d+\.?\d*', expr))
        if not calc_nums:
            errors.append(f"CALC: 第{i+1}步 calculate 表达式无数字")
            continue
        prior_obs = " ".join(
            s.get("observation", "") for s in steps[:i] if s.get("observation")
        )
        prior_nums = set(re.findall(r'\d+\.?\d*', prior_obs))
        unmatched = calc_nums - prior_nums
        CALC_CONSTANTS = {"100", "1000", "10000"}
        significant_unmatched = {n for n in unmatched if float(n) > 10 and n not in CALC_CONSTANTS}
        if significant_unmatched:
            errors.append(f"CALC: 第{i+1}步数字 {significant_unmatched} 未在前序 Observation 中出现")

    return len(errors) == 0, errors


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
    parser = argparse.ArgumentParser(description="FinAgent SFT 数据生成（Final: V1引擎+V2格式）")
    parser.add_argument("--test", type=int, default=0, help="测试模式，只生成N条")
    parser.add_argument("--type", type=str, default="", help="只生成指定类型")
    parser.add_argument("--resume", action="store_true", help="从断点续传")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("FinAgent SFT 数据生成（Final: V1引擎+V2格式）")
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
            pre_info = pre_retrieve(question, retriever, qtype)

            # Phase 1: V1 ReAct Loop 生成
            plan = react_loop(client, tools, question, qtype, pre_info, seed_grouped)

            if not plan:
                stats["failed_react"] += 1
                logger.warning(f"[{idx}] ReAct Loop 失败: {question[:50]}")
                continue

            if not plan.get("retrieval_quality", True) and qtype != "reject":
                stats["low_retrieval"] += 1

            # Phase 2: 验证（V1 格式验证）
            passed, validation_errors = validate_sample(plan)
            if not passed:
                stats["failed_validation"] += 1
                logger.warning(f"[{idx}] 验证失败: {validation_errors}")
                plan["validation_errors"] = validation_errors
                plan["validation_passed"] = False
            else:
                plan["validation_passed"] = True
                plan["validation_errors"] = []

            # Phase 3: V2 格式转换
            sft_sample = format_as_sft_sample(plan)
            sft_sample["validation_passed"] = plan["validation_passed"]
            sft_sample["validation_errors"] = plan.get("validation_errors", [])

            results.append(sft_sample)
            stats["success"] += 1

            if len(results) % 20 == 0:
                save_checkpoint(results, stats)

            time.sleep(0.3)

        except Exception as e:
            logger.error(f"[{idx}] 未预期错误: {e}")
            stats["failed_react"] += 1
            continue

    # 最终保存
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
        n = sample.get("num_tool_steps", 0)
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

    stats_path = os.path.join(OUTPUT_DIR, "generation_stats_final.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("SFT 数据生成完成（Final: V1引擎+V2格式）")
    logger.info(f"总数: {len(results)} 条")
    logger.info(f"验证通过: {validation_passed} ({validation_passed/max(len(results),1):.1%})")
    logger.info(f"类型分布: {type_dist}")
    logger.info(f"工具分布: {tool_dist}")
    logger.info(f"步数分布: {step_dist}")
    logger.info(f"输出路径: {FINAL_OUTPUT_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
