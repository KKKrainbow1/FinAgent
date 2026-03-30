"""
FinAgent SFT 数据生成脚本（V2 Native Tool Calling 版）

V1 → V2 核心改动：
  1. 使用 Qwen3-Max 原生 tools 参数生成数据，不再使用 response_format: json_object
     - Thought 自然输出在 content 字段
     - Action 自然输出在 tool_calls 字段
     - 格式与 Qwen2.5 chat_template 完全对齐
  2. 移除 STEP_PROMPT / CALC_FILL_PROMPT / ANSWER_PROMPT 三大模板
     - 改为标准 messages + tools 参数，由模型原生决定工具调用
     - calculate 不再需要两步流程（先生成 thought 再生成表达式）
  3. 移除 finish 工具，最终回答是普通 assistant 消息（无 tool_calls）
  4. 输出格式从纯文本轨迹改为 OpenAI messages 格式，训练时直接 apply_chat_template

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
from tools import FinAgentTools, TOOLS_NATIVE, TOOL_NAMES_NATIVE

# ============ 配置 ============

OUTPUT_DIR = "./data/sft"
SEED_DATA_PATH = "./sft_seed_data_v3.jsonc"
QUESTIONS_PATH = "./data/sft/all_questions_v2.jsonl"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint_native.json")
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sft_data_native.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, f"gen_native_{datetime.now():%Y%m%d_%H%M%S}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============ 步数控制 ============

MIN_STEPS = {
    "financial_query": 2,
    "single_company_simple": 2,
    "single_company_medium": 3,
    "company_comparison": 3,
    "industry_analysis": 3,
    "risk_analysis": 3,
    "reject": 2,
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
MAX_RETRY = 3


# ============ System Prompt（数据生成专用） ============

# 与 08_prompts.py 的 SYSTEM_PROMPT 内容一致，但额外加入数据生成的行为引导
DATAGEN_SYSTEM_PROMPT = """你是"金融翻译官"，一个专业的A股上市公司分析助手。你的任务是将复杂的财务数据和研报信息转化为清晰、有依据的分析报告。

## 核心原则
1. **有据可依**：所有结论必须基于检索到的数据，不编造数据
2. **数字精确**：涉及计算时使用 calculate 工具，不要心算
3. **简洁高效**：用最少的步骤获取最关键的信息，避免重复检索
4. **诚实拒绝**：如果检索不到相关数据，诚实说明，不要编造
5. **回答问题**：最终报告应围绕用户的问题展开，只分析与问题相关的维度，不要每次都套用全部框架

## 数据库覆盖范围
- 财务数据：覆盖约300家A股公司，时间范围 2022年半年报 ~ 2024年年报（共6期）
- 券商研报元数据：约39,000篇，时间范围 2017年 ~ 2026年3月，包含评级、EPS预测、目标价等
- 研报PDF正文：约64,000个文本片段，来自券商深度研报的分析论证内容
- 重要提示：数据库中存在部分2017-2021年的老研报，评级和目标价已过时。
  除非用户明确询问历史数据，否则应优先引用2024年之后的最新研报观点。

## 时效性检索原则
- 用户问"XX怎么样"等一般性问题 → 查最新数据（search_financial query 加 "2024"）
- 用户问"近几年趋势/变化" → 不限定年份，让检索返回多期数据做对比
- 用户问"目标价/评级" → search_report query 加 "2025 2026"，获取最新券商观点
- 如果检索返回明显过时的研报（如2017-2019年），在思考中标注其时效性

## 工具使用规范
- search_financial 的 query 中只写一家公司名，不要同时查多家
- 所有数值计算必须通过 calculate 工具完成，禁止心算
- 信息收集完毕后，直接输出最终分析报告（不调用工具）

## 思考过程要求（重要）
每次调用工具前，先输出你的思考过程（作为回复内容），然后调用工具：
- **第一步**：说明分析框架和需要获取什么数据
- **后续步骤**：必须引用上一步返回的至少2个具体数字，说明含义，然后解释下一步动机
  - 好："检索结果显示宁德时代2024年 ROE 为21.3%，毛利率22.4%，但营收增速仅7.8%（较2023年大幅放缓）。盈利能力强但增长乏力，需检索券商研报了解市场对其未来增长的预期。"
  - 坏："已获取数据，接下来查研报。"（禁止）
- **最终回答**：综合所有数据给出明确判断，使用断言式（"ROE为21.3%"），不要用假设式

## 时间一致性要求
- 如果检索返回了多个年份的数据，在思考中明确选择使用哪一期
- 杜邦分析的三个指标必须来自同一期数据
- 不要在同一段分析中混用年报和半年报数据
- 优先使用最新的年报数据（2024年报 > 2023年报 > 半年报）

## 分析方法（德勤财务分析方法论）
根据用户问题选择合适的分析方法和维度，不需要每次覆盖全部内容：
- 用户问盈利相关 → 重点分析盈利能力
- 用户问风险/债务相关 → 重点分析偿债能力
- 用户问经营效率相关 → 重点分析营运能力
- 用户要求全面评估 → 三个维度综合分析
- 用户问简单事实（"ROE是多少"） → 直接回答，适度展开即可

### 财务比率分析（三维度）
1. 盈利能力：毛利率、净利率、ROA、ROE、EPS。ROE 应同行业对比。
2. 偿债能力：流动比率(2:1)、速动比率(1:1)、资产负债率(非金融40-60%，金融85-95%)。
3. 营运能力：总资产周转率、存货周转率、应收账款周转率。

### 杜邦分析法
- ROE = 销售净利率 × 总资产周转率 × 权益乘数（三因子同期数据）

### 行业适配
- 金融行业（银行/保险/证券）：无存货周转率、毛利率等指标，应解释行业特性
- 风险提示需具体（包含数字或事件），简单查询不需要强制添加

## 按问题类型调整回答深度
- financial_query（如"ROE是多少"）：先直接回答数字，适度展开。150-300字。
- single_company_simple（如"目标价多少"）：直接回答核心信息。200-350字。
- single_company_medium（如"全面评估财务状况"）：选择相关维度深入分析。400-600字。
- company_comparison：用表格对比核心指标，重点分析差异原因。400-600字。
- risk_analysis：重点分析偿债能力和风险因素。400-600字。
- industry_analysis：选取代表性公司，分析行业趋势。400-600字。
- reject：简短说明数据库不覆盖该内容。100-200字。"""


# ============ Question 加载 ============

def load_questions() -> list:
    """
    从预处理好的 all_questions_v2.jsonl 加载所有 question。
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


# ============ 预检索 ============

def pre_retrieve(question: str, retriever: FinAgentRetriever) -> str:
    """预检索：探测检索系统能返回什么数据"""
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


# ============ ReAct Loop 核心（V2 Native Tool Calling） ============

def generate_step_hint(step_num: int, min_steps: int, max_steps: int) -> str:
    """生成步数提示，注入到 user 消息中引导模型行为"""
    remaining_min = max(0, min_steps - step_num)
    remaining_max = max_steps - step_num

    if remaining_min > 1:
        return f"[提示：还需要至少 {remaining_min} 步检索数据，请继续调用工具，不要直接输出最终回答。]"
    elif remaining_max <= 1:
        return "[提示：这是最后一步，请直接输出最终分析报告，不要再调用工具。]"
    else:
        return "[提示：如果信息已足够，可以直接输出最终分析报告；否则继续调用工具检索。]"


def react_loop(client: OpenAI, tools_executor: FinAgentTools,
               question: str, question_type: str, pre_info: str) -> dict:
    """
    核心 ReAct Loop（V2 - 原生 Tool Calling）

    使用 Qwen3-Max 的 tools 参数生成数据：
    - 模型天然在 content 中输出 Thought，在 tool_calls 中输出 Action
    - 不再需要 STEP_PROMPT / CALC_FILL_PROMPT 模板
    - calculate 的 expression 直接在 tool_calls.arguments 中生成

    Returns:
        完成的 plan dict，或 None 表示失败
    """
    min_steps = MIN_STEPS.get(question_type, 2)
    max_steps = MAX_STEPS.get(question_type, 4)

    # 构建初始 messages
    # 将预检索信息和步数提示放入 user 消息
    user_content = question
    if pre_info and pre_info != "[预检索结果为空]":
        user_content += f"\n\n[系统提示：数据库中有以下相关数据可供检索]\n{pre_info}"

    messages = [
        {"role": "system", "content": DATAGEN_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    steps = []  # 记录每一步的结构化数据
    retrieval_quality = True
    call_counter = 0
    no_tool_retries = 0  # 模型拒绝调用工具的连续次数

    for step_num in range(max_steps):
        # 添加步数引导（作为额外的 user 消息）
        # 工具调用步数 = 实际执行了工具的步数，不是循环计数
        tool_steps_done = len([s for s in steps if "tool_name" in s])
        hint = generate_step_hint(tool_steps_done + 1, min_steps, max_steps)
        hint_messages = messages.copy()
        # 始终添加步数提示（第一步也需要引导模型调用工具）
        hint_messages.append({"role": "user", "content": hint})

        # 调用 Qwen3-Max（原生 tool calling）
        retry_count = 0
        response = None

        while retry_count < MAX_RETRY:
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=hint_messages,
                    tools=TOOLS_NATIVE,
                    temperature=0.7,
                    max_tokens=1500,
                    extra_body={"enable_thinking": False},
                )
                break
            except Exception as e:
                logger.warning(f"API 调用失败 (retry {retry_count}): {e}")
                retry_count += 1
                time.sleep(1)

        if response is None:
            logger.error(f"API 调用失败，已重试 {MAX_RETRY} 次")
            return None

        msg = response.choices[0].message

        # 判断模型行为
        if msg.tool_calls:
            # 模型选择调用工具
            tc = msg.tool_calls[0]
            tool_name = tc.function.name
            tool_call_id = tc.id or f"call_{call_counter}"
            call_counter += 1

            # 解析参数
            try:
                tool_arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.warning(f"tool arguments 解析失败: {tc.function.arguments}")
                retry_count += 1
                continue

            # 校验工具名
            if tool_name not in TOOL_NAMES_NATIVE:
                logger.warning(f"非法工具 '{tool_name}'，跳过")
                continue

            # 防护：未达最少步数时不允许直接结束（无 tool_calls 才算结束）
            # 原生 tool calling 下模型调用工具就不会结束，所以这里不需要额外防护

            thought = msg.content or ""

            # 执行工具
            observation = tools_executor.call(tool_name, tool_arguments)

            # 检查检索质量
            if tool_name in ("search_report", "search_financial"):
                if "未找到" in observation or len(observation) < 50:
                    retrieval_quality = False
                    logger.warning(f"检索质量低: {observation[:100]}")

            # 追加到 messages
            assistant_msg = {"role": "assistant", "tool_calls": [
                {"id": tool_call_id, "type": "function",
                 "function": {"name": tool_name, "arguments": tc.function.arguments}}
            ]}
            if thought:
                assistant_msg["content"] = thought
            messages.append(assistant_msg)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": observation,
            })

            # 记录步骤
            steps.append({
                "thought": thought,
                "tool_name": tool_name,
                "tool_arguments": tool_arguments,
                "tool_call_id": tool_call_id,
                "observation": observation,
            })

            logger.info(f"  Step {step_num+1}: {tool_name}({json.dumps(tool_arguments, ensure_ascii=False)[:80]})")

        else:
            # 模型没有调用工具 → 输出最终回答
            final_answer = msg.content or ""

            # 防护：未达最少工具调用步数时不允许直接结束
            tool_steps_done = len([s for s in steps if "tool_name" in s])
            if tool_steps_done < min_steps:
                no_tool_retries += 1
                if no_tool_retries >= 3:
                    logger.error(f"模型连续 {no_tool_retries} 次拒绝调用工具，放弃该条数据")
                    return None
                logger.warning(f"工具调用步数不足 ({tool_steps_done} < {min_steps})，第 {no_tool_retries} 次重试")
                messages.append({"role": "user", "content":
                    f"[你还没有调用任何检索工具。请先使用 search_financial 或 search_report 检索数据，"
                    f"至少还需要 {min_steps - tool_steps_done} 步工具调用。"
                    f"不要凭记忆回答，必须基于检索结果。请立即调用工具。]"})
                continue

            # 记录最终回答
            steps.append({
                "thought": "",  # 最终回答的 thought 在 content 开头
                "final_answer": final_answer,
            })

            logger.info(f"  Final: 回答 {len(final_answer)} 字")
            break

        # API 限频
        time.sleep(0.3)

    # 确保有最终回答
    tool_steps_done = len([s for s in steps if "tool_name" in s])
    if tool_steps_done < min_steps:
        logger.error(f"循环结束但工具调用不足 ({tool_steps_done} < {min_steps})，丢弃该条")
        return None

    if not steps or "final_answer" not in steps[-1]:
        logger.warning("循环结束但未生成最终回答，强制请求")
        try:
            messages.append({"role": "user", "content":
                "[请根据已获取的数据，直接输出最终分析报告。]"})
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
                extra_body={"enable_thinking": False},
                # 不传 tools 参数，强制模型输出纯文本
            )
            final_answer = response.choices[0].message.content or ""
            steps.append({
                "thought": "",
                "final_answer": final_answer,
            })
        except Exception as e:
            logger.error(f"强制生成答案失败: {e}")
            return None

    return {
        "question": question,
        "type": question_type,
        "steps": steps,
        "retrieval_quality": retrieval_quality,
    }


# ============ 验证 ============

def validate_sample(sample: dict) -> tuple:
    """验证生成的样本质量，返回 (通过?, 错误列表)"""
    errors = []
    steps = sample.get("steps", [])

    # 1. 格式完整性
    if not sample.get("question"):
        errors.append("FORMAT: 缺少 question")
    if not steps:
        errors.append("FORMAT: 缺少 steps")

    # 2. 必须有最终回答
    has_final = any("final_answer" in s for s in steps)
    if not has_final:
        errors.append("FORMAT: 缺少最终回答")

    # 3. 工具调用步骤验证
    tool_steps = [s for s in steps if "tool_name" in s]
    for i, step in enumerate(tool_steps):
        tool_name = step.get("tool_name", "")
        if tool_name not in TOOL_NAMES_NATIVE:
            errors.append(f"TOOL: 第{i+1}步使用非法工具 '{tool_name}'")
        if not step.get("observation"):
            errors.append(f"OBS: 第{i+1}步缺少 observation")

    # 4. 答案-证据一致性
    qtype = sample.get("type", "")
    if qtype != "reject":
        final_step = next((s for s in steps if "final_answer" in s), None)
        if final_step:
            answer = final_step.get("final_answer", "")
            if answer:
                answer_nums = set(re.findall(r'\d+\.?\d*', answer))
                obs_text = " ".join(
                    s.get("observation", "") for s in tool_steps if s.get("observation")
                )
                obs_nums = set(re.findall(r'\d+\.?\d*', obs_text))
                if answer_nums:
                    coverage = len(answer_nums & obs_nums) / len(answer_nums)
                    if coverage < 0.5:
                        errors.append(f"CONSISTENCY: 答案数字溯源率仅 {coverage:.0%}")

    # 5. 步数合理性
    num_tool_steps = len(tool_steps)
    if num_tool_steps < 1 and qtype != "reject":
        errors.append(f"STEPS: 工具调用步数过少({num_tool_steps})")
    if num_tool_steps > 6:
        errors.append(f"STEPS: 工具调用步数过多({num_tool_steps})")

    # 6. 检索质量
    if not sample.get("retrieval_quality", True):
        errors.append("RETRIEVAL: 检索返回结果相关性低")

    # 7. calculate 验证
    for i, step in enumerate(tool_steps):
        if step.get("tool_name") != "calculate":
            continue
        expr = step.get("tool_arguments", {}).get("expression", "")
        calc_nums = set(re.findall(r'\d+\.?\d*', expr))
        if not calc_nums:
            errors.append(f"CALC: 第{i+1}步 calculate 表达式无数字")
            continue
        prior_obs = " ".join(
            s.get("observation", "") for s in tool_steps[:i] if s.get("observation")
        )
        prior_nums = set(re.findall(r'\d+\.?\d*', prior_obs))
        unmatched = calc_nums - prior_nums
        CALC_CONSTANTS = {"100", "1000", "10000"}
        significant_unmatched = {n for n in unmatched if float(n) > 10 and n not in CALC_CONSTANTS}
        if significant_unmatched:
            errors.append(f"CALC: 第{i+1}步数字 {significant_unmatched} 未在前序 Observation 中出现")

    return len(errors) == 0, errors


# ============ 格式转换 ============

def format_as_sft_sample(plan: dict) -> dict:
    """
    将完成的 plan 转换为 SFT 训练格式（V2 messages 格式）

    输出的 messages 可直接用 tokenizer.apply_chat_template(messages, tools=TOOLS_NATIVE)
    渲染为 Qwen2.5 原生 token 序列。
    """
    from prompts import build_system_message

    messages = [
        build_system_message(),
        {"role": "user", "content": plan["question"]},
    ]

    tool_steps = [s for s in plan["steps"] if "tool_name" in s]
    final_step = next((s for s in plan["steps"] if "final_answer" in s), None)

    for step in tool_steps:
        # assistant 消息：content=Thought + tool_calls=Action
        assistant_msg = {
            "role": "assistant",
            "tool_calls": [{
                "id": step["tool_call_id"],
                "type": "function",
                "function": {
                    "name": step["tool_name"],
                    "arguments": json.dumps(step["tool_arguments"], ensure_ascii=False),
                }
            }]
        }
        if step.get("thought"):
            assistant_msg["content"] = step["thought"]
        messages.append(assistant_msg)

        # tool 消息：Observation
        messages.append({
            "role": "tool",
            "tool_call_id": step["tool_call_id"],
            "content": step["observation"],
        })

    # 最终回答
    if final_step:
        messages.append({
            "role": "assistant",
            "content": final_step["final_answer"],
        })

    tools_used = [s["tool_name"] for s in tool_steps]

    return {
        "question": plan["question"],
        "type": plan["type"],
        "messages": messages,
        "num_tool_steps": len(tool_steps),
        "tools_used": tools_used,
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
    parser = argparse.ArgumentParser(description="FinAgent SFT 数据生成（Native Tool Calling 版）")
    parser.add_argument("--test", type=int, default=0, help="测试模式，只生成N条")
    parser.add_argument("--type", type=str, default="", help="只生成指定类型")
    parser.add_argument("--resume", action="store_true", help="从断点续传")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("FinAgent SFT 数据生成（Native Tool Calling 版）开始")
    logger.info(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info("=" * 60)

    # 初始化
    client = OpenAI()
    logger.info("[1/3] 初始化 OpenAI 客户端完成")

    retriever = FinAgentRetriever()
    retriever.load_index()
    tools_executor = FinAgentTools(retriever)
    logger.info("[2/3] 加载检索索引完成")

    # 加载 question
    questions = load_questions()
    if args.type:
        questions = [q for q in questions if q["type"] == args.type]
    if args.test > 0:
        questions = questions[:args.test]
    logger.info(f"[3/3] 待生成: {len(questions)} 条")

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

            # Phase 1: ReAct Loop（原生 Tool Calling）
            plan = react_loop(client, tools_executor, question, qtype, pre_info)

            if not plan:
                stats["failed_react"] += 1
                logger.warning(f"[{idx}] ReAct Loop 失败: {question[:50]}")
                continue

            # 检索质量检查
            if not plan.get("retrieval_quality", True) and qtype != "reject":
                stats["low_retrieval"] += 1
                logger.warning(f"[{idx}] 检索质量低: {question[:50]}")

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

            # 格式化为 SFT 样本
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

    stats_path = os.path.join(OUTPUT_DIR, "generation_stats_native.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("SFT 数据生成（Native Tool Calling）完成")
    logger.info(f"总数: {len(results)} 条")
    logger.info(f"验证通过: {validation_passed} ({validation_passed/max(len(results),1):.1%})")
    logger.info(f"类型分布: {type_dist}")
    logger.info(f"工具分布: {tool_dist}")
    logger.info(f"步数分布: {step_dist}")
    logger.info(f"输出路径: {FINAL_OUTPUT_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
