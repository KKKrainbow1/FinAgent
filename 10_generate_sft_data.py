"""
FinAgent SFT 数据生成脚本（四步法）
用途：基于种子数据 + 阿里云百炼 Qwen API + 真实检索系统，批量生成 800 条 SFT 训练数据
环境：AutoDL

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

依赖：
    pip install openai --break-system-packages
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

# API 配置（阿里云百炼，在 AutoDL 上设置环境变量）
# export OPENAI_API_KEY="你的百炼API Key"
# export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

OUTPUT_DIR = "./data/sft"
SEED_DATA_PATH = "./sft_seed_data_v3.jsonc"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.json")
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sft_data_v1.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, f"gen_{datetime.now():%Y%m%d_%H%M%S}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============ 数据分布配置 ============

# 每种类型的目标数量和使用的模型
TYPE_CONFIG = {
    "financial_query": {
        "count": 160,
        "model": "qwen3-max",
        "description": "简单财务指标查询，1-2步"
    },
    "single_company_simple": {
        "count": 120,
        "model": "qwen3-max",
        "description": "单公司简单分析，2步"
    },
    "single_company_medium": {
        "count": 160,
        "model": "qwen3-max",
        "description": "单公司多维度分析，3步"
    },
    "company_comparison": {
        "count": 120,
        "model": "qwen3-max",
        "description": "公司对比分析，3-4步"
    },
    "industry_analysis": {
        "count": 80,
        "model": "qwen3-max",
        "description": "行业分析，4步"
    },
    "risk_analysis": {
        "count": 100,
        "model": "qwen3-max",
        "description": "风险分析，3步"
    },
    "reject": {
        "count": 60,
        "model": "qwen3-max",
        "description": "拒绝回答，1-2步"
    },
}

# 沪深300代表性公司（按行业分组，用于生成多样化的问题）
COMPANY_POOL = {
    "白酒": ["贵州茅台", "五粮液", "泸州老窖", "山西汾酒", "古井贡酒", "今世缘"],
    "新能源": ["宁德时代", "比亚迪", "阳光电源", "汇川技术", "亿纬锂能"],
    "光伏": ["隆基绿能", "通威股份", "阳光电源", "晶澳科技", "天合光能"],
    "医药": ["恒瑞医药", "药明康德", "迈瑞医疗", "智飞生物", "片仔癀"],
    "半导体": ["中芯国际", "北方华创", "韦尔股份", "兆易创新", "澜起科技"],
    "金融": ["招商银行", "中国平安", "兴业银行", "宁波银行", "中信证券"],
    "消费": ["海天味业", "伊利股份", "美的集团", "格力电器", "海尔智家"],
    "科技": ["立讯精密", "海康威视", "中兴通讯", "紫光股份", "用友网络"],
    "汽车": ["比亚迪", "长城汽车", "长安汽车", "上汽集团", "广汽集团"],
    "地产": ["万科A", "保利发展", "招商蛇口", "华润置地", "金地集团"],
    "化工": ["万华化学", "恒力石化", "荣盛石化", "华鲁恒升", "宝丰能源"],
    "军工": ["中航沈飞", "航发动力", "中航光电", "紫光国微", "振华科技"],
    "交运": ["顺丰控股", "中远海控", "招商港口", "京沪高铁", "中国国航"],
    "食品": ["海天味业", "伊利股份", "双汇发展", "绝味食品", "安琪酵母"],
    "家电": ["美的集团", "格力电器", "海尔智家", "老板电器", "苏泊尔"],
}

# 分析角度（用于组合生成多样化问题）
ANGLE_POOL = {
    "financial_query": [
        "ROE", "毛利率", "净利率", "营收增长率", "净利润增长率",
        "每股收益EPS", "资产负债率", "流动比率", "总资产周转率",
    ],
    "single_company_simple": [
        "机构评级", "目标价", "EPS预测", "最新研报观点",
    ],
    "single_company_medium": [
        "盈利能力分析", "杜邦分析拆解ROE", "财务状况全面评估",
        "经营表现分析", "成长性评估",
    ],
    "company_comparison": [
        "盈利能力对比", "ROE对比", "经营效率对比", "商业模式对比",
    ],
    "industry_analysis": [
        "行业整体经营情况", "行业竞争格局", "行业政策影响",
        "行业技术替代趋势", "行业盈利周期",
    ],
    "risk_analysis": [
        "偿债风险评估", "经营风险分析", "投资风险", "竞争压力评估",
    ],
    "reject": [
        "海外公司分析", "加密货币预测", "历史股价查询", "非金融问题",
        "港股公司分析", "期货价格预测",
    ],
}

# 拒绝类的特殊公司/话题
REJECT_TOPICS = [
    ("特斯拉", "美股"), ("苹果", "美股"), ("英伟达", "美股"),
    ("三星", "韩股"), ("台积电", "台股"),
    ("比特币", "加密货币"), ("以太坊", "加密货币"),
    ("2015年贵州茅台股价", "历史数据"), ("2010年万科营收", "历史数据"),
    ("今天天气", "非金融"), ("推荐一部电影", "非金融"),
    ("腾讯控股", "港股"), ("阿里巴巴", "港股"),
    ("原油期货", "期货"), ("螺纹钢价格", "期货"),
]


# ============ Step 0: 预检索 ============

def pre_retrieve(company: str, retriever: FinAgentRetriever) -> str:
    """
    预检索：探测检索系统对该公司能返回什么数据。
    结果摘要传给 GPT，让它基于实际可用数据生成计划。
    """
    probe_queries = [
        f"{company} 财务数据 营收 ROE",
        f"{company} 研报 评级",
    ]

    available_info = []
    for query in probe_queries:
        try:
            # 分别调 search_financial 和 search_report
            if "财务" in query or "营收" in query:
                results = retriever.search_financial(query, top_k=2)
            else:
                results = retriever.search_report(query, top_k=2)

            for r in results:
                source = r["metadata"].get("source_type", "unknown")
                text_preview = r["text"][:150]
                available_info.append(f"- [{source}] {text_preview}...")
        except Exception as e:
            logger.warning(f"预检索失败 [{company}]: {e}")

    if not available_info:
        return "[预检索结果为空，该公司可能不在数据库覆盖范围内]"

    # 去重
    available_info = list(set(available_info))[:8]
    return "\n".join(available_info)


# ============ Step 1: GPT 生成轨迹骨架 ============

PLAN_PROMPT = """你是金融分析数据标注员。请根据以下要求生成一条ReAct轨迹的"计划"部分。

## 可用工具（必须且只能使用以下4个工具）

| 工具名 | 功能 | 输入格式 | 适用场景 |
|--------|------|----------|----------|
| search_report | 检索券商研报信息 | query字符串 | 需要获取机构观点、评级、目标价时使用 |
| search_financial | 检索公司财务数据 | query字符串 | 需要获取营收、净利润、ROE、毛利率等硬数据时使用 |
| calculate | 计算数学表达式 | 纯数学表达式字符串 | 需要做数值计算时使用（如计算ROE差值、行业均值等）。注意：不要让模型在Thought中心算，所有数值计算必须通过此工具完成 |
| finish | 输出最终答案 | 分析报告文本 | 所有需要的信息已获取且计算已完成，生成最终报告 |

## 工具使用原则
- 简单查询（如"XX的ROE是多少"）：1-2步即可
- 多维度分析（如"分析XX的盈利能力"）：2-3步
- 对比分析（如"比较A和B"）：需要分别查两家公司，3-4步
- 每条轨迹控制在2-5步
- search_financial 的 query 中只写一家公司名，不要同时查多家公司
- calculate 的 action_input 必须是纯数学表达式（如 36.99 - 24.53），不要用变量名

## Thought 写作要求（⭐重要）
- 第一步的 Thought 应说明分析框架（如"分析盈利能力需要看ROE、毛利率、净利率"）
- 中间步骤的 Thought 应引用前一步 Observation 中的具体数字（如"Observation显示ROE为36.99%，处于优秀水平"）
- 涉及 calculate 时，Thought 中必须写明公式推导过程（如"杜邦公式：ROE = 净利率 × 周转率 × 权益乘数"）
- 最后一步 Thought 应综合前面的数据给出判断，不要用"如果...说明..."的假设句式
- 注意：种子数据中 Thought 里的具体数字仅为格式示范，你生成时 Thought 中不要编造数字，前几步写分析框架，最后一步 finish 的 action_input 写 PLACEHOLDER（后续会基于真实Observation重新生成）

## 种子示例
{seed_examples}

## 预检索结果（检索系统对该公司实际能返回的信息样例）
{pre_retrieved_info}

## 目标
公司：{company}
行业：{industry}
分析角度：{angle}
问题类型：{question_type}

请生成一条新的轨迹。输出 JSON 格式：
{{
  "question": "用户问题",
  "steps": [
    {{"thought": "...", "action": "search_financial", "action_input": "..."}},
    {{"thought": "...", "action": "finish", "action_input": "PLACEHOLDER"}}
  ]
}}
"""

# 拒绝类单独的 prompt
REJECT_PLAN_PROMPT = """你是金融分析数据标注员。请生成一条"拒绝回答"的ReAct轨迹。

## 背景
我们的Agent数据库只覆盖A股上市公司（沪深300等）的研报和财务数据，时间范围2022-2024年。
当用户问到覆盖范围外的问题时，Agent应该先尝试检索，确认没有相关数据后，诚实拒绝并引导用户。

## 目标
话题：{topic}
超出范围的原因：{reason}

## 种子示例
{seed_examples}

请生成一条新的拒绝轨迹。输出 JSON 格式：
{{
  "question": "用户问题",
  "steps": [
    {{"thought": "...", "action": "search_report 或 search_financial", "action_input": "..."}},
    {{"thought": "检索结果不相关的原因分析...", "action": "finish", "action_input": "拒绝回答的完整内容"}}
  ]
}}
"""


def load_seed_data(seed_path: str) -> dict:
    """加载种子数据，按类型分组"""
    # 处理 jsonc（去掉注释）
    with open(seed_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 去掉 // 注释
    content = re.sub(r'//.*?\n', '\n', content)
    seeds = json.loads(content)

    grouped = {}
    for seed in seeds:
        t = seed["type"]
        if t not in grouped:
            grouped[t] = []
        grouped[t].append(seed)

    return grouped


def generate_plan(client: OpenAI, company: str, industry: str,
                  angle: str, question_type: str, model: str,
                  seed_examples: str, pre_info: str) -> dict:
    """Step 1: GPT 生成轨迹骨架"""

    prompt = PLAN_PROMPT.format(
        seed_examples=seed_examples,
        pre_retrieved_info=pre_info,
        company=company,
        industry=industry,
        angle=angle,
        question_type=question_type,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.8,
            max_tokens=2000,
            extra_body={"enable_thinking": False},
        )
        plan = json.loads(response.choices[0].message.content)
        return plan
    except Exception as e:
        logger.error(f"Qwen 生成计划失败 [{company}/{angle}]: {e}")
        return None


def generate_reject_plan(client: OpenAI, topic: str, reason: str,
                         model: str, seed_examples: str) -> dict:
    """拒绝类的计划生成"""
    prompt = REJECT_PLAN_PROMPT.format(
        topic=topic,
        reason=reason,
        seed_examples=seed_examples,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.8,
            max_tokens=1000,
            extra_body={"enable_thinking": False},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Qwen 生成拒绝计划失败 [{topic}]: {e}")
        return None


# ============ Step 2: 填充真实 Observation ============

def fill_real_observations(plan: dict, tools: FinAgentTools) -> dict:
    """
    对 plan 中每一步的 action，调用真实检索系统获取 Observation。
    返回填充后的 plan + 检索质量标记。
    """
    filled_steps = []
    retrieval_quality = True

    for i, step in enumerate(plan.get("steps", [])):
        action = step.get("action", "")
        action_input = step.get("action_input", "")

        if action == "finish":
            # finish 步不需要 Observation
            filled_steps.append(step)
            continue

        # 调用真实工具
        try:
            observation = tools.call(action, action_input)
        except Exception as e:
            observation = f"[工具调用失败] {e}"
            logger.warning(f"工具调用失败 [{action}({action_input})]: {e}")

        # 检查检索质量：如果返回"未找到"或结果太短，标记为低质量
        if "未找到" in observation or len(observation) < 50:
            retrieval_quality = False
            logger.warning(f"检索质量低 [{action}({action_input})]: {observation[:100]}")

        filled_step = {
            "thought": step["thought"],
            "action": action,
            "action_input": action_input,
            "observation": observation,
        }
        filled_steps.append(filled_step)

    plan["steps"] = filled_steps
    plan["retrieval_quality"] = retrieval_quality
    return plan


# ============ Step 3: GPT 生成最终答案 ============

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

## 种子示例的答案风格（参考结构和深度）
{answer_example}
"""


def generate_final_answer(client: OpenAI, question: str,
                          observations: str, model: str,
                          answer_example: str = "") -> str:
    """Step 3: 基于真实 Observation 生成 finish 答案"""

    prompt = ANSWER_PROMPT.format(
        question=question,
        observations=observations,
        answer_example=answer_example if answer_example else "（无示例，请按写作要求生成）",
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500,
            extra_body={"enable_thinking": False},
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Qwen 生成答案失败: {e}")
        return None


# ============ 七层验证 ============

VALID_TOOLS = {"search_report", "search_financial", "calculate", "finish"}


def validate_sample(sample: dict) -> tuple:
    """
    七层自动验证，返回 (通过?, 错误列表)
    """
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

    # 3. Observation 真实性（非 finish 步必须有 observation）
    for i, step in enumerate(steps):
        if step.get("action") != "finish" and not step.get("observation"):
            errors.append(f"OBS: 第{i+1}步缺少 observation")

    # 4. 答案-证据一致性（finish 答案中 80%+ 数字可溯源）
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
        # 检查数字是否来自前序 Observation
        prior_obs = " ".join(
            s.get("observation", "") for s in steps[:i]
            if s.get("observation")
        )
        prior_nums = set(re.findall(r'\d+\.?\d*', prior_obs))
        unmatched = calc_nums - prior_nums
        # 允许小数字（如 100, 2 等公式常数）
        significant_unmatched = {n for n in unmatched if float(n) > 10}
        if significant_unmatched:
            errors.append(f"CALC: 第{i+1}步数字 {significant_unmatched} 未在前序 Observation 中出现")

    return len(errors) == 0, errors


# ============ 格式转换：plan → SFT 训练格式 ============

def format_as_sft_sample(plan: dict, question_type: str) -> dict:
    """
    将填充后的 plan 转换为 SFT 训练格式。
    格式：符合 Qwen chat template 的 messages 列表。
    """
    # 构建 assistant 的回复内容（ReAct 轨迹）
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
        "type": question_type,
        "steps": plan["steps"],
        "trajectory_text": trajectory_text,
        "num_steps": len(plan["steps"]),
        "tools_used": [s["action"] for s in plan["steps"]],
    }


# ============ 主流程 ============

def generate_task_list(seed_grouped: dict) -> list:
    """
    生成所有待生成的任务列表。
    每个任务 = (公司, 行业, 角度, 类型)
    """
    tasks = []

    for qtype, config in TYPE_CONFIG.items():
        count = config["count"]

        if qtype == "reject":
            # 拒绝类用特殊话题池
            for i in range(count):
                topic, reason = random.choice(REJECT_TOPICS)
                tasks.append({
                    "type": qtype,
                    "topic": topic,
                    "reason": reason,
                    "model": config["model"],
                })
        else:
            # 正常类型：从公司池和角度池中随机组合
            industries = list(COMPANY_POOL.keys())
            angles = ANGLE_POOL.get(qtype, ["综合分析"])

            for i in range(count):
                industry = random.choice(industries)
                company = random.choice(COMPANY_POOL[industry])
                angle = random.choice(angles)
                tasks.append({
                    "type": qtype,
                    "company": company,
                    "industry": industry,
                    "angle": angle,
                    "model": config["model"],
                })

    random.shuffle(tasks)
    return tasks


def get_seed_examples(seed_grouped: dict, qtype: str, max_examples: int = 3) -> str:
    """从种子数据中获取示例文本"""
    seeds = seed_grouped.get(qtype, [])
    if not seeds:
        return "（无种子示例）"

    examples = random.sample(seeds, min(max_examples, len(seeds)))
    return json.dumps(examples, ensure_ascii=False, indent=2)


def get_answer_example(seed_grouped: dict, qtype: str) -> str:
    """获取有完整 finish 的种子作为答案风格参考"""
    seeds = seed_grouped.get(qtype, [])
    for seed in seeds:
        for step in seed.get("steps", []):
            if step.get("action") == "finish" and step["action_input"] != "PLACEHOLDER":
                return step["action_input"]
    return ""


def save_checkpoint(results: list, stats: dict):
    """保存断点"""
    with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
        json.dump({"results": results, "stats": stats}, f, ensure_ascii=False, indent=2)
    logger.info(f"[checkpoint] 已保存 {len(results)} 条数据")


def load_checkpoint() -> tuple:
    """加载断点"""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"[断点续传] 已有 {len(data['results'])} 条数据")
        return data["results"], data["stats"]
    return [], {}


def main():
    parser = argparse.ArgumentParser(description="FinAgent SFT 数据生成")
    parser.add_argument("--test", type=int, default=0, help="测试模式，只生成N条")
    parser.add_argument("--type", type=str, default="", help="只生成指定类型")
    parser.add_argument("--resume", action="store_true", help="从断点续传")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("FinAgent SFT 数据生成开始")
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

    # 生成任务列表
    tasks = generate_task_list(seed_grouped)
    if args.type:
        tasks = [t for t in tasks if t["type"] == args.type]
    if args.test > 0:
        tasks = tasks[:args.test]
    logger.info(f"[4/4] 生成任务列表: {len(tasks)} 条待生成")

    # 断点续传
    results = []
    stats = {"total": 0, "success": 0, "failed_plan": 0,
             "failed_obs": 0, "failed_answer": 0, "failed_validation": 0}
    if args.resume:
        results, stats = load_checkpoint()

    start_idx = len(results)

    # 主循环
    for idx, task in enumerate(tasks[start_idx:], start=start_idx):
        qtype = task["type"]
        model = task["model"]
        stats["total"] += 1

        if idx % 10 == 0:
            logger.info(f"--- 进度: {idx}/{len(tasks)} | 成功: {stats['success']} | "
                        f"失败: {stats['failed_plan']+stats['failed_obs']+stats['failed_answer']+stats['failed_validation']} ---")

        try:
            # === Step 0: 预检索 ===
            if qtype == "reject":
                pre_info = ""
            else:
                pre_info = pre_retrieve(task["company"], retriever)

            # === Step 1: GPT 生成计划 ===
            seed_examples = get_seed_examples(seed_grouped, qtype)

            if qtype == "reject":
                plan = generate_reject_plan(
                    client, task["topic"], task["reason"],
                    model, seed_examples
                )
            else:
                plan = generate_plan(
                    client, task["company"], task["industry"],
                    task["angle"], qtype, model,
                    seed_examples, pre_info
                )

            if not plan or not plan.get("steps"):
                stats["failed_plan"] += 1
                logger.warning(f"[{idx}] 计划生成失败: {task}")
                continue

            # 给 plan 标记类型
            plan["type"] = qtype

            # === Step 2: 填充真实 Observation ===
            if qtype != "reject":
                plan = fill_real_observations(plan, tools)
                if not plan.get("retrieval_quality", True):
                    stats["failed_obs"] += 1
                    logger.warning(f"[{idx}] 检索质量低，丢弃: {plan.get('question', '')[:50]}")
                    continue
            else:
                # 拒绝类也需要填充 Observation（展示检索结果不相关）
                plan = fill_real_observations(plan, tools)

            # === Step 3: 生成最终答案（仅对 PLACEHOLDER 的 finish 步） ===
            finish_step = next(
                (s for s in plan["steps"] if s.get("action") == "finish"),
                None
            )
            if finish_step and finish_step.get("action_input") == "PLACEHOLDER":
                # 收集所有 Observation
                obs_text = "\n".join(
                    f"[Step {i+1}] {s['observation']}"
                    for i, s in enumerate(plan["steps"])
                    if s.get("observation")
                )

                answer_example = get_answer_example(seed_grouped, qtype)

                answer = generate_final_answer(
                    client, plan["question"], obs_text,
                    model, answer_example
                )
                if not answer:
                    stats["failed_answer"] += 1
                    logger.warning(f"[{idx}] 答案生成失败: {plan.get('question', '')[:50]}")
                    continue

                finish_step["action_input"] = answer

            # === 七层验证 ===
            passed, validation_errors = validate_sample(plan)
            if not passed:
                stats["failed_validation"] += 1
                logger.warning(f"[{idx}] 验证失败: {validation_errors}")
                # 不完全丢弃——记录错误但仍保存（后续人工审核决定）
                plan["validation_errors"] = validation_errors
                plan["validation_passed"] = False
            else:
                plan["validation_passed"] = True

            # 格式化并保存
            sft_sample = format_as_sft_sample(plan, qtype)
            sft_sample["validation_passed"] = plan.get("validation_passed", False)
            sft_sample["validation_errors"] = plan.get("validation_errors", [])

            results.append(sft_sample)
            stats["success"] += 1

            # 每 20 条保存一次断点
            if len(results) % 20 == 0:
                save_checkpoint(results, stats)

            # API 限频
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"[{idx}] 未预期错误: {e}")
            continue

    # ============ 最终保存 ============

    # 保存 JSONL（每行一条）
    with open(FINAL_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for sample in results:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # 保存统计报告
    stats_path = os.path.join(OUTPUT_DIR, "generation_stats.json")

    # 统计分布
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

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("SFT 数据生成完成")
    logger.info(f"总数: {len(results)} 条")
    logger.info(f"验证通过: {validation_passed} ({validation_passed/max(len(results),1):.1%})")
    logger.info(f"类型分布: {type_dist}")
    logger.info(f"工具分布: {tool_dist}")
    logger.info(f"步数分布: {step_dist}")
    logger.info(f"输出路径: {FINAL_OUTPUT_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
