"""
FinAgent SFT 数据生成脚本(V4 - Mode B 原生 tool calling)

============================================================================
 V4 设计思路
============================================================================

Teacher qwen3-max 直接用 OpenAI 原生 tool calling(tools=TOOLS_NATIVE + tool_choice="auto"),
通过 system prompt 硬约束"必须在 content 写 thought"稳定输出 thought + tool_calls。

  生成 = 训练 = 推理 三者统一
    - 生成期 teacher 输出:msg.content(thought) + msg.tool_calls[0](工具调用)
    - 训练期 assistant messages 直接这样存(arguments=dict,对齐 Qwen2.5 chat_template)
    - 推理期模型按同样格式输出
    → 不需要 V1→V2 格式转换层,不需要 arguments string→dict 后处理

  相对 V3 的核心变化
    1. 废弃 STEP_PROMPT + response_format=json_object(JSON mode)
       → SYSTEM_PROMPT_V4 + tools=TOOLS_NATIVE
    2. 废弃 CALC_FILL_PROMPT 二阶段填充
       → Mode B 原生直出 expression + validate + eval(单阶段)
    3. 废弃 ANSWER_PROMPT 单独生成答案
       → 不调工具时的 content 就是最终分析报告
    4. 废弃 plan_to_messages / format_as_sft_sample(V1→V2 转换层)
       → generate_trajectory_v4 直接产 messages,build_sft_sample 只做元数据聚合
    5. 废弃 MIN_STEPS/MAX_STEPS 按 type 硬约束
       → 全局 MAX_STEPS=10,teacher 自主判断 finish 时机
    6. 废弃 pre_retrieve 预检索注入
       → SYSTEM_PROMPT_V4 明确写数据库覆盖范围,teacher 直接按 Level 1/2 reject 判断

  为什么 V3 当初选 JSON mode?
    V3 观察到 qwen3-max 用原生 tool calling 时 content 经常被吞(thought 丢失),
    退回 json_object 强制输出 thought/action/action_input 结构化字段。
    V4 通过 system prompt 强约束"必须在 content 写 thought",实验(n=30)验证
    content 非空率 100%,解决了 V3 当时的问题,可以回到更自然的原生 tool calling。

  历史版本见 backup/sft_data_generation_v3_archived.py

============================================================================
 运行方式
============================================================================

    python 10_generate_sft_data.py --test 5
    python 10_generate_sft_data.py
    python 10_generate_sft_data.py --type financial_query
    python 10_generate_sft_data.py --resume

环境变量:
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
from llm_grounding import llm_grounding_check

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


# ============ 步数控制(V4) ============
#
# V3 按 type 硬编码 MIN_STEPS/MAX_STEPS,导致 single_company_medium 87.7% 集中在 3 步
# → Thought 模板化根因之一。V4 改为全局 max_steps,由 teacher 自主决定 finish 时机
# (tool_choice="auto" 不调工具即 finish,第 10 步强制 tool_choice="none" 兜底)。
MAX_STEPS = 10

MODEL = "qwen3-max"
JUDGE_MODEL = "qwen3-max"
JUDGE_THRESHOLD = 4
MAX_REGEN_ATTEMPTS = 3      # 被 Judge 驳回后最多重新生成几次答案
VALID_TOOLS = {"search_report", "search_financial", "search_industry", "calculate"}   # V4 删除 "finish" 虚拟工具
MAX_RETRY = 3

# V4: calculate expression 校验正则(纯数学表达式)
CALC_EXPR_RE = re.compile(r'^[\d\.\+\-\*\/\(\)\s%]+$')


# ============ V4 Mode B 主 Prompt(原生 tool calling + content 强约束 thought)============

SYSTEM_PROMPT_V4 = """你是"金融翻译官",一个专业的 A 股上市公司分析助手。通过调用工具获取数据,逐步分析,给出带数据支撑的答案。

# 数据库覆盖范围

| 数据源 | 覆盖范围 |
|---|---|
| 财务数据 | 沪深 300 × 4 期(2024H1 / 2024 年报 / 2025H1 全量;2025 年报约 53%/158 家披露)|
| 行业对比 | 30 大类行业聚合(银行/保险/证券/白酒/食品饮料/医药/医疗/汽车/半导体/消费电子/光伏/电池/电力/煤炭/钢铁/有色金属/化工/房地产/建筑建材/家电/软件/通信/传媒/交通运输/军工/机械/农业/石油石化/纺织服装/零售)|
| 研报 | 2024-01-01 ~ 2026-03(硬过滤,~10K 元数据 + 1.4K PDF 正文)|

**不覆盖**:港股 / 美股 / 加密货币 / 期货 / 非金融问题 / 主观荐股 / 财务数据 < 2024H1 / 研报 < 2024-01 / > 2026-03 任何

# 硬约束:必须在 content 写 thought

每次调用工具前,在 assistant message 的 content 字段用自然语言写出 thought。**content 绝对不能为空**。
- 第一步:说明分析框架 + 要获取的数据(2-3 句)
- 中间步:必须引用上一步 observation 中至少 2 个具体数字,说明意义,解释下一步动机
- 最终步(不调工具时):content 直接是分析报告

# 时效性 fallback 链(用户不指定时间 = 问"最新可得")

财务数据:**2025 年报 > 2025H1 > 2024 年报 > 2024H1**(2025 年报缺失则自然 fallback)
研报观点:**2026 年研报 > 2025 年研报**

**答案必须显式标注期别**:"贵州茅台 2025 年报 ROE 31.8%"(✓);"贵州茅台 ROE 31.8%"(✗ 跨期混用)

用户显式指定时间时直接按指定期查,不做 fallback;期别不在库则按 Reject Level 1 拒识。

# 数字溯源硬规则(反编造核心)

最终答案中**每个具体数字必须能在前面任一 tool message 的 observation 中字面找到**(允许 ±1 位四舍五入,如 36.99% → 37%)。注意:calculate 工具返回的计算结果**也属于 observation**,可直接引用。

- 合规:"茅台 2025 年报 ROE 31.8%"(31.8% 在 obs 里);"营收增速 16.31%(由 calculate 算出,1505 和 1294 均在 obs)"
- 违规:"茅台 ROE 约 35%"(obs 只有 36.99%);"茅台员工 5 万人"(obs 没这数字,编造)

**无法溯源的数字一律删除或改为定性描述**("营收增速较快"/"盈利能力强")。禁止用模型先验/经验值填充。

# Edge case 处理

**Observation 多源矛盾**(同一指标在不同 chunk 数字不同):
- 优先级 search_financial(财报底层) > search_report(机构观点)
- thought 显式标注差异;多机构预测分歧列**区间 + 中位数 + 归因**,不取均值当唯一答案

**检索为空 / 数据不足**:至多重试 1 次,仍无果如实标注 "X 数据未检索到",**禁止用模型先验填补**。部分数据足够时给部分答案 + 明确说明缺失维度。

**Reject Level 1 直接拒绝(0 步 tool_call)**:时间越界 / 已知非 A 股 / 非金融 / 主观荐股 / 其他金融品种。即使 query 含数字也不调 calculate。

**Reject Level 2 先搜后拒(1-2 步)**:不确定是否在覆盖范围内(冷门股、曾用名、子公司、别名),先 search_* 尝试检索,无数据后再拒。

# 时间一致性

- 杜邦三因子(净利率/总资产周转率/权益乘数)必须来自同一期
- 不要在同一段分析中混用年报和半年报数据
- 跨期对比(如 2024 vs 2025)必须同时标注两期期别

# 行业适配

金融行业(银行/保险/证券)无毛利率、存货周转率等指标,应解释行业特性而非"数据缺失"。银行用 NIM/拨备/不良率;保险用综合成本率/承保利润;证券用经纪业务收入/自营收益率。
"""


# V4 种子示例参考(按 type 提供 few-shot,可选注入到 user message)
# V3 用 {answer_example} placeholder 塞 ANSWER_PROMPT;V4 不强制,teacher 自主判断
# seed_grouped 仍由 load_seed_data 加载,需要时塞到第一条 user message 末尾


# ============ 生成期专用提示(只在 teacher 生成数据时叠加到 user prompt)============
# 推理时 react_agent 不注入此段,保持训-推 system 一致
# 这段告诉 teacher "按 type 控制答案长度 + 写作格式要求",仅作生成期 hint

GENERATION_HINTS = """
## 生成期写作要求(本提示仅在生成训练数据时使用)
按问题类型控制最终答案长度:
- financial_query: 150-300 字,先答数字再适度展开
- single_company_simple: 200-350 字,直接答核心信息
- single_company_medium: 400-600 字,按德勤框架选相关维度
- company_comparison: 400-600 字,用表格对比
- risk_analysis: 400-600 字,重点分析风险
- industry_analysis: 400-600 字,基于 search_industry 数据
- finance_concept: 200-400 字,纯知识答(0 步 tool_call)
- reject: 100-200 字,简短拒答 + 建议调整问题

德勤分析方法(深度题用):
- 盈利:毛利率/净利率/ROA/ROE(同行业内对比,不用绝对阈值)
- 偿债:流动比率 ~2:1 / 速动比率 ~1:1 / 资产负债率(非金融 40-60%,金融 85-95%)
- 营运:总资产/存货/应收账款周转率(因行业而异)
- 杜邦:ROE = 净利率 × 总资产周转率 × 权益乘数(三因子同期,深度题用)

写作核心:
1. 数字溯源(见 system 规则)
2. 与问题相关维度展开,不相关不展开
3. 风险提示需具体(含数字或事件),简单查询不强制
4. 数据缺失时不要硬凑,不写"数据有限"等空话
"""


# ============ 内联 LLM-as-Judge ============

JUDGE_PROMPT_INLINE = """你是金融分析质量审核专家。当前日期：2026年4月。请对以下金融分析进行**严格**评审。
注意：数据库中的2024年、2025年数据均为真实已发布的历史数据，不是"未来数据"。你的职责是客观评估质量。评分标准：
- 4分：合格的分析回答，逻辑通顺、回答了问题。这是大部分正常回答应得的分数。
- 5分：优秀的回答，分析深度和精度都超出预期。
- 3分：有明显问题但尚可挽救（过度分析、逻辑小瑕疵）。
- 1-2分：严重错误（逻辑矛盾、答非所问、行业错配）。

注意：**数字准确性 / 心算溯源已由独立的 LLM Grounding 模块审查,本提示不要重复评分**;但以下语义级错配仍需在 D2/D4 严格扣分:
- 时间错位(把 2023 年的数字说成是 2024 年的)
- 行业错配(用银行不该有的指标如银行毛利率)

## 用户问题
{question}

## 问题类型
{question_type}

## 检索到的原始数据（来自真实检索系统）
{observations}

## 模型的分析回答
{answer}

## 评分维度（每个维度 1-5 分，请逐条严格检查）

### D2. 逻辑连贯性
检查分析推理链条是否自洽，有无前后矛盾。
- 5分：逻辑严密，因果关系清晰，结论由数据支撑
- 4分：整体逻辑通顺，无矛盾
- 3分：基本合理，但个别判断缺乏充分依据
- 2分：存在明显矛盾或不合理判断
- 1分：逻辑混乱，前后矛盾严重

**常见扣分点：**
- 杠杆判断矛盾：资产负债率 42%（非金融行业正常范围）却说"杠杆较高"
- 趋势判断矛盾：ROE 从 34% 降到 30% 说"盈利能力持续提升"
- 行业错配：对银行/保险分析存货周转率（金融行业无此指标）
- 时间混用：同一段分析中混用 2023 年报和 2024 半年报数据
- 异常数字不质疑：制造业净利率 79% 不做合理性检查就直接使用

### D3. 问题匹配度
检查回答是否精准回答了用户问题，不多不少。
- 5分：精准回答问题核心，展开恰到好处
- 4分：回答了问题，展开基本合理
- 3分：回答了问题但有明显冗余（如问 ROE 是多少却分析了三个维度）
- 2分：偏离问题，或严重套用"盈利/偿债/营运"三段式模板
- 1分：答非所问

**按问题类型的合理回答标准：**
- 简单查询（如"ROE是多少"）：直接给数字 + 简短展开，150-300字为宜
- 深度分析（如"全面评估财务状况"）：多维度分析，400-600字为宜
- 对比类（如"A和B谁更好"）：需要有明确对比结论
- 行业分析类（如"光伏行业盈利能力"）：应包含多家公司对比和行业均值，400-600字为宜
- 拒绝类（数据库外的问题）：简短说明数据库不覆盖，100-200字为宜

### D4. 分析专业性
检查金融分析方法和行业知识的运用是否正确。
- 5分：使用了恰当的分析框架（如杜邦分析、行业对比），行业知识准确
- 4分：分析框架基本正确，行业判断合理
- 3分：分析方法一般，未使用针对性框架
- 2分：分析框架使用不当（如对简单查询做了完整杜邦分析）
- 1分：金融知识错误（如杜邦三因子不同期、权益乘数计算错误）

**常见扣分点：**
- 杜邦分析的净利率、周转率、权益乘数来自不同报告期
- ROE 对比未考虑行业差异（轻资产 vs 重资产行业 ROE 天然不同）
- 对银行/保险说"毛利率数据缺失"而非解释行业特性
- 风险提示空泛（"存在一定经营风险"），缺乏具体数字支撑

### D5. 工具使用合理性
检查工具调用策略是否高效合理（注意：calculate 心算问题由独立模块检测，此处不需要关注）。
- 5分：工具选择精准，无冗余检索
- 4分：工具选择正确，有轻微冗余但不影响效率
- 3分：工具选择基本合理，但有明显可优化的地方
- 2分：工具选择不当（如查财务数据用了 search_report），或有大量冗余检索
- 1分：完全不调用工具就给出了带具体数字的回答（纯靠模型内部知识编造）

**常见扣分点：**
- 连续两次检索几乎相同的内容（如先搜"茅台研报"又搜"茅台分析师观点"）
- 该查财务数据的查了研报，或反过来
- 行业分析/行业对比类问题没有使用 search_industry，而是逐个公司用 search_financial 检索（效率低）
- 回答中出现了具体数字但没有任何检索步骤

请严格按以下 JSON 格式输出，不要输出其他内容：
```json
{{"d2_coherence": 1-5, "d3_relevance": 1-5, "d4_professionalism": 1-5, "d5_tools": 1-5, "total": 1-5, "issues": ["问题1", "问题2"], "reason": "一句话总评", "regen_type": "answer_only 或 full_trajectory 或 discard"}}
```

评分规则：
- total 是综合评分，不是简单平均，而是你的整体判断
- issues 列出发现的所有具体问题（空数组表示无问题）
- 合格的回答给 4 分，有明显问题的给 3 分，优秀的给 5 分
- regen_type 判断标准（仅在 total < 4 时需要填写，total >= 4 时填 "answer_only" 即可）：
  - "answer_only"：检索数据正确但答案有问题（过度分析、表述不当、排名搞反、数字引用错误等），只需重写答案
  - "full_trajectory"：检索数据本身有问题（返回了错误公司的数据、关键数据缺失、工具选择错误导致答案无法修复），需要重新检索
  - "discard"：问题与数据库严重不匹配（如检索系统完全无法提供有效数据），重新生成也无法改善"""


REGEN_ANSWER_PROMPT = """你是金融分析师。根据以下检索到的真实数据，重新生成最终分析报告。

用户问题：{question}
问题类型：{question_type}

已获取的数据（来自真实检索系统）：
{observations}

## 上次生成的答案被评审驳回，请避免以下问题：
{judge_feedback}

## 按问题类型控制回答深度（⭐最重要）
- financial_query（如"ROE是多少"）：先直接回答数字，简短展开即可，不要做杜邦分析或多维度分析。150-300字。
- single_company_simple（如"目标价多少"）：直接回答核心信息，少量背景。200-350字。
- single_company_medium（如"全面评估财务状况"）：选择相关维度深入分析。400-600字。
- company_comparison：表格对比核心指标，分析差异原因。400-600字。
- risk_analysis：重点分析偿债能力和风险因素。400-600字。
- industry_analysis：基于行业对比数据分析趋势和排名。400-600字。
- reject：简短说明数据库不覆盖。100-200字。

## 写作要求
1. 所有数字必须来自上述数据，不能编造
2. 只分析与问题相关的维度，不要套模板
3. 杜邦分析的三个指标必须来自同一期数据
4. 不要在同一段分析中混用年报和半年报的数据
5. 风险提示需具体（包含数字或事件）
6. 金融行业无存货周转率等指标，应解释行业特性而非说"数据缺失"
7. 做数据排名对比时仔细核对大小关系，不要搞反（如"高于"和"低于"）

请重新生成高质量的分析报告。只输出正式报告，不要输出思考过程。"""


# ============ 规则质检（D1-D10，零成本） ============

# 从 quality_check_v2.py 导入检查函数
import sys as _sys
# 兼容两种目录结构：服务器（代码在根目录）和本地（代码在 finagent_repo/）
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 同级目录
try:
    from quality_check_v2 import (
        check_thought_coherence,    # D1
        check_finish_depth,         # D2
        check_assertive_vs_hypothetical,  # D3
        check_observation_match,    # D4
        check_number_consistency,   # D5
        check_industry_adaptation,  # D6
        check_standard_consistency, # D7
        check_dupont_correctness,   # D8
        check_time_consistency,     # D9
        check_risk_specificity,     # D10
        score_sample,
    )
    QUALITY_CHECK_AVAILABLE = True
except ImportError:
    logger.warning("quality_check_v2.py 导入失败，跳过规则质检")
    QUALITY_CHECK_AVAILABLE = False

# 规则质检严重问题类型：命中则必须重新生成答案
CRITICAL_ISSUE_TYPES = {
    "number_consistency",     # D5: 数字不一致
    "industry_adaptation",    # D6: 行业错配
    "dupont",                 # D8: 杜邦拆解错误
    "time_consistency",       # D9: 时间混用
}

RULE_CHECK_MIN_SCORE = 4.0   # 规则质检最低分

MENTAL_MATH_PROMPT = """你是金融分析质量审核员。请判断以下回答中是否存在"心算"问题。

"心算"定义：回答中出现了需要通过数值计算才能得出的结果，但这个数字在检索数据中并未直接出现，说明模型是自己心算的，而非通过 calculate 工具验算。

## 检索到的原始数据
{observations}

## 最终回答
{answer}

## 判断标准
算心算（返回 true）——回答中的数字在检索数据中找不到，且需要计算才能得出：
- 回答中出现差值（如"下降4.17个百分点"），但检索数据中只有两个原始值，没有差值本身
- 回答中出现增长率/变化率（如"同比增长15.3%"），但检索数据中没有这个增长率，只有两期的原始数据
- 回答中出现由其他指标反推的数字（如从权益乘数推算资产负债率）

不算心算（返回 false）——数字可以直接在检索数据中找到：
- 检索数据中已经包含"营收增长率7.76%"，回答引用"营收增长7.76%"→ 纯引用
- 检索数据中已经包含"ROE 29.90%（上年同期34.07%）"，回答引用这两个数字 → 纯引用
- 检索数据中的杜邦chunk已经算好"杜邦验算ROE≈X%"，回答引用 → 纯引用
- 回答中没有任何需要计算的内容（纯定性分析）

关键：逐一检查回答中的每个数字，看它是否能在检索数据中直接找到。能找到就是引用，找不到且需要计算才能得出就是心算。

请只输出 JSON：{{"has_mental_math": true/false, "evidence": "简述判断依据（20字内）"}}"""


# 缓存 client，避免重复创建（在 main 中赋值）
_mental_math_client = None


def check_mental_math(steps: list) -> list:
    """用 LLM 检测答案中是否存在心算（对照检索数据判断）"""
    global _mental_math_client

    used_calculate = any(s.get("action") == "calculate" for s in steps)

    # 提取 finish 步的答案
    answer = ""
    for step in reversed(steps):
        if step.get("action") == "finish":
            answer = step.get("action_input", "")
            break
    if not answer:
        return []

    # 提取检索数据（Observations）
    observations = extract_obs_from_steps(steps)
    if not observations:
        return []

    prompt = MENTAL_MATH_PROMPT.format(
        observations=observations[:3000],  # 截断避免 token 过长
        answer=answer[:1500],
    )

    if _mental_math_client is None:
        return []  # client 未初始化时跳过

    try:
        response = _mental_math_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=100,
            extra_body={"enable_thinking": False},
        )
        result = json.loads(response.choices[0].message.content)
        if result.get("has_mental_math", False):
            evidence = result.get("evidence", "")
            if used_calculate:
                detail = (
                    f"答案中存在无法从检索结果或 calculate 结果直接溯源的数字"
                    f"（{evidence}），应重新生成完整轨迹"
                )
            else:
                detail = f"答案中存在心算但未使用 calculate 工具（{evidence}），应重新生成完整轨迹"
            return [{
                "type": "mental_math",
                "severity": "critical",
                "detail": detail,
            }]
    except Exception as e:
        logger.warning(f"心算检测 LLM 调用失败: {e}")

    return []


def rule_based_quality_check(plan: dict) -> tuple:
    """
    规则质检：对 V1 plan 跑 D1-D10 检查（零 API 成本）

    Returns:
        (passed: bool, issues: list, score: float)
    """
    if not QUALITY_CHECK_AVAILABLE:
        return True, [], 5.0

    question = plan["question"]
    steps = plan["steps"]

    all_issues = []
    all_issues.extend(check_thought_coherence(steps))
    all_issues.extend(check_finish_depth(steps, question))
    all_issues.extend(check_assertive_vs_hypothetical(steps))
    all_issues.extend(check_observation_match(steps, question))
    all_issues.extend(check_number_consistency(steps))
    all_issues.extend(check_industry_adaptation(steps, question))
    all_issues.extend(check_standard_consistency(steps))
    all_issues.extend(check_dupont_correctness(steps))
    all_issues.extend(check_time_consistency(steps))
    all_issues.extend(check_risk_specificity(steps, question))

    score = score_sample(all_issues)
    has_critical = any(i["type"] in CRITICAL_ISSUE_TYPES for i in all_issues)
    passed = score >= RULE_CHECK_MIN_SCORE and not has_critical

    return passed, all_issues, score


def judge_single_inline(client: OpenAI, question: str, observations: str,
                        answer: str, question_type: str) -> dict:
    """内联 LLM-as-Judge：对单条结果评分"""
    prompt = JUDGE_PROMPT_INLINE.format(
        question=question,
        question_type=question_type,
        observations=observations if observations else "（无检索数据）",
        answer=answer if answer else "（无回答）",
    )

    for retry in range(3):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=800,
                extra_body={"enable_thinking": False},
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            logger.warning(f"Judge 调用失败 (retry {retry}): {e}")
            time.sleep(1)

    return {"total": 0, "issues": ["Judge调用失败"], "reason": "调用失败"}


def regen_answer_with_feedback(client: OpenAI, question: str, question_type: str,
                               observations: str, judge_feedback: str) -> str:
    """根据 Judge 反馈重新生成答案"""
    prompt = REGEN_ANSWER_PROMPT.format(
        question=question,
        question_type=question_type,
        observations=observations,
        judge_feedback=judge_feedback,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1500,
        extra_body={"enable_thinking": False},
    )
    return response.choices[0].message.content


def extract_obs_from_steps(steps: list) -> str:
    """从 V1 steps 中提取 observations"""
    parts = []
    for i, step in enumerate(steps):
        obs = step.get("observation", "")
        if obs:
            parts.append(f"[Step {i+1} - {step.get('action', '')}] {obs}")
    return "\n".join(parts)


def extract_answer_from_steps(steps: list) -> str:
    """从 V1 steps 中提取最终答案"""
    for step in reversed(steps):
        if step.get("action") == "finish":
            return step.get("action_input", "")
    return ""


def judge_and_regen(client: OpenAI, plan: dict, seed_grouped: dict,
                    tools: FinAgentTools = None, retriever: FinAgentRetriever = None) -> dict:
    """
    两关 Judge + 重新生成闭环(V4):
      [第 1 关] LLM Grounding(数字溯源专项,基于 V3 ground truth 同款 prompt)
        - label=1        → regen_type='full_trajectory'(跳过第 2 关)
        - label=0        → 进第 2 关
        - label=uncertain→ 直接丢弃(plan['judge_passed']=False)
        - label=failed   → 直接丢弃
      [第 2 关] LLM Judge(D2-D5,删 D1 数字溯源)
        - 与原版逻辑一致(answer_only / full_trajectory / discard)
    """
    question = plan["question"]
    qtype = plan["type"]
    steps = plan["steps"]
    observations = extract_obs_from_steps(steps)
    answer = extract_answer_from_steps(steps)

    # 记录所有轮次的 Judge 评语
    all_judge_rounds = []

    # ========== 第 1 关:LLM Grounding ==========
    grounding = llm_grounding_check(client, plan, n_samples=1, model=JUDGE_MODEL)
    plan["grounding_check"] = grounding
    g_label = grounding.get('label')
    logger.info(f"    Grounding: label={g_label}, votes={grounding.get('votes')}, reason={grounding.get('reasons')}")

    # uncertain / failed → 直接丢弃,不进第 2 关
    if g_label in ('uncertain', 'failed'):
        logger.info(f"    Grounding label={g_label},丢弃该条")
        plan["judge_score"] = {"total": 0, "reason": f"grounding_{g_label}", "regen_type": "discard"}
        plan["judge_passed"] = False
        plan["judge_all_rounds"] = []
        return plan

    # label=1 → 直接走 full_trajectory 重生成,不调用第 2 关
    if g_label == 1:
        logger.info(f"    Grounding label=1(编造/心算),直接 full_trajectory 重生成")
        score = {
            "total": 0,
            "issues": [f"grounding: {grounding.get('reasons', [''])[0]}"],
            "reason": "grounding_failed",
            "regen_type": "full_trajectory",
        }
        all_judge_rounds.append(score)
        total = 0
        regen_type = "full_trajectory"
        plan["judge_score"] = score
        plan["judge_passed"] = False
    else:
        # label=0 → 进第 2 关
        score = judge_single_inline(client, question, observations, answer, qtype)
        total = score.get("total", 0)
        regen_type = score.get("regen_type", "answer_only")
        logger.info(f"    Judge: total={total}, regen_type={regen_type}, issues={score.get('issues', [])}")
        all_judge_rounds.append(score)

        plan["judge_score"] = score
        plan["judge_passed"] = total >= JUDGE_THRESHOLD
        plan["judge_all_rounds"] = all_judge_rounds

        if total >= JUDGE_THRESHOLD:
            return plan

        # discard: 直接放弃
        if regen_type == "discard":
            logger.info(f"    Judge 建议 discard,直接放弃")
            return plan

    plan["judge_all_rounds"] = all_judge_rounds

    # 重新生成
    for attempt in range(MAX_REGEN_ATTEMPTS):
        # 只传维度级别的问题描述，不传具体数字，避免模型把 feedback 当数据源
        feedback_parts = []
        # D1 已由第 1 关 LLM Grounding 接管,Judge 不再产 d1_accuracy
        if score.get("d2_coherence", 5) < 4:
            feedback_parts.append("- 逻辑连贯性不足：分析推理存在矛盾或判断缺乏依据")
        if score.get("d3_relevance", 5) < 4:
            feedback_parts.append("- 问题匹配度不足：回答偏离了用户问题，或过度展开了不相关的分析")
        if score.get("d4_professionalism", 5) < 4:
            feedback_parts.append("- 分析专业性不足：金融分析方法使用不当或行业知识有误")
        if score.get("d5_tools", 5) < 4:
            feedback_parts.append("- 工具使用不当：工具选择或使用策略有问题")
        # full_trajectory 由 grounding 触发时,补 grounding reason
        if score.get("reason") == "grounding_failed":
            g_reason = (plan.get("grounding_check") or {}).get("reasons", [""])[0]
            feedback_parts.append(f"- 数字溯源不足(grounding):{g_reason}")
        elif score.get("reason"):
            feedback_parts.append(f"- 总体问题：{score['reason']}")
        feedback = "\n".join(feedback_parts) if feedback_parts else "请提高整体分析质量"

        if regen_type == "full_trajectory" and tools is None:
            logger.warning(f"    Judge 建议 full_trajectory 但 tools 未传入，放弃")
            break

        if regen_type == "full_trajectory":
            # V4: 重新跑整个 generate_trajectory_v4(原 react_loop)
            logger.info(f"    Judge 建议 full_trajectory，重新生成完整轨迹 (attempt {attempt+1})...")
            try:
                # V4: 不再需要 pre_retrieve(SYSTEM_PROMPT_V4 已含数据库信息)
                new_plan = generate_trajectory_v4(client, tools, question, qtype, seed_grouped)
                if new_plan and any(s.get("action") == "finish" for s in new_plan["steps"]):
                    # V4: 完整替换 plan 的 steps + messages(保持下游同步)
                    plan["steps"] = new_plan["steps"]
                    plan["messages"] = new_plan["messages"]
                    plan["num_tool_steps"] = new_plan["num_tool_steps"]
                    plan["tools_used"] = new_plan["tools_used"]
                    plan["retrieval_quality"] = new_plan.get("retrieval_quality", True)
                    steps = plan["steps"]
                    # 新轨迹再过 LLM Grounding(替代 check_mental_math)
                    g2 = llm_grounding_check(client, plan, n_samples=1, model=JUDGE_MODEL)
                    plan["grounding_check"] = g2
                    if g2.get('label') == 1:
                        g2_evidence = (g2.get('reasons') or [''])[0]
                        mm_hint = (
                            f"上一次生成被驳回,原因(grounding):{g2_evidence}\n"
                            "本次生成的强制要求:\n"
                            "1. 答案中每个具体数字必须能在 observation 中字面找到(允许 ±1 位四舍五入)\n"
                            "2. 任何数值运算结果(差值/增长率/比率)必须通过 calculate 工具得出,引用其返回值\n"
                            "3. 在 Thought(content 中)写明要计算什么,然后调用 calculate 工具填入纯数学表达式\n"
                            "4. 无法溯源的数字一律删除或改为定性描述"
                        )
                        logger.info(f"    full_trajectory 新轨迹 grounding 仍 fail,带提示再次重跑...")
                        new_plan2 = generate_trajectory_v4(client, tools, question, qtype,
                                                           seed_grouped, extra_hint=mm_hint)
                        if new_plan2 and any(s.get("action") == "finish" for s in new_plan2["steps"]):
                            plan["steps"] = new_plan2["steps"]
                            plan["messages"] = new_plan2["messages"]
                            plan["num_tool_steps"] = new_plan2["num_tool_steps"]
                            plan["tools_used"] = new_plan2["tools_used"]
                            plan["retrieval_quality"] = new_plan2.get("retrieval_quality", True)
                            steps = plan["steps"]
                        else:
                            logger.warning(f"    full_trajectory 二次重跑失败")
                            break
                    elif g2.get('label') in ('uncertain', 'failed'):
                        logger.warning(f"    full_trajectory 重跑后 grounding={g2.get('label')},放弃")
                        break
                    observations = extract_obs_from_steps(steps)
                    answer = extract_answer_from_steps(steps)
                else:
                    logger.warning(f"    full_trajectory 重新生成失败")
                    break
            except Exception as e:
                logger.warning(f"    full_trajectory 重新生成异常: {e}")
                break
        elif regen_type == "answer_only":
            logger.info(f"    Judge 建议 answer_only，重新生成答案 (attempt {attempt+1})...")
            try:
                new_answer = regen_answer_with_feedback(
                    client, question, qtype, observations, feedback
                )
                # 更新 plan 中的 finish 步答案(V1-like steps)
                for step in reversed(steps):
                    if step.get("action") == "finish":
                        step["action_input"] = new_answer
                        break
                # V4: 同步更新 plan["messages"] 最后一条 assistant content(无 tool_calls 的那条)
                if plan.get("messages"):
                    for m in reversed(plan["messages"]):
                        if m.get("role") == "assistant" and not m.get("tool_calls"):
                            m["content"] = new_answer
                            break
                answer = new_answer
            except Exception as e:
                logger.warning(f"    重新生成答案失败: {e}")
                break

        # 再次 Judge
        score = judge_single_inline(client, question, observations, answer, qtype)
        total = score.get("total", 0)
        regen_type = score.get("regen_type", "answer_only")
        logger.info(f"    Re-Judge: total={total}, regen_type={regen_type}, issues={score.get('issues', [])}")
        all_judge_rounds.append(score)

        plan["judge_score"] = score
        plan["judge_passed"] = total >= JUDGE_THRESHOLD
        plan["judge_all_rounds"] = all_judge_rounds

        if total >= JUDGE_THRESHOLD:
            return plan

        # 如果 Re-Judge 建议 discard，直接退出
        if regen_type == "discard":
            logger.info(f"    Re-Judge 建议 discard，停止重试")
            break

    return plan


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


# [V4 REMOVED] pre_retrieve 函数整体删除。
# V3 的 pre_retrieve 用 probe query 预查一次数据库,把可用信息拼进 STEP_PROMPT 的
# {pre_retrieved_info} placeholder,让 teacher 知道"数据库大概能返回什么",防止编造
# 不存在的指标。V4 在 SYSTEM_PROMPT_V4 已经明确写出:
#   1. 沪深 300 成分股 + 2024H1~2025 财务数据 + 2024-01~2026-03 研报(均硬过滤)
#   2. 30 个行业名称列表
#   3. 不覆盖的品类(港股/美股/加密货币/期货/非金融/主观荐股)
# 所以 teacher 不需要"边生成边探查"库里有啥,直接按 SYSTEM_PROMPT 定义的覆盖范围做
# Level 1 直拒 / Level 2 先搜 的分级判断即可。


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

# ============ V4 辅助:tool_call 转换 + calculate 校验 ============

def _tool_call_to_v2(tc, args_dict: dict) -> dict:
    """把 OpenAI 原生 ChatCompletionMessageToolCall 转成 V2 messages 中的 tool_calls 项。
    关键:arguments 存 dict(不是 string),对齐 Qwen2.5 chat_template 的 items() 要求。"""
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": args_dict,   # ← dict,不是 json.dumps 后的 string
        }
    }


def _validate_calc_expression(expr: str) -> str:
    """返回归一化后的纯数学表达式(Unicode/百分号转换);不合法则 raise ValueError。"""
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError(f"calculate expression 为空或非字符串: {expr!r}")
    # Unicode 运算符转 ASCII
    expr = expr.translate(str.maketrans({
        "×": "*", "÷": "/", "−": "-", "－": "-",
        "（": "(", "）": ")",
    }))
    # 百分号:5% → (5/100)
    expr = re.sub(r'(\d+\.?\d*)%', r'(\1/100)', expr)
    expr = expr.strip()
    # 纯数学表达式校验
    if not CALC_EXPR_RE.match(expr):
        raise ValueError(f"calculate expression 非纯数学: {expr!r}")
    return expr


def _eval_calc(expr: str) -> str:
    """eval 纯数学表达式,返回 Observation 文本格式与 V3 一致。"""
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return f"计算结果: {result}"
    except Exception as e:
        raise ValueError(f"calculate eval 失败: {expr!r} ({e})")


# [V4 REMOVED] fill_calculate(CALC_FILL_PROMPT 二阶段填充)
# [V4 REMOVED] generate_answer(ANSWER_PROMPT 单独调用)
# V4 在 generate_trajectory_v4 里单阶段完成:calculate 直接 validate + eval,
# 最终答案由模型"不调工具时的 content"直接给出(不需要二次 API)


def generate_trajectory_v4(client: OpenAI, tools: FinAgentTools, question: str,
                           question_type: str, seed_grouped: dict = None,
                           extra_hint: str = "",
                           max_steps: int = MAX_STEPS) -> dict:
    """V4 Mode B 生成引擎:原生 tool calling + content 强约束 thought + 全局 max_steps。

    流程:
      1. system = SYSTEM_PROMPT_V4(含详细数据库范围 + 分级 reject + 工具原则)
      2. user   = question + question_type(+ 可选 extra_hint)
      3. loop(step_num in 0..max_steps):
           - tool_choice = "none"(最后一步强制收尾)| "auto"(其余)
           - 调 qwen3-max with tools=TOOLS_NATIVE
           - msg.tool_calls 为空 → finish,content 是最终答案,break
           - 有 tool_calls:取第 1 个(parallel 只取第一个,保持一步一工具)
               - calculate: _validate_calc_expression + _eval_calc
               - search_*: tools.call(name, args_dict)
           - append assistant / tool 到 messages;同步 V1-like steps
      4. 返回字段对齐 V3 react_loop + format_as_sft_sample 的组合

    V4 vs V3:
      - 废弃 MIN_STEPS(teacher 自主判断 finish 时机)
      - 废弃 pre_retrieve 注入(SYSTEM_PROMPT_V4 已自带覆盖范围)
      - 废弃 seed_grouped 注入 prompt(保留 seed_grouped 参数仅为下游 judge 兼容)
      - calculate 单阶段(不再调 CALC_FILL_PROMPT 二次)
      - 不再有"finish"虚拟工具(不调工具即 finish)
    """
    # 构造首轮 user message
    # 注意:GENERATION_HINTS 只在生成期注入 user prompt,推理时 react_agent 不注入,
    # 保持训-推 system prompt 一致。但 user 末尾的 hint 会进训练数据,模型见过。
    user_content = f"## 用户问题\n{question}\n\n## 问题类型\n{question_type}"
    user_content += f"\n{GENERATION_HINTS}"
    if extra_hint:
        user_content += f"\n\n## 重要提示\n{extra_hint}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V4},
        {"role": "user", "content": user_content},
    ]

    steps = []                    # V1-like [{thought, action, action_input, observation}]
    tools_used = []
    retrieval_quality = True

    finished = False
    for step_num in range(max_steps):
        # 最后一步强制收尾(tool_choice=none)
        tool_choice = "none" if step_num == max_steps - 1 else "auto"

        # 调 API(retry on transient errors)
        msg = None
        for retry in range(MAX_RETRY):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=TOOLS_NATIVE,
                    tool_choice=tool_choice,
                    parallel_tool_calls=False,   # API 层硬约束一次一个工具,替代 prompt 软约束
                    temperature=0.7,
                    max_tokens=1500,
                    extra_body={"enable_thinking": False},
                )
                msg = response.choices[0].message
                break
            except Exception as e:
                logger.warning(f"LLM 调用失败 (step {step_num+1} retry {retry}): {e}")
                time.sleep(1)
        if msg is None:
            logger.error(f"step {step_num+1} 生成失败,retry 耗尽")
            return None

        content = (msg.content or "").strip()
        raw_tcs = msg.tool_calls or []

        # ---- 分支 1:不调工具 → finish(content 是最终答案)----
        if not raw_tcs:
            if not content:
                logger.warning(f"step {step_num+1} 既无 tool_call 又无 content,视为失败")
                return None
            # V1-like steps(action=finish,action_input=答案),兼容下游 judge
            steps.append({
                "thought": "",
                "action": "finish",
                "action_input": content,
            })
            # V2 messages
            messages.append({"role": "assistant", "content": content})
            logger.info(f"  Step {step_num+1}: finish (len={len(content)})")
            finished = True
            break

        # ---- 分支 2:有 tool_calls → 取第 1 个,忽略 parallel ----
        tc = raw_tcs[0]
        if len(raw_tcs) > 1:
            logger.info(f"  step {step_num+1}: 收到 {len(raw_tcs)} 个 parallel tool_calls,只取第 1 个")

        name = tc.function.name
        if name not in VALID_TOOLS:
            logger.warning(f"非法工具 '{name}',step {step_num+1} 跳过")
            continue

        # arguments:OpenAI SDK 返回 str(JSON),解析为 dict
        try:
            args_raw = tc.function.arguments
            args_dict = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            if not isinstance(args_dict, dict):
                raise ValueError("arguments 不是 dict")
        except Exception as e:
            logger.warning(f"step {step_num+1} tool_call arguments 解析失败: {e}")
            continue

        # content 非空校验(⭐V4 硬约束)
        if not content:
            logger.warning(f"step {step_num+1} tool_call 时 content 为空(V4 要求必须有 thought)")
            content = "(thought 缺失)"

        # ---- 执行工具 ----
        if name == "calculate":
            expr = args_dict.get("expression", "")
            try:
                expr_normalized = _validate_calc_expression(expr)
                observation = _eval_calc(expr_normalized)
                args_dict = {"expression": expr_normalized}   # 归一化后回写
            except ValueError as e:
                logger.warning(f"step {step_num+1} calculate 失败: {e}")
                continue
            action_input_v1 = expr_normalized
        else:
            query = args_dict.get("query", "")
            if not query:
                logger.warning(f"step {step_num+1} {name} query 为空,跳过")
                continue
            try:
                obs_and_meta = tools.call(name, args_dict)
                observation = obs_and_meta[0] if isinstance(obs_and_meta, tuple) else obs_and_meta
            except Exception as e:
                observation = f"[工具调用失败] {e}"
                logger.warning(f"工具调用失败 [{name}({args_dict})]: {e}")
            if "未找到" in observation or len(observation) < 50:
                retrieval_quality = False
                logger.warning(f"检索质量低: {observation[:100]}")
            action_input_v1 = query

        tools_used.append(name)

        # ---- 记录 steps(V1 风格)和 messages(V2)----
        steps.append({
            "thought": content,
            "action": name,
            "action_input": action_input_v1,
            "observation": observation,
        })
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [_tool_call_to_v2(tc, args_dict)],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": observation,
        })
        logger.info(f"  Step {step_num+1}: {name}({str(action_input_v1)[:60]})")
        time.sleep(0.3)

    if not finished:
        # 跑满 max_steps 但从未产出 finish(通常是连续 continue 到底),视为失败
        logger.warning(f"max_steps={max_steps} 耗尽仍未 finish")
        return None

    return {
        "question": question,
        "type": question_type,
        "messages": messages,
        "steps": steps,
        "num_tool_steps": len(tools_used),
        "tools_used": tools_used,
        "retrieval_quality": retrieval_quality,
    }


# ============ V4 SFT sample 落库(不需要 V1→V2 转换)============

def build_sft_sample(plan: dict) -> dict:
    """V4 直接从 plan 提取训练样本字段。

    相对 V3 format_as_sft_sample:
      - V3 收 V1 steps → 转 V2 messages(70 行转换层)
      - V4 generate_trajectory_v4 已产 messages,此处只是 re-project
    """
    return {
        "question": plan["question"],
        "type": plan["type"],
        "messages": plan["messages"],
        "num_tool_steps": plan["num_tool_steps"],
        "tools_used": plan["tools_used"],
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

def save_checkpoint(results: list, stats: dict, processed_count: int):
    with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
        json.dump(
            {
                "results": results,
                "stats": stats,
                "processed_count": processed_count,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"[checkpoint] 已保存 {len(results)} 条成功样本，已处理 {processed_count} 条问题")


def load_checkpoint() -> tuple:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        processed_count = data.get(
            "processed_count",
            data.get("stats", {}).get("total", len(data.get("results", []))),
        )
        logger.info(
            f"[断点续传] 已有 {len(data['results'])} 条成功数据，"
            f"累计处理 {processed_count} 条问题"
        )
        return data["results"], data["stats"], processed_count
    return [], {}, 0


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="FinAgent SFT 数据生成（Final: V1引擎+V2格式）")
    parser.add_argument("--test", type=int, default=0, help="测试模式，只生成N条")
    parser.add_argument("--type", type=str, default="", help="只生成指定类型")
    parser.add_argument("--resume", action="store_true", help="从断点续传")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--questions", type=str, default="", help="自定义问题文件路径（默认用 QUESTIONS_PATH）")
    parser.add_argument("--output", type=str, default="", help="自定义输出文件路径（默认用 FINAL_OUTPUT_PATH）")
    args = parser.parse_args()

    # 覆盖全局路径
    global QUESTIONS_PATH, FINAL_OUTPUT_PATH, CHECKPOINT_PATH
    if args.questions:
        QUESTIONS_PATH = args.questions
    if args.output:
        FINAL_OUTPUT_PATH = args.output
        CHECKPOINT_PATH = args.output.replace(".jsonl", "_checkpoint.json")

    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("FinAgent SFT 数据生成（Final: V1引擎+V2格式）")
    logger.info(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info("=" * 60)

    # 初始化
    client = OpenAI()
    global _mental_math_client
    _mental_math_client = client
    logger.info("[1/4] 初始化 OpenAI 客户端完成")

    retriever = FinAgentRetriever()
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
             "failed_validation": 0, "failed_rule_check": 0,
             "failed_judge": 0, "low_retrieval": 0}
    processed_count = 0
    if args.resume:
        results, stats, processed_count = load_checkpoint()

    # 收集所有 Judge 评语（通过+未通过），用于最终根因分析
    judge_feedback_log = []  # [{question, type, passed, score, issues, reason}, ...]

    random.shuffle(questions)
    start_idx = min(processed_count, len(questions))

    # 主循环
    for idx, task in enumerate(questions[start_idx:], start=start_idx):
        question = task["question"]
        qtype = task["type"]
        stats["total"] += 1

        if idx % 10 == 0:
            total_failed = stats['failed_react'] + stats['failed_validation'] + stats.get('failed_rule_check', 0) + stats.get('failed_judge', 0)
            logger.info(f"--- 进度: {idx}/{len(questions)} | 成功: {stats['success']} | 失败: {total_failed} ---")

        try:
            # V4: 废弃 Phase 0 预检索(SYSTEM_PROMPT_V4 已含完整数据库范围)
            # Phase 1: V4 Mode B 生成轨迹
            plan = generate_trajectory_v4(client, tools, question, qtype, seed_grouped)

            if not plan:
                stats["failed_react"] += 1
                logger.warning(f"[{idx}] 轨迹生成失败: {question[:50]}")
                continue

            if not plan.get("retrieval_quality", True) and qtype != "reject":
                stats["low_retrieval"] += 1

            # Phase 2: 验证(V1-like steps 字段验证)
            passed, validation_errors = validate_sample(plan)
            if not passed:
                stats["failed_validation"] += 1
                logger.warning(f"[{idx}] 验证失败: {validation_errors}")
                continue  # 验证失败直接跳过
            plan["validation_passed"] = True
            plan["validation_errors"] = []

            # Phase 2.5: 规则质检（D1-D10，零 API 成本）
            if plan["validation_passed"]:  # 验证失败已 continue，此处恒为 True
                rule_passed, rule_issues, rule_score = rule_based_quality_check(plan)
                plan["rule_check_score"] = rule_score
                plan["rule_check_issues"] = [i["detail"] for i in rule_issues]

                if not rule_passed:
                    # 有严重问题，尝试重新生成答案
                    feedback = "\n".join(f"- {i['detail']}" for i in rule_issues)
                    logger.info(f"    规则质检未通过(score={rule_score})，重新生成答案...")

                    try:
                        observations = extract_obs_from_steps(plan["steps"])
                        new_answer = regen_answer_with_feedback(
                            client, question, qtype, observations, feedback
                        )
                        for step in reversed(plan["steps"]):
                            if step.get("action") == "finish":
                                step["action_input"] = new_answer
                                break

                        # 再次规则质检
                        rule_passed2, rule_issues2, rule_score2 = rule_based_quality_check(plan)
                        plan["rule_check_score"] = rule_score2
                        plan["rule_check_issues"] = [i["detail"] for i in rule_issues2]

                        if not rule_passed2:
                            stats["failed_rule_check"] = stats.get("failed_rule_check", 0) + 1
                            logger.warning(f"[{idx}] 规则质检二次未通过(score={rule_score2}): {[i['detail'] for i in rule_issues2[:2]]}")
                            continue
                    except Exception as e:
                        stats["failed_rule_check"] = stats.get("failed_rule_check", 0) + 1
                        logger.warning(f"[{idx}] 规则质检重新生成失败: {e}")
                        continue

                logger.info(f"    规则质检通过(score={plan['rule_check_score']})")

            # Phase 2.6: 已废弃独立心算检测,改由 Phase 2.75 judge_and_regen 第 1 关 LLM Grounding 统一处理
            # (规则质检留作零成本兜底信号)

            # Phase 2.75: LLM-as-Judge + 重新生成
            if plan["validation_passed"]:  # 验证/规则质检失败已 continue，此处恒为 True
                plan = judge_and_regen(client, plan, seed_grouped, tools=tools, retriever=retriever)
                # 收集所有轮次的 Judge 评语（包括中间被驳回的）
                for round_score in plan.get("judge_all_rounds", []):
                    judge_feedback_log.append({
                        "question": question,
                        "type": qtype,
                        "passed": round_score.get("total", 0) >= JUDGE_THRESHOLD,
                        "total": round_score.get("total", 0),
                        "issues": round_score.get("issues", []),
                        "reason": round_score.get("reason", ""),
                    })
                if not plan.get("judge_passed", False):
                    stats["failed_judge"] = stats.get("failed_judge", 0) + 1
                    logger.warning(f"[{idx}] Judge 未通过: {plan.get('judge_score', {}).get('reason', '')}")
                    continue  # 丢弃该条

                # Judge 可能重写答案或整条轨迹，需重新走硬验证
                passed_after_judge, validation_errors_after_judge = validate_sample(plan)
                plan["validation_passed"] = passed_after_judge
                plan["validation_errors"] = validation_errors_after_judge
                if not passed_after_judge:
                    stats["failed_validation"] += 1
                    logger.warning(f"[{idx}] Judge 重生成后验证失败: {validation_errors_after_judge}")
                    continue

                rule_passed3, rule_issues3, rule_score3 = rule_based_quality_check(plan)
                plan["rule_check_score"] = rule_score3
                plan["rule_check_issues"] = [i["detail"] for i in rule_issues3]
                if not rule_passed3:
                    stats["failed_rule_check"] = stats.get("failed_rule_check", 0) + 1
                    logger.warning(
                        f"[{idx}] Judge 重生成后规则质检未通过(score={rule_score3}): "
                        f"{[i['detail'] for i in rule_issues3[:2]]}"
                    )
                    continue

                # 心算检测已下沉到 judge_and_regen 第 1 关 LLM Grounding,这里不再重复

            # Phase 3: V4 SFT sample 落库(不需要 V1→V2 转换,plan 已带 messages)
            sft_sample = build_sft_sample(plan)
            sft_sample["validation_passed"] = plan["validation_passed"]
            sft_sample["validation_errors"] = plan.get("validation_errors", [])
            sft_sample["judge_score"] = plan.get("judge_score", {})

            results.append(sft_sample)
            stats["success"] += 1

            time.sleep(0.3)

        except Exception as e:
            logger.error(f"[{idx}] 未预期错误: {e}")
            stats["failed_react"] += 1
            continue
        finally:
            if stats["total"] > 0 and stats["total"] % 20 == 0:
                save_checkpoint(results, stats, processed_count=stats["total"])

    # 最终保存
    save_checkpoint(results, stats, processed_count=stats["total"])

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

    stats_path = os.path.splitext(FINAL_OUTPUT_PATH)[0] + "_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    judge_scores = [s.get("judge_score", {}).get("total", 0) for s in results if s.get("judge_score")]
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else 0

    logger.info("SFT 数据生成完成（V1引擎+V2格式+规则质检+内联Judge）")
    logger.info(f"总数: {len(results)} 条")
    logger.info(f"验证通过: {validation_passed} ({validation_passed/max(len(results),1):.1%})")
    logger.info(f"规则质检淘汰: {stats.get('failed_rule_check', 0)} 条")
    logger.info(f"Judge 平均分: {avg_judge:.2f}")
    logger.info(f"Judge 未通过: {stats.get('failed_judge', 0)} 条")
    logger.info(f"类型分布: {type_dist}")
    logger.info(f"工具分布: {tool_dist}")
    logger.info(f"步数分布: {step_dist}")
    logger.info(f"输出路径: {FINAL_OUTPUT_PATH}")
    logger.info("=" * 60)

    # ============ Judge 评语根因分析 ============
    if judge_feedback_log:
        # 保存原始评语日志（路径跟随 --output）
        output_base = os.path.splitext(FINAL_OUTPUT_PATH)[0]
        feedback_path = output_base + "_judge_feedback.json"
        with open(feedback_path, 'w', encoding='utf-8') as f:
            json.dump(judge_feedback_log, f, ensure_ascii=False, indent=2)
        logger.info(f"Judge 评语日志已保存: {feedback_path}")

        # 收集未通过的评语做根因分析
        failed_feedbacks = [fb for fb in judge_feedback_log if not fb["passed"]]
        if failed_feedbacks and len(failed_feedbacks) >= 3:
            logger.info(f"正在对 {len(failed_feedbacks)} 条未通过评语做根因分析...")

            # 按问题类型分组统计
            type_issues = {}
            for fb in failed_feedbacks:
                t = fb["type"]
                if t not in type_issues:
                    type_issues[t] = []
                for issue in fb["issues"]:
                    type_issues[t].append(issue)

            # 拼接给 LLM 做总结
            summary_parts = []
            for t, issues in sorted(type_issues.items()):
                summary_parts.append(f"### {t}（{len(issues)} 个问题）")
                for issue in issues[:10]:  # 每类最多10条
                    summary_parts.append(f"- {issue}")
            issues_text = "\n".join(summary_parts)

            root_cause_prompt = f"""以下是 SFT 数据生成过程中 LLM-as-Judge 发现的质量问题汇总。
请分析这些问题的根因，区分以下三类：

1. **Prompt 设计问题**：生成 prompt（STEP_PROMPT / ANSWER_PROMPT）的指令不够明确，导致模型行为偏差
2. **检索/知识库问题**：检索系统返回的数据不够或不准确，导致答案质量差
3. **Chunk 设计问题**：chunk 的内容组织方式有缺陷，导致关键信息缺失或混淆

## 未通过的问题按类型分组：
{issues_text}

请输出 JSON 格式的根因分析：
```json
{{"prompt_issues": ["问题1", "问题2"], "retrieval_issues": ["问题1"], "chunk_issues": ["问题1"], "summary": "一段话总结最核心的问题和建议"}}
```"""

            try:
                response = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": root_cause_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=2000,
                    extra_body={"enable_thinking": False},
                )
                root_cause = json.loads(response.choices[0].message.content)

                # 保存根因分析（路径跟随 --output）
                root_cause_path = output_base + "_root_cause.json"
                with open(root_cause_path, 'w', encoding='utf-8') as f:
                    json.dump(root_cause, f, ensure_ascii=False, indent=2)

                logger.info("=" * 60)
                logger.info("Judge 根因分析：")
                logger.info(f"  Prompt 问题: {root_cause.get('prompt_issues', [])}")
                logger.info(f"  检索问题: {root_cause.get('retrieval_issues', [])}")
                logger.info(f"  Chunk 问题: {root_cause.get('chunk_issues', [])}")
                logger.info(f"  总结: {root_cause.get('summary', '')}")
                logger.info(f"  详细报告: {root_cause_path}")
                logger.info("=" * 60)
            except Exception as e:
                logger.warning(f"根因分析调用失败: {e}")


if __name__ == "__main__":
    main()
