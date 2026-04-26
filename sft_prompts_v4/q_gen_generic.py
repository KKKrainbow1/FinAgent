"""
Q_GEN_GENERIC — 通用 ReAct 调工具类 question 生成

覆盖 ~444 条 skeleton:
  - 6 个 existing type 扩充(305 条)
  - reject 44 条(含 time_boundary 40)
  - D1-D4 深度 seed 95 条

核心思路:
  skeleton 的 template_class 字段(7 种)承载句式约束,teacher 按 class 生成不同风格。
  7 种 few-shot 各 3-4 条,让 teacher 看到多样范本。

不覆盖:
  - finance_concept 8 子类 → q_gen_concept
  - clarify / anomaly / multi_source / insuf L1-L4 → q_gen_boundary
"""

SYSTEM_PROMPT = """你是 FinAgent SFT 数据生成助手中的"用户提问模拟器"。你的任务是**根据下面提供的 skeleton(元标签集合),生成一条真实用户会问的 A 股金融分析 question**。

## 知识库覆盖范围(⭐生成 question 时必须严格遵守)

- **财务数据**:沪深 300 成分股 × 4 期(**2024H1 / 2024 年报 / 2025H1 全量 + 2025 年报约 53%**)
- **研报正文**:**2024-01-01 ~ 2026-03**(硬过滤)
- **股票池**:**仅沪深 300 成分股**(港股/美股/未上市/非沪深 300 均不在库)
- **行业**:30 大类(银行/保险/证券/白酒/食品饮料/医药/医疗/汽车/半导体/消费电子/光伏/电池/电力/煤炭/钢铁/有色金属/化工/房地产/建筑建材/家电/软件/通信/传媒/交通运输/军工/机械/农业/石油石化/纺织服装/零售)

**时间生成规则**:
- skeleton.period_value **在库内** → question 用该期别(预期 teacher 正常查询)
- skeleton.period_value **越界** → question 显式提该越界年份(预期 teacher 触发 time_boundary 拒答)
- skeleton.period=None → question 不提时间(用户自然省略,teacher 按 fallback 最新)
- **❌ 严禁**:在 skeleton 没给 period_value 时,自己编造库外年份(如 2020 / 2030 / 2019 等)

## 硬规则

1. **只输出 question 文本**,不要输出解释、编号、引号或其他任何内容
2. **question 长度 15-35 字**,简洁但信息清晰
3. **严格遵守 skeleton 指定的**:
   - `template_class`:问题句式风格
   - `stock_name`(如有):公司名必须出现在 question 里
   - `period_value`(如有):时间表达必须准确显式(如 "2025 年报" / "2024 半年报")
   - `metric_tag`:question 应自然涉及至少 1 个指定指标(但不需要全部列出)
4. **period 字段含义**:
   - `None` → 不提时间(用户自然省略,默认最新)
   - "单期" + period_value → 必须显式提到 period_value
   - "跨期" + period_value(list)→ 必须两期都提
   - "越界" + period_value → 必须显式提到越界年份
5. **避免模板化**:**严禁**使用以下禁用句式(产生这些句式必须重写):
   - ❌ "XX 的 YY 是多少?" / "XX YY 多少?"
   - ❌ "XX YY 表现如何?" / "XX 的 YY 情况如何?"
   - ❌ "XX 目前 YY 怎么样?"(除 clarify_positive 外)
   - ✅ 替代:"XX 年报 YY 跑赢同业了吗?" / "XX 近期 YY 压力多大?" / "XX YY 在行业里排第几?" / "XX 2025 业绩会重现去年的 Z 吗?"

## template_class 句式定义

- `query_simple`:直接查询数字/事实("茅台 2025 年报 ROE?" / "招行 2024 年报净利率多少")
- `causal_why`:因果追问("为什么 XX 下滑" / "XX 下滑的驱动因子")
- `comparison_ab`:多公司对比("A 和 B 的 X 比较" / "A 与 B 谁的 Y 更好")
- `status_how`:状态评价("XX 的 Y 表现如何" / "XX Y 能力情况")
- `risk_query`:风险分析("XX 有哪些风险" / "XX 的经营脆弱点")
- `industry_query`:行业问题("XX 行业整体 Y" / "XX 行业 Y 格局")
- `counterfactual`:假设/反事实("如果 X,Y 会怎样" / "假设 Y 下滑 N%,对 Z 影响")

## 难度 hint(`difficulty_tag` 字段,可选)

- `D1 推理深度`:含因果追问 + 至少 2 个指标的关联 + 要求追问"驱动因子"
- `D2 答案结构`:要求全面评估(杜邦 / 敏感性 / 多维)
- `D3 数据融合`:含 3+ 公司或多期数据交叉
- `D4 反事实`:含"假设" / "如果" + 量化传导要求
"""

# ============ Few-shot 示例(21 条,每 template_class 3 条)============

FEW_SHOTS = [
    # --- query_simple(3 条,P1 反模板化:避免 "XX YY 多少?" 单一句式)---
    {
        "skel": {
            "template_class": "query_simple",
            "stock_name": "贵州茅台",
            "period_value": "2025 年报",
            "metric_tag": ["ROE"],
            "period": "单期",
        },
        "question": "茅台 2025 年报 ROE 跑赢白酒同业了吗?"
    },
    {
        "skel": {
            "template_class": "query_simple",
            "stock_name": "招商银行",
            "period_value": None,
            "metric_tag": ["净息差"],
            "period": None,
        },
        "question": "招行最新 NIM 有没有跌破 2?"
    },
    {
        "skel": {
            "template_class": "query_simple",
            "stock_name": "宁德时代",
            "period_value": "2025H1",
            "metric_tag": ["毛利率"],
            "period": "单期",
        },
        "question": "宁德 2025 半年报毛利率回升到什么水平"
    },

    # --- causal_why(3)---
    {
        "skel": {
            "template_class": "causal_why",
            "stock_name": "招商银行",
            "period_value": "2025 年报",
            "metric_tag": ["ROE", "净利率"],
            "period": "单期",
            "difficulty_tag": "D1",
        },
        "question": "招行 2025 年报 ROE 下滑的主要驱动因子是净利率还是周转率?"
    },
    {
        "skel": {
            "template_class": "causal_why",
            "stock_name": "比亚迪",
            "metric_tag": ["毛利率", "营收增速"],
            "period": None,
        },
        "question": "为什么比亚迪近期毛利率承压而营收增速仍保持高位?"
    },
    {
        "skel": {
            "template_class": "causal_why",
            "stock_name": "隆基绿能",
            "period_value": "2024 年报",
            "metric_tag": ["净利润"],
            "period": "单期",
        },
        "question": "隆基绿能 2024 年报净利润为什么大幅下滑?"
    },

    # --- comparison_ab(3)---
    {
        "skel": {
            "template_class": "comparison_ab",
            "stock_name": ["招商银行", "兴业银行"],
            "period_value": "2025H1",
            "metric_tag": ["ROE", "NIM"],
            "period": "单期",
        },
        "question": "招行和兴业 2025 半年报 ROE 和净息差谁更优?"
    },
    {
        "skel": {
            "template_class": "comparison_ab",
            "stock_name": ["宁德时代", "比亚迪", "国轩高科"],
            "period_value": ["2024 年报", "2025H1"],
            "metric_tag": ["毛利率"],
            "period": "跨期",
            "difficulty_tag": "D3",
        },
        "question": "对比宁德、比亚迪、国轩 2024 年报和 2025 半年报毛利率变化趋势"
    },
    {
        "skel": {
            "template_class": "comparison_ab",
            "stock_name": ["万科A", "保利发展"],
            "metric_tag": ["资产负债率", "经营现金流"],
            "period": None,
        },
        "question": "万科和保利谁更可能先现金流告急"
    },

    # --- status_how(3 条,P1 反模板化:避免 "XX YY 表现如何?" 单一句式)---
    {
        "skel": {
            "template_class": "status_how",
            "stock_name": "海天味业",
            "period_value": "2025 年报",
            "metric_tag": ["盈利能力", "营收增速"],
            "period": "单期",
        },
        "question": "海天味业 2025 年报还撑得住调味品龙头地位吗?"
    },
    {
        "skel": {
            "template_class": "status_how",
            "stock_name": "三一重工",
            "metric_tag": ["周转率", "流动比率"],
            "period": None,
        },
        "question": "三一重工当下流动性绷不绷?周转有没有放慢"
    },
    {
        "skel": {
            "template_class": "status_how",
            "stock_name": "恒瑞医药",
            "period_value": "2025H1",
            "metric_tag": ["研发投入", "净利率"],
            "period": "单期",
        },
        "question": "恒瑞 2025 半年报研发费用压利润了吗?"
    },

    # --- risk_query(3 条,P1 反模板化:多变体,避免 "XX 风险在哪些方面" 单一句式)---
    {
        "skel": {
            "template_class": "risk_query",
            "stock_name": "中国国航",
            "metric_tag": ["资产负债率", "利息覆盖"],
            "period": None,
        },
        "question": "国航的高杠杆会不会在利率上行时爆?"
    },
    {
        "skel": {
            "template_class": "risk_query",
            "stock_name": "荣盛石化",
            "period_value": "2025H1",
            "metric_tag": ["经营现金流", "负债"],
            "period": "单期",
        },
        "question": "荣盛石化 2025H1 现金流恶化到什么程度了"
    },
    {
        "skel": {
            "template_class": "risk_query",
            "stock_name": "长城汽车",
            "metric_tag": ["库存", "应收"],
            "period": None,
        },
        "question": "长城汽车库存和应收有没有积压信号"
    },

    # --- industry_query(3)---
    {
        "skel": {
            "template_class": "industry_query",
            "industry": "钢铁",
            "period_value": "2025H1",
            "metric_tag": ["毛利率", "营收增速"],
            "period": "单期",
        },
        "question": "钢铁行业 2025 半年报整体毛利率和营收表现如何?"
    },
    {
        "skel": {
            "template_class": "industry_query",
            "industry": "银行",
            "metric_tag": ["NIM", "拨备覆盖率"],
            "period": None,
        },
        "question": "银行业最新的息差压力和资产质量怎么样?"
    },
    {
        "skel": {
            "template_class": "industry_query",
            "industry": "光伏",
            "period_value": "2024 年报",
            "metric_tag": ["盈利能力"],
            "period": "单期",
        },
        "question": "光伏行业 2024 年报盈利能力分化明显吗?"
    },

    # --- counterfactual(3)---
    {
        "skel": {
            "template_class": "counterfactual",
            "stock_name": "招商银行",
            "metric_tag": ["NIM", "ROE"],
            "period": None,
            "difficulty_tag": "D4",
        },
        "question": "如果 LPR 再下调 20bp,招行 NIM 和 ROE 会受多少影响?"
    },
    {
        "skel": {
            "template_class": "counterfactual",
            "stock_name": "宁德时代",
            "metric_tag": ["毛利率"],
            "period": None,
            "difficulty_tag": "D4",
        },
        "question": "假设碳酸锂价格再跌 10%,宁德时代毛利率能维持在多少水平?"
    },
    {
        "skel": {
            "template_class": "counterfactual",
            "stock_name": "茅台",
            "metric_tag": ["ROE", "净利率"],
            "period": None,
            "difficulty_tag": "D4",
        },
        "question": "如果茅台净利率下滑 5pct,对 ROE 的传导影响大概几个 pct?"
    },
]


# ============ 渲染 ============

USER_PROMPT_TEMPLATE = """请基于以下 skeleton 生成一条真实用户 question:

## Skeleton

{skeleton_str}

## 提示

- template_class = `{template_class}`,句式必须按此类风格
- difficulty_tag = `{difficulty_tag}`,{difficulty_guidance}
- period 风格 = `{period_style}`,{period_guidance}
- 自然、不模板化,不要用"XX的YY是多少"这种烂套式

## 输出

只输出 question 文本(15-35 字),**不要任何其他内容**。"""


DIFFICULTY_GUIDANCE = {
    "D1": "问题要含因果追问或驱动因子拆解,不要只问单数字",
    "D2": "问题要求全面评估或结构化分析(可提'杜邦'/'敏感性'等)",
    "D3": "问题应涉及 3+ 公司或多期数据交叉",
    "D4": "问题必须含'如果'/'假设',并要求量化传导",
    None: "无难度特殊要求",
}

PERIOD_GUIDANCE = {
    None: "不要提时间(用户自然省略,默认最新)",
    "单期": "必须显式提到指定期别",
    "跨期": "必须两期都提(对比语境)",
    "越界": "必须提到越界年份(teacher 会触发 time_boundary 拒答)",
}


def _format_skeleton(skel):
    """格式化 skeleton 成可读字符串,跳过 null 字段"""
    lines = []
    def fmt(k, v):
        if v is None or v == [] or v == "":
            return None
        if isinstance(v, list):
            v = " / ".join(str(x) for x in v)
        return f"  - {k}: {v}"
    for k in ["type", "subtype", "difficulty_tag",
              "stock_name", "industry",
              "period", "period_value",
              "metric_tag", "template_class",
              "hint"]:
        line = fmt(k, skel.get(k))
        if line:
            lines.append(line)
    return "\n".join(lines)


def _format_fewshots(n=5, rng=None):
    """从 FEW_SHOTS 里采样 n 条,格式化成 prompt 前缀(P0: 用独立 RNG 防 seed 污染)"""
    import random as _random
    rng = rng or _random.Random()
    shots = rng.sample(FEW_SHOTS, min(n, len(FEW_SHOTS)))
    out = ["## 参考示例(仅供风格参考,不要照抄)\n"]
    for i, s in enumerate(shots, 1):
        skel_str = _format_skeleton(s["skel"])
        out.append(f"**示例 {i}**")
        out.append(skel_str)
        out.append(f"→ {s['question']}\n")
    return "\n".join(out)


def render(skeleton, n_fewshot=5):
    """
    渲染完整 messages,直接用于 OpenAI client.chat.completions.create
    返回 list[dict] 格式的 messages

    P0 修:few-shot 采样用独立 RNG(基于 skeleton_id hash + time_ns),避免全局 seed 污染
    """
    import random as _random, time as _time
    seed = hash(skeleton.get("skeleton_id", "")) ^ _time.time_ns()
    rng = _random.Random(seed)

    user = USER_PROMPT_TEMPLATE.format(
        skeleton_str=_format_skeleton(skeleton),
        template_class=skeleton.get("template_class", "status_how"),
        difficulty_tag=skeleton.get("difficulty_tag", None),
        difficulty_guidance=DIFFICULTY_GUIDANCE.get(skeleton.get("difficulty_tag"), "无"),
        period_style=skeleton.get("period"),
        period_guidance=PERIOD_GUIDANCE.get(skeleton.get("period"), "无"),
    )

    # few-shot 作为 user prompt 前缀(in-context learning)
    fewshots = _format_fewshots(n_fewshot, rng=rng)
    user = fewshots + "\n---\n\n" + user

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


if __name__ == "__main__":
    # smoke test
    test_skel = {
        "skeleton_id": "v4_skel_0001",
        "prompt_bucket": "medium_expand",
        "type": "single_company_medium",
        "subtype": None,
        "difficulty_tag": "D1",
        "stock_code": "300750",
        "stock_name": "宁德时代",
        "industry": "电池",
        "period": "单期",
        "period_value": "2025H1",
        "metric_tag": ["ROE", "净利率"],
        "template_class": "causal_why",
        "hint": "含因果追问,要求推理深度",
    }
    msgs = render(test_skel)
    print("=== SYSTEM ===")
    print(msgs[0]["content"][:500])
    print("\n=== USER ===")
    print(msgs[1]["content"])
