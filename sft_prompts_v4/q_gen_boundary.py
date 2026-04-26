"""
Q_GEN_BOUNDARY — 边界 subtype question 生成

覆盖 150 条 skeleton:
  clarify_positive  30   故意设计歧义(主体/时间/口径/对比维度)
  clarify_negative  15   明确无歧义(负样本,防过度反问)
  insuf_L1          15   字段不支持(员工人数/ESG 评分等 schema 外)
  insuf_L2          15   时间越界(2027/2022 超出库覆盖)
  insuf_L3          10   部分覆盖(缺 Q1/Q3 季报)
  insuf_L4          20   主体不在库(港股/美股/未上市)
  anomaly_seed      20   query 含潜在异常数字或数量级
  multi_source_seed 25   涉及多研报观点或预测分歧

核心:每种 subtype 的 question 有独特语境,不是普通 ReAct 问,需要 teacher **故意设计**成触发特定行为。
"""

SYSTEM_PROMPT = """你是 FinAgent SFT 数据生成助手中的"边界情境问题模拟器"。你的任务是**根据 skeleton 生成一条能触发特定边界行为的 user question**。

边界行为包括:歧义澄清 / 知识边界承认 / 异常数据质疑 / 多源分歧处理。teacher 会按 question 触发的 subtype 执行对应策略。

## 硬规则

1. **只输出 question 文本**,不要输出解释、编号、引号
2. **question 长度 10-35 字**
3. **严格按 skeleton 的 subtype 生成对应语境**:

### subtype 语境定义

#### clarify_positive(故意设计歧义)
- 必须包含**至少一种歧义**:
  - **主体歧义**:"平安" 可指平安银行 vs 中国平安;"长城" 可指长城汽车 vs 长城银行
  - **时间歧义**:"最近" / "近期" / "最新"(没说具体年份)
  - **口径歧义**:"EPS"(实际 vs 预测)/ "ROE"(加权 vs 扣非)
  - **对比维度歧义**:"X 和 Y 谁更好"(没说比什么)
- 示例:"平安 EPS 多少"(主体 + 时间 + 口径三重歧义)

#### clarify_negative(故意无歧义的负样本)
- 明确、无歧义的 query,teacher **不应该反问**,应该直接查
- 示例:"贵州茅台 2025 年报 ROE 多少"(明确主体 + 明确时间 + 明确指标)
- **用途**:让模型学会"有 clarify 行为但不要过度反问"

#### insuf_L1(字段不支持)
- 问 FinAgent 工具 schema **没有**的字段:
  - 员工人数 / 员工福利 / 工资水平
  - ESG 评分 / 碳排放 / 社保缴纳
  - 股东结构 / 高管持股 / 董监高变动
  - 具体业务线细分数据
- 示例:"茅台 2024 年员工人数是多少?"

#### insuf_L2(时间越界)
- 问库**不覆盖**的时间:
  - **早于 2024H1 的财务**(2023 年报 / 2023H1 / 2022 年报 / 2022H1 等,所有 2024-06-30 之前数据均越界)
  - **研报早于 2024-01-01**(2020/2019 及更早)
  - **2026-03 之后的未来**(2027 / 2028 / 2029 预测)
- 示例:"工商银行 2027 年 ROE 多少?"
- **❌ 严禁**:把 2024H1 / 2024 年报 / 2025H1 / 2025 年报 当作越界(这些在库内)

#### insuf_L3(部分覆盖)
- 问**部分可查部分不可查**的组合:
  - 问各季度 ROE,但库只有半年报/年报(没 Q1/Q3)
  - 跨期对比但其中一期越界("2022 vs 2025 ROE")
- 示例:"贵州茅台 2025 年各季度 ROE 趋势"(库只有 H1/年报,Q1/Q3 缺)

#### insuf_L4(主体不在库)
- 问**非 A 股沪深 300** 主体:
  - 港股(腾讯控股/友邦保险/华润置地)
  - 美股(特斯拉/英伟达/苹果)
  - 非沪深 300 A 股(创业板小盘)
- 示例:"特斯拉 2025 年营收情况如何?"

#### anomaly_seed(诱发异常数据质疑)
- Query **暗含潜在异常值** 或询问 "是否真的如此":
  - "XX 同比增长 XXX% 是真的吗"(暗示异常高)
  - "XX ROE 是否真的达到 80%"(暗示超高)
  - "XX 半年报净利润大幅下滑的原因"(触发 teacher 查异常值)
- 示例:"XX 2024 年报 ROE 据说有 120%,请验证"

#### multi_source_seed(诱发多源分歧处理)
- Query **涉及多个机构/研报的预测**:
  - "多家券商对 XX 的 EPS 预测"
  - "XX 评级分歧"
  - "市场对 XX 未来盈利预期"
  - "综合研报看 XX 目标价"
- 示例:"多家券商对平安银行 2025 EPS 的预测区间是多少?"

## 避免的反模式

- 不要生成普通的 query_simple("XX 的 ROE 多少")类 —— 那是 GENERIC 的任务
- 不要在 clarify_negative 里加任何歧义词("最近"/"EPS"不加限定等)
- insuf_L2 的越界年份必须真的越界(不要用 2024/2025)

## ⭐ subtype 优先级高于 template_class

**如果 skeleton 同时给了 subtype 和 template_class,subtype 强制覆盖**:
- subtype=anomaly_seed,即使 tc=status_how,也必须生成含异常信号的 question(不能生成普通"XX 表现如何")
- subtype=multi_source_seed,即使 tc=comparison_ab,也必须生成含"多机构分歧"语境的 question
- subtype=clarify_negative,必须显式给**主体 + 时间 + 指标** 三要素齐全(不能省略时间)
- subtype=insuf_L1,即使 skeleton 的 metric_tag 给了合法字段(周转率/ROE 等),也要**改用 schema 外字段**(员工人数 / ESG / 股东结构 / 高管持股 等)作为问题内容

## anomaly_seed 数字生成规则

- **不要复制 few-shot 里的具体数字(120% / +500%)**,根据 metric 自己设计合理但极端的异常数字
- 示例数字参考:
  - ROE: 真实 10-40%,异常值 80-150%(触发质疑)
  - 营收同比增速: 真实 -20%~+50%,异常值 +300%~+2000%
  - 毛利率: 真实 5-70%,异常值 90-120%
- 数字必须贴合被问公司的行业常识(不要银行问毛利率 120%,因为银行根本不用毛利率)
"""

# ============ Few-shot(15 条)============

FEW_SHOTS = [
    # --- clarify_positive(3 条,展示不同歧义类型)---
    {"skel": {"subtype": "clarify_positive", "ambiguity_type": "主体+口径+时间"},
     "question": "平安 EPS 多少?"},
    {"skel": {"subtype": "clarify_positive", "ambiguity_type": "时间+口径"},
     "question": "茅台最近 ROE 怎么样?"},
    {"skel": {"subtype": "clarify_positive", "ambiguity_type": "对比维度"},
     "question": "宁德时代和比亚迪哪个更好?"},

    # --- clarify_negative(3 条,P1: 强口径限定避免隐含歧义)---
    {"skel": {"subtype": "clarify_negative"},
     "question": "贵州茅台 2025 年报加权 ROE 多少?"},
    {"skel": {"subtype": "clarify_negative"},
     "question": "比亚迪 2024 年报营业总收入同比增速"},
    {"skel": {"subtype": "clarify_negative"},
     "question": "招商银行 2025H1 扣非净利润同比"},

    # --- insuf_L1(2 条,schema 外字段)---
    {"skel": {"subtype": "insuf_L1", "unsupported_field": "员工人数"},
     "question": "贵州茅台 2024 年员工总数是多少?"},
    {"skel": {"subtype": "insuf_L1", "unsupported_field": "ESG 评分"},
     "question": "宁德时代 ESG 评分是多少?"},

    # --- insuf_L2(2 条,时间越界)---
    {"skel": {"subtype": "insuf_L2", "period_value": "2022 年报"},
     "question": "工商银行 2022 年报 ROE 是多少?"},
    {"skel": {"subtype": "insuf_L2", "period_value": "2027 年报"},
     "question": "平安银行 2027 年 EPS 预测多少?"},

    # --- insuf_L3(1 条,部分覆盖)---
    {"skel": {"subtype": "insuf_L3"},
     "question": "贵州茅台 2025 年各季度 ROE 趋势"},

    # --- insuf_L4(2 条,非 A 股)---
    {"skel": {"subtype": "insuf_L4"},
     "question": "特斯拉 2024 年毛利率多少?"},
    {"skel": {"subtype": "insuf_L4"},
     "question": "华润置地 2025 年经营表现如何?"},

    # --- anomaly_seed(2 条,触发质疑)---
    {"skel": {"subtype": "anomaly_seed"},
     "question": "有研报说宁德时代 2025H1 ROE 达到 120%,是否靠谱?"},
    {"skel": {"subtype": "anomaly_seed"},
     "question": "比亚迪 2024 年营收同比增长 500%,请核实一下"},

    # --- multi_source_seed(2 条,多机构分歧)---
    {"skel": {"subtype": "multi_source_seed"},
     "question": "多家券商对平安银行 2025 EPS 的预测分别是多少?"},
    {"skel": {"subtype": "multi_source_seed"},
     "question": "市场对宁德时代 2025 年净利润的共识预期范围"},
]


# ============ 渲染 ============

USER_PROMPT_TEMPLATE = """请基于以下 skeleton 生成一条边界情境 question:

## Skeleton

{skeleton_str}

## 子类要求

**subtype = `{subtype}`**:{subtype_desc}

{specific_hint}

## 输出

只输出 question 文本(10-35 字),**不要任何其他内容**。"""


SUBTYPE_DESC = {
    # 全名(skeleton 新约定)
    "clarify_positive": "故意设计歧义(主体/时间/口径/对比维度,至少一种)",
    "clarify_negative": "明确无歧义(负样本,teacher 不应反问)",
    "insuf_L1": "问 schema 不支持的字段(员工/ESG/股东结构等)",
    "insuf_L2": "问库不覆盖的时间(早于 2024H1 或 2026-03 之后)",   # P1: 2026Q3 → 2026-03 之后
    "insuf_L3": "问部分可查部分不可查(Q1/Q3 季报缺失 或 跨期部分越界)",
    "insuf_L4": "问非 A 股主体(港美股/非沪深 300)",
    "anomaly_seed": "Query 暗含潜在异常值,诱发 teacher 数据质疑",
    "multi_source_seed": "涉及多机构/多研报预测或分歧",
    # P0: 短名别名(build_v4_skeletons.py 实际产出的 subtype 值)
    "clarify": "故意设计歧义(主体/时间/口径/对比维度,至少一种)",
    "clarify_neg": "明确无歧义(负样本,teacher 不应反问)",
    "anomaly": "Query 暗含潜在异常值,诱发 teacher 数据质疑",
    "multi_source": "涉及多机构/多研报预测或分歧",
}

# P0: 短名 → 全名 归一化表(用于 _specific_hint / _format_fewshots filter)
SUBTYPE_ALIAS = {
    "clarify": "clarify_positive",
    "clarify_neg": "clarify_negative",
    "anomaly": "anomaly_seed",
    "multi_source": "multi_source_seed",
}


def _canonicalize_subtype(sub):
    """把短名归一化为全名,兼容 skeleton 的历史命名"""
    if sub is None:
        return None
    return SUBTYPE_ALIAS.get(sub, sub)


def _specific_hint(skeleton):
    """按 subtype 给额外 hint(P0: 先归一化短名→全名)"""
    sub = _canonicalize_subtype(skeleton.get("subtype"))
    if sub == "clarify_positive":
        return "提示:歧义类型可选 主体 / 时间 / 口径 / 对比维度,必须至少含一种"
    if sub == "clarify_negative":
        return "提示:明确公司名 + 明确时间 + 明确指标 + 无歧义口径,teacher 直接查即可"
    if sub == "insuf_L1":
        return "提示:选 schema 外字段(员工人数/ESG/股东结构/高管持股等)"
    if sub == "insuf_L2":
        period = skeleton.get("period_value", "2022 年报")
        return f"提示:必须用越界时间 {period}(库不覆盖)"
    if sub == "insuf_L3":
        return "提示:问各季度 ROE / 跨期有一期越界 等"
    if sub == "insuf_L4":
        sn = skeleton.get("stock_name")
        if sn:
            return f"提示:主体必须是 {sn}(非 A 股 / 未上市)"
        return "提示:用港股/美股/未上市公司名"
    if sub == "anomaly_seed":
        return "提示:query 暗含异常值(ROE>100% / 同比+500% / 净利润骤变)或主动质疑"
    if sub == "multi_source_seed":
        return "提示:query 必须涉及'多家券商'/'市场共识'/'研报分歧'等关键词"
    return ""


def _format_skeleton(skel):
    """简洁格式化,只展示非 null 字段(P2: 加 period 对齐 generic)"""
    lines = []
    for k in ["type", "subtype", "stock_name", "industry",
              "period", "period_value", "metric_tag", "hint"]:
        v = skel.get(k)
        if v in (None, [], ""):
            continue
        if isinstance(v, list):
            v = " / ".join(str(x) for x in v)
        lines.append(f"  - {k}: {v}")
    return "\n".join(lines) if lines else "  (minimal)"


def _format_fewshots(subtype, n=3, rng=None):
    """
    优先同 subtype,混 1 条其他做风格参考
    P0: 归一化短名→全名后再 filter,保证 clarify/anomaly/multi_source 能取到同 subtype few-shot
    """
    import random as _random
    rng = rng or _random.Random()
    canon = _canonicalize_subtype(subtype)
    same = [s for s in FEW_SHOTS if s["skel"].get("subtype") == canon]
    other = [s for s in FEW_SHOTS if s["skel"].get("subtype") != canon]
    picked = rng.sample(same, min(2, len(same)))
    if other and n > 2:
        picked += rng.sample(other, 1)
    out = ["## 参考示例(仅供风格参考,不要照抄)\n"]
    for i, s in enumerate(picked, 1):
        sub = s["skel"].get("subtype", "?")
        out.append(f"**示例 {i}** ({sub})")
        out.append(f"→ {s['question']}\n")
    return "\n".join(out)


def render(skeleton, n_fewshot=3):
    """渲染完整 messages(P0: 独立 RNG + 短名归一化)"""
    import random as _random, time as _time
    seed = hash(skeleton.get("skeleton_id", "")) ^ _time.time_ns()
    rng = _random.Random(seed)

    raw_subtype = skeleton.get("subtype", "clarify_positive")
    canon = _canonicalize_subtype(raw_subtype) or "clarify_positive"

    user = USER_PROMPT_TEMPLATE.format(
        skeleton_str=_format_skeleton(skeleton),
        subtype=canon,   # prompt 里始终展示全名,teacher 对齐 SYSTEM_PROMPT 里的子类规则
        subtype_desc=SUBTYPE_DESC.get(canon, "边界类"),
        specific_hint=_specific_hint(skeleton),
    )
    fewshots = _format_fewshots(raw_subtype, n_fewshot, rng=rng)
    user = fewshots + "\n---\n\n" + user

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


if __name__ == "__main__":
    test = {
        "skeleton_id": "v4_skel_0500",
        "prompt_bucket": "clarify_positive",
        "type": "single_company_simple",
        "subtype": "clarify_positive",
        "stock_name": None,
        "metric_tag": ["EPS"],
    }
    msgs = render(test)
    print("=== USER ===")
    print(msgs[1]["content"])
