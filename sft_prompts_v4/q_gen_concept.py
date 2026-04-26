"""
Q_GEN_CONCEPT — finance_concept 纯知识 question 生成

覆盖 200 条 skeleton (fc_C1-C8):
  C1_concept       概念定义     20 条   "什么是 ROE?" / "TTM 是什么意思?"
  C2_formula       公式计算     20 条   "杜邦三因子公式" / "自由现金流怎么算"
  C3_methodology   分析方法论   45 条   "如何判断现金流质量" / "怎么识别财报造假信号"
  C4_industry_norm 行业惯例     45 条   "银行 NIM 合理区间" / "白酒毛利率正常水平"
  C5_distinguish   指标辨析     20 条   "ROE vs ROIC" / "合并 vs 母公司报表"
  C6_pitfall       误区陷阱     20 条   "高 ROE 就是好公司吗"
  C7_three_statement 三表联动  15 条   "利润亏损但现金流好怎么解读"
  C8_calibration   解读口径    15 条   "加权 ROE vs 扣非 ROE"

核心:所有 concept 类 question 都是 **纯知识问答,0 步 tool_call**。
  - 不涉及具体公司(stock_name = null)
  - 不涉及具体时间(period = null)
  - 答案由 teacher 凭预训练金融知识回答
"""

SYSTEM_PROMPT = """你是 FinAgent SFT 数据生成助手中的"金融知识问题模拟器"。你的任务是**为 finance_concept 类型生成一条用户 question**,这类 question **不需要调用任何工具查数据**,teacher 会凭金融专业知识直接回答。

## ⭐ 语境硬约束

- **必须是中国 A 股语境下的金融分析问题**
- **中国会计准则(CAS)** 口径,不是 US-GAAP / IFRS
- **涉及行业惯例(C4)** 时,必须是中国 A 股相关行业的基准(如"中国银行业 NIM 合理区间",不是美股银行)
- **涉及报表口径(C7/C8)** 时,必须是中国 A 股报表(合并/母公司/扣非/加权等中国口径)

### ❌ 严禁生成以下美股/国际概念

- 10-K / 10-Q / 8-K 等 SEC 填报
- NOPAT / WACC 在 US-GAAP 下的计算
- DCF 按 US 折现口径
- 美股 Buy-Back / Dividend Yield 等术语
- 非财务分析领域(临床试验 / 技术路径 / 市场营销分析等)

### ✅ 鼓励的中国 A 股特有主题

- **A 股报表特色**:合并报表 vs 母公司 / 扣非净利润 / 加权 ROE / 营业总收入
- **A 股监管**:业绩快报 / 业绩预告 / 减持公告 / 一季报季报(虽然库内只有 H1/年报,但概念层可讨论)
- **中国行业基准**:银行 NIM(中国水平) / 白酒毛利率(茅五洋区间) / 地产去化率
- **中国财报造假信号**:关联交易异常 / 存货周转激变 / 应收账款激增等

## 硬规则

1. **只输出 question 文本**,不要输出解释、编号、引号
2. **question 长度 6-30 字**,具体由 subtype 决定(concept/formula 偏短,methodology/pitfall 偏长)
3. **严格按 skeleton 的 subtype 分类风格生成**:
   - `C1_concept`(概念定义):"什么是 X" / "X 是什么意思" / "怎么理解 X"
   - `C2_formula`(公式计算):"X 怎么算" / "X 的公式是什么" / "X 如何计算"
   - `C3_methodology`(分析方法论):"如何判断 X" / "怎么识别 X 信号" / "X 的分析框架"
   - `C4_industry_norm`(行业惯例):"XX 行业 X 合理区间" / "XX 行业 X 正常水平"
   - `C5_distinguish`(指标辨析):"X 和 Y 有什么区别" / "X 与 Y 如何选"
   - `C6_pitfall`(误区陷阱):"X 就是好 Y 吗" / "看 X 有什么误区"
   - `C7_three_statement`(三表联动):"X 但 Y 好/坏,怎么解读"
   - `C8_calibration`(解读口径):"X vs Y 的区别" / "X 怎么看待"
4. **绝不涉及具体公司名或具体时间**(如果 skeleton 里出现,忽略)
   - skeleton.industry = "银行" 时,只能生成**"银行业 X"** 级别的问题,**严禁**出现具体银行名(招行/工行/平安银行等)
   - 如果 skeleton 意外传入 stock_name,直接忽略不用
5. **避免模板化**:同一 subtype 的多条 question 要换不同词序和问法
6. **metric 与 subtype 兼容性**:
   - C5_distinguish(指标辨析)的两个 metric 必须**同维度**(都是盈利类 / 都是偿债类 / 都是成长类);如果给的 metric 不同维度(如 "净利率 vs 营收增速"),**只选其一,单独出一个辨析题**
   - C7_three_statement(三表联动)必须涉及**2+ 张报表**的指标;如果只给 1 个指标,自动扩展相关三表信号
   - C4_industry_norm 如果 skeleton.industry 为 None 但 metric 有明显行业属性(如 NIM→银行 / 拨备→银行 / 去化率→地产),自动使用对应行业

## 金融知识主题提示(可选用)

- **盈利**:ROE / ROIC / ROA / 毛利率 / 净利率 / EPS / 杜邦分解
- **成长**:营收增速 / 利润增速 / CAGR
- **现金流**:经营现金流 / 自由现金流 / 现金含量 / 收现比
- **偿债**:资产负债率 / 流动比率 / 速动比率 / 利息保障倍数
- **估值**:PE / PB / EV/EBITDA / PEG
- **行业特殊**:NIM(银行)/ 拨备覆盖率(银行)/ 去化率(地产)/ 单位成本(电池)/ 研发强度(医药)
- **方法论**:杜邦分析 / 波特五力 / 三阶段现金流折现 / 波特钻石模型
- **财报造假信号**:存货周转异常 / 应收款激增 / 经营现金流与利润背离 / 关联交易
"""

# ============ Few-shot(16 条,每 C 2 条)============

FEW_SHOTS = [
    # --- C1_concept(3 条,P2 加变体)---
    {"skel": {"subtype": "C1_concept", "metric_tag": ["ROE"]},
     "question": "什么是 ROE?"},
    {"skel": {"subtype": "C1_concept", "metric_tag": ["EV/EBITDA"]},
     "question": "EV/EBITDA 怎么理解?和 PE 有什么关系?"},
    {"skel": {"subtype": "C1_concept", "metric_tag": ["自由现金流"]},
     "question": "自由现金流和经营现金流的核心差别在哪?"},

    # --- C2_formula(2)---
    {"skel": {"subtype": "C2_formula", "metric_tag": ["杜邦"]},
     "question": "杜邦三因子分解的公式是什么?"},
    {"skel": {"subtype": "C2_formula", "metric_tag": ["自由现金流"]},
     "question": "自由现金流应该怎么算?"},

    # --- C3_methodology(3)---
    {"skel": {"subtype": "C3_methodology", "metric_tag": ["现金流"]},
     "question": "如何判断一家公司的现金流质量?"},
    {"skel": {"subtype": "C3_methodology", "metric_tag": []},
     "question": "识别财报造假的 5 个关键信号是什么?"},
    {"skel": {"subtype": "C3_methodology", "metric_tag": []},
     "question": "怎么用三张表交叉验证财务健康度?"},

    # --- C4_industry_norm(3)---
    {"skel": {"subtype": "C4_industry_norm", "industry": "银行", "metric_tag": ["NIM"]},
     "question": "银行净息差合理区间是多少?"},
    {"skel": {"subtype": "C4_industry_norm", "industry": "白酒", "metric_tag": ["毛利率"]},
     "question": "白酒行业毛利率的正常水平在多少?"},
    {"skel": {"subtype": "C4_industry_norm", "industry": "地产", "metric_tag": []},
     "question": "地产去化率多少算警戒线?"},

    # --- C5_distinguish(3 条)---
    {"skel": {"subtype": "C5_distinguish", "metric_tag": ["ROE", "ROIC"]},
     "question": "ROE 和 ROIC 有什么区别?"},
    {"skel": {"subtype": "C5_distinguish", "metric_tag": []},
     "question": "合并报表和母公司报表,分析时怎么选?"},
    {"skel": {"subtype": "C5_distinguish", "metric_tag": ["毛利率", "净利率"]},
     "question": "毛利率和净利率看哪个更反映真实盈利?"},

    # --- C6_pitfall(3 条)---
    {"skel": {"subtype": "C6_pitfall", "metric_tag": ["ROE"]},
     "question": "高 ROE 就是好公司吗?"},
    {"skel": {"subtype": "C6_pitfall", "metric_tag": ["毛利率"]},
     "question": "毛利率持续提升一定是好事吗?"},
    {"skel": {"subtype": "C6_pitfall", "metric_tag": ["PE"]},
     "question": "低市盈率是不是就代表估值便宜?"},

    # --- C7_three_statement(3 条,P1 扩容)---
    {"skel": {"subtype": "C7_three_statement", "metric_tag": ["净利润", "现金流"]},
     "question": "利润亏损但经营现金流很好,这种公司怎么解读?"},
    {"skel": {"subtype": "C7_three_statement", "metric_tag": ["应收账款", "营收"]},
     "question": "应收账款激增但营收也涨,是真增长还是虚增?"},
    {"skel": {"subtype": "C7_three_statement", "metric_tag": ["资产", "负债"]},
     "question": "资产大幅重估对三张表的影响路径是什么?"},

    # --- C8_calibration(3 条,P1 扩容)---
    {"skel": {"subtype": "C8_calibration", "metric_tag": ["ROE"]},
     "question": "加权平均 ROE 和扣非 ROE,看哪个更能反映真实经营?"},
    {"skel": {"subtype": "C8_calibration", "metric_tag": ["营收"]},
     "question": "营业收入和营业总收入有什么差别?"},
    {"skel": {"subtype": "C8_calibration", "metric_tag": ["现金流"]},
     "question": "经营现金流和自由现金流,分析时怎么取?"},
]


# ============ 渲染 ============

USER_PROMPT_TEMPLATE = """请基于以下 skeleton 生成一条 finance_concept 类 question:

## Skeleton

  - subtype: {subtype}
  - 子类风格: {subtype_desc}
  - 涉及指标/主题: {topic_str}

## 提示

- subtype = `{subtype}`,严格按此子类风格生成
- 长度 6-30 字,短而明确
- **不要涉及具体公司名或时间**
- 题目要自然,避免模板化重复

## 输出

只输出 question 文本,**不要任何其他内容**。"""


SUBTYPE_DESC = {
    "C1_concept": "概念定义(问'什么是 X' / '怎么理解 X')",
    "C2_formula": "公式计算(问'X 怎么算' / 'X 公式')",
    "C3_methodology": "分析方法论(问'如何判断 X' / '怎么识别 X')",
    "C4_industry_norm": "行业惯例基准(问'XX 行业 X 合理区间')",
    "C5_distinguish": "指标辨析(问'X 和 Y 区别')",
    "C6_pitfall": "误区陷阱(问'X 就是好 Y 吗?')",
    "C7_three_statement": "三表联动(问'X 但 Y 好/坏,怎么解读')",
    "C8_calibration": "解读口径(问'X vs Y 的差异,看哪个合理')",
}


def _format_fewshots(subtype, n=3, rng=None):
    """优先采样同 subtype 的 few-shot,混 1-2 条其他 C 子类做风格多样性(P0: 用独立 RNG)"""
    import random as _random
    rng = rng or _random.Random()
    same = [s for s in FEW_SHOTS if s["skel"].get("subtype") == subtype]
    other = [s for s in FEW_SHOTS if s["skel"].get("subtype") != subtype]
    picked = rng.sample(same, min(2, len(same))) + rng.sample(other, min(1, n - 2))
    out = ["## 参考示例(仅供风格参考,不要照抄)\n"]
    for i, s in enumerate(picked, 1):
        sub = s["skel"].get("subtype", "?")
        metrics = s["skel"].get("metric_tag", [])
        ind = s["skel"].get("industry", "")
        topic = ""
        if metrics:
            topic += f"涉及:{' / '.join(metrics)}"
        if ind:
            topic += f" (行业:{ind})"
        out.append(f"**示例 {i}** ({sub}){topic}")
        out.append(f"→ {s['question']}\n")
    return "\n".join(out)


def render(skeleton, n_fewshot=3):
    """
    渲染完整 messages,用于生成 finance_concept question(P0: 独立 RNG)
    """
    import random as _random, time as _time
    seed = hash(skeleton.get("skeleton_id", "")) ^ _time.time_ns()
    rng = _random.Random(seed)

    subtype = skeleton.get("subtype", "C1_concept")
    metrics = skeleton.get("metric_tag", [])
    industry = skeleton.get("industry")

    topic_parts = []
    if metrics:
        topic_parts.append(" / ".join(metrics))
    if industry:
        topic_parts.append(f"行业={industry}")
    topic_str = "; ".join(topic_parts) if topic_parts else "通用金融概念"

    user = USER_PROMPT_TEMPLATE.format(
        subtype=subtype,
        subtype_desc=SUBTYPE_DESC.get(subtype, "通用金融知识"),
        topic_str=topic_str,
    )
    fewshots = _format_fewshots(subtype, n_fewshot, rng=rng)
    user = fewshots + "\n---\n\n" + user

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


if __name__ == "__main__":
    test = {
        "skeleton_id": "v4_skel_0100",
        "prompt_bucket": "fc_C3_methodology",
        "type": "finance_concept",
        "subtype": "C3_methodology",
        "metric_tag": ["现金流", "经营现金流"],
        "industry": None,
    }
    msgs = render(test)
    print("=== USER ===")
    print(msgs[1]["content"])
