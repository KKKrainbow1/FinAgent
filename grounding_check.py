"""
V4 数字溯源校验(Grounding Check)

核心思路(YIMING 的创新):
  所有数据都有源头,不应凭空生成。如果 answer/thought 里出现了 observation 中
  找不到的数字,就是编造,触发 full_trajectory 重生成。

  这比"心算检测"更根本 —— 心算检测是溯源校验的子集(覆盖 ~90%):
    心算检测 ⊂ 溯源校验
    心算抓:做了运算但没用 calculate(且中间值在 obs)
    溯源抓:任何凭空数字(无论是算的还是直接编的)

工作流:
  1. extract_numerics(text):从 text 里提取数字事实(NumericFact)
  2. collect_obs_facts(plan):聚合所有 tool step 的 observation 数字
     ⭐ 关键:calculate 工具返回的"计算结果: 16.31"也是 observation 的一部分,
        它的数字必须进 obs_facts(由 calculate 算出的数字直接命中 'direct')
  3. is_grounded(num, obs_facts):五级容差判断
     - direct:    精确字面匹配(含 calculate 返回值)
     - fuzzy:     ±0.05% 百分比 / ±1% 相对值容差
     - whitelist: 经验阈值(2.0 / 100 等行业基准)
     - derivable: 由 obs 两数 ±/×÷ 一步还原(救"心算合规但工具未用")
     - missing:   纯编造(触发 regen)
  4. grounding_check(plan):整轨迹扫描,返回 issues 列表

使用:
  from grounding_check import grounding_check
  issues = grounding_check(plan)
  fabricated = [i for i in issues if i['type'] == 'fabricated']
  if fabricated:
      # 触发 full_trajectory regen
"""
from __future__ import annotations

import re
import itertools
from dataclasses import dataclass
from typing import Optional


# ============ 配置 ============

# 容差阈值(对齐 FinQA EMNLP 2021 标准)
# Bug 4 修复:0.05% 太严(30.66% 写 30% 差 0.66 误判),改为分级容差
EPSILON_REL = 0.01      # 比率/绝对值的相对容差(1%)


def get_pct_tolerance(value: float) -> float:
    """
    分级百分比容差:
    - < 1% (精确数字): 0.05  (1.23% → 1.2% 严格)
    - 1-10% (单位数): 0.5    (7.76% → 8% 容许)
    - >= 10% (双位数): 1.0   (36.99% → 37% / 30.66% → 30% 容许整数舍入)
    """
    abs_v = abs(value)
    if abs_v < 1:
        return 0.05
    if abs_v < 10:
        return 0.5
    return 1.0


# 兼容旧引用(默认中等容差)
EPSILON_PCT = 1.0

# 经验阈值白名单(中国 A 股财务分析常见基准/排名/家数)
WHITELIST_INT = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                 15, 20, 25, 30, 40, 50, 60, 80, 100,
                 200, 300, 500, 1000}
WHITELIST_FLOAT = {0.5, 1.5, 2.0, 3.0, 5.0,    # 流动比率/速动比率经验值
                   0.2, 0.4, 0.6, 0.85, 0.95}  # 资产负债率行业基准


# Iteration 1: 在数字提取阶段过滤伪数字,源头降低误判
# 量词列表:数字后跟这些 token 通常是业务量,非金融指标,跳过提取
QUANTIFIER_CHARS = (
    '年月日天周届个只种次款版期号位人名股家季'
    '　'  # 全角空格
)
# 但"年报"等组合是 date,不是量词,优先级要在 _DATE_RE 之后判断


def _is_after_quantifier(text: str, end_pos: int) -> bool:
    """end_pos 之后是否紧跟量词字符"""
    if end_pos >= len(text):
        return False
    next_char = text[end_pos]
    return next_char in QUANTIFIER_CHARS


def _is_in_chinese_word(text: str, start: int, end: int) -> bool:
    """数字前后是否都是中文(嵌在中文产品名/词组中)"""
    if start <= 0 or end >= len(text):
        return False
    prev_char = text[start - 1]
    next_char = text[end] if end < len(text) else ''
    is_chinese = lambda c: '一' <= c <= '鿿'
    return is_chinese(prev_char) and is_chinese(next_char)


def _is_after_rank_prefix(text: str, start: int) -> bool:
    """数字前是否有排名前缀(第/排名/TOP/No.)"""
    prefix_window = text[max(0, start - 5): start]
    return any(p in prefix_window for p in ['第', '排名', 'TOP', 'top', 'No.', 'no.'])


def _is_year_value(value: float) -> bool:
    """整数是否在年份范围"""
    return 1990 <= value <= 2050


# ============ 数字事实数据结构 ============

@dataclass(frozen=True)
class NumericFact:
    """归一化后的数字事实"""
    raw: str               # 原始字符串(用于报告 issue)
    value: float           # 归一化后的数值
    kind: str              # 'pct' / 'amount' / 'ratio' / 'date' / 'int'
    unit: str = ""         # 'yuan' / '%' / '元' / '亿元' 等

    def __hash__(self):
        return hash((round(self.value, 4), self.kind))

    def __eq__(self, other):
        if not isinstance(other, NumericFact):
            return False
        # 同 kind 且数值容差内
        if self.kind != other.kind:
            return False
        eps = EPSILON_PCT if self.kind == 'pct' else EPSILON_REL * max(abs(self.value), 1)
        return abs(self.value - other.value) <= eps


# ============ 数字提取 ============

# 5 种原子模式(贪婪不重叠)
_PCT_RE = re.compile(r'(-?\d+(?:\.\d+)?)\s*(?:个百分点|百分点|pct|%)')   # Bug 2 扩展:"个百分点"也算 pct
# Bug 1 修复:裸"亿"/"万"(没"元"字)在研报口语化中常见,"元"改可选
_AMOUNT_RE = re.compile(r'(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(亿|万|百万|千万|万亿)\s*元?')
# Iter 2 Bug A 修复:区间表达 "247-323 亿元" / "14-16 倍 PE",前段共享后段单位
# 例:"247-323 亿元" 应识别为 247 amount + 323 amount,而不是 247 int + 323 amount
_RANGE_AMOUNT_RE = re.compile(r'(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*[-~–至到]\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(亿|万|百万|千万|万亿)\s*元?')
# Iter 3 Bug A2: "15%-16%" 这种"前段也带%"的范围,中间的%可选
_RANGE_PCT_RE = re.compile(
    r'(-?\d+(?:\.\d+)?)\s*%?\s*[-~–至到]\s*(-?\d+(?:\.\d+)?)\s*(?:个百分点|百分点|pct|%)'
)
# 日期:含年份的各种格式(2025-12-31 / 2025年12月 / 2025 年报 / 2025年 / 2025H1 / 2025Q1)
# "2025 年报"中 "2025" 和 "年" 间允许空格
_DATE_RE = re.compile(r'(?:20\d{2})[年\-/]\d{1,2}[月\-/]\d{1,2}日?|'
                      r'(?:20\d{2})[年\-/]\d{1,2}月?|'
                      r'(?:20\d{2})\s*年(?:报|度)?(?:[一二三四上下]?半?年?报?)?|'
                      r'(?:20\d{2})\s*[Hh][12]|'
                      r'(?:20\d{2})\s*[Qq][1-4]|'
                      r'(?<!\d)(?:20\d{2})(?!\d)')   # 裸 4 位年份(20XX)
_RATIO_RE = re.compile(r'(?<![%元万亿\d.])(-?\d+\.\d+)(?!\s*[%元万亿])')   # 浮点数,前后不接单位
_INT_RE = re.compile(r'(?<![%元万亿\d.])(\d{1,4})(?!\s*[%元万亿\d])')      # 1-4 位整数,前后不接单位

UNIT_TO_YUAN = {'亿': 1e8, '万': 1e4, '百万': 1e6, '千万': 1e7, '万亿': 1e12}


def extract_numerics(text: str) -> list[NumericFact]:
    """
    从 text 提取所有数字事实。

    支持 5 种类型:
      - pct:    "36.99%" → kind='pct', value=36.99
      - amount: "1505 亿元" → kind='amount', value=150500000000
      - date:   "2024年" / "2024H1" / "2025-12-31" → kind='date'(独立时间一致性检查)
      - ratio:  "4.83" / "1.22" → kind='ratio'(浮点数,无单位)
      - int:    "300" / "5" → kind='int'(纯整数,白名单豁免)
    """
    facts = []
    # 用 mask 标记已被消费的字符,避免重复提取
    mask = [False] * len(text)

    def consume(start, end):
        for i in range(start, end):
            if i < len(mask):
                mask[i] = True

    def is_consumed(start, end):
        return any(mask[i] for i in range(start, min(end, len(mask))))

    # 0. 区间表达式优先(Iter 2 Bug A:247-323 亿元 / 14-16% / 2.0-2.2)
    # 否则前段会被 _INT_RE / _RATIO_RE 误抓
    for m in _RANGE_AMOUNT_RE.finditer(text):
        if is_consumed(m.start(), m.end()):
            continue
        try:
            for grp in (m.group(1), m.group(2)):
                num_str = grp.replace(',', '')
                value = float(num_str) * UNIT_TO_YUAN.get(m.group(3), 1)
                facts.append(NumericFact(
                    raw=f"{grp}{m.group(3)}元",
                    value=value,
                    kind='amount',
                    unit=m.group(3) + '元',
                ))
            consume(m.start(), m.end())
        except ValueError:
            continue

    for m in _RANGE_PCT_RE.finditer(text):
        if is_consumed(m.start(), m.end()):
            continue
        try:
            for grp in (m.group(1), m.group(2)):
                value = float(grp)
                facts.append(NumericFact(
                    raw=f"{grp}%",
                    value=value,
                    kind='pct',
                    unit='%',
                ))
            consume(m.start(), m.end())
        except ValueError:
            continue

    # 1. 百分比(优先级最高)
    for m in _PCT_RE.finditer(text):
        if is_consumed(m.start(), m.end()):
            continue
        try:
            value = float(m.group(1))
            facts.append(NumericFact(
                raw=m.group(0).strip(),
                value=value,
                kind='pct',
                unit='%',
            ))
            consume(m.start(), m.end())
        except ValueError:
            continue

    # 2. 金额(Bug 1 修:支持千分位逗号 "1,505 亿元",也支持裸"亿"/"万")
    for m in _AMOUNT_RE.finditer(text):
        if is_consumed(m.start(), m.end()):
            continue
        try:
            num_str = m.group(1).replace(',', '')   # 去千分位
            value = float(num_str) * UNIT_TO_YUAN.get(m.group(2), 1)
            facts.append(NumericFact(
                raw=m.group(0).strip(),
                value=value,
                kind='amount',
                unit=m.group(2) + '元',
            ))
            consume(m.start(), m.end())
        except ValueError:
            continue

    # 3. 日期
    for m in _DATE_RE.finditer(text):
        if is_consumed(m.start(), m.end()):
            continue
        # date 仅占位,不提取数值(由 D9 时间一致性检查处理)
        facts.append(NumericFact(
            raw=m.group(0).strip(),
            value=0.0,
            kind='date',
            unit='',
        ))
        consume(m.start(), m.end())

    # 4. 浮点数(ratio 类,如 "4.83" / "1.22")
    for m in _RATIO_RE.finditer(text):
        if is_consumed(m.start(), m.end()):
            continue
        try:
            value = float(m.group(1))
            facts.append(NumericFact(
                raw=m.group(0).strip(),
                value=value,
                kind='ratio',
                unit='',
            ))
            consume(m.start(), m.end())
        except ValueError:
            continue

    # 5. 纯整数(白名单豁免常见经验值/家数)
    # Iter 1 源头过滤伪数字:YEAR / QUANTIFIER / RANK / PRODUCT 一律跳过(不进 obs/answer 数字池)
    for m in _INT_RE.finditer(text):
        if is_consumed(m.start(), m.end()):
            continue
        try:
            value = float(m.group(1))
        except ValueError:
            continue

        # YEAR 过滤:年份范围整数(1990-2050)直接跳过,这些是日期不是数据
        if _is_year_value(value):
            consume(m.start(), m.end())   # 占位 mask 防其他 pattern 抓
            continue

        # QUANTIFIER 过滤:数字后跟量词字符(年/月/家/位/股 等)
        if _is_after_quantifier(text, m.end()):
            consume(m.start(), m.end())
            continue

        # PRODUCT 过滤:数字嵌中文词中(国窖 1573 / 片仔癀 188 等)
        if _is_in_chinese_word(text, m.start(), m.end()):
            consume(m.start(), m.end())
            continue

        # RANK 过滤:数字前有排名前缀(第/TOP/排名)
        if _is_after_rank_prefix(text, m.start()):
            consume(m.start(), m.end())
            continue

        # 通过过滤,作为真整数加入
        facts.append(NumericFact(
            raw=m.group(0).strip(),
            value=value,
            kind='int',
            unit='',
        ))
        consume(m.start(), m.end())

    return facts


# ============ 出处定义(五级)============

def is_grounded(num: NumericFact, obs_facts: list[NumericFact]) -> str:
    """
    判断 num 是否在 obs_facts 中找到出处。

    返回:
      'direct'     精确字面匹配(含 calculate 返回值)
      'fuzzy'      容差内匹配(±0.05% 百分比 / ±1% 相对值)
      'whitelist'  经验阈值白名单(2.0 / 100 等行业基准)
      'derivable'  由 obs 两数 ±/×÷ 一步还原(救"心算合规")
      'missing'    纯编造
    """
    # date 不需要溯源(由 D9 时间一致性单独检查)
    if num.kind == 'date':
        return 'direct'

    # Level 1: 直接命中(归一后字面相等)
    for o in obs_facts:
        if o.kind == num.kind and abs(o.value - num.value) < 1e-6:
            return 'direct'

    # Level 2: 容差(同 kind)
    # Iter 2 Bug B 修复:小数 ratio (如 -0.6 vs -0.61) 相对容差 0.6%=0.006 太严
    # 改为 max(相对 1%, 绝对 0.05) — 救小数舍入
    eps = get_pct_tolerance(num.value) if num.kind == 'pct' else max(EPSILON_REL * abs(num.value), 0.05)
    for o in obs_facts:
        if o.kind == num.kind and abs(o.value - num.value) <= eps:
            return 'fuzzy'

    # Level 3: 经验阈值白名单
    if num.kind == 'int' and int(num.value) in WHITELIST_INT:
        return 'whitelist'
    if num.kind == 'ratio' and num.value in WHITELIST_FLOAT:
        return 'whitelist'

    # Level 4: 由 obs 两数运算 ±/×÷ 一步还原
    # 救"心算合规但工具未用"场景:中间值在 obs 但结果不在 obs
    # Bug 5 修复:校验 op 结果 kind 与 target 一致,避免 amount 之间运算误命中 pct target
    target = num.value
    target_kind = num.kind

    # 按 kind 分桶,避免跨 kind 运算(amount-amount 不能产生 pct)
    obs_numeric = [o for o in obs_facts if o.kind in ('pct', 'amount', 'ratio')]

    # 定义 op 及结果 kind:input_kind 决定 output_kind
    # add/sub/mul/div: 同 kind 运算结果保持 kind(amount-amount=amount, pct-pct=pct)
    # pct_change: 任意两数算变化率 → 结果一定是 pct
    same_kind_ops = [
        ('add', lambda x, y: x + y),
        ('sub', lambda x, y: x - y),
    ]
    pct_output_ops = [
        ('pct_change', lambda x, y: (x - y) / y * 100 if y != 0 else None),
        ('pct_change_r', lambda x, y: (y - x) / x * 100 if x != 0 else None),
        # mul/div 结果 kind 不确定,只在同 kind=ratio 时尝试
    ]
    ratio_output_ops = [
        ('div', lambda x, y: x / y if y != 0 else None),
        ('rdiv', lambda x, y: y / x if x != 0 else None),
        ('mul', lambda x, y: x * y),
    ]

    tol = get_pct_tolerance(target) if target_kind == 'pct' else max(EPSILON_REL * abs(target), 0.01)

    for a, b in itertools.combinations(obs_numeric, 2):
        # same_kind_ops:输入输出 kind 一致,只在 a.kind == b.kind == target_kind 才用
        if a.kind == b.kind == target_kind:
            for _, op in same_kind_ops:
                try:
                    val = op(a.value, b.value)
                    if val is not None and abs(val - target) <= tol:
                        return 'derivable'
                except Exception:
                    continue

        # pct_output_ops:输出一定是 pct,只在 target_kind=='pct' 时尝试(任意输入 kind)
        if target_kind == 'pct':
            for _, op in pct_output_ops:
                try:
                    val = op(a.value, b.value)
                    if val is not None and abs(val - target) <= tol:
                        return 'derivable'
                except Exception:
                    continue

        # ratio_output_ops(div/mul):输出一般是 ratio,只在 target_kind=='ratio' 时尝试
        if target_kind == 'ratio':
            for _, op in ratio_output_ops:
                try:
                    val = op(a.value, b.value)
                    if val is not None and abs(val - target) <= tol:
                        return 'derivable'
                except Exception:
                    continue

    return 'missing'


# ============ 主检查函数 ============

def collect_obs_facts(plan: dict) -> list[NumericFact]:
    """
    收集所有 tool step 的 observation 数字 —— 不区分 search_* 还是 calculate。

    关键(YIMING 强调):calculate 的返回 "计算结果: 16.3100" 也走 extract_numerics,
    16.3100 进入 obs_facts 池,直接命中 answer 里的 16.31%(fuzzy 容差)。

    Bug 2 修复:calculate 返回值是裸数(kind=ratio),但 answer 引用时常加 %(kind=pct),
    kind 不同会导致 direct/fuzzy 不命中。所以 calculate observation 提取的数字 **双登记**
    (同时进 ratio 和 pct 两个 kind 池),保证 answer 写"16.31%"时能命中。

    Args:
        plan: dict,有 'steps' 字段,每个 step 是 {action, action_input, observation, ...}

    Returns:
        所有 step observation 提取出的 NumericFact 列表(去重)
    """
    obs_facts = []
    seen = set()

    def _add(fact):
        key = (fact.kind, round(fact.value, 4))
        if key not in seen:
            seen.add(key)
            obs_facts.append(fact)

    for step in plan.get('steps', []):
        action = step.get('action', '')
        observation = step.get('observation', '')

        # 跳过 finish step(action_input 是 answer,不是 observation)
        if action == 'finish' or not observation:
            continue

        is_calc = (action == 'calculate')

        for fact in extract_numerics(observation):
            _add(fact)
            # Bug 2 修:calculate 返回的 ratio 数字双登记为 pct(answer 可能加 %)
            if is_calc and fact.kind == 'ratio':
                _add(NumericFact(
                    raw=fact.raw,
                    value=fact.value,
                    kind='pct',
                    unit='%',
                ))
            # 反向:如果 calculate 算出来的是 pct(罕见,但理论可能),也登记 ratio
            if is_calc and fact.kind == 'pct':
                _add(NumericFact(
                    raw=fact.raw,
                    value=fact.value,
                    kind='ratio',
                    unit='',
                ))

    return obs_facts


def extract_answer_text(plan: dict) -> str:
    """从 plan 提取最终答案文本(finish step 的 action_input,或 messages 最后一个 assistant)"""
    # 优先从 steps 找 finish
    for step in plan.get('steps', []):
        if step.get('action') == 'finish':
            return step.get('action_input', '')
    # fallback:从 messages 找最后一个无 tool_calls 的 assistant
    for msg in reversed(plan.get('messages', [])):
        if msg.get('role') == 'assistant' and not msg.get('tool_calls'):
            return msg.get('content', '')
    return ''


def used_calculate(plan: dict) -> bool:
    """判断轨迹是否调用过 calculate 工具"""
    return any(step.get('action') == 'calculate' for step in plan.get('steps', []))


def grounding_check(plan: dict) -> list[dict]:
    """
    主检查函数:扫描 plan 的最终答案,标记编造和心算违规。

    Bug 3 修复:reject 类轨迹(0 步 tool_call,answer 中含编码/年份/家数等数字)
    本身就是合理拒答 demo,answer 里的数字与 obs 无溯源关系,跳过检查避免误报。
    适用类型:reject / time_boundary / non_a_share / insuf_L1-L4 / clarify(0 步)/ finance_concept(纯知识)

    Args:
        plan: SFT 轨迹 plan dict

    Returns:
        issues list,每个 dict 含:
          - type: 'fabricated' | 'mental_math'
          - where: 'answer' | 'thought'
          - value: 原始数字字符串
          - severity: 'high' | 'low'
    """
    # Bug 3 修复:reject 类 / 0-1 步轨迹 跳过检查
    # 这些题型本身就不依赖工具检索数据(拒答/概念题/边界拒识),不该用溯源校验
    plan_type = plan.get('type', '')
    plan_subtype = plan.get('subtype', '') or ''
    skip_types = {'reject', 'finance_concept'}
    skip_subtypes = {'time_boundary', 'non_a_share',
                     'insuf_L1', 'insuf_L2', 'insuf_L3', 'insuf_L4',
                     'clarify', 'clarify_neg',
                     'C1_concept', 'C2_formula', 'C3_methodology', 'C4_industry_norm',
                     'C5_distinguish', 'C6_pitfall', 'C7_three_statement', 'C8_calibration'}

    if plan_type in skip_types or plan_subtype in skip_subtypes:
        return []

    # 0 步轨迹也跳过(纯拒答 demo,无 obs 可溯源)
    tool_steps = [s for s in plan.get('steps', []) if s.get('action') != 'finish']
    if len(tool_steps) == 0:
        return []

    obs_facts = collect_obs_facts(plan)
    answer = extract_answer_text(plan)

    if not answer:
        return [{'type': 'no_answer', 'where': 'answer', 'severity': 'high'}]

    issues = []
    has_calculate = used_calculate(plan)

    for num in extract_numerics(answer):
        if num.kind == 'date':
            continue   # 时间一致性独立检查

        status = is_grounded(num, obs_facts)

        if status == 'missing':
            # 纯编造,严重
            issues.append({
                'type': 'fabricated',
                'where': 'answer',
                'value': num.raw,
                'kind': num.kind,
                'severity': 'high',
            })
        elif status == 'derivable' and not has_calculate:
            # 心算了但中间值在 obs:严重性低,数字本身合规
            # 优秀的轨迹应该走 calculate → 直接命中 direct
            issues.append({
                'type': 'mental_math',
                'where': 'answer',
                'value': num.raw,
                'kind': num.kind,
                'severity': 'low',
            })

    return issues


def grounding_check_debug(plan: dict, verbose: bool = True) -> dict:
    """
    调试工具:逐数字打印 5 级判断结果,事后审计用。

    使用:
        from grounding_check import grounding_check_debug
        grounding_check_debug(plan)
    """
    plan_type = plan.get('type', '')
    plan_subtype = plan.get('subtype', '') or ''
    obs_facts = collect_obs_facts(plan)
    answer = extract_answer_text(plan)

    trace = []
    for num in extract_numerics(answer):
        if num.kind == 'date':
            continue
        status = is_grounded(num, obs_facts)
        # 找最近邻 obs 数字(同 kind,数值最接近)便于人工对比
        candidates = sorted(
            [o for o in obs_facts if o.kind == num.kind],
            key=lambda o: abs(o.value - num.value)
        )[:3]
        trace.append({
            'raw': num.raw,
            'value': num.value,
            'kind': num.kind,
            'status': status,
            'closest_obs': [(o.raw, o.value) for o in candidates],
        })

    issues = grounding_check(plan)
    summary = {
        'plan_type': plan_type,
        'plan_subtype': plan_subtype,
        'tool_steps': len([s for s in plan.get('steps', []) if s.get('action') != 'finish']),
        'has_calculate': used_calculate(plan),
        'obs_facts_count': len(obs_facts),
        'answer_len': len(answer),
        'trace': trace,
        'issues': issues,
        'regen_type': decide_regen_type(issues),
    }

    if verbose:
        print(f"=== Grounding Check Debug ===")
        print(f"plan: type={plan_type} subtype={plan_subtype} tool_steps={summary['tool_steps']}")
        print(f"has_calculate={summary['has_calculate']} obs_facts={len(obs_facts)} answer_len={len(answer)}")
        print(f"\n--- 数字逐条追溯 ---")
        for t in trace:
            print(f"  [{t['status']:10s}] {t['raw']:15s} kind={t['kind']:6s} value={t['value']}")
            for r, v in t['closest_obs']:
                print(f"      closest: {r}({v})")
        print(f"\n--- issues ---")
        for i in issues:
            print(f"  {i}")
        print(f"\n--- regen_type --- {summary['regen_type']}")

    return summary


def decide_regen_type(issues: list[dict]) -> Optional[str]:
    """
    根据 grounding_check 的 issues 决定 regen 策略。

    Returns:
      None              通过(无问题)
      'answer_only'     单点心算违规(轻微,只重写答案)
      'full_trajectory' 编造严重(数据本身有问题,重跑整个轨迹)
      'discard'         无答案 / 严重编造 ≥ 5
    """
    if not issues:
        return None

    fabricated = [i for i in issues if i['type'] == 'fabricated']
    mental = [i for i in issues if i['type'] == 'mental_math']

    if any(i.get('type') == 'no_answer' for i in issues):
        return 'discard'

    if len(fabricated) >= 5:
        return 'discard'

    if len(fabricated) >= 1:
        return 'full_trajectory'

    if len(mental) >= 1:
        return 'answer_only'

    return None


# ============ 单元测试(命令行可跑)============

def _self_test():
    """简单单测,验证主要场景"""
    print("=" * 60)
    print("Grounding Check Self Test")
    print("=" * 60)

    # 场景 1:完美轨迹(数字全在 obs)
    plan1 = {
        'steps': [
            {
                'action': 'search_financial',
                'observation': '[financial · profitability | 贵州茅台 | 2025-12-31] ROE 31.8%, 毛利率 91.93%, 营收 1505 亿元',
            },
            {
                'action': 'finish',
                'action_input': '茅台 2025 年报 ROE 31.8%,毛利率 91.93%,营收 1505 亿元',
            },
        ],
    }
    issues1 = grounding_check(plan1)
    print(f"\n[场景 1] 完美轨迹:")
    print(f"  issues: {issues1}")
    assert len(issues1) == 0, "完美轨迹不应该有 issues"
    print("  ✅ 通过")

    # 场景 2:编造数字(用 28% 这种非白名单数字)
    plan2 = {
        'steps': [
            {
                'action': 'search_financial',
                'observation': '[financial · profitability | 贵州茅台 | 2025-12-31] ROE 31.8%, 毛利率 91.93%',
            },
            {
                'action': 'finish',
                'action_input': '茅台 2025 年报 ROE 31.8%,营收增速 28%(observation 没这数字!)',
            },
        ],
    }
    issues2 = grounding_check(plan2)
    print(f"\n[场景 2] 编造员工人数:")
    print(f"  issues: {issues2}")
    fabricated = [i for i in issues2 if i['type'] == 'fabricated']
    assert len(fabricated) >= 1, "应该抓到编造"
    print(f"  ✅ 通过(抓到 {len(fabricated)} 条编造)")

    # 场景 3:心算了但 calculate 未调
    plan3 = {
        'steps': [
            {
                'action': 'search_financial',
                'observation': '[financial | 茅台 | 2025-12-31] 营收 1505 亿元,2024 年营收 1294 亿元',
            },
            {
                'action': 'finish',
                'action_input': '茅台 2025 年营收 1505 亿元,同比 2024 年 1294 亿增长 16.3%',
            },
        ],
    }
    issues3 = grounding_check(plan3)
    print(f"\n[场景 3] 心算了但没用 calculate(中间值在 obs):")
    print(f"  issues: {issues3}")
    mental = [i for i in issues3 if i['type'] == 'mental_math']
    print(f"  抓到 {len(mental)} 条心算违规(严重性 low)")

    # 场景 4:用了 calculate(完美场景)
    plan4 = {
        'steps': [
            {
                'action': 'search_financial',
                'observation': '[financial | 茅台 | 2025-12-31] 营收 1505 亿元,2024 年营收 1294 亿元',
            },
            {
                'action': 'calculate',
                'observation': '计算结果: 16.31',
            },
            {
                'action': 'finish',
                'action_input': '茅台 2025 年营收 1505 亿元,同比 2024 年 1294 亿增长 16.31%',
            },
        ],
    }
    issues4 = grounding_check(plan4)
    print(f"\n[场景 4] 用了 calculate(16.31 在 calculate observation 里):")
    print(f"  issues: {issues4}")
    assert len(issues4) == 0, "用了 calculate 后 16.31 应该 direct 命中"
    print("  ✅ 通过(calculate 返回值进入 obs_facts)")

    # 场景 5:经验阈值白名单
    plan5 = {
        'steps': [
            {
                'action': 'search_financial',
                'observation': '[financial | 茅台 | 2025-12-31] 流动比率 4.83,资产负债率 17.85%',
            },
            {
                'action': 'finish',
                'action_input': '茅台流动比率 4.83(行业经验值 2:1 左右,远高于警戒线)',
            },
        ],
    }
    issues5 = grounding_check(plan5)
    print(f"\n[场景 5] 答案含经验阈值 '2'(白名单):")
    print(f"  issues: {issues5}")
    print(f"  ✅ 通过(2 命中 whitelist,不算编造)")

    # 场景 6:容差(36.99% ≈ 37%)
    plan6 = {
        'steps': [
            {
                'action': 'search_financial',
                'observation': '[financial | 茅台 | 2025-12-31] ROE 36.99%',
            },
            {
                'action': 'finish',
                'action_input': '茅台 ROE 约 37%(精确值 36.99%)',
            },
        ],
    }
    issues6 = grounding_check(plan6)
    print(f"\n[场景 6] 36.99% 写成 37%(容差内):")
    print(f"  issues: {issues6}")
    fabricated = [i for i in issues6 if i['type'] == 'fabricated']
    assert len(fabricated) == 0, "37% 应该 fuzzy 命中 36.99%"
    print(f"  ✅ 通过(容差 ±0.05% 内)")

    # 场景 7:regen_type 决策
    print(f"\n[场景 7] regen_type 决策:")
    print(f"  完美:           {decide_regen_type([])}")
    print(f"  心算 1 条:       {decide_regen_type([{'type': 'mental_math', 'severity': 'low'}])}")
    print(f"  编造 1 条:       {decide_regen_type([{'type': 'fabricated', 'severity': 'high'}])}")
    print(f"  编造 5 条:       {decide_regen_type([{'type': 'fabricated', 'severity': 'high'}] * 5)}")

    # ============ Phase 1 Bug 修复后新增测试 ============

    # 场景 8(Bug 1):裸"亿"无"元",obs 有"亿元",应同源
    plan8 = {
        'steps': [
            {'action': 'search_financial',
             'observation': '[financial | 茅台 | 2025-12-31] 营收 1505 亿元'},
            {'action': 'finish',
             'action_input': '茅台营收 1505 亿(口语化,无元字)'},
        ],
    }
    issues8 = grounding_check(plan8)
    print(f"\n[场景 8] Bug 1 修复:裸'1505 亿'命中 obs '1505 亿元':")
    print(f"  issues: {issues8}")
    fab = [i for i in issues8 if i['type'] == 'fabricated']
    assert len(fab) == 0, "1505 亿 应该和 1505 亿元 同源"
    print("  ✅ 通过(_AMOUNT_RE '元' 改可选)")

    # 场景 9(Bug 2):calculate 返回 ratio 16.31,answer 写 16.31% pct,应命中
    plan9 = {
        'steps': [
            {'action': 'search_financial',
             'observation': '营收 1505 亿元, 2024 营收 1294 亿元'},
            {'action': 'calculate',
             'observation': '计算结果: 16.31'},   # 裸数 ratio
            {'action': 'finish',
             'action_input': '茅台营收同比增长 16.31%'},   # answer 加 % → pct
        ],
    }
    issues9 = grounding_check(plan9)
    print(f"\n[场景 9] Bug 2 修复:calculate 返回值双登记(ratio + pct):")
    print(f"  issues: {issues9}")
    fab = [i for i in issues9 if i['type'] == 'fabricated']
    assert len(fab) == 0, "calculate 16.31 应双登记,answer 16.31% 命中"
    print("  ✅ 通过(calculate ratio 双登记 pct)")

    # 场景 10(Bug 3):reject 类轨迹 0 步,answer 含数字,应跳过
    plan10 = {
        'type': 'reject',
        'subtype': 'non_a_share',
        'steps': [
            {'action': 'finish',
             'action_input': '抱歉,我们的数据库不支持港股 00700 腾讯控股的查询。'},
        ],
    }
    issues10 = grounding_check(plan10)
    print(f"\n[场景 10] Bug 3 修复:reject 类轨迹跳过 grounding_check:")
    print(f"  issues: {issues10}")
    assert len(issues10) == 0, "reject 类应跳过,不报 fabricated"
    print("  ✅ 通过(reject 类跳过)")

    # 场景 10b:finance_concept 类纯知识,跳过
    plan10b = {
        'type': 'finance_concept',
        'subtype': 'C1_concept',
        'steps': [
            {'action': 'finish',
             'action_input': 'ROE 即净资产收益率,公式 ROE = 净利润 / 净资产 × 100%。一般 15% 以上为优秀。'},
        ],
    }
    issues10b = grounding_check(plan10b)
    print(f"\n[场景 10b] finance_concept 类跳过:")
    print(f"  issues: {issues10b}")
    assert len(issues10b) == 0
    print("  ✅ 通过(finance_concept 跳过)")

    # 场景 11(Bug 4):粗略表达,30.66% 写成 30%,容差从 0.05 → 0.5,应通过
    plan11 = {
        'steps': [
            {'action': 'search_financial',
             'observation': '[financial | 茅台 | 2025-12-31] ROE 30.66%'},
            {'action': 'finish',
             'action_input': '茅台 ROE 约 30%(粗略表达)'},
        ],
    }
    issues11 = grounding_check(plan11)
    print(f"\n[场景 11] Bug 4 修复:30% ≈ 30.66%(EPSILON_PCT 0.05→0.5):")
    print(f"  issues: {issues11}")
    fab = [i for i in issues11 if i['type'] == 'fabricated']
    assert len(fab) == 0, "差 0.66 应在 0.5 容差内"
    print("  ✅ 通过(EPSILON_PCT=0.5)")

    # 场景 12(Bug 5):amount 之间相减结果不应误命中 pct target
    plan12 = {
        'steps': [
            {'action': 'search_financial',
             'observation': '茅台营收 1505 亿元, 净利润 745 亿元'},
            {'action': 'finish',
             # 760(凭空)写成 ratio,如果 derivable 不校验 kind,1505-745=760
             # 但 760 也可以是 amount 减出来,实际 760 是真编造数字(无单位,不在 obs)
             'action_input': '茅台某指标为 760'},
        ],
    }
    # 760(ratio kind)应该被标 fabricated(amount-amount=amount,target=ratio,kind 不一致)
    issues12 = grounding_check(plan12)
    print(f"\n[场景 12] Bug 5 修复:derivable 校验 kind 一致(amount 相减不应误命中 ratio):")
    print(f"  issues: {issues12}")
    print("  (760 是 ratio,1505 amount 减 745 amount = 760 amount,kind 不同,应标 fabricated)")
    fab = [i for i in issues12 if i['type'] == 'fabricated' and '760' in str(i.get('value', ''))]
    print(f"  抓到 760 fabricated: {len(fab) > 0}")

    # 场景 13:个百分点(Bug 2 扩展)
    plan13 = {
        'steps': [
            {'action': 'search_financial',
             'observation': '茅台 2024 ROE 36.99%, 2023 ROE 34.07%'},
            {'action': 'finish',
             'action_input': '茅台 ROE 同比提升 2.92 个百分点'},
        ],
    }
    issues13 = grounding_check(plan13)
    print(f"\n[场景 13] '个百分点' 识别为 pct kind,可由两 pct 相减 derivable:")
    print(f"  issues: {issues13}")
    # 36.99 - 34.07 = 2.92,derivable 应命中,但没用 calculate → mental_math
    print(f"  (36.99 - 34.07 = 2.92,derivable 命中 + 无 calculate → mental_math)")

    print(f"\n{'='*60}")
    print("所有 self test 通过 ✅(8 + 7 = 15 场景)")
    print(f"{'='*60}")


if __name__ == '__main__':
    _self_test()
