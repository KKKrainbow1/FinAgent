"""
FinAgent Reward V4.1 知识库

所有 reward 维度共享的映射表、关键词库、提取函数。
被 grpo_plugin.py（V4.1 reward）和 reward_calibration 脚本引用。

设计文档：docs/FinAgent_Reward_V4.1_Design.md
"""

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Set, List, Dict, Optional, Tuple


# ============================================================
# 一、金融指标-分析维度映射（V4.1 Section 3.1）
# ============================================================

DIMENSION_CONFIG = {
    "盈利能力": {
        "keywords": ["盈利", "利润", "赚钱"],
        "metrics": [
            "ROE", "roe", "净利率", "毛利率", "营业利润率",
            "净资产收益率", "ROIC", "ROA", "roa",
        ],
    },
    "偿债/杠杆": {
        "keywords": ["偿债", "风险", "杠杆", "负债", "稳健"],
        "metrics": [
            "资产负债率", "流动比率", "速动比率",
            "利息保障倍数", "权益乘数",
        ],
    },
    "营运效率": {
        "keywords": ["营运", "运营", "效率", "周转"],
        "metrics": [
            "总资产周转率", "存货周转率", "应收账款周转率",
            "周转天数", "周转率",
        ],
    },
    "成长性": {
        "keywords": ["成长", "增长", "发展"],
        "metrics": [
            "营收增长率", "净利润增长率", "同比", "环比",
            "CAGR", "复合增长",
        ],
    },
    "估值": {
        "keywords": ["估值", "价值", "贵不贵"],
        "metrics": [
            "PE", "PB", "PS", "EPS",
            "市盈率", "市净率", "PEG",
        ],
    },
    "分红/现金流": {
        "keywords": ["分红", "股息", "现金流"],
        "metrics": [
            "股息率", "分红比例", "经营性现金流", "自由现金流",
        ],
    },
}

# ---- 自动生成的派生数据 ----

# 扁平化的全指标关键词列表
ALL_METRIC_KEYWORDS: List[str] = []
for _config in DIMENSION_CONFIG.values():
    ALL_METRIC_KEYWORDS.extend(_config["metrics"])

# 指标 → 维度 反向映射
METRIC_TO_DIMENSION: Dict[str, str] = {}
for _dim_name, _config in DIMENSION_CONFIG.items():
    for _metric in _config["metrics"]:
        METRIC_TO_DIMENSION[_metric] = _dim_name

# 有效工具集
VALID_TOOLS = {"search_financial", "search_report", "search_industry", "calculate"}


# ============================================================
# 二、question_type 工具需求配置（V4.1 Section 3.2）
# ============================================================

# 每个 check 项的结构：
#   check: 检查类型（字符串，对应 _check_dispatch 中的函数名）
#   weight: 权重
#   tool / tools: 工具名（部分 check 类型需要）
#   condition: 触发条件（conditional 级别才有，如 "needs_calc"）

TOOL_REQUIREMENTS = {
    "single_company_medium": {
        "must": [
            {"check": "has_tool", "tool": "search_financial", "weight": 0.40},
        ],
        "should": [
            {"check": "has_any_tool", "tools": ["search_report", "search_industry"], "weight": 0.20},
            {"check": "query_has_metric", "weight": 0.10},
        ],
        "conditional": [
            {"check": "has_tool", "tool": "calculate", "condition": "needs_calc", "weight": 0.30},
        ],
    },
    "company_comparison": {
        "must": [
            {"check": "unique_companies_ge_2", "weight": 0.60},
        ],
        "should": [
            {"check": "has_any_tool", "tools": ["search_industry", "calculate"], "weight": 0.20},
            {"check": "query_has_metric", "weight": 0.10},
            {"check": "companies_in_separate_queries", "weight": 0.10},
        ],
        "conditional": [],
    },
    "risk_analysis": {
        "must": [
            {"check": "has_tool", "tool": "search_financial", "weight": 0.40},
        ],
        "should": [
            {"check": "has_any_tool", "tools": ["search_report", "search_industry"], "weight": 0.20},
            {"check": "query_has_leverage_metric", "weight": 0.20},
        ],
        "conditional": [
            {"check": "has_tool", "tool": "calculate", "condition": "needs_calc", "weight": 0.20},
        ],
    },
    "industry_analysis": {
        "must": [
            {"check": "has_tool", "tool": "search_industry", "weight": 0.40},
        ],
        "should": [
            {"check": "has_tool", "tool": "search_financial", "weight": 0.20},
            {"check": "has_tool", "tool": "search_report", "weight": 0.20},
            {"check": "query_has_industry_name", "weight": 0.20},
        ],
        "conditional": [],
    },
    "financial_query": {
        "must": [
            {"check": "has_tool", "tool": "search_financial", "weight": 0.50},
        ],
        "should": [
            {"check": "query_has_metric", "weight": 0.30},
            {"check": "query_has_company_name", "weight": 0.20},
        ],
        "conditional": [],
    },
    "single_company_simple": {
        "must": [
            {"check": "has_tool", "tool": "search_report", "weight": 0.50},
        ],
        "should": [
            {"check": "query_has_company_and_keyword", "weight": 0.30},
            {"check": "has_tool", "tool": "search_financial", "weight": 0.20},
        ],
        "conditional": [],
    },
    # reject 类型有独立评分逻辑，不用此配置
}


# ============================================================
# 三、needs_calc 判定（V4.1 Section 3.3）
# ============================================================

# 强触发：单独出现即判定 needs_calc = True
CALC_STRONG_TRIGGERS = [
    "杜邦", "拆解", "分解",
    "增长率", "同比增长", "环比增长",
    "差值", "差距",
    "比率", "比值",
    "加权", "平均",
    "复合增长", "CAGR",
]

# 心算检测正则（只检查涉及运算符+等号的模式，避免误匹配引用数据）
MENTAL_CALC_PATTERNS = [
    re.compile(r'[\d.]+%?\s*[×*]\s*[\d.]+%?\s*[×*]\s*[\d.]+'),     # 杜邦三因子 A×B×C
    re.compile(r'1\s*/\s*[\d.]+%?\s*[=≈]\s*[\d.]+'),                # 权益乘数 1/X%≈Y
    re.compile(r'[\d.]+\s*[-−]\s*[\d.]+\s*[=≈]\s*[\d.]+'),          # 差值 A-B=C
    re.compile(r'[\d.]+\s*[/÷]\s*[\d.]+\s*[=≈]\s*[\d.]+'),          # 除法 A/B=C
    re.compile(r'[\d.]+\s*[+]\s*[\d.]+\s*[=≈]\s*[\d.]+'),           # 加法 A+B=C
]


def check_needs_calc(question: str, answer: str = "") -> bool:
    """
    判断是否需要 calculate（双路径）。

    路径 1a：question 中有强触发关键词
    路径 2：answer 中检测到心算模式
    路径 1b（弱触发）：暂不启用，后续补充

    Returns:
        True 如果判定"需要计算"
    """
    # 路径 1a：强触发
    for keyword in CALC_STRONG_TRIGGERS:
        if keyword in question:
            return True

    # 路径 1b：弱触发（实验 1 阶段暂不启用）
    # TODO: 补充 requires_two_entities / requires_time_range / requires_numeric_context

    # 路径 2：answer 心算检测
    if answer:
        for pattern in MENTAL_CALC_PATTERNS:
            if pattern.search(answer):
                return True

    return False


def detect_mental_calc(answer: str) -> bool:
    """单独的心算检测（用于 calc_behavior 评分）"""
    if not answer:
        return False
    return any(p.search(answer) for p in MENTAL_CALC_PATTERNS)


# ============================================================
# 四、维度提取（V4.1 Section 3.4）
# ============================================================

COMPREHENSIVE_KEYWORDS = [
    "全面评估", "综合分析", "财务状况", "投资价值",
    "整体", "全方位", "全面分析",
]


def extract_dimensions(question: str) -> Set[str]:
    """
    从 question 中提取涉及的分析维度集合。

    两条提取路径取并集：
    - 路径 1：匹配维度关键词（如"盈利"→盈利能力）
    - 路径 2：匹配具体指标名，反查维度（如"ROE"→盈利能力）

    特殊规则：
    - "全面评估"等 → 扩展为 {盈利能力, 偿债/杠杆, 营运效率}
    - "杜邦"/"ROE 拆解" → {盈利能力, 营运效率, 偿债/杠杆}
    - 未匹配到 → 返回空集（调用方跳过匹配检查）
    """
    # 特殊规则优先
    if any(kw in question for kw in COMPREHENSIVE_KEYWORDS):
        return {"盈利能力", "偿债/杠杆", "营运效率"}
    if "杜邦" in question or "ROE拆解" in question or "ROE 拆解" in question:
        return {"盈利能力", "营运效率", "偿债/杠杆"}

    dims = set()

    # 路径 1：维度关键词
    for dim_name, config in DIMENSION_CONFIG.items():
        for kw in config["keywords"]:
            if kw in question:
                dims.add(dim_name)

    # 路径 2：指标名 → 反查维度
    for metric, dim_name in METRIC_TO_DIMENSION.items():
        if metric in question:
            dims.add(dim_name)

    return dims


def is_comprehensive(question: str) -> bool:
    """判断是否是全面分析类问题"""
    return any(kw in question for kw in COMPREHENSIVE_KEYWORDS)


# ============================================================
# 五、Query 分析工具函数
# ============================================================

def count_metrics_in_query(query: str) -> int:
    """统计 query 中包含的具体指标名数量"""
    count = 0
    found = set()
    for metric in ALL_METRIC_KEYWORDS:
        if metric in query and metric not in found:
            count += 1
            found.add(metric)
    return count


def query_has_metric(query: str) -> bool:
    """query 是否包含至少一个具体指标名"""
    return count_metrics_in_query(query) > 0


def query_has_year(query: str) -> bool:
    """query 是否包含年份（2020-2029）"""
    return bool(re.search(r'20[2]\d', query))


# 常见非公司名前缀（分析动词、描述词等，不应被识别为公司名）
_NON_COMPANY_PREFIXES = [
    "分析", "评估", "对比", "比较", "查询", "搜索", "投资", "全面",
    "综合", "深度", "简要", "详细", "最新", "目前", "当前", "如何",
    "怎么", "价值", "风险", "盈利", "偿债", "成长", "估值",
]

_NON_COMPANY_WORDS = [
    "盈利能力", "偿债能力", "营运效率", "成长性", "财务状况",
    "投资价值", "风险分析", "行业分析", "价值分析", "能力指标",
    "能力", "分析", "指标", "数据", "评估", "状况", "趋势",
]


@lru_cache(maxsize=1)
def _load_known_stock_names() -> Tuple[str, ...]:
    """
    加载已知股票名列表，优先用真实股票名做公司识别。

    当前仓库是沪深300数据管线，优先读取本地 hs300 股票池。
    文件缺失时退化为空列表，继续走规则兜底。
    """
    csv_path = Path(__file__).resolve().parent / "data/raw/hs300_stocks.csv"
    if not csv_path.exists():
        return tuple()

    names = set()
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = (row.get("stock_name") or "").strip()
            if name:
                names.add(name)

    return tuple(sorted(names, key=lambda x: (-len(x), x)))


def query_has_company_name(query: str) -> bool:
    """
    query 是否包含公司名。

    比 ≥4 字连续中文更严格：排除常见非公司名前缀。
    """
    name = extract_company_name(query)
    if not name:
        return False

    known_names = set(_load_known_stock_names())
    if name in known_names:
        return True

    # 规则兜底时放宽到 3 字，兼容"比亚迪"这类三字公司名
    return len(name) >= 3


def extract_company_name(query: str) -> Optional[str]:
    """
    从 query 开头提取公司名（取前 2-8 个连续汉字）。

    排除常见分析动词开头的情况：
    - "分析贵州茅台盈利能力" → 跳过"分析"，提取"贵州茅台"
    - "格力电器 ROE 2024" → 提取"格力电器"
    - "投资价值分析" → 跳过"投资"，剩余"价值分析"不是公司名 → None
    """
    text = query.strip()

    # 循环去掉开头的非公司名前缀（可能叠加："全面评估" → 去"全面" → 去"评估"）
    changed = True
    while changed:
        changed = False
        for prefix in _NON_COMPANY_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                changed = True
                break

    if not text:
        return None

    # 先用已知股票名精确匹配，避免"招商银行"这类真实公司被行业词规则误伤
    for stock_name in _load_known_stock_names():
        if text.startswith(stock_name):
            return stock_name

    match = re.match(r'([\u4e00-\u9fa5]{2,8})', text)
    if not match:
        return None

    name = match.group(1)

    # 如果提取的"名字"包含分析维度词或常见非公司词，截断或排除
    # 尝试在 name 中找到非公司词的位置并截断
    for nw in _NON_COMPANY_WORDS:
        pos = name.find(nw)
        if pos >= 0:
            name = name[:pos]
            break

    # 截断后如果剩余 < 2 字，认为不是公司名
    if len(name) < 2:
        return None

    # 只排除"行业短语本身"，不要误杀"招商银行"这类包含行业词的真实公司名
    if name in INDUSTRY_NAMES or name.endswith(("行业", "板块")):
        return None
    if text.startswith(name + "行业") or text.startswith(name + "板块"):
        return None

    if name in _NON_COMPANY_PREFIXES or name in _NON_COMPANY_WORDS:
        return None

    return name


def count_unique_companies(tool_steps: list) -> set:
    """
    统计 search_financial / search_report query 中出现的不同公司名。

    只从这两类工具的 query 中提取，避免 search_industry 的行业名被误判。
    返回公司名集合。
    """
    names = set()
    for step in tool_steps:
        if step.get("tool") in ("search_financial", "search_report"):
            name = extract_company_name(step.get("query", ""))
            if name:
                names.add(name)
    return names


def extract_metrics_from_queries(queries: List[str]) -> Set[str]:
    """从多条 query 中提取所有出现的指标名"""
    found = set()
    for q in queries:
        for metric in ALL_METRIC_KEYWORDS:
            if metric in q:
                found.add(metric)
    return found


def get_dimensions_from_metrics(metrics: Set[str]) -> Set[str]:
    """从指标集合反查维度集合"""
    dims = set()
    for m in metrics:
        if m in METRIC_TO_DIMENSION:
            dims.add(METRIC_TO_DIMENSION[m])
    return dims


def query_has_leverage_metric(query: str) -> bool:
    """query 是否包含偿债/杠杆类指标"""
    leverage_metrics = DIMENSION_CONFIG["偿债/杠杆"]["metrics"]
    return any(m in query for m in leverage_metrics)


# ============================================================
# 六、行业名称库
# ============================================================

@lru_cache(maxsize=1)
def _build_industry_names() -> Tuple[str, ...]:
    """
    从 hybrid_search.py 的真实 alias 源自动生成行业名称集合。

    这样 reward 和检索器永远共用同一套行业支持范围，避免手写列表漂移。
    """
    from hybrid_search import FinAgentRetriever

    names = set()

    for standard_name, sub_industries in FinAgentRetriever._INDUSTRY_MAP.items():
        names.add(standard_name)
        for sub in sub_industries:
            sub = sub.strip()
            if not sub:
                continue
            names.add(sub)
            clean = sub.replace("Ⅱ", "").strip()
            if clean:
                names.add(clean)

    for aliases in FinAgentRetriever._EXTRA_ALIASES.values():
        for alias in aliases:
            alias = alias.strip()
            if alias:
                names.add(alias)

    return tuple(sorted(names, key=lambda x: (-len(x), x)))


INDUSTRY_NAMES = _build_industry_names()


def query_has_industry_name(query: str) -> bool:
    """query 是否包含行业名称"""
    return any(name in query for name in INDUSTRY_NAMES)


# ============================================================
# 七、Query 质量评分
# ============================================================

def query_length_score(query: str) -> float:
    """
    query 长度合理性评分。

    8-30 字符 → 1.0（最佳区间）
    5-7 或 31-50 → 0.5（偏短/偏长但可接受）
    其他 → 0.0
    """
    n = len(query)
    if 8 <= n <= 30:
        return 1.0
    elif (5 <= n <= 7) or (31 <= n <= 50):
        return 0.5
    else:
        return 0.0


def compute_single_query_score(query: str, tool_name: str) -> float:
    """
    计算单条 query 的质量得分（0-1）。

    子项权重：
    - 包含具体指标名：0.35（计数制：0个→0, 1个→0.20, 2个→0.30, 3+→0.35）
    - 包含年份：0.25
    - 包含公司全称或行业名：0.20
    - 长度合理性：0.20

    search_industry 特殊处理：
    - "包含准确行业名"权重从 0.20 提升到 0.35，其他子项等比缩小
    """
    if tool_name == "calculate":
        # calculate 的 query 是数学表达式，不用这个评分逻辑
        return 0.5  # 中性值，由 calc_behavior 维度单独评

    # ---- 子项评分 ----

    # 指标名（计数制）
    metric_count = count_metrics_in_query(query)
    if metric_count >= 3:
        metric_score = 1.0
    elif metric_count == 2:
        metric_score = 0.85
    elif metric_count == 1:
        metric_score = 0.57
    else:
        metric_score = 0.0

    # 年份
    year_score = 1.0 if query_has_year(query) else 0.0

    # 公司名或行业名
    entity_score = 0.0
    if query_has_company_name(query):
        entity_score = 1.0
    elif extract_company_name(query):  # 有短名（2-3 字）
        entity_score = 0.5

    # 行业名（search_industry 用）
    industry_score = 1.0 if query_has_industry_name(query) else 0.0

    # 长度
    length_score = query_length_score(query)

    # ---- 加权 ----
    if tool_name == "search_industry":
        # search_industry 精确匹配，行业名权重提升
        score = (
            0.25 * metric_score +
            0.20 * year_score +
            0.35 * industry_score +   # 权重提升
            0.20 * length_score
        )
    else:
        # search_financial / search_report
        score = (
            0.35 * metric_score +
            0.25 * year_score +
            0.20 * entity_score +
            0.20 * length_score
        )

    return score


# ============================================================
# 八、Reject 类型辅助
# ============================================================

REJECT_KEYWORDS = [
    "无法回答", "抱歉", "无法提供", "超出范围",
    "不在数据库", "没有相关数据", "无法分析",
    "不具备", "超出能力", "不在覆盖范围",
]


def is_reject_response(answer: str) -> bool:
    """检查最终输出是否包含拒绝类关键词"""
    if not answer:
        return False
    return any(kw in answer for kw in REJECT_KEYWORDS)


# ============================================================
# 九、指标-维度匹配检查（Anti-Hacking, V4.1 Section 9）
# ============================================================

def compute_metric_dimension_match(query: str, target_dims: Set[str]) -> float:
    """
    计算 query 中指标与问题目标维度的匹配率。

    返回匹配率 0-1。如果 query 中无指标或目标维度为空，返回 1.0（不惩罚）。
    """
    if not target_dims:
        return 1.0  # 未识别到目标维度，跳过检查

    query_metrics = extract_metrics_from_queries([query])
    if not query_metrics:
        return 1.0  # query 中无具体指标名，不做匹配检查

    query_dims = get_dimensions_from_metrics(query_metrics)
    if not query_dims:
        return 1.0

    matched = query_dims & target_dims
    match_rate = len(matched) / len(query_dims)
    return match_rate


def apply_anti_hacking_penalty(query_score: float, match_rate: float) -> float:
    """
    如果匹配率 < 0.5（超过一半的指标与问题维度不匹配），
    对 query_quality 得分乘以 0.5 作为惩罚。
    """
    if match_rate < 0.5:
        return query_score * 0.5
    return query_score


# ============================================================
# 十、tool_coverage 检查分发器
# ============================================================

def check_item(item: dict, tool_steps: list, queries: List[str],
               needs_calc: bool) -> float:
    """
    根据 check 项的类型执行对应检查，返回 0.0 或 1.0（或中间值）。

    Args:
        item: TOOL_REQUIREMENTS 中的一个 check 项
        tool_steps: 环境记录的工具调用列表
        queries: 所有 query 文本列表
        needs_calc: 是否需要计算
    """
    check_type = item["check"]
    tools_used = [s.get("tool") for s in tool_steps]

    if check_type == "has_tool":
        return 1.0 if item["tool"] in tools_used else 0.0

    elif check_type == "has_any_tool":
        return 1.0 if any(t in tools_used for t in item["tools"]) else 0.0

    elif check_type == "unique_companies_ge_2":
        # company_comparison 的 graded must：≥2→1.0, =1→0.5, =0→0.0
        companies = count_unique_companies(tool_steps)
        n = len(companies)
        if n >= 2:
            return 1.0
        elif n == 1:
            return 0.5
        else:
            return 0.0

    elif check_type == "companies_in_separate_queries":
        # 检查是否有单条 query 同时包含两家公司名
        companies = count_unique_companies(tool_steps)
        if len(companies) < 2:
            return 0.0
        company_list = list(companies)
        for q in queries:
            if sum(1 for c in company_list if c in q) >= 2:
                return 0.0  # 两家公司合并在同一条 query 中
        return 1.0

    elif check_type == "query_has_metric":
        # 任意一条 query 包含指标名即满足
        return 1.0 if any(query_has_metric(q) for q in queries) else 0.0

    elif check_type == "query_has_leverage_metric":
        return 1.0 if any(query_has_leverage_metric(q) for q in queries) else 0.0

    elif check_type == "query_has_industry_name":
        return 1.0 if any(query_has_industry_name(q) for q in queries) else 0.0

    elif check_type == "query_has_company_name":
        return 1.0 if any(query_has_company_name(q) for q in queries) else 0.0

    elif check_type == "query_has_company_and_keyword":
        # single_company_simple：query 包含公司名 + 研报相关关键词
        report_keywords = ["评级", "目标价", "前景", "投资", "研报", "推荐", "买入", "增持"]
        for q in queries:
            has_company = bool(extract_company_name(q))
            has_keyword = any(kw in q for kw in report_keywords)
            if has_company and has_keyword:
                return 1.0
        return 0.0

    else:
        # 未知 check 类型，返回 0 避免加分
        return 0.0


def compute_tool_coverage(type_config: dict, tool_steps: list,
                          queries: List[str], needs_calc: bool) -> float:
    """
    计算工具覆盖完整性得分。

    score = 各满足项的权重之和 / 所有适用项的权重之和
    conditional 项只在条件满足时才计入分母。
    """
    total_weight = 0.0
    earned_weight = 0.0

    for item in type_config.get("must", []):
        w = item["weight"]
        total_weight += w
        result = check_item(item, tool_steps, queries, needs_calc)
        earned_weight += w * result

    for item in type_config.get("should", []):
        w = item["weight"]
        total_weight += w
        result = check_item(item, tool_steps, queries, needs_calc)
        earned_weight += w * result

    for item in type_config.get("conditional", []):
        condition = item.get("condition", "")
        if condition == "needs_calc" and not needs_calc:
            continue  # 条件不满足，不计入分母
        w = item["weight"]
        total_weight += w
        result = check_item(item, tool_steps, queries, needs_calc)
        earned_weight += w * result

    if total_weight == 0:
        return 0.5  # 无检查项时给中间值

    return earned_weight / total_weight
