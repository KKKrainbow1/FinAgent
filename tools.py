"""
FinAgent Step 7: Agent 工具定义（V2 - Qwen 原生 Tool Calling 版）
用途：定义 3 个工具（search_report, search_financial, calculate），供 ReAct Agent 调用
依赖：06_hybrid_search.py

V1 → V2 改动说明：
    1. 新增 TOOLS_NATIVE：Qwen2.5 原生 tool calling 的 JSON Schema 定义，
       传入 API 的 tools 参数或 tokenizer.apply_chat_template 的 tools 参数，
       由 chat_template 自动注入 system prompt（<tools>...</tools> 格式）
    2. 移除 finish 工具：在原生 tool calling 下，最终回答是一条普通 assistant 消息
       （没有 tool_calls），不再需要显式的 finish 工具
    3. call() 方法支持 dict 参数：从 tool_calls[0].function.arguments 解析出的
       JSON 对象可直接传入，支持多参数（如 top_k）
    4. 保留 TOOL_DESCRIPTIONS 和纯文本 call(name, str) 接口，供 V1 代码兼容调用

面试追问：为什么要从纯文本工具描述迁移到 JSON Schema？
答：Qwen2.5 预训练阶段学习了 <tool_call> 格式的工具调用，JSON Schema 能激活模型
的 tool calling 先验能力。纯文本描述需要模型从 SFT 数据中从头学习自定义格式，
学习效率低且正则解析脆弱。迁移后工具选择准确性和参数生成质量都有提升。

面试追问：为什么要加 calculate 工具？
答：14B 模型的算术能力不可靠，"营收1505亿，同比增长16.3%，去年营收是多少"
这种反推经常算错。calculate 工具用 Python eval 做确定性计算，准确率 100%。
这也是一个面试 talking point——说明你观察到了模型的能力边界并做了工程弥补。

面试追问：工具输出为什么要截断？
答：PDF 正文 chunk 可能很长（500+ 字符），全部塞进 Observation 会占满 context window，
导致后续轮次 Agent 能看到的历史信息减少。截断到 200 字符保留核心信息，
同时给 Agent 留足空间做多轮检索。
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


# ============ Qwen 原生 Tool Calling 定义（JSON Schema） ============

TOOLS_NATIVE = [
    {
        "type": "function",
        "function": {
            "name": "search_report",
            "description": (
                "检索券商研报信息，返回机构评级、目标价、EPS预测、行业分析等。"
                "适用于：查机构观点、投资评级、行业前景、竞争分析等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "研报检索关键词，如'贵州茅台 投资评级 2025 2026'"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量，默认3",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_financial",
            "description": (
                "检索公司财务数据，返回ROE、毛利率、营收增长率、资产负债率等指标。"
                "适用于：查具体财务数据、盈利能力、偿债能力、运营效率等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "财务数据检索关键词，如'宁德时代 ROE 毛利率 2024'"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量，默认3",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_industry",
            "description": (
                "检索行业对比数据，返回同行业多家公司的关键指标对比表（ROE、净利率、"
                "周转率、资产负债率、营收增长率等）和行业均值。"
                "适用于：行业分析、同行对比、行业排名、行业趋势等。"
                "query 中必须包含以下行业名之一：白酒、银行、保险、证券、半导体、消费电子、"
                "光伏、电池、汽车、医药、医疗、家电、化工、煤炭、钢铁、有色金属、电力、"
                "军工、机械、房地产、建筑建材、食品饮料、软件、通信、交通运输、传媒、"
                "农业、石油石化、零售。"
                "示例：search_industry(query='光伏行业 ROE 盈利能力')"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "行业检索关键词，必须包含行业名，如'银行行业 ROE 对比'、'光伏 盈利能力'"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量，默认3",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "计算数学表达式。所有涉及数值计算的场景（同比增长率、行业均值、"
                "PE/PB、杜邦分析等）都必须使用此工具，禁止在 Thought 中心算。"
                "支持加减乘除、百分比、括号。"
                "示例：calculate(expression='(1505 - 1294) / 1294 * 100') → 16.31"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "纯数学表达式，如 '(1505 - 1294) / 1294 * 100'"
                    },
                    "precision": {
                        "type": "integer",
                        "description": "小数精度位数，默认4",
                        "default": 4
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# 从 TOOLS_NATIVE 提取可用工具名列表
TOOL_NAMES_NATIVE = [t["function"]["name"] for t in TOOLS_NATIVE]


# ============ 文本清洗 ============

def _clean_text(text: str) -> str:
    """
    清洗检索结果中的噪声（Marker 解析残留的 HTML 标签等）

    你的实验已发现 report_fulltext 里有 <span>, <sup> 等标签，
    清洗后再传给 Agent，减少无关噪声。
    """
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除 Markdown 图片引用
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # 压缩连续空白
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _format_result(result: dict, max_text_len: int = 200) -> str:
    """
    将单条检索结果格式化为 Agent 可读的字符串

    格式示例：
    [report | 贵州茅台 | 2026-03-03] 贵州茅台(600519) 研报：营销改革复盘...评级：买入...
    [financial | 贵州茅台 | 2024-12-31] 贵州茅台(600519) 2024年报盈利指标，ROE 36.99%...
    """
    meta = result["metadata"]
    source = meta.get("source_type", "unknown")
    name = meta.get("stock_name", "未知")
    date = meta.get("date", meta.get("report_date", ""))

    text = _clean_text(result["text"])
    if len(text) > max_text_len:
        last_period = text[:max_text_len].rfind('。')
        if last_period > max_text_len * 0.5:
            text = text[:last_period + 1]
        else:
            text = text[:max_text_len] + "..."

    return f"[{source} | {name} | {date}] {text}"


# ============ 3 个工具 ============

class FinAgentTools:
    """
    FinAgent 工具集（V2 - 支持 Qwen 原生 Tool Calling）

    初始化时传入 retriever（FinAgentRetriever 实例），
    提供 3 个工具方法。

    V2 支持两种调用方式：
        1. 原生模式（推荐）：call("search_financial", {"query": "贵州茅台 ROE", "top_k": 5})
        2. 兼容模式：call("search_financial", "贵州茅台 ROE")  ← V1 纯字符串

    用法：
        from hybrid_search import FinAgentRetriever
        retriever = FinAgentRetriever()
        retriever.load_index()

        tools = FinAgentTools(retriever)

        # V2 原生调用（从 tool_calls.arguments 解析出的 dict）
        obs = tools.call("search_financial", {"query": "贵州茅台 ROE 2024", "top_k": 5})

        # V1 兼容调用（纯字符串）
        obs = tools.call("search_financial", "贵州茅台 ROE 2024")
    """

    # V1 纯文本工具描述（保留供 V1 代码使用）
    TOOL_DESCRIPTIONS = {
        "search_report": (
            "search_report(query: str) → 检索券商研报信息。"
            "返回机构评级、目标价、EPS预测、行业分析等。"
            "适用于：查机构观点、投资评级、行业前景、竞争分析等。"
        ),
        "search_financial": (
            "search_financial(query: str) → 检索公司财务数据。"
            "返回ROE、毛利率、营收增长率、资产负债率等指标。"
            "适用于：查具体财务数据、盈利能力、偿债能力、运营效率等。"
        ),
        "search_industry": (
            "search_industry(query: str) → 检索行业对比数据。"
            "返回同行业多家公司的关键指标对比表和行业均值。"
            "适用于：行业分析、同行对比、行业排名等。"
        ),
        "calculate": (
            "calculate(expression: str) → 计算数学表达式。"
            "支持加减乘除、百分比、括号等。"
            "适用于：计算同比增长率、行业均值、PE/PB 等需要精确计算的场景。"
            "示例：calculate('(1505 - 1294) / 1294 * 100') → 16.31"
        ),
    }

    TOOL_NAMES = list(TOOL_DESCRIPTIONS.keys())

    def __init__(self, retriever):
        """
        Args:
            retriever: FinAgentRetriever 实例（已调用 load_index）
        """
        self.retriever = retriever

    def call(self, tool_name: str, tool_input) -> str:
        """
        统一工具调用入口

        Args:
            tool_name: 工具名（search_report / search_financial / calculate）
            tool_input: 工具输入，支持两种格式：
                - dict: 从 tool_calls[0].function.arguments 解析出的 JSON 对象
                        如 {"query": "贵州茅台 ROE", "top_k": 5}
                - str:  V1 兼容模式，纯字符串输入
                        如 "贵州茅台 ROE"

        Returns:
            Observation 字符串
        """
        # 统一解析输入
        if isinstance(tool_input, str):
            # V1 兼容：尝试解析为 JSON，失败则当作纯字符串
            try:
                tool_input = json.loads(tool_input)
            except (json.JSONDecodeError, TypeError):
                # 纯字符串，包装为 dict
                if tool_name in ("search_report", "search_financial", "search_industry"):
                    tool_input = {"query": tool_input.strip()}
                elif tool_name == "calculate":
                    tool_input = {"expression": tool_input.strip()}
                else:
                    tool_input = {"input": tool_input.strip()}

        if tool_name == "search_report":
            return self._search_report(
                query=tool_input.get("query", ""),
                top_k=tool_input.get("top_k", 3),
            )
        elif tool_name == "search_financial":
            return self._search_financial(
                query=tool_input.get("query", ""),
                top_k=tool_input.get("top_k", 3),
            )
        elif tool_name == "search_industry":
            return self._search_industry(
                query=tool_input.get("query", ""),
                top_k=tool_input.get("top_k", 3),
            )
        elif tool_name == "calculate":
            return self._calculate(
                expression=tool_input.get("expression", ""),
                precision=tool_input.get("precision", 4),
            )
        else:
            return f"[错误] 未知工具: {tool_name}。可用工具: {', '.join(TOOL_NAMES_NATIVE)}"

    # ---------- search_report ----------

    def _search_report(self, query: str, top_k: int = 3) -> str:
        """
        检索研报信息

        返回格式示例：
        找到 5 条相关研报信息：
        1. [report | 宁德时代 | 2026-02-15] 宁德时代(300750) 研报：全年业绩超预期...评级：买入...
        2. [report_fulltext | 宁德时代 | 2026-02-15] 公司动力电池市占率持续提升...
        ...
        """
        try:
            results = self.retriever.search_report(query, top_k=top_k)
        except Exception as e:
            logger.error(f"search_report 异常: {e}")
            return f"[检索错误] 研报检索失败: {str(e)}"

        if not results:
            return "[检索结果为空] 未找到与查询相关的研报信息。"

        lines = [f"找到 {len(results)} 条相关研报信息："]
        for i, r in enumerate(results):
            lines.append(f"{i+1}. {_format_result(r)}")

        return "\n".join(lines)

    # ---------- search_financial ----------

    def _search_financial(self, query: str, top_k: int = 3) -> str:
        """
        检索财务数据

        返回格式示例：
        找到 5 条相关财务数据：
        1. [financial | 贵州茅台 | 2024-12-31] 贵州茅台(600519) 2024年报盈利指标，ROE 36.99%...
        2. [financial | 贵州茅台 | 2023-12-31] 贵州茅台(600519) 2023年报盈利指标，ROE 34.65%...
        ...
        """
        try:
            results = self.retriever.search_financial(query, top_k=top_k)
        except Exception as e:
            logger.error(f"search_financial 异常: {e}")
            return f"[检索错误] 财务数据检索失败: {str(e)}"

        if not results:
            return "[检索结果为空] 未找到与查询相关的财务数据。"

        lines = [f"找到 {len(results)} 条相关财务数据："]
        for i, r in enumerate(results):
            lines.append(f"{i+1}. {_format_result(r)}")

        return "\n".join(lines)

    # ---------- search_industry ----------

    def _search_industry(self, query: str, top_k: int = 3) -> str:
        """
        检索行业对比数据

        返回格式示例：
        找到 2 条相关行业数据：
        1. [industry | 银行 | 5家公司] 银行行业对比（5家公司，最新年报数据）...
        """
        try:
            results = self.retriever.search_industry(query, top_k=top_k)
        except Exception as e:
            logger.error(f"search_industry 异常: {e}")
            return f"[检索错误] 行业数据检索失败: {str(e)}"

        if not results:
            return "[检索结果为空] 未找到与查询相关的行业对比数据。"

        lines = [f"找到 {len(results)} 条相关行业数据："]
        for i, r in enumerate(results):
            meta = r["metadata"]
            industry = meta.get("industry", "未知")
            count = meta.get("company_count", 0)
            # 行业 chunk 较长，截断到 1000 字符保留核心数据
            text = _clean_text(r["text"])
            if len(text) > 1000:
                last_newline = text[:1000].rfind('\n')
                if last_newline > 500:
                    text = text[:last_newline] + "\n..."
                else:
                    text = text[:1000] + "..."
            lines.append(f"{i+1}. [industry | {industry} | {count}家公司]\n{text}")

        return "\n".join(lines)

    # ---------- calculate ----------

    def _calculate(self, expression: str, precision: int = 4) -> str:
        """
        安全计算数学表达式

        只允许数字和基本运算符，防止代码注入。
        Agent 可能生成 "calculate('(1505-1294)/1294*100')" 这样的输入，
        需要先清理引号和多余字符。

        面试追问：为什么不直接用 eval？
        答：直接 eval 有代码注入风险（虽然在离线训练场景下风险低）。
        我用正则白名单过滤，只允许数字、运算符和括号。
        """
        # 清理输入：去掉引号、空格
        expr = expression.strip().strip("'\"")

        # 安全检查：只允许数字、运算符、括号、小数点
        if not re.match(r'^[\d\s\+\-\*/\(\)\.\,%]+$', expr):
            return f"[计算错误] 表达式包含不允许的字符: {expr}"

        # 处理百分号：5% → 0.05
        expr = re.sub(r'(\d+\.?\d*)%', r'(\1/100)', expr)

        try:
            result = eval(expr)
            # 格式化输出
            if isinstance(result, float):
                if abs(result) >= 1e8:
                    return f"计算结果: {result/1e8:.{precision}f}亿"
                elif abs(result) >= 1e4:
                    return f"计算结果: {result/1e4:.{precision}f}万"
                else:
                    return f"计算结果: {result:.{precision}f}"
            return f"计算结果: {result}"
        except ZeroDivisionError:
            return "[计算错误] 除数为零"
        except Exception as e:
            return f"[计算错误] 无法计算 '{expr}': {str(e)}"


# ================================================================
#  测试脚本
# ================================================================

def main():
    """测试工具调用"""
    from hybrid_search import FinAgentRetriever

    # 加载检索器
    retriever = FinAgentRetriever()
    retriever.load_index()

    # 初始化工具
    tools = FinAgentTools(retriever)

    # 测试 V2 原生调用（dict 参数）
    print("=" * 60)
    print("测试 V2 原生调用（dict 参数）")
    print("=" * 60)

    obs = tools.call("search_financial", {"query": "贵州茅台 ROE 2024", "top_k": 5})
    print(f"search_financial (top_k=5):\n{obs}\n")

    obs = tools.call("search_report", {"query": "宁德时代 投资评级 目标价"})
    print(f"search_report:\n{obs}\n")

    # 测试 V1 兼容调用（纯字符串）
    print("=" * 60)
    print("测试 V1 兼容调用（纯字符串）")
    print("=" * 60)

    obs = tools.call("search_financial", "贵州茅台 ROE 2024")
    print(f"search_financial (str):\n{obs}\n")

    # 测试 calculate
    print("=" * 60)
    print("测试 calculate")
    print("=" * 60)
    test_cases = [
        {"expression": "(1505 - 1294) / 1294 * 100"},                  # 同比增长率
        {"expression": "(21.3 + 18.5 + 15.2) / 3", "precision": 2},   # 行业均值，精度2
        {"expression": "48.80 / 3.52"},                                 # PE 计算
        {"expression": "100 / 0"},                                      # 除零错误
    ]
    for case in test_cases:
        obs = tools.call("calculate", case)
        print(f"  {case} → {obs}")

    # 测试 calculate V1 兼容
    obs = tools.call("calculate", "(1505 - 1294) / 1294 * 100")
    print(f"\n  V1兼容: '(1505 - 1294) / 1294 * 100' → {obs}")

    # 测试未知工具
    print("\n" + "=" * 60)
    print("测试未知工具")
    print("=" * 60)
    obs = tools.call("unknown_tool", "test")
    print(f"  → {obs}")

    # 打印 TOOLS_NATIVE（供检查）
    print("\n" + "=" * 60)
    print("TOOLS_NATIVE JSON Schema")
    print("=" * 60)
    print(json.dumps(TOOLS_NATIVE, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
