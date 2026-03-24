"""
FinAgent Step 7: Agent 工具定义
用途：定义 4 个工具（search_report, search_financial, calculate, finish），供 ReAct Agent 调用
依赖：06_hybrid_search.py

工具设计原则：
    1. 每个工具输入都是字符串（Agent 生成的 action_args）
    2. 每个工具输出都是字符串（拼进 Observation 给 Agent 看）
    3. 输出要简洁但信息完整，太长会占用 context window
    4. 异常情况返回友好的错误提示，不要让 Agent 看到 Python traceback

面试追问：为什么要加 calculate 工具？
答：14B 模型的算术能力不可靠，"营收1505亿，同比增长16.3%，去年营收是多少"
这种反推经常算错。calculate 工具用 Python eval 做确定性计算，准确率 100%。
这也是一个面试 talking point——说明你观察到了模型的能力边界并做了工程弥补。

面试追问：工具输出为什么要截断？
答：PDF 正文 chunk 可能很长（500+ 字符），全部塞进 Observation 会占满 context window，
导致后续轮次 Agent 能看到的历史信息减少。截断到 200 字符保留核心信息，
同时给 Agent 留足空间做多轮检索。
"""

import re
import logging

logger = logging.getLogger(__name__)


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
        text = text[:max_text_len] + "..."

    return f"[{source} | {name} | {date}] {text}"


# ============ 4 个工具 ============

class FinAgentTools:
    """
    FinAgent 工具集

    初始化时传入 retriever（FinAgentRetriever 实例），
    提供 4 个工具方法，每个工具接受字符串输入、返回字符串输出。

    用法：
        from hybrid_search import FinAgentRetriever
        retriever = FinAgentRetriever()
        retriever.load_index()

        tools = FinAgentTools(retriever)
        obs = tools.call("search_financial", "贵州茅台 ROE 2024")
    """

    # 工具名 → 工具描述（供 prompt 使用）
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
        "calculate": (
            "calculate(expression: str) → 计算数学表达式。"
            "支持加减乘除、百分比、括号等。"
            "适用于：计算同比增长率、行业均值、PE/PB 等需要精确计算的场景。"
            "示例：calculate('(1505 - 1294) / 1294 * 100') → 16.31"
        ),
        "finish": (
            "finish(answer: str) → 输出最终分析报告并结束。"
            "当已获取足够信息时调用，answer 为完整的分析报告文本。"
        ),
    }

    # 可用工具名列表
    TOOL_NAMES = list(TOOL_DESCRIPTIONS.keys())

    def __init__(self, retriever):
        """
        Args:
            retriever: FinAgentRetriever 实例（已调用 load_index）
        """
        self.retriever = retriever

    def call(self, tool_name: str, tool_input: str) -> str:
        """
        统一工具调用入口

        Args:
            tool_name: 工具名（search_report / search_financial / calculate / finish）
            tool_input: 工具输入（字符串）

        Returns:
            Observation 字符串，拼进 Agent 的 context
        """
        tool_input = tool_input.strip()

        if tool_name == "search_report":
            return self._search_report(tool_input)
        elif tool_name == "search_financial":
            return self._search_financial(tool_input)
        elif tool_name == "calculate":
            return self._calculate(tool_input)
        elif tool_name == "finish":
            return self._finish(tool_input)
        else:
            return f"[错误] 未知工具: {tool_name}。可用工具: {', '.join(self.TOOL_NAMES)}"

    # ---------- search_report ----------

    def _search_report(self, query: str, top_k: int = 5) -> str:
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

    def _search_financial(self, query: str, top_k: int = 5) -> str:
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

    # ---------- calculate ----------

    def _calculate(self, expression: str) -> str:
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
                    return f"计算结果: {result/1e8:.2f}亿"
                elif abs(result) >= 1e4:
                    return f"计算结果: {result/1e4:.2f}万"
                else:
                    return f"计算结果: {result:.4f}"
            return f"计算结果: {result}"
        except ZeroDivisionError:
            return "[计算错误] 除数为零"
        except Exception as e:
            return f"[计算错误] 无法计算 '{expr}': {str(e)}"

    # ---------- finish ----------

    def _finish(self, answer: str) -> str:
        """
        结束工具 —— 直接返回答案文本

        finish 的特殊之处：它不产生 Observation，而是标记轨迹结束。
        在 react_agent.py 中会检测到 finish 并跳出循环。
        这里只做简单的透传。
        """
        return answer


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

    # 测试 search_financial
    print("=" * 60)
    print("测试 search_financial")
    print("=" * 60)
    obs = tools.call("search_financial", "贵州茅台 ROE 2024")
    print(obs)

    # 测试 search_report
    print("\n" + "=" * 60)
    print("测试 search_report")
    print("=" * 60)
    obs = tools.call("search_report", "宁德时代 投资评级 目标价")
    print(obs)

    # 测试 calculate
    print("\n" + "=" * 60)
    print("测试 calculate")
    print("=" * 60)
    test_exprs = [
        "(1505 - 1294) / 1294 * 100",   # 同比增长率
        "(21.3 + 18.5 + 15.2) / 3",     # 行业均值
        "48.80 / 3.52",                  # PE 计算
        "100 / 0",                       # 除零错误
    ]
    for expr in test_exprs:
        obs = tools.call("calculate", expr)
        print(f"  {expr} → {obs}")

    # 测试 finish
    print("\n" + "=" * 60)
    print("测试 finish")
    print("=" * 60)
    obs = tools.call("finish", "贵州茅台是一家优秀的白酒企业...")
    print(f"  finish → {obs[:80]}...")

    # 测试未知工具
    print("\n" + "=" * 60)
    print("测试未知工具")
    print("=" * 60)
    obs = tools.call("unknown_tool", "test")
    print(f"  → {obs}")


if __name__ == "__main__":
    main()
