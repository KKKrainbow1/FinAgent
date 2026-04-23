"""
批量 smoke test:跑 20 条 query,人工验证 general 质量。

覆盖维度:
  - 三个工具(search_financial / search_industry / search_report)各 6-7 条
  - 期间关键词 match / 不 match / 混合三类(验证 time boost P0d)
  - 不同行业、不同股票(泛化)
  - 边界 case(空期间、多期间)

用法:
    python eval_retrieval_smoke.py
"""
import logging

logging.basicConfig(level=logging.WARNING)   # 只看结果,不看 Milvus 内部 log

from hybrid_search import FinAgentRetriever


TEST_CASES = [
    # ============ search_financial(6 条:数据查询) ============
    ("financial", "茅台 2024 年 ROE",              "股票 + 年份 + 指标,预期 ROE 37% 附近"),
    ("financial", "宁德时代 2024 年营收",           "股票 + 年份 + 指标"),
    ("financial", "平安银行 2025 净利润",          "2025 但无期间词,期望年度数据"),
    ("financial", "美的集团 2024 毛利率",           "消费股,毛利率应 20-30%"),
    ("financial", "比亚迪 2024 资产负债率",         "新能源,负债率应 70%+"),
    ("financial", "招商银行 2025 ROE",             "银行股"),

    # ============ search_industry(5 条:行业对比) ============
    ("industry",  "白酒行业对比",                   "已知 satisfied case"),
    ("industry",  "新能源车行业",                   "别名 → 汽车整车"),
    ("industry",  "半导体产业链",                   "别名 → 半导体"),
    ("industry",  "银行业对比",                     "应 match 股份制银行或城商行"),
    ("industry",  "消费电子",                       "别名"),

    # ============ search_report(9 条:研报文本) ============
    # time boost 核心验证(含期间关键词)
    ("report",    "平安银行 2025 三季报营收",       "★ P0d 目标:Q3 pdf 必须 top-1"),
    ("report",    "贵州茅台 2024 年报业绩",          "年报期间,boost 应 match 2024 年报 pdf"),
    ("report",    "宁德时代 2025 中报",              "中报期间"),
    ("report",    "招商银行 2025 三季报",            "另一股票的 Q3"),
    ("report",    "美的集团 2024 年报",              "年报"),
    ("report",    "比亚迪 2025 半年报",              "半年报 = h1"),
    # 无期间词 control(验证 time boost 不影响普通 query)
    ("report",    "平安银行营收",                    "无期间词,time_boosts={},不加分"),
    ("report",    "茅台业绩表现",                    "无期间词,普通 query"),
    # 多期间词(边界)
    ("report",    "宁德时代 2025 三季报 利润",       "同 Q3,带额外指标词"),
]


def _truncate(s: str, n: int = 80) -> str:
    s = (s or '').replace('\n', ' ').strip()
    return s[:n] + ('…' if len(s) > n else '')


def main():
    print("=" * 100)
    print("批量 smoke test(20 query)")
    print("=" * 100)
    r = FinAgentRetriever()
    print()

    for i, (tool, q, note) in enumerate(TEST_CASES, 1):
        print(f"\n{'─' * 100}")
        print(f"[{i:2d}/{len(TEST_CASES)}] tool={tool}  query={q!r}")
        print(f"       说明: {note}")
        print(f"{'─' * 100}")

        if tool == "financial":
            results = r.search_financial(q, top_k=3)
        elif tool == "industry":
            results = r.search_industry(q, top_k=3)
        elif tool == "report":
            results = r.search_report(q, top_k=3)
        else:
            print(f"  未知 tool: {tool}")
            continue

        if not results:
            print("  ❌ 返回空")
            continue

        for j, c in enumerate(results, 1):
            m = c.get('metadata', {})
            score = c.get('score', 0)
            title = m.get('report_title') or ''
            method = m.get('chunk_method') or m.get('source_type') or ''
            text = _truncate(c.get('text', ''))
            print(f"  [{j}] score={score:.4f}  [{method:15s}] {title[:40]}")
            print(f"       {text}")


if __name__ == '__main__':
    main()
