"""
FinAgent Step 4: Chunk 构建
用途：将所有数据源（研报元数据 + PDF正文 + 财务指标）统一构建为检索chunks
环境：Google Colab
依赖：pip install pandas tqdm

运行方式：
    python 04_build_chunks.py                                    # 处理全部数据（默认用pdfplumber解析结果）
    python 04_build_chunks.py --parser marker                    # 用Marker解析结果
    python 04_build_chunks.py --parser mineru                    # 用MinerU解析结果
    python 04_build_chunks.py --report_only                      # 只处理研报
    python 04_build_chunks.py --financial_only                   # 只处理财务

面试追问：你的chunk是怎么设计的？
答：研报数据本身是摘要级别（标题+评级+盈利预测），天然适合做单条chunk。
财务数据是结构化的，我转成自然语言描述后作为chunk，这样向量检索能匹配到。
PDF正文用滑动窗口切分（512字符，64重叠），在句号/换行处断句。
比如query="茅台盈利能力"能匹配到"贵州茅台2024年ROE为15.3%"。
"""

import pandas as pd
import json
import os
import re
import logging
import argparse
from datetime import datetime
from tqdm import tqdm

# ============ 配置 ============
RAW_DIR = "./data/raw"
CHUNK_DIR = "./data/processed"
os.makedirs(CHUNK_DIR, exist_ok=True)

# 解析器 → 结果文件的映射
PARSER_RESULT_FILES = {
    "pdfplumber": os.path.join(RAW_DIR, "report_parsed", "pdfplumber_all_results.json"),
    "marker": os.path.join(RAW_DIR, "report_parsed", "marker_all_results.json"),
    "mineru": os.path.join(RAW_DIR, "report_parsed", "mineru_200_results.json"),
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============ 研报元数据 Chunk 构建 ============

def build_report_chunks(report_path: str) -> list[dict]:
    """
    将研报数据转为检索chunks

    每篇研报 → 1个chunk，包含：
    - 报告标题（核心语义信息）
    - 机构 + 评级（机构观点）
    - 盈利预测数据（硬数据）
    - 行业 + 日期（元信息）

    面试追问：为什么不每个字段单独做chunk？
    答：研报摘要本身就很短（一行标题+几个字段），拆开后每个chunk语义太稀疏，
    检索时会匹配到大量无关结果。合在一起语义更完整。
    """
    df = pd.read_csv(report_path, dtype={'股票代码': str})
    chunks = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建研报chunks"):
        code = str(row.get('股票代码', '')).strip()
        name = str(row.get('股票简称', '')).strip()
        title = str(row.get('报告名称', '')).strip()
        rating = str(row.get('东财评级', '')).strip()
        institution = str(row.get('机构', '')).strip()
        industry = str(row.get('行业', '')).strip()
        date = str(row.get('日期', '')).strip()

        # 构建自然语言描述
        text_parts = [f"{name}({code}) 研报：{title}"]

        if institution and institution != 'nan':
            text_parts.append(f"出品机构：{institution}")
        if rating and rating != 'nan':
            text_parts.append(f"评级：{rating}")

        # 盈利预测
        eps_parts = []
        for year in ['2025', '2026', '2027']:
            eps = row.get(f'{year}-盈利预测-收益')
            pe = row.get(f'{year}-盈利预测-市盈率')
            if pd.notna(eps) and pd.notna(pe):
                eps_parts.append(f"{year}年预测EPS {eps}元，PE {pe}倍")
        if eps_parts:
            text_parts.append("盈利预测：" + "；".join(eps_parts))

        text = "\n".join(text_parts)

        chunk = {
            "text": text,
            "metadata": {
                "source_type": "report",
                "stock_code": code,
                "stock_name": name,
                "institution": institution,
                "rating": rating,
                "industry": industry,
                "date": date,
                "report_title": title,
            }
        }
        chunks.append(chunk)

    logger.info(f"研报chunks: {len(chunks)} 条")
    return chunks


# ============ PDF正文 Chunk 构建 ============

def build_fulltext_chunks(parser: str, pdf_map_path: str = None,
                          chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """
    加载解析器输出的全文JSON，切分为检索chunks（滑动窗口）

    面试追问：chunk_size怎么定的？
    答：512是经验值。256太短一段分析可能被切断，1024太长检索时混入无关信息。
    消融实验R-1会验证 {256, 512, 768, 1024} 的效果差异。
    overlap=64（约12.5%）保证上下文不断裂。
    """
    result_file = PARSER_RESULT_FILES.get(parser)
    if not result_file or not os.path.exists(result_file):
        logger.warning(f"解析结果不存在: {result_file}，跳过PDF正文chunks")
        return []

    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    logger.info(f"加载 {parser} 解析结果: {len(results)} 篇")

    # 加载 pdf_map 获取元数据（股票代码、机构等）
    pdf_map = {}
    if pdf_map_path and os.path.exists(pdf_map_path):
        with open(pdf_map_path, 'r', encoding='utf-8') as f:
            pdf_map = json.load(f)

    all_chunks = []
    for item in tqdm(results, desc=f"构建{parser}正文chunks"):
        filename = item["file"]
        text = item["text"]

        if not text or len(text) < 50:
            continue

        # 清理 markdown 图片标记（Marker 输出的残留，如 ![](_page_0_Picture_0.jpeg)）
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        if len(text) < 50:
            continue

        # 从 pdf_map 获取元数据
        meta = pdf_map.get(filename, {})
        base_metadata = {
            "source_type": "report_fulltext",
            "parser": parser,
            "stock_code": meta.get("stock_code", ""),
            "stock_name": meta.get("stock_name", ""),
            "institution": meta.get("institution", ""),
            "rating": meta.get("rating", ""),
            "industry": meta.get("industry", ""),
            "date": meta.get("date", ""),
            "report_title": meta.get("report_title", ""),
            "pdf_file": filename,
        }

        chunks = _sliding_window_chunks(text, base_metadata, chunk_size, overlap)
        all_chunks.extend(chunks)

    logger.info(f"PDF正文chunks ({parser}): {len(all_chunks)} 条")
    return all_chunks


def _sliding_window_chunks(text: str, metadata: dict,
                           chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """将全文切分为定长chunks（滑动窗口）"""
    chunks = []

    if len(text) <= chunk_size:
        if len(text) >= 50:
            chunks.append({
                "text": text,
                "metadata": {**metadata, "chunk_method": "full_text"}
            })
        return chunks

    start = 0
    chunk_idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # 尽量在句号/换行处截断
        if end < len(text):
            last_break = max(
                chunk_text.rfind('。'),
                chunk_text.rfind('\n'),
                chunk_text.rfind('；'),
            )
            if last_break > chunk_size * 0.5:
                chunk_text = chunk_text[:last_break + 1]
                end = start + last_break + 1

        chunk_text = chunk_text.strip()
        if len(chunk_text) >= 50:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_method": "sliding_window",
                    "chunk_index": chunk_idx,
                }
            })
            chunk_idx += 1

        start = end - overlap
        if start >= len(text):
            break

    return chunks


# ============ 财务数据 Chunk 构建 ============

def build_financial_chunks(financial_path: str) -> list[dict]:
    """
    将 stock_financial_analysis_indicator 的结构化数据转为自然语言chunks

    数据来源：ak.stock_financial_analysis_indicator，每期包含86个字段

    策略：每期数据生成2个chunk（盈利+结构），而不是原来的4个chunk（利润表/资产负债/现金流/指标）。
    原因：新接口的数据已经是汇总指标，不需要按报表拆分。合并后每个chunk语义更完整，
    检索时一次就能拿到盈利能力+成长性的全貌。

    面试追问：为什么转成自然语言而不是直接存结构化数据？
    答：因为Agent通过自然语言检索，向量检索对结构化数据不友好。
    将"ROE: 15.3%"转成"贵州茅台2024年ROE为15.3%"后，
    query="茅台盈利能力"就能检索到。这是一个工程上的trade-off。
    """
    with open(financial_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    chunks = []

    for code, data in tqdm(all_data.items(), desc="构建财务chunks"):
        name = data.get("stock_name", code)

        if "financial_indicators" not in data:
            continue

        for record in data["financial_indicators"]:
            date = str(record.get("日期", "未知日期"))

            # 标注报告期类型，避免Agent混淆年报和半年报
            if date.endswith("12-31"):
                period_label = "年报"
            elif date.endswith("06-30"):
                period_label = "半年报"
            else:
                period_label = ""

            # Chunk 1: 盈利能力 + 成长性 + 每股指标
            text = _profitability_to_text(code, name, date, record, period_label)
            if text:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "source_type": "financial",
                        "data_type": "profitability",
                        "stock_code": code,
                        "stock_name": name,
                        "report_date": date,
                    }
                })

            # Chunk 2: 偿债能力 + 运营效率 + 资产结构
            text = _structure_to_text(code, name, date, record, period_label)
            if text:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "source_type": "financial",
                        "data_type": "balance_structure",
                        "stock_code": code,
                        "stock_name": name,
                        "report_date": date,
                    }
                })

    logger.info(f"财务chunks: {len(chunks)} 条")
    return chunks


def _safe_fmt(value, suffix="", decimals=2) -> str:
    """安全格式化数值，处理NaN和None"""
    if value is None:
        return None
    try:
        v = float(value)
        if pd.isna(v):
            return None
        return f"{v:.{decimals}f}{suffix}"
    except (ValueError, TypeError):
        return None


def _format_amount(value, unit="元") -> str:
    """将大数字转为可读格式：亿/万"""
    if value is None:
        return None
    try:
        v = float(value)
        if pd.isna(v):
            return None
    except (ValueError, TypeError):
        return None
    if abs(v) >= 1e8:
        return f"{v/1e8:.2f}亿{unit}"
    elif abs(v) >= 1e4:
        return f"{v/1e4:.2f}万{unit}"
    else:
        return f"{v:.2f}{unit}"


def _profitability_to_text(code: str, name: str, date: str, record: dict, period_label: str = "") -> str:
    """盈利能力 + 成长性 + 每股指标 → 自然语言"""
    header = f"{name}({code}) {date}"
    if period_label:
        header += f"({period_label})"
    parts = [f"{header}盈利与成长指标"]
    valid = False

    fields = [
        ("净资产收益率(%)", "净资产收益率(ROE)", "%"),
        ("加权净资产收益率(%)", "加权ROE", "%"),
        ("主营业务利润率(%)", "主营业务利润率(毛利率)", "%"),
        ("销售净利率(%)", "销售净利率", "%"),
        ("营业利润率(%)", "营业利润率", "%"),
        ("总资产利润率(%)", "总资产利润率(ROA)", "%"),
    ]
    for key, label, suffix in fields:
        val = _safe_fmt(record.get(key), suffix)
        if val:
            parts.append(f"{label}{val}")
            valid = True

    growth_fields = [
        ("主营业务收入增长率(%)", "营收增长率", "%"),
        ("净利润增长率(%)", "净利润增长率", "%"),
        ("净资产增长率(%)", "净资产增长率", "%"),
        ("总资产增长率(%)", "总资产增长率", "%"),
    ]
    for key, label, suffix in growth_fields:
        val = _safe_fmt(record.get(key), suffix)
        if val:
            parts.append(f"{label}{val}")
            valid = True

    eps_fields = [
        ("摊薄每股收益(元)", "每股收益(EPS)", "元"),
        ("每股净资产_调整后(元)", "每股净资产(BPS)", "元"),
        ("每股经营性现金流(元)", "每股经营现金流", "元"),
        ("每股未分配利润(元)", "每股未分配利润", "元"),
    ]
    for key, label, suffix in eps_fields:
        val = _safe_fmt(record.get(key), suffix)
        if val:
            parts.append(f"{label}{val}")
            valid = True

    total_assets = _format_amount(record.get("总资产(元)"))
    if total_assets:
        parts.append(f"总资产{total_assets}")
        valid = True

    return "，".join(parts) + "。" if valid else None


def _structure_to_text(code: str, name: str, date: str, record: dict, period_label: str = "") -> str:
    """偿债能力 + 运营效率 + 资产结构 → 自然语言"""
    header = f"{name}({code}) {date}"
    if period_label:
        header += f"({period_label})"
    parts = [f"{header}偿债与运营指标"]
    valid = False

    fields = [
        ("资产负债率(%)", "资产负债率", "%"),
        ("流动比率", "流动比率", ""),
        ("速动比率", "速动比率", ""),
        ("现金比率(%)", "现金比率", "%"),
        ("股东权益比率(%)", "股东权益比率", "%"),
        ("产权比率(%)", "产权比率", "%"),
    ]
    for key, label, suffix in fields:
        val = _safe_fmt(record.get(key), suffix)
        if val:
            parts.append(f"{label}{val}")
            valid = True

    efficiency_fields = [
        ("总资产周转率(次)", "总资产周转率", "次"),
        ("存货周转率(次)", "存货周转率", "次"),
        ("存货周转天数(天)", "存货周转天数", "天"),
        ("应收账款周转率(次)", "应收账款周转率", "次"),
        ("应收账款周转天数(天)", "应收账款周转天数", "天"),
    ]
    for key, label, suffix in efficiency_fields:
        val = _safe_fmt(record.get(key), suffix)
        if val:
            parts.append(f"{label}{val}")
            valid = True

    cf_fields = [
        ("经营现金净流量对销售收入比率(%)", "经营现金流/营收比", "%"),
        ("经营现金净流量与净利润的比率(%)", "经营现金流/净利润比", "%"),
    ]
    for key, label, suffix in cf_fields:
        val = _safe_fmt(record.get(key), suffix)
        if val:
            parts.append(f"{label}{val}")
            valid = True

    return "，".join(parts) + "。" if valid else None


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="FinAgent Chunk 构建")
    parser.add_argument("--parser", type=str, default="pdfplumber",
                        choices=["pdfplumber", "marker", "mineru"],
                        help="选择PDF解析器的结果（默认pdfplumber）")
    parser.add_argument("--report_only", action="store_true")
    parser.add_argument("--financial_only", action="store_true")
    args = parser.parse_args()

    all_chunks = []

    # 1. 研报元数据chunks（标题+评级+EPS预测）
    report_path = os.path.join(RAW_DIR, "reports", "all_reports.csv")
    if not args.financial_only and os.path.exists(report_path):
        report_chunks = build_report_chunks(report_path)
        all_chunks.extend(report_chunks)
    elif not args.financial_only:
        logger.warning(f"研报元数据不存在: {report_path}，跳过")

    # 2. 研报PDF正文chunks（从解析器结果 → 滑动窗口切分）
    pdf_map_path = os.path.join(RAW_DIR, "report_pdfs", "pdf_map.json")
    if not args.financial_only:
        fulltext_chunks = build_fulltext_chunks(args.parser, pdf_map_path)
        all_chunks.extend(fulltext_chunks)

    # 3. 财务数据chunks
    financial_path = os.path.join(RAW_DIR, "financial", "all_financial.json")
    if not args.report_only and os.path.exists(financial_path):
        financial_chunks = build_financial_chunks(financial_path)
        all_chunks.extend(financial_chunks)
    elif not args.report_only:
        logger.warning(f"财务数据不存在: {financial_path}，跳过")

    # 保存
    if all_chunks:
        output_path = os.path.join(CHUNK_DIR, "all_chunks.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        source_counts = {}
        for c in all_chunks:
            st = c["metadata"]["source_type"]
            source_counts[st] = source_counts.get(st, 0) + 1

        stats = {
            "total_chunks": len(all_chunks),
            "by_source_type": source_counts,
            "parser": args.parser,
            "unique_stocks": len(set(c["metadata"]["stock_code"] for c in all_chunks)),
            "avg_text_length": sum(len(c["text"]) for c in all_chunks) / len(all_chunks),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(os.path.join(CHUNK_DIR, "chunk_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info("=" * 50)
        logger.info("Chunk 构建完成！")
        logger.info(f"  PDF解析器: {args.parser}")
        logger.info(f"  总chunks: {stats['total_chunks']}")
        for st, cnt in source_counts.items():
            logger.info(f"  {st}: {cnt} 条")
        logger.info(f"  覆盖股票: {stats['unique_stocks']} 只")
        logger.info(f"  平均chunk长度: {stats['avg_text_length']:.0f} 字符")
        logger.info(f"  保存路径: {output_path}")
        logger.info("=" * 50)
        logger.info("下一步: 运行 05_build_index.py 构建 FAISS + BM25 索引")
    else:
        logger.error("没有生成任何chunk，请检查原始数据是否存在")


if __name__ == "__main__":
    main()
