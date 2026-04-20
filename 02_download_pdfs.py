"""
FinAgent Step 2: 研报PDF下载
用途：从东财下载研报PDF文件
环境：Google Colab
依赖：pip install pandas tqdm requests

运行顺序：
    1. 先运行 01_fetch_raw_data.py 获取研报元数据（all_reports.csv）
    2. 运行本脚本下载PDF
    3. 运行 03a/03b/03c 任一解析脚本解析PDF

运行方式：
    python 02_download_pdfs.py                     # 全量下载
    python 02_download_pdfs.py --max_per_stock 3   # 每只股票最多3篇（默认5）
    python 02_download_pdfs.py --max_total 100     # 总共最多100篇（测试用）
"""

import pandas as pd
import requests
import json
import time
import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm

# ============ 配置 ============
RAW_DIR = "./data/raw"
REPORT_CSV = os.path.join(RAW_DIR, "reports", "all_reports.csv")
PDF_DIR = os.path.join(RAW_DIR, "report_pdfs")
LOG_DIR = "./logs"

MAX_PER_STOCK = 5           # 每只股票最多下载几篇
SLEEP_BETWEEN_DOWNLOADS = 1.0
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30        # PDF下载超时秒数
MAX_PDF_SIZE = 20 * 1024 * 1024  # 跳过超过20MB的PDF

# PDF请求头（模拟浏览器，防反爬）
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://data.eastmoney.com/",
}

for d in [PDF_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"download_pdf_{datetime.now():%Y%m%d_%H%M%S}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============ 筛选要下载的PDF ============

def select_pdfs_to_download(max_per_stock: int = 5, max_total: int = 0) -> pd.DataFrame:
    """
    从 all_reports.csv 筛选要下载的研报
    策略：每只股票取最新的 max_per_stock 篇

    面试追问：为什么不全部下载？
    答：沪深300 × 平均20篇 = 6000+ PDF，Colab存储和时间都不够。
    每只股票取最新5篇，约1500篇，已经能提供足够的语料多样性。
    """
    df = pd.read_csv(REPORT_CSV, dtype={'股票代码': str})

    # 确保日期列可排序
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df = df.dropna(subset=['日期', '报告PDF链接'])

    # 过滤无效链接
    df = df[df['报告PDF链接'].str.startswith('http')].copy()

    # 每只股票取最新 max_per_stock 篇
    df = df.sort_values(['股票代码', '日期'], ascending=[True, False])
    df = df.groupby('股票代码').head(max_per_stock).reset_index(drop=True)

    # 总量限制
    if max_total > 0:
        df = df.head(max_total)

    logger.info(f"筛选完成: {len(df)} 篇PDF待下载，覆盖 {df['股票代码'].nunique()} 只股票")
    return df


# ============ 下载PDF ============

def download_pdf(url: str, save_path: str) -> bool:
    """下载单个PDF文件"""
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return True  # 已下载，跳过

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()

            # 检查content-type
            content_type = resp.headers.get('content-type', '')
            if 'pdf' not in content_type.lower() and 'octet-stream' not in content_type.lower():
                logger.warning(f"  非PDF响应: {content_type}")
                return False

            # 检查大小
            content_length = int(resp.headers.get('content-length', 0))
            if content_length > MAX_PDF_SIZE:
                logger.warning(f"  PDF过大: {content_length / 1024 / 1024:.1f}MB, 跳过")
                return False

            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 2)
            else:
                logger.error(f"  下载失败: {e}")
                return False


def batch_download_pdfs(df: pd.DataFrame) -> dict:
    """
    批量下载PDF
    返回: {pdf_filename: row_data} 的映射
    """
    stats = {"success": 0, "skipped": 0, "failed": 0}
    pdf_map = {}

    logger.info(f"===== 开始下载 {len(df)} 篇研报PDF =====")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="下载PDF"):
        code = row['股票代码']
        date = pd.to_datetime(row['日期']).strftime('%Y%m%d')
        url = row['报告PDF链接']

        # 文件名: 股票代码_日期_序号.pdf
        filename = f"{code}_{date}_{idx}.pdf"
        save_path = os.path.join(PDF_DIR, filename)

        success = download_pdf(url, save_path)

        if success:
            stats["success"] += 1
            # 字段做 NaN-safe + strip,和 04 _load_pdf_reverse_index 的 4 元 key 对齐
            def _f(v):
                return '' if pd.isna(v) else str(v).strip()
            pdf_map[filename] = {
                "stock_code": _f(code),
                "stock_name": _f(row.get('股票简称', '')),
                "report_title": _f(row.get('报告名称', '')),
                "institution": _f(row.get('机构', '')),
                "rating": _f(row.get('东财评级', '')),
                "industry": _f(row.get('行业', '')),
                "date": pd.to_datetime(row['日期']).strftime('%Y-%m-%d'),
                "url": url,
                "pdf_path": save_path,
            }
        else:
            stats["failed"] += 1

        time.sleep(SLEEP_BETWEEN_DOWNLOADS)

        # 每100篇保存一次映射
        if (idx + 1) % 100 == 0:
            _save_pdf_map(pdf_map, stats)

    _save_pdf_map(pdf_map, stats)
    logger.info(f"===== PDF下载完成: 成功{stats['success']}, 失败{stats['failed']} =====")
    return pdf_map


def _save_pdf_map(pdf_map: dict, stats: dict):
    """保存PDF映射和统计"""
    with open(os.path.join(PDF_DIR, "pdf_map.json"), 'w', encoding='utf-8') as f:
        json.dump(pdf_map, f, ensure_ascii=False, indent=2)
    with open(os.path.join(PDF_DIR, "download_stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="研报PDF下载")
    parser.add_argument("--max_per_stock", type=int, default=5, help="每只股票最多下载几篇")
    parser.add_argument("--max_total", type=int, default=0, help="总共最多下载几篇（0=不限）")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("研报PDF下载")
    logger.info(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info("=" * 60)

    if not os.path.exists(REPORT_CSV):
        logger.error(f"研报元数据不存在: {REPORT_CSV}")
        logger.error("请先运行 01_fetch_raw_data.py 获取研报元数据")
        return

    selected = select_pdfs_to_download(
        max_per_stock=args.max_per_stock,
        max_total=args.max_total
    )
    batch_download_pdfs(selected)

    logger.info("=" * 60)
    logger.info("下载完成！下一步:")
    logger.info("  运行 03a_parse_pdfplumber.py / 03b_parse_marker.py / 03c_parse_mineru.py 解析PDF")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
