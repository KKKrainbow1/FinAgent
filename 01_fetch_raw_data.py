"""
FinAgent 数据拉取脚本
用途：批量拉取沪深300研报 + 财务数据，构建检索语料库
依赖：pip install akshare pandas tqdm

运行方式：
    python 01_fetch_raw_data.py                        # 拉取全部（研报+财务）
    python 01_fetch_raw_data.py --only financial        # 只拉财务数据
    python 01_fetch_raw_data.py --only reports          # 只拉研报
    python 01_fetch_raw_data.py --only financial --start_year 2024  # 只拉2024年起的财务数据
    python 01_fetch_raw_data.py --max_stocks 10         # 先拉10只测试
    python 01_fetch_raw_data.py --stocks 600519,000858  # 只拉指定股票
    python 01_fetch_raw_data.py --update                # 增量更新（只拉新数据，不覆盖旧数据）
    python 01_fetch_raw_data.py --resume                # 断点续传
"""

import akshare as ak
import pandas as pd
import json
import time
import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm

# ============ 配置 ============
OUTPUT_DIR = "./data/raw"
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
FINANCIAL_DIR = os.path.join(OUTPUT_DIR, "financial")
LOG_DIR = "./logs"

SLEEP_BETWEEN_STOCKS = 1.0      # 每只股票间隔（秒）
SLEEP_BETWEEN_REQUESTS = 0.5    # 每个API请求间隔（秒）
MAX_RETRIES = 3                 # 单个请求最大重试次数

# 创建目录
for d in [REPORT_DIR, FINANCIAL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"fetch_{datetime.now():%Y%m%d_%H%M%S}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============ Step 1: 获取沪深300成分股列表 ============

def get_hs300_stocks() -> pd.DataFrame:
    """
    获取沪深300成分股列表
    尝试多个接口，确保至少一个能用
    """
    # 方案1: 中证指数官网
    try:
        df = ak.index_stock_cons_csindex(symbol="000300")
        logger.info(f"[csindex] 获取沪深300成分股 {len(df)} 只")
        # 统一字段名
        if '成分券代码' in df.columns:
            df = df.rename(columns={'成分券代码': 'stock_code', '成分券名称': 'stock_name'})
        elif '证券代码' in df.columns:
            df = df.rename(columns={'证券代码': 'stock_code', '证券简称': 'stock_name'})
        return df[['stock_code', 'stock_name']].reset_index(drop=True)
    except Exception as e:
        logger.warning(f"[csindex] 失败: {e}")

    # 方案2: 东财接口
    try:
        df = ak.index_stock_cons(symbol="000300")
        logger.info(f"[eastmoney] 获取沪深300成分股 {len(df)} 只")
        if '品种代码' in df.columns:
            df = df.rename(columns={'品种代码': 'stock_code', '品种名称': 'stock_name'})
        return df[['stock_code', 'stock_name']].reset_index(drop=True)
    except Exception as e:
        logger.warning(f"[eastmoney] 失败: {e}")

    # 方案3: 从本地文件读取（备用）
    fallback_path = os.path.join(OUTPUT_DIR, "hs300_stocks.csv")
    if os.path.exists(fallback_path):
        df = pd.read_csv(fallback_path, dtype={'stock_code': str})
        logger.info(f"[本地文件] 读取沪深300成分股 {len(df)} 只")
        return df

    raise RuntimeError("无法获取沪深300成分股列表，请手动准备 hs300_stocks.csv")


# ============ Step 2: 拉取研报数据 ============

def fetch_reports_for_stock(stock_code: str) -> pd.DataFrame:
    """
    拉取单只股票的研报数据
    返回: DataFrame，包含研报标题、评级、机构、盈利预测等
    """
    for attempt in range(MAX_RETRIES):
        try:
            df = ak.stock_research_report_em(symbol=stock_code)
            if df is not None and len(df) > 0:
                return df
            return pd.DataFrame()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 2
                logger.warning(f"  [{stock_code}] 研报请求失败(第{attempt+1}次): {e}, {wait}秒后重试")
                time.sleep(wait)
            else:
                logger.error(f"  [{stock_code}] 研报请求最终失败: {e}")
                return pd.DataFrame()


def batch_fetch_reports(stocks: pd.DataFrame) -> dict:
    """
    批量拉取研报数据
    返回: {stock_code: DataFrame} 的字典
    """
    results = {}
    stats = {"success": 0, "empty": 0, "failed": 0, "total_reports": 0}

    logger.info(f"===== 开始拉取研报数据，共 {len(stocks)} 只股票 =====")

    for idx, row in tqdm(stocks.iterrows(), total=len(stocks), desc="拉取研报"):
        code = row['stock_code']
        name = row['stock_name']

        df = fetch_reports_for_stock(code)

        if df is not None and len(df) > 0:
            # 添加股票名称（原始数据可能没有）
            if '股票简称' not in df.columns:
                df['股票简称'] = name
            results[code] = df
            stats["success"] += 1
            stats["total_reports"] += len(df)
            logger.info(f"  [{code} {name}] 获取 {len(df)} 篇研报")
        else:
            stats["empty"] += 1
            logger.info(f"  [{code} {name}] 无研报数据")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

        # 每50只股票保存一次checkpoint
        if (idx + 1) % 50 == 0:
            _save_report_checkpoint(results, stats)

    # 最终保存
    _save_report_checkpoint(results, stats)
    logger.info(f"===== 研报拉取完成: 成功{stats['success']}, 空{stats['empty']}, "
                f"失败{stats['failed']}, 共{stats['total_reports']}篇 =====")
    return results


def _save_report_checkpoint(results: dict, stats: dict):
    """保存研报数据checkpoint"""
    if not results:
        return
    all_reports = pd.concat(results.values(), ignore_index=True)
    save_path = os.path.join(REPORT_DIR, "all_reports.csv")
    all_reports.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    # 保存统计信息
    with open(os.path.join(REPORT_DIR, "fetch_stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"  [checkpoint] 已保存 {len(all_reports)} 篇研报到 {save_path}")


# ============ Step 3: 拉取财务数据 ============

def fetch_financial_for_stock(stock_code: str, start_year: str = "2022") -> dict:
    """
    拉取单只股票的关键财务数据

    改用 stock_financial_analysis_indicator（东财源）统一拉取。
    stock_financial_analysis_indicator 一个接口包含86个字段，覆盖：
    盈利能力（ROE/毛利率/净利率）、成长能力（营收增长率/利润增长率）、
    偿债能力（资产负债率/流动比率）、运营效率（周转率）、每股指标（EPS/BPS）、
    总资产等。

    Args:
        stock_code: 股票代码
        start_year: 起始年份（默认"2022"）
    """
    result = {}

    try:
        df = ak.stock_financial_analysis_indicator(
            symbol=stock_code, start_year=start_year
        )
        if df is not None and len(df) > 0:
            # 只保留年报(12-31)和半年报(06-30)，去掉Q1(03-31)和Q3(09-30)
            df['日期'] = pd.to_datetime(df['日期'])
            df = df[df['日期'].dt.month.isin([6, 12])].copy()

            if len(df) > 0:
                records = df.to_dict(orient='records')
                # 日期转回字符串便于JSON序列化
                for r in records:
                    r['日期'] = r['日期'].strftime('%Y-%m-%d')
                result["financial_indicators"] = records
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    except Exception as e:
        result["financial_error"] = str(e)

    return result


def batch_fetch_financial(stocks: pd.DataFrame, start_year: str = "2022",
                          update_mode: bool = False) -> dict:
    """
    批量拉取财务数据

    Args:
        stocks: 股票列表 DataFrame
        start_year: 起始年份
        update_mode: 增量更新模式，合并新旧数据

    返回: {stock_code: dict} 的字典
    """
    # 增量更新：先加载已有数据
    existing_data = {}
    if update_mode:
        existing_path = os.path.join(FINANCIAL_DIR, "all_financial.json")
        if os.path.exists(existing_path):
            with open(existing_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            logger.info(f"[增量更新] 已加载 {len(existing_data)} 只股票的现有数据")

    results = {}
    stats = {"success": 0, "partial": 0, "failed": 0}

    logger.info(f"===== 开始拉取财务数据，共 {len(stocks)} 只股票，起始年份 {start_year} =====")

    for idx, row in tqdm(stocks.iterrows(), total=len(stocks), desc="拉取财务"):
        code = row['stock_code']
        name = row['stock_name']

        data = fetch_financial_for_stock(code, start_year=start_year)

        # 统计成功/失败
        has_data = "financial_indicators" in data
        has_error = "financial_error" in data

        if has_data and not has_error:
            stats["success"] += 1
        elif has_data:
            stats["partial"] += 1
        else:
            stats["failed"] += 1

        data["stock_code"] = code
        data["stock_name"] = name

        # 增量更新：合并新旧数据
        if update_mode and code in existing_data:
            old_indicators = existing_data[code].get("financial_indicators", [])
            new_indicators = data.get("financial_indicators", [])
            # 按日期去重合并
            old_dates = {r['日期'] for r in old_indicators}
            merged = old_indicators.copy()
            added = 0
            for r in new_indicators:
                if r['日期'] not in old_dates:
                    merged.append(r)
                    added += 1
            merged.sort(key=lambda x: x['日期'])
            data["financial_indicators"] = merged
            if added > 0:
                logger.info(f"  [{code} {name}] 新增 {added} 期数据（总共 {len(merged)} 期）")

        results[code] = data

        n_periods = len(data.get("financial_indicators", []))
        logger.info(f"  [{code} {name}] 财务指标: "
                    f"{'✓ ' + str(n_periods) + '期' if has_data else '✗'}"
                    f"{' (有错误)' if has_error else ''}")

        time.sleep(SLEEP_BETWEEN_STOCKS)

        # 每50只股票保存一次checkpoint
        if (idx + 1) % 50 == 0:
            # 增量模式下合并未处理的旧数据
            save_results = {**existing_data, **results} if update_mode else results
            _save_financial_checkpoint(save_results, stats)

    # 最终保存
    save_results = {**existing_data, **results} if update_mode else results
    _save_financial_checkpoint(save_results, stats)
    logger.info(f"===== 财务数据拉取完成: 成功{stats['success']}, "
                f"部分{stats['partial']}, 失败{stats['failed']} =====")
    return save_results


def _save_financial_checkpoint(results: dict, stats: dict):
    """保存财务数据checkpoint"""
    if not results:
        return
    save_path = os.path.join(FINANCIAL_DIR, "all_financial.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    with open(os.path.join(FINANCIAL_DIR, "fetch_stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"  [checkpoint] 已保存 {len(results)} 只股票财务数据到 {save_path}")


# ============ Step 4: 断点续传支持 ============

def load_report_progress() -> set:
    """
    返回已完成研报拉取的股票代码集合。

    研报 checkpoint 只会保存实际拉取到的研报记录，因此以 all_reports.csv
    中出现过的股票代码作为“研报已完成”的判定。
    """
    done_codes = set()
    report_path = os.path.join(REPORT_DIR, "all_reports.csv")
    if not os.path.exists(report_path):
        return done_codes

    df = pd.read_csv(report_path, dtype={'股票代码': str})
    if '股票代码' in df.columns:
        done_codes.update(df['股票代码'].dropna().astype(str).unique())
        logger.info(f"[断点续传] 已有 {len(done_codes)} 只股票的研报数据")

    return done_codes


def load_financial_progress() -> set:
    """
    返回已完成财务拉取的股票代码集合。

    财务 checkpoint 会保存成功、失败或增量合并后的结果，因此这里只将
    “存在 financial_indicators 且没有 financial_error”的股票视为完成，
    避免把失败/异常的股票也在 resume 时跳过。
    """
    done_codes = set()
    financial_path = os.path.join(FINANCIAL_DIR, "all_financial.json")
    if not os.path.exists(financial_path):
        return done_codes

    with open(financial_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return done_codes

    for stock_code, item in data.items():
        if not isinstance(item, dict):
            continue
        indicators = item.get("financial_indicators") or []
        has_error = bool(item.get("financial_error"))
        if indicators and not has_error:
            done_codes.add(str(stock_code))

    logger.info(f"[断点续传] 已有 {len(done_codes)} 只股票的财务数据")
    return done_codes


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="FinAgent 数据拉取")
    parser.add_argument("--max_stocks", type=int, default=0,
                        help="最多拉取股票数（0=全部）")
    parser.add_argument("--only", type=str, choices=["financial", "reports"],
                        help="只拉取指定类型（financial=财务数据, reports=研报）")
    parser.add_argument("--skip_reports", action="store_true",
                        help="跳过研报拉取（等价于 --only financial）")
    parser.add_argument("--skip_financial", action="store_true",
                        help="跳过财务数据拉取（等价于 --only reports）")
    parser.add_argument("--start_year", type=str, default="2022",
                        help="财务数据起始年份（默认2022）")
    parser.add_argument("--stocks", type=str, default="",
                        help="只拉指定股票，逗号分隔（如 600519,000858）")
    parser.add_argument("--update", action="store_true",
                        help="增量更新模式：合并新旧数据，不覆盖已有期数")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传模式")
    args = parser.parse_args()

    # --only 参数转换为 skip 标记
    if args.only == "financial":
        args.skip_reports = True
    elif args.only == "reports":
        args.skip_financial = True

    logger.info("=" * 60)
    logger.info("FinAgent 数据拉取开始")
    logger.info(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    if args.only:
        logger.info(f"模式: 只拉取 {args.only}")
    if args.start_year != "2022":
        logger.info(f"财务数据起始年份: {args.start_year}")
    if args.update:
        logger.info(f"增量更新模式: 合并新旧数据")
    logger.info("=" * 60)

    # 1. 获取股票列表
    if args.stocks:
        # 指定股票模式
        stock_codes = [s.strip() for s in args.stocks.split(",")]
        stocks = pd.DataFrame({"stock_code": stock_codes, "stock_name": [""] * len(stock_codes)})
        logger.info(f"指定股票模式: {len(stocks)} 只 ({args.stocks})")
    else:
        stocks = get_hs300_stocks()
        # 保存股票列表（备用）
        stocks.to_csv(os.path.join(OUTPUT_DIR, "hs300_stocks.csv"), index=False, encoding='utf-8-sig')
        logger.info(f"沪深300成分股: {len(stocks)} 只")

    report_stocks = stocks
    financial_stocks = stocks

    # 2. 断点续传：按任务类型分别过滤已完成的股票
    if args.resume:
        if not args.skip_reports:
            report_done = load_report_progress()
            report_stocks = stocks[~stocks['stock_code'].isin(report_done)].reset_index(drop=True)
            logger.info(f"断点续传[研报]: 跳过 {len(report_done)} 只, 剩余 {len(report_stocks)} 只")
        if not args.skip_financial:
            financial_done = load_financial_progress()
            financial_stocks = stocks[~stocks['stock_code'].isin(financial_done)].reset_index(drop=True)
            logger.info(f"断点续传[财务]: 跳过 {len(financial_done)} 只, 剩余 {len(financial_stocks)} 只")

    # 3. 限制数量（测试用）
    if args.max_stocks > 0:
        report_stocks = report_stocks.head(args.max_stocks)
        financial_stocks = financial_stocks.head(args.max_stocks)
        logger.info(f"测试模式: 每类任务最多拉取前 {args.max_stocks} 只")

    # 4. 拉取研报
    if not args.skip_reports:
        report_results = batch_fetch_reports(report_stocks)

    # 5. 拉取财务数据
    if not args.skip_financial:
        financial_results = batch_fetch_financial(
            financial_stocks,
            start_year=args.start_year,
            update_mode=args.update,
        )

    # 6. 汇总统计
    logger.info("=" * 60)
    logger.info("数据拉取完成！")
    logger.info(f"数据保存目录: {os.path.abspath(OUTPUT_DIR)}")
    if not args.skip_reports:
        logger.info(f"  研报: {REPORT_DIR}/all_reports.csv")
    if not args.skip_financial:
        logger.info(f"  财务: {FINANCIAL_DIR}/all_financial.json")
    logger.info("=" * 60)
    if not args.skip_reports:
        logger.info("下一步: 运行 02_download_pdfs.py 下载研报PDF")
    if not args.skip_financial:
        logger.info("下一步: 运行 04_build_chunks.py 重新构建索引")


if __name__ == "__main__":
    main()
