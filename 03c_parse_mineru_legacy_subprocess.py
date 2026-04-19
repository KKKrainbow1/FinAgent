import os
import json
import time
import argparse
import logging
import subprocess
import glob
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============ 配置 ============
PDF_DIR = "./data/raw/report_pdfs"
OUTPUT_DIR = "./data/raw/mineru_output"  # MinerU 临时输出目录
RESULT_FILE = "./data/raw/report_parsed/mineru_200_results.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)


def run_mineru_single(pdf_path: str, output_dir: str) -> dict:
    """
    用 MinerU 命令行处理单个 PDF
    返回: {"file": str, "text": str, "time": float} 或 None
    """
    start = time.time()
    
    try:
        # MinerU 2.x 命令行：mineru（不是 magic-pdf）
        result = subprocess.run(
            ["mineru", "-p", pdf_path, "-o", output_dir],
            capture_output=True,
            text=True,
            timeout=300  # 单篇超时5分钟
        )
        
        elapsed = time.time() - start
        
        # MinerU 2.x 输出路径：{output_dir}/{pdf_name}/hybrid_auto/*.md
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        md_pattern = os.path.join(output_dir, pdf_name, "hybrid_auto", "*.md")
        md_files = glob.glob(md_pattern)
        
        # 也试试其他可能的路径
        if not md_files:
            md_pattern = os.path.join(output_dir, pdf_name, "**", "*.md")
            md_files = glob.glob(md_pattern, recursive=True)
        
        if md_files:
            with open(md_files[0], 'r', encoding='utf-8') as f:
                text = f.read()
            
            # 跑完后删除 images 目录节省空间
            images_dir = os.path.join(output_dir, pdf_name, "hybrid_auto", "images")
            if os.path.exists(images_dir):
                import shutil
                shutil.rmtree(images_dir, ignore_errors=True)
            
            return {"file": os.path.basename(pdf_path), "text": text, "time": elapsed}
        else:
            logger.warning(f"  未找到输出文件: {md_pattern}")
            if result.stderr:
                logger.warning(f"  stderr: {result.stderr[:300]}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.warning(f"  超时: {pdf_path}")
        return None
    except Exception as e:
        logger.error(f"  失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="MinerU PDF 提取")
    parser.add_argument("--max_pdfs", type=int, default=0, help="最多处理几篇（0=全部）")
    parser.add_argument("--sample", action="store_true", help="随机抽样而非按顺序")
    args = parser.parse_args()
    
    # 获取所有 PDF
    all_pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    logger.info(f"共 {len(all_pdfs)} 篇 PDF")
    
    # 加载已完成的结果（断点续传）
    results = []
    done_files = set()
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
            done_files = {r['file'] for r in results}
        logger.info(f"已完成 {len(done_files)} 篇，跳过")
    
    # 过滤已完成的
    remaining = [f for f in all_pdfs if f not in done_files]
    
    # 随机抽样
    if args.sample and args.max_pdfs > 0:
        remaining = random.sample(remaining, min(args.max_pdfs, len(remaining)))
    elif args.max_pdfs > 0:
        remaining = remaining[:args.max_pdfs]
    
    logger.info(f"本次处理: {len(remaining)} 篇")
    
    # 处理
    total_start = time.time()
    success = 0
    fail = 0
    for i, pdf_file in enumerate(remaining):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        
        result = run_mineru_single(pdf_path, OUTPUT_DIR)
        
        if result:
            results.append(result)
            success += 1
            total_elapsed = time.time() - total_start
            avg = total_elapsed / (i + 1)
            remain_min = avg * (len(remaining) - i - 1) / 60
            logger.info(f"[{i+1}/{len(remaining)}] {result['time']:.1f}秒 | "
                       f"{len(result['text'])}字符 | 剩余{remain_min:.0f}分钟 | {pdf_file}")
        else:
            fail += 1
            logger.warning(f"[{i+1}/{len(remaining)}] 失败 | {pdf_file}")
        
        # 每 20 篇保存 checkpoint
        if (i + 1) % 20 == 0:
            with open(RESULT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"  [checkpoint] 已保存 {len(results)} 篇 | 成功{success} 失败{fail}")
    
    # 最终保存
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 统计
    if results:
        times = [r['time'] for r in results]
        texts = [len(r['text']) for r in results]
        logger.info("=" * 60)
        logger.info(f"完成! 成功{success}篇, 失败{fail}篇")
        logger.info(f"平均耗时: {sum(times)/len(times):.1f}秒/篇")
        logger.info(f"总耗时: {sum(times)/3600:.1f}小时")
        logger.info(f"平均字符数: {sum(texts)/len(texts):.0f}")
        logger.info(f"保存到: {RESULT_FILE}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
