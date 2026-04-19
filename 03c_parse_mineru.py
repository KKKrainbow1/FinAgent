"""
FinAgent Step 3c: MinerU VLM 解析(vllm-server + vlm-http-client 架构)

## 为什么用 vllm-server 架构?
单进程 `mineru` CLI 每次冷启动 ~30s 加载 VLM 权重,批量处理 1,482 份 PDF 根本不现实。
改用 `mineru-vllm-server` 常驻 GPU + `vlm-http-client` 并发 HTTP 调用的架构:
  - Server 一次加载,一直常驻,服务端 `--max-num-seqs 32` 自动 batching
  - 客户端是轻量 mineru CLI 子进程,可以本机 / 远程并发启多个
  - 实测吞吐:0.58 秒/页(RTX PRO 6000 96GB,batch 自动填满)
  - 外推:1,482 份全量 ~4.3 小时

## 架构图

```
    Python ThreadPool (本脚本,并发 8~16 份 PDF)
            ↓ subprocess
    mineru -b vlm-http-client -u http://xxx   (CLI 客户端)
            ↓ HTTP POST PDF 图像
    mineru-vllm-server (服务器常驻,GPU 独占)
            ↓ VLM 推理
    content_list.json / *.md / images/ 写入本地磁盘
```

## 前置步骤(服务器端一次性操作,不在本脚本)

```bash
# 1. 起 conda env(一次性)
conda activate mineru_env

# 2. 拉起 vllm-server(前台启动,ctrl-C 停)
mineru-vllm-server \\
    --model /root/autodl-tmp/models/MinerU2.5-Pro-2604-1.2B \\
    --gpu-memory-utilization 0.85 \\
    --max-model-len 8192 \\
    --max-num-seqs 32 \\
    --host 0.0.0.0 --port 30000
```

首次启动需要 30-60s warmup(加载权重 + torch compile),之后常驻。

## 本脚本使用

```bash
# 全量
python 03c_parse_mineru.py \\
    --pdf_dir ./data/raw/report_pdfs \\
    --output_dir ./data/raw/report_parsed/mineru \\
    --server_url http://localhost:30000 \\
    --concurrency 8

# 原型(只跑 20 份)
python 03c_parse_mineru.py --limit 20

# 断点续跑(跳过已产出 content_list.json 的 PDF)
python 03c_parse_mineru.py --skip_existing
```

## 输出目录结构

```
{output_dir}/
├── {stem_1}/
│   └── vlm/
│       ├── {stem_1}_content_list.json    ← 核心产出,下游 03d 消费
│       ├── {stem_1}.md
│       ├── {stem_1}_middle.json
│       └── images/
├── {stem_2}/
│   └── vlm/
│       └── ...
└── ...
```

产出汇总统计:`{output_dir}/../mineru_parse_results.json`

## 依赖

- 服务器端:`pip install "mineru[core]==3.1.0"`(装 vllm / 权重)
- 客户端(本脚本):`pip install requests tqdm`(轻量,mineru CLI 走 HTTP,不需要本地模型)
"""
import argparse
import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============ server 健康检查 ============

def check_server_alive(server_url: str, timeout: float = 5.0) -> bool:
    """
    ping 几个常见端点确认 vllm-server 活着。
    mineru-vllm-server 基于 vllm,OpenAI 兼容接口通常在 /v1/models。
    """
    for path in ("/v1/models", "/health", "/"):
        try:
            r = requests.get(f"{server_url.rstrip('/')}{path}", timeout=timeout)
            if r.status_code < 500:
                return True
        except requests.RequestException:
            continue
    return False


# ============ 单份 PDF 处理 ============

def parse_one_pdf(pdf_path: Path, output_dir: Path, server_url: str,
                  timeout: int = 300) -> dict:
    """
    启一个 mineru CLI 子进程,用 vlm-http-client backend 处理一份 PDF。

    Returns: {file, stem, status, time_s, content_list_path, error}
      status ∈ {'ok', 'failed', 'timeout', 'no_output'}
    """
    start = time.time()
    stem = pdf_path.stem

    cmd = [
        "mineru",
        "-p", str(pdf_path),
        "-o", str(output_dir),
        "-b", "vlm-http-client",
        "-u", server_url,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            'file': pdf_path.name, 'stem': stem, 'status': 'timeout',
            'time_s': round(time.time() - start, 2),
            'content_list_path': None,
            'error': f'超时 {timeout}s',
        }

    elapsed = round(time.time() - start, 2)
    expected = output_dir / stem / "vlm" / f"{stem}_content_list.json"

    if result.returncode != 0:
        return {
            'file': pdf_path.name, 'stem': stem, 'status': 'failed',
            'time_s': elapsed, 'content_list_path': None,
            'error': (result.stderr or result.stdout or '')[-500:],
        }

    if not expected.exists():
        return {
            'file': pdf_path.name, 'stem': stem, 'status': 'no_output',
            'time_s': elapsed, 'content_list_path': None,
            'error': f'CLI returncode=0 但未产出 {expected.name}',
        }

    return {
        'file': pdf_path.name, 'stem': stem, 'status': 'ok',
        'time_s': elapsed, 'content_list_path': str(expected),
        'error': None,
    }


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(
        description="MinerU VLM 解析(vllm-server + vlm-http-client 架构)",
    )
    parser.add_argument("--pdf_dir", type=str, default="./data/raw/report_pdfs",
                        help="PDF 输入目录")
    parser.add_argument("--output_dir", type=str, default="./data/raw/report_parsed/mineru",
                        help="MinerU 输出根目录")
    parser.add_argument("--server_url", type=str, default="http://localhost:30000",
                        help="mineru-vllm-server URL(--host --port 对应)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="并发 mineru CLI 子进程数(server max-num-seqs=32,建议 4-16)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="单份 PDF 超时秒数")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制 PDF 处理数量(0=全量,用于原型 smoke)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="跳过已产出 content_list.json 的 PDF(断点续跑)")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 检查 server 健康
    logger.info(f"检查 mineru-vllm-server: {args.server_url}")
    if not check_server_alive(args.server_url):
        logger.error(f"❌ server 不可达: {args.server_url}")
        logger.error("请先在服务器端启动(见脚本 docstring 前置步骤):")
        logger.error("  mineru-vllm-server --model <MinerU2.5-Pro-2604-1.2B path> \\")
        logger.error("    --gpu-memory-utilization 0.85 --max-model-len 8192 \\")
        logger.error("    --max-num-seqs 32 --host 0.0.0.0 --port 30000")
        return
    logger.info("✅ server 健康")

    # 2. 扫 PDF 清单
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.error(f"❌ {pdf_dir} 下未找到 PDF")
        return
    logger.info(f"发现 {len(pdfs)} 份 PDF")

    if args.skip_existing:
        before = len(pdfs)
        pdfs = [
            p for p in pdfs
            if not (output_dir / p.stem / "vlm" / f"{p.stem}_content_list.json").exists()
        ]
        logger.info(f"断点续跑: 跳过已解析 {before - len(pdfs)} 份,剩 {len(pdfs)} 份")

    if args.limit > 0:
        pdfs = pdfs[:args.limit]
        logger.info(f"原型模式: 限流至 {len(pdfs)} 份(--limit)")

    if not pdfs:
        logger.info("无待处理 PDF,退出")
        return

    # 3. 并发处理
    logger.info(f"并发 {args.concurrency} × mineru CLI 子进程,timeout={args.timeout}s/份")
    wall_start = time.time()
    results = []
    stats = {'ok': 0, 'failed': 0, 'timeout': 0, 'no_output': 0}
    total_cli_time = 0.0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(parse_one_pdf, pdf, output_dir, args.server_url, args.timeout): pdf
            for pdf in pdfs
        }
        with tqdm(total=len(futures), desc="MinerU parse") as pbar:
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                stats[r['status']] = stats.get(r['status'], 0) + 1
                total_cli_time += r['time_s']
                pbar.update(1)
                pbar.set_postfix(ok=stats['ok'], fail=stats['failed'])

    wall_elapsed = time.time() - wall_start

    # 4. 写汇总结果
    result_path = output_dir.parent / "mineru_parse_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'server_url': args.server_url,
            'concurrency': args.concurrency,
            'total_pdfs': len(pdfs),
            'stats': stats,
            'wall_clock_seconds': round(wall_elapsed, 1),
            'sum_cli_time_seconds': round(total_cli_time, 1),
            'avg_cli_time_per_pdf': round(total_cli_time / max(len(pdfs), 1), 2),
            'speedup_vs_serial': round(total_cli_time / max(wall_elapsed, 1), 2),
            'results': results,
        }, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"✅ 完成: ok={stats['ok']} / failed={stats['failed']} / "
                f"timeout={stats['timeout']} / no_output={stats['no_output']}")
    logger.info(f"   wall-clock: {wall_elapsed:.1f}s, "
                f"CLI 累计: {total_cli_time:.1f}s, "
                f"并发加速比: {total_cli_time/max(wall_elapsed,1):.2f}x")
    logger.info(f"   平均每份(CLI): {total_cli_time/max(len(pdfs),1):.2f}s")
    logger.info(f"   汇总已存: {result_path}")

    # 5. 错误样本前 3 个,便于诊断
    errors = [r for r in results if r['status'] != 'ok']
    if errors:
        logger.warning(f"⚠️  {len(errors)} 份失败,前 3 个错误:")
        for r in errors[:3]:
            err_msg = (r['error'] or '')[:200]
            logger.warning(f"   [{r['stem']}] {r['status']}: {err_msg}")

    logger.info("=" * 60)
    logger.info("下一步: 运行 03d_clean_content_list.py --input-dir <output_dir> 清洗")


if __name__ == '__main__':
    main()
