"""
FinAgent Step 6: 表格自然语言化（Parent-Child Retrieval 的 Child 生成）
用途：把 PDF 解析出的表格转成两种 Child chunk，修复"表格直接 embedding 效果差"
支持两种输入源:
    - source=marker       : 从 marker_all_results.json 正则抠出 Markdown 表
    - source=mineru_cleaned: 从 table_parents.jsonl 读 HTML 表（2026-04-19 加;parents 不进召回索引）
依赖：pip install openai tqdm

运行方式：
    # 原型（先跑 50 张验证 prompt 稳定性 + 质量）
    python 06_tabularize_fulltext.py --max_tables 50

    # 全量（默认 Marker 源）
    python 06_tabularize_fulltext.py

    # 用 MinerU 清洗版本
    python 06_tabularize_fulltext.py --source mineru_cleaned

    # 断点续跑
    python 06_tabularize_fulltext.py --resume

环境变量（已在 10_generate_sft_data.py 用过同一套）：
    OPENAI_API_KEY=sk-xxx
    OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

产出：
    data/processed/tabular_chunks.jsonl    最终的 Child chunks（narrative + row_facts 双存）
    data/processed/tabularize_raw.jsonl    qwen3-max 原始输出（断点续跑用）
    data/processed/tabularize_stats.json   统计 + 失败样本

设计要点：
    1. 每张表 → 2 种 Child：
        - table_narrative：整段叙述，给 LLM 阅读（回答"XX 盈利预测如何"这类整体问题）
        - table_row_fact：原子事实（公司 + 年份 + 指标 = 值），给 embedding 匹配具体数字
    2. 每个 Child 的 metadata 带 parent_id 回指原 Markdown 表，检索命中时可取回原表
    3. 并发 10，断点续跑（append 到 .ckpt.jsonl）
    4. JSON 格式失败重试 1 次；二次失败打 fallback_needed=True 标记
"""

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============ 配置 ============
def _find_project_root() -> Path:
    """兼容 Mac(backup/finagent_repo/ 两层嵌套)和服务器(Finagent/ 一层)两种结构"""
    here = Path(__file__).resolve()
    for cand in (here.parent, here.parent.parent):
        if (cand / "data").is_dir():
            return cand
    return here.parent

ROOT = _find_project_root()
MARKER_PATH = ROOT / "data/raw/report_parsed/marker_all_results.json"
PDF_MAP_PATH = ROOT / "data/raw/report_pdfs/pdf_map.json"

# Marker 路径输出(历史 170K Child,保持不动)
OUT_RAW = ROOT / "data/processed/tabularize_raw.jsonl"
OUT_CHUNKS = ROOT / "data/processed/tabular_chunks.jsonl"
OUT_STATS = ROOT / "data/processed/tabularize_stats.json"

# MinerU 路径输入(04 产出)和输出(独立文件,避免和 Marker 混淆)
# 2026-04-19 改:读独立的 table_parents.jsonl,不再从 all_chunks.jsonl 过滤
# Parent 不进召回索引 → all_chunks.jsonl 只含会被召回的 chunk
TABLE_PARENTS_PATH = ROOT / "data/processed/table_parents.jsonl"
OUT_RAW_MINERU = ROOT / "data/processed/tabularize_raw_mineru.jsonl"
OUT_CHUNKS_MINERU = ROOT / "data/processed/tabular_chunks_mineru.jsonl"
OUT_STATS_MINERU = ROOT / "data/processed/tabularize_stats_mineru.json"

MODEL = "qwen3-max"
CONCURRENCY = 25   # qwen3-max 60 QPM,平均表处理 ~30s,25 并发 ≈ 50 QPM(留 17% 余量抗抖动)
CONTEXT_CHARS = 300  # 表格前 300 字作为上下文

# Markdown 表格识别
RE_TABLE_LINE = re.compile(r'^\s*\|.+\|\s*$')
RE_TABLE_SEP_LINE = re.compile(r'^\s*\|[\s:\-|]+\|\s*$')


# ============ Prompt ============

SYSTEM_PROMPT = """你是金融研报表格解析专家。输入一段 A 股研报文本 + 1 张表格（Markdown pipe 语法或 HTML）+ 研报元数据。
你需要理解这张表在讲什么，并输出严格的 JSON。

输出要求：
1. narrative 是给 LLM 阅读的一段叙述，200-400 字，必须带公司名、年份/场景、单位，语句通顺
2. row_facts 是给向量检索的原子事实列表。每条事实独立语义（公司 + 年份/场景 + 指标 = 值 + 单位），用于 embedding 匹配具体查询
3. 所有数字必须从原表精确保留，不要改写
4. 表头如果是年份列（2024A/2025E 等），header_type="year"；如果是情景列（乐观/中性/悲观），header_type="scenario"；如果是业务分部，header_type="business"
5. 如果是情景分析表，用"乐观情景/中性情景/悲观情景"代替年份
6. 输出严格的 JSON object，不要解释，不要 markdown 包裹
7. row_facts 数量控制在 30-50 条以内（总数上限 50）：
   - 优先包含：营收、净利润、毛利率、净利率、ROE、ROA、EPS、PE、PB、EV/EBITDA、营收增长率、净利润增长率
   - 每个核心指标 × 所有年份/场景 = 若干条
   - 略过次要明细（如"其他应收款"、"预付账款"、"存货周转天数"等）
   - 对于超大合并表（比如"资产负债表+利润表"并排，或含 20+ 指标），只取骨架核心指标，不要全展开
8. narrative 不受此限制，保持语义完整覆盖所有关键数据

输出 JSON 字段：
{
  "table_type": "profit_forecast|financial_ratio|valuation|balance_sheet|income_statement|cashflow|business_segment|scenario_analysis|comparable_peer|key_indicators|other",
  "caption": "本表主题（一句话），若上下文有图表标题直接引用",
  "header_type": "year|scenario|business|company|other",
  "narrative": "整段叙述…",
  "row_facts": ["…事实1…", "…事实2…", ...]
}
"""

FEW_SHOT_EXAMPLES = [
    {
        "user": """<context>
## 盈利预测

我们预计藏格矿业 2025-2027 年营业收入分别为 3206/3267/3326 百万元，归母净利润 3711/5477/5958 百万元。
</context>

<chunk_metadata>
公司: 藏格矿业(000408)
机构: 中邮证券
日期: 2025-08-07
</chunk_metadata>

<table>
| 项目               | 2024A | 2025E | 2026E | 2027E |
|-------------------|-------|-------|-------|-------|
| 营业收入(百万元)    | 3251  | 3206  | 3267  | 3326  |
| 归属母公司净利润(百万元)| 2580  | 3711  | 5477  | 5958  |
| EPS(元/股)         | 1.64  | 2.36  | 3.49  | 3.79  |
| 市盈率(P/E)        | 29.70 | 20.65 | 13.99 | 12.86 |
</table>""",
        "assistant": """{
  "table_type": "profit_forecast",
  "caption": "藏格矿业 2024-2027 年盈利预测与估值",
  "header_type": "year",
  "narrative": "藏格矿业(000408) 2024-2027 年盈利预测与估值（中邮证券 2025-08-07）：2024 年实际营业收入 3251 百万元，2025-2027 年预测营收分别为 3206、3267、3326 百万元；2024 年归母净利润 2580 百万元，2025-2027 年预测归母净利润分别为 3711、5477、5958 百万元；对应 EPS 分别为 1.64、2.36、3.49、3.79 元/股；按当前股价对应市盈率 P/E 分别为 29.70、20.65、13.99、12.86 倍。",
  "row_facts": [
    "藏格矿业(000408) 2024年 营业收入 3251 百万元",
    "藏格矿业(000408) 2025年预测 营业收入 3206 百万元",
    "藏格矿业(000408) 2026年预测 营业收入 3267 百万元",
    "藏格矿业(000408) 2027年预测 营业收入 3326 百万元",
    "藏格矿业(000408) 2024年 归母净利润 2580 百万元",
    "藏格矿业(000408) 2025年预测 归母净利润 3711 百万元",
    "藏格矿业(000408) 2026年预测 归母净利润 5477 百万元",
    "藏格矿业(000408) 2027年预测 归母净利润 5958 百万元",
    "藏格矿业(000408) 2024年 EPS 1.64 元/股",
    "藏格矿业(000408) 2025年预测 EPS 2.36 元/股",
    "藏格矿业(000408) 2026年预测 EPS 3.49 元/股",
    "藏格矿业(000408) 2027年预测 EPS 3.79 元/股",
    "藏格矿业(000408) 2024年 市盈率P/E 29.70 倍",
    "藏格矿业(000408) 2025年预测 市盈率P/E 20.65 倍",
    "藏格矿业(000408) 2026年预测 市盈率P/E 13.99 倍",
    "藏格矿业(000408) 2027年预测 市盈率P/E 12.86 倍"
  ]
}"""
    },
    {
        "user": """<context>
## 估值与关键指标
</context>

<chunk_metadata>
公司: 招商蛇口(001979)
机构: 中信证券
日期: 2026-02-03
</chunk_metadata>

<table>
| 收盘价(元)        | 10.08      |
|------------------|------------|
| 一年最低/最高价    | 8.43/10.98 |
| 市净率(倍)        | 0.92       |
| 流通A股市值(百万元)| 84,813.35  |
| 总市值(百万元)     | 90,881.60  |
</table>""",
        "assistant": """{
  "table_type": "key_indicators",
  "caption": "招商蛇口股价与市值关键指标",
  "header_type": "other",
  "narrative": "招商蛇口(001979) 关键指标（中信证券 2026-02-03）：收盘价 10.08 元；过去一年股价区间 8.43 至 10.98 元；市净率 0.92 倍；流通 A 股市值 84,813.35 百万元；总市值 90,881.60 百万元。",
  "row_facts": [
    "招商蛇口(001979) 收盘价 10.08 元",
    "招商蛇口(001979) 一年最低价 8.43 元",
    "招商蛇口(001979) 一年最高价 10.98 元",
    "招商蛇口(001979) 市净率 0.92 倍",
    "招商蛇口(001979) 流通A股市值 84,813.35 百万元",
    "招商蛇口(001979) 总市值 90,881.60 百万元"
  ]
}"""
    }
]


# ============ 数据结构 ============

@dataclass
class TableTask:
    """单张待解析的表格"""
    task_id: str            # 唯一 id: "{pdf_file}::table_{idx}"
    pdf_file: str
    table_idx: int          # 该 report 内第几张表
    context: str            # 前 300 字上下文
    table_md: str           # Markdown 表格原文
    metadata: dict          # 继承的 chunk 元数据（stock_code/name/date/institution）


# ============ 表格提取 ============

def _extract_tables_from_text(text: str) -> list[tuple[str, str, int, int]]:
    """
    从 Marker 一份 report 的 text 里找所有 Markdown 表格。
    返回 [(context, table_md, start_line, end_line), ...]
    """
    lines = text.split('\n')
    n = len(lines)
    tables = []
    i = 0
    while i < n:
        is_start = (
            i + 1 < n
            and RE_TABLE_LINE.match(lines[i])
            and RE_TABLE_SEP_LINE.match(lines[i + 1])
        )
        if is_start:
            j = i + 2
            while j < n and RE_TABLE_LINE.match(lines[j]):
                j += 1
            # 过滤：至少 3 行（表头 + 分隔 + 数据 ≥1 行）
            if j - i >= 3:
                table_md = '\n'.join(lines[i:j])
                # 上下文：表格前 CONTEXT_CHARS 字符
                prefix_text = '\n'.join(lines[max(0, i - 30):i])
                context = prefix_text[-CONTEXT_CHARS:]
                tables.append((context, table_md, i, j))
            i = j
        else:
            i += 1
    return tables


def extract_all_tables(marker_results: list[dict], pdf_map: dict) -> list[TableTask]:
    """扫全部 report，返回所有 TableTask"""
    tasks = []
    for r in marker_results:
        pdf_file = r['file']
        text = r.get('text', '')
        if not text:
            continue
        meta = pdf_map.get(pdf_file, {})
        # 跳过元数据完全缺失的
        if not meta.get('stock_code'):
            continue

        tables = _extract_tables_from_text(text)
        for idx, (context, table_md, _, _) in enumerate(tables):
            tasks.append(TableTask(
                task_id=f"{pdf_file}::table_{idx}",
                pdf_file=pdf_file,
                table_idx=idx,
                context=context.strip(),
                table_md=table_md,
                metadata={
                    'stock_code': meta.get('stock_code', ''),
                    'stock_name': meta.get('stock_name', ''),
                    'institution': meta.get('institution', ''),
                    'date': str(meta.get('date', ''))[:10],
                    'report_title': meta.get('report_title', ''),
                    'industry': meta.get('industry', ''),
                },
            ))
    return tasks


def extract_all_tables_mineru(parents_path: Path) -> list[TableTask]:
    """
    从 04 产出的 table_parents.jsonl 读 parent records → TableTask

    每行是一个扁平 record(见 04_build_chunks.py::_build_table_parent_record):
      parent_id / pdf_file / page_idx / table_md / table_caption / table_footnote
      current_section / stock_code / stock_name / institution / date / report_title / industry
    """
    tasks = []
    with open(parents_path, encoding='utf-8') as f:
        for line in f:
            p = json.loads(line)
            if not p.get('stock_code'):
                continue
            table_md = (p.get('table_md') or '').strip()
            if not table_md:
                continue

            context_parts = []
            if p.get('current_section'):
                context_parts.append(p['current_section'])
            if p.get('table_caption'):
                context_parts.append(p['table_caption'])
            context = '\n'.join(context_parts)

            tasks.append(TableTask(
                task_id=p['parent_id'],
                pdf_file=p.get('pdf_file', ''),
                table_idx=0,   # parent_id 已编号
                context=context,
                table_md=table_md,
                metadata={
                    'stock_code': p.get('stock_code', ''),
                    'stock_name': p.get('stock_name', ''),
                    'institution': p.get('institution', ''),
                    'date': str(p.get('date') or '')[:10],  # None 转 '' 再切,避免 'None'[:10]='None'
                    'report_title': p.get('report_title', ''),
                    'industry': p.get('industry', ''),
                    'table_caption': p.get('table_caption', ''),
                    'table_footnote': p.get('table_footnote', ''),
                    'current_section': p.get('current_section', ''),
                    'page_idx': p.get('page_idx', -1),
                },
            ))
    return tasks


# ============ qwen3-max 调用 ============

def build_user_prompt(task: TableTask) -> str:
    return f"""<context>
{task.context}
</context>

<chunk_metadata>
公司: {task.metadata['stock_name']}({task.metadata['stock_code']})
机构: {task.metadata['institution']}
日期: {task.metadata['date']}
</chunk_metadata>

<table>
{task.table_md}
</table>"""


def build_messages(task: TableTask) -> list[dict]:
    """system + few-shot + 当前 user"""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        msgs.append({"role": "user", "content": ex["user"]})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": build_user_prompt(task)})
    return msgs


async def tabularize_one(client: AsyncOpenAI, sem: asyncio.Semaphore,
                         task: TableTask, retries: int = 1) -> dict:
    """调 qwen3-max 把一张表转 JSON。返回 {task_id, status, data, error}"""
    async with sem:
        msgs = build_messages(task)
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=msgs,
                    response_format={"type": "json_object"},
                    temperature=0 if attempt == 0 else 0.1,
                    max_tokens=5000,  # 兜底上限；主控靠 prompt 的 row_facts ≤50 约束
                )
                raw = resp.choices[0].message.content
                data = json.loads(raw)
                # 基本字段检查
                if not all(k in data for k in ("table_type", "narrative", "row_facts")):
                    raise ValueError(f"缺字段: {list(data.keys())}")
                return {
                    'task_id': task.task_id,
                    'status': 'ok',
                    'data': data,
                    'metadata': task.metadata,
                    'parent_md': task.table_md,
                    'pdf_file': task.pdf_file,
                }
            except (json.JSONDecodeError, ValueError, Exception) as e:
                last_err = str(e)
                if attempt < retries:
                    await asyncio.sleep(1.5)
                continue
        return {
            'task_id': task.task_id,
            'status': 'failed',
            'error': last_err,
            'metadata': task.metadata,
            'parent_md': task.table_md,
            'pdf_file': task.pdf_file,
        }


# ============ 断点续跑 ============

def load_done_ids(ckpt_path: Path) -> set[str]:
    done = set()
    if ckpt_path.exists():
        with open(ckpt_path, encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add(d['task_id'])
                except Exception:
                    continue
    return done


def append_result(ckpt_path: Path, result: dict):
    """
    断点续跑写入。单 consumer 顺序调用(main_async:535-537 的 `async for coro in
    as_completed` 在主协程里顺序执行),不会出现多协程并发 append,无需加 asyncio.Lock。
    tabularize_one 本身不碰 ckpt_path。如果以后改成实时写(放进 tabularize_one 内),
    必须在这里加锁,且用 `open('a', buffering=0)` 或整行一次性写入。
    """
    with open(ckpt_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


# ============ Child chunk 生成 ============

def build_child_chunks(raw_path: Path, parser: str = 'marker') -> list[dict]:
    """读 tabularize raw.jsonl，生成 narrative + row_fact 双版本 chunk

    2026-04-20:marker 和 mineru_cleaned 统一用 parent_md 字段存 Markdown 表格
      - marker: 原本就是 Markdown pipe syntax
      - mineru_cleaned: 04 建库时已经 HTML→Markdown
    """
    # 防御:旧版 raw jsonl 没有 pdf_file 字段,会让 tabular Child 的 pdf_file 空串 → None
    # → 方案 B 按 pdf_file 聚合时所有 None 折同桶,完全错乱。要求用户重跑 06。
    with open(raw_path, encoding='utf-8') as f:
        first_line = f.readline()
    if not first_line:
        raise RuntimeError(
            f"{raw_path} 为空文件,rm 后重跑 06_tabularize_fulltext.py"
        )
    try:
        first_record = json.loads(first_line)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"{raw_path} 首行 JSON 损坏({e}),rm 后重跑 06_tabularize_fulltext.py"
        ) from e
    if 'pdf_file' not in first_record:
        raise RuntimeError(
            f"{raw_path} 是旧版本 raw jsonl(缺 pdf_file 字段),"
            f"rm 后重跑 06_tabularize_fulltext.py 重新生成"
        )

    chunks = []
    with open(raw_path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('status') != 'ok':
                continue
            task_id = r['task_id']
            data = r['data']
            meta = r['metadata']
            parent_text = r['parent_md']

            base_meta = {
                'source_type': 'report_tabular',
                'parser': parser,
                'stock_code': meta['stock_code'],
                'stock_name': meta['stock_name'],
                'institution': meta['institution'],
                'date': meta['date'],
                'report_title': meta['report_title'],
                'industry': meta['industry'],
                'table_type': data.get('table_type', ''),
                'caption': data.get('caption', ''),
                'header_type': data.get('header_type', ''),
                'parent_id': task_id,
                'parent_md': parent_text,
                'page_idx': meta.get('page_idx', -1),
                'pdf_file': r.get('pdf_file', ''),
            }
            if parser == 'mineru_cleaned':
                base_meta['table_caption'] = meta.get('table_caption', '')
                base_meta['table_footnote'] = meta.get('table_footnote', '')
                base_meta['current_section'] = meta.get('current_section', '')

            # Child 1: narrative(整段叙述,给 LLM 阅读用)—— 每表 1 条
            narr = data.get('narrative', '').strip()
            if narr and len(narr) >= 30:
                chunks.append({
                    'text': narr,
                    'metadata': {
                        **base_meta,
                        'chunk_id':     f"{task_id}_narrative",
                        'chunk_method': 'table_narrative',
                    },
                })

            # Child 2: row_facts(原子事实,给 embedding 匹配用)—— 每表 N 条
            for idx, fact in enumerate(data.get('row_facts', [])):
                fact = fact.strip()
                if fact and len(fact) >= 10:
                    chunks.append({
                        'text': fact,
                        'metadata': {
                            **base_meta,
                            'chunk_id':     f"{task_id}_rowfact_{idx}",
                            'chunk_method': 'table_row_fact',
                        },
                    })

    return chunks


# ============ 主流程 ============

async def main_async(args):
    # 根据 source 选择输入 + 输出路径
    if args.source == 'marker':
        logger.info(f"加载 Marker: {MARKER_PATH}")
        with open(MARKER_PATH, encoding='utf-8') as f:
            marker_results = json.load(f)
        with open(PDF_MAP_PATH, encoding='utf-8') as f:
            pdf_map = json.load(f)
        logger.info(f"  reports: {len(marker_results)}, pdf_map: {len(pdf_map)}")
        all_tasks = extract_all_tables(marker_results, pdf_map)
        out_raw, out_chunks, out_stats = OUT_RAW, OUT_CHUNKS, OUT_STATS

    elif args.source == 'mineru_cleaned':
        logger.info(f"加载 table_parents: {TABLE_PARENTS_PATH}")
        if not TABLE_PARENTS_PATH.exists():
            raise FileNotFoundError(
                f"{TABLE_PARENTS_PATH} 不存在,请先跑 04_build_chunks.py --parser mineru_cleaned"
            )
        all_tasks = extract_all_tables_mineru(TABLE_PARENTS_PATH)
        out_raw, out_chunks, out_stats = OUT_RAW_MINERU, OUT_CHUNKS_MINERU, OUT_STATS_MINERU

    else:
        raise ValueError(f"未知 source: {args.source}")

    logger.info(f"抽取表格: {len(all_tasks)} 张")

    # 断点续跑 + 原型限制
    done_ids = load_done_ids(out_raw) if args.resume else set()
    if done_ids:
        logger.info(f"断点续跑：已完成 {len(done_ids)} 张，跳过")
    pending = [t for t in all_tasks if t.task_id not in done_ids]
    if args.max_tables > 0:
        pending = pending[:args.max_tables]
        logger.info(f"原型模式：只跑 {len(pending)} 张")
    if not pending:
        logger.info("没有待处理的表格，进入 Child 生成阶段")
    else:
        logger.info(f"待处理: {len(pending)} 张，并发 {CONCURRENCY}")

    # 并发调 qwen3-max
    if pending:
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(CONCURRENCY)
        coros = [tabularize_one(client, sem, t) for t in pending]

        ok, fail = 0, 0
        for coro in atqdm.as_completed(coros, total=len(coros), desc="tabularize"):
            result = await coro
            append_result(out_raw, result)
            if result['status'] == 'ok':
                ok += 1
            else:
                fail += 1

        logger.info(f"处理完成: 成功 {ok}, 失败 {fail}")

    # 生成 Child chunks
    logger.info("生成 Child chunks（narrative + row_facts）...")
    chunks = build_child_chunks(out_raw, parser=args.source)
    with open(out_chunks, 'w', encoding='utf-8') as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    logger.info(f"写入 {len(chunks)} 条 Child chunks → {out_chunks}")

    # 统计
    from collections import Counter
    method_dist = Counter()
    type_dist = Counter()
    for c in chunks:
        method_dist[c['metadata']['chunk_method']] += 1
        type_dist[c['metadata'].get('table_type', 'unknown')] += 1

    fail_samples = []
    with open(out_raw, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('status') == 'failed' and len(fail_samples) < 10:
                fail_samples.append({'task_id': r['task_id'], 'error': r.get('error', '')})

    stats = {
        'source': args.source,
        'total_tables_extracted': len(all_tasks),
        'total_tasks_processed': len(pending) + len(done_ids),
        'total_chunks_generated': len(chunks),
        'chunk_method_dist': dict(method_dist),
        'table_type_dist': dict(type_dist),
        'failed_samples': fail_samples,
    }
    with open(out_stats, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"统计已存 {out_stats}")
    logger.info(f"  chunk_method: {dict(method_dist)}")
    logger.info(f"  table_type:   {dict(type_dist)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='marker',
                        choices=['marker', 'mineru_cleaned'],
                        help='表格来源: marker=从 Marker markdown 抽取, mineru_cleaned=从 04 产出的 table_parent chunks')
    parser.add_argument('--max_tables', type=int, default=0,
                        help='限制处理表格数（0=全量），用于原型验证')
    parser.add_argument('--resume', action='store_true', help='从 .ckpt.jsonl 断点续跑')
    args = parser.parse_args()

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
