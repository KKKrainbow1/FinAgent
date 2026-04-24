"""
FinAgent Step 4: Chunk 构建
用途：将所有数据源（研报元数据 + PDF正文 + 财务指标）统一构建为检索chunks
环境：Google Colab
依赖：pip install pandas tqdm

运行方式：
    python 04_build_chunks.py                                    # 处理全部数据（默认用 Marker 解析结果）
    python 04_build_chunks.py --parser mineru                    # 用MinerU md结果（备选）
    python 04_build_chunks.py --parser mineru_cleaned            # 用 03d 清洗后的 content_list（Block-native + Rule-based）
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
import hashlib
import logging
import argparse
from datetime import datetime
from tqdm import tqdm

# ============ 配置 ============
RAW_DIR = "./data/raw"
CHUNK_DIR = "./data/processed"
os.makedirs(CHUNK_DIR, exist_ok=True)

# 解析器 → 结果文件的映射
# 当前生产:MinerU VLM + 03d 清洗(mineru_cleaned)
# Marker 保留作 A/B 实验(V2 时代的主用)
# pdfplumber 已于 2026-04-17 下线(Marker 在表格保真度上完胜),03a 脚本保留作历史对比证据
# mineru_cleaned 指目录(含 03d 清洗后的 content_list_cleaned.json),2026-04-19 新增
# 2026-04-23 路径修正:从 ./data/processed/mineru_compare (24 份样本实验目录) 改成生产路径 ./data/raw/report_parsed/mineru
PARSER_RESULT_FILES = {
    "marker":         os.path.join(RAW_DIR, "report_parsed", "marker_all_results.json"),
    "mineru":         os.path.join(RAW_DIR, "report_parsed", "mineru_200_results.json"),
    "mineru_cleaned": os.path.join(RAW_DIR, "report_parsed", "mineru"),  # 目录
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============ 研报元数据 Chunk 构建 ============

def _safe_str(v) -> str:
    """
    CSV 字段 NaN-safe:NaN → '',其他 str()+strip。
    额外去除内部空格:akshare 对"五粮液"/"新希望"等 3 字股票名返回"五 粮 液"/"新 希 望",
    query 检索时会因为空格导致 BGE character token 和 BM25 切词混乱,召回错位(召回其他白酒)。
    """
    if pd.isna(v):
        return ''
    return str(v).replace(' ', '').replace('\u3000', '').strip()


def _load_pdf_reverse_index(pdf_map_path: str) -> dict:
    """
    读 02_download_pdfs.py 产出的 pdf_map.json,构建反查:
      (stock_code, YYYYMMDD, institution, report_title) → [filename, ...]

    4 元 key(加 title):同日同机构可能发多份(如"系列深度一" + "系列点评十一"),
    3 元 key 会错配第二份 meta 到第一个 filename。加 title 做消歧,冲突率接近 0。
    未下载的 meta row 查不到 → pdf_file = None,和 body 无法聚合(反正也没 body)。
    """
    if not os.path.exists(pdf_map_path):
        raise FileNotFoundError(
            f"pdf_map 不存在: {pdf_map_path}\n"
            f"报告 meta chunk 的 pdf_file 依赖此文件反查,缺失会让 _external_rrf 方案 B 完全失效。\n"
            f"解决:先跑 02_download_pdfs.py 生成 pdf_map.json 再跑 04。"
        )
    with open(pdf_map_path, encoding='utf-8') as f:
        pdf_map = json.load(f)
    reverse = {}
    missing_date = 0
    for filename, meta in pdf_map.items():
        try:
            date_key = pd.to_datetime(meta['date']).strftime('%Y%m%d')
        except (ValueError, TypeError):
            missing_date += 1
            continue
        except KeyError:
            missing_date += 1
            continue
        key = (
            meta.get('stock_code', ''),
            date_key,
            meta.get('institution', ''),
            meta.get('report_title', ''),
        )
        reverse.setdefault(key, []).append(filename)
    if missing_date:
        logger.warning(f"pdf_map 反查:{missing_date} 条 record 缺 date 被跳过")
    logger.info(f"pdf_map 反查:{len(pdf_map)} 个 filename → {len(reverse)} 个 4 元组")
    return reverse


def build_report_chunks(report_path: str,
                        pdf_map_path: str = None,
                        since: str = '2024-01-01') -> list[dict]:
    """
    将研报数据转为检索chunks

    每篇研报 → 1个chunk,包含:
    - 报告标题(核心语义信息)
    - 机构 + 评级(机构观点)
    - 盈利预测数据(硬数据)
    - 行业 + 日期(元信息)
    - pdf_file(_external_rrf 方案 B 按 PDF 聚合必需,从 pdf_map.json 反查)

    2026-04-20 加日期过滤(默认 >= 2024-01-01):
    akshare 拉回的 CSV 覆盖 2017-2026 全时段(~39K),但实际 PDF 下载 MAX_PER_STOCK=5
    按日期倒序,89% 正文集中在 2025-2026。2017-2023 的元数据没有对应正文 chunk,
    召回这些只能给 LLM 孤立的标题+评级+EPS 预测,对"最新业绩"类查询反而是污染。
    过滤后保留 ~10K 条(2024 年后),和实际 PDF 正文时间窗对齐 + 留 1 年缓冲给对比类查询。

    面试追问:为什么不每个字段单独做chunk?
    答:研报摘要本身就很短(一行标题+几个字段),拆开后每个chunk语义太稀疏,
    检索时会匹配到大量无关结果。合在一起语义更完整。
    """
    df = pd.read_csv(report_path, dtype={'股票代码': str})

    # 日期过滤
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    before = len(df)
    df = df[df['日期'] >= pd.Timestamp(since)].copy()
    logger.info(f"研报元数据日期过滤 (>= {since}): {before} → {len(df)} ({len(df)/before*100:.1f}%)")

    # 加载 pdf_map 反查(用默认路径)
    if pdf_map_path is None:
        pdf_map_path = os.path.join(RAW_DIR, "report_pdfs", "pdf_map.json")
    pdf_reverse = _load_pdf_reverse_index(pdf_map_path)

    chunks = []
    hit_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建研报chunks"):
        code = _safe_str(row.get('股票代码'))
        name = _safe_str(row.get('股票简称'))
        title = _safe_str(row.get('报告名称'))
        rating = _safe_str(row.get('东财评级'))
        institution = _safe_str(row.get('机构'))
        # 归一化到大类(东财细分"白酒Ⅱ" → 大类"白酒"),和 industry_alias / build_industry_chunks 对齐
        industry = _get_major_industry(_safe_str(row.get('行业')))
        # date 之前是 str(Timestamp)='2024-01-01 00:00:00';统一成 pdf_map 的 %Y-%m-%d
        date_ts = row.get('日期')
        date = date_ts.strftime('%Y-%m-%d') if pd.notna(date_ts) else ''

        # 反查 pdf_file(没下载的 meta 返回 None,方案 B 时与 body 无法聚合)
        try:
            date_key = pd.to_datetime(date).strftime('%Y%m%d')
        except (ValueError, TypeError):
            date_key = ''
        filenames = pdf_reverse.get((code, date_key, institution, title), [])
        pdf_file = filenames[0] if filenames else None
        if pdf_file:
            hit_count += 1

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

        # 显式 chunk_id(code + date + institution + title 的 md5 前 12 字符)
        # md5 确定性(不受 PYTHONHASHSEED 影响),48-bit 空间,10K 条全库碰撞概率 <0.001%
        title_hash = hashlib.md5(title.encode('utf-8')).hexdigest()[:12]
        cid = f"report_{code}_{date_key}_{institution}_{title_hash}"
        chunk = {
            "text": text,
            "metadata": {
                "chunk_id": cid,
                "source_type": "report",
                "stock_code": code,
                "stock_name": name,
                "institution": institution,
                "rating": rating,
                "industry": industry,
                "date": date,
                "report_title": title,
                "pdf_file": pdf_file,
            }
        }
        chunks.append(chunk)

    logger.info(f"pdf_file 命中率: {hit_count}/{len(chunks)} ({hit_count/max(len(chunks),1)*100:.1f}%)")

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
        # 深度清洗：HTML 标签 / OCR 模板标签 / 页码 / 免责声明截断（2026-04-17 加）
        text = _clean_marker_text(text)

        if len(text) < 50:
            continue

        # 从 pdf_map 获取元数据
        meta = pdf_map.get(filename, {})
        base_metadata = {
            "source_type": "report_fulltext",
            "parser": parser,
            "stock_code": _safe_str(meta.get("stock_code", "")),
            "stock_name": _safe_str(meta.get("stock_name", "")),
            "institution": _safe_str(meta.get("institution", "")),
            "rating": _safe_str(meta.get("rating", "")),
            # industry 归一化到大类,和 build_report_chunks / build_industry_chunks 对齐
            "industry": _get_major_industry(meta.get("industry", "")),
            # date 防御性 normalize:历史 pdf_map 可能是 str(Timestamp)='YYYY-MM-DD 00:00:00'
            # 或 ISO 'YYYY-MM-DDTHH:MM:SS',截到纯 date 避免 Milvus filter/dedup 失败
            "date": (meta.get("date") or "").split(" ")[0].split("T")[0],
            "report_title": _safe_str(meta.get("report_title", "")),
            "pdf_file": filename,
        }

        # 表格感知 chunking（2026-04-17 加）：识别 Markdown 表格 + 正文差异化切分
        chunks = _table_aware_chunks(text, base_metadata, chunk_size, overlap)
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


# ============ Marker 文本清洗 + 表格感知 chunking（2026-04-17 加） ============
#
# 背景：症状扫描显示 64,008 个 fulltext chunk 中：
#   - 30.63% 表格行无分隔行（表头丢失）
#   - 16.27% HTML 标签残留
#   - 8.13% 免责声明样板
# 直接影响 GPT-5.4 审计里 25% 的"跨报告期混用"编造
# 详见 docs/Chunk_Symptom_Scan_20260417.md

_RE_HTML_BR = re.compile(r'<br\s*/?>', re.IGNORECASE)
_RE_HTML_TABLE_TAG = re.compile(r'</?(table|tr|td|th|span|div|p)[^>]*>', re.IGNORECASE)
_RE_HTML_ENTITY = re.compile(r'&(nbsp|amp|lt|gt|quot|emsp|ensp);')
_RE_OCR_TMPL_TAG = re.compile(
    r'\[[a-zA-Z_\u4e00-\u9fff0-9]*?'
    r'(?:finchina|wind|ths|table_[a-zA-Z0-9_]+|asset_table|finance|introduction)'
    r'[a-zA-Z_\u4e00-\u9fff0-9]*?\]',
    re.IGNORECASE
)
_RE_PAGE_NUM = re.compile(r'(?:^|\n)\s*(?:P\s*\d+|第\s*\d+\s*页|page\s*\d+)\s*', re.IGNORECASE)
_RE_DISCLAIMER_HEAD = re.compile(
    r'^(#{1,4}\s*)?(免责声明|分析师声明|评级说明|法律声明|分析师保证|证券评级标准|重要声明)\s*$',
    re.MULTILINE
)


def _clean_marker_text(text: str) -> str:
    """
    Marker 输出的文本清洗
    清洗规则 + 症状扫描触发率：
      B1: HTML 标签残留        16.27%
      B2: OCR 模板标签泄漏      0.20%
      B3: 页码残留              0.05%
      B5: 免责声明样板          8.13%
    """
    # 1. HTML 残留
    text = _RE_HTML_BR.sub(' ', text)
    text = _RE_HTML_TABLE_TAG.sub('', text)
    text = _RE_HTML_ENTITY.sub(' ', text)

    # 2. OCR 模板标签（如 [项ta目ble_FinchinaSimple]、[TABLE_FINANCE]）
    text = _RE_OCR_TMPL_TAG.sub('', text)

    # 3. 页码/页眉（保守：只删独占一行的页码标记）
    text = _RE_PAGE_NUM.sub('\n', text)

    # 4. 免责声明截断：只在文档后 60% 位置触发，避免误伤开头提及风险的段落
    m = _RE_DISCLAIMER_HEAD.search(text)
    if m and m.start() > len(text) * 0.6:
        text = text[:m.start()].rstrip()

    # 5. 空白规整
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ============ Markdown 表格感知 chunking ============

_RE_TABLE_LINE = re.compile(r'^\s*\|.+\|\s*$')
_RE_TABLE_SEP_LINE = re.compile(r'^\s*\|[\s:\-|]+\|\s*$')


def _split_into_blocks(text: str) -> list[dict]:
    """
    把文本切成 [{type: "table"|"prose", content, header}, ...]
    表格 = 连续 |...| 行 + 至少一行 |---|---| 分隔行
    每个 table block 附带 header 字段（表头 + 分隔行），用于超长表按行切时前缀复用
    """
    lines = text.split('\n')
    blocks = []
    i, n = 0, len(lines)

    while i < n:
        is_table_start = (
            i + 1 < n
            and _RE_TABLE_LINE.match(lines[i])
            and _RE_TABLE_SEP_LINE.match(lines[i + 1])
        )
        if is_table_start:
            j = i + 2
            while j < n and _RE_TABLE_LINE.match(lines[j]):
                j += 1
            blocks.append({
                'type': 'table',
                'content': '\n'.join(lines[i:j]),
                'header': lines[i] + '\n' + lines[i + 1],
            })
            i = j
        else:
            j = i
            while j < n:
                if (j + 1 < n
                        and _RE_TABLE_LINE.match(lines[j])
                        and _RE_TABLE_SEP_LINE.match(lines[j + 1])):
                    break
                j += 1
            prose = '\n'.join(lines[i:j]).strip()
            if prose:
                blocks.append({'type': 'prose', 'content': prose, 'header': None})
            i = j

    return blocks


def _table_chunks(block: dict, base_metadata: dict, chunk_size: int = 512) -> list[dict]:
    """
    表格 block → chunks
      整表 ≤ chunk_size → 一个原子 chunk          (chunk_method=table_atomic)
      整表 > chunk_size → 按行切，每块复用表头    (chunk_method=table_split)
    """
    content = block['content']
    header = block['header']

    if len(content) <= chunk_size:
        return [{
            'text': content,
            'metadata': {**base_metadata, 'chunk_method': 'table_atomic'}
        }]

    lines = content.split('\n')
    body_lines = lines[2:]
    chunks, current_rows = [], []
    current_len = len(header)
    chunk_idx = 0

    for row in body_lines:
        row_len = len(row) + 1
        if current_rows and current_len + row_len > chunk_size:
            chunks.append({
                'text': header + '\n' + '\n'.join(current_rows),
                'metadata': {
                    **base_metadata,
                    'chunk_method': 'table_split',
                    'chunk_index': chunk_idx,
                }
            })
            chunk_idx += 1
            current_rows = [row]
            current_len = len(header) + row_len
        else:
            current_rows.append(row)
            current_len += row_len

    if current_rows:
        chunks.append({
            'text': header + '\n' + '\n'.join(current_rows),
            'metadata': {
                **base_metadata,
                'chunk_method': 'table_split',
                'chunk_index': chunk_idx,
            }
        })
    return chunks


def _table_aware_chunks(text: str, metadata: dict,
                       chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """
    表格感知 chunking 主入口
    流程：split_into_blocks → table block 走 _table_chunks；prose block 走 _sliding_window_chunks
    """
    blocks = _split_into_blocks(text)
    chunks = []
    for block in blocks:
        if block['type'] == 'table':
            chunks.extend(_table_chunks(block, metadata, chunk_size))
        else:
            chunks.extend(_sliding_window_chunks(block['content'], metadata, chunk_size, overlap))
    return chunks


# ============ MinerU Prose Parent-Child + fixed_window Chunking（2026-04-20 V3） ============
#
# 历史演进:
#   V1 (Marker)  : 512 字滑窗 + 64 overlap
#   V2           : Block-native + Rule-based(35 个主题词切)
#   V3 (当前)    : **Parent-Child 解耦** — section 整段作 Parent,section 内 fixed_window 300+60 切 Child
#
# 为什么 V3:
#   - Embedding 和 LLM 阅读的目标天然冲突(embedding 想短 / LLM 想长)
#   - V2 切碎后 LLM 看不到 "首先→其次→综上" 完整论述链,容易编造结论
#   - V3 和表格 Parent-Child 完全同构:Child 做 embedding / Parent 给 LLM
#
# Section 边界:MinerU 的 text_level=1 标记
# 详见 docs/Finagent项目介绍.md 3.6 / Chunk_Alignment 12 节

# 复用 03d 的作者签名正则,避免假 text_level=1 作 section 锚点
_RE_AUTHOR_TITLE_SECTION = re.compile(
    r'^(分析师|研究员|联系人|证券分析师|首席分析师)\s*[:：]\s*\S{2,6}$'
)
_RE_DATE_SECTION = re.compile(r'^\d{4}\s*年\s*\d{1,2}\s*月')
_RE_RATING_SECTION = re.compile(r'^(强烈推荐|买入|增持|中性|回避|卖出)')


def _is_real_section(section: dict) -> bool:
    """过滤假 text_level=1(作者签名 / 评级简写 / 日期等)"""
    title = (section.get('title') or '').strip()
    if len(title) < 3:
        return False
    if _RE_AUTHOR_TITLE_SECTION.match(title):
        return False
    if _RE_RATING_SECTION.match(title):
        return False
    if _RE_DATE_SECTION.match(title):
        return False
    return True


def _collect_sections(blocks: list[dict], pdf_stem: str) -> list[dict]:
    """
    按 text_level=1 聚合 blocks 为 sections
    返回: [{id, title, text, page_idx}, ...]
    """
    sections = []
    current_title = ""
    current_texts = []
    current_page = -1

    def flush():
        if current_texts:
            sid = f"{pdf_stem}_sec_{len(sections)}"
            sections.append({
                'id': sid,
                'title': current_title,
                'text': '\n\n'.join(current_texts),
                'page_idx': current_page,
            })

    for b in blocks:
        if b.get('type') != 'text':
            continue
        text = (b.get('text') or '').strip()
        if not text:
            continue
        if b.get('text_level') == 1:
            flush()
            current_title = text
            current_texts = []
            current_page = b.get('page_idx', -1)
        else:
            if not current_texts:
                current_page = b.get('page_idx', -1)
            current_texts.append(text)
    flush()
    return sections


def _fixed_window_chunks(section: dict, base_metadata: dict,
                         chunk_size: int = 300, overlap: int = 60) -> list[dict]:
    """
    Section 内固定窗口 + overlap 切 Child。
    每个 Child 的 metadata 冗余存 section 的 id / title / text(Parent-Child zero-join)。
    """
    text = section['text'].strip()
    sid = section['id']
    section_meta = {
        **base_metadata,
        'section_id':    sid,
        'section_title': section['title'],
        'section_text':  text[:3000],       # 冗余存 Parent,最长 3000 字防过长
        'page_idx':      section.get('page_idx', -1),
    }

    chunks = []

    # section 本身短于 chunk_size → 整块作 1 个 Child,不切
    if len(text) <= chunk_size:
        if len(text) >= 20:
            chunks.append({
                'text': text,
                'metadata': {**section_meta,
                             'chunk_id':     f"{sid}_c0",
                             'chunk_method': 'fixed_window',
                             'chunk_index':  0},
            })
        return chunks

    # section 长于 chunk_size → 滑窗切
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        ct = text[start:end].strip()
        if len(ct) >= 50:   # 尾部太短丢弃
            chunks.append({
                'text': ct,
                'metadata': {**section_meta,
                             'chunk_id':     f"{sid}_c{idx}",
                             'chunk_method': 'fixed_window',
                             'chunk_index':  idx},
            })
            idx += 1
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def _html_table_to_md(html: str) -> str:
    """
    HTML table → Markdown,密度提升 3-5 倍(消除 `<tr><td>` 开销)。
    - 首选 BeautifulSoup(嵌套 / colspan / 空单元格更稳)
    - 降级 regex(无 bs4 时也能跑)
    - 非表格 HTML 原样返回
    """
    if not html or '<table' not in html.lower():
        return html
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        rows = []
        for tr in soup.find_all('tr'):
            cells = [td.get_text(separator=' ', strip=True)
                     for td in tr.find_all(['td', 'th'])]
            if cells:
                rows.append('| ' + ' | '.join(cells) + ' |')
        if not rows:
            return html
        if len(rows) > 1:
            col = max(r.count('|') - 1 for r in rows)
            rows.insert(1, '|' + '|'.join([' --- '] * col) + '|')
        return '\n'.join(rows)
    except ImportError:
        # 先剥掉 thead/tbody/tfoot,避免被后面的 t[hd] regex 误匹配(word boundary 不够)
        text = re.sub(r'</?(?:thead|tbody|tfoot)\b[^>]*>', '', html, flags=re.IGNORECASE)
        text = re.sub(r'</?(?:div|p|span)[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<tr[^>]*>', '\n| ', text, flags=re.IGNORECASE)
        text = re.sub(r'</tr>', ' |', text, flags=re.IGNORECASE)
        text = re.sub(r'</?t[hd][^>]*>', ' | ', text, flags=re.IGNORECASE)
        text = re.sub(r'</?table[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\|\s*\|', '|', text)
        text = re.sub(r'\n\s*\n+', '\n', text)
        return text.strip()


def _build_table_parent_record(table_block: dict, base_metadata: dict,
                                current_section: str, parent_id: str) -> dict:
    """
    MinerU table block → Parent record (写 data/processed/table_parents.jsonl,不进 all_chunks)

    设计决策(2026-04-19): Parent 是纯 metadata 载体,不应进 embedding 索引
      - 避免 HTML 表污染 embedding(标签分词化、数字碎片)
      - 避免 BM25 召回 `<table><tr><td>` 这类通用 token
      - 只被 Child(narrative/row_fact)通过 parent_id 回取,在 hybrid_search.enrich_with_parent 展开

    06_tabularize_fulltext.py 读这些 record,生成 narrative / row_fact Child,
    Child metadata 会带 parent_md 冗余字段(zero-join 回取)

    2026-04-20:HTML 改存 Markdown(建库时一次性转,密度 ×3-5,LLM 阅读友好,Milvus VARCHAR 压力降)
    """
    caption = ' '.join(table_block.get('table_caption', []))
    footnote = ' '.join(table_block.get('table_footnote', []))
    md = _html_table_to_md(table_block.get('table_body', ''))

    return {
        'parent_id': parent_id,
        'pdf_file': base_metadata.get('pdf_file', ''),
        'page_idx': table_block.get('page_idx', -1),
        'current_section': current_section,
        'table_caption': caption,
        'table_footnote': footnote,
        'table_md': md,
        # 继承研报元数据(便于 06 调 LLM 时构造 prompt + Child metadata)
        'stock_code': base_metadata.get('stock_code', ''),
        'stock_name': base_metadata.get('stock_name', ''),
        'institution': base_metadata.get('institution', ''),
        'rating': base_metadata.get('rating', ''),
        'industry': base_metadata.get('industry', ''),
        'date': base_metadata.get('date', ''),
        'report_title': base_metadata.get('report_title', ''),
    }


def build_fulltext_chunks_mineru(cleaned_dir: str, pdf_map_path: str = None,
                                 chunk_size: int = 300,
                                 overlap: int = 60) -> tuple[list[dict], list[dict]]:
    """
    消费 03d 输出的 *_content_list_cleaned.json,按 **Prose Parent-Child + fixed_window** 切 chunks。

    处理规则:
      1. 先聚合所有 text block 为 sections(按 text_level=1 边界)
      2. 过滤假 section(_is_real_section 排除作者签名/评级/日期)
      3. 每个 section 内走 fixed_window 300+60 overlap 切 Child
         Child metadata 冗余存 section_text(Parent)
      4. table block → 独立 table_parents.jsonl
      5. chart / image → 跳过(文字模型无视觉能力)

    Returns:
        (chunks, parent_records)
          chunks         : 进 all_chunks.jsonl / Milvus 索引(fixed_window Child)
          parent_records : 写 table_parents.jsonl(不进索引,仅供 06 消费)
    """
    from pathlib import Path

    pdf_map = {}
    if pdf_map_path and os.path.exists(pdf_map_path):
        with open(pdf_map_path, 'r', encoding='utf-8') as f:
            pdf_map = json.load(f)

    cleaned_files = sorted(Path(cleaned_dir).rglob('*_content_list_cleaned.json'))
    cleaned_files = [f for f in cleaned_files
                     if '_cleaned_v' not in f.stem and f.stem.endswith('_content_list_cleaned')]
    logger.info(f"MinerU cleaned content_list 文件数: {len(cleaned_files)}")

    all_chunks = []
    all_parents = []
    stats = {'chart': 0, 'image': 0, 'empty_table': 0,
             'sections_total': 0, 'sections_kept': 0}

    for json_path in tqdm(cleaned_files, desc="构建 MinerU fulltext chunks"):
        stem = json_path.name.replace('_content_list_cleaned.json', '')
        pdf_file = f"{stem}.pdf"

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                blocks = json.load(f)
        except Exception as e:
            logger.error(f"跳过 {json_path.name}: {e}")
            continue

        meta = pdf_map.get(pdf_file, {})
        base_metadata = {
            "source_type": "report_fulltext",
            "parser": "mineru_cleaned",
            "stock_code": _safe_str(meta.get("stock_code", "")),
            "stock_name": _safe_str(meta.get("stock_name", "")),
            "institution": _safe_str(meta.get("institution", "")),
            "rating": _safe_str(meta.get("rating", "")),
            "industry": _get_major_industry(meta.get("industry", "")),
            # date 防御性 normalize(同 build_fulltext_chunks 路径,防 pdf_map 残留脏格式)
            "date": (meta.get("date") or "").split(" ")[0].split("T")[0],
            "report_title": _safe_str(meta.get("report_title", "")),
            "pdf_file": pdf_file,
        }

        # --- Prose 路径:聚合 sections → 过滤 → fixed_window 切 ---
        sections = _collect_sections(blocks, stem)
        stats['sections_total'] += len(sections)
        for section in sections:
            if not _is_real_section(section):
                continue
            stats['sections_kept'] += 1
            all_chunks.extend(_fixed_window_chunks(
                section, base_metadata, chunk_size=chunk_size, overlap=overlap,
            ))

        # --- 表格路径:产出独立 parent records ---
        # 遍历时同步追踪 current_section(最近一次遇到的 text_level=1 标题),
        # 传给 _build_table_parent_record 让表格知道自己属于哪个章节,便于 06 LLM 构造 prompt
        # 用 _is_real_section 过滤假 section(作者名 / 日期 / 评级等),和 _collect_sections 一致
        table_counter = 0
        current_section_title = ''
        for block in blocks:
            btype = block.get('type')
            if btype == 'text' and block.get('text_level') == 1:
                raw = (block.get('text') or '').strip()
                if _is_real_section({'title': raw}):
                    current_section_title = raw
                continue
            if btype == 'table':
                if not block.get('table_body', '').strip():
                    stats['empty_table'] += 1
                    continue
                table_counter += 1
                parent_id = f"{stem}_table_{table_counter}"
                all_parents.append(_build_table_parent_record(
                    block, base_metadata, current_section_title, parent_id,
                ))
            elif btype == 'chart':
                stats['chart'] += 1
            elif btype == 'image':
                stats['image'] += 1

    logger.info(f"MinerU fulltext chunks: {len(all_chunks)} 条(fixed_window Child,进索引)")
    logger.info(f"MinerU table parents:  {len(all_parents)} 条(不进索引,06 消费)")
    logger.info(f"  sections: {stats['sections_kept']}/{stats['sections_total']} "
                f"(过滤假 text_level=1 后保留)")
    logger.info(f"  跳过: chart={stats['chart']}, image={stats['image']}, "
                f"empty_table={stats['empty_table']}")
    # 方法分布(预期全是 fixed_window)
    method_counts = {}
    for c in all_chunks:
        m = c['metadata'].get('chunk_method', 'unknown')
        method_counts[m] = method_counts.get(m, 0) + 1
    logger.info(f"  chunk_method 分布: {method_counts}")
    return all_chunks, all_parents


# ============ 财务数据 Chunk 构建 ============

def build_financial_chunks(financial_path: str) -> list[dict]:
    """
    将 stock_financial_analysis_indicator 的结构化数据转为自然语言chunks

    数据来源：ak.stock_financial_analysis_indicator，每期包含86个字段

    策略：每期数据生成3个chunk（盈利+结构+杜邦），而不是原来的4个chunk（利润表/资产负债/现金流/指标）。
    原因：新接口的数据已经是汇总指标，不需要按报表拆分。合并后每个chunk语义更完整，
    检索时一次就能拿到盈利能力+成长性的全貌。
    杜邦分析chunk将净利率+周转率+权益乘数聚合在一起，Agent一次检索即可拿全三因子。

    面试追问：为什么转成自然语言而不是直接存结构化数据？
    答：因为Agent通过自然语言检索，向量检索对结构化数据不友好。
    将"ROE: 15.3%"转成"贵州茅台2024年ROE为15.3%"后，
    query="茅台盈利能力"就能检索到。这是一个工程上的trade-off。
    """
    with open(financial_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    chunks = []

    for code, data in tqdm(all_data.items(), desc="构建财务chunks"):
        # 走 _safe_str 去内部空格("五 粮 液" → "五粮液")
        name = _safe_str(data.get("stock_name", code)) or code

        if "financial_indicators" not in data:
            continue

        # 按日期排序，构建上年同期索引（年报对年报，半年报对半年报）
        records = sorted(data["financial_indicators"],
                         key=lambda r: str(r.get("日期", "")))
        prev_annual = {}    # year -> record（年报）
        prev_semi = {}      # year -> record（半年报）

        for record in records:
            date = str(record.get("日期", "未知日期"))

            # 标注报告期类型，避免Agent混淆年报和半年报
            if date.endswith("12-31"):
                period_label = "年报"
                year = date[:4]
                prev_record = prev_annual.get(str(int(year) - 1))
                prev_annual[year] = record
            elif date.endswith("06-30"):
                period_label = "半年报"
                year = date[:4]
                prev_record = prev_semi.get(str(int(year) - 1))
                prev_semi[year] = record
            else:
                period_label = ""
                prev_record = None

            # Chunk 1: 盈利能力 + 成长性 + 每股指标
            text = _profitability_to_text(code, name, date, record, period_label, prev_record)
            if text:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "chunk_id":   f"financial_{code}_{date}_profitability",
                        "source_type": "financial",
                        "data_type": "profitability",
                        "stock_code": code,
                        "stock_name": name,
                        "date": date,
                    }
                })

            # Chunk 2: 偿债能力 + 运营效率 + 资产结构
            text = _structure_to_text(code, name, date, record, period_label, prev_record)
            if text:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "chunk_id":   f"financial_{code}_{date}_balance_structure",
                        "source_type": "financial",
                        "data_type": "balance_structure",
                        "stock_code": code,
                        "stock_name": name,
                        "date": date,
                    }
                })

            # Chunk 3: 杜邦分析专属（净利率+周转率+权益乘数+ROE）
            text = _dupont_to_text(code, name, date, record, period_label, prev_record)
            if text:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "chunk_id":   f"financial_{code}_{date}_dupont",
                        "source_type": "financial",
                        "data_type": "dupont",
                        "stock_code": code,
                        "stock_name": name,
                        "date": date,
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


def _fmt_with_prev(record: dict, prev_record: dict | None, key: str, label: str, suffix: str) -> str | None:
    """格式化指标值，如果有上年同期数据则附加对比"""
    val = _safe_fmt(record.get(key), suffix)
    if not val:
        return None
    if prev_record is not None:
        prev_val = _safe_fmt(prev_record.get(key), suffix)
        if prev_val:
            return f"{label}{val}（上年同期{prev_val}）"
    return f"{label}{val}"


# 需要附加上年同期对比的核心指标
_YOY_KEYS = {
    "净资产收益率(%)", "销售净利率(%)", "资产负债率(%)",
    "总资产周转率(次)", "主营业务收入增长率(%)", "净利润增长率(%)",
}


def _profitability_to_text(code: str, name: str, date: str, record: dict,
                           period_label: str = "", prev_record: dict = None) -> str:
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
        if key in _YOY_KEYS:
            text = _fmt_with_prev(record, prev_record, key, label, suffix)
        else:
            val = _safe_fmt(record.get(key), suffix)
            text = f"{label}{val}" if val else None
        if text:
            parts.append(text)
            valid = True

    growth_fields = [
        ("主营业务收入增长率(%)", "营收增长率", "%"),
        ("净利润增长率(%)", "净利润增长率", "%"),
        ("净资产增长率(%)", "净资产增长率", "%"),
        ("总资产增长率(%)", "总资产增长率", "%"),
    ]
    for key, label, suffix in growth_fields:
        if key in _YOY_KEYS:
            text = _fmt_with_prev(record, prev_record, key, label, suffix)
        else:
            val = _safe_fmt(record.get(key), suffix)
            text = f"{label}{val}" if val else None
        if text:
            parts.append(text)
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


def _structure_to_text(code: str, name: str, date: str, record: dict,
                       period_label: str = "", prev_record: dict = None) -> str:
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
        if key in _YOY_KEYS:
            text = _fmt_with_prev(record, prev_record, key, label, suffix)
        else:
            val = _safe_fmt(record.get(key), suffix)
            text = f"{label}{val}" if val else None
        if text:
            parts.append(text)
            valid = True

    # 从股东权益比率直接计算权益乘数，避免模型心算出错
    equity_ratio = record.get("股东权益比率(%)")
    if equity_ratio is not None:
        try:
            er = float(equity_ratio)
            if not pd.isna(er) and er > 0:
                equity_multiplier = 100 / er
                parts.append(f"权益乘数{equity_multiplier:.2f}倍")
                valid = True
        except (ValueError, TypeError):
            pass

    efficiency_fields = [
        ("总资产周转率(次)", "总资产周转率", "次"),
        ("存货周转率(次)", "存货周转率", "次"),
        ("存货周转天数(天)", "存货周转天数", "天"),
        ("应收账款周转率(次)", "应收账款周转率", "次"),
        ("应收账款周转天数(天)", "应收账款周转天数", "天"),
    ]
    for key, label, suffix in efficiency_fields:
        if key in _YOY_KEYS:
            text = _fmt_with_prev(record, prev_record, key, label, suffix)
        else:
            val = _safe_fmt(record.get(key), suffix)
            text = f"{label}{val}" if val else None
        if text:
            parts.append(text)
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


def _dupont_to_text(code: str, name: str, date: str, record: dict,
                    period_label: str = "", prev_record: dict = None) -> str:
    """杜邦分析专属chunk：净利率 + 总资产周转率 + 权益乘数 + ROE，一次检索拿全三因子"""
    # 提取三因子 + ROE
    net_margin = record.get("销售净利率(%)")
    turnover = record.get("总资产周转率(次)")
    equity_ratio = record.get("股东权益比率(%)")
    roe = record.get("净资产收益率(%)")

    factors = {}
    try:
        if net_margin is not None and not pd.isna(float(net_margin)):
            factors["net_margin"] = float(net_margin)
        if turnover is not None and not pd.isna(float(turnover)):
            factors["turnover"] = float(turnover)
        if equity_ratio is not None and not pd.isna(float(equity_ratio)) and float(equity_ratio) > 0:
            factors["equity_multiplier"] = 100 / float(equity_ratio)
        if roe is not None and not pd.isna(float(roe)):
            factors["roe"] = float(roe)
    except (ValueError, TypeError):
        pass

    # 三因子必须全部有才生成杜邦chunk（ROE是附加信息，可选）
    if not all(k in factors for k in ("net_margin", "turnover", "equity_multiplier")):
        return None

    header = f"{name}({code}) {date}"
    if period_label:
        header += f"({period_label})"
    parts = [f"{header}杜邦分析"]

    if "roe" in factors:
        text = f"净资产收益率(ROE){factors['roe']:.2f}%"
        if prev_record:
            prev_val = _safe_fmt(prev_record.get("净资产收益率(%)"), "%")
            if prev_val:
                text += f"（上年同期{prev_val}）"
        parts.append(text)

    if "net_margin" in factors:
        text = f"销售净利率{factors['net_margin']:.2f}%"
        if prev_record:
            prev_val = _safe_fmt(prev_record.get("销售净利率(%)"), "%")
            if prev_val:
                text += f"（上年同期{prev_val}）"
        parts.append(text)

    if "turnover" in factors:
        text = f"总资产周转率{factors['turnover']:.2f}次"
        if prev_record:
            prev_val = _safe_fmt(prev_record.get("总资产周转率(次)"), "次")
            if prev_val:
                text += f"（上年同期{prev_val}）"
        parts.append(text)

    if "equity_multiplier" in factors:
        text = f"权益乘数{factors['equity_multiplier']:.2f}倍"
        if prev_record and prev_record.get("股东权益比率(%)"):
            try:
                prev_er = float(prev_record["股东权益比率(%)"])
                if not pd.isna(prev_er) and prev_er > 0:
                    text += f"（上年同期{100/prev_er:.2f}倍）"
            except (ValueError, TypeError):
                pass
        parts.append(text)

    # 验算：净利率 × 周转率 × 权益乘数 + 自洽性校验
    if all(k in factors for k in ("net_margin", "turnover", "equity_multiplier")):
        calc_roe = (factors["net_margin"] / 100) * factors["turnover"] * factors["equity_multiplier"] * 100

        if "roe" in factors:
            abs_diff = abs(calc_roe - factors["roe"])
            # 相对偏差：ROE 接近 0 时用绝对阈值保护
            rel_diff = abs_diff / max(abs(factors["roe"]), 1.0)

            if abs_diff > 3.0 or rel_diff > 0.3:
                # 🔴 差距过大：数据口径不一致，不生成杜邦chunk
                return None
            elif abs_diff > 1.0:
                # 🟡 有差距：生成但附加说明
                parts.append(f"杜邦验算ROE≈{calc_roe:.2f}%（净利率×周转率×权益乘数），"
                             f"与报告ROE（{factors['roe']:.2f}%）存在{abs_diff:.2f}个百分点差异，"
                             f"可能源于加权平均口径差异")
            else:
                # 🟢 差距小：正常生成
                parts.append(f"杜邦验算ROE≈{calc_roe:.2f}%（净利率×周转率×权益乘数）")
        else:
            parts.append(f"杜邦验算ROE≈{calc_roe:.2f}%（净利率×周转率×权益乘数）")

    return "，".join(parts) + "。"


# ============ 行业汇总 Chunk 构建 ============

# 东财细分行业 → 大类映射（130 → ~25 大类）
INDUSTRY_MAP = {
    "银行": ["银行", "银行Ⅱ"],
    "保险": ["保险", "保险Ⅱ"],
    "证券": ["证券", "证券Ⅱ", "券商信托", "多元金融"],
    "白酒": ["白酒Ⅱ", "酿酒行业", "非白酒"],
    "食品饮料": ["食品饮料", "饮料乳品", "食品加工", "调味发酵品Ⅱ"],
    "医药": ["化学制药", "中药Ⅱ", "中药", "生物制品", "医药制造", "医药商业", "医疗行业"],
    "医疗": ["医疗器械", "医疗服务", "医疗美容"],
    "汽车": ["乘用车", "商用车", "汽车整车", "汽车行业", "汽车零部件"],
    "半导体": ["半导体"],
    "消费电子": ["消费电子", "元件", "光学光电子", "电子元件", "电子信息"],
    "光伏": ["光伏设备"],
    "电池": ["电池", "能源金属"],
    "电力": ["电力", "电力行业", "电网设备"],
    "煤炭": ["煤炭开采", "煤炭行业", "煤炭采选"],
    "钢铁": ["普钢", "特钢Ⅱ", "钢铁行业"],
    "有色金属": ["工业金属", "小金属", "贵金属", "有色金属", "金属制品"],
    "化工": ["化学制品", "化学原料", "化工行业", "化纤行业", "化肥行业", "农化制品"],
    "房地产": ["房地产", "房地产开发"],
    "建筑建材": ["基础建设", "工程建设", "房屋建设Ⅱ", "专业工程", "水泥", "水泥建材",
                "装修建材", "玻璃玻纤", "玻璃陶瓷"],
    "家电": ["白色家电", "小家电", "家电行业", "家电零部件Ⅱ"],
    "软件": ["软件开发", "软件服务", "IT服务Ⅱ", "计算机设备"],
    "通信": ["通信设备", "通信服务", "通讯行业", "电信运营"],
    "传媒": ["数字媒体", "广告营销", "影视院线", "文化传媒"],
    "交通运输": ["航空机场", "民航机场", "航运港口", "铁路公路", "物流", "物流行业", "交运物流"],
    "军工": ["航天装备Ⅱ", "航空装备Ⅱ", "航海装备Ⅱ", "军工电子Ⅱ", "船舶制造",
            "航天航空", "交运设备"],
    "机械": ["工程机械", "自动化设备", "机械行业", "轨交设备Ⅱ", "仪器仪表"],
    "农业": ["养殖业", "农产品加工", "农牧饲渔", "饲料"],
    "石油石化": ["油服工程", "油气开采Ⅱ", "炼化及贸易", "石油行业", "燃气", "燃气Ⅱ"],
    "纺织服装": ["纺织制造", "纺织服装"],
    "零售": ["一般零售", "旅游零售Ⅱ", "家居用品"],
    "安防": ["安防设备"],
}

# 反向映射：细分行业 → 大类
_INDUSTRY_REVERSE = {}
for major, subs in INDUSTRY_MAP.items():
    for sub in subs:
        _INDUSTRY_REVERSE[sub] = major


def _get_major_industry(sub_industry: str) -> str:
    """从东财细分行业获取大类行业名"""
    return _INDUSTRY_REVERSE.get(sub_industry, sub_industry)


def build_industry_chunks(financial_path: str, report_path: str = None) -> list[dict]:
    """
    构建行业汇总 chunks：按行业聚合多家公司的关键指标对比表

    数据源：从 all_financial.json 中按行业分组聚合
    行业标签来源：all_reports.csv 中的行业字段（东财分类）

    每个行业生成 1 个 chunk，包含该行业内所有公司的核心指标对比。
    """
    # 1. 从研报数据获取 stock_code → industry 映射
    stock_industry = {}
    if report_path and os.path.exists(report_path):
        df = pd.read_csv(report_path, dtype={'股票代码': str})
        for _, row in df.iterrows():
            code = str(row.get('股票代码', '')).strip()
            industry = str(row.get('行业', '')).strip()
            if code and industry and industry != 'nan':
                stock_industry[code] = _get_major_industry(industry)

    if not stock_industry:
        logger.warning("无法获取行业映射（研报数据不存在或为空），跳过行业汇总chunks")
        return []

    # 2. 加载财务数据，按行业分组聚合
    with open(financial_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    # 按行业聚合最新年报数据
    industry_data = {}  # {industry: [{name, code, roe, net_margin, ...}, ...]}

    for code, data in all_data.items():
        if code not in stock_industry:
            continue
        industry = stock_industry[code]
        name = _safe_str(data.get("stock_name", code)) or code

        if "financial_indicators" not in data:
            continue

        # 取最新年报
        annual_records = [
            r for r in data["financial_indicators"]
            if str(r.get("日期", "")).endswith("12-31")
        ]
        if not annual_records:
            continue
        latest = max(annual_records, key=lambda r: str(r.get("日期", "")))
        report_date = str(latest.get("日期", ""))

        # 提取关键指标
        entry = {"name": name, "code": code, "report_date": report_date}
        for key, label in [
            ("净资产收益率(%)", "ROE"),
            ("销售净利率(%)", "净利率"),
            ("总资产周转率(次)", "周转率"),
            ("资产负债率(%)", "资产负债率"),
            ("主营业务收入增长率(%)", "营收增长率"),
            ("净利润增长率(%)", "净利润增长率"),
            ("主营业务利润率(%)", "毛利率"),
        ]:
            val = latest.get(key)
            if val is not None:
                try:
                    v = float(val)
                    if not pd.isna(v):
                        entry[label] = v
                except (ValueError, TypeError):
                    pass

        if len(entry) > 3:  # 至少有一个指标
            if industry not in industry_data:
                industry_data[industry] = []
            industry_data[industry].append(entry)

    # 3. 为每个行业生成汇总 chunk
    chunks = []
    for industry, companies in sorted(industry_data.items()):
        if len(companies) < 2:  # 至少2家公司才有对比价值
            continue

        # 按 ROE 降序排列
        companies.sort(key=lambda x: x.get("ROE", 0), reverse=True)

        # 标注数据年份范围
        years = set(c["report_date"][:4] for c in companies if c.get("report_date"))
        year_label = "/".join(sorted(years)) + "年报" if years else "最新年报"

        parts = [f"{industry}行业对比（{len(companies)}家公司，{year_label}数据）"]

        for comp in companies:
            line_parts = [f"{comp['name']}({comp['code']})"]
            for label, suffix in [
                ("ROE", "%"), ("净利率", "%"), ("毛利率", "%"),
                ("周转率", "次"), ("资产负债率", "%"),
                ("营收增长率", "%"), ("净利润增长率", "%"),
            ]:
                if label in comp:
                    line_parts.append(f"{label}{comp[label]:.2f}{suffix}")
            parts.append("，".join(line_parts))

        # 行业均值
        avg_parts = ["行业均值"]
        for label, suffix in [
            ("ROE", "%"), ("净利率", "%"), ("毛利率", "%"),
            ("周转率", "次"), ("资产负债率", "%"),
            ("营收增长率", "%"), ("净利润增长率", "%"),
        ]:
            vals = [c[label] for c in companies if label in c]
            if vals:
                avg_parts.append(f"{label}{sum(vals)/len(vals):.2f}{suffix}")
        parts.append("，".join(avg_parts))

        parts.append(f"（数据来源：沪深300成分股中{industry}行业代表性公司）")
        text = "。\n".join(parts) + "。"

        chunks.append({
            "text": text,
            "metadata": {
                "source_type": "industry",
                "industry": industry,
                "company_count": len(companies),
                "stock_codes": [c["code"] for c in companies],
            }
        })

    logger.info(f"行业汇总chunks: {len(chunks)} 条（覆盖 {len(industry_data)} 个行业）")
    return chunks


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="FinAgent Chunk 构建")
    parser.add_argument("--parser", type=str, default="mineru_cleaned",
                        choices=["marker", "mineru", "mineru_cleaned"],
                        help="选择PDF解析器的结果(默认 mineru_cleaned = 03d 清洗后的 Block-native 路径,"
                             "MinerU VLM 3:0 胜 Marker 见 docs/Finagent项目RAG数据库部分.md §2;"
                             "marker 路径保留作 fallback)")
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
        if args.parser == "mineru_cleaned":
            cleaned_dir = PARSER_RESULT_FILES["mineru_cleaned"]
            fulltext_chunks, table_parents = build_fulltext_chunks_mineru(cleaned_dir, pdf_map_path)
            # 单独持久化 table parents(不进召回索引,仅供 06_tabularize_fulltext.py 消费)
            if table_parents:
                parents_path = os.path.join(CHUNK_DIR, "table_parents.jsonl")
                with open(parents_path, 'w', encoding='utf-8') as f:
                    for p in table_parents:
                        f.write(json.dumps(p, ensure_ascii=False) + '\n')
                logger.info(f"table_parents: {len(table_parents)} 条 → {parents_path}")
        else:
            fulltext_chunks = build_fulltext_chunks(args.parser, pdf_map_path)
        all_chunks.extend(fulltext_chunks)

    # 3. 财务数据chunks
    financial_path = os.path.join(RAW_DIR, "financial", "all_financial.json")
    if not args.report_only and os.path.exists(financial_path):
        financial_chunks = build_financial_chunks(financial_path)
        all_chunks.extend(financial_chunks)
    elif not args.report_only:
        logger.warning(f"财务数据不存在: {financial_path}，跳过")

    # 4. 行业汇总chunks（需要研报的行业标签 + 财务数据）
    if not args.report_only and os.path.exists(financial_path):
        industry_chunks = build_industry_chunks(financial_path, report_path)
        all_chunks.extend(industry_chunks)

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
            "unique_stocks": len(set(c["metadata"].get("stock_code", "") for c in all_chunks if c["metadata"].get("stock_code"))),
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
