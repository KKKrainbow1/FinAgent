"""
FinAgent Step 5 (V3.5 · 多 Collection 版): 建 5 个独立 Milvus collection

V3 单 collection → V3.5 多 collection 演进(2026-05-03):
  - 5 工具(search_report_meta / search_report_content / search_financial /
            search_industry / calculate)对应 4 个独立 collection(meta + section + tabular + financial + industry)
  - 每 collection schema 精简(只放该类用得着的字段),索引体积锐减
  - 工具语义 ↔ collection 语义 1:1 对应,运维 / 调试 / 重建独立
  - prose 切分改 section 整段(删 fixed_window),table_row_fact 仍 filter

输入:
  data/processed/all_chunks.jsonl              (04 产出:report meta / report_fulltext section / financial / industry)
  data/processed/tabular_chunks_mineru.jsonl   (06 产出:table_narrative + table_row_fact;后者 filter)
  data/processed/industry_alias_entities.jsonl (Phase 3 产出:255 别名 entity,可选)

输出(5 个独立 .db):
  data/processed/milvus_report_meta.db        → collection 'report_meta'      (~10K 条)
  data/processed/milvus_report_section.db     → collection 'report_section'   (~7K 条,section 整段 + chunk head)
  data/processed/milvus_report_tabular.db     → collection 'report_tabular'   (~10K 条,table_narrative)
  data/processed/milvus_financial.db          → collection 'financial'        (~6K 条)
  data/processed/milvus_industry.db           → collection 'industry_alias'   (~285 条)

总:~33K chunks(单 collection 版 ~42K),索引体积 2.3 GB → ~300 MB。

用法:
    python 05_build_index.py
    python 05_build_index.py --hnsw          # Standalone 部署切 HNSW(Lite 只能用 FLAT)
    python 05_build_index.py --limit 1000    # 原型:每 collection 限流 1000 条
    python 05_build_index.py --only report_section  # 只重建指定 collection
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for cand in (here.parent, here.parent.parent):
        if (cand / "data").is_dir():
            return cand
    return here.parent


ROOT = _find_project_root()
DIM = 1024
EMBEDDING_MODEL = "./models/bge-m3"

DATA_DIR = ROOT / "data/processed"
ALL_CHUNKS = DATA_DIR / "all_chunks.jsonl"
TAB_CHUNKS = DATA_DIR / "tabular_chunks_mineru.jsonl"
INDUSTRY_ALIAS_FILE = DATA_DIR / "industry_alias_entities.jsonl"

DB_PATHS = {
    "report_meta":     DATA_DIR / "milvus_report_meta.db",
    "report_section":  DATA_DIR / "milvus_report_section.db",
    "report_tabular":  DATA_DIR / "milvus_report_tabular.db",
    "financial":       DATA_DIR / "milvus_financial.db",
    "industry_alias":  DATA_DIR / "milvus_industry.db",
}
COLL_NAMES = {k: k for k in DB_PATHS}


# ============ 数据收集(从 jsonl → 5 个分桶 list) ============

def _make_section_head(m: dict) -> str:
    """prose section chunk head,V3.5 BGE-m3 dense 锚定:
        [公司名(代码) · 日期 · 报告标题 · 章节:章节标题]
    """
    name = (m.get("stock_name") or "").strip()
    code = (m.get("stock_code") or "").strip()
    date = ((m.get("date") or "")[:10]).strip()
    rtitle = (m.get("report_title") or "").strip()
    stitle = (m.get("section_title") or "").strip()
    parts = []
    if name and code:
        parts.append(f"{name}({code})")
    elif name:
        parts.append(name)
    if date:
        parts.append(date)
    if rtitle:
        parts.append(rtitle[:80])
    if stitle:
        parts.append(f"章节:{stitle[:60]}")
    return "[" + " · ".join(parts) + "]" if parts else ""


def collect_buckets() -> dict:
    """从 3 个 jsonl 源读取,按 source_type / chunk_method 分到 5 个桶。
    table_row_fact filter 掉(05 历史决策,索引瘦身 80%,实证 P@5 持平)。
    section dedup by section_id(同一 section 多个 fixed_window child → 合并为 1 个 section chunk)。
    """
    buckets = {k: [] for k in DB_PATHS.keys()}
    seen_sections: set[str] = set()

    # all_chunks.jsonl: report meta + financial + industry + report_fulltext (section 源)
    if ALL_CHUNKS.exists():
        with open(ALL_CHUNKS, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                m = d.get("metadata", {})
                st = m.get("source_type")

                if st == "report":
                    buckets["report_meta"].append({
                        "chunk_id":     m.get("chunk_id") or "",
                        "text":         (d.get("text") or "")[:1000],
                        "stock_code":   m.get("stock_code") or "",
                        "stock_name":   m.get("stock_name") or "",
                        "institution":  m.get("institution") or "",
                        "date":         m.get("date") or "",
                        "rating":       m.get("rating") or "",
                        "report_title": (m.get("report_title") or "")[:512],
                        "pdf_file":     m.get("pdf_file") or "",
                    })
                elif st == "report_fulltext":
                    sid = m.get("section_id")
                    sec_text = m.get("section_text") or ""
                    if not sid or not sec_text or sid in seen_sections:
                        continue
                    seen_sections.add(sid)
                    head = _make_section_head(m)
                    full_text = f"{head} {sec_text}" if head else sec_text
                    buckets["report_section"].append({
                        "chunk_id":     sid,
                        "text":         full_text[:8000],
                        "stock_code":   m.get("stock_code") or "",
                        "stock_name":   m.get("stock_name") or "",
                        "industry":     m.get("industry") or "",
                        "date":         m.get("date") or "",
                        "report_title": (m.get("report_title") or "")[:512],
                        "section_id":   sid,
                        "section_title":(m.get("section_title") or "")[:512],
                        "page_idx":     int(m.get("page_idx") or -1),
                        "pdf_file":     m.get("pdf_file") or "",
                    })
                elif st == "financial":
                    buckets["financial"].append({
                        "chunk_id":   m.get("chunk_id") or "",
                        "text":       (d.get("text") or "")[:2000],
                        "stock_code": m.get("stock_code") or "",
                        "stock_name": m.get("stock_name") or "",
                        "date":       m.get("date") or "",
                        "data_type":  m.get("data_type") or "",
                        "metric_class": m.get("metric_class") or m.get("data_type") or "",
                    })
                elif st == "industry":
                    # industry chunk 也走 industry_alias collection(每行业一条聚合 chunk)
                    # industry_alias_entities.jsonl 是 alias entity(255 条,每个 alias 一行),
                    # 我们也灌进去,保持向量索引检索能力
                    pass  # industry comparison chunk 由 industry_alias_entities 文件主导

    # tabular_chunks_mineru.jsonl: table_narrative(row_fact filter)
    if TAB_CHUNKS.exists():
        with open(TAB_CHUNKS, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                m = d.get("metadata", {})
                if m.get("source_type") != "report_tabular":
                    continue
                if m.get("chunk_method") != "table_narrative":
                    continue   # row_fact 不灌库
                buckets["report_tabular"].append({
                    "chunk_id":     m.get("parent_id") or "",  # narrative 1 个 / 表 → parent_id 当 chunk_id
                    "text":         (d.get("text") or "")[:4000],
                    "stock_code":   m.get("stock_code") or "",
                    "stock_name":   m.get("stock_name") or "",
                    "industry":     m.get("industry") or "",
                    "date":         m.get("date") or "",
                    "report_title": (m.get("report_title") or "")[:512],
                    "chunk_method": "table_narrative",
                    "parent_id":    m.get("parent_id") or "",
                    "parent_md":    (m.get("parent_md") or m.get("table_md") or "")[:8000],
                    "table_caption":(m.get("caption") or m.get("table_caption") or "")[:512],
                    "table_footnote":(m.get("footnote") or m.get("table_footnote") or "")[:1024],
                    "page_idx":     int(m.get("page_idx") or -1),
                    "pdf_file":     m.get("pdf_file") or "",
                })

    # industry_alias_entities.jsonl: 255 alias entity
    if INDUSTRY_ALIAS_FILE.exists():
        with open(INDUSTRY_ALIAS_FILE, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                m = d.get("metadata", {})
                stock_codes = m.get("stock_codes")
                if isinstance(stock_codes, list):
                    stock_codes = ",".join(stock_codes)
                buckets["industry_alias"].append({
                    "chunk_id":     m.get("chunk_id") or "",
                    "text":         (d.get("text") or "")[:512],
                    "industry":     m.get("industry") or "",
                    "stock_codes":  (stock_codes or "")[:1024],
                    "company_count":int(m.get("company_count") or 0),
                    "section_text": (m.get("section_text") or "")[:4000],
                    "pdf_file":     m.get("pdf_file") or "",
                })

    for k, v in buckets.items():
        logger.info(f"  bucket {k}: {len(v)} 条")
    return buckets


# ============ 5 个 collection schema ============

def _build_schema_meta(client):
    from pymilvus import DataType
    s = client.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field("chunk_id",     DataType.VARCHAR, max_length=192, is_primary=True)
    s.add_field("dense",        DataType.FLOAT_VECTOR, dim=DIM)
    s.add_field("sparse",       DataType.SPARSE_FLOAT_VECTOR)
    s.add_field("text",         DataType.VARCHAR, max_length=1024)
    s.add_field("stock_code",   DataType.VARCHAR, max_length=16)
    s.add_field("stock_name",   DataType.VARCHAR, max_length=64)
    s.add_field("institution",  DataType.VARCHAR, max_length=64)
    s.add_field("date",         DataType.VARCHAR, max_length=32)
    s.add_field("rating",       DataType.VARCHAR, max_length=32)
    s.add_field("report_title", DataType.VARCHAR, max_length=512)
    s.add_field("pdf_file",     DataType.VARCHAR, max_length=128)
    return s


def _build_schema_section(client):
    from pymilvus import DataType
    s = client.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field("chunk_id",     DataType.VARCHAR, max_length=192, is_primary=True)
    s.add_field("dense",        DataType.FLOAT_VECTOR, dim=DIM)
    s.add_field("sparse",       DataType.SPARSE_FLOAT_VECTOR)
    s.add_field("text",         DataType.VARCHAR, max_length=8000)
    s.add_field("stock_code",   DataType.VARCHAR, max_length=16)
    s.add_field("stock_name",   DataType.VARCHAR, max_length=64)
    s.add_field("industry",     DataType.VARCHAR, max_length=64)
    s.add_field("date",         DataType.VARCHAR, max_length=32)
    s.add_field("report_title", DataType.VARCHAR, max_length=512)
    s.add_field("section_id",   DataType.VARCHAR, max_length=192)
    s.add_field("section_title",DataType.VARCHAR, max_length=512)
    s.add_field("page_idx",     DataType.INT32)
    s.add_field("pdf_file",     DataType.VARCHAR, max_length=128)
    return s


def _build_schema_tabular(client):
    from pymilvus import DataType
    s = client.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field("chunk_id",     DataType.VARCHAR, max_length=192, is_primary=True)
    s.add_field("dense",        DataType.FLOAT_VECTOR, dim=DIM)
    s.add_field("sparse",       DataType.SPARSE_FLOAT_VECTOR)
    s.add_field("text",         DataType.VARCHAR, max_length=4000)
    s.add_field("stock_code",   DataType.VARCHAR, max_length=16)
    s.add_field("stock_name",   DataType.VARCHAR, max_length=64)
    s.add_field("industry",     DataType.VARCHAR, max_length=64)
    s.add_field("date",         DataType.VARCHAR, max_length=32)
    s.add_field("report_title", DataType.VARCHAR, max_length=512)
    s.add_field("chunk_method", DataType.VARCHAR, max_length=32)
    s.add_field("parent_id",    DataType.VARCHAR, max_length=192)
    s.add_field("parent_md",    DataType.VARCHAR, max_length=8000)
    s.add_field("table_caption",DataType.VARCHAR, max_length=512)
    s.add_field("table_footnote",DataType.VARCHAR, max_length=1024)
    s.add_field("page_idx",     DataType.INT32)
    s.add_field("pdf_file",     DataType.VARCHAR, max_length=128)
    return s


def _build_schema_financial(client):
    from pymilvus import DataType
    s = client.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field("chunk_id",     DataType.VARCHAR, max_length=192, is_primary=True)
    s.add_field("dense",        DataType.FLOAT_VECTOR, dim=DIM)
    s.add_field("sparse",       DataType.SPARSE_FLOAT_VECTOR)
    s.add_field("text",         DataType.VARCHAR, max_length=2000)
    s.add_field("stock_code",   DataType.VARCHAR, max_length=16)
    s.add_field("stock_name",   DataType.VARCHAR, max_length=64)
    s.add_field("date",         DataType.VARCHAR, max_length=32)
    s.add_field("data_type",    DataType.VARCHAR, max_length=64)
    s.add_field("metric_class", DataType.VARCHAR, max_length=64)
    return s


def _build_schema_industry(client):
    from pymilvus import DataType
    s = client.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field("chunk_id",     DataType.VARCHAR, max_length=128, is_primary=True)
    s.add_field("dense",        DataType.FLOAT_VECTOR, dim=DIM)
    s.add_field("sparse",       DataType.SPARSE_FLOAT_VECTOR)
    s.add_field("text",         DataType.VARCHAR, max_length=512)
    s.add_field("industry",     DataType.VARCHAR, max_length=64)
    s.add_field("stock_codes",  DataType.VARCHAR, max_length=1024)
    s.add_field("company_count",DataType.INT32)
    s.add_field("section_text", DataType.VARCHAR, max_length=4000)
    s.add_field("pdf_file",     DataType.VARCHAR, max_length=128)
    return s


SCHEMA_BUILDERS = {
    "report_meta":     _build_schema_meta,
    "report_section":  _build_schema_section,
    "report_tabular":  _build_schema_tabular,
    "financial":       _build_schema_financial,
    "industry_alias":  _build_schema_industry,
}

INDEX_FIELDS = {
    "report_meta":     ["stock_code", "date"],
    "report_section":  ["stock_code", "section_id"],
    "report_tabular":  ["stock_code", "parent_id", "date"],
    "financial":       ["stock_code", "data_type"],
    "industry_alias":  ["industry"],
}


def _sparse_row_to_dict(sp_raw, j: int) -> dict:
    """从 BGE-m3 单 batch sparse(scipy csr / list[dict])里取第 j 行 → {int: float} 升序 dict。"""
    if isinstance(sp_raw, list):
        return {int(k): float(v) for k, v in sp_raw[j].items()}
    if hasattr(sp_raw, "tocsr"):
        csr = sp_raw.tocsr()
        if hasattr(csr, "indptr") and csr.shape[0] > j:
            start = int(csr.indptr[j]); end = int(csr.indptr[j + 1])
            return dict(sorted(((int(k), float(v)) for k, v in
                                zip(csr.indices[start:end], csr.data[start:end])),
                               key=lambda x: x[0]))
    if hasattr(sp_raw, "indices") and hasattr(sp_raw, "data"):
        return dict(sorted(((int(k), float(v)) for k, v in zip(sp_raw.indices, sp_raw.data)),
                           key=lambda x: x[0]))
    return {}


def build_one(coll_key: str, rows: list[dict], ef, batch: int = 32,
              use_hnsw: bool = False) -> None:
    from pymilvus import MilvusClient
    db_path = DB_PATHS[coll_key]
    coll_name = COLL_NAMES[coll_key]

    if not rows:
        logger.warning(f"[{coll_key}] 0 rows,skip")
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    client = MilvusClient(str(db_path))
    if coll_name in client.list_collections():
        client.drop_collection(coll_name)
    schema = SCHEMA_BUILDERS[coll_key](client)
    index_params = client.prepare_index_params()
    if use_hnsw:
        index_params.add_index(field_name="dense", index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 200})
    else:
        index_params.add_index(field_name="dense", index_type="FLAT", metric_type="COSINE")
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    for f in INDEX_FIELDS[coll_key]:
        index_params.add_index(field_name=f, index_type="INVERTED")
    client.create_collection(coll_name, schema=schema, index_params=index_params)
    logger.info(f"  [{coll_key}] '{coll_name}' @ {db_path} 创建,{len(rows)} 条待灌")

    inserted = 0
    t0 = time.time()
    for i in range(0, len(rows), batch):
        batch_rows = rows[i : i + batch]
        texts = [r["text"] for r in batch_rows]
        out = ef.encode_documents(texts)
        rows_to_insert = []
        for j, r in enumerate(batch_rows):
            d_vec = out["dense"][j]
            d_list = d_vec.tolist() if hasattr(d_vec, "tolist") else list(d_vec)
            sp = _sparse_row_to_dict(out["sparse"], j)
            if not sp:
                sp = {0: 1e-6}
            row = dict(r)
            row["dense"] = d_list
            row["sparse"] = sp
            rows_to_insert.append(row)
        client.insert(coll_name, rows_to_insert)
        inserted += len(rows_to_insert)
        if (inserted % (batch * 10) == 0) or inserted == len(rows):
            elapsed = time.time() - t0
            rate = inserted / max(elapsed, 1e-3)
            eta = (len(rows) - inserted) / max(rate, 1e-3)
            logger.info(f"    [{coll_key}] inserted {inserted}/{len(rows)}  ({rate:.1f}/s, ETA {eta:.0f}s)")

    client.flush(coll_name)
    stats = client.get_collection_stats(coll_name)
    logger.info(f"  ✅ [{coll_key}] done. row_count = {stats.get('row_count')}")


def main():
    parser = argparse.ArgumentParser(description="V3.5 多 collection Milvus build")
    parser.add_argument('--hnsw', action='store_true',
                        help='用 HNSW dense 索引(Standalone),默认 FLAT(Lite)')
    parser.add_argument('--limit', type=int, default=0,
                        help='每 collection 限流(原型用)')
    parser.add_argument('--batch', type=int, default=32,
                        help='encode + insert batch size')
    parser.add_argument('--only', type=str, default=None,
                        help='只 build 指定 collection (report_meta / report_section / report_tabular / financial / industry_alias)')
    parser.add_argument('--device', type=str, default="cuda:0",
                        help='BGE-m3 device,默认 cuda:0')
    args = parser.parse_args()

    logger.info(f"V3.5 多 collection build 开始,output dir={DATA_DIR}")
    buckets = collect_buckets()

    if args.limit:
        for k in buckets:
            buckets[k] = buckets[k][: args.limit]
        logger.info(f"limit={args.limit} 生效")

    from pymilvus.model.hybrid import BGEM3EmbeddingFunction
    logger.info(f"加载 bge-m3: {EMBEDDING_MODEL} (device={args.device})")
    ef = BGEM3EmbeddingFunction(model_name=EMBEDDING_MODEL, device=args.device, use_fp16=True)

    # 由小到大顺序,先验证 schema 没问题再跑大集合
    order = ["industry_alias", "financial", "report_meta", "report_tabular", "report_section"]
    for key in order:
        if args.only and args.only != key:
            continue
        build_one(key, buckets[key], ef, batch=args.batch, use_hnsw=args.hnsw)

    logger.info("✅ 全部 collection build 完成")


if __name__ == "__main__":
    main()
