"""
FinAgent Step 5 (V3 · Milvus Lite 版): 建 Milvus 索引

目标架构(对照 docs/Finagent项目介绍.md 第 4 节):
  单 collection + 5 种 source_type + scalar filter 路由 + Parent 冗余在 Child metadata

输入:
  data/processed/all_chunks.jsonl                     (04 产出:report/financial/industry + prose Child)
  data/processed/tabular_chunks_mineru.jsonl          (06 --source mineru_cleaned 产出:table Child)
  data/processed/industry_alias_entities.jsonl        (Phase 3 产出:255 别名,可选)

输出:
  data/processed/milvus_finagent.db                   (Milvus Lite 单文件,含 dense + learned sparse)

  注:旧版 data/processed/bm25_ef.pkl 已废弃(bge-m3 自带 learned sparse 替代 jieba/BM25)。

Schema / index 策略详见 docs/Chunk_Alignment_20260420.md 14 节。

用法:
    python 05_build_index.py
    python 05_build_index.py --hnsw          # Standalone 部署切 HNSW(Lite 只能用 FLAT)
    python 05_build_index.py --limit 1000    # 原型:只插 1000 条
    python 05_build_index.py --uri http://localhost:19530   # 连 Standalone
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def _find_project_root() -> Path:
    """兼容 Mac(backup/finagent_repo/ 两层嵌套)和服务器(Finagent/ 一层)两种结构"""
    here = Path(__file__).resolve()
    for cand in (here.parent, here.parent.parent):
        if (cand / "data").is_dir():
            return cand
    return here.parent

ROOT = _find_project_root()

# ============ 配置 ============
COLLECTION = "finagent"
DB_PATH = ROOT / "data/processed/milvus_finagent.db"
EMBEDDING_MODEL = "./models/bge-m3"
DIM = 1024

CHUNK_SOURCES = [
    ROOT / "data/processed/all_chunks.jsonl",
    ROOT / "data/processed/tabular_chunks_mineru.jsonl",
    ROOT / "data/processed/industry_alias_entities.jsonl",
]


# ============ 加载 chunks ============

def load_all_chunks() -> list[dict]:
    """从所有源加载 chunks"""
    all_chunks = []
    for p in CHUNK_SOURCES:
        if not p.exists():
            logger.warning(f"⚠️  {p.name} 不存在,跳过")
            continue
        n = 0
        with open(p, encoding='utf-8') as f:
            for line in f:
                all_chunks.append(json.loads(line))
                n += 1
        logger.info(f"  + {p.name}: {n} 条")
    return all_chunks


# ============ Schema ============

def build_schema(client):
    """Schema 见 docs/Chunk_Alignment_20260420.md 14.2 节"""
    from pymilvus import DataType
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

    # 主键 + 双向量
    schema.add_field("chunk_id",      DataType.VARCHAR, is_primary=True, max_length=256)
    schema.add_field("dense",         DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("sparse",        DataType.SPARSE_FLOAT_VECTOR)

    # 内容
    schema.add_field("text",          DataType.VARCHAR, max_length=4096)

    # Scalar (filter/group_by 用,各 source_type 共享)
    # milvus-lite 不支持 nullable(PR #332 至今未合并),全部用 sentinel:VARCHAR="",INT16=-1
    schema.add_field("source_type",   DataType.VARCHAR, max_length=32)
    schema.add_field("chunk_method",  DataType.VARCHAR, max_length=32)
    schema.add_field("stock_code",    DataType.VARCHAR, max_length=16)
    schema.add_field("stock_name",    DataType.VARCHAR, max_length=64)
    schema.add_field("institution",   DataType.VARCHAR, max_length=128)
    schema.add_field("date",          DataType.VARCHAR, max_length=24)
    schema.add_field("industry",      DataType.VARCHAR, max_length=64)
    schema.add_field("rating",        DataType.VARCHAR, max_length=32)
    schema.add_field("report_title",  DataType.VARCHAR, max_length=256)
    schema.add_field("page_idx",      DataType.INT16)
    # 原始 PDF 文件名(_external_rrf 按 pdf_file 聚合用,同 PDF 多 chunk 共享 RRF 分数)
    schema.add_field("pdf_file",      DataType.VARCHAR, max_length=256)
    # 解析器 / 数据类型 / 表格类型 等语义标签(之前被 enable_dynamic_field=False silent drop)
    schema.add_field("parser",          DataType.VARCHAR, max_length=32)
    schema.add_field("data_type",       DataType.VARCHAR, max_length=32)
    schema.add_field("chunk_index",     DataType.INT16)
    schema.add_field("table_type",      DataType.VARCHAR, max_length=64)
    schema.add_field("header_type",     DataType.VARCHAR, max_length=32)
    schema.add_field("current_section", DataType.VARCHAR, max_length=256)

    # Parent-Child 关联 + 冗余(Table;建库时 HTML→Markdown,密度 ×3-5,8192 够用)
    schema.add_field("parent_id",      DataType.VARCHAR, max_length=256)
    schema.add_field("parent_md",      DataType.VARCHAR, max_length=8192)
    schema.add_field("table_caption",  DataType.VARCHAR, max_length=512)
    schema.add_field("table_footnote", DataType.VARCHAR, max_length=512)

    # Parent-Child 关联 + 冗余(Prose)
    schema.add_field("section_id",    DataType.VARCHAR, max_length=256)
    schema.add_field("section_title", DataType.VARCHAR, max_length=256)
    schema.add_field("section_text",  DataType.VARCHAR, max_length=8192)

    # Industry(V3 alias + industry_comparison 聚合)
    schema.add_field("stock_codes",   DataType.VARCHAR, max_length=2048)
    schema.add_field("company_count", DataType.INT16)

    return schema


def build_index_params(client, use_hnsw: bool = False):
    """
    Lite 默认用 FLAT(精确,~20ms / 335K 单查)
    Standalone 传 --hnsw 切 HNSW(~5ms,98%+ 召回)
    """
    params = client.prepare_index_params()

    if use_hnsw:
        params.add_index(field_name="dense", index_type="HNSW", metric_type="COSINE",
                         params={"M": 16, "efConstruction": 200})
        logger.info("  dense 用 HNSW(Standalone)")
    else:
        params.add_index(field_name="dense", index_type="FLAT", metric_type="COSINE")
        logger.info("  dense 用 FLAT(Lite)")

    params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    # Scalar 索引:Lite 支持 INVERTED,加速 filter / group_by
    for scalar in ("source_type", "stock_code", "parent_id", "section_id", "industry"):
        params.add_index(field_name=scalar, index_type="INVERTED")

    return params


# ============ chunk → Milvus row ============

def _truncate(s, max_len):
    """None-safe 截断"""
    if s is None:
        return None
    s = str(s)
    return s[:max_len] if len(s) > max_len else s


def _varchar(s, max_len):
    """VARCHAR sentinel:None / 空串 → "",避免 milvus-lite 不支持 nullable 的限制"""
    if s is None:
        return ""
    return _truncate(s, max_len)


# INT16 sentinel:page_idx / chunk_index / company_count 缺失时用 -1
# 业务上 page_idx 最小 0、chunk_index 最小 0、company_count 最小 2,-1 不会与真实值冲突
_INT_MISSING = -1


def _csr_row_to_dict(csr, row_idx):
    """
    scipy.sparse.csr_array 的一行 → Milvus 可接受的 {col_idx: value} dict
    Milvus 要求 sparse key 升序,csr.indices 对单行通常升序(scipy 规范),但 sorted 兜底更安全
    """
    start = csr.indptr[row_idx]
    end = csr.indptr[row_idx + 1]
    pairs = sorted(
        ((int(k), float(v)) for k, v in zip(csr.indices[start:end], csr.data[start:end])),
        key=lambda kv: kv[0],
    )
    return dict(pairs)


def chunk_to_row(chunk: dict, global_idx: int,
                 dense_vec, sparse_csr, sparse_row_idx: int) -> dict:
    """把 chunk dict 摊平成 Milvus row(14.4 节字段映射)"""
    m = chunk.get('metadata', {})
    source_type = m.get('source_type', 'unknown')

    # chunk_id:优先 metadata 里的,否则用 global_idx 保证唯一
    chunk_id = m.get('chunk_id') or f"{source_type}_{global_idx:08d}"

    # stock_codes 可能是 list[str],转成逗号分隔字符串
    stock_codes = m.get('stock_codes')
    if isinstance(stock_codes, list):
        stock_codes = ','.join(stock_codes)

    # INT16 缺失用 -1 sentinel(page_idx/chunk_index 业务范围 ≥0,company_count ≥2)
    page_idx = m.get('page_idx')
    page_idx = int(page_idx) if page_idx is not None else _INT_MISSING
    company_count = m.get('company_count')
    company_count = int(company_count) if company_count is not None else _INT_MISSING
    chunk_index = m.get('chunk_index')
    chunk_index = int(chunk_index) if chunk_index is not None else _INT_MISSING

    dense_list = dense_vec.tolist() if hasattr(dense_vec, 'tolist') else list(dense_vec)

    return {
        # 必填:chunk_id / text / source_type
        "chunk_id":       _truncate(chunk_id, 256),
        "dense":          dense_list,
        "sparse":         _csr_row_to_dict(sparse_csr, sparse_row_idx),
        "text":           _truncate(chunk.get('text', ''), 4096),
        "source_type":    _truncate(source_type, 32),
        # 可选 VARCHAR:缺失用 "" sentinel(查询侧 `x or ''` / `if x:` 已是 null-safe)
        "chunk_method":   _varchar(m.get('chunk_method'),   32),
        "stock_code":     _varchar(m.get('stock_code'),     16),
        "stock_name":     _varchar(m.get('stock_name'),     64),
        "institution":    _varchar(m.get('institution'),    128),
        "date":           _varchar(m.get('date'),           24),
        "industry":       _varchar(m.get('industry'),       64),
        "rating":         _varchar(m.get('rating'),         32),
        "report_title":   _varchar(m.get('report_title'),   256),
        "page_idx":       page_idx,
        "pdf_file":       _varchar(m.get('pdf_file'),       256),
        "parser":         _varchar(m.get('parser'),          32),
        "data_type":      _varchar(m.get('data_type'),       32),
        "chunk_index":    chunk_index,
        "table_type":     _varchar(m.get('table_type'),      64),
        "header_type":    _varchar(m.get('header_type'),     32),
        "current_section":_varchar(m.get('current_section'), 256),
        "parent_id":      _varchar(m.get('parent_id'),      256),
        "parent_md":      _varchar(m.get('parent_md'),        8192),
        "table_caption":  _varchar(m.get('table_caption') or m.get('caption'), 512),
        "table_footnote": _varchar(m.get('table_footnote'),   512),
        "section_id":     _varchar(m.get('section_id'),     256),
        "section_title":  _varchar(m.get('section_title'),  256),
        "section_text":   _varchar(m.get('section_text'),   8192),
        "stock_codes":    _varchar(stock_codes,             2048),
        "company_count":  company_count,
    }


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="建 Milvus Lite 索引")
    parser.add_argument('--hnsw', action='store_true',
                        help='用 HNSW dense 索引(Standalone),默认 FLAT(Lite)')
    parser.add_argument('--limit', type=int, default=0,
                        help='限制 chunks 数量(用于原型)')
    parser.add_argument('--batch', type=int, default=500,
                        help='每批 insert 大小')
    parser.add_argument('--uri', type=str, default=None,
                        help='覆盖 URI(连 Standalone 用 http://localhost:19530)')
    args = parser.parse_args()

    # ─── 1. 加载 chunks ───
    logger.info("加载 chunks:")
    chunks = load_all_chunks()
    if not chunks:
        logger.error("❌ 没有 chunks 可以导入,退出")
        return
    logger.info(f"总 chunks: {len(chunks)}")
    if args.limit > 0:
        chunks = chunks[:args.limit]
        logger.info(f"原型模式:限流到 {len(chunks)}")

    dist = Counter(c.get('metadata', {}).get('source_type', '?') for c in chunks)
    logger.info(f"source_type 分布: {dict(dist)}")

    # 与 Milvus text 字段(VARCHAR 4096)对齐,避免 encode 看到的内容比存储的短
    texts = [(c.get('text') or '')[:4096] for c in chunks]

    # ─── 2. bge-m3 一次输出 dense + native sparse(替代 BGE + jieba/BM25 两路) ───
    logger.info(f"加载 bge-m3: {EMBEDDING_MODEL}")
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction
    m3_ef = BGEM3EmbeddingFunction(
        model_name=EMBEDDING_MODEL, device="cuda:0", use_fp16=True,
    )

    logger.info(f"encode {len(texts)} 条 dense+sparse(bge-m3)...")
    docs_emb = m3_ef.encode_documents(texts)

    # 兼容:dense 可能是 list[np.ndarray] 或 np.ndarray(pymilvus 版本差异)
    import numpy as np
    dense_raw = docs_emb["dense"]
    dense_vecs = np.array(dense_raw) if isinstance(dense_raw, list) else dense_raw

    # 兼容:sparse 可能是 list[dict] 或 scipy csr/coo,统一转 csr_matrix
    sparse_raw = docs_emb["sparse"]
    if isinstance(sparse_raw, list) and sparse_raw and isinstance(sparse_raw[0], dict):
        from scipy.sparse import csr_matrix
        rows_i, cols_i, data_v = [], [], []
        max_col = 0
        for i_row, d in enumerate(sparse_raw):
            for k, v in d.items():
                rows_i.append(i_row)
                cols_i.append(int(k))
                data_v.append(float(v))
                if int(k) > max_col:
                    max_col = int(k)
        sparse_csr = csr_matrix((data_v, (rows_i, cols_i)),
                                 shape=(len(sparse_raw), max_col + 1))
    else:
        # 已经是 scipy sparse,转 csr 以便 chunk_to_row 里的 _csr_row_to_dict 使用 indptr
        sparse_csr = sparse_raw.tocsr() if hasattr(sparse_raw, 'tocsr') else sparse_raw

    logger.info(f"encode 完成 dense={dense_vecs.shape} sparse_nnz={sparse_csr.nnz}")

    # ─── 4. 建 Milvus collection ───
    from pymilvus import MilvusClient
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    uri = args.uri or str(DB_PATH)
    logger.info(f"连接 Milvus: {uri}")
    client = MilvusClient(uri=uri)

    if client.has_collection(COLLECTION):
        logger.info(f"⚠️  collection {COLLECTION!r} 已存在,drop 重建")
        client.drop_collection(COLLECTION)

    schema = build_schema(client)
    index_params = build_index_params(client, use_hnsw=args.hnsw)
    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        index_params=index_params,
    )
    logger.info(f"✅ collection {COLLECTION!r} 创建完成")

    # ─── 5. 批量 insert ───
    logger.info(f"批量 insert(batch={args.batch})...")
    buffer = []
    n_inserted = 0
    for i, c in enumerate(tqdm(chunks, desc="insert")):
        row = chunk_to_row(c, i, dense_vecs[i], sparse_csr, i)
        buffer.append(row)
        if len(buffer) >= args.batch:
            client.insert(collection_name=COLLECTION, data=buffer)
            n_inserted += len(buffer)
            buffer = []
    if buffer:
        client.insert(collection_name=COLLECTION, data=buffer)
        n_inserted += len(buffer)

    logger.info(f"✅ insert 完成:{n_inserted} 条")

    # ─── 6. flush + sanity check ───
    try:
        client.flush(collection_name=COLLECTION)
    except Exception:
        pass   # Lite 可能没有 flush API

    try:
        sample = client.query(
            collection_name=COLLECTION,
            filter='source_type != ""',
            output_fields=["source_type"],
            limit=5,
        )
        logger.info(f"sanity check: 命中 {len(sample)} 条 source_type 样本")
    except Exception as e:
        logger.warning(f"sanity check 跳过: {e}")

    logger.info("=" * 60)
    logger.info("✅ 建索引完成")
    logger.info(f"   DB 路径:     {DB_PATH}")
    logger.info(f"   Embedding:   bge-m3 (1024 维, dense+learned sparse)")
    logger.info(f"   索引 backend: {'HNSW(Standalone)' if args.hnsw else 'FLAT(Lite)'}")
    logger.info("下一步: 跑 hybrid_search.py 做召回验证")


if __name__ == '__main__':
    main()
