"""
FinAgent 检索索引构建脚本
用途：将 all_chunks.jsonl 构建为 FAISS 向量索引 + BM25 稀疏索引
依赖：pip install sentence-transformers faiss-gpu rank_bm25 jieba

运行方式：
    python 05_build_index.py                    # 构建全部索引
    python 05_build_index.py --test_query "茅台盈利能力"  # 构建完后测试检索

面试追问：为什么用混合检索而不是纯向量检索？
答：向量检索擅长语义匹配（"盈利能力" ↔ "ROE"），
但对精确实体匹配差（"宁德时代"可能和"比亚迪"的向量距离不够远）。
BM25擅长精确关键词匹配。混合后互补。

面试追问：embedding 模型为什么选 BGE？
答：BGE-base-zh-v1.5 是中文 embedding 的 SOTA 之一，
在 C-MTEB 榜单上排名前列，对中文金融文本的语义理解好。
base 版本 768 维，推理速度快，58000 条 chunk 在 GPU 上几分钟编码完。
"""

import json
import os
import pickle
import time
import argparse
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============ 配置 ============
CHUNK_PATH = "./data/processed/all_chunks.jsonl"
INDEX_DIR = "./data/index"
EMBEDDING_MODEL = "./models/bge-base-zh-v1.5"
BATCH_SIZE = 256  # GPU 上可以用大 batch

os.makedirs(INDEX_DIR, exist_ok=True)


# ============ Step 1: 加载 chunks ============

def load_chunks(chunk_path: str) -> list[dict]:
    """加载 JSONL 格式的 chunks"""
    chunks = []
    with open(chunk_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info(f"加载 {len(chunks)} 条 chunks")
    return chunks


# ============ Step 2: 构建 FAISS 向量索引 ============

def build_faiss_index(chunks: list[dict], model_name: str = EMBEDDING_MODEL):
    """
    用 BGE 模型对所有 chunk 做 embedding，构建 FAISS 索引
    
    面试追问：为什么用 IndexFlatIP 而不是 IndexIVFFlat？
    答：58000 条数据量不大，暴力搜索（Flat）的延迟在毫秒级，
    不需要近似搜索（IVF）。IVF 适合百万级以上的数据量。
    用 IP（Inner Product）是因为 embedding 已归一化，IP = cosine similarity。
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    logger.info(f"加载 embedding 模型: {model_name}")
    encoder = SentenceTransformer(model_name)
    
    # 检查是否使用 GPU
    device = encoder.device
    logger.info(f"模型运行设备: {device}")

    texts = [c["text"] for c in chunks]
    
    logger.info(f"开始编码 {len(texts)} 条文本 (batch_size={BATCH_SIZE})")
    start = time.time()
    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # 归一化后 IP = cosine similarity
    )
    elapsed = time.time() - start
    logger.info(f"编码完成: {elapsed:.1f}秒, 向量维度: {embeddings.shape}")

    # 构建 FAISS 索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    logger.info(f"FAISS 索引构建完成: {index.ntotal} 条向量")

    # 保存
    faiss_path = os.path.join(INDEX_DIR, "faiss_index.bin")
    faiss.write_index(index, faiss_path)
    logger.info(f"FAISS 索引已保存: {faiss_path}")

    # 保存 encoder 路径供后续加载
    with open(os.path.join(INDEX_DIR, "encoder_config.json"), 'w') as f:
        json.dump({"model_name": model_name, "dim": dim}, f)

    return index, encoder


# ============ Step 3: 构建 BM25 稀疏索引 ============

def build_bm25_index(chunks: list[dict]):
    """
    用 jieba 分词后构建 BM25 索引
    
    面试追问：为什么用 jieba 而不是 BGE 的 tokenizer？
    答：BM25 需要的是"词"级别的 token，不是 subword。
    jieba 分出来的是中文词（如"净资产收益率"），
    BGE tokenizer 分出来的是 subword（如"净资"+"产收"+"益率"），
    BM25 用 subword 效果很差。
    """
    import jieba
    from rank_bm25 import BM25Okapi

    texts = [c["text"] for c in chunks]
    
    logger.info(f"开始 jieba 分词 {len(texts)} 条文本")
    start = time.time()
    tokenized = []
    for text in tqdm(texts, desc="jieba分词"):
        tokenized.append(list(jieba.cut(text)))
    elapsed = time.time() - start
    logger.info(f"分词完成: {elapsed:.1f}秒")

    logger.info("构建 BM25 索引")
    bm25 = BM25Okapi(tokenized)

    # 保存
    bm25_path = os.path.join(INDEX_DIR, "bm25_index.pkl")
    with open(bm25_path, 'wb') as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)
    logger.info(f"BM25 索引已保存: {bm25_path}")

    return bm25


# ============ Step 4: 保存元数据 ============

def save_metadata(chunks: list[dict]):
    """
    保存 chunks 的文本和元数据，供检索时使用
    索引中只存向量，文本和元数据需要单独存储
    """
    metadata = {
        "texts": [c["text"] for c in chunks],
        "metadatas": [c["metadata"] for c in chunks],
    }
    
    meta_path = os.path.join(INDEX_DIR, "chunk_metadata.pkl")
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"元数据已保存: {meta_path} ({len(chunks)} 条)")


# ============ Step 5: 测试检索 ============

def test_search(query: str, index, encoder, bm25, chunks: list[dict],
                top_k: int = 5, alpha: float = 0.6,
                source_type_filter: str = None):
    """
    测试混合检索
    
    alpha: 向量检索权重（1-alpha 为 BM25 权重）
    source_type_filter: 按数据类型过滤（"report", "report_fulltext", "financial"）
    """
    import jieba
    import faiss

    # FAISS 检索
    q_emb = encoder.encode([query], normalize_embeddings=True).astype('float32')
    faiss_scores, faiss_ids = index.search(q_emb, top_k * 3)
    
    # BM25 检索
    q_tokens = list(jieba.cut(query))
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_top_ids = np.argsort(bm25_scores)[-top_k * 3:][::-1]

    # 分数归一化
    def normalize(scores):
        min_s, max_s = scores.min(), scores.max()
        if max_s == min_s:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    faiss_norm = normalize(faiss_scores[0])
    bm25_norm = normalize(bm25_scores[bm25_top_ids])

    # 融合
    combined = {}
    for rank, idx in enumerate(faiss_ids[0]):
        if idx >= 0:  # FAISS 可能返回 -1
            combined[int(idx)] = combined.get(int(idx), 0) + alpha * faiss_norm[rank]
    for rank, idx in enumerate(bm25_top_ids):
        combined[int(idx)] = combined.get(int(idx), 0) + (1 - alpha) * bm25_norm[rank]

    # 按分数排序
    sorted_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)

    # source_type 过滤
    results = []
    for idx in sorted_ids:
        meta = chunks[idx]["metadata"]
        if source_type_filter:
            if isinstance(source_type_filter, str):
                if meta["source_type"] != source_type_filter:
                    continue
            elif isinstance(source_type_filter, list):
                if meta["source_type"] not in source_type_filter:
                    continue
        
        results.append({
            "text": chunks[idx]["text"],
            "metadata": meta,
            "score": combined[idx],
        })
        
        if len(results) >= top_k:
            break

    return results


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="FinAgent 检索索引构建")
    parser.add_argument("--test_query", type=str, default=None,
                        help="构建完后测试检索（可选）")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FinAgent 检索索引构建")
    logger.info("=" * 60)

    # 1. 加载 chunks
    chunks = load_chunks(CHUNK_PATH)

    # 2. 构建 FAISS 索引
    faiss_index, encoder = build_faiss_index(chunks)

    # 3. 构建 BM25 索引
    bm25 = build_bm25_index(chunks)

    # 4. 保存元数据
    save_metadata(chunks)

    # 5. 保存索引统计
    stats = {
        "total_chunks": len(chunks),
        "faiss_vectors": faiss_index.ntotal,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": faiss_index.d,
        "index_files": os.listdir(INDEX_DIR),
    }
    with open(os.path.join(INDEX_DIR, "index_stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("索引构建完成！")
    logger.info(f"  FAISS: {faiss_index.ntotal} 条向量, {faiss_index.d} 维")
    logger.info(f"  BM25: {len(chunks)} 条文档")
    logger.info(f"  索引目录: {os.path.abspath(INDEX_DIR)}")
    logger.info("=" * 60)

    # 6. 可选：测试检索
    if args.test_query:
        logger.info(f"\n测试检索: '{args.test_query}'")
        
        # 测试1：不过滤
        print(f"\n===== 全量检索（不过滤）=====")
        results = test_search(args.test_query, faiss_index, encoder, bm25, chunks)
        for i, r in enumerate(results):
            print(f"\n[{i+1}] score={r['score']:.3f} | {r['metadata']['source_type']} | {r['metadata'].get('stock_name', '')}")
            print(f"    {r['text'][:150]}...")

        # 测试2：只搜研报
        print(f"\n===== search_report（只搜研报）=====")
        results = test_search(args.test_query, faiss_index, encoder, bm25, chunks,
                            source_type_filter=["report", "report_fulltext"])
        for i, r in enumerate(results):
            print(f"\n[{i+1}] score={r['score']:.3f} | {r['metadata']['source_type']} | {r['metadata'].get('stock_name', '')}")
            print(f"    {r['text'][:150]}...")

        # 测试3：只搜财务
        print(f"\n===== search_financial（只搜财务）=====")
        results = test_search(args.test_query, faiss_index, encoder, bm25, chunks,
                            source_type_filter="financial")
        for i, r in enumerate(results):
            print(f"\n[{i+1}] score={r['score']:.3f} | {r['metadata']['source_type']} | {r['metadata'].get('stock_name', '')}")
            print(f"    {r['text'][:150]}...")


if __name__ == "__main__":
    main()
