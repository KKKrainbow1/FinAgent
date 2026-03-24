"""
FinAgent Step 6: 统一检索模块
用途：封装 FAISS + BM25 检索逻辑，供 Agent 工具层调用
依赖：pip install sentence-transformers faiss-gpu rank_bm25 jieba numpy

设计要点：
    1. search_financial: 混合检索（FAISS 为主），只搜 financial chunks
       - 你的实验已证明：纯 BM25 对 financial 有系统性偏差（总返回偿债 chunk）
       - FAISS 能正确区分"盈利能力"和"偿债能力"的语义
       - BM25 辅助精确匹配公司名和指标名
    2. search_report: 两路合并
       - 研报元数据（report）→ BM25 为主（结构化短文本，关键词精确匹配）
       - PDF 正文（report_fulltext）→ FAISS 为主（自由文本，需要语义理解）
       - 两路各取 top-K，交替合并返回

面试追问：为什么 search_financial 不用纯 BM25？
答：最初假设结构化数据用 BM25 就够了，但实验发现 BM25 对文档长度有偏置——
偿债 chunk 字段更多、文本更长，BM25 的 TF-IDF 打分倾向给长文档更高分。
搜"贵州茅台盈利能力"时 BM25 top-3 全是偿债指标，而 FAISS 能正确返回盈利指标。
所以 financial 也改用了混合检索，FAISS 权重调高（alpha=0.7）以语义匹配为主。

面试追问：search_report 为什么用两路合并而不是统一混合？
答：研报元数据和 PDF 正文的文本特征差异太大。元数据是结构化短文本（标题+评级+EPS），
PDF 正文是自由长文本（分析论证）。统一混合时 PDF 正文的 chunk 数量远多于元数据（14,684 vs 39,044），
但元数据里的评级、EPS 预测等信息是高价值的。两路分别检索再合并，保证两种数据都能被召回。
"""

import json
import os
import pickle
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============ 配置 ============
INDEX_DIR = "./data/index"
EMBEDDING_MODEL = "./models/BAAI/bge-base-zh-v1___5"


class FinAgentRetriever:
    """
    FinAgent 统一检索器

    加载一套索引（FAISS + BM25 + 元数据），
    根据工具类型选择不同的检索策略和 source_type 过滤。

    用法：
        retriever = FinAgentRetriever()
        retriever.load_index()

        # Agent 工具调用
        results = retriever.search_financial("贵州茅台 ROE 2024")
        results = retriever.search_report("宁德时代 投资评级 目标价")
    """

    def __init__(self, index_dir: str = INDEX_DIR, model_path: str = EMBEDDING_MODEL):
        self.index_dir = index_dir
        self.model_path = model_path

        # 延迟加载，调用 load_index() 后才可用
        self.faiss_index = None
        self.bm25 = None
        self.encoder = None
        self.texts = []       # 所有 chunk 的文本
        self.metadatas = []   # 所有 chunk 的元数据

        # 预计算的 source_type → chunk 下标映射，加速过滤
        self._source_type_ids = {}

    def load_index(self):
        """加载 FAISS 索引、BM25 索引、embedding 模型、chunk 元数据"""
        import faiss
        from sentence_transformers import SentenceTransformer

        logger.info("加载检索索引...")

        # 1. FAISS
        faiss_path = os.path.join(self.index_dir, "faiss_index.bin")
        self.faiss_index = faiss.read_index(faiss_path)
        logger.info(f"  FAISS: {self.faiss_index.ntotal} 条向量, {self.faiss_index.d} 维")

        # 2. BM25
        bm25_path = os.path.join(self.index_dir, "bm25_index.pkl")
        with open(bm25_path, 'rb') as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        logger.info(f"  BM25: 已加载")

        # 3. Embedding 模型
        self.encoder = SentenceTransformer(self.model_path)
        logger.info(f"  Encoder: {self.model_path} ({self.encoder.device})")

        # 4. Chunk 元数据（文本 + metadata）
        meta_path = os.path.join(self.index_dir, "chunk_metadata.pkl")
        with open(meta_path, 'rb') as f:
            meta_data = pickle.load(f)
        self.texts = meta_data["texts"]
        self.metadatas = meta_data["metadatas"]
        logger.info(f"  Chunks: {len(self.texts)} 条")

        # 5. 预计算 source_type → 下标集合（避免每次检索都遍历）
        self._build_source_type_index()

        logger.info("检索索引加载完成")

    def _build_source_type_index(self):
        """预计算每种 source_type 对应的 chunk 下标集合"""
        self._source_type_ids = {}
        for idx, meta in enumerate(self.metadatas):
            st = meta.get("source_type", "unknown")
            if st not in self._source_type_ids:
                self._source_type_ids[st] = set()
            self._source_type_ids[st].add(idx)

        for st, ids in self._source_type_ids.items():
            logger.info(f"  source_type '{st}': {len(ids)} 条")

    # ================================================================
    #  核心检索方法：供 tools.py 调用
    # ================================================================

    def search_financial(self, query: str, top_k: int = 5) -> list[dict]:
        """
        检索财务数据

        策略：混合检索（FAISS 权重 0.7 + BM25 权重 0.3）
        范围：只返回 source_type == "financial" 的 chunk

        为什么 FAISS 权重更高？
        因为 financial chunk 分为"盈利"和"偿债"两类，BM25 无法区分，
        FAISS 能通过语义匹配正确区分"盈利能力"和"偿债能力"。
        BM25 保留 0.3 的权重用于精确匹配公司名。
        """
        valid_ids = self._source_type_ids.get("financial", set())
        results = self._hybrid_search(
            query,
            valid_ids=valid_ids,
            alpha=0.7,        # FAISS 权重高，解决 BM25 偏债务 chunk 的问题
            top_k=top_k,
        )
        return results

    def search_report(self, query: str, top_k: int = 5) -> list[dict]:
        """
        检索研报信息（元数据 + PDF 正文）

        策略：两路分别检索，交替合并
        - 路线1：研报元数据（report）→ 混合检索，BM25 权重高（alpha=0.4）
          因为元数据是结构化短文本，关键词精确匹配更有效
        - 路线2：PDF 正文（report_fulltext）→ 混合检索，FAISS 权重高（alpha=0.7）
          因为正文是自由文本，需要语义理解（"护城河" ↔ "品牌壁垒"）
        - 合并：两路各取 top_k 条，交替插入，去重后返回 top_k
        """
        # 路线1：研报元数据
        meta_ids = self._source_type_ids.get("report", set())
        meta_results = self._hybrid_search(
            query,
            valid_ids=meta_ids,
            alpha=0.4,         # BM25 权重高，精确匹配标题、评级、机构名
            top_k=top_k,
        )

        # 路线2：PDF 正文
        text_ids = self._source_type_ids.get("report_fulltext", set())
        text_results = self._hybrid_search(
            query,
            valid_ids=text_ids,
            alpha=0.7,         # FAISS 权重高，语义匹配
            top_k=top_k,
        )

        # 交替合并去重
        combined = self._interleave_results(meta_results, text_results, top_k)
        return combined

    # ================================================================
    #  底层检索实现
    # ================================================================

    def _hybrid_search(self, query: str, valid_ids: set,
                       alpha: float = 0.6, top_k: int = 5) -> list[dict]:
        """
        混合检索的底层实现

        Args:
            query: 检索词
            valid_ids: 允许返回的 chunk 下标集合（用于 source_type 过滤）
            alpha: FAISS 权重（1-alpha 为 BM25 权重）
            top_k: 返回条数

        Returns:
            [{"text": str, "metadata": dict, "score": float}, ...]
        """
        import jieba

        if not valid_ids:
            return []

        # ---------- FAISS 检索 ----------
        # 多召回一些候选（top_k * 5），过滤后再截断
        n_candidates = min(top_k * 5, self.faiss_index.ntotal)
        q_emb = self.encoder.encode([query], normalize_embeddings=True).astype('float32')
        faiss_scores_raw, faiss_ids_raw = self.faiss_index.search(q_emb, n_candidates)

        # 只保留 valid_ids 内的结果
        faiss_scores = []
        faiss_ids = []
        for score, idx in zip(faiss_scores_raw[0], faiss_ids_raw[0]):
            if idx >= 0 and int(idx) in valid_ids:
                faiss_scores.append(score)
                faiss_ids.append(int(idx))

        # ---------- BM25 检索 ----------
        q_tokens = list(jieba.cut(query))
        bm25_all_scores = self.bm25.get_scores(q_tokens)

        # 只保留 valid_ids 内的，取 top 候选
        bm25_candidates = [(idx, bm25_all_scores[idx]) for idx in valid_ids]
        bm25_candidates.sort(key=lambda x: x[1], reverse=True)
        bm25_ids = [c[0] for c in bm25_candidates[:n_candidates]]
        bm25_scores = [c[1] for c in bm25_candidates[:n_candidates]]

        # ---------- 分数归一化 ----------
        faiss_norm = self._normalize(np.array(faiss_scores)) if faiss_scores else {}
        bm25_norm = self._normalize(np.array(bm25_scores)) if bm25_scores else {}

        # 转成 {idx: normalized_score} 的字典
        faiss_dict = {faiss_ids[i]: faiss_norm[i] for i in range(len(faiss_ids))}
        bm25_dict = {bm25_ids[i]: bm25_norm[i] for i in range(len(bm25_ids))}

        # ---------- 加权融合 ----------
        all_ids = set(faiss_dict.keys()) | set(bm25_dict.keys())
        combined = {}
        for idx in all_ids:
            f_score = faiss_dict.get(idx, 0.0)
            b_score = bm25_dict.get(idx, 0.0)
            combined[idx] = alpha * f_score + (1 - alpha) * b_score

        # ---------- 排序返回 ----------
        sorted_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)

        results = []
        for idx in sorted_ids[:top_k]:
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadatas[idx],
                "score": float(combined[idx]),
            })
        return results

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Min-Max 归一化到 [0, 1]"""
        if len(scores) == 0:
            return scores
        min_s, max_s = scores.min(), scores.max()
        if max_s == min_s:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    @staticmethod
    def _interleave_results(list_a: list[dict], list_b: list[dict],
                            top_k: int) -> list[dict]:
        """
        交替合并两路结果，去重（按 text 去重）

        为什么用交替而不是按分数排序？
        因为两路的分数不在同一尺度上（不同的 alpha、不同的 source_type），
        直接比较分数没有意义。交替插入保证两种数据都有露出。
        """
        seen_texts = set()
        combined = []

        i, j = 0, 0
        while len(combined) < top_k and (i < len(list_a) or j < len(list_b)):
            # 先插 list_a
            if i < len(list_a):
                text_key = list_a[i]["text"][:100]  # 用前100字符做去重key
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    combined.append(list_a[i])
                i += 1

            # 再插 list_b
            if j < len(list_b) and len(combined) < top_k:
                text_key = list_b[j]["text"][:100]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    combined.append(list_b[j])
                j += 1

        return combined


# ================================================================
#  测试脚本：直接运行此文件可测试检索效果
# ================================================================

def main():
    """测试检索效果"""
    retriever = FinAgentRetriever()
    retriever.load_index()

    test_queries = [
        ("search_financial", "贵州茅台盈利能力"),
        ("search_financial", "宁德时代ROE"),
        ("search_financial", "比亚迪营收增长率"),
        ("search_financial", "招商银行资产负债率"),
        ("search_report", "宁德时代 投资评级 目标价"),
        ("search_report", "贵州茅台 竞争优势 护城河"),
        ("search_report", "隆基绿能 行业前景"),
    ]

    for tool, query in test_queries:
        print(f"\n{'='*60}")
        print(f"工具: {tool} | Query: {query}")
        print('='*60)

        if tool == "search_financial":
            results = retriever.search_financial(query)
        else:
            results = retriever.search_report(query)

        for i, r in enumerate(results):
            source = r["metadata"]["source_type"]
            name = r["metadata"].get("stock_name", "")
            score = r["score"]
            text_preview = r["text"][:120].replace("\n", " ")
            print(f"  [{i+1}] {score:.3f} | {source} | {name}")
            print(f"      {text_preview}...")


if __name__ == "__main__":
    main()
