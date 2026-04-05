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

        # 6. 预计算行业名 → chunk 下标的精确映射
        self._build_industry_index()

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

    # 东财细分行业 → 大类映射（复制自 04_build_chunks.py 的 INDUSTRY_MAP）
    # ⚠️ 如果 04_build_chunks.py 的 INDUSTRY_MAP 有更新，这里必须同步修改
    # 不直接 import 是因为两个文件的使用场景不同（build_chunks 只在数据构建时跑）
    _INDUSTRY_MAP = {
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
        "其他电源设备Ⅱ": [],  # 东财没有细分，只有大类自身
    }

    # 模型口语化别名（东财没有，但模型实际会搜的说法）
    # 基于 100 条推理结果中 84 次 search_industry 的实际 query 验证
    _EXTRA_ALIASES = {
        "白酒": ["白酒", "酒类", "高端白酒"],
        "银行": ["银行", "商业银行", "金融"],
        "保险": ["保险", "寿险", "财险"],
        "证券": ["证券", "券商", "投行"],
        "半导体": ["半导体", "芯片", "半导体设备", "半导体显示", "集成电路", "IC"],
        "消费电子": ["消费电子", "面板", "显示", "PCB", "模组", "电子元器件",
                    "电子制造", "电子行业", "面板显示器件"],
        "光伏": ["光伏", "太阳能", "硅片", "组件"],
        "电池": ["电池", "锂电池", "储能", "动力电池", "锂电材料"],
        "汽车": ["汽车", "新能源汽车", "新能源车", "整车"],
        "医药": ["医药", "生物医药", "创新药", "制药", "疫苗"],
        "医疗": ["医疗", "CXO", "体外诊断"],
        "家电": ["家电", "白电", "空调"],
        "化工": ["化工", "精细化工", "化纤", "钛白粉"],
        "煤炭": ["煤炭", "动力煤", "焦煤"],
        "钢铁": ["钢铁", "特钢", "螺纹钢"],
        "有色金属": ["有色金属", "铜", "铝", "黄金", "矿业", "黄金采选"],
        "电力": ["电力", "火电", "水电", "核电", "新能源发电", "新能源行业"],
        "军工": ["军工", "国防", "航天", "军工电子", "航空装备"],
        "机械": ["机械", "机器人", "自动化", "工控"],
        "房地产": ["房地产", "地产", "房企", "楼市"],
        "建筑建材": ["建筑建材", "建筑", "建材", "水泥", "玻璃"],
        "食品饮料": ["食品饮料", "食品", "饮料", "乳制品", "调味品", "食品加工"],
        "软件": ["软件", "IT", "信息技术", "云计算", "SaaS", "计算机", "计算机硬件"],
        "通信": ["通信", "5G", "光通信", "运营商"],
        "交通运输": ["交通运输", "物流", "航运", "航空运输", "铁路", "高速公路",
                    "快递", "货运", "铁路运输"],
        "传媒": ["传媒", "广告", "游戏", "影视"],
        "农业": ["农业", "养殖", "种植", "饲料", "畜牧"],
        "石油石化": ["石油石化", "石油", "石化", "天然气", "油气", "石油化工", "炼化"],
        "零售": ["零售", "商超", "电商"],
        "其他电源设备Ⅱ": ["其他电源设备", "电源设备", "电力设备", "发电设备", "电气设备"],
    }

    def _build_industry_index(self):
        """
        预计算行业名 → chunk 下标的精确映射。

        别名来源分两层：
        1. 东财 INDUSTRY_MAP 的反向映射（130 个细分行业名，官方分类）
        2. _EXTRA_ALIASES 的口语化别名（基于模型实际 query 验证）
        3. 标准大类名自身（"银行"→"银行"）

        以后 INDUSTRY_MAP 加了新行业，这里自动跟上。
        """
        import re
        self._industry_name_to_idx = {}   # 标准行业名 → chunk 下标
        self._alias_to_industry = {}       # 别名 → 标准行业名

        # Step 1: 从 chunk 文本提取标准行业名
        industry_ids = self._source_type_ids.get("industry", set())
        for idx in industry_ids:
            text = self.texts[idx]
            match = re.match(r'(\S+?)行业对比', text)
            if match:
                name = match.group(1)
                self._industry_name_to_idx[name] = idx

        # Step 2: 从 INDUSTRY_MAP 构建反向映射（东财细分 → 大类）
        for standard_name, sub_industries in self._INDUSTRY_MAP.items():
            if standard_name in self._industry_name_to_idx:
                # 标准大类名自身
                self._alias_to_industry[standard_name] = standard_name
                # 东财细分行业名
                for sub in sub_industries:
                    # 去掉Ⅱ后缀也作为别名（模型不会打Ⅱ）
                    self._alias_to_industry[sub] = standard_name
                    clean = sub.replace("Ⅱ", "").strip()
                    if clean and clean != sub:
                        self._alias_to_industry[clean] = standard_name

        # Step 3: 合并口语化别名（不覆盖已有的东财映射）
        for standard_name, aliases in self._EXTRA_ALIASES.items():
            if standard_name in self._industry_name_to_idx:
                for alias in aliases:
                    if alias not in self._alias_to_industry:
                        self._alias_to_industry[alias] = standard_name

        logger.info(f"  行业索引: {len(self._industry_name_to_idx)} 个行业, "
                    f"{len(self._alias_to_industry)} 个别名")

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

    def search_industry(self, query: str, top_k: int = 5) -> list[dict]:
        """
        检索行业汇总数据

        策略：行业名精确匹配（不走 FAISS/BM25 混合检索）

        为什么不用混合检索？
        行业 chunk 只有 30 条，所有 chunk 都包含相同的指标关键词（ROE、净利率等），
        BM25 在这 30 条中区分度极差（实测 50% 的查询返回"农业"而非目标行业）。
        30 条数据不需要向量检索，直接按行业名精确匹配即可。

        匹配逻辑：
        1. 从 query 中提取行业关键词
        2. 通过别名映射找到标准行业名
        3. 直接返回对应的行业 chunk
        4. 匹配不上则返回空（不返回错误行业）
        """
        if not self._industry_name_to_idx:
            return []

        # 从 query 中匹配行业名（优先匹配最长的别名，避免"石油石化"被"石油"截断）
        matched_industries = []
        sorted_aliases = sorted(self._alias_to_industry.keys(), key=len, reverse=True)
        for alias in sorted_aliases:
            if alias in query:
                standard_name = self._alias_to_industry[alias]
                if standard_name not in matched_industries:
                    matched_industries.append(standard_name)
                if len(matched_industries) >= top_k:
                    break

        # 如果没有匹配到任何行业，尝试直接匹配标准行业名
        if not matched_industries:
            for name in self._industry_name_to_idx:
                if name in query:
                    matched_industries.append(name)

        # 构建返回结果
        results = []
        for name in matched_industries[:top_k]:
            idx = self._industry_name_to_idx[name]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadatas[idx],
                "score": 1.0,
            })

        if not results:
            logger.warning(f"search_industry 未匹配到行业: query='{query}'")
            available = "、".join(sorted(self._industry_name_to_idx.keys()))
            return [{
                "text": (
                    f"未找到匹配的行业数据。请在 query 中使用以下标准行业名之一：{available}。"
                    f"例如：search_industry(query='半导体 ROE 盈利能力')。"
                ),
                "metadata": {"source_type": "industry", "match_failed": True},
                "score": 0.0,
            }]

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

        # 交替合并去重（保证两路都有露出）
        combined = self._interleave_results(meta_results, text_results, top_k)

        # 按 score 重排：time_bonus 已经加进 score 里了，
        # 现在两路的分数可比（都包含了时效性加分），按分数统一排序
        # 这样 2017 老研报即使 BM25 分高，也会被 time_bonus 更大的新研报压下去
        combined.sort(key=lambda x: x["score"], reverse=True)

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

        # ---------- 时效性加分 ----------
        # 只对 report / report_fulltext 生效，financial 不加
        # 解决的问题：BM25 关键词匹配可能把 7 年前的老研报排在最新研报前面
        for idx in combined:
            combined[idx] += self._time_bonus(self.metadatas[idx])

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
    def _time_bonus(metadata: dict, half_life_days: int = 365,
                    max_bonus: float = 0.4) -> float:
        """
        时效性加分：越新的文档 bonus 越高，指数衰减

        bonus = max_bonus * exp(-0.693 * days_ago / half_life_days)

        参数选择：
        - half_life_days=365：一年前的文档 bonus 衰减到一半（0.20）
        - max_bonus=0.4：初始 0.3 时 2017 老研报仍偶尔排进 top-3，
          调到 0.4 后新研报 bonus 约 0.34（半年内），老研报接近 0，拉开足够差距

        设计决策：
        - 只对 report / report_fulltext 生效
        - financial 不加（用户可能需要旧财务数据做趋势分析）

        面试话术：
        "SFT 数据测试中发现 search_report('贵州茅台 目标价') 返回了 2017 年的老研报，
        因为标题里恰好有'目标价'三个字，BM25 给了高分。加了指数衰减的 time_bonus 后，
        7 年前的研报 bonus 接近 0，半年内的研报能拿到 0.2+ 的 bonus，排序直接翻转。
        half_life 设 365 天是因为金融研报时效性强，一年前的评级和目标价参考价值大幅下降。"
        """
        import math
        from datetime import datetime

        source_type = metadata.get("source_type", "")
        if source_type not in ("report", "report_fulltext"):
            return 0.0

        # 兼容两种日期字段名：report/report_fulltext 用 "date"，financial 用 "report_date"
        date_str = metadata.get("date") or metadata.get("report_date") or ""
        date_str = str(date_str).strip()[:10]  # 处理 "2025-11-05 00:00:00" 格式

        if not date_str:
            return 0.0

        try:
            doc_date = datetime.strptime(date_str, "%Y-%m-%d")
            days_ago = max((datetime.now() - doc_date).days, 0)
            decay = math.exp(-0.693 * days_ago / half_life_days)
            return max_bonus * decay
        except (ValueError, TypeError):
            return 0.0

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
        ("search_report", "贵州茅台 目标价"),  # 时效性测试：之前返回 2017 老研报
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
            date = r["metadata"].get("date") or r["metadata"].get("report_date", "")
            score = r["score"]
            text_preview = r["text"][:120].replace("\n", " ")
            print(f"  [{i+1}] {score:.3f} | {source} | {name} | {str(date)[:10]}")
            print(f"      {text_preview}...")


if __name__ == "__main__":
    main()
