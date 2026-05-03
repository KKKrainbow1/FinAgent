"""
FinAgent Retriever (V3 · 多 collection 版)

架构演进:V1 FAISS(单 collection) → V3 Milvus 单 collection + scalar filter →
        V3.5 Milvus 多 collection(本文件,2026-05-03)

设计变更(对比 V3 单 collection):
  1. 5 个独立 collection,5 个独立 .db 文件,schema 各自精简(字段减半)
  2. 工具数 4 → 5:search_report 拆为 search_report_meta + search_report_content
  3. 全 RRFRanker(60),废 WeightedRanker
  4. prose 不再切 fixed_window child + parent dedup;改 section 整段(章节级,带 head 锚定)
  5. table_row_fact 不再灌库(05 已 filter,索引瘦身 80%),只保留 table_narrative
  6. enrich_with_parent 删 prose 分支,只保留 table 分支(每 parent_id 1 个 child,trivial dedup)
  7. 删 stock_code 字典硬抓(LLM 记不住 + 字典覆盖不全),保留期间词解析(4 个词,真硬约束)

对外 API(变化):
    retriever = FinAgentRetriever()
    retriever.search_report_meta(query, top_k=5)         # 新,评级摘要
    retriever.search_report_content(query, top_k=5)      # 新,正文+表格(text section + tabular)
    retriever.search_financial(query, top_k=5, stock_code=None)
    retriever.search_industry(query, top_k=5)

    # 旧 search_report 保留作兼容(meta + content 二路融合,新代码不应再调)
    retriever.search_report(query, top_k=5)

每个 dict 格式:{text, metadata, score},和 V2/V3 保持一致。

Collection 布局:
    data/processed/milvus_report_meta.db     → collection 'report_meta'
    data/processed/milvus_report_section.db  → collection 'report_section'
    data/processed/milvus_report_tabular.db  → collection 'report_tabular'
    data/processed/milvus_financial.db       → collection 'financial'
    data/processed/milvus_industry.db        → collection 'industry_alias'

依赖:
  pip install pymilvus "pymilvus[model]" FlagEmbedding milvus-lite

V1 FAISS 版本备份在 hybrid_search_legacy_faiss.py。
旧单 collection V3 版本备份在 hybrid_search_legacy_single_collection.py(由 git history 保留)。
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

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

# ============ 5 collection 文件路径(默认值) ============
DEFAULT_DB_DIR = ROOT / "data/processed"
DB_PATHS = {
    "report_meta":     DEFAULT_DB_DIR / "milvus_report_meta.db",
    "report_section":  DEFAULT_DB_DIR / "milvus_report_section.db",
    "report_tabular":  DEFAULT_DB_DIR / "milvus_report_tabular.db",
    "financial":       DEFAULT_DB_DIR / "milvus_financial.db",
    "industry_alias":  DEFAULT_DB_DIR / "milvus_industry.db",
}
COLLECTION_NAMES = {
    "report_meta":     "report_meta",
    "report_section":  "report_section",
    "report_tabular":  "report_tabular",
    "financial":       "financial",
    "industry_alias":  "industry_alias",
}
DEFAULT_ENCODER = "./models/bge-m3"

# search_report_content 上下文预算(字符数):Qwen2.5-14B 32K 窗口,留给单次 observation ~12K 字符
SEARCH_REPORT_BUDGET_CHARS = 12000

# query 期间关键词 → 研报点评发布日期区间(A 股惯例):
#   年报:        3-5 月点评(报告期年 + 1 = 发布年)
#   一季报:      4-6 月点评
#   中报:        7-9 月点评
#   三季报:      10-12 月点评
# query 含期间词 + year(如 "2025 三季报") → date 区间硬过滤进 expr
_PERIOD_TO_DATE_RANGE = {
    "年报":   ("-03-01", "-05-15"),
    "一季报": ("-04-15", "-06-01"),
    "中报":   ("-07-01", "-10-01"),
    "三季报": ("-10-01", "-12-15"),
}


def _extract_period_clause(query: str) -> str:
    """query 含 'YYYY' + 期间词 → 返回 ` and date >= "lo" and date < "hi"` 子句,
    否则返回空字符串(纯 RRF 兜底)。

    语义:query 里的 YYYY 是**报告期年**,不是**发布年**。
        一季报 / 中报 / 三季报 → 同年(YYYY)发布
        年报                   → 次年(YYYY + 1)发布
    "贵州茅台 2025 三季报点评" → ` and date >= "2025-10-01" and date < "2025-12-15"`
    "格力电器 2024 年报点评"   → ` and date >= "2025-03-01" and date < "2025-05-15"`
    """
    import re
    ym = re.search(r"(20\d{2})", query)
    if not ym:
        return ""
    report_year = int(ym.group(1))
    for kw in sorted(_PERIOD_TO_DATE_RANGE.keys(), key=len, reverse=True):
        if kw in query:
            lo_suf, hi_suf = _PERIOD_TO_DATE_RANGE[kw]
            publish_year = report_year + 1 if kw == "年报" else report_year
            return f' and date >= "{publish_year}{lo_suf}" and date < "{publish_year}{hi_suf}"'
    return ""


def _sparse_to_dict(sp):
    """单行 sparse → {int: float} dict(Milvus SPARSE 要求升序)"""
    if isinstance(sp, dict):
        return {int(k): float(v) for k, v in sp.items()}
    if hasattr(sp, 'tocsr'):
        try:
            sp = sp.tocsr()
        except Exception:
            pass
    if hasattr(sp, 'indices') and hasattr(sp, 'data'):
        pairs = sorted(
            ((int(k), float(v)) for k, v in zip(sp.indices, sp.data)),
            key=lambda kv: kv[0],
        )
        return dict(pairs)
    if hasattr(sp, 'col') and hasattr(sp, 'data'):
        pairs = sorted(
            ((int(k), float(v)) for k, v in zip(sp.col, sp.data)),
            key=lambda kv: kv[0],
        )
        return dict(pairs)
    raise ValueError(f"unsupported sparse format: {type(sp)}")


def _sparse_from_row(sp_raw, row_idx: int = 0):
    """从 embedding 模型返回的 sparse 里取第 row_idx 行 → dict{int: float}。"""
    if isinstance(sp_raw, list):
        return {int(k): float(v) for k, v in sp_raw[row_idx].items()}
    if hasattr(sp_raw, 'tocsr'):
        try:
            csr = sp_raw.tocsr()
            if hasattr(csr, 'indptr') and getattr(csr, 'ndim', 2) == 2 and csr.shape[0] > 1:
                start = int(csr.indptr[row_idx])
                end = int(csr.indptr[row_idx + 1])
                pairs = sorted(
                    ((int(k), float(v)) for k, v in zip(csr.indices[start:end], csr.data[start:end])),
                    key=lambda kv: kv[0],
                )
                return dict(pairs)
        except Exception:
            pass
    return _sparse_to_dict(sp_raw)


class FinAgentRetriever:
    """
    V3.5 Milvus 多 collection 检索器。无参构造自动加载 5 个独立 .db。
    """

    # ============ 行业别名字典(给 search_industry 字典快路径 / 05a / reward_knowledge_base 用) ============

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
        "其他电源设备Ⅱ": [],
    }

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

    # ============ 构造函数 ============

    def __init__(
        self,
        db_dir: Optional[str] = None,
        encoder_model: str = DEFAULT_ENCODER,
        device: str = "cuda:0",
        use_fp16: bool = True,
    ):
        from pymilvus import MilvusClient

        # 5 个 db 路径(允许 db_dir 覆盖默认 data/processed)
        if db_dir:
            base = Path(db_dir)
            paths = {k: base / Path(v).name for k, v in DB_PATHS.items()}
        else:
            paths = DB_PATHS

        # 5 个 client(每 client 句柄轻量,共享一份 BGE-m3 encoder)
        self.clients = {}
        for key, p in paths.items():
            if not Path(p).exists():
                logger.warning(f"db 不存在: {p}(对应 collection {key} 将不可用)")
            self.clients[key] = MilvusClient(str(p))
            logger.info(f"Milvus Lite 连接: {p}")
        self.collections = COLLECTION_NAMES

        # bge-m3:dense + native sparse 一体化(替代 BGE dense + jieba/BM25 两路)
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction
        logger.info(f"加载 bge-m3 encoder: {encoder_model} (device={device}, fp16={use_fp16})")
        self.m3_ef = BGEM3EmbeddingFunction(
            model_name=encoder_model, device=device, use_fp16=use_fp16,
        )

        # 扁平 alias → industry map(快路径:字典命中优先,否则向量 fallback)
        self._alias_to_industry = self._flatten_aliases()
        logger.info(f"行业别名字典: {len(self._alias_to_industry)} 个别名")

        logger.info("✅ FinAgentRetriever V3.5(5 collection)就绪")

    # ============ helpers ============

    @classmethod
    def _flatten_aliases(cls) -> dict:
        """扁平化 _INDUSTRY_MAP + _EXTRA_ALIASES 为 alias → industry"""
        mp = {}
        for standard in cls._INDUSTRY_MAP:
            mp[standard] = standard
        for standard, subs in cls._INDUSTRY_MAP.items():
            for sub in subs:
                mp[sub] = standard
                clean = sub.replace("Ⅱ", "").strip()
                if clean and clean != sub:
                    mp[clean] = standard
        for standard, aliases in cls._EXTRA_ALIASES.items():
            for a in aliases:
                if a not in mp:
                    mp[a] = standard
        return mp

    def _embed(self, query: str) -> tuple[list, dict]:
        """query → (dense 向量 list, sparse {col: weight} dict)"""
        result = self.m3_ef.encode_queries([query])
        d0 = result["dense"][0]
        dense = d0.tolist() if hasattr(d0, 'tolist') else list(d0)
        sparse = _sparse_from_row(result["sparse"])
        if not sparse:
            logger.warning(f"bge-m3 query sparse 空: {query!r},用占位 sparse")
            sparse = {0: 1e-6}
        return dense, sparse

    @staticmethod
    def _hit_to_chunk(hit) -> dict:
        """Milvus hit → {text, metadata, score} 标准格式。"""
        if isinstance(hit, dict):
            entity = dict(hit.get('entity', {}))
            pk = hit.get('id') or hit.get('pk')
            distance = hit.get('distance', 0.0)
        else:
            entity = dict(getattr(hit, 'entity', {}) or {})
            pk = getattr(hit, 'id', None) or getattr(hit, 'pk', None)
            distance = getattr(hit, 'distance', 0.0)

        text = entity.pop('text', '')
        if pk is not None:
            entity['chunk_id'] = pk
        return {
            'text': text,
            'metadata': entity,
            'score': float(distance),
        }

    # ============ search_report_meta(评级摘要主路) ============

    def search_report_meta(self, query: str, top_k: int = 5) -> list[dict]:
        """检索研报评级摘要(机构观点 / 评级 / 目标价 / EPS 预测)。
        RRF 融合 dense + sparse,期间词 + 年份齐全才加 date expr。
        """
        from pymilvus import AnnSearchRequest, RRFRanker
        q_dense, q_sparse = self._embed(query)
        # 整个 collection 都是 source_type='report' 类型,不需要 source_type filter
        period_clause = _extract_period_clause(query)
        # 这里 expr 整体可以是 None(无 period 时);Milvus 接受
        expr = period_clause.removeprefix(" and ") if period_clause else None

        results = self.clients["report_meta"].hybrid_search(
            collection_name=self.collections["report_meta"],
            reqs=[
                AnnSearchRequest(data=[q_dense],  anns_field="dense",
                                 param={"metric_type": "COSINE"}, limit=30, expr=expr),
                AnnSearchRequest(data=[q_sparse], anns_field="sparse",
                                 param={"metric_type": "IP"},     limit=30, expr=expr),
            ],
            ranker=RRFRanker(k=60),
            output_fields=["text", "stock_code", "stock_name", "institution",
                           "date", "rating", "report_title", "pdf_file"],
            limit=top_k,
        )
        return [self._hit_to_chunk(h) for h in results[0]]

    # ============ search_report_content(正文 section + 表格 二路 RRF) ============

    def search_report_content(self, query: str, top_k: int = 5) -> list[dict]:
        """检索研报正文章节 + 表格内容。
        内部:section + tabular 两个 collection 独立查 → 应用层 RRF 融合 → table 走 enrich_with_parent。
        """
        from pymilvus import AnnSearchRequest, RRFRanker
        q_dense, q_sparse = self._embed(query)
        period_clause = _extract_period_clause(query)
        period_expr = period_clause.removeprefix(" and ") if period_clause else None

        # 1) section 路(prose 整段 chunk)
        section_hits = self.clients["report_section"].hybrid_search(
            collection_name=self.collections["report_section"],
            reqs=[
                AnnSearchRequest(data=[q_dense],  anns_field="dense",
                                 param={"metric_type": "COSINE"}, limit=30, expr=period_expr),
                AnnSearchRequest(data=[q_sparse], anns_field="sparse",
                                 param={"metric_type": "IP"},     limit=30, expr=period_expr),
            ],
            ranker=RRFRanker(k=60),
            output_fields=["text", "stock_code", "stock_name", "industry", "date",
                           "report_title", "section_id", "section_title",
                           "page_idx", "pdf_file"],
            limit=top_k * 2,
        )
        section_chunks = [self._hit_to_chunk(h) for h in section_hits[0]]

        # 2) tabular 路(table_narrative + parent_md)
        tab_hits = self.clients["report_tabular"].hybrid_search(
            collection_name=self.collections["report_tabular"],
            reqs=[
                AnnSearchRequest(data=[q_dense],  anns_field="dense",
                                 param={"metric_type": "COSINE"}, limit=60, expr=period_expr),
                AnnSearchRequest(data=[q_sparse], anns_field="sparse",
                                 param={"metric_type": "IP"},     limit=60, expr=period_expr),
            ],
            ranker=RRFRanker(k=60),
            output_fields=["text", "chunk_method", "stock_code", "stock_name",
                           "industry", "date", "report_title",
                           "parent_id", "parent_md", "table_caption", "table_footnote",
                           "page_idx", "pdf_file"],
            limit=top_k * 2,
        )
        tab_chunks = [self._hit_to_chunk(h) for h in tab_hits[0]]
        tab_chunks = self._enrich_table(tab_chunks)

        # 3) 应用层 RRF 二路融合(以 pdf_file 为聚合 key)
        merged = self._rrf_merge_by_pdf(section_chunks, tab_chunks, k=60, top_k=top_k * 6)

        # 4) per-pdf cap=3 + budget 预算截取
        MAX_OUT_PER_PDF = 3
        pdf_count: dict = {}
        out, total = [], 0
        for chunk in merged:
            m = chunk.get('metadata', {})
            pdf = m.get('pdf_file') or m.get('chunk_id', '') or '_no_pdf'
            if pdf_count.get(pdf, 0) >= MAX_OUT_PER_PDF:
                continue
            tlen = len(chunk.get('text', ''))
            if out and total + tlen > SEARCH_REPORT_BUDGET_CHARS:
                break
            out.append(chunk)
            total += tlen
            pdf_count[pdf] = pdf_count.get(pdf, 0) + 1
            if len(out) >= top_k:
                break
        return out

    # ============ search_financial ============

    def search_financial(self, query: str, top_k: int = 5,
                         stock_code: Optional[str] = None) -> list[dict]:
        """检索财务指标(单公司财报 80+ 指标),可选 stock_code 精确过滤。"""
        import re
        from pymilvus import AnnSearchRequest, RRFRanker

        q_dense, q_sparse = self._embed(query)
        # collection 内部全是 source_type='financial',不需 source_type filter
        expr = f'stock_code == "{stock_code}"' if stock_code else None

        results = self.clients["financial"].hybrid_search(
            collection_name=self.collections["financial"],
            reqs=[
                AnnSearchRequest(data=[q_dense],  anns_field="dense",
                                 param={"metric_type": "COSINE"}, limit=30, expr=expr),
                AnnSearchRequest(data=[q_sparse], anns_field="sparse",
                                 param={"metric_type": "IP"},     limit=30, expr=expr),
            ],
            ranker=RRFRanker(k=60),
            output_fields=["text", "stock_code", "stock_name",
                           "date", "data_type", "metric_class"],
            limit=top_k * 4,
        )
        hits = [self._hit_to_chunk(h) for h in results[0]]

        # 年份 tie-break:query 含 "2024" 时,date 年份 == 2024 的 chunk 优先
        years = set(re.findall(r'(20\d{2})', query))
        if years:
            for h in hits:
                if (h.get('metadata', {}).get('date') or '')[:4] in years:
                    h['score'] = h.get('score', 0) + 0.01
            hits.sort(key=lambda h: -h.get('score', 0))
        return hits[:top_k]

    # ============ search_industry(字典快路径 + 向量 fallback) ============

    def search_industry(self, query: str, top_k: int = 5,
                        sim_threshold: float = 0.75) -> list[dict]:
        """字典快路径(255 别名 → 标准名 → query 取整个行业聚合 chunk),
        没命中再走 ANN fallback。"""
        # 阶段 1:字典精确匹配
        matched_industries = []
        for alias in sorted(self._alias_to_industry.keys(), key=len, reverse=True):
            if alias in query:
                industry = self._alias_to_industry[alias]
                if industry not in matched_industries:
                    matched_industries.append(industry)
                if len(matched_industries) >= top_k:
                    break

        if matched_industries:
            return self._fetch_industry_parents(matched_industries, via='dict')

        # 阶段 2:向量 fallback
        q_dense, _ = self._embed(query)
        results = self.clients["industry_alias"].search(
            collection_name=self.collections["industry_alias"],
            anns_field="dense",
            data=[q_dense],
            output_fields=["chunk_id", "text", "industry", "section_text",
                           "stock_codes", "company_count", "pdf_file"],
            limit=top_k * 3,
        )
        seen_industries = set()
        hits = []
        for h in results[0]:
            c = self._hit_to_chunk(h)
            sim = c['score']
            if sim < sim_threshold:
                continue
            ent = c['metadata']
            industry = ent.get('industry')
            if industry in seen_industries:
                continue
            seen_industries.add(industry)
            meta = {
                **ent,
                'source_type':   'industry',
                'match_via':     'vector_fallback',
                'sim':           sim,
            }
            hits.append({
                'text':     c['text'],
                'metadata': meta,
                'score':    sim,
            })
            if len(hits) >= top_k:
                break
        return hits

    def _fetch_industry_parents(self, industries: list, via: str) -> list[dict]:
        """字典命中后,直接 query 取该 industry 的整个聚合 chunk。"""
        hits = []
        for ind in industries:
            rows = self.clients["industry_alias"].query(
                collection_name=self.collections["industry_alias"],
                filter=f'industry == "{ind}"',
                output_fields=["chunk_id", "industry", "section_text",
                               "stock_codes", "company_count", "pdf_file", "text"],
                limit=1,
            )
            if not rows:
                continue
            r = rows[0]
            def _g(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)
            hits.append({
                'text': _g(r, 'section_text') or _g(r, 'text') or '',
                'metadata': {
                    'chunk_id':     _g(r, 'chunk_id'),
                    'source_type':  'industry',
                    'industry':      _g(r, 'industry'),
                    'stock_codes':   _g(r, 'stock_codes'),
                    'company_count': _g(r, 'company_count'),
                    'pdf_file':      _g(r, 'pdf_file'),
                    'match_via':     via,
                },
                'score': 1.0,
            })
        return hits

    # ============ 兼容老 API:search_report = meta + content 二路融合 ============

    def search_report(self, query: str, top_k: int = 5) -> list[dict]:
        """兼容 V2/V3 旧调用。新代码请改用 search_report_meta / search_report_content。"""
        meta_hits    = self.search_report_meta(query, top_k=top_k * 2)
        content_hits = self.search_report_content(query, top_k=top_k * 2)
        return self._rrf_merge_by_pdf(meta_hits, content_hits, k=60, top_k=top_k)

    # ============ RRF / table parent 拼接 ============

    @staticmethod
    def _rrf_merge_by_pdf(*lists, k: int = 60, top_k: int = 5) -> list[dict]:
        """按 pdf_file 聚合 RRF。同 PDF 多 chunk 共享 PDF-level 分。
        无 pdf_file 的 chunk fallback 到 chunk_id。"""
        def _key(c):
            m = c.get("metadata", {})
            return m.get("pdf_file") or m.get("chunk_id") or id(c)

        pdf_hits: dict = {}
        for lst in lists:
            for rank, c in enumerate(lst):
                pdf_hits.setdefault(_key(c), []).append((rank, c))

        scores = {}
        items = {}
        for key, hits in pdf_hits.items():
            hits_sorted = sorted(hits, key=lambda x: x[0])
            scores[key] = sum(1 / (k + r + 1) for r, _ in hits_sorted[:5])
            items[key] = [c for _, c in hits_sorted]

        ordered = sorted(scores.keys(), key=lambda x: -scores[x])
        out = []
        for key in ordered:
            for c in items[key]:
                c['score'] = scores[key]
                out.append(c)
                if len(out) >= top_k:
                    return out
        return out

    @staticmethod
    def _enrich_table(hits: list) -> list[dict]:
        """命中 table_narrative child → 通过 parent_id dedup + 拼 parent_md(整表 Markdown)给 LLM。
        实际上 milvus 只灌了 table_narrative(05 把 row_fact filter 了),每 parent_id 1 个 child,
        dedup 是 trivial 的;保留兼容是为了万一未来重启 row_fact。
        """
        final = []
        groups: OrderedDict = OrderedDict()
        for h in hits:
            m = h.get("metadata", {})
            pid = m.get("parent_id")
            method = m.get("chunk_method", "")
            if pid and method in ("table_narrative", "table_row_fact"):
                if pid not in groups:
                    groups[pid] = {"score": h.get("score", 0.0), "metadata": m,
                                   "child_texts": [h["text"]], "_insert_at": len(final)}
                    final.append(None)
                else:
                    groups[pid]["score"] = max(groups[pid]["score"], h.get("score", 0.0))
                    groups[pid]["child_texts"].append(h["text"])
            else:
                final.append(h)
        for pid, info in groups.items():
            m = info["metadata"]
            caption = m.get("table_caption") or ""
            footnote = m.get("table_footnote") or ""
            page = m.get("page_idx")
            parent_md = m.get("parent_md") or ""
            header_parts = ["表格"]
            if caption:
                header_parts.append(caption)
            if isinstance(page, int) and page >= 0:
                header_parts.append(f"第 {page + 1} 页")
            header = "【" + " · ".join(header_parts) + "】"
            parts = [header]
            if parent_md:
                parts.append(parent_md)
            if footnote:
                parts.append(f"【附注】{footnote}")
            n = len(info["child_texts"])
            safe = [t.replace("\n", " ").strip() for t in info["child_texts"]]
            if n == 1:
                parts.append(f"【检索命中】\n- {safe[0]}")
            else:
                parts.append(f"【检索命中 {n} 条事实】\n" + "\n".join(f"- {t}" for t in safe))
            final[info["_insert_at"]] = {
                "text": "\n\n".join(parts),
                "metadata": {**m, "chunk_method": "table_enriched", "n_child_hits": n},
                "score": info["score"],
            }
        return [h for h in final if h is not None]


# ============ CLI 快速冒烟测 ============

def _cli_smoke():
    """python hybrid_search.py -- 跑一组测试 query"""
    retriever = FinAgentRetriever()
    queries = [
        ("search_report_meta",    "贵州茅台 投资评级 目标价"),
        ("search_report_content", "宁德时代 海外业务"),
        ("search_financial",      "茅台 2024 年 ROE", {"stock_code": "600519"}),
        ("search_industry",       "白酒行业对比"),
        ("search_industry",       "酒类赛道"),
        ("search_report",         "平安银行 2025 三季报营收"),
    ]
    for entry in queries:
        method, q = entry[0], entry[1]
        kwargs = entry[2] if len(entry) > 2 else {}
        logger.info(f"\n══ {method}({q!r}, {kwargs}) ══")
        fn = getattr(retriever, method)
        results = fn(q, top_k=3, **kwargs)
        for i, r in enumerate(results, 1):
            t = r.get('text', '')[:100]
            score = r.get('score', 0)
            logger.info(f"  [{i}] score={score:.3f}  {t}")


if __name__ == '__main__':
    _cli_smoke()
