"""
FinAgent Retriever (V3 · Milvus 版)

架构对照: docs/Finagent项目介绍.md 第 4 节 / docs/Chunk_Alignment_20260420.md 第 14 节

对外 API 保持不变(grpo_plugin / react_agent / tools / sft_data 等无需改):
    retriever = FinAgentRetriever()
    retriever.search_financial(query, top_k=5)  -> list[dict]
    retriever.search_industry(query, top_k=5)   -> list[dict]
    retriever.search_report(query, top_k=10)    -> list[dict]

每个 dict 格式:{text, metadata, score},和 V2 保持一致。

内部实现(全部从 FAISS/BM25 双轨 + 应用层 RRF + enrich_with_parent 迁移到 Milvus):
  - 所有 chunk 在一个 Milvus collection,scalar filter 路由 3 工具
  - dense (BGE 768) + sparse (BM25 via pymilvus) 两路 hybrid_search
  - RRFRanker 融合 dense/sparse 两路(search_report 元数据路仍用 WeightedRanker)
  - Parent-Child dedup 在应用层做(enrich_with_parent,Lite 不支持 group_by_field,
    且 NULL 字段分组陷阱,统一应用层处理更稳)
  - Parent 数据全部冗余在 Child metadata,1 次 search 拿完
  - search_industry 走 V3 Parent-Child 复用(255 别名 Child + industry Parent)

依赖:
  pip install pymilvus sentence-transformers

V1 FAISS 版本备份在 hybrid_search_legacy_faiss.py,不建议继续使用。
"""
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ============ 默认路径配置 ============
DEFAULT_COLLECTION = "finagent"
DEFAULT_DB_PATH = ROOT / "data/processed/milvus_finagent.db"
DEFAULT_BM25_PATH = ROOT / "data/processed/bm25_ef.pkl"
DEFAULT_ENCODER = "./models/bge-base-zh-v1.5"


class FinAgentRetriever:
    """
    V3 Milvus 版统一检索器。无参构造自动加载默认 DB + BM25 + BGE。

    三种部署:
        Lite (默认):    MilvusClient("./data/processed/milvus_finagent.db")
        Standalone:     MilvusClient(uri="http://localhost:19530") — 传 uri 参数
    """

    # ============ 行业别名字典(保留,给 05a + reward_knowledge_base 用) ============
    # ⚠️ 更新别名时注意:
    #   - 05a_build_industry_aliases.py 会 import 这两个 dict 构建 industry_alias entity
    #   - reward_knowledge_base.py 也引用

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
        collection: str = DEFAULT_COLLECTION,
        db_path: Optional[str] = None,
        uri: Optional[str] = None,
        bm25_path: Optional[str] = None,
        encoder_model: str = DEFAULT_ENCODER,
        enrich_parents: bool = True,
    ):
        from pymilvus import MilvusClient

        # Milvus 连接:uri 优先,否则用 Lite 文件
        if uri:
            self.client = MilvusClient(uri=uri)
            logger.info(f"Milvus 连接: {uri}")
        else:
            path = db_path or str(DEFAULT_DB_PATH)
            self.client = MilvusClient(path)
            logger.info(f"Milvus Lite 连接: {path}")

        self.collection = collection
        self.enrich_parents = enrich_parents

        # BGE dense encoder
        from sentence_transformers import SentenceTransformer
        logger.info(f"加载 BGE encoder: {encoder_model}")
        self.encoder = SentenceTransformer(encoder_model)

        # BM25 sparse(加载 05 时 fit 保存的模型)
        from pymilvus.model.sparse import BM25EmbeddingFunction
        from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
        self.bm25_ef = BM25EmbeddingFunction(build_default_analyzer(language="zh"))
        bm25p = bm25_path or str(DEFAULT_BM25_PATH)
        if Path(bm25p).exists():
            # pymilvus 2.5+ 是 load(),2.4.x 是 load_from_file(),兼容两种
            if hasattr(self.bm25_ef, 'load'):
                self.bm25_ef.load(bm25p)
            elif hasattr(self.bm25_ef, 'load_from_file'):
                self.bm25_ef.load_from_file(bm25p)
            else:
                import pickle
                with open(bm25p, 'rb') as f:
                    self.bm25_ef = pickle.load(f)
            logger.info(f"BM25 模型加载: {bm25p}")
        else:
            logger.warning(f"⚠️  BM25 模型 {bm25p} 不存在,sparse 检索不可用(只有 dense)")

        # 扁平 alias → industry map(快路径:字典命中优先,否则向量 fallback)
        self._alias_to_industry = self._flatten_aliases()
        logger.info(f"行业别名字典: {len(self._alias_to_industry)} 个别名")

        logger.info("✅ FinAgentRetriever (V3 Milvus) 就绪")

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
        dense = self.encoder.encode(
            [query], normalize_embeddings=True
        )[0].tolist()

        sparse_csr = self.bm25_ef.encode_queries([query])
        start, end = sparse_csr.indptr[0], sparse_csr.indptr[1]
        sparse = {
            int(k): float(v)
            for k, v in zip(sparse_csr.indices[start:end],
                            sparse_csr.data[start:end])
        }
        return dense, sparse

    @staticmethod
    def _hit_to_chunk(hit) -> dict:
        """
        Milvus hit → {text, metadata, score} 标准格式
        hit['id'] 是主键 chunk_id(不在 output_fields 里自动返回),手动放进 metadata 让 RRF 能 dedup
        """
        entity = dict(hit.get('entity', {}))
        text = entity.pop('text', '')
        # 主键 → metadata.chunk_id,避免 _external_rrf 用 id() 导致 dedup 失效
        pk = hit.get('id') or hit.get('pk')
        if pk is not None:
            entity['chunk_id'] = pk
        return {
            'text': text,
            'metadata': entity,
            'score': float(hit.get('distance', 0.0)),
        }

    # ============ search_financial ============

    def search_financial(self, query: str, top_k: int = 5,
                         stock_code: Optional[str] = None) -> list[dict]:
        """
        hybrid_search + RRFRanker,filter source_type == financial
        可选 stock_code 做精确过滤
        """
        from pymilvus import AnnSearchRequest, RRFRanker

        q_dense, q_sparse = self._embed(query)
        filter_expr = 'source_type == "financial"'
        if stock_code:
            filter_expr += f' and stock_code == "{stock_code}"'

        results = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=[
                AnnSearchRequest(data=[q_dense],  anns_field="dense",  limit=30),
                AnnSearchRequest(data=[q_sparse], anns_field="sparse", limit=30),
            ],
            ranker=RRFRanker(k=60),
            filter=filter_expr,
            output_fields=[
                "text", "source_type", "stock_code", "stock_name",
                "date", "chunk_method",
            ],
            limit=top_k,
        )
        return [self._hit_to_chunk(h) for h in results[0]]

    # ============ search_industry ============

    def search_industry(self, query: str, top_k: int = 5,
                        sim_threshold: float = 0.75) -> list[dict]:
        """
        V3 架构:字典快路径 + 向量 fallback,都走 Milvus 取 Parent 冗余数据

        两阶段:
          1) 字典精确匹配(主流 95% 查询):从 query 抓最长别名 → standard industry
             → client.query(filter=industry=='XX') 取任意一条 alias entity 拿 Parent
          2) 字典没中 → ANN 搜 255 个 industry_alias 向量(多拉 top_k*3 条,
             应用层按 industry dedup —— Lite 不支持 group_by_field)
             → 直接拿 Parent 冗余数据
        """
        # 阶段 1:字典匹配(保留,因为它是"0 延迟 + 100% 精确")
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

        # 阶段 2:向量 fallback(ANN,Milvus Lite 不支持 group_by_field,应用层 dedup)
        q_dense, _ = self._embed(query)
        results = self.client.search(
            collection_name=self.collection,
            anns_field="dense",
            data=[q_dense],
            filter='source_type == "industry_alias"',
            output_fields=["industry", "section_text", "stock_codes", "company_count"],
            limit=top_k * 3,    # 多拉一些,应用层按 industry dedup
        )
        seen_industries = set()
        hits = []
        for h in results[0]:
            sim = float(h.get('distance', 0))
            if sim < sim_threshold:
                continue
            ent = h.get('entity', {})
            industry = ent.get('industry')
            if industry in seen_industries:        # 应用层 dedup(代替 group_by_field)
                continue
            seen_industries.add(industry)
            hits.append({
                'text': ent.get('section_text', '') or '',
                'metadata': {
                    'source_type': 'industry',
                    'industry':      industry,
                    'stock_codes':   ent.get('stock_codes'),
                    'company_count': ent.get('company_count'),
                    'match_via':     'vector_fallback',
                    'sim':           sim,
                },
                'score': sim,
            })
            if len(hits) >= top_k:
                break
        return hits

    def _fetch_industry_parents(self, industries: list, via: str) -> list[dict]:
        """按标准 industry 名直接 query 取 Parent 冗余字段"""
        hits = []
        for ind in industries:
            rows = self.client.query(
                collection_name=self.collection,
                filter=f'source_type == "industry_alias" and industry == "{ind}"',
                output_fields=["industry", "section_text", "stock_codes", "company_count"],
                limit=1,
            )
            if not rows:
                continue
            r = rows[0]
            hits.append({
                'text': r.get('section_text', '') or '',
                'metadata': {
                    'source_type':  'industry',
                    'industry':      r.get('industry'),
                    'stock_codes':   r.get('stock_codes'),
                    'company_count': r.get('company_count'),
                    'match_via':     via,
                },
                'score': 1.0,   # 字典精确匹配满分
            })
        return hits

    # ============ search_report ============

    def search_report(self, query: str, top_k: int = 5) -> list[dict]:
        """
        两路 hybrid_search + 外部 RRF:
          元数据路 (source_type='report'):       WeightedRanker(0.4, 0.6) BM25 主
          正文路   (report_fulltext/tabular):    RRFRanker(k=60),多拉 top_k*6 后应用层 dedup
        然后外部 RRF 融合,enrich_with_parent 展开 section
        """
        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

        q_dense, q_sparse = self._embed(query)

        # 元数据路
        meta_hits = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=[
                AnnSearchRequest(data=[q_dense],  anns_field="dense",  limit=30),
                AnnSearchRequest(data=[q_sparse], anns_field="sparse", limit=30),
            ],
            ranker=WeightedRanker(0.4, 0.6),
            filter='source_type == "report"',
            output_fields=[
                "text", "source_type", "stock_code", "stock_name",
                "institution", "date", "rating", "report_title",
            ],
            limit=30,
        )

        # 正文路(prose Child + table Child)
        # 注意:不用 group_by_field —— Milvus Lite 不支持,且 prose Child 的 parent_id=NULL 会把
        # 所有 prose 合并到同一 NULL 组导致召回丢失。dedup 全在应用层 enrich_with_parent 做。
        body_hits = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=[
                AnnSearchRequest(data=[q_dense],  anns_field="dense",  limit=60),
                AnnSearchRequest(data=[q_sparse], anns_field="sparse", limit=60),
            ],
            ranker=RRFRanker(k=60),
            filter='source_type in ["report_fulltext", "report_tabular"]',
            output_fields=[
                "text", "source_type", "chunk_method",
                "stock_code", "stock_name", "institution", "date",
                "parent_id", "parent_html", "table_caption",
                "section_id", "section_title", "section_text",
                "page_idx",
            ],
            limit=60,   # 多拉,enrich_with_parent 合并后再截 top_k
        )

        meta_chunks = [self._hit_to_chunk(h) for h in meta_hits[0]]
        body_chunks = [self._hit_to_chunk(h) for h in body_hits[0]]

        merged = self._external_rrf(meta_chunks, body_chunks, k=60, top_k=top_k * 3)

        if self.enrich_parents:
            merged = self.enrich_with_parent(merged)

        return merged[:top_k]

    # ============ 外部 RRF 融合 + Parent 展开 ============

    @staticmethod
    def _external_rrf(list_a: list, list_b: list, k: int = 60,
                      top_k: int = 30) -> list[dict]:
        """两路 chunk 列表用 RRF 排序融合(对量纲免疫)"""
        scores = {}
        items = {}
        for rank, c in enumerate(list_a):
            key = c.get('metadata', {}).get('chunk_id') or id(c)
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            items.setdefault(key, c)
        for rank, c in enumerate(list_b):
            key = c.get('metadata', {}).get('chunk_id') or id(c)
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            items.setdefault(key, c)

        ordered = sorted(scores.keys(), key=lambda x: -scores[x])
        out = []
        for key in ordered[:top_k]:
            c = items[key]
            c['score'] = scores[key]
            out.append(c)
        return out

    @staticmethod
    def enrich_with_parent(hits: list) -> list:
        """
        命中 Child 展开 Parent 给 LLM(表格走 parent_id dedup,prose 走 section_id dedup)。
        所有 dedup 都在这里做(Milvus Lite 不支持 group_by_field,且 NULL 字段分组有陷阱,
        统一在应用层处理更稳定)。
        """
        final = []
        table_parents = OrderedDict()   # parent_id → {score, metadata, child_texts, _insert_at}
        prose_groups = OrderedDict()    # section_id → 同上

        for h in hits:
            m = h.get('metadata', {})
            method = m.get('chunk_method', '')
            pid = m.get('parent_id')
            sid = m.get('section_id')

            is_table_child = pid and method in ('table_narrative', 'table_row_fact')
            is_prose_child = sid and method == 'fixed_window'

            if is_table_child:
                if pid not in table_parents:
                    table_parents[pid] = {
                        'score': h.get('score', 0.0),
                        'metadata': m,
                        'child_texts': [h['text']],
                        '_insert_at': len(final),
                    }
                    final.append(None)
                else:
                    table_parents[pid]['score'] = max(table_parents[pid]['score'],
                                                      h.get('score', 0.0))
                    table_parents[pid]['child_texts'].append(h['text'])
            elif is_prose_child:
                if sid not in prose_groups:
                    prose_groups[sid] = {
                        'score': h.get('score', 0.0),
                        'metadata': m,
                        'child_texts': [h['text']],
                        '_insert_at': len(final),
                    }
                    final.append(None)
                else:
                    prose_groups[sid]['score'] = max(prose_groups[sid]['score'],
                                                     h.get('score', 0.0))
                    prose_groups[sid]['child_texts'].append(h['text'])
            else:
                # 元数据 / 财务 / 行业 等原样
                final.append(h)

        # 表格 Parent 合并
        for pid, info in table_parents.items():
            m = info['metadata']
            caption = m.get('table_caption') or ''
            page = m.get('page_idx')
            parent_html = m.get('parent_html') or ''

            header_parts = ['表格']
            if caption:
                header_parts.append(caption)
            if isinstance(page, int) and page >= 0:
                header_parts.append(f"第 {page + 1} 页")
            header = '【' + ' · '.join(header_parts) + '】'

            parts = [header]
            if parent_html:
                parts.append(parent_html)

            n = len(info['child_texts'])
            if n == 1:
                parts.append(f"【检索命中】\n- {info['child_texts'][0]}")
            else:
                parts.append(f"【检索命中 {n} 条事实】\n"
                             + "\n".join(f"- {t}" for t in info['child_texts']))

            final[info['_insert_at']] = {
                'text': "\n\n".join(parts),
                'metadata': {**m, 'chunk_method': 'table_enriched', 'n_child_hits': n},
                'score': info['score'],
            }

        # Prose Section 合并
        for sid, info in prose_groups.items():
            m = info['metadata']
            title = m.get('section_title') or ''
            section_text = m.get('section_text') or ''
            page = m.get('page_idx')

            header_parts = ['章节']
            if title:
                header_parts.append(title)
            if isinstance(page, int) and page >= 0:
                header_parts.append(f"第 {page + 1} 页")
            header = '【' + ' · '.join(header_parts) + '】'

            parts = [header]
            if section_text:
                parts.append(section_text)

            n = len(info['child_texts'])
            if n == 1:
                parts.append(f"【检索命中】\n- {info['child_texts'][0]}")
            else:
                parts.append(f"【本段检索命中 {n} 句】\n"
                             + "\n".join(f"- {t}" for t in info['child_texts']))

            final[info['_insert_at']] = {
                'text': "\n\n".join(parts),
                'metadata': {**m, 'chunk_method': 'prose_enriched', 'n_child_hits': n},
                'score': info['score'],
            }

        return [h for h in final if h is not None]


# ============ CLI 快速冒烟测 ============

def _cli_smoke():
    """python hybrid_search.py -- 跑一组测试 query"""
    retriever = FinAgentRetriever()
    queries = [
        ("search_financial", "茅台 2024 年 ROE",    {"stock_code": "600519"}),
        ("search_industry",  "白酒行业对比",          {}),
        ("search_industry",  "酒类赛道",              {}),   # 字典没中,走向量 fallback
        ("search_report",    "平安银行 2025 三季报营收", {}),
    ]
    for method, q, kwargs in queries:
        logger.info(f"\n══ {method}({q!r}) ══")
        fn = getattr(retriever, method)
        results = fn(q, top_k=3, **kwargs)
        for i, r in enumerate(results, 1):
            t = r.get('text', '')[:100]
            score = r.get('score', 0)
            logger.info(f"  [{i}] score={score:.3f}  {t}")


if __name__ == '__main__':
    _cli_smoke()
