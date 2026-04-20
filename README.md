# FinAgent - 金融研报 RAG Pipeline(V3 · Milvus)

基于沪深300的金融研报检索增强生成(RAG)系统,从数据采集到检索索引到 ReAct Agent 全流程。底层存储从 V1 FAISS + BM25 双轨迁移到 V3 **Milvus Lite 单 collection**。

## Pipeline 概览

```
01_fetch_raw_data.py              # 拉取沪深300研报元数据 + 财务指标
        ↓
02_download_pdfs.py               # 下载研报PDF文件,写 pdf_map.json
        ↓
03b_parse_marker.py   ─┐
03c_parse_mineru.py   ─┼─ PDF → 结构化(Marker text / MinerU content_list.json,三选一)
03a_parse_pdfplumber  ─┘   (pdfplumber 已下线,仅作历史对比)
        ↓
03d_clean_content_list.py         # (MinerU 专用)10 条规则清洗 content_list.json
        ↓
04_build_chunks.py                # 构建所有 chunks:报告 meta + Prose Parent-Child(fixed_window)+ 财务 + table parent records
        ↓
06_tabularize_fulltext.py         # (MinerU table parent → LLM 生成)narrative + row_fact Child
        ↓
05a_build_industry_aliases.py     # 255 条行业别名 entity(Parent-Child 第 3 次复用)
        ↓
05_build_index.py                 # 构建 Milvus collection(dense + sparse BM25 + scalar filter)
        ↓
(下游)tools.py / react_agent.py / grpo_plugin.py 消费 hybrid_search
```

## 数据源

| 数据 | 来源 | 说明 |
|------|------|------|
| 研报元数据 | 东方财富(akshare) | 标题、评级、机构、盈利预测、PDF URL |
| 财务指标 | 东方财富(akshare) | ROE、毛利率、资产负债率等 86 个字段 |
| 研报正文 | 研报 PDF | Marker 或 MinerU 解析,得到 text + table_body(HTML) |

## 检索设计(V3 Milvus)

- **存储**:单 Milvus Lite collection `finagent`(~305K entity,单文件 `.db`)
- **向量**:BGE-base-zh-v1.5(768 维)+ BM25EmbeddingFunction(pymilvus 内置,jieba 分词)
- **BGE query 端加 instruction prefix**(asymmetric retrieval,+1-2% 召回)
- **索引**:Lite 默认 dense FLAT + sparse SPARSE_INVERTED_INDEX;Standalone 可切 HNSW
- **混合检索**:`client.hybrid_search` + `RRFRanker(k=60)` / `WeightedRanker(0.4, 0.6)` 两选一
- **3 工具路由**:`scalar filter source_type ==` 路由 `search_financial` / `search_industry` / `search_report`
- **Parent-Child 4x 复用**:table / prose / industry_alias / research_report_by_pdf_file
- **search_report budget**:12K 字符预算截取,不硬截 Parent

## 快速开始

### 环境依赖

```bash
# 基础
pip install akshare pandas tqdm requests

# 向量 + 检索
pip install pymilvus sentence-transformers jieba

# PDF 解析
pip install marker-pdf       # Marker
pip install mineru           # MinerU(可选,中文 SOTA)
```

### 运行

```bash
# 1. 拉原始数据
python 01_fetch_raw_data.py

# 2. 下载 PDF(写 data/raw/report_pdfs/pdf_map.json)
python 02_download_pdfs.py

# 3. 解析 PDF(当前生产用 MinerU + 03d 清洗)
python 03c_parse_mineru.py
python 03d_clean_content_list.py

# 4. 构建 chunks
python 04_build_chunks.py --parser mineru_cleaned

# 5. 表格 Parent → Child(调 qwen3-max 做 narrative + row_fact 生成)
python 06_tabularize_fulltext.py --source mineru_cleaned

# 6. 行业别名 entity
python 05a_build_industry_aliases.py

# 7. 建 Milvus 索引
python 05_build_index.py              # Lite,本地开发
# python 05_build_index.py --hnsw     # Standalone + HNSW
# python 05_build_index.py --uri http://localhost:19530   # 连 Standalone
```

## 输出目录结构

```
data/
├── raw/
│   ├── financial/all_financial.json            # 沪深300 财务指标
│   ├── reports/all_reports.csv                 # 研报元数据
│   ├── report_pdfs/pdf_map.json                # filename → row meta 映射
│   └── report_parsed/                          # PDF 解析结果(统一格式)
│       ├── marker_all_results.json             # Marker 产物
│       └── mineru_200_results.json             # MinerU 产物
├── processed/
│   ├── all_chunks.jsonl                        # 所有进索引的 chunk
│   ├── table_parents.jsonl                     # 04 产出的表格 Parent record(不进索引,06 消费)
│   ├── tabular_chunks_mineru.jsonl             # 06 产出的 narrative + row_fact Child
│   ├── industry_alias_entities.jsonl           # 05a 产出的 255 条别名 entity
│   ├── milvus_finagent.db                      # ★ Milvus Lite 单文件索引
│   └── bm25_ef.pkl                             # BM25EmbeddingFunction pickle(query 端复用)
└── ...
```

## PDF 解析器选型(历史)

| 解析器 | 文本准确率 | 表格准确率 | 速度 | 当前状态 |
|--------|-----------|-----------|------|---------|
| pdfplumber | 90%+ | ~70% | 快 | ❌ 已下线(表格保真度不够) |
| Marker | 85%+ | 85%+ | 慢(需 GPU) | ✅ 生产备选 |
| **MinerU** | 85%+ | 85%+ | 慢(需 GPU) | ✅ **当前生产**(+ 03d 清洗产结构化 content_list) |

**选型结论**:MinerU 输出结构化 `content_list.json`(分 text / table / image / chart blocks + text_level),便于 03d 做 10 条规则清洗 + 04 做 Parent-Child 切分。Marker 只给 text,少了元数据维度。

## 文档索引

- `docs/Finagent项目介绍.md` — 数据 / Chunk / 召回 完整设计文档(面试主交付)
- `docs/Chunk_Alignment_20260420.md` — chunk 结构对齐 + 样本 + schema 细节
- `docs/下一步优化路径_V2.md` — SFT V4 / GRPO V3 roadmap
- `docs/daily_reports/` — 每日工作报告
