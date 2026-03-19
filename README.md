# FinAgent - 金融研报 RAG Pipeline

基于沪深300的金融研报检索增强生成(RAG)系统，完整覆盖从数据采集到检索索引构建的全流程。

## Pipeline 概览

```
01_fetch_raw_data.py        # 拉取沪深300研报元数据 + 财务指标
        ↓
02_download_pdfs.py         # 下载研报PDF文件
        ↓
03a_parse_pdfplumber.py  ─┐
03b_parse_marker.py      ─┼─ PDF → 纯文本（统一输出格式，三选一）
03c_parse_mineru.py      ─┘
        ↓
04_build_chunks.py          # 统一构建所有chunks（研报元数据 + PDF正文切分 + 财务数据）
        ↓
05_build_index.py           # 构建 FAISS向量索引 + BM25稀疏索引
```

## 数据源

| 数据 | 来源 | 说明 |
|------|------|------|
| 研报元数据 | 东方财富（akshare） | 标题、评级、机构、盈利预测 |
| 财务指标 | 东方财富（akshare） | ROE、毛利率、资产负债率等86个字段 |
| 研报正文 | 研报PDF | 通过 pdfplumber/Marker/MinerU 解析 |

## 检索设计

- **向量检索**: BGE-base-zh-v1.5 编码 + FAISS IndexFlatIP
- **稀疏检索**: jieba 分词 + BM25Okapi
- **混合检索**: 加权融合（默认 alpha=0.6 向量 + 0.4 BM25）

## 快速开始

### 环境依赖

```bash
# 基础依赖
pip install akshare pandas tqdm requests pdfplumber

# 索引构建
pip install sentence-transformers faiss-gpu rank_bm25 jieba

# PDF解析（可选，按需安装）
pip install marker-pdf    # Marker
pip install mineru        # MinerU
```

### 运行

```bash
# 1. 拉取原始数据
python 01_fetch_raw_data.py

# 2. 下载研报PDF
python 02_download_pdfs.py

# 3. 解析PDF（三选一）
python 03a_parse_pdfplumber.py
# python 03b_parse_marker.py
# python 03c_parse_mineru.py

# 4. 构建chunks（通过 --parser 指定使用哪个解析器的结果）
python 04_build_chunks.py --parser pdfplumber
# python 04_build_chunks.py --parser marker
# python 04_build_chunks.py --parser mineru

# 5. 构建索引
python 05_build_index.py

# 测试检索
python 05_build_index.py --test_query "茅台盈利能力"
```

## 输出目录结构

```
data/
├── raw/
│   ├── financial/all_financial.json          # 沪深300财务指标
│   ├── reports/all_reports.csv               # 研报元数据
│   └── report_parsed/                        # PDF解析结果（统一格式）
│       ├── pdfplumber_all_results.json
│       ├── marker_all_results.json
│       └── mineru_200_results.json
├── processed/
│   ├── all_chunks.jsonl                      # 全部chunks
│   └── chunk_stats.json                      # 统计信息
└── index/
    ├── faiss_index.bin                       # FAISS向量索引
    ├── bm25_index.pkl                        # BM25稀疏索引
    └── chunk_metadata.pkl                    # chunk元数据
```

## PDF解析器对比

| 解析器 | 文本准确率 | 表格准确率 | 速度 | 适用场景 |
|--------|-----------|-----------|------|---------|
| pdfplumber | 90%+ | ~70% | 快 | 大批量处理 |
| Marker | 85%+ | 85%+ | 慢（需GPU） | 高质量解析 |
| MinerU | 85%+ | 85%+ | 慢（需GPU） | 高质量解析 |

## 解析器输出格式

三个解析器输出统一的 JSON 格式，chunk 切分统一在 04_build_chunks.py 中完成：

```json
[
  {"file": "000001_20250101_1.pdf", "text": "解析后的全文...", "time": 1.23},
  ...
]
```
