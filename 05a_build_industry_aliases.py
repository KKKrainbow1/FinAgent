"""
FinAgent Step 5a (V3): 构建 industry_alias entity

V3 search_industry 架构(详见 docs/Chunk_Alignment_20260420.md 13b):
  别名 = Child / industry_comparison = Parent,255 个别名各自作 entity,
  每条冗余存对应 industry 的 comparison 数据(section_text / stock_codes / company_count)。
  查询时 Milvus ANN + group_by_field='industry' 一次搞定。

输入:
  data/processed/all_chunks.jsonl  (过滤 source_type='industry' 的 30 条作 Parent 数据源)
  hybrid_search.py::FinAgentRetriever._INDUSTRY_MAP / _EXTRA_ALIASES  (别名字典)

输出:
  data/processed/industry_alias_entities.jsonl

下一步:
  python 05_build_index.py   # 自动读本文件一起灌进 Milvus

用法:
  python 05a_build_industry_aliases.py
"""
import importlib.util
import json
import logging
import re
from pathlib import Path

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
ALL_CHUNKS_PATH = ROOT / "data/processed/all_chunks.jsonl"
OUT_PATH = ROOT / "data/processed/industry_alias_entities.jsonl"
# hybrid_search.py 和 05a 总是在同一目录(Mac:finagent_repo/ 服务器:Finagent/),用 __file__.parent 定位
HYBRID_SEARCH_PATH = Path(__file__).resolve().parent / "hybrid_search.py"


# ============ 从 hybrid_search.py import 别名字典(单一真相) ============

def load_industry_dicts() -> tuple[dict, dict]:
    """
    从 hybrid_search.py 读取 _INDUSTRY_MAP + _EXTRA_ALIASES,避免两份 dict 不同步。
    hybrid_search.py 的 import 有少量副作用(class 定义 + 模块级 logger),但不会跑实际 IO。
    """
    spec = importlib.util.spec_from_file_location("hs", HYBRID_SEARCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    retriever = mod.FinAgentRetriever
    return retriever._INDUSTRY_MAP, retriever._EXTRA_ALIASES


def build_alias_to_industry(industry_map: dict, extra_aliases: dict) -> dict[str, str]:
    """复制 hybrid_search._build_industry_index 里的扁平化逻辑"""
    alias_to_industry = {}

    # Step 1: 标准大类名自身
    for standard_name in industry_map:
        alias_to_industry[standard_name] = standard_name

    # Step 2: 东财细分 → 大类映射
    for standard_name, sub_industries in industry_map.items():
        for sub in sub_industries:
            alias_to_industry[sub] = standard_name
            clean = sub.replace("Ⅱ", "").strip()
            if clean and clean != sub:
                alias_to_industry[clean] = standard_name

    # Step 3: 口语化别名(不覆盖已有的东财映射)
    for standard_name, aliases in extra_aliases.items():
        for alias in aliases:
            if alias not in alias_to_industry:
                alias_to_industry[alias] = standard_name

    return alias_to_industry


# ============ 加载 industry_comparison Parent ============

def load_industry_parents() -> dict[str, dict]:
    """
    从 all_chunks.jsonl 读 source_type='industry' 的 chunk
    返回 {industry_name: {text, stock_codes, company_count}}
    """
    if not ALL_CHUNKS_PATH.exists():
        logger.error(f"❌ {ALL_CHUNKS_PATH} 不存在,请先跑 04_build_chunks.py")
        return {}

    parents = {}
    with open(ALL_CHUNKS_PATH, encoding='utf-8') as f:
        for line in f:
            c = json.loads(line)
            m = c.get('metadata', {})
            if m.get('source_type') != 'industry':
                continue
            # text 首行 "{行业}行业对比(...)" 提取行业名
            match = re.match(r'(\S+?)行业对比', c.get('text', ''))
            if not match:
                continue
            industry_name = match.group(1)
            stock_codes = m.get('stock_codes', [])
            if isinstance(stock_codes, list):
                stock_codes = ','.join(stock_codes)
            parents[industry_name] = {
                'text': c['text'],
                'stock_codes': stock_codes,
                'company_count': m.get('company_count', 0),
            }
    return parents


# ============ 主流程 ============

def main():
    logger.info("=" * 60)
    logger.info("构建 industry_alias entities(V3)")
    logger.info("=" * 60)

    logger.info(f"[1/3] 加载别名字典(从 hybrid_search.py import)...")
    industry_map, extra_aliases = load_industry_dicts()
    alias_to_industry = build_alias_to_industry(industry_map, extra_aliases)
    logger.info(f"  共 {len(alias_to_industry)} 个别名,覆盖 {len(industry_map)} 个标准行业")

    logger.info(f"[2/3] 加载 industry_comparison Parent 数据...")
    parents = load_industry_parents()
    logger.info(f"  从 all_chunks.jsonl 找到 {len(parents)} 个 industry_comparison")

    # 产出 entity
    logger.info(f"[3/3] 产出 industry_alias entity...")
    entities = []
    unmatched_aliases = []
    matched_industries = set()

    for alias, industry in sorted(alias_to_industry.items()):
        if industry not in parents:
            unmatched_aliases.append((alias, industry))
            continue
        parent = parents[industry]
        entities.append({
            'text': alias,
            'metadata': {
                'chunk_id':      f"alias_{alias}",
                'source_type':   'industry_alias',
                'industry':      industry,              # ← group_by 锚点
                'section_text':  parent['text'],         # Parent 冗余
                'stock_codes':   parent['stock_codes'],
                'company_count': parent['company_count'],
            }
        })
        matched_industries.add(industry)

    logger.info(f"  产出: {len(entities)} 条 entity,覆盖 {len(matched_industries)} 个 industry")

    # 报告未匹配
    if unmatched_aliases:
        missing_industries = set(i for _, i in unmatched_aliases)
        logger.warning(f"⚠️  {len(unmatched_aliases)} 个别名未找到对应的 industry_comparison")
        logger.warning(f"   缺失 industry: {sorted(missing_industries)}")
        logger.warning(f"   → 可能原因:04_build_chunks.py 没生成该 industry 的对比 chunk(公司数 <2)")
        for a, i in unmatched_aliases[:5]:
            logger.warning(f"     例: {a!r} → 期望 industry={i!r}(没找到)")

    # 写入
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        for e in entities:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    logger.info(f"✅ 写入 {OUT_PATH}")
    logger.info(f"   下一步: python 05_build_index.py(会自动读这个文件)")


if __name__ == '__main__':
    main()
