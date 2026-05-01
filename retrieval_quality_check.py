"""
FinAgent SFT V4 — Retrieval Grounding Check

============================================================================
 设计目标
============================================================================

V4 question 池每条已带 (stock_code, stock_name, industry, period, metric_tag)
ground truth 字段(question 生成阶段编入)。每个 search_* 工具调用返回的
chunk meta 也带 (stock_code, source_type, pdf_file, date)。

本模块在 candidate 落库前 0 成本比对两侧元数据,过滤掉"召回明显错"的
trajectory:
  - 单公司类问题(financial_query / single_company_*)→ 必须召回到目标 stock_code
  - 行业类(industry_analysis)→ 至少 1 chunk 的 industry 字段匹配
  - 公司对比(company_comparison)→ 跨公司放宽,不强制(downstream Judge 兜)
  - 概念/拒答(finance_concept / reject)→ 0 步直答,跳过

跟规则 4(check_observation_match)的区别:
  - 规则 4 看 obs **文本**是否含目标指标关键词(粗粒度)
  - 本模块看 chunk **元数据**是否匹配 question metadata(细粒度,免误判)

============================================================================
 接口
============================================================================

check_retrieval_grounding(plan, question_meta) → (passed: bool, issues: list[str])

  passed:
    True  - 召回符合 ground truth 期望
    False - 召回错误(单公司类未命中目标公司 / 行业类未命中目标行业)
  issues:
    [str, ...] 失败原因(passed=True 时为空)
"""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)


# 不强制召回精确公司的 type(跨公司或 0 步直答)
LENIENT_TYPES = frozenset({
    "company_comparison",   # 跨公司,可能召回多家,不限定哪家
    "finance_concept",      # 0 步直答金融概念,无 retrieval
    "reject",               # 拒答类,0-2 步,可能 search 失败也合法
})

# 行业类:不查 stock_code,查 industry
INDUSTRY_TYPES = frozenset({
    "industry_analysis",
})


def _collect_retrieved_chunks(plan: dict) -> list[dict]:
    """收集 plan 中所有 step 的 retrieved_chunks(扁平化)。"""
    chunks: list[dict] = []
    for step in plan.get("steps", []):
        rc = step.get("retrieved_chunks")
        if rc:
            chunks.extend(rc)
    return chunks


def _chunk_matches_stock(chunk: dict, expected_code: str) -> bool:
    """单 chunk 是否包含目标公司。

    支持两种 schema:
      - financial / report chunk: stock_code 单值
      - industry chunk: stock_codes list(行业聚合)
    """
    if chunk.get("stock_code") == expected_code:
        return True
    codes_list = chunk.get("stock_codes") or []
    if isinstance(codes_list, str):
        codes_list = [c.strip() for c in codes_list.split(",") if c.strip()]
    return expected_code in codes_list


def check_retrieval_grounding(plan: dict, question_meta: dict) -> tuple[bool, list[str]]:
    """V4 retrieval grounding check 主入口。

    Args:
        plan: generate_trajectory 产出的 plan dict(含 steps[*].retrieved_chunks)
        question_meta: question 池中的元数据(stock_code/industry/period/...)

    Returns:
        (passed, issues)
    """
    qtype = question_meta.get("type", "")
    expected_code = question_meta.get("stock_code")
    expected_industry = question_meta.get("industry")

    # 跨公司类 / 0 步直答类 跳过(downstream Judge 兜)
    if qtype in LENIENT_TYPES:
        return True, []

    chunks = _collect_retrieved_chunks(plan)

    # 没有任何 retrieved_chunks(可能是 calculate-only 路径或解析失败)→ 不在这一关拒
    if not chunks:
        return True, []

    # 行业类:industry 字段必须匹配
    if qtype in INDUSTRY_TYPES:
        if not expected_industry:
            return True, []
        matched = any(c.get("industry") == expected_industry for c in chunks)
        if not matched:
            seen = sorted({c.get("industry") for c in chunks if c.get("industry")})
            return False, [
                f"industry_mismatch: 问 {expected_industry},召回 {seen}"
            ]
        return True, []

    # 单公司类:必须命中目标 stock_code
    if not expected_code:
        return True, []      # question 池没编 stock_code 则跳过

    if any(_chunk_matches_stock(c, expected_code) for c in chunks):
        return True, []

    # 收集召回到的公司,作 issue 信息
    seen_codes: set[str] = set()
    for c in chunks:
        sc = c.get("stock_code")
        if sc:
            seen_codes.add(sc)
        for s in (c.get("stock_codes") or []):
            if isinstance(s, str) and s:
                seen_codes.add(s)
    return False, [
        f"stock_mismatch: 问 {expected_code}({question_meta.get('stock_name', '')}),"
        f"召回 stock_code 集合 {sorted(seen_codes)[:5]}"
    ]


def _self_test():
    """轻量自测,无 LLM / 无 retriever 依赖。"""
    # 单公司命中
    plan = {"steps": [
        {"action": "search_financial", "retrieved_chunks": [
            {"chunk_id": "x", "stock_code": "600519"},
        ]}
    ]}
    qm = {"type": "single_company_simple", "stock_code": "600519"}
    assert check_retrieval_grounding(plan, qm) == (True, [])

    # 单公司未命中
    plan = {"steps": [
        {"action": "search_financial", "retrieved_chunks": [
            {"chunk_id": "x", "stock_code": "600276"},   # 召回了恒瑞医药
        ]}
    ]}
    qm = {"type": "single_company_simple", "stock_code": "600519",
          "stock_name": "贵州茅台"}
    passed, issues = check_retrieval_grounding(plan, qm)
    assert not passed and "stock_mismatch" in issues[0]

    # 行业命中
    plan = {"steps": [
        {"action": "search_industry", "retrieved_chunks": [
            {"chunk_id": "x", "industry": "白酒", "stock_codes": ["600519", "000568"]},
        ]}
    ]}
    qm = {"type": "industry_analysis", "industry": "白酒"}
    assert check_retrieval_grounding(plan, qm) == (True, [])

    # 行业未命中
    plan = {"steps": [
        {"action": "search_industry", "retrieved_chunks": [
            {"chunk_id": "x", "industry": "医药"},
        ]}
    ]}
    qm = {"type": "industry_analysis", "industry": "白酒"}
    passed, issues = check_retrieval_grounding(plan, qm)
    assert not passed and "industry_mismatch" in issues[0]

    # comparison 类放宽
    plan = {"steps": [
        {"action": "search_financial", "retrieved_chunks": [
            {"chunk_id": "x", "stock_code": "600276"},
        ]}
    ]}
    qm = {"type": "company_comparison", "stock_code": "600519"}
    assert check_retrieval_grounding(plan, qm) == (True, [])

    # finance_concept 0 步直答放宽
    plan = {"steps": []}
    qm = {"type": "finance_concept"}
    assert check_retrieval_grounding(plan, qm) == (True, [])

    # industry chunk stock_codes list 单公司命中
    plan = {"steps": [
        {"action": "search_industry", "retrieved_chunks": [
            {"chunk_id": "x", "industry": "白酒",
             "stock_codes": ["600519", "000568", "000858"]},
        ]}
    ]}
    qm = {"type": "single_company_simple", "stock_code": "600519"}
    assert check_retrieval_grounding(plan, qm) == (True, [])

    print("✅ retrieval_quality_check self-test all passed")


if __name__ == "__main__":
    _self_test()
