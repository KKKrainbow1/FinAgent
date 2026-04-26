"""
V4 SFT Question 生成 Prompt 包

3 个 question 生成 prompt + 1 个 thought 改写 prompt:
  q_gen_generic    通用 ReAct 调工具类 question 生成(覆盖 ~444 条 skeleton)
  q_gen_concept    finance_concept C1-C8 纯知识 question 生成(覆盖 200 条)
  q_gen_boundary   边界 subtype(clarify/anomaly/multi_source/insuf)(覆盖 150 条)
  thought_rewrite  V3 存量 Thought 去模板化改写(覆盖 300 条)

使用:
    from finagent_repo.sft_prompts_v4 import q_gen_generic
    messages = q_gen_generic.render(skeleton)
    resp = client.chat.completions.create(model="qwen3-max", messages=messages, ...)
    question = resp.choices[0].message.content.strip()

路由规则(scripts/generate_v4_questions.py 里实现):
    prompt_bucket in {simple/medium/fq/comparison/industry/risk}_expand → q_gen_generic
    prompt_bucket in reject_time_boundary / reject_misc → q_gen_generic
    prompt_bucket in D1_deep_thinking / D2_structure / D3_fusion / D4_counterfactual → q_gen_generic
    prompt_bucket in fc_C1-C8 → q_gen_concept
    prompt_bucket in clarify_positive / clarify_negative → q_gen_boundary
    prompt_bucket in insuf_L1/L2/L3/L4 → q_gen_boundary
    prompt_bucket in anomaly_seed / multi_source_seed → q_gen_boundary
"""

from . import q_gen_generic, q_gen_concept, q_gen_boundary, thought_rewrite

# Bucket → 生成模块 路由表
BUCKET_TO_MODULE = {
    # GENERIC(ReAct 调工具类)
    "simple_expand": q_gen_generic,
    "medium_expand": q_gen_generic,
    "fq_expand": q_gen_generic,
    "comparison_expand": q_gen_generic,
    "industry_expand": q_gen_generic,
    "risk_expand": q_gen_generic,
    "reject_time_boundary": q_gen_generic,
    "reject_misc": q_gen_generic,
    "D1_deep_thinking": q_gen_generic,
    "D2_structure": q_gen_generic,
    "D3_fusion": q_gen_generic,
    "D4_counterfactual": q_gen_generic,
    # CONCEPT
    "fc_C1_concept": q_gen_concept,
    "fc_C2_formula": q_gen_concept,
    "fc_C3_methodology": q_gen_concept,
    "fc_C4_industry_norm": q_gen_concept,
    "fc_C5_distinguish": q_gen_concept,
    "fc_C6_pitfall": q_gen_concept,
    "fc_C7_three_statement": q_gen_concept,
    "fc_C8_calibration": q_gen_concept,
    # BOUNDARY
    "clarify_positive": q_gen_boundary,
    "clarify_negative": q_gen_boundary,
    "insuf_L1": q_gen_boundary,
    "insuf_L2": q_gen_boundary,
    "insuf_L3": q_gen_boundary,
    "insuf_L4": q_gen_boundary,
    "anomaly_seed": q_gen_boundary,
    "multi_source_seed": q_gen_boundary,
}


# Subtype → 模块 路由(P0 F12 修:subtype 优先级 > bucket)
# 当 skeleton 带特殊 subtype 时,必须强制路由到 boundary,避免漏拒答信号
# 同时收录短名(skeleton 实际产出)+ 全名(prompt 内部规范名)防漏
SUBTYPE_FORCE_BOUNDARY = {
    "clarify", "clarify_positive",
    "clarify_neg", "clarify_negative",   # Agent3 建议:补齐 clarify_neg 短名
    "insuf_L1", "insuf_L2", "insuf_L3", "insuf_L4",
    "anomaly", "anomaly_seed",
    "multi_source", "multi_source_seed",
}
SUBTYPE_FORCE_CONCEPT = {
    "C1_concept", "C2_formula", "C3_methodology", "C4_industry_norm",
    "C5_distinguish", "C6_pitfall", "C7_three_statement", "C8_calibration",
}


def route(skeleton):
    """
    根据 skeleton.subtype > prompt_bucket 优先级路由到对应模块,返回 (module, messages)

    路由优先级(P0 修):
      1. subtype ∈ SUBTYPE_FORCE_BOUNDARY → q_gen_boundary(避免 insuf_L1 等被漏到 generic)
      2. subtype ∈ SUBTYPE_FORCE_CONCEPT → q_gen_concept
      3. 否则按 prompt_bucket 查 BUCKET_TO_MODULE
    """
    subtype = skeleton.get("subtype")
    if subtype in SUBTYPE_FORCE_BOUNDARY:
        return q_gen_boundary, q_gen_boundary.render(skeleton)
    if subtype in SUBTYPE_FORCE_CONCEPT:
        return q_gen_concept, q_gen_concept.render(skeleton)

    bucket = skeleton.get("prompt_bucket")
    if bucket not in BUCKET_TO_MODULE:
        raise ValueError(f"Unknown bucket: {bucket}, and no subtype force-route matched")
    mod = BUCKET_TO_MODULE[bucket]
    return mod, mod.render(skeleton)
