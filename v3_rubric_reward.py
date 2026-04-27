"""
V3 Rubric-based Reward 函数(V5 修订版)

V5 关键改动:
  1. finance_concept 走 ground truth 路径(GPT-5.4 比对 student vs GT,不进 LLooM)
  2. GiGPO PRM/ORM 混合 reward(state lookback=3 工具序列前缀)
     - finance_concept 跳过 GiGPO,纯 ORM
  3. Rubric YAML 加载时跳过 review_status != 'keep' 的 rubric(支持人工 review 流程)
  4. 路径 A:trajectory-level weighting 模拟 step-level credit assignment

每条 trajectory reward 计算:
  if type == 'finance_concept':
    Step 0: 0 步硬约束(调工具了 → 0.1 大幅惩罚)
    Step 1: GPT-5.4 比对 student answer vs ground truth(0.0 / 0.5 / 1.0)
  else:
    Step 0: 格式硬约束(继承 V2)
    Step 1: 加载 type rubric,过滤 review_status='keep'
    Step 2: 拆 rule-judgable form / llm-judgable fact/reasoning
    Step 3: batch GPT-5.5 judge
    Step 4: 加权求和 → ORM
    Step 5: 硬约束扣分 + DAPO overlong

GiGPO(可选,USE_V3_GIGPO=1 启用):
  - 计算每 step 的 state(最近 3 步工具序列)
  - 同 state 组内 ORM advantage = PRM
  - final_reward = α × ORM + (1-α) × mean_PRM × Indicator(|state group| ≥ 4)
  - finance_concept 跳过 GiGPO

环境变量:
  - V3_REWARD_RUBRIC_DIR / V3_REWARD_JUDGE_MODEL / V3_REWARD_API_BASE / V3_REWARD_API_KEY
  - V3_QUESTIONS_PATH:RL 数据集路径(默认 data/grpo/questions/grpo_questions_v3.jsonl)
  - V3_REWARD_GT_MODEL:ground truth 比对模型(默认 gpt-4o)
  - USE_V3_GIGPO:1 启用 GiGPO(默认 0)
  - V3_GIGPO_ALPHA:ORM/PRM 混合权重(默认 0.5)
  - V3_GIGPO_LOOKBACK:state 工具序列长度(默认 3)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from grpo_plugin import (
    _apply_overlong_penalty,
    _call_quality_penalty,
    _completion_to_str,
    _estimate_token_count,
    _extract_final_answer,
    _extract_question_and_type,
    _is_format_valid,
    finagent_reward as v2_reward_fallback,
    FORMAT_INVALID_PENALTY,
    VALID_TOOLS,
)


logger = logging.getLogger(__name__)


# ============ 配置 ============

V3_RUBRIC_DIR = Path(os.environ.get(
    "V3_REWARD_RUBRIC_DIR",
    str(Path(__file__).resolve().parent.parent / "data/grpo/rubric_v3/final_rubrics"),
))
V3_QUESTIONS_PATH = Path(os.environ.get(
    "V3_QUESTIONS_PATH",
    str(Path(__file__).resolve().parent.parent / "data/grpo/questions/grpo_questions_v3.jsonl"),
))
V3_JUDGE_MODEL = os.environ.get("V3_REWARD_JUDGE_MODEL", "gpt-5.5")
V3_GT_MODEL = os.environ.get("V3_REWARD_GT_MODEL", "gpt-4o")
V3_API_BASE = os.environ.get(
    "V3_REWARD_API_BASE",
    os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
)
V3_API_KEY = os.environ.get(
    "V3_REWARD_API_KEY",
    os.environ.get("OPENAI_API_KEY", ""),
)
V3_MAX_WORKERS = int(os.environ.get("V3_REWARD_MAX_WORKERS", "16"))
V3_CACHE_SIZE = int(os.environ.get("V3_REWARD_CACHE_SIZE", "10000"))
V3_JUDGE_TEMPERATURE = 0.1
V3_JUDGE_MAX_TOKENS = 800
V3_JUDGE_RETRY = 2

# GiGPO 配置
USE_V3_GIGPO = os.environ.get("USE_V3_GIGPO", "0") == "1"
V3_GIGPO_ALPHA = float(os.environ.get("V3_GIGPO_ALPHA", "0.5"))
V3_GIGPO_LOOKBACK = int(os.environ.get("V3_GIGPO_LOOKBACK", "3"))
V3_GIGPO_MIN_GROUP_SIZE = int(os.environ.get("V3_GIGPO_MIN_GROUP_SIZE", "4"))

# finance_concept 0 步惩罚(软,不到 -1.0)
FINANCE_CONCEPT_TOOL_USE_PENALTY = 0.1


# ============ Rubric YAML 加载(全局缓存)============

_RUBRIC_BY_TYPE: dict[str, dict] | None = None
_RUBRIC_LOAD_LOCK = threading.Lock()


def _load_rubrics() -> dict[str, dict]:
    """加载所有 task_type 的 final rubric YAML。

    V5 修订:支持 review_status 过滤
      - review_status_overall == 'reviewed' 时,只加载 review_status='keep' 的 rubric
      - review_status_overall == 'pending' 时,全部加载(开发期)
    """
    global _RUBRIC_BY_TYPE
    if _RUBRIC_BY_TYPE is not None:
        return _RUBRIC_BY_TYPE
    with _RUBRIC_LOAD_LOCK:
        if _RUBRIC_BY_TYPE is not None:
            return _RUBRIC_BY_TYPE
        loaded: dict[str, dict] = {}
        if not V3_RUBRIC_DIR.exists():
            logger.warning(f"V3 rubric 目录不存在: {V3_RUBRIC_DIR}")
            _RUBRIC_BY_TYPE = {}
            return _RUBRIC_BY_TYPE

        for yaml_path in sorted(V3_RUBRIC_DIR.glob("*.yaml")):
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not data or "task_type" not in data:
                continue

            # 过滤 review_status
            rubrics = data.get("rubrics", [])
            review_overall = data.get("review_status_overall", "pending")
            if review_overall == "reviewed":
                rubrics = [r for r in rubrics if r.get("review_status") == "keep"]
                logger.info(
                    f"  [{data['task_type']}] reviewed,保留 {len(rubrics)}/{len(data.get('rubrics', []))}"
                    f" rubric (review_status=keep)"
                )
            data["rubrics"] = rubrics

            loaded[data["task_type"]] = data

        _RUBRIC_BY_TYPE = loaded
        logger.info(
            f"V3 rubric 加载: {len(loaded)} task_type "
            f"({list(loaded.keys())})"
        )
        return _RUBRIC_BY_TYPE


# ============ Ground Truth 加载(全局缓存)============

_GROUND_TRUTHS: dict[str, str] | None = None
_GROUND_TRUTHS_LOCK = threading.Lock()


def _load_ground_truths() -> dict[str, str]:
    """加载 grpo_questions_v3.jsonl 中 finance_concept 题的 ground_truth_answer
    返回 question 文本 → GT 字符串映射
    """
    global _GROUND_TRUTHS
    if _GROUND_TRUTHS is not None:
        return _GROUND_TRUTHS
    with _GROUND_TRUTHS_LOCK:
        if _GROUND_TRUTHS is not None:
            return _GROUND_TRUTHS
        gt_map: dict[str, str] = {}
        if not V3_QUESTIONS_PATH.exists():
            logger.warning(f"V3 questions 文件不存在: {V3_QUESTIONS_PATH}")
            _GROUND_TRUTHS = gt_map
            return _GROUND_TRUTHS

        with open(V3_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gt = rec.get("ground_truth_answer")
                if gt and rec.get("type") == "finance_concept":
                    gt_map[rec["question"].strip()] = gt

        _GROUND_TRUTHS = gt_map
        logger.info(f"V3 ground truths 加载: {len(gt_map)} 条 finance_concept GT")
        return _GROUND_TRUTHS


# ============ Trace 格式化与缓存 ============

def _format_trace_for_judge(env, completion_text: str) -> str:
    lines = []
    for i, step in enumerate(env.tool_steps, start=1):
        lines.append(f"[Step {i}]")
        thought = step.get("thought") or ""
        if thought:
            lines.append(f"Thought: {thought}")
        tool = step.get("tool") or step.get("tool_name")
        if tool:
            args = step.get("query") or step.get("tool_arguments") or ""
            lines.append(f"Action: {tool}({args})")
        obs = step.get("observation") or ""
        if obs:
            obs_short = obs[:1500] + ("..." if len(obs) > 1500 else "")
            lines.append(f"Observation: {obs_short}")
        lines.append("")
    final = _extract_final_answer(completion_text) or ""
    if final:
        lines.append("[Final Answer]")
        lines.append(final[:3000])
    return "\n".join(lines)


def _trace_hash(question: str, trace_text: str) -> str:
    h = hashlib.sha1()
    h.update(question.encode("utf-8"))
    h.update(b"|")
    h.update(trace_text.encode("utf-8"))
    return h.hexdigest()


# ============ Rule-judgable Form 类规则 ============

def _rule_judge(rubric: dict, env, completion_text: str) -> bool | None:
    """Form 类 rubric 的规则预判;不可判时返回 None 退到 LLM"""
    crit = rubric.get("criterion", "")
    tools_used = {s.get("tool") or s.get("tool_name") for s in env.tool_steps}

    for tool in VALID_TOOLS:
        if f"调用了 {tool}" in crit or f"调了 {tool}" in crit:
            return tool in tools_used
    if ("calculate" in crit.lower()) and ("是否" in crit or "调" in crit):
        return "calculate" in tools_used
    return None


# ============ GPT-5.5 Batch Judge(其他 type 路径)============

_judge_cache: dict[tuple[str, int], bool] = {}
_judge_cache_lock = threading.Lock()
_openai_client = None
_openai_client_lock = threading.Lock()


def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    with _openai_client_lock:
        if _openai_client is None:
            from openai import OpenAI
            _openai_client = OpenAI(base_url=V3_API_BASE, api_key=V3_API_KEY)
        return _openai_client


def _cache_get(trace_h: str, rubric_id: int) -> bool | None:
    with _judge_cache_lock:
        return _judge_cache.get((trace_h, rubric_id))


def _cache_set(trace_h: str, rubric_id: int, value: bool):
    with _judge_cache_lock:
        if len(_judge_cache) >= V3_CACHE_SIZE:
            for k in list(_judge_cache.keys())[: V3_CACHE_SIZE // 10]:
                del _judge_cache[k]
        _judge_cache[(trace_h, rubric_id)] = value


BATCH_JUDGE_SYSTEM_PROMPT = """你是金融分析任务的客观评判者。给你一条 Agent 轨迹和一组 0/1 二元 rubric,
请同时判断轨迹是否满足每条 rubric。

判定规则:
- 严格按 rubric 文字执行
- 不参考写作风格、长度、格式
- 不确定时倾向 false(防 false positive)

输出严格 JSON。"""


BATCH_JUDGE_USER_TEMPLATE = """## 任务
问题: {question}
问题类型: {task_type}

## Agent 轨迹
{trace_text}

## Rubric 列表(逐条判断 satisfied: true/false)
{rubric_block}

## 输出 JSON
{{
  "judgments": [
    {{"id": 1, "satisfied": true,  "rationale": "..."}},
    ...
  ]
}}
只输出 JSON。"""


def _format_rubric_block(rubrics_to_judge: list[dict]) -> str:
    return "\n".join(f"{r['id']}. [{r['type']}] {r['criterion']}" for r in rubrics_to_judge)


def _parse_judgments(text: str, expected_ids: list[int]) -> dict[int, bool]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON: {text[:200]}")
    parsed = json.loads(cleaned[start : end + 1])
    out: dict[int, bool] = {}
    for j in parsed.get("judgments", []):
        rid = int(j.get("id"))
        if rid in expected_ids:
            out[rid] = bool(j.get("satisfied", False))
    return out


def _llm_judge_one_trace(question, task_type, trace_text, rubrics_to_judge):
    if not rubrics_to_judge:
        return {}
    expected_ids = [r["id"] for r in rubrics_to_judge]
    user = BATCH_JUDGE_USER_TEMPLATE.format(
        question=question, task_type=task_type,
        trace_text=trace_text[:6000],
        rubric_block=_format_rubric_block(rubrics_to_judge),
    )
    messages = [
        {"role": "system", "content": BATCH_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    client = _get_openai_client()
    last_error = None
    for attempt in range(1, V3_JUDGE_RETRY + 2):
        try:
            response = client.chat.completions.create(
                model=V3_JUDGE_MODEL,
                messages=messages,
                temperature=V3_JUDGE_TEMPERATURE,
                max_tokens=V3_JUDGE_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or ""
            return _parse_judgments(text, expected_ids)
        except Exception as e:
            last_error = e
            time.sleep(0.8 * attempt)
    logger.warning(f"V3 LLM judge 失败 ({V3_JUDGE_RETRY+1} 次): {last_error}")
    return {rid: False for rid in expected_ids}


# ============ finance_concept Ground Truth 路径 ============

GT_COMPARE_SYSTEM_PROMPT = """你是金融知识题客观评判者。给定标准答案和学生答案,判断学生答案是否
正确表达了标准答案的核心概念。

评判标准(三档):
- 1.0 = 学生答案完全正确表达了核心概念,可以有表述差异但要点齐全
- 0.5 = 部分正确(核心对但有重要遗漏 / 部分表述错误 / 概念区分不清)
- 0.0 = 错误、偏题或泛谈

注意:
- 不参考写作风格、长度、格式美观
- 表述差异(用词不同但意思一致)不扣分
- 编造具体公司数据(标准答案没有的)扣分
- 偏题或答非所问 → 0.0

输出严格 JSON。"""


GT_COMPARE_USER_TEMPLATE = """## 问题
{question}

## 标准答案(参考)
{ground_truth}

## 学生答案
{student_answer}

## 输出 JSON
{{
  "score": 0.0 / 0.5 / 1.0,
  "reason": "一句话理由"
}}"""


def _gpt_compare_to_ground_truth(question: str, student: str, gt: str) -> float:
    """GPT 比对 student answer vs ground truth,返回 0.0 / 0.5 / 1.0"""
    user = GT_COMPARE_USER_TEMPLATE.format(
        question=question,
        ground_truth=gt[:2000],
        student_answer=student[:2000],
    )
    messages = [
        {"role": "system", "content": GT_COMPARE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    client = _get_openai_client()
    last_error = None
    for attempt in range(1, V3_JUDGE_RETRY + 2):
        try:
            response = client.chat.completions.create(
                model=V3_GT_MODEL,
                messages=messages,
                temperature=V3_JUDGE_TEMPERATURE,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or ""
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1:
                continue
            parsed = json.loads(cleaned[start : end + 1])
            score = float(parsed.get("score", 0.0))
            # 限定为 0.0 / 0.5 / 1.0
            if score not in (0.0, 0.5, 1.0):
                if score < 0.25:
                    score = 0.0
                elif score < 0.75:
                    score = 0.5
                else:
                    score = 1.0
            return score
        except Exception as e:
            last_error = e
            time.sleep(0.8 * attempt)
    logger.warning(f"GT compare 失败: {last_error},兜底 0.0")
    return 0.0


def _finance_concept_reward(env, completion_text, question) -> tuple[float, dict]:
    """finance_concept 专属 reward:0 步硬约束 + GT 比对"""
    # Step 0: 0 步硬约束
    if env.tool_steps:
        return FINANCE_CONCEPT_TOOL_USE_PENALTY, {
            "reason": "finance_concept_tool_used",
            "tool_steps_count": len(env.tool_steps),
        }

    # Step 1: 提取 student answer
    student = _extract_final_answer(completion_text) or ""
    if not student or len(student.strip()) < 30:
        return 0.0, {"reason": "answer_too_short", "answer_length": len(student)}

    # Step 2: 加载 ground truth
    gt_map = _load_ground_truths()
    gt = gt_map.get(question.strip())
    if not gt:
        # 没有 GT(可能 build_grpo_v3_questions.py 还没跑 --gen-ground-truth)
        # 兜底:中性分 0.5(避免 reward=0 误导训练)
        logger.warning(f"finance_concept 无 GT: {question[:30]}...,reward=0.5 中性兜底")
        return 0.5, {"reason": "missing_ground_truth"}

    # Step 3: GPT 比对
    score = _gpt_compare_to_ground_truth(question, student, gt)
    return score, {
        "reason": "finance_concept_gt_compare",
        "gt_score": score,
        "answer_length": len(student),
    }


# ============ 单 trajectory rubric reward 计算(其他 type)============

def _compute_v3_rubric_reward(
    env, completion_text, question, question_type, rubric_set,
) -> tuple[float, dict]:
    """计算单 trajectory 的 V3 rubric reward(rubric 路径,非 finance_concept)"""
    rubrics = rubric_set.get("rubrics", [])
    if not rubrics:
        return 0.0, {"error": "empty rubric"}

    trace_text = _format_trace_for_judge(env, completion_text)
    trace_h = _trace_hash(question, trace_text)

    rule_results: list[tuple[dict, bool]] = []
    llm_pending: list[dict] = []

    for rubric in rubrics:
        rid = rubric["id"]
        cached = _cache_get(trace_h, rid)
        if cached is not None:
            rule_results.append((rubric, cached))
            continue
        if rubric.get("judge_method") == "rule" or rubric.get("type") == "form":
            ruled = _rule_judge(rubric, env, completion_text)
            if ruled is not None:
                rule_results.append((rubric, ruled))
                _cache_set(trace_h, rid, ruled)
                continue
        llm_pending.append(rubric)

    if llm_pending:
        llm_results = _llm_judge_one_trace(question, question_type, trace_text, llm_pending)
        for rubric in llm_pending:
            rid = rubric["id"]
            satisfied = llm_results.get(rid, False)
            rule_results.append((rubric, satisfied))
            _cache_set(trace_h, rid, satisfied)

    total_weight = sum(r["weight"] for r, _ in rule_results)
    base_reward = sum(r["weight"] * (1.0 if sat else 0.0) for r, sat in rule_results)
    if total_weight > 0:
        base_reward = base_reward / total_weight

    debug_info = {
        "satisfactions": [
            {"id": r["id"], "type": r["type"], "weight": r["weight"], "satisfied": sat}
            for r, sat in rule_results
        ],
        "rule_count": sum(1 for r, _ in rule_results
                          if r.get("judge_method") == "rule" or r.get("type") == "form"),
        "llm_count": len(llm_pending),
    }
    return base_reward, debug_info


# ============ GiGPO state + PRM(Stage A + 路径 A) ============

def _compute_state_for_step(env, step_idx: int, lookback: int = V3_GIGPO_LOOKBACK) -> tuple | None:
    """state = 最近 lookback 步的工具序列前缀"""
    if step_idx == 0:
        return ("START",)
    window = env.tool_steps[max(0, step_idx - lookback + 1) : step_idx + 1]
    return tuple(s.get("tool") or s.get("tool_name") for s in window)


def _aggregate_gigpo_state_groups(
    batch_envs: list, batch_orms: list[float], batch_question_types: list[str],
) -> dict:
    """对 batch 内所有 (env_idx, step_idx) 按 state 分组,跳过 finance_concept"""
    state_groups: dict[tuple, list] = defaultdict(list)
    for env_idx, env in enumerate(batch_envs):
        if batch_question_types[env_idx] == "finance_concept":
            continue   # finance_concept 跳过 GiGPO
        orm = batch_orms[env_idx]
        for step_idx in range(len(env.tool_steps)):
            state = _compute_state_for_step(env, step_idx)
            if state is None:
                continue
            state_groups[state].append({
                "env_idx": env_idx,
                "step_idx": step_idx,
                "orm": orm,
            })
    return state_groups


def _compute_gigpo_prm_advantages(state_groups: dict) -> dict:
    """同 state 组内算 PRM advantage,Indicator(|state|≥4) 控制"""
    prm_advantages: dict[tuple[int, int], float] = {}
    for state, members in state_groups.items():
        if len(members) < V3_GIGPO_MIN_GROUP_SIZE:
            for m in members:
                prm_advantages[(m["env_idx"], m["step_idx"])] = 0.0
            continue
        rewards = np.array([m["orm"] for m in members], dtype=np.float64)
        mean = rewards.mean()
        std = max(rewards.std(), 0.01)
        for m in members:
            prm_advantages[(m["env_idx"], m["step_idx"])] = float(
                (m["orm"] - mean) / std
            )
    return prm_advantages


def _apply_gigpo_weighting(
    batch_envs: list, batch_orms: list[float], batch_question_types: list[str],
    alpha: float = V3_GIGPO_ALPHA,
) -> list[float]:
    """路径 A:trajectory-level weighting 模拟 step-level credit assignment

    final_reward_traj_i = alpha * orm_traj_i + (1-alpha) * mean_PRM_over_steps
        finance_concept 走 ORM-only(纯 orm)
    """
    if not USE_V3_GIGPO:
        return list(batch_orms)

    state_groups = _aggregate_gigpo_state_groups(batch_envs, batch_orms, batch_question_types)
    prm_advs = _compute_gigpo_prm_advantages(state_groups)

    # 每个 trajectory 求其内部 step 的平均 PRM
    n = len(batch_envs)
    final_rewards = list(batch_orms)
    for env_idx, env in enumerate(batch_envs):
        if batch_question_types[env_idx] == "finance_concept":
            continue   # 纯 ORM
        steps_n = len(env.tool_steps)
        if steps_n == 0:
            continue
        step_prms = [
            prm_advs.get((env_idx, step_idx), 0.0)
            for step_idx in range(steps_n)
        ]
        mean_prm = sum(step_prms) / max(len(step_prms), 1)
        # 路径 A:trajectory-level weighting
        final_rewards[env_idx] = alpha * batch_orms[env_idx] + (1 - alpha) * (
            batch_orms[env_idx] + mean_prm * 0.1   # PRM 作为微调,*0.1 防 advantage 过大
        )

    return final_rewards


# ============ 主 reward 函数 ============

def v3_rubric_reward(completions, environments=None, **kwargs) -> list[float]:
    """V3 Rubric-based Reward(GRPOTrainer reward_funcs 入口)

    路径分流:
      - finance_concept → ground truth 比对(GPT-4o)
      - 其他 type → rubric judge(GPT-5.5 batch judge)

    可选 GiGPO PRM/ORM 混合(USE_V3_GIGPO=1)
    """
    if environments is None:
        environments = kwargs.get("environments", kwargs.get("envs", []))

    completion_texts = [_completion_to_str(c) for c in completions]

    rubric_db = _load_rubrics()

    # 第一遍:计算每条 trajectory 的 ORM
    batch_orms = [0.0] * len(completion_texts)
    batch_question_types = [""] * len(completion_texts)
    debug_info_all: list[dict | None] = [None] * len(completion_texts)

    def _compute_one(idx: int, env, completion_text: str):
        question, question_type = _extract_question_and_type(idx, kwargs)

        # finance_concept 路径
        if question_type == "finance_concept":
            orm, debug = _finance_concept_reward(env, completion_text, question)
            debug["question_type"] = question_type
            debug["path"] = "finance_concept_gt"
            return idx, orm, question_type, debug

        # 其他 type:格式硬约束
        if not _is_format_valid(env, completion_text, question_type):
            return idx, FORMAT_INVALID_PENALTY, question_type, {
                "reason": "format_invalid", "path": "rubric"
            }

        # 加载 rubric;若缺失,fallback V2
        rubric_set = rubric_db.get(question_type)
        if rubric_set is None:
            v2_rewards = v2_reward_fallback(
                completions=[completion_text], environments=[env],
                question=[question], question_type=[question_type],
            )
            return idx, v2_rewards[0], question_type, {
                "reason": "v2_fallback", "missing_type": question_type,
                "path": "v2_fallback",
            }

        # rubric reward
        base_reward, debug = _compute_v3_rubric_reward(
            env, completion_text, question, question_type, rubric_set,
        )

        # 硬约束扣分 + DAPO overlong
        penalty = _call_quality_penalty(env)
        base_reward = max(base_reward + penalty, 0.0)
        token_est = _estimate_token_count(completion_text)
        final_orm = _apply_overlong_penalty(base_reward, token_est)

        debug.update({
            "penalty": penalty,
            "token_est": token_est,
            "question_type": question_type,
            "path": "rubric",
        })
        return idx, final_orm, question_type, debug

    with ThreadPoolExecutor(max_workers=V3_MAX_WORKERS) as ex:
        futures = {
            ex.submit(_compute_one, i, env, ct): i
            for i, (env, ct) in enumerate(zip(environments, completion_texts))
        }
        for fut in as_completed(futures):
            idx, orm, qtype, debug = fut.result()
            batch_orms[idx] = orm
            batch_question_types[idx] = qtype
            debug_info_all[idx] = debug

    # 第二遍(GiGPO):trajectory-level weighting(路径 A)
    if USE_V3_GIGPO:
        final_rewards = _apply_gigpo_weighting(
            list(environments), batch_orms, batch_question_types,
            alpha=V3_GIGPO_ALPHA,
        )
        for i, (orm, fr) in enumerate(zip(batch_orms, final_rewards)):
            if debug_info_all[i] is not None:
                debug_info_all[i]["orm"] = orm
                debug_info_all[i]["gigpo_final"] = fr
    else:
        final_rewards = list(batch_orms)

    _log_v3_metrics(debug_info_all, final_rewards)
    return final_rewards


# ============ V3 metrics ============

_v3_metrics: dict[str, list] = {
    "fact_satisfactions": [],
    "reasoning_satisfactions": [],
    "form_satisfactions": [],
    "finance_concept_count": [],
    "finance_concept_gt_score": [],
    "v2_fallback_count": [],
    "format_invalid_count": [],
    "rule_judge_count": [],
    "llm_judge_count": [],
}
_v3_metrics_lock = threading.Lock()


def _log_v3_metrics(debug_info_all, rewards):
    """收集 V3 维度指标"""
    with _v3_metrics_lock:
        for debug, reward in zip(debug_info_all, rewards):
            if not debug:
                continue
            path = debug.get("path", "")
            if debug.get("reason") == "format_invalid":
                _v3_metrics["format_invalid_count"].append(1)
                continue
            if path == "v2_fallback":
                _v3_metrics["v2_fallback_count"].append(1)
                continue
            if path == "finance_concept_gt":
                _v3_metrics["finance_concept_count"].append(1)
                gt_score = debug.get("gt_score")
                if gt_score is not None:
                    _v3_metrics["finance_concept_gt_score"].append(gt_score)
                continue
            sats = debug.get("satisfactions", [])
            for s in sats:
                key = f"{s['type']}_satisfactions"
                if key in _v3_metrics:
                    _v3_metrics[key].append(1.0 if s["satisfied"] else 0.0)
            _v3_metrics["rule_judge_count"].append(debug.get("rule_count", 0))
            _v3_metrics["llm_judge_count"].append(debug.get("llm_count", 0))


def get_and_reset_v3_metrics() -> dict[str, float]:
    """供 GRPOMetricsCallback 调用,返回 mean 后重置"""
    with _v3_metrics_lock:
        out = {}
        for k, vals in _v3_metrics.items():
            if vals:
                if k.endswith("_satisfactions") or k.endswith("_score"):
                    out[f"v3_rubric/{k}_mean"] = sum(vals) / len(vals)
                elif k.endswith("_count"):
                    out[f"v3_rubric/{k}_per_traj"] = sum(vals) / max(len(vals), 1)
            _v3_metrics[k] = []
        return out


# ============ Self-test ============

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    db = _load_rubrics()
    print(f"\nLoaded rubric task_types: {list(db.keys())}")
    for task_type, data in db.items():
        rubrics = data.get("rubrics", [])
        types = [r["type"] for r in rubrics]
        weights_sum = sum(r["weight"] for r in rubrics)
        print(
            f"  [{task_type}] total={len(rubrics)} "
            f"fact={types.count('fact')} reasoning={types.count('reasoning')} "
            f"form={types.count('form')} weights_sum={weights_sum:.4f}"
        )

    gts = _load_ground_truths()
    print(f"\nLoaded ground truths: {len(gts)} 条 finance_concept")
    if gts:
        sample_q = next(iter(gts))
        print(f"  样例 question: {sample_q[:60]}...")
        print(f"  样例 GT (前 100 字): {gts[sample_q][:100]}...")

    print(f"\nGiGPO 配置:")
    print(f"  USE_V3_GIGPO = {USE_V3_GIGPO}")
    print(f"  ALPHA = {V3_GIGPO_ALPHA}")
    print(f"  LOOKBACK = {V3_GIGPO_LOOKBACK}")
    print(f"  MIN_GROUP_SIZE = {V3_GIGPO_MIN_GROUP_SIZE}")
