"""
V3 Rubric-based Reward 函数

设计:per-task_type rubric(7 套,从 LLooM pipeline 生成)+ GPT-5.5 batch judge
+ rule-judgable form 类规则预过滤 + 缓存。

每条 trajectory reward 计算:
  1. 格式硬约束(复用 V2 的 _is_format_valid / _call_quality_penalty / _apply_overlong_penalty)
  2. 加载 question_type 对应 rubric set(若缺失,fallback 到 V2 reward)
  3. 拆 rule-judgable form 类(查 env.tool_steps)和 llm-judgable fact/reasoning 类
  4. batch GPT-5.5 judge:同一 batch 内所有 trace × LLM rubric 合并并发(8 rubric 合并 1 prompt)
  5. 加权求和 + 硬约束扣分 + DAPO overlong

依赖:
  - finagent_repo/grpo_plugin.py(复用 _is_format_valid 等辅助)
  - data/grpo/rubric_v3/final_rubrics/{task_type}.yaml(7 套 rubric,LLooM 产出)
  - GPT-5.5 API(judge fact/reasoning 类 rubric)

环境变量:
  - V3_REWARD_RUBRIC_DIR:rubric YAML 目录(默认 data/grpo/rubric_v3/final_rubrics)
  - V3_REWARD_JUDGE_MODEL:judge 模型名(默认 gpt-5.5)
  - V3_REWARD_API_BASE / V3_REWARD_API_KEY:OpenAI 兼容 API
  - V3_REWARD_MAX_WORKERS:GPT-5.5 调用并发(默认 16)
  - V3_REWARD_CACHE_SIZE:trace_hash × rubric_id 缓存上限(默认 10000)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

# 复用 V2 的硬约束 + 辅助函数
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
V3_JUDGE_MODEL = os.environ.get("V3_REWARD_JUDGE_MODEL", "gpt-5.5")
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


# ============ Rubric 加载(全局缓存,启动一次)============

_RUBRIC_BY_TYPE: dict[str, dict] | None = None
_RUBRIC_LOAD_LOCK = threading.Lock()


def _load_rubrics() -> dict[str, dict]:
    """加载所有 task_type 的 final rubric YAML 到全局缓存"""
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
            loaded[data["task_type"]] = data

        _RUBRIC_BY_TYPE = loaded
        logger.info(
            f"V3 rubric 加载: {len(loaded)} task_type "
            f"({list(loaded.keys())})"
        )
        return _RUBRIC_BY_TYPE


# ============ Trace 格式化与缓存 key ============

def _format_trace_for_judge(env, completion_text: str) -> str:
    """把 trajectory 序列化成可读文本(供 GPT-5.5 judge)"""
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
    """缓存 key 基础"""
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

    # 简单匹配:criterion 中明确提"调用了 X"
    for tool in VALID_TOOLS:
        if f"调用了 {tool}" in crit or f"调了 {tool}" in crit:
            return tool in tools_used

    # criterion 提"是否包含 calculate"
    if ("calculate" in crit.lower()) and ("是否" in crit or "调" in crit):
        return "calculate" in tools_used

    # 不可由规则判断
    return None


# ============ GPT-5.5 Batch Judge(8 rubric 合并 1 prompt)============

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
            # 简单淘汰:删除最早的 10%
            for k in list(_judge_cache.keys())[: V3_CACHE_SIZE // 10]:
                del _judge_cache[k]
        _judge_cache[(trace_h, rubric_id)] = value


BATCH_JUDGE_SYSTEM_PROMPT = """你是金融分析任务的客观评判者。给你一条 Agent 轨迹和一组 0/1 二元 rubric,
请同时判断轨迹是否满足每条 rubric。

判定规则:
- 严格按 rubric 文字执行(数字溯源就核对数字,工具调用就检查 tool_steps)
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
    {{"id": 2, "satisfied": false, "rationale": "..."}},
    ...
  ]
}}

只输出 JSON。"""


def _format_rubric_block(rubrics_to_judge: list[dict]) -> str:
    lines = []
    for r in rubrics_to_judge:
        lines.append(f"{r['id']}. [{r['type']}] {r['criterion']}")
    return "\n".join(lines)


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
    judgments = parsed.get("judgments", [])
    out: dict[int, bool] = {}
    for j in judgments:
        rid = int(j.get("id"))
        if rid in expected_ids:
            out[rid] = bool(j.get("satisfied", False))
    return out


def _llm_judge_one_trace(
    question: str,
    task_type: str,
    trace_text: str,
    rubrics_to_judge: list[dict],
) -> dict[int, bool]:
    """一条 trace × N 条 rubric 合并成 1 次 LLM 调用"""
    if not rubrics_to_judge:
        return {}

    expected_ids = [r["id"] for r in rubrics_to_judge]
    user = BATCH_JUDGE_USER_TEMPLATE.format(
        question=question,
        task_type=task_type,
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
    # 失败兜底:全部 False(保守)
    return {rid: False for rid in expected_ids}


# ============ 单 trajectory reward 计算 ============

def _compute_v3_single_reward(
    env,
    completion_text: str,
    question: str,
    question_type: str,
    rubric_set: dict,
) -> tuple[float, dict]:
    """计算单 trajectory 的 V3 rubric reward,返回 (reward, debug_info)"""
    rubrics = rubric_set.get("rubrics", [])
    if not rubrics:
        return 0.0, {"error": "empty rubric"}

    trace_text = _format_trace_for_judge(env, completion_text)
    trace_h = _trace_hash(question, trace_text)

    rule_results: list[tuple[dict, bool]] = []
    llm_pending: list[dict] = []

    for rubric in rubrics:
        rid = rubric["id"]

        # 1. 缓存命中
        cached = _cache_get(trace_h, rid)
        if cached is not None:
            rule_results.append((rubric, cached))
            continue

        # 2. Rule-judgable(form 类常见模式)
        if rubric.get("judge_method") == "rule" or rubric.get("type") == "form":
            ruled = _rule_judge(rubric, env, completion_text)
            if ruled is not None:
                rule_results.append((rubric, ruled))
                _cache_set(trace_h, rid, ruled)
                continue

        # 3. 留给 LLM
        llm_pending.append(rubric)

    # 4. 批量 LLM judge(同一 trace 内 N 条 rubric 合并 1 次调用)
    if llm_pending:
        llm_results = _llm_judge_one_trace(
            question, question_type, trace_text, llm_pending,
        )
        for rubric in llm_pending:
            rid = rubric["id"]
            satisfied = llm_results.get(rid, False)
            rule_results.append((rubric, satisfied))
            _cache_set(trace_h, rid, satisfied)

    # 5. 加权求和
    total_weight = sum(r["weight"] for r, _ in rule_results)
    base_reward = sum(
        r["weight"] * (1.0 if sat else 0.0)
        for r, sat in rule_results
    )
    if total_weight > 0:
        base_reward = base_reward / total_weight   # 归一化(防 weight 总和不为 1)

    debug_info = {
        "satisfactions": [
            {
                "id": r["id"],
                "type": r["type"],
                "weight": r["weight"],
                "satisfied": sat,
            }
            for r, sat in rule_results
        ],
        "rule_count": sum(1 for r, _ in rule_results if r.get("judge_method") == "rule" or r.get("type") == "form"),
        "llm_count": len(llm_pending),
    }
    return base_reward, debug_info


# ============ 主 reward 函数(GRPOTrainer 入口)============

def v3_rubric_reward(completions, environments=None, **kwargs) -> list[float]:
    """V3 Rubric-based Reward(GRPOTrainer reward_funcs 入口)

    完整流程:
      Step 0: 加载 rubrics(全局缓存)
      Step 1: 对每条 (env, completion):
        - 格式硬约束(继承 V2)
        - 加载 question_type 对应 rubric;若缺失,fallback v2_reward
        - 单 trajectory reward = Σ rubric_i × weight_i / Σ weight_i
        - 硬约束扣分(继承 V2)
        - DAPO overlong(继承 V2)
    """
    if environments is None:
        environments = kwargs.get("environments", kwargs.get("envs", []))

    completion_texts = [_completion_to_str(c) for c in completions]

    rubric_db = _load_rubrics()

    rewards: list[float] = []
    debug_info_all: list[dict] = []

    # 同 batch 内的所有 trajectory 并发(每条 trace 一次 LLM 调用)
    def _compute_one(idx: int, env, completion_text: str):
        question, question_type = _extract_question_and_type(idx, kwargs)

        # Step 0: 格式硬约束
        if not _is_format_valid(env, completion_text, question_type):
            return idx, FORMAT_INVALID_PENALTY, {"reason": "format_invalid"}

        # Step 1: 加载 rubric;若缺失,fallback V2
        rubric_set = rubric_db.get(question_type)
        if rubric_set is None:
            # 调用 V2 reward 兜底(它会处理这一条)
            v2_rewards = v2_reward_fallback(
                completions=[completion_text],
                environments=[env],
                question=[question],
                question_type=[question_type],
            )
            return idx, v2_rewards[0], {"reason": "v2_fallback", "missing_type": question_type}

        # Step 2: 计算 rubric reward
        base_reward, debug = _compute_v3_single_reward(
            env, completion_text, question, question_type, rubric_set,
        )

        # Step 3: 硬约束扣分(继承 V2)
        penalty = _call_quality_penalty(env)
        base_reward = max(base_reward + penalty, 0.0)

        # Step 4: DAPO overlong(继承 V2)
        token_est = _estimate_token_count(completion_text)
        final_reward = _apply_overlong_penalty(base_reward, token_est)

        debug["penalty"] = penalty
        debug["token_est"] = token_est
        debug["final_reward"] = final_reward
        return idx, final_reward, debug

    # 占位
    rewards = [0.0] * len(completion_texts)
    debug_info_all = [None] * len(completion_texts)

    with ThreadPoolExecutor(max_workers=V3_MAX_WORKERS) as ex:
        futures = {
            ex.submit(_compute_one, i, env, ct): i
            for i, (env, ct) in enumerate(zip(environments, completion_texts))
        }
        for fut in as_completed(futures):
            idx, reward, debug = fut.result()
            rewards[idx] = reward
            debug_info_all[idx] = debug

    # 自定义指标 logging(供 GRPOMetricsCallback 消费)
    _log_v3_metrics(debug_info_all, rewards)

    return rewards


# ============ V3 metrics(供 BC-4 修复后的 GRPOMetricsCallback)============

_v3_metrics: dict[str, list] = {
    "fact_satisfactions": [],
    "reasoning_satisfactions": [],
    "form_satisfactions": [],
    "v2_fallback_count": [],
    "format_invalid_count": [],
    "rule_judge_count": [],
    "llm_judge_count": [],
}
_v3_metrics_lock = threading.Lock()


def _log_v3_metrics(debug_info_all: list[dict | None], rewards: list[float]):
    """收集 V3 维度指标到全局,供 callback 写 TensorBoard"""
    with _v3_metrics_lock:
        for debug, reward in zip(debug_info_all, rewards):
            if not debug:
                continue
            if debug.get("reason") == "format_invalid":
                _v3_metrics["format_invalid_count"].append(1)
                continue
            if debug.get("reason") == "v2_fallback":
                _v3_metrics["v2_fallback_count"].append(1)
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
                if k.endswith("_satisfactions"):
                    out[f"v3_rubric/{k.replace('_satisfactions', '_satisfaction_rate')}"] = (
                        sum(vals) / len(vals)
                    )
                elif k.endswith("_count"):
                    out[f"v3_rubric/{k}_per_traj"] = sum(vals) / max(len(vals), 1)
            _v3_metrics[k] = []
        return out


# ============ Self-test(直接运行该文件做最小烟雾测试)============

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    # 加载 rubric 库
    db = _load_rubrics()
    print(f"\nLoaded rubric task_types: {list(db.keys())}")

    # 检查每个 rubric 配比
    for task_type, data in db.items():
        rubrics = data.get("rubrics", [])
        types = [r["type"] for r in rubrics]
        weights_sum = sum(r["weight"] for r in rubrics)
        print(
            f"  [{task_type}]"
            f" total={len(rubrics)}"
            f" fact={types.count('fact')}"
            f" reasoning={types.count('reasoning')}"
            f" form={types.count('form')}"
            f" weights_sum={weights_sum:.4f}"
        )
