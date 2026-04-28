"""
FinAgent SFT V4 Quality Filter(GRAPE 范式 Phase 2)

============================================================================
 设计目标
============================================================================

读 10a 输出的 candidates,跑两层 LLM 质检串联(对应 V4 故事 §3.1):

  Phase 2.75 第 1 关 LLM Grounding(数字溯源专项)
    - label=0 → 进第 2 关
    - label=1 → 丢弃(数字编造)
    - uncertain / failed → 丢弃
  Phase 2.75 第 2 关 简化 Judge D2-D5(逻辑/匹配/专业/工具)
    - total >= judge_threshold → 通过(qualified=True)
    - 其余丢弃

  注意:
    - reject / finance_concept / 0 步轨迹在 grounding 内部已 skip,自动判 label=0
    - 没有 regen 闭环(10 才 regen,10a/b/c pipeline 不重试,直接走 ppl 选低)
    - 每个 candidate 独立判,不跨 candidate 投票

============================================================================
 输出
============================================================================

  v4_candidates_qualified.jsonl
    candidate + 增加字段 grounding_check / judge_score / qualified=True

  v4_candidates_rejected.jsonl(debug 用)
    qualified=False 的 candidate + reject_reason

  v4_quality_filter_stats.json

============================================================================
 运行方式
============================================================================

    # pilot
    python 10b_v4_quality_filter.py \\
        --candidates ./data/sft/v4_pipeline/v4_candidates_pilot.jsonl \\
        --output_qualified ./data/sft/v4_pipeline/v4_candidates_qualified_pilot.jsonl \\
        --output_rejected  ./data/sft/v4_pipeline/v4_candidates_rejected_pilot.jsonl \\
        --workers 8

    # 全量
    python 10b_v4_quality_filter.py \\
        --candidates ./data/sft/v4_pipeline/v4_candidates.jsonl \\
        --output_qualified ./data/sft/v4_pipeline/v4_candidates_qualified.jsonl \\
        --output_rejected  ./data/sft/v4_pipeline/v4_candidates_rejected.jsonl \\
        --workers 16
"""

import argparse
import importlib.util
import json
import logging
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from llm_grounding import llm_grounding_check


# ============ 加载 10 模块以复用 judge_single_inline / extract_*_from_steps ============

_HERE = Path(__file__).resolve().parent
_GEN_SFT_PATH = _HERE / "10_generate_sft_data.py"
_spec = importlib.util.spec_from_file_location("gen_sft_v4", str(_GEN_SFT_PATH))
gen_sft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_sft)


# ============ 日志 ============

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("v4_quality_filter")


# ============ 文件写入(带线程锁)============

_WRITE_LOCK = threading.Lock()

def append_jsonl(path: str, item: dict):
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def candidate_to_plan(candidate: dict) -> dict:
    """从 candidate 构造 grounding/judge 需要的 plan dict。"""
    return {
        "question":       candidate["question"],
        "type":           candidate["type"],
        "subtype":        candidate.get("subtype"),
        "steps":          candidate["steps"],
        "messages":       candidate["messages"],
        "tools_used":     candidate.get("tools_used", []),
        "num_tool_steps": candidate.get("num_tool_steps", 0),
    }


def call_judge_inline(client: OpenAI, question: str, observations: str,
                      answer: str, question_type: str, model: str,
                      max_retry: int = 3) -> dict:
    """简化 Judge D2-D5 — 线程安全版(model 参数化,不动 gen_sft 全局变量)。

    复用 gen_sft.JUDGE_PROMPT_INLINE,自己发 API 调用,避免修改 gen_sft.JUDGE_MODEL
    在 ThreadPoolExecutor 多线程下产生 race condition。
    """
    prompt = gen_sft.JUDGE_PROMPT_INLINE.format(
        question=question,
        question_type=question_type,
        observations=observations if observations else "(无检索数据)",
        answer=answer if answer else "(无回答)",
    )
    last_err = None
    for attempt in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=800,
                extra_body={"enable_thinking": False},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            last_err = e
    return {"total": 0, "issues": [f"Judge 调用失败: {last_err}"], "reason": "调用失败"}


# ============ 处理单个 candidate ============

def process_candidate(candidate: dict, client: OpenAI, judge_threshold: float,
                      grounding_model: str, judge_model: str,
                      qualified_path: str, rejected_path: str) -> str:
    """
    Returns:
        'qualified' | 'rejected_grounding' | 'rejected_judge' | 'error'
    """
    qid = candidate.get("question_id", -1)
    rid = candidate.get("rollout_idx", -1)
    plan = candidate_to_plan(candidate)

    # ---- 第 1 关 LLM Grounding ----
    try:
        grounding = llm_grounding_check(
            client, plan,
            n_samples=1,
            model=grounding_model,
            temperature=0.3,
        )
    except Exception as e:
        logger.error(f"[q{qid} r{rid}] grounding 异常: {e}")
        candidate["grounding_check"] = {"label": "failed", "votes": {"0": 0, "1": 0},
                                        "reasons": [str(e)], "regen_type": None}
        candidate["qualified"] = False
        candidate["reject_reason"] = "grounding_exception"
        append_jsonl(rejected_path, candidate)
        return "error"

    candidate["grounding_check"] = grounding
    g_label = grounding.get("label")

    if g_label != 0:
        # label=1 / 'uncertain' / 'failed' 全部丢弃
        candidate["qualified"] = False
        candidate["reject_reason"] = f"grounding_label_{g_label}"
        append_jsonl(rejected_path, candidate)
        return "rejected_grounding"

    # ---- 第 2 关 简化 Judge D2-D5 ----
    obs = gen_sft.extract_obs_from_steps(plan["steps"])
    answer = gen_sft.extract_answer_from_steps(plan["steps"])
    qtype = plan["type"]
    question = plan["question"]

    # finance_concept / reject 类 0 步直答,Judge 仍能跑(JUDGE_PROMPT_INLINE 不区分 type)
    # 但答案 / observation 可能很短,judge 评分体系仍能给出合理分
    try:
        judge_result = call_judge_inline(client, question, obs, answer, qtype, judge_model)
    except Exception as e:
        logger.error(f"[q{qid} r{rid}] judge 异常: {e}")
        candidate["judge_score"] = {"total": 0, "issues": [str(e)], "reason": "judge_exception"}
        candidate["qualified"] = False
        candidate["reject_reason"] = "judge_exception"
        append_jsonl(rejected_path, candidate)
        return "error"

    candidate["judge_score"] = judge_result
    total = judge_result.get("total", 0)

    if total < judge_threshold:
        candidate["qualified"] = False
        candidate["reject_reason"] = f"judge_total_{total}"
        append_jsonl(rejected_path, candidate)
        return "rejected_judge"

    candidate["qualified"] = True
    append_jsonl(qualified_path, candidate)
    return "qualified"


# ============ 主流程 ============

def load_candidates(path: str) -> list:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_resume_keys(qualified_path: str, rejected_path: str) -> set:
    """从已写入文件读 (question_id, rollout_idx) 集合,断点续传。"""
    seen = set()
    for p in (qualified_path, rejected_path):
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    seen.add((d["question_id"], d["rollout_idx"]))
                except Exception:
                    continue
    return seen


def main():
    parser = argparse.ArgumentParser(description="V4 Quality Filter(LLM Grounding + 简化 Judge)")
    parser.add_argument("--candidates", type=str, required=True,
                        help="10a 输出的 candidates jsonl")
    parser.add_argument("--output_qualified", type=str, required=True,
                        help="qualified candidates 输出路径")
    parser.add_argument("--output_rejected", type=str, default="",
                        help="rejected candidates 输出路径(空=同目录 _rejected.jsonl)")
    parser.add_argument("--judge_threshold", type=float, default=4.0,
                        help="简化 Judge total 阈值(>=阈值通过,默认 4.0 与 10 一致)")
    parser.add_argument("--grounding_model", type=str, default="qwen3-max")
    parser.add_argument("--judge_model", type=str, default="qwen3-max")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数")
    parser.add_argument("--resume", action="store_true",
                        help="跳过 qualified/rejected 文件中已处理的 (qid, rid)")
    args = parser.parse_args()

    # 输出路径
    qualified_path = args.output_qualified
    rejected_path = args.output_rejected or qualified_path.replace(
        "_qualified", "_rejected"
    ).replace(".jsonl", "_rejected.jsonl")
    os.makedirs(os.path.dirname(qualified_path), exist_ok=True)
    os.makedirs(os.path.dirname(rejected_path), exist_ok=True)

    log_path = os.path.join(
        os.path.dirname(qualified_path),
        f"10b_quality_filter_{datetime.now():%Y%m%d_%H%M%S}.log",
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("V4 Quality Filter(GRAPE Phase 2)")
    logger.info(f"Candidates: {args.candidates}")
    logger.info(f"Qualified:  {qualified_path}")
    logger.info(f"Rejected:   {rejected_path}")
    logger.info(f"Threshold:  judge_total >= {args.judge_threshold}")
    logger.info(f"Workers:    {args.workers}")
    logger.info("=" * 60)

    # 加载 candidates
    candidates = load_candidates(args.candidates)
    logger.info(f"加载 {len(candidates)} 条 candidates")

    # Resume
    seen = set()
    if args.resume:
        seen = load_resume_keys(qualified_path, rejected_path)
        logger.info(f"[Resume] 跳过 {len(seen)} 条已处理")
    else:
        # 非 resume:重命名旧文件
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for p in (qualified_path, rejected_path):
            if os.path.exists(p):
                os.rename(p, p + ".bak." + ts)

    todo = [c for c in candidates if (c["question_id"], c["rollout_idx"]) not in seen]
    logger.info(f"待处理 {len(todo)} 条")

    # OpenAI client(线程安全)
    client = OpenAI()

    stats = Counter({
        "total": 0, "qualified": 0,
        "rejected_grounding": 0, "rejected_judge": 0, "error": 0,
    })

    # 并发处理
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                process_candidate, c, client, args.judge_threshold,
                args.grounding_model, args.judge_model,
                qualified_path, rejected_path,
            ): c
            for c in todo
        }
        for i, fut in enumerate(as_completed(futures), 1):
            c = futures[fut]
            try:
                outcome = fut.result()
                stats[outcome] += 1
                stats["total"] += 1
            except Exception as e:
                logger.error(f"[q{c['question_id']} r{c['rollout_idx']}] worker 异常: {e}",
                             exc_info=True)
                stats["error"] += 1
                stats["total"] += 1

            if i % 20 == 0 or i == len(todo):
                qrate = stats["qualified"] / stats["total"] * 100 if stats["total"] else 0
                logger.info(
                    f"[Progress {i}/{len(todo)}] "
                    f"qualified={stats['qualified']} ({qrate:.1f}%) | "
                    f"rej_grounding={stats['rejected_grounding']} | "
                    f"rej_judge={stats['rejected_judge']} | err={stats['error']}"
                )

    # ---- 终结统计 ----
    # 按 question_id 看 qualified 分布(做 ppl 选低前的容量预估)
    if os.path.exists(qualified_path):
        qcount_by_qid = Counter()
        with open(qualified_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    qcount_by_qid[d["question_id"]] += 1
                except Exception:
                    continue
        nq_with_at_least_1 = sum(1 for v in qcount_by_qid.values() if v >= 1)
        nq_with_at_least_2 = sum(1 for v in qcount_by_qid.values() if v >= 2)
        avg_qualified = (sum(qcount_by_qid.values()) / len(qcount_by_qid)) if qcount_by_qid else 0
    else:
        nq_with_at_least_1 = nq_with_at_least_2 = 0
        avg_qualified = 0

    final_stats = {
        **dict(stats),
        "n_questions_with_at_least_1_qualified": nq_with_at_least_1,
        "n_questions_with_at_least_2_qualified": nq_with_at_least_2,
        "avg_qualified_per_question": round(avg_qualified, 2),
        "judge_threshold": args.judge_threshold,
        "candidates_file": args.candidates,
        "qualified_file": qualified_path,
        "rejected_file": rejected_path,
        "timestamp": datetime.now().isoformat(),
    }
    stats_path = qualified_path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("V4 Quality Filter 完成")
    for k, v in final_stats.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
