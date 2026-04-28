"""
FinAgent SFT V4 N-rollout 生成脚本(GRAPE 范式 Phase 1)

============================================================================
 设计目标
============================================================================

对 v4_questions_dedup.jsonl 中每条 question 跑 N=4 次独立 rollout(temperature=1.0),
每个 rollout 通过规则 D1-D10 初筛,输出 candidates 文件供下游 10b/10c 消费。

  GRAPE 范式(arxiv 2502.04194, NeurIPS 2025 Spotlight):
    - 单 teacher × N 次采样 → 候选间风格可比(避免双 teacher 风格差异污染 ppl 比较)
    - 给质检过滤更大的选择空间(N=4 → 预期 2-4 条进 ppl select)
    - 后续 10b 跑 LLM Grounding + 简化 Judge 过滤错误峰
    - 后续 10c 用 base model(Qwen2.5-14B) 算 trajectory ppl,选最低
      → q_SFT 在每个 mode 内部锐利对齐预训练分布

  与 10_generate_sft_data.py 的区别:
    - 10:  N=1 rollout + 完整三层 Judge + regen,产 final SFT 数据
    - 10a: N=4 rollout + 仅规则 D1-D10 初筛,产 candidates(不做 LLM Judge)
           交由 10b 做 LLM Grounding + 简化 Judge,10c 做 ppl 选低

============================================================================
 输出 schema(每行一个 candidate)
============================================================================

{
  "question_id":      int,           # v4_questions_dedup 行号(0-indexed)
  "rollout_idx":      int,           # 0..N-1
  "question":         str,
  "type":             str,
  "subtype":          str | None,
  "metadata":         {              # 完整 question 元数据(stock/industry/metric_tag/...)
    "stock_code":     str,
    "stock_name":     str,
    "industry":       str,
    "metric_tag":     list,
    "template_class": str,
    "period":         str | None,
    "source":         str,
    ...
  },
  "messages":         list,          # V4 chat messages(可直接进训练)
  "steps":            list,          # V1-like steps(兼容下游 Judge)
  "num_tool_steps":   int,
  "tools_used":       list,
  "retrieval_quality": bool,
  "rule_check": {
    "passed":         bool,          # True = 进 10b
    "score":          float,
    "issues":         list           # [str, ...] 命中的规则点
  }
}

============================================================================
 运行方式
============================================================================

    # pilot(50 条 × N=4 = 200 rollout)
    python 10a_v4_rollout.py --questions ./data/sft/questions/v4_questions_dedup.jsonl \\
        --output ./data/sft/v4_pipeline/v4_candidates_pilot.jsonl \\
        --n_rollouts 4 --temperature 1.0 --pilot 50

    # 全量(1540 × N=4 = 6160 rollout)
    python 10a_v4_rollout.py --questions ./data/sft/questions/v4_questions_dedup.jsonl \\
        --output ./data/sft/v4_pipeline/v4_candidates.jsonl \\
        --n_rollouts 4 --temperature 1.0

    # 断点续传
    python 10a_v4_rollout.py --questions ... --output ... --n_rollouts 4 --resume

环境变量:
    export OPENAI_API_KEY="百炼 API Key"
    export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
"""

import argparse
import importlib.util
import json
import logging
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from openai import OpenAI


# ============ 动态加载 10_generate_sft_data.py(文件名以数字开头无法直接 import)============

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
logger = logging.getLogger("v4_rollout")


# ============ 辅助 ============

def load_questions_full(path: str) -> list:
    """加载 question 完整 schema(保留 metadata 进 candidate)。

    每条 question 注入 __source_line__ = 0-indexed 行号,作为后续 question_id 主源,
    保证无论 pilot / shuffle 后 question_id 仍能追回 dedup 原始行号。
    """
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            q["__source_line__"] = line_no
            questions.append(q)
    return questions


def build_candidate(question_id: int, rollout_idx: int, question_meta: dict,
                    plan: dict, rule_check: dict) -> dict:
    """从 plan + rule_check 组装 candidate 行。"""
    metadata = {
        k: v for k, v in question_meta.items()
        if k not in ("question", "type", "subtype", "__source_line__")
    }
    return {
        "question_id":       question_id,
        "rollout_idx":       rollout_idx,
        "question":          plan["question"],
        "type":              plan["type"],
        "subtype":           question_meta.get("subtype"),
        "metadata":          metadata,
        "messages":          plan["messages"],
        "steps":             plan["steps"],
        "num_tool_steps":    plan["num_tool_steps"],
        "tools_used":        plan["tools_used"],
        "retrieval_quality": plan["retrieval_quality"],
        "rule_check":        rule_check,
    }


def append_jsonl(path: str, item: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_resume_state(output_path: str) -> tuple:
    """从已写入的 candidates 文件读取已完成的 (question_id, rollout_idx) 集合。

    用 set 而非 max_qid:question_id 是 dedup 行号(注入自 load_questions_full),
    主循环顺序是 shuffle 后的,行号不单调,不能用 max_qid <= 截断。
    """
    seen_pairs = set()
    seen_qids = set()
    if not os.path.exists(output_path):
        return seen_pairs, seen_qids
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                qid = d.get("question_id", -1)
                rid = d.get("rollout_idx", -1)
                seen_pairs.add((qid, rid))
                seen_qids.add(qid)
            except Exception:
                continue
    return seen_pairs, seen_qids


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="V4 N-rollout 生成 + 规则 D1-D10 初筛")
    parser.add_argument("--questions", type=str, required=True,
                        help="question jsonl 路径(v4_questions_dedup.jsonl)")
    parser.add_argument("--output", type=str, required=True,
                        help="candidates jsonl 输出路径")
    parser.add_argument("--n_rollouts", type=int, default=4, help="每条 question 采样次数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="teacher 采样温度(0.7 让 teacher 稳定输出;ppl 计算与此解耦,在 10c 里固定用 temp=1 标准)")
    parser.add_argument("--pilot", type=int, default=0,
                        help="只跑前 N 条 question(0=全量)")
    parser.add_argument("--type", type=str, default="",
                        help="只跑指定 type(空=全 type)")
    parser.add_argument("--resume", action="store_true", help="从已写入 candidates 续跑")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rule_check_keep_failed", action="store_true",
                        help="规则质检 fail 也写入 candidates(默认丢弃,debug 用)")
    args = parser.parse_args()

    random.seed(args.seed)

    # 输出路径准备
    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_dir = os.path.dirname(out_path)
    log_path = os.path.join(log_dir, f"10a_rollout_{datetime.now():%Y%m%d_%H%M%S}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("FinAgent V4 N-rollout(GRAPE Phase 1)")
    logger.info(f"Questions: {args.questions}")
    logger.info(f"Output:    {out_path}")
    logger.info(f"N rollouts: {args.n_rollouts}  |  Temperature: {args.temperature}")
    logger.info("=" * 60)

    # 初始化 client / tools
    client = OpenAI()
    gen_sft._mental_math_client = client          # 兼容 gen_sft 内部 hook
    retriever = gen_sft.FinAgentRetriever()
    tools = gen_sft.FinAgentTools(retriever)
    seed_grouped = gen_sft.load_seed_data(gen_sft.SEED_DATA_PATH)
    logger.info(f"种子数据: {', '.join(f'{k}={len(v)}' for k, v in seed_grouped.items())}")

    # 加载 question
    questions = load_questions_full(args.questions)
    if args.type:
        questions = [q for q in questions if q.get("type") == args.type]
    if args.pilot > 0:
        # pilot:按 type 分层抽样,而非简单 head
        random.shuffle(questions)
        type2qs = {}
        for q in questions:
            type2qs.setdefault(q["type"], []).append(q)
        # 平均每 type 取 ceil(pilot / num_types)
        num_types = len(type2qs)
        per_type = max(1, args.pilot // num_types)
        sampled = []
        for t, qs in type2qs.items():
            sampled.extend(qs[:per_type])
        # 截到 pilot
        random.shuffle(sampled)
        questions = sampled[:args.pilot]
        # 重新按 source 分布查看
        type_dist = Counter(q["type"] for q in questions)
        logger.info(f"Pilot {len(questions)} 条,type 分布: {dict(type_dist)}")
    else:
        logger.info(f"全量 {len(questions)} 条")

    # 断点续传
    seen_pairs = set()    # {(qid, rid)} 已写入 candidate 的精确集合
    if args.resume:
        seen_pairs, seen_qids = load_resume_state(out_path)
        logger.info(f"[Resume] 已写 {len(seen_pairs)} candidates,覆盖 {len(seen_qids)} 个 question")
    else:
        # 非 resume 模式,output 已存在则直接覆盖
        if os.path.exists(out_path):
            os.rename(out_path, out_path + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S"))

    # 主循环
    stats = Counter({
        "questions_total": 0,
        "rollouts_total": 0,
        "rollouts_succeeded": 0,
        "rollouts_failed_react": 0,
        "rollouts_failed_validation": 0,
        "rollouts_failed_rule_check": 0,
        "rollouts_kept": 0,
    })

    for q_idx, qmeta in enumerate(questions):
        # question_id = dedup 原始文件行号(load_questions_full 已注入 __source_line__)
        question_id = qmeta["__source_line__"]

        question = qmeta["question"]
        qtype = qmeta["type"]
        stats["questions_total"] += 1

        if stats["questions_total"] % 10 == 1:
            kept = stats["rollouts_kept"]
            tried = stats["rollouts_total"]
            keep_rate = (kept / tried * 100) if tried else 0
            logger.info(
                f"[Progress] q={stats['questions_total']}/{len(questions)}"
                f" | rollouts kept={kept}/{tried} ({keep_rate:.1f}%)"
            )

        # 每条 question 跑 N rollout
        for r_idx in range(args.n_rollouts):
            # Resume:精确跳过 (qid, rid) 已写入的 rollout
            if (question_id, r_idx) in seen_pairs:
                continue
            stats["rollouts_total"] += 1
            try:
                plan = gen_sft.generate_trajectory_v4(
                    client=client,
                    tools=tools,
                    question=question,
                    question_type=qtype,
                    seed_grouped=seed_grouped,
                    temperature=args.temperature,
                )
                if plan is None:
                    stats["rollouts_failed_react"] += 1
                    logger.warning(f"[q{question_id} r{r_idx}] react 失败")
                    continue

                # validate(format)
                passed_validation, validation_errors = gen_sft.validate_sample(plan)
                if not passed_validation:
                    stats["rollouts_failed_validation"] += 1
                    logger.warning(f"[q{question_id} r{r_idx}] validation 失败: {validation_errors[:2]}")
                    continue

                # 规则质检 D1-D10
                rule_passed, rule_issues, rule_score = gen_sft.rule_based_quality_check(plan)
                rule_check = {
                    "passed": rule_passed,
                    "score":  float(rule_score),
                    "issues": [i["detail"] for i in rule_issues],
                }

                if not rule_passed and not args.rule_check_keep_failed:
                    stats["rollouts_failed_rule_check"] += 1
                    logger.warning(
                        f"[q{question_id} r{r_idx}] rule_check 失败(score={rule_score:.2f}): "
                        f"{rule_check['issues'][:2]}"
                    )
                    continue

                # 写入 candidate
                candidate = build_candidate(question_id, r_idx, qmeta, plan, rule_check)
                append_jsonl(out_path, candidate)
                stats["rollouts_succeeded"] += 1
                stats["rollouts_kept"] += 1
                logger.info(
                    f"[q{question_id} r{r_idx}] keep "
                    f"(steps={plan['num_tool_steps']}, rule={rule_score:.2f})"
                )
            except Exception as e:
                stats["rollouts_failed_react"] += 1
                logger.error(f"[q{question_id} r{r_idx}] 未预期异常: {e}", exc_info=True)
                continue

            time.sleep(0.2)   # 防 rate limit

    # 最终 stats
    stats_path = out_path.replace(".jsonl", "_stats.json")
    final_stats = {
        **dict(stats),
        "n_rollouts_per_question": args.n_rollouts,
        "temperature":             args.temperature,
        "questions_file":          args.questions,
        "output_file":             out_path,
        "timestamp":               datetime.now().isoformat(),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("V4 N-rollout 完成")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    keep_rate = (stats['rollouts_kept'] / stats['rollouts_total'] * 100) if stats['rollouts_total'] else 0
    logger.info(f"  整体 keep_rate: {keep_rate:.1f}%")
    logger.info(f"输出: {out_path}")
    logger.info(f"Stats: {stats_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
