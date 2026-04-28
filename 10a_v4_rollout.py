"""
FinAgent SFT V4 N-rollout 生成脚本(GRAPE 范式 Phase 1, AsyncOpenAI)

============================================================================
 设计目标
============================================================================

对 v4_questions_dedup.jsonl 中每条 question 跑 N=4 次独立 rollout(temperature=0.7),
每个 rollout 通过规则 D1-D10 初筛,输出 candidates 文件供下游 10b/10c 消费。

  并发模型(AsyncOpenAI + asyncio.Semaphore):
    - (qmeta, rollout_idx) 全部展开为 jobs
    - asyncio.gather + Semaphore 限制并发数(默认 8)
    - 每个 job 内部 ReAct 多步串行 await(单 rollout 内顺序不能乱)
    - 跨 jobs 完全并发(I/O bound,单 OpenAI client 复用)

  GRAPE 范式(arxiv 2502.04194, NeurIPS 2025 Spotlight):
    - 单 teacher × N 次采样 → 候选间风格可比
    - 给质检过滤更大的选择空间(N=4 → 预期 2-4 条进 ppl select)
    - 后续 10b 跑 LLM Grounding + 简化 Judge 过滤错误峰
    - 后续 10c 用 base model(Qwen2.5-14B) 算 trajectory ppl,选最低

  与 10_generate_sft_data.py 的区别:
    - 10:  N=1 rollout + 完整三层 Judge + regen,产 final SFT 数据(同步 OpenAI)
    - 10a: N=4 rollout + 仅规则 D1-D10 初筛,产 candidates(AsyncOpenAI 并发)

  为什么不复用 gen_sft.generate_trajectory_v4:
    - 10 用同步 OpenAI 客户端,API call 串行
    - 10a 用 AsyncOpenAI,需要 await 调用
    - 在 10a 内复制 ReAct 循环逻辑,常量(SYSTEM_PROMPT_V4 / TOOLS_NATIVE / VALID_TOOLS
      / MAX_RETRY / MAX_STEPS / GENERATION_HINTS / _validate_calc_expression / _eval_calc
      / _tool_call_to_v2)全部复用 gen_sft 模块,逻辑等价

============================================================================
 输出 schema(每行一个 candidate)
============================================================================

{
  "question_id":      int,           # v4_questions_dedup 行号(0-indexed)
  "rollout_idx":      int,           # 0..N-1
  "question":         str,
  "type":             str,
  "subtype":          str | None,
  "metadata":         {...},         # 完整 question 元数据(stock/industry/...)
  "messages":         list,          # V4 chat messages(可直接进训练)
  "steps":            list,          # V1-like steps(兼容下游 Judge)
  "num_tool_steps":   int,
  "tools_used":       list,
  "retrieval_quality": bool,
  "rule_check": {
    "passed":         bool,
    "score":          float,
    "issues":         list
  }
}

============================================================================
 运行方式
============================================================================

    # pilot(50 条 × N=4 = 200 rollout, concurrency=8)
    python 10a_v4_rollout.py --questions ./data/sft/questions/v4_questions_dedup.jsonl \\
        --output ./data/sft/v4_pipeline/v4_candidates_pilot.jsonl \\
        --n_rollouts 4 --temperature 0.7 --pilot 50 --concurrency 8

    # 全量(1540 × N=4 = 6160 rollout, concurrency=16)
    python 10a_v4_rollout.py --questions ./data/sft/questions/v4_questions_dedup.jsonl \\
        --output ./data/sft/v4_pipeline/v4_candidates.jsonl \\
        --n_rollouts 4 --temperature 0.7 --concurrency 16

    # 断点续传
    python 10a_v4_rollout.py --questions ... --output ... --n_rollouts 4 --resume

环境变量:
    export OPENAI_API_KEY="百炼 API Key"
    export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
"""

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI


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


# ============ AsyncOpenAI 版 generate_trajectory_v4 ============
# 复制 gen_sft.generate_trajectory_v4 的逻辑,把同步 client.chat.completions.create
# 换成 AsyncOpenAI 的 await 调用。常量与辅助函数(_validate_calc_expression / _eval_calc
# / _tool_call_to_v2 / SYSTEM_PROMPT_V4 / GENERATION_HINTS / TOOLS_NATIVE / MODEL /
# MAX_RETRY / MAX_STEPS / VALID_TOOLS)全部从 gen_sft 模块取,与 10 严格等价。

async def generate_trajectory_async(async_client: AsyncOpenAI, tools, question: str,
                                    question_type: str, temperature: float = 0.7,
                                    max_steps: int = None) -> dict:
    """V4 Mode B 异步生成 — 与 gen_sft.generate_trajectory_v4 行为等价,
    仅把 OpenAI 同步调用替换为 AsyncOpenAI await。"""
    if max_steps is None:
        max_steps = gen_sft.MAX_STEPS

    user_content = f"## 用户问题\n{question}\n\n## 问题类型\n{question_type}"
    user_content += f"\n{gen_sft.GENERATION_HINTS}"

    messages = [
        {"role": "system", "content": gen_sft.SYSTEM_PROMPT_V4},
        {"role": "user", "content": user_content},
    ]

    steps = []
    tools_used = []
    retrieval_quality = True
    finished = False

    for step_num in range(max_steps):
        tool_choice = "none" if step_num == max_steps - 1 else "auto"

        msg = None
        for retry in range(gen_sft.MAX_RETRY):
            try:
                response = await async_client.chat.completions.create(
                    model=gen_sft.MODEL,
                    messages=messages,
                    tools=gen_sft.TOOLS_NATIVE,
                    tool_choice=tool_choice,
                    parallel_tool_calls=False,
                    temperature=temperature,
                    max_tokens=1500,
                    extra_body={"enable_thinking": False},
                )
                msg = response.choices[0].message
                break
            except Exception as e:
                logger.warning(f"LLM 调用失败 (step {step_num+1} retry {retry}): {e}")
                await asyncio.sleep(1)
        if msg is None:
            return None

        content = (msg.content or "").strip()
        raw_tcs = msg.tool_calls or []

        # 分支 1:不调工具 → finish
        if not raw_tcs:
            if not content:
                return None
            steps.append({"thought": "", "action": "finish", "action_input": content})
            messages.append({"role": "assistant", "content": content})
            finished = True
            break

        # 分支 2:有 tool_calls
        tc = raw_tcs[0]
        name = tc.function.name
        if name not in gen_sft.VALID_TOOLS:
            continue

        try:
            args_raw = tc.function.arguments
            args_dict = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            if not isinstance(args_dict, dict):
                raise ValueError("arguments 不是 dict")
        except Exception as e:
            logger.warning(f"step {step_num+1} args 解析失败: {e}")
            continue

        if not content:
            content = "(thought 缺失)"

        # 执行工具(本地操作,直接同步调,不阻塞 event loop 太久)
        if name == "calculate":
            expr = args_dict.get("expression", "")
            try:
                expr_normalized = gen_sft._validate_calc_expression(expr)
                observation = gen_sft._eval_calc(expr_normalized)
                args_dict = {"expression": expr_normalized}
            except ValueError:
                continue
            action_input_v1 = expr_normalized
        else:
            query = args_dict.get("query", "")
            if not query:
                continue
            try:
                obs_and_meta = tools.call(name, args_dict)
                observation = obs_and_meta[0] if isinstance(obs_and_meta, tuple) else obs_and_meta
            except Exception as e:
                observation = f"[工具调用失败] {e}"
            if "未找到" in observation or len(observation) < 50:
                retrieval_quality = False
            action_input_v1 = query

        tools_used.append(name)
        steps.append({
            "thought": content,
            "action": name,
            "action_input": action_input_v1,
            "observation": observation,
        })
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [gen_sft._tool_call_to_v2(tc, args_dict)],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": observation,
        })

    if not finished:
        return None

    return {
        "question": question,
        "type": question_type,
        "messages": messages,
        "steps": steps,
        "num_tool_steps": len(tools_used),
        "tools_used": tools_used,
        "retrieval_quality": retrieval_quality,
    }


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


# ============ 单 (qmeta, rollout_idx) 处理 worker ============

async def process_one_rollout(qmeta: dict, r_idx: int,
                              async_client: AsyncOpenAI, tools,
                              args, write_lock: asyncio.Lock,
                              out_path: str, stats: Counter):
    """处理单个 (qmeta, r_idx) job:rollout → validate → rule_check → 写入。

    并发安全:async_client / tools(retrieval) 是线程/协程安全;
    stats 更新与文件写入在 write_lock 保护下做。
    """
    question_id = qmeta["__source_line__"]
    question = qmeta["question"]
    qtype = qmeta["type"]

    async with write_lock:
        stats["rollouts_total"] += 1

    try:
        plan = await generate_trajectory_async(
            async_client=async_client,
            tools=tools,
            question=question,
            question_type=qtype,
            temperature=args.temperature,
        )
        if plan is None:
            async with write_lock:
                stats["rollouts_failed_react"] += 1
            logger.warning(f"[q{question_id} r{r_idx}] react 失败")
            return

        passed_validation, validation_errors = gen_sft.validate_sample(plan)
        if not passed_validation:
            async with write_lock:
                stats["rollouts_failed_validation"] += 1
            logger.warning(f"[q{question_id} r{r_idx}] validation 失败: {validation_errors[:2]}")
            return

        rule_passed, rule_issues, rule_score = gen_sft.rule_based_quality_check(plan)
        rule_check = {
            "passed": rule_passed,
            "score":  float(rule_score),
            "issues": [i["detail"] for i in rule_issues],
        }
        if not rule_passed and not args.rule_check_keep_failed:
            async with write_lock:
                stats["rollouts_failed_rule_check"] += 1
            logger.warning(
                f"[q{question_id} r{r_idx}] rule_check 失败(score={rule_score:.2f}): "
                f"{rule_check['issues'][:2]}"
            )
            return

        candidate = build_candidate(question_id, r_idx, qmeta, plan, rule_check)
        async with write_lock:
            append_jsonl(out_path, candidate)
            stats["rollouts_succeeded"] += 1
            stats["rollouts_kept"] += 1
        logger.info(
            f"[q{question_id} r{r_idx}] keep "
            f"(steps={plan['num_tool_steps']}, rule={rule_score:.2f})"
        )
    except Exception as e:
        async with write_lock:
            stats["rollouts_failed_react"] += 1
        logger.error(f"[q{question_id} r{r_idx}] 未预期异常: {e}", exc_info=True)


# ============ 主流程 ============

async def main_async():
    parser = argparse.ArgumentParser(description="V4 N-rollout(AsyncOpenAI)+ 规则 D1-D10 初筛")
    parser.add_argument("--questions", type=str, required=True,
                        help="question jsonl 路径(v4_questions_dedup.jsonl)")
    parser.add_argument("--output", type=str, required=True,
                        help="candidates jsonl 输出路径")
    parser.add_argument("--n_rollouts", type=int, default=4, help="每条 question 采样次数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="teacher 采样温度(0.7 让 teacher 稳定输出;ppl 在 10c 内固定 T=1)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="asyncio.Semaphore 并发数(默认 8,全量推荐 16)")
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
    logger.info("FinAgent V4 N-rollout(GRAPE Phase 1, AsyncOpenAI)")
    logger.info(f"Questions:    {args.questions}")
    logger.info(f"Output:       {out_path}")
    logger.info(f"N rollouts:   {args.n_rollouts}  |  Temperature: {args.temperature}")
    logger.info(f"Concurrency:  {args.concurrency}")
    logger.info("=" * 60)

    # 初始化 client / tools
    async_client = AsyncOpenAI()
    retriever = gen_sft.FinAgentRetriever()
    tools = gen_sft.FinAgentTools(retriever)
    logger.info("[1/3] AsyncOpenAI / retriever / tools 初始化完成")

    # 加载 question
    questions = load_questions_full(args.questions)
    if args.type:
        questions = [q for q in questions if q.get("type") == args.type]
    if args.pilot > 0:
        random.shuffle(questions)
        type2qs = {}
        for q in questions:
            type2qs.setdefault(q["type"], []).append(q)
        num_types = len(type2qs)
        per_type = max(1, args.pilot // num_types)
        sampled = []
        for _, qs in type2qs.items():
            sampled.extend(qs[:per_type])
        # 不足 pilot 时从剩余里随机补足
        if len(sampled) < args.pilot:
            sampled_set = set(id(q) for q in sampled)
            remaining = [q for q in questions if id(q) not in sampled_set]
            random.shuffle(remaining)
            sampled.extend(remaining[:args.pilot - len(sampled)])
        random.shuffle(sampled)
        questions = sampled[:args.pilot]
        type_dist = Counter(q["type"] for q in questions)
        logger.info(f"[2/3] Pilot {len(questions)} 条,type 分布: {dict(type_dist)}")
    else:
        logger.info(f"[2/3] 全量 {len(questions)} 条")

    # 断点续传
    seen_pairs = set()
    if args.resume:
        seen_pairs, seen_qids = load_resume_state(out_path)
        logger.info(f"[Resume] 已写 {len(seen_pairs)} candidates,覆盖 {len(seen_qids)} 个 question")
    else:
        if os.path.exists(out_path):
            os.rename(out_path, out_path + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S"))

    # 构造 jobs:展开 (qmeta, r_idx),跳过已写入
    jobs = []
    for qmeta in questions:
        qid = qmeta["__source_line__"]
        for r in range(args.n_rollouts):
            if (qid, r) not in seen_pairs:
                jobs.append((qmeta, r))
    logger.info(f"[3/3] 总 jobs: {len(jobs)} (跳过 resume {sum(1 for q in questions for r in range(args.n_rollouts) if (q['__source_line__'], r) in seen_pairs)})")

    # Stats + write lock
    stats = Counter({
        "rollouts_total": 0,
        "rollouts_succeeded": 0,
        "rollouts_failed_react": 0,
        "rollouts_failed_validation": 0,
        "rollouts_failed_rule_check": 0,
        "rollouts_kept": 0,
    })
    write_lock = asyncio.Lock()

    # Semaphore + gather
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded_worker(qmeta, r_idx):
        async with semaphore:
            await process_one_rollout(qmeta, r_idx, async_client, tools,
                                      args, write_lock, out_path, stats)

    # 进度报告:每 N 个 done 打一次
    async def progress_reporter():
        last_done = 0
        while True:
            await asyncio.sleep(15)
            done = stats["rollouts_total"]
            if done == last_done:
                continue
            last_done = done
            kept = stats["rollouts_kept"]
            keep_rate = (kept / done * 100) if done else 0
            logger.info(
                f"[Progress] done={done}/{len(jobs)}, kept={kept} ({keep_rate:.1f}%) "
                f"| react_fail={stats['rollouts_failed_react']} "
                f"| valid_fail={stats['rollouts_failed_validation']} "
                f"| rule_fail={stats['rollouts_failed_rule_check']}"
            )

    reporter_task = asyncio.create_task(progress_reporter())
    try:
        await asyncio.gather(*[bounded_worker(q, r) for q, r in jobs])
    finally:
        reporter_task.cancel()
        try:
            await reporter_task
        except asyncio.CancelledError:
            pass
        await async_client.close()

    # 最终 stats
    stats_path = out_path.replace(".jsonl", "_stats.json")
    final_stats = {
        **dict(stats),
        "n_rollouts_per_question": args.n_rollouts,
        "temperature":             args.temperature,
        "concurrency":             args.concurrency,
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


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
