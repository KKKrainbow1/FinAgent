"""
FinAgent SFT V4 Base PPL Selection(GRAPE 范式 Phase 3)

============================================================================
 设计目标
============================================================================

读 10b 输出的 qualified candidates,按 question_id group by,
用 base model(Qwen2.5-14B-Instruct)对每条 trajectory 算 perplexity,
每个 question 选 ppl 最低的 candidate → 最终 SFT 训练数据。

  GRAPE 论文(arxiv 2502.04194, NeurIPS 2025 Spotlight)核心结论:
    用 base model 算 response 的 perplexity,选最低的(GRAPE)
    让 SFT 数据更对齐 base 的预训练分布,SFT 学习效率显著提升
    (LLaMA3.1-8B,1/3 数据 + 半 epoch 超过 Tulu3-SFT 3.5%)

  关键认知:
    - 多峰性来源于 question 之间(数据集层),不是同一 question 多 response 内
    - 每个 question 选 1 个低 ppl response → 每个 mode 内部锐利
    - 数据集仍保持 28 bucket 多峰结构,不会单峰塌陷

  Ppl 算什么:
    用 V4 patched chat_template + return_assistant_tokens_mask=True 渲染整条对话,
    只对 assistant tokens(content + tool_calls,即 mask=1 的位置)算 cross-entropy,
    ppl = exp(mean_assistant_token_loss)
    → 与 SFT 训练时的 loss 范围完全一致(训-评一致)

============================================================================
 输出
============================================================================

  v4_sft_train.jsonl
    每个 question 选 ppl 最低的 candidate,V4 SFT sample 格式:
    {
      "question": str,
      "type": str,
      "subtype": str | null,
      "metadata": {...},
      "messages": [...],          # 训练直接用
      "num_tool_steps": int,
      "tools_used": list,
      "rule_check_score": float,
      "judge_score": dict,
      "grounding_check": dict,
      "base_ppl": float,
      "selected_rollout_idx": int,
      "n_qualified_candidates": int,
    }

  v4_ppl_all.jsonl(debug)
    所有 candidate 的 ppl 记录,便于事后审计 + ppl 分布分析

  v4_ppl_select_stats.json

============================================================================
 运行方式(必须在服务器 GPU 上)
============================================================================

    python 10c_v4_ppl_select.py \\
        --qualified ./data/sft/v4_pipeline/v4_candidates_qualified_pilot.jsonl \\
        --output    ./data/sft/v4_pipeline/v4_sft_train_pilot.jsonl \\
        --output_all_ppl ./data/sft/v4_pipeline/v4_ppl_all_pilot.jsonl \\
        --model_path /root/autodl-tmp/Finagent/models/Qwen2.5-14B-Instruct
"""

import argparse
import importlib.util
import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============ 复用 11_sft_train.py 的 patched template loader ============

_HERE = Path(__file__).resolve().parent
_TRAIN_PATH = _HERE / "11_sft_train.py"
_spec = importlib.util.spec_from_file_location("sft_train_v4", str(_TRAIN_PATH))
sft_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sft_train)


# ============ 日志 ============

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("v4_ppl_select")


# ============ Ppl 计算核心 ============

@torch.no_grad()
def compute_assistant_ppl(model, tokenizer, messages: list, tools_native: list,
                          max_length: int, device: str) -> tuple:
    """对单条 messages 渲染 → forward → 只在 assistant tokens 算 cross-entropy。

    Returns:
        (ppl, n_assistant_tokens, n_total_tokens, truncated)
    """
    out = tokenizer.apply_chat_template(
        messages,
        tools=tools_native,
        return_assistant_tokens_mask=True,
        return_dict=True,
        tokenize=True,
    )
    input_ids = out["input_ids"]
    mask = out["assistant_masks"]
    truncated = False

    if len(input_ids) > max_length:
        # 训练时也是 max_length=12288,这里直接截断尾部(若发生说明轨迹过长,极少见)
        input_ids = input_ids[:max_length]
        mask = mask[:max_length]
        truncated = True

    if sum(mask) == 0:
        # 没有 assistant token(理论不可能,防御一下)
        return float("inf"), 0, len(input_ids), truncated

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    mask_tensor = torch.tensor([mask], dtype=torch.float, device=device)

    outputs = model(input_tensor)
    logits = outputs.logits   # [1, L, V]

    # next-token prediction:第 t 位的 label 是 input_ids[t+1],loss 算在预测下一 token 的位置上
    # 第 t+1 个 token 是 assistant token(mask[t+1]=1) → 第 t 位 logits 的 loss 进入统计
    shift_logits = logits[:, :-1, :].contiguous()    # [1, L-1, V]
    shift_labels = input_tensor[:, 1:].contiguous()  # [1, L-1]
    shift_mask = mask_tensor[:, 1:].contiguous()     # [1, L-1]

    loss_per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size())   # [1, L-1]

    masked_sum = (loss_per_tok * shift_mask).sum()
    masked_count = shift_mask.sum()
    if masked_count.item() == 0:
        return float("inf"), 0, len(input_ids), truncated

    mean_loss = (masked_sum / masked_count).item()
    # 防 JSON 序列化:超大 loss 用 1e10 代替 inf(json.dumps inf 会报错)
    ppl = math.exp(mean_loss) if mean_loss < 30 else 1e10

    return ppl, int(masked_count.item()), len(input_ids), truncated


# ============ 主流程 ============

def load_qualified(path: str) -> list:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("qualified", False):
                items.append(d)
    return items


def append_jsonl(path: str, item: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="V4 Base Ppl Selection(GRAPE Phase 3)")
    parser.add_argument("--qualified", type=str, required=True,
                        help="10b 输出的 qualified candidates jsonl")
    parser.add_argument("--output", type=str, required=True,
                        help="最终 SFT 训练 jsonl(每 question 一条 ppl 最低)")
    parser.add_argument("--output_all_ppl", type=str, default="",
                        help="所有 candidate 的 ppl 记录(空=同目录 _ppl_all.jsonl)")
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/Finagent/models/Qwen2.5-14B-Instruct",
                        help="base model 路径(Qwen2.5-14B-Instruct)")
    parser.add_argument("--max_length", type=int, default=12288,
                        help="与训练 max_length 一致")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--resume", action="store_true",
                        help="跳过 output_all_ppl 中已记录的 (qid, rid)")
    args = parser.parse_args()

    # 输出路径
    output_path = args.output
    output_all_ppl = args.output_all_ppl or output_path.replace(
        ".jsonl", "_ppl_all.jsonl"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_all_ppl), exist_ok=True)

    log_path = os.path.join(
        os.path.dirname(output_path),
        f"10c_ppl_select_{datetime.now():%Y%m%d_%H%M%S}.log",
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("V4 Base Ppl Selection(GRAPE Phase 3)")
    logger.info(f"Qualified:    {args.qualified}")
    logger.info(f"Output:       {output_path}")
    logger.info(f"Output all:   {output_all_ppl}")
    logger.info(f"Base model:   {args.model_path}")
    logger.info(f"Max length:   {args.max_length}")
    logger.info("=" * 60)

    # ---- 加载 model + tokenizer(patched template)----
    logger.info("[1/4] 加载 tokenizer + patched chat_template")
    tokenizer = sft_train.load_tokenizer_with_patched_template(args.model_path)
    tools_native = sft_train.get_tools_native()

    logger.info(f"[2/4] 加载 base model({args.dtype})")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    logger.info(f"  Model loaded on {args.device} ({args.dtype})")

    # ---- 加载 qualified candidates ----
    candidates = load_qualified(args.qualified)
    logger.info(f"[3/4] 加载 qualified candidates: {len(candidates)} 条")

    # 按 question_id group by
    by_qid = defaultdict(list)
    for c in candidates:
        by_qid[c["question_id"]].append(c)
    logger.info(f"  覆盖 question 数: {len(by_qid)}")
    for n_cand in sorted(set(len(v) for v in by_qid.values())):
        n_qs = sum(1 for v in by_qid.values() if len(v) == n_cand)
        logger.info(f"  qualified={n_cand}: {n_qs} 个 question")

    # Resume:读已写入的 ppl_all,把 (qid, rid) → ppl 提前回填到 qid_to_ppls
    # 这样后面 select best 时已算的 candidate 也参与排序,不会"只在新算的里选 best"
    qid_to_ppls = defaultdict(list)   # {qid: [(ppl, candidate)]}
    resumed_keys = {}                 # {(qid, rid): ppl}
    if args.resume and os.path.exists(output_all_ppl):
        with open(output_all_ppl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    resumed_keys[(d["question_id"], d["rollout_idx"])] = d["base_ppl"]
                except Exception:
                    continue
        # 把 resumed ppl 与 candidate join,append 进 qid_to_ppls
        n_resumed = 0
        for qid, cands in by_qid.items():
            for cand in cands:
                key = (cand["question_id"], cand["rollout_idx"])
                if key in resumed_keys:
                    qid_to_ppls[qid].append((resumed_keys[key], cand))
                    n_resumed += 1
        logger.info(f"  [Resume] 已算 {len(resumed_keys)} 条 ppl,join 回 {n_resumed} 个 candidate")
    else:
        # 非 resume:备份旧文件
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for p in (output_path, output_all_ppl):
            if os.path.exists(p):
                os.rename(p, p + ".bak." + ts)

    # ---- 算 ppl ----
    logger.info("[4/4] 开始算 base model ppl")
    n_total = sum(len(v) for v in by_qid.values())
    n_done = 0
    n_truncated = 0
    n_failed = 0

    for qid in sorted(by_qid.keys()):
        for cand in by_qid[qid]:
            n_done += 1
            key = (cand["question_id"], cand["rollout_idx"])
            if key in resumed_keys:
                # 已算过(resume 模式),ppl 已 join 回 qid_to_ppls,跳过 forward
                continue
            try:
                ppl, n_assistant, n_total_tok, truncated = compute_assistant_ppl(
                    model, tokenizer, cand["messages"], tools_native,
                    max_length=args.max_length, device=args.device,
                )
                if truncated:
                    n_truncated += 1
                ppl_record = {
                    "question_id":   cand["question_id"],
                    "rollout_idx":   cand["rollout_idx"],
                    "type":          cand["type"],
                    "base_ppl":      ppl,
                    "n_assistant_tokens": n_assistant,
                    "n_total_tokens":     n_total_tok,
                    "truncated":     truncated,
                }
                append_jsonl(output_all_ppl, ppl_record)
                qid_to_ppls[qid].append((ppl, cand))
            except Exception as e:
                n_failed += 1
                logger.error(f"[q{qid} r{cand['rollout_idx']}] ppl 计算失败: {e}", exc_info=True)
                continue

            if n_done % 25 == 0:
                logger.info(
                    f"[Progress {n_done}/{n_total}] "
                    f"truncated={n_truncated}, failed={n_failed}"
                )

    # ---- 选每个 question 的最低 ppl candidate ----
    logger.info("=" * 60)
    logger.info("选 ppl 最低的 candidate")

    n_selected = 0
    n_questions_with_only_1 = 0
    ppl_distribution = []
    type_dist = defaultdict(int)

    for qid in sorted(qid_to_ppls.keys()):
        ppl_list = qid_to_ppls[qid]
        if not ppl_list:
            continue
        ppl_list.sort(key=lambda x: x[0])
        best_ppl, best_cand = ppl_list[0]
        ppl_distribution.append(best_ppl)

        if len(ppl_list) == 1:
            n_questions_with_only_1 += 1

        sft_sample = {
            "question":       best_cand["question"],
            "type":           best_cand["type"],
            "subtype":        best_cand.get("subtype"),
            "metadata":       best_cand.get("metadata", {}),
            "messages":       best_cand["messages"],
            "num_tool_steps": best_cand["num_tool_steps"],
            "tools_used":     best_cand["tools_used"],
            "rule_check":     best_cand.get("rule_check"),
            "judge_score":    best_cand.get("judge_score"),
            "grounding_check": best_cand.get("grounding_check"),
            "base_ppl":              best_ppl,
            "selected_rollout_idx":  best_cand["rollout_idx"],
            "n_qualified_candidates": len(ppl_list),
            "ppl_all": [round(p, 4) for p, _ in ppl_list],
        }
        append_jsonl(output_path, sft_sample)
        n_selected += 1
        type_dist[best_cand["type"]] += 1

    # ---- Stats ----
    avg_ppl = (sum(ppl_distribution) / len(ppl_distribution)) if ppl_distribution else 0
    median_ppl = sorted(ppl_distribution)[len(ppl_distribution) // 2] if ppl_distribution else 0

    final_stats = {
        "n_qualified_candidates":         len(candidates),
        "n_questions_processed":          len(by_qid),
        "n_questions_selected":           n_selected,
        "n_questions_with_only_1":        n_questions_with_only_1,
        "n_truncated":                    n_truncated,
        "n_failed":                       n_failed,
        "avg_ppl_selected":               round(avg_ppl, 4),
        "median_ppl_selected":            round(median_ppl, 4),
        "type_distribution":              dict(type_dist),
        "model_path":                     args.model_path,
        "max_length":                     args.max_length,
        "qualified_file":                 args.qualified,
        "output_file":                    output_path,
        "timestamp":                      datetime.now().isoformat(),
    }
    stats_path = output_path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("V4 Ppl Selection 完成")
    for k, v in final_stats.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"输出: {output_path}")
    logger.info(f"Stats: {stats_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
