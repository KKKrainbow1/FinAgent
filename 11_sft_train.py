#!/usr/bin/env python3
"""
FinAgent Step 11: SFT 训练脚本（带 Observation Loss Mask）

核心设计：
    - 使用 LoRA 微调 Qwen2.5-14B-Instruct
    - Observation token 的 loss 被 mask 掉（设为 -100）
    - 模型只学习生成 Thought / Action / Action Input / finish 答案
    - Observation 由检索系统在推理时提供，不需要模型学会生成

用法：
    # 默认配置训练
    python 11_sft_train.py

    # 自定义参数
    python 11_sft_train.py \
        --model_path ./models/Qwen2.5-14B-Instruct \
        --data_path ./data/sft/sft_data_train.jsonl \
        --output_dir ./outputs/sft_lora \
        --epochs 3 \
        --lr 2e-4

    # 只做数据预处理检查（不训练）
    python 11_sft_train.py --dry-run

面试追问：为什么要 mask Observation？
答：Observation 是检索系统返回的真实数据，推理时由工具提供，不是模型该生成的。
如果不 mask，模型会花大量参数去"记忆"检索结果的格式和内容，
既浪费模型容量，又会导致推理时生成虚假的 Observation。
mask 之后，模型只学习：给定 Observation → 如何在 Thought 中分析它 →
选择什么 Action → 最终如何生成 finish 报告。
"""

import json
import os
import re
import sys
import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============ System Prompt（与 08_prompts.py 保持一致） ============

def get_system_prompt():
    """获取 system prompt，与推理时使用的完全一致"""
    from tools import FinAgentTools
    from prompts import build_system_prompt
    return build_system_prompt(FinAgentTools.TOOL_DESCRIPTIONS)


SYSTEM_PROMPT_FALLBACK = """你是"金融翻译官"，一个专业的A股上市公司分析助手。你的任务是将复杂的财务数据和研报信息转化为清晰、有依据的分析报告。

## 可用工具
1. search_report(query: str) → 检索券商研报信息。返回机构评级、目标价、EPS预测、行业分析等。
2. search_financial(query: str) → 检索公司财务数据。返回ROE、毛利率、营收增长率、资产负债率等指标。
3. calculate(expression: str) → 计算数学表达式。支持加减乘除、百分比、括号等。
4. finish(answer: str) → 输出最终分析报告并结束。

## 输出格式（严格遵守）
每一步必须按以下格式输出：

Thought: <你的思考过程>
Action: <工具名>
Action Input: <工具输入参数>

当信息收集完毕时：

Thought: <总结信息>
Action: finish
Action Input: <完整的分析报告>"""


# ============ 数据格式化 ============

def format_trajectory(sample: dict) -> str:
    """
    将一条 SFT 样本格式化为 assistant 的完整回复文本。
    与 09_react_agent.py 的输出格式完全一致。
    """
    parts = []
    for step in sample["steps"]:
        parts.append(f"Thought: {step['thought']}")
        parts.append(f"Action: {step['action']}")
        parts.append(f"Action Input: {step['action_input']}")
        if step.get("observation"):
            parts.append(f"Observation: {step['observation']}")
    return "\n".join(parts)


def build_observation_spans(trajectory_text: str) -> list[tuple[int, int]]:
    """
    找出 trajectory 文本中所有 Observation 内容的字符级 span。
    返回 [(start, end), ...] 列表，这些区间的 token 需要被 mask。

    Observation 格式：
        Observation: xxxxxx（多行内容）
        Thought:  ← 下一个 Thought 标记结束

    最后一步（finish）没有 Observation，不需要处理。
    """
    spans = []
    # 匹配 "Observation: " 开头到下一个 "Thought: " 或文本末尾
    pattern = r'(Observation: .*?)(?=\nThought: |\Z)'
    for m in re.finditer(pattern, trajectory_text, re.DOTALL):
        spans.append((m.start(), m.end()))
    return spans


# ============ Dataset ============

class SFTDataset(Dataset):
    """
    SFT 训练数据集，支持 Observation loss mask。

    每条数据格式化为 Qwen2.5 的 chat template：
        <|im_start|>system\n{system_prompt}<|im_end|>\n
        <|im_start|>user\n问题：{question}<|im_end|>\n
        <|im_start|>assistant\n{trajectory}<|im_end|>

    Loss mask 策略：
        - system + user 部分：全部 mask（labels = -100）
        - assistant 部分：Observation 内容 mask，其余保留
    """

    def __init__(self, data_path: str, tokenizer, system_prompt: str,
                 max_length: int = 4096):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_length = max_length

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = [json.loads(line) for line in f]

        logger.info(f"加载 {len(self.samples)} 条训练数据")

        # 预处理
        self.processed = []
        skipped = 0
        for i, sample in enumerate(self.samples):
            result = self._process_sample(sample)
            if result is not None:
                self.processed.append(result)
            else:
                skipped += 1

        logger.info(f"预处理完成: {len(self.processed)} 条可用, {skipped} 条跳过")

    def _process_sample(self, sample: dict):
        """处理单条样本：tokenize + 构建 loss mask"""

        question = sample["question"]
        trajectory = format_trajectory(sample)

        # 构建 messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"问题：{question}"},
            {"role": "assistant", "content": trajectory},
        ]

        # 用 chat template tokenize 完整序列
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        input_ids = full_tokens["input_ids"]
        offset_mapping = full_tokens["offset_mapping"]

        if len(input_ids) >= self.max_length:
            return None  # 超长跳过

        # ---- 构建 labels ----
        # 默认全部 mask（-100）
        labels = [-100] * len(input_ids)

        # 找到 assistant 回复在原文中的起始位置
        # Qwen2.5 的 chat template: ...<|im_start|>assistant\n{content}<|im_end|>
        assistant_marker = "<|im_start|>assistant\n"
        assistant_start_char = full_text.find(assistant_marker)
        if assistant_start_char < 0:
            return None

        assistant_content_start_char = assistant_start_char + len(assistant_marker)

        # assistant 内容的结束位置（<|im_end|> 之前）
        assistant_end_marker = "<|im_end|>"
        assistant_end_char = full_text.find(assistant_end_marker, assistant_content_start_char)
        if assistant_end_char < 0:
            assistant_end_char = len(full_text)

        # 找出 Observation 在 trajectory 中的 char spans
        obs_spans_in_traj = build_observation_spans(trajectory)

        # 转换为 full_text 中的 char spans
        obs_spans_in_full = [
            (s + assistant_content_start_char, e + assistant_content_start_char)
            for s, e in obs_spans_in_traj
        ]

        # 对每个 token，判断是否应该计算 loss
        for token_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:
                continue  # special token

            # 只对 assistant 内容区间的 token 计算 loss
            if tok_start < assistant_content_start_char:
                continue  # system + user 部分，保持 -100

            if tok_start >= assistant_end_char:
                continue  # <|im_end|> 及之后，保持 -100

            # 检查是否在 Observation span 内
            in_obs = False
            for obs_start, obs_end in obs_spans_in_full:
                if tok_start >= obs_start and tok_end <= obs_end:
                    in_obs = True
                    break

            if not in_obs:
                labels[token_idx] = input_ids[token_idx]  # 保留 loss

        # 统计 mask 比例
        total_assistant = sum(1 for l in labels if l != -100 or True)
        masked = sum(1 for l in labels if l == -100)
        active = len(labels) - masked

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "_stats": {
                "total_tokens": len(input_ids),
                "active_tokens": active,
                "masked_tokens": masked,
                "mask_ratio": masked / len(input_ids) if input_ids else 0,
            }
        }

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        item = self.processed[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }

    def print_stats(self):
        """打印数据集统计"""
        total_tokens = sum(p["_stats"]["total_tokens"] for p in self.processed)
        active_tokens = sum(p["_stats"]["active_tokens"] for p in self.processed)
        masked_tokens = sum(p["_stats"]["masked_tokens"] for p in self.processed)
        lengths = [p["_stats"]["total_tokens"] for p in self.processed]

        print(f"\n{'='*50}")
        print(f"数据集统计")
        print(f"{'='*50}")
        print(f"样本数: {len(self.processed)}")
        print(f"总 token 数: {total_tokens:,}")
        print(f"有效 token（计算loss）: {active_tokens:,} ({active_tokens/total_tokens:.1%})")
        print(f"Masked token（Obs+prompt）: {masked_tokens:,} ({masked_tokens/total_tokens:.1%})")
        print(f"序列长度: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")


# ============ 训练 ============

def train(args):
    logger.info("=" * 60)
    logger.info("FinAgent SFT 训练")
    logger.info("=" * 60)

    # 1. 加载 tokenizer
    logger.info(f"[1/5] 加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 构建 system prompt
    logger.info("[2/5] 构建 system prompt")
    try:
        system_prompt = get_system_prompt()
        logger.info("  从 08_prompts.py 加载成功")
    except Exception as e:
        logger.warning(f"  无法从 08_prompts.py 加载 ({e})，使用内置 fallback")
        system_prompt = SYSTEM_PROMPT_FALLBACK

    # 3. 构建数据集
    logger.info(f"[3/5] 构建数据集: {args.data_path}")
    dataset = SFTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_length=args.max_length,
    )
    dataset.print_stats()

    # 验证 loss mask
    logger.info("\n[验证] Loss mask 样本检查:")
    sample = dataset.processed[0]
    tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
    labels = sample["labels"]
    # 找连续的 masked/active 区间
    active_segments = []
    current = None
    for i, (tok, lab) in enumerate(zip(tokens, labels)):
        is_active = lab != -100
        if current is None or current["active"] != is_active:
            if current:
                active_segments.append(current)
            current = {"active": is_active, "start": i, "count": 1}
        else:
            current["count"] += 1
    if current:
        active_segments.append(current)

    for seg in active_segments[:15]:
        status = "LEARN" if seg["active"] else "MASK "
        start_tok = tokens[seg["start"]] if seg["start"] < len(tokens) else "?"
        print(f"  {status} | {seg['count']:>4d} tokens | starts: {start_tok[:20]}")

    if args.dry_run:
        logger.info("\n[dry-run] 数据预处理检查完成，不执行训练")
        return

    # 4. 加载模型 + LoRA
    logger.info(f"\n[4/5] 加载模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()  # LoRA 需要

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 训练
    logger.info(f"\n[5/5] 开始训练")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # 保存 LoRA adapter
    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"\nLoRA adapter 已保存: {final_path}")
    logger.info("训练完成")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="FinAgent SFT 训练")
    parser.add_argument("--model_path", type=str,
                        default="./models/Qwen2.5-14B-Instruct",
                        help="基座模型路径")
    parser.add_argument("--data_path", type=str,
                        default="./data/sft/sft_data_train.jsonl",
                        help="训练数据路径")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/sft_lora",
                        help="输出目录")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="最大序列长度")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="每卡 batch size")
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="梯度累积步数（等效 batch=16）")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                        help="LoRA alpha（通常为 2*rank）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只做数据预处理检查，不训练")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
