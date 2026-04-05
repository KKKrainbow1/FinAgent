#!/usr/bin/env python3
"""
FinAgent Step 11: SFT 训练脚本（V2 - Qwen 原生 Tool Calling + 结构化 Loss Mask）

V1 → V2 核心改动：
    1. 训练数据格式：从纯文本轨迹改为 OpenAI messages 格式
       每条数据包含 messages 列表，直接用 apply_chat_template(tools=TOOLS_NATIVE) 渲染
    2. Loss Mask 策略：从正则匹配 "Observation: ..." 改为按 message role 精确切分
       - role=system / role=user / role=tool → mask=0（不学习）
       - role=assistant（content + tool_calls）→ mask=1（学习）
       不再依赖正则，边界精确，不可能切错
    3. chat_template 自动注入工具定义（<tools>...</tools>），system prompt 不再手动拼接

用法：
    # 默认配置训练
    python 11_sft_train.py

    # 自定义参数
    python 11_sft_train.py \
        --model_path ./models/Qwen2.5-14B-Instruct \
        --data_path ./data/sft/sft_data_native.jsonl \
        --output_dir ./outputs/sft_lora_native \
        --epochs 3 \
        --lr 2e-4

    # 只做数据预处理检查（不训练）
    python 11_sft_train.py --dry-run

面试追问：为什么要 mask Observation（role=tool）？
答：Observation 是检索系统返回的真实数据，推理时由工具提供，不是模型该生成的。
如果不 mask，模型会花大量参数去"记忆"检索结果的格式和内容，
既浪费模型容量，又会导致推理时生成虚假的 Observation。
mask 之后，模型只学习：给定 Observation → 如何在 Thought 中分析它 →
选择什么工具和参数 → 最终如何生成分析报告。

面试追问：V2 的 loss mask 相比 V1 有什么优势？
答：V1 用正则匹配 "Observation: ..." 文本来确定 mask 边界，如果模型输出格式
稍有偏差（如多空格、换行变化），正则可能切错，导致 Observation 的 token 泄漏进
loss 计算。V2 基于 message role 切分，role=tool 的消息对应的 token 一定是
Observation，不可能切错。而且 chat_template 渲染后 role=tool 变成
<|im_start|>user\n<tool_response>...\n</tool_response><|im_end|>，
边界由特殊 token 和固定标签确定，非常精确。
"""

import json
import os
import sys
import argparse
import logging

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


# ============ 工具定义（训练时需要传入 chat_template） ============

def get_tools_native():
    """获取 TOOLS_NATIVE，用于 apply_chat_template 的 tools 参数"""
    try:
        from tools import TOOLS_NATIVE
        return TOOLS_NATIVE
    except ImportError:
        # fallback：内置工具定义
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_report",
                    "description": "检索券商研报信息，返回机构评级、目标价、EPS预测、行业分析等",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "研报检索关键词"},
                            "top_k": {"type": "integer", "description": "返回结果数量", "default": 3}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_financial",
                    "description": "检索公司财务数据，返回ROE、毛利率、营收增长率、资产负债率等指标",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "财务数据检索关键词"},
                            "top_k": {"type": "integer", "description": "返回结果数量", "default": 3}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_industry",
                    "description": "检索行业对比数据，返回同行业多家公司的关键指标对比表和行业均值",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "行业检索关键词"},
                            "top_k": {"type": "integer", "description": "返回结果数量", "default": 3}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "计算数学表达式。所有涉及数值计算的场景都必须使用此工具，禁止心算",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "纯数学表达式"},
                            "precision": {"type": "integer", "description": "小数精度位数", "default": 4}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]


# ============ Dataset（V2 - 结构化 Loss Mask） ============

class SFTDataset(Dataset):
    """
    SFT 训练数据集（V2 - messages 格式 + 结构化 loss mask）

    输入数据格式（10_generate_sft_data.py 生成的 sft_data_native.jsonl）：
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "分析贵州茅台..."},
            {"role": "assistant", "content": "思考...", "tool_calls": [...]},
            {"role": "tool", "tool_call_id": "call_0", "content": "检索结果..."},
            {"role": "assistant", "content": "继续思考...", "tool_calls": [...]},
            {"role": "tool", "tool_call_id": "call_1", "content": "检索结果..."},
            {"role": "assistant", "content": "最终分析报告..."}
        ],
        ...
    }

    Loss Mask 策略（按 message role 精确切分）：
        role=system    → mask=0（不学习）
        role=user      → mask=0（不学习）
        role=assistant → mask=1（学习 Thought + tool_calls + 最终回答）
        role=tool      → mask=0（不学习，这是 Observation）

    实现方式：
        1. 用 apply_chat_template 渲染完整序列（不含 labels 信息）
        2. 分别渲染「不含最后一条 assistant 消息」的前缀，找到每段 assistant 的起始 token 位置
        3. 利用渐进式渲染确定每条 assistant 消息的 token 范围，设为 loss active
    """

    def __init__(self, data_path: str, tokenizer, tools_native: list,
                 max_length: int = 7168):
        self.tokenizer = tokenizer
        self.tools_native = tools_native
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
        """
        处理单条样本：tokenize + 构建结构化 loss mask

        核心思路：渐进式渲染
        1. 渲染完整 messages 得到 full_tokens
        2. 逐步添加消息并渲染，确定每条消息对应的 token 范围
        3. role=assistant 的 token 范围设为 loss active，其余 mask
        """
        messages = sample.get("messages")
        if not messages:
            return None

        # 1. 渲染完整序列
        try:
            # 传入 tools 参数：让 chat_template 注入工具定义（<tools>...</tools>）
            # 确保训练和推理的输入格式一致，避免模型对 </tool_call> 停止符不稳定
            # 每条数据多 ~1000 token，但 RTX PRO 6000 96GB 显存足够
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tools=self.tools_native,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            logger.warning(f"apply_chat_template 失败: {e}")
            return None

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

        # 2. 确定每条 assistant 消息在 full_text 中的字符范围
        # 方法：使用 Qwen2.5 chat_template 的固定格式标记
        #   assistant 消息格式：<|im_start|>assistant\n{content}\n<tool_call>...\n</tool_call><|im_end|>
        #   tool 消息格式：<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>
        # 我们只需要找到所有 <|im_start|>assistant 到 <|im_end|> 的范围

        assistant_char_ranges = []
        search_start = 0
        while True:
            # 找 <|im_start|>assistant
            marker = "<|im_start|>assistant"
            pos = full_text.find(marker, search_start)
            if pos == -1:
                break

            # 找对应的 <|im_end|>
            content_start = pos + len(marker) + 1  # +1 for \n
            end_pos = full_text.find("<|im_end|>", content_start)
            if end_pos == -1:
                end_pos = len(full_text)

            # 记录 assistant 内容的字符范围（不含 <|im_start|>assistant\n 和 <|im_end|>）
            assistant_char_ranges.append((content_start, end_pos))
            search_start = end_pos + len("<|im_end|>")

        # 3. 构建 labels：assistant 内容区域 active，其余 mask
        labels = [-100] * len(input_ids)

        for token_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:
                continue  # special token，保持 -100

            # 检查 token 是否在某个 assistant 内容范围内
            for (a_start, a_end) in assistant_char_ranges:
                if tok_start >= a_start and tok_end <= a_end:
                    labels[token_idx] = input_ids[token_idx]
                    break

        # 统计
        active = sum(1 for l in labels if l != -100)
        masked = len(labels) - active

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "_stats": {
                "total_tokens": len(input_ids),
                "active_tokens": active,
                "masked_tokens": masked,
                "mask_ratio": masked / len(input_ids) if input_ids else 0,
                "num_assistant_segments": len(assistant_char_ranges),
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
        num_segments = [p["_stats"]["num_assistant_segments"] for p in self.processed]

        print(f"\n{'='*50}")
        print(f"数据集统计（V2 Native Tool Calling）")
        print(f"{'='*50}")
        print(f"样本数: {len(self.processed)}")
        print(f"总 token 数: {total_tokens:,}")
        print(f"有效 token（计算 loss）: {active_tokens:,} ({active_tokens/total_tokens:.1%})")
        print(f"Masked token（system+user+tool）: {masked_tokens:,} ({masked_tokens/total_tokens:.1%})")
        print(f"序列长度: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")
        print(f"Assistant 段数: min={min(num_segments)}, max={max(num_segments)}, avg={sum(num_segments)/len(num_segments):.1f}")


# ============ 训练 ============

def train(args):
    logger.info("=" * 60)
    logger.info("FinAgent SFT 训练（V2 - Native Tool Calling）")
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

    # 2. 加载工具定义
    logger.info("[2/5] 加载工具定义")
    tools_native = get_tools_native()
    logger.info(f"  工具数量: {len(tools_native)}")
    for t in tools_native:
        logger.info(f"  - {t['function']['name']}")

    # 3. 构建数据集
    logger.info(f"[3/5] 构建数据集: {args.data_path}")
    dataset = SFTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        tools_native=tools_native,
        max_length=args.max_length,
    )
    dataset.print_stats()

    # 验证 loss mask
    logger.info("\n[验证] Loss mask 样本检查:")
    if dataset.processed:
        sample = dataset.processed[0]
        tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
        labels = sample["labels"]

        # 找连续的 masked/active 区间
        segments = []
        current = None
        for i, (tok, lab) in enumerate(zip(tokens, labels)):
            is_active = lab != -100
            if current is None or current["active"] != is_active:
                if current:
                    segments.append(current)
                current = {"active": is_active, "start": i, "count": 1}
            else:
                current["count"] += 1
        if current:
            segments.append(current)

        for seg in segments[:20]:
            status = "LEARN" if seg["active"] else "MASK "
            start_tok = tokens[seg["start"]] if seg["start"] < len(tokens) else "?"
            # 显示该段的角色类型
            print(f"  {status} | {seg['count']:>4d} tokens | starts: {repr(start_tok[:30])}")

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
    parser = argparse.ArgumentParser(description="FinAgent SFT 训练（V2 Native Tool Calling）")
    parser.add_argument("--model_path", type=str,
                        default="./models/Qwen2.5-14B-Instruct",
                        help="基座模型路径")
    parser.add_argument("--data_path", type=str,
                        default="./data/sft/sft_data_native.jsonl",
                        help="训练数据路径（messages 格式）")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/sft_lora_native",
                        help="输出目录")
    parser.add_argument("--max_length", type=int, default=7168,
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
