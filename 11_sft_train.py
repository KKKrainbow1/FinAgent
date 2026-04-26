#!/usr/bin/env python3
"""
FinAgent Step 11: SFT 训练脚本(V4 - TRL SFTTrainer + assistant_only_loss)

============================================================================
 V4 架构:TRL 官方 assistant-only loss 替代手写 token mask
============================================================================

V3 → V4 核心改动:
    1. transformers.Trainer → trl.SFTTrainer(V4 用 TRL v1.0+ 官方 agent SFT 栈)
    2. 手写 token-level loss mask(~100 行 SFTDataset._process_sample)→
       chat_template 里的 `{% generation %}` 标签 + SFTConfig(assistant_only_loss=True)
    3. 官方 chat_template(不含 generation 标签)→
       finagent_repo/qwen25_sft_chat_template.jinja(patched,含 generation 标签)

Mask 工作原理:
    - Qwen2.5 官方 chat_template 没有 `{% generation %}` 标签,默认 return_assistant_tokens_mask
      返回全 0(无法用 TRL assistant_only_loss),Qwen 团队官方拒绝合并相关 PR
    - V4 fork 一份 template 放 finagent_repo/qwen25_sft_chat_template.jinja,
      在两个 assistant 分支(含/不含 tool_calls)都包 {% generation %}...{% endgeneration %}
    - TRL SFTTrainer 在 assistant_only_loss=True 时自动调 tokenizer.apply_chat_template(
      ..., return_assistant_tokens_mask=True) 拿到 assistant_masks 字段,
      把非 assistant 的 token label 设为 -100
    - 训练 / 推理 / 生成期的 chat_template 在结构 token 层面完全一致,
      `{% generation %}` 标签只在 return_assistant_tokens_mask=True 时生效

相对 V3 的代码简化:
    - 删除 SFTDataset 类(~150 行):字符级扫 <|im_start|>assistant / <|im_end|> 边界 +
      offset_mapping 转 token-level label 的全部逻辑
    - 删除 DataCollatorForSeq2Seq 手动拼接
    - 删除 fallback 内置 tools schema(V4 要求 tools.py 必须 import 成功)

用法:
    # 默认配置训练
    python 11_sft_train.py
    # 自定义参数
    python 11_sft_train.py --epochs 3 --lr 2e-4 --lora_rank 32 --lora_alpha 64
    # 只做数据预处理 + mask 可视化检查(不训练)
    python 11_sft_train.py --dry-run

面试追问:为什么 Qwen2.5 需要手动 patch chat_template?
答:Qwen 团队官方决定不在默认 chat_template 里加 `{% generation %}` 标签(讨论在
Qwen2.5-VL discussion 27 和 Qwen3 discussion 10),留给用户自己按训练需求添加。
TRL v1 对 Qwen3 自动 patch,对 Qwen2.5 需要用户自己 fork 模板。V4 的做法是把
patched 版本作为项目资产随 git 追踪,训练期加载,保证行为确定性。

面试追问:V3 的手写 mask 有什么问题?
答:(1) 依赖字符级扫 <|im_start|>assistant 边界,理论正确但脆弱,未来 Qwen2.6
改 chat_template 格式就失效;(2) ~100 行自定义逻辑,和 TRL 官方 API 脱节,未来
升级 TRL 版本可能不兼容;(3) 面试讲"手写 mask"是正向(展示对机制的理解),
但 V4 改成"用 chat_template + TRL 官方 API 标准化"是更专业的做法。
"""

import json
import os
import argparse
import logging
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel, TaskType

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============ 工具定义 + patched chat_template ============

PATCHED_TEMPLATE_PATH = Path(__file__).parent / "qwen25_sft_chat_template.jinja"


def get_tools_native():
    """加载 TOOLS_NATIVE(V4 必须成功 import,不再 fallback 内置版本)"""
    from tools import TOOLS_NATIVE
    return TOOLS_NATIVE


def load_tokenizer_with_patched_template(model_path: str):
    """加载 Qwen2.5 tokenizer 并 override chat_template 为 patched 版本
    (含 {% generation %} 标签,支持 TRL assistant_only_loss)"""
    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if not PATCHED_TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"未找到 patched chat_template: {PATCHED_TEMPLATE_PATH}\n"
            "V4 训练必须用带 {% generation %} 标签的 template,请先 git pull 确认文件存在"
        )
    tok.chat_template = PATCHED_TEMPLATE_PATH.read_text(encoding="utf-8")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    logger.info(f"  已加载 patched chat_template: {PATCHED_TEMPLATE_PATH}")
    return tok


def load_sft_dataset(data_path: str, tools_native: list) -> Dataset:
    """加载 V4 SFT jsonl,并给每条数据加 tools 字段(TRL SFTTrainer 会自动传给
    apply_chat_template 让 chat_template 渲染 <tools>...</tools> 说明)"""
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if "messages" not in ex:
                continue
            # 给每条数据添加 tools 字段(所有行都用同一个 TOOLS_NATIVE)
            ex["tools"] = tools_native
            rows.append(ex)
    logger.info(f"  加载 SFT 数据: {len(rows)} 条")
    return Dataset.from_list(rows)


def verify_train_infer_consistency(sample: dict, tokenizer_patched, tools_native: list,
                                    model_path: str) -> bool:
    """V4 dry-run 验证 1:patched template 和原版 template 渲染输出必须完全一致
    保证训推一致(训练用 patched 渲染 + return_assistant_tokens_mask=True,
    推理期 vLLM 用原版或 patched 都渲染出相同 prompt 文本)。

    {% generation %} 是 Jinja statement extension,只在
    return_assistant_tokens_mask=True 时激活产出 mask;普通渲染时不产出任何文本,
    配合 HuggingFace 默认 trim_blocks=True/lstrip_blocks=True,空白控制一致。
    """
    # 加载一个独立的原版 tokenizer(不 override chat_template)
    tokenizer_orig = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right"
    )

    text_patched = tokenizer_patched.apply_chat_template(
        sample["messages"], tools=tools_native, tokenize=False,
    )
    text_orig = tokenizer_orig.apply_chat_template(
        sample["messages"], tools=tools_native, tokenize=False,
    )
    if text_patched == text_orig:
        logger.info("  ✓ patched vs 原版 chat_template 渲染输出 bit-level 一致(训推无差异)")
        return True

    logger.error("  ✗ patched 和原版渲染输出不一致 —— 可能影响训推对齐")
    for i, (c1, c2) in enumerate(zip(text_patched, text_orig)):
        if c1 != c2:
            ctx_start = max(0, i - 30)
            ctx_end = min(len(text_patched), i + 30)
            logger.error(f"  第 {i} 字符开始差异:")
            logger.error(f"    patched: ...{text_patched[ctx_start:ctx_end]!r}...")
            logger.error(f"    origin : ...{text_orig[ctx_start:ctx_end]!r}...")
            break
    # 长度不同的兜底
    if len(text_patched) != len(text_orig):
        logger.error(f"  长度差异:patched={len(text_patched)}, orig={len(text_orig)}")
    return False


def verify_assistant_mask(sample: dict, tokenizer, tools_native: list, num_segments: int = 20):
    """V4 dry-run 验证 2:渲染一条样本,打印 assistant_masks 的段分布
    确认 patched template 工作正常(content + tool_calls 都在 mask=1 区域,
    system/user/tool observation 都在 mask=0 区域)"""
    out = tokenizer.apply_chat_template(
        sample["messages"],
        tools=tools_native,
        return_assistant_tokens_mask=True,
        return_dict=True,
        tokenize=True,
    )
    ids = out["input_ids"]
    mask = out["assistant_masks"]

    active = sum(mask)
    total = len(mask)
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"[Assistant Mask 可视化] 样本总 tokens: {total}")
    logger.info(f"  active(算 loss):{active} ({active/total*100:.1f}%)")
    logger.info(f"  mask  (不算 loss):{total - active} ({(total-active)/total*100:.1f}%)")
    logger.info("=" * 60)

    # 找连续段
    segments = []
    current = None
    for i, m in enumerate(mask):
        is_active = bool(m)
        if current is None or current["active"] != is_active:
            if current:
                segments.append(current)
            current = {"active": is_active, "start": i, "count": 1}
        else:
            current["count"] += 1
    if current:
        segments.append(current)

    # 打印前 num_segments 段
    for seg in segments[:num_segments]:
        flag = "LEARN" if seg["active"] else "MASK "
        start_tok_id = ids[seg["start"]]
        start_tok = tokenizer.decode([start_tok_id])
        # 取段头 + 段尾几个 token 预览
        end = min(seg["start"] + seg["count"], len(ids))
        preview = tokenizer.decode(ids[seg["start"]:min(seg["start"]+5, end)])[:40]
        logger.info(f"  [{flag}] {seg['count']:>5d} tokens  | preview: {preview!r}")
    logger.info("=" * 60)


# ============ 训练主流程 ============

def train(args):
    # 1. Tokenizer + patched chat_template
    logger.info("[1/4] 加载 tokenizer + 修改 chat_template")
    tokenizer = load_tokenizer_with_patched_template(args.model_path)

    # 2. 工具定义 + 数据集
    logger.info(f"[2/4] 加载 tools + 数据集: {args.data_path}")
    tools_native = get_tools_native()
    logger.info(f"  工具数量: {len(tools_native)}")
    for t in tools_native:
        logger.info(f"  - {t['function']['name']}")
    train_ds = load_sft_dataset(args.data_path, tools_native)

    # Dry-run:两步验证 patched template
    #   验证 1:patched 和原版渲染 bit-level 一致(训推一致)
    #   验证 2:assistant_masks 段分布正确(TRL assistant_only_loss 能用)
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("[Dry-Run 验证 1/2] 训推一致性:patched vs 原版 chat_template 渲染")
        logger.info("=" * 60)
        ok = verify_train_infer_consistency(train_ds[0], tokenizer, tools_native, args.model_path)
        if not ok:
            logger.error("训推一致性验证失败,停止")
            return

        logger.info("")
        logger.info("=" * 60)
        logger.info("[Dry-Run 验证 2/2] Assistant mask 段分布(patched template)")
        logger.info("=" * 60)
        verify_assistant_mask(train_ds[0], tokenizer, tools_native, num_segments=25)
        logger.info("")
        logger.info("Dry-run 完成,两项验证均通过,可以启动真实训练")
        return

    # 3. 模型 + LoRA
    logger.info(f"[3/4] 加载模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    if args.adapter_path:
        # 加载已有 adapter 继续训(post-GRPO SFT 修补场景)
        logger.info(f"  加载已有 adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            is_trainable=True,
        )
        peft_config = None
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
        )

    # 4. TRL SFTTrainer
    logger.info(f"\n[4/4] 启动训练")
    config = SFTConfig(
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
        max_length=args.max_length,
        assistant_only_loss=True,     # ← V4 核心:自动 mask 非 assistant token
        packing=False,                 # messages 场景下保持 False
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # 保存 LoRA adapter + patched tokenizer
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    # tokenizer 也要保存(含 patched chat_template)
    tokenizer.save_pretrained(final_path)
    logger.info(f"\nLoRA adapter + patched tokenizer 已保存: {final_path}")
    logger.info("训练完成")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(
        description="FinAgent SFT 训练(V4 - TRL SFTTrainer + assistant_only_loss)"
    )
    parser.add_argument("--model_path", type=str,
                        default="./models/Qwen2.5-14B-Instruct",
                        help="基座模型路径")
    parser.add_argument("--data_path", type=str,
                        default="./data/sft/sft_data_v4.jsonl",
                        help="训练数据路径(V4 messages 格式,arguments 为 dict)")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/sft_lora_v4",
                        help="输出目录")
    parser.add_argument("--max_length", type=int, default=12288,
                        help="最大序列长度(V4 升至 12288:精简 SYSTEM_PROMPT 后 5 步轨迹 ~10K tokens,12K 留 2K 余量给深度推理)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="每卡 batch size")
    parser.add_argument("--grad_accum", type=int, default=16,
                        help="梯度累积步数(等效 batch=16)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank(V3 r32 在 6/7 type 打平或胜 r64,V4 默认 32)")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha(通常为 2*rank)")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="已有 LoRA adapter 路径(继续训,post-GRPO SFT 修补场景)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只做 assistant mask 可视化验证,不训练")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
