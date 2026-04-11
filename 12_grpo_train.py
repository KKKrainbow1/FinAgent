#!/usr/bin/env python3
"""
FinAgent Step 12: GRPO 训练脚本

基于 TRL v1.0 GRPOTrainer + environment_factory，
在 SFT v3_native_r32 基础上进行 GRPO 训练。

Reward 设计：V4.1
  - tool_coverage / query_quality / calc_behavior / strategy_match
  - 可选 llm_judge（二元判断）
  - 格式不合规 → -1.0（前置硬约束）
  - DAPO overlong penalty（后处理）
  - 实现见 grpo_plugin.py

用法：
    # 正式训练
    python 12_grpo_train.py

    # 自定义参数
    python 12_grpo_train.py \
        --model_path ./models/Qwen2.5-14B-Instruct \
        --adapter_path ./outputs/sft_lora_v3_native_r32/final \
        --data_path ./data/sft/questions/grpo_questions.jsonl \
        --output_dir ./outputs/grpo_v1 \
        --num_generations 4 \
        --epochs 2

    # Dry-run（只加载模型和数据，不训练）
    python 12_grpo_train.py --dry-run

依赖：
    - trl >= 1.0.0（需要 environment_factory 支持）
    - transformers >= 5.2.0
    - peft
    - grpo_plugin.py（FinAgentEnv + finagent_reward）
"""

import os
import sys
import json
import argparse
import logging
from collections import defaultdict

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ============ 默认配置 ============

DEFAULTS = {
    "model_path": "./models/Qwen2.5-14B-Instruct",
    "adapter_path": "./outputs/sft_lora_v3_native_r32_v3/final",
    "data_path": "./data/grpo/grpo_questions.jsonl",
    "output_dir": "./outputs/grpo_v2",

    # GRPO 核心参数
    # V2: num_generations 4→8（C(8,2)=28 对 vs C(4,2)=6 对，GRPO 信号更准确）
    "num_generations": 8,
    "temperature": 0.9,
    "max_completion_length": 12288,
    "beta": 0.0,            # 不使用 KL 正则化（参考 ToolRL）

    # 训练参数
    # V2: lr 从 5e-7 提到 1e-6（V1 太低几乎不更新，Qwen2.5 DPO 用 7e-7，LoRA 需稍高）
    # V2: grad_accum 保持 8（TRL 要求 generation_batch_size 能被 num_generations 整除）
    # effective batch = 1 × 8 = 8 questions × 8 rollouts = 64 samples/step
    "learning_rate": 1e-6,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 2,
    "max_grad_norm": 0.5,
    "warmup_ratio": 0.05,

    # 监控
    "logging_steps": 5,
    "save_steps": 50,
    "log_completions": True,

    # Prompt 复用上限
    "max_prompt_reuse": 3,

    # vLLM（如果显存允许）
    "use_vllm": False,
    "vllm_gpu_memory_utilization": 0.5,
}


# ============ 数据加载 ============

def load_grpo_dataset(data_path: str) -> Dataset:
    """
    加载 GRPO 训练问题集。

    数据格式：
    {"question": "...", "type": "...", "source": "sft", "difficulty": "complex"}

    GRPOTrainer 需要的格式：
    {"prompt": [{"role": "user", "content": "..."}], "question": "...", "type": "..."}

    Args:
        data_path: grpo_questions.jsonl 路径
    """
    raw_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line.strip())
            raw_data.append(d)

    logger.info(f"加载 {len(raw_data)} 条 GRPO 问题")

    # 统计分布
    type_counts = defaultdict(int)
    diff_counts = defaultdict(int)
    for d in raw_data:
        type_counts[d.get("type", "unknown")] += 1
        diff_counts[d.get("difficulty", "unknown")] += 1

    logger.info("问题类型分布:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {t}: {c}")
    logger.info(f"难度分布: {dict(diff_counts)}")

    # 构建 HuggingFace Dataset
    # prompt 字段：GRPOTrainer 用来生成 rollout 的输入
    # question / type 字段：传给 reward 函数的 kwargs
    dataset_dict = {
        "prompt": [],
        "question": [],
        "type": [],
    }

    for d in raw_data:
        # GRPOTrainer 的 prompt 格式：list of messages
        dataset_dict["prompt"].append(
            [{"role": "user", "content": d["question"]}]
        )
        dataset_dict["question"].append(d["question"])
        dataset_dict["type"].append(d.get("type", ""))

    dataset = Dataset.from_dict(dataset_dict)
    logger.info(f"Dataset 构建完成: {len(dataset)} 条")

    return dataset


# ============ 模型加载 ============

def load_model_and_tokenizer(model_path: str, adapter_path: str):
    """
    加载 base model + SFT LoRA adapter。

    GRPO 直接在 SFT 的 LoRA 上继续训练，不 merge、不重置。
    """
    logger.info(f"加载 base model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": 0},  # 单卡 96GB 不需要 auto 分片
        trust_remote_code=True,
    )

    logger.info(f"加载 SFT LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    # GRPO 需要 adapter 可训练
    model.enable_adapter_layers()
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()  # LoRA 参数用 float32 训练

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    logger.info(f"加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",  # GRPO 生成时需要 left padding
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ============ 训练回调 ============

class GRPOMetricsCallback(TrainerCallback):
    """
    自定义回调：定期输出 grpo_plugin 的监控指标。
    继承 TrainerCallback 以被 Trainer 正确调用。
    """

    def __init__(self, log_interval: int = 10):
        super().__init__()
        self.log_interval = log_interval
        self.step_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次 logging 时输出自定义指标 + 注入 TensorBoard"""
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            try:
                from grpo_plugin import get_and_reset_metrics
                metrics = get_and_reset_metrics()
                if metrics:
                    # 打印到终端
                    logger.info(f"[GRPO V4.1 指标] Step {state.global_step}:")
                    logger.info(f"  calculate_rate: {metrics.get('calculate_rate', 0):.3f}")
                    logger.info(f"  mental_math_rate: {metrics.get('mental_math_rate', 0):.3f}")
                    logger.info(f"  tool_call_count: {metrics.get('tool_call_count', 0):.1f}")
                    logger.info(f"  tool_coverage: {metrics.get('tool_coverage_scores', 0):.3f}")
                    logger.info(f"  query_quality: {metrics.get('query_quality_scores', 0):.3f}")
                    logger.info(f"  calc_behavior: {metrics.get('calc_behavior_scores', 0):.3f}")
                    logger.info(f"  strategy_match: {metrics.get('strategy_match_scores', 0):.3f}")
                    if metrics.get('llm_judge_scores', 0) > 0:
                        logger.info(f"  llm_judge: {metrics['llm_judge_scores']:.3f}")

                    # 注入 logs → TensorBoard 自动写入
                    if logs is not None:
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                logs[f"custom/{key}"] = value
            except Exception as e:
                logger.warning(f"获取自定义指标失败: {e}")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="FinAgent GRPO Training")

    # 模型和数据
    parser.add_argument("--model_path", default=DEFAULTS["model_path"])
    parser.add_argument("--adapter_path", default=DEFAULTS["adapter_path"])
    parser.add_argument("--data_path", default=DEFAULTS["data_path"])
    parser.add_argument("--output_dir", default=DEFAULTS["output_dir"])

    # GRPO 参数
    parser.add_argument("--num_generations", type=int, default=DEFAULTS["num_generations"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--max_completion_length", type=int, default=DEFAULTS["max_completion_length"])
    parser.add_argument("--beta", type=float, default=DEFAULTS["beta"])

    # 训练参数
    parser.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["per_device_train_batch_size"])
    parser.add_argument("--grad_accum", type=int, default=DEFAULTS["gradient_accumulation_steps"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["num_train_epochs"])
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULTS["max_grad_norm"])
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULTS["warmup_ratio"])

    # 监控
    parser.add_argument("--logging_steps", type=int, default=DEFAULTS["logging_steps"])
    parser.add_argument("--save_steps", type=int, default=DEFAULTS["save_steps"])

    # vLLM
    parser.add_argument("--use_vllm", action="store_true", default=DEFAULTS["use_vllm"])

    # 其他
    parser.add_argument("--max_prompt_reuse", type=int, default=DEFAULTS["max_prompt_reuse"])
    parser.add_argument("--dry-run", action="store_true", help="只加载模型和数据，不训练")
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 恢复训练，如 ./outputs/grpo_v1/checkpoint-250")

    args = parser.parse_args()

    # ---- 检查依赖版本 ----
    try:
        import trl
        logger.info(f"TRL version: {trl.__version__}")
        from trl import GRPOTrainer, GRPOConfig
    except ImportError:
        logger.error("TRL 未安装或版本过低。需要 trl >= 1.0.0")
        logger.error("请执行: pip install trl>=1.0.0")
        sys.exit(1)

    # ---- 加载数据 ----
    if args.max_prompt_reuse < 1:
        logger.error("--max_prompt_reuse 必须 >= 1")
        sys.exit(1)

    effective_epochs = min(args.epochs, args.max_prompt_reuse)
    if effective_epochs < args.epochs:
        logger.warning(
            "epochs=%s 超过 max_prompt_reuse=%s，"
            "为保证每个 prompt 的累计使用次数不超过上限，训练轮数将按 %s 执行",
            args.epochs, args.max_prompt_reuse, effective_epochs,
        )

    dataset = load_grpo_dataset(args.data_path)

    # ---- 加载模型 ----
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.adapter_path)

    # ---- 导入 reward 和环境 ----
    from grpo_plugin import FinAgentEnv, finagent_reward

    # ---- 手动设置 Qwen2.5 的 response_schema ----
    # TRL v1.0 的 environment_factory 只自动识别 Qwen3/3.5 的 chat template。
    # 但 Qwen2.5 和 Qwen3 用相同的 <tool_call> XML 格式，可以复用 qwen3_schema。
    # 同时 monkey-patch add_response_schema 防止它覆盖我们手动设置的值。
    from trl.chat_template_utils import qwen3_schema
    tokenizer.response_schema = qwen3_schema
    import trl.trainer.grpo_trainer
    trl.trainer.grpo_trainer.add_response_schema = lambda x: x

    # ---- parse_response 超时保护 ----
    # qwen3_schema 的正则在某些 Qwen2.5 长输出上会触发灾难性回溯，导致 re.search 死循环。
    # 用 signal.alarm 包一层 10 秒超时，超时返回空 dict（等同于"无 tool_calls"）。
    import signal
    import functools
    import trl.chat_template_utils

    _original_parse_response = trl.chat_template_utils.parse_response

    def _timeout_handler(signum, frame):
        raise TimeoutError("parse_response regex timeout")

    @functools.wraps(_original_parse_response)
    def _safe_parse_response(tokenizer_obj, ids):
        # signal.alarm 只在主线程可用，子线程调用会抛 ValueError
        # 如果不是主线程，直接调用原函数（退化为无超时保护，但至少不崩）
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(10)  # 10 秒超时
        except ValueError:
            return _original_parse_response(tokenizer_obj, ids)

        try:
            return _original_parse_response(tokenizer_obj, ids)
        except TimeoutError:
            logger.warning(f"parse_response 超时，返回空 dict (ids len={len(ids) if ids else 0})")
            return {}
        finally:
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass

    trl.chat_template_utils.parse_response = _safe_parse_response
    trl.trainer.grpo_trainer.parse_response = _safe_parse_response

    # ---- 配置 GRPO ----
    effective_batch_size = max(args.batch_size * args.grad_accum, 1)
    steps_per_epoch = max((len(dataset) + effective_batch_size - 1) // effective_batch_size, 1)
    total_train_steps = max(steps_per_epoch * effective_epochs, 1)
    warmup_steps = int(args.warmup_ratio * total_train_steps)
    if args.warmup_ratio > 0 and warmup_steps == 0:
        warmup_steps = 1

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,

        # GRPO 核心
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=6,  # ReAct 最多 6 轮工具调用（和推理时一致）
        beta=args.beta,

        # 训练
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=effective_epochs,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=warmup_steps,

        # 监控
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="tensorboard",
        log_completions=True,
        # logging_dir 通过环境变量 TENSORBOARD_LOGGING_DIR 设置

        # 其他
        bf16=True,
        remove_unused_columns=False,  # 保留 question/type 字段传给 reward
    )

    logger.info("GRPO 配置:")
    logger.info(f"  num_generations: {args.num_generations}")
    logger.info(f"  temperature: {args.temperature}")
    logger.info(f"  max_completion_length: {args.max_completion_length}")
    logger.info(f"  beta: {args.beta}")
    logger.info(f"  learning_rate: {args.lr}")
    logger.info(f"  effective_batch_size: {args.batch_size * args.grad_accum}")
    logger.info(f"  epochs: {effective_epochs}")
    logger.info(f"  warmup_steps: {warmup_steps}")
    logger.info(f"  output_dir: {args.output_dir}")

    if args.dry_run:
        logger.info("=== DRY-RUN 模式 ===")
        logger.info("模型和数据加载成功，检查通过。")
        logger.info(f"Dataset 样本数: {len(dataset)}")
        logger.info(f"Dataset 字段: {dataset.column_names}")
        logger.info(f"样本示例: {dataset[0]}")

        # 测试 reward 函数
        logger.info("测试 reward 函数...")
        test_completions = ["这是一个测试回答"]
        test_env = FinAgentEnv()
        test_env.reset()
        try:
            test_rewards = finagent_reward(
                completions=test_completions,
                environments=[test_env],
                question=["测试问题"],
                type=["financial_query"],
            )
            logger.info(f"Reward 函数测试通过: {test_rewards}")
        except Exception as e:
            logger.error(f"Reward 函数测试失败: {e}")

        logger.info("=== DRY-RUN 完成 ===")
        return

    # ---- 创建 Trainer ----
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[finagent_reward],
        environment_factory=FinAgentEnv,
        args=grpo_config,
        callbacks=[GRPOMetricsCallback()],
    )

    # ---- 开始训练 ----
    logger.info("=" * 60)
    logger.info("开始 GRPO 训练")
    logger.info("=" * 60)

    if args.resume:
        logger.info(f"从 checkpoint 恢复: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ---- 保存最终模型 ----
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"最终模型已保存到: {final_dir}")

    # ---- 输出最终指标 ----
    from grpo_plugin import get_and_reset_metrics
    final_metrics = get_and_reset_metrics()
    logger.info("最终自定义指标:")
    for k, v in final_metrics.items():
        logger.info(f"  {k}: {v}")

    logger.info("GRPO 训练完成！")


if __name__ == "__main__":
    main()
