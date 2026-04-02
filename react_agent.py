"""
FinAgent Step 9: ReAct Agent 核心（V2 - Qwen 原生 Tool Calling 版）
用途：实现 Thought → Action → Observation 的循环逻辑
依赖：06_hybrid_search.py, 07_tools.py, 08_prompts.py

运行方式：
    python 09_react_agent.py                                    # 交互模式
    python 09_react_agent.py --question "分析贵州茅台的投资价值"   # 单次提问
    python 09_react_agent.py --batch data/sft/test_questions.jsonl  # 批量测试

V1 → V2 改动说明：
    1. 推理时传入 tools 参数（TOOLS_NATIVE），由 chat_template 自动注入工具定义
    2. 模型输出解析：从正则解析 Thought/Action/Action Input 改为读取
       assistant.content（Thought）+ assistant.tool_calls（Action）
    3. Observation 返回方式：从拼接纯文本改为构造 role=tool 消息
    4. 轨迹结束判断：从检测 "Action: finish" 改为检测 assistant 消息无 tool_calls
    5. 移除 finish 工具，最终回答是普通 assistant 消息

面试追问：为什么不用 LangChain？
答：ReAct 循环本身只有几十行代码，自己写可以完全控制 prompt 格式、
轨迹结构、Observation 的 loss mask。LangChain 的抽象层反而会阻碍
后续 SFT/GRPO 训练时对轨迹格式的精确控制。

面试追问：V2 相比 V1 推理阶段有什么优势？
答：1. 不再需要正则解析，tool_calls 是结构化 JSON，解析零失败率
    2. 复用 Qwen2.5 预训练的 tool calling 先验，工具选择更准确
    3. 支持多参数（如 top_k），V1 只能传单个字符串
"""

import json
import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 最大交互轮数（防止死循环）
MAX_STEPS = 6


# ============ 模型推理（V2 - 原生 Tool Calling） ============

def load_model(model_path: str = None):
    """
    加载 Qwen2.5-14B-Instruct 模型（或 LoRA 微调后的版本）

    现阶段用原始 Instruct 模型跑 baseline，
    SFT 训练后替换为 LoRA adapter 即可。

    Args:
        model_path: 模型路径，None 则使用默认路径

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    if model_path is None:
        model_path = "./models/Qwen2.5-14B-Instruct"

    logger.info(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info(f"模型加载完成，设备: {model.device}")

    return model, tokenizer


def generate_next_step(model, tokenizer, messages: list[dict], tools: list[dict],
                       max_new_tokens: int = 512, temperature: float = 0.1) -> dict:
    """
    调用模型生成下一步

    模型输出两种情况：
    1. 工具调用：content=Thought + tool_calls=Action（继续循环）
    2. 最终回答：content=分析报告，无 tool_calls（结束循环）

    Args:
        model: 语言模型
        tokenizer: 分词器
        messages: 当前完整的 messages 列表
        tools: TOOLS_NATIVE 工具定义
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度

    Returns:
        解析后的 assistant 消息 dict：
        {
            "content": str or None,      # Thought 或最终回答
            "tool_calls": list or None,   # 工具调用列表
        }
    """
    # 使用 chat_template 渲染（传入 tools 参数）
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    # 只取新生成的部分
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # 解析模型输出
    return parse_native_output(response)


def parse_native_output(response: str) -> dict:
    """
    解析 Qwen2.5 原生 tool calling 格式的模型输出

    模型输出示例（工具调用）：
        需要先获取贵州茅台的最新财务数据。
        <tool_call>
        {"name": "search_financial", "arguments": {"query": "贵州茅台 ROE 2024"}}
        </tool_call><|im_end|>

    模型输出示例（最终回答，无 tool_call）：
        贵州茅台2024年ROE高达36.99%，毛利率76.18%...<|im_end|>

    Returns:
        {
            "content": str or None,
            "tool_calls": [{"id": str, "type": "function",
                            "function": {"name": str, "arguments": str}}] or None,
        }
    """
    # 清理结尾特殊 token
    clean = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

    # 提取 tool_call 块
    tool_calls = []
    import re
    tool_call_pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
    matches = tool_call_pattern.findall(clean)

    for i, match in enumerate(matches):
        try:
            tc = json.loads(match)
            tool_calls.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                }
            })
        except json.JSONDecodeError:
            logger.warning(f"tool_call JSON 解析失败: {match[:200]}")

    # 提取 content（tool_call 标签之前的文本）
    content = tool_call_pattern.sub("", clean).strip()
    if not content:
        content = None

    return {
        "content": content,
        "tool_calls": tool_calls if tool_calls else None,
    }


# ============ ReAct 主循环（V2） ============

def run_agent(question: str, model, tokenizer, tools_executor, tools_schema: list[dict],
              max_steps: int = MAX_STEPS, verbose: bool = True) -> dict:
    """
    运行 ReAct Agent 主循环（V2 - 原生 Tool Calling）

    Args:
        question: 用户问题
        model: 语言模型
        tokenizer: 分词器
        tools_executor: FinAgentTools 实例（执行工具）
        tools_schema: TOOLS_NATIVE（工具 JSON Schema 定义）
        max_steps: 最大步数
        verbose: 是否打印每步详情

    Returns:
        {
            "question": str,
            "steps": [{"thought", "tool_name", "tool_arguments", "tool_call_id", "observation"}, ...],
            "final_answer": str or None,
            "finished": bool,
            "total_steps": int,
            "elapsed_seconds": float,
        }
    """
    from prompts import build_system_message

    # 初始化 messages
    messages = [
        build_system_message(),
        {"role": "user", "content": question},
    ]

    steps = []
    final_answer = None
    finished = False
    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}")

    for step_idx in range(max_steps):
        if verbose:
            print(f"\n--- Step {step_idx + 1} ---")

        # 1. 模型生成下一步
        assistant_output = generate_next_step(model, tokenizer, messages, tools_schema)

        content = assistant_output.get("content")
        tool_calls = assistant_output.get("tool_calls")

        # 2. 判断是否为最终回答（无 tool_calls）
        if not tool_calls:
            # 分离 Thought 和 Answer：训练数据中 finish 步格式是 "Thought\n\nAnswer"
            # 模型学到了这个模式，输出时 content 前半段是 Thought 总结，后半段是正式报告
            raw_content = content or ""
            if "\n\n" in raw_content:
                parts = raw_content.split("\n\n", 1)
                thought_part = parts[0]
                answer_part = parts[1]
                # 验证：如果第一段以 Observation/根据/结合/综合 开头，大概率是 Thought
                thought_starters = ("Observation", "根据", "结合", "综合", "已获取", "基于",
                                    "通过", "从", "上述", "以上", "目前", "当前")
                if any(thought_part.strip().startswith(s) for s in thought_starters):
                    final_answer = answer_part
                else:
                    final_answer = raw_content  # 无法确认，保留完整内容
            else:
                final_answer = raw_content
            finished = True

            # 追加到 messages（保留完整 content 用于轨迹记录）
            messages.append({"role": "assistant", "content": raw_content})

            if verbose:
                print(f"[完成] Agent 输出最终回答")
                if content:
                    print(f"  回答: {content[:200]}...")
            break

        # 3. 有 tool_calls → 执行工具
        tc = tool_calls[0]  # 当前只支持单工具调用
        tc_func = tc["function"]
        tool_name = tc_func["name"]
        tool_call_id = tc.get("id", f"call_{step_idx}")

        # 解析参数
        try:
            tool_arguments = json.loads(tc_func["arguments"])
        except json.JSONDecodeError:
            tool_arguments = {"query": tc_func["arguments"]}
            logger.warning(f"tool arguments JSON 解析失败，回退为字符串: {tc_func['arguments'][:100]}")

        if verbose:
            print(f"  Thought: {(content or '(无)')[:150]}...")
            print(f"  Action: {tool_name}")
            print(f"  Arguments: {json.dumps(tool_arguments, ensure_ascii=False)}")

        # 4. 追加 assistant 消息到 messages
        assistant_msg = {"role": "assistant", "tool_calls": tool_calls}
        if content:
            assistant_msg["content"] = content
        messages.append(assistant_msg)

        # 5. 调用工具获取 Observation
        observation = tools_executor.call(tool_name, tool_arguments)

        if verbose:
            print(f"  Observation: {observation[:200]}...")

        # 6. 追加 tool 消息到 messages
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": observation,
        })

        # 7. 记录步骤
        steps.append({
            "thought": content or "",
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
            "tool_call_id": tool_call_id,
            "observation": observation,
        })

    elapsed = time.time() - start_time

    # 如果达到最大步数但没有结束
    if not finished:
        logger.warning(f"达到最大步数 {max_steps}，Agent 未输出最终回答")
        if verbose:
            print(f"\n[超时] 达到最大步数 {max_steps}，强制结束")

    result = {
        "question": question,
        "steps": steps,
        "final_answer": final_answer,
        "finished": finished,
        "total_steps": len(steps),
        "elapsed_seconds": elapsed,
    }

    # 打印最终报告
    if verbose and final_answer:
        print(f"\n{'='*60}")
        print("最终分析报告")
        print(f"{'='*60}")
        print(final_answer)
        print(f"\n[统计] {len(steps)} 步工具调用, {elapsed:.1f}秒")

    return result


# ============ 导出轨迹（供 SFT 数据使用） ============

def export_trajectory(result: dict) -> dict:
    """
    将 Agent 运行结果导出为 V2 SFT 训练数据格式（messages 格式）

    Returns:
        {
            "question": str,
            "steps": [{"thought", "tool_name", "tool_arguments", "tool_call_id", "observation"}, ...],
            "final_answer": str,
        }
    """
    return {
        "question": result["question"],
        "steps": result["steps"],
        "final_answer": result["final_answer"],
    }


# ============ 批量测试 ============

def run_batch(input_path: str, output_path: str, model, tokenizer, tools_executor,
              tools_schema: list[dict], max_steps: int = MAX_STEPS):
    """
    批量运行测试集

    读取 test_questions.jsonl（每行 {question, type}），
    逐条跑 Agent，结果追加写入 output_path。
    支持断点续跑：如果 output_path 已有结果，跳过已完成的条目。
    """
    # 读取测试集
    with open(input_path, "r", encoding="utf-8") as f:
        test_items = [json.loads(line) for line in f]

    # 断点续跑：检查已完成的条目
    done_questions = set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done_questions.add(r["question"])
        logger.info(f"发现已有结果 {len(done_questions)} 条，跳过已完成")
    except FileNotFoundError:
        pass

    total = len(test_items)
    pending = [item for item in test_items if item["question"] not in done_questions]
    logger.info(f"测试集共 {total} 条，待运行 {len(pending)} 条")

    with open(output_path, "a", encoding="utf-8") as f_out:
        for idx, item in enumerate(pending):
            question = item["question"]
            q_type = item.get("type", "unknown")
            completed = total - len(pending) + idx

            print(f"\n[{completed + 1}/{total}] ({q_type}) {question[:50]}...")

            result = run_agent(question, model, tokenizer, tools_executor,
                               tools_schema, max_steps=max_steps, verbose=False)

            # 写入结果
            output_record = {
                "question": question,
                "type": q_type,
                "steps": result["steps"],
                "final_answer": result["final_answer"],
                "finished": result["finished"],
                "total_steps": result["total_steps"],
                "elapsed_seconds": round(result["elapsed_seconds"], 1),
            }
            f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            f_out.flush()

            # 打印摘要
            status = "OK" if result["finished"] else "TIMEOUT"
            answer_len = len(result["final_answer"]) if result["final_answer"] else 0
            tools_used = [s["tool_name"] for s in result["steps"]]
            print(f"  {status} | {result['total_steps']}步 | {result['elapsed_seconds']:.1f}s | "
                  f"答案{answer_len}字 | 工具: {tools_used}")

    logger.info(f"批量运行完成，结果保存至 {output_path}")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="FinAgent ReAct Agent (V2 - Native Tool Calling)")
    parser.add_argument("--question", type=str, default=None,
                        help="单次提问模式")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型路径（默认 ./models/Qwen2.5-14B-Instruct）")
    parser.add_argument("--interactive", action="store_true",
                        help="交互模式（循环提问）")
    parser.add_argument("--batch", type=str, default=None,
                        help="批量模式：指定测试集路径（jsonl），如 data/sft/test_questions.jsonl")
    parser.add_argument("--output", type=str, default=None,
                        help="批量模式输出路径（默认在输入文件同目录生成 _results.jsonl）")
    args = parser.parse_args()

    # 1. 加载检索器
    from hybrid_search import FinAgentRetriever
    from tools import FinAgentTools, TOOLS_NATIVE

    retriever = FinAgentRetriever()
    retriever.load_index()

    # 2. 初始化工具
    tools_executor = FinAgentTools(retriever)

    # 3. 加载模型
    model, tokenizer = load_model(args.model_path)

    # 4. 运行
    if args.batch:
        # 批量模式
        output_path = args.output
        if output_path is None:
            output_path = args.batch.replace(".jsonl", "_results.jsonl")
        run_batch(args.batch, output_path, model, tokenizer,
                  tools_executor, TOOLS_NATIVE)

    elif args.question:
        # 单次提问
        run_agent(args.question, model, tokenizer, tools_executor, TOOLS_NATIVE)

    elif args.interactive:
        # 交互模式
        print("\nFinAgent 交互模式 V2（输入 quit 退出）")
        while True:
            question = input("\n请输入问题: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            run_agent(question, model, tokenizer, tools_executor, TOOLS_NATIVE)

    else:
        # 默认：运行几个测试问题
        test_questions = [
            "贵州茅台2024年的盈利能力怎么样？",
            "分析宁德时代的投资价值",
            "比较比亚迪和宁德时代的ROE",
        ]
        for q in test_questions:
            run_agent(q, model, tokenizer, tools_executor, TOOLS_NATIVE)
            print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
