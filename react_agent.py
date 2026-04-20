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
import os
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

    # 检测是否为 LoRA adapter
    import os
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)
    base_model = model_path

    if is_lora:
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_model = adapter_cfg.get("base_model_name_or_path", "./models/Qwen2.5-14B-Instruct")
        logger.info(f"  LoRA adapter 检测到, base model: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
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

    支持两种模式：API 模式（调 vLLM server）和 HF 模式（本地 generate）。
    """
    # 判断推理模式
    is_api = isinstance(model, dict) and model.get('_finagent_api_mode')

    if is_api:
        # API 模式：调 vLLM server 的 OpenAI 兼容接口
        client = model['_api_client']
        api_model = model['_api_model']

        response = client.chat.completions.create(
            model=api_model,
            messages=messages,
            tools=tools,
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0.01,
            top_p=0.9,
        )

        msg = response.choices[0].message

        # 构建返回格式
        result = {"content": msg.content, "tool_calls": None}
        if msg.tool_calls:
            result["tool_calls"] = [{
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            } for tc in msg.tool_calls]

        # 兜底：如果 server 没解析出 tool_calls 但 content 中有 <tool_call>，
        # 用 parse_native_output 从 content 中提取
        if not result["tool_calls"] and msg.content and "<tool_call>" in msg.content:
            logger.info("  API 返回 tool_calls 为空但 content 含 <tool_call>，用 parse_native_output 兜底")
            result = parse_native_output(msg.content)

        return result

    else:
        # HF generate
        text = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
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
        # 清理嵌套的 tool_call 标签（模型过拟合时可能重复输出）
        match_clean = match.replace("<tool_call>", "").replace("</tool_call>", "").strip()
        try:
            tc = json.loads(match_clean)
            tool_calls.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                }
            })
            break  # 只取第一个成功解析的 tool_call
        except json.JSONDecodeError:
            logger.warning(f"tool_call JSON 解析失败: {match_clean[:200]}")

    # 如果正则没匹配到，尝试直接从文本中提取 JSON（兜底）
    if not tool_calls:
        json_pattern = re.compile(r'\{"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})', re.DOTALL)
        json_match = json_pattern.search(clean)
        if json_match:
            try:
                name = json_match.group(1)
                args = json.loads(json_match.group(2))
                tool_calls.append({
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    }
                })
                logger.info(f"  兜底提取 tool_call: {name}")
            except json.JSONDecodeError:
                pass

    # 提取 content（tool_call 标签之前的文本）
    content = tool_call_pattern.sub("", clean).strip()
    # 清理残留的 tool_call JSON
    content = re.sub(r'\{"name"\s*:.*$', '', content, flags=re.DOTALL).strip()
    if not content:
        content = None

    return {
        "content": content,
        "tool_calls": tool_calls if tool_calls else None,
    }


# ============ ReAct 主循环（V2） ============

def run_agent(question: str, model, tokenizer, tools_executor, tools_schema: list[dict],
              max_steps: int = MAX_STEPS, verbose: bool = True, temperature: float = 0.1) -> dict:
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

        # 1. 模型生成下一步（最后一步给更多 token 用于生成完整回答）
        is_last_possible_step = (step_idx == max_steps - 1) or (step_idx >= 2)
        max_tokens = 1500 if is_last_possible_step else 512
        assistant_output = generate_next_step(model, tokenizer, messages, tools_schema,
                                              max_new_tokens=max_tokens, temperature=temperature)

        content = assistant_output.get("content")
        tool_calls = assistant_output.get("tool_calls")

        # 2. 判断是否为最终回答（无 tool_calls）
        # 防止 tool_call 解析失败时把残缺内容当成 final_answer
        raw_text = content or ""
        if not tool_calls and ("<tool_call>" in raw_text or '"name": "search_' in raw_text or '"name": "calculate"' in raw_text):
            logger.warning(f"Step {step_idx+1}: 检测到 tool_call 残留但解析失败，跳过本步")
            # 构造一个错误步骤让模型重试
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({
                "role": "tool",
                "tool_call_id": f"error_{step_idx}",
                "content": "[格式错误] 工具调用格式不正确，请重新生成。",
            })
            continue

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

        # 解析参数（兜底处理双重转义和字符串类型）
        try:
            tool_arguments = json.loads(tc_func["arguments"])
            # 双重转义兜底：json.loads 后得到 str 而非 dict，再 loads 一次
            if isinstance(tool_arguments, str):
                try:
                    tool_arguments = json.loads(tool_arguments)
                except json.JSONDecodeError:
                    tool_arguments = {"query": tool_arguments}
            # 最终确保是 dict
            if not isinstance(tool_arguments, dict):
                tool_arguments = {"query": str(tool_arguments)}
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

        # 5. 调用工具获取 Observation + retrieval 元信息
        # tuple 返回:observation 给 LLM,retrieved 给 trajectory 记录(search_* 填 list,calculate 是 None)
        # 用于 bad case 追溯:重建索引后 chunk_id 稳定,可重回查具体 chunk 看数据质量
        observation, retrieved = tools_executor.call(tool_name, tool_arguments)

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
            "retrieved": retrieved,   # list[{chunk_id,score,source_type,...}] 或 None
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

    # 判断是否为 API 模式（可并发）
    is_api = isinstance(model, dict) and model.get('_finagent_api_mode')
    num_workers = 8 if is_api else 1  # API 模式并发 8 条，HF 模式串行

    if num_workers > 1:
        logger.info(f"API 模式：启用 {num_workers} 并发推理")
        _run_batch_parallel(pending, total, output_path, model, tokenizer,
                           tools_executor, tools_schema, max_steps, num_workers)
    else:
        _run_batch_serial(pending, total, output_path, model, tokenizer,
                         tools_executor, tools_schema, max_steps)

    logger.info(f"批量运行完成，结果保存至 {output_path}")


def _run_batch_serial(pending, total, output_path, model, tokenizer,
                      tools_executor, tools_schema, max_steps):
    """串行推理（HF generate 模式）"""
    with open(output_path, "a", encoding="utf-8") as f_out:
        for idx, item in enumerate(pending):
            question = item["question"]
            q_type = item.get("type", "unknown")
            completed = total - len(pending) + idx

            print(f"\n[{completed + 1}/{total}] ({q_type}) {question[:50]}...")

            result = run_agent(question, model, tokenizer, tools_executor,
                               tools_schema, max_steps=max_steps, verbose=False)

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

            status = "OK" if result["finished"] else "TIMEOUT"
            answer_len = len(result["final_answer"]) if result["final_answer"] else 0
            tools_used = [s["tool_name"] for s in result["steps"]]
            print(f"  {status} | {result['total_steps']}步 | {result['elapsed_seconds']:.1f}s | "
                  f"答案{answer_len}字 | 工具: {tools_used}")


def _run_batch_parallel(pending, total, output_path, model, tokenizer,
                        tools_executor, tools_schema, max_steps, num_workers):
    """并发推理（API 模式，多条问题同时走各自的 ReAct 循环）"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    write_lock = threading.Lock()
    completed_count = [total - len(pending)]  # 用 list 方便在闭包中修改

    def _process_one(item):
        question = item["question"]
        q_type = item.get("type", "unknown")

        # 每个线程需要独立的 tools_executor（检索器是线程安全的，共享即可）
        result = run_agent(question, model, tokenizer, tools_executor,
                           tools_schema, max_steps=max_steps, verbose=False)

        output_record = {
            "question": question,
            "type": q_type,
            "steps": result["steps"],
            "final_answer": result["final_answer"],
            "finished": result["finished"],
            "total_steps": result["total_steps"],
            "elapsed_seconds": round(result["elapsed_seconds"], 1),
        }
        return output_record

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_one, item): item for item in pending}

        with open(output_path, "a", encoding="utf-8") as f_out:
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                    with write_lock:
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f_out.flush()
                        completed_count[0] += 1

                    status = "OK" if record["finished"] else "TIMEOUT"
                    answer_len = len(record["final_answer"]) if record["final_answer"] else 0
                    tools_used = [s["tool_name"] for s in record["steps"]]
                    print(f"  [{completed_count[0]}/{total}] {status} | "
                          f"{record['total_steps']}步 | {record['elapsed_seconds']:.1f}s | "
                          f"答案{answer_len}字 | 工具: {tools_used} | {item['question'][:30]}...")
                except Exception as e:
                    logger.error(f"推理失败: {item['question'][:50]}... | {e}")
                    with write_lock:
                        completed_count[0] += 1


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
    parser.add_argument("--use_api", type=str, default=None,
                        help="使用 vLLM API server 推理，传入 server URL（如 http://localhost:8000）")
    parser.add_argument("--api_model", type=str, default="sft_adapter",
                        help="API server 的模型名（默认 sft_adapter）")
    args = parser.parse_args()

    # 1. 加载检索器
    from hybrid_search import FinAgentRetriever
    from tools import FinAgentTools, TOOLS_NATIVE

    retriever = FinAgentRetriever()

    # 2. 初始化工具
    tools_executor = FinAgentTools(retriever)

    # 3. 加载模型
    if args.use_api:
        # API 模式：不加载本地模型，用 vLLM server
        from openai import OpenAI
        api_client = OpenAI(base_url=f"{args.use_api}/v1", api_key="dummy")
        model = {"_api_client": api_client, "_api_model": args.api_model,
                 "_finagent_api_mode": True}
        # tokenizer 仍需加载（用于 apply_chat_template）
        from transformers import AutoTokenizer
        base_model_path = args.model_path or "./models/Qwen2.5-14B-Instruct"
        # 如果是 LoRA adapter 路径，从 adapter_config 找 base model
        adapter_cfg_path = os.path.join(base_model_path, "adapter_config.json") if base_model_path else ""
        if os.path.exists(adapter_cfg_path):
            with open(adapter_cfg_path) as f:
                _cfg = json.load(f)
            base_model_path = _cfg.get("base_model_name_or_path", base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        logger.info(f"API 模式：server={args.use_api}, model={args.api_model}")
    else:
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
