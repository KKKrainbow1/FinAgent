"""
FinAgent Step 9: ReAct Agent 核心
用途：实现 Thought → Action → Observation 的循环逻辑
依赖：06_hybrid_search.py, 07_tools.py, 08_prompts.py

运行方式：
    python 09_react_agent.py                                    # 交互模式
    python 09_react_agent.py --question "分析贵州茅台的投资价值"   # 单次提问

设计要点：
    1. 模型生成 Thought + Action + Action Input
    2. 正则解析出 action 和 action_input
    3. 调用对应工具拿到 Observation
    4. 把 Observation 拼回历史，让模型继续生成
    5. 遇到 finish 或达到最大步数时结束

面试追问：为什么不用 LangChain？
答：ReAct 循环本身只有几十行代码，自己写可以完全控制 prompt 格式、
轨迹结构、Observation 的 loss mask。LangChain 的抽象层反而会阻碍
后续 SFT/GRPO 训练时对轨迹格式的精确控制。
"""

import re
import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 最大交互轮数（防止死循环）
MAX_STEPS = 6


# ============ 模型推理 ============

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


def generate_next_step(model, tokenizer, system_prompt: str, user_prompt: str,
                       max_new_tokens: int = 512, temperature: float = 0.1) -> str:
    """
    调用模型生成下一步（Thought + Action + Action Input）

    用 chat template 格式：
    [system] system_prompt
    [user] user_prompt（问题 + 历史轨迹）
    [assistant] 模型续写

    temperature 设低（0.1）因为推理阶段要确定性，
    采样阶段（GRPO）再调高。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
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
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()


# ============ 解析模型输出 ============

def parse_agent_output(text: str) -> dict:
    """
    从模型输出中解析 Thought / Action / Action Input

    模型可能输出：
        Thought: 需要查询茅台的财务数据
        Action: search_financial
        Action Input: 贵州茅台 ROE 2024

    也可能格式不完美（多余空格、换行、冒号变体等），
    正则要尽量容错。

    Returns:
        {
            "thought": str,
            "action": str,
            "action_input": str,
            "parse_success": bool,
            "raw_output": str,
        }
    """
    result = {
        "thought": "",
        "action": "",
        "action_input": "",
        "parse_success": False,
        "raw_output": text,
    }

    # 解析 Thought（匹配到 Action 之前的所有内容）
    thought_match = re.search(
        r'Thought[:：]\s*(.*?)(?=\nAction[:：]|\Z)',
        text, re.DOTALL
    )
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # 解析 Action（工具名，单行）
    action_match = re.search(
        r'Action[:：]\s*(\S+)',
        text
    )
    if action_match:
        result["action"] = action_match.group(1).strip()

    # 解析 Action Input（可能多行，匹配到下一个 Thought 或文本末尾）
    input_match = re.search(
        r'Action\s*Input[:：]\s*(.*?)(?=\nThought[:：]|\nObservation[:：]|\Z)',
        text, re.DOTALL
    )
    if input_match:
        result["action_input"] = input_match.group(1).strip()

    # 判断解析是否成功
    if result["action"] and result["action_input"]:
        result["parse_success"] = True

    return result


# ============ ReAct 主循环 ============

def run_agent(question: str, model, tokenizer, tools, system_prompt: str,
              max_steps: int = MAX_STEPS, verbose: bool = True) -> dict:
    """
    运行 ReAct Agent 主循环

    Args:
        question: 用户问题
        model: 语言模型
        tokenizer: 分词器
        tools: FinAgentTools 实例
        system_prompt: 系统 prompt
        max_steps: 最大步数
        verbose: 是否打印每步详情

    Returns:
        {
            "question": str,
            "steps": [{"thought", "action", "action_input", "observation"}, ...],
            "final_answer": str or None,
            "finished": bool,          # 是否正常结束（调用了 finish）
            "total_steps": int,
            "elapsed_seconds": float,
        }
    """
    from prompts import build_user_prompt

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

        # 1. 拼接历史轨迹，构建 user prompt
        user_prompt = build_user_prompt(question, steps)

        # 2. 模型生成下一步
        raw_output = generate_next_step(model, tokenizer, system_prompt, user_prompt)

        if verbose:
            print(f"[模型输出]\n{raw_output}")

        # 3. 解析 Thought / Action / Action Input
        parsed = parse_agent_output(raw_output)

        if not parsed["parse_success"]:
            logger.warning(f"Step {step_idx+1}: 解析失败，原始输出: {raw_output[:200]}")
            # 解析失败时构造一个错误步骤，让模型有机会纠正
            steps.append({
                "thought": parsed.get("thought", "（解析失败）"),
                "action": "parse_error",
                "action_input": "",
                "observation": "[格式错误] 请严格按照 Thought/Action/Action Input 格式输出。",
            })
            if verbose:
                print("[!] 解析失败，提示模型重新生成")
            continue

        thought = parsed["thought"]
        action = parsed["action"]
        action_input = parsed["action_input"]

        if verbose:
            print(f"  Thought: {thought[:100]}...")
            print(f"  Action: {action}")
            print(f"  Action Input: {action_input[:100]}...")

        # 4. 检查是否结束
        if action == "finish":
            final_answer = action_input
            steps.append({
                "thought": thought,
                "action": action,
                "action_input": action_input,
            })
            finished = True
            if verbose:
                print(f"\n[完成] Agent 调用 finish，生成最终报告")
            break

        # 5. 调用工具获取 Observation
        observation = tools.call(action, action_input)

        if verbose:
            print(f"  Observation: {observation[:200]}...")

        # 6. 记录这一步
        steps.append({
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": observation,
        })

    elapsed = time.time() - start_time

    # 如果达到最大步数但没有 finish，强制结束
    if not finished:
        logger.warning(f"达到最大步数 {max_steps}，Agent 未调用 finish")
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
        print(f"\n[统计] {len(steps)} 步, {elapsed:.1f}秒")

    return result


# ============ 导出轨迹（供 SFT 数据使用） ============

def export_trajectory(result: dict) -> dict:
    """
    将 Agent 运行结果导出为 SFT 训练数据格式

    这是后续 SFT 数据构建的基础——Agent 跑通后，
    可以批量运行并导出轨迹作为训练数据的骨架。

    Returns:
        {
            "question": str,
            "steps": [{"thought", "action", "action_input", "observation"}, ...],
            "final_answer": str,
        }
    """
    return {
        "question": result["question"],
        "steps": result["steps"],
        "final_answer": result["final_answer"],
    }


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="FinAgent ReAct Agent")
    parser.add_argument("--question", type=str, default=None,
                        help="单次提问模式")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型路径（默认 ./models/Qwen2.5-14B-Instruct）")
    parser.add_argument("--interactive", action="store_true",
                        help="交互模式（循环提问）")
    args = parser.parse_args()

    # 1. 加载检索器
    from hybrid_search import FinAgentRetriever
    from tools import FinAgentTools
    from prompts import build_system_prompt

    retriever = FinAgentRetriever()
    retriever.load_index()

    # 2. 初始化工具
    tools = FinAgentTools(retriever)

    # 3. 构建 system prompt
    system_prompt = build_system_prompt(FinAgentTools.TOOL_DESCRIPTIONS)

    # 4. 加载模型
    model, tokenizer = load_model(args.model_path)

    # 5. 运行
    if args.question:
        # 单次提问
        result = run_agent(args.question, model, tokenizer, tools, system_prompt)

    elif args.interactive:
        # 交互模式
        print("\nFinAgent 交互模式（输入 quit 退出）")
        while True:
            question = input("\n请输入问题: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            result = run_agent(question, model, tokenizer, tools, system_prompt)

    else:
        # 默认：运行几个测试问题
        test_questions = [
            "贵州茅台2024年的盈利能力怎么样？",
            "分析宁德时代的投资价值",
            "比较比亚迪和宁德时代的ROE",
        ]
        for q in test_questions:
            result = run_agent(q, model, tokenizer, tools, system_prompt)
            print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
