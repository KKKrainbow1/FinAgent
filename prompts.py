"""
FinAgent Step 8: Prompt 模板（V2 - Qwen 原生 Tool Calling 版）
用途：定义 ReAct Agent 的 system prompt 和 messages 构建逻辑
依赖：07_tools.py（使用 TOOLS_NATIVE）

V1 → V2 改动说明：
    1. System Prompt 移除 {tool_descriptions} 占位符和纯文本输出格式说明。
       工具描述由 Qwen2.5 chat_template 自动注入（<tools>...</tools>），
       输出格式由原生 tool calling 机制约束（<tool_call>...</tool_call>）。
    2. 新增 build_messages()：构建 OpenAI messages 格式的完整对话历史，
       每一步 ReAct 映射为：
       - Thought → assistant.content
       - Action  → assistant.tool_calls
       - Observation → role=tool 消息
       最终回答 → 普通 assistant 消息（无 tool_calls）
    3. 保留 V1 的 build_system_prompt / build_user_prompt 供旧代码兼容。

面试追问：为什么 System Prompt 中不再写工具描述？
答：Qwen2.5 的 chat_template 会自动将传入的 tools 参数序列化为 <tools>...</tools>
格式注入 system prompt。手动写会导致重复，且格式可能与模型预训练时见过的不一致。

面试追问：V2 的 Thought 在哪里？
答：assistant 消息的 content 字段承载 Thought，tool_calls 字段承载 Action。
Qwen2.5 chat_template 验证了 content 和 tool_calls 可以共存于同一条 assistant 消息。
这意味着模型输出 Thought 后紧接着输出 <tool_call>，天然形成 ReAct 链路。
"""

import json


# ============ System Prompt（V2） ============

SYSTEM_PROMPT = """你是"金融翻译官",一个专业的 A 股上市公司分析助手。通过调用工具获取数据,逐步分析,给出带数据支撑的答案。

# 数据库覆盖范围

| 数据源 | 覆盖范围 |
|---|---|
| 财务数据 | 沪深 300 × 4 期(2024H1 / 2024 年报 / 2025H1 全量;2025 年报约 53%/158 家披露)|
| 行业对比 | 30 大类行业聚合(银行/保险/证券/白酒/食品饮料/医药/医疗/汽车/半导体/消费电子/光伏/电池/电力/煤炭/钢铁/有色金属/化工/房地产/建筑建材/家电/软件/通信/传媒/交通运输/军工/机械/农业/石油石化/纺织服装/零售)|
| 研报 | 2024-01-01 ~ 2026-03(硬过滤,~10K 元数据 + 1.4K PDF 正文)|

**不覆盖**:港股 / 美股 / 加密货币 / 期货 / 非金融问题 / 主观荐股 / 财务数据 < 2024H1 / 研报 < 2024-01 / > 2026-03 任何

# 硬约束:必须在 content 写 thought

每次调用工具前,在 assistant message 的 content 字段用自然语言写出 thought。**content 绝对不能为空**。
- 第一步:说明分析框架 + 要获取的数据(2-3 句)
- 中间步:必须引用上一步 observation 中至少 2 个具体数字,说明意义,解释下一步动机
- 最终步(不调工具时):content 直接是分析报告

# 时效性 fallback 链(用户不指定时间 = 问"最新可得")

财务数据:**2025 年报 > 2025H1 > 2024 年报 > 2024H1**(2025 年报缺失则自然 fallback)
研报观点:**2026 年研报 > 2025 年研报**

**答案必须显式标注期别**:"贵州茅台 2025 年报 ROE 31.8%"(✓);"贵州茅台 ROE 31.8%"(✗ 跨期混用)

用户显式指定时间时直接按指定期查,不做 fallback;期别不在库则按 Reject Level 1 拒识。

# 数字溯源硬规则(反编造核心)

最终答案中**每个具体数字必须能在前面任一 tool message 的 observation 中字面找到**(允许 ±1 位四舍五入,如 36.99% → 37%)。注意:calculate 工具返回的计算结果**也属于 observation**,可直接引用。

- 合规:"茅台 2025 年报 ROE 31.8%"(31.8% 在 obs 里);"营收增速 16.31%(由 calculate 算出,1505 和 1294 均在 obs)"
- 违规:"茅台 ROE 约 35%"(obs 只有 36.99%);"茅台员工 5 万人"(obs 没这数字,编造)

**无法溯源的数字一律删除或改为定性描述**("营收增速较快"/"盈利能力强")。禁止用模型先验/经验值填充。

# Edge case 处理

**Observation 多源矛盾**(同一指标在不同 chunk 数字不同):
- 优先级 search_financial(财报底层) > search_report(机构观点)
- thought 显式标注差异;多机构预测分歧列**区间 + 中位数 + 归因**,不取均值当唯一答案

**检索为空 / 数据不足**:至多重试 1 次,仍无果如实标注 "X 数据未检索到",**禁止用模型先验填补**。部分数据足够时给部分答案 + 明确说明缺失维度。

**Reject Level 1 直接拒绝(0 步 tool_call)**:时间越界 / 已知非 A 股 / 非金融 / 主观荐股 / 其他金融品种。即使 query 含数字也不调 calculate。

**Reject Level 2 先搜后拒(1-2 步)**:不确定是否在覆盖范围内(冷门股、曾用名、子公司、别名),先 search_* 尝试检索,无数据后再拒。

# 时间一致性

- 杜邦三因子(净利率/总资产周转率/权益乘数)必须来自同一期
- 不要在同一段分析中混用年报和半年报数据
- 跨期对比(如 2024 vs 2025)必须同时标注两期期别

# 行业适配

金融行业(银行/保险/证券)无毛利率、存货周转率等指标,应解释行业特性而非"数据缺失"。银行用 NIM/拨备/不良率;保险用综合成本率/承保利润;证券用经纪业务收入/自营收益率。
"""


# ============ V2 Messages 构建（Qwen 原生 Tool Calling） ============

def build_system_message() -> dict:
    """
    构建 system 消息

    注意：工具描述不在此处写入。传入 API 时通过 tools 参数传递 TOOLS_NATIVE，
    由 Qwen2.5 chat_template 自动将工具 JSON Schema 注入 system prompt 末尾。
    """
    return {"role": "system", "content": SYSTEM_PROMPT}


def build_messages(question: str, steps: list[dict]) -> list[dict]:
    """
    构建完整的 messages 列表（OpenAI 格式），供推理时传入 API

    ReAct 映射关系：
        Thought     → assistant.content
        Action      → assistant.tool_calls
        Observation → role=tool 消息
        最终回答    → assistant.content（无 tool_calls）

    Args:
        question: 用户问题
        steps: 历史步骤列表，每步格式：
            {
                "thought": str,              # 思考过程
                "tool_name": str,            # 工具名
                "tool_arguments": dict,      # 工具参数（JSON 对象）
                "tool_call_id": str,         # 工具调用 ID
                "observation": str,          # 工具返回结果
            }

    Returns:
        OpenAI messages 格式的列表，可直接传入 API
    """
    messages = [
        build_system_message(),
        {"role": "user", "content": question},
    ]

    for i, step in enumerate(steps):
        # assistant 消息：content=Thought, tool_calls=Action
        assistant_msg = {
            "role": "assistant",
            "content": step["thought"],
            "tool_calls": [{
                "id": step.get("tool_call_id", f"call_{i}"),
                "type": "function",
                "function": {
                    "name": step["tool_name"],
                    "arguments": json.dumps(step["tool_arguments"], ensure_ascii=False),
                }
            }]
        }
        messages.append(assistant_msg)

        # tool 消息：Observation
        tool_msg = {
            "role": "tool",
            "tool_call_id": step.get("tool_call_id", f"call_{i}"),
            "content": step["observation"],
        }
        messages.append(tool_msg)

    return messages


def build_messages_with_final_answer(question: str, steps: list[dict],
                                     final_thought: str, final_answer: str) -> list[dict]:
    """
    构建包含最终回答的完整 messages 列表（用于 SFT 训练数据）

    在 build_messages() 基础上，追加最终的 assistant 消息（无 tool_calls）。

    Args:
        question: 用户问题
        steps: 工具调用步骤（同 build_messages）
        final_thought: 最终思考（综合所有数据的判断）
        final_answer: 最终分析报告

    Returns:
        完整的 messages 列表，最后一条是最终回答
    """
    messages = build_messages(question, steps)

    # 最终回答：思考 + 报告合并为 content，无 tool_calls
    # 格式：先写思考过程，空行后写正式报告
    final_content = f"{final_thought}\n\n{final_answer}" if final_thought else final_answer
    messages.append({
        "role": "assistant",
        "content": final_content,
    })

    return messages


# ============ V1 兼容函数（供旧代码使用） ============

def build_tool_descriptions(tool_descriptions: dict) -> str:
    """
    [V1 兼容] 将工具描述字典拼接成 prompt 中的工具说明文本
    """
    lines = []
    for i, (name, desc) in enumerate(tool_descriptions.items(), 1):
        lines.append(f"{i}. {desc}")
    return "\n".join(lines)


# V1 的 SYSTEM_PROMPT 需要 {tool_descriptions} 占位符
_SYSTEM_PROMPT_V1 = SYSTEM_PROMPT + """

## 可用工具
{tool_descriptions}

## 输出格式（严格遵守）
每一步必须按以下格式输出，不要添加额外内容：

Thought: <你的思考过程，分析需要什么信息、下一步该做什么>
Action: <工具名>
Action Input: <工具输入参数>

等待工具返回 Observation 后，继续下一步思考。

当信息收集完毕时，使用 finish 工具输出最终报告：

Thought: <总结已获取的信息，准备生成报告>
Action: finish
Action Input: <完整的分析报告>"""


def build_system_prompt(tool_descriptions: dict) -> str:
    """
    [V1 兼容] 构建完整的 system prompt（填入工具描述）
    """
    desc_text = build_tool_descriptions(tool_descriptions)
    return _SYSTEM_PROMPT_V1.format(tool_descriptions=desc_text)


def build_user_prompt(question: str, steps: list[dict]) -> str:
    """
    [V1 兼容] 构建当前轮次的 user prompt（问题 + 历史轨迹）
    """
    parts = [f"问题：{question}\n"]

    for step in steps:
        parts.append(f"Thought: {step['thought']}")
        parts.append(f"Action: {step['action']}")
        parts.append(f"Action Input: {step['action_input']}")
        if "observation" in step:
            parts.append(f"Observation: {step['observation']}")
        parts.append("")  # 空行分隔

    return "\n".join(parts)


# ============ 测试 ============

def main():
    """测试 V2 messages 构建"""
    import json as _json

    # 测试 V2 messages 构建
    print("=" * 60)
    print("V2 MESSAGES 构建测试")
    print("=" * 60)

    steps = [
        {
            "thought": "用户想了解贵州茅台的投资价值，需要从盈利能力和机构观点两个维度分析。先获取最新财务数据。",
            "tool_name": "search_financial",
            "tool_arguments": {"query": "贵州茅台 ROE 毛利率 净利率 2024"},
            "tool_call_id": "call_0",
            "observation": "找到 5 条相关财务数据：\n1. [financial · profitability | 贵州茅台 | 2024-12-31] ROE 36.99%...",
        },
        {
            "thought": "检索结果显示贵州茅台2024年ROE高达36.99%，毛利率76.18%，净利率52.27%，盈利能力极强。但仅有财务数据不够，还需了解机构对未来增长的预期和风险判断。",
            "tool_name": "search_report",
            "tool_arguments": {"query": "贵州茅台 投资评级 目标价 2025 2026"},
            "tool_call_id": "call_1",
            "observation": "找到 5 条相关研报片段（按 12K 上下文预算截取，数据库可能还有更多匹配）：\n1. [report | 贵州茅台 | 2026-03-03] 评级：买入...",
        },
    ]

    # 构建推理时的 messages
    messages = build_messages("分析贵州茅台的投资价值", steps)
    print("推理时 messages（不含最终回答）：")
    print(_json.dumps(messages, ensure_ascii=False, indent=2))

    # 构建包含最终回答的 messages（SFT 训练用）
    print("\n" + "=" * 60)
    print("SFT 训练数据 messages（含最终回答）：")
    print("=" * 60)

    messages_full = build_messages_with_final_answer(
        question="分析贵州茅台的投资价值",
        steps=steps,
        final_thought="综合财务数据和券商研报，贵州茅台盈利能力卓越，机构一致看好。",
        final_answer="贵州茅台2024年ROE高达36.99%，毛利率76.18%，净利率52.27%，在A股中处于顶尖水平...",
    )
    print(_json.dumps(messages_full, ensure_ascii=False, indent=2))

    # 打印 System Prompt（不含工具描述，工具由 chat_template 注入）
    print("\n" + "=" * 60)
    print("V2 SYSTEM PROMPT（工具描述由 chat_template 自动注入）")
    print("=" * 60)
    print(SYSTEM_PROMPT)


if __name__ == "__main__":
    main()
