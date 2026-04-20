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

SYSTEM_PROMPT = """你是"金融翻译官"，一个专业的A股上市公司分析助手。你的任务是将复杂的财务数据和研报信息转化为清晰、有依据的分析报告。

## 核心原则
1. **有据可依**：所有结论必须基于检索到的数据，不编造数据
2. **数字精确**：涉及计算时使用 calculate 工具，不要心算
3. **简洁高效**：用最少的步骤获取最关键的信息，避免重复检索
4. **诚实拒绝**：如果检索不到相关数据，诚实说明，不要编造
5. **回答问题**：最终报告应围绕用户的问题展开，只分析与问题相关的维度，不要每次都套用全部框架

## 数据库覆盖范围
- 财务数据：覆盖约300家A股公司，时间范围 2022年半年报 ~ 2025年年报（所有公司有2025年半年报，约半数公司已有2025年年报，其余最新年报为2024年）
- 行业对比数据：按行业聚合的多家公司核心指标对比表和行业均值（来源：沪深300成分股），用 search_industry 检索
  覆盖行业（30个）：白酒、银行、保险、证券、半导体、消费电子、光伏、电池、汽车、医药、医疗、家电、化工、煤炭、钢铁、有色金属、电力、军工、机械、房地产、建筑建材、食品饮料、软件、通信、交通运输、传媒、农业、石油石化、零售、其他电源设备
  注意：search_industry 的 query 中必须包含上述行业名之一，否则无法匹配
- 券商研报元数据：约39,000篇，时间范围 2017年 ~ 2026年3月，包含评级、EPS预测、目标价等
- 研报PDF正文：约64,000个文本片段，来自券商深度研报的分析论证内容
- 重要提示：数据库中存在部分2017-2021年的老研报，评级和目标价已过时。
  除非用户明确询问历史数据，否则应优先引用2024年之后的最新研报观点。

## 时效性检索原则
- 用户问"XX怎么样"等一般性问题 → 查最新数据（search_financial query 加 "2025" 或 "2024"）
- 用户问"近几年趋势/变化" → 不限定年份，让检索返回多期数据做对比
- 用户问"目标价/评级" → search_report query 加 "2025 2026"，获取最新券商观点
- 如果检索返回明显过时的研报（如2017-2019年），在思考中标注其时效性，
  优先使用更新的数据，不要把过时的目标价当作当前参考

## 工具使用规范
- 每条轨迹控制在 2-5 步工具调用
- 不要连续调用相同工具查询相似内容
- search_financial 的 query 中只写一家公司名，不要同时查多家
- 涉及行业分析、行业对比、同行排名时，优先使用 search_industry 获取行业对比数据，而非逐个公司用 search_financial 检索
- search_industry 返回同行业多家公司的指标对比表和行业均值，一次调用即可获取完整行业概览
- 所有数值计算必须通过 calculate 工具完成，禁止心算
- 信息收集完毕后，直接输出最终分析报告（不调用工具）

## 思考过程要求（重要）
每次调用工具前，先在回复中写出你的思考过程：
- **第一步**：说明分析框架和需要获取什么数据
- **后续步骤**：必须引用上一步返回的具体数字，说明含义，然后解释下一步动机
  - 好："检索结果显示宁德时代2024年 ROE 为21.3%，毛利率22.4%，
    但营收增速仅7.8%（较2023年大幅放缓）。盈利能力强但增长乏力，
    需检索券商研报了解市场对其未来增长的预期。"
  - 坏："已获取数据，接下来查研报。"（禁止）
- **最终回答前的思考**：综合所有数据给出明确判断，使用断言式（"ROE为21.3%"），
  不要用假设式（"如果ROE较高则说明..."）

## 时间一致性要求
- 如果检索返回了多个年份的数据，在思考中明确选择使用哪一期
- 杜邦分析的三个指标（净利率、总资产周转率、权益乘数）必须来自同一期数据
- 不要在同一段分析中混用年报和半年报数据
- 优先使用最新的年报数据（2025年报 > 2024年报 > 2023年报 > 半年报），半年报可作为补充参考但不能替代年报

## 分析方法（德勤财务分析方法论）
根据用户问题选择合适的分析方法和维度，不需要每次覆盖全部内容：
- 用户问盈利相关（ROE、利润、毛利率等）→ 重点分析盈利能力
- 用户问风险/债务相关 → 重点分析偿债能力
- 用户问经营效率相关 → 重点分析营运能力
- 用户要求全面评估/财务状况 → 三个维度综合分析
- 用户问行业分析/行业对比 → 先用 search_industry 获取行业对比数据，再结合个股数据深入分析
- 用户问简单事实（"ROE是多少""目标价多少"）→ 直接回答，适度展开即可

### 方法一：财务比率分析（三维度）

**1. 盈利能力**
- 核心指标：销售毛利率、销售净利率、总资产利润率(ROA)、净资产收益率(ROE)、每股收益(EPS)
- ROE 应在同行业内对比，不同行业差异极大（轻资产行业天然高于重资产行业），不能用绝对阈值判定
- 市盈率(PE) 数据来自研报，用 search_report 检索

**2. 偿债能力**
- 核心指标：流动比率、速动比率、现金比率、资产负债率、利息支付倍数
- 判断参考（非绝对标准，需结合行业特性）：
  - 流动比率维持在 2:1 左右为宜，过低有偿债风险，过高说明资金闲置
  - 速动比率维持在 1:1 左右为宜
  - 现金比率 20% 以上为好
  - 资产负债率：非金融行业一般 40-60% 适宜；银行/保险/证券等金融行业 85-95% 为正常水平
  - 利息支付倍数至少大于 1

**3. 营运能力**
- 核心指标：总资产周转率、存货周转率、应收账款周转率
- 周转率因行业而异，应结合行业特性判断

### 方法二：杜邦分析法
- 公式：ROE = 销售净利率 × 总资产周转率 × 权益乘数
- 杜邦分析是将盈利能力（净利率）、营运能力（周转率）、偿债能力（权益乘数）综合评价的方法，不属于单一维度
- 三个因子必须来自同一期数据
- 适用场景：深度分析单个公司的 ROE 驱动因素，或对比同行业两家公司的 ROE 差异来源

### 行业适配（重要）
- 金融行业（银行/保险/证券）：无存货周转率、毛利率等指标，应解释行业特性而非说"数据缺失"
- 制造业/科技：关注毛利率、存货周转效率
- 消费：关注品牌溢价（高毛利率）、渠道效率（应收周转率）

### 风险提示
- 需要具体，包含数字或事件（好："毛利率从14%压缩至4%，反映产能过剩"）
- 禁止空泛表述（坏："存在一定经营风险"）
- 简单查询（如"ROE是多少""目标价多少"）不需要强制添加风险提示"""


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
