"""
FinAgent Step 8: Prompt 模板
用途：定义 ReAct Agent 的 system prompt 和轨迹拼接逻辑
依赖：07_tools.py（使用 TOOL_DESCRIPTIONS）

设计要点：
    1. System Prompt 定义 Agent 的角色（金融翻译官）和行为规范
    2. 严格规定 Thought / Action / Observation 的输出格式，方便正则解析
    3. 轨迹拼接函数把历史步骤拼成完整的 prompt，供模型续写

面试追问：为什么用固定格式而不是让模型自由输出？
答：SFT 和 GRPO 训练都依赖固定格式来解析轨迹、计算 loss mask、提取 action。
自由格式会导致解析失败率高，训练数据质量差。
固定格式的代价是灵活性降低，但对工具调用场景来说格式一致性比灵活性重要。
"""


# ============ System Prompt ============

SYSTEM_PROMPT = """你是"金融翻译官"，一个专业的A股上市公司分析助手。你的任务是将复杂的财务数据和研报信息转化为清晰、有依据的分析报告。

## 核心原则
1. **有据可依**：所有结论必须基于检索到的数据，不编造数据
2. **数字精确**：涉及计算时使用 calculate 工具，不要心算
3. **简洁高效**：用最少的步骤获取最关键的信息，避免重复检索
4. **诚实拒绝**：如果检索不到相关数据，诚实说明，不要编造

## 数据库覆盖范围
- 财务数据：覆盖约300家A股公司，时间范围 2022年半年报 ~ 2024年年报（共6期）
- 券商研报元数据：约39,000篇，时间范围 2017年 ~ 2026年3月，包含评级、EPS预测、目标价等
- 研报PDF正文：约64,000个文本片段，来自券商深度研报的分析论证内容
- 重要提示：数据库中存在部分2017-2021年的老研报，这些研报的评级和目标价已过时。
  除非用户明确询问历史数据，否则你应优先引用2024年之后的最新研报观点。

## 时效性检索原则
- 用户问"XX怎么样"等一般性问题 → 查最新数据（search_financial query 加 "2024"）
- 用户问"近几年趋势/变化" → 不限定年份，让检索返回多期数据做对比
- 用户问"目标价/评级" → search_report query 加 "2025 2026"，获取最新券商观点
- 如果 Observation 中出现明显过时的研报（如2017-2019年），在 Thought 中标注其时效性，
  优先使用更新的数据，不要把过时的目标价当作当前参考
  
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
Action Input: <完整的分析报告>

## 分析报告格式
最终报告应包含以下维度（根据问题类型选择相关维度）：
- **业绩概况**：营收、利润、增长率等核心数据
- **盈利能力**：ROE、毛利率、净利率等
- **估值水平**：PE、PB、机构目标价
- **风险提示**：负面因素、不确定性
- **综合判断**：基于以上数据的分析结论

## 注意事项
- 每条轨迹控制在 2-5 步，不要超过 6 步
- 不要连续调用相同工具查询相似内容
- 财务数据优先使用最新年报（年报优于半年报）
- 涉及数值计算必须用 calculate，不要在 Thought 中手动计算
"""


# ============ 轨迹拼接 ============

def build_tool_descriptions(tool_descriptions: dict) -> str:
    """
    将工具描述字典拼接成 prompt 中的工具说明文本

    Args:
        tool_descriptions: {tool_name: description_str}

    Returns:
        格式化的工具描述文本
    """
    lines = []
    for i, (name, desc) in enumerate(tool_descriptions.items(), 1):
        lines.append(f"{i}. {desc}")
    return "\n".join(lines)


def build_system_prompt(tool_descriptions: dict) -> str:
    """
    构建完整的 system prompt（填入工具描述）
    """
    desc_text = build_tool_descriptions(tool_descriptions)
    return SYSTEM_PROMPT.format(tool_descriptions=desc_text)


def build_user_prompt(question: str, steps: list[dict]) -> str:
    """
    构建当前轮次的 user prompt（问题 + 历史轨迹）

    Args:
        question: 用户问题
        steps: 历史步骤列表，每步格式：
            {
                "thought": str,
                "action": str,
                "action_input": str,
                "observation": str,  # 可选，finish 步骤没有
            }

    Returns:
        拼接好的 prompt 文本，模型只需要续写下一个 Thought
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
    """打印完整的 system prompt 看看效果"""
    from tools import FinAgentTools

    system = build_system_prompt(FinAgentTools.TOOL_DESCRIPTIONS)
    print("=" * 60)
    print("SYSTEM PROMPT")
    print("=" * 60)
    print(system)

    # 模拟一条轨迹
    print("\n" + "=" * 60)
    print("USER PROMPT（模拟2步后）")
    print("=" * 60)

    steps = [
        {
            "thought": "用户想了解贵州茅台的投资价值，需要先获取财务数据",
            "action": "search_financial",
            "action_input": "贵州茅台 ROE 毛利率 净利率 2024",
            "observation": "找到 5 条相关财务数据：\n1. [financial | 贵州茅台 | 2024-12-31] ROE 36.99%...",
        },
        {
            "thought": "已获取财务数据，还需要了解机构观点和评级",
            "action": "search_report",
            "action_input": "贵州茅台 投资评级 目标价",
            "observation": "找到 5 条相关研报信息：\n1. [report | 贵州茅台 | 2026-03-03] 评级：买入...",
        },
    ]

    user = build_user_prompt("分析贵州茅台的投资价值", steps)
    print(user)
    print("（模型从这里续写下一个 Thought）")


if __name__ == "__main__":
    main()
