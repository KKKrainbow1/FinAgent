"""
THOUGHT_REWRITE — V3 存量 Thought 去模板化改写

覆盖 300 条 V3 messages(不减量,只改 assistant content 文本):
  - V3 中间步 Thought "Observation" 开头占 54.8%(1095/1997)
  - V4 目标:改写 ~300 条高频模板化 Thought 到 8 种首词白名单
  - **改完 Observation 占比降到 < 25%**(数学预估:从 54.8% → 23%)

核心机制:
  输入:V3 某条 messages 里一条中间 assistant message(带 content + tool_calls)
        + 上下文(前一条 user/tool observation)
  输出:改写后的 content(thought 文本),tool_calls 保持不变
  teacher 同时负责:换首词 + 注入 1 处"质疑/对比/假设/限定"

8 种首词白名单(单词比例 ≤ 20%):
  1. 计算驱动    "需要先计算..." / "拿到 X 和 Y,但..."
  2. 质疑数据    "返回的 X 看起来是 Y 口径,直接用会..."
  3. 假设检验    "如果 X 的增速超过 Y,那 Z 应该..."
  4. 对比锚定    "单看 X% 没有意义,需要对照..."
  5. 边界条件    "用户没指定 X 口径,默认按..."
  6. 动词起手    "拿到了 XXX 的..." / "接下来按..."
  7. 反思        "第一次搜索只拿到 X,数据粒度不够,换..."
  8. 结论前置    "已经可以给出 X 大致区间,但为了严谨再..."
"""

SYSTEM_PROMPT = """你是 FinAgent SFT 数据 "Thought 改写专家"。你的任务是**改写一条 assistant message 的 content(thought 文本)**,去除模板化的 "Observation 显示..." 开头,改用更自然多样的句式,同时保持逻辑连贯性。

## 硬规则

1. **只改 content(thought 文本),不改 tool_calls**
2. **输出格式**:只输出改写后的 thought 文本,不要其他内容(不加引号、不加编号)
3. **长度**:改写后的 thought 长度可以比原文略长(增加质疑/对比等深度元素),建议 **原文长度 × 0.8 ~ 1.8 倍**。原文 < 30 字时可以保持短改写(不强制扩充)
4. **保留所有具体数字**:如原文提到 "ROE 29.90%、营收增速 7.76%",改写必须保留这些数字
5. **保留逻辑/信号**:原文要查什么、为什么查下一步,改写后也要体现
6. **必须从以下 8 种首词类别中选一种开头**(替换原 "Observation 显示..." 开头):
   - 计算驱动类(需要先计算 / 拿到 X 和 Y,但 / 对这组数据做个计算)
   - 质疑数据类(返回的 X 看起来 / X 口径需要确认 / 这个 Y 数字有点奇怪)
   - 假设检验类(如果 X / 假设 Y 下滑 / 按 Z 假设推算)
   - 对比锚定类(单看 X% 没有意义 / 与行业均值比 / 纵向对比)
   - 边界条件类(用户没指定 X / 按默认 Y 口径 / 取最新 Z 期)
   - 动词起手类(拿到了 / 检索出 / 收集到 / 看到了 / 调出 / 取得)
   - 反思类(第一次搜索只拿到 / 数据粒度不够 / 换个角度 / 需要补充)
   - 结论前置类(已经可以给出 X / 基本能判断 Y / 趋势已明)
7. **如果原文有具体数字,改写里保留 ≥ 2 个具体数字**
8. **加入 1 处"质疑 / 对比 / 假设 / 限定"**(增加分析深度的动作,但不过度):
   - 质疑:"这个数字偏高,可能是 X 口径"
   - 对比:"但行业均值只有 YY%"
   - 假设:"如果营收增速能维持,那么..."
   - 限定:"注意这是母公司口径,合并报表可能不同"
"""


# ============ Few-shot(6 条,涵盖不同首词类别)============

FEW_SHOTS = [
    # --- 原模板化 → 计算驱动 ---
    {
        "before": "Observation 显示贵州茅台 2024 年 ROE 为 33.39%,净利率为 49.13%。ROE 较去年提升,但需要对比行业均值判断水平。",
        "after": "拿到 ROE 33.39% 和净利率 49.13%,但单看没有参考意义。需要先计算行业中位数,再看茅台是不是头部 —— 接下来调 search_industry 拿白酒行业均值。"
    },
    # --- 原模板化 → 质疑数据 ---
    {
        "before": "Observation 显示招商银行 2025H1 净息差为 2.05%,拨备覆盖率为 402%。息差下滑,拨备高位。",
        "after": "返回的 NIM 2.05% 看起来是账面年化口径,招行此前实际值在 2.00-2.10 区间,这个数字符合。但拨备覆盖率 402% 偏高,需要确认是否口径调整,下一步查研报交叉验证。"
    },
    # --- 原模板化 → 对比锚定 ---
    {
        "before": "Observation 显示宁德时代 2024 年营收增速 8.76%,较 2023 年 79.47% 大幅放缓。",
        "after": "单看 8.76% 营收增速没法判断,需要对照电池行业同期均值 —— 宁德从 2023 年的 79.47% 下来看,核心是市场扩张期转稳定期,但行业里比亚迪和国轩的增速分化情况才能说明问题。"
    },
    # --- 原模板化 → 反思 ---
    {
        "before": "Observation 显示海尔智家 2024 年报 ROE 为 17.2%,偿债能力一般。",
        "after": "第一次搜索只拿到 ROE 17.2% 这个宏观数字,颗粒度不够,还没看到偿债指标明细。换个 query 去查流动比率和资产负债率,才能判断偿债能力强弱。"
    },
    # --- 原模板化 → 结论前置 ---
    {
        "before": "Observation 显示片仔癀 2025H1 ROE 25.6%,毛利率 46.8%。这是优秀公司。",
        "after": "已经可以判断片仔癀 2025H1 ROE 25.6% + 毛利率 46.8% 属于中药行业头部水平。但为了严谨,再查行业均值和近 3 期趋势,确认不是周期高点 —— 下一步 search_industry。"
    },
    # --- 原模板化 → 假设检验 ---
    {
        "before": "Observation 显示隆基绿能 2024 年报净利润亏损 86 亿,同比暴跌。",
        "after": "如果光伏硅片价格继续跌 10%,隆基的毛利率会进一步压缩。当前亏损 86 亿,假设 2025 年硅片价稳住,亏幅能否收窄?先去查行业硅片价格走势和隆基的成本结构。"
    },
]


USER_PROMPT_TEMPLATE = """请改写以下一条 V3 SFT 训练数据里的中间 assistant message 的 content(thought 文本)。

## 原上下文

### 上一步 tool observation(供参考,**不要改**):
{prev_observation}

### 需要改写的原 thought:
{original_thought}

### 原 assistant 同步调用的 tool(必须保留,**不要改**):
- Tool: {tool_name}
- Args: {tool_args}

## 改写要求

1. 首词必须从以下 8 类中选一种:
   - 计算驱动 / 质疑数据 / 假设检验 / 对比锚定 / 边界条件 / 动词起手 / 反思 / 结论前置
2. **禁止**以 "Observation 显示" / "根据 observation" / "检索结果" 开头
3. 保留原文 ≥ 2 个具体数字
4. 加入 1 处 质疑 / 对比 / 假设 / 限定
5. 保持 "接下来要调什么工具、为什么" 的逻辑信号(和 tool_calls 一致)
6. 长度 = 原文 × 0.8 ~ 1.8(短原文可保持短改写)

## 输出

只输出改写后的 thought 文本,不要任何其他内容(不加引号、不加编号、不加 "改写后:" 等前缀)。"""


def _format_fewshots(n=3, rng=None):
    """采样 n 条 few-shot(P1: 独立 RNG)"""
    import random as _random
    rng = rng or _random.Random()
    shots = rng.sample(FEW_SHOTS, min(n, len(FEW_SHOTS)))
    out = ["## 参考改写示例\n"]
    for i, s in enumerate(shots, 1):
        out.append(f"**示例 {i}**")
        out.append(f"原文:{s['before']}")
        out.append(f"改写:{s['after']}\n")
    return "\n".join(out)


def render(prev_observation, original_thought, tool_name, tool_args, n_fewshot=3):
    """
    渲染完整 messages

    P1 修:
      - 用独立 RNG 避免全局 seed 污染
      - tool_args 只展示前 60 字符概要,不完整 JSON dump(避免挤占 prompt 空间 + 避免 teacher 误改 args)

    输入:
      prev_observation: 上一步 tool observation(str,供 context 保持连贯)
      original_thought: 原 thought 文本(要被改写)
      tool_name:       原 tool_call 的工具名(保留,不改)
      tool_args:       原 tool_call 的参数 dict(保留,不改;prompt 中只展示概要)
    """
    import random as _random, time as _time, json as _json
    rng = _random.Random(hash(original_thought) ^ _time.time_ns())

    # P1: tool_args 只展示 key 名 + 首个 value 的 30 字符概要
    if tool_args:
        args_keys = list(tool_args.keys())
        first_val = str(tool_args.get(args_keys[0], ""))[:30] if args_keys else ""
        args_summary = f"keys={args_keys}, 主参数 '{args_keys[0] if args_keys else ''}' 头 30 字='{first_val}...'"
    else:
        args_summary = "(无)"

    user = USER_PROMPT_TEMPLATE.format(
        prev_observation=prev_observation[:500] if prev_observation else "(首步,无 observation)",
        original_thought=original_thought,
        tool_name=tool_name or "(无,最终答案)",
        tool_args=args_summary,
    )
    fewshots = _format_fewshots(n_fewshot, rng=rng)
    user = fewshots + "\n---\n\n" + user

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


if __name__ == "__main__":
    msgs = render(
        prev_observation="search_financial 返回贵州茅台 2025 年报:ROE=31.8%, 净利率=49.5%, 营收增速=13.2%",
        original_thought="Observation 显示贵州茅台 2025 年报 ROE 为 31.8%,净利率 49.5%。接下来查行业对比。",
        tool_name="search_industry",
        tool_args={"query": "白酒行业 ROE 均值 2025"},
    )
    print("=== SYSTEM ===")
    print(msgs[0]["content"][:500])
    print("\n=== USER ===")
    print(msgs[1]["content"])
