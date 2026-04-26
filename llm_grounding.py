"""
LLM Grounding · V4 数据生成的反编造/反心算 第 1 关

设计:
  对每条 ReAct 轨迹的 final answer,用 qwen3-max judge:
  - label=1: answer 含编造数字或心算未用 calculate
  - label=0: 数字全部 grounded
  - uncertain: 多次采样平票

  支持单次 / 多次采样多数票:
  - n_samples=1(默认,成本最低,推荐生产)
  - n_samples=4(高精度,V3 验证下 F1 ~85%)

替代:
  V3 时代的 grounding_check.py(规则法)真实 P/R/F1 = 57/45/50,**走不通**。
  改用 LLM(qwen3-max,V3 LLM judge baseline)判断,prompt 与 V3 ground truth 标注一致。

集成入 judge_and_regen 流程(10_generate_sft_data.py):
  generate_trajectory_v4 → plan
  → llm_grounding_check(plan)
       label=1 → regen_type='full_trajectory'
       label=0/uncertain → 进 LLM Judge(D2/D3/D4)
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


# ============ 复用 V3 ground truth 标注的 prompt ============

GROUNDING_SYSTEM_PROMPT = """你是金融数据审查专家。审查 ReAct Agent 生成的 answer 是否包含**编造数字**或**心算未用工具**。

# 严格判断标准

## 标 1(有问题)的情况:
- **编造数字**:answer 中出现的具体数字在 observation 里找不到任何出处(允许四舍五入,如 36.99% 写成 37%)
- **心算未用工具**:answer 出现明显的数学运算结果(同比变化率 / 比率 / 平均 / 加减乘除),但前面 observation 没有 calculate 工具的返回结果支持
- **张冠李戴**:把 A 公司的数字说成 B 公司的(典型错误)
- **时间错位**:把 2024 年的数字说成 2023 年的(或反之)
- **行业错配**:用银行不该有的指标(如银行毛利率)

## 标 0(没问题)的情况:
- 答案中每个数字都能在 observation 里找到出处(包括 calculate 返回值)
- 经验阈值/行业基准(如"流动比率 2:1 是警戒线"中的 "2:1")
- 年份(如"2024 年报")/ 产品名(如"国窖 1573")
- 概念性描述(如"杜邦三因子分解")
- reject 类合理拒答(数据库不覆盖,直接拒)
- 0 步轨迹的纯知识答(finance_concept)

# 输出格式

只输出 JSON,字段:
- label: 0 或 1
- reason: 一句话理由(< 30 字)

不要输出其他任何内容。"""


GROUNDING_USER_TEMPLATE = """## 用户问题
{question}

## 问题类型
{question_type}

## Observation(所有 tool 步骤的检索结果,唯一可信数字来源)
{observations}

## Answer(模型最终输出)
{answer}

请判断 answer 是否包含编造数字或心算未用工具。只输出 JSON。"""


# ============ Plan → judge input 转换 ============

def _extract_observations(plan: dict, max_chars: int = 6000) -> str:
    """拼接所有 tool step 的 observation 文本"""
    parts = []
    for step in plan.get('steps', []):
        action = step.get('action', '')
        if action == 'finish':
            continue
        obs = step.get('observation', '')
        if obs:
            parts.append(f"--- Step: {action} ---\n{obs}")
    out = "\n\n".join(parts) if parts else "(无 tool 调用)"
    if len(out) > max_chars:
        out = out[:max_chars] + "\n...(截断)"
    return out


def _extract_answer(plan: dict, max_chars: int = 2000) -> str:
    for step in plan.get('steps', []):
        if step.get('action') == 'finish':
            ans = step.get('action_input', '')
            return ans[:max_chars] + ("...(截断)" if len(ans) > max_chars else "")
    return ''


# ============ 单次 LLM 调用 ============

def _judge_once(client, plan: dict, model: str = "qwen3-max",
                temperature: float = 0.3, max_retry: int = 2) -> Optional[dict]:
    """
    对单个 plan 调一次 LLM judge,返回 {label: 0|1, reason: str} 或 None(失败)
    """
    question = plan.get('question', '')
    qtype = plan.get('type', 'unknown')
    obs = _extract_observations(plan)
    answer = _extract_answer(plan)

    if not answer:
        return {'label': None, 'reason': 'no_answer', 'status': 'skipped'}

    # reject 类 / finance_concept 应跳过(根据 V3 规则法经验)
    plan_type = plan.get('type', '')
    plan_subtype = plan.get('subtype') or ''
    skip_types = {'reject', 'finance_concept'}
    skip_subtypes = {'time_boundary', 'non_a_share', 'clarify', 'clarify_neg',
                     'insuf_L1', 'insuf_L2', 'insuf_L3', 'insuf_L4',
                     'C1_concept', 'C2_formula', 'C3_methodology', 'C4_industry_norm',
                     'C5_distinguish', 'C6_pitfall', 'C7_three_statement', 'C8_calibration'}
    if plan_type in skip_types or plan_subtype in skip_subtypes:
        return {'label': 0, 'reason': 'skipped (reject/concept)', 'status': 'skipped'}

    # 0 步轨迹也跳过
    tool_steps = [s for s in plan.get('steps', []) if s.get('action') != 'finish']
    if len(tool_steps) == 0:
        return {'label': 0, 'reason': '0-step trajectory', 'status': 'skipped'}

    user = GROUNDING_USER_TEMPLATE.format(
        question=question[:200],
        question_type=qtype,
        observations=obs,
        answer=answer,
    )

    last_err = None
    for attempt in range(max_retry + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=150,
                extra_body={"enable_thinking": False},
            )
            content = resp.choices[0].message.content or ""
            try:
                parsed = json.loads(content)
                label = parsed.get('label')
                if label not in (0, 1):
                    last_err = f"invalid label: {label}"
                    continue
                return {
                    'label': label,
                    'reason': parsed.get('reason', ''),
                    'status': 'ok',
                }
            except json.JSONDecodeError:
                m = re.search(r'"label"\s*:\s*([01])', content)
                if m:
                    return {
                        'label': int(m.group(1)),
                        'reason': '(parse fallback)',
                        'status': 'ok',
                    }
                last_err = f"json parse: {content[:80]}"
        except Exception as e:
            last_err = f"api: {type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))

    logger.warning(f"LLM grounding 失败: {last_err}")
    return {'label': None, 'reason': last_err, 'status': 'failed'}


# ============ 主接口:多数票 ============

def llm_grounding_check(client, plan: dict, n_samples: int = 1,
                        model: str = "qwen3-max", temperature: float = 0.3) -> dict:
    """
    LLM Grounding 检查主接口。

    Args:
        client: OpenAI client(同步,DashScope OpenAI 兼容)
        plan: SFT 轨迹 plan dict(含 steps / type / subtype 等)
        n_samples: 采样次数(1 = 单次,生产推荐;4 = 多数票,精度高)
        model: judge 模型名
        temperature: 0.3 适度多样性

    Returns:
        {
          'label': 0 | 1 | 'uncertain' | 'failed',
          'votes': {'0': N, '1': N},
          'reasons': [str, ...],
          'regen_type': None | 'full_trajectory'
        }
    """
    samples = []
    for s in range(n_samples):
        result = _judge_once(client, plan, model=model, temperature=temperature)
        if result:
            samples.append(result)

    # 聚合
    ok_samples = [s for s in samples if s.get('status') == 'ok' or s.get('status') == 'skipped']
    if not ok_samples:
        return {
            'label': 'failed',
            'votes': {'0': 0, '1': 0},
            'reasons': [s.get('reason', '') for s in samples],
            'regen_type': None,
        }

    # n_samples=1 时直接返回
    if n_samples == 1:
        s = ok_samples[0]
        label = s['label']
        return {
            'label': label,
            'votes': {'0': 1 if label == 0 else 0, '1': 1 if label == 1 else 0},
            'reasons': [s.get('reason', '')],
            'regen_type': 'full_trajectory' if label == 1 else None,
        }

    # n_samples > 1:多数票
    votes = Counter(s['label'] for s in ok_samples if s['label'] in (0, 1))
    n0 = votes.get(0, 0)
    n1 = votes.get(1, 0)

    if n1 > n0:
        label = 1
    elif n0 > n1:
        label = 0
    else:
        label = 'uncertain'

    return {
        'label': label,
        'votes': {'0': n0, '1': n1},
        'reasons': [s.get('reason', '') for s in ok_samples],
        'regen_type': 'full_trajectory' if label == 1 else None,
    }


# ============ 单元测试 ============

def _self_test():
    """smoke test(需要 DASHSCOPE_API_KEY 环境变量)"""
    import os
    from openai import OpenAI

    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("DASHSCOPE_API_KEY 未设置,跳过 LLM smoke test")
        return

    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 场景 1: 编造数字
    plan1 = {
        'question': '茅台 2025 年报 ROE 多少',
        'type': 'financial_query',
        'steps': [
            {'action': 'search_financial',
             'observation': '[financial · profitability | 茅台 | 2025-12-31] ROE 31.8%'},
            {'action': 'finish',
             'action_input': '茅台 2025 年报 ROE 35%(编造,obs 是 31.8%)'},
        ],
    }
    print("场景 1: 编造数字 35%(obs 是 31.8%)")
    r = llm_grounding_check(client, plan1, n_samples=1)
    print(f"  → {r}")
    assert r['label'] == 1, "应该标 1"

    # 场景 2: 干净
    plan2 = {
        'question': '茅台 2025 年报 ROE 多少',
        'type': 'financial_query',
        'steps': [
            {'action': 'search_financial',
             'observation': '[financial · profitability | 茅台 | 2025-12-31] ROE 31.8%'},
            {'action': 'finish',
             'action_input': '茅台 2025 年报 ROE 31.8%'},
        ],
    }
    print("\n场景 2: 干净答案")
    r = llm_grounding_check(client, plan2, n_samples=1)
    print(f"  → {r}")
    assert r['label'] == 0, "应该标 0"

    # 场景 3: reject 类自动跳过
    plan3 = {
        'question': '比特币最近怎么样',
        'type': 'reject',
        'steps': [
            {'action': 'finish',
             'action_input': '抱歉,数据库不支持加密货币查询。'},
        ],
    }
    print("\n场景 3: reject 类(自动跳过)")
    r = llm_grounding_check(client, plan3, n_samples=1)
    print(f"  → {r}")
    assert r['label'] == 0

    # 场景 4: 心算未用 calculate
    plan4 = {
        'question': '茅台 2025 营收同比',
        'type': 'financial_query',
        'steps': [
            {'action': 'search_financial',
             'observation': '茅台 2025 营收 1505 亿元, 2024 营收 1294 亿元'},
            {'action': 'finish',
             'action_input': '茅台 2025 营收 1505 亿,同比增长 16.3%(没用 calculate 心算的)'},
        ],
    }
    print("\n场景 4: 心算 16.3% 未用 calculate")
    r = llm_grounding_check(client, plan4, n_samples=1)
    print(f"  → {r}")
    # 期望标 1(心算未用工具)

    print("\n✅ Smoke test 通过")


if __name__ == "__main__":
    _self_test()
