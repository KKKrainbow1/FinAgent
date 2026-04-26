# SFT V4 Prompt 包

3 个 question 生成 prompt + 1 个 thought 改写 prompt,**覆盖 794 条 V4 skeleton + 300 条 V3 改写**。

## 文件

| 模块 | 覆盖 skeleton | 覆盖数 |
|---|---|---:|
| `q_gen_generic.py` | simple/medium/fq/comparison/industry/risk 扩充 + reject + D1-D4 | ~444 |
| `q_gen_concept.py` | fc_C1-C8 纯知识 | 200 |
| `q_gen_boundary.py` | clarify_pos/neg / insuf_L1-L4 / anomaly / multi_source | 150 |
| `thought_rewrite.py` | V3 存量 Thought 首词模板化改写 | 300 |

## 使用

```python
from finagent_repo.sft_prompts_v4 import route

for skel in load_v4_skeletons():
    mod, messages = route(skel)
    resp = client.chat.completions.create(
        model="qwen3-max",  # 或切 GPT-5
        messages=messages,
        temperature=0.8,    # 0.7-0.9 保证多样性
        max_tokens=200,
    )
    question = resp.choices[0].message.content.strip()
    save_v4_question({**skel, "question": question})
```

## 路由(skeleton.prompt_bucket → module)

见 `__init__.py::BUCKET_TO_MODULE` 字典,22 个 bucket 到 3 个 question 生成模块的路由。

## Smoke test

每个模块都有 `__main__` 块可以直接跑:
```bash
python -m finagent_repo.sft_prompts_v4.q_gen_generic
python -m finagent_repo.sft_prompts_v4.q_gen_concept
python -m finagent_repo.sft_prompts_v4.q_gen_boundary
python -m finagent_repo.sft_prompts_v4.thought_rewrite
```

## 设计原则(参考)

- **Skeleton 驱动**:每条 skeleton 的 8 字段已承载 stock/period/metric 等信息,prompt 不再重复
- **Few-shot 多样化**:每个 prompt 内嵌 15-20 条跨风格 few-shot,比"22 个独立 prompt × 3 条 few-shot"多样性更高
- **双 teacher 接入**:prompt 格式是 OpenAI messages 风格,可无缝切 qwen3-max / GPT-5
- **Temperature 0.7-0.9**:同一 skeleton 多次采样可产出不同措辞

详细分布设计见 `docs/sft/Finagent项目SFT_V4_Question分布.md`。
