# Long Context 评估 — NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro 宣称 10M token 上下文。在 1M token 时，8-needle MRCR 降到 26.3%。宣称 ≠ 可用。Long-context 评估告诉你正在部署的模型的实际容量。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60 minutes

## 问题

你有一份 200 页的合同。模型声称有 1M-token 上下文。你把合同粘贴进去问："终止条款是什么？"模型回答了——但答案来自封面页，因为终止条款在 120k token 深处，超出了模型实际注意力的范围。

这就是 2026 年的上下文容量差距。规格表说 1M 或 10M。现实是其中 60-70% 可用，而"可用"取决于任务。

- **检索（单 needle in haystack）：** 在前沿模型上，直到宣称的最大值都接近完美。
- **多跳 / 聚合：** 在大多数模型上超过 ~128k 后急剧退化。
- **对分散事实的推理：** 最先失败的任务。

Long-context 评估测量这些维度。本课命名基准、每个基准实际测量什么，以及如何为你的领域构建自定义 needle 测试。

## 概念

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack（NIAH, 2023）。** 在长上下文的受控深度放置一个事实（"the magic word is pineapple"）。让模型检索它。扫描深度 × 长度。最初的 long-context 基准。前沿模型现在已经饱和；它是必要但不充分的基线。

**RULER（Nvidia, 2024）。** 13 种任务类型跨 4 个类别：检索（单/多键/多值）、多跳追踪（变量跟踪）、聚合（常见词频率）、QA。可配置上下文长度（4k 到 128k+）。揭示那些饱和 NIAH 但在多跳上失败的模型。在 2024 发布中，17 个声称 32k+ 上下文的模型中只有一半在 32k 时保持了质量。

**LongBench v2（2024）。** 503 道多选题，8k-2M 词上下文，六个任务类别：单文档 QA、多文档 QA、长 in-context learning、长对话、代码仓库、长结构化数据。真实世界 long-context 行为的生产基准。

**MRCR（Multi-Round Coreference Resolution）。** 大规模多轮共指。8-needle、24-needle、100-needle 变体。暴露模型在注意力退化前能同时处理多少事实。

**NoLiMa。** "Non-lexical needle。" Needle 和 query 没有字面重叠；检索需要一步语义推理。比 NIAH 更难。

**HELMET。** 拼接多个文档，从任意一个中提问。测试选择性注意力。

**BABILong。** 将 bAbI 推理链嵌入无关的 haystack 中。测试 haystack 中的推理，而不仅仅是检索。

### 实际应该报告什么

- **宣称的上下文窗口。** 规格表数字。
- **有效检索长度。** NIAH 在某个阈值（如 90%）下通过。
- **有效推理长度。** 多跳或聚合在该阈值下通过。
- **退化曲线。** 准确率 vs 上下文长度，按任务类型绘制。

规格表上的两个数字：检索有效长度和推理有效长度。通常推理有效长度是宣称窗口的 25-50%。

## 动手构建

### 第 1 步：为你的领域构建自定义 NIAH

参见 `code/main.py`。骨架：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

扫描 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制热力图。这就是你目标模型的 NIAH 卡片。

### 第 2 步：multi-needle 变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

像 "What are the three magic words?" 这样的问题需要检索全部三个。单 needle 成功不能预测 multi-needle 成功。

### 第 3 步：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要链接三个赋值。前沿模型在 128k 时通常降到 50-70% 准确率。

### 第 4 步：在你的技术栈上跑 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

报告每个类别的准确率。聚合分数隐藏了大的任务级差异。

## 常见陷阱

- **仅 NIAH 评估。** 在 1M token 通过 NIAH 对多跳什么也说明不了。始终运行 RULER 或自定义多跳测试。
- **均匀深度采样。** 许多实现只测试 depth=0.5。测试 depth=0, 0.25, 0.5, 0.75, 1.0——"lost in the middle" 效应是真实的。
- **与 filler 的词汇重叠。** 如果 needle 与 filler 共享关键词，检索就变得简单。使用 NoLiMa 风格的无重叠 needle。
- **忽略延迟。** 1M-token prompt 需要 30-120 秒预填充。在准确率之外测量 time-to-first-token。
- **厂商自报数字。** OpenAI、Google、Anthropic 都发布自己的分数。始终在你的用例上独立重新运行。

## 实际应用

2026 技术栈：

| 场景 | 基准 |
|-----------|-----------|
| 快速健全性检查 | Custom NIAH at 3 depths × 3 lengths |
| 生产模型选择 | RULER (13 tasks) at your target length |
| 真实世界 QA 质量 | LongBench v2 single-doc-QA subset |
| 多跳推理 | BABILong or custom variable-tracing |
| 对话 / dialogue | MRCR 8-needle at your target length |
| 模型升级回归 | Fixed in-house NIAH + RULER harness, run on every new model |

生产经验法则：在你预期的长度上有 NIAH + 1 个推理任务通过之前，永远不要信任一个上下文窗口。

## 交付

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习

1. **简单。** 构建 3 个深度（0.25, 0.5, 0.75）× 3 个长度（1k, 4k, 16k）的 NIAH。在任意模型上运行。将通过率绘制为 3×3 热力图。
2. **中等。** 添加 3-needle 变体。在每个长度测量全部 3 个的检索。与同长度的单 needle 通过率比较。
3. **困难。** 构造一个变量追踪任务（X1 → X2 → X3，3 跳）嵌入 64k filler 中。在 3 个前沿模型上测量准确率。报告每个模型的有效推理长度。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| NIAH | Needle in haystack | 在 filler 中植入一个事实，让模型检索它。 |
| RULER | NIAH 加强版 | 13 种任务类型跨检索 / 多跳 / 聚合 / QA。 |
| Effective context | 真实容量 | 准确率仍保持在阈值以上的长度。 |
| Lost in the middle | 深度偏差 | 模型对长输入中间部分的内容注意力不足。 |
| Multi-needle | 同时多个事实 | 多个植入；测试注意力调度，不仅仅是检索。 |
| MRCR | Multi-round coref | 8、24 或 100-needle 共指；暴露注意力饱和。 |
| NoLiMa | Non-lexical needle | Needle 和 query 没有字面 token 重叠；需要推理。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — the original NIAH repo.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — the multi-task benchmark.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — real-world long-context eval.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — harder needles.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — reasoning-in-haystack.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — the depth-bias paper.
