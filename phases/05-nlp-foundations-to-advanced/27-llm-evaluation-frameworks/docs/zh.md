# LLM 评估 — RAGAS, DeepEval, G-Eval

> Exact-match 和 F1 无法捕捉语义等价。人工审核无法规模化。LLM-as-judge 是生产环境的答案——前提是有足够的校准来信任这个数字。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## 问题

你的 RAG 系统回答："June 29th, 2007."
Gold reference 是："June 29, 2007."
Exact Match 得 0 分。F1 得 ~75%。人类会给 100%。

现在乘以 10,000 个测试用例。再乘以每次对检索器、chunking、prompt 或模型的修改。你需要一个理解语义、能低成本大规模运行、不会在回归上撒谎、并能暴露正确失败模式的评估器。

2026 年有三个框架主导这个问题。

- **RAGAS。** Retrieval-Augmented Generation ASsessment。四个 RAG 指标（faithfulness、answer-relevance、context-precision、context-recall），后端是 NLI + LLM-judge。有研究支撑，轻量。
- **DeepEval。** LLM 的 Pytest。G-Eval、task-completion、hallucination、bias 指标。原生 CI/CD。
- **G-Eval。** 一种方法（也是 DeepEval 的一个指标）：LLM-as-judge + chain-of-thought + 自定义标准 + 0-1 分数。

三者都依赖 LLM-as-judge。本课建立对该方法及其信任层的直觉。

## 概念

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM-as-judge。** 用 LLM 替代静态指标，根据评分标准对输出打分。给定 `(query, context, answer)`，提示 judge LLM："Score 0-1 on faithfulness." 返回分数。

为什么有效：LLM 以极小的成本近似人类判断。GPT-4o-mini 每个评分案例约 $0.003，1000 样本的回归评估运行不到 $5。

为什么会静默失败：

1. **Judge 偏差。** Judge 偏好更长的答案、来自自己模型家族的答案、匹配 prompt 风格的答案。
2. **JSON 解析失败。** 坏 JSON → NaN 分数 → 从聚合中静默排除。RAGAS 用户深知这个痛点。用 try/except + 显式失败模式来把关。
3. **模型版本漂移。** 升级 judge 会改变每个指标。冻结 judge 模型 + 版本。

**RAG 四指标。**

| 指标 | 问题 | 后端 |
|--------|----------|---------|
| Faithfulness | 答案中的每个声明都来自检索到的上下文吗？ | 基于 NLI 的蕴含 |
| Answer relevance | 答案回应了问题吗？ | 从答案生成假设问题；与真实问题比较 |
| Context precision | 检索到的 chunk 中，多少比例是相关的？ | LLM-judge |
| Context recall | 检索是否返回了所有需要的内容？ | LLM-judge against gold answer |

**G-Eval。** 定义自定义标准："答案是否引用了正确的来源？"框架自动展开为 chain-of-thought 评估步骤，然后打 0-1 分。适合 RAGAS 未覆盖的领域特定质量维度。

**校准。** 在与人工标签做相关性验证之前，永远不要信任原始 judge 分数。运行 100 个手工标注的样本。绘制 judge vs human。计算 Spearman rho。如果 rho < 0.7，你的 judge rubric 需要改进。

## 动手构建

### 第 1 步：用 NLI 做 faithfulness（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为原子声明。对每个声明用 NLI 检查是否被检索上下文蕴含。Faithfulness = 被支持的比例。

### 第 2 步：answer relevance

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题与实际问题不同，relevance 就会下降。

### 第 3 步：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤就是 rubric。显式步骤比隐式的 "score 0-1" prompt 更稳定。

### 第 4 步：CI 门禁

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件部署。每个 PR 运行。在回归时阻止合并。

### 第 5 步：从零手写玩具评估

参见 `code/main.py`。仅标准库的 faithfulness（答案声明与上下文的重叠）和 relevance（答案 token 与问题 token 的重叠）近似。非生产级。展示形状。

## 常见陷阱

- **没有校准。** 与人工标签相关性只有 0.3 的 judge 就是噪声。部署前要求校准运行。
- **自我评估。** 用同一个 LLM 生成和评判会使分数膨胀 10-20%。judge 用不同的模型家族。
- **成对评判中的位置偏差。** Judge 偏好先呈现的选项。始终随机化顺序并双向运行。
- **原始聚合隐藏失败。** 平均分 0.85 往往隐藏了 5% 的灾难性失败。始终检查底部分位数。
- **Golden dataset 腐化。** 未版本化的评估集随时间漂移会破坏纵向比较。每次变更都给数据集打标签。
- **LLM 成本。** 大规模时，judge 调用主导成本。使用满足校准阈值的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## 实际应用

2026 技术栈：

| 用例 | 框架 |
|---------|-----------|
| RAG 质量监控 | RAGAS (4 metrics) |
| CI/CD 回归门禁 | DeepEval + pytest |
| 自定义领域标准 | G-Eval within DeepEval |
| 在线实时流量监控 | RAGAS with reference-free mode |
| Human-in-the-loop 抽检 | LangSmith or Phoenix with annotation UI |
| Red-teaming / 安全评估 | Promptfoo + DeepEval |

典型技术栈：RAGAS 用于监控，DeepEval 用于 CI，G-Eval 用于新维度。三者都跑；它们的分歧是有用的。

## 交付

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **简单。** 在 10 个包含已知幻觉的 RAG 样本上使用 RAGAS。验证 faithfulness 指标是否捕获了每一个。
2. **中等。** 手动标注 50 个 QA 答案的正确性（0-1）。用 G-Eval 打分。测量 judge 与人工之间的 Spearman rho。
3. **困难。** 用 DeepEval 构建 pytest CI 门禁。故意让检索器退化。验证门禁是否失败。通过对最低 10% 的阈值检查添加底部分位数告警。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| LLM-as-judge | 用 LLM 打分 | 提示 judge 模型根据 rubric 对输出打 0-1 分。 |
| RAGAS | RAG 指标库 | 开源评估框架，4 个无参考 RAG 指标。 |
| Faithfulness | 答案有据吗？ | 答案声明被检索上下文蕴含的比例。 |
| Context precision | 检索到的 chunk 相关吗？ | top-K chunk 中实际有用的比例。 |
| Context recall | 检索找全了吗？ | gold-answer 声明被检索 chunk 支持的比例。 |
| G-Eval | 自定义 LLM judge | Rubric + chain-of-thought 评估步骤 + 0-1 分数。 |
| Calibration | 信任但验证 | Judge 分数与人工分数之间的 Spearman 相关性。 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — the RAGAS paper.
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — the G-Eval paper.
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — open production stack.
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — biases, calibration, limits.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — unifying framework that integrates RAGAS, DeepEval, Phoenix.
