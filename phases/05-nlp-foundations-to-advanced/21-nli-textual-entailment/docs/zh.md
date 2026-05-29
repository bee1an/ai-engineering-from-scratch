# 自然语言推理 — Textual Entailment

> "t 蕴含 h" 意味着一个人读了 t 之后会得出 h 为真的结论。NLI 的任务是预测蕴含 / 矛盾 / 中立。表面上看很无聊，但在生产环境中是承重结构。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time:** ~60 minutes

## 问题

你构建了一个摘要系统。它生成了一段摘要。你怎么知道摘要里没有幻觉？

你构建了一个聊天机器人。它回答了"是的"。你怎么知道这个答案有检索到的段落支撑？

你需要按主题分类 10,000 篇新闻文章。你没有训练标签。能复用一个模型吗？

这三个问题都可以归结为 Natural Language Inference。NLI 的问题是：给定前提 `t` 和假设 `h`，`h` 是被 `t` 蕴含、矛盾、还是中立（无关）？

- **幻觉检测：** `t` = 源文档，`h` = 摘要中的声明。非蕴含 = 幻觉。
- **有据问答：** `t` = 检索到的段落，`h` = 生成的答案。非蕴含 = 捏造。
- **Zero-shot 分类：** `t` = 文档，`h` = 语言化的标签（"This is about sports"）。蕴含 = 预测标签。

一个任务，三种生产用途。这就是为什么每个 RAG 评估框架底层都内置了一个 NLI 模型。

## 概念

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**三个标签。**

- **Entailment（蕴含）。** `t` → `h`。"The cat is on the mat" 蕴含 "There is a cat."
- **Contradiction（矛盾）。** `t` → ¬`h`。"The cat is on the mat" 与 "There is no cat." 矛盾。
- **Neutral（中立）。** 无法推断。"The cat is on the mat" 对 "The cat is hungry." 是中立的。

**不是逻辑蕴含。** NLI 是*自然*语言推理——一个普通人类读者会推断出什么，而不是严格逻辑。"John walked his dog" 在 NLI 中蕴含 "John has a dog"，但严格的一阶逻辑只有在你公理化"拥有"关系时才能推出。

**数据集。**

- **SNLI**（2015）。570k 人工标注的句对，图片描述作为前提。领域较窄。
- **MultiNLI**（2017）。433k 句对，跨 10 个体裁。2026 年的标准训练语料。
- **ANLI**（2019）。对抗性 NLI。人类专门编写了用来击败现有模型的样本。更难。
- **DocNLI, ConTRoL**（2020–21）。文档级前提。测试多跳和长距离推理。

**架构。** Transformer 编码器（BERT、RoBERTa、DeBERTa）读入 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 表示送入 3-way softmax。在 MNLI 上训练，在留出的基准上评估，分布内句对可达 90%+ 准确率。

**通过 NLI 实现 Zero-shot。** 给定一个文档和候选标签，将每个标签转化为假设（"This text is about sports"）。计算每个假设的蕴含概率。取最大值。这就是 Hugging Face `zero-shot-classification` pipeline 背后的机制。

## 动手构建

### 第 1 步：运行预训练 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

生产环境中的 NLI，`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是开源默认选择。DeBERTa-v3 在排行榜上领先。

### 第 2 步：zero-shot 分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板是 "This example is about {label}."。可以通过 `hypothesis_template` 自定义。不需要训练数据。不需要微调。开箱即用。

### 第 3 步：RAG 忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这就是 RAGAS faithfulness 的核心。将生成的答案拆分为原子声明。逐一检查每个声明是否被检索到的上下文蕴含。报告蕴含的比例。

### 第 4 步：手写 NLI 分类器（概念性）

参见 `code/main.py`，一个仅用标准库的玩具实现：通过词汇重叠 + 否定检测来比较前提和假设。无法与 transformer 模型竞争——但它展示了任务的形状：两段文本输入，3-way 标签输出，损失 = `{entail, contradict, neutral}` 上的交叉熵。

## 常见陷阱

- **仅假设捷径。** 模型仅从假设就能在 SNLI 上达到 ~60% 准确率，因为 "not"、"nobody"、"never" 与矛盾标签相关。这是检测标签泄漏的强基线。
- **词汇重叠启发式。** 子序列启发式（"每个子序列都被蕴含"）能通过 SNLI 但在 HANS/ANLI 上失败。使用对抗性基准。
- **文档级退化。** 单句 NLI 模型在文档级前提上 F1 下降 20+。对长上下文使用 DocNLI 训练的模型。
- **Zero-shot 模板敏感性。** "This example is about {label}" vs "{label}" vs "The topic is {label}" 可以导致准确率波动 10+ 个百分点。调优模板。
- **领域不匹配。** MNLI 在通用英语上训练。法律、医学和科学文本需要领域特定的 NLI 模型（如 SciNLI、MedNLI）。

## 实际应用

2026 技术栈：

| 用例 | 模型 |
|---------|-------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速 / 边缘设备 | `cross-encoder/nli-deberta-v3-base` |
| Zero-shot 分类（轻量） | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 中的幻觉检测 | RAGAS / DeepEval 内置的 NLI 层 |

2026 元模式：NLI 是文本理解的万能胶带。每当你需要"A 是否支持 B？"或"A 是否与 B 矛盾？"——先用 NLI，再考虑另一次 LLM 调用。

## 交付

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 练习

1. **简单。** 在 20 个手工构造的 (premise, hypothesis, label) 三元组上运行 `facebook/bart-large-mnli`，覆盖所有三个类别。测量准确率。加入对抗性"子序列启发式"陷阱（"I did not eat the cake" vs "I ate the cake"），看看模型是否会出错。
2. **中等。** 在 100 条 AG News 标题上比较 zero-shot 模板 `"This text is about {label}"` 与 `"The topic is {label}"` 和 `"{label}"`。报告准确率波动。
3. **困难。** 构建一个 RAG 忠实度检查器：原子声明分解 + 逐声明 NLI。在 50 个带有 gold context 的 RAG 生成答案上评估。测量相对于人工标签的假阳性和假阴性率。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| NLI | Natural Language Inference | 前提-假设关系的三分类。 |
| RTE | Recognizing Textual Entailment | NLI 的旧称；同一任务。 |
| Entailment | "t 蕴含 h" | 一个普通读者会根据 t 得出 h 为真。 |
| Contradiction | "t 排除 h" | 一个普通读者会根据 t 得出 h 为假。 |
| Neutral | "无法判断" | 从 t 到 h 无法做出任何推断。 |
| Zero-shot classification | NLI 当分类器用 | 将标签语言化为假设，取蕴含概率最大的。 |
| Faithfulness | 答案有支撑吗？ | 对 (检索上下文, 生成答案) 做 NLI。 |

## 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — the ANLI benchmark.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI-as-classifier.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — the 2026 NLI workhorse.
