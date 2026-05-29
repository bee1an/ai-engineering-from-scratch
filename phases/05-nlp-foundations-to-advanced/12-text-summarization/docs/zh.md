# 文本摘要

> 抽取式系统告诉你文档说了什么。生成式系统告诉你作者想表达什么。不同的任务，不同的陷阱。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**Time:** ~75 minutes

## 问题

一篇 2000 词的新闻文章出现在你的信息流中。你需要 120 个词来概括它。你可以从文章中挑出三个最重要的句子（抽取式），也可以用自己的话重写内容（生成式）。两者都叫摘要，但它们是完全不同的问题。

抽取式摘要是一个排序问题。给每个句子打分，返回 top-`k`。输出始终语法正确，因为它是原文逐字提取的。风险是遗漏分散在文章各处的内容。

生成式摘要是一个生成问题。Transformer 基于输入产生新文本。输出流畅且压缩度高，但可能幻觉出源文本中没有的事实。风险是自信的编造。

本课构建两种方法，并指出各自的失败模式。

## 概念

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**抽取式。** 将文章视为一个图，节点是句子，边是相似度。在图上运行 PageRank（或类似算法）来按句子与其他所有句子的连接程度打分。得分最高的句子就是摘要。经典实现是 **TextRank**（Mihalcea and Tarau, 2004）。

**生成式。** 在文档-摘要对上微调 transformer encoder-decoder（BART, T5, Pegasus）。推理时，模型读入文档并通过 cross-attention 逐 token 生成摘要。Pegasus 特别使用了 gap-sentence 预训练目标，使其无需太多微调就能出色地完成摘要任务。

使用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）评估。ROUGE-1 和 ROUGE-2 衡量 unigram 和 bigram 重叠。ROUGE-L 衡量最长公共子序列。越高越好，但 40 ROUGE-L 是"好"，50 是"优秀"。每篇论文都报告这三个。使用 `rouge-score` 包。

## 动手构建

### 第 1 步：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

两点值得说明。相似度函数使用 log 归一化的词重叠，这是原始 TextRank 变体。TF-IDF 向量的余弦相似度也可以。阻尼因子 0.85 和迭代次数是 PageRank 的默认值。

### 第 2 步：用 BART 做生成式摘要

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料上微调。开箱即用生成新闻风格的摘要。对于其他领域（科学论文、对话、法律），使用对应的 Pegasus checkpoint 或在目标数据上微调。

### 第 3 步：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

务必使用 stemming。不用的话，"running" 和 "run" 会被当作不同的词，ROUGE 会低估。

### 超越 ROUGE（2026 摘要评估）

ROUGE 统治摘要评估指标二十年了，但在 2026 年单独使用已经不够。一项大规模 NLG 论文元分析显示：

- **BERTScore**（上下文嵌入相似度）在 2023 年前后获得广泛采用，现在大多数摘要论文都与 ROUGE 一起报告。
- **BARTScore** 将评估视为生成：用预训练 BART 给定源文本时对摘要的生成概率来打分。
- **MoverScore**（上下文嵌入上的 Earth Mover's Distance）在 2025 年摘要基准中排名第一，因为它比 ROUGE 更好地捕捉语义重叠。
- **FactCC** 和 **基于 QA 的忠实度** 在 2021-2023 年常见，现在常被 **G-Eval** 取代（一个 GPT-4 prompt chain，用 chain-of-thought 推理对连贯性、一致性、流畅度、相关性打分）。
- **G-Eval** 和类似的 LLM-judge 方法在评分标准设计良好时，与人类判断的一致性约 80%。

生产建议：报告 ROUGE-L 用于历史对比，BERTScore 用于语义重叠，G-Eval 用于连贯性和事实性。用 50-100 个人工标注摘要进行校准。

### 第 4 步：事实性问题

生成式摘要容易产生幻觉。抽取式摘要的幻觉风险低得多，因为输出是从源文本逐字提取的，但如果源句子被去语境化、过时或引用顺序错误，仍然可能产生误导。这是生产系统在合规相关内容上仍然偏好抽取式方法的最大原因。

需要命名的幻觉类型：

- **实体替换。** 源文本说 "John Smith"。摘要说 "John Brown"。
- **数字漂移。** 源文本说 "25,000"。摘要说 "2500 万"。
- **极性翻转。** 源文本说"拒绝了报价"。摘要说"接受了报价"。
- **事实编造。** 源文本没有提到 CEO。摘要说 CEO 批准了。

有效的评估方法：

- **FactCC。** 在源句子和摘要句子之间的蕴含关系上训练的二分类器。预测事实/非事实。
- **基于 QA 的事实性。** 向 QA 模型提问，答案在源文本中。如果摘要支持不同的答案，标记。
- **实体级 F1。** 比较源文本和摘要中的命名实体。仅出现在摘要中的实体是可疑的。

对于任何事实性重要的面向用户场景（新闻、医疗、法律、金融），抽取式是更安全的默认选择。生成式需要在流程中加入事实性检查。

## 实际应用

2026 年技术栈：

| 用例 | 推荐 |
|---------|-------------|
| 新闻，3-5 句摘要，英文 | `facebook/bart-large-cnn` |
| 科学论文 | `google/pegasus-pubmed` 或微调的 T5 |
| 多文档，长篇 | 任何 32k+ 上下文的 LLM，用提示 |
| 对话摘要 | `philschmid/bart-large-cnn-samsum` |
| 抽取式，构造上低幻觉风险 | TextRank 或 `sumy` 的 LSA / LexRank |

2026 年长上下文 LLM 在计算不受限时通常能击败专用模型。代价是成本和可复现性；专用模型给出更一致的输出。

## 交付

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习

1. **简单。** 在 5 篇新闻文章上运行 TextRank。将 top-3 句子与参考摘要比较。测量 ROUGE-L。在 CNN/DailyMail 风格的文章上你应该看到 30-45 ROUGE-L。
2. **中等。** 实现实体级事实性检查：从源文本和摘要中提取命名实体（spaCy），计算源实体在摘要中的召回率和摘要实体相对于源的精确率。高精确率低召回率意味着安全但简短；低精确率意味着幻觉实体。
3. **困难。** 在 50 篇 CNN/DailyMail 文章上比较 BART-large-CNN 和 LLM（Claude 或 GPT-4）。报告 ROUGE-L、事实性（按实体 F1）和每条摘要的成本。记录各自在哪些方面胜出。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| Extractive | 挑句子 | 从源文本逐字返回句子。不会产生幻觉。 |
| Abstractive | 改写 | 基于源文本生成新文本。可能产生幻觉。 |
| ROUGE | 摘要指标 | 系统输出与参考之间的 N-gram / LCS 重叠。 |
| TextRank | 基于图的抽取式 | 在句子相似度图上运行 PageRank。 |
| Factuality | 对不对 | 摘要中的声明是否被源文本支持。 |
| Hallucination | 编造内容 | 摘要中源文本不支持的内容。 |

## 延伸阅读

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 和 gap-sentence 目标。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 事实性全景论文。
