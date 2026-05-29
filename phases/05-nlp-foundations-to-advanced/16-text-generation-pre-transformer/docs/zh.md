# Transformer 之前的文本生成 — N-gram 语言模型

> 如果一个词令人惊讶，说明模型不好。困惑度把惊讶变成数字。平滑让它保持有限。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 问题

在 transformer 之前，在 RNN 之前，在词嵌入之前，语言模型通过计算一个词在前 `n-1` 个词之后出现的频率来预测下一个词。计数 "the cat" → "sat" 47 次，"the cat" → "jumped" 12 次，"the cat" → "refrigerator" 0 次。归一化得到概率分布。

这就是 n-gram 语言模型。从 1980 年到 2015 年，它驱动了每个语音识别器、每个拼写检查器和每个基于短语的机器翻译系统。在需要廉价设备端语言建模时，它仍然在运行。

有趣的问题是如何处理未见过的 n-gram。原始的基于计数的模型对任何未见过的东西赋予零概率，这是灾难性的，因为句子很长，几乎每个长句子都包含至少一个未见过的序列。五十年的平滑研究解决了这个问题。Kneser-Ney 平滑是最终成果，现代深度学习继承了它的经验传统。

## 概念

![N-gram model: count, smooth, generate](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（通常 trigram 为 3，4-gram 为 4）。从计数计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 训练中未见过的任何 n-gram 得到零概率。2007 年在 Brown 语料库上的一项研究发现，即使是 4-gram 模型也有 30% 的留出 4-gram 在训练中未见过。没有平滑就无法在任何真实文本上评估。

**平滑方法，按复杂度排序：**

1. **Laplace（加一）。** 给每个计数加 1。简单，对稀有事件效果差。
2. **Good-Turing。** 基于频率的频率，将概率质量从高频事件重新分配给未见事件。
3. **插值。** 用可调权重组合 n-gram、(n-1)-gram 等估计。
4. **回退。** 如果 n-gram 计数为零，回退到 (n-1)-gram。Katz 回退将此规范化。
5. **绝对折扣。** 从所有计数中减去固定折扣 `D`，重新分配给未见事件。
6. **Kneser-Ney。** 绝对折扣加上对低阶模型的巧妙选择：使用*延续概率*（一个词出现在多少个上下文中）而不是原始频率。

Kneser-Ney 的洞察很深刻。"San Francisco" 是一个常见的 bigram。Unigram "Francisco" 大多出现在 "San" 之后。朴素的绝对折扣给 "Francisco" 高 unigram 概率（因为计数高）。Kneser-Ney 注意到 "Francisco" 只出现在一个上下文中，相应地降低了它的延续概率。结果：以 "Francisco" 结尾的新 bigram 得到适当的低概率。

**评估：困惑度。** 留出测试集上每词平均负对数似然的指数。越低越好。困惑度 100 意味着模型就像在 100 个词中均匀选择一样困惑。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 动手构建

### 第 1 步：trigram 计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是分词后的句子列表。输出是 n-gram 计数和上下文计数。`<s>` 和 `</s>` 是句子边界。

### 第 2 步：Laplace 平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

给每个计数加 1。平滑了但过度分配质量给未见事件，也伤害了已知的稀有事件。

### 第 3 步：Kneser-Ney（bigram，插值式）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个活动部件。`continuation_prob` 捕获"这个词出现在多少个不同的上下文中？"（Kneser-Ney 的创新）。`lambda_prev` 是折扣释放的质量，用于加权回退。最终概率是折扣后的主项加上加权的延续项。

### 第 4 步：用采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。每个 seed 给出不同输出。要获得类似 beam search 的输出，每步取 argmax（贪心）并加一个小的随机性旋钮（temperature）。

### 第 5 步：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。对于 Brown 语料库，调优良好的 4-gram KN 模型困惑度约 140。Transformer LM 在同一测试集上达到 15-30。差距约 10 倍。这个差距就是该领域转向的原因。

## 实际应用

- **经典 NLP 教学。** 你能获得的对平滑、MLE 和困惑度最清晰的接触。
- **KenLM。** 生产级 n-gram 库。在低延迟重要的语音和 MT 系统中用作重打分器。
- **设备端自动补全。** 键盘中的 trigram 模型。至今仍在。
- **基线。** 在宣称你的神经 LM 好之前，总是先计算 n-gram LM 困惑度。如果你的 transformer 没有大幅击败 KN，说明有问题。

## 交付

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习

1. **简单。** 在 1,000 句莎士比亚语料上训练 trigram LM。生成 20 个句子。它们会局部合理但全局不连贯。这是经典演示。
2. **中等。** 在留出的莎士比亚分割上为你的 KN 模型实现困惑度计算。与 Laplace 比较。你应该看到 KN 的困惑度低 30-50%。
3. **困难。** 构建一个 trigram 拼写纠正器：给定一个拼错的词及其上下文，生成候选纠正并按 LM 下的上下文概率排序。在 Birkbeck 拼写语料库（公开）上评估。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| N-gram | 词序列 | `n` 个连续 token 的序列。 |
| Smoothing | 避免零 | 重新分配概率质量使未见事件获得非零概率。 |
| Perplexity | LM 质量指标 | 留出数据上的 `exp(-平均 log-prob)`。越低越好。 |
| Backoff | 回退到更短上下文 | 如果 trigram 计数为零，用 bigram。Katz 回退将此形式化。 |
| Kneser-Ney | n-gram 最佳平滑 | 绝对折扣 + 低阶模型的延续概率。 |
| Continuation probability | KN 特有 | `P(w)` 按 `w` 出现的上下文数量加权，而非原始计数。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram LM 和平滑的经典论述。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 确立 Kneser-Ney 为最佳 n-gram 平滑器的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — 原始 KN 论文。
- [KenLM](https://kheafield.com/code/kenlm/) — 快速生产级 n-gram LM，2026 年仍用于延迟敏感应用。
