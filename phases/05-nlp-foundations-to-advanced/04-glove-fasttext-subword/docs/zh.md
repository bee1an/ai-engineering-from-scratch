# GloVe、FastText 与子词 Embedding

> Word2Vec 为每个词训练一个 embedding。GloVe 分解了共现矩阵。FastText 嵌入了词的碎片。BPE 架起了通往 transformer 的桥梁。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 5 · 03（从零实现 Word2Vec）
**时间：** 约 45 分钟

## 问题

Word2Vec 留下了两个未解问题。

第一，有一条平行的研究路线直接分解共现矩阵（LSA、HAL），而不是做在线 skip-gram 更新。Word2Vec 的迭代方法本质上更好，还是差异只是两种方法处理计数方式的产物？**GloVe** 回答了这个问题：精心设计损失函数的矩阵分解能匹配或超越 Word2Vec，而且训练成本更低。

第二，两种方法都没有处理从未见过的词的方案。`Zoomer-approved`、`dogecoin`、上周才造出的任何专有名词、稀有词根的每种屈折形式。**FastText** 通过嵌入字符 n-gram 修复了这个问题：一个词是其组成部分的总和（包括词素），所以即使未登录词也能得到合理的向量。

第三，transformer 到来后，问题再次转变。词级词汇表上限约一百万条目；真实语言比这更开放。**字节对编码（BPE）** 及其变体通过学习一个覆盖一切的高频子词单元词汇表解决了这个问题。每个现代 LLM 的 tokenizer 都是子词 tokenizer。

本课逐一讲解这三者，然后解释何时该选哪个。

## 概念

**GloVe（Global Vectors）。** 构建词-词共现矩阵 `X`，其中 `X[i][j]` 是词 `j` 出现在词 `i` 上下文中的频率。训练向量使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失加权使高频对不会主导。完成。

**FastText。** 一个词是其字符 n-gram 加上词本身的总和。`where` 变成 `<wh, whe, her, ere, re>, <where>`。词向量是这些组件向量的总和。像 Word2Vec 一样训练。好处：未见过的词（`whereupon`）从已知 n-gram 组合而来。

**BPE（字节对编码）。** 从单个字节（或字符）的词汇表开始。统计语料库中每个相邻对的频率。将最频繁的对合并为新 token。重复 `k` 次迭代。结果：一个 `k + 256` 个 token 的词汇表，其中高频序列（`ing`、`tion`、`the`）是单个 token，稀有词被拆成熟悉的片段。任何句子都能被分词。

## 动手构建

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

两个值得说明的关键部分。加权函数 `f(x) = (x/x_max)^alpha` 降低非常高频对（如 `(the, and)`）的权重，使它们不会主导损失。最终 embedding 是 `W`（中心）和 `W_tilde`（上下文）表的总和。两者相加是一个已发表的技巧，通常比只用一个效果更好。

### FastText：子词感知的 embedding

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词由其 n-gram 集合表示（通常 3 到 6 个字符）。词 embedding 是其 n-gram embedding 的总和。对于 skip-gram 训练，在 Word2Vec 使用单个向量的地方插入这个。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于未见过的词，只要它的一些 n-gram 是已知的，你仍然能得到一个向量。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere` 和 `<where`，所以两者在空间中靠近。

### BPE：学习得到的子词词汇表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代合并最常见的相邻对。经过足够多的迭代，高频子串（`low`、`est`、`tion`）变成单个 token，稀有词被干净地拆分。

真正的 GPT / BERT / T5 tokenizer 学习 30k-100k 次合并。结果：任何文本都能被分词为有界长度的已知 ID 序列，永远没有 OOV。

## 使用现成工具

实践中，你很少自己训练这些。你加载预训练检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

对于 transformer 时代的 BPE 风格子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀标记词边界（GPT-2 的惯例）。每个现代 tokenizer 都是 BPE 变体、WordPiece（BERT）或 SentencePiece（T5、LLaMA）。

### 如何选择

| 场景 | 选择 |
|------|------|
| 预训练通用词向量，不需要 OOV 容忍度 | GloVe 300d |
| 预训练通用词向量，必须处理拼写错误/新词/形态丰富的语言 | FastText |
| 任何进入 transformer 的东西（训练或推理） | 模型自带的 tokenizer。永远不要换。 |
| 从零训练自己的语言模型 | 先在你的语料库上训练 BPE 或 SentencePiece tokenizer |
| 用线性模型做生产文本分类 | 仍然是 TF-IDF。第 02 课。 |

## 交付

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 练习

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个 n-gram 集合的 Jaccard 重叠。你应该看到大量共享片段（`pla`、`lay`、`play`），这就是为什么 FastText 在形态变体之间迁移良好。
2. **中等。** 扩展 `learn_bpe` 以跟踪词汇表增长。绘制每语料库字符的 token 数作为合并次数的函数。你应该看到开始时快速压缩，渐近到约 2-3 字符/token。
3. **困难。** 在莎士比亚全集上训练 1k 次合并的 BPE。比较常见词与稀有专有名词的分词结果。测量前后每个词的平均 token 数。写下让你惊讶的发现。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 共现矩阵 | 词-词频率表 | `X[i][j]` = 词 `j` 在词 `i` 的窗口中出现的频率。 |
| 子词 | 词的片段 | 字符 n-gram（FastText）或学习得到的 token（BPE/WordPiece/SentencePiece）。 |
| BPE | 字节对编码 | 迭代合并最频繁的相邻对，直到词汇表达到目标大小。 |
| OOV | 未登录词 | 模型从未见过的词。Word2Vec/GloVe 失败。FastText 和 BPE 能处理。 |
| Byte-level BPE | 基于原始字节的 BPE | GPT-2 的方案。词汇表从 256 个字节开始，所以没有任何东西是 OOV。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 论文，七页，仍然是损失函数最好的推导。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 将 BPE 引入现代 NLP 的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE、WordPiece 和 SentencePiece 在实践中到底有什么不同。
