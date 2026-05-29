# 预训练数据管线

> 模型是一面镜子。你喂给它什么数据，它就反映什么。喂垃圾进去，它会用完美的流畅度反映垃圾。

**类型：** Build
**语言：** Python
**前置课程：** Phase 10, Lessons 01-02（Tokenizers、构建 Tokenizer）
**时间：** 约 90 分钟

## 学习目标

- 构建一个流式数据管线，能够对 TB 级文本进行 tokenize、分块、打乱和批处理，而无需将全部数据加载到内存
- 实现真实预训练管线中使用的数据质量过滤器（去重、语言检测、内容过滤）
- 创建固定长度的训练序列，正确处理 attention mask 和文档边界
- 对管线吞吐量进行 profiling，确保 dataloader 能跟上 GPU 训练速度

## 问题

你已经有了 tokenizer。现在你需要数据。

不是一个 dataset。不是一个 CSV 文件。而是 TB 级的文本——清洗过的、去重过的、经过质量过滤的、tokenize 成固定长度序列的，并且以随机化 batch 的形式足够快地供给，让你的 8-GPU 集群永远不需要等待下一个 batch。

大多数人认为训练 LLM 的关键在于模型架构。并非如此。Llama 3 使用了 15.6 万亿 token。GPT-3 使用了 3000 亿。DeepSeek-V2 使用了 8.1 万亿。三者的架构大致相同：堆叠的 transformer block，包含 attention 和 feedforward 层。输出质量的差异绝大部分来自数据。

DeepMind 的 Chinchilla 论文将这一点量化了。对于给定的计算预算，模型参数量和训练 token 数之间存在最优比例。Chinchilla 表明 2022 年的大多数模型都严重欠训练——相对于它们看到的数据量，参数太多了。一个 70B 参数的模型在 1.4 万亿 token 上训练（Chinchilla 最优）的表现优于一个 280B 模型在 3000 亿 token 上训练（Gopher）。

你的数据管线决定了模型学到的是语言还是噪声。

## 概念

### 数据从哪里来

每个大语言模型都在多种来源的混合数据上训练。确切的组成比例对大多数实验室来说是严格保密的，但我们已经了解足够多来理解这些类别。

| 来源 | 规模 | 质量 | 使用者 |
|--------|------|---------|---------|
| Common Crawl | 原始约 250 TB | 低（需要大量过滤） | GPT-3、Llama、大多数开源模型 |
| Wikipedia | 约 20 GB | 高 | 所有主流 LLM |
| GitHub 代码 | 1 TB+ | 中等（大量重复、死代码） | StarCoder、CodeLlama、DeepSeek-Coder |
| 书籍（BookCorpus、Pile） | 约 100 GB | 高 | GPT-2、GPT-3、早期模型 |
| 学术论文（arXiv、S2ORC） | 约 100 GB | STEM 领域高 | Llama、Galactica |
| StackOverflow、Reddit | 约 100 GB | 中等 | Llama、Falcon |
| 精选网页（C4、RefinedWeb） | 约 5 TB | 中高（预过滤） | T5、Falcon |

Llama 3 公开了其数据配比：大约 50% 网页数据、25% 代码、13% 书籍和学术论文、8% 数学数据、4% 多语言网页数据。总计 15.6 万亿 token，来自超过 5 TB 的原始文本。

配比和总量同样重要。网页数据太多，模型就变成 Reddit 复读机。代码太少，它就不会编程。数学太少，它就无法推理。找到正确的配比是训练 LLM 最难的部分之一，没有公式——需要实验和评估。

### 数据清洗

原始网页数据非常脏。一个典型的 Common Crawl dump 包含：

- HTML 标签和 JavaScript
- 样板化的页头、页脚、导航菜单
- 重复页面（精确重复和近似重复）
- 机器生成的垃圾内容
- 个人身份信息（PII）
- 低质量文本（关键词列表、SEO 垃圾）
- 编码为文本的非文本内容

清洗不是可选的。它决定了模型是生成连贯的段落，还是输出混杂着产品列表的 HTML 标签。

```mermaid
graph TD
    A[Raw Text] --> B[HTML Strip]
    B --> C[Language Detection]
    C --> D[Quality Filter]
    D --> E[Deduplication]
    E --> F[PII Removal]
    F --> G[Clean Text]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff
```

每一步消除一类噪声：

**HTML 剥离：** 移除所有标记。只保留可见文本内容。`trafilatura` 或 `readability` 等库可以提取文章内容，同时丢弃导航、广告和样板内容。

**语言检测：** 使用 fastText 的语言识别模型（lid.176.bin）对每个文档进行分类。过滤到目标语言。一个被分类为英语但置信度低于 0.8 的文档，大概率不是干净的英语。

**质量过滤：** 这里开始变得有趣。RefinedWeb（Falcon 背后的数据集）使用基于 perplexity 的过滤器：在 Wikipedia 上训练一个小型语言模型，然后对每个文档打分。高 perplexity 意味着文档不像 Wikipedia——很可能是垃圾内容、关键词列表或机器生成的内容。perplexity 超过阈值的文档会被移除。

**去重：** 最具影响力的单一清洗步骤。Common Crawl 包含大量重复页面——法律声明、cookie 通知、服务条款。在重复数据上训练浪费算力，还可能导致模型逐字记忆并复述特定段落。

**PII 移除：** 姓名、电子邮件地址、电话号码、社会安全号码。对结构化 PII 使用正则检测，对上下文中的姓名使用 NER 模型。

### 使用 MinHash 去重

精确去重很简单：对每个文档做 hash，移除重复。但近似重复才是真正的问题。同一篇新闻文章的两个副本，周围的广告略有不同，就是近似重复。内容 95% 相同，但逐字节比较是不同的。

MinHash + Locality-Sensitive Hashing (LSH) 可以高效解决这个问题。

```mermaid
graph LR
    A[Document] --> B[Shingling]
    B --> C[MinHash Signature]
    C --> D[LSH Buckets]
    D --> E[Candidate Pairs]
    E --> F[Jaccard Similarity]
    F --> G[Deduplicated Set]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style G fill:#1a1a2e,stroke:#e94560,color:#fff
```

核心思路：

1. **Shingling：** 将每个文档转换为 n-gram 集合（例如 5-gram 的词或字符）。"the quick brown fox" 用 3-word shingle 变成 {"the quick brown", "quick brown fox"}。

2. **MinHash：** 对每个文档的 shingle 集合，计算 k 个 hash 值。每个 hash 值是所有 shingle 在不同 hash 函数下的最小 hash。这创建了一个固定大小的"签名"，可以近似任意两个文档之间的 Jaccard 相似度。

3. **LSH：** 根据 MinHash 签名的 band 将文档分组到桶中。同一个桶中的文档是候选近似重复。这避免了比较每一对——你只需比较候选对。

4. **验证：** 对每个候选对，计算精确的 Jaccard 相似度。如果相似度超过阈值（通常 0.8），移除其中一个副本。

Llama 团队报告通过去重移除了大约 38% 的网页数据。这不是一个小数字。Common Crawl 中超过三分之一是重复或近似重复内容。

### 序列打包

你的模型期望固定长度的输入序列。你的文档长度不一。有些是 50 个 token。有些是 50,000 个 token。

朴素方法：将每个文档 pad 到最大序列长度。这在对学习毫无贡献的 padding token 上浪费了大量算力。

更好的方法：将多个文档打包到一个序列中，用 end-of-sequence token 分隔。一个 2048-token 的序列可能包含三个短文档，用 [EOS] token 连接。

```mermaid
graph TD
    subgraph Naive Packing
        A1["Doc A (200 tokens)"] --> P1["[PAD] x 1848"]
        A2["Doc B (500 tokens)"] --> P2["[PAD] x 1548"]
        A3["Doc C (100 tokens)"] --> P3["[PAD] x 1948"]
    end

    subgraph Efficient Packing
        B1["Doc A (200) | Doc B (500) | Doc C (100) | Doc D (400) | Doc E (848)"]
    end

    style A1 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A2 fill:#1a1a2e,stroke:#e94560,color:#fff
    style A3 fill:#1a1a2e,stroke:#e94560,color:#fff
    style P1 fill:#333,stroke:#666,color:#999
    style P2 fill:#333,stroke:#666,color:#999
    style P3 fill:#333,stroke:#666,color:#999
    style B1 fill:#1a1a2e,stroke:#16c784,color:#fff
```

attention mask 必须正确设置。同一个打包序列中，Document A 的 token 不应该 attend 到 Document B 的 token。这需要一个 block-diagonal attention mask。

长文档在序列边界处被截断或分割成块。分割点很重要：在句子中间分割会迫使模型看到不完整的思想。一些管线会尽可能在段落或句子边界对齐分割。

### Chinchilla Scaling Law

对于固定的计算预算 C（以 FLOPs 衡量），最优模型大小 N 和数据集大小 D 遵循：

```
N_opt ~ C^0.5
D_opt ~ C^0.5
```

实际上，这意味着你应该大致等比例地扩展模型大小和数据集大小。参数量多 10 倍的模型需要大约 10 倍的训练 token 才能达到相同的 loss。

| 模型 | 参数量 | 训练 Token 数 | Chinchilla 最优？ |
|-------|-----------|----------------|-------------------|
| GPT-3 | 175B | 300B | 否（欠训练 3-4 倍） |
| Chinchilla | 70B | 1.4T | 是（设计如此） |
| Llama 2 | 70B | 2T | 过度训练（有意为之） |
| Llama 3 | 70B | 15T | 大幅过度训练 |

Llama 3 故意违反了 Chinchilla 定律。Meta 发现在更多数据上过度训练——远超计算最优比例——能产生更好的推理模型。额外的训练成本只需支付一次，但更小的模型永远更便宜地服务。这有时被称为"推理最优"的 scaling 方法，自 2024 年以来已成为行业标准。

## 动手构建

### 第 1 步：文本清洗

剥离 HTML，规范化空白字符，移除非文本内容。我们将使用公共领域文本（Project Gutenberg）作为小型语料库。

```python
import re

def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def quality_filter(text, min_words=50, max_ratio_caps=0.3, max_ratio_special=0.1):
    words = text.split()
    if len(words) < min_words:
        return False
    caps_ratio = sum(1 for w in words if w.isupper()) / len(words)
    if caps_ratio > max_ratio_caps:
        return False
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / max(len(text), 1) > max_ratio_special:
        return False
    return True
```

质量过滤器捕获 SEO 垃圾（全大写）、机器生成的噪声（高特殊字符比例）和存根页面（太短）。仅这三项检查就能从网页爬取中移除惊人数量的垃圾。

### 第 2 步：MinHash 去重

从零实现 MinHash。不需要外部库——只需 `hashlib`。

```python
import hashlib
from collections import defaultdict

def get_shingles(text, k=5):
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}

def minhash_signature(shingles, num_hashes=128):
    signature = []
    for i in range(num_hashes):
        min_hash = float("inf")
        for shingle in shingles:
            h = int(hashlib.sha256(f"{i}:{shingle}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    return signature

def lsh_buckets(signature, bands=16):
    rows_per_band = len(signature) // bands
    buckets = []
    for b in range(bands):
        start = b * rows_per_band
        band_data = tuple(signature[start:start + rows_per_band])
        bucket_hash = hashlib.md5(str(band_data).encode()).hexdigest()
        buckets.append((b, bucket_hash))
    return buckets

def deduplicate(documents, threshold=0.8, num_hashes=128, bands=16):
    signatures = []
    shingle_sets = []
    for doc in documents:
        shingles = get_shingles(doc)
        shingle_sets.append(shingles)
        signatures.append(minhash_signature(shingles, num_hashes))

    bucket_map = defaultdict(list)
    for doc_idx, sig in enumerate(signatures):
        for band_id, bucket_hash in lsh_buckets(sig, bands):
            bucket_map[(band_id, bucket_hash)].append(doc_idx)

    duplicate_pairs = set()
    for bucket_docs in bucket_map.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                duplicate_pairs.add((bucket_docs[i], bucket_docs[j]))

    removed = set()
    for i, j in duplicate_pairs:
        if i in removed or j in removed:
            continue
        s1, s2 = shingle_sets[i], shingle_sets[j]
        if not s1 or not s2:
            continue
        jaccard = len(s1 & s2) / len(s1 | s2)
        if jaccard >= threshold:
            removed.add(j)

    return [doc for idx, doc in enumerate(documents) if idx not in removed], len(removed)
```

`num_hashes=128` 和 `bands=16` 参数控制精确率-召回率的权衡。更多 hash 给出更准确的相似度估计。更多 band 增加召回率（捕获更多重复）但代价是更多误报。这些值对典型网页文本效果良好。

### 第 3 步：Tokenize 和打包序列

将清洗过的、去重后的文本进行 tokenize，并打包成固定长度的训练序列。

```python
def tokenize_corpus(documents, tokenizer):
    all_tokens = []
    for doc in documents:
        tokens = tokenizer.encode(doc)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_id)
    return all_tokens

def pack_sequences(token_ids, seq_length, pad_id=0):
    sequences = []
    attention_masks = []
    for i in range(0, len(token_ids), seq_length):
        seq = token_ids[i:i + seq_length]
        mask = [1] * len(seq)
        if len(seq) < seq_length:
            pad_count = seq_length - len(seq)
            seq = seq + [pad_id] * pad_count
            mask = mask + [0] * pad_count
        sequences.append(seq)
        attention_masks.append(mask)
    return sequences, attention_masks
```

### 第 4 步：训练用 DataLoader

生成打包序列的随机化 batch。这是训练循环消费的内容。

```python
import random

class PreTrainingDataLoader:
    def __init__(self, sequences, attention_masks, batch_size, shuffle=True):
        self.sequences = sequences
        self.attention_masks = attention_masks
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.sequences)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batch_seqs = [self.sequences[i] for i in batch_idx]
            batch_masks = [self.attention_masks[i] for i in batch_idx]
            yield batch_seqs, batch_masks
```

### 第 5 步：数据集统计

计算重要的数字：总 token 数、唯一 token 数、压缩比、文档长度分布。

```python
from collections import Counter

def compute_statistics(documents, token_ids, sequences, tokenizer_vocab_size):
    total_chars = sum(len(d) for d in documents)
    total_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))
    compression_ratio = total_chars / total_tokens

    doc_lengths = [len(d.split()) for d in documents]
    avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)
    max_doc_length = max(doc_lengths) if doc_lengths else 0
    min_doc_length = min(doc_lengths) if doc_lengths else 0

    token_counts = Counter(token_ids)
    top_tokens = token_counts.most_common(10)

    non_pad_tokens = sum(sum(1 for t in seq if t != 0) for seq in sequences)
    total_positions = sum(len(seq) for seq in sequences)
    utilization = non_pad_tokens / max(total_positions, 1)

    stats = {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "vocab_utilization": unique_tokens / tokenizer_vocab_size,
        "compression_ratio": compression_ratio,
        "avg_doc_length_words": avg_doc_length,
        "max_doc_length_words": max_doc_length,
        "min_doc_length_words": min_doc_length,
        "num_sequences": len(sequences),
        "sequence_utilization": utilization,
        "top_10_tokens": top_tokens,
    }
    return stats
```

压缩比告诉你 tokenizer 在这个语料库上的效率。英文文本通常压缩到每个 token 约 3-4 个字符。如果你看到每个 token 1.5 个字符，说明 tokenizer 切分得太激进了。如果看到 8+，说明它学到了非常领域特定的合并。

序列利用率告诉你打包序列中有多少是真实数据，多少是 padding。低于 90% 意味着你的打包效率不高——你在 padding token 上浪费了算力。

## 实际使用

### 与 HuggingFace Datasets 对比

通过 HuggingFace 的 datasets 库加载相同的语料库，比较管线速度。

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

import time

start = time.time()
tokenized = ds.map(
    lambda x: tokenizer(x["text"], truncation=True, max_length=2048),
    batched=True,
    num_proc=4,
)
hf_time = time.time() - start
total_tokens = sum(len(t) for t in tokenized["input_ids"])
print(f"HuggingFace: {total_tokens:,} tokens in {hf_time:.2f}s ({total_tokens/hf_time:,.0f} tokens/sec)")
```

HuggingFace 管线底层使用 Rust tokenizer 和跨 4 核的并行处理。你的纯 Python 管线会慢 10-50 倍。这个差距就是为什么生产团队使用编译型 tokenizer。算法是一样的。实现语言是区别所在。

## 交付产出

本课程产出一个用于验证和调试 LLM 训练管线数据质量的 prompt。见 `outputs/prompt-data-quality-checker.md`。

## 练习

1. **简单：** 使用简单的启发式方法（字符集分析）为清洗管线添加语言检测。只过滤英文文档，并测量有多少文档被移除。
2. **中等：** 在 MinHash 近似去重之外，使用 SHA-256 hash 实现精确去重。在网页爬取语料库上比较两种方法各自捕获的重复数量。
3. **困难：** 构建基于 perplexity 的质量过滤器。在 Wikipedia 文本上训练一个小型 bigram 语言模型，对每个文档按 perplexity 打分，移除最差的 20%。比较在过滤后数据和未过滤数据上训练时的模型输出质量。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| Common Crawl | "互联网" | 一个每月爬取网页的非营利组织——原始约 250TB，是大多数 LLM 训练数据的起点 |
| MinHash | "某种 hash 技巧" | 一种使用固定大小签名估计集合间 Jaccard 相似度的技术——使大规模近似重复检测成为可能 |
| LSH | "Locality-Sensitive Hashing" | 一种将相似项分到同一个桶中的方法——将成对比较从 O(n^2) 降低到近线性 |
| Sequence packing | "拼接文档" | 将多个文档装入固定长度序列并设置正确的 attention mask——消除 padding 浪费 |
| Chinchilla scaling | "用更多数据训练" | 对于固定计算预算，最优性能要求模型大小和训练 token 数大致等比例扩展 |
| Fertility | "每词 token 数" | 每个词的平均 token 数——GPT-4 中英文为 1.3，非拉丁文字更高 |
| Data mixing | "选择训练数据" | 代码 vs 文本 vs 数学 vs 多语言数据的比例——没有公式，需要实验 |
| Perplexity filter | "质量打分" | 使用小型语言模型对文档打分——高 perplexity 意味着文本不像干净的参考数据 |
| Deduplication | "移除副本" | 消除精确和近似重复的文档——通常移除 30-40% 的原始网页数据 |
| Attention mask | "看哪些 token" | 一个二值 mask，防止在打包序列中跨文档边界进行 attention |

## 延伸阅读

- [Hoffmann et al., 2022 -- Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) -- 改变了我们对数据规模思考方式的论文
- [Penedo et al., 2023 -- The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) -- 如何将 Common Crawl 过滤为高质量数据
- [Touvron et al., 2023 -- Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) -- Llama 2 的数据管线细节
- [Lee et al., 2022 -- Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) -- 为什么去重比你想象的更重要
- [Broder, 1997 -- On the Resemblance and Containment of Documents](https://ieeexplore.ieee.org/document/666900) -- 原始 MinHash 论文
- [Meta, 2024 -- Llama 3 Technical Report](https://arxiv.org/abs/2407.21783) -- 15.6T token、数据混合比例、过滤管线
