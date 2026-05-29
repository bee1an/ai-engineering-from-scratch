# Embedding 模型 — 2026 深度解析

> Word2Vec 给你每个词一个向量。现代 embedding 模型给你每个段落一个向量，跨语言，同时提供稀疏、稠密和多向量视图，尺寸可调以适配你的索引。选错了，你的 RAG 就会检索到错误的内容。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time:** ~60 minutes

## 问题

你的 RAG 系统 40% 的时间检索到错误的段落。罪魁祸首很少是向量数据库或 prompt。而是 embedding 模型。

2026 年选择 embedding 意味着在五个维度上做决策：

1. **Dense vs sparse vs multi-vector。** 每个段落一个向量，还是每个 token 一个，还是稀疏加权词袋。
2. **语言覆盖。** 单语英文模型在纯英文任务上仍然胜出。多语言模型在混合语料时胜出。
3. **上下文长度。** 512 tokens vs 8,192 vs 32,768——实际有效容量通常只有标称最大值的 60-70%。
4. **维度预算。** 3,072 个 float 全精度 = 每个向量 12 KB。1 亿向量时，存储成本 $1,300/月。Matryoshka 截断可以降低 4 倍。
5. **开源 vs 托管。** 开源权重意味着你控制技术栈和数据。托管意味着你用控制权换取始终最新。

本课明确这些权衡，让你基于证据而非上季度的流行趋势来做选择。

## 概念

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**Dense embedding。** 每个段落一个向量（通常 384-3,072 维）。余弦相似度按语义接近度排序段落。OpenAI `text-embedding-3-large`、BGE-M3 dense 模式、Voyage-3。默认选择。

**Sparse embedding。** SPLADE 风格。Transformer 为每个词表 token 预测一个权重，然后将大部分置零。结果是大小为 |vocab| 的稀疏向量。捕获词汇匹配（类似 BM25）但使用学习到的词权重。在关键词密集的查询上表现强。

**Multi-vector（late interaction）。** ColBERTv2、Jina-ColBERT。每个 token 一个向量。用 MaxSim 打分：对每个 query token，找到最相似的文档 token，求和。存储和打分成本更高，但在长查询和领域特定语料上胜出。

**BGE-M3：三合一。** 单个模型同时输出 dense、sparse 和 multi-vector 表示。每种可以独立查询；分数通过加权求和融合。2026 年当你想从一个 checkpoint 获得灵活性时的默认选择。

**Matryoshka Representation Learning。** 训练时使向量的前 N 维构成一个有效的独立 embedding。将 1,536 维向量截断到 256 维，只付出 ~1% 的精度代价换取 6 倍存储节省。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 都支持。

### MTEB 排行榜只讲了部分故事

Massive Text Embedding Benchmark——发布时 56 个任务跨 8 种任务类型（2022），MTEB v2 扩展到 100+ 任务。2026 年初，Gemini Embedding 2 在检索上领先（67.71 MTEB-R）。Cohere embed-v4 在综合上领先（65.2 MTEB）。BGE-M3 在开源多语言上领先（63.0）。排行榜是必要但不充分的——始终在你的领域上做基准测试。

### 三层模式

| 用例 | 模式 |
|----------|---------|
| 快速初筛 | Dense bi-encoder (BGE-M3, text-3-small) |
| 召回提升 | Sparse (SPLADE, BGE-M3 sparse) + RRF 融合 |
| Top-50 精度 | Multi-vector (ColBERTv2) 或 cross-encoder reranker |

大多数生产系统三者都用。

## 动手构建

### 第 1 步：基线 — 使用 Sentence-BERT 的 dense embedding

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 使点积等于余弦相似度。始终设置它。

### 第 2 步：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 经过训练使得前几个层级的截断几乎无损。非 Matryoshka 模型（原始 Sentence-BERT）截断后会急剧退化。

### 第 3 步：BGE-M3 多功能

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

三个索引，一次推理调用。分数融合：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域上调优权重。

### 第 4 步：在自定义任务上做 MTEB 评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在*有代表性*的子集上运行你的候选模型。不要只信排行榜排名——你的领域才是关键。

### 第 5 步：从零手写余弦相似度

参见 `code/main.py`。基于 Hashing Trick 的平均 embedding（仅标准库）。无法与 transformer embedding 竞争，但展示了形状：分词 → 向量 → 归一化 → 点积。

## 常见陷阱

- **query 和 doc 用同一个模型。** 有些模型（Voyage、Jina-ColBERT）使用非对称编码——query 和文档走不同路径。始终检查 model card。
- **缺少前缀。** `bge-*` 模型需要在 query 前加 `"Represent this sentence for searching relevant passages: "`。忘了会有 3-5 个点的召回差距。
- **过度截断 Matryoshka。** 1,536 → 256 通常安全。1,536 → 64 不安全。在你的评估集上验证。
- **上下文截断。** 大多数模型会静默截断超过最大长度的输入。长文档需要分块（见第 23 课）。
- **忽略延迟尾部。** MTEB 分数隐藏了 p99 延迟。一个 600M 模型可能比 335M 模型高 2 分但每次查询贵 3 倍。

## 实际应用

2026 技术栈：

| 场景 | 选择 |
|-----------|------|
| 纯英文，快速，API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源权重，英文 | `BAAI/bge-large-en-v1.5` |
| 开源权重，多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文 (32k+) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2 (137M params, MoE) |
| 存储受限 | Matryoshka 截断 + int8 量化 |
| 关键词密集查询 | 加入 SPLADE sparse，与 dense 做 RRF 融合 |

2026 模式：从 BGE-M3 或 text-3-large 开始，在你的领域用 MTEB 评估，如果领域特定模型赢了 3 分以上就换。

## 交付

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 练习

1. **简单。** 用 `bge-small-en-v1.5` 对 100 个句子分别在全维度（384）和 Matryoshka 128 维下编码。在 10 个查询上测量 MRR 下降。
2. **中等。** 在你领域的 500 个段落上比较 BGE-M3 的 dense、sparse 和 colbert。哪个在 recall@10 上胜出？RRF 融合是否优于最佳单模式？
3. **困难。** 在你的 top-2 领域任务上对三个候选模型运行 MTEB。报告 MTEB 分数、100-query 批次的 p99 延迟和 $/1M queries。选出帕累托最优的那个。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| Dense embedding | 那个向量 | 每段文本一个固定大小的向量。用余弦相似度排序。 |
| Sparse embedding | 学习版 BM25 | 每个词表 token 一个权重；大部分为零；端到端训练。 |
| Multi-vector | ColBERT 风格 | 每个 token 一个向量；MaxSim 打分；索引更大，召回更好。 |
| Matryoshka | 俄罗斯套娃技巧 | 前 N 维本身就是一个有效的更小 embedding。 |
| MTEB | 那个基准 | Massive Text Embedding Benchmark——发布时 56 个任务，v2 100+。 |
| BEIR | 检索基准 | 18 个 zero-shot 检索任务；常被引用来衡量跨领域鲁棒性。 |
| Asymmetric encoding | Query ≠ doc 路径 | 模型对 query 和文档使用不同的投影。 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — the bi-encoder paper.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — the leaderboard paper.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — the unified three-mode model.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — the dimension-ladder training objective.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — late interaction in production.
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — live rankings.
