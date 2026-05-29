# 信息检索与搜索

> BM25 精确但脆弱。稠密检索撒大网但漏关键词。混合检索是 2026 年的默认方案。其余都是调优。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## 问题

用户输入"骗钱会怎样"，期望找到实际涵盖这一行为的法条："印度刑法第 420 条"。关键词搜索完全找不到（没有共享词汇）。语义搜索如果嵌入模型没有在法律文本上训练过也会找不到。真正的搜索必须同时处理两种情况。

IR 是每个 RAG 系统、每个搜索栏、每个文档站模糊查找背后的流水线。2026 年在生产中有效的架构不是单一方法，而是一系列互补方法的链条，每个方法捕获前一个方法的失败。

本课逐一构建每个组件，并指出每个组件捕获的是哪类失败。

## 概念

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

四层。按需选择。

1. **稀疏检索（BM25）。** 快速，精确匹配准确，语义理解差。运行在倒排索引上。百万文档级每查询低于 10ms。能正确找到法条引用、产品编码、错误信息、命名实体。
2. **稠密检索。** 将查询和文档编码为向量。最近邻搜索。捕获改述和语义相似性。会漏掉仅差一个字符的精确关键词匹配。使用 FAISS 或向量数据库每查询 50-200ms。
3. **融合。** 合并稀疏和稠密的排序列表。Reciprocal Rank Fusion (RRF) 是简单的默认选择，因为它忽略原始分数（处于不同尺度）而只使用排名位置。当你知道某个信号在你的领域中占主导时，加权融合是一个选项。
4. **Cross-encoder rerank。** 取融合后的 top-30。运行 cross-encoder（查询 + 文档一起，对每对打分）。保留 top-5。Cross-encoder 每对比 bi-encoder 慢，但准确得多。你通过只在 top-30 上运行来摊销成本。

三路检索（BM25 + 稠密 + 学习型稀疏如 SPLADE）在 2026 年基准上优于两路，但需要学习型稀疏索引的基础设施。对大多数团队来说，两路加 cross-encoder rerank 是最佳平衡点。

## 动手构建

### 第 1 步：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个值得了解的参数。`k1=1.5` 控制词频饱和度；越高意味着对词重复赋予更多权重。`b=0.75` 控制长度归一化；0 忽略文档长度，1 完全归一化。默认值是 Robertson 在原始论文中的推荐，很少需要调整。

### 第 2 步：用 bi-encoder 做稠密检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2 归一化嵌入使得点积等于余弦相似度。`all-MiniLM-L6-v2` 是 384 维，快速，对大多数英文检索足够强。多语言工作用 `paraphrase-multilingual-MiniLM-L12-v2`。追求最高准确率用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 第 3 步：Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 常数来自原始 RRF 论文。更高的 `k` 会平滑排名差异的贡献；更低的 `k` 让顶部排名占主导。60 是发表的默认值，很少需要调整。

### 第 4 步：混合搜索 + rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合。BM25 找词汇匹配。稠密检索找语义匹配。RRF 合并两个排序列表而无需分数校准。Cross-encoder 使用查询-文档对一起重新打分 top-30，捕获 bi-encoder 遗漏的细粒度相关性。保留 top-5。

### 第 5 步：评估

| 指标 | 含义 |
|--------|---------|
| Recall@k | 在正确文档存在的查询中，它出现在 top-k 中的频率？ |
| MRR (Mean Reciprocal Rank) | 第一个相关文档的 1/rank 的平均值。 |
| nDCG@k | 考虑相关性梯度，不仅仅是二元的相关/不相关。 |

对于 RAG 来说，检索器的 **Recall@k** 是最重要的数字。如果正确段落不在检索集中，你的阅读器就无法回答。

调试技巧：对于失败的查询，对比稀疏和稠密的排序。如果一个找到了正确文档而另一个没有，你有词汇不匹配（修复：加上缺失的那一半）或语义歧义（修复：更好的嵌入或 reranker）。

## 实际应用

2026 年技术栈：

| 规模 | 技术栈 |
|-------|-------|
| 1k-100k 文档 | 内存中 BM25 + `all-MiniLM-L6-v2` 嵌入 + RRF。无需单独数据库。 |
| 100k-10M 文档 | FAISS 或 pgvector 做稠密 + Elasticsearch / OpenSearch 做 BM25。并行运行。 |
| 10M+ 文档 | Qdrant / Weaviate / Vespa / Milvus 带混合支持。Cross-encoder rerank top-30。 |
| 最高质量前沿 | 三路（BM25 + 稠密 + SPLADE）+ ColBERT late-interaction reranking |

无论选什么，都要为评估留出预算。在基准测试端到端 RAG 准确率之前先基准测试检索召回率。阅读器无法修复检索器遗漏的内容。

### 2026 年生产 RAG 的血泪教训

- **80% 的 RAG 失败追溯到摄入和分块，而不是模型。** 团队花数周换 LLM 和调 prompt，而检索每三个查询就悄悄返回错误上下文。先修分块。
- **分块策略比分块大小更重要。** 固定大小切分会破坏表格、代码和嵌套标题。句子感知是默认值；语义或基于 LLM 的分块对技术文档和产品手册有回报。
- **父文档模式。** 检索小的"子"块以获得精确性。当同一父节的多个子块出现时，换入父块以保留上下文。这在不重新训练的情况下持续提升答案质量。
- **k_rerank=3 通常是最优的。** 超过这个数的每个额外块都增加 token 成本和生成延迟，而不提升答案质量。如果 k=8 对你仍然比 k=3 好，说明 reranker 表现不佳。
- **HyDE / 查询扩展。** 从查询生成一个假设答案，嵌入它，检索。弥合短问题和长文档之间的措辞差距。无需训练的免费精确率提升。
- **上下文预算低于 8K token。** 持续触及该限制意味着 reranker 阈值太松。
- **版本化一切。** Prompt、分块规则、嵌入模型、reranker。任何漂移都会悄悄破坏答案质量。CI 门控忠实度、上下文精确率和未回答问题率，在用户看到之前阻止回归。
- **三路检索（BM25 + 稠密 + 学习型稀疏如 SPLADE）优于两路**，在 2026 年基准上尤其对混合专有名词和语义的查询。当基础设施支持 SPLADE 索引时就上线。

合理的检索设计根据 2026 年行业测量可以减少 70-90% 的幻觉。大多数 RAG 性能提升来自更好的检索，而不是模型微调。

## 交付

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习

1. **简单。** 在 500 文档语料库上实现上述 `hybrid_search`。测试 20 个查询。比较 BM25-only、dense-only 和混合的 recall@5。
2. **中等。** 添加 MRR 计算。对每个有已知正确文档的测试查询，找到正确文档在 BM25、稠密和混合排序中的排名。报告每种方法的 MRR。
3. **困难。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）在你的领域上微调稠密编码器。从 500 个查询-文档对构建训练集。比较微调前后的召回率。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| BM25 | 关键词搜索 | Okapi BM25。按词频、IDF 和长度对文档打分。 |
| Dense retrieval | 向量搜索 | 将查询 + 文档编码为向量，找最近邻。 |
| Bi-encoder | 嵌入模型 | 独立编码查询和文档。查询时快速。 |
| Cross-encoder | Reranker 模型 | 将查询 + 文档一起编码。慢但准确。 |
| RRF | 排名融合 | 通过求和 `1/(k + rank)` 来合并两个排序。 |
| Recall@k | 检索指标 | 相关文档出现在 top-k 中的查询比例。 |

## 延伸阅读

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 的权威论述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，经典 bi-encoder。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 缩小与稠密差距的学习型稀疏检索器。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — late-interaction 检索。
