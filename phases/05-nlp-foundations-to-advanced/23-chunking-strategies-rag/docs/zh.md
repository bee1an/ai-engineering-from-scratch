# RAG 的 Chunking 策略

> Chunking 配置对检索质量的影响与 embedding 模型的选择一样大（Vectara NAACL 2025）。Chunking 搞错了，再多的 reranking 也救不了你。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## 问题

你把一份 50 页的合同放进 RAG 系统。用户问："终止条款是什么？"检索器返回了封面页。为什么？因为模型是在 512-token 的 chunk 上训练的，而终止条款在第 20 页深处，跨越了一个分页符，且没有本地关键词将其与查询关联。

解决方案不是"买一个更好的 embedding 模型"。解决方案是 chunking。多大？重叠多少？在哪里切分？带不带周围上下文？

2026 年 2 月的基准测试显示了令人意外的结果：

- Vectara 2026 研究：递归 512-token chunking 击败了语义 chunking，69% → 54% 准确率。
- SPLADE + Mistral-8B 在 Natural Questions 上：overlap 提供了零可测量收益。
- Context cliff：响应质量在约 2,500 token 的上下文处急剧下降。

"显而易见"的答案（语义 chunking、20% overlap、1000 tokens）往往是错的。本课为六种策略建立直觉，告诉你什么时候该用哪种。

## 概念

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking。** 每 N 个字符或 token 切一刀。最简单的基线。会在句子中间断开。压缩率好，连贯性差。

**Recursive。** LangChain 的 `RecursiveCharacterTextSplitter`。先尝试在 `\n\n` 处切分，然后 `\n`，然后 `.`，然后空格。优雅降级。2026 年的默认选择。

**Semantic。** 对每个句子做 embedding。计算相邻句子之间的余弦相似度。在相似度低于阈值处切分。保持主题连贯性。更慢；有时会产生 40-token 的小碎片，损害检索效果。

**Sentence。** 在句子边界切分。每个 chunk 一个句子或 N 个句子的窗口。在 ~5k tokens 以内与语义 chunking 效果相当，成本只是其零头。

**Parent-document。** 存储小的 child chunk 用于检索，*同时*存储更大的 parent chunk 用于上下文。按 child 检索；返回 parent。优雅降级：差的 child chunk 仍然返回合理的 parent。

**Late chunking（2024）。** 先在 token 级别对整个文档做 embedding，然后将 token embedding 池化为 chunk embedding。保留跨 chunk 上下文。适用于长上下文 embedder（BGE-M3、Jina v3）。计算量更高。

**Contextual retrieval（Anthropic, 2024）。** 在每个 chunk 前面加上 LLM 生成的摘要，描述其在文档中的位置（"This chunk is section 3.2 of the termination clauses..."）。在 Anthropic 自己的基准上检索提升 35-50%。索引成本高。

### 击败所有默认配置的规则

将 chunk 大小匹配到查询类型：

| 查询类型 | Chunk 大小 |
|------------|-----------|
| 事实型（"CEO 叫什么名字？"） | 256-512 tokens |
| 分析型 / 多跳 | 512-1024 tokens |
| 整节理解 | 1024-2048 tokens |

NVIDIA 2026 基准。Chunk 应该大到能包含答案加上本地上下文，小到检索器的 top-K 返回聚焦于答案而非上下文噪声。

## 动手构建

### 第 1 步：fixed 和 recursive chunking

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 第 2 步：semantic chunking

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的领域上调优 `threshold`。太高 → 碎片。太低 → 一个巨大的 chunk。

### 第 3 步：parent-document

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

关键洞察：对 parent 去重。多个 child 可能映射到同一个 parent；全部返回会浪费上下文。

### 第 4 步：contextual retrieval（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

索引上下文化后的 chunk。查询时，检索受益于额外的周围信号。

### 第 5 步：评估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

始终做基准测试。对你的语料来说"最佳"策略可能与任何博客文章都不一致。

## 常见陷阱

- **只在事实型查询上评估 chunking。** 多跳查询会揭示非常不同的赢家。使用按查询类型分层的评估集。
- **语义 chunking 没有最小尺寸。** 会产生 40-token 的碎片，损害检索。始终强制 `min_tokens`。
- **Overlap 作为货物崇拜。** 2026 研究发现 overlap 通常提供零收益且使索引成本翻倍。测量，不要假设。
- **没有 min/max 限制。** 5 token 或 5000 token 的 chunk 都会破坏检索。做截断。
- **跨文档 chunking。** 永远不要让一个 chunk 跨越两个文档。始终按文档分块，然后合并。

## 实际应用

2026 技术栈：

| 场景 | 策略 |
|-----------|----------|
| 首次构建，未知语料 | Recursive，512 tokens，无 overlap |
| 事实型 QA | Recursive，256-512 tokens |
| 分析型 / 多跳 | Recursive，512-1024 tokens + parent-document |
| 大量交叉引用（合同、论文） | Late chunking 或 contextual retrieval |
| 对话语料 | Turn 级 chunk + 说话人元数据 |
| 短文本（推文、评论） | 一个文档 = 一个 chunk |

从 recursive 512 开始。在 50-query 评估集上测量 recall@5。从那里调优。

## 交付

保存为 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 练习

1. **简单。** 用 fixed(512, 0)、recursive(512, 0) 和 recursive(512, 100) 对一份 20 页文档做分块。比较 chunk 数量和边界质量。
2. **中等。** 在 5 个文档上构建 30-query 评估集。测量 recursive、semantic 和 parent-document 的 recall@5。哪个赢了？与博客文章的结论一致吗？
3. **困难。** 实现 contextual retrieval。测量相对于基线 recursive 的 MRR 提升。报告索引成本（LLM 调用次数）vs 精度收益。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| Chunk | 文档的一块 | 被 embedding、索引和检索的子文档单元。 |
| Overlap | 安全边距 | 相邻 chunk 共享的 N 个 token；在 2026 基准中通常无用。 |
| Semantic chunking | 智能分块 | 在相邻句子 embedding 相似度下降处切分。 |
| Parent-document | 两级检索 | 检索小的 child，返回更大的 parent。 |
| Late chunking | 先 embedding 再分块 | 在 token 级别对整个文档做 embedding，池化为 chunk 向量。 |
| Contextual retrieval | Anthropic 的技巧 | 索引前在每个 chunk 前加上 LLM 生成的摘要。 |
| Context cliff | 2500-token 墙 | 在 RAG 中约 2.5k 上下文 token 处观察到的质量下降（2026 年 1 月）。 |

## 延伸阅读

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — the default in production.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — chunking matters as much as embedding choice.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — the late chunking paper.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 35-50% retrieval improvement with LLM-generated context prefixes.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — chunk size by query type.
