# 毕业项目 02 — 代码库RAG（跨仓库语义搜索）

> 2026 年，每个认真的工程组织都在运行一个理解语义而非仅匹配字符串的内部代码搜索。Sourcegraph Amp、Cursor 的代码库问答、Augment 的企业图谱、Aider 的 repomap、Pinterest 的内部 MCP——同一个形态。摄入多个仓库，用 tree-sitter 解析，在函数和类级别分块嵌入，混合搜索，重排序，带引用地回答。这个毕业项目要求你构建一个能处理 10 个仓库共 200 万行代码、并在每次 git push 时支持增量重索引的系统。

**类型：** 毕业项目
**语言：** Python（摄入），TypeScript（API + UI）
**前置要求：** Phase 5（NLP 基础）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 17（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P17
**时间：** 30 小时

## 问题

到 2026 年，每个前沿编码智能体都自带代码库检索层，因为仅靠上下文窗口无法解决跨仓库问题。Claude 的 100 万 token 上下文有帮助，但并不能消除对排序检索的需求。对原始分块做朴素余弦搜索会在生成代码、monorepo 重复和长尾罕用符号上污染结果。生产级答案是对 AST 感知分块做混合（dense + BM25）搜索加重排序，背后有一个符号引用图。

你通过索引一个真实的仓库集群来学习这些——不是一个教程仓库——并衡量 MRR@10、引用忠实度和增量新鲜度。失败模式是基础设施层面的：一个 10 万文件的 monorepo、一次修改了一半文件的 push、一个需要跨四个仓库才能正确回答的查询。

## 概念

一个 AST 感知的摄入管道用 tree-sitter 解析每个文件，提取函数和类节点，在节点边界而非固定 token 窗口处分块。每个分块获得三种表示：稠密嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项，以及一段简短的自然语言摘要。摘要增加了第三种可检索模态——用户问"X 是如何授权的"，摘要会提到"authz"，即使代码里只有 `check_permission`。

检索是混合的。一个查询同时触发稠密和 BM25 搜索，合并 top-k，将并集交给交叉编码器重排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序后的列表送入长上下文合成器（Claude Sonnet 4.7 带 prompt caching，或自托管的 Llama 3.3 70B），指令要求每个声明都引用文件和行范围。没有引用的回答会被后置过滤器拒绝。

增量新鲜度是基础设施问题。Git push 触发一个 diff：哪些文件变了，哪些符号变了。只有受影响的分块重新嵌入。受影响的跨文件符号边（imports、方法调用）被重新计算。索引保持一致，无需每次提交都重新处理 200 万行。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析：tree-sitter 支持 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- 稠密嵌入：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 备选
- 稀疏索引：Tantivy（Rust）带 BM25F，按符号名 vs 函数体加权
- 向量数据库：Qdrant 1.12 带混合搜索，或 pgvector + pgvectorscale（适合 5000 万向量以下的团队）
- 分块摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，prompt-cached
- 重排序器：Cohere rerank-3 或自托管的 bge-reranker-v2-gemma-2b
- 编排：LlamaIndex Workflows 用于摄入，LangGraph 用于查询智能体
- 合成器：Claude Sonnet 4.7（100 万上下文）带 prompt caching
- 符号图：Neo4j（托管）或 kuzu（嵌入式），存储 import 和调用边
- 可观测性：Langfuse span 覆盖每个检索 + 合成步骤

## 构建步骤

1. **摄入遍历器。** 在每次 push hook 时遍历 git 历史。收集变更文件。对每个文件用 tree-sitter 解析，提取函数和类节点及其完整源码范围。输出分块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **分块摘要器。** 将分块批量送入 Haiku 4.5 调用，system preamble 使用 prompt caching。提示词："用一句话总结这个函数，说明其公开契约和副作用。"将摘要与分块一起存储。

3. **嵌入池。** 两个并行队列：稠密（Voyage-code-3 batch 128）和摘要（同一模型，但对摘要字符串）。将向量写入 Qdrant，payload 为 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权的 Tantivy 索引：符号名权重 4，符号体权重 1，摘要权重 2。既支持"找到名为 X 的函数"查询，也支持"找到做 X 事情的函数"查询。

5. **符号图。** 对每个分块记录边：imports（此文件使用仓库 Z 中的符号 Y）、calls（此函数调用类 C 的方法 M）、继承。存入 kuzu。查询时用于跨仓库边界扩展检索。

6. **查询智能体。** LangGraph 三节点。`retrieve` 并行触发 dense + BM25，按 (repo, path, symbol) 去重。`rerank` 对 top-50 运行交叉编码器，保留 top-10。`synth` 调用 Claude Sonnet 4.7，将重排序后的分块放入上下文，缓存 system prompt，要求 file:line 引用。

7. **引用强制。** 解析模型输出；任何没有 `(repo/path:start-end)` 锚点的声明被标记为需要重新询问或直接丢弃。只返回有引用的回答给用户。

8. **增量重索引。** 每次 webhook 时计算符号级 diff。只重新嵌入文本变更的分块。重新计算 imports 变更的分块的符号边。衡量标准：200 万行代码集群中，50 文件的 push 在 60 秒内完成重索引。

9. **评估。** 标注 100 个跨仓库问题及其黄金 file:line 答案。衡量 MRR@10、nDCG@10、引用忠实度（有可验证锚点的声明比例）和 p50/p99 延迟。

## 使用示例

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付标准

交付技能文件 `outputs/skill-codebase-rag.md`。给定一组仓库，它搭建摄入管道、混合索引和查询智能体，并为任何跨仓库问题返回带引用的回答。评分标准：

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 检索质量 | 100 问保留集上的 MRR@10 和 nDCG@10 |
| 20 | 引用忠实度 | 回答声明中有可验证 file:line 锚点的比例 |
| 20 | 延迟与规模 | 在索引语料规模下 10k QPS 时的 p95 查询延迟 |
| 20 | 增量索引正确性 | 50 文件提交从 git push 到可搜索的时间 |
| 15 | 用户体验与回答格式 | 引用可点击性、代码片段预览、后续追问支持 |
| **100** | | |

## 练习

1. 将 Voyage-code-3 换成自托管的 nomic-embed-code。衡量 MRR@10 差异。报告启用重排序后差距是否缩小。

2. 向语料中注入 20% 的生成代码（LLM 生成的样板代码）并重新评估。观察检索污染。在 payload 中添加"generated"标志并降低这些命中的权重。

3. 在你的语料规模下对比 Qdrant 混合搜索 vs pgvector + pgvectorscale。报告 batch size 1 时的 p99。

4. 添加基于采样的漂移检查：每周重跑 100 问评估。MRR@10 下降超过 5% 时告警。

5. 扩展到跨语言符号解析：一个 Python 函数通过 gRPC 调用一个 Go 服务。使用符号图将它们关联起来。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| AST 感知分块 | "函数级切分" | 在 tree-sitter 节点边界而非固定 token 窗口处切割代码 |
| 混合搜索 | "Dense + sparse" | 并行运行 BM25 和向量搜索，合并 top-k，重排序 |
| 交叉编码器重排序 | "二阶段排序" | 对每个 (query, candidate) 对联合打分的模型，比余弦更准确 |
| Prompt caching | "缓存 system prompt" | 2026 Claude / OpenAI 特性，对重复前缀 token 最高打 90% 折扣 |
| 符号图 | "代码图" | 跨文件和仓库的 import、调用、继承边 |
| 引用忠实度 | "有据回答率" | 用户可以通过点击锚点并阅读引用范围来验证的声明比例 |
| 增量重索引 | "Push-to-search 时间" | 从 git push 到变更符号可查询的挂钟时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓库代码智能
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本毕业项目的参考深度解析
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter 排序仓库视图
- [Augment Code enterprise graph](https://www.augmentcode.com) — 商业符号图 RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 详情
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器参考
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — 内部平台参考
