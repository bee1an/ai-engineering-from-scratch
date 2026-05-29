# 混合记忆：向量 + 图 + KV（Mem0）

> Mem0（Chhikara et al., 2025）将记忆视为三个并行存储 — 向量用于语义相似度，KV 用于快速事实查找，图用于实体关系推理。一个打分层在检索时融合三者。这是 2026 外部记忆的生产标准。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## 学习目标

- 解释为什么单一存储（仅向量、仅图、仅 KV）不足以支撑智能体记忆。
- 说出 Mem0 的三个并行存储及各自优化的目标。
- 描述 Mem0 的融合打分 — relevance、importance、recency — 以及为什么它是加权和而非层级结构。
- 用 stdlib 实现一个玩具三存储记忆，带 `add()` 写入全部三个和 `search()` 融合结果。

## 问题

单一存储对三类查询中的某一类是错误的：

- **语义相似度** — "我们上周讨论了什么关于 agent drift 的内容？"向量胜出；KV 和图无法回答。
- **事实查找** — "用户的电话号码是什么？"KV 胜出；向量浪费，图过度。
- **关系推理** — "哪些客户共享同一个计费实体？"图胜出；向量和 KV 无法回答。

生产智能体在一个会话中会发出全部三种查询。单存储记忆对其中两种总是错误的。Mem0 的贡献是将三者接在单一 `add`/`search` 面后面，用打分函数融合它们。

## 核心概念

### 三个并行存储

Mem0（arXiv:2504.19413, April 2025）在 `add(text, user_id, metadata)` 时：

1. 从文本中提取候选事实（LLM 驱动的步骤）。
2. 将每个事实写入向量存储（embedding）用于语义搜索。
3. 将每个事实写入 KV 存储，键为 (user_id, fact_type, entity)，用于 O(1) 查找。
4. 将每个事实写入图存储（Mem0g）作为类型化边，用于关系查询。

在 `search(query, user_id)` 时：

1. 向量存储返回 embedding cosine 的 top-k。
2. KV 存储返回基于查询派生的 (user_id, type, entity) 的直接命中。
3. 图存储返回从查询实体可达的子图。
4. 打分层融合三者。

### 融合打分

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance** — 向量 cosine、KV 精确匹配、图路径权重。
- **Importance** — 写入时标记或学习得到（某些事实更重要：姓名、ID、策略）。
- **Recency** — 自上次写入或读取以来的指数衰减。

权重按产品调节。聊天智能体用更高的 `w_recency`；合规智能体用更高的 `w_importance`；检索智能体用更高的 `w_relevance`。

### Mem0g 与时序推理

Mem0g 添加了冲突检测器。当新事实与现有边矛盾时，现有边被标记为无效但不删除。时序查询（"用户三月份住在哪个城市？"）遍历在该时间点有效的子图。

这是 Letta 失效模式泛化后的合规级行为。

### 基准数据

Mem0 论文报告（2025）：

- **LoCoMo**（长对话记忆）：91.6
- **LongMemEval**（长程情景记忆）：93.4
- **BEAM 1M**（1M token 记忆基准）：64.1

对比基线（全上下文 128k LLM、扁平向量存储、扁平 KV）全部落后 10+ 分。基准本身不能证明选择合理 — 运营形态才能 — 但数据表明融合设计不是舍入误差。

### 范围分类

Mem0 按范围拆分记忆：

- **User memory** — 跨会话持久，键为 `user_id`。
- **Session memory** — 在一个线程内持久。
- **Agent memory** — 每个智能体实例的状态。

每次写入选择一个范围。检索可以跨范围查询，带每范围权重。不加思考地混合范围就是"助手把 Bob 的项目告诉了 Alice"事故的来源。

### 这个模式哪里会出错

- **Embedding 漂移。** 前一百次查询看起来正确的向量结果随着语料增长而退化。定期对使用最多的 top-N 记录重新 embedding。
- **KV schema 蔓延。** `(user_id, type, entity)` 看起来简单，直到每个团队都添加自己的 `type`。每季度审计 type 集合。
- **图爆炸。** 一个有噪声的提取器每条消息添加 50 条边。限制每次 `add` 调用的图写入数；丢弃低置信度边。

## Build It

`code/main.py` 用 stdlib 实现三存储模式：

- `VectorStore` — 朴素 token 重叠相似度作为 embedding 替代。
- `KVStore` — 以 `(user_id, fact_type, entity)` 为键的字典。
- `GraphStore` — 类型化边 (subject, relation, object, valid)。
- `Mem0` — 顶层门面，带 `add()`、`search()`、融合打分和范围感知检索。
- 一个多用户、多会话对话的完整 trace。

运行：

```
python3 code/main.py
```

输出展示三条独立的召回路径加上融合后的 top-k。翻转 `main()` 顶部的打分权重，观察排名变化。

## Use It

- **Mem0（Apache 2.0）** — 生产就绪。用 Postgres + Qdrant + Neo4j 自托管，或使用托管云。
- **Letta** — 三层 core/recall/archival；自带向量和图后端。
- **Zep** — 商业替代方案，带时序知识图谱和事实提取。
- **自定义构建** — 当你需要精确控制提取器（合规）或融合权重（recency 主导的语音智能体）时。

## Ship It

`outputs/skill-hybrid-memory.md` 生成一个三存储记忆脚手架，带融合打分器、范围分类和时序失效。

## 练习

1. 将玩具向量相似度替换为真实 embedding 模型（sentence-transformers、Ollama、OpenAI embeddings）。在合成长对话上衡量 recall@10。排名在 1000 次写入后会漂移吗？
2. 添加时序查询：`search(query, as_of=timestamp)`。只返回在该时间点或之前有效的记录。哪个存储需要最多工作？
3. 实现冲突检测器：如果传入事实与图边矛盾，失效旧边并记录两者。在"用户住在柏林" -> "用户住在里斯本"上测试。
4. 将融合打分器扩展为包含 `user_feedback` 维度（对检索记录的点赞）。如何防止博弈（智能体只返回它已经喜欢的记录）？
5. 阅读 Mem0 文档（`docs.mem0.ai`）。将玩具移植到 `mem0` 客户端调用。在相同的 20 个测试查询上对比检索质量。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 混合记忆 | "向量加图加 KV" | 三个存储并行写入，检索时融合 |
| 事实提取 | "记忆摄入" | LLM 步骤，将文本拆分为 (entity, relation, fact) 元组 |
| 融合打分 | "相关性排序" | relevance、importance、recency 的加权和 |
| Scope | "记忆命名空间" | user / session / agent — 决定谁看到什么 |
| Mem0g | "记忆图" | 带时序有效性的类型化边，用于关系查询 |
| 时序失效 | "软删除" | 将矛盾的边标记为无效；永不删除 |
| Embedding 漂移 | "检索腐化" | 向量质量随语料增长而退化；定期重新 embedding |

## 延伸阅读

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — 原始论文
- [Mem0 docs](https://docs.mem0.ai/platform/overview) — 生产 API、SDK、托管云
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — 虚拟上下文前身
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — 三层兄弟设计
