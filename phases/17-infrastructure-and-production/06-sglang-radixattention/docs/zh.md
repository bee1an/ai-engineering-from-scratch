# SGLang 与 RadixAttention：前缀密集型负载

> SGLang 将 KV cache 视为一等可复用资源，存储在 radix tree 中。vLLM 按 FCFS（先到先服务）调度请求，而 SGLang 的 cache-aware 调度器优先处理共享更长前缀的请求——实际上是深度优先的 radix 遍历，让热分支驻留在 HBM 中。在 Llama 3.1 8B 配合 ShareGPT 风格的 1K prompt 上，SGLang 达到 ~16,200 tok/s，vLLM 为 ~12,500，领先 ~29%。在前缀密集的 RAG 负载上优势达到 6.4x。在语音克隆类负载上 cache 命中率超过 86%。2026 年部署在 400,000+ GPU 上，覆盖 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS。但要注意：当前缀顺序不一致时，6.4x 的数字会蒸发——顺序是工程师的杠杆。

**Type:** Learn
**Languages:** Python (stdlib, toy radix-tree cache + cache-aware scheduler)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 14 (Agentic RAG)
**Time:** ~75 minutes

## 学习目标

- 画出 RadixAttention 的结构图：前缀如何存储在 radix tree 中，KV block 如何在根于同一分支的序列间共享。
- 解释 cache-aware 调度以及为什么 FCFS 对前缀密集流量是错误的。
- 给定前缀缓存命中率和 prompt 长度分布，计算预期加速比。
- 说出让 6.4x 数字成为现实（而非错失收益）的 prompt 排序纪律。

## 问题

经典 serving 将每个请求的 prompt 视为不透明的。即使 5,000 个 RAG 请求都以相同的 2,000-token system prompt 加相同的检索前言开头，vLLM 也会对那个 2,000-token 前缀做 5,000 次 prefill。GPU 反复做相同的工作。

观察：在 agentic 和 RAG 负载中，prompt 几乎总是共享长前缀。System prompt、tool schemas、few-shot 示例、检索头部、对话历史——都在请求间重复。如果你把那个前缀的 KV cache 存一次并复用，就不需要再次 prefill。

RadixAttention 正是这样做的。Token 被索引在 radix tree 中；每个节点拥有从根到该节点路径上 token 序列的 KV block。新请求沿树行走：任何 token 匹配的节点都复用该节点的 KV block。Prefill 代价变成与"新"后缀成正比，而非整个 prompt。

挑战在于调度。如果两个请求共享 2,000-token 前缀，第三个只共享同一前缀的 200 token，你希望一起服务那两个长共享请求，让长前缀驻留在 HBM 中。FCFS 做的恰恰相反——它服务先到的，可能在下一个长前缀请求到来之前就驱逐了热分支。

## 核心概念

### Radix tree 作为 KV 索引

Radix tree（紧凑 trie）存储 token 序列。每个节点拥有一个 token 范围及为该范围计算的 KV block。子节点将序列延伸一个或多个 token。

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

一个新请求进来，包含 system prompt + "Context: <doc A>" + "Question: Carol"。调度器沿树行走：system 前缀匹配（124 block 复用），doc-A 分支匹配（31 block 复用），然后只为 "Question: Carol" 分配新 block（4 block）。Prefill 代价：4 block 的新 token。没有树的话：160 block。prefill 节省 ~40x。

### Cache-aware 调度

Radix-tree 支持的复用如果缓存频繁翻转就毫无意义。两个关键策略：

1. **深度优先分发**。从队列中选下一个请求时，优先选与当前运行集根于同一分支的请求。这让热分支保持驻留。
2. **分支级 LRU，而非 block 级**。驱逐整个分支（从最久未用的叶子开始），而非单个 block，使缓存形状匹配 radix 形状。

FCFS 违反了这两条。一个共享 2,000 token 的请求排在一个共享 50 token 的后面，然后 2,000-token 分支被驱逐来容纳那个 50-token 的。

### 你应该记住的 benchmark 数字

- Llama 3.1 8B, H100, ShareGPT 1K prompts: SGLang ~16,200 tok/s vs vLLM ~12,500（~29% 领先）。
- 前缀密集 RAG（相同 system + 相同 doc，变化 question）：SGLang 上最高 6.4x。
- 语音克隆负载：86.4% 前缀缓存命中率。
- SGLang 客户的生产命中率：50-99%，取决于 prompt 纪律。
- 2026 年部署在 400,000+ GPU 上。

### 排序陷阱

6.4x 的数字依赖于一致的 prompt 模板排序。如果你的客户端在某些请求中构造 prompt 为 `[system, tools, context, history, question]`，在另一些中为 `[system, context, tools, history, question]`，树就找不到共享前缀。对人类来说看起来是共享前缀的东西，对 radix tree 来说是两个不同的序列。

工程师的杠杆：你的 prompt 模板就是 cache key。固定顺序。把所有不变的（system、tools、schemas）放最前面。检索上下文放中间。用户问题放最后。不要在前缀中穿插动态内容。

研究中的真实案例：将动态内容移出可缓存前缀，一次变更就让一个部署从 7% 提升到 74% 缓存命中率。

### RadixAttention 的优势和劣势

优势：
- RAG（相同检索前言，变化 question）。
- Agent（相同 tool schemas，变化 query）。
- 带长 system prompt 的对话。
- 带重复前言的语音/视觉负载。

劣势（回到 vLLM 级别吞吐）：
- 唯一 prompt 的单次生成（代码补全、无 system prompt 的开放对话）。
- 每个请求在前缀中穿插唯一内容的动态 prompt。

### 为什么这是调度问题，不仅是 kernel 问题

你可以把 KV 复用实现为 kernel 技巧。SGLang 的洞察是：复用只有在调度器保持热分支驻留时才有回报。朴素的"有就复用"策略在混合负载下会翻转缓存。Radix-tree 索引的调度器才是把 kernel 技巧变成 29% 生产优势的关键。

### 与 vLLM 的关系

两个系统不是严格竞争关系。2026 年 vLLM 添加了前缀缓存（`--enable-prefix-caching`）和 cache-aware 路由器（Rust 实现的 vLLM Router）。差距缩小但没有完全消失——SGLang 的整个栈是 radix-first 的；vLLM 是后来嫁接的。对于前缀复用主导的负载，SGLang 仍是默认选择。对于没有强前缀模式的通用 serving，vLLM 保持相当或更好。

## Use It

`code/main.py` 实现了一个 toy radix-tree KV cache 加带两种策略的调度器：FCFS 和 cache-aware。在相同负载上运行两者，报告前缀缓存命中率和吞吐差异。然后运行一个"打乱顺序"的负载来展示 6.4x 的崩塌。

## Ship It

本课产出 `outputs/skill-radix-scheduler-advisor.md`。给定负载描述（prompt 模板形状、检索模式、并发租户数），产出 prompt 排序处方和 SGLang 采用的 go/no-go 决策。

## 练习

1. 运行 `code/main.py`。比较 FCFS 和 cache-aware 在相同负载上的表现。差异来自哪里——prefill 节省、decode 节省、还是队列延迟？
2. 修改负载使 prompt 随机排列 `[system, tools, context]`。重新运行。命中率发生了什么？为什么？
3. 计算在 Llama 3.1 8B 上将 2,000-token system prompt 作为一个 radix 分支驻留的 HBM 开销。与没有前缀复用的 16 序列 batch 的开销对比。
4. 阅读 SGLang RadixAttention 论文。用三句话解释为什么树形 LRU 驱逐在前缀密集负载下优于 block 形 LRU。
5. 一个客户报告只有 8% 缓存命中率。说出三个可能原因以及你会为每个原因运行的诊断。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| RadixAttention | "the SGLang thing" | KV cache 以 radix tree 索引，共享前缀复用 block |
| Radix tree | "compact trie" | 每个节点拥有一个 token 范围及其 KV block 的树 |
| Cache-aware scheduler | "hot-branch-first" | 优先选择共享驻留分支的请求的调度器 |
| Prefix-cache hit rate | "多少 prompt 是免费的" | 从复用 KV block 服务的 prompt token 比例 |
| FCFS | "first-come first-served" | 破坏前缀局部性的默认调度 |
| Branch-level LRU | "evict the leaf" | 匹配 radix 形状的驱逐策略 |
| Prompt template ordering | "the cache key" | Prompt 的组件顺序决定了树能共享什么 |
| System prompt pinning | "resident prefix" | 保持不变的 system 部分驻留以避免驱逐抖动 |

## 延伸阅读

- [SGLang GitHub](https://github.com/sgl-project/sglang) — 源码和文档。
- [SGLang documentation](https://sgl-project.github.io/) — RadixAttention 和调度细节。
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — 设计参考。
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — benchmark 数字和调度器原理。
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — vLLM 自己的类 radix 实现，供对比。
