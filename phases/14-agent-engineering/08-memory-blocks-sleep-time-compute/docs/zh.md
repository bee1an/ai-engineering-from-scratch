# Memory Blocks 与 Sleep-Time Compute（Letta）

> MemGPT 在 2024 年变成了 Letta。2026 的演进增加了两个想法：模型可以直接编辑的离散功能性 memory blocks，以及在主智能体空闲时异步整合记忆的 sleep-time agent。这就是如何将记忆扩展到单次对话之外。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT)
**Time:** ~75 minutes

## 学习目标

- 说出 Letta 使用的三个记忆层（core、recall、archival）及各自的作用。
- 解释 memory-block 模式：Human block、Persona block 和用户自定义 blocks 作为一等类型化对象。
- 描述 sleep-time compute 是什么、为什么它在关键路径之外、为什么它可以运行比主智能体更强的模型。
- 实现一个脚本化的双智能体循环，主智能体服务响应，sleep-time agent 在轮次之间整合 blocks。

## 问题

MemGPT（Lesson 07）解决了虚拟内存控制流。三个生产问题浮现了：

1. **延迟。** 每个记忆操作都在关键路径上。如果智能体需要在用户等待时修剪、摘要或调和，尾延迟会爆炸。
2. **记忆腐化。** 写入不断积累。矛盾的事实留存。检索被过时内容淹没。
3. **结构丢失。** 扁平的 archival store 无法表达"Human block 始终在 prompt 中；Persona block 始终在 prompt 中；Task block 按会话切换。"

Letta（letta.com）是 2026 的重写。Memory blocks 使结构显式化；sleep-time compute 将整合移出关键路径。

## 核心概念

### 三层

| 层 | 范围 | 存储位置 | 写入者 |
|---|------|---------|--------|
| Core | 始终可见 | 主 prompt 内部 | 智能体工具调用 + sleep-time 重写 |
| Recall | 对话历史 | 可检索 | 自动轮次记录 |
| Archival | 任意事实 | 向量 + KV + 图 | 智能体工具调用 + sleep-time 摄入 |

Core 是 MemGPT 的 core。Recall 是对话缓冲区及其被驱逐的尾部。Archival 是外部存储。这个拆分清理了 MemGPT 两层的过载问题。

### Memory blocks

Block 是 core 层的一个类型化、持久、可编辑的部分。原始 MemGPT 论文定义了两个：

- **Human block** — 关于用户的事实（姓名、角色、偏好、目标）。
- **Persona block** — 智能体的自我概念（身份、语气、约束）。

Letta 泛化为任意用户自定义 blocks：用于当前目标的 `Task` block、用于代码库事实的 `Project` block、用于硬约束的 `Safety` block。每个 block 有 `id`、`label`、`value`、`limit`（字符上限）、`description`（让模型知道何时编辑它）。

Blocks 通过工具面可编辑：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` — 当 block 接近上限时压缩。

### Sleep-time compute

2025 年 Letta 的新增：在后台运行第二个智能体，不在关键路径上。Sleep-time agents 处理对话记录和代码库上下文，将 `learned_context` 写入共享 blocks，并整合或失效 archival 记录。

由此产生的特性：

- **无延迟成本。** 主响应不等待记忆操作。
- **允许更强的模型。** Sleep-time agent 可以是更贵、更慢的模型，因为它不受延迟约束。
- **自然的整合窗口。** 在用户不等待时去重、摘要、失效矛盾事实。

这个形态与人类的工作方式匹配：你做任务，你睡一觉，长期记忆在夜间沉淀。

### Letta V1 与原生推理

Letta V1（`letta_v1_agent`，2026）弃用了 `send_message`/heartbeat 和内联 `Thought:` token，转而使用原生推理。Responses API（OpenAI）和带 extended thinking 的 Messages API（Anthropic）在单独通道上输出推理，跨轮传递（生产中跨提供商加密）。控制循环仍然是 ReAct。思考 trace 是结构性的，不是 prompt 形状的。

### 这个模式哪里会出错

- **Block 膨胀。** 无限 `block_append` 很快达到上限。在会导致超限的写入之前接入 block 摘要器。
- **静默漂移。** Sleep-time agent 重写了一个 block 而主智能体从未注意到。对 blocks 做版本控制并在 trace 中展示 diff。
- **投毒整合。** Sleep-time agent 将攻击者可达的内容处理进 core。Lesson 27 同样适用于 sleep-time 面。

## Build It

`code/main.py` 实现了：

- `Block` — id、label、value、limit、description。
- `BlockStore` — CRUD + `near_limit(label)` 辅助方法。
- 两个脚本化智能体 — `PrimaryAgent` 服务一个轮次，`SleepTimeAgent` 在轮次之间整合。
- 一个 trace 展示三轮对话中的 block 写入，加上一次 sleep-time pass 摘要一个 block 并失效一个过时事实。

运行：

```
python3 code/main.py
```

记录展示了分工：主轮次快速且产出原始写入；sleep pass 压缩和清理。

## Use It

- **Letta**（letta.com）作为参考实现。自托管或托管云。
- **Claude Agent SDK skills** 作为 block 形状的知识 — skill 是一个命名的、版本化的、可检索的指令块，智能体按需加载。
- **自定义构建** 适合想控制存储后端的团队。使用 Letta API 契约以便日后迁移。

## Ship It

`outputs/skill-memory-blocks.md` 为任何运行时生成一个 Letta 形状的 block 系统，带 sleep-time hooks，包含安全规则和引用接线。

## 练习

1. 添加一个 `block_summarize` 工具，当 `near_limit` 返回 true 时用模型生成的摘要替换 block 值。哪个触发阈值能同时最小化摘要调用和 block 溢出？
2. 实现 sleep-time 对 archival 的去重：文本 token 重叠 >90% 的两条记录合并为一条。只在 sleep pass 中做，永远不在关键路径上。
3. 对 blocks 做版本控制。每次写入记录旧值和 diff。暴露 `block_history(label)` 让运维人员可以调试"为什么智能体忘了 X"。
4. 将 sleep-time agents 视为不可信写入者。当它们触碰 Persona 或 Safety block 时，要求第二个智能体审查后才提交。
5. 将示例移植到使用 Letta API（`letta_v1_agent`）。block schema 有什么变化，原生推理如何改变 trace 形态？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Memory block | "可编辑的 prompt 部分" | 类型化、持久、LLM 可编辑的 core memory 段 |
| Human block | "用户记忆" | 关于用户的事实，固定在 core 中 |
| Persona block | "智能体身份" | 自我概念、语气、约束，固定在 core 中 |
| Sleep-time compute | "异步记忆工作" | 第二个智能体在关键路径之外做整合 |
| Core / Recall / Archival | "层" | 三层记忆拆分：始终可见 / 对话 / 外部 |
| Block limit | "上限" | 每个 block 的字符限制；强制摘要 |
| Native reasoning | "思考通道" | 提供商级别的推理输出，不是 prompt 级别的 `Thought:` |
| Learned context | "Sleep 输出" | Sleep-time agent 写入共享 blocks 的事实 |

## 延伸阅读

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — block 模式
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) — 异步整合
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) — 原生推理重写
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — 起源
