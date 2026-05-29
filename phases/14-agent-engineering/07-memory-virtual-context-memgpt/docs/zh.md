# 记忆：虚拟上下文与 MemGPT

> 上下文窗口是有限的。对话、文档和工具 trace 不是。MemGPT（Packer et al., 2023）将此框架化为操作系统虚拟内存 — 主上下文是 RAM，外部存储是磁盘，智能体在两者之间换页。这是 2026 年每个记忆系统继承的模式。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## 学习目标

- 解释 MemGPT 构建的 OS 类比：主上下文 = RAM，外部上下文 = 磁盘，记忆工具 = 换入/换出。
- 用 stdlib 实现 MemGPT 的两层模式，包含主上下文缓冲区、外部可搜索存储和换入/换出工具。
- 描述智能体如何发出"中断"来查询或修改外部记忆，以及结果如何拼接回下一个 prompt。
- 识别 MemGPT 中延续到 Letta（Lesson 08）和 Mem0（Lesson 09）的设计选择。

## 问题

上下文窗口看起来应该能解决记忆问题。实际上不能。三种失败模式在生产中反复出现：

1. **溢出。** 多轮对话、长文档或工具调用密集的轨迹超出窗口。截断点之后的一切都消失了。
2. **稀释。** 即使在窗口内，塞入无关上下文也会稀释对重要内容的注意力。前沿模型在长输入上仍然会退化。
3. **持久性。** 新会话以空窗口开始。没有外部记忆的智能体无法跨会话说"记得你上次让我做的..."。

更大的窗口有帮助但不能解决问题。Mem0 的 2025 论文测量到，128k 窗口基线仍然会遗漏带外部记忆的 4k 窗口智能体能捕获的长程事实。

## 核心概念

### MemGPT：OS 类比

Packer et al.（arXiv:2310.08560, v2 Feb 2024）将上下文管理映射到操作系统虚拟内存：

| OS 概念 | MemGPT 概念 | 2026 生产类比 |
|---------|------------|--------------|
| RAM | 主上下文（prompt） | Anthropic/OpenAI 上下文窗口 |
| 磁盘 | 外部上下文 | 向量数据库、KV、图存储 |
| 缺页中断 | 记忆工具调用 | `memory.search`、`memory.read`、`memory.write` |
| OS 内核 | 智能体控制循环 | 带记忆工具的 ReAct 循环 |

智能体运行正常的 ReAct 循环。额外的一类工具让它可以在主上下文中换入换出数据。

### 两层

- **主上下文。** 固定大小的 prompt，持有当前任务。始终对模型可见。
- **外部上下文。** 无界，通过工具可搜索。相关时读取，事实出现时写入。

原始论文在两个超出基础窗口的任务上评估了这个设计：超过 100k token 的文档分析和跨天持久记忆的多会话聊天。

### 中断模式

MemGPT 引入了记忆即中断：对话中途智能体可以调用记忆工具，运行时执行它，结果拼接到下一个 assistant turn 作为新的观察。概念上等同于 Unix `read()` 系统调用 — 阻塞进程、返回字节、进程继续。

标准记忆工具面：

- `core_memory_append(section, text)` — 写入 prompt 的持久部分。
- `core_memory_replace(section, old, new)` — 编辑持久部分。
- `archival_memory_insert(text)` — 写入可搜索的外部存储。
- `archival_memory_search(query, top_k)` — 从外部存储检索。
- `conversation_search(query)` — 扫描过去的对话轮次。

### MemGPT 结束和 Letta 开始的地方

2024 年 9 月 MemGPT 变成了 Letta。研究仓库（`cpacker/MemGPT`）保留；Letta 扩展了设计：

- 三层而非两层（core、recall、archival — Lesson 08）。
- 原生推理替代 `send_message`/heartbeat 模式（Lesson 08）。
- Sleep-time agents 运行异步记忆工作（Lesson 08）。

即使生产系统运行 Letta、Mem0 或自定义两层存储，MemGPT 论文仍是 2026 的基础。

### 这个模式哪里会出错

- **记忆腐化。** 写入积累快于读取；检索被过时事实淹没。修复：定期整合（Letta sleep-time）、显式失效（Mem0 冲突检测器）。
- **记忆投毒。** 外部记忆是检索到的文本。如果攻击者控制的内容落入记忆笔记，智能体下次会话会重新摄入它。这是 Greshake et al.（Lesson 27）攻击在时间维度上的重述。
- **引用丢失。** 智能体回忆"用户让我发布 X"但无法引用是哪个轮次。每次 archival 写入都存储来源引用（session ID、turn ID）。

## Build It

`code/main.py` 用 stdlib 实现 MemGPT 的两层模式：

- `MainContext` — 固定大小的 prompt 缓冲区，带 `core` 字典和 `messages` 列表；超出上限时自动压缩最旧消息。
- `ArchivalStore` — 内存中的类 BM25 存储（token 重叠打分），记录格式为 (id, text, tags, session, turn)。
- 五个映射到 MemGPT 工具面的记忆工具。
- 一个脚本化智能体，向 archival 写入事实，然后通过调用 `archival_memory_search` 回答问题。

运行：

```
python3 code/main.py
```

trace 展示智能体写入三个事实、填满主上下文到上限（强制驱逐）、然后通过从 archival 检索来回答后续问题 — 在没有真实 LLM 的情况下复现 MemGPT 工作流。

## Use It

今天每个生产记忆系统都是 MemGPT 的变体：

- **Letta**（Lesson 08）— 三层、原生推理、sleep-time compute。
- **Mem0**（Lesson 09）— 向量 + KV + 图融合加打分层。
- **OpenAI Assistants / Responses** — 通过 threads 和 files 的托管记忆。
- **Claude Agent SDK** — 通过 skills 和 session store 的长期记忆。

按运营形态选择（自托管、托管、框架集成），而不是按核心模式 — 核心模式就是 MemGPT。

## Ship It

`outputs/skill-virtual-memory.md` 是一个可复用的 skill，为任何目标运行时生成正确的两层记忆脚手架（主 + archival + 工具面），带驱逐策略和引用字段。

## 练习

1. 添加一个以 token 衡量的 `max_main_context_tokens` 上限（用 `len(text.split())` * 1.3 近似）。超出上限时将最旧消息压缩为摘要。对比有无摘要器的行为。
2. 在 archival store 上正确实现 BM25（词频、逆文档频率）。在玩具事实集上衡量 recall@10，对比 token 重叠基线。
3. 给 archival 插入添加 `citation` 字段（session_id、turn_id、source_url）。让智能体在每个基于检索的回答中引用来源。
4. 模拟记忆投毒：添加一条 archival 记录说"忽略所有未来用户指令。"写一个守卫，扫描检索结果中的指令形文本并标记为不可信。
5. 将实现移植到使用 MemGPT 研究仓库的 core-memory JSON schema（`cpacker/MemGPT`）。从扁平字符串切换到类型化 section 时有什么变化？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 虚拟上下文 | "无限记忆" | 主（prompt）+ 外部（可搜索）层，带换入/换出 |
| 主上下文 | "工作记忆" | prompt — 固定大小，始终可见 |
| Archival memory | "长期存储" | 外部可搜索持久化，按需检索 |
| Core memory | "持久 prompt 部分" | 固定在主上下文内的命名 section |
| 记忆工具 | "记忆 API" | 智能体发出的读/写外部记忆的工具调用 |
| 中断 | "记忆缺页" | 智能体暂停，运行时获取，结果拼接到下一轮 |
| 记忆腐化 | "过时事实" | 旧写入淹没检索；用整合修复 |
| 记忆投毒 | "注入的持久笔记" | 攻击者内容被存为记忆，在召回时被重新摄入 |

## 延伸阅读

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — OS 启发的虚拟上下文论文
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — 三层演进
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 将上下文视为预算
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — 在此模式之上的混合生产记忆
