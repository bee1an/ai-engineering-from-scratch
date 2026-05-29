# 共享记忆与 Blackboard 模式

> 2026 年多智能体系统中两种方法共存：**消息池**（所有人看到所有人的消息，如 AutoGen GroupChat 或 MetaGPT）和**带订阅的 blackboard**（agent 订阅相关事件，如 Context-Aware MCP 或 Matrix 框架）。两者都是多 agent 系统中唯一有状态的部分——这意味着有趣的 bug 都在这里。参考失败模式是**记忆投毒**：一个 agent 幻觉出一个"事实"，其他 agent 将其视为已验证，准确性逐渐衰减，比立即崩溃更难调试。本课从 stdlib 构建两种结构，注入投毒攻击，并展示在生产中真正有效的三种缓解措施。

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## 问题

多 agent 系统需要一个地方让 agent 共享事实。一个字面选项是"在消息中传递一切"——但这用额外复制重新发明了共享状态。另一个是"给所有人一个全局日志"——但全局日志无限增长且容易被投毒。第三个是"为每个 agent 投影一个视图"——可扩展但 schema 重。

当一个 agent 幻觉并将幻觉写入共享状态时，每个读取该状态的下游 agent 都将幻觉作为事实采纳。等人类注意到时，推理链已经深入五步，根因是第三条写入的消息。调试多 agent 准确性衰减比调试崩溃更难。

这就是记忆投毒。它是 MAST 分类法（Cemri 等人，arXiv:2503.13657）中记录第二多的失败族，而且是结构性的：任何没有溯源和不可写验证者的共享记忆设计最终都会表现出来。

## 概念

### 两种主要拓扑

**全消息池。** 每个 agent 读取每条消息。AutoGen GroupChat 和 MetaGPT 使用这种。简单、透明、可检查，但超过约 10 个 agent 就无法扩展，因为每个 agent 的上下文被其他 agent 的工作填满。

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**带订阅的 Blackboard。** Agent 声明对主题的兴趣；基础设施只路由相关消息。CA-MCP（arXiv:2601.11595）和 Matrix 去中心化框架（arXiv:2511.21686）使用这种。扩展更远，但需要前期 schema 设计使订阅有意义。

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### 各自何时胜出

- **全池**在 agent 少（< 10）、异构、对话短期时胜出。当所有人看到一切时，推理谁说了什么是简单的。
- **Blackboard**在 agent 多、角色同质但实例众多（swarm）、对话长期运行时胜出。路由节省 token 成本和上下文污染。

生产系统通常混合：顶部一个小全池（规划层），下面是 blackboard（工作者层）。

### 记忆投毒，一个场景

三个 agent 处理一个研究任务。Agent A 是检索 agent。Agent B 是摘要器。Agent C 是分析师。

1. A 获取一个页面并向共享状态写入消息："该研究报告了 42% 的准确率提升。"
2. 获取的页面实际上说的是"4.2% 提升"。A 幻觉了一个小数点。
3. B 读取共享状态，写入："报告了大幅 42% 准确率提升（来源：A）。"
4. C 读取共享状态，写入："建议采用——42% 提升是变革性的。"
5. 最终报告引用了一个从未存在的 42% 数字。

没有 agent 崩溃。没有测试失败。系统"工作了"。幻觉通过共享状态从一个 agent 的上下文跨越到每个下游 agent 的推理中。

### 为什么这是结构性的

没有共享状态，agent A 的幻觉留在 A 的上下文中。下游 agent 会重新获取或重新推导，可能捕获错误。有了朴素共享状态，A 的上下文变成所有人的上下文，幻觉被洗白为事实。

问题不是共享状态本身——而是**没有溯源和没有独立验证者**的共享状态。三种缓解措施解决这个问题：

1. **在每次写入时标注溯源。** 共享状态中的每个条目记录谁写的、何时、在什么提示下，以及（如适用）agent 引用了什么来源。下游 agent 带着基于溯源的怀疑来读取。
2. **版本化写入；将其视为仅追加。** 更正是一个取代旧条目的新条目，而不是就地更新。审计追踪被保留。
3. **保持至少一个不能写入共享状态的 agent。** 一个只读验证者 agent 抽样条目、重新获取来源并标记不一致。因为它不能写入池，所以不能被池投毒。

### Blackboard 先例（Hayes-Roth，1985）

Blackboard 模式比 LLM agent 早了四十年。Hayes-Roth（1985，"A Blackboard Architecture for Control"）描述了观察全局 blackboard、贡献部分解决方案并触发其他来源的专家知识源。2026 年的 blackboard（CA-MCP、Matrix）是相同模式，以 LLM agent 作为知识源，JSON blob 作为部分解决方案。旧文献有关于写竞争、机会主义控制和一致性的已记录解决方案，现代系统正在重新发现。

### 投影 vs 全视图

纯 blackboard 给每个订阅者相同的投影（主题范围）。更激进的设计是**每 agent 投影**：每个 agent 获得为其角色定制的视图。LangGraph 的 state reducer 是 2026 年的规范实现——reducer 函数将全局状态折叠为角色特定的切片。

每 agent 投影扩展更远但需要 schema。没有 schema，你在每个 agent 的提示中重建临时投影。

### 写竞争模式

多个 agent 同时写入是并发问题，不仅仅是 LLM 问题。三种模式有效：

- **顺序写入者（单生产者）。** 所有写入通过一个协调者 agent 序列化。简单，但是瓶颈。
- **带版本的乐观并发。** 每个条目有版本；写入者在版本不匹配时失败并重试。经典数据库技术。
- **主题分区。** 不同 agent 拥有不同主题。没有跨主题竞争。需要设计的分区边界。

大多数 2026 框架默认顺序写入者，因为 LLM 调用足够慢，竞争罕见且瓶颈不伤。

### 不可写验证者

最承重的缓解措施是只读验证者。实现规则：

- 验证者与团队共享状态（读取 blackboard 或池）。
- 验证者没有共享状态的写句柄——只有到单独验证通道的。
- 验证者独立获取写入中引用的来源。标记不一致。
- 验证者自己的输出路由到人类或单独的决策 agent，永远不反馈到池中。

没有这种分离，验证者的输出变成池中的新条目，这意味着被投毒的池投毒验证者，验证者投毒其验证。

## Build It

`code/main.py` 用 stdlib Python 实现两种拓扑加一个玩具投毒攻击和三种缓解措施。

- `MessagePool` — 线程安全的仅追加日志，带完整读出。
- `Blackboard` — 主题键控的 pub/sub，带每 agent 订阅。
- `ProvenanceEntry` — 每次写入记录 (writer, timestamp, prompt_hash, source_uri)。
- `PoisoningScenario` — 运行一个三 agent 研究任务，其中 agent A 幻觉一个小数点。打印最终报告。
- `Verifier` — 一个只读 agent，重新获取来源并标记不一致。用验证者在场运行相同场景。

运行：

```
python3 code/main.py
```

预期输出：
- 运行 1（无验证者）：幻觉的 42% 传播到最终报告。
- 运行 2（有验证者）：验证者标记不一致，池被标记为"flagged"，最终报告包含撤回。

## Use It

`outputs/skill-memory-auditor.md` 是一个技能，审计任何多 agent 系统的共享记忆设计的溯源、版本控制和验证者分离。在生产前对新多 agent 架构运行它。

## Ship It

对于任何共享记忆设计：

- 在每次写入时记录溯源：`(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`。
- 使日志仅追加。更正是引用被取代条目的新条目。
- 部署至少一个具有独立来源访问的只读验证者 agent。
- 将验证者输出路由到单独通道，而不是反馈到共享池。
- 记录取代写入的比率——上升的比率是幻觉模式的早期证据。

## 练习

1. 运行 `code/main.py`。确认运行 1 传播幻觉，运行 2 捕获它。
2. 添加第二个幻觉：agent B 编造一个数据集大小。验证者应该在不为任何一个手动调优的情况下捕获两者。
3. 将全池切换为带主题分区（`prices`、`summaries`、`analyses`）的 blackboard。主题分区使哪些投毒场景更难实施，哪些没有帮助？
4. 阅读 Hayes-Roth（1985，"A Blackboard Architecture for Control"）。识别论文中本课未讨论的两种控制模式，2026 系统会从中受益。
5. 阅读 CA-MCP（arXiv:2601.11595）。将其 Shared Context Store 映射到 `code/main.py` 中的 MessagePool 或 Blackboard 类。CA-MCP 在其之上添加了什么原语？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 消息池 | "共享聊天历史" | 每个 agent 读取的仅追加日志。完全透明，扩展性差。 |
| Blackboard | "共享工作区" | 主题键控的 pub/sub。Agent 订阅相关主题。扩展更远。 |
| 溯源 | "谁写了什么" | 每次写入的元数据：写入者、时间戳、提示、来源。 |
| 记忆投毒 | "幻觉扩散" | 一个 agent 的错误进入共享状态，下游 agent 将其作为事实采纳。 |
| 仅追加 | "无就地更新" | 更正是取代的新条目。保留审计追踪。 |
| 不可写验证者 | "独立审计员" | 重新获取来源并标记不一致的只读 agent。 |
| 投影 | "范围视图" | 从全局状态计算的每 agent 视图。LangGraph reducer 是规范案例。 |
| 知识源 | "专家 agent" | Hayes-Roth 1985 年对 blackboard 参与者的术语。 |

## 延伸阅读

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST 分类法；记忆投毒是协调失败子族
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — 协调 MCP 服务器的共享上下文存储
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) — 无中央编排器的基于消息队列的 blackboard
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 生产中的每 agent 投影模式
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 生产部署中的溯源和验证笔记
