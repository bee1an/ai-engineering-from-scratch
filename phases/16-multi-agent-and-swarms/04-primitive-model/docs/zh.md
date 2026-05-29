# 多智能体原语模型

> 2026 年发布的每个多 agent 框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是四维设计空间中的一个点。四个原语，仅此而已：agent、handoff、共享状态、编排器。本课从零构建它们，在所有四个上运行一个玩具系统，然后将每个主要框架映射到相同的坐标轴上，让你能用一段话读懂任何新发布。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## 问题

每六个月就有一个新的多 agent 框架发布。2023 年 AutoGen。2024 年 CrewAI。2024 年 LangGraph 和 OpenAI Swarm。2025 年 4 月 Google ADK。2026 年 2 月 Microsoft Agent Framework RC。每个新闻稿都声称是"正确的抽象"。

如果你试图逐个学习，你会精疲力竭。API 看起来不同。文档对"agent"是什么意见不一。一个框架把共享记忆叫"blackboard"，另一个叫"message pool"，第三个叫"StateGraph"。你开始怀疑这个领域只是在空转。

不是的。在营销之下，四个原语是稳定的。学一次，用一段话读懂每个新框架。

## 概念

### 四个原语

1. **Agent** — 一个系统提示加一个工具列表。无状态；每次运行从其系统提示和当前消息历史开始。
2. **Handoff** — 从一个 agent 到另一个的结构化控制转移。机制上是一个返回新 agent 的工具调用，或一条跟随条件的图边。
3. **共享状态** — 任何超过一个 agent 可以读取（有时写入）的数据结构。消息池、blackboard、键值存储、向量记忆。
4. **编排器** — 决定谁下一个发言的角色。选项：显式图（确定性）、LLM 发言者选择器（软性）、上一个发言者的 handoff 调用（OpenAI Swarm）、或队列上的调度器（swarm 架构）。

这就是整个设计空间。每个框架为每个轴选择默认值；其余是表面语法。

### 2026 年每个框架如何映射

| 框架 | Agent | Handoff | 共享状态 | 编排器 |
|------|-------|---------|---------|--------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回 Agent | 调用者的问题 | LLM 的下一个 handoff 调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的 speaker-selector | 消息池 | 选择器函数（LLM 或轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Task 输出链接 | 管理者 LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` reducer | 图，确定性 |
| Microsoft Agent Framework | agent + 编排模式 | 模式特定 | 线程 / 上下文 | 模式特定 |
| Google ADK | agent + A2A card | A2A task | A2A artifacts | 宿主决定 |

表面差异看起来很大。底层：相同的四个旋钮。

### 为什么这很重要

一旦你看到原语，框架比较就变成一个简短的检查清单：

- 编排器是信任 LLM 来路由（Swarm）还是在代码中固定路由（LangGraph）？
- 共享状态是全历史（GroupChat）还是投影的（StateGraph reducer）？
- Agent 能修改彼此的提示（CrewAI manager）还是只能 handoff（Swarm）？

这三个问题回答了 80% 的"哪个框架适合给定问题"。你不再购物"最好的多 agent 框架"，而是开始为你真正关心的轴进行设计。

### 无状态洞察

除共享状态外，每个原语都是无状态的。Agent 是 (prompt, tools) 的函数。Handoff 是一个函数调用。编排器是一个调度器。**系统中唯一有状态的东西是共享状态。** 这就是所有有趣 bug 的所在：记忆投毒（Lesson 15）、消息排序、版本控制、写竞争。

隐藏共享状态的框架（Swarm）把问题推给调用者。集中化共享状态的框架（LangGraph checkpoint、AutoGen pool）使其可检查，但将协调成本转移到共享状态实现上。

### 单个原语的解剖

#### Agent

```
Agent = (system_prompt, tools, model, optional_name)
```

没有记忆。没有状态。两个具有相同系统提示和工具的 agent 是可互换的。看起来像 per-agent 状态的一切实际上在共享状态或 handoff 协议中。

#### Handoff

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现占主导：

- **函数返回** — 工具返回下一个 agent。这是 OpenAI Swarm 模式。Agent 在其工具 schema 中携带路由。
- **图边** — LangGraph。边是声明式的。LLM 产生一个值；条件选择下一个节点。
- **发言者选择** — AutoGen GroupChat。一个选择器函数（有时本身是一次 LLM 调用）读取池并选择下一个发言者。

#### 共享状态

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

最少是一个消息列表。通常更多：结构化制品（CrewAI Task 输出）、类型化上下文（LangGraph reducers）、外部记忆（MCP、向量数据库）。

两种拓扑：**全池**（每个 agent 看到每条消息）和**投影**（agent 看到角色范围的视图）。全池简单但扩展性差。投影池可扩展但需要前期 schema 设计。

#### 编排器

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种风味：

- **静态** — 图在构建时固定（LangGraph 确定性、CrewAI Sequential）。
- **LLM 选择** — LLM 读取池并选择下一个发言者（AutoGen、CrewAI Hierarchical）。
- **Handoff 驱动** — 当前 agent 通过调用 handoff 工具来决定（Swarm）。
- **队列驱动** — 工作者从共享队列拉取；没有显式的下一发言者（swarm 架构、Matrix）。

### 框架之间的差异

一旦原语固定，剩余的设计决策是：

- **记忆策略** — 临时 vs 持久检查点（LangGraph checkpointer）。
- **安全边界** — 谁可以批准 handoff（人在回路中）。
- **成本核算** — 每 agent token 预算。
- **可观测性** — 追踪 handoff，持久化状态以供重放。

所有这些都可以在原语之上实现。它们都不是新原语。

## Build It

`code/main.py` 用约 150 行 stdlib Python 实现了四个原语。没有真正的 LLM——每个 agent 是一个脚本化策略，这样焦点保持在协调结构上。

文件导出：

- `Agent` — 名称、系统提示、工具、策略函数的 dataclass。
- `Handoff` — 返回新 agent 的函数。
- `SharedState` — 线程安全的消息池。
- `Orchestrator` — 三个变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示通过所有三种编排器类型运行相同的三 agent 流水线（research → write → review），并在最后打印消息池。你可以看到输出仅在*谁选择下一个*上不同；agent 和共享状态在各次运行中是相同的。

运行：

```
python3 code/main.py
```

预期输出：三次编排器运行，每种模式一次。每次打印最终消息池。handoff 驱动的运行如果研究员决定提前完成则到达更少的 agent——这就是 LLM 路由权衡的缩影。

## Use It

`outputs/skill-primitive-mapper.md` 是一个技能，读取任何多 agent 代码库或框架文档并返回四原语映射。在新框架发布时运行它，在深入阅读文档之前获得一段话的理解。

## Ship It

在采用新框架之前，为它写原语映射。如果你写不出来，说明文档不完整或框架在发明第五个原语（罕见——检查是否有你没见过的共享状态变体）。

将映射固定在你的架构文档中。当新团队成员加入时，在 API 文档之前发给他们映射。当框架版本变化时，diff 映射，而不是 changelog。

## 练习

1. 用不同的 agent 策略运行 `code/main.py` 三次。观察编排器选择如何改变哪些 agent 运行。
2. 实现第四种编排器类型：队列驱动的，agent 轮询共享状态获取工作。什么死锁可能发生，如何检测？
3. 取 LangGraph 快速入门 (https://docs.langchain.com/oss/python/langgraph/workflows-agents) 并将其重写为四个原语。LangGraph 的哪些抽象 1:1 映射，哪些是便利包装？
4. 阅读 OpenAI Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents)。识别四个原语中 Swarm 使哪个最符合人体工学，哪个推给了调用者。
5. 在此表中找一个完全隐藏共享状态的框架。解释当 agent 需要跨 handoff 协调而不重新读取历史时会出什么问题。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Agent | "带工具的 LLM" | 一个 `(system_prompt, tools, model)` 三元组。无状态。 |
| Handoff | "控制转移" | 一个命名下一个 agent 和可选有效载荷的结构化调用。三种实现：函数返回、图边、发言者选择。 |
| 共享状态 | "记忆" / "上下文" | 多 agent 系统中唯一有状态的部分。消息池或 blackboard。 |
| 编排器 | "协调者" | 决定谁下一个运行的角色。静态图、LLM 选择器、handoff 驱动或队列驱动。 |
| 原语 | "抽象" | 每个框架参数化的四个轴之一。不是框架特性。 |
| 消息池 | "共享聊天历史" | 全历史共享状态。容易推理，扩展性差。 |
| 投影状态 | "范围视图" | 角色特定的共享状态视图。可扩展，需要 schema 设计。 |
| 发言者选择 | "谁下一个说话" | 编排器模式，一个函数（通常是 LLM）从群组中选择下一个 agent。 |

## 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — handoff 驱动编排的最清晰阐述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) — GroupChat + speaker selection 是 LLM 选择编排的参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 图边编排和基于 reducer 的共享状态
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — role-goal-backstory agent，Sequential / Hierarchical 流程
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — 微软将 v0.4 转入维护后的活跃 AutoGen v0.2 分支
