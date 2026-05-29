# Agent 框架权衡 — LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都在卖同一个 demo（研究 agent 生成报告），也都藏着同一个 bug（state schema 和编排层打架）。选那个核心抽象与你问题形状匹配的框架；其他的都是你要写两遍的胶水代码。

**类型：** 学习
**语言：** Python
**前置课程：** Phase 11 · 09 (Function Calling), Phase 11 · 16 (LangGraph)
**时长：** ~45 分钟

## 问题

你有一个需要多次 LLM 调用的任务。可能是研究工作流（规划、搜索、总结、引用），可能是代码审查流水线（解析 diff、评审、修补、验证），也可能是一个多轮助手——订机票、写邮件、报销。你选了一个框架。

三天后，你发现框架的抽象在漏水。CrewAI 给你角色，但当"研究员"需要把结构化计划交给"写手"时就打架了。AutoGen 给你 agent 之间的对话，但没有一等公民的 state，所以你的 checkpoint 是一个对话日志的 pickle。LangGraph 给你状态图，但强迫你在还不知道 agent 会做什么之前就命名每个转换。Agno 给你单 agent 抽象，当你想扇出到三个并发 worker 时就崩了。

解决方案不是"选最好的框架"，而是让框架的核心抽象匹配你问题的形状。这节课画出这张地图。

## 概念

![Agent 框架矩阵：核心抽象 vs 问题形状](../assets/framework-matrix.svg)

2026 年有四个框架主导市场。它们的核心抽象并不相同。

| 框架 | 核心抽象 | 最佳适配 | 最差适配 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — 类型化 state、节点、条件边、checkpointer。 | 需要显式 state 和 human-in-the-loop 中断的工作流；需要时间旅行调试的生产 agent。 | 拓扑未知的松散角色驱动头脑风暴。 |
| **CrewAI** | `Crew` — 角色（目标、背景故事）、任务、流程（顺序或层级）。 | 角色扮演或人设驱动的工作流，短线性/层级计划。 | 超出 crew 轮次历史的有状态场景；复杂分支。 |
| **AutoGen** | `ConversableAgent` 对 — 两个或多个 agent 轮流对话直到退出条件。 | 多 agent *对话*（师生、提议-批评、演员-审阅者），思考从聊天中涌现。 | 已知 DAG 的确定性工作流；需要跨重启持久 state 的场景。 |
| **Agno** | `Agent` — 单个 LLM + tools + memory，可组合成 team。 | 快速构建的单 agent 和轻量 team；强多模态和内置存储驱动。 | 需要自定义 reducer 的深度显式分支图。 |

### "抽象"到底意味着什么

框架的核心抽象就是你在白板上画架构时画的那个东西。

- **LangGraph** → 你画一张图。节点是步骤，边是转换，每个点的 state 对象都有类型。心智模型是状态机。
- **CrewAI** → 你画一张组织架构图。每个角色有职位描述，经理路由任务。心智模型是一个小型专家团队。
- **AutoGen** → 你画一个 Slack 私信。两个 agent 互发消息；需要时第三个加入当主持人。心智模型是聊天。
- **Agno** → 你画一个方框，工具挂在上面。把方框并排放就是 team。心智模型是"自带电池的 agent"。

### State 问题

State 是大多数框架选择在生产中崩溃的地方。

- **LangGraph。** 类型化 state（`TypedDict` 或 Pydantic model），每字段 reducer，一等公民 checkpointer（SQLite/Postgres/Redis）。恢复、中断和时间旅行开箱即用。*（见 Phase 11 · 16。）*
- **CrewAI。** State 通过 `context` 字段以字符串形式在任务间流动，或通过 `output_pydantic` 结构化传递。没有开箱即用的持久化 per-crew 存储；如果 crew 需要在重启后存活，你得自己加。
- **AutoGen。** State 是聊天历史和用户定义的 `context`。对话记录可持久化；任意工作流 state 不行，除非你写适配器。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB）通过 `storage=` 附加到 `Agent` — 对话会话和用户记忆自动持久化。不是完整的图 checkpointer；是会话存储。

### 分支问题

每个非平凡 agent 都会分支。谁决定分支很重要。

- **LangGraph** — 你决定，通过条件边。路由是一个带命名分支的 Python 函数。分支是编译图中的一等公民；checkpointer 记录走了哪个分支。
- **CrewAI** — 层级模式下经理决定；顺序模式下你在构建时决定。路由隐含在任务列表中；经理 prompt 之外没有一等公民的"if"。
- **AutoGen** — agent 通过聊天决定。分支从谁下一个发言中涌现。`GroupChatManager` 选择下一个发言者；你可以手写 `speaker_selection_method`，但默认是 LLM 驱动的。
- **Agno** — agent 通过调用哪个 tool 来决定。Team 有 coordinator/router/collaborator 模式；超出这些的分支是开发者的责任。

### 可观测性问题

- **LangGraph** — 通过 LangSmith 或任何 OTel exporter 的 OpenTelemetry。每个节点转换是一个 trace span；checkpoint 同时是可重放的 trace。LangSmith 是第一方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI** — 2025 年底起一等公民 OpenTelemetry；与 Langfuse、Phoenix、Opik、AgentOps 集成。
- **AutoGen** — 通过 `autogen-core` 的 OpenTelemetry 集成；AgentOps 和 Opik 有连接器。追踪粒度是 per-agent-message，不是 per-node。
- **Agno** — 内置 `monitoring=True` 标志加 OpenTelemetry exporter；与 Langfuse 紧密集成用于会话 trace。

### 成本和延迟

四个框架都增加每次调用的开销（框架逻辑、验证、序列化）。开销从小到大的粗略排序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要取决于框架做了多少额外的 LLM 路由。CrewAI 的层级经理花 token 决定谁下一个；AutoGen 的 `GroupChatManager` 同理。LangGraph 只在你写 `llm.invoke` 的地方花 token。Agno 的单 agent 路径很薄。

当每次运行的成本很重要时，优先选择显式路由（LangGraph 边、AutoGen `speaker_selection_method`）而非 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** tools、retrievers、LLMs。一等公民 MCP 适配器（tools 作为 MCP servers 导入）。
- **CrewAI** ↔ tools 继承自 `BaseTool`；LangChain tools、LlamaIndex tools 和 MCP tools 都能适配进来。Crew 间委托通过 `allow_delegation=True`。
- **AutoGen** → `FunctionTool` 包装任何 Python callable；MCP 适配器可用。与 AG2 生态系统紧密耦合用于 agent-to-agent 模式。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；MCP 适配器；tools 可在 agents 和 teams 间共享。

## 技能

> 你能用一句话解释为什么某个框架适合某个 agent 问题。

构建前检查清单：

1. **画出形状。** 这是一张图（类型化 state、命名转换）？一个角色扮演（专家交接工作）？一个聊天（agents 聊到结束）？一个带 tools 的单 agent？
2. **决定谁分支。** 开发者决定分支 → LangGraph。经理 agent 决定 → CrewAI 层级模式。聊天涌现 → AutoGen。Tool 调用决定 → Agno。
3. **检查 state 预算。** 你需要从 checkpoint 恢复？时间旅行？运行中人工中断？如果是，LangGraph 是默认选择；Agno sessions 覆盖对话范围的 state。
4. **检查成本预算。** LLM 选择的路由每轮都花额外 token。如果 agent 每天运行数千次，优先选择显式路由。
5. **预算框架开销。** 每个框架都是一个额外依赖。如果任务只是两次 LLM 调用和一个 tool，写 30 行纯 Python；没有框架比没有框架更便宜。

在你能画出图、组织架构图、聊天或 agent 方框之前，拒绝使用框架。拒绝选一个让你为了实际需要的东西而与其 state 模型搏斗的框架。

## 决策矩阵

| 问题形状 | 推荐框架 | 原因 |
|---------------|---------------------|-----|
| 带类型化 state、人工审批、长时间运行的工作流 DAG | LangGraph | 一等公民 state、checkpointer、中断、时间旅行。 |
| 有明确角色的研究/写作流水线 | CrewAI（顺序）或 LangGraph 子图 | 每任务一角色在 CrewAI 中表达成本低；分支变复杂时升级到 LangGraph。 |
| 提议者-批评者或师生对话 | AutoGen | 双 agent 聊天是其原生形状。 |
| 带 tools、sessions、memory 的单 agent | Agno | 最薄的设置，内置存储和 memory。 |
| 数千个并行扇出加 reducer | LangGraph + `Send` | 唯一有一等公民并行分发 API 的。 |
| 快速原型，不想绑定框架 | 纯 Python + provider SDK | 没有框架是最快的框架。 |

## 练习

1. **简单。** 用同一个任务 — "研究 Anthropic 总部，写一份 200 字简报，引用来源" — 分别在 LangGraph（四个节点：plan、search、write、cite）和 CrewAI（三个角色：researcher、writer、editor）中实现。报告每次运行的 token 成本和代码行数。
2. **中等。** 在 AutoGen（researcher ↔ writer 聊天，editor 通过 `GroupChat` 加入）和 Agno（单 agent 带 `search_tools` 和 `write_tools`，加 session store）中构建同一任务。按 (a) 每次运行成本、(b) 崩溃后恢复能力、(c) 在写入步骤前注入人工审批的能力 对四个实现排名。
3. **困难。** 构建一个决策树脚本 `pick_framework.py`，接受简短问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`）并返回推荐和一句话理由。用你自己设计的六个案例验证。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Orchestration | "agents 怎么协调" | 决定下一个运行哪个节点/角色/agent 的层。 |
| Durable state | "重启后恢复" | 在进程死亡后存活的 state，附加到 checkpoint 或 session store。 |
| LLM-selected routing | "让模型决定" | 一个 planner LLM 每轮选择下一步；灵活但每次决策都花 token。 |
| Explicit routing | "开发者决定" | 一个 Python 函数或静态边选择下一步；便宜且可审计。 |
| Crew | "一个 CrewAI 团队" | 角色 + 任务 + 流程（顺序或层级）绑定成一个可运行单元。 |
| GroupChat | "AutoGen 的多 agent 聊天" | 带发言者选择器的 N 个 agent 之间的受管对话。 |
| Team (Agno) | "多 agent Agno" | Route / coordinate / collaborate 模式下的一组 agents。 |
| StateGraph | "LangGraph 的图" | 类型化 state、节点、条件边、checkpointer 抽象。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph、checkpointers、interrupts、time-travel。
- [CrewAI documentation](https://docs.crewai.com/) — Crews、Flows、Agents、Tasks、Processes。
- [AutoGen documentation](https://microsoft.github.io/autogen/) — ConversableAgent、GroupChat、teams、tools。
- [Agno documentation](https://docs.agno.com/) — Agent、Team、Workflow、storage、memory。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 框架无关的模式库（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer）。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每个框架都在包装的循环。
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — AutoGen 的设计论文。
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — CrewAI 风格人设栈所基于的角色扮演基础。
- Phase 11 · 16 (LangGraph) — 本课对标的框架。
- Phase 11 · 19 (Reflexion) — 一个能干净映射到 LangGraph 但在 CrewAI 中很别扭的模式。
- Phase 11 · 22 (Production observability) — 如何为你选的框架加装可观测性。
