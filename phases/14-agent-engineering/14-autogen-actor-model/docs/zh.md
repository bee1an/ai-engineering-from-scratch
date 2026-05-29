# AutoGen v0.4：Actor 模型与智能体框架

> AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕 Actor 模型重新设计了智能体编排。异步消息交换、事件驱动智能体、故障隔离、原生并发。该框架现已进入维护模式，Microsoft Agent Framework（2025 年 10 月公开预览）成为继任者。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## 学习目标

- 描述 Actor 模型：智能体作为 actor，消息作为唯一的 IPC，每个 actor 故障隔离。
- 列举 AutoGen v0.4 的三个 API 层——Core、AgentChat、Extensions——以及各自的用途。
- 解释为什么将消息投递与处理解耦能带来故障隔离和原生并发。
- 用 Python stdlib 实现一个 actor 运行时，并将一个双智能体代码审查流移植到上面。

## 问题

大多数智能体框架是同步的：一个智能体产出，一个智能体消费，在调用栈中。故障会崩溃整个栈。并发是后加的。分布式需要重写。

AutoGen v0.4 的答案：Actor 模型。每个智能体是一个拥有私有收件箱的 actor。消息是唯一的交互方式。运行时将投递与处理解耦。故障隔离到单个 actor。并发是原生的。分布式只是不同的传输层。

## 概念

### Actor

一个 actor 拥有：

- 私有状态（外部永远不能直接触碰）。
- 收件箱（消息队列）。
- 处理器：`receive(message) -> effects`，其中 effects 可以是"回复"、"发送给其他 actor"、"生成新 actor"、"更新状态"、"停止自身"。

两个 actor 不能共享内存。它们只能发送消息。

### AutoGen v0.4 的三个 API 层

1. **Core。** 低层 actor 框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat。** 任务驱动的高层 API（替代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 集成——OpenAI、Anthropic、Azure、工具、记忆。

### 为什么解耦很重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 会同步阻塞 agent_a 直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 将消息放入 agent_b 的收件箱并返回。运行时稍后投递。三个后果：

- **故障隔离。** Agent B 崩溃不会崩溃 Agent A——运行时在 B 的处理器中捕获故障并决定怎么做（日志、重试、死信）。
- **原生并发。** 多条消息同时在途；actor 并发处理其收件箱。
- **分布式就绪。** 收件箱 + 传输层是相同的抽象，无论 actor 在进程内还是在另一台主机上。

### 拓扑

- **RoundRobinGroupChat。** 智能体按固定轮转顺序发言。
- **SelectorGroupChat。** 一个选择器智能体根据对话上下文选择下一个发言者。
- **Magentic-One。** 用于网页浏览、代码执行、文件处理的参考多智能体团队。基于 AgentChat 构建。

### 可观测性

内置 OpenTelemetry 支持。每条消息发出一个 span；工具调用携带 `gen_ai.*` 属性，遵循 2026 OTel GenAI 语义约定（Lesson 23）。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 对研究和原型开发是稳定的。Microsoft 已将活跃开发转移到 Microsoft Agent Framework（2025 年 10 月 1 日公开预览；1.0 GA 目标 2026 年 Q1 末）。AutoGen 模式可以干净地向前移植——Actor 模型是持久的理念。

## Build It

`code/main.py` 实现了一个 stdlib actor 运行时：

- `Message` — 带 `sender`、`recipient`、`topic`、`body` 的类型化载荷。
- `Actor` — 抽象类，带 `receive(message, runtime)`。
- `Runtime` — 带共享队列、投递、故障隔离的事件循环。
- 双 actor 演示：`ReviewerAgent` 审查代码，`ChecklistAgent` 运行检查清单；它们交换消息直到达成共识。

运行：

```
python3 code/main.py
```

轨迹展示消息投递、一个 actor 中的模拟故障不会崩溃另一个、以及在共享裁决上的收敛。

## Use It

- **AutoGen v0.4/v0.7**（维护模式）— 对研究、原型开发、多智能体模式是稳定的。
- **Microsoft Agent Framework**（公开预览）— 前进路径；相同的 actor 模型理念在刷新的 API 中。
- **LangGraph swarm 拓扑**（Lesson 13）— 通过共享工具交接的类似模式。
- **自定义 actor 运行时** — 当你需要特定传输层（NATS、RabbitMQ、gRPC）时。

## Ship It

`outputs/skill-actor-runtime.md` 为给定的多智能体任务生成一个最小 actor 运行时加团队模板（RoundRobin 或 Selector）。

## 练习

1. 添加死信队列：当处理器抛出异常时，将失败消息停放供人工检查。在你的玩具中 DLQ 被命中的频率如何？
2. 实现 `SelectorGroupChat`：一个选择器 actor 根据对话状态选择谁处理下一条消息。
3. 添加分布式传输：将进程内队列替换为 JSON-over-HTTP 服务器，使 actor 可以在不同进程中运行。
4. 为每条消息接入一个 OTel span（或无操作替代）。发出 `gen_ai.agent.name`、`gen_ai.operation.name`，遵循 Lesson 23。
5. 阅读 AutoGen v0.4 的架构文章。将你的玩具移植到真正的 `autogen_core` API。你跳过了什么在生产中重要的东西？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Actor | "智能体" | 私有状态 + 收件箱 + 处理器；无共享内存 |
| Message | "事件" | 类型化载荷；actor 交互的唯一方式 |
| 收件箱 | "邮箱" | 每个 actor 的待处理消息队列 |
| Runtime | "智能体宿主" | 路由消息并隔离故障的事件循环 |
| Topic | "频道" | actor 之间的命名发布-订阅路由 |
| 故障隔离 | "Let it crash" | 一个 actor 失败不会崩溃其他 actor |
| RoundRobinGroupChat | "固定轮转团队" | 智能体按顺序轮流 |
| SelectorGroupChat | "上下文路由团队" | 选择器选择下一个发言者 |
| Magentic-One | "参考团队" | 用于 web + 代码 + 文件的多智能体小队 |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 重新设计文章
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 图形替代方案
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen 默认发出的 span
