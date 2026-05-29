# LangGraph：有状态图与持久执行

> LangGraph 是 2026 年低层有状态编排的参考实现。智能体是状态机；节点是函数；边是转换；状态不可变且每步之后都有检查点。从任何故障点精确恢复。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## 学习目标

- 描述 LangGraph 的核心模型：带不可变状态、函数节点、条件边和步后检查点的状态机。
- 列举文档强调的四种能力：持久执行、流式输出、人机协作、全面记忆。
- 解释 LangGraph 支持的三种编排拓扑：supervisor、点对点（swarm）、层级式（嵌套子图）。
- 用 stdlib 实现一个带不可变状态、条件边和检查点/恢复循环的状态图。

## 问题

智能体和工作流共享一个问题：当一个 40 步的运行在第 38 步失败时，你想从第 38 步恢复，而不是从头开始。二等公民的状态模型让运维人员在一个假设全新运行的库周围 hack 重试逻辑。

LangGraph 的设计答案：状态是一等公民的类型化对象，变更是显式的，检查点在每个节点后持久化。恢复就是一个 `load_state(session_id)` 调用。

## 概念

### 图

图由以下部分定义：

- **状态类型。** 一个 typed dict（或 Pydantic 模型），每个节点都读取和变更它。
- **节点。** 纯函数 `(state) -> state_update`。更新在返回后合并到状态中。
- **边。** 条件或直接的节点间转换。
- **入口和出口。** `START` 和 `END` 哨兵节点标记边界。

示例：一个带 `classify`、`refund`、`bug`、`sales`、`done` 节点的智能体——路由工作流作为图。

### 持久执行

每个节点返回后，运行时序列化状态并写入检查点器（SQLite、Postgres、Redis、自定义）。在第 N 步失败时，运行时可以 `resume(session_id)` 并从第 N+1 步以精确状态继续。

LangGraph 文档明确强调了这一点对生产用户的重要性：Klarna、Uber、J.P. Morgan。核心不是图的形状；而是图的形状加上检查点使恢复变得廉价。

### 流式输出

每个节点可以 yield 部分输出。图向调用者流式传输每节点增量事件，使 UI 在图运行时实时更新。

### 人机协作

在节点之间检查和修改状态。实现方式：在关键节点前暂停，将状态展示给人类，接受修改，恢复。检查点器使这变得容易，因为状态已经序列化了。

### 记忆

短期（运行内——状态中的对话历史）和长期（跨运行——通过检查点器加独立的长期存储持久化）。LangGraph 通过工具与外部记忆系统（Mem0、自定义）集成。

### 三种拓扑

1. **Supervisor。** 中央路由 LLM 分发给专家子智能体。`langgraph-supervisor` 中的 `create_supervisor()`（不过 LangChain 团队在 2026 年建议通过工具调用直接实现，以获得更多上下文控制）。
2. **Swarm / 点对点。** 智能体通过共享工具表面直接交接。没有中央路由器。
3. **层级式。** Supervisor 管理子 supervisor，实现为嵌套子图。

### 这种模式出错的地方

- **检查点太小。** 只检查点对话轮次会让工具状态和记忆写入不可恢复。完整状态必须序列化。
- **非确定性节点。** 恢复假设节点输入产生相同的状态更新。随机种子、墙钟时间、外部 API 必须被捕获。
- **过度使用条件边。** 每条边都是条件的图是一个无法推理的状态机。优先使用线性链加偶尔的分支。

## Build It

`code/main.py` 实现了一个 stdlib 有状态图：

- `State` — 带 `messages`、`step`、`route`、`output`、`human_approval` 的 typed dict。
- `Node` — 接受 state 并返回更新 dict 的 callable。
- `StateGraph` — 节点 + 边 + 条件边 + run + resume。
- `SQLiteCheckpointer`（内存模拟）— 每个节点后序列化状态；`load(session_id)` 恢复。
- 演示图：classify -> branch(refund / bug / sales) -> human gate -> send。

运行：

```
python3 code/main.py
```

轨迹展示第一次运行在 human gate 失败、持久化、然后恢复产生最终输出。

## Use It

- **LangGraph** — 参考实现，生产就绪。使用 `create_react_agent`、`create_supervisor`，或构建自己的图。
- **AutoGen v0.4**（Lesson 14）— 高并发场景的 Actor 模型替代方案。
- **Claude Agent SDK**（Lesson 17）— 带内置 session store 的托管 harness。
- **自定义** — 当你需要精确控制状态形状或检查点器后端时。

## Ship It

`outputs/skill-state-graph.md` 在任何目标运行时中生成 LangGraph 形状的状态图，带检查点和恢复。

## 练习

1. 从 `classify` 到 `end` 添加一条条件边，当分类置信度低于阈值时触发。在人工手动设置 `route` 后恢复运行。
2. 将类 SQLite 模拟替换为真正的 SQLite 检查点器。测量每步序列化开销。
3. 实现并行边：两个节点并发运行，通过自定义 reducer 合并。不可变状态在这里带来了什么？
4. 阅读 `langgraph-supervisor` 参考。将玩具移植到 `create_supervisor`。比较轨迹形状。
5. 添加流式输出：每个节点在运行时 yield 部分状态。打印到达的增量。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 状态图 | "智能体即状态机" | 类型化状态 + 节点 + 边 + reducer |
| 检查点器 | "持久化后端" | 每个节点后序列化状态；启用恢复 |
| Reducer | "状态合并器" | 将当前状态与节点更新组合的函数 |
| 条件边 | "分支" | 由状态函数选择的边 |
| 子图 | "嵌套图" | 作为另一个图中节点使用的图 |
| 持久执行 | "从故障恢复" | 以精确状态在最后成功的节点重启 |
| Supervisor | "路由 LLM" | 专家子智能体的中央调度器 |
| Swarm | "P2P 智能体" | 智能体通过共享工具交接；无中央路由器 |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 参考文档
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) — supervisor 模式 API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — Actor 模型替代方案
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — session store 和子智能体
