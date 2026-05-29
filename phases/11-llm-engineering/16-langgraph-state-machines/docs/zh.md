# LangGraph — Agent 的状态机

> 手写的 ReAct 循环是一个 `while True`。用 LangGraph 写的 ReAct 循环是一张你可以 checkpoint、中断、分支和时间旅行的图。Agent 没变。围绕它的 harness 变了。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**时长：** ~75 分钟

## 问题

你上线了一个 function-calling agent。它工作了三轮，然后出了问题：模型尝试了一个返回 500 的 tool，用户在任务中途改了主意，或者 agent 决定在没有人工签字的情况下退款。`while True:` 循环没有钩子。你不能暂停它，不能回退它，也不能分支出"如果模型选了另一个 tool 会怎样"。一旦你把它推过 demo 阶段，agent 就变成了一个要么成功要么失败的黑盒。

下一步一旦看到就很明显。Agent 本身已经是一个状态机 — system prompt 加消息历史加待处理的 tool calls 加下一个动作。把状态机显式化：为"模型思考"、"tool 运行"、"人工审批"设节点，为它们之间的条件转换设边。一旦图是显式的，harness 就免费获得四样东西：checkpointing（在步骤间保存 state）、interrupts（为人工暂停）、streaming（流式传输 token 和中间事件）、time-travel（回退到先前 state 并尝试不同分支）。

LangGraph 就是提供这个抽象的库。它不是 LangChain 意义上的 agent 框架（"这是一个 AgentExecutor，祝你好运"）。它是一个带一等公民 state、一等公民持久化和一等公民中断的图运行时。Agent 循环是你画出来的，不是手写的。

## 概念

![LangGraph StateGraph：节点、边和 checkpointer](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 有三样东西。

1. **State。** 一个类型化 dict（TypedDict 或 Pydantic model），在图中流动。每个节点接收完整 state 并返回部分更新，LangGraph 使用每字段的 *reducer* 来合并 — 对于需要累积的列表用 `operator.add`，默认是覆盖。
2. **Nodes。** Python 函数 `state -> partial_state`。每个是一个离散步骤："调用模型"、"运行 tools"、"总结"。
3. **Edges。** 节点间的转换。静态边去一个地方。条件边接受一个路由函数 `state -> next_node_name`，这样图可以根据模型输出分支。

你编译图。编译绑定拓扑，附加 checkpointer（可选但生产必需），返回一个 runnable。你用初始 state 和 `thread_id` 调用它。每一步执行都持久化一个以 `(thread_id, checkpoint_id)` 为键的 checkpoint。

### 四大超能力

**Checkpointing。** 每次节点转换都把新 state 写入存储（测试用内存，生产用 Postgres/Redis/SQLite）。用相同 `thread_id` 再次调用图即可恢复。图从暂停处继续。

**Interrupts。** 用 `interrupt_before=["human_review"]` 标记一个节点，执行在该节点运行前停止。State 持久化。你的 API 回复用户"等待审批"。之后对同一 `thread_id` 发送 `Command(resume=...)` 的请求恢复执行。

**Streaming。** `graph.stream(state, mode="updates")` 在 state delta 发生时 yield 它们。`mode="messages"` 流式传输模型节点内的 LLM token。`mode="values"` yield 完整快照。你选择在 UI 中展示什么。

**Time-travel。** `graph.get_state_history(thread_id)` 返回完整的 checkpoint 日志。把任何先前的 `checkpoint_id` 传给 `graph.invoke`，你就从那个点 fork。适合调试（"如果模型选了 tool B 会怎样？"）和回放生产 trace 的回归测试。

### Reducer 是关键

每个 state 字段都有一个 reducer。大多数默认值没问题 — 新值覆盖旧值。但消息列表需要 `operator.add` 这样新消息追加而不是替换。并行边通过 reducer 合并它们的更新。如果两个节点都更新 `messages` 而你忘了 `Annotated[list, add_messages]`，第二个会静默胜出，你丢失半个轮次。Reducer 是库中唯一微妙的东西；搞对它，其余的就能组合。

### 四节点 ReAct 图

一个生产 ReAct agent 是四个节点和两条边：

1. `agent` — 用当前消息历史调用 LLM。返回 assistant 消息（可能包含 tool_calls）。
2. `tools` — 执行最后一条 assistant 消息中的任何 tool_calls，把 tool 结果作为 tool 消息追加。
3. 从 `agent` 出发的条件边，如果最后一条消息有 tool_calls 则路由到 `tools`，否则到 `END`。
4. 从 `tools` 回到 `agent` 的静态边。

就这些。你用大约 40 行代码获得完整的 ReAct 循环（Thought → Action → Observation → Thought → …），带 checkpointing、interrupts 和 streaming。

### StateGraph vs Send（扇出）

`Send(node_name, state)` 让一个节点分发并行子图。例如：agent 决定同时查询三个 retriever。每个 `Send` 生成目标节点的一个并行执行；它们的输出通过 state reducer 合并。这就是 LangGraph 不用线程原语表达 orchestrator-workers 模式的方式。

### 子图

一个编译好的图可以作为另一个图中的节点。外部图看到一个单节点；内部图有自己的 state 和自己的 checkpoints。这就是团队构建 supervisor-worker agents 的方式：supervisor 图把用户意图路由到每个领域的 worker 子图。

## 构建

### 步骤 1：state 和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是让消息列表累积而非覆盖的 reducer。忘记它是最常见的 LangGraph bug。

### 步骤 2：用 thread 运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每个 update 是一个 dict `{node_name: state_delta}`。你的前端可以把这些流式传输到 UI，让用户看到"agent 在思考… 调用 search_web… 得到结果… 回答中。"

### 步骤 3：添加 human-in-the-loop 中断

标记一个节点，让执行在它运行前暂停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

State、checkpoint 和 thread 都跨中断持久化。除了执行期间，没有东西在内存中。

### 步骤 4：用于调试的时间旅行

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

传 `None` 作为输入从给定 checkpoint 重放；传一个值则在恢复前把它作为更新追加到该 checkpoint 的 state。这就是你不用重跑整个对话就能复现一次坏的 agent 运行的方式。

### 步骤 5：为生产切换 checkpointer

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 都已内置。`MemorySaver` 用于测试。任何需要跨重启持久化的都需要真正的存储。

## 技能

> 你把 agents 构建为图，而不是 `while True` 循环。

在使用 LangGraph 之前，做一个 60 秒设计：

1. **命名节点。** 每个离散决策或有副作用的动作是一个节点。"Agent 思考"、"tool 运行"、"审阅者批准"、"响应流式传输"。如果你列不出来，任务还不是 agent 形状的。
2. **声明 state。** 最小 TypedDict，每个列表字段都有 reducer。不要把所有东西塞进 `messages`；把任务特定字段（一个工作中的 `plan`、一个 `budget` 计数器、一个 `retrieved_docs` 列表）提升到顶层。
3. **画边。** 除非下一步取决于模型输出，否则用静态边。每条条件边需要一个带命名分支的路由函数。
4. **预先选择 checkpointer。** 测试用 `MemorySaver`，其他用 Postgres/Redis/SQLite。不带 checkpointer 就不要上线 — 没有 checkpointer 意味着没有恢复、没有中断、没有时间旅行。
5. **在 tools 运行前决定中断，而不是之后。** 审批放在进入有副作用节点的边上，这样你可以在造成伤害前取消；验证放在模型输出的边上，这样你可以廉价地拒绝坏调用。
6. **默认流式传输。** UI 用 `mode="updates"`，模型节点内的 token 级流式传输用 `mode="messages"`，eval 期间的完整快照用 `mode="values"`。

拒绝上线没有 checkpointer 的 LangGraph agent。拒绝上线在副作用*之后*才中断的。拒绝上线没有 `add_messages` 作为 reducer 的 `messages` 字段。

## 练习

1. **简单。** 用计算器 tool 和网页搜索 tool 实现上面的四节点 ReAct 图。验证 `list(app.get_state_history(config))` 对一个两轮对话返回至少四个 checkpoints。
2. **中等。** 添加一个 `planner` 节点，在 `agent` 之前运行并把结构化的 `plan: list[str]` 写入 state。让 `agent` 标记计划步骤为已完成。如果 `plan` 在 checkpoint 恢复后丢失（错误的 reducer），测试失败。
3. **困难。** 构建一个 supervisor 图，使用 `Send` 在三个子图（`researcher`、`writer`、`reviewer`）之间路由。每个子图有自己的 state 和 checkpointer。在外部图上添加 `interrupt_before=["writer"]`，这样人工可以在研究简报前审批。确认从先前 checkpoint 的时间旅行只重跑 fork 的分支。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| StateGraph | "LangGraph 的图" | 你在编译前添加节点和边的构建器对象。 |
| Reducer | "字段怎么合并" | 当节点返回该字段的更新时应用的函数 `(old, new) -> merged`；默认是覆盖，`add_messages` 追加。 |
| Thread | "一个对话 ID" | 一个 `thread_id` 字符串，限定一个会话的所有 checkpoints 的范围。 |
| Checkpoint | "一个暂停的 state" | 节点转换后完整图 state 的持久化快照，以 `(thread_id, checkpoint_id)` 为键。 |
| Interrupt | "为人工暂停" | `interrupt_before` / `interrupt_after` 在节点边界停止执行；用 `Command(resume=...)` 恢复。 |
| Time-travel | "从先前步骤 fork" | `graph.invoke(None, config_with_old_checkpoint_id)` 从该 checkpoint 向前重放。 |
| Send | "并行子图分发" | 一个节点可以返回的构造器，用于生成目标节点的 N 个并行执行。 |
| Subgraph | "编译图作为节点" | 一个编译好的 StateGraph 用作另一个图中的节点；保留自己的 state 范围。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph、reducers、checkpointers 和 interrupts 的权威参考。
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 本课使用的心智模型，直接来自源头。
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Postgres/SQLite/Redis 存储、checkpoint 命名空间和 thread IDs 的细节。
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)` 和编辑 state 模式。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每个 LangGraph agent 实现的模式；阅读它了解推理 trace 的原理。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 何时优先选择哪种图形状（chain、router、orchestrator-workers、evaluator-optimizer）。
- Phase 11 · 09 (Function Calling) — 每个 LangGraph agent 节点复用的 tool-call 原语。
- Phase 11 · 14 (Model Context Protocol) — 通过 MCP 适配器插入 LangGraph `ToolNode` 的外部 tool 发现。
- Phase 11 · 17 (Agent framework tradeoffs) — 何时选 LangGraph 而非 CrewAI、AutoGen 或 Agno。
