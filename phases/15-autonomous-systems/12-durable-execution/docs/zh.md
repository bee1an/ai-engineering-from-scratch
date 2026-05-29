# 长时运行后台 Agent：持久执行

> 生产级长时 agent 不会跑在 `while True` 里。每次 LLM 调用都变成一个带 checkpoint、重试和重放的 activity。Temporal 的 OpenAI Agents SDK 集成于 2026 年 3 月正式 GA。Claude Code Routines（Anthropic）可以在没有持久本地进程的情况下运行定时 Claude Code 调用。会话在等待人工输入时暂停，能跨部署存活，并从以 `thread_id` 为键的最新 checkpoint 恢复。新的开发体验背后是一个老模式——工作流编排——只是多了一个新输入：LLM 调用作为非确定性 activity，必须在恢复时被确定性地重放。

**Type:** Learn
**Languages:** Python (stdlib, minimal durable-execution state machine)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~60 minutes

## 问题

假设一个 agent 运行了四个小时。它调用了三个工具，两次提示用户，发起了四十次 LLM 调用。运行到一半时，宿主机重启了。会发生什么？

- 在朴素的 `while True` 循环中：一切丢失。运行从头开始。三次工具调用（带有真实副作用）再次执行。用户被再次要求批准已经批准过的事情。四十次 LLM 调用重新计费。
- 有持久执行：运行从最近的 checkpoint 恢复。已完成的 activity 不会重新执行；它们的结果从持久日志中重放。用户不需要重新批准已批准的内容。已完成的 LLM 调用不会重新计费。

这和工作流引擎十年来一直在做的事情是同一个模式（Temporal、Cadence、Uber 的 Cherami）。新的地方在于 LLM 调用现在是一种 activity——非确定性的、昂贵的、有副作用的——而且它们完美契合这个模式。

本课的主线：长时可靠性会衰减（METR 观察到"35 分钟退化"——成功率大致随时间跨度呈二次方下降）。持久执行让你能运行超出可靠性曲线所支持的时长，如果设计正确这是一种安全的失败方式，如果设计错误则是不安全的。

## 概念

### Activity、Workflow 和重放

- **Workflow**：确定性编排代码。定义 activity 的顺序、分支、等待。必须是确定性的，这样才能从事件日志重放而不产生意外分歧。
- **Activity**：非确定性的、可能失败的工作单元。LLM 调用、工具调用、文件写入、HTTP 请求。每个 activity 的输入和（完成后的）输出都会被记录。
- **Event log**：持久后端存储。每个 activity 的开始、完成、失败、重试，以及每个 workflow 决策都被记录。
- **Replay**：恢复时，workflow 代码从头重新运行；每个已完成的 activity 返回其记录的结果而不重新执行。只有未完成的 activity 才会真正运行。

这和 React 对虚拟 DOM 的重新渲染，或 Git 从 commit 重建工作树是同一个形状。编排器的确定性是让持久化变得廉价的关键。

### 为什么 LLM 调用适合这个模式

LLM 调用是：
- 非确定性的（temperature > 0；即使 temperature 0 也会跨模型版本漂移）。
- 昂贵的（金钱和延迟）。
- 可能失败的（速率限制、超时）。
- 有副作用的（如果它们调用工具）。

这正是 activity 的特征。将每次 LLM 调用包装为 activity，你就获得了指数退避重试、跨重启的 checkpoint，以及可重放的调试 trace。

### 以 `thread_id` 为键的 Checkpoint

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 和 Claude Code Routines 都收敛到了相同的 API 形状：一个 `thread_id`（或等价物）标识会话；每次状态转换持久化到后端（PostgreSQL 为默认，SQLite 用于开发，Redis 用于缓存）；恢复时读取最新 checkpoint。

后端选择很重要：

- **PostgreSQL**：持久、可查询、跨部署存活。LangGraph 的默认选择。
- **SQLite**：仅限本地开发；跨主机丢失数据。
- **Redis**：快但短暂，除非配置了 AOF/快照。
- **Cloudflare Durable Objects**：透明分布式；按唯一键作用域；存活数小时到数周。

### 人工输入作为一等状态

Propose-then-commit（第 15 课）需要一个持久的"等待人工"状态。Workflow 暂停，外部队列持有待处理请求，审批到达后从该点精确恢复。没有持久化这只是尽力而为；有了持久化，隔夜到达的审批仍能让 workflow 在早上继续。

### 35 分钟退化

METR 观察到，所有被测量的 agent 类别在连续运行超过约 35 分钟后都表现出可靠性衰减。任务时长翻倍，失败率大约翻四倍。持久执行不能修复这个问题；它让你能运行超出可靠性曲线所支持的时长。安全的模式是将持久化与需要在重新进入时进行新鲜 HITL 的 checkpoint 结合，并配合预算 kill switch（第 13 课）来限制总计算量，不论挂钟时间。

### 持久执行不适用的场景

- 运行时间短于几分钟且没有人工输入。开销 > 收益。
- 纯只读信息检索。
- 正确性要求端到端在一个上下文窗口内完成的任务（某些推理任务；某些一次性生成）。

## Use It

`code/main.py` 用 stdlib Python 实现了一个最小持久执行引擎。它支持：

- `@activity` 装饰器，将输入和输出记录到 JSON event log。
- 一个 workflow 函数，按顺序编排 activity。
- 一个 `run_or_replay(workflow, event_log)` 函数，重放已完成的 activity 而不重新执行。

驱动程序模拟一个三步 activity workflow，中途崩溃，展示 (a) 朴素重试重新执行所有内容 vs (b) 重放只运行缺失的 activity。

## Ship It

`outputs/skill-durable-execution-review.md` 审查一个拟议的长时运行 agent 部署是否具备正确的持久执行形状：activity、确定性、checkpoint 后端、人工输入状态，以及恢复时的 HITL 策略。

## 练习

1. 运行 `code/main.py`。观察朴素重试和重放之间 activity 执行次数的差异。改变崩溃点，展示重放计数相应变化。

2. 将玩具引擎改为显式使用 `thread_id`。模拟两个并发会话共享引擎，确认它们的 event log 不会冲突。

3. 取玩具引擎中的一个 activity。引入一个非确定性（在 workflow 决策中使用挂钟时间戳）。演示重放时的分歧。解释真实引擎如何处理这个问题（副作用注册、`Workflow.now()` API）。

4. 阅读 LangChain 的"Runtime behind production deep agents"文章。列出运行时持久化的每个状态，并说明每个覆盖了哪种失败模式。

5. 为一个 6 小时的自主编码任务设计 checkpoint 策略。在哪里做 checkpoint？崩溃恢复是什么样的？什么需要新鲜的 HITL？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| Workflow | "Agent 的脚本" | 确定性编排代码；可从 event log 重放 |
| Activity | "一个步骤" | 非确定性单元（LLM 调用、工具调用）；执行前后都记录 |
| Event log | "后端存储" | 每次状态转换的持久记录 |
| Replay | "恢复" | 重新运行 workflow；已完成的 activity 返回记录结果而不重新执行 |
| Checkpoint | "存档点" | 以 thread_id 为键的持久化状态；恢复时取最新的 |
| thread_id | "会话键" | 界定持久状态作用域的标识符 |
| 35 分钟退化 | "可靠性衰减" | METR：成功率随时间跨度大致呈二次方下降 |
| 非确定性 | "重放漂移" | 挂钟时间、随机数、LLM 输出；必须注册为副作用 |

## 延伸阅读

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 预算、轮次和恢复语义。
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent 形状。
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 具体运行时需求。
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — LLM 调用的 activity 形状。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 35 分钟退化的参考来源。
