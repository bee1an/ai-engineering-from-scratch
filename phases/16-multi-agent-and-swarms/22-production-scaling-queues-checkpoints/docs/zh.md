# 生产扩展 — 队列、检查点、持久性

> 将多智能体系统扩展到数千个并发运行需要**持久执行**。LangGraph 的运行时在每个 super-step 后写入检查点，以 `thread_id` 为键（默认 Postgres）；worker 崩溃释放租约，另一个 worker 恢复。智能体可以无限期休眠等待人工输入。**MegaAgent**（arXiv:2408.09955）运行每智能体的生产者-消费者队列，有三种状态（Idle / Processing / Response）和两层协调（组内聊天 + 组间管理聊天）。**Fiber/async** 在 LLM 流式传输上击败 thread-per-job：线程 99% 的时间闲置等待 token，fiber 在 I/O 上协作让出。反面观点：Ashpreet Bedi 的"Scaling Agentic Software"主张 **FastAPI + Postgres + 别的都不要**直到负载证明需要——简单架构比预期走得更远。本课构建一个持久检查点日志、一个带状态转换的每智能体工作队列、一个 async-vs-thread 演示，并落地务实的"从简单开始"规则。

**Type:** Learn + Build
**Languages:** Python (stdlib, `asyncio`, `sqlite3`)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## 问题

一个原型多智能体系统在一台笔记本上用三个智能体在内存事件循环中工作。你转向生产：

- 智能体有时运行数小时（长期研究、人在回路等待）。
- Worker 进程崩溃。重启丢失状态。
- 峰值负载是平均的 10 倍；你需要水平扩展。
- 用户按智能体运行付费；你需要精确一次语义来计费。

内存事件循环做不到这些。你需要底层的持久执行层。2026 年的规范选项是：

1. 带检查点的工作流引擎（Temporal, LangGraph runtime）。
2. 消息队列加状态存储（Postgres + SQS/RabbitMQ）。
3. Actor 模型框架（MegaAgent 的每智能体生产者-消费者）。
4. 手写 FastAPI + Postgres（Bedi 的论点）。

本课构建每种的微缩版。

## 概念

### 持久执行，模式

持久执行引擎在每个"步骤"（LangGraph 语言中的 super-step）后持久化完整程序状态。崩溃时：

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

使其工作的要求：

- **可序列化状态。** 所有智能体状态必须可持久化。带活数据库连接的函数闭包无法存活。
- **确定性恢复。** 给定相同状态和相同输入，智能体产生相同动作（或委托给外部确定性预言机处理 LLM 调用）。
- **幂等副作用。** 外部调用（工具调用、支付）必须是幂等的或使用去重键。

LangGraph 在每个 super-step 后写检查点；Temporal 在每个 activity 后写；Restate 使用事件溯源日志。三者实现相同模式。

### LangGraph 的运行时

每个智能体有一个 `thread_id`；状态是类型化字典；每个 super-step 向检查点表写一行。恢复时，运行时从最后一个检查点重放，而非从头开始。智能体可以 `interrupt()` 等待人工输入；运行时持久化并释放 worker。当输入到达时，任何 worker 都可以恢复。

这是 2026 年 4 月的参考生产设计。

### MegaAgent 的每智能体队列

arXiv:2408.09955 描述了一个规模实验：一个集群中数千个并发智能体。架构：

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

两层协调让组内对话密集进行，而组间保持稀疏——用于在数千个智能体中保持成本线性的模式。

### Async vs thread-per-job

LLM 调用是 I/O 密集的。等待下一个 token 的线程 99% 的时间闲置。线程每个约 1MB RAM；在 10,000 个并发调用时，仅栈就是 10GB。

Fiber（Python `asyncio`、Go goroutine、Rust `tokio`）在 I/O 上协作让出。同样的 10,000 个调用舒适地在进程内运行。在 LLM 智能体规模上，async 不是优化——它是架构。

例外：CPU 密集的后处理（嵌入、tokenizer 技巧）仍然需要线程或进程。将你的 I/O 层与 CPU 层分开。

### Bedi 的反面观点

"Scaling Agentic Software"（Ashpreet Bedi, 2026）论证大多数团队在测量负载之前过度工程化。务实的默认：

- FastAPI + Postgres。
- 每个智能体运行是一行；状态用乐观并发就地更新。
- 后台作业通过 `pg_notify` 或简单的 Celery worker。
- 应用代码中的重试策略。

对于可管理任务上 ~100 个并发智能体运行以下的负载，这通常就够了。当你测量到它失败时再升级。

规则：当你遇到简单架构无法解决的具体问题时才采用持久执行框架。过早采用浪费时间在不产生回报的仪式上。

### 精确一次语义

对于付费智能体运行，你需要"有效精确一次"（至少一次投递 + 幂等消费者）。工程手段：

- **每次运行的去重键。** 在每个副作用调用中包含它。
- **Outbox 模式。** 副作用先写入表，然后单独的进程执行它们。两步都幂等。
- **补偿事务。** 当副作用成功但其跟踪写入失败时，调度补偿。

这些是数据库工程模式，不是 LLM 特定的。LLM 税只是 LLM 调用慢；其他一切都是标准分布式系统。

### Rainbow 部署

Anthropic 的多智能体研究系统使用"rainbow deployments"：多个版本的智能体运行时并发运行，这样长期运行的智能体不必在每次代码部署时被杀死。在一部分流量上金丝雀新版本；当旧版本的智能体完成时退役旧版本。

这对长期运行的有状态系统是标准的；2026 年的适配是智能体可以存活数小时，所以部署周期必须适应。

### 规范生产检查清单

- 持久状态（检查点、快照，或 outbox + 可重放日志）。
- 幂等副作用。
- LLM 调用的 async I/O 层。
- 至少一次投递加去重。
- 有状态工作负载的 rainbow/canary 部署。
- 可观测性：每智能体追踪、super-step 审计、重试计数器。

## 动手构建

`code/main.py` 实现：

- `CheckpointStore` — SQLite 支持的检查点日志，以 thread-id 为键。每个 super-step 追加一行。
- `run_with_checkpoint(agent, thread_id)` — 模拟运行中崩溃；第二个 worker 从最后检查点恢复。
- `AgentQueue` — 每智能体 Idle / Processing / Response 状态机，带小型工作队列。
- `demo_async_vs_threads()` — 通过 asyncio 和线程运行 500 个并发模拟"LLM 调用"；报告时钟时间和峰值内存（近似）。

运行：

```
python3 code/main.py
```

预期输出：模拟崩溃后检查点恢复成功；async 版本在 < 1s 内处理 500 个并发调用；线程版本需要几秒且每并发单元使用数量级更多的内存。

## 使用方式

`outputs/skill-scaling-advisor.md` 建议持久执行选择：FastAPI + Postgres、LangGraph runtime、Temporal 或自定义。按负载、状态保留需求和部署频率校准。

## 上线清单

规范生产加固：

- **从简单开始（Bedi 规则）。** FastAPI + Postgres 直到你测量到它失败。
- **优化前先检测一切。** 每次运行延迟直方图、每步时间、重试计数、失败分类。
- **副作用的 Outbox 模式。** 特别是支付和外部 API 调用。
- **Rainbow 部署。** 永远不要在部署期间杀死进行中的智能体运行。
- **当**你遇到特定问题时**采用持久执行引擎（Temporal / LangGraph / Restate）**：数小时的人在回路等待、跨区域协调、复杂的重试/补偿策略。
- **I/O 层用 Async。** 线程仅用于 CPU 密集的后处理。

## 练习

1. 运行 `code/main.py`。确认检查点恢复工作；测量 async vs thread 并发差异。
2. 实现一个 **outbox** 表：每个工具调用先写入 outbox，然后单独的 goroutine/task 执行。通过运行工具调用两次验证幂等性。
3. 模拟 **rainbow 部署**：两个并发运行时版本；将一半新 thread_id 路由到每个；确认旧版本上进行中的线程不被中断。
4. 阅读 LangGraph 的运行时文档（链接如下）。识别运行时的哪些功能在手写 FastAPI + Postgres 版本中复制最耗时。这是采用的理由，还是可以推迟？
5. 阅读 MegaAgent（arXiv:2408.09955）第 3 节。两层协调（组内 + 组间管理聊天）是显式的。草拟你如何将其映射到带两个队列族的消息队列。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Durable execution | "持久化程序状态" | 引擎在每个 super-step 后写状态；崩溃恢复是确定性的。 |
| Super-step | "事务边界" | 检查点之间的工作单元。LangGraph 术语。 |
| thread_id | "智能体运行标识符" | 绑定检查点和恢复逻辑的键。 |
| Idempotency | "安全重试" | 重复副作用产生与一次尝试相同的结果。 |
| Outbox pattern | "解耦副作用" | 将意图写入表；单独的执行器执行并标记完成。 |
| At-least-once delivery | "可能重复" | 消息队列语义；去重键使消费者有效一次。 |
| Rainbow deploy | "重叠版本" | 长期运行工作负载期间多个运行时版本并发。 |
| Async fiber | "协作让出" | 用户态并发；对 I/O 密集负载比线程便宜。 |
| Checkpoint | "状态快照" | super-step 边界处的序列化状态；恢复的关键。 |

## 延伸阅读

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — LangGraph 运行时设计
- [MegaAgent](https://arxiv.org/abs/2408.09955) — 每智能体生产者-消费者队列；数千并发智能体的两层协调
- [Matrix](https://arxiv.org/abs/2511.21686) — 以消息队列为协调基底的去中心化框架
- [Temporal docs](https://docs.temporal.io/) — 持久执行的参考工作流引擎
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 包括 rainbow 部署的生产经验
