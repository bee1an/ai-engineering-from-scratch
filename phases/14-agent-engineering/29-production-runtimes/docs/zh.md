# 生产运行时：Queue、Event、Cron

> 生产环境的 agent 运行在六种 runtime 形状上：request-response、streaming、durable execution、queue-based background、event-driven、scheduled。先选形状，再选框架。可观测性在每种形状上都是承重的。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Voice)
**Time:** ~60 minutes

## 学习目标

- 说出六种生产 runtime 形状，并将每种匹配到框架/产品模式。
- 解释为什么 durable execution（LangGraph）对长周期任务很重要。
- 描述 event-driven runtime 以及 Claude Managed Agents 何时适用。
- 解释"可观测性是承重的"这一论断对多步骤 agent 的意义。

## 问题

生产环境的 agent 会以 Jupyter notebook 无法暴露的方式失败：第 37 步的网络超时、用户在语音通话中途挂断、cron job 在机器重启时死掉、后台 worker 内存耗尽。Runtime 形状决定了哪些故障是可恢复的。

## 概念

### Request-response

- 同步 HTTP。用户等待完成。
- 只适用于短任务（<30s）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性：标准 HTTP access log + OTel span。

### Streaming

- SSE 或 WebSocket 用于渐进式输出。
- LiveKit 将其扩展到 WebRTC 用于语音/视频（Lesson 22）。
- 技术栈：任何支持 streaming 的框架 + 处理 SSE/WS 的前端。
- 可观测性：per-chunk timing、first-token latency、tail latency。

### Durable execution

- 每一步之后 checkpoint state；故障时自动恢复。
- AutoGen v0.4 actor 模型将故障隔离到单个 agent（Lesson 14）。
- LangGraph 的核心差异化能力（Lesson 13）。
- 当步数未知且恢复成本高时必不可少。

### Queue-based / background

- Job 进入队列，worker 拾取，结果通过 webhook 或 pub/sub 回流。
- 对长周期 agent 必不可少（每个任务数十到数百步，参见 Anthropic 的 computer use 公告）。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自定义。
- 可观测性：queue depth、per-job latency distribution、DLQ size。

### Event-driven

- Agent 订阅触发器：新邮件、PR 打开、cron 触发。
- Claude Managed Agents 开箱即用地覆盖这个场景（Lesson 17）。
- CrewAI Flows（Lesson 15）结构化事件驱动的确定性工作流。
- 可观测性：trigger source、event-to-start latency、agent latency。

### Scheduled

- Cron 形状的 agent，周期性运行。
- 与 durable execution 结合，这样失败的夜间运行可以在下一个 tick 恢复。
- 技术栈：Kubernetes CronJob + durable 框架；托管方案（Render cron、Vercel cron）。

### 2026 部署模式

- **CrewAI Flows** 用于事件驱动生产。
- **Agno** 无状态 FastAPI 用于 Python 微服务。
- **Mastra** server adapter（Express、Hono、Fastify、Koa）用于嵌入。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（Lesson 22）。
- **Claude Managed Agents** 用于托管的长时间运行异步。

### 可观测性是承重的

没有 OpenTelemetry GenAI span（Lesson 23）加上 Langfuse/Phoenix/Opik 后端（Lesson 24），你无法调试一个在第 40 步失败的多步骤 agent。这在生产环境中不是可选的。它是"我们快速调试"和"我们从头重放并加更多日志"之间的区别。

### 生产 runtime 在哪里失败

- **选错形状。** 为 5 分钟的任务选 request-response。用户挂断；worker 堆积；重试叠加。
- **没有 DLQ。** Queue worker 没有 dead-letter。失败的 job 消失了。
- **不透明的后台工作。** 后台 agent 运行没有 trace 导出。故障在用户报告之前不可见。
- **跳过 durable state。** 任何超过 30 秒且你承受不起重启的运行都需要 durable execution。

## Build It

`code/main.py` 是一个 stdlib 多形状演示：

- Request-response endpoint（普通函数）。
- Streaming handler（generator）。
- Queue-based worker with DLQ。
- Event trigger registry。
- Cron-shaped scheduler。

运行：

```bash
python3 code/main.py
```

输出：五个 trace 展示每种形状在相同任务上的行为。相同的 agent 逻辑，不同的外壳。Durable execution（第六种形状）有意在 Lesson 13 中用 LangGraph checkpointing 覆盖。

## Use It

- **Request-response** 用于聊天式 UX。
- **Streaming** 用于渐进式响应。
- **Durable** 用于长周期任务。
- **Queue** 用于批处理 / 异步 / 长时间运行。
- **Event** 用于 agent 响应性。
- **Cron** 用于日常维护（memory consolidation、evals、成本报告）。

## Ship It

`outputs/skill-runtime-shape.md` 为任务选择 runtime 形状并接入可观测性需求。

## 练习

1. 把你的 Lesson 01 ReAct 循环移植到你技术栈中的全部六种形状。哪种形状适合哪个产品界面？
2. 给 queue-based 演示加一个 DLQ。模拟 10% 的 job 失败；暴露 DLQ size。
3. 写一个 cron 触发的 eval agent，每晚对当天 top 20 trace 运行。
4. 实现带背压的 streaming：如果客户端慢，暂停 agent。这与 turn budget 如何交互？
5. 阅读 Claude Managed Agents 文档。什么时候你会把自托管的长周期 agent 迁移到托管方案？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Request-response | "同步" | 用户等待；仅限短任务 |
| Streaming | "SSE / WS" | 渐进式输出；更好的 UX；per-chunk 可观测延迟 |
| Durable execution | "从故障恢复" | Checkpoint state；从最后一步重启 |
| Queue-based | "后台 job" | Producer / worker pool / DLQ |
| Event-driven | "基于触发器" | Agent 响应外部事件 |
| DLQ | "Dead-letter queue" | 失败 job 的停放区 |
| Claude Managed Agents | "托管 harness" | Anthropic 托管的长时间运行异步，带 caching + compaction |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — durable execution details
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — hosted long-running async
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "dozens-to-hundreds of steps per task"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model fault isolation
