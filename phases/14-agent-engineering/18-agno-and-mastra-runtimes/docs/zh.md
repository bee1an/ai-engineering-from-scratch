# Agno 和 Mastra：生产运行时

> Agno（Python）和 Mastra（TypeScript）是 2026 年的生产运行时搭档。Agno 目标是微秒级智能体实例化和无状态 FastAPI 后端。Mastra 在 Vercel AI SDK 基底上提供智能体、工具、工作流、统一模型路由和复合存储。

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## 学习目标

- 识别 Agno 的性能目标及其何时重要。
- 列举 Mastra 的三个原语——Agents、Tools、Workflows——以及支持的服务器适配器。
- 解释为什么无状态 session 作用域的 FastAPI 后端是推荐的 Agno 生产路径。
- 为给定技术栈选择 Agno vs Mastra（Python 优先 vs TypeScript 优先）。

## 问题

LangGraph、AutoGen、CrewAI 都是重框架。想要"只要 agent loop，快，在我的运行时里"的团队会选择 Agno（Python）或 Mastra（TypeScript）。两者都用一些框架拥有的原语换取原始速度和与周围技术栈更紧密的契合。

## 概念

### Agno

- Python 运行时，前身是 Phi-data。
- "没有图、链或复杂模式——只有纯 Python。"
- 文档中的性能目标：~2μs 智能体实例化、~3.75 KiB 每智能体内存、~23 个模型提供商。
- 生产路径：无状态 session 作用域的 FastAPI 后端。每个请求启动一个新智能体；session 状态存在数据库中。
- 原生多模态（文本、图像、音频、视频、文件）和 agentic RAG。

速度目标在你每秒有数千个短生命周期智能体时重要（聊天扇入、评估管道）。当一个智能体运行 10 分钟时就不那么重要了。

### Mastra

- TypeScript，基于 Vercel AI SDK 构建。
- 三个原语：**Agents**、**Tools**（Zod 类型化）、**Workflows**。
- Unified Model Router — 跨 94 个提供商的 3,300+ 模型（2026 年 3 月）。
- 复合存储：记忆、工作流、可观测性到不同后端；大规模可观测性推荐 ClickHouse。
- Apache 2.0，`ee/` 目录在 source-available 企业许可下。
- Express、Hono、Fastify、Koa 的服务器适配器；一等 Next.js 和 Astro 集成。
- 提供 Mastra Studio（localhost:4111）用于调试。
- 22k+ GitHub stars，300k+ 每周 npm 下载量，1.0 版（2026 年 1 月）。

### 定位

两者都不试图成为 LangGraph。它们在以下方面竞争：

- **语言契合。** Agno 面向 Python 优先团队；Mastra 面向 TypeScript 优先。
- **运行时人体工学。** Agno = 近零开销；Mastra = 与 Vercel 生态系统集成。
- **可观测性。** 两者都与 Langfuse/Phoenix/Opik（Lesson 24）集成，但 Mastra Studio 是第一方的。

### 何时选择哪个

- **Agno** — Python 后端、大量短生命周期智能体、强性能要求、FastAPI 技术栈。
- **Mastra** — TypeScript 后端、Next.js / Vercel 部署、统一多提供商模型路由、Zod 类型化工具。
- **LangGraph**（Lesson 13）— 当持久状态和显式图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK** — 当你想要提供商的产品化形态时（Lessons 16–17）。

### 这种模式出错的地方

- **为性能而性能。** 因为"2μs"听起来好就选 Agno，而工作负载是每请求一个慢智能体调用。开销不是瓶颈。
- **生态系统锁定。** Mastra 的 Vercel 风格集成在 Vercel 上是加分，在其他地方是减分。
- **企业许可混淆。** Mastra 的 `ee/` 目录是 source-available，不是 Apache 2.0。如果你计划 fork，请阅读许可证。

## Build It

本课主要是比较性的——没有单一代码产物能公正对待两个框架。见 `code/main.py` 的并排玩具：一个最小的"运行智能体、流式输出、持久化 session"流程实现两次（一次 Agno 形状，一次 Mastra 形状）。

运行：

```
python3 code/main.py
```

两个结构不同但功能等价的轨迹。

## Use It

- **Agno** — 需要速度和 FastAPI 形状的 Python 后端。
- **Mastra** — 有多提供商和工作流原语的 TypeScript 后端。
- 两者都提供第一方可观测性钩子。两者都与 Langfuse 集成。

## Ship It

`outputs/skill-runtime-picker.md` 基于技术栈、延迟预算和运维形状选择 Agno、Mastra、LangGraph 或提供商 SDK。

## 练习

1. 阅读 Agno 的文档。将 stdlib ReAct 循环（Lesson 01）移植到 Agno。什么消失了？什么保留了？
2. 阅读 Mastra 的文档。将同一循环移植到 Mastra。工具类型化有什么变化（Zod vs 无）？
3. 基准测试：测量你技术栈上的智能体实例化延迟。Agno 的 2μs 对你的工作负载重要吗？
4. 设计一次迁移：如果你一直在 Python 中运行 CrewAI，迁移到 Agno 会破坏什么？
5. 阅读 Mastra 的 `ee/` 许可条款。什么限制会影响开源 fork？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Agno | "快速 Python 智能体" | 无状态 session 作用域的智能体运行时 |
| Mastra | "Vercel AI SDK 上的 TypeScript 智能体" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "多提供商访问" | 跨 94 个提供商 3,300+ 模型的单一客户端 |
| 复合存储 | "多后端" | 记忆/工作流/可观测性各自到不同存储 |
| Mastra Studio | "本地调试器" | localhost:4111 用于内省智能体的 UI |
| Source-available | "非 OSS" | 许可证允许阅读源码但限制商业使用 |

## 延伸阅读

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — 性能目标、FastAPI 集成
- [Mastra docs](https://mastra.ai/docs) — 原语、服务器适配器、Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 集成引用的可观测性比较
