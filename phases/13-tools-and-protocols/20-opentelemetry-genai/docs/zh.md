# OpenTelemetry GenAI — 端到端追踪工具调用

> 一个 agent 调用五个工具、三个 MCP 服务器和两个子 agent。你需要一条贯穿所有这些的 trace。OpenTelemetry GenAI 语义约定（v1.37 及以上版本中的稳定属性）是 2026 年的标准，被 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 原生支持。本课命名必需的属性，走通 span 层次结构（agent → LLM → tool），并提供一个可插入任何 OTel exporter 的 stdlib span 发射器。

**Type:** Build
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## 学习目标

- 命名 LLM span 和 tool-execution span 所需的 OTel GenAI 属性。
- 构建覆盖 agent 循环、LLM 调用、工具调用和 MCP 客户端分发的 trace 层次结构。
- 决定哪些内容要捕获（opt-in）vs 脱敏（默认）。
- 向本地 collector（Jaeger、Langfuse）发射 span，无需重写工具代码。

## 问题

2026 年 2 月的一次调试：用户报告"我的 agent 有时 30 秒才响应；有时 3 秒"。没有 trace。日志显示了 LLM 调用，但没有工具分发、没有 MCP 服务器往返、没有子 agent。你只能猜。最终你发现：一个 MCP 服务器偶尔在冷启动时挂起。

没有端到端追踪，你找不到这个问题。OTel GenAI 解决它。

这些约定在 2025-2026 年在 OpenTelemetry semantic-conventions 工作组下确定。它们定义了稳定的属性名称，使 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都解析相同的 span。一次埋点；发送到任何后端。

## 概念

### Span 层次结构

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

整个结构嵌套在一个 trace id 下。Span id 链接父子关系。

### 必需属性

按 2025-2026 semconv：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — 请求的模型字符串（如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 实际服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 用于关联的 provider 响应 id。

工具 span：

- `gen_ai.tool.name` — 工具标识符。
- `gen_ai.tool.call.id` — 特定的调用 id。
- `gen_ai.tool.description` — 工具描述（可选）。

Agent span：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span 类型

- `SpanKind.CLIENT` 用于跨进程边界的调用（LLM provider、MCP 服务器）。
- `SpanKind.INTERNAL` 用于 agent 自身的循环步骤和工具执行。

### Opt-in 内容捕获

默认情况下，span 携带指标和计时 — 不包含 prompt 或补全。大 payload 和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 和特定的内容捕获环境变量来包含内容。在生产中启用前仔细审查。

### Span 上的事件

Token 级别的事件可以作为 span events 添加：

- `gen_ai.content.prompt` — 输入消息。
- `gen_ai.content.completion` — 输出消息。
- `gen_ai.content.tool_call` — 记录的工具调用。

事件在 span 内按时间排序，用于详细回放。

### Exporters

OTel span 导出到：

- **Jaeger / Tempo。** 开源，本地部署。
- **Langfuse。** LLM 可观测性专用；可视化 token 使用。
- **Arize Phoenix。** 评估 + 追踪结合。
- **Datadog。** 商业；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 列式存储；查询友好。

全部使用 OTLP 线格式。你的代码不关心。

### 跨 MCP 的传播

当 MCP 客户端调用服务器时，将 W3C traceparent header 注入请求。Streamable HTTP 支持标准 header。Stdio 原生不携带 HTTP header；规范的 2026 路线图讨论在 JSON-RPC 调用上添加 `_meta.traceparent` 字段。

在那之前：手动在每个请求的 `_meta` 中包含 traceparent。服务器记录 trace id。

### 指标

除了 span，GenAI semconv 还定义了指标：

- `gen_ai.client.token.usage` — histogram。
- `gen_ai.client.operation.duration` — histogram。
- `gen_ai.tool.execution.duration` — histogram。

用于不需要每次调用详情的仪表盘。

### AgentOps 层

AgentOps（2024 年成立）专注于 GenAI 可观测性。它包装流行框架（LangGraph、Pydantic AI、CrewAI）以自动发射 OTel span。如果你的技术栈使用支持的框架则有用；否则使用手动埋点。

## 动手实践

`code/main.py` 为一个调用 LLM、分发两个工具并进行一次 MCP 往返的 agent 向 stdout 发射 OTel 形状的 span（OTLP-JSON-like 格式）。没有真实 exporter — 本课聚焦 span 形状和属性集。将输出粘贴到 OTLP 兼容的查看器中或直接阅读。

关注点：

- Trace id 在所有 span 间共享。
- 父子链接通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；一个场景通过环境变量开启。

## 交付产出

本课产出 `outputs/skill-otel-genai-instrumentation.md`。给定一个 agent 代码库，该技能产出埋点计划：在哪里添加 span、填充哪些属性、以及目标 exporter。

## 练习

1. 运行 `code/main.py`。计算 span 数量并识别哪些是 CLIENT vs INTERNAL。

2. 开启内容捕获（环境变量）并确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出现。注意对 PII 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration` 并将其作为每次调用的 histogram 样本发射。

4. 将 traceparent 从父 agent span 传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器会看到相同的 trace id。

5. 阅读 OTel GenAI semconv 规范。找出 semconv 中列出但本课代码未发射的一个属性。添加它。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| OTel | "OpenTelemetry" | traces、metrics、logs 的开放标准 |
| GenAI semconv | "GenAI 语义约定" | LLM / tool / agent span 的稳定属性名称 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性共享此前缀 |
| Span | "计时操作" | 带开始、结束和属性的工作单元 |
| Trace | "跨 span 祖先关系" | 共享 trace id 的 span 树 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 关于 span 方向的提示 |
| OTLP | "OpenTelemetry Line Protocol" | exporter 的线格式 |
| Opt-in content | "Prompt / completion 捕获" | 默认关闭；环境变量启用 |
| traceparent | "W3C header" | 跨服务传播 trace 上下文 |
| Exporter | "后端特定的发送器" | 将 span 发送到 Jaeger / Datadog 等的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI span、metrics 和 events 的规范约定
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和 tool-execution span 属性列表
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — agent 级别的 `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 托管的权威来源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产集成详解
