# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了 agent 遥测的标准 schema。Span 名称、属性和内容捕获规则在各厂商间趋于统一，使得 agent trace 在 Datadog、Grafana、Jaeger 和 Honeycomb 中含义一致。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 24 (Observability Platforms)
**Time:** ~60 minutes

## 学习目标

- 列举 GenAI span 类别：model/client、agent、tool。
- 区分 `invoke_agent` CLIENT 与 INTERNAL span 及各自适用场景。
- 列出顶层 GenAI 属性：provider name、request model、data-source ID。
- 解释内容捕获契约：opt-in、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用建议。

## 问题

每个厂商都发明自己的 span 名称。运维团队最终要为每个框架构建独立的 dashboard。OpenTelemetry 的 GenAI SIG 通过定义一个全生态系统统一的标准来解决这个问题。

## 概念

### Span 类别

1. **Model / client spans。** 覆盖原始 LLM 调用。由 provider SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2. **Agent spans。** `create_agent`（agent 构建时）和 `invoke_agent`（agent 运行时）。
3. **Tool spans。** 每次工具调用一个；通过父子关系连接到 agent span。

### Agent span 命名

- Span 名称：如果有名称则为 `invoke_agent {gen_ai.agent.name}`；否则回退到 `invoke_agent`。
- Span kind：
  - **CLIENT** — 用于远程 agent 服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内 agent 框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型 ID。
- `gen_ai.response.model` — 解析后的模型（可能因路由与请求不同）。
- `gen_ai.agent.name` — agent 标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 用于 RAG：查询了哪个语料库或存储。

针对 Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 存在技术特定约定。

### 内容捕获

默认规则：instrumentation 默认不应捕获输入/输出。捕获通过以下方式 opt-in：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：将内容存储在外部（S3、你的日志存储），在 span 上记录引用（指针 ID，而非原文）。这是 Lesson 27 内容投毒防御接入可观测性的方式。

### 稳定性

截至 2026 年 3 月，大多数约定仍处于实验阶段。通过以下方式 opt-in 稳定预览：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 原生将 GenAI 属性映射到其 LLM Observability schema。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 这个模式容易出错的地方

- **在 span 中捕获完整 prompt。** PII、密钥、客户数据出现在运维可读的 trace 中。应存储在外部。
- **没有 `gen_ai.provider.name`。** 多 provider dashboard 在缺少归属时会崩溃。
- **Span 没有父链接。** 孤立的 tool span。始终传播 context。
- **没有设置 stability opt-in。** 后端升级时你的属性可能被重命名。

## Build It

`code/main.py` 实现了一个符合 GenAI 约定的 stdlib span 发射器：

- 带 GenAI 属性 schema 的 `Span`。
- 带 `start_span`、嵌套 context 的 `Tracer`。
- 一个脚本化 agent 运行，发出：`create_agent`、`invoke_agent`（INTERNAL）、per-tool span、LLM 调用的 `chat` span。
- 一个内容捕获模式，将 prompt 存储在外部并在 span 上记录 ID。

运行：

```
python3 code/main.py
```

输出：一棵带有所有必需 GenAI 属性的 span 树，以及一个展示 opt-in 内容引用的"外部存储"。

## Use It

- **Datadog LLM Observability**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（Lesson 24）— 自动 instrument 生态系统。
- **Jaeger / Honeycomb / Grafana Tempo** — 原始 OTel trace；基于 GenAI 属性构建 dashboard。
- **Self-hosted** — 运行带 GenAI processor 的 OTel Collector。

## Ship It

`outputs/skill-otel-genai.md` 将 OTel GenAI span 接入现有 agent，配置内容捕获默认值和外部引用存储。

## 练习

1. 用 `invoke_agent`（INTERNAL）+ per-tool span instrument 你的 Lesson 01 ReAct 循环。发送到 Jaeger 实例。
2. 以"仅引用"模式添加内容捕获：prompt 存入 SQLite，span 属性只携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。将其接入你的 Lesson 09 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，验证你的属性不会被 collector 重命名。
5. 构建一个 dashboard："哪些工具错误与哪些模型相关"，仅基于 GenAI 属性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| GenAI SIG | "OpenTelemetry GenAI group" | 定义 schema 的 OTel 工作组 |
| invoke_agent | "Agent span" | 代表 agent 运行的 span 名称 |
| CLIENT span | "Remote call" | 调用远程 agent 服务的 span |
| INTERNAL span | "In-process" | 进程内 agent 运行的 span |
| gen_ai.provider.name | "Provider" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG source" | 检索命中了哪个语料库/存储 |
| Content capture | "Prompt logging" | Opt-in 捕获消息；生产环境存储在外部 |
| Stability opt-in | "Preview mode" | 固定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the spec
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — GenAI spans by default
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — OTel spans built in
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace context propagation
