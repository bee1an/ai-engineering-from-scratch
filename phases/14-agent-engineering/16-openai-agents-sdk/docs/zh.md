# OpenAI Agents SDK：Handoff、Guardrail、Tracing

> OpenAI Agents SDK 是基于 Responses API 构建的轻量级多智能体框架。五个原语：Agent、Handoff、Guardrail、Session、Tracing。Handoff 是名为 `transfer_to_<agent>` 的工具。Guardrail 在输入或输出上触发。Tracing 默认开启。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## 学习目标

- 列举 OpenAI Agents SDK 的五个原语。
- 解释 handoff：为什么建模为工具、模型看到什么名称形状、上下文如何转移。
- 区分 input guardrail、output guardrail 和 tool guardrail；解释 `run_in_parallel` vs 阻塞模式。
- 用 stdlib 实现一个带 handoff + guardrail + span 风格 tracing 的运行时。

## 问题

无法干净委派的智能体最终把所有东西塞进一个提示词。没有 guardrail 的智能体会泄露 PII、产出违反策略的输出、或无限循环。OpenAI 的 SDK 将使多智能体工作可控的三个原语编纂成型。

## 概念

### 五个原语

1. **Agent。** LLM + instructions + tools + handoffs。
2. **Handoff。** 委派给另一个智能体。对模型表示为名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail。** 对输入（仅第一个智能体）、输出（仅最后一个智能体）或工具调用（每个 function tool）的验证。
4. **Session。** 跨轮次的自动对话历史。
5. **Tracing。** 为 LLM 生成、工具调用、handoff、guardrail 内置 span。

### Handoff 作为工具

模型在其工具列表中看到 `transfer_to_billing_agent`。调用它向运行时发出信号：

1. 复制对话上下文（或通过 `nest_handoff_history` beta 折叠它）。
2. 用目标智能体的 instructions 初始化它。
3. 用目标智能体继续运行。

这是 supervisor 模式（Lesson 13 / Lesson 28）的产品化。

### Guardrail

三种类型：

- **Input guardrail。** 在第一个智能体的输入上运行。在任何 LLM 调用之前拒绝不安全或超范围的请求。
- **Output guardrail。** 在最后一个智能体的输出上运行。捕获 PII 泄露、策略违规、格式错误的响应。
- **Tool guardrail。** 每个 function tool 运行。验证参数、检查权限、审计执行。

模式：

- **并行**（默认）。Guardrail LLM 与主 LLM 并行运行。更低的尾部延迟。如果触发，主 LLM 的工作被丢弃（token 浪费）。
- **阻塞**（`run_in_parallel=False`）。Guardrail LLM 先运行。如果触发，主调用不浪费 token。

触发器抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### Tracing

默认开启。每次 LLM 生成、工具调用、handoff 和 guardrail 都发出一个 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 可关闭。`add_trace_processor(processor)` 将 span 扇出到你自己的后端，与 OpenAI 的并行。

### Session

`Session` 将对话历史存储在后端（SQLite、Redis、自定义）。`Runner.run(agent, input, session=session)` 自动加载和追加。

### 这种模式出错的地方

- **Handoff 漂移。** Agent A 交接给 Agent B，Agent B 又交接回 Agent A。添加跳数计数器。
- **Guardrail 绕过。** Tool guardrail 只在 function tool 上触发；内置工具（文件读取器、web fetch）需要单独的策略。
- **过度 tracing。** span 中的敏感内容。配合 OTel GenAI 内容捕获规则（Lesson 23）——存储在外部，通过 ID 引用。

## Build It

`code/main.py` 用 stdlib 实现了 SDK 的形状：

- `Agent`、`FunctionTool`、`Handoff`（作为带转移语义的 function tool）。
- `Runner` 带 input/output/tool guardrail、handoff 分发和跳数计数器。
- 一个简单的 span 发射器展示 trace 形状。
- 一个分诊智能体根据用户查询交接给 billing 或 support；guardrail 在一个输入上触发。

运行：

```
python3 code/main.py
```

轨迹展示两次成功的 handoff、一次 input guardrail 触发、以及一个镜像真实 SDK 发出的 span 树。

## Use It

- **OpenAI Agents SDK** 用于 OpenAI 优先的产品。
- **Claude Agent SDK**（Lesson 17）用于 Claude 优先的产品。
- **LangGraph**（Lesson 13）当你想要显式状态和持久恢复时。
- **自定义** 当你需要精确控制（语音、多提供商、联邦部署）时。

## Ship It

`outputs/skill-agents-sdk-scaffold.md` 搭建一个 Agents SDK 应用，带分诊智能体、handoff、input/output/tool guardrail、session store 和 trace processor。

## 练习

1. 添加 handoff 跳数计数器：N 次转移后拒绝。追踪行为。
2. 实现 `nest_handoff_history` 作为选项——在转移前将先前消息折叠为一个摘要。
3. 编写一个阻塞 output guardrail。比较会触发它的提示词 vs 通过的提示词的延迟。
4. 将 `add_trace_processor` 接入 JSON 日志器。每个 span 发出什么形状？
5. 阅读 SDK 文档。将你的 stdlib 玩具移植到 `openai-agents-python`。你建模错了什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Agent | "LLM + instructions" | SDK 中的 Agent 类型；拥有 tools 和 handoffs |
| Handoff | "转移" | 模型调用以委派给另一个智能体的工具 |
| Guardrail | "策略检查" | 对输入/输出/工具调用的验证 |
| Tripwire | "Guardrail 触发" | guardrail 拒绝时抛出的异常 |
| Session | "历史存储" | 在运行之间持久化的对话记忆 |
| Tracing | "Span" | 覆盖 LLM + 工具 + handoff + guardrail 的内置可观测性 |
| 阻塞 guardrail | "顺序检查" | Guardrail 先运行；触发时不浪费 token |
| 并行 guardrail | "并发检查" | Guardrail 并行运行；更低延迟，触发时浪费 token |

## 延伸阅读

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 原语、handoff、guardrail、tracing
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 风格的对应物
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 何时需要 handoff
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK span 映射到的标准
