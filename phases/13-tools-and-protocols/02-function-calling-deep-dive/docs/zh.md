# Function Calling 深入 — OpenAI、Anthropic、Gemini

> 三大前沿提供商在 2024 年收敛到了相同的工具调用循环，然后在其他所有方面分道扬镳。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 id 关联。本课将三者并排对比，让在一个提供商上发布的代码在移植时不会崩溃。

**类型：** 构建
**语言：** Python（标准库，schema 转换器）
**前置课程：** Phase 13 · 01（工具接口）
**时长：** 约 75 分钟

## 学习目标

- 说出 OpenAI、Anthropic 和 Gemini function-calling 载荷之间的三个形状差异（声明、调用、结果）。
- 将一个工具声明翻译为三种提供商格式，并预测 strict mode 约束在哪里会有差异。
- 在每个提供商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解每个提供商的硬限制（工具数量、schema 深度、参数长度）以及违反限制时各自发出的错误签名。

## 问题

Function-calling 请求的形状因提供商而异。来自 2026 年生产栈的三个具体例子：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是一个你必须解析的 JSON 字符串。Strict mode（`strict: true`）通过受约束解码强制 schema 合规。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 形式返回。`input` 已经是解析好的对象（不是字符串）。你用一条新的 `user` 消息回复，其中包含 `{type: "tool_result", tool_use_id, content}` 块。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 形式到达，其中 `id` 在 Gemini 3 及以上版本中是唯一的，用于并行调用关联。你用 `{functionResponse: {name, id, response}}` 回复。

相同的循环。不同的字段名、不同的嵌套、不同的字符串与对象约定、不同的关联机制。一个团队在 OpenAI 上写了一个天气 agent，移植到 Anthropic 要花两天，再到 Gemini 又要一天，仅仅是管道工作。

本课构建一个转换器，将三种格式统一为一个规范的工具声明，并在边缘路由。Phase 13 · 17 将同一模式泛化为 LLM 网关。

## 概念

### 共同结构

每个提供商需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入 schema。
2. **工具选择。** 强制特定工具、禁止工具或让模型决定。
3. **调用发出。** 命名工具和参数的结构化输出。
4. **调用 id。** 将响应关联到正确的调用（对并行调用很重要）。
5. **结果注入。** 将结果绑定回调用的消息或块。

### 形状差异，逐字段对比

| 方面 | OpenAI | Anthropic | Gemini |
|------|--------|-----------|--------|
| 声明信封 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | assistant 消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | 字符串化的 JSON | 已解析的对象 | 已解析的对象 |
| Id 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | 角色 `tool`，`tool_call_id` | `user` 中的 `tool_result`，`tool_use_id` | 带匹配 `id` 的 `functionResponse` |
| 强制某工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格 schema | `strict: true` | schema 即 schema（始终强制） | 请求级别的 `responseSchema` |

### 你实际会遇到的限制

- **OpenAI。** 每个请求 128 个工具。Schema 深度 5。参数字符串 <= 8192 字节。Strict mode 要求无 `$ref`、无重叠的 `oneOf`/`anyOf`/`allOf`、每个属性都列在 `required` 中。
- **Anthropic。** 每个请求 64 个工具。Schema 深度实际上无限但实用限制为 10。无 strict-mode 标志；schema 是契约，模型倾向于遵守。
- **Gemini。** 每个请求 64 个函数。Schema 类型是 OpenAPI 3.0 子集（与 JSON Schema 2020-12 略有偏差）。Gemini 3 起并行调用有唯一 id。

### `tool_choice` 行为

三种所有人都支持的模式，命名不同。

- **Auto。** 模型选择工具或文本。默认。
- **Required / Any。** 模型必须调用至少一个工具。
- **None。** 模型不得调用工具。

加上每个提供商独有的一种模式：

- **OpenAI。** 按名称强制特定工具。
- **Anthropic。** 按名称强制特定工具；`disable_parallel_tool_use` 标志区分单次与多次。
- **Gemini。** `mode: "VALIDATED"` 无论模型意图如何都将每个响应路由通过 schema 验证器。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一条 assistant 消息中发出多个调用。你运行所有调用并用一条批量 tool 角色消息回复，每个 `tool_call_id` 一个条目。Anthropic 历史上是单次调用；`disable_parallel_tool_use: false`（Claude 3.5 起默认）启用多次。Gemini 2 允许并行调用但没有给出稳定 id；Gemini 3 添加了 UUID，使乱序响应能干净地关联。

### Streaming

三者都支持流式工具调用。线上格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的 delta 块增量到达。你累积直到 `finish_reason: "tool_calls"`。
- **Anthropic。** Block-start / block-delta / block-stop 事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）发出带有 `functionCallId` 的块，使多个并行调用可以交错。

Phase 13 · 03 深入并行 + streaming 重组。本课聚焦于声明和单次调用形状。

### 错误和修复

无效参数错误看起来也不同。

- **OpenAI（非 strict）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入错误消息并重新调用。
- **OpenAI（strict）。** 验证在解码期间发生；无效 JSON 不可能出现，但 `refusal` 可能出现。
- **Anthropic。** `input` 可能包含意外字段；schema 是建议性的。在服务端验证。
- **Gemini。** OpenAPI 3.0 怪癖：对象字段上的 `enum` 被静默忽略；自己验证。

### 转换器模式

你代码中的规范工具声明看起来像这样（你选择形状）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其翻译为三种提供商形状。`code/main.py` 中的框架正是这样做的，然后通过每个提供商的响应形状往返一个假工具调用。不需要网络 — 本课教的是形状，不是 HTTP。

生产团队将这个转换器包装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。Phase 13 · 17 发布一个网关，在三者中任何一个前面暴露 OpenAI 形状的 API。

## 动手试试

`code/main.py` 定义了一个规范的 `Tool` 数据类和三个转换器，分别发出 OpenAI、Anthropic 和 Gemini 的声明 JSON。然后它将每种形状的手工制作的提供商响应解析为相同的规范调用对象，展示语义在表面之下是相同的。运行它并并排对比三个声明。

关注点：

- 三个声明块仅在信封和字段名上不同。
- 三个响应块在调用所在位置上不同（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数从所有三种响应形状中提取 `{id, name, args}`。

## 交付物

本课产出 `outputs/skill-provider-portability-audit.md`。给定一个针对某个提供商的 function-calling 集成，该技能产出可移植性审计：它依赖哪些提供商限制、哪些字段需要重命名、以及移植到其他每个提供商时什么会崩溃。

## 练习

1. 运行 `code/main.py` 并验证三个提供商声明 JSON 都序列化了相同的底层 `Tool` 对象。修改规范工具以添加一个 enum 参数，确认只有 Gemini 转换器需要处理 OpenAPI 怪癖。

2. 为每个提供商添加一个 `ListToolsResponse` 解析器，提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 原生没有这个；注意这种不对称性。

3. 实现 `tool_choice` 转换：将规范的 `ToolChoice(mode="force", tool_name="x")` 映射到所有三种提供商形状。然后映射 `mode="any"` 和 `mode="none"`。对照本课的差异表。

4. 选择三个提供商之一，从头到尾阅读其 function-calling 指南。找出其 schema 规范中一个其他两个不支持的字段。候选：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明 schema 的工具调用。通过每个提供商的验证器运行它（第 01 课的标准库验证器可以作为代理），记录哪些错误触发。记录你在生产中会使用哪个提供商来获得严格性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Function calling | "Tool use" | 提供商级别的 API，用于结构化工具调用发出 |
| Tool declaration（工具声明） | "Tool spec" | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | "强制 / 禁止" | Auto / required / none / 指定名称模式 |
| Strict mode | "Schema 强制" | OpenAI 标志，约束解码以匹配 schema |
| `tool_use` 块 | "Anthropic 的调用形状" | 带有 id、name、input 的内联内容块 |
| `functionCall` part | "Gemini 的调用形状" | 包含 name、args 和 id 的 `parts[]` 条目 |
| Arguments-as-string | "字符串化的 JSON" | OpenAI 将 args 作为 JSON 字符串返回，而非对象 |
| Parallel tool calls（并行工具调用） | "一个回合中的扇出" | 一条 assistant 消息中的多个工具调用 |
| Refusal（拒绝） | "模型拒绝" | Strict-mode 专有的拒绝块，替代调用 |
| OpenAPI 3.0 子集 | "Gemini schema 怪癖" | Gemini 使用与 JSON Schema 类似但有细微差异的方言 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 权威参考，包括 strict mode 和并行调用
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 和 `tool_result` 块语义
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — 并行调用、唯一 id 和 OpenAPI 子集
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企业级表面
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict-mode schema 强制细节
