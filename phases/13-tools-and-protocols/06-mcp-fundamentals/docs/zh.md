# MCP 基础 — 原语、生命周期、JSON-RPC 基础

> MCP 之前的每个集成都是一次性的。Model Context Protocol 由 Anthropic 于 2024 年 11 月首次发布，现由 Linux 基金会的 Agentic AI Foundation 管理，它标准化了发现和调用，使任何 client 都能与任何 server 通信。2025-11-25 规范命名了六个原语（三个服务端、三个客户端）、一个三阶段生命周期和一个 JSON-RPC 2.0 线上格式。学会这些，本 phase 的 MCP 章节其余部分就是阅读了。

**类型：** 学习
**语言：** Python（标准库，JSON-RPC 解析器）
**前置课程：** Phase 13 · 01 到 05（工具接口和 function calling）
**时长：** 约 45 分钟

## 学习目标

- 命名所有六个 MCP 原语（服务端的 tools、resources、prompts；客户端的 roots、sampling、elicitation）并各给出一个用例。
- 走过三阶段生命周期（initialize、operation、shutdown），说明每个阶段谁发送什么消息。
- 解析和发出 JSON-RPC 2.0 的 request、response 和 notification 信封。
- 解释 `initialize` 时的能力协商是什么，以及没有它会出什么问题。

## 问题

MCP 之前，每个使用工具的 agent 都有自己的协议。Cursor 有一个 MCP 形状但不兼容的工具系统。Claude Desktop 用了另一个。VS Code 的 Copilot 扩展有第三个。一个团队构建了"Postgres 查询"工具，要写三次相同的工具，每次对应不同宿主的 API。复用需要复制代码。

结果是一次性集成的寒武纪大爆发和生态系统速度的天花板。

MCP 通过标准化线上格式来解决这个问题。一个 MCP server 在每个 MCP client 中都能工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，到 2026 年 4 月有 300+ 个 client。每月 1.1 亿次 SDK 下载。10,000+ 个公开 server。Linux 基金会于 2025 年 12 月在新的 Agentic AI Foundation 下接管管理。

本 phase 使用的规范版本是 **2025-11-25**。它添加了异步 Tasks（SEP-1686）、URL 模式 elicitation（SEP-1036）、带工具的 sampling（SEP-1577）、增量范围同意（SEP-835）和 OAuth 2.1 resource-indicator 语义。Phase 13 · 09 到 16 讲解这些扩展。本课止步于基础。

## 概念

### 三个服务端原语

1. **Tools。** 可调用的动作。与 Phase 13 · 01 相同的四步循环。
2. **Resources。** 暴露的数据。通过 URI 可寻址的只读内容：`file:///path`、`db://query/...`、自定义 scheme。
3. **Prompts。** 可复用的模板。宿主 UI 中的斜杠命令；server 提供模板，client 填充参数。

### 三个客户端原语

4. **Roots。** server 被允许触及的 URI 集合。Client 声明它们；server 遵守。
5. **Sampling。** Server 请求 client 的模型执行一次 completion。使 server 托管的 agent 循环无需服务端 API 密钥。
6. **Elicitation。** Server 在执行中途向 client 的用户请求结构化输入。表单或 URL（SEP-1036）。

MCP 中的每个能力恰好属于这六个之一。Phase 13 · 10 到 14 深入讲解每一个。

### 线上格式：JSON-RPC 2.0

每条消息是一个 JSON 对象，包含以下字段：

- Requests：`{jsonrpc: "2.0", id, method, params}`。
- Responses：`{jsonrpc: "2.0", id, result | error}`。
- Notifications：`{jsonrpc: "2.0", method, params}` — 无 `id`，不期望响应。

基础规范有约 15 个方法，按原语分组。重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（server 到 client）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**阶段 1：initialize。**

Client 发送 `initialize`，带有其 `capabilities` 和 `clientInfo`。Server 响应自己的 `capabilities`、`serverInfo` 和它使用的规范版本。Client 在消化响应后发送 `notifications/initialized`。从此以后，双方可以按协商的能力发送请求。

**阶段 2：operation。**

双向的。Client 调用 `tools/list` 来发现，然后 `tools/call` 来调用。如果 server 声明了该能力，它可以发送 `sampling/createMessage`。当工具集变化时 server 可以发送 `notifications/tools/list_changed`。当用户更改 root 范围时 client 可以发送 `notifications/roots/list_changed`。

**阶段 3：shutdown。**

任一方关闭传输。MCP 中没有结构化的 shutdown 方法；传输（stdio 或 Streamable HTTP，Phase 13 · 09）承载连接结束信号。

### 能力协商

`initialize` 握手中的 `capabilities` 是契约。来自 server 的示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

Server 声明它可以发出 `tools/list_changed` 通知并支持 `resources/subscribe`。Client 通过声明自己的来同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果 client 没有声明 `sampling`，server 不得调用 `sampling/createMessage`。对称地：如果 server 没有声明 `resources.subscribe`，client 不得尝试订阅。

这就是防止生态系统漂移的机制。一个不支持 sampling 的 client 仍然是有效的 MCP client；一个不调用 `sampling` 的 server 仍然是有效的 MCP server。它们只是不一起使用那个功能。

### 结构化内容和错误形状

`tools/call` 返回一个类型化块的 `content` 数组：`text`、`image`、`resource`。Phase 13 · 14 将 MCP Apps（`ui://` 交互式 UI）添加到该列表。

错误使用 JSON-RPC 错误码。规范定义的补充：`-32002` "Resource not found"、`-32603` "Internal error"，加上 MCP 特定的错误数据作为 `error.data`。

### Client capabilities vs tool call 细节

一个常见困惑：`capabilities.tools` 是 client 是否支持 tool-list-changed 通知。Client 是否会调用特定工具是由其模型驱动的运行时选择，不是能力标志。能力标志是规范级契约。模型的选择是正交的。

### 为什么是 JSON-RPC 而不是 REST？

JSON-RPC 2.0（2010）是一个轻量级双向协议。REST 是客户端发起的。MCP 需要服务端发起的消息（sampling、notifications），所以 JSON-RPC 及其对称的 request/response 形状是自然的选择。JSON-RPC 也能干净地组合在 stdio 和 WebSocket/Streamable HTTP 之上，无需重新发明 HTTP 的请求形状。

## 动手试试

`code/main.py` 提供一个最小的 JSON-RPC 2.0 解析器和发射器，然后手动走过 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。没有真实传输；只有消息形状。与延伸阅读中链接的规范对比以验证每个信封。

关注点：

- `initialize` 双向声明能力；响应有 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应 `content` 是 `{type, text}` 块的数组。

## 交付物

本课产出 `outputs/skill-mcp-handshake-tracer.md`。给定一个 pcap 风格的 MCP client-server 交互记录，该技能注释每条消息属于哪个原语、哪个生命周期阶段、以及它依赖哪个能力。

## 练习

1. 运行 `code/main.py`。找到能力协商发生的那一行，描述如果 server 没有声明 `tools.listChanged` 会有什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形状：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时间运行的 `tools/call` 进行中发出它，确认 client 处理器会显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范 — 整个文档约 80 页。找出大多数 server 不需要的那个能力标志。提示：它与资源订阅有关。

4. 在纸上画出一个假设的"定时任务"功能应该属于哪个原语。（提示：server 想让 client 在预定时间调用它。今天六个原语中没有一个适合。）MCP 的 2026 路线图有一个相关的 SEP 草案。

5. 解析 GitHub 上一个开放 MCP server 的一个会话日志。计算 request vs response vs notification 消息数。计算生命周期 vs 操作流量的比例。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MCP | "Model Context Protocol" | 用于模型到工具发现和调用的开放协议 |
| Server primitive（服务端原语） | "Server 暴露什么" | tools（动作）、resources（数据）、prompts（模板） |
| Client primitive（客户端原语） | "Client 让 server 使用什么" | roots（范围）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | "线上格式" | 对称的 request/response/notification 信封 |
| `initialize` 握手 | "能力协商" | 第一对消息；server 和 client 声明各自支持的功能 |
| `tools/list` | "发现" | Client 向 server 请求其当前工具集 |
| `tools/call` | "调用" | Client 请求 server 用参数执行一个工具 |
| `notifications/*_changed` | "变更事件" | Server 告诉 client 其原语列表已变化 |
| Content block（内容块） | "类型化结果" | tool result 中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | "Spec Evolution Proposal" | 命名的草案提案（如 SEP-1686 用于异步 Tasks） |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 权威规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — 六原语心智模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024 年 11 月发布文章
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 一周年回顾和 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686、1036、1577、835 和 1724 的总结
