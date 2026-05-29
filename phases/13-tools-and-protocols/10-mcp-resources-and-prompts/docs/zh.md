# MCP Resources 和 Prompts — 工具之外的上下文暴露

> Tools 获得了 MCP 90% 的关注。另外两个服务端原语解决不同的问题。Resources 暴露数据供读取；prompts 暴露可复用模板作为斜杠命令。许多 server 应该用 resources 而非将读取包装在 tools 中，用 prompts 而非在 client prompt 中硬编码工作流。本课命名决策规则并走过 `resources/*` 和 `prompts/*` 消息。

**类型：** 构建
**语言：** Python（标准库，resource + prompt 处理器）
**前置课程：** Phase 13 · 07（MCP server）
**时长：** 约 45 分钟

## 学习目标

- 针对给定领域，决定将能力暴露为 tool、resource 还是 prompt。
- 实现 `resources/list`、`resources/read`、`resources/subscribe` 并处理 `notifications/resources/updated`。
- 实现 `prompts/list` 和 `prompts/get`，带参数模板。
- 识别宿主何时将 prompts 作为斜杠命令呈现 vs 自动注入上下文。

## 问题

一个朴素的笔记应用 MCP server 将所有东西都暴露为 tools：`notes_read`、`notes_list`、`notes_search`。这将每次数据访问都包装在模型驱动的工具调用中。后果：

- 模型必须决定是否为每个可能受益于上下文的查询调用 `notes_read`。
- 只读内容无法被订阅或流式传输到宿主的侧边栏。
- Client UI（Claude Desktop 的资源附件面板、Cursor 的"Include file"选择器）无法呈现数据。

正确的拆分：将数据暴露为 resource，将变更或计算动作暴露为 tool，将可复用的多步工作流暴露为 prompt。每个原语有其 UX 可供性和访问模式。

## 概念

### Tools vs resources vs prompts — 决策规则

| 能力 | 原语 |
|------|------|
| 用户想搜索、过滤或转换数据 | tool |
| 用户想让宿主将此数据作为上下文包含 | resource |
| 用户想要一个可重复运行的模板化工作流 | prompt |

指导原则：如果模型在每个相关查询上调用它都有好处，它是 tool。如果用户将它附加到对话中有好处，它是 resource。如果整个多步工作流是用户想复用的单元，它是 prompt。

### Resources

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接收 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的东西：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义 scheme）
- `memory://session-2026-04-22/recent`（server 特定）

`contents[]` 支持文本和二进制。二进制使用 `blob` 作为 base64 编码字符串加 `mimeType`。

### Resource 订阅

在 capabilities 中声明 `{resources: {subscribe: true}}`。Client 调用 `resources/subscribe {uri}`。当资源变化时 server 发送 `notifications/resources/updated {uri}`。Client 重新读取。

用例：一个笔记 server，其 resources 是磁盘上的文件；文件监视器触发更新通知；Claude Desktop 在文件在宿主外被编辑时重新拉取到上下文中。

### Resource 模板（2025-11-25 新增）

`resourceTemplates` 让你暴露参数化的 URI 模式：`notes://{id}`，`id` 作为补全目标。Client 可以在资源选择器中自动补全 id。

### Prompts

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接收 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

Prompt 是一个填充为消息列表的模板，宿主将其馈入模型。例如，一个 `code_review` prompt 接收 `file_path` 参数并返回三消息序列：一条 system 消息、一条带文件内容的 user 消息、和一条带推理模板的 assistant 开头。

### 宿主和 prompts

Claude Desktop、VS Code 和 Cursor 在聊天 UI 中将 prompts 暴露为斜杠命令。用户输入 `/code_review` 并从表单中选择参数。Server 的 prompt 是"用户快捷方式"和"发送给模型的完整 prompt"之间的契约。

并非每个 client 都支持 prompts — 检查能力协商。一个声明了 prompt 能力的 server 配上一个没有 prompt 支持的 client，简单地看不到斜杠命令。

### "list changed" 通知

Resources 和 prompts 在集合变化时都发出 `notifications/list_changed`。一个刚导入 20 条新笔记的笔记 server 发出 `notifications/resources/list_changed`；client 重新调用 `resources/list` 来获取新增内容。

### 内容类型约定

文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
二进制：`image/png`、`application/pdf`，加 `blob` 字段。
MCP Apps（第 14 课）：`ui://` URI 中的 `text/html;profile=mcp-app`。

### 动态 resources

Resource URI 不必对应静态文件。`notes://recent` 可以在每次读取时返回最新五条笔记。`db://query/users/active` 可以执行参数化查询。Server 可以自由地动态计算内容。

规则：如果 client 可以按 URI 缓存，URI 必须稳定。如果计算是一次性的，URI 应包含时间戳或 nonce，这样 client 缓存不会过期。

### 订阅 vs 轮询

支持订阅的 client 通过 `notifications/resources/updated` 获得 server 推送。不支持订阅的 client 或宿主通过重新读取来轮询。两者都符合规范。Server 的能力声明告诉 client 它支持哪种。

订阅的成本：server 上的每会话状态（谁订阅了什么）。保持订阅集有界；断开连接的 client 应超时。

### Prompts vs system prompts

MCP 中的 prompts 不是 system prompts。宿主的 system prompt（其自身的操作指令）和 MCP prompts（用户调用的 server 提供的模板）并存。一个行为良好的 client 永远不会让 server prompt 覆盖其自身的 system prompt；它将它们分层。

## 动手试试

`code/main.py` 扩展了第 07 课的笔记 server，添加了：

- 每笔记 resources（`notes://note-1` 等），带 `resources/subscribe` 支持。
- 一个 `review_note` prompt，渲染为三消息模板。
- 一个文件监视器模拟，在笔记被修改时发出 `notifications/resources/updated`。
- 一个 `notes://recent` 动态 resource，始终返回最新五条笔记。

运行演示查看完整流程。

## 交付物

本课产出 `outputs/skill-primitive-splitter.md`。给定一个提议的 MCP server，该技能将每个能力分类为 tool / resource / prompt 并附带理由。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，然后触发笔记编辑并验证 `notifications/resources/updated` 事件触发。

2. 添加一个 `resources/list_changed` 发射器：当新笔记被创建时，发送通知使 client 重新发现。

3. 为 GitHub MCP server 设计三个 prompts：`summarize_pr`、`triage_issue`、`release_notes`。每个带参数 schema。Prompt 主体应无需进一步编辑即可运行。

4. 取第 07 课 server 中的一个现有 tool，分类它应该保持为 tool 还是拆分为 resource 加 tool 对。用一句话证明。

5. 阅读规范的 `server/resources` 和 `server/prompts` 部分。找出 `resources/read` 中很少填充但规范支持的那个字段。提示：看 resource content 上的 `_meta`。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Resource | "暴露的数据" | 宿主可以读取的 URI 可寻址内容 |
| Resource URI | "数据指针" | Scheme 前缀的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | "监视变化" | Client 选择加入的 server 推送更新，针对特定 URI |
| `notifications/resources/updated` | "资源变了" | 通知 client 订阅的资源有新内容的信号 |
| Resource template（资源模板） | "参数化 URI" | 带补全提示的 URI 模式，用于宿主选择器 |
| Prompt | "斜杠命令模板" | 带参数槽的命名多消息模板 |
| Prompt arguments（Prompt 参数） | "模板输入" | 宿主在渲染前收集的类型化参数 |
| `prompts/get` | "渲染模板" | Server 返回填充好的消息列表 |
| Content block（内容块） | "类型化块" | `{type: text \| image \| resource \| ui_resource}` |
| Slash-command UX（斜杠命令 UX） | "用户快捷方式" | 宿主将 prompts 作为以 `/` 开头的命令呈现 |

## 延伸阅读

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — resource URI、订阅和模板
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — prompt 模板和斜杠命令集成
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 完整 `resources/*` 消息参考
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 完整 `prompts/*` 消息参考
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — 扩展官方文档的社区指南
