# 构建 MCP Client — 发现、调用、会话管理

> 大多数 MCP 内容发布 server 教程，对 client 一笔带过。Client 代码才是复杂编排所在：进程启动、能力协商、跨多个 server 的工具列表合并、sampling 回调、重连和命名空间冲突解决。本课构建一个多 server client，将三个不同的 MCP server 提升到一个扁平的工具命名空间中供模型使用。

**类型：** 构建
**语言：** Python（标准库，多 server MCP client）
**前置课程：** Phase 13 · 07（构建 MCP server）
**时长：** 约 75 分钟

## 学习目标

- 将 MCP server 作为子进程启动，完成 `initialize`，并发送 `notifications/initialized`。
- 维护每 server 的会话状态（capabilities、工具列表、最后看到的通知 id）。
- 将多个 server 的工具列表合并到一个命名空间中，处理冲突。
- 将工具调用路由到拥有它的 server 并重组响应。

## 问题

一个真实的 agent 宿主（Claude Desktop、Cursor、Goose、Gemini CLI）同时加载多个 MCP server。用户可能同时运行文件系统 server、Postgres server 和 GitHub server。Client 的工作：

1. 启动每个 server。
2. 独立握手每一个。
3. 对每个调用 `tools/list` 并扁平化结果。
4. 当模型发出 `notes_search` 时，在合并的命名空间中查找并路由到正确的 server。
5. 处理来自任何 server 的通知（`tools/list_changed`）而不阻塞。
6. 传输失败时重连。

手动实现所有这些就是"玩具"和"可用"之间的区别。官方 SDK 包装了这些，但心智模型必须是你自己的。

## 概念

### 子进程启动

`subprocess.Popen` 带 `stdin=PIPE, stdout=PIPE, stderr=PIPE`。设置 `bufsize=1` 并使用文本模式逐行读取。每个 server 是一个进程；client 为每个 server 持有一个 `Popen` 句柄。

### 每 server 的会话状态

每个 server 一个 `Session` 对象，持有：

- `process` — Popen 句柄。
- `capabilities` — server 在 `initialize` 时声明的内容。
- `tools` — 最后一次 `tools/list` 的结果。
- `pending` — 请求 id 到等待响应的 promise/future 的映射。

请求本质上是异步的；发送给 server A 的 `tools/call` 在 server B 正在调用中时不得阻塞。要么使用带队列的线程，要么使用 asyncio。

### 合并的命名空间

当 client 看到聚合的工具列表时，名称可能冲突。两个 server 可能都暴露 `search`。Client 有三个选项：

1. **按 server 名称前缀。** `notes/search`、`files/search`。清晰但不美观。
2. **静默先到先得。** 后来 server 的 `search` 覆盖前一个。有风险；隐藏冲突。
3. **冲突拒绝。** 拒绝加载第二个 server；通知用户。对安全敏感的宿主最安全。

Claude Desktop 使用按 server 前缀。Cursor 使用冲突拒绝并给出清晰错误。VS Code MCP 也采用按 server 前缀。

### 路由

合并后，一个分发表将 `tool_name -> session` 映射。模型按名称发出调用；client 找到 session 并向该 server 的 stdin 写入 `tools/call` 消息，然后等待响应。

### Sampling 回调

如果 server 在 `initialize` 时声明了 `sampling` 能力，它可能发送 `sampling/createMessage` 请求 client 运行其 LLM。Client 必须：

1. 阻塞对该 server 的进一步请求直到 sample 解决，或者如果其实现支持并发则流水线化。
2. 调用其 LLM 提供商。
3. 将响应发回 server。

第 11 课端到端讲解 sampling。本课为完整性做了桩实现。

### 通知处理

`notifications/tools/list_changed` 意味着重新调用 `tools/list`。`notifications/resources/updated` 意味着如果资源正在使用则重新读取。Notifications 不得产生响应 — 不要尝试确认它们。

一个常见的 client bug：在 `tools/call` 上阻塞读取循环，而通知坐在流中。使用后台读取线程将每条消息推入队列；主线程出队并分发。

### 重连

传输可能失败：server 崩溃、OS 杀死进程、stdio 管道断裂。Client 检测 stdout 上的 EOF 并将会话标记为死亡。选项：

- 静默重启 server 并重新握手。对纯只读 server 可以。
- 将失败暴露给用户。对有用户可见会话的有状态 server 可以。

Phase 13 · 09 讲解 Streamable HTTP 的重连语义；stdio 更简单。

### Keepalive 和 session id

Streamable HTTP 使用 `Mcp-Session-Id` 头。Stdio 没有 session id — 进程身份就是会话。Keepalive ping 是可选的；stdio 管道在不活动时不会断开。

## 动手试试

`code/main.py` 将三个模拟 MCP server 作为子进程启动，握手每一个，合并它们的工具列表，并将工具调用路由到正确的那个。"Server"实际上是运行玩具响应器的其他 Python 进程（没有真实 LLM）。运行它可以看到：

- 三次初始化，每个有自己的能力集。
- 三个 `tools/list` 结果合并为一个 7 工具的命名空间。
- 基于工具名称的路由决策。
- 通过命名空间前缀防止的冲突。

关注点：

- `Session` 数据类干净地持有每 server 的状态。
- 后台读取线程在不阻塞主线程的情况下出队 stdout 上的每一行。
- 分发表是一个简单的 `dict[str, Session]`。
- 冲突处理是显式的：当两个 server 声明相同名称时，后来的那个用前缀重命名。

## 交付物

本课产出 `outputs/skill-mcp-client-harness.md`。给定一个声明式的 MCP server 列表（name、command、args），该技能产出一个框架，启动它们、合并工具列表、并提供带冲突解决的路由函数。

## 练习

1. 运行 `code/main.py` 并观察 server 启动日志。用 SIGTERM 杀死一个模拟 server 进程，观察 client 如何检测 EOF 并将该会话标记为死亡。

2. 实现命名空间前缀。当两个 server 暴露 `search` 时，将第二个重命名为 `<server>/search`。更新分发表并验证工具调用正确路由。

3. 添加连接池风格的退避用于 server 重启：连续失败时指数退避，上限 30 秒，三次失败后向用户发出通知。

4. 设计一个支持 100 个并发 MCP server 的 client。什么数据结构替代简单的分发字典？（提示：用于前缀命名空间的 trie，加上每 server 工具数量的指标。）

5. 将 client 移植到官方 MCP Python SDK。SDK 包装了 `stdio_client` 和 `ClientSession`。代码应从约 200 行缩减到约 40 行，同时保留多 server 路由。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MCP client | "Agent 宿主" | 启动 server 并编排工具调用的进程 |
| Session（会话） | "每 server 的状态" | Capabilities、工具列表和待处理请求的簿记 |
| Merged namespace（合并命名空间） | "一个工具列表" | 跨所有活跃 server 的扁平工具名称集 |
| Namespace collision（命名空间冲突） | "两个 server 同名工具" | Client 必须前缀、拒绝或先到先得处理重复 |
| Routing（路由） | "谁接这个调用？" | 从工具名称到拥有它的 server 的分发 |
| Background reader（后台读取器） | "非阻塞 stdout" | 将 server stdout 排入队列的线程或任务 |
| Sampling callback | "LLM 即服务" | Client 对 server 发来的 `sampling/createMessage` 的处理器 |
| `notifications/*_changed` | "原语变更了" | Client 必须重新发现或重新读取的信号 |
| Reconnection policy（重连策略） | "Server 死了怎么办" | 传输失败时的重启语义 |
| Stdio session | "进程 = 会话" | 无 session id；子进程生命周期就是会话 |

## 延伸阅读

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — 权威 client 行为
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) — 使用 Python SDK 的 hello-world client 教程
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) — 参考 `ClientSession` 和 `stdio_client`
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — TS 并行实现
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code 如何在单个编辑器宿主中多路复用多个 MCP server
