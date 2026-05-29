# MCP 传输 — stdio vs Streamable HTTP vs SSE 迁移

> stdio 在本地工作，其他地方不行。Streamable HTTP（2025-03-26）是远程标准。旧的 HTTP+SSE 传输已弃用，将在 2026 年中移除。选错传输要付出迁移代价；选对了就能获得一个带会话连续性和 DNS 重绑定防护的可远程托管 MCP server。

**类型：** 学习
**语言：** Python（标准库，Streamable HTTP 端点骨架）
**前置课程：** Phase 13 · 07、08（MCP server 和 client）
**时长：** 约 45 分钟

## 学习目标

- 根据部署形态（本地 vs 远程、单进程 vs 集群）在 stdio 和 Streamable HTTP 之间选择。
- 实现 Streamable HTTP 单端点模式：POST 用于请求，GET 用于会话流。
- 强制 `Origin` 验证和 session-id 语义以防御 DNS 重绑定。
- 在 2026 年中移除截止日期前将遗留 HTTP+SSE server 迁移到 Streamable HTTP。

## 问题

第一个 MCP 远程传输（2024-11）是 HTTP+SSE：两个端点，一个用于 client 的 POST，一个 Server-Sent-Events 通道用于 server 到 client 的流。它能工作。但也很笨拙：每个会话两个端点、某些 CDN 前面的缓存损坏、以及对长连接 SSE 的硬依赖（某些 WAF 会激进地终止它）。

2025-03-26 规范用 Streamable HTTP 替换了它：一个端点，POST 用于 client 请求，GET 用于建立会话流，两者共享 `Mcp-Session-Id` 头。此后构建或迁移的每个 server 都使用 Streamable HTTP。旧的 SSE 模式正在被弃用 — Atlassian Rovo 于 2026 年 6 月 30 日移除；Keboola 于 2026 年 4 月 1 日；大多数剩余企业 server 在 2026 年底前。

而 stdio 对本地 server 仍然重要。Claude Desktop、VS Code 和每个 IDE 形态的 client 通过 stdio 启动 server。正确的心智模型：stdio 用于"这台机器"，Streamable HTTP 用于"通过网络"。没有交叉。

## 概念

### stdio

- 子进程传输。Client 启动 server，通过 stdin/stdout 通信。
- 每行一个 JSON 对象。换行符分隔。
- 无 session id；进程身份就是会话。
- 不需要认证（子进程继承父进程的信任边界）。
- 永远不要用于远程 server — 你需要 SSH 或 socat 来隧道，到那时就该用 Streamable HTTP 了。

### Streamable HTTP

单端点 `/mcp`（或任何路径）。支持三种 HTTP 方法：

- **POST /mcp。** Client 发送 JSON-RPC 消息。Server 回复单个 JSON 响应，或一个包含一个或多个响应的 SSE 流（对批量响应和与该请求相关的通知有用）。
- **GET /mcp。** Client 打开长连接 SSE 通道。Server 用它发送 server 到 client 的请求（sampling、notifications、elicitation）。
- **DELETE /mcp。** Client 显式终止会话。

会话由 server 在第一个响应上设置的 `Mcp-Session-Id` 头标识，client 在每个后续请求上回显它。Session id 必须是加密随机的（128+ 位）；client 选择的 id 出于安全被拒绝。

### 单端点 vs 双端点

旧规范的双端点模式在 2026 年仍可调用 — 规范声明它"遗留兼容"。但所有新 server 应该是单端点的。官方 SDK 发出单端点；只在与未迁移的远程通信时使用遗留模式。

### `Origin` 验证和 DNS 重绑定

浏览器（目前）不是 MCP client，但攻击者可以制作一个网页说服浏览器 POST 到 `localhost:1234/mcp` — 用户的本地 MCP server 监听的地方。如果 server 不检查 `Origin`，浏览器的同源策略不会救它，因为 `Origin: http://evil.com` 是有效的跨域。

2025-11-25 规范要求 server 拒绝 `Origin` 不在允许列表上的请求。允许列表通常包含 MCP client 宿主（`https://claude.ai`、`vscode-webview://*`）和本地 UI 的 localhost 变体。

### Session id 生命周期

1. Client 发送第一个请求，不带 `Mcp-Session-Id`。
2. Server 分配一个随机 id，在响应头上设置 `Mcp-Session-Id`。
3. Client 在所有后续请求和 `GET /mcp` 的流上回显该头。
4. Session 可以被 server 撤销；client 在后续请求上看到 404，必须重新初始化。
5. Client 可以显式 DELETE 会话以干净关闭。

### Keepalive 和重连

SSE 连接会断开。Client 通过用相同的 `Mcp-Session-Id` 重新 GET 来重建。Server 必须将中断期间错过的事件排队（在合理窗口内）并通过 client 回显的 `last-event-id` 头重放。

Phase 13 · 13 讲解 Tasks，它让长时间运行的工作即使在完整会话重连后也能存活。

### 向后兼容探测

一个想同时支持新旧 server 的 client：

1. POST 到 `/mcp`。
2. 如果响应是 `200 OK` 带 JSON 或 SSE，这是 Streamable HTTP。
3. 如果响应是 `200 OK` 带 `Content-Type: text/event-stream` 且有指向第二端点的 `Location` 头，这是遗留 HTTP+SSE；跟随 `Location`。

### Cloudflare、ngrok 和托管

2026 年的生产远程 MCP server 运行在 Cloudflare Workers（带其 MCP Agents SDK）、Vercel Functions 或容器化的 Node/Python 上。关键：你的托管必须支持长连接 HTTP 用于 SSE GET。Vercel 免费层上限 10 秒，不适合。Cloudflare Workers 支持无限流。

### 网关组合

当你用网关前置多个 MCP server（Phase 13 · 17）时，网关是一个单 Streamable HTTP 端点，重写 session id 并多路复用上游。工具在网关层合并；client 看到一个单一逻辑 server。

### 传输失败模式

- **stdio SIGPIPE。** 子进程在写入中途死亡引发 SIGPIPE；server 应干净退出。Client 应检测 EOF 并标记会话为死亡。
- **HTTP 502 / 504。** Cloudflare、nginx 和其他代理在上游失败时发出这些。Streamable HTTP client 应在短暂退避后重试一次。
- **SSE 连接断开。** TCP RST、代理超时或 client 网络变化关闭流。Client 用 `Mcp-Session-Id` 和可选的 `last-event-id` 重连以恢复。
- **Session 撤销。** Server 使 session id 无效；client 在下一个请求上看到 404。Client 必须重新握手。
- **时钟偏差。** Client 上的资源 TTL 计算与 server 偏离。Client 应将 server 时间戳视为权威。

### 何时绕过 Streamable HTTP

一些企业在自己的网络内将 MCP server 部署在 gRPC 或消息队列传输后面。这是非标准的 — MCP 的规范没有正式定义这些。网关可以向 MCP client 暴露 Streamable HTTP 表面，同时内部使用 gRPC。保持外部表面符合规范；网关负责翻译。

## 动手试试

`code/main.py` 使用 `http.server`（标准库）实现一个最小的 Streamable HTTP 端点。它处理 `/mcp` 上的 POST、GET 和 DELETE，在第一个响应上设置 `Mcp-Session-Id`，验证 `Origin`，并拒绝来自非允许列表来源的请求。处理器复用第 07 课笔记 server 的分发逻辑。

关注点：

- POST 处理器读取 JSON-RPC 主体，分发，并写入 JSON 响应（单响应变体；SSE 变体结构类似）。
- `Origin` 检查拒绝默认的 `http://evil.example` 探测但接受 `http://localhost`。
- Session id 是随机 128 位十六进制字符串；server 在内存中保持每会话状态。

## 交付物

本课产出 `outputs/skill-mcp-transport-migrator.md`。给定一个 HTTP+SSE（遗留）MCP server，该技能产出一个迁移计划到 Streamable HTTP，带有 session-id 连续性、Origin 检查和向后兼容探测支持。

## 练习

1. 运行 `code/main.py`。从 `curl` POST 一个 `initialize` 并观察 `Mcp-Session-Id` 响应头。POST 第二个请求回显该头并验证会话连续性。

2. 添加一个 GET 处理器打开 SSE 流。每五秒发送一个 `notifications/progress` 事件。用相同 session id 重新 GET 来重连，确认 server 接受它。

3. 实现 `last-event-id` 重放逻辑。重连时，重放自该 id 以来生成的所有事件。

4. 扩展 `Origin` 验证以支持通配符模式（`https://*.example.com`），确认它接受 `https://app.example.com` 但拒绝 `https://evil.example.com.attacker.net`。

5. 从官方注册表中取一个遗留 HTTP+SSE server（有好几个），草拟迁移方案：端点处理、session id 生成和头语义有什么变化。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| stdio transport | "本地子进程" | 通过 stdin/stdout 的 JSON-RPC，换行符分隔 |
| Streamable HTTP | "远程传输" | 单端点 POST + GET + 可选 SSE，2025-03-26 规范 |
| HTTP+SSE | "遗留" | 正在 2026 年中移除的双端点模型 |
| `Mcp-Session-Id` | "Session 头" | Server 分配的随机 id，在每个后续请求上回显 |
| `Origin` 允许列表 | "DNS 重绑定防御" | 拒绝 Origin 未被批准的请求 |
| Single endpoint（单端点） | "一个 URL" | `/mcp` 处理所有会话操作的 POST / GET / DELETE |
| `last-event-id` | "SSE 重放" | 用于恢复断开流而不丢失事件的头 |
| Backwards-compat probe（向后兼容探测） | "新旧检测" | Client 响应形状检查，自动选择传输 |
| Long-lived HTTP（长连接 HTTP） | "SSE streaming" | Server 在一个 TCP 连接上推送事件数分钟或数小时 |
| Session revocation（Session 撤销） | "强制重新初始化" | Server 使 session id 无效；client 必须重新握手 |

## 延伸阅读

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 和 Streamable HTTP 的权威参考
- [MCP — Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — 引入 Streamable HTTP 的修订版
- [Cloudflare — MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers 托管的 Streamable HTTP 模式
- [AWS — MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 跨部署形态的比较
- [Atlassian — HTTP+SSE deprecation notice](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 具体迁移截止日期示例
