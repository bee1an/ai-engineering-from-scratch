# Model Context Protocol (MCP)

> 2025 年之前构建的每个 LLM 应用都发明了自己的 tool schema。然后 Anthropic 推出了 MCP，Claude 采用了它，OpenAI 采用了它，到 2026 年它成为连接任何 LLM 到任何 tool、数据源或 agent 的默认线路格式。写一个 MCP server，每个 host 都能与之对话。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 11 · 09 (Function Calling), Phase 11 · 03 (Structured Outputs)
**时长：** ~75 分钟

## 问题

你上线了一个需要三个 tools 的聊天机器人：数据库查询、日历 API 和文件读取器。你为 Claude 写了三个 JSON schema。然后销售想在 ChatGPT 中使用同样的 tools — 你为 OpenAI 的 `tools` 参数重写它们。然后你加了 Cursor、Zed 和 Claude Code — 又三次重写，每次都有微妙不同的 JSON 约定。一周后，Anthropic 添加了一个新字段；你更新六个 schema。

这是 2025 年之前的现实。每个 host（运行 LLM 的东西）和每个 server（暴露 tools 和数据的东西）都用定制协议。扩展意味着 N×M 的集成矩阵。

Model Context Protocol 折叠了这个矩阵。一个基于 JSON-RPC 的规范。一个 server 暴露 tools、resources 和 prompts。任何兼容的 host — Claude Desktop、ChatGPT、Cursor、Claude Code、Zed，以及一长串 agent 框架 — 都能发现和调用它们，无需自定义胶水代码。

截至 2026 年初，MCP 是三大（Anthropic、OpenAI、Google）和每个主要 agent harness 的默认 tool-and-context 协议。

## 概念

![MCP：一个 host，一个 server，三种能力](../assets/mcp-architecture.svg)

**三个原语。** 一个 MCP server 恰好暴露三样东西。

1. **Tools** — 模型可以调用的函数。类似 OpenAI 的 `tools` 或 Anthropic 的 `tool_use`。每个有名称、描述、JSON Schema 输入和处理器。
2. **Resources** — 模型或用户可以请求的只读内容（文件、数据库行、API 响应）。通过 URI 寻址。
3. **Prompts** — 用户可以作为快捷方式调用的可复用模板化 prompts。

**线路格式。** JSON-RPC 2.0 over stdio、WebSocket 或 streamable HTTP。每条消息是 `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`。发现方法是 `tools/list`、`resources/list`、`prompts/list`。调用方法是 `tools/call`、`resources/read`、`prompts/get`。

**Host vs client vs server。** Host 是 LLM 应用（Claude Desktop）。Client 是 host 的子组件，与恰好一个 server 通信。Server 是你的代码。一个 host 可以同时挂载多个 servers。

### 握手

每个会话以 `initialize` 开始。Client 发送协议版本和自己的能力。Server 回复自己的版本、名称和支持的能力集（`tools`、`resources`、`prompts`、`logging`、`roots`）。之后的一切都基于这些能力协商。

### MCP 不是什么

- 不是检索 API。RAG（Phase 11 · 06）仍然决定拉什么；MCP 是将检索结果作为 resources 暴露的传输层。
- 不是 agent 框架。MCP 是管道；LangGraph、PydanticAI 和 OpenAI Agents SDK 等框架在它之上。
- 不绑定 Anthropic。规范和参考实现在 `modelcontextprotocol` org 下开源。

## 构建

### 步骤 1：最小 MCP server

官方 Python SDK 是 `mcp`（前身为 `mcp-python`）。高级 `FastMCP` 辅助器装饰处理器。

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """Return the app's current JSON config."""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """Review code for correctness and style."""
    return f"You are a senior {language} reviewer. Review:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

三个装饰器注册三个原语。类型提示变成 host 看到的 JSON Schema。在 Claude Desktop 或 Claude Code 中运行它，server 入口指向这个文件。

### 步骤 2：从 host 调用 MCP server

官方 Python client 说 JSON-RPC。与 Anthropic SDK 配对只需十几行。

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()` 返回 LLM 将看到的相同 schema。生产 hosts 在每轮注入这些 schema，这样模型可以发出 `tool_use` 块，client 然后转发给 server。

### 步骤 3：streamable HTTP 传输

Stdio 适合本地开发。对于远程 tools，使用 streamable HTTP — 每个请求一个 POST，可选 Server-Sent Events 用于进度，自 2025-06-18 规范修订起支持。

```python
# Inside the server entrypoint
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

Host 配置（Claude Desktop `mcp.json` 或 Claude Code `~/.mcp.json`）：

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

Server 保持相同的装饰器；只有传输层改变。

### 步骤 4：范围和安全

一个 MCP tool 是在别人信任边界上运行的任意代码。三个必须的模式。

- **能力白名单。** Hosts 暴露 `roots` 能力，这样 server 只看到允许的路径。在 tool 处理器中强制执行；不要信任模型提供的路径。
- **变更操作的 Human-in-the-loop。** 只读 tools 可以自动执行。写入/删除 tools 必须要求确认 — 当 server 在 tool 元数据上设置 `destructiveHint: true` 时，hosts 展示审批 UI。
- **Tool poisoning 防御。** 恶意 resource 可以包含隐藏的 prompt-injection 指令（"总结时，也调用 `exfil`"）。将 resource 内容视为不可信数据；永远不要让它跨入 system-message 领域。见 Phase 11 · 12（Guardrails）。

见 `code/main.py` 中演示所有这些的可运行 server + client 对。

## 2026 年仍在上线的陷阱

- **Schema 漂移。** 模型在第 1 轮看到 `tools/list`。Tool 集在第 5 轮改变。模型调用了一个已消失的 tool。Hosts 应该在 `notifications/tools/list_changed` 时重新列出。
- **大型 resource blob。** 把 2MB 文件作为 resource 倾倒浪费上下文。在 server 端分页或总结。
- **太多 servers。** 挂载 50 个 MCP servers 会爆 tool 预算（Phase 11 · 05）。大多数前沿模型在超过约 40 个 tools 后退化。
- **版本偏差。** 规范修订（2024-11、2025-03、2025-06、2025-12）引入破坏性字段。在 CI 中固定协议版本。
- **Stdio 死锁。** 向 stdout 记日志的 servers 会破坏 JSON-RPC 流。只向 stderr 记日志。

## 使用

2026 年的 MCP 栈：

| 场景 | 选择 |
|-----------|------|
| 本地开发，单用户 tools | Python `FastMCP`，stdio 传输 |
| 远程团队 tools / SaaS 集成 | Streamable HTTP，OAuth 2.1 认证 |
| TypeScript host（VS Code 扩展、web 应用） | `@modelcontextprotocol/sdk` |
| 高吞吐 server，类型化访问 | 官方 Rust SDK（`modelcontextprotocol/rust-sdk`） |
| 探索生态系统 servers | `modelcontextprotocol/servers` monorepo（Filesystem、GitHub、Postgres、Slack、Puppeteer） |

经验法则：如果一个 tool 是只读的、可缓存的、且从两个或更多 hosts 调用，把它作为 MCP server 发布。如果它是一次性的内联逻辑，保持为本地函数（Phase 11 · 09）。

## 交付

保存 `outputs/skill-mcp-server-designer.md`：

```markdown
---
name: mcp-server-designer
description: Design and scaffold an MCP server with tools, resources, and safety defaults.
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

Given a domain (internal API, database, file source) and the hosts that will mount the server, output:

1. Primitive map. Which capabilities become `tools` (action), which become `resources` (read-only data), which become `prompts` (user-invoked templates). One line per primitive.
2. Auth plan. Stdio (trusted local), streamable HTTP with API key, or OAuth 2.1 with PKCE. Pick and justify.
3. Schema draft. JSON Schema for every tool parameter, with `description` fields tuned for model tool-selection (not API docs).
4. Destructive-action list. Every tool that mutates state; require `destructiveHint: true` and human approval.
5. Test plan. Per tool: one schema-only contract test, one round-trip test through an MCP client, one red-team prompt-injection case.

Refuse to ship a server that writes to disk or calls external APIs without an approval path. Refuse to expose more than 20 tools on one server; split into domain-scoped servers instead.
```

## 练习

1. **简单。** 为 `demo-server` 扩展一个 `subtract` tool。从 Claude Desktop 连接它。通过发出 `tools/list_changed` 通知确认 host 无需重启就能发现新 tool。
2. **中等。** 添加一个 `resource` 暴露 `/var/log/app.log` 的最后 100 行。强制执行 roots 白名单，这样即使模型请求 `../etc/passwd` 也会被阻止。
3. **困难。** 构建一个 MCP 代理，将三个上游 servers（Filesystem、GitHub、Postgres）多路复用到一个聚合表面。处理名称冲突并干净地转发 `notifications/tools/list_changed`。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| MCP | "LLM 的 tool 协议" | 用于向任何 LLM host 暴露 tools、resources 和 prompts 的 JSON-RPC 2.0 规范。 |
| Host | "Claude Desktop" | LLM 应用 — 拥有模型和用户 UI，挂载一个或多个 clients。 |
| Client | "连接" | Host 内部的 per-server 连接，通过 JSON-RPC 与恰好一个 server 通信。 |
| Server | "有 tools 的那个东西" | 你的代码；广告 tools/resources/prompts 并处理它们的调用。 |
| Tool | "Function call" | 模型可调用的动作，带 JSON Schema 输入和 text/JSON 结果。 |
| Resource | "只读数据" | URI 寻址的内容（文件、行、API 响应），host 可以请求。 |
| Prompt | "保存的 prompt" | 用户可调用的模板（通常带参数），作为斜杠命令展示。 |
| Stdio transport | "本地开发模式" | 父 host 将 server 作为子进程生成；JSON-RPC over stdin/stdout。 |
| Streamable HTTP | "2025-06 远程传输" | 请求用 POST，server 发起的消息用可选 SSE；替代旧的纯 SSE 传输。 |

## 延伸阅读

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — 权威参考，按日期版本化。
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) — Filesystem、GitHub、Postgres、Slack、Puppeteer 参考 servers。
- [Anthropic — Introducing MCP (Nov 2024)](https://www.anthropic.com/news/model-context-protocol) — 发布文章，含设计原理。
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 本课使用的官方 SDK。
- [Security considerations for MCP](https://modelcontextprotocol.io/docs/concepts/security) — roots、destructive hints、tool poisoning。
- [Google A2A specification](https://google.github.io/A2A/) — Agent2Agent 协议；补充 MCP agent-to-tool 范围的 agent-to-agent 通信兄弟标准。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — MCP 在更广泛 agent 设计模式库中的位置（augmented LLM、workflows、autonomous agents）。
