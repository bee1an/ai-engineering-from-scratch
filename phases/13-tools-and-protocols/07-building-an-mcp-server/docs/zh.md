# 构建 MCP Server — Python + TypeScript SDK

> 大多数 MCP 教程只展示 stdio hello-world。一个真正的 server 暴露 tools 加 resources 加 prompts，处理能力协商，发出结构化错误，并且跨 SDK 工作方式相同。本课端到端构建一个笔记 server：标准库 stdio 传输、JSON-RPC 分发、三个服务端原语，以及一种纯函数风格，可以直接放入 Python SDK 的 FastMCP 或 TypeScript SDK。

**类型：** 构建
**语言：** Python（标准库，stdio MCP server）
**前置课程：** Phase 13 · 06（MCP 基础）
**时长：** 约 75 分钟

## 学习目标

- 实现 `initialize`、`tools/list`、`tools/call`、`resources/list`、`resources/read`、`prompts/list` 和 `prompts/get` 方法。
- 编写一个从 stdin 读取 JSON-RPC 消息并将响应写入 stdout 的分发循环。
- 按照 JSON-RPC 2.0 规范和 MCP 的附加错误码发出结构化错误响应。
- 将标准库实现升级到 FastMCP（Python SDK）或 TypeScript SDK，无需重写工具逻辑。

## 问题

在使用远程传输（Phase 13 · 09）或认证层（Phase 13 · 16）之前，你需要一个干净的本地 server。本地意味着 stdio：server 由 client 作为子进程启动，消息通过 stdin/stdout 以换行符分隔流动。

2025-11-25 规范规定 stdio 消息编码为带有显式 `\n` 分隔符的 JSON 对象。这里没有 SSE；SSE 是旧的远程模式，正在 2026 年中被移除（Atlassian 的 Rovo MCP server 于 2026 年 6 月 30 日弃用；Keboola 于 2026 年 4 月 1 日）。对于 stdio，每行一个 JSON 对象就是全部线上格式。

笔记 server 是一个好的形状，因为它练习了所有三个服务端原语。Tools 做变更（`notes_create`）。Resources 暴露数据（`notes://{id}`）。Prompts 提供模板（`review_note`）。本课的形状可泛化到任何领域。

## 概念

### 分发循环

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

三条规则：

- 不要向 stdout 打印任何不是 JSON-RPC 信封的内容。调试日志发到 stderr。
- 每个 request 必须匹配一个携带相同 `id` 的 response。
- Notifications 不得被响应。

### 实现 `initialize`

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

只声明你支持的。Client 依赖能力集来门控功能。

### 实现 `tools/list` 和 `tools/call`

`tools/list` 返回 `{tools: [...]}`，每个条目有 `name`、`description`、`inputSchema`。`tools/call` 接收 `{name, arguments}` 并返回 `{content: [blocks], isError: bool}`。

内容块是类型化的。最常见的：

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

工具错误有两种形状。协议级错误（未知方法、错误参数）是 JSON-RPC 错误。工具级错误（有效调用但工具失败）作为 `{content: [...], isError: true}` 返回。这让模型在其上下文中看到失败。

### 实现 resources

Resources 设计上是只读的。`resources/list` 返回清单；`resources/read` 返回内容。URI 可以是 `file://...`、`http://...` 或自定义 scheme 如 `notes://`。

当你将数据作为 resource 而非 tool 暴露时：

- 模型不"调用"它；client 可以在用户请求时将其注入上下文。
- 订阅让 server 在资源变化时推送更新（Phase 13 · 10）。
- Phase 13 · 14 用 `ui://` 扩展了交互式资源。

### 实现 prompts

Prompts 是带有命名参数的模板。宿主将它们作为斜杠命令呈现。一个 `review_note` prompt 可能接收 `note_id` 参数并产生一个多消息 prompt 模板，client 将其馈入模型。

### Stdio 传输细节

- 换行符分隔的 JSON。无长度前缀帧。
- 不要缓冲。每次写入后 `sys.stdout.flush()`。
- Client 控制生命周期。当 stdin 关闭（EOF）时，干净退出。
- 不要静默处理 SIGPIPE；记录日志并退出。

### 注解

每个工具可以携带描述安全属性的 `annotations`：

- `readOnlyHint: true` — 纯读取，可安全重试。
- `destructiveHint: true` — 不可逆副作用；client 应确认。
- `idempotentHint: true` — 相同输入产生相同输出。
- `openWorldHint: true` — 与外部系统交互。

Client 使用这些来决定 UX（确认对话框、状态指示器）和路由（Phase 13 · 17）。

### 升级路径

`code/main.py` 中的标准库 server 约 180 行。FastMCP（Python）将相同逻辑压缩为装饰器风格：

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK 有等价的形状。升级路径是即插即用的；概念（capabilities、dispatch、content blocks）是相同的。

## 动手试试

`code/main.py` 是一个完整的笔记 MCP server，基于 stdio，仅使用标准库。它处理 `initialize`、三个工具（`notes_list`、`notes_search`、`notes_create`）的 `tools/list` 和 `tools/call`、每个笔记的 `resources/list` 和 `resources/read`、以及一个 `review_note` prompt。你可以通过管道传入 JSON-RPC 消息来驱动它：

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

关注点：

- 分发器是一个以方法名为键的 `dict[str, Callable]`。
- 每个工具执行器返回内容块列表，而非裸字符串。
- 当执行器抛出异常时设置 `isError: true`。

## 交付物

本课产出 `outputs/skill-mcp-server-scaffolder.md`。给定一个领域（笔记、工单、文件、数据库），该技能脚手架一个 MCP server，带有正确的 tools / resources / prompts 拆分和 SDK 升级路径。

## 练习

1. 运行 `code/main.py` 并用手工构建的 JSON-RPC 消息驱动它。执行 `notes_create`，然后 `resources/read` 来检索新笔记。

2. 添加一个带有 `annotations: {destructiveHint: true}` 的 `notes_delete` 工具。验证 client 会弹出确认对话框（这需要真实宿主；Claude Desktop 可以）。

3. 实现 `resources/subscribe`，使 server 在笔记被修改时推送 `notifications/resources/updated`。添加一个 keepalive 任务。

4. 将 server 移植到 FastMCP。Python 文件应缩减到 80 行以下。线上行为必须相同；用相同的 JSON-RPC 测试框架验证。

5. 阅读规范的 `server/tools` 部分，找出本课 server 未实现的工具定义的一个字段。（提示：有好几个；选一个并添加。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MCP server | "暴露工具的那个东西" | 通过 stdio 或 HTTP 说 MCP JSON-RPC 的进程 |
| stdio transport | "子进程模型" | Server 由 client 启动；通过 stdin/stdout 通信 |
| Dispatcher（分发器） | "方法路由器" | JSON-RPC 方法名到处理函数的映射 |
| Content block（内容块） | "工具结果块" | 工具响应 `content` 数组中的类型化元素 |
| `isError` | "工具级失败" | 信号工具失败；区别于 JSON-RPC 错误 |
| Annotations（注解） | "安全提示" | readOnly / destructive / idempotent / openWorld 标志 |
| FastMCP | "Python SDK" | 基于装饰器的 MCP 协议高级框架 |
| Resource URI | "可寻址数据" | `file://`、`db://` 或自定义 scheme 标识资源 |
| Prompt template（Prompt 模板） | "斜杠命令简介" | Server 提供的带参数槽的模板，用于宿主 UI |
| Capability declaration（能力声明） | "功能开关" | 在 `initialize` 中声明的每原语标志 |

## 延伸阅读

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 参考 Python 实现
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 并行的 TS 实现
- [FastMCP — server framework](https://gofastmcp.com/) — 装饰器风格的 MCP server Python API
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — 使用任一 SDK 的端到端教程
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* 消息的完整参考
