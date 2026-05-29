# MCP Apps — 通过 `ui://` 提供交互式 UI 资源

> 纯文本工具输出限制了 agent 能展示的内容。MCP Apps（SEP-1724，2026 年 1 月 26 日正式发布）让工具返回沙箱化的交互式 HTML，在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中内联渲染。仪表盘、表单、地图、3D 场景，全部通过一个扩展实现。本课讲解 `ui://` 资源方案、`text/html;profile=mcp-app` MIME、iframe 沙箱 postMessage 协议，以及让服务器渲染 HTML 带来的安全面。

**Type:** Build
**Languages:** Python (stdlib, UI resource emitter), HTML (sample app)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Time:** ~75 minutes

## 学习目标

- 从工具调用返回 `ui://` 资源并设置正确的 MIME 和元数据。
- 使用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具关联的 UI。
- 实现 iframe 沙箱 postMessage JSON-RPC 用于 UI 到宿主的通信。
- 应用 CSP 和 permissions-policy 默认值来防御 UI 发起的攻击。

## 问题

2025 年的 `visualize_timeline` 工具只能返回"这里有 14 条按时间排列的笔记：..."。那只是一段文字。用户实际想要的是交互式时间线。在 MCP Apps 之前，选项是：客户端特定的 widget API（Claude artifacts、OpenAI Custom GPT HTML），或者完全没有 UI。

MCP Apps（SEP-1724，2026 年 1 月 26 日发布）标准化了这个契约。工具结果包含一个 `resource`，其 URI 是 `ui://...`，MIME 是 `text/html;profile=mcp-app`。宿主在受限 CSP 和无网络访问（除非显式授予）的沙箱 iframe 中渲染它。iframe 内的 UI 通过一个小型 postMessage JSON-RPC 方言向宿主发送消息。

每个兼容客户端（Claude Desktop、ChatGPT、Goose、VS Code）以相同方式渲染相同的 `ui://` 资源。一个服务器，一个 HTML 包，通用 UI。

## 概念

### `ui://` 资源方案

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

宿主随后对 `ui://notes/timeline` URI 调用 `resources/read`，获得：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe 沙箱

宿主在沙箱化的 `<iframe>` 中渲染 HTML：

- `sandbox="allow-scripts allow-same-origin"`（或按服务器声明更严格）
- 通过响应头应用服务器声明的 CSP。
- 没有来自宿主 origin 的 cookies 或 localStorage。
- 网络访问限制在 CSP 的 `connectSrc` 内。

### postMessage 协议

iframe 通过 `window.postMessage` 与宿主通信。一个小型 JSON-RPC 2.0 方言：

始终将 `targetOrigin` 固定为对端的确切 origin，接收端在处理任何 payload 之前验证 `event.origin` 是否在允许列表中。永远不要对此通道的任何一端使用 `"*"` — 消息体携带工具调用和资源读取。

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

UI 可以调用的宿主端方法：

- `host.callTool(name, arguments)` — 调用服务器工具。
- `host.readResource(uri)` — 读取 MCP 资源。
- `host.getPrompt(name, arguments)` — 获取 prompt 模板。
- `host.close()` — 关闭 UI。

每个调用仍然通过 MCP 协议并继承服务器的权限。

### 权限

`_meta.ui.permissions` 列表请求额外能力：

- `camera` — 访问用户摄像头（用于扫描文档的 UI）。
- `microphone` — 语音输入。
- `geolocation` — 位置。
- `network:*` — 比 `connectSrc` 单独允许的更广泛的网络访问。

每个权限都是用户在 UI 渲染前看到的提示。

### 安全风险

iframe 中的 HTML 仍然是 HTML。新的攻击面：

- **通过 UI 的 prompt 注入。** 恶意服务器 UI 可以显示看起来像系统消息的文本来欺骗用户。宿主渲染应明显区分服务器 UI 和宿主 UI。
- **通过 `connectSrc` 的数据外泄。** 如果 CSP 允许 `connect-src: *`，UI 可以向任何地方发送数据。默认应该严格。
- **点击劫持。** UI 覆盖宿主 chrome。宿主必须防止 z-index 操纵并强制执行不透明度规则。
- **窃取焦点。** UI 获取键盘焦点并捕获下一条消息。宿主必须拦截。

Phase 13 · 15 作为 MCP 安全的一部分深入涵盖了这些；本课介绍它们。

### `ui/initialize` 握手

iframe 加载后，通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

宿主响应能力和会话令牌。UI 在后续每次宿主调用中使用该会话令牌。

### AppRenderer / AppFrame SDK 原语

ext-apps SDK 暴露两个便利原语：

- `AppRenderer`（服务端）— 包装 React / Vue / Solid 组件并发出带正确 MIME 和元数据的 `ui://` 资源。
- `AppFrame`（客户端）— 接收资源，挂载 iframe，并中介 postMessage。

你可以使用这些或手写 HTML 和 JSON-RPC。

### 生态系统状态

MCP Apps 于 2026 年 1 月 26 日发布。截至 2026 年 4 月的客户端支持：

- **Claude Desktop.** 自 2026 年 1 月起完全支持。
- **ChatGPT.** 通过 Apps SDK 完全支持（底层相同的 MCP Apps 协议）。
- **Cursor.** Beta；通过设置启用。
- **VS Code.** 仅 Insider 构建。
- **Goose.** 完全支持。
- **Zed, Windsurf.** 已列入路线图。

生产中的服务器：仪表盘、地图可视化、数据表、图表构建器、沙箱 IDE 预览。

## 动手实践

`code/main.py` 扩展了笔记服务器，添加了一个 `visualize_timeline` 工具，返回 `ui://notes/timeline` 资源，加上一个处理该 URI 的 `resources/read` handler，返回一个小而完整的带 SVG 时间线的 HTML 包。HTML 使用 stdlib 模板 — 无构建系统。postMessage 在 JS 注释中草拟，因为 stdlib 无法驱动浏览器。

关注点：

- 工具响应上的 `_meta.ui` 携带 resourceUri、CSP、permissions。
- HTML 无需网络访问即可渲染；所有数据内联。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（在此 stdlib 演示中有文档但不活跃）。

## 交付产出

本课产出 `outputs/skill-mcp-apps-spec.md`。给定一个可以从交互式 UI 中受益的工具，该技能产出完整的 MCP Apps 契约：`ui://` URI、CSP、permissions、postMessage 入口点和安全检查清单。

## 练习

1. 运行 `code/main.py` 并检查发出的 HTML。直接在浏览器中打开 HTML；验证 SVG 渲染。然后草拟 UI 调用 `host.callTool("notes_update", ...)` 时使用的 postMessage 契约。

2. 收紧 CSP：移除 `'unsafe-inline'` 并使用基于 nonce 的脚本策略。HTML 生成代码需要什么变化？

3. 添加第二个 UI 资源 `ui://notes/editor`，带有就地编辑笔记的表单。用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击面。恶意服务器可以在哪里注入内容？iframe 沙箱防御了什么，没防御什么？

5. 阅读 SEP-1724 规范，找出 MCP Apps SDK 中此玩具实现未使用的一个能力。（提示：组件级状态同步。）

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| MCP Apps | "交互式 UI 资源" | SEP-1724 扩展，2026-01-26 发布 |
| `ui://` | "App URI 方案" | UI 包的资源方案 |
| `text/html;profile=mcp-app` | "那个 MIME" | MCP App HTML 的 Content-type |
| Iframe sandbox | "渲染容器" | 带 CSP 和权限的浏览器沙箱化 UI |
| postMessage JSON-RPC | "UI 到宿主的通信线" | 用于宿主调用的小型 JSON-RPC-over-postMessage 方言 |
| `_meta.ui` | "工具-UI 绑定" | 将工具结果链接到 UI 资源的元数据 |
| CSP | "Content-Security-Policy" | 声明脚本、网络、样式的允许来源 |
| AppRenderer | "服务端 SDK 原语" | 将框架组件转换为 `ui://` 资源 |
| AppFrame | "客户端 SDK 原语" | 挂载 iframe 并中介 postMessage 的辅助器 |
| `ui/initialize` | "握手" | UI 到宿主的第一条 postMessage |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 参考实现和 SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 正式规范文档
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — 高层文档
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026 年 1 月发布帖
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc 风格的 SDK 参考
