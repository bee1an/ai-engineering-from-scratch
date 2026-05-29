# MCP 安全性 II — OAuth 2.1、Resource Indicators、增量 Scopes

> 远程 MCP 服务器需要授权，而不仅仅是认证。2025-11-25 规范与 OAuth 2.1 + PKCE + resource indicators（RFC 8707）+ protected-resource metadata（RFC 9728）对齐。SEP-835 通过 403 WWW-Authenticate 上的 step-up authorization 添加了增量 scope 同意。本课将 step-up 流程实现为状态机，让你看到每一跳。

**Type:** Build
**Languages:** Python (stdlib, OAuth state machine simulator)
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 15 (security I)
**Time:** ~75 minutes

## 学习目标

- 区分资源服务器和授权服务器的职责。
- 走通 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和 protected-resource metadata（RFC 9728）防止 confused-deputy 攻击。
- 实现 step-up authorization：服务器响应 403 并通过 WWW-Authenticate 请求更高 scope；客户端重新提示用户同意并重试。

## 问题

早期 MCP（2025 年之前）的远程服务器使用临时 API key 甚至没有认证。2025-11-25 规范用完整的 OAuth 2.1 profile 填补了这个空白。

三个现实需求：

- **普通远程服务器。** 用户安装一个访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。OAuth 2.1 with PKCE 是正确的形式。
- **Scope 升级。** 一个被授予 `notes:read` 的笔记服务器后来需要 `notes:write` 来执行特定操作。不必重做整个流程，step-up（SEP-835）请求额外的 scope。
- **Confused deputy 防护。** 客户端持有一个 audience 限定为 Server A 的 token。Server A 是恶意的，试图将该 token 提交给 Server B。Resource indicators（RFC 8707）将 token 固定到其预期受众。

OAuth 2.1 不是新的。新的是 MCP 的 profile：特定的必需流程（仅授权码 + PKCE；默认无 implicit、无 client credentials）、每个 token 请求强制 resource indicators、以及发布 protected-resource metadata 让客户端知道去哪里。

## 概念

### 角色

- **Client。** MCP 客户端（Claude Desktop、Cursor 等）。
- **Resource server。** MCP 服务器（notes、GitHub、Postgres 等）。
- **Authorization server。** 签发 token。可以与资源服务器是同一服务，也可以是独立的 IdP（Auth0、Keycloak、Cognito）。

在 MCP 的 profile 中，资源服务器和授权服务器可以是同一主机，但应该通过 URL 区分。

### 授权码 + PKCE

流程：

1. 客户端生成 `code_verifier`（随机）和 `code_challenge`（SHA256）。
2. 客户端将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意。授权服务器重定向到 `redirect_uri?code=...`。
4. 客户端 POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. 授权服务器验证 verifier 的哈希与存储的 challenge 匹配，签发 access token。
6. 客户端使用 token：对资源服务器的每个请求带上 `Authorization: Bearer ...`。

PKCE 防止授权码拦截攻击。Resource indicators 防止 token 在其他地方有效。

### Protected-resource metadata（RFC 9728）

资源服务器发布 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器。减少配置 — 客户端只需要资源 URL。

### Resource indicators（RFC 8707）

token 请求中的 `resource` 参数固定 token 的预期受众。签发的 token 包含 `aud: "https://notes.example.com"`。另一个 MCP 服务器收到此 token 时检查 `aud` 并拒绝。

### Scope 模型

Scopes 是空格分隔的字符串。常见 MCP 约定：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 用于管理能力（谨慎使用）
- `profile:read` 用于身份

Scope 选择应遵循最小权限：请求当前需要的，需要更多时再 step up。

### Step-up authorization（SEP-835）

用户授予 `notes:read`。后来他们要求 agent 删除一个笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端看到 insufficient_scope 错误，向用户弹出额外 scope 的同意对话框，为其执行一个小型 OAuth 流程，用新 token 重试请求。

### Token audience 验证

每个请求：服务器检查 `token.aud == self.resource_url`。不匹配 = 401。这阻止了跨服务器 token 重用。

### 短期 token 和轮换

Access token 应该是短期的（默认 1 小时）。Refresh token 在每次刷新时轮换。客户端在后台处理静默刷新。

### 禁止 token 透传

Sampling 服务器（Phase 13 · 11）禁止将客户端的 token 透传给其他服务。Sampling 请求是边界。

### Confused deputy 防护

Token 绑定到 `aud`。客户端绑定到 `client_id`。每个请求都对两者进行验证。规范明确禁止了在 MCP 之前的远程工具生态系统中常见的旧"传递 token"模式。

### Client ID 发现

每个 MCP 客户端在固定 URL 发布其元数据。授权服务器可以获取客户端的元数据文档来发现 redirect URI 和联系信息。这消除了手动客户端注册。

### 网关与 OAuth

Phase 13 · 17 展示了企业网关如何处理 OAuth：网关持有上游服务器的凭证，给客户端的 token 由网关签发，上游 token 永远不离开网关。这翻转了信任模型 — 用户对网关认证一次；网关处理 N 个服务器的授权。

## 动手实践

`code/main.py` 将完整的 OAuth 2.1 step-up 流程模拟为状态机。它实现了：

- PKCE code-verifier / challenge 生成。
- 带 resource indicator 的授权码流程。
- Protected-resource metadata 端点。
- 带 audience 检查的 token 验证。
- `insufficient_scope` 上的 step-up。

本课没有 HTTP 服务器；状态机在内存中运行，让你可以追踪每一跳。Phase 13 · 17 的网关课将其连接到实际传输。

## 交付产出

本课产出 `outputs/skill-oauth-scope-planner.md`。给定一个带工具的远程 MCP 服务器，该技能设计 scope 集合、pinning 规则和 step-up 策略。

## 练习

1. 运行 `code/main.py`。追踪两个 scope 的 step-up 流程。注意哪些跳在 step-up 时重复。

2. 添加 refresh-token 轮换：每次刷新签发新的 refresh token 并使旧的失效。模拟被盗的 refresh token 在轮换后被使用并确认失败。

3. 使用 stdlib http.server 将 protected-resource metadata 端点实现为真实的 HTTP 响应。镜像 Lesson 09 的 /mcp 端点。

4. 为 GitHub MCP 服务器设计 scope 层次结构：read repo、write PR、approve PR、merge PR、admin。在每个级别之间使用 step-up。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中 MCP 与 RFC 示例使用方式不同的一个字段。（提示：与 `scopes_supported` 有关。）

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| OAuth 2.1 | "现代 OAuth" | 强制 PKCE 并禁止 implicit flow 的整合 RFC |
| PKCE | "持有证明" | Code verifier + challenge 击败授权码拦截 |
| Resource indicator | "Token audience" | RFC 8707 `resource` 参数将 token 固定到一个服务器 |
| Protected-resource metadata | "发现文档" | RFC 9728 `.well-known/oauth-protected-resource` |
| Step-up authorization | "增量同意" | SEP-835 按需添加 scope 的流程 |
| `insufficient_scope` | "403 with WWW-Authenticate" | 服务器信号要求重新同意更大的 scope |
| Confused deputy | "跨服务 token 重用" | 受信持有者不当转发 token 的攻击 |
| Short-lived token | "Access token TTL" | 快速过期的 Bearer；refresh token 续期 |
| Scope hierarchy | "最小权限栈" | 带级别间 step-up 的分级 scope 集合 |
| Client ID metadata | "客户端发现文档" | 客户端发布其 OAuth 元数据的 URL |

## 延伸阅读

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 规范的 MCP OAuth profile
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更详解
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience-pinning RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档 RFC
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 实用的 step-up 流程详解
