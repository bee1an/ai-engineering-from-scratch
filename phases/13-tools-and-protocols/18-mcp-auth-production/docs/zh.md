# MCP 生产环境认证 — DCR、JWKS 轮换、基于 iii 原语的 Audience-Pinned Tokens

> Lesson 16 在内存中搭建了 OAuth 2.1 状态机。到 2026 年，你发布给真实组织的每个 MCP 服务器都位于生产认证之后：动态客户端注册（RFC 7591）、授权服务器元数据发现（RFC 8414）、不会在凌晨 3 点破坏 token 验证的 JWKS 轮换，以及拒绝 confused-deputy 重用的 audience-pinned tokens。本课通过 iii 原语串联所有这些 — `iii.registerTrigger` 用于 HTTP 和 cron，`iii.registerFunction` 用于认证逻辑，`state::set/get` 用于缓存的密钥 — 使认证面可观测、可重启、可重放，就像引擎中的每个其他工作负载一样。

**Type:** Build
**Languages:** Python (stdlib, iii primitives mocked for the lesson environment)
**Prerequisites:** Phase 13 · 16 (OAuth 2.1 state machine), Phase 13 · 17 (gateways)
**Time:** ~90 minutes

## 学习目标

- 通过 RFC 8414 元数据发现授权服务器并验证契约。
- 实现 RFC 7591 动态客户端注册，使 MCP 客户端无需管理员干预即可注册。
- 使用 cron trigger 缓存和轮换 JWKS 密钥，使签名验证在密钥轮换后仍然有效。
- 使用 RFC 8707 resource indicators 将 token 固定到单个 MCP 资源，拒绝 confused-deputy 重用。
- 将每个端点和后台任务作为 iii 原语连接 — HTTP triggers、cron triggers、命名函数和 `state::*` 读取 — 使单次重启即可重建认证面。
- 阅读 IdP 能力矩阵，当 IdP 无法满足 MCP 的认证 profile 时拒绝部署。

## 问题

Lesson 16 的模拟器在内存中运行 OAuth 2.1。生产环境有三个纯内存模拟器看不到的运维缺口。

第一个缺口是注册。真实组织运行数百个 MCP 服务器和数千个 MCP 客户端。运维人员不会为每个 Cursor 用户手动注册 OAuth 客户端。RFC 7591 动态客户端注册让客户端对授权服务器 `POST /register` 并当场获得 `client_id`（以及可选的 `client_secret`）。服务器在其 RFC 8414 元数据中发布 `registration_endpoint`；客户端无需带外配置即可发现它。

第二个缺口是密钥轮换。JWT 验证依赖授权服务器的签名密钥，以 JSON Web Key Set (JWKS) 发布。授权服务器按计划轮换这些密钥（通常每小时，有时在事件响应下更快）。启动时只获取一次 JWKS 的 MCP 服务器在轮换窗口之前验证正常 — 然后每个请求都失败直到重启。生产环境将 JWKS 作为带刷新任务的缓存值，在前一个密钥过期前覆盖缓存，加上缓存未命中时的回退获取，用于 token 由比缓存更新的密钥签名的情况。

第三个缺口是 audience 绑定。Lesson 16 引入了 RFC 8707 resource indicators。在生产中，该 indicator 成为每个请求上的硬 claim 检查。MCP 服务器将 `token.aud` 与自己的规范资源 URL 比较，不匹配则返回 HTTP 401。这是在同一信任网格中防止上游 MCP 服务器（或持有针对一个服务器的 token 的恶意客户端）将该 token 重放到另一个服务器的唯一防御。

本课将每个缺口都视为 iii 原语。元数据文档是返回函数输出的 HTTP trigger。JWKS 轮换是调用 `auth::rotate-jwks` 的 cron trigger，它写入 `state::set("auth/jwks/<issuer>", ...)`。JWT 验证是其他人通过 `iii.trigger("auth::validate-jwt", token)` 调用的函数。MCP 服务器本身只是另一个在分发前调用验证的 HTTP trigger。重启引擎：trigger 注册表重建；状态存活；认证面无需手动协调即可运行。

## 概念

### RFC 8414 — OAuth Authorization Server Metadata

`/.well-known/oauth-authorization-server` 处的文档描述了客户端需要的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

给定 MCP 资源 URL 的客户端链式发现：RFC 9728 的 `oauth-protected-resource`（资源服务器的文档）命名 issuer，然后 `oauth-authorization-server`（此 RFC）命名每个端点。客户端永远不硬编码授权 URL。

信任 IdP 用于 MCP 之前你验证的契约：

- `code_challenge_methods_supported` 包含 `S256`（RFC 7636 的 PKCE）。
- `grant_types_supported` 包含 `authorization_code` 且拒绝 `password` 和 `implicit`。
- `registration_endpoint` 存在（RFC 7591 支持）。
- `response_types_supported` 对 OAuth 2.1 恰好是 `["code"]`。

如果任何一项缺失，MCP 服务器拒绝对此 IdP 部署。部署清单有问题，不是代码。

### RFC 9728（回顾）— Protected Resource Metadata

Lesson 16 涵盖了 RFC 9728。生产中的增量：此文档是客户端查找*此* MCP 服务器信任的授权服务器的唯一位置。单个 MCP 服务器可以接受来自多个 IdP 的 token（一个用于员工，一个用于合作伙伴）。RFC 9728 声明该集合；RFC 8414 记录每个 IdP 支持什么。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591 — Dynamic Client Registration

没有 DCR，每个 MCP 客户端（Cursor、Claude Desktop、自定义 agent）都需要与 IdP 管理员的带外交换。有了 DCR，客户端发送：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器响应 `client_id` 和用于后续更新的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` 是运行在用户设备上的 MCP 客户端的正确默认值。它们只获得 `client_id` — 没有可被窃取的 `client_secret`。PKCE 提供公共客户端需要的持有证明。

三个生产陷阱：

- 注册端点必须按源 IP 限速。否则，恶意行为者可以脚本化数百万假注册并耗尽 `client_id` 命名空间。iii 使这变得简单：注册 HTTP trigger 在分发到注册器之前调用 `auth::rate-limit` 函数。
- `software_statement`（为客户端担保的签名 JWT）被某些企业 IdP 要求。本课的 mock 跳过它；生产环境连接一个验证步骤，拒绝来自非 localhost redirect URI 的未签名注册。
- `registration_access_token` 必须以哈希形式存储，而非明文。此 token 被盗意味着攻击者可以重写客户端的 redirect URI。

### RFC 8707（回顾）— Resource Indicators

Lesson 16 建立了形状。生产规则：每个 token 请求包含 `resource=<canonical-mcp-url>`，MCP 服务器在每次调用时验证 `token.aud` 与自己的资源 URL 匹配。如果 MCP 服务器可通过 `https://notes.example.com/mcp` 访问，规范 URL 是 `https://notes.example.com` — 排除路径组件，使单个服务器在一个 audience 下托管多个路径。

### RFC 7636（回顾）— PKCE

PKCE 在 OAuth 2.1 中是强制的。本课的授权码流程始终携带 `code_challenge` 和 `code_verifier`。服务器拒绝任何没有 verifier 或 verifier 哈希与存储的 challenge 不匹配的 token 请求。

### MCP Spec 2025-11-25 Auth Profile

MCP 规范（2025-11-25）精确规定了 MCP 服务器的授权层必须做什么：

- 发布 `/.well-known/oauth-protected-resource`（RFC 9728）。
- 仅通过 `Authorization: Bearer ...` 接受 token。
- 每个请求验证 `aud`、`iss`、`exp` 和所需 scope。
- 对每个 401 和 403 响应 `WWW-Authenticate`，携带 `Bearer error=...`，包括适用的 `scope=` 和 `resource=` 参数。
- 拒绝 `aud` 与规范资源不匹配的 token。
- 拒绝 `iss` 不在 protected-resource metadata 的 `authorization_servers` 列表中的 token。

OAuth 2.1 草案是基底；RFC 8414/7591/8707/9728 + RFC 7636 是表面；MCP 规范是 profile。

### IdP 能力矩阵

不是每个 IdP 都支持完整的 MCP profile。下面的矩阵记录了截至 2025-11-25 规范的事实能力声明。它是*部署门控*，不是推荐。

| IdP 类别 | RFC 8414 metadata | RFC 7591 DCR | RFC 8707 resource | RFC 7636 S256 PKCE | 备注 |
|---|---|---|---|---|---|
| 自托管 (Keycloak) | yes | yes | yes (since 24.x) | yes | 本课 MCP profile 的参考 IdP；端到端支持每个 RFC。 |
| 企业 SSO (Microsoft Entra ID) | yes | yes (premium tiers) | yes | yes | DCR 可用性因租户层级而异；部署前在目标租户中验证。 |
| 企业 SSO (Okta) | yes | yes (Okta CIC / Auth0) | yes | yes | DCR 在 Auth0（现 Okta CIC）上可用；经典 Okta org 需要管理员预注册。 |
| 社交登录 IdP (通用) | varies | rarely | rarely | yes | 大多数社交 IdP 将客户端视为静态合作伙伴；不要依赖 DCR。仅作为身份源使用，在其上层叠你自己的 MCP 感知授权服务器。 |
| 自定义 / 自建 | depends | depends | depends | depends | 如果你自己发布，发布完整 profile。跳过上述四个 RFC 中的任何一个都会破坏 MCP 认证契约。 |

部署清单的拒绝规则：如果选择的 IdP 不返回 `registration_endpoint` 且不在 `code_challenge_methods_supported` 中列出 `S256`，MCP 服务器拒绝启动。没有降级模式。

### 使用 iii 的 JWKS 轮换模式

生产故障模式是过期的 JWKS 缓存。用 cron trigger 和 `state::*` 缓存解决：

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```

每六小时，cron trigger 调用 `auth::rotate-jwks`，它获取 `<issuer>/.well-known/jwks.json` 并写入 `state::set("auth/jwks/<issuer>", {keys, fetched_at})`。验证器从 `state::get` 读取。`kid` 不在缓存中的 token 触发同步 `auth::rotate-jwks` 调用作为回退。这同时处理两种情况：计划轮换（cron）和密钥重叠窗口（同步回退）。

状态形状：

```json
{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同时有两个密钥是稳态。授权服务器通过在退役前一个密钥（`k_2026_03`）之前引入下一个密钥（`k_2026_04`）来轮换，因此在旧密钥下签发的 token 在过期前仍然有效。缓存持有并集；验证器按 `kid` 选择。

### iii 原语连接（本课真正要讲的部分）

五个原语组成认证面：

```python
# 1. RFC 8414 metadata document
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 dynamic client registration
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT validation as a callable function (the resource server triggers it)
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. Step-up issuance for incremental scope (SEP-835 from L16)
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. Cron-driven JWKS rotation
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

MCP 服务器本身从不直接调用验证。它做的是：

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

这种间接性是 iii 的赌注。明天你把验证器换成并行咨询两个 IdP 的 fanout，或者你添加 span 发射器，或者你缓存正向验证。MCP 服务器不变。

### 带 audience 绑定的 Confused-deputy 演练

Server A（`notes.example.com`）和 Server B（`tasks.example.com`）都注册到同一授权服务器。Server A 被入侵。攻击者拿到用户的 notes token 并重放到 Server B。

Server B 的验证器：

1. 解码 JWT，按 `kid` 获取 JWKS，验证签名。
2. 检查 `iss` 是否在其 protected-resource metadata 的 `authorization_servers` 中。（通过 — 同一 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败 — token 的 `aud` 是 `https://notes.example.com`。）
4. 返回 401，`WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`。

audience claim 是协议层面对此攻击的唯一防御。为性能跳过它是最常见的生产错误；验证器必须在每个请求上运行，而不仅仅在会话开始时。

### 故障模式

- **过期 JWKS。** 验证器在密钥轮换后拒绝有效 token。修复是上面的 cron+回退模式。永远不要在没有刷新任务的情况下缓存 JWKS。
- **缺失 `aud` claim。** 某些 IdP 默认省略 `aud`，除非 token 请求中有 `resource`。验证器必须拒绝缺失 `aud` 的 token，而非将缺失视为通配符。
- **Scope 升级竞态。** 同一用户的两个并发 step-up 流程都可能成功并产生两个不同 scope 的 access token。验证器必须使用请求上呈现的 token，而非查找"用户当前的 scope" — 那会创建 TOCTOU 窗口。
- **Registration token 被盗。** 泄露的 `registration_access_token` 让攻击者重写 redirect URI。静态存储时哈希它们；要求客户端在每次更新时呈现明文；怀疑时轮换。
- **`iss` 未固定。** 接受任何 `iss` 的验证器让攻击者搭建自己的授权服务器，为目标 audience 注册客户端，并签发 token。Protected-resource metadata 的 `authorization_servers` 列表是允许列表；强制执行它。

## 动手实践

`code/main.py` 使用 stdlib Python 和一个小型 `iii_mock` 注册表走通完整的生产流程，该注册表模拟 `iii.registerFunction`、`iii.registerTrigger`、`iii.trigger` 和 `state::set/get`。流程：

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点，发现注册端点。
3. MCP 客户端 POST 到 `/register`（RFC 7591）并获得 `client_id`。
4. MCP 客户端运行带 `resource` indicator（RFC 8707）的 PKCE 保护授权码流程（RFC 7636）。
5. MCP 客户端带 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器触发 `auth::validate-jwt`，它从 `state::get` 读取 JWKS。
7. Cron trigger 触发 `auth::rotate-jwks`，替换 state 中的 JWKS。
8. 下一次调用对新密钥验证，无需重启。
9. 对不同 MCP 资源的 confused-deputy 尝试获得 401，audience mismatch。

这里的 mock JWT 使用 HS256 和共享密钥（使本课仅在 stdlib 上运行）。生产使用 RS256 或 EdDSA 配合上面的 JWKS 模式；验证逻辑在其他方面相同。

## 交付产出

本课产出 `outputs/skill-mcp-auth-iii.md`。给定 MCP 服务器配置和 IdP 能力集，该技能发出要注册的 iii 原语、JWKS 轮换计划、scope 映射，以及当 IdP 不支持完整 RFC profile 时要应用的拒绝规则。

## 练习

1. 运行 `code/main.py`。追踪 9 步流程。注意 `state::get` 在 `auth::rotate-jwks` 覆盖之前返回过期数据的位置，以及下一个请求如何对新密钥验证。

2. 向 protected-resource metadata 的 `authorization_servers` 列表添加新 IdP。签发由新 IdP 签名的 token 并确认验证器接受。签发由未列出 IdP 签名的 token 并确认验证器以 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` 拒绝。

3. 将 `auth::rate-limit` 实现为 iii 函数，并在注册 HTTP trigger 内部、注册器运行之前调用它。使用保存在 `state::set("auth/ratelimit/<ip>", ...)` 中的每源 IP token-bucket。

4. 阅读 RFC 7591 并找出本课 `/register` handler 未验证的两个字段。添加验证。（提示：`software_statement` 和 `redirect_uris` URI scheme。）

5. 阅读 MCP spec 2025-11-25 授权部分。找出本课验证器当前未发出的一个关于 `WWW-Authenticate` 头的规范性要求。添加它。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| ASM | "OAuth 元数据文档" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| DCR | "自助客户端注册" | RFC 7591 `POST /register` 流程 |
| JWKS | "JWT 验证的公钥" | JSON Web Key Set，从 `jwks_uri` 获取，按 `kid` 索引 |
| Resource indicator | "Audience 参数" | RFC 8707 `resource` 参数将 token 固定到一个服务器 |
| `aud` claim | "Audience" | 验证器与规范资源 URL 比较的 JWT claim |
| Confused deputy | "Token 重放" | 为 Server A 签发的 token 被提交给 Server B 的攻击 |
| `iss` allow-list | "受信授权服务器" | protected-resource metadata 的 `authorization_servers` 中命名的集合 |
| Key rotation | "滚动 JWKS" | 带重叠窗口的签名密钥定期替换 |
| Public client | "原生或浏览器客户端" | 没有 `client_secret` 的 OAuth 客户端；PKCE 补偿 |
| `WWW-Authenticate` | "401/403 响应头" | 携带驱动客户端恢复的 `Bearer error=...` 指令 |

## 延伸阅读

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 本课实现的 MCP auth profile
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — 发现契约
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 公共客户端持有证明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience pinning
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 资源服务器发现
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 整合的 OAuth 基底
