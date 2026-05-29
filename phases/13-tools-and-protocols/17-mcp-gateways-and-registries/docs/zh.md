# MCP 网关与注册中心 — 企业控制平面

> 企业不能让每个开发者随意安装 MCP 服务器。网关集中化认证、RBAC、审计、速率限制、缓存和 tool poisoning 检测，然后将合并的工具面作为单一 MCP 端点暴露。Official MCP Registry（Anthropic + GitHub + PulseMCP + Microsoft，命名空间验证）是规范的上游。本课说明网关的位置、走通一个最小实现，并调研 2026 年的供应商格局。

**Type:** Learn
**Languages:** Python (stdlib, minimal gateway)
**Prerequisites:** Phase 13 · 15 (tool poisoning), Phase 13 · 16 (OAuth 2.1)
**Time:** ~45 minutes

## 学习目标

- 解释 MCP 网关的位置（在 MCP 客户端和多个后端 MCP 服务器之间）。
- 实现网关的五项职责：auth、RBAC、audit、rate limit、policy。
- 在网关层强制执行 pinned-tool-hash manifest。
- 区分 Official MCP Registry 和元注册中心（Glama、MCPMarket、MCP.so、Smithery、LobeHub）。

## 问题

一家财富 500 强公司有 30 个已批准的 MCP 服务器、5000 名开发者、合规和审计要求，以及一个想要集中策略的安全团队。让每个开发者在 IDE 中安装任意服务器是不可接受的。

网关模式：

1. 网关作为单一 Streamable HTTP 端点运行，开发者连接到它。
2. 网关持有每个后端 MCP 服务器的凭证。
3. 每个开发者请求通过网关自己的 OAuth 进行认证和作用域限定。
4. 网关将调用路由到后端服务器，应用策略。
5. 所有调用记录用于审计。

Cloudflare MCP Portals、Kong AI Gateway、IBM ContextForge、MintMCP、TrueFoundry、Envoy AI Gateway — 都在 2025-2026 年发布了网关或网关功能。

同时，Official MCP Registry 作为规范上游启动：经过策展、命名空间验证、反向 DNS 命名的服务器，网关可以从中拉取。元注册中心（Glama、MCPMarket、MCP.so、Smithery、LobeHub）聚合多个来源的服务器。

## 概念

### 网关的五项职责

1. **Auth。** OAuth 2.1 识别开发者；映射到用户角色。
2. **RBAC。** 每用户策略：哪些服务器、哪些工具、哪些 scope。
3. **Audit。** 每次调用记录 who、what、when、result。
4. **Rate limit。** 每用户 / 每工具 / 每服务器的上限防止滥用。
5. **Policy。** 拒绝毒化描述、强制 Rule of Two、脱敏 PII。

### 网关作为单一端点

对开发者来说，网关看起来像一个 MCP 服务器。内部它路由到 N 个后端。Session id（Phase 13 · 09）在边界处被重写。

### 凭证保管

开发者永远看不到后端 token。网关持有它们（或代理到持有它们的身份提供者）。在网关上有 `notes:read` 的开发者可以传递性地使用网关自己的后端凭证访问笔记 MCP 服务器 — 但仅在绑定传递性访问的策略下。

### 网关层的 Tool-hash pinning

网关持有已批准工具描述的 manifest（SHA256 哈希）。在发现时，它获取每个后端的 `tools/list`，将哈希与 manifest 比较，移除任何描述已变更的工具。这是 Phase 13 · 15 的 rug-pull 防御在中心化层面的应用。

### Policy-as-code

高级网关用 OPA/Rego、Kyverno 或 Styra 表达策略。像"用户 `alice` 只能在 org `acme` 的仓库上调用 `github.open_pr`"这样的规则以声明式编码。简单网关使用手写 Python。两种形式都有效。

### 会话感知路由

当用户的会话包含多个服务器时，网关进行多路复用：开发者的单个 MCP 会话持有 N 个后端会话，每个服务器一个。来自任何后端的通知通过网关路由到开发者的会话。

### 命名空间合并

网关合并所有后端的工具命名空间，通常在冲突时加前缀。`github.open_pr`、`notes.search`。这使路由无歧义。

### 注册中心

- **Official MCP Registry (`registry.modelcontextprotocol.io`)。** 在 Anthropic、GitHub、PulseMCP、Microsoft 管理下启动。命名空间验证（反向 DNS：`io.github.user/server`）。预过滤基本质量。
- **Glama。** 以搜索为中心的元注册中心，聚合多个来源。
- **MCPMarket。** 偏商业的目录，有供应商列表。
- **MCP.so。** 社区目录；开放提交。
- **Smithery。** 包管理器风格的安装流程。
- **LobeHub。** 在其 LobeChat 应用中集成的注册中心。

企业网关默认从 Official Registry 拉取，允许管理员从元注册中心策展添加，拒绝任何未 pin 的内容。

### 反向 DNS 命名

Official Registry 要求公共服务器使用反向 DNS 名称：`io.github.alice/notes`。命名空间防止抢注并使信任委托更清晰。

### 供应商调研，2026 年 4 月

| 供应商 | 优势 |
|--------|------|
| Cloudflare MCP Portals | 边缘托管；OAuth 集成；免费层 |
| Kong AI Gateway | K8s 原生；细粒度策略；日志到 OpenTelemetry |
| IBM ContextForge | 企业 IAM；合规；审计导出 |
| TrueFoundry | DevOps 导向；指标优先 |
| MintMCP | 开发者平台导向 |
| Envoy AI Gateway | 开源；可定制过滤器 |

Phase 17（生产基础设施）更深入地探讨网关运维。

## 动手实践

`code/main.py` 提供了一个约 150 行的最小网关：通过假 Bearer token 认证用户，持有每用户 RBAC 策略，将请求路由到两个后端 MCP 服务器，将每次调用写入审计日志，强制速率限制，并拒绝任何描述哈希与 pinned manifest 不匹配的后端工具。

关注点：

- `RBAC` 字典按 `user_id` 键控，包含允许的 `server_tool` 条目。
- `AUDIT_LOG` 是只追加的事件列表。
- 速率限制使用每用户的 token bucket。
- Pinned manifest 是 `server::tool -> hash` 的字典。

## 交付产出

本课产出 `outputs/skill-gateway-bootstrap.md`。给定一个企业 MCP 计划（用户、后端、合规），该技能产出网关配置规范。

## 练习

1. 运行 `code/main.py`。以允许的用户发起调用；然后以不允许的用户；然后超出速率限制的突发。验证三种流程。

2. 添加一个策略，在返回给客户端之前脱敏结果中的 PII。使用简单的正则匹配 SSN 格式的字符串；注意差距（邮箱、电话号码）。

3. 扩展审计日志以发出 OpenTelemetry GenAI span。Phase 13 · 20 涵盖了确切的属性。

4. 为一个 50 人开发团队设计 RBAC 策略，有五个后端（notes、github、postgres、jira、slack）。谁对每个有只读权限？谁有写权限？

5. 从头到尾阅读 Cloudflare enterprise MCP 帖子。找出 Cloudflare 提供而此 stdlib 网关没有的一个功能。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Gateway | "MCP 代理" | 客户端和后端之间的集中化服务器 |
| Credential vaulting | "后端 token 留在服务端" | 开发者永远看不到上游 token |
| Session-aware routing | "多后端会话" | 网关为每个开发者会话多路复用 N 个后端会话 |
| Tool-hash pinning | "已批准 manifest" | 每个已批准工具描述的 SHA256；集中阻止 rug-pulls |
| RBAC | "每用户策略" | 工具和服务器的基于角色的访问控制 |
| Policy-as-code | "声明式规则" | 在网关强制执行的 OPA/Rego、Kyverno、Styra 策略 |
| Audit log | "Who, what, when" | 用于合规的只追加事件日志 |
| Rate limit | "每用户 token bucket" | 防止滥用的每分钟上限 |
| Official MCP Registry | "规范上游" | `registry.modelcontextprotocol.io`，命名空间验证 |
| Reverse-DNS naming | "注册中心命名空间" | `io.github.user/server` 约定 |

## 延伸阅读

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — 规范上游，命名空间验证
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) — 带 OAuth 和策略的网关模式
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) — 开源参考网关
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — 功能对比文章
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) — IBM 的企业网关
