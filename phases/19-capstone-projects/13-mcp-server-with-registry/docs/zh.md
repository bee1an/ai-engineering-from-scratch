# 毕业项目 13 — MCP Server 与注册中心及治理

> Model Context Protocol 不再是未来，而是 2026 年默认的工具使用规范。Anthropic、OpenAI、Google 和每个主要 IDE 都发布 MCP 客户端。Pinterest 公开了其内部 MCP 服务器生态。AAIF Registry 在 `.well-known` 上形式化了能力元数据。AWS ECS 发布了参考无状态部署。Block 的 goose-agent 将同一协议放入托管助手。2026 年的生产形态是：StreamableHTTP 传输、OAuth 2.1 scope、OPA 策略门控，以及让平台团队发现、验证和启用服务器的注册中心。端到端构建它。

**类型：** 毕业项目
**语言：** Python（服务器，via FastMCP）或 TypeScript（@modelcontextprotocol/sdk），Go（注册中心服务）
**前置要求：** Phase 11（LLM 工程）、Phase 13（工具与 MCP）、Phase 14（智能体）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段：** P11 · P13 · P14 · P17 · P18
**时间：** 25 小时

## 问题

MCP 成为了工具使用的通用语言。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 和每个托管智能体现在都消费 MCP 服务器。生产挑战不在于编写服务器（FastMCP 让这很容易），而在于以企业要求大规模部署：每租户 OAuth scope、对破坏性工具的 OPA 策略、StreamableHTTP 无状态扩展、用于发现的注册中心、每工具调用的审计日志。Pinterest 的内部 MCP 生态和 AAIF Registry 规范设定了 2026 年的标准。

你将构建一个暴露 10 个内部工具（Postgres 只读、S3 列表、Jira、Linear、Datadog 等）的 MCP 服务器、一个用于平台发现的注册中心 UI，以及破坏性工具的人工审批门。负载测试演示 StreamableHTTP 水平扩展。审计轨迹满足企业安全审查。

## 概念

MCP 2026 修订版将 StreamableHTTP 作为默认传输。与早期的 stdio-and-SSE 形态不同，StreamableHTTP 默认无状态：单个 HTTP 端点接受 JSON-RPC 请求，流式响应，支持长连接用于通知。无状态意味着可以在负载均衡器后水平扩展。

授权是 OAuth 2.1 带每工具 scope。Token 携带如 `jira:read`、`s3:list`、`postgres:query:readonly` 的 scope。MCP 服务器在工具调用时检查 scope，而非仅在会话开始时。对高风险工具，服务器拒绝任何 scope 未在最近 N 分钟内提升到 `approved:by:human` 的调用——该提升来自 Slack 审查卡片。

注册中心是独立服务。每个 MCP 服务器暴露一个 `.well-known/mcp-capabilities` 文档，包含工具清单、传输 URL、认证要求。注册中心轮询、验证并索引。平台团队使用注册中心 UI 查看有哪些工具可用、需要什么 scope、哪个团队拥有它们。

## 架构

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## 技术栈

- 服务器框架：FastMCP（Python）或 `@modelcontextprotocol/sdk`（TypeScript）
- 传输：StreamableHTTP over HTTPS（无状态）
- 认证：OAuth 2.1 带工作负载身份 via SPIFFE / SPIRE
- 策略：OPA / Rego 规则每工具；每请求策略决策服务
- 注册中心：自托管，消费 `.well-known/mcp-capabilities` 清单
- 人工审批：Slack 交互式消息用于破坏性工具
- 部署：AWS ECS Fargate 或 Fly.io，每租户一个服务器或共享带租户范围
- 审计：结构化 JSONL 每租户 bucket 带每调用血缘

## 构建步骤

1. **工具面。** 暴露 10 个内部工具：Postgres 只读查询、S3 列表对象、Jira 搜索/获取、Linear 搜索/获取、Datadog 指标查询、PagerDuty 值班查询、GitHub 只读、Notion 搜索、Slack 搜索、Salesforce 读取。每个工具有类型化 schema 和 scope 标签。

2. **FastMCP 服务器。** 挂载工具。配置 StreamableHTTP 传输。添加 OAuth token 内省和 scope 强制的中间件。

3. **OPA 策略。** 每工具 Rego 策略：什么 scope 允许调用、什么 PII 脱敏适用、什么 payload 大小上限适用。每次工具调用时调用决策服务。

4. **注册中心服务。** 独立 Go 或 TS 服务，轮询注册服务器的 `.well-known/mcp-capabilities`，用 JSON Schema 验证，暴露列表/搜索/验证/启用-禁用 UI。

5. **能力清单。** 每个服务器暴露 `.well-known/mcp-capabilities`，包含：工具列表、认证要求、传输 URL、所有者团队、SLO。

6. **破坏性工具分离。** 变更状态的工具（Jira 创建、Linear 创建、Postgres 写入）在第二个 MCP 服务器上，有更严格的认证流程：token 必须有在 15 分钟内通过 Slack 卡片提升的 `approved:by:human` scope。

7. **审计日志。** 每租户 append-only JSONL：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写入前通过 Presidio 进行 PII 脱敏。

8. **负载测试。** 100 个并发客户端在 StreamableHTTP 上。通过添加第二个副本演示水平扩展；展示负载均衡器在无会话粘性的情况下重新分配。

9. **一致性测试。** 对两个服务器运行官方 MCP 一致性套件。通过所有必需部分。

## 使用示例

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## 交付标准

`outputs/skill-mcp-server.md` 描述交付物。一个生产级 MCP 服务器 + 注册中心 + 审计层，用于内部工具，带 OAuth 2.1 scope 和 OPA 门控。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 规范一致性 | StreamableHTTP + 能力清单通过 MCP 一致性测试 |
| 20 | 安全性 | Scope 强制、OPA 覆盖每个工具、密钥卫生 |
| 20 | 可观测性 | 每工具调用审计日志带 PII 脱敏 |
| 20 | 规模 | 100 客户端负载测试水平扩展演示 |
| 15 | 注册中心体验 | 发现/验证/启用-禁用工作流 |
| **100** | | |

## 练习

1. 添加新工具（Confluence 搜索）。通过注册中心验证流程发布，不触碰核心服务器。

2. 编写 OPA 策略，脱敏包含名为 `email`、`ssn` 或 `phone` 列的 Postgres 查询结果。用探测查询验证。

3. 基准测试 StreamableHTTP vs stdio 的本地延迟。报告每调用 p50/p95。

4. 实现每租户配额：每租户每工具每分钟最大 N 次调用。通过第二条 OPA 规则强制执行。

5. 从 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 运行 MCP 一致性套件并修复每个失败。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| StreamableHTTP | "2026 MCP 传输" | 无状态 HTTP + 流式；取代 SSE + stdio 用于网络服务器 |
| 能力清单 | "Well-known 文档" | `.well-known/mcp-capabilities` 带工具列表、认证、传输 URL |
| OPA / Rego | "策略引擎" | Open Policy Agent，用于对外部规则授权工具调用 |
| Scope 提升 | "Approved-by-human" | 通过 Slack 审批授予的短期 scope，破坏性工具必需 |
| 注册中心 | "工具发现" | 从能力清单索引 MCP 服务器的服务 |
| 工作负载身份 | "SPIFFE / SPIRE" | 用于 OAuth token 签发的加密服务身份 |
| 一致性套件 | "规范测试" | 官方 MCP 测试集，用于 StreamableHTTP + 工具清单正确性 |

## 延伸阅读

- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据、注册中心
- [AAIF MCP Registry spec](https://github.com/modelcontextprotocol/registry) — 2026 注册中心规范
- [AWS ECS reference deployment](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — 参考生产部署
- [Pinterest internal MCP ecosystem](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — 参考内部部署
- [Block `goose` MCP usage](https://block.github.io/goose/) — 参考智能体消费模式
- [FastMCP](https://github.com/jlowin/fastmcp) — Python 服务器框架
- [Open Policy Agent](https://www.openpolicyagent.org/) — 策略引擎参考
- [SPIFFE / SPIRE](https://spiffe.io) — 工作负载身份参考
