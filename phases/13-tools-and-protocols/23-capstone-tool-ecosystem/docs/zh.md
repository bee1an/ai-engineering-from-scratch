# Capstone — 构建完整的工具生态系统

> Phase 13 教了每个组件。本 capstone 将它们连接成一个生产级系统：一个带 tools + resources + prompts + tasks + UI 的 MCP server，边缘的 OAuth 2.1，一个 RBAC 网关，一个多服务器客户端，一次 A2A 子 agent 调用，OTel tracing 到 collector，CI 中的 tool-poisoning 检测，以及 AGENTS.md + SKILL.md 包。完成后你能为每个架构决策辩护。

**Type:** Build
**Languages:** Python (stdlib, end-to-end ecosystem harness)
**Prerequisites:** Phase 13 · 01 through 21
**Time:** ~120 minutes

## 学习目标

- 组合一个暴露 tools、resources、prompts 和带 `ui://` app 的 task 的 MCP server。
- 在 server 前面放置一个执行 RBAC 和 pinned hashes 的 OAuth 2.1 网关。
- 编写一个端到端使用 OTel GenAI attributes 进行 tracing 的多服务器客户端。
- 将部分工作负载委托给 A2A 子 agent；验证不透明性得到保持。
- 用 AGENTS.md + SKILL.md 打包整个栈，使其他 agent 能驱动它。

## 问题

交付"研究与报告"系统：

- 用户提问："总结 2026 年 arXiv 上关于 agent 协议的三篇被引最多的论文。"
- 系统：通过 MCP 搜索 arXiv；通过 A2A 将论文摘要委托给专门的 writer agent；聚合结果；将交互式报告渲染为 MCP Apps `ui://` resource；将每一步记录到 OTel。

Phase 13 的所有原语都出现了。这不是玩具——2026 年 Anthropic（Claude Research 产品）、OpenAI（GPTs with Apps SDK）和第三方发布的生产级研究助手系统正是这个形态。

## 概念

### 架构

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### Trace 层级

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

一个 trace id。每个 span 都有正确的 `gen_ai.*` attributes。

### 安全态势

- OAuth 2.1 + PKCE，resource indicator 将 audience 固定到网关。
- 网关持有上游凭证；用户永远看不到。
- RBAC：`alice` 有 `research:read`、`research:write`，可调用所有 tools。`bob` 有 `research:read`，不能调用 `generate_report`。
- Pinned description manifest：丢弃任何 tool hash 变化的 server。
- Rule of Two 审计：没有 tool 同时组合不可信输入、敏感数据和有后果的操作。

### 渲染

最终的 `generate_report` task 返回 content blocks 加一个 `ui://report/current` resource。客户端的 host（Claude Desktop 等）在沙箱 iframe 中渲染交互式仪表板。仪表板包含排序的论文列表、引用计数，以及一个按钮，用户点击时调用 `host.callTool('summarize_paper', {arxiv_id})`。

### 打包

整个系统以如下形式发布：

```
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

用户通过 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户通过调用 `run-research` skill 来驱动系统。

### Phase 13 每课的贡献

| 课程 | Capstone 使用的内容 |
|--------|------------------------|
| 01-05 | Tool 接口、供应商可移植性、并行调用、schema、linting |
| 06-10 | MCP 原语、server、client、transports、resources + prompts |
| 11-14 | Sampling、roots + elicitation、异步 tasks、`ui://` apps |
| 15-17 | Tool poisoning、OAuth 2.1、gateway + registry |
| 18 | A2A 子 agent 委托 |
| 19 | OTel GenAI tracing |
| 20 | LLM 层的路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## Use It

`code/main.py` 将前面课程的模式拼接成一个可运行的 demo。全部 stdlib，全部进程内，可以端到端阅读。它运行研究与报告场景的完整流程：与网关握手、模拟 OAuth 2.1、tools/list 合并、generate_report 作为 task、A2A 调用 writer、返回 ui:// resource、发出 OTel spans。

关注点：

- 每一跳共享一个 trace id。
- 网关策略阻止第二个用户写入。
- Task 生命周期从 working → completed，返回文本和 ui:// 内容。
- A2A 调用的内部状态对编排器不透明。
- AGENTS.md 和 SKILL.md 是另一个 agent 复现工作流所需的唯一文件。

## Ship It

本课产出 `outputs/skill-ecosystem-blueprint.md`。给定产品需求（研究、摘要、自动化），该 skill 生成完整架构：哪些 MCP 原语、哪些网关控制、哪些 A2A 调用、哪些遥测、哪种打包方式。

## 练习

1. 运行 `code/main.py`。注意单一 trace id 和 spans 的嵌套方式。数一数 demo 触及了 Phase 13 的多少个原语。

2. 扩展 demo：添加第二个后端 MCP server（如 `bibliography`），确认网关将其 tools 合并到同一命名空间。

3. 用一个运行在子进程中的真实 agent 替换假的 A2A writer agent。使用 Lesson 19 的 harness。

4. 在路由网关中添加 PII 脱敏步骤，位于编排器和 LLM 之间。确认用户查询中的邮箱被清洗。

5. 为将维护此系统的队友编写一个 AGENTS.md。阅读时间应在五分钟以内，并提供他们在 Cursor 或 Codex 中驱动 capstone 所需的一切。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Capstone | "Phase-13 集成 demo" | 使用每个原语的端到端系统 |
| Research and report | "场景" | 搜索、摘要、渲染模式 |
| Ecosystem | "所有组件组合" | Server + client + gateway + 子 agent + 遥测 + 打包 |
| Trace hierarchy | "单一 trace id" | 每一跳的 span 共享 trace；通过 span id 建立父子关系 |
| Gateway-issued token | "传递式认证" | 客户端只看到网关的 token；网关持有上游凭证 |
| Merged namespace | "所有 tools 在一个扁平列表" | 网关处多服务器合并，冲突时加前缀 |
| Opacity boundary | "A2A 调用隐藏内部" | 子 agent 的推理对编排器不可见 |
| Three-layer stack | "AGENTS.md + SKILL.md + MCP" | 项目上下文 + 工作流 + 工具 |
| Defense-in-depth | "多层安全" | Pinned hashes、OAuth、RBAC、Rule of Two、审计日志 |
| Spec compliance matrix | "我们交付的 vs 规范要求的" | 将交付物映射到 2025-11-25 要求的清单 |

## 延伸阅读

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 统一参考
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议发展方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范 tracing 约定
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — 生产级 agent 运行时模式
