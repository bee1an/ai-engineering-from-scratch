# A2A — Agent-to-Agent Protocol

> MCP 是 agent 到工具。A2A（Agent2Agent）是 agent 到 agent — 一个让基于不同框架构建的不透明 agent 协作的开放协议。由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux Foundation，2026 年 4 月达到 v1.0，拥有 150+ 支持者，包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。它吸收了 IBM 的 ACP 并添加了 AP2 支付扩展。本课讲解 Agent Card、Task 生命周期和两种传输绑定。

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## 学习目标

- 区分 agent-to-tool（MCP）和 agent-to-agent（A2A）的用例。
- 在 `/.well-known/agent.json` 发布 Agent Card，包含 skills 和端点元数据。
- 走通 Task 生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用带 Parts（text、file、data）的 Messages 和作为输出的 Artifacts。

## 问题

一个客服 agent 需要将报告撰写委托给专门的写作 agent。A2A 之前的选项：

- 自定义 REST API。可行但每对组合都是一次性的。
- 共享代码库。要求两个 agent 运行相同框架。
- MCP。不适合：MCP 用于调用工具，不适合两个 agent 在保持各自不透明内部推理的同时协作。

A2A 填补了这个空白。它将交互建模为一个 agent 向另一个发送 Task，带有生命周期、消息和产物。被调用 agent 的内部状态保持不透明 — 调用者只看到 task 状态转换和最终输出。

A2A 是"让跨框架的 agent 互相对话"的协议。它不替代 MCP；两者互补。

## 概念

### Agent Card

每个 A2A 兼容的 agent 在 `/.well-known/agent.json` 发布一张卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现是基于 URL 的：获取卡片，了解 A2A 端点的 URL，枚举 skills。

### 签名 Agent Cards（AP2）

AP2 扩展（2025 年 9 月）为 Agent Cards 添加了加密签名。发布者用 JWT 签名自己的卡片；消费者验证。防止冒充。

### Task 生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

客户端通过 `tasks/send` 发起。被调用 agent 在状态间转换；客户端通过 SSE 订阅状态更新或轮询。

### Messages 和 Parts

一条消息携带一个或多个 Parts：

- `text` — 纯文本内容。
- `file` — 带 mimeType 的 base64 blob。
- `data` — 类型化 JSON payload（给被调用 agent 的结构化输入）。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

输出是 Artifacts，不是原始字符串。Artifact 是命名的、类型化的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以作为块流式传输。调用者累积。

### 两种传输绑定

1. **JSON-RPC over HTTP。** `/a2a` 端点，POST 用于请求，可选 SSE 用于流式传输。默认绑定。
2. **gRPC。** 用于 gRPC 是原生的企业环境。

两种绑定携带相同的逻辑消息形状。

### 不透明性保持

一个关键设计原则：被调用 agent 的内部状态是不透明的。调用者看到 task 状态和 artifacts。被调用 agent 的思维链、它的工具调用、它的子 agent 委托 — 全部不可见。这与 MCP 不同，MCP 中工具调用是透明的。

原理：A2A 使竞争对手能在不暴露内部的情况下协作。A2A 可以是"调用这个客服 agent"而调用者不了解该 agent 如何实现服务。

### 时间线

- **2025-04-09。** Google 宣布 A2A。
- **2025-06-23。** 捐赠给 Linux Foundation。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（Agent Payments）发布。
- **2026-04。** v1.0 发布，150+ 支持组织。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|------|-----|-----|
| 用例 | Agent-to-tool | Agent-to-agent |
| 不透明性 | 透明的工具调用 | 不透明的内部推理 |
| 典型调用者 | Agent 运行时 | 另一个 agent |
| 状态 | 工具调用结果 | 带生命周期的 Task |
| 授权 | OAuth 2.1 (Phase 13 · 16) | JWT 签名的 Agent Cards (AP2) |
| 传输 | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

当你想调用特定工具时使用 MCP。当你想将整个任务委托给另一个 agent 时使用 A2A。许多生产系统同时使用两者：agent 使用 MCP 作为工具层，使用 A2A 作为协作层。

## 动手实践

`code/main.py` 实现了一个最小的 A2A harness：一个研究 agent 发布其卡片，一个写作 agent 接收带有 PDF 和文本指令的 `tasks/send`，经过 working → input_required → working → completed 的转换，返回文本 artifact。全部 stdlib；使用内存传输以聚焦消息形状。

关注点：

- Agent Card JSON 形状。
- Task id 分配和状态转换。
- 带混合类型 parts 的 Messages。
- Task 中途的 input-required 分支。
- 完成时的 Artifact 返回。

## 交付产出

本课产出 `outputs/skill-a2a-agent-spec.md`。给定一个应该可被其他 agent 调用的新 agent，该技能产出 Agent Card JSON、skills schema 和端点蓝图。

## 练习

1. 运行 `code/main.py`。追踪完整的 Task 生命周期，包括被调用 agent 请求澄清的 input-required 暂停。

2. 添加签名 Agent Card。用 HMAC 对卡片的规范 JSON 签名。编写验证器并确认在卡片被篡改时失败。

3. 实现 task 流式传输：写作 agent 通过 SSE 发出三个增量 artifact 块，调用者累积它们。

4. 设计一个包装 MCP 服务器的 A2A agent。将每个 MCP 工具映射到 A2A skill。注意权衡 — 失去了什么不透明性？

5. 阅读 A2A v1.0 公告，找出截至 2026 年 4 月尚未被任何框架实现的一个功能。（提示：与多跳 task 委托有关。）

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| A2A | "Agent-to-Agent 协议" | 用于不透明 agent 协作的开放协议 |
| Agent Card | "`.well-known/agent.json`" | 描述 agent 的 skills 和端点的已发布元数据 |
| Skill | "可调用单元" | agent 支持的命名操作（类似 MCP tool） |
| Task | "委托单元" | 带生命周期和最终 artifact 的工作项 |
| Message | "Task 输入" | 携带 Parts（text、file、data） |
| Part | "类型化块" | 消息的 `text` / `file` / `data` 元素 |
| Artifact | "Task 输出" | 完成时返回的命名、类型化输出 |
| AP2 | "Agent Payments Protocol" | 用于信任和支付的签名 Agent Cards 扩展 |
| Opacity | "黑盒协作" | 被调用 agent 的内部对调用者隐藏 |
| Input-required | "Task 暂停" | agent 需要更多信息时的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 规范的 A2A 规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月治理转移
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路线图和合作伙伴动力
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 发布说明和向后兼容指南
