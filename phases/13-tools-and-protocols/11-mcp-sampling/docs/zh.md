# MCP Sampling — 服务端请求 LLM 补全与 Agent 循环

> 大多数 MCP 服务器只是"哑执行器"：接收参数、运行代码、返回内容。Sampling 让服务器反转方向：它请求客户端的 LLM 来做决策。这使得服务器可以托管 agent 循环而无需拥有任何模型凭证。SEP-1577 于 2025-11-25 合并，在 sampling 请求中加入了 tools，使循环可以包含更深层的推理。漂移风险提示：SEP-1577 的 tool-in-sampling 结构在 2026 Q1 之前仍处于实验阶段，SDK API 仍在调整中。

**Type:** Build
**Languages:** Python (stdlib, sampling harness)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 minutes

## 学习目标

- 解释 `sampling/createMessage` 解决了什么问题（无需服务端 API key 的服务器托管循环）。
- 实现一个服务器，请求客户端对多轮 prompt 进行 sample 并返回补全结果。
- 使用 `modelPreferences`（cost / speed / intelligence 优先级）来引导客户端的模型选择。
- 构建一个 `summarize_repo` 工具，内部通过 sampling 迭代而非硬编码行为。

## 问题

一个有用的代码摘要 MCP 服务器需要：遍历文件树、选择要读取的文件、合成摘要并返回。LLM 推理发生在哪里？

方案 A：服务器调用自己的 LLM。需要 API key，服务端计费，每用户成本高。

方案 B：服务器返回原始内容；客户端的 agent 做推理。可行但把服务器逻辑移到了客户端 prompt 中，很脆弱。

方案 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法（读哪些文件、做几轮），客户端保留计费和模型选择。服务器完全没有凭证。

Sampling 就是方案 C。它是受信任的服务器在不成为完整 LLM 宿主的情况下托管 agent 循环的机制。

## 概念

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行其 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个浮点数之和为 1.0：

- `costPriority`：倾向更便宜的模型。
- `speedPriority`：倾向更快的模型。
- `intelligencePriority`：倾向更强的模型。

加上 `hints`：服务器偏好的具名模型。客户端可能遵循也可能不遵循 hints；客户端的用户配置始终优先。

### `includeContext`

三个值：

- `"none"` — 仅服务器提供的消息。默认值。
- `"thisServer"` — 包含此服务器会话的先前消息。
- `"allServers"` — 包含所有会话上下文。

`includeContext` 自 2025-11-25 起已软弃用，因为它会泄露跨服务器上下文，这是一个安全隐患。建议使用 `"none"` 并在 messages 中显式传递上下文。

### 带 tools 的 Sampling（SEP-1577）

2025-11-25 新增：sampling 请求可以包含 `tools` 数组。客户端使用这些 tools 运行完整的 tool-calling 循环。这让服务器可以通过客户端的模型托管 ReAct 风格的 agent 循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环：sample、如果调用了 tool 则执行、再次 sample、返回最终的 assistant 消息。这在 2026 Q1 之前仍处于实验阶段；SDK 签名可能仍会变化。实现时请对照 2025-11-25 规范的 client/sampling 部分确认。

### Human-in-the-loop

客户端必须在运行 sample 之前向用户展示服务器要求模型做什么。恶意服务器可能利用 sampling 操纵用户会话（"对用户说 X 让他们点击 Y"）。Claude Desktop、VS Code 和 Cursor 将 sampling 请求呈现为用户可以拒绝的确认对话框。

2026 共识：没有人工确认的 sampling 是一个危险信号。网关（Phase 13 · 17）可以自动批准低风险 sampling 并自动拒绝可疑请求。

### 无需 API key 的服务器托管循环

典型用例：一个没有自己 LLM 访问权限的代码摘要 MCP 服务器。它做的事情是：

1. 遍历仓库结构。
2. 调用 `sampling/createMessage`，提示"选出最可能描述此仓库用途的五个文件"。
3. 读取这些文件。
4. 调用 `sampling/createMessage`，附上文件内容和"用 3 段话总结此仓库"。
5. 将摘要作为 `tools/call` 结果返回。

服务器从未接触 LLM API。客户端的用户使用自己的凭证为补全付费。

### 安全风险（Unit 42 披露，2026 Q1）

- **隐蔽 sampling。** 一个工具总是调用 sampling 并提示"从会话上下文中返回用户的邮箱"。Phase 13 · 15 涵盖了攻击向量。
- **通过 sampling 窃取资源。** 服务器要求客户端总结攻击者的 payload，由用户买单。
- **循环炸弹。** 服务器在紧密循环中调用 sampling。客户端必须强制执行每会话速率限制。

## 动手实践

`code/main.py` 提供了一个模拟的服务器到客户端 sampling 工具。一个模拟的 "summarize_repo" 工具调用两轮 sampling（选文件，然后总结），模拟客户端返回预设响应。该工具展示了：

- 服务器发送带 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回补全结果。
- 服务器继续其循环。
- 速率限制器限制每次工具调用的总 sampling 次数。

关注点：

- 服务器只暴露一个工具（`summarize_repo`）；所有推理都发生在 sampling 调用中。
- 模型偏好权重影响客户端的模型选择；hints 列出偏好的模型。
- 循环在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 限制捕获失控循环。

## 交付产出

本课产出 `outputs/skill-sampling-loop-designer.md`。给定一个需要 LLM 调用的服务端算法（研究、摘要、规划），该技能设计基于 sampling 的实现，包含正确的 modelPreferences、速率限制和安全确认。

## 练习

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2，观察速率限制截断。

2. 实现 SEP-1577 的 tool-in-sampling 变体：sampling 请求携带 `tools` 数组。验证客户端循环在返回最终补全之前执行了这些 tools。注意漂移风险：SDK 签名在 2026 H1 之前可能仍会变化。

3. 添加 human-in-the-loop 确认：在服务器的第一次 `sampling/createMessage` 之前暂停并等待用户批准。被拒绝的调用返回类型化的拒绝响应。

4. 添加按客户端会话键控的每用户速率限制器。同一服务器同一用户的循环应共享预算。

5. 设计一个 `summarize_pdf` 工具，使用 sampling 来选择要包含的块。草拟发送的消息。`modelPreferences.intelligencePriority` 在 0.1 和 0.9 时行为有何不同？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Sampling | "服务器到客户端的 LLM 调用" | 服务器请求客户端的模型生成补全 |
| `sampling/createMessage` | "那个方法" | 用于 sampling 请求的 JSON-RPC 方法 |
| `modelPreferences` | "模型优先级" | Cost / speed / intelligence 权重加名称提示 |
| `includeContext` | "跨会话泄露" | 已软弃用的上下文包含模式 |
| SEP-1577 | "Sampling 中的 tools" | 允许在 sampling 中使用 tools 以实现服务器托管的 ReAct |
| Human-in-the-loop | "用户确认" | 客户端在运行前向用户展示 sampling 请求 |
| Loop bomb | "失控 sampling" | 服务端无限 sampling 循环；客户端必须限速 |
| Covert sampling | "隐藏推理" | 恶意服务器在 sampling prompt 中隐藏意图 |
| Resource theft | "使用用户的 LLM 预算" | 服务器强制客户端在不需要的 sampling 上花费 |
| `stopReason` | "生成停止原因" | `endTurn`、`stopSequence` 或 `maxTokens` |

## 延伸阅读

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — sampling 高层概述
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — 规范的 `sampling/createMessage` 结构
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — Sampling 中 tools 的 Spec Evolution Proposal（实验性）
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 隐蔽 sampling 和资源窃取模式
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) — 带客户端代码示例的详解
