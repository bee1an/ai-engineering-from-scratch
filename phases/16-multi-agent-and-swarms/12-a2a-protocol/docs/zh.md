# A2A — Agent-to-Agent 协议

> Google 在 2025 年 4 月宣布 A2A；到 2026 年 4 月规范在 https://a2a-protocol.org/latest/specification/，150+ 组织支持。A2A 是 MCP（Lesson 13）的水平补充：MCP 是垂直的（agent ↔ 工具），A2A 是点对点的（agent ↔ agent）。它定义了 Agent Card（发现）、带制品的任务（文本、结构化数据、视频）、不透明任务生命周期和认证。生产系统越来越多地将 MCP 与 A2A 配对。Google Cloud 在 2025-2026 年间将 A2A 支持集成到 Vertex AI Agent Builder。

**Type:** Learn + Build
**Languages:** Python (stdlib, `http.server`, `json`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## 问题

你的 agent 需要调用另一个系统上的另一个 agent。怎么做？你可以暴露一个 HTTP 端点，定义一个定制 JSON schema，然后希望对方也说同样的语言。每对 agent 都变成一个定制集成。

A2A 是该调用的通用线路协议。标准发现、标准任务模型、标准传输、标准制品。就像 HTTP+REST 但以 agent 为一等公民。

## 概念

### 四个要素

**Agent Card。** 一个位于 `/.well-known/agent.json` 的 JSON 文档，描述 agent：名称、技能、端点、支持的模态、认证要求。发现通过读取 card 进行。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task。** 工作单元。一个异步、有状态的对象，具有生命周期：`submitted → working → completed / failed / canceled`。客户端发送任务，轮询或订阅更新。

**Artifact。** 任务产出的结果类型。文本、结构化 JSON、图像、视频、音频。制品是类型化的，所以不同模态是一等公民。

**不透明生命周期。** A2A 不规定远程 agent *如何*解决任务。客户端看到状态转换和制品；实现可以自由使用任何框架。

### MCP/A2A 分工

- **MCP**（Lesson 13）：agent ↔ 工具。Agent 通过 JSON-RPC 读写工具服务器。默认无状态。
- **A2A**：agent ↔ agent。对等协议；双方都是有自己推理的 agent。

生产多 agent 系统两者都用。A2A 对等方在自己这边调用 MCP 工具。分工保持两个关注点干净。

### 发现流程

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或使用流式：SSE 订阅 `/tasks/{id}/events` 获取推送更新。

### 认证

A2A 支持三种常见模式：

- **Bearer token** — OAuth2 或不透明。
- **mTLS** — 双向 TLS；组织互相证明身份。
- **签名请求** — 对有效载荷的 HMAC。

认证在 Agent Card 中声明；客户端发现并遵守。

### 到 2026 年 4 月 150+ 组织

企业采用推动了 A2A 规模。标题：A2A 成为企业 agent 系统跨越信任边界的方式。Google Cloud 发布了 Vertex AI Agent Builder A2A 支持；Microsoft Agent Framework 支持它；大多数主要框架（LangGraph、CrewAI、AutoGen）发布 A2A 适配器。

### A2A 优势

- **跨组织调用。** 公司 A 的 agent 调用公司 B 的 agent。没有 A2A，每对都是定制契约。
- **异构框架。** LangGraph agent 调用 CrewAI agent 调用自定义 Python agent。A2A 标准化。
- **类型化制品。** 视频结果、结构化 JSON、音频——都是一等公民。
- **长时间运行任务。** 不透明生命周期 + 轮询使数小时的任务变得简单。

### A2A 困难场景

- **延迟敏感的微调用。** A2A 的生命周期是异步的。亚毫秒 agent 到 agent 不适合；使用直接 RPC。
- **紧耦合进程内 agent。** 如果两个 agent 在同一个 Python 进程中运行，A2A 的 HTTP 往返是过度的。
- **小团队。** 规范开销是真实的；仅内部 agent 可能不需要这种正式性。

### A2A vs ACP、ANP、NLIP

2024-2026 年出现了几个相关规范：

- **ACP**（IBM/Linux Foundation）— A2A 的前身，范围更窄。
- **ANP**（Agent Network Protocol）— 重对等发现，去中心化优先。
- **NLIP**（Ecma Natural Language Interaction Protocol，2025 年 12 月标准化）— 自然语言内容类型。

截至 2026 年 4 月，A2A 是最被采用的对等协议。见 arXiv:2505.02279（Liu 等人，"A Survey of Agent Interoperability Protocols"）的比较。

## Build It

`code/main.py` 使用 `http.server` 和 JSON 实现了一个 A2A 最小服务器和客户端。服务器：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理任务状态，
- 在 `GET /tasks/{id}` 返回制品。

客户端：

- 获取 Agent Card，
- 提交任务，
- 轮询直到完成，
- 读取制品。

运行：

```
python3 code/main.py
```

脚本在后台线程启动服务器，然后对其运行客户端。你看到完整流程：发现、提交、轮询、制品。

## Use It

`outputs/skill-a2a-integrator.md` 设计 A2A 集成：Agent Card 内容、任务 schema、认证选择、流式 vs 轮询。

## Ship It

检查清单：

- **固定规范版本。** A2A 仍在演进；Agent Card 应声明协议版本。
- **幂等任务创建。** 重复提交（网络重试）应产出一个任务。
- **制品 schema。** 声明 agent 返回什么形状；消费者应验证。
- **速率限制 + 认证。** A2A 是面向公众的；应用标准 web 安全。
- **失败任务死信。** 随时间检查模式以发现重复失败类型。

## 练习

1. 运行 `code/main.py`。确认客户端发现服务器并接收正确制品。
2. 向服务器添加第二个技能（例如"summarize"）。更新 Agent Card。编写一个根据任务类型选择技能的客户端。
3. 实现 SSE 流式端点：`/tasks/{id}/events` 发出状态变化。客户端需要做什么不同？
4. 阅读 A2A 规范 (https://a2a-protocol.org/latest/specification/)。识别规范要求但此演示未实现的三件事。
5. 比较 A2A（Agent Card 发现）和 MCP（服务器端通过 `listTools` 的能力列表）。自描述 agent 和能力探测之间的权衡是什么？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| A2A | "Agent 到 agent" | 跨系统 agent 调用其他 agent 的对等协议。Google 2025。 |
| Agent Card | "Agent 的名片" | 位于 `/.well-known/agent.json` 的 JSON，描述技能、端点、认证。 |
| Task | "工作单元" | 带生命周期的异步有状态对象；完成时产出制品。 |
| Artifact | "结果" | 类型化输出：文本、结构化 JSON、图像、视频、音频。一等媒体。 |
| 不透明生命周期 | "怎么解决是 agent 的事" | 客户端看到状态转换；服务器可以自由选择框架/工具。 |
| 发现 | "找到 agent" | `GET /.well-known/agent.json` 返回 card。 |
| MCP vs A2A | "工具 vs 对等" | MCP：垂直 agent ↔ 工具。A2A：水平 agent ↔ agent。 |
| ACP / ANP / NLIP | "兄弟协议" | 相邻规范；A2A 是 2026 年最被采用的。 |

## 延伸阅读

- [A2A specification](https://a2a-protocol.org/latest/specification/) — 规范规范
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月发布博文
- [A2A GitHub repo](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP 比较
