# Claude Agent SDK：子智能体与 Session Store

> Claude Agent SDK 是 Claude Code harness 的库形式。内置工具、用于上下文隔离的子智能体、hooks、W3C trace 传播、session store 对等。Claude Managed Agents 是用于长时间异步工作的托管替代方案。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time:** ~75 minutes

## 学习目标

- 解释 Anthropic Client SDK（原始 API）和 Claude Agent SDK（harness 形态）之间的区别。
- 描述子智能体——并行化和上下文隔离——以及何时使用它们。
- 列举 Python SDK 的 session store 接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 用 stdlib 实现一个带内置工具、隔离上下文的子智能体生成、生命周期 hooks 和 session store 的 harness。

## 问题

原始 LLM API 给你一次往返。生产智能体需要工具执行、MCP 服务器、生命周期 hooks、子智能体生成、会话持久化、trace 传播。Claude Agent SDK 将这种形态作为库发布——Claude Code 使用的同一个 harness，暴露给自定义智能体。

## 概念

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`)。** 原始 Messages API。你掌控循环、工具、状态。
- **Agent SDK (`claude-agent-sdk`)。** 内置工具执行、MCP 连接、hooks、子智能体生成、session store。Claude Code 循环作为库。

### 内置工具

SDK 开箱即用提供 10+ 工具：文件读写、shell、grep、glob、web fetch 等。自定义工具通过标准 tool-schema 接口注册。

### 子智能体

Anthropic 文档记录的两个用途：

1. **并行化。** 并发运行独立工作。"为这 20 个模块中的每一个找到测试文件"是 20 个并行子智能体任务。
2. **上下文隔离。** 子智能体使用自己的上下文窗口；只有结果返回给编排器。编排器的预算得以保留。

Python SDK 近期新增：`list_subagents()`、`get_subagent_messages()` 用于读取子智能体转录。

### Session store

与 TypeScript 的协议对等：

- `append(session_id, message)` — 添加一轮。
- `load(session_id)` — 恢复对话。
- `list_sessions()` — 枚举。
- `delete(session_id)` — 级联删除子智能体会话。
- `list_subkeys(session_id)` — 列出子智能体键。

`--session-mirror`（CLI 标志）在流式传输时将转录镜像到外部文件，用于调试。

### Hooks

你可以注册的生命周期 hooks：

- `PreToolUse`、`PostToolUse` — 门控或审计工具调用。
- `SessionStart`、`SessionEnd` — 设置和拆除。
- `UserPromptSubmit` — 在模型看到用户输入之前对其执行操作。
- `PreCompact` — 在上下文压缩前运行。
- `Stop` — 智能体退出时清理。
- `Notification` — 侧信道告警。

Hooks 是 pro-workflow（Phase 14 课程参考）和类似系统添加横切行为的方式。

### W3C trace context

调用者上活跃的 OTel span 通过 W3C trace context headers 传播到 CLI 子进程中。整个多进程 trace 在你的后端显示为一个 trace。

### Claude Managed Agents

托管替代方案（beta header `managed-agents-2026-04-01`）。长时间异步工作、内置 prompt caching、内置压缩。用控制权换取托管基础设施。

### 这种模式出错的地方

- **子智能体过度生成。** 为 100 个小任务生成 100 个子智能体。开销占主导。改用批处理。
- **Hook 蔓延。** 每个团队都添加 hooks；启动时间膨胀。每季度审查 hooks。
- **Session 膨胀。** Session 累积；大小增长。使用 `list_sessions` + 过期策略。

## Build It

`code/main.py` 用 stdlib 实现了 SDK 的形状：

- `Tool`、`ToolRegistry` 带内置 `read_file`、`write_file`、`list_dir`。
- `Subagent` — 私有上下文、隔离运行、结果返回。
- `SessionStore` — append、load、list、delete、list_subkeys。
- `Hooks` — `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 演示：主智能体并行生成 3 个子智能体（各自隔离），聚合结果，持久化 session。

运行：

```
python3 code/main.py
```

轨迹展示子智能体上下文隔离（编排器上下文大小保持有界）、hook 执行和 session 持久化。

## Use It

- **Claude Agent SDK** 用于想要 Claude Code harness 形态的 Claude 优先产品。
- **Claude Managed Agents** 用于托管的长时间异步工作。
- **OpenAI Agents SDK**（Lesson 16）用于 OpenAI 优先的对应物。
- **LangGraph + 自定义工具** 如果你想要图形状态机。

## Ship It

`outputs/skill-claude-agent-scaffold.md` 搭建一个 Claude Agent SDK 应用，带子智能体、hooks、session store、MCP 服务器附加和 W3C trace 传播。

## 练习

1. 添加一个子智能体生成器，将 20 个任务分批为 5 个并行子智能体一组。测量编排器上下文大小 vs 每任务一个。
2. 实现一个 `PreToolUse` hook，限制 `write_file` 调用速率（每 session 每分钟 5 次）。追踪行为。
3. 将 `list_subkeys` 接入渲染子智能体树。深层嵌套看起来是什么样的？
4. 将玩具移植到真正的 `claude-agent-sdk` Python 包。工具注册有什么变化？
5. 阅读 Claude Managed Agents 文档。什么时候你会从自托管切换到托管？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Agent SDK | "Claude Code 作为库" | Harness 形态：工具、MCP、hooks、子智能体、session store |
| 子智能体 | "子智能体" | 独立上下文、自有预算；结果向上冒泡 |
| Session store | "对话数据库" | 持久化、加载、列出、删除轮次，带子智能体级联 |
| Hook | "生命周期回调" | Pre/post tool、session、prompt submit、compact、stop |
| W3C trace context | "跨进程 trace" | 父 span 传播到 CLI 子进程 |
| Managed Agents | "托管 harness" | Anthropic 托管的长时间异步工作 |
| `--session-mirror` | "转录镜像" | 在流式传输时将 session 轮次写入外部文件 |
| MCP server | "工具表面" | 附加到智能体的外部工具/资源源 |

## 延伸阅读

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库形式
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产模式
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 对应物
