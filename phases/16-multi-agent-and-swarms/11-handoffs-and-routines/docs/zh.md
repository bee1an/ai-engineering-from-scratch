# Handoff 与 Routine — 无状态编排

> OpenAI 的 Swarm（2024 年 10 月）将多 agent 编排提炼为两个原语：**routine**（指令 + 工具作为系统提示）和 **handoff**（返回另一个 Agent 的工具）。没有状态机，没有分支 DSL——LLM 通过调用正确的 handoff 工具来路由。OpenAI Agents SDK（2025 年 3 月）是生产继任者。Swarm 本身仍然是最干净的概念参考——其全部源码只有几百行。这个模式之所以流行是因为 API 表面大约是"agent = prompt + tools; handoff = 返回 agent 的函数"。限制：无状态，所以记忆是调用者的问题。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## 问题

每个多 agent 框架都想让你学习它的 DSL：LangGraph 节点和边、CrewAI crew 和 task、AutoGen GroupChat 和 manager。这些 DSL 是真正的抽象，但它们让事情感觉比需要的更重。

Swarm 推向相反方向：使用模型已有的工具调用能力。Handoff 变成工具调用。编排器是当前持有对话的 agent。状态机隐含在 agent 的系统提示中。

## 概念

### 两个原语

**Routine。** 定义 agent 角色和可用工具的系统提示。把它想象成一组有范围的指令："你是分诊 agent；如果用户问退款，handoff 给退款 agent。"

**Handoff。** agent 可以调用的一个工具，返回一个新的 Agent 对象。Swarm 运行时检测到 Agent 返回值并为下一轮切换活跃 agent。

这就是全部抽象。

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

分诊 agent 的系统提示使其根据用户消息选择正确的 handoff。LLM 的工具调用做路由。

### 为什么它流行

- **小 API。** 两个概念要学。
- **使用模型已有的能力。** 工具调用在各提供商中已经是生产级的。
- **没有状态机负担。** 你不描述图；agent 的提示描述它们 handoff 给谁。

### 无状态的代价

Swarm 在运行之间明确无状态。框架在运行期间保持消息历史，但不持久化任何东西。记忆、连续性、长时间运行的任务——都是调用者的问题。

在生产中（OpenAI Agents SDK，2025 年 3 月）这是改变的主要事项之一：SDK 添加了内置会话管理、护栏和追踪，同时保留 handoff 原语。

### Swarm/handoff 适用场景

- **分诊模式。** 前线 agent 将用户路由到专家。
- **基于技能的 handoff。** "如果任务需要代码，调用编码器；如果需要研究，调用研究员。"
- **短的、有界的对话。** 客户支持、FAQ 转工单、简单工作流。

### Swarm 困难场景

- **带共享记忆的长会话。** Handoff 将对话状态重置为新 agent 的提示加历史。没有调用者管理的记忆就没有跨 agent 的持久状态。
- **并行执行。** Handoff 是一次一个——活跃 agent 切换。并行需要调用者编排多个 Swarm 运行。
- **审计和重放。** 无状态运行难以精确重放；LLM 的 handoff 选择不是确定性的。

### OpenAI Agents SDK（2025 年 3 月）

生产继任者添加了：

- **会话状态。** 跨运行的持久线程。
- **护栏。** 输入/输出验证钩子。
- **追踪。** 每个工具调用和 handoff 都被记录。
- **Handoff 过滤器。** 控制 handoff 时什么上下文转移。

Handoff 原语存活；生产人体工学被添加在其周围。

### Swarm vs GroupChat

两者都使用 LLM 驱动的路由，但它们在**谁选择下一个**上不同：

- GroupChat：一个选择器（函数或 LLM）从外部选择下一个发言者。
- Swarm：当前 agent 通过调用 handoff 工具选择其继任者。

Swarm 是"agent 决定下一步"；GroupChat 是"管理者决定下一步"。Swarm 的决策在活跃 agent 的工具调用中；GroupChat 的在 `GroupChatManager` 中。

## Build It

`code/main.py` 从头实现 Swarm：一个 Agent dataclass、一个 handoff 机制（工具返回 Agent）和一个检测 agent 切换的运行循环。

演示：一个分诊 agent 路由到退款、销售或支持专家。每个专家有自己的工具。运行循环打印每次 handoff。

运行：

```
python3 code/main.py
```

## Use It

`outputs/skill-handoff-designer.md` 为给定任务设计 handoff 拓扑：哪些 agent 存在、它们可以调用哪些 handoff、什么上下文转移。

## Ship It

检查清单：

- **Handoff 日志。** 每次 handoff 写一个追踪事件，包含 from-agent、to-agent、上下文快照。
- **上下文转移规则。** 决定 handoff 时什么移动：完整历史（昂贵）、最后 N 条消息或摘要。
- **Handoff 护栏。** 到具有不同工具权限的专家的 handoff 必须经过认证——否则提示注入可以强制不想要的 handoff。
- **循环检测。** 两个 agent 来回 handoff 是常见失败；用简单的 last-K 环检查检测。
- **回退 agent。** 如果 handoff 目标不存在，回退到安全默认。

## 练习

1. 运行 `code/main.py`，分诊到退款 agent。确认第二轮的活跃 agent 是退款。
2. 添加循环检测规则：如果相同两个 agent 连续 handoff 3 次，强制退出。设计回退。
3. 阅读 OpenAI Agents SDK 关于 handoff 过滤器的文档。实现一个"handoff 时摘要"版本：离开的 agent 在接手 agent 接管前将上下文压缩为要点摘要。
4. 比较 Swarm handoff 和 GroupChatManager 选择器。哪种模式使提示注入更严重，为什么？
5. 阅读 Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents)。识别 Swarm 做出的一个显式设计决策，OpenAI Agents SDK 改变了还是保留了。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Routine | "agent 提示" | 系统提示 + 工具列表。定义角色和可用 handoff。 |
| Handoff | "转移到另一个 agent" | 活跃 agent 可以调用的工具，返回新 Agent。运行时切换活跃 agent。 |
| 无状态 | "运行间无记忆" | Swarm 不持久化任何东西；记忆是调用者的责任。 |
| 活跃 agent | "谁在说话" | 当前持有对话的 agent。Handoff 改变这个。 |
| 上下文转移 | "handoff 时什么移动" | 接手 agent 看到什么历史的策略：完整、最后 N 条或摘要。 |
| Handoff 循环 | "Agent 乒乓" | 两个 agent 持续互相 handoff 的失败模式。 |
| OpenAI Agents SDK | "生产 Swarm" | 2025 年 3 月继任者；在 handoff 原语之上添加会话、护栏、追踪。 |
| Handoff 过滤器 | "转移门控" | SDK 特性，在 handoff 边界检查和修改上下文。 |

## 延伸阅读

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 参考阐述
- [OpenAI Swarm repo](https://github.com/openai/swarm) — 原始实现，保留为概念参考
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 带会话和追踪的生产继任者
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — Claude Code 子 agent 如何通过 `Task` 使用类 handoff 模式
