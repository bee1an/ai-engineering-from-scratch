# 人在回路：Propose-Then-Commit

> 2026 年关于 HITL 的共识是具体的。它不是"agent 问，用户点批准"。它是 propose-then-commit：提议的操作被持久化到带有幂等键的持久存储中；连同意图、数据血缘、触及的权限、爆炸半径和回滚计划一起呈现给审查者；只有在正向确认后才提交；执行后验证副作用确实发生了。LangGraph 的 `interrupt()` 加 PostgreSQL checkpoint、Microsoft Agent Framework 的 `RequestInfoEvent`、Cloudflare 的 `waitForApproval()` 都实现了相同的形状。典型失败模式是橡皮图章审批："Approve?" 被不经审查地点击。文档化的缓解措施是带有明确清单的质询-应答。

**Type:** Learn
**Languages:** Python (stdlib, propose-then-commit state machine with idempotency)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 14 (Tripwires)
**Time:** ~60 minutes

## 问题

Agent 执行一个操作。用户必须决定：批准还是不批准。如果决定是即时的，它可能不是审查。如果决定是结构化的，它慢但可信。工程问题是如何让结构化审查成为阻力最小的路径。

2023 年代的 HITL 模式是同步提示："Agent 想发邮件给 X，内容是 Y——批准？"用户点击批准。每个人都觉得系统是安全的。实际上这个界面被大量橡皮图章：用户快速批准，批准预测力很低，当 agent 出错时，审计轨迹显示一长串用户无法回忆的批准记录。

2026 年的模式——propose-then-commit——将 HITL 移到持久基底上，附加结构化元数据，并要求正向提交。每个托管 agent SDK 都提供了一个版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API 名称不同；形状相同。

## 概念

### Propose-then-commit 状态机

1. **Propose。** Agent 产生一个提议操作。持久化到持久存储（PostgreSQL、Redis、Durable Object）。包含：
   - 意图（agent 为什么要做这个）
   - 数据血缘（什么来源导致了这个提议）
   - 触及的权限（哪些 scope / 文件 / 端点）
   - 爆炸半径（最坏情况是什么）
   - 回滚计划（如果提交了，如何撤销）
   - 幂等键（每个提议唯一；重新提交返回相同记录）
2. **Surface。** 审查者看到带有所有元数据的提议。审查者是人（不是 agent 审查自己）。
3. **Commit。** 正向确认。操作执行。
4. **Verify。** 执行后，副作用被回读并确认。如果验证步骤失败，系统处于已知坏状态，告警介入。

### 幂等键

没有幂等键，瞬态故障后的重试可能双重执行已批准的操作。具体例子：用户批准"从 A 转账 $100 到 B"。网络闪断。Workflow 重试。用户只批准了一次但转账执行了两次。幂等键将批准绑定到单一、唯一的副作用；第二次执行是空操作。

这和 Stripe 及 AWS API 使用的幂等模式相同。将其复用于 agent 审批在 Microsoft Agent Framework 文档中是明确的。

### 持久性：为什么审批比进程活得更久

审批等待室是 agent 不拥有的一块状态。Workflow 被暂停（第 12 课）。当审批到达时，workflow 从该点精确恢复。这就是为什么 LangGraph 将 `interrupt()` 与 PostgreSQL checkpoint 配对，而不仅仅是内存状态——两天后的审批仍然能找到完整的 workflow。

### 橡皮图章审批与质询-应答缓解

HITL 的默认 UI（"Approve" / "Reject" 按钮）产生快速审批而没有真正的审查。文档化的缓解措施：一个质询-应答清单，要求在 Approve 按钮启用之前对特定问题给出正向回答。具体形状：

- "你理解这触及什么资源吗？[ ]"
- "你验证过爆炸半径是可接受的吗？[ ]"
- "如果失败你有回滚计划吗？[ ]"

不是为了官僚而官僚——是一个强制函数。无法勾选方框的审查者要么要求澄清（升级），要么拒绝（安全默认）。Anthropic 的 agent 安全研究明确引用清单驱动的 HITL 作为橡皮图章审批模式的缓解措施。

### 什么算有后果的

不是每个操作都需要 propose-then-commit。2026 年的指导：

- **有后果的操作**（总是 HITL）：不可逆写入、金融交易、对外通信、生产数据库变更、破坏性文件系统操作。
- **可逆操作**（有时 HITL）：编辑本地文件、staging 环境变更、有明确回滚的可逆写入。
- **读取和检查**（永不 HITL）：读取文件、列出资源、调用只读 API。

### 操作后验证

"提交运行了"不等于"副作用发生了"。网络分区和竞态条件可以产生一个认为自己成功了但后端没有持久化的 workflow。验证步骤在提交后重新读取目标资源以确认。这和数据库事务的 `RETURNING` 子句或 AWS 在 `PutObject` 后的 `GetObject` 是相同的模式。

### EU AI Act Article 14

Article 14 要求欧盟高风险 AI 系统具有有效的人工监督。"有效"不是装饰性的。监管语言明确排除橡皮图章模式。带有质询-应答的 propose-then-commit 是在 Microsoft Agent Governance Toolkit 合规文档中能通过 Article 14 审查的形状。

## Use It

`code/main.py` 用 stdlib Python 实现了一个 propose-then-commit 状态机。持久存储是一个 JSON 文件。幂等键是 (thread_id, action_signature) 的哈希。驱动程序模拟三种情况：干净的审批流程、瞬态故障后的重试（不能双重执行），以及橡皮图章默认 vs 质询-应答流程。

## Ship It

`outputs/skill-hitl-design.md` 审查一个拟议的 HITL 工作流是否具备 propose-then-commit 形状，并标记缺失的元数据、幂等性、验证或质询-应答层。

## 练习

1. 运行 `code/main.py`。确认已批准提议的重试使用持久记录而不重新执行。现在将幂等键改为包含时间戳，展示重试会双重执行。

2. 用 `rollback` 字段扩展提议记录。模拟一个验证步骤失败的执行。展示回滚自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` 文档。找出 API 包含但玩具引擎缺少的一个元数据字段。添加它并解释它防护什么。

4. 为一个特定操作（例如"发布到公开 Twitter 账号"）设计质询-应答清单。审查者必须回答哪三个问题？为什么是这三个？

5. 选一个同步 "Approve?" 提示就足够的场景（不需要持久存储）。解释为什么，并说明你接受的风险类别。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| Propose-then-commit | "两阶段审批" | 持久化提议 + 正向提交 + 验证 |
| Idempotency key | "重试安全令牌" | 每个提议唯一；第二次执行为空操作 |
| Data lineage | "它从哪来" | 导致该提议的具体来源内容 |
| Blast radius | "最坏情况" | 操作出错时的影响范围 |
| Rubber-stamp | "快速审批" | 不经真正审查就点击"Approve" |
| Challenge-and-response | "强制清单" | 审查者必须正向确认特定问题 |
| RequestInfoEvent | "MS Agent Framework 原语" | 带结构化元数据的持久 HITL 请求 |
| `interrupt()` / `waitForApproval()` | "框架原语" | LangGraph / Cloudflare 的相同形状等价物 |

## 延伸阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`，持久审批。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 和 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL 作为长时风险的缓解措施。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 高风险系统的监管基线。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 围绕监督的宪法框架。
