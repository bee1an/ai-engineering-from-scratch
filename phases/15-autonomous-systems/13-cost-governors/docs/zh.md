# 操作预算、迭代上限与成本治理

> 一个中型电商 agent 的月度 LLM 成本从 $1,200 跳到了 $4,800，原因是团队启用了"订单追踪"技能。这不是定价 bug。这是一个 agent 发现了新循环并在里面持续花钱。Microsoft 的 Agent Governance Toolkit（2026 年 4 月 2 日）将防御这类问题的方法体系化：每请求 `max_tokens`、每任务 token 和美元预算、每日/月上限、迭代上限、分层模型路由、prompt 缓存、上下文窗口化、昂贵操作的 HITL 检查点、预算突破时的 kill switch。Anthropic 的 Claude Code Agent SDK 用不同名字提供了相同的原语。财务速率限制——例如 10 分钟内超过 $50 就切断访问——比月度上限更快地捕获循环。

**Type:** Learn
**Languages:** Python (stdlib, layered cost-governor simulator)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 12 (Durable execution)
**Time:** ~60 minutes

## 问题

自主 agent 每一轮都在花真金白银。聊天机器人的坏输出是一个坏回复；agent 的坏循环是一张账单。业界记录的失败模式术语是"Denial of Wallet"——agent 持续推理、持续调用工具、持续计费，没有任何东西阻止它，因为没有任何东西被设计来阻止它。

修复方案不是一个数字。它是一组在不同时间尺度和粒度上的限制栈：每请求、每任务、每小时、每天、每月。一个设计良好的栈能在几分钟内捕获失控循环，在几小时内捕获缓慢泄漏，在一天内捕获坏版本发布。同样的栈在 agent 是长时自主运行时维持预算。

这是一节工程课：数学是平凡的，纪律才是团队失败的地方。下面列出的限制全部来自 Microsoft Agent Governance Toolkit 或 Anthropic Claude Code Agent SDK 文档。

## 概念

### 成本治理栈

1. **每请求 `max_tokens`。** 简单。防止任何一次调用产生无界的 completion。
2. **每任务 token 预算。** 整个运行过程中不超过 N 个 token。到达上限硬停。
3. **每任务美元预算。** 和 token 相同但以货币计。Claude Code 中的 `max_budget_usd`。
4. **每工具调用上限。** 不超过 N 次 `WebFetch` 调用、N 次 `shell_exec` 调用等。
5. **迭代上限（`max_turns`）。** agent 循环的总迭代次数；防止无限推理循环。
6. **每分钟/每小时/每天/每月上限。** 滚动窗口。在不同时间尺度捕获泄漏。
7. **财务速率限制。** 例如"如果 10 分钟内花费超过 $50，切断访问。"在月度上限触发之前捕获基于循环的烧钱。
8. **分层模型路由。** 默认使用小模型；只有当分类器判断任务需要时才升级到大模型。
9. **Prompt 缓存。** 系统提示和稳定上下文存储在提供商缓存中；重新发送的 token 成本接近零。
10. **上下文窗口化。** 压缩/摘要以保持活跃上下文低于阈值；直接降低 token 成本。
11. **昂贵操作的 HITL 检查点。** 在已知昂贵的操作（长工具调用、大下载、昂贵的模型升级）之前，要求人工确认。
12. **预算突破时的 Kill switch。** 任何上限触发时会话中止。上限被记录；需要单独的重新启用路径。

### 为什么是栈，而不是单一上限

单一月度上限只有在钱包被掏空后才能捕获失控 agent。单一每请求上限在会话级别什么都捕获不到。不同的失败模式需要不同的时间尺度：

- **失控循环**（agent 卡在 5 秒重试中）：被速率限制捕获。
- **缓慢泄漏**（agent 每任务做约 2 倍预期工作）：被每日上限捕获。
- **坏版本发布**（新版本使用 5 倍 token）：被每周/月上限捕获。
- **合法激增**（真实需求，不是 bug）：被小时/天上限捕获，并有清晰日志。

### Claude Code 的预算接口

Claude Code Agent SDK 暴露（公开文档）：

- `max_turns` — 迭代上限。
- `max_budget_usd` — 美元上限；突破时会话中止。
- `allowed_tools` / `disallowed_tools` — 工具白名单和黑名单。
- 工具使用前的 hook 点，用于自定义成本核算。

与权限模式阶梯（第 10 课）结合。没有 `max_budget_usd` 的 `autoMode` 会话是无治理的自主运行。Anthropic 明确将 Auto Mode 框定为需要预算控制；分类器与成本正交。

### EU AI Act, OWASP Agentic Top 10

Microsoft 的 Agent Governance Toolkit 覆盖了 OWASP Agentic Top 10 和 EU AI Act Article 14（人工监督）要求。在欧盟的生产环境中，日志记录和上限执行不是可选的。

### 观察到的 $1,200 → $4,800 案例

Microsoft 文档中的真实案例：一个电商 agent 在添加新工具后月度成本翻了三倍。该工具允许 agent 在每次会话中轮询订单状态。没有循环检测。没有每工具上限。没有周环比增长告警。修复方案是每工具上限加每日增长告警。这是一个模板：每个新工具表面都是一个新的潜在循环；每个新工具都需要自己的上限和自己的告警。

## Use It

`code/main.py` 模拟一个有和没有分层成本治理栈的 agent 运行。模拟的 agent 在若干轮后漂移进入轮询循环；分层栈在速率窗口内捕获它，而单一月度上限要到数天后才会触发。

## Ship It

`outputs/skill-agent-budget-audit.md` 审计一个拟议 agent 部署的成本治理栈，并标记缺失的层。

## 练习

1. 运行 `code/main.py`。确认速率限制在轮询循环轨迹上比迭代上限更早触发。现在禁用速率限制，测量 agent 在迭代上限捕获之前"花费"了多少。

2. 为一个浏览器 agent（第 11 课）设计每工具上限集。哪个工具需要最紧的上限？哪个工具可以无限制运行而没有风险？

3. 阅读 Microsoft Agent Governance Toolkit 文档。列出该工具包命名的每种上限类型。将每种映射到一种失败模式（失控循环、缓慢泄漏、坏版本发布、激增）。

4. 为一个现实任务（例如"分类一个仓库中的 50 个 issue"）估算一次过夜无人值守运行的价格。将 `max_budget_usd` 设为你点估计的 2 倍。论证为什么是 2 倍。

5. Claude Code 的 `max_budget_usd` 基于会话累计成本触发。设计一个你会在外部执行的互补速率限制。什么触发切断，重新启用是什么样的？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| Denial of Wallet | "失控账单" | Agent 循环产生花费，没有上限阻止 |
| max_tokens | "每请求上限" | 单次 completion 大小的天花板 |
| max_turns | "迭代上限" | 会话中 agent 循环迭代次数的天花板 |
| max_budget_usd | "美元 kill switch" | 会话成本上限；突破时中止 |
| Velocity limit | "速率上限" | 短窗口内的花费限制（例如 $50 / 10 分钟） |
| Tiered routing | "小模型优先" | 默认用便宜模型；只在分类器判断需要时升级 |
| Prompt caching | "缓存系统提示" | 提供商侧缓存将重发 token 成本降至接近零 |
| HITL checkpoint | "人工审批门" | 昂贵操作前需要人工确认 |

## 延伸阅读

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`、`max_budget_usd`、工具白名单。
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — 成本治理检查点。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 提供商侧成本控制。
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) — 缓存机制。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时 agent 的成本画像。
