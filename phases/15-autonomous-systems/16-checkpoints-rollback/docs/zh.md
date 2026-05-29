# 检查点与回滚

> 每次图状态转换都会持久化。当 worker 崩溃时，其租约过期，另一个 worker 从最新 checkpoint 接手。Cloudflare Durable Objects 可以跨数小时或数周保持状态。Propose-then-commit（第 15 课）为每个动作定义了回滚计划。执行后验证闭合了循环。EU AI Act Article 14 要求高风险系统必须有有效的人类监督——实践中这意味着 checkpoint 必须可查询、回滚必须经过演练、审计轨迹必须在部署中存活。尖锐的失败模式：没有幂等键和前置条件检查，瞬态故障后的重试可能会重复执行一个已批准的动作。执行后验证是捕获这种情况的手段。

**Type:** Learn
**Languages:** Python (stdlib, checkpoint 和回滚状态机)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 15 (Propose-then-commit)
**Time:** ~60 minutes

## 问题

持久执行（第 12 课）让崩溃的 agent 可以恢复。Propose-then-commit（第 15 课）让已批准的动作可审计。本课将两者结合：当一个已批准的动作部分执行、崩溃并恢复时会发生什么？回滚何时运行，针对什么状态？

真实系统的接线方式各不相同：

- **LangGraph** 将每次图状态转换 checkpoint 到 PostgreSQL。Worker 崩溃时，租约释放，另一个 worker 从最新 checkpoint 恢复。Workflow 在 `interrupt()` 处暂停，暂停本身也会持久化。
- **Cloudflare Durable Objects** 按键保持状态，可跨数小时到数周。将计算与已批准动作的存储共置。
- **Microsoft Agent Framework** 在 workflow API 中暴露 `Checkpoint` 原语；重放加幂等覆盖重试。

在所有情况下，真正有效的组合是：幂等键（防止重复执行）+ 前置条件检查（状态仍然是我们批准时的状态）+ 执行后验证（副作用确实发生了）+ 验证失败时回滚。

## 概念

### 每次转换都持久化

图状态转换是将 workflow 从一个命名状态移动到另一个的任何步骤。朴素实现只在特定提交点持久化；生产实现持久化每次转换。成本（多几次写入）相对于可靠性收益（重放可以落在任何位置，租约恢复是精确的）来说很小。

### 租约恢复

当 worker 崩溃时，workflow 不会丢失；租约（一个短期声明，表示这个 worker 正在执行这个运行）只是过期了。另一个 worker 接手最新的 checkpoint 并恢复。租约机制是让生产系统在滚动部署中存活而不丢失进行中工作的关键。

### 幂等加前置条件

仅有幂等是不够的。考虑：一个 workflow 被批准"当余额 > $1000 时从 A 转账 $100 到 B"。Workflow 被提交，执行中崩溃，然后恢复。如果只检查幂等键，执行恢复后转账运行一次（正确）。但考虑在崩溃和恢复之间，A 的余额通过另一个 workflow 降到了 $500。幂等检查仍然通过；前置条件不通过。没有前置条件检查，我们就会产生透支。

每个有后果的动作都需要两者：

- **幂等键**：防止重复执行。
- **前置条件检查**：确认状态仍然与批准时一致。

### 执行后验证

"工具返回了 200"不是验证。真正的验证是重新读取目标状态并确认副作用确实发生了。模式：

- 数据库更新：`UPDATE ... RETURNING *` 然后断言返回的行匹配预期状态。
- 邮件发送：提交后检查已发送文件夹中的消息 ID。
- 文件写入：读回文件并计算哈希。
- API 调用：对目标资源进行后续 `GET`。

如果验证失败，workflow 处于已知的坏状态。回滚启动。

### 回滚计划

Propose-then-commit（第 15 课）中每个有后果的动作都携带一个回滚计划。类型：

- **带内回滚**：直接逆转副作用（`INSERT` 后 `DELETE`，发送后发送更正邮件）。
- **补偿事务**：一个新动作来中和原始动作（标准 SAGA 模式）。
- **带外回滚**：通知人类，暂停 workflow，保留坏状态以供调查。

无操作回滚（"我们无法撤销这个"）必须在提案中命名。没有回滚的动作在提交时需要更强的 HITL（第 15 课的 challenge-and-response）。

### EU AI Act Article 14 的操作性解读

Article 14 要求高风险系统有"有效的人类监督"。在操作层面，实施者将其解读为：

- Checkpoint 可被审计员查询。
- 回滚经过演练（至少端到端测试过一次）。
- 审计轨迹在部署中存活（checkpoint 后端不是临时的）。
- 验证失败会触发告警，而不是静默记录。

一个在提交中途崩溃、恢复并完成副作用而没有验证 + 回滚路径的 workflow 无法通过 Article 14 的检验。

### 尖锐的失败模式：重复执行

这个领域最常见的生产事故：

1. 动作被批准，幂等键 k。
2. 提交开始，执行，返回 200。
3. Workflow 在持久化"已提交"状态之前崩溃。
4. Workflow 恢复；看到"已批准但未提交"；重新执行。
5. 副作用触发两次。

缓解方案：在执行前持久化一个"进行中"意图，用幂等键执行，然后只在执行后验证成功后才标记"已提交"。如果动作触发了但状态写入失败，你知道需要验证并（如有必要）重新触发。如果状态写入成功但动作失败，你验证并通过恢复路径精确触发一次。

## Use It

`code/main.py` 实现了一个带幂等、前置条件、验证和回滚的 checkpoint workflow。驱动程序模拟四个场景：正常运行、崩溃后重试（幂等捕获）、前置条件失败（workflow 不触发就中止）、验证失败（回滚触发）。

## Ship It

`outputs/skill-rollback-rehearsal.md` 为一个拟议 workflow 设计回滚演练测试，并审计 checkpoint 后端的审计轨迹持久性。

## 练习

1. 运行 `code/main.py`。验证四个场景。对于提交中崩溃的情况，确认动作在重试中精确触发一次。

2. 修改"先标记完成再执行"模式，使状态写入在动作之后触发。重新运行崩溃场景。测量产生了多少重复动作。

3. 为一个具体的生产动作（例如"发布到 Slack 频道"）设计回滚计划。分类为带内、补偿或带外。论证你的选择。

4. 取一个你了解的 workflow。识别每次状态转换。为每个标记持久性需求（持久化/不持久化）。数一数你目前没有持久化的那些。

5. 回滚演练测试：设计一个端到端测试，运行一个真实 workflow，使其崩溃，并确认回滚路径触发。测试断言什么？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| Checkpoint | "存档点" | 每次图状态转换持久化到持久存储 |
| Lease | "Worker 声明" | 短期声明表示 worker 正在执行一个运行；崩溃时过期 |
| Precondition | "状态门" | 断言状态仍然与已批准动作一致 |
| Post-action verify | "回读检查" | 确认副作用确实在目标系统中发生了 |
| In-band rollback | "直接撤销" | 用逆操作逆转副作用 |
| Compensating transaction | "SAGA 撤销" | 一个新动作来中和原始动作 |
| Mark-as-done-first | "状态写入顺序" | 在从提交返回之前持久化已提交状态 |
| Article 14 | "EU AI Act 人类监督" | 操作性：可查询的 checkpoint、经过演练的回滚、可审计的轨迹 |

## 延伸阅读

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — checkpoint 原语和租约恢复。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — Durable Objects 作为状态基底。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 监管基线。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时域 workflow 的可靠性框架。
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — Claude Code Routines 的 workflow 形状。
