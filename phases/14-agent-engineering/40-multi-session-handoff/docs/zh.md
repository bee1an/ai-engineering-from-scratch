# 多会话交接

> 会话会结束。工作不会。交接包是将"智能体工作了一小时"变成"下一个会话在第一分钟就高效"的制品。有意构建它，而不是事后补救。

**类型：** 构建
**语言：** Python (stdlib)
**前置课程：** Phase 14 · 34（仓库记忆）、Phase 14 · 38（验证）、Phase 14 · 39（审查者）
**时间：** ~50 分钟

## 学习目标

- 识别每个交接包需要的七个字段。
- 从 workbench 制品生成交接，无需手写散文。
- 将大型反馈日志裁剪为交接大小的摘要。
- 使下一个会话的第一个动作是确定性的。

## 问题

会话结束。智能体说"很好，我们取得了进展。"下一个会话打开。下一个智能体问"我们上次到哪了？"第一个智能体的答案已经消失了。下一个智能体重新发现、重新运行相同的命令、重新问人类相同的问题，花三十分钟恢复上一个会话最后三十秒的内容。

糟糕交接的代价在任务生命周期的每个会话中都要支付。修复方法是在会话结束时自动生成的包：什么变了、为什么、尝试了什么、什么失败了、还剩什么、下次先做什么。

## 概念

```mermaid
flowchart LR
  State[agent_state.json] --> Generator[generate_handoff.py]
  Verdict[verification_report.json] --> Generator
  Review[review_report.json] --> Generator
  Feedback[feedback_record.jsonl] --> Generator
  Generator --> Handoff[handoff.md + handoff.json]
  Handoff --> Next[Next Session]
```

### 每个交接携带的七个字段

| 字段 | 回答的问题 |
|-------|---------------------|
| `summary` | 做了什么的一段话 |
| `changed_files` | 一览 diff |
| `commands_run` | 实际执行了什么 |
| `failed_attempts` | 尝试了什么以及为什么没成功 |
| `open_risks` | 下一个会话可能遇到什么，带严重性 |
| `next_action` | 下一个会话采取的第一个具体步骤 |
| `verdict_pointer` | 验证 + 审查报告的路径 |

`next_action` 字段是承重的。一个有其他所有内容但没有 `next_action` 的交接是状态报告，不是交接。

### 交接是生成的，不是手写的

手写的交接是在困难的一天会被跳过的交接。生成器读取 workbench 制品并输出包。智能体的工作是让 workbench 处于生成器可以总结的状态，而不是写总结。

### 两种形式：人类可读和机器可读

`handoff.md` 是人类读的。`handoff.json` 是下一个智能体加载的。两者来自相同的源制品。如果它们分歧，JSON 赢。

### 反馈日志裁剪

完整的 `feedback_record.jsonl` 可能有数百条。交接只携带最后 K 条加上每条非零退出的条目。下一个会话如果需要可以加载完整日志，但包保持小。

## 构建

`code/main.py` 实现：

- 一个加载器，将状态、裁决、审查和反馈收集到单个 `WorkbenchSnapshot` 中。
- 一个 `generate_handoff(snapshot) -> (markdown, payload)` 函数。
- 一个过滤器，选取最后 K 条反馈条目加上所有非零退出。
- 一个演示运行，在脚本旁边写入 `handoff.md` 和 `handoff.json`。

运行：

```
python3 code/main.py
```

输出：打印的交接正文，加上磁盘上的两个文件。

## 生产环境中的实践模式

Codex CLI、Claude Code 和 OpenCode 各自有不同的压缩方案；结构化交接包位于所有三者之上。

**压缩策略各异；包 schema 不变。** Codex CLI 的 POST /v1/responses/compact 是服务端不透明的 AES blob（OpenAI 模型的快速路径）；回退是作为 `_summary` user-role 消息追加的本地"交接摘要"。Claude Code 在 95% 上下文时运行五阶段渐进压缩。OpenCode 做基于时间戳的消息隐藏加 5 标题 LLM 摘要。三种不同机制，相同需求：将压缩后存活的内容序列化为可移植制品。包就是那个制品。

**新会话交接不是压缩。** 压缩延长会话；交接干净地关闭一个并开始下一个。Hermes Issue #20372 的框架（2026 年 4 月）是对的：当就地压缩开始退化时，智能体应该写一个紧凑的交接，结束会话，在新鲜上下文中恢复。包是使这个转换廉价的东西。错误是持续压缩直到质量崩溃；修复是预算一个早期、干净的交接。

**每个分支和主题一个活跃交接。** 多智能体协调在过时交接上崩溃的频率高于在糟糕模型输出上。始终包含 `branch`、`last_known_good_commit` 和 `active | superseded | archived` 的 `status`。过时的交接被归档；只有活跃的驱动下一个会话。这是交接作为笔记与交接作为状态之间的区别。

**在 50-75% 上下文时收尾，不是在墙边。** 手写模式 playbook（CLAUDE.md + HANDOVER.md）报告最佳结果是会话在 50-75% 上下文预算时结束而不是 95%。包生成器在压缩制品污染源状态之前干净运行。上下文完整时写入廉价；模型已经迷失时写入昂贵。

## 使用

生产模式：

- **Session-end hook。** 运行时在用户关闭聊天时触发生成器。包进入 `outputs/handoff/<session_id>/`。
- **PR 模板。** 生成器的 markdown 也是 PR body。审查者无需打开五个其他文件就能阅读。
- **跨智能体交接。** 用一个产品构建（Claude Code），用另一个继续（Codex）。包是通用语言。

包小、规则、生产成本低。成本节省随每个会话复合。

## 交付

`outputs/skill-handoff-generator.md` 产出一个调优到项目制品路径的生成器、一个运行它的会话结束 hook，以及下一个智能体在启动时读取的 `handoff.json` schema。

## 练习

1. 添加 `assumptions_to_validate` 字段，浮出构建者记录但审查者未评分超过 1 的每个假设。
2. 对失败运行和通过运行不同地裁剪反馈摘要。论证这种不对称。
3. 包含"给人类的问题"列表。问题进入包而非聊天消息的阈值是什么？
4. 使生成器幂等：运行两次产生相同的包。什么需要稳定才能成立？
5. 添加"下一会话前置条件"部分，精确列出下一个会话在行动前必须加载的制品。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|------------------------|
| Handoff packet | "会话摘要" | 携带七个字段的生成制品，markdown 和 JSON 两种形式 |
| Next action | "先做什么" | 启动下一个会话的一个具体步骤 |
| Feedback trim | "日志摘要" | 最后 K 条记录加每条非零退出 |
| Status report | "我们做了什么" | 缺少 `next_action` 的文档；有用，但不是交接 |
| Verdict pointer | "收据" | 验证 + 审查报告的路径，用于可追溯性 |

## 延伸阅读

- [Anthropic, Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [OpenAI Agents SDK handoffs](https://platform.openai.com/docs/guides/agents-sdk/handoffs)
- [Codex Blog, Codex CLI Context Compaction: Architecture, Configuration, Managing Long Sessions](https://codex.danielvaughan.com/2026/03/31/codex-cli-context-compaction-architecture/) — POST /v1/responses/compact 和本地回退
- [Justin3go, Shedding Heavy Memories: Context Compaction in Codex, Claude Code, OpenCode](https://justin3go.com/en/posts/2026/04/09-context-compaction-in-codex-claude-code-and-opencode) — 三供应商压缩比较
- [JD Hodges, Claude Handoff Prompt: How to Keep Context Across Sessions (2026)](https://www.jdhodges.com/blog/ai-session-handoffs-keep-context-across-conversations/) — CLAUDE.md + HANDOVER.md，50-75% 上下文预算
- [Mervin Praison, Managing Handoffs in Multi-Agent Coding Sessions: Fresh Context Without Losing Continuity](https://mer.vin/2026/04/managing-handoffs-in-multi-agent-coding-sessions-fresh-context-without-losing-continuity/) — 分布式系统框架
- [Hermes Issue #20372 — automatic fresh-session handoff when compression becomes risky](https://github.com/NousResearch/hermes-agent/issues/20372)
- [Hermes Issue #499 — Context Compaction Quality Overhaul](https://github.com/NousResearch/hermes-agent/issues/499) — Codex CLI 中面向交接的提示
- [Microsoft Agent Framework, Compaction](https://learn.microsoft.com/en-us/agent-framework/agents/conversations/compaction)
- [OpenCode, Context Management and Compaction](https://deepwiki.com/sst/opencode/2.4-context-management-and-compaction)
- [LangChain, Context Engineering for Agents](https://www.langchain.com/blog/context-engineering-for-agents)
- Phase 14 · 34 — 生成器读取的状态文件
- Phase 14 · 38 — 包指向的验证裁决
- Phase 14 · 39 — 打包进包的审查报告
