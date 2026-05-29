# 失败模式：Agent 为什么会崩

> MASFT（Berkeley, 2025）将 14 种多智能体失败模式归入 3 个类别。Microsoft 的 Taxonomy 记录了现有 AI 失败如何在 agentic 场景中被放大。行业现场数据收敛于五种反复出现的模式：幻觉动作、范围蔓延、级联错误、上下文丢失、工具误用。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 05 (Self-Refine and CRITIC), Phase 14 · 24 (Observability)
**Time:** ~60 minutes

## 学习目标

- 列举 MASFT 的三个失败类别及每个类别中至少四种具体模式。
- 解释为什么 agentic 失败会放大现有 AI 失败模式（偏见、幻觉）。
- 描述五种行业反复出现的模式及其缓解措施。
- 实现一个 stdlib 检测器，为 agent trace 打上失败模式标签。

## 问题

团队上线的 agent 在 90% 的 trace 上正常工作。那 10% 的失败不是随机噪声 — 它们落入少数反复出现的类别。一旦你能命名它们，就能监控并修复它们。

## 概念

### MASFT (Berkeley, arXiv:2503.13657)

Multi-Agent System Failure Taxonomy。14 种失败模式聚为 3 个类别。标注者间 Cohen's Kappa 0.88 — 类别可靠可区分。

核心主张：失败是多智能体系统的根本设计缺陷，而非可以通过更好基础模型修复的 LLM 局限。

### Microsoft Taxonomy of Failure Mode in Agentic AI Systems

- 现有 AI 失败（偏见、幻觉、数据泄露）在 agentic 场景中被放大。
- 自主性带来新的失败：大规模意外动作、工具误用、任务漂移。
- 该白皮书是 agentic 产品的风险登记册。

### Characterizing Faults in Agentic AI (arXiv:2603.06847)

- 失败源于编排、内部状态演化和环境交互。
- 不仅仅是"代码差"或"模型输出差"。

### LLM Agent Hallucinations Survey (arXiv:2509.18970)

两种主要表现：

1. **Instruction-following Deviation** — agent 不遵循 system prompt。
2. **Long-range Contextual Misuse** — agent 遗忘或误用早期轮次的上下文。

Sub-intention 错误：Omission（遗漏步骤）、Redundancy（重复步骤）、Disorder（步骤乱序）。

### 五种行业反复出现的模式

Arize、Galileo、NimbleBrain 2024-2026 现场分析收敛于：

1. **幻觉动作。** Agent 调用不存在的工具或捏造参数。
2. **范围蔓延。** Agent 将任务扩展到用户请求之外（创建额外 PR、发送额外邮件）。
3. **级联错误。** 一次错误调用触发下游连锁反应。一个虚构的 SKU 幻觉触发四次 API 调用 — 一个多系统事故。
4. **上下文丢失。** 长周期任务遗忘早期轮次的约束。
5. **工具误用。** 用错误参数调用正确工具，或完全调用错误工具。

级联是杀手。Agent 无法区分"我失败了"和"任务不可能完成"，经常在 400 错误上幻觉出成功消息来关闭循环。

### 缓解：每步设门

在推理链的每一步设置自动化验证门，对照环境状态检查事实基础。具体来说：

- Per-step safety classifier（Lesson 21）。
- Tool-call 参数验证（Lesson 06）。
- 将检索内容与已知事实交叉检查（Lesson 05, CRITIC）。
- 通过重新探测状态检测成功幻觉（文件真的创建了吗？）。

### 失败监控容易出错的地方

- **只标记崩溃。** 大多数 agent 失败产出看起来合法的输出。需要内容级检查。
- **没有基线。** 漂移检测需要 last-known-good；没有它你无法说"这在变差"。
- **过度告警。** 每次失败都产生一个 page。应聚类和限流。

## Build It

`code/main.py` 实现一个 stdlib 失败模式标记器：

- 覆盖五种模式的合成 trace 数据集。
- 每种模式的检测函数（tool call、输出、重复动作的签名模式）。
- 一个标记器，为每条 trace 打标签并报告模式分布。

运行：

```
python3 code/main.py
```

输出：per-trace 标签 + 聚合分布，廉价复现 Phoenix trace 聚类所展示的内容。

## Use It

- **Phoenix** 用于生产漂移聚类（Lesson 24）。
- **Langfuse** 用于 session replay + 标注。
- **Custom** 用于你的可观测性平台无法检测的领域特定签名。

## Ship It

`outputs/skill-failure-detector.md` 生成针对你领域的失败模式检测器，接入 trace 存储。

## 练习

1. 添加"成功幻觉"检测器：agent 返回成功但目标状态未改变。
2. 标记你构建的产品中 100 条真实 trace。哪种模式占主导？修复它的成本是多少？
3. 实现"级联半径"指标：给定步骤 N 的失败，它影响了多少下游步骤？
4. 阅读 MASFT 的 14 种失败模式。选三种适用于你产品的。编写检测器。
5. 将一个检测器接入 CI 任务：如果 >=5% 的 trace 标记了某种模式则构建失败。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| MASFT | "Multi-agent failure taxonomy" | Berkeley 14 模式分类 |
| Cascading error | "Ripple failure" | 一个早期错误传播经过 N 步 |
| Context loss | "Forgot the constraint" | 长周期轮次丢弃早期事实 |
| Tool misuse | "Wrong tool / wrong args" | 合法调用，错误调用方式 |
| Success hallucination | "Faked completion" | Agent 在 400 上声称成功；状态未变 |
| Scope creep | "Overreach" | Agent 做了超出请求的事 |
| Instruction-following deviation | "Disobedience" | 忽略 system prompt 或用户约束 |
| Sub-intention errors | "Plan bugs" | 计划执行中的遗漏、冗余、乱序 |

## 延伸阅读

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 failure modes, 3 categories
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — risk register
- [Arize Phoenix](https://docs.arize.com/phoenix) — drift clustering in practice
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — when simpler patterns avoid modes entirely
