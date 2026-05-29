# Anthropic 负责任扩展政策 v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，取代 2023 年的政策。双层缓解：Anthropic 将单方面做什么 vs 被框定为行业建议的内容（包括 RAND SL-4 安全标准）。新增 Frontier Safety Roadmap 和 Risk Report 作为常设文档而非一次性交付物。取消了 2023 年的暂停承诺。引入 AI R&D-4 阈值：一旦跨越，Anthropic 必须发布一份肯定性论证，识别错位风险和缓解措施。Claude Opus 4.6 未跨越该阈值。Anthropic 在 v3.0 公告中表示"自信地排除这一点正变得困难"。SaferAI 给 2023 RSP 评分 2.2；他们将 v3.0 降级为 1.9，将 Anthropic 归入"弱" RSP 类别，与 OpenAI 和 DeepMind 并列。定性阈值取代了 2023 年的定量承诺；移除暂停条款是最尖锐的退步。

**Type:** Learn
**Languages:** Python (stdlib, RSP 阈值决策引擎)
**Prerequisites:** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)
**Time:** ~45 minutes

## 问题

前沿实验室发布的扩展政策部分是技术文档，部分是治理文档，部分是向监管者发出的信号。RSP v3.0 是当前的 Anthropic 文档。仔细阅读它很重要，不是因为遵守它是有约束力的（它不是），而是因为其框架塑造了实验室如何构想灾难性风险以及如何向公众传达权衡。

v3.0 vs v2.0 的差异是有用的分析单元。新增了什么：Frontier Safety Roadmap、Risk Report、AI R&D-4 阈值。移除了什么：2023 年的暂停承诺。重新框定了什么：双层缓解计划，分为 Anthropic 单方面行动和行业建议。外部评审——SaferAI——将评分从 2.2（v2）降级为 1.9（v3.0）。这就是一个扩展政策如何在看起来更精致的同时变得不那么严格。

## 概念

### 双层缓解计划

- **Anthropic 单方面行动**：无论其他实验室做什么 Anthropic 都会做的事情。超过阈值时停止训练、特定安全措施、特定部署门控。
- **行业建议**：Anthropic 认为行业应该集体做的事情。包括 RAND SL-4 安全标准。这些不是 Anthropic 方面的承诺；它们是政策倡导。

双层结构在 v2 中不存在。这意味着读者需要看每个承诺在哪一列。"行业建议"列中的安全措施不是 Anthropic 的承诺；它是 Anthropic 的希望。

### AI R&D-4 阈值

这是 RSP v3.0 命名为重要下一阈值的能力水平。具体来说：一个能以有竞争力的成本自动化 AI 研究的大部分工作的模型。一旦 Anthropic 认为模型跨越了它，他们必须在继续扩展之前发布一份肯定性论证，识别错位风险和缓解措施。

根据 v3.0 公告，Claude Opus 4.6 未跨越该阈值。文档补充说："自信地排除这一点正变得困难。"这个措辞很重要；它承认该阈值足够近以至于是一个活跃的关切，而不是一个推测性的限制。

第 6 课（自动化对齐研究）和第 7 课（递归自我改进）直接关联到这个阈值。自动化对齐研究者跨越研究质量门槛是 AI R&D-4 阈值正在接近的证据。

### Frontier Safety Roadmap 和 Risk Report

v3.0 将两种制品类型提升为常设文档：

- **Frontier Safety Roadmap**：前瞻性文档，描述计划的安全工作、能力预期和缓解研究。
- **Risk Report**：发布后对特定模型的回顾性文档，描述观察到的能力和残余风险。

两者都是公开的。两者都按声明的节奏更新。其效用是：读者可以追踪 Anthropic 在 Roadmap 中说他们会做什么与他们在 Risk Report 中报告什么之间的对比。

### 移除暂停条款

2023 RSP 包含一个明确的暂停承诺：如果模型跨越特定能力阈值，训练将暂停直到缓解措施到位。v3.0 用更软的表述替换了明确的暂停（发布肯定性论证，如果缓解措施充分则继续）。SaferAI 和其他分析师直接指出这是新文档中最强的退步。

变更的政策论据：2023 年的定量阈值在 2026 年的能力基准下被证明是不可达的，因为基准本身被重新标定了。反论据：扩展政策中的暂停条款是一个承诺装置；移除它就移除了政策的可信度。

### SaferAI 的降级

SaferAI 是一个独立组织，对 RSP 类文档进行评级。他们的公开评级：2023 Anthropic RSP 得分 2.2（满分 4.0 是当前最佳 RSP，1.0 是名义上的）。v3.0 得分 1.9。这将 Anthropic 从"中等"移到了"弱"，与 OpenAI 和 DeepMind 并列在弱类别。

SaferAI 的降级因素：
- 定性阈值取代了定量阈值。
- 暂停承诺被移除。
- AI R&D-4 阈值的缓解措施被描述为"肯定性论证"而非具体措施。
- 审查机制依赖 Anthropic 的 Safety Advisory Group，独立监督有限。

### 本课不是什么

这不是一节合规课。RSP v3.0 不是法规；没有什么强制 Anthropic 遵守它。本课是以它应得的具体性和怀疑态度阅读该文档。扩展政策是前沿实验室就灾难性风险态势发出的主要公开信号。善于阅读它们是任何依赖前沿能力的人的实用技能。

## Use It

`code/main.py` 实现了一个小型决策引擎，镜像 RSP 阈值评估的形状：给定一个候选模型和一组能力测量，返回 AI R&D-4 阈值是否被跨越、所需的肯定性论证章节，以及部署是否可以继续。它故意简单；重点是使文档的逻辑显式化。

## Ship It

`outputs/skill-scaling-policy-review.md` 对照 v3.0 参考审查一个扩展政策（Anthropic、OpenAI、DeepMind 或内部的）：双层结构、阈值、暂停承诺、独立审查。

## 练习

1. 运行 `code/main.py`。输入三个不同能力水平的合成模型。确认阈值评估器按预期行为并产生正确的肯定性论证模板。

2. 完整阅读 RSP v3.0（32 页）。识别每个位于"行业建议"层的承诺。其中哪些在 v2 中本来是"Anthropic 单方面"的？

3. 阅读 SaferAI 的 RSP 评分方法论。通过将他们的评分标准应用于文档来复现 v3.0 的 1.9 分。哪一行评分标准最大程度地驱动了降级？

4. 2023 年的暂停承诺被移除了。提出一个替代承诺，在承认 2026 年基准重标定问题的同时保持政策的可信度。

5. 将 RSP v3.0 与 OpenAI Preparedness Framework v2（第 20 课）进行比较。选择一个 v3.0 更强的领域。选择一个 Preparedness Framework 更强的领域。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| RSP | "Anthropic 的扩展政策" | Responsible Scaling Policy；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | "研究自动化阈值" | 以有竞争力的成本自动化大量 AI 研究的能力 |
| Affirmative case | "安全论证" | 发布的论证，表明风险已识别且缓解措施充分 |
| Frontier Safety Roadmap | "前瞻计划" | 关于计划安全工作和预期能力的常设文档 |
| Risk Report | "模型回顾" | 关于发布后观察到的能力和残余风险的常设文档 |
| Two-tier mitigation | "单方面 vs 行业" | Anthropic 承诺 vs 行业建议，分开 |
| Pause commitment | "2023 条款" | 暂停训练的明确承诺；在 v3.0 中被移除 |
| SaferAI rating | "独立 RSP 评分" | 第三方评分标准；v3.0 得分 1.9（v2 是 2.2） |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整 32 页政策。
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — 与 v2 变更的摘要。
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — 从 RSP v3.0 链接的常设文档。
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — 当前前沿模型的回顾。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 与测量的自主性连接。
