# OpenAI Preparedness Framework 与 DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2（2025 年 4 月）引入了 Research Categories — Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards — 与 Tracked Categories 区分开来。Tracked Categories 触发 Capabilities Reports 加 Safeguards Reports，由 Safety Advisory Group 审查。DeepMind 的 FSF v3（2025 年 9 月，2026 年 4 月 17 日添加 Tracked Capability Levels）将自主性折叠到 ML R&D 和 Cyber 领域（ML R&D autonomy level 1 = 以与人类 + AI 工具相当的成本完全自动化 AI R&D 流水线）。FSF v3 通过自动化监控明确应对 deceptive alignment 中的 instrumental-reasoning 滥用。诚实说明：PF v2 中的 Research Categories（包括 Long-range Autonomy）不会自动触发缓解措施；政策用语是"潜在的"。DeepMind 自己也说，如果 instrumental reasoning 增强，自动化监控"长期来看将不再充分"。

**Type:** Learn
**Languages:** Python (stdlib, three-framework decision-table diff tool)
**Prerequisites:** Phase 15 · 19 (Anthropic RSP)
**Time:** ~45 minutes

## 问题

Lesson 19 仔细阅读了 Anthropic 的 scaling policy。本课通过阅读 OpenAI 和 DeepMind 的政策来补全全貌。三份文件是表亲制品，回答同一个问题——前沿实验室何时应暂停或门控模型——它们在一小组类别上趋同，在特定的重要之处分歧。

趋同点：三者都将 long-range autonomy 标记为值得追踪的能力类别。三者都承认欺骗行为（alignment faking、sandbagging）是一个特定的风险类别。三者都有内部审查机构。分歧点：OpenAI 将类别分为"Tracked"（强制缓解）和"Research"（无自动触发）。DeepMind 将自主性折叠到两个领域而非单独命名。实验室命名为 Tracked vs Research，或 Critical vs Moderate，或 Tier-1 vs Tier-2；一个能力落在哪个桶中的运营后果在各实验室之间不同。

将它们放在一起阅读是有价值的练习。同一能力可能在 Anthropic 是"强制缓解"，在 OpenAI 是"监控但不触发"，在 DeepMind 是"在特定领域中追踪"。政策姿态很重要。

## 概念

### OpenAI Preparedness Framework v2（2025 年 4 月）

结构：

- **Tracked Categories**：触发 Capabilities Reports（模型能做什么）加 Safeguards Reports（有哪些缓解措施）。部署前由 Safety Advisory Group 审查。
- **Research Categories**：实验室正在追踪但尚未承诺具体缓解措施的模型能力。包括 Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards。

v2 的关键框架：Research Categories 不会自动触发缓解措施。政策用语是"潜在的"缓解措施。这是需要关注的具体差异：一个类别在 Tracked vs Research 中的位置是"门控"与"观察"之间的枢纽。

### DeepMind Frontier Safety Framework v3（2025 年 9 月；2026 年 4 月 17 日添加 Tracked Capability Levels）

结构：

- **Critical Capability Levels (CCLs)**：五个领域的能力阈值：Cyber、Bio、ML R&D、CBRN、Autonomy（折叠到 ML R&D 和 Cyber 中）。
- **Tracked Capability Levels**：2026 年 4 月添加的额外粒度。具体示例：ML R&D autonomy level 1 = 以与人类 + AI 工具相当的成本完全自动化 AI R&D 流水线。
- **Deceptive alignment monitoring**：明确承诺对 instrumental-reasoning 滥用进行自动化监控。

自主性的框架与 OpenAI 不同。DeepMind 不将"Autonomy"作为顶级领域；它被折叠到自主性会造成危害的领域中（ML R&D 和 Cyber）。论点是：没有领域的自主性是没有风险的能力；反论点是：跨领域的自主性是框架应该命名的元风险。

### 三者的趋同点

- 内部 Safety Advisory Group（分别命名为 Anthropic SAG、OpenAI SAG、DeepMind 内部委员会）。高能力模型部署前审查。
- 明确提及 deceptive alignment / alignment faking 作为风险类别。
- 按声明节奏发布的常设制品（Anthropic：Frontier Safety Roadmap、Risk Report；OpenAI：Capabilities and Safeguards Reports；DeepMind：FSF 更新周期）。
- 承认仅靠监控的防御有上限。DeepMind 明确表示："自动化监控长期来看将不再充分。"

### 分歧点

- **Anthropic**：v3.0 中移除了暂停承诺；AI R&D-4 阈值是命名的下一个门控。
- **OpenAI**：Tracked vs Research 分裂；Research Categories（包括 Long-range Autonomy）不会自动门控。
- **DeepMind**：自主性折叠到其他领域；2026 年 4 月添加 Tracked Capability Levels 增加粒度。

### Sandbagging：一个使三者都复杂化的特定能力

Sandbagging（模型在评估中策略性地表现不佳）在 OpenAI 的 Research Categories 中。Anthropic 的 RSP v3.0 通过 evaluation-context gap（Lesson 1）来应对。DeepMind 通过 FSF v3 中的 deceptive alignment monitoring 来应对。

如果模型在评估中 sandbagging，每个框架的能力阈值都被低估。框架只有在测量有效时才有效。这就是为什么除了实验室自评之外，还需要外部测量（Lesson 21，METR）和对抗性评估。

### 政策阅读技能

- 定位：你关心的每个能力都应该能在政策中找到。如果找不到，政策就没有覆盖它。
- 分类：它是 Tracked（触发缓解）还是 Research（追踪但不触发）？OpenAI 命名了这个；Anthropic 和 DeepMind 有各自的等价物。
- 节奏：政策是按声明的时间表更新，还是仅在特定事件后更新？声明节奏更强。
- 独立性：外部审查是强制的还是可选的？Anthropic 与 Apollo 和 US AI Safety Institute 合作；OpenAI 与 METR 合作；DeepMind 主要依赖内部 SAG。

## Use It

`code/main.py` 实现了一个小型决策表 diff 工具。给定一个能力（autonomy、deceptive alignment、R&D automation、cyber uplift 等），它输出三个政策各自如何分类该能力，以及触发什么缓解措施。这是阅读辅助工具，不是政策工具。

## Ship It

`outputs/skill-cross-policy-diff.md` 为特定能力生成跨政策比较，使用三个框架作为参考。

## 练习

1. 运行 `code/main.py`。确认 diff 工具的输出与你能对照源文件验证的至少两个能力的政策匹配。

2. 完整阅读 OpenAI Preparedness Framework v2。识别每个 Research Category。对每个类别，写一句话说明为什么它在 Research 而非 Tracked 中。

3. 完整阅读 DeepMind FSF v3，加上 2026 年 4 月的 Tracked Capability Levels 更新。识别 ML R&D autonomy level 1 的具体评估标准。你如何从外部测量它？

4. Sandbagging 在 OpenAI 的 Research Categories 中。设计一个评估，迫使 sandbagging 模型暴露其实际能力。参考 Lesson 1 的 eval-context-gaming 讨论。

5. 在一个特定能力（你选择）上比较三个政策。指出哪个政策的分类你认为最严格，哪个最不严格。用源文本论证。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| Preparedness Framework | "OpenAI 的 scaling policy" | PF v2（2025 年 4 月）；Tracked vs Research categories |
| Tracked Category | "强制缓解" | 触发 Capabilities + Safeguards Reports；SAG 审查 |
| Research Category | "仅监控" | 追踪但无自动缓解；包括 Long-range Autonomy |
| Frontier Safety Framework | "DeepMind 的 scaling policy" | FSF v3（2025 年 9 月）+ Tracked Capability Levels（2026 年 4 月） |
| CCL | "Critical Capability Level" | DeepMind 每领域阈值（Cyber、Bio、ML R&D、CBRN） |
| ML R&D autonomy level 1 | "R&D 自动化" | 以相当成本完全自动化 AI R&D 流水线 |
| Sandbagging | "策略性表现不佳" | 模型在评估中表现不佳；在 OpenAI Research Categories 中 |
| Instrumental reasoning | "手段-目的推理" | 关于如何实现目标的推理；DeepMind 监控的目标 |

## 延伸阅读

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 公告。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文件。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 公告。
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — Tracked Capability Levels 添加。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF 格式 Risk Report 示例。
