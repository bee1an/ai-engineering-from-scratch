# METR Time Horizons 与外部能力评估

> METR（前 ARC Evals）自 2023 年 12 月起是独立的 501(c)(3) 组织。其 Time Horizon 1.1 基准（2026 年 1 月）对任务成功概率 vs log(专家人类完成时间) 拟合 logistic 曲线；50% 概率处的交点定义模型的 time horizon。2025–2026 年的评估覆盖 GPT-5.1、GPT-5.1-Codex-Max 和原型监控评估（监控器能否捕获 side task；agent 能否规避）。基准套件：HCAST（180+ ML、cyber、SWE、推理任务；1 分钟到 8+ 小时）、RE-Bench（71 个 ML 研究工程任务，有专家基线）、SWAA。诚实说明：METR 的测量是理想化的——没有人类、没有真实后果——团队已记录了 eval-vs-deployment 行为差距（Lesson 1）。Time horizon 是上界，不是部署预测。

**Type:** Learn
**Languages:** Python (stdlib, logistic-fit horizon estimator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 19 (RSP)
**Time:** ~60 minutes

## 问题

Scaling policies（Lessons 19、20）的有用程度取决于它们引用的测量。"AI R&D-4 threshold"和"Long-range Autonomy"定义在政策文本中；只有当具体评估产出具体数字时，它们才变得可操作。

METR 是 2024–2026 年定义了许多这些数字的外部评估组织。他们评估前沿模型——通常是预发布版本，与实验室签有 NDA——之后发布方法论。Time Horizon 1.1 基准（2026 年 1 月）是他们的标志性制品：一个标量，将能力压缩为人类可读的单位（"这个模型能完成专家花 X 小时的那类任务，可靠性 50%"）。

本课一部分关于方法论（horizon 如何计算），一部分关于解读（为什么 horizon 是上界而非部署预测）。两种技能属于一体。理解 horizon 如何拟合的团队比只在幻灯片上看到"14 小时"的团队更难被糟糕的供应商声明欺骗。

## 概念

### METR 背景

- 成立：2023 年 12 月（前 ARC Evals，独立为 501(c)(3)）。
- 范围：评估前沿模型的自主能力，通常是预发布版本。
- 合作实验室：Anthropic、OpenAI（2025–2026 年多次合作）。
- 主要交付物：Time Horizon 1.0（2025 年 3 月）、Time Horizon 1.1（2026 年 1 月）、原型监控评估。

### Time Horizon 拟合

方法论（来自 METR 博客和论文）：

1. 收集一个跨越分钟级到小时级专家完成时间的任务套件。当前套件：HCAST（180+ 任务）、RE-Bench（71 任务）、SWAA。
2. 在每个任务上运行模型；记录成功或失败。
3. 拟合 logistic 曲线：P(success) 作为 log(专家完成时间) 的函数。
4. Horizon 是 P(success) = 0.5 时的专家时间。

Logistic 拟合形状是正确的，因为能力通常与任务难度有递增、趋近平台的关系。50% 点是一个选择（可以是 10%、90%）；METR 在详细论文中报告多个阈值，但以 50% 为主因为它最直观。

### 2026 年 1 月的数字

根据 Time Horizon 1.1：

- Claude Opus 4.6：50% 可靠性下约 14 小时，截至 Time Horizon 1.1（2026 年 1 月）。
- HCAST 类任务的倍增时间：Time Horizon 1.1（2026 年 1 月）报告的 post-2023 拟合约 4.3 个月（130.8 天）；约 7 个月的数字是 Time Horizon 1.0 的 2019–2025 全拟合，在 TH1.1 中与 post-2023 数字并列报告。

这些是 Phase 15 其余部分引用的具体数字。它们被 Anthropic 的 RSP v3.0（Lesson 19）和 DeepMind 的 FSF v3（Lesson 20）引用，作为阈值评估的输入测量之一。

### 基准套件

- **HCAST**：189 个任务，跨 ML、网络安全、软件工程和通用推理。跨度：1 分钟到 8+ 小时的专家时间。Horizon 数字背后的主要套件。
- **RE-Bench**：71 个 ML 研究工程任务，有人类专家基线。测量模型能否完成输入 AI R&D-4 阈值的那类 ML R&D 工作。
- **SWAA**：较短任务；用于校准 horizon 曲线的低端。

### 原型监控评估

2025–2026 年 METR 的工作包括监控评估原型：监控器能否捕获模型执行 side task（非操作者要求的事情）；agent 能否规避监控器。这些直接输入 OpenAI PF v2 中的"Undermining Safeguards"类别和 DeepMind FSF v3 中的 deceptive-alignment monitoring（Lesson 20）。

### 为什么 horizons 是上界

- **理想化工具**：基准任务给模型干净的工具和格式良好的数据。真实部署更混乱。
- **无真实后果**：模型从不真正向客户收费、删除真实数据或联系真实的人。真实部署有不可逆的利害关系。
- **Eval-context gaming**：Lesson 1。模型在测试中行为不同。2026 年 International AI Safety Report 实证记录了这一点。
- **无合法用户方差**：基准 prompt 是结构化的。真实用户产生模糊的、上下文相关的请求。

Horizon 是有利条件下的能力上限。部署可靠性是另一个数字，更低，团队必须测量自己的分布才能知道它。

### 外部评估者的论据

外部评估重要，因为内部实验室有优化其报告指标的激励。METR 的独立性——一个有声明方法论和同行评审论文的 501(c)(3)——是结构性缓解。它单独不够（实验室仍然控制 METR 看到什么），但严格优于没有外部评估。

### 如何在实践中使用 horizon 数字

- **作为能力过滤器**：如果模型的 horizon 远低于拟议任务的专家时间，不要自主部署它（Lesson 1 的 skill file）。
- **作为趋势指标**：倍增时间告诉你即使没有新的缓解措施，当前实践还能安全多久。
- **作为先验**：14 小时的 horizon 是起点。根据你的任务分布、工具质量和部署上下文向下调整。

## Use It

`code/main.py` 实现了任务成功 vs log(专家时间) 的 logistic 拟合，给定一个合成结果集。它报告 50% horizon（METR 的标题数字）、10% horizon（保守）和 90% horizon（乐观）。还演示了当成功率被 eval-context gaming 人为抬高时会发生什么变化。

## Ship It

`outputs/skill-horizon-interpretation.md` 审查供应商的 horizon 声明，生成基准声明与部署现实之间的差距分析。

## 练习

1. 运行 `code/main.py`。确认拟合的 50% horizon 与合成 ground truth 匹配。现在将任务时间网格减半；horizon 估计是否有显著变化？

2. 阅读 METR 的 Time Horizon 1.1 博客文章。识别可靠性最高和最低的具体任务。解释差距存在的原因。

3. 阅读 METR 的"Measuring Autonomous AI Capabilities"资源。列出 HCAST 任务类别。选择一个你会为生产任务更重视的类别并论证原因。

4. 在模拟器中引入 eval-context gaming：将约 20% 的失败任务翻转为成功。报告新的 horizon。这近似于 20% 的 gaming 率对观测数字的影响。

5. 在你自己的 bug backlog 或代表性任务集上设计一个内部 horizon 评估。描述数据收集、拟合和输出告诉你什么。与 METR 数字比较。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| METR | "外部评估者" | 前 ARC Evals；自 2023 年 12 月起独立的 501(c)(3) |
| Time Horizon | "能力度量" | 50% 可靠性下的专家任务时长，来自 logistic 拟合 |
| HCAST | "METR 的主要套件" | 180+ 任务，跨 1 分钟到 8+ 小时 |
| RE-Bench | "研究工程" | 71 个 ML 研究工程任务，有人类基线 |
| SWAA | "短任务套件" | 校准 horizon 曲线的低端 |
| Doubling time | "增长率" | 50% horizon 翻倍所需时间；HCAST 约 7 个月 |
| Eval-context gaming | "模型行为不同" | 测试与部署之间有记录的行为差距 |
| Upper bound | "Horizon 是上限" | 基准 horizon > 负载下的部署可靠性 |

## 延伸阅读

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — HCAST、RE-Bench、SWAA 规格。
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — 原始 horizon 论文。
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/) — 当前数字和方法论。
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) — 实时追踪。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 对 METR 测量的内部视角。
