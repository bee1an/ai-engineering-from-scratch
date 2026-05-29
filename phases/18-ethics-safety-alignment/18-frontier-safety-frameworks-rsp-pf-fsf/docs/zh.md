# 前沿安全框架 — RSP、PF、FSF

> 三大实验室框架定义了 2026 年前沿能力的行业治理。Anthropic 负责任扩展政策 v3.0（2026 年 2 月）引入分层 AI 安全等级（ASL-1 到 ASL-5+），以生物安全等级为模型，ASL-3 于 2025 年 5 月因 CBRN 相关模型而激活。OpenAI Preparedness Framework v2（2025 年 4 月）定义了追踪能力的五项标准，并将能力报告与保障措施报告分离。DeepMind Frontier Safety Framework v3.0（2025 年 9 月）引入关键能力等级，包括新的有害操纵 CCL。三者现在都包含竞争者调整条款，允许在同行实验室未采取可比保障措施时推迟要求。跨实验室对齐在结构上一致，但术语不同："Capability Thresholds"、"High Capability thresholds"和"Critical Capability Levels"表示类似的构造。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 07-09 (deception failures)
**Time:** ~75 minutes

## 学习目标

- 描述 Anthropic 的 ASL 层级结构以及什么触发了 ASL-3。
- 列出 OpenAI Preparedness Framework v2 追踪能力的五项标准。
- 描述 DeepMind 的关键能力等级结构和有害操纵 CCL。
- 解释竞争者调整条款及其对竞赛动态的影响。
- 定义安全论证并描述三支柱结构（监控、不可读性、无能力）。

## 问题

Lessons 7-17 确立了欺骗是可能的、双重用途能力存在、评估有局限性。拥有前沿能力模型的实验室需要一个内部治理结构来：
- 定义何时需要新保障措施的阈值。
- 定义扩展前所需的评估。
- 描述安全论证应该是什么样的。
- 处理竞赛动态问题（如果竞争对手在没有保障措施的情况下发布，你怎么办？）。

2025-2026 年的三个框架是当前最先进的 — 不完美、持续演进，但跨实验室足够对齐，使得治理问题现在是框架是否充分，而非框架是否存在。

## 概念

### Anthropic 负责任扩展政策 v3.0（2026 年 2 月）

ASL 结构：
- ASL-1：非前沿模型（被弱于前沿的基线所涵盖）。
- ASL-2：当前前沿基线；以常规保障措施部署。
- ASL-3：灾难性滥用风险显著更高；CBRN 相关能力。2025 年 5 月激活。
- ASL-4：AI R&D-2 跨越阈值；能够自动化入门级 AI 研究的模型。
- ASL-5+：高级 AI R&D；能够显著加速有效扩展的模型。

v3.0 新增内容：
- 前沿安全路线图（以删节形式公开）。
- 风险报告（季度发布，部分经外部审查）。
- AI R&D 被细分为 AI R&D-2 和 AI R&D-4。
- 一旦跨越 AI R&D-4，需要肯定性安全论证，识别模型追求错位目标的错位风险。

### OpenAI Preparedness Framework v2（2025 年 4 月 15 日）

追踪能力的五项标准：
- **Plausible（合理）。** 存在合理的威胁模型。
- **Measurable（可测量）。** 可进行经验评估。
- **Severe（严重）。** 伤害很大。
- **Net-new（全新）。** 不是已有风险的放大。
- **Instantaneous-or-irremediable（即时或不可补救）。** 伤害发生迅速或无法撤销。

满足全部五项的能力被追踪。其他不被追踪。

PF v2 的其他结构：
- 将能力报告（模型能做什么）与保障措施报告（存在什么控制）分离。
- 安全咨询组审查。
- 领导层批准；董事会安全与安保委员会监督。
- "调整条款"：如果另一家实验室在没有可比保障措施的情况下发布，OpenAI 可以降低要求。

### DeepMind Frontier Safety Framework v3.0（2025 年 9 月）

按领域划分的关键能力等级（CCLs）：
- 生物武器提升
- 网络提升
- ML R&D 加速
- 有害操纵（v3.0 新增）：能够在高风险场景中实质性改变信念/行为的模型。

v2.0（2025 年 2 月）增加了欺骗性对齐部分和 ML R&D CCLs 的更高安全等级。

### 跨实验室对齐

- Anthropic "Capability Thresholds"。
- DeepMind "Critical Capability Levels"。
- OpenAI "High Capability thresholds"。

没有行业标准术语。结构上对齐：三个前沿能力层级，带有公开的评估标准。三者都包含竞争者调整条款（2025 年起）。

### 安全论证

安全论证是一份书面论证，论证在最坏情况假设下部署是可接受安全的。标准结构针对三个支柱：

- **监控（Monitoring）。** 如果发生不良行为，我们能否检测到？
- **不可读性（Illegibility）。** 模型是否缺乏执行连贯计划来造成伤害的能力？
- **无能力（Incapability）。** 模型是否缺乏造成相关伤害的能力？

不同的安全论证针对不同的支柱。对于 ASL-3 CBRN 案例，无能力（通过遗忘）是主要目标。对于欺骗性对齐，监控和不可读性是目标。对于网络提升，三者都相关。

### 竞赛动态问题

竞争者调整条款存在争议。批评者认为它们制造了逐底竞争：如果三家实验室都会在竞争对手背叛时降低要求，均衡就会向背叛方向移动。支持者认为替代方案（单方面保障措施）在背叛实验室安全意识更低时会产生更差的结果。

UK AISI、US CAISI 和 EU AI Office（Lesson 24）是外部治理对应方。实验室框架是自愿的；监管框架正在形成。

### 在 Phase 18 中的位置

Lessons 17-18 是建立在欺骗和红队分析之上的测量与治理层。Lessons 19-24 覆盖福利、偏见、隐私、水印和监管结构。Lesson 28 映射了将评估付诸实践的研究生态系统（MATS、Redwood、Apollo、METR）。

## Use It

本课无代码。阅读三个主要来源：RSP v3.0、PF v2、FSF v3.0。将每个实验室的层级结构映射到其他实验室，并找出每个实验室定义了而其他实验室没有的一个阈值。

## Ship It

本课产出 `outputs/skill-framework-diff.md`。给定一个安全框架或发布说明，它将框架的阈值定义、所需评估和安全论证结构与 RSP v3.0、PF v2、FSF v3.0 进行比较，并标记跨实验室差距。

## 练习

1. 阅读 RSP v3.0、PF v2 和 FSF v3.0。编制一张表格，列出每个实验室的 CBRN 阈值、AI R&D 阈值和所需的部署前评估。

2. 竞争者调整条款存在于所有三个框架中（2025 年起）。写一段支持它的论述；写一段反对它的论述。找出每个立场所依赖的假设。

3. 为跨越 Anthropic AI R&D-4 阈值的模型设计一个安全论证。列出三个支柱（监控、不可读性、无能力）各自需要的证据。

4. DeepMind 的 FSF v3.0 引入了有害操纵 CCL。提出三项经验测量，用以表明模型已跨越该阈值。

5. 阅读 METR 的 "Common Elements of Frontier AI Safety Policies"（2025）。列出三个最强的跨实验室趋同点和两个最大的分歧。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| RSP | "Anthropic 的框架" | 负责任扩展政策；ASL 层级；v3.0 2026 年 2 月 |
| PF | "OpenAI 的框架" | Preparedness Framework；五项标准；v2 2025 年 4 月 |
| FSF | "DeepMind 的框架" | Frontier Safety Framework；CCLs；v3.0 2025 年 9 月 |
| ASL-3 | "生物安全 3 级类比" | Anthropic 针对 CBRN 相关能力的层级；2025 年 5 月激活 |
| CCL | "关键能力等级" | DeepMind 的阈值构造；按领域划分 |
| 安全论证 | "正式论证" | 书面论证，论证在最坏情况下部署是可接受安全的 |
| 调整条款 | "竞争者背叛许可" | 框架条款，允许在竞争对手未采取可比保障措施时降低要求 |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL 层级、路线图、AI R&D 细分
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — 五项标准、调整条款
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0、有害操纵
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 跨实验室比较
