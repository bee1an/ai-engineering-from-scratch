# 对齐研究生态系统 — MATS、Redwood、Apollo、METR

> 五个组织定义了 2026 年非实验室对齐研究层。MATS（ML Alignment & Theory Scholars）：自 2021 年底以来 527+ 名研究者，180+ 篇论文，10K+ 引用，h-index 47；2024 年夏季队列注册为 501(c)(3)，约 90 名学者和 40 名导师；80% 的 2025 年前校友从事安全/安保工作，200+ 人在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。Redwood Research：由 Buck Shlegeris 创立的应用对齐实验室；引入了 AI Control（Lesson 10）；与 UK AISI 合作开展控制安全论证。Apollo Research：为前沿实验室进行部署前阴谋评估；撰写了 In-Context Scheming（Lesson 8）和 Towards Safety Cases for AI Scheming。METR（Model Evaluation and Threat Research）：基于任务的能力评估、自主任务时间跨度研究；"Common Elements of Frontier AI Safety Policies" 比较实验室框架。Eleos AI Research：模型福利部署前评估（Lesson 19）；进行了 Claude Opus 4 福利评估。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 01-27 (prior Phase 18 lessons)
**Time:** ~45 minutes

## 学习目标

- 识别非实验室对齐研究生态系统的五个组织及其核心产出。
- 描述 MATS 的规模（学者、论文、h-index）及其作为人才管道的角色。
- 描述 Redwood 的 AI Control 议程及其与 UK AISI 的合作。
- 描述 METR 基于任务的评估方法论。

## 问题

前沿实验室（Lesson 18）在内部进行安全评估并发布选定结果。实验室外部的生态系统是评估被验证的地方、新型失败模式首先被发现的地方、人才被培训的地方。理解生态系统有助于解读哪些研究发现被谁信任。

## 概念

### MATS（ML Alignment & Theory Scholars）

2021 年底启动。研究导师项目；学者与资深研究者共度 10-12 周，研究特定对齐问题。

规模（2026）：
- 自成立以来 527+ 名研究者。
- 180+ 篇论文发表。
- 10K+ 引用。
- h-index 47。
- 2024 年夏季：90 名学者 + 40 名导师；注册为 501(c)(3)。

职业成果：约 80% 的 2025 年前校友从事安全/安保工作。200+ 人在 Anthropic、DeepMind、OpenAI、UK AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

应用对齐实验室。由 Buck Shlegeris 创立。引入了 AI Control 议程（Lesson 10）。与 UK AISI 合作开展控制安全论证。为 DeepMind 和 Anthropic 提供评估设计建议。

代表性论文：Greenblatt, Shlegeris et al., "AI Control" (arXiv:2312.06942, ICML 2024)；Alignment Faking (Greenblatt, Denison, Wright et al., arXiv:2412.14093, 与 Anthropic 联合)。

风格：具体威胁模型、最坏情况对手、可压力测试的具体协议。

### Apollo Research

为前沿实验室进行部署前阴谋评估。撰写了 In-Context Scheming（Lesson 8, arXiv:2412.04984）。2025 年 OpenAI 反阴谋训练合作的伙伴。产出 Towards Safety Cases for AI Scheming（2024）。

风格：欺骗可能出现的 agent 设置评估；三支柱分解（错位、目标导向性、情境意识）。

### METR（Model Evaluation and Threat Research）

基于任务的能力评估。自主任务完成时间跨度研究。"Common Elements of Frontier AI Safety Policies"（metr.org/common-elements, 2025）比较实验室框架。

与 Apollo 共同撰写 AI Scheming 安全论证草案。

风格：长时间跨度任务评估、经验能力测量、框架综合。

### Eleos AI Research

模型福利部署前评估。进行了系统卡第 5.3 节中记录的 Claude Opus 4 福利评估。为 Lesson 19 的福利相关声明提供外部方法论检查。

### 流动

MATS 培训研究者。毕业生去 Anthropic、DeepMind、OpenAI（实验室安全团队）或去 Redwood、Apollo、METR、Eleos（外部评估）。外部评估者与实验室和 UK AISI / CAISI 合作。出版物反馈到生态系统中，为 MATS 下一队列提供素材。

### 为什么这一层很重要

单一来源的评估不可靠：实验室评估自己的模型存在结构性利益冲突。外部评估者可以提出和验证实验室可能低报的失败模式。2024 年 Sleeper Agents 论文（Lesson 7）是 Anthropic + Redwood；Alignment Faking 是 Anthropic + Redwood；In-Context Scheming 是 Apollo；Anti-Scheming 是 Apollo + OpenAI。多组织结构是质量控制。

### 在 Phase 18 中的位置

Lessons 7-11 引用 Redwood 和 Apollo 的工作；Lesson 18 引用 METR 的框架比较；Lesson 19 引用 Eleos。Lesson 28 是 Phase 其余部分所依赖的生态系统的显式组织地图。

## Use It

无代码。阅读 METR 的 "Common Elements of Frontier AI Safety Policies" 作为外部综合如何为实验室内部政策工作增加价值的示例。

## Ship It

本课产出 `outputs/skill-ecosystem-map.md`。给定一个对齐声明或评估，它识别组织、发表场所和方法论风格，并与已知对应组织进行交叉检查。

## 练习

1. 从 Lessons 7-15 中选一篇论文，识别涉及的组织。将作者与 MATS 校友和当前生态系统隶属关系进行交叉检查。

2. 阅读 METR 的 "Common Elements of Frontier AI Safety Policies"。找出他们强调的三个跨实验室趋同点和两个最大分歧。

3. MATS 职业成果约 80% 是安全/安保。论证这种选择压力是适应性的（培训领域）还是有偏的（过滤掉异端立场）。

4. Redwood 和 Apollo 都做控制/阴谋工作但风格不同。选择一个失败模式，描述每个组织会如何调查它。

5. Eleos AI 是唯一纯粹的模型福利组织。设计一个假设的第二个组织，聚焦于不同的福利相关问题（认知自由、机器人具身化等），并阐述其方法论。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| MATS | "导师项目" | ML Alignment & Theory Scholars；自 2021 年以来 527+ 名研究者 |
| Redwood Research | "控制实验室" | 应用对齐；AI Control 作者；UK AISI 合作伙伴 |
| Apollo Research | "阴谋评估" | 为前沿实验室进行部署前阴谋评估 |
| METR | "任务时间跨度评估" | 基于任务的能力评估；框架综合 |
| Eleos AI | "福利实验室" | 模型福利部署前评估 |
| 人才管道 | "MATS -> 实验室" | MATS 毕业生流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| 外部评估 | "非实验室检查" | 不由模型生产者进行的评估；增加可信度 |

## 延伸阅读

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — 导师项目
- [Redwood Research](https://www.redwoodresearch.org/) — AI Control 论文
- [Apollo Research](https://www.apolloresearch.ai/) — 阴谋评估
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 框架比较
- [Eleos AI Research](https://www.eleosai.org/research) — 模型福利方法论
