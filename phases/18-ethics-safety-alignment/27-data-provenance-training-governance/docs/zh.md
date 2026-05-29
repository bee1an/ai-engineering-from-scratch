# 数据溯源与训练数据治理

> EU AI Act 要求 GPAI 在 2025 年 8 月前实现机器可读的退出标准（通过 EU Copyright Directive TDM 例外）。加州 AB 2013（2024 年签署）— 生成式 AI 训练数据透明度要求开发者发布包含 12 个法定字段的数据集摘要。2025 年 DPA 在合法利益上的趋同：爱尔兰 DPC（2025 年 5 月 21 日）在 EDPB 意见后接受 Meta 对第一方公开 EU/EEA 成年用户内容进行 LLM 训练的计划（附保障措施）；科隆高等地区法院（2025 年 5 月 23 日）驳回禁令；汉堡 DPA 放弃紧急程序；UK ICO（2025 年 9 月 23 日）对 LinkedIn 的 AI 训练保障措施（透明度、简化退出、延长异议窗口）发出积极监管回应并继续监控 — 不是正式许可。巴西 ANPD（2024 年 7 月 2 日）因信息透明度不足暂停 Meta 的处理；预防措施在 Meta 提交合规计划后于 2024 年 8 月 30 日解除。关键不可逆性问题：cookie 同意框架设计用于实时、可逆的追踪；一旦数据进入模型权重，手术式擦除不可能 — 训练过的神经网络没有实际的 GDPR 删除权。合规窗口在收集时。Data Provenance Initiative (dataprovenance.org, Longpre, Mahari, Lee et al., "Consent in Crisis", 2024 年 7 月)：大规模审计显示随着出版商添加 robots.txt 限制，AI 数据公地正在快速萎缩。

**Type:** Learn
**Languages:** Python (stdlib, 12-field California AB 2013 scaffolding generator)
**Prerequisites:** Phase 18 · 24 (regulatory), Phase 18 · 26 (cards)
**Time:** ~60 minutes

## 学习目标

- 描述加州 AB 2013 生成式 AI 训练数据透明度的 12 个法定字段。
- 陈述 2025 年 DPA 对合法利益 LLM 训练的立场（爱尔兰 DPC、UK ICO、汉堡、科隆）。
- 描述不可逆性问题：为什么 GDPR 删除权对训练过的神经网络没有实际等价物。
- 陈述 Data Provenance Initiative 的 "Consent in Crisis" 发现。

## 问题

训练数据治理是每个模型卡（Lesson 26）和监管义务（Lesson 24）的上游。2024-2025 年，监管格局在三个原则上趋同：退出基础设施、每数据集披露、对公开可用数据的合法利益适应。在收集时不合规的提供者无法在下游补救。

## 概念

### 加州 AB 2013

2024 年签署。文档必须在 2026 年 1 月 1 日前发布，适用于 2022 年 1 月 1 日后发布的系统。第 3111(a) 条要求开发者发布训练中使用的数据集的高层摘要，包含 12 个法定项目：
1. 数据集的来源或所有者。
2. 数据集如何促进 AI 系统预期目的的描述。
3. 数据集中的数据点数量（可接受一般范围；动态数据集可估计）。
4. 数据点类型的描述（标注数据集的标签类型；未标注数据集的一般特征）。
5. 数据集是否包含受版权、商标或专利保护的数据，或完全属于公共领域。
6. 数据集是否被购买或许可。
7. 数据集是否包含个人信息（依据 Cal. Civ. Code §1798.140(v)）。
8. 数据集是否包含聚合消费者信息（依据 Cal. Civ. Code §1798.140(b)）。
9. 开发者的清洗、处理或其他修改及其预期目的。
10. 数据收集的时间段，如收集持续进行需注明。
11. 数据集首次用于开发的日期。
12. 系统是否使用或持续使用合成数据生成。

第 12 项（合成数据）相对于 Gebru et al. 2018 数据手册是新的。第 7 项（个人信息）触发隐私权法案（CPRA）义务。该法规豁免安全/完整性、航空器运营和仅联邦国家安全系统（第 3111(b) 条）。

### EU AI Act（Lesson 24）和 TDM 退出

EU Copyright Directive 的文本和数据挖掘例外允许对公开可用内容进行训练，除非权利人退出。EU AI Act GPAI 行为准则版权章节要求 GPAI 提供者尊重机器可读的退出信号（robots.txt、C2PA "No AI Training" 声明等）。

### 2025 年 DPA 在合法利益上的趋同

爱尔兰 DPC（2025 年 5 月 21 日）：在 EDPB 意见后接受 Meta 对第一方公开 EU/EEA 成年用户内容进行训练的计划（附保障措施）。科隆高等地区法院（2025 年 5 月 23 日）驳回对 Meta 的禁令：退出即足够。汉堡 DPA 为 EU 范围一致性放弃紧急程序。UK ICO（2025 年 9 月 23 日）发出积极监管回应 — 不是正式许可 — 对 LinkedIn 恢复 AI 训练的类似保障措施和持续监控。

趋同原则：合法利益可以为对公开可用第一方内容的训练提供正当理由（附退出机制）。不需要同意。

### 巴西 ANPD（2024 年 6 月）

因信息透明度不足暂停 Meta 对巴西用户数据的 AI 训练处理。与 EU DPA 不同的结果 — ANPD 优先考虑透明度而非合法利益的可接受性。

### 不可逆性问题

Cookie 同意设计用于实时、可逆的追踪。训练数据不同：一旦数据进入模型权重，手术式擦除不可能。从头重新训练是唯一完整的补救措施，而且成本高得令人望而却步。

部分补救：
- **遗忘（Unlearning）。** 近似移除；通过 MIA 测量（Lesson 22）。
- **基于影响函数的定位。** 识别受数据影响最大的权重；选择性更新。
- **微调抑制。** 训练模型拒绝源自该数据的输出。

没有一种完全解决问题。合规窗口在收集时。

### Data Provenance Initiative

dataprovenance.org。Longpre, Mahari, Lee et al. "Consent in Crisis"（2024 年 7 月）：AI 训练数据公地的大规模审计。发现：出版商正在以加速的速度添加 robots.txt 限制。可公开训练的公地正在快速萎缩。2023 -> 2024 年约 25% 的顶级训练来源添加了某种限制。含义：未来训练数据的可用性取决于新的获取范式（许可、合成生成、激励参与）。

### 在 Phase 18 中的位置

Lesson 26 是模型级文档。Lesson 27 是数据集级治理。两者共同定义了透明度层。Lesson 28 映射了研究这些问题的研究生态系统。

## Use It

`code/main.py` 为玩具数据集生成符合加州 AB 2013 的 12 字段数据集摘要脚手架。你可以填写字段并观察哪些字段触发隐私或版权后续义务。

## Ship It

本课产出 `outputs/skill-provenance-check.md`。给定训练中使用的数据集，它检查 AB 2013 12 字段覆盖、退出基础设施合规、DPA 对齐和不可逆性风险评估。

## 练习

1. 运行 `code/main.py`。为玩具数据集生成 12 字段摘要，并找出哪些字段规格不足。

2. EU Copyright Directive TDM 退出是机器可读的。提出退出信号的标准格式，并与 robots.txt 和 C2PA "No AI Training" 进行比较。

3. 阅读 Data Provenance Initiative 的 "Consent in Crisis"（2024 年 7 月）。描述三个限制最快的内容类别，并论证一个经济后果。

4. 2025 年 DPA 趋同接受公开内容训练的合法利益。构造一个合法利益不足的场景，并找出提供者需要的法律基础。

5. 草拟一个训练数据溯源清单，与 AB 2013 字段和每个数据集的 C2PA 签名溯源链组合。找出一个技术障碍和一个法律障碍。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| AB 2013 | "加州法律" | 生成式 AI 训练数据透明度；12 个法定字段 |
| TDM 例外 | "文本和数据挖掘" | EU Copyright Directive 训练数据例外（附退出） |
| 合法利益 | "EU 法律基础" | GDPR 第 6 条基础，可为公开内容训练提供正当理由 |
| 退出信号 | "机器可读的禁止训练" | robots.txt、C2PA "No AI Training"、TDM.Reservation |
| 不可逆性 | "无法反训练" | 模型权重中的数据不可手术式移除 |
| 遗忘 | "近似移除" | 减少模型对特定数据依赖的训练后干预 |
| Consent in Crisis | "DPI 审计" | 2024 年 7 月发现 robots.txt 限制加速增长 |

## 延伸阅读

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — 生成式 AI 训练数据透明度法
- [EU AI Act + GPAI Code of Practice (Lesson 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — 版权章节
- [Longpre, Mahari, Lee et al. — Consent in Crisis (dataprovenance.org, July 2024)](https://www.dataprovenance.org/consent-in-crisis-paper) — DPI 审计
- [IAPP — EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — 监管背景
