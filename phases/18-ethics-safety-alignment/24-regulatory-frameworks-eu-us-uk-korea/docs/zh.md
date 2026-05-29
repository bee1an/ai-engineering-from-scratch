# 监管框架 — 欧盟、美国、英国、韩国

> 四个主要监管体制定义了 2026 年的 AI 治理格局。EU AI Act（2024 年 8 月 1 日生效）— 2025 年 2 月 2 日起禁止性做法和 AI 素养；2025 年 8 月 2 日起 GPAI 义务；2026 年 8 月 2 日全面适用和第 50 条透明度；2027 年 8 月 2 日遗留 GPAI 和嵌入式高风险系统；罚款最高 1500 万欧元或全球营业额的 3%。GPAI 行为准则（2025 年 7 月 10 日）：三章 — 透明度、版权、安全与安保 — 12 项承诺；2026 年 8 月开始执行。UK AISI -> AI Security Institute（2025 年 2 月）：更名标志着更窄的范围。US AISI -> CAISI（2025 年 6 月）：NIST 下的 Center for AI Standards and Innovation；转向促增长立场。韩国 AI 框架法（2024 年 12 月通过，2026 年 1 月生效）：第 12 条在 MSIT 下设立 AISI；要求外国 AI 公司设立本地代表、风险评估、高影响和生成式 AI 的安全措施。

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 18 (frontier frameworks), Phase 18 · 27 (data governance)
**Time:** ~75 minutes

## 学习目标

- 描述 EU AI Act 的风险层级（禁止、高风险、通用、有限风险）和 2025 年 8 月 / 2026 年 8 月 / 2027 年 8 月的时间线。
- 描述 GPAI 行为准则的三章以及每章约束哪些提供者。
- 描述 2025 年的更名：UK AISI -> AI Security Institute；US AISI -> CAISI；每次更名对政策方向的含义。
- 陈述韩国 AI 框架法的核心条款。

## 问题

实验室框架（Lesson 18）是自愿的。监管框架是强制的。2024-2026 年期间，第一波全面 AI 监管开始生效。部署者必须将技术控制映射到监管义务；映射因司法管辖区而异。

## 概念

### EU AI Act

**2024 年 8 月 1 日生效。** 风险层级结构：

- **禁止性做法**（第 5 条）。社会评分、公共场所实时远程生物识别（有执法例外）、对弱势群体的剥削性操纵。2025 年 2 月 2 日适用。
- **高风险系统**（附件 III）。就业、教育、信贷、执法、司法、移民。需要合规评估、风险管理、日志记录、透明度。
- **通用 AI（GPAI）模型**。2025 年 8 月 2 日适用。所有 GPAI 提供者有义务；系统性风险 GPAI（训练计算 >1e25 FLOP）有额外义务。
- **有限风险系统**。第 50 条下的透明度义务（AI 生成内容标注）。2026 年 8 月 2 日适用。

时间线：
- 2025 年 2 月 2 日：禁止性做法 + AI 素养。
- 2025 年 8 月 2 日：GPAI + 治理。
- 2026 年 8 月 2 日：全面适用 + 第 50 条透明度 + 罚款最高 1500 万欧元 / 全球营业额 3%。
- 2027 年 8 月 2 日：遗留 GPAI + 嵌入式高风险。

欧盟委员会在 2025 年底提议将高风险时间线调整为 16 个月。

### GPAI 行为准则

2025 年 7 月 10 日发布。三章：

- **透明度。** 所有 GPAI 提供者。
- **版权。** 所有 GPAI 提供者。
- **安全与安保。** 系统性风险 GPAI 提供者（估计 5-15 家公司）。

共 12 项承诺。由 AI Office 主持的签署方工作组管理实施。2026 年 8 月 2 日开始执行；在此之前，善意合规即可。

### 第 50 条透明度守则

2025 年 12 月 17 日首稿。2026 年 3 月第二稿。2026 年 6 月最终版。覆盖 AI 生成内容标注，包括深度伪造 — 这是要求 Lesson 23 水印技术的监管层。

### UK AI Security Institute（2025 年 2 月）

从 AI Safety Institute 更名。更名缩窄了范围：放弃算法偏见和言论自由框架；聚焦前沿能力安全。开源了 Inspect 评估工具（2024 年 5 月）。与 Redwood（Lesson 10）合作开展控制安全论证。

### US CAISI（2025 年 6 月）

特朗普政府将 NIST 的 AI Safety Institute 转变为 Center for AI Standards and Innovation。转向"促增长 AI 政策"，引用副总统 Vance 在巴黎 AI 行动峰会上的发言。减少对部署前评估的强调；强调标准和创新支持。作为 EU AI Act 监管立场的国内对冲。

### 韩国 AI 框架法

2024 年 12 月通过。2025 年 1 月颁布。2026 年 1 月生效。整合了 19 项独立的 AI 法案。

第 12 条在科学技术信息通信部（MSIT）下设立 AISI。要求：
- 在韩国运营的外国 AI 公司设立本地代表。
- 对"高影响"AI 系统进行风险评估。
- 生成式 AI 和高影响 AI 的安全措施。

首个拥有全面水平 AI 监管的亚洲司法管辖区。

### 跨司法管辖区动态

- 欧盟：严格、风险分层、重罚。隐私相关监管的基准。
- 美国：偏向创新、去中心化，各州（如加州 AB 2013 — Lesson 27）填补联邦空白。
- 英国：聚焦安全、强评估基础设施。
- 韩国：MSIT 主导、聚焦外国提供者。

竞争性的监管哲学。在多个司法管辖区运营的部署者必须遵守最严格的规定，2026 年通常是 EU AI Act。

### 在 Phase 18 中的位置

Lesson 18 是实验室自愿治理；Lesson 24 是监管；Lesson 25 是 AI 系统的新兴 CVE 类别；Lessons 26-27 覆盖文档（卡片）和训练数据治理。

## Use It

无代码。阅读 EU AI Act 主要来源：法规文本、GPAI 行为准则、UK AISI Inspect 框架。将你的部署映射到每个司法管辖区的适用义务。

## Ship It

本课产出 `outputs/skill-regulatory-map.md`。给定一个部署描述，它映射适用的司法管辖区、每个管辖区中的层级分类、每个管辖区的义务，以及截止日期结构。

## 练习

1. 阅读 EU AI Act（法规 2024/1689）和 GPAI 行为准则（2025 年 7 月 10 日）。找出三项适用于所有 GPAI 提供者的义务和三项仅适用于系统性风险 GPAI 的义务。

2. 一个部署由美国公司制作，运行在欧盟基础设施上，服务韩国用户。哪三个司法管辖区的规则适用，每个实质性问题上哪条规则约束？

3. UK AI Security Institute 的更名缩窄了范围。论证支持和反对更窄框架。找出每个立场所依赖的政策假设。

4. CAISI 的"促增长"框架是对 2022-2024 年 AI 安全研究所模式的偏离。找出两个可测量的政策转变。

5. 韩国 AI 框架法要求外国提供者设立本地代表。描述对一家服务韩国用户的湾区公司的运营影响。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| EU AI Act | "那个法规" | 基于风险层级的水平 AI 监管；2024 年 8 月生效 |
| GPAI | "通用 AI" | 大型基础模型；系统性风险子集有额外义务 |
| 第 50 条 | "透明度义务" | AI 生成内容标注；2026 年 8 月适用 |
| UK AISI | "AI Security Institute" | 2025 年 2 月更名；更窄的前沿安全聚焦 |
| CAISI | "美国 AI 标准中心" | 2025 年 6 月从 AI Safety Institute 更名；促增长立场 |
| 韩国 AI 框架法 | "MSIT 水平监管" | 首部亚洲全面 AI 法律；2026 年 1 月生效 |
| 系统性风险 GPAI | "1e25 FLOP 阈值" | 额外义务层级；估计约束 5-15 家公司 |

## 延伸阅读

- [EU AI Act text (Regulation 2024/1689)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — 法规和时间线
- [GPAI Code of Practice (10 July 2025)](https://digital-strategy.ec.europa.eu/en/library/final-version-general-purpose-ai-code-practice) — 三章准则
- [UK AI Security Institute (renamed Feb 2025)](https://www.gov.uk/government/organisations/ai-security-institute) — 官方页面
- [CSET — South Korea AI Framework Act Analysis (2025)](https://cset.georgetown.edu/publication/south-korea-ai-law-2025/) — 韩国框架分析
