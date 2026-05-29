# 模型卡、系统卡与数据集卡

> 三种文档格式构成了 AI 透明度的结构。Model Cards（Mitchell et al. 2019）— 模型的"营养标签"：训练数据、量化分解分析、伦理考量、注意事项；Hugging Face 模型卡中仅 0.3% 记录了伦理考量（Oreamuno et al. 2023）。Datasheets for Datasets（Gebru et al. 2018, CACM）— 动机、组成、收集过程、标注、分发、维护；电子元件数据手册类比。Data Cards（Pushkarna et al., Google 2022）— 模块化分层细节（望远镜、潜望镜、显微镜）作为面向不同读者的边界对象。2024-2025 发展：通过 LLM 自动生成（CardGen, Liu et al. 2024）；模型卡详细程度与 HF 上最高 29% 的下载量增长相关（Liang et al. 2024）；可验证证明（Laminator, Duddu et al. 2024）；碳/水可持续性报告补充（Jouneaux et al. 2025 年 7 月）；EU/ISO 监管卡正在形成。System Cards（Sidhpurwala 2024；Meta 系统级透明度；"Blueprints of Trust" arXiv:2509.20394）— 端到端 AI 系统文档，覆盖安全能力、提示注入防护、数据外泄检测、与人类价值观的对齐。

**Type:** Build
**Languages:** Python (stdlib, model-card + datasheet + system-card generator)
**Prerequisites:** Phase 18 · 18 (safety frameworks), Phase 18 · 24 (regulatory)
**Time:** ~60 minutes

## 学习目标

- 描述 Mitchell et al. 2019 的原始模型卡和 Gebru et al. 2018 的数据手册。
- 描述 Data Cards 的望远镜/潜望镜/显微镜分层。
- 描述 System Cards 及其端到端覆盖。
- 陈述三个 2024-2025 发展（自动生成、可验证证明、可持续性报告）。

## 问题

监管框架（Lesson 24）和实验室安全政策（Lesson 18）都要求文档。文档格式从模型特定（模型卡）演进到数据集特定（数据手册）再到系统特定（系统卡）。每种针对不同范围的透明度。2024-2025 年的自动化和可验证证明工作解决了长期存在的采用问题。

## 概念

### Model Cards（Mitchell et al. 2019）

章节：
- 模型详情。
- 预期用途。
- 因素（评估相关的人口统计或环境因素）。
- 指标。
- 评估数据。
- 训练数据。
- 量化分析（按因素分解）。
- 伦理考量。
- 注意事项和建议。

采用问题：Oreamuno et al. 2023 对 Hugging Face 模型卡的审计发现仅 0.3% 记录了伦理考量。

### Datasheets for Datasets（Gebru et al. 2018）

电子元件数据手册类比。章节：
- 动机（为什么创建数据集）。
- 组成（里面有什么）。
- 收集过程（如何组装）。
- 标注（如适用）。
- 用途（预期、禁止、风险）。
- 分发。
- 维护。

2021 年发表于 CACM。数据手册是上游文档；模型卡依赖于数据手册的准确性。

### Data Cards（Pushkarna et al., Google 2022）

模块化分层细节。三个缩放级别：
- **望远镜（Telescopic）。** 面向非专家的高层摘要。
- **潜望镜（Periscopic）。** 面向 ML 从业者的中层概览。
- **显微镜（Microscopic）。** 面向审计员的详细特征级文档。

边界对象框架：不同读者从同一文档中提取不同信息。

### System Cards

范围：端到端 AI 系统，包括模型 + 安全栈 + 部署上下文。章节通常包括：
- 安全能力。
- 提示注入防护。
- 数据外泄检测。
- 与声明的人类价值观的对齐。
- 事件响应。

Sidhpurwala 2024 和 Meta 系统级透明度工作。"Blueprints of Trust"（arXiv:2509.20394）将 System Card 形式化为 Model Cards 的部署层补充。

### 2024-2025 发展

- **CardGen（Liu et al. 2024）。** 通过 LLM 自动生成模型卡；报告在标准化 Mitchell 2019 字段上比许多人工撰写的卡片具有更高的客观性。
- **下载量相关性（Liang et al. 2024）。** 详细的模型卡与 HF 上最高 29% 的下载量增长相关 — 采用压力现在是市场驱动的，不仅仅是合规驱动的。
- **Laminator（Duddu et al. 2024）。** 通过硬件 TEE / 加密签名的可验证证明 — 允许模型卡携带声明的证明，而不仅仅是声明。
- **可持续性（Jouneaux et al. 2025 年 7 月）。** 碳、水和计算能源足迹的补充；新兴 ISO 标准。
- **监管卡。** EU AI Act（Lesson 24）GPAI 行为准则透明度章节要求模型卡作为合规产物。

### 在 Phase 18 中的位置

Lessons 24-25 是监管和 CVE 层。Lesson 26 是文档层。Lesson 27 是训练数据治理，即数据手册的上游。Lesson 28 是产生卡片中引用的评估的研究生态系统。

## Use It

`code/main.py` 为一个玩具部署生成最小的模型卡、数据手册和系统卡。每个遵循规范的章节结构。你可以检查格式并比较三种范围。

## Ship It

本课产出 `outputs/skill-card-audit.md`。给定一个模型卡、数据手册或系统卡，它审计章节覆盖、数值分解，以及是否存在可验证证明。

## 练习

1. 运行 `code/main.py`。检查生成的卡片。找出薄弱的章节（仅占位符），并指定什么证据能加强它们。

2. 用跨两个人口统计群体的量化分解分析扩展模型卡（Lesson 20）。

3. 阅读 Oreamuno et al. 2023 关于 0.3% 采用率的论文。提出一项模型卡规范的结构性变更，以提高伦理考量的采用率。

4. Laminator（Duddu et al. 2024）使用 TEE 进行可验证证明。设计一个携带评估结果加密证明的模型卡字段，并描述验证者的角色。

5. 为你过去的一个项目或假设部署编写一个 System Card（系统卡，不是模型卡）。找出对第三方审计员最有价值的章节。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Model Card | "Mitchell 卡" | Mitchell et al. 2019 ML 模型标准文档 |
| Datasheet | "Gebru 数据手册" | Gebru et al. 2018 数据集标准文档 |
| Data Card | "Pushkarna 卡" | Google 2022 模块化分层数据文档 |
| System Card | "部署卡" | 端到端 AI 系统文档，包括安全栈 |
| 边界对象 | "不同读者，一份文档" | Data Cards 框架：同一文档服务不同受众 |
| 可验证证明 | "Laminator 证明" | 附加到文档声明的加密或 TEE 证明 |
| 可持续性字段 | "碳/水足迹" | 2025 年新兴的环境核算补充 |

## 延伸阅读

- [Mitchell et al. — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — 经典模型卡
- [Gebru et al. — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — 数据手册论文
- [Pushkarna et al. — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — 分层数据文档
- [Sidhpurwala et al. — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — System Card 形式化
