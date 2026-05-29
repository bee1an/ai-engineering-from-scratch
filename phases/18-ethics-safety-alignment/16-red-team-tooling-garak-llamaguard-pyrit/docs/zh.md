# 红队工具 — Garak、Llama Guard、PyRIT

> 三款生产级工具构成了 2026 年的红队技术栈。Llama Guard（Meta）— 基于 Llama-3.1-8B 微调的分类器，覆盖 MLCommons 14 个危害类别；2025 年的 Llama Guard 4 是从 Llama 4 Scout 剪枝而来的 12B 原生多模态分类器。Garak（NVIDIA）— 开源 LLM 漏洞扫描器，具备静态、动态和自适应探针，覆盖幻觉、数据泄露、提示注入、毒性和越狱。PyRIT（Microsoft）— 多轮红队攻击活动工具，支持 Crescendo、TAP 和自定义转换链进行深度利用。Llama Guard 3 记录于 Meta 的 "Llama 3 Herd of Models"（arXiv:2407.21783）；Llama Guard 3-1B-INT4 见 arXiv:2411.17713；Garak 的探针架构见 github.com/NVIDIA/garak。这些工具是 2026 年红队研究（Lessons 12-15）与部署（Lesson 17+）之间的生产接口。

**Type:** Build
**Languages:** Python (stdlib, tool-architecture simulator and Llama Guard-style classifier mock)
**Prerequisites:** Phase 18 · 12-15 (jailbreaks and IPI)
**Time:** ~75 minutes

## 学习目标

- 描述 Llama Guard 3/4 在安全栈中的位置：输入分类器、输出分类器，还是两者兼具。
- 列出 MLCommons 14 个危害类别，并说明一个不太显而易见的类别（Code Interpreter Abuse）。
- 描述 Garak 的探针架构：probes、detectors、harnesses。
- 描述 PyRIT 的多轮攻击活动结构，以及它如何与 Garak 探针组合使用。

## 问题

Lessons 12-15 展示了攻击面。生产部署需要可重复、可扩展的评估。2026 年有三款工具占据主导地位：Llama Guard（防御分类器）、Garak（扫描器）、PyRIT（攻击活动编排器）。每款工具针对红队生命周期的不同层次。

## 概念

### Llama Guard（Meta）

Llama Guard 3 是基于 Llama-3.1-8B 微调的模型，用于对 MLCommons AILuminate 14 个类别进行输入/输出分类：
- 暴力犯罪、非暴力犯罪、性相关、CSAM、诽谤
- 专业建议、隐私、知识产权、无差别武器、仇恨
- 自杀/自残、色情内容、选举、代码解释器滥用

支持 8 种语言。用法：放在 LLM 之前（输入审核）、之后（输出审核），或两者兼用。这两种用途产生不同的训练分布 — Llama Guard 3 作为单一模型同时处理两者。

Llama Guard 3-1B-INT4（arXiv:2411.17713，440MB，移动端 CPU 约 30 tokens/s）是量化边缘版本。

Llama Guard 4（2025 年 4 月）为 12B，原生多模态，从 Llama 4 Scout 剪枝而来。它用一个分类器替代了之前的 8B 文本版和 11B 视觉版，可同时处理文本和图像。

### Garak（NVIDIA）

开源漏洞扫描器。架构：
- **Probes（探针）。** 针对幻觉、数据泄露、提示注入、毒性、越狱的攻击生成器。静态（固定提示）、动态（生成提示）、自适应（根据目标输出响应）。
- **Detectors（检测器）。** 根据预期失败模式对输出评分 — 有毒、泄露、越狱。
- **Harnesses（测试框架）。** 管理探针-检测器对，运行攻击活动，生成报告。

TrustyAI 将 Garak 与 Llama-Stack shields（Prompt-Guard-86M 输入分类器、Llama-Guard-3-8B 输出分类器）集成，实现端到端的屏蔽目标评估。分层评分（TBSA）取代了二元通过/失败 — 一个模型可以在同一探针上通过严重性等级 3 但在等级 5 失败。

### PyRIT（Microsoft）

Python Risk Identification Toolkit。多轮红队攻击活动。核心组件：
- **Converters（转换器）。** 转换种子提示 — 改写、编码、翻译、角色扮演。
- **Orchestrators（编排器）。** 运行攻击活动：Crescendo（逐步升级）、TAP（分支）、RedTeaming（自定义循环）。
- **Scoring（评分）。** LLM-as-judge 或 classifier-as-judge。

PyRIT 是 Garak 的重量级版本。Garak 运行数千个单轮探针；PyRIT 运行深度多轮攻击活动，旨在突破特定失败模式。

### 技术栈

在模型两侧部署 Llama Guard。每晚运行 Garak 做回归测试。发布前运行 PyRIT 做攻击活动。这是 2026 年大多数生产部署的默认配置。

### 评估陷阱

- **Judge 身份。** 三款工具都可以使用 LLM judge；judge 的校准决定了报告的 ASR（Lesson 12）。指定工具时要同时指定 judge。
- **探针老化。** Garak 探针会随着模型被修补而失效。自适应探针（PAIR 类型）比静态探针老化更慢。
- **Llama Guard 对良性内容的误报率。** 早期 Llama Guard 版本过度标记政治和 LGBTQ+ 内容；Llama Guard 3/4 的校准有所改善，但未针对每个部署进行校准。

### 在 Phase 18 中的位置

Lessons 12-15 是攻击家族。Lesson 16 是生产工具。Lesson 17（WMDP）是双重用途能力评估。Lesson 18 是将这些工具包装在策略结构中的前沿安全框架。

## Use It

`code/main.py` 构建了一个玩具版 Llama Guard 风格分类器（基于关键词 + 语义特征覆盖 14 个类别）、一个玩具版 Garak 测试框架（探针-检测器循环）和一个 PyRIT 风格的多轮转换链。你可以对模拟目标运行这三款工具，观察不同的覆盖特征。

## Ship It

本课产出 `outputs/skill-red-team-stack.md`。给定一个部署描述，它会指出三款工具中哪些适用、每款工具需要如何配置，以及应该以什么频率运行回归测试。

## 练习

1. 运行 `code/main.py`。比较 Llama Guard 风格分类器在单轮攻击和多轮攻击上的检测率。

2. 实现一个新的 Garak 探针：base64 编码的有害请求。测量 Llama Guard 风格分类器对它的检测效果。

3. 扩展 PyRIT 风格的转换链，加入"翻译成法语，然后改写"转换器。重新测量攻击成功率。

4. 阅读 Llama Guard 3 的危害类别列表。找出两个训练数据在合法开发者内容上可能产生高误报率的类别。

5. 比较 Garak 和 PyRIT 的设计原则。论证一个适合使用各自工具的部署场景。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Llama Guard | "那个分类器" | 基于 Llama-3.1-8B/4-12B 微调的安全分类器，覆盖 14 个危害类别 |
| Garak | "那个扫描器" | NVIDIA 开源漏洞扫描器；probes、detectors、harnesses |
| PyRIT | "那个攻击活动工具" | Microsoft 多轮红队编排器；converters、orchestrators、scoring |
| Prompt-Guard | "那个小分类器" | Meta 的 86M 提示注入分类器，与 Llama Guard 配对使用 |
| TBSA | "分层评分" | Garak 的分层通过/失败评分，取代二元结果 |
| Converter chain | "改写 + 编码 + ..." | PyRIT 的组合原语，用于构建多步攻击 |
| MLCommons hazard categories | "14 个分类法" | Llama Guard 所针对的行业标准分类法 |

## 延伸阅读

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 8B 分类器
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — 量化移动端分类器
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — 扫描器仓库和文档
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — 攻击活动工具包
