# 内容审核系统 — OpenAI、Perspective、Llama Guard

> 生产内容审核系统将 Lessons 12-16 中定义的安全策略付诸实践。OpenAI Moderation API：`omni-moderation-latest`（2024）基于 GPT-4o 构建，一次调用分类文本+图像；在多语言测试集上比前一版本好 42%；响应 schema 返回 13 个类别布尔值 — harassment、harassment/threatening、hate、hate/threatening、illicit、illicit/violent、self-harm、self-harm/intent、self-harm/instructions、sexual、sexual/minors、violence、violence/graphic；对大多数开发者免费。分层模式：输入审核（生成前）、输出审核（生成后）、自定义审核（领域规则）。异步并行调用隐藏延迟；标记时使用占位响应。Llama Guard 3/4（Lesson 16）：14 个 MLCommons 危害类别、Code Interpreter Abuse、8 种语言（v3）、多图像（v4）。Perspective API（Google Jigsaw）：早于 LLM-as-moderator 浪潮的毒性评分；主要是单维度毒性，带有 severe-toxicity/insult/profanity 变体；内容审核研究的基线。弃用：Azure Content Moderator 2024 年 2 月弃用，2027 年 2 月退役，由 Azure AI Content Safety 替代。

**Type:** Build
**Languages:** Python (stdlib, three-layer moderation harness)
**Prerequisites:** Phase 18 · 16 (Llama Guard / Garak / PyRIT)
**Time:** ~60 minutes

## 学习目标

- 描述 OpenAI Moderation API 的类别分类法及其与 Llama Guard 3 MLCommons 集合的区别。
- 描述三层审核模式（输入、输出、自定义）并列出每层的一个失败模式。
- 描述 Perspective API 作为 LLM 前时代基线的定位及其在研究中仍被使用的原因。
- 陈述 Azure 弃用时间线。

## 问题

Lessons 12-16 描述了攻击和防御工具。Lesson 29 覆盖了在用户接触产品的表面将防御付诸实践的已部署审核系统。三层模式是 2026 年的默认配置。

## 概念

### OpenAI Moderation API

`omni-moderation-latest`（2024）。基于 GPT-4o 构建。一次调用分类文本+图像。对大多数开发者免费。

类别（响应 schema 中的 13 个布尔值）：
- harassment、harassment/threatening
- hate、hate/threatening
- self-harm、self-harm/intent、self-harm/instructions
- sexual、sexual/minors
- violence、violence/graphic
- illicit、illicit/violent

多模态支持适用于 `violence`、`self-harm` 和 `sexual`，但不适用于 `sexual/minors`；其余仅文本。

在 `code/main.py` 的代码框架中，我们为教学简洁性将 `/threatening`、`/intent`、`/instructions` 和 `/graphic` 子类别折叠到其顶级父类别中。生产代码应使用完整的 13 类别 schema。

在多语言测试集上比前一代审核端点好 42%。每类别分数；应用设置阈值。

### Llama Guard 3/4

在 Lesson 16 中覆盖。14 个 MLCommons 危害类别（与 OpenAI 的 13 个响应 schema 布尔值组织方式不同）。支持 8 种语言（v3）。Llama Guard 4（2025 年 4 月）原生多模态，12B。

OpenAI 和 Llama Guard 的分类法重叠但有分歧。OpenAI 有"illicit"作为宽泛类别；Llama Guard 分别有"violent crimes"和"non-violent crimes"。部署根据策略分类法的匹配度选择。

### Perspective API（Google Jigsaw）

早于 LLM-as-moderator 浪潮的毒性评分系统（2020 年前）。类别：TOXICITY、SEVERE_TOXICITY、INSULT、PROFANITY、THREAT、IDENTITY_ATTACK。单维度主分数（TOXICITY）带子维度变体。

作为内容审核研究基线被广泛使用，因为 API 稳定、文档完善，且有多年校准数据。对于现代 LLM 相关用例，Llama Guard 或 OpenAI Moderation 通常更合适。

### 三层模式

1. **输入审核。** 在生成前分类用户提示。如标记则拒绝。延迟：一次分类器调用。
2. **输出审核。** 在交付前分类模型输出。如标记则替换为拒绝。延迟：生成后一次分类器调用。
3. **自定义审核。** 领域特定规则（正则、白名单、业务策略）。在输入或输出端运行。

三层在设计上是顺序的：输入审核必须在生成前完成，输出审核在生成后运行。并行性适用于层内 — 在同一文本上并发运行多个分类器（如 OpenAI Moderation + Llama Guard + Perspective）可隐藏每个分类器的延迟。作为可选优化，可在输入审核完成和 token-1 流式传输延迟期间显示占位响应（"稍等，正在检查..."）。标记行为可配置：拒绝、净化、升级到人工审查。

### 失败模式

- **仅输入。** 无法捕获输出幻觉（Lesson 12-14 编码攻击绕过输入分类器）。
- **仅输出。** 允许任何输入到达模型；增加成本；向攻击者暴露内部推理。
- **仅自定义。** 跨类别不鲁棒；正则表达式脆弱。

分层是默认。双保险。

### Azure 弃用

Azure Content Moderator：2024 年 2 月弃用，2027 年 2 月退役。由 Azure AI Content Safety 替代，后者基于 LLM 并与 Azure OpenAI 集成。迁移是 Azure 部署的 2024-2027 年领域级项目。

### 在 Phase 18 中的位置

Lesson 16 在红队上下文中覆盖审核工具。Lesson 29 覆盖已部署的审核。Lesson 30 以当前双重用途能力证据收尾。

## Use It

`code/main.py` 构建了一个三层审核框架：输入审核器（关键词 + 类别分数）、输出审核器（对输出使用相同分类器）、自定义审核器（领域规则）。你可以运行输入通过并观察哪一层捕获了什么。

## Ship It

本课产出 `outputs/skill-moderation-stack.md`。给定一个部署，它推荐审核栈配置：输入端用哪个分类器、输出端用哪个、哪些自定义规则，以及边缘情况用什么 judge。

## 练习

1. 运行 `code/main.py`。将良性、边界和有害输入通过所有三层运行。报告每层对每个输入的触发情况。

2. 用 Perspective-API 风格的特定类别毒性评分扩展框架。比较其阈值行为与类别分数。

3. 阅读 OpenAI Moderation API 文档和 Llama Guard 3 类别列表。将每个 OpenAI 类别映射到最接近的 Llama Guard 类别。找出三个不能干净映射的类别。

4. 为代码助手部署（如 GitHub Copilot）设计审核栈。找出最相关和最不相关的类别，并提出自定义规则。

5. Azure Content Moderator 2027 年 2 月退役。规划迁移到 Azure AI Content Safety。找出迁移中风险最高的元素。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| OpenAI Moderation | "omni-moderation-latest" | 基于 GPT-4o 的 13 类别（文本）分类器，部分多模态支持 |
| Perspective API | "Google Jigsaw 毒性" | LLM 前时代的毒性评分基线 |
| Llama Guard | "MLCommons 14 类别" | Meta 的危害分类器（v3: 8B 文本, 8 语言; v4: 12B 多模态） |
| 输入审核 | "生成前过滤" | 模型调用前对用户提示的分类器 |
| 输出审核 | "生成后过滤" | 交付前对模型输出的分类器 |
| 自定义审核 | "领域规则" | 部署特定规则（正则、白名单、策略） |
| 分层审核 | "三层全用" | 标准生产部署模式 |

## 延伸阅读

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation 端点
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard 仓库
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 毒性评分
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure 替代方案
