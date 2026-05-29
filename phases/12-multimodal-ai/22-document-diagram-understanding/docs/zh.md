# 文档与图表理解

> 文档不是照片。PDF、科学论文、发票或手写表单有版面、表格、图表、脚注、标题和语义结构，这些是普通图像理解无法捕获的。VLM 之前的技术栈是流水线：Tesseract OCR + LayoutLMv3 + 表格提取启发式。VLM 浪潮用 OCR-free 模型取代了它——Donut (2022)、Nougat (2023)、DocLLM (2023)——直接输出结构化标记。到 2026 年前沿就是"把页面图像以 2576px 原生分辨率喂给 Claude Opus 4.7"，结构化标记输出免费获得。本课阅读文档 AI 的三个时代演进。

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180 minutes

## 学习目标

- 解释文档 AI 的三个时代：OCR 流水线、OCR-free、VLM 原生。
- 描述 LayoutLMv3 的三个输入流：文本、版面（bbox）、图像 patch，带统一 masking。
- 对比 Donut（OCR-free，图像 → 标记）、Nougat（科学论文 → LaTeX）、DocLLM（版面感知生成式）、PaliGemma 2（VLM 原生）。
- 为新任务（发票、科学论文、手写表单、中文收据）选择文档模型。

## 问题

"理解这个 PDF"看似简单实则很难。信息存在于：

- 文本内容（90% 的信号）。
- 版面（标题、脚注、侧边栏、双栏格式）。
- 表格（行、列、合并单元格）。
- 图形和图表。
- 手写注释。
- 字体和排版（标题 vs 正文）。

原始 OCR 倾倒文本并丢失其余。一个关心发票的系统需要知道"Total: $1,245"来自右下角，而不是脚注。

## 核心概念

### 时代 1——OCR 流水线（2021 年前）

经典技术栈：

1. PDF → 每页图像。
2. Tesseract（或商业 OCR）提取文本及每词边界框。
3. 版面分析器识别块（标题、表格、段落）。
4. 表格结构识别器解析表格。
5. 领域规则 + 正则提取字段。

对干净印刷文本有效。在手写、倾斜扫描、复杂表格、非英语脚本上失败。每种故障模式都需要自定义异常路径。

### TrOCR (2021)

TrOCR（Li et al.，arXiv:2109.10282）用在合成 + 真实文本图像上训练的 transformer encoder-decoder 替换了 Tesseract 的经典 CNN-CTC。在手写和多语言文本上干净胜出。仍然是流水线（检测器然后 TrOCR 然后版面），但 OCR 步骤大幅改善。

### 时代 2——OCR-free (2022-2023)

第一批 OCR-free 模型说：完全跳过检测，直接将图像像素映射到结构化输出。

Donut（Kim et al.，arXiv:2111.15664）：
- Encoder-decoder transformer，encoder 是 Swin-B。
- 输出是表单理解的 JSON、摘要的 markdown 或任何任务专属 schema。
- 无 OCR、无版面、无检测。

Nougat（Blecher et al.，arXiv:2308.13418）：
- 专门在科学论文上训练。
- 输出是 LaTeX / markdown。
- 处理公式、多栏版面、图形。
- 每个 arXiv 解析器都调用的模型。

这些是专家，不是通才。Donut 在科学论文上失败；Nougat 在发票上失败。

### LayoutLMv3 (2022)

另一条路线。LayoutLMv3（Huang et al.，arXiv:2204.08387）保留 OCR 但添加版面理解：

- 三个输入流：OCR 文本 token、每 token 的 2D 边界框、图像 patch。
- 跨三个模态的 masked 训练目标（masked 文本、masked patch、masked 版面）。
- 下游：分类、实体提取、表格 QA。

LayoutLMv3 是基于 OCR 的文档理解的巅峰。在表单和发票上强。需要上游 OCR。在标准化文档 benchmark 上是 VLM 之前最佳精度。

### DocLLM (2023)

DocLLM（Wang et al.，arXiv:2401.00908）是 LayoutLM 的生成式兄弟。根据版面 token 生成自由形式答案。对文档 QA 更好；仍依赖 OCR 输入。

### 时代 3——VLM 原生 (2024+)

2024 年 VLM 变得足够好，可以完全替代流水线。将完整页面图像以高分辨率喂给 VLM，问问题，得到答案。

- LLaVA-NeXT 336-tile AnyRes 适用于小文档。
- Qwen2.5-VL 动态分辨率原生处理 2048+ 像素。
- Claude Opus 4.7 支持 2576px 文档。
- PaliGemma 2（2025 年 4 月）专门为文档 + 手写训练。

VLM 原生与 OCR 流水线之间的差距迅速缩小。到 2026 年，VLM 原生在以下方面胜出：

- 场景文本（手写 + 印刷，混合脚本）。
- 带合并单元格的复杂表格。
- 嵌入文本中的数学公式。
- 带文本注释的图形。

OCR 流水线仍在以下方面胜出：

- 大规模纯扫描工作负载，每页延迟很重要。
- 流水线可靠性（确定性故障 vs VLM 幻觉）。
- 需要可审计 OCR 输出的监管环境。

### Claude 4.7 / GPT-5 前沿

在 2576 像素原生输入下，前沿 VLM 以接近人类的精度做文档理解。2026 年初的 benchmark 数字：

- DocVQA：Claude 4.7 ~95.1，PaliGemma 2 ~88.4，Nougat ~77.3，流水线 LayoutLMv3 ~83。
- ChartQA：Claude 4.7 ~92.2，GPT-4V ~78。
- VisualMRC：Claude 4.7 ~94。

闭源模型差距主要是分辨率和基础 LLM 规模。7B 开源模型落后几个点但在追赶。

### 数学公式和 LaTeX 输出

科学论文需要精确的 LaTeX 公式输出。Nougat 在此上训练。用 LaTeX 目标训练的 VLM（Qwen2.5-VL-Math、Nougat 衍生物）产出可用的 LaTeX。没有显式 LaTeX 训练的 VLM 产出可读但不精确的转录。

2026 年科学论文流水线：在 PDF 上链式调用 Nougat，然后对棘手页面用 VLM。

### 手写

仍然是最难的子任务。混合印刷 + 手写（医生笔记、填写的表单）是 OCR 流水线在成本上仍优于 VLM 的地方。纯手写 VLM 在改善（Claude 4.7、PaliGemma 2）。

### 2026 配方

对于新的文档 AI 项目：

- 大规模纯印刷发票：LayoutLMv3 + 规则，成本高效。
- 混合文档（科学 + 手写 + 表单）：VLM 原生（PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv 摄入：Nougat 处理数学，VLM 处理图形。
- 监管：OCR 流水线 + VLM 验证器交叉检查。

## Use It

`code/main.py`：

- 一个玩具版面感知 tokenizer：给定（文本，bbox）对，产出 LayoutLMv3 风格输入。
- 一个 Donut 风格任务 schema 生成器：表单的 JSON 模板。
- 跨 OCR 流水线、Donut、Nougat 和 VLM 原生的每页 token 预算对比。

## Ship It

本课产出 `outputs/skill-document-ai-stack-picker.md`。给定一个文档 AI 项目（领域、规模、质量、监管），在 OCR 流水线、OCR-free 专家和 VLM 原生之间选择。

## 练习

1. 你的项目是每天 1000 万张发票。哪个技术栈在不丢失精度的情况下最小化每页成本？

2. 为什么 LayoutLMv3 在表单 QA 上优于纯 CLIP-VLM 但在场景文本上不如？bbox 流放弃了什么？

3. Nougat 生成 LaTeX。提出一个 VLM 原生输出在 LaTeX 保真度上超过 Nougat 的测试用例，和一个 Nougat 胜出的用例。

4. 阅读 PaliGemma 2 论文（Google，2024）。相比 PaliGemma 1，提升文档精度的关键训练数据添加是什么？

5. 设计一个监管安全的混合方案：OCR 流水线为主，VLM 为辅交叉检查。如何解决分歧？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| OCR 流水线 | "Tesseract 风格" | 分阶段栈：检测 -> OCR -> 版面 -> 规则；确定性，脆弱 |
| OCR-free | "Donut 风格" | 跳过显式 OCR 的图像到输出 transformer；单一模型 |
| 版面感知 | "LayoutLM" | 输入包含每 token bbox 坐标；跨模态统一 masking |
| VLM 原生 | "前沿 VLM" | 将页面图像直接以高分辨率喂给 Claude/GPT/Qwen VLM；无流水线 |
| DocVQA | "文档 benchmark" | 文档 VQA 标准；最常引用的分数 |
| Markup 输出 | "LaTeX / MD" | 结构化输出格式而非自由形式文本；支持下游自动化 |

## 延伸阅读

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
