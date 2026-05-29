# 毕业项目 04 — 多模态文档问答（视觉优先 PDF、表格、图表）

> 2026 年的文档问答前沿从 OCR-then-text 转向了视觉优先的 late interaction。ColPali、ColQwen2.5 和 ColQwen3-omni 将每个 PDF 页面视为图像，用多向量 late interaction 嵌入，让查询直接关注 patch。在金融 10-K、科学论文和手写笔记上，这种模式大幅超越 OCR 优先方案。端到端构建这个管道，索引 1 万页，并发布与 OCR-then-text 的对比报告。

**类型：** 毕业项目
**语言：** Python（管道），TypeScript（查看器 UI）
**前置要求：** Phase 4（计算机视觉）、Phase 5（NLP）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 17（基础设施）
**涉及阶段：** P4 · P5 · P7 · P11 · P12 · P17
**时间：** 30 小时

## 问题

企业坐拥大量被 OCR 管道搞乱的 PDF：旋转表格的扫描 10-K、方程密集的科学论文、只有作为图像才有意义的图表、手写批注。将这些视为文本优先意味着丢失一半信号。2026 年的答案是对原始页面图像做 late-interaction 多向量检索。ColPali（Illuin Tech）引入了这一方法；ColQwen2.5-v0.2 和 ColQwen3-omni 推高了准确率。在 ViDoRe v3 上，视觉优先检索以显著优势超越 OCR-then-text——在图表、表格和手写内容上差距更大。

代价是存储和延迟。一个 ColQwen 嵌入是每页约 2048 个 patch 向量，而非单个 1024 维向量。原始存储膨胀。DocPruner（2026）在几乎无精度损失的情况下实现 50% 剪枝。你将索引 1 万页，衡量 ViDoRe v3 nDCG@5，在 2 秒内提供答案，并直接与 OCR-then-text 基线对比。

## 概念

Late interaction 意味着每个查询 token 对每个 patch token 打分，每个查询 token 取最大分数后求和。你获得细粒度匹配而无需单个池化向量。多向量索引（Vespa、Qdrant multi-vector 或 AstraDB）存储每个 patch 的嵌入，在检索时运行 MaxSim。

回答器是一个视觉语言模型，接收查询加 top-k 检索页面作为图像，写出带证据区域（边界框或页面引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于方程和科学符号，OCR 回退（Nougat、dots.ocr）作为可选文本通道拼接进来。

评估是一个二维矩阵。一个轴：内容类型（纯文本段落、密集表格、柱状/折线图、手写笔记、方程）。另一个轴：检索方法（视觉优先 late interaction vs OCR-then-text vs 混合）。每个单元格获得 nDCG@5 和回答准确率。报告就是交付物。

## 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

- 页面渲染：PyMuPDF (fitz)，180 DPI，纵向归一化
- Late-interaction 模型：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 vidore 团队）
- 索引：Vespa 多向量字段，或 Qdrant multi-vector，或 AstraDB 带 MaxSim
- 剪枝：DocPruner 2026 策略（保留高方差 patch，50% 压缩，精度损失 < 0.5%）
- OCR 回退（方程/密集表格）：dots.ocr 或 Nougat
- VLM 回答器：自托管 Qwen3-VL-30B 或托管 Gemini 2.5 Pro；InternVL3 作为备选
- 评估：ViDoRe v3 基准，M3DocVQA 用于多页推理
- 查看器 UI：Next.js 15 带 canvas overlay 用于证据区域

## 构建步骤

1. **摄入。** 遍历 1 万页 PDF 语料，涵盖 10-K、科学论文和扫描文档。将每页渲染为 1536x2048 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **嵌入。** 对每个页面图像运行 ColQwen2.5-v0.2。输出形状约 2048 个 patch 嵌入，维度 128。应用 DocPruner 保留信号最强的一半。写入 Vespa 多向量字段或 Qdrant multi-vector。

3. **查询。** 对每个传入查询，用查询塔嵌入（token 级嵌入）。对索引运行 MaxSim：对每个查询 token，取与页面 patch 嵌入的最大点积，求和。返回 top-k 页面。

4. **合成。** 用查询和 top-5 页面图像调用 Qwen3-VL-30B。提示词："仅使用提供的页面回答。每个声明引用 (doc_id, page) 并命名区域（图表、表格、段落）。"

5. **证据区域。** 后处理答案以提取引用区域。如果 VLM 输出边界框（Qwen3-VL 支持），在查看器中渲染为叠加层。

6. **OCR 回退。** 对被识别为方程密集的页面（基于图像方差的启发式），运行 Nougat 或 dots.ocr，将 OCR 文本作为额外通道与图像一起传入。

7. **评估。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页 QA 准确率）。同时在相同语料上用相同合成器运行 OCR-then-text 管道。生成内容类型 × 方法矩阵。

8. **UI。** 先做 Streamlit 原型；然后 Next.js 15 生产查看器，带逐页证据区域叠加。

## 使用示例

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付标准

`outputs/skill-doc-qa.md` 描述交付物：一个视觉优先的多模态文档问答系统，针对特定语料调优，并在 ViDoRe v3 上与 OCR-then-text 基线对比评估。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 基准数字 vs OCR-text 基线和公开排行榜 |
| 20 | 证据区域定位 | 引用区域实际包含答案片段的比例 |
| 20 | 存储与延迟工程 | DocPruner 压缩比、索引 p95、回答 p95 |
| 20 | 多页推理 | 手工标注的 100 问多页集上的准确率 |
| 15 | 源文档检查体验 | 查看器清晰度、叠加保真度、并排对比工具 |
| **100** | | |

## 练习

1. 在相同语料上衡量 ColQwen2.5-v0.2 vs ColQwen3-omni。哪些页面一个对了另一个错了？在索引中添加"内容类别"标签以按类型路由。

2. 激进剪枝嵌入（75%、90%）。找到压缩悬崖：ViDoRe nDCG@5 降到 OCR 基线以下的点。

3. 构建混合方案：并行运行 OCR-then-text 和 ColQwen，用 RRF 融合，用交叉编码器重排序。混合方案是否超越任一单独方案？在哪里帮助最大？

4. 将 Qwen3-VL-30B 换成更小的 VLM（Qwen2.5-VL-7B）。衡量准确率/美元曲线。

5. 添加手写笔记支持。渲染手写语料，用 ColQwen 嵌入，衡量检索。与手写 OCR 管道对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Late interaction | "ColPali 风格检索" | 查询 token 独立对页面 patch 打分；MaxSim 聚合 |
| Multi-vector | "Per-patch 嵌入" | 每个文档有多个向量，而非一个池化向量 |
| MaxSim | "Late-interaction 打分" | 对每个查询 token，取与文档向量的最大相似度；求和 |
| DocPruner | "Patch 压缩" | 2026 剪枝方法，保留 50% patch 且精度损失可忽略 |
| ViDoRe v3 | "文档检索基准" | 2026 年衡量视觉文档检索的标准 |
| 证据区域 | "引用边界框" | 源页面上定位答案片段的 bbox |
| OCR 回退 | "方程通道" | 与视觉并行使用的文本管道，用于方程或表格密集页面 |

## 延伸阅读

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — 参考 late-interaction 文档检索
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — 基础方法论文
- [ColQwen family on Hugging Face](https://huggingface.co/vidore) — 生产就绪的 checkpoint
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html) — 参考服务栈
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 备选索引
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 备选托管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持方程的 OCR 回退
