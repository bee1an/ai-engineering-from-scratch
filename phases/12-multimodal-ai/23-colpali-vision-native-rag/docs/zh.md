# ColPali 与视觉原生文档 RAG

> 传统 RAG 将 PDF 解析为文本，分割成 chunk，嵌入 chunk，存储向量。每一步都丢失信号：OCR 丢弃图表数据，分块打断表格行，文本嵌入忽略图形。ColPali（Faysse et al.，2024 年 7 月）问了一个更简单的问题：为什么要提取文本？直接通过 PaliGemma 嵌入页面图像，用 ColBERT 风格的 late interaction 做检索，保留文档携带的所有版面、图形、字体和格式信号。公开 benchmark：在视觉丰富文档上端到端精度比 text-RAG 好 20-40%。ColQwen2、ColSmol 和 VisRAG 扩展了这个模式。本课阅读视觉原生 RAG 论点并构建一个小型 ColPali 风格索引器。

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Time:** ~180 minutes

## 学习目标

- 解释 bi-encoder 检索（每文档一个向量）与 late-interaction 检索（每文档多个向量）的区别。
- 描述 ColBERT 的 MaxSim 操作以及 ColPali 如何将其从文本 token 泛化到图像 patch。
- 构建一个小型 ColPali 风格索引器：页面 → patch embedding → 对 query-term embedding 做 MaxSim → top-k 页面。
- 对比 ColPali + Qwen2.5-VL 生成器 vs text-RAG + GPT-4 在发票/财务报告用例上的表现。

## 问题

PDF 上的 text-RAG 丢弃了文档的大部分内容。财务报告的 Q3 营收增长通常在图表中；医疗报告的发现在带注释的图像中；法律合同的签名栏是版面事实，不是文本事实。

Text-RAG 流水线：

1. PDF → 通过 OCR / pdftotext 提取文本。
2. 文本 → 300-500 token chunk。
3. Chunk → bi-encoder embedding（一个向量）。
4. 用户查询 → embedding → 余弦相似度 → top-k chunk。
5. Chunk + 查询 → LLM。

五个有损步骤。图表未捕获。表格跨 chunk 断裂。多栏版面被展平。图形注释消失。

ColPali 的修复：跳过 OCR，直接嵌入页面图像。用 ColBERT 风格的 late interaction 做检索，这样模型在查询时可以 attend 到细粒度 patch。

## 核心概念

### ColBERT (2020)

ColBERT（Khattab & Zaharia，arXiv:2004.12832）是一种文本检索方法。不是每文档一个向量，而是每 token 一个向量。查询时：

- 查询 token 获得自己的 embedding（N_q 个向量）。
- 文档 token 获得 embedding（N_d 个向量，通常缓存）。
- 分数 = 对查询 token 求和，对文档 token 取最大余弦相似度：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim 操作。每个查询 token "选择"其最佳匹配的文档 token。最终分数是总和。

优点：强召回，处理词级语义。缺点：每文档 N_d 个向量，存储昂贵。

### ColPali

ColPali（Faysse et al.，arXiv:2407.01449）将 ColBERT 模式应用于图像。

- 每页由 PaliGemma（ViT + language）编码为 patch embedding：每页 N_p 个向量。
- 每个用户查询（文本）编码为 query-token embedding：N_q 个向量。
- 分数 = Σ_i max_j cos(q_i, p_j)，即对 query-text-token 和 page-image-patch 做 MaxSim。
- 按总分检索 top-k 页面。

文档摄入时：用 PaliGemma 嵌入每页，存储所有 patch embedding。查询时：嵌入查询 token，对所有存储的页面 embedding 计算 MaxSim，返回 top-k 页面。

优点：在视觉丰富文档上端到端比 text-RAG 好 20-40%。每个 patch 向量捕获局部版面和内容。

缺点：每页 N_p patch × 4 字节浮点 × D 维向量 = 存储增长快。通过 PQ / OPQ 量化缓解。

### ColQwen2 和 ColSmol

ColQwen2（illuin-tech，2024-2025）将 PaliGemma 换为 Qwen2-VL。更好的基础编码器，更好的检索。

ColSmol 是用于本地/边缘的更小规模变体。约 1B 参数的 ColSmol 检索器可在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu et al.，arXiv:2410.10594）是另一种变体：不是对 patch 做 MaxSim，而是用 VLM 将每页池化为单一向量然后 bi-encoder 检索。更快的索引 + 更小的存储，更弱的召回。

质量 vs 成本权衡：ColPali 追求质量，VisRAG 追求规模。

### M3DocRAG

M3DocRAG（Cho et al.，arXiv:2411.04952）将多模态检索扩展到多页多文档推理。跨文档检索页面，为 VLM 组合多页上下文。

### ViDoRe——benchmark

ColPali 的配套 benchmark。Visual Document Retrieval Evaluation。任务包括财务报告、科学论文、行政文档、医疗记录、手册。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上得分约 80% nDCG@5；同一文档上的 text-RAG 得分约 50-60%。

### 端到端 RAG 流水线

视觉原生 RAG：

1. 摄入：PDF → 页面图像 → PaliGemma 编码 → 存储所有 patch embedding。
2. 查询：用户文本 → query-token embedding → 对所有索引页面做 MaxSim → top-k 页面。
3. 生成：top-k 页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 答案。

全程无 OCR。图形、图表、字体、版面全部流入答案。

### 存储数学

50 页财务报告，每页 729 patch，128 维 embedding：

- ColPali：50 * 729 * 128 * 4 bytes = 约 18 MB 原始，PQ 后约 4 MB。
- Text-RAG：50 chunk * 768 维 * 4 bytes = 约 150 kB。

ColPali 每文档约 30 倍存储。大规模下 OPQ / PQ 将其降到约 5-10 倍，通常可接受。

### Text-RAG 仍然胜出的场景

- 无版面信号的纯文本文档（wiki 文章、聊天记录）。Text-RAG 更简单且存储更便宜。
- 存储主导成本的百万页级档案。
- 严格监管要求可提取 OCR 文本伴随检索。

2026 年其他所有场景——财务报告、科学论文、法律合同、医疗记录、UX 文档——视觉原生 RAG 胜出。

## Use It

`code/main.py`：

- 玩具 patch 编码器：将"页面"（小型特征向量网格）映射为 patch embedding 数组。
- MaxSim 评分器：计算查询 token embedding 集和页面 patch 集之间的 ColBERT 风格分数。
- 索引 5 个玩具页面，运行 3 个查询，返回带分数的 top-k。

## Ship It

本课产出 `outputs/skill-vision-rag-designer.md`。给定一个文档 RAG 项目，选择 ColPali / ColQwen2 / VisRAG / text-RAG 并估算存储。

## 练习

1. 200 页年报，每页 729 patch，128 维 emb，4 字节浮点。计算原始存储和 PQ 压缩（8x）存储。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。这个求和捕获了简单均值相似度没有的什么？

3. ColPali 将页面索引为 patch 集。如果我们改为在词级别索引（如 ColBERT 所做）会怎样？权衡是什么？

4. 为 1M 页语料设计端到端流水线，每查询延迟预算 500ms。选择 ColQwen2 / VisRAG 并论证。

5. 阅读 M3DocRAG (arXiv:2411.04952)。描述多页注意力模式以及它与单页 ColPali 检索的区别。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Late interaction | "ColBERT 风格" | 使用 per-token 或 per-patch embedding + MaxSim 的检索，而非单一文档向量 |
| MaxSim | "Max-over-patches" | 对每个查询 token，选择最高相似度的文档 token；跨查询求和 |
| Bi-encoder | "单向量" | 每文档一个向量；更快但丢失粒度 |
| Multi-vector | "每文档多向量" | 每文档/页面存储 N_p 个向量；存储成本增长但召回提升 |
| Patch embedding | "页面特征" | VLM 编码器每个图像 patch 一个向量，按页缓存 |
| ViDoRe | "视觉文档 bench" | ColPali 的视觉文档检索 benchmark 套件 |
| PQ quantization | "乘积量化" | 在保持向量相似度的同时将存储缩小约 8 倍的压缩方法 |

## 延伸阅读

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
