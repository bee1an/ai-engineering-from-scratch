# Visual Autoregressive Modeling (VAR): Next-Scale Prediction

> Diffusion 模型在时间维度上迭代采样（去噪步骤）。VAR 在尺度维度上迭代采样——它先预测一个 1x1 token，然后 2x2，再 4x4，逐步提升到最终分辨率，每个尺度都以前一个尺度为条件。2024 年的论文表明 VAR 在图像生成上符合 GPT 式的 scaling law，并在相同计算预算下超越 DiT。本课构建其核心机制。

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## 问题

Autoregressive 生成主导了语言建模，因为它的扩展性可预测：更多计算、更多参数、更低困惑度、更好的输出。2024 年之前，图像生成有两种主要的 AR 尝试：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE code 上逐 token）。

两者都受困于生成顺序问题。像素和 token 排列在 2D 网格中，但 AR 模型必须按 1D 光栅顺序逐个访问。早期角落的像素完全不知道图像最终会变成什么样。生成质量的扩展性不如 GPT-on-text，在匹配计算量下也从未达到 diffusion 模型的质量。

VAR 通过改变生成对象来解决生成顺序问题。VAR 不是在空间中逐个预测图像 token，而是以递增分辨率预测整幅图像。第 1 步：预测一个 1x1 token（图像的整体"摘要"）。第 2 步：预测一个 2x2 token 网格（较粗的特征）。第 3 步：预测一个 4x4 网格。第 K 步：预测最终的 (H/8)x(W/8) 网格。

每个尺度都关注所有先前尺度（在"尺度顺序"上因果），并在自身尺度内并行。顺序问题消失了：尺度 k 的整幅图像在一次 transformer pass 中生成。

## 概念

### VQ-VAE Multi-Scale Tokenizer

VAR 需要一个 **multi-scale discrete tokenizer**。对于图像 x，它生成一系列分辨率逐步提高的 token 网格：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 z_k 使用相同的 codebook（典型大小 4096-16384）。各尺度的 tokenization 并非独立——它经过训练使得各尺度残差之和能重建 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是一种 **residual VQ** 变体。尺度 k 捕获尺度 1..k-1 遗漏的内容。Decoder 对所有尺度的 embedding 求和后生成图像。

Multi-scale VQ tokenizer 训练一次（类似 VQGAN）然后冻结。所有生成工作由上层的 autoregressive 模型完成。

### Next-Scale Prediction

生成模型是一个 transformer，它看到所有先前尺度的 token 并预测下一个尺度的 token。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Position embedding 同时编码尺度索引和尺度内的空间位置。Attention 在尺度顺序上是因果的：尺度 k、位置 (i, j) 的 token 可以关注尺度 1..k 的所有 token，以及尺度 k 内按某种尺度内顺序排在前面的 token（VAR 使用固定位置 attention，无尺度内因果性——同一尺度内所有位置并行预测）。

训练损失：在每个尺度 k，给定所有先前尺度的 token 预测 z_k。对离散 VQ code 计算 cross-entropy loss。结构与 GPT 相同，只是"序列"现在是尺度结构化的。

### 生成

推理时：
```
generate z_1 = sample from p(z_1)                    # 1 个 token
generate z_2 = sample from p(z_2 | z_1)              # 4 个 token 并行
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 个 token 并行
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

对于 K = 10 个尺度，生成需要 10 次 transformer forward pass。每次 pass 并行生成整个尺度——尺度内无逐 token autoregression。对于 256x256 图像，大约是 10 次 pass，而 DiT 需要 28-50 次。

### 为什么 Next-Scale 优于 Next-Token

三个结构性优势：
1. **从粗到细符合自然图像统计特性。** 人类视觉感知和图像数据集都表现出尺度相关的规律性：低频结构稳定且可预测；高频细节以低频内容为条件。Next-scale prediction 利用了这一点。
2. **尺度内并行生成。** 与 GPT 式逐 token AR 不同，VAR 在一步中生成一个尺度的所有 token。有效生成长度是对数级而非线性级。
3. **无生成顺序偏差。** 尺度 k 的 token 能看到尺度 k-1 的全部内容；不存在"左侧"或"上方"偏差迫使早期 token 在晚期上下文可用之前就做出承诺。

### Scaling Law

Tian et al. 证明 VAR 在 ImageNet 上的 FID 遵循幂律 scaling 曲线——就像 GPT 的困惑度一样。参数或计算量翻倍可靠地将误差减半。这是第一个像语言模型一样清晰地展现这种 scaling 行为的图像生成模型。结果是 VAR 的预测可以从计算量推导，而非依赖每种架构的经验猜测。

### 与 Diffusion 的关系

VAR 和 diffusion 共享相同的数据压缩故事：两者都将生成问题分解为一系列更简单的子问题。

- Diffusion：逐步添加噪声，学习撤销一步。
- VAR：逐步增加分辨率，学习预测下一个尺度。

它们是穿越问题的不同轴。两者都产生可处理的条件分布。实验上 VAR 推理更快（更少的 pass，尺度内全部并行），在 class-conditional ImageNet 上匹配或超越 DiT。Text-conditional VAR（VARclip、HART）是活跃的研究方向。

## Build It

在 `code/main.py` 中你将：
1. 在合成"图像"数据（2D Gaussian rings）上构建一个小型 **multi-scale VQ tokenizer**。
2. 训练一个 **VAR 式 transformer** 进行 next-scale prediction。
3. 调用 transformer 4 次（4 个尺度）进行采样并解码。
4. 验证尺度有序训练使得尺度内生成可以并行。

这是一个 toy 实现。重点是看到尺度结构化的 attention mask 和尺度内并行生成实际运作。

## Ship It

本课产出 `outputs/skill-var-tokenizer-designer.md`——一个用于设计 multi-scale tokenizer 的 skill：尺度数量、尺度比例、codebook 大小、残差共享、decoder 架构。

## 练习

1. **尺度数量消融。** 用 4、6、8、10 个尺度训练 VAR。测量重建质量与 autoregressive pass 数量的关系。更多尺度 = 更细的残差 = 更好的质量但更多 pass。

2. **Codebook 大小。** 用 codebook 大小 512、4096、16384 训练 tokenizer。更大的 codebook 重建更好但预测更难。找到拐点。

3. **尺度内并行验证。** 对训练好的 VAR，显式测量 attention pattern。在尺度 k 内，模型是否关注跨尺度位置而非尺度内位置？验证 mask 实现。

4. **VAR vs DiT scaling。** 对相同的 ImageNet class-conditional 任务，在匹配参数预算（如 33M、130M、458M）下训练 VAR 和 DiT。绘制 FID vs 计算量。VAR 应在每个规模上领先 DiT——在小规模上复现论文结果。

5. **Text conditioning。** 扩展 VAR 以通过 adaLN 接收 text embedding（CLIP pooled）作为额外条件输入。这是 HART 的方案。FID 在 text-aligned sampling 上改善多少？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| VAR | "Visual AutoRegressive" | 通过在 VQ token 网格金字塔上进行 next-scale prediction 来生成图像 |
| Next-scale prediction | "先预测粗的，再预测细的" | 模型以递增分辨率尺度预测 token，以所有先前尺度为条件 |
| Multi-scale VQ tokenizer | "Residual VQ" | 产生 K 个分辨率递增的 token 网格的 VQ-VAE，decoder 对所有尺度求和 |
| Scale k | "金字塔第 k 层" | K 个分辨率级别之一，从 k=1 时的 1x1 到 k=K 时的 (H/p)x(W/p) |
| Parallel-within-scale | "每个尺度一次 forward" | 尺度 k 的所有 token 在一次 transformer pass 中预测，非 autoregressively |
| Causal-across-scales | "尺度有序 attention" | 尺度 k 的 token 可以关注尺度 1..k 但不能关注尺度 k+1..K |
| Residual VQ | "加法式 tokenization" | 每个尺度的 token 编码低尺度遗留的残差；decoder 对所有尺度 embedding 求和 |
| VAR scaling law | "图像 GPT scaling" | FID 在计算量上遵循可预测的幂律，类似语言模型的困惑度 |
| HART | "Hybrid VAR + text" | Text-conditional VAR 变体，结合 MaskGIT 式迭代解码与 VAR 的尺度结构 |
| Scale position embedding | "(scale, row, col) 三元组" | 位置编码同时携带尺度索引和尺度内的空间坐标 |

## 延伸阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 论文，权威参考
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，diffusion 对比基线
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 的 multi-scale tokenizer 所扩展的 tokenizer 家族
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，离散图像 tokenization 的基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — text-conditional VAR
