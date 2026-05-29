# Chameleon 与 Early-Fusion 纯 Token 多模态模型

> 到目前为止我们看到的每个 VLM 都将图像和文本分开处理。视觉 token 来自视觉编码器，流经投影层，然后在 LLM 内部与文本相遇。视觉和文本词表从不重叠。Chameleon（Meta, 2024 年 5 月）提出了一个问题：如果它们重叠呢？训练一个 VQ-VAE 将图像转换为来自共享词表的离散 token 序列。每个多模态文档现在都是一个序列——文本 token 和图像 token 交织在一起，一个自回归 loss。副作用：模型可以生成混合模态输出——在单次推理调用中交替产生文本和图像 token。本课阅读 early-fusion 论点并端到端构建一个玩具版本。

**Type:** Build
**Languages:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Phase 12 · 05, Phase 8 (Generative AI)
**Time:** ~180 minutes

## 学习目标

- 解释为什么共享词表 + 单一 loss 改变了模型能做什么。
- 描述 VQ-VAE 如何将图像 tokenize 为与 transformer next-token 目标兼容的离散序列。
- 说出 Chameleon 的训练稳定性技巧：QK-Norm、dropout 放置、LayerNorm 顺序。
- 比较 Chameleon 与 BLIP-2 的 Q-Former 方法，描述各自适用的场景。

## 问题

基于 adapter 的 VLM（LLaVA、BLIP-2、Qwen-VL）将文本和图像视为两种不同的东西。文本 token 经过 `embed(text_token)`；图像经过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两条输入路径，在中途合并。

三个后果：

1. LLM 只能消费图像，不能输出图像。输出仅限文本。
2. 混合模态文档（如文章中交替出现的段落和图片）处理起来很别扭——你要么在模型外部解析多模态输入，要么链式生成。
3. 分布不匹配。视觉 token 和文本 token 处于隐藏空间的不同区域，产生微妙的对齐问题。

Chameleon 拒绝了这个前提：图像就是来自共享词表的离散 token 序列。在交织文档上训练模型，一个 loss，一个自回归解码器，你就免费解锁了混合模态生成。

## 概念

### VQ-VAE 作为图像 tokenizer

tokenizer 是一个向量量化变分自编码器。架构：

- 编码器：CNN + ViT 将图像映射为空间特征图，比如 32x32 个 256 维特征。
- Codebook：K 个学习到的向量（Chameleon 使用 8192 个），同样是 256 维。
- 量化：对每个空间特征，通过 L2 距离查找最近的 codebook 条目。用整数索引替换连续特征。
- 解码器：CNN 将量化特征还原为像素。

训练：VAE 重建 loss + commitment loss + codebook loss。Codebook 索引构成图像的离散字母表。

对于 Chameleon：一张图像变成 32*32 = 1024 个 token，取自 8192 大小的词表。与文本 token（来自 LLM 的 BPE 词表，比如 32000）拼接。最终词表：40192。Transformer 看到一个序列，一个 loss。

### 共享词表

Chameleon 的词表组合了文本 token、图像 token 和模态分隔符。每个 token 有一个唯一 ID。输入 embedding 层将每个 ID 映射为 D 维隐向量。输出投影将隐向量映射回词表 logits。Softmax 选择下一个 token，无论什么模态。

分隔符很重要：`<image>` 和 `</image>` 标签括住图像 token 序列。在生成时，如果模型输出 `<image>`，下游软件就知道接下来的 1024 个 token 是 VQ 索引，需要送到解码器进行像素渲染。

### 混合模态生成

推理就是在共享词表中做 next-token prediction。示例 prompt："Draw a cat and describe it." Chameleon 输出：

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

模型自主选择顺序——它可能先产生图像再产生文本，先文本再图像，或者交织。同一个解码器，同一个 loss。

对比 adapter VLM 中生成仅限文本的情况。Chameleon 重新打开了模型输出模态的问题。

### 训练稳定性——QK-Norm、dropout、LayerNorm 顺序

Early-fusion 训练在大规模下不稳定。Chameleon 的论文记录了三个技巧：

- QK-Norm。在 attention 内部对 query 和 key 投影应用 LayerNorm，在点积之前。防止 logit 幅度在深层爆炸。被 2024 年后的多个大模型采用。
- Dropout 放置。在每个 residual-add 之后都加 dropout，而不仅仅在 attention 和 MLP 之后。当图像 token 的梯度可能主导时需要更多正则化。
- LayerNorm 顺序。残差分支上的 Pre-LN（标准做法），加上最后一个 block 的 skip connection 上额外的 LN。稳定最后一层的梯度流。

没有这些技巧，34B 参数的 Chameleon 训练在多个 checkpoint 处发散。有了它们，就能收敛。训练方案与架构本身同样重要。

### Tokenizer 的重建上限

VQ-VAE 是有损的。在 8192 个 codebook 条目和每张 512x512 图像 1024 个 token 的设置下，重建 PSNR 上限约为 26-28 dB。这足以生成可辨认的图像，但明显不如连续空间扩散模型（Stable Diffusion 3 达到 32+ dB）。

Tokenizer 是瓶颈。更好的 tokenizer（MAGVIT-v2、IBQ、SBER-MoVQGAN）可以提升上限。Emu3（Lesson 12.12）仅通过更好的 tokenizer 就达到了 SDXL 级别的生成质量。

### Chameleon vs BLIP-2 / LLaVA

Chameleon（early fusion，共享词表）：
- 一个 loss，一个解码器。
- 生成混合模态输出。
- Tokenizer 是质量上限。
- 昂贵：推理路径上每张生成图像都需要 VQ-VAE 解码器。

BLIP-2 / LLaVA（late fusion，独立塔）：
- 视觉输入，仅文本输出。
- 复用预训练 LLM。
- 理解任务没有 tokenizer 瓶颈。
- 便宜：单次前向传播。

按任务选择。如果你需要图像生成，选 Chameleon 家族。如果只需要理解，adapter-VLM 更简单且复用更多预训练计算。

### Fuyu 和 AnyGPT

Fuyu（Adept, 2023）是一种相关方法：完全跳过独立的视觉编码器，将原始图像 patch 通过 LLM 的输入投影层当作 token 输入，没有 tokenizer。比 Chameleon 更简单，但失去了共享词表的输出生成能力。

AnyGPT（Zhan et al., 2024）将 Chameleon 扩展到四种模态：文本、图像、语音、音乐。每种模态使用相同的 VQ-VAE 技巧，共享 transformer。任意到任意生成。在 Lesson 12.16 中有更多介绍。

## Use It

`code/main.py` 构建了一个玩具级端到端 early-fusion 模型：

- 一个微型 VQ-VAE 风格量化器，将 8x8 patch 映射到 codebook 索引（K=16）。
- 一个共享词表：（text ids 0..31）+（image ids 32..47）+（separators 48, 49）。
- 一个玩具自回归解码器（bigram 表），在合成 caption + 图像 token 序列上训练。
- 给定 prompt 输出交替文本 + 图像 token 的采样循环。

代码故意将 transformer 保持极小（bigrams），以便你可以端到端追踪信号流。

## Ship It

本课产出 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定产品规格（仅理解 vs 理解 + 生成、所需图像质量、成本预算），它在 Chameleon 家族（early fusion）和 LLaVA 家族（late fusion）之间选择，并用定量经验法则进行论证。

## 练习

1. Chameleon 使用 K=8192 个 codebook 条目和每张 512x512 图像 1024 个 token。估算相对于 24 位 RGB 图像的压缩比。是有损的吗？损失多大？

2. 一张 4K 图像（3840x2160）在相同 VQ-VAE 密度下产生多少图像 token？Chameleon 风格的模型能在一次推理调用中生成 4K 图像吗？什么先崩溃——context、tokenizer 质量还是 KV cache？

3. 用纯 Python 实现 QK-Norm。给定 64 维的 query 和 key，展示 LayerNorm 前后的点积。为什么幅度控制在深层很重要？

4. 阅读 Chameleon 第 2.3 节关于训练稳定性的内容。描述论文在 34B 规模下没有 QK-Norm 时观察到的确切失败模式。"norm explosion" 的特征是什么？

5. 扩展玩具解码器，使其在给定纯文本 prompt 时输出混合模态响应。测量在训练数据分布为 60% text-first / 40% image-first 的情况下，模型选择 image-first vs text-first 的频率。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Early fusion | "统一 token" | 图像从第一步就被转换为与 transformer 词表共享的离散 token |
| VQ-VAE | "图像 tokenizer" | CNN + ViT + codebook，将图像映射为 transformer 可预测的整数索引 |
| Shared vocabulary | "一本字典" | 覆盖文本 + 图像 + 模态分隔符的单一 token ID 空间 |
| QK-Norm | "Attention 稳定器" | 在 query 和 key 点积之前应用 LayerNorm，防止 norm 爆炸 |
| Mixed-modality generation | "文本 + 图像输出" | 在一次推理中自主产生交织的文本和图像 token |
| Codebook size | "K 个条目" | VQ-VAE 可量化到的离散向量数量；在压缩和保真度之间权衡 |
| Tokenizer ceiling | "重建上限" | 解码 VQ token 可达到的最佳 PSNR；限定了模型的图像质量 |

## 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
