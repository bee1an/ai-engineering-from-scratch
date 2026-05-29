# Emu3：用 Next-Token Prediction 进行图像和视频生成

> BAAI 的 Emu3（Wang et al., 2024 年 9 月）是 2024 年本应终结扩散 vs 自回归之争的结果。一个单一的 Llama 风格 decoder-only transformer，仅在 next-token prediction 目标上训练，跨文本 + VQ 图像 token + 3D VQ 视频 token 的统一词表，在图像生成上击败 SDXL，在感知上击败 LLaVA-1.6。没有 CLIP loss。没有扩散调度。推理时使用 classifier-free guidance 提升质量，但核心训练目标是带 teacher forcing 的 next-token prediction。发表于 Nature。本课阅读 Emu3 的论点——为什么更好的 tokenizer 加上规模就是你所需要的一切——并与扩散方法对比。

**Type:** Learn
**Languages:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites:** Phase 12 · 11 (Chameleon)
**Time:** ~120 minutes

## 学习目标

- 解释为什么 Emu3 的单 loss next-token 目标能够工作，尽管长期以来人们认为图像质量需要扩散。
- 描述 3D 视频 tokenizer：时空 VQ codebook 是什么样的，为什么 patch 跨越时间。
- 比较 Emu3 vs Stable Diffusion XL 在（训练计算量、推理成本、质量上限）上的差异。
- 说出同一个 Emu3 模型扮演的三个角色：Emu3-Gen（图像生成）、Emu3-Chat（感知）、Emu3-Stage2（视频生成）。

## 问题

2024 年之前的传统观点：图像生成需要扩散。论据是：离散图像 token 丢失了太多信息无法重建细节，而自回归采样在数千个 token 上累积误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都使用某种形式的扩散。Chameleon（Lesson 12.11）在小规模上部分推翻了这一点，但在质量上没有匹配 SDXL。

Emu3 正面攻击了这个论点。声明是：更好的视觉 tokenizer + 足够的规模 + next-token loss = 在同一个也做感知的模型中击败扩散的图像生成。

这个赌注在发表时是有争议的。两年后，开源统一生成家族（Emu3、Show-o、Janus-Pro、Transfusion）已成为研究的默认路径；生产前沿模型似乎使用某种变体。

## 概念

### Emu3 tokenizer

关键成分是视觉 tokenizer。Emu3 训练了一个自定义的 IBQ 类 tokenizer（Inverse Bottleneck Quantizer，SBER-MoVQGAN 家族），每个 token 做 8x8 分辨率缩减。一张 512x512 图像变成 64x64 = 4096 个 token，codebook 大小为 32768。

这比 Chameleon 的每张 512x512 图像 1024 个 token（K=8192）更多，但每个 token 更便宜（更小的 codebook 查找，更简单的编解码器）。关键指标：重建 PSNR 为 30.5 dB，与 Stable Diffusion 的连续潜空间 32 dB 具有竞争力。

对于视频：3D VQ tokenizer 将一个时空 patch（4x4x4 像素）编码为一个整数。一个 4 秒片段在 8 FPS 下有 32 帧；在 256x256 分辨率下，4x 空间和 4x 时间缩减，token 数为 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 个 token。

Tokenizer 质量是上限。Emu3 的贡献部分在于"我们训练了一个非常好的 tokenizer。"

### 单 loss 训练

Emu3 使用一个目标：在跨文本 token、2D 图像 token 和 3D 视频 token 的共享词表上做 next-token prediction。训练时权重乘以模态特定因子来平衡贡献，但 loss 函数是相同的。

训练数据混合：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：类似。
- 纯文本：标准 NTP。

模型从数据分布中学习何时输出图像 token vs 文本 token。生成能力来自模型在 `<image>` 标签后预测图像 token。

### Classifier-free guidance 和温度

自回归图像生成在推理时使用 classifier-free guidance (CFG) 效果好得多。Emu3 使用它：生成两次，一次用完整 caption，一次用空 caption，用引导权重（典型值 3.0-7.0）混合 logits。这与扩散使用的 CFG 技巧相同，借用到了自回归设置中。

温度很重要：太高产生伪影；太低导致模式坍缩。Emu3 推荐的温度是感知 1.0，图像生成 0.8。

### 三个角色，一个模型

Emu3 作为三个功能不同的 API 发布，但底层是同一套权重：

- Emu3-Gen。图像生成。输入文本，输出图像 token。
- Emu3-Chat。VQA 和 captioning。输入图像（token），输出文本。
- Emu3-Stage2。视频生成和视频 VQA。输入文本或视频，输出文本或视频。

没有任务特定的 head。只是不同的 prompt 模板。同一个 checkpoint。

### Benchmark

来自 Emu3 论文（2024 年 9 月）：

- 图像生成：在 MJHQ-30K FID 上击败 SDXL（5.4 vs 5.6），GenEval 总分（0.54 vs 0.55——统计平局），Deep-Eval 综合分持平。
- 图像感知：在 VQAv2 上击败 LLaVA-1.6（75.1 vs 72.4），在 MMMU 上大致匹配。
- 视频生成：4 秒片段质量在 FVD 上与 Sora 时代公开 benchmark 的模型具有竞争力。

数字并非总是赢——Emu3 在这里赢一分那里输一分——但"next-token prediction is all you need"的声明在各模态上是站得住脚的。

### 计算成本

Emu3 在约 3000 亿多模态 token 上训练了一个 7B 参数模型。GPU 小时大致相当于 Llama-2-7B 预训练（在 A100 级硅片上 2k-4k GPU 年）。像 Stable Diffusion 3 这样的扩散模型在类似预算下训练，但需要独立的文本编码器和更复杂的流水线。

在推理时，Emu3 每张图像比 SDXL 慢：4096 个图像 token 在 30 tok/s 下约需 2 分钟生成一张 512x512 图像，而 SDXL 需要 2-5 秒。投机解码和 KV-cache 优化缩小了差距但没有消除。自回归图像生成是计算密集型的；这是持续存在的权衡。

### 为什么重要

Emu3 的深层贡献是概念性的。如果 next-token prediction 能扩展到在图像生成上匹配扩散，那么统一模型路径（一个 loss，一个 backbone，任意模态）就是可行的。未来的模型不需要独立的文本编码器、独立的扩散调度器、独立的 VAE。一个 transformer，每种模态一个 tokenizer，扩展规模。

Show-o、Janus-Pro 和 InternVL-U 都在此论点上构建或挑战。中国实验室（BAAI、DeepSeek）在这个方向上比美国实验室发表得更积极，贯穿 2025 年。

## Use It

`code/main.py` 构建了两个玩具组件：

- 一个 2D vs 3D VQ tokenizer token 数计算器：给定（分辨率、patch、片段长度、FPS），计算图像 vs 视频的 token 数。
- 一个带 classifier-free guidance 和温度的自回归图像 token 采样器。

CFG 实现匹配 Emu3 的方案——用引导权重混合条件和无条件 logits。

## Ship It

本课产出 `outputs/skill-token-gen-cost-analyzer.md`。给定生成产品规格（图像或视频、目标分辨率、质量层级、延迟预算），它计算 token 数、推理成本，并在 Emu3 家族 vs 扩散之间选择。

## 练习

1. Emu3 在 8x8 缩减下每张 512x512 图像产生 4096 个 token。计算 1024x1024 和 2048x2048 的等效值。推理延迟会怎样？

2. 阅读 Emu3 第 3.3 节关于视频 tokenizer 的内容。描述 3D VQ patch 形状以及为什么是 4x4x4 而不是 8x8x1。

3. Classifier-free guidance 权重 5.0 vs 3.0：视觉效果有什么不同？在 `code/main.py` 中追踪数学计算。

4. 计算 Emu3-7B 在 300B token 下的训练 FLOPs 并与 Stable Diffusion 3 比较。哪个训练更贵？

5. Emu3 在 FID 上击败 SDXL 但在 VQAv2 上不如专用 VLM。解释为什么统一 loss 方法在不同 benchmark 上相对专用模型表现出不同的优势。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Next-token prediction | "NTP" | 标准自回归 loss：给定 token[0..i] 预测 token[i+1]；tokenize 后适用于所有模态 |
| IBQ tokenizer | "Inverse bottleneck quantizer" | 一类具有更大 codebook（32768+）和比 Chameleon 更好重建质量的 VQ-VAE |
| 3D VQ | "时空量化器" | 按（时间、行、列）索引的 codebook；一个 token 覆盖一个 4x4x4 像素立方体 |
| Classifier-free guidance | "CFG" | 用权重 gamma 混合条件和无条件 logits；在推理时提升图像质量 |
| Unified vocabulary | "共享 token" | 文本 + 图像 + 视频都从同一个整数空间取值；模型预测接下来出现的任何模态 |
| MJHQ-30K | "图像生成 benchmark" | 包含 30k prompt 的 Midjourney 质量 benchmark；Emu3 在此报告 FID |

## 延伸阅读

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
