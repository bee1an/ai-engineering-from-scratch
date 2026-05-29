# 生成模型 — 分类与历史

> 每个图像模型、文本模型、视频模型和 3D 模型都属于五大类之一。选错了类别，你会跟数学搏斗好几周。选对了，这个领域过去十二年的进展就能在你脑中清晰地叠加起来。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 2 (ML Fundamentals), Phase 3 (Deep Learning Core), Phase 7 · 14 (Transformers)
**Time:** ~45 minutes

## 问题

生成模型做一件事：给定从某个未知分布 `p_data(x)` 中抽取的训练样本，输出看起来像是来自同一分布的新样本。人脸、句子、MIDI 文件、蛋白质结构——如果你眯起眼看，都是同一个问题。

难点在于 `p_data` 存在于一个百万维的空间中（一张 512x512 RGB 图像约 786k 维），样本集中在这个空间内的一个薄流形上，而你可能只有 1000 万个样本。暴力估计密度是无望的。每个生成模型都是一种妥协，用一个困难问题换一个稍微不那么困难的问题。

五大家族在过去十二年中存活了下来。了解每个家族做了什么妥协，就能告诉你为什么它在某些任务上胜出，在另一些任务上崩溃。

## 概念

![五大生成模型家族 — 按建模方式分类](../assets/taxonomy.svg)

**1. 显式密度，可计算。** 把 `log p(x)` 写成一个你能实际计算的求和。Autoregressive 模型（PixelCNN、WaveNet、GPT）将 `p(x) = ∏ p(x_i | x_<i)` 分解。Normalizing flows（RealNVP、Glow）将 `p(x)` 构建为简单基分布的可逆变换。优点：精确似然，干净的训练损失。缺点：autoregressive 推理是顺序的（长序列很慢），flows 需要可逆架构（架构受限）。

**2. 显式密度，近似。** 对 `log p(x)` 求下界（ELBO）并优化该下界。VAE（Kingma 2013）使用编码器-解码器加变分后验。扩散模型（DDPM，Ho 2020）训练一个去噪器，隐式优化加权 ELBO。扩散模型是 2026 年图像、视频和 3D 的主流骨干。

**3. 隐式密度。** 完全跳过密度；学习一个生成器 `G(z)` 产生样本，一个判别器 `D(x)` 区分真假。GAN（Goodfellow 2014）。推理快（一次前向传播）但训练时出了名的不稳定。StyleGAN 1/2/3 在 2026 年仍然是固定领域照片级真实感（人脸、卧室）的最强模型。

**4. 基于 score / 连续时间。** 直接学习对数密度的梯度 `∇_x log p(x)`（score）。Song & Ermon（2019）证明 score matching 将扩散推广为 SDE。Flow matching（Lipman 2023）是 2024-2026 的热点：无需模拟的训练、更直的路径、比 DDPM 快 4-10 倍的采样。Stable Diffusion 3、Flux、AudioCraft 2 都使用 flow matching。

**5. 基于 token 的 autoregressive（离散编码）。** 用 VQ-VAE 或残差量化器将高维数据压缩为短的离散 token 序列，然后用 Transformer 建模 token 序列。Parti、MuseNet、AudioLM、VALL-E、Sora 的 patch tokenizer 都用这种方式。这是第 1 类加上一个学习到的 tokenizer。

## 简史

| Year | Model | Why it mattered |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | 第一个有可用训练损失的深度生成模型。 |
| 2014 | GAN (Goodfellow) | 隐式密度，无似然——样本锐利得惊人。 |
| 2015 | DRAW, PixelCNN | 顺序图像生成。 |
| 2017 | Glow, RealNVP | 可逆 flows；精确似然加深度。 |
| 2017 | Progressive GAN | 第一个百万像素人脸。 |
| 2019 | StyleGAN / StyleGAN2 | 照片级真实人脸，至今在该领域难以超越。 |
| 2020 | DDPM (Ho) | 扩散变得实用。 |
| 2021 | CLIP, DALL-E 1, VQGAN | 文本到图像走向主流。 |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Latent diffusion + 文本条件 = 商品化。 |
| 2022 | ControlNet, LoRA | 对预训练扩散模型的精细控制。 |
| 2023 | SDXL, Midjourney v5, Flow matching | 规模 + 更好的训练动态。 |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | 视频扩散；flow matching 胜出。 |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | 生产级视频。 |
| 2026 | Consistency + Rectified Flow | 从扩散骨干实现单步采样。 |

## 五问分诊法

当一篇新的生成模型论文出来时，在阅读方法部分之前先回答这五个问题。

1. **建模什么？** 像素、latent、离散 token、3D Gaussians、mesh、波形？
2. **密度是显式还是隐式？** 他们写出了 `log p(x)` 吗？
3. **采样：一次性还是迭代？** 迭代意味着推理更慢；一次性通常意味着对抗式或蒸馏。
4. **条件：无条件、类别、文本、图像、姿态？** 这决定了损失和架构脚手架。
5. **评估：FID、CLIP score、IS、人类偏好、任务准确率？** 每个都有已知的失败模式（见第 14 课）。

你将在本阶段的每一课中重新回答这五个问题。到最后，它们会成为条件反射。

## Build It

本课的代码是一个轻量级可视化：用三种玩具方法（核密度估计、离散直方图和一个最近邻"GAN 式"生成器）从样本拟合一个 1-D 高斯混合分布，让你在一个屏幕上就能看到显式密度与隐式密度的区别。

运行 `code/main.py`。它从一个双模高斯混合中抽取 2000 个样本，然后打印：

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

注意：前两个让你问"这个点有多大可能？"第三个不能。这就是*显式 vs 隐式*的区别，在未来每一课中都很重要。

## Use It

2026 年，哪个家族用于哪个任务？

| Task | Best family | Why |
|------|-------------|-----|
| 照片级真实人脸，窄领域 | StyleGAN 2/3 | 仍然最锐利，推理最快。 |
| 通用文本到图像 | Latent diffusion + flow matching | SD3, Flux.1, DALL-E 3. |
| 快速文本到图像 | Rectified flow + distillation | SDXL-Turbo, SD3-Turbo, LCM. |
| 文本到视频 | Diffusion Transformer + flow matching | Sora, Veo 2, Kling. |
| 语音 + 音乐 | Token-based AR (AudioLM, VALL-E, MusicGen) 或 flow matching (AudioCraft 2) | 离散 token 扩展成本低。 |
| 3D 场景 | Gaussian Splatting fit, diffusion prior | 3D-GS 用于重建，diffusion 用于新视角。 |
| 密度估计（不采样） | Flows | 唯一有精确 `log p(x)` 的家族。 |
| 仿真 / 物理 | Flow matching, score SDE | 直线路径，平滑向量场。 |

## Ship It

保存为 `outputs/skill-model-chooser.md`。

该 skill 接收任务描述并输出：(1) 使用哪个家族，(2) 三个开源和三个托管选项的排序列表，(3) 你应该注意的可能失败模式，(4) 计算/时间预算。

## Exercises

1. **Easy.** 对以下五个产品，识别其家族和骨干：ChatGPT image、Midjourney v7、Sora、Runway Gen-3、ElevenLabs。证据应来自公开技术报告。
2. **Medium.** 你明天要读的论文声称比扩散快 100 倍。写下三个问题来检验这个加速在条件生成和高分辨率下是否成立。
3. **Hard.** 选一个你关心的领域（如蛋白质结构、CAD、分子、轨迹）。对该领域当前 SOTA 模型回答五问分诊法，并勾画一个更好的模型会改变什么。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 生成模型 | "它能造新东西" | 学习 `p_data(x)` 的采样器，可选地暴露 `log p(x)`。 |
| 显式密度 | "你能算出来" | 模型提供封闭形式或可计算的 `log p(x)`。 |
| 隐式密度 | "GAN 式" | 只有采样器——无法评估给定点的 `p(x)`。 |
| ELBO | "Evidence lower bound" | `log p(x)` 的可计算下界；VAE 和扩散模型优化它。 |
| Score | "对数密度的梯度" | `∇_x log p(x)`；扩散和 SDE 模型学习这个场。 |
| 流形假设 | "数据在一个曲面上" | 高维数据集中在低维流形上；这是降维有效的原因。 |
| Autoregressive | "预测下一个片段" | 将联合分布分解为条件分布的乘积。 |
| Latent | "压缩编码" | 低维表示，解码器可以从中重建输入。 |

## 生产笔记：五大家族，五种推理形态

每个家族对应不同的推理服务器成本曲线。生产推理文献将 LLM 推理框架化为 prefill + decode；同样的分解适用于此：

- **Autoregressive（第 1 和第 5 类）。** 顺序 decode 主导延迟；KV-cache、continuous batching 和 speculative decoding 都直接适用。
- **VAE / diffusion / flow-matching（第 2 和第 4 类）。** 没有 LLM 意义上的 decode。成本 = `num_steps × step_cost`，而 `step_cost` 是在完整 latent 分辨率上的 transformer 或 U-Net 前向传播。生产旋钮是步数（DDIM / DPM-Solver / distillation）、batch size 和精度（bf16 / fp8 / int4）。
- **GAN（第 3 类）。** 一次前向传播。没有 schedule，没有 KV-cache。TTFT ≈ 总延迟。这就是为什么 StyleGAN 在窄领域 UX 上仍然胜出。

当你在论文摘要中看到"比扩散更快"时，翻译为"更少步数 × 相同步成本"或"相同步数 × 更低步成本"。其他都是营销。

## Further Reading

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — the GAN paper.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — the VAE paper.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — the DDPM paper.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — diffusion as an SDE.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — the flow matching paper.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3.
