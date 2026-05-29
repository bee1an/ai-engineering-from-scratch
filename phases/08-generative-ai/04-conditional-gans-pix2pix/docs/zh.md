# 条件 GAN 与 Pix2Pix

> 2014-2017 年的第一个重大突破是控制 GAN 生成什么。附加一个标签、一张图像或一句话。Pix2Pix 做了图像版本，在窄领域的图像到图像任务上它至今仍然击败每个通用文本到图像模型。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**Time:** ~75 minutes

## 问题

无条件 GAN 采样任意人脸。对演示有用，在生产中无用。你想要的是：*将草图映射到照片*、*将地图映射到航拍照片*、*将白天场景映射到夜晚*、*给灰度图像上色*。在所有这些任务中，你有一个输入图像 `x`，必须输出具有某种语义对应关系的 `y`。每个 `x` 有许多合理的 `y`。均方误差将它们压平成糊状。对抗损失不会，因为"看起来真实"是锐利的。

条件 GAN（Mirza & Osindero, 2014）将条件 `c` 作为 `G` 和 `D` 的输入。Pix2Pix（Isola et al., 2017）将其特化：条件是完整输入图像，生成器是 U-Net，判别器是*基于 patch 的*分类器（PatchGAN），损失是对抗 + L1。这个配方在窄领域图像到图像任务上甚至在 2026 年仍然优于从头训练的文本到图像模型，因为它是在*配对数据*上训练的——你有你需要的确切信号。

## 概念

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**条件 G。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（没有输入噪声——Isola 发现显式噪声会被忽略）。

**条件 D。** `D(x, y) → [0, 1]`。输入是*对*（条件，输出）。这是关键区别：D 必须判断 `y` 是否与 `x` 一致，而不仅仅是 `y` 是否看起来真实。

**U-Net 生成器。** 带有跨瓶颈跳跃连接的编码器-解码器。对于输入和输出共享低级结构（边缘、轮廓）的任务至关重要。没有跳跃连接，高频细节就会消失。

**PatchGAN 判别器。** D 不是输出单个真/假分数，而是输出一个 `N×N` 网格，每个单元判断约 70×70 像素的感受野。取平均。这是马尔可夫随机场假设：真实性是局部的。训练更快，参数更少，输出更锐利。

**损失。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练并将 G 推向已知目标。L1 比 L2 给出更锐利的边缘（中位数，而非均值）。`λ = 100` 是 Pix2Pix 的默认值。

## CycleGAN — 当你没有配对数据时

Pix2Pix 需要配对的 `(x, y)` 数据。CycleGAN（Zhu et al., 2017）以额外损失为代价放弃了这个要求：*循环一致性*损失。两个生成器 `G: X → Y` 和 `F: Y → X`。训练它们使得 `F(G(x)) ≈ x` 且 `G(F(y)) ≈ y`。这让你可以将马翻译成斑马、夏天翻译成冬天，而不需要配对样本。

2026 年，无配对图像到图像主要通过扩散（ControlNet、IP-Adapter）而非 CycleGAN 完成，但循环一致性的思想几乎存活在每篇无配对域适应论文中。

## Build It

`code/main.py` 在 1-D 数据上实现了一个小型条件 GAN。条件 `c` 是类别标签（0 或 1）。任务：为给定类别从条件分布中产生样本。

### Step 1: append condition to both G and D inputs

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot 编码是最简单的方式。更大的模型使用学习的嵌入、FiLM 调制或交叉注意力。

### Step 2: train conditional

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器必须匹配*给定条件下*的真实分布，而不是边际分布。

### Step 3: verify per-class output

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## Pitfalls

- **条件被忽略。** G 学会边际化，D 从不惩罚因为条件信号太弱。修复：更积极地条件化 D（早期层，而非仅晚期），使用 projection discriminator（Miyato & Koyama 2018）。
- **L1 权重太低。** G 漂移到任意看起来真实的输出，而非忠实的。Pix2Pix 式任务从 λ≈100 开始。
- **L1 权重太高。** G 产生模糊输出因为 L1 仍然是 L_p 范数。训练稳定后逐步降低。
- **D 中的 ground-truth 泄漏。** 将 `(x, y)` 拼接作为 D 输入，而不仅仅是 `y`。没有这个 D 无法检查一致性。
- **每类模式坍缩。** 每个类可以独立坍缩。运行类条件多样性检查。

## Use It

2026 年图像到图像任务的现状：

| Task | Best approach |
|------|---------------|
| 草图 → 照片，同领域，配对数据 | Pix2Pix / Pix2PixHD（仍然快，仍然锐利） |
| 草图 → 照片，无配对 | ControlNet with a Scribble conditioning model |
| 语义分割 → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移 | Diffusion with IP-Adapter 或 LoRA；GAN 方法已是遗产 |
| 深度 → 照片 | ControlNet-Depth over Stable Diffusion |
| 超分辨率 | Real-ESRGAN (GAN), ESRGAN-Plus, 或 SD-Upscale (diffusion) |
| 上色 | ColTran, diffusion-based colorizers, 或 Pix2Pix-color |
| 白天 → 夜晚，季节，天气 | CycleGAN 或 ControlNet-based |

Pix2Pix 在以下情况仍然是正确的工具：(a) 你有数千个配对样本，(b) 任务窄且可重复，(c) 你需要快速推理。在通用开放领域任务上，扩散胜出。

## Ship It

保存 `outputs/skill-img2img-chooser.md`。Skill 接收任务描述、数据可用性（配对 vs 无配对，N 个样本）和延迟/质量预算，输出：方法（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据需求、推理成本和评估协议（LPIPS、FID、任务特定）。

## Exercises

1. **Easy.** 修改 `code/main.py` 添加第三个类。确认 G 仍然将每个类的噪声映射到正确的模式。
2. **Medium.** 在 1-D 设置中用感知式损失替换 L1（例如一个小型冻结 D 作为特征提取器）。它是否改变了条件分布的锐度？
3. **Hard.** 在 1-D 设置中勾画一个 CycleGAN：两个分布，两个生成器，循环损失。展示它在没有配对数据的情况下学会了在它们之间映射。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 条件 GAN | "带标签的 GAN" | G(z, c), D(x, c)。两个网络都看到条件。 |
| Pix2Pix | "图像到图像 GAN" | 配对 cGAN，U-Net G 和 PatchGAN D + L1 损失。 |
| U-Net | "带跳跃的编码器-解码器" | 对称卷积网络；跳跃保留高频。 |
| PatchGAN | "局部真实性分类器" | D 输出每 patch 分数而非全局分数。 |
| CycleGAN | "无配对图像翻译" | 两个 G + 循环一致性损失；无需配对数据。 |
| SPADE | "GauGAN" | 用语义图归一化中间激活；分割到图像。 |
| FiLM | "Feature-wise linear modulation" | 来自条件的逐特征仿射变换；廉价的条件化。 |

## 生产笔记：Pix2Pix 作为延迟受限的基线

当你有配对数据和窄任务（草图 → 渲染、语义图 → 照片、白天 → 夜晚）时，Pix2Pix 的一次性推理在延迟上比扩散快一个数量级。生产比较通常是：

| Path | Steps | Typical latency at 512² on a single L4 |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net forward) | 1 | ~30 ms |
| SD-Inpaint or SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix 在静态 batch 的吞吐量上胜出（每个请求相同 FLOPs）。扩散在质量和泛化上胜出。现代做法通常是为窄任务部署 Pix2Pix 式蒸馏模型，为尾部输入提供扩散回退。

## Further Reading

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — the cGAN paper.
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix.
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN.
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD.
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN.
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — the projection D.
