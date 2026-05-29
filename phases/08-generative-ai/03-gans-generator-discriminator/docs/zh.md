# GAN — 生成器 vs 判别器

> Goodfellow 在 2014 年的技巧是完全跳过密度。两个网络。一个造假。一个抓假。它们对抗直到假的和真的无法区分。这不应该有效。它经常确实无效。但当它有效时，在窄领域中样本仍然是文献中最锐利的。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 08 (Optimizers), Phase 8 · 02 (VAE)
**Time:** ~75 minutes

## 问题

VAE 产生模糊样本，因为它们的 MSE 解码器损失对*均值*图像是贝叶斯最优的——而许多合理数字的均值是一个模糊的数字。你想要一个奖励*合理性*的损失，而不是与任何单一目标的逐像素接近度。合理性没有封闭形式。你必须学习它。

Goodfellow 的想法：训练一个分类器 `D(x)` 来区分真实图像和假图像。训练一个生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号是 `D` 当前认为什么使东西看起来真实。这个信号随着 `G` 的改进而更新，追逐一个移动的目标。如果两个网络都收敛，`G` 就学会了数据分布，而从未写下 `log p(x)`。

这就是对抗训练。数学上是一个极小极大博弈：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

2026 年 GAN 不再是 SOTA 生成器（扩散和 flow matching 夺走了那顶皇冠）。但 StyleGAN 2/3 仍然是有史以来最锐利的人脸模型，GAN 判别器被用作扩散训练中的*感知损失*，对抗训练驱动着快速 1 步蒸馏（SDXL-Turbo、SD3-Turbo、LCM），让你能部署实时扩散。

## 概念

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**生成器 `G(z)`。** 将噪声向量 `z ~ N(0, I)` 映射到样本 `x̂`。解码器形状的网络（全连接或转置卷积）。

**判别器 `D(x)`。** 将样本映射到标量概率（或分数）。真 → 1，假 → 0。

**损失。** 两个交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。对 real=1, fake=0 的二元交叉熵。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和*形式（原始的 `log(1 - D(G(z)))` 在 `D` 自信时饱和并杀死梯度）。

**训练循环。** `D` 一步，`G` 一步。重复。

**为什么有效。** 如果 `G` 完美匹配 `p_data`，那么 `D` 不能比随机猜测更好，到处输出 0.5；`G` 不再获得梯度。均衡。

**为什么会崩溃。** 模式坍缩（`G` 找到一个 `D` 无法分类的模式并不断生成它）、梯度消失（`D` 学得太快，`log D` 饱和）、训练不稳定（学习率、batch size、任何东西）。

## 让 GAN 有效的变体

| Year | Innovation | Fix |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv, batch norm, LeakyReLU — 第一个稳定的架构。 |
| 2017 | WGAN, WGAN-GP | 用 Wasserstein 距离 + 梯度惩罚替换 BCE。修复梯度消失。 |
| 2017 | Spectral normalization | 对判别器做 Lipschitz 约束。2026 年仍在使用。 |
| 2018 | Progressive GAN | 先训练低分辨率，再加层。第一个百万像素结果。 |
| 2019 | StyleGAN / StyleGAN2 | Mapping network + adaptive instance norm。固定领域照片级真实感的 SOTA。 |
| 2021 | StyleGAN3 | Alias-free，平移等变——2026 年仍是人脸金标准。 |
| 2022 | StyleGAN-XL | 条件式，类别感知，更大规模。 |
| 2024 | R3GAN | 更强正则化的重新品牌；无需技巧即可在 1024² 上工作。 |

## Build It

`code/main.py` 在 1-D 数据上训练一个小型 GAN：两个高斯的混合。生成器和判别器是单隐藏层 MLP。我们手动实现前向、反向和极小极大循环。目标是看到两个关键失败模式（模式坍缩 + 梯度消失）的发生。

### Step 1: non-saturating loss

原始 Goodfellow 损失 `log(1 - D(G(z)))` 在 D 以高置信度将 G 的假样本分类为假时趋向 0。此时 G 的梯度基本为零——G 无法改进。非饱和形式 `-log D(G(z))` 有相反的渐近线：当 D 自信时它爆炸，给 G 强信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Step 2: one discriminator step per generator step

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

G 用新鲜的假样本，否则梯度是过时的。

### Step 3: watch for mode collapse

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

典型症状：两个真实模式中的一个不再被生成。判别器停止纠正它，因为它从未被视为假样本。

## Pitfalls

- **判别器太强。** 将 D 的学习率降低 2-5 倍，或添加 instance/layer noise。如果 D 达到 >95% 准确率，G 就死了。
- **生成器记住一个模式。** 给 D 输入添加噪声，使用 minibatch-discriminator 层，或切换到 WGAN-GP。
- **Batch norm 泄漏统计量。** 真实 batch + 假 batch 流过同一个 BN 层会混合它们的统计量。改用 instance norm 或 spectral norm。
- **Inception-score 作弊。** FID 和 IS 在低样本数时噪声很大。评估时使用 ≥10k 样本。
- **条件任务中一次性采样是谎言。** 你仍然需要 CFG scales、truncation tricks 和重采样来获得可用输出。

## Use It

2026 年的 GAN 技术栈：

| Situation | Pick |
|-----------|------|
| 照片级真实人脸，固定姿态 | StyleGAN3（最锐利，最小） |
| 动漫 / 风格化人脸 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像翻译 | Pix2Pix / CycleGAN (Phase 8 · 04) 或 ControlNet (Phase 8 · 08) |
| 快速 1 步文本到图像 | 扩散的对抗蒸馏 (SDXL-Turbo, SD3-Turbo) |
| 扩散训练器中的感知损失 | 图像裁剪上的小型 GAN 判别器 |
| 任何多模态、开放式的 | 不要——用扩散或 flow matching |

GAN 锐利但狭窄。一旦你的领域打开——照片、任意文本提示、视频——切换到扩散。对抗技巧作为组件（感知损失、蒸馏）存活下来，而不是独立的生成器。

## Ship It

保存 `outputs/skill-gan-debugger.md`。Skill 接收一个失败的 GAN 运行（损失曲线、样本网格、数据集大小），输出可能原因的排序列表、一行修复和重跑协议。

## Exercises

1. **Easy.** 用默认设置运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 并重跑。G 的损失多快坍缩为常数？
2. **Medium.** 将 Goodfellow BCE 损失替换为 WGAN 损失：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并将 D 的权重裁剪到 `[-0.01, 0.01]`。训练更稳定吗？比较墙钟收敛时间。
3. **Hard.** 将 1-D 示例扩展到 2-D 数据（环上 8 个高斯的混合）。跟踪生成器在步骤 1k、5k、10k 时捕获了 8 个模式中的多少个。实现 minibatch discrimination 并重新测量。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 生成器 | "G" | 噪声到样本的网络，`G: z → x̂`。 |
| 判别器 | "D" | 分类器 `D: x → [0, 1]`，真 vs 假。 |
| Minimax | "博弈" | 联合目标的 `min_G max_D`。 |
| 非饱和损失 | "修复" | G 用 `-log D(G(z))` 而不是 `log(1 - D(G(z)))`。 |
| 模式坍缩 | "G 记住了一个东西" | 尽管数据多样，生成器只产生少数不同的输出。 |
| WGAN | "Wasserstein" | 用 Earth-Mover 距离 + 梯度惩罚替换 BCE；更平滑的梯度。 |
| Spectral norm | "Lipschitz 技巧" | 约束 D 的权重范数以限制其斜率；稳定训练。 |
| StyleGAN | "那个有效的" | Mapping network + AdaIN；2026 年仍是人脸最佳。 |

## 生产笔记：一次性推理是 GAN 持久的优势

GAN 在开放领域生成的样本质量上不再胜出，但在推理成本上仍然胜出。用生产推理文献的术语来说，GAN 具有：

- **没有 prefill，没有 decode 阶段。** 单次 `G(z)` 前向传播。TTFT ≈ 总延迟。
- **没有 KV-cache 压力。** 唯一的状态是权重。Batch size 受激活内存限制，而非缓存。
- **简单的 continuous batching。** 由于每个请求消耗相同的固定 FLOPs，在服务器目标占用率下的静态 batch 通常是最优的。不需要在途调度器。

这就是为什么 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）是 2026 年快速文本到图像的主导技术：它将 20-50 步扩散管线压缩为 1-4 次 GAN 式前向传播，同时保持扩散基础的分布。对抗损失作为训练时的旋钮存活下来，用于将慢生成器变成快生成器。

## Further Reading

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — the original GAN paper.
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — the first stable architecture.
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN.
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo.
