# StyleGAN

> 大多数生成器将 `z` 同时搅入每一层。StyleGAN 将其拆开：先将 `z` 映射到中间 `w`，然后通过 AdaIN 在每个分辨率级别*注入* `w`。这一个改变解纠缠了 latent 空间，并让照片级真实人脸成为一个已解决的问题，持续了七年。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 08 (Normalization), Phase 3 · 07 (CNNs)
**Time:** ~45 minutes

## 问题

DCGAN 通过一叠转置卷积将 `z` 映射到图像。问题是：`z` 控制一切——姿态、光照、身份、背景——纠缠在一起。沿 `z` 的一个轴移动，四个都变。你无法要求模型"同一个人，不同姿态"，因为表示没有这样分解。

Karras et al.（2019，NVIDIA）提出：停止将 `z` 直接送入卷积层。将一个常量 `4×4×512` 张量作为网络输入。学习一个 8 层 MLP 将 `z ∈ Z → w ∈ W`。通过*自适应实例归一化*（AdaIN）在每个分辨率注入 `w`：归一化每个卷积特征图，然后用 `w` 的仿射投影进行缩放和偏移。为随机细节（皮肤毛孔、发丝）添加逐层噪声。

结果：`W` 对"高级风格"（姿态、身份）vs "精细风格"（光照、颜色）有大致正交的轴。你可以通过对低分辨率级别使用图像 A 的 `w`、对高分辨率级别使用图像 B 的 `w` 来交换两张图像之间的风格。这解锁了编辑、跨域风格化和整个"StyleGAN-inversion"研究线。

## 概念

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**Mapping network。** `f: Z → W`，8 层 MLP。`Z = N(0, I)^512`。`W` 不被强制为高斯——它学习数据适应的形状。

**Synthesis network。** 从学习的常量 `4×4×512` 开始。每个分辨率块：`upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`。分辨率翻倍：4, 8, 16, 32, 64, 128, 256, 512, 1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。逐特征图归一化，然后重新风格化。这里的"风格"是特征图的一阶和二阶统计量。

**逐层噪声。** 单通道高斯噪声添加到每个特征图，由学习的逐通道因子缩放。控制随机细节而不影响全局结构。

**Truncation trick。** 推理时，采样 `z`，计算 `w = mapping(z)`，然后 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是许多样本上 `w` 的均值。`ψ < 1` 以多样性换质量。几乎每个 StyleGAN 演示都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| Version | Year | Innovation |
|---------|------|------------|
| StyleGAN | 2019 | Mapping network + AdaIN + noise + progressive growing. |
| StyleGAN2 | 2020 | Weight demodulation 替换 AdaIN（修复水滴伪影）；skip/residual 架构；path-length regularization。 |
| StyleGAN3 | 2021 | Alias-free convolution + equivariant kernels；消除纹理粘在像素网格上的问题。 |
| StyleGAN-XL | 2022 | 类条件，1024²，ImageNet。 |
| R3GAN | 2024 | 更强正则化的重新品牌；以 20 倍更少参数在 FFHQ-1024 上缩小与扩散的差距。 |

2026 年 StyleGAN3 仍然是以下场景的默认选择：(a) 高 FPS 的窄领域照片级真实感，(b) 少样本域适应（在 100 张图像的新数据集上训练，冻结 mapping），(c) 基于反演的编辑（找到重建真实照片的 `w`，然后编辑该 `w`）。对于开放领域文本到图像，它不是工具——扩散才是。

## Build It

`code/main.py` 在 1-D 中实现了一个玩具"style-GAN lite"：一个 mapping MLP，一个合成函数接收学习的常量向量并用 `w` 派生的 scale/bias 调制它，以及逐层噪声。它展示了通过仿射调制注入 `w` 匹配或击败将 `z` 拼接到生成器输入中的方式。

### Step 1: mapping network

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Step 2: adaptive instance normalization

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

逐特征图的 scale 和 bias 来自 `w` 通过线性投影。

### Step 3: per-layer noise

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

逐通道的 Sigma 是可学习的。

## Pitfalls

- **水滴伪影。** StyleGAN 1 在特征图中产生斑点状水滴，因为 AdaIN 将均值归零。StyleGAN 2 的 weight demodulation 通过缩放卷积权重来修复。
- **纹理粘连。** StyleGAN 1 和 2 的纹理跟随像素坐标而非物体坐标（插值时可见）。StyleGAN 3 的 alias-free 卷积用窗口化 sinc 滤波器修复。
- **模式覆盖。** Truncation `ψ < 0.7` 看起来干净但从窄锥中采样；如果需要多样性使用 `ψ = 1.0`。
- **反演是有损的。** 将真实照片反演到 `W` 通常通过优化或编码器（e4e、ReStyle、HyperStyle）完成。结果在多次迭代后会漂移。

## Use It

| Use case | Approach |
|----------|----------|
| 照片级真实人脸（动漫、产品、窄领域） | StyleGAN3 FFHQ / 自定义微调 |
| 从照片编辑人脸 | e4e inversion + StyleSpace / InterFaceGAN directions |
| 换脸 / 重演 | StyleGAN + encoder + blending |
| Avatar 管线 | StyleGAN3 w/ ADA 用于低数据微调 |
| 从少量图像做域适应 | 冻结 mapping network，微调 synthesis |
| 多模态或文本条件生成 | 不要——用扩散 |

对于答案是"一个人脸照片"的产品级演示，StyleGAN 在推理成本（单次前向传播，4090 上 <10ms）和相同质量标准下的锐度上击败扩散。

## Ship It

保存 `outputs/skill-stylegan-inversion.md`。Skill 接收一张真实照片，输出：反演方法（e4e / ReStyle / HyperStyle）、预期 latent 损失、编辑预算（在 `W` 中能移动多远才出现伪影）、以及已知有效的编辑方向列表（年龄、表情、姿态）。

## Exercises

1. **Easy.** 用 `adain_on=True` 和 `adain_on=False` 运行 `code/main.py`。比较固定 latent vs 扰动 latent 的输出分布。
2. **Medium.** 实现 mixing regularization：对一个训练 batch，计算 `w_a`、`w_b`，对合成的前半部分应用 `w_a`，后半部分应用 `w_b`。解码器是否学到了解纠缠的风格？
3. **Hard.** 取一个预训练的 StyleGAN3 FFHQ 模型（ffhq-1024.pkl）。通过在标注样本上训练 SVM 找到控制"微笑"的 `w` 方向；报告推多远身份会漂移。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Mapping network | "The MLP" | `f: Z → W`，8 层，将 latent 几何与数据统计解耦。 |
| W space | "风格空间" | Mapping network 的输出；大致解纠缠。 |
| AdaIN | "Adaptive instance norm" | 归一化特征图，然后用 `w` 投影做 scale + shift。 |
| Truncation trick | "Psi" | `w = mean + ψ·(w - mean)`，ψ<1 以多样性换质量。 |
| Path-length regularization | "PL reg" | 惩罚 `w` 单位变化引起的图像大变化；使 `W` 更平滑。 |
| Weight demodulation | "StyleGAN2 的修复" | 归一化卷积权重而非激活；消除水滴伪影。 |
| Alias-free | "StyleGAN3 的技巧" | 窗口化 sinc 滤波器；消除纹理粘在像素网格上。 |
| Inversion | "为真实图像找 w" | 优化或编码 `x → w` 使得 `G(w) ≈ x`。 |

## 生产笔记：为什么 StyleGAN 在 2026 年仍在部署

StyleGAN3 在 4090 上生成一张 1024² FFHQ 人脸不到 10 ms——`num_steps = 1`，没有 VAE decode，没有 cross-attention pass。在生产术语中这是任何图像生成器的延迟下限。同分辨率下 50 步 SDXL + VAE-decode 管线约 3 秒。这是 **300 倍的差距**，对于窄领域产品（头像服务、证件管线、库存人脸生成）它在 TCO 上胜出。

两个运营后果：

- **无需调度器，无需 batcher。** 在目标占用率下的静态 batch 是最优的。Continuous batching（对 LLM 和扩散至关重要）提供零收益，因为每个请求消耗相同 FLOPs。
- **Truncation `ψ` 是安全旋钮。** `ψ < 0.7` 从 mapping network 范围的窄锥中采样。这是服务层对样本方差的唯一杠杆。峰值负载时降低 `ψ`，为高级用户提高它。

## Further Reading

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e inversion.
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL.
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — modern minimal GAN recipe.
