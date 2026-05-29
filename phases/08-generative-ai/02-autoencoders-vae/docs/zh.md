# 自编码器与变分自编码器 (VAE)

> 普通自编码器先压缩再重建。它在记忆，不在生成。加一个技巧——强制编码看起来像高斯分布——你就得到了一个采样器。这个技巧，即 `z = μ + σ·ε` 的重参数化，正是 2026 年你使用的每个 latent-diffusion 和 flow-matching 图像模型在输入端都有一个 VAE 的原因。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Time:** ~75 minutes

## 问题

将一个 784 像素的 MNIST 数字压缩到 16 个数字的编码，然后重建。普通自编码器在重建 MSE 上表现优秀，但编码空间是一团乱麻。在编码空间中随机选一个点，解码它，你得到的是噪声。它没有采样器。它只是一个伪装成生成模型的压缩模型。

你真正想要的是：(a) 编码空间是一个干净、平滑的分布，你可以从中采样——比如各向同性高斯 `N(0, I)`，(b) 解码任何样本都能产生一个合理的数字，(c) 编码器和解码器仍然压缩得好。三个目标，一个架构，一个损失。

Kingma 的 2013 年 VAE 通过训练编码器输出一个*分布* `q(z|x) = N(μ(x), σ(x)²)` 来解决这个问题，通过 KL 惩罚将该分布拉向先验 `N(0, I)`，然后在解码前从 `q(z|x)` 中采样 `z`。推理时，丢弃编码器，采样 `z ~ N(0, I)`，解码。KL 惩罚就是强制编码空间结构化的东西。

2026 年 VAE 很少单独部署——在原始图像质量上已被扩散模型超越——但它们是每个 latent-diffusion 模型（SD 1/2/XL/3、Flux、AudioCraft）的首选编码器。学会 VAE，你就学会了你使用的每个图像管线中看不见的第一层。

## 概念

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**自编码器。** `z = encoder(x)`，`x̂ = decoder(z)`，loss = `||x - x̂||²`。编码空间无结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们定义 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧。** 从 `q(z|x)` 采样是不可微的。将样本改写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 的确定性函数加上一个非参数噪声——梯度可以流过 `μ` 和 `σ`。

**损失。** Evidence Lower BOund (ELBO)，两项：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建项将 `x̂` 推向 `x`。KL 项将 `q(z|x)` 推向先验。它们相互权衡。小 β (<1) = 更锐利的样本，编码空间不那么高斯。大 β (>1) = 更干净的编码空间，更模糊的样本。β-VAE（Higgins 2017）让这个旋钮出名，并开启了解纠缠研究。

**采样。** 推理时：抽取 `z ~ N(0, I)`，通过解码器前向传播。一次前向传播——不像扩散那样需要迭代采样。

## Build It

`code/main.py` 实现了一个不使用 numpy 或 torch 的小型 VAE。输入是从 8-D 中的 2 分量高斯混合中抽取的 8 维合成数据。编码器和解码器是单隐藏层 MLP。我们实现了 tanh 激活、前向传播、损失和手写反向传播。不是生产代码——是教学。

### Step 1: encoder forward

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

用 `log σ²` 而不是 `σ`，这样网络输出是无约束的（σ 的 softplus 是个陷阱——梯度在 σ ≈ 0 时消失）。

### Step 2: reparameterize and decode

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Step 3: the ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

精确的封闭形式 KL，因为两个分布都是高斯的。不要数值积分。2026 年仍有人用蒙特卡洛 KL 估计——慢 3 倍，毫无理由。

### Step 4: generate

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行代码。

## Pitfalls

- **后验坍缩。** KL 项过于激进地驱动 `q(z|x) → N(0, I)`，使得 `z` 不携带关于 `x` 的信息。修复：β-annealing（从 β=0 开始，逐步升到 1）、free bits、或跳过不活跃维度的 KL。
- **模糊样本。** 高斯解码器似然意味着 MSE 重建，这对 L2 是贝叶斯最优的（均值）——一组合理数字的均值是一个模糊的数字。修复：离散解码器（VQ-VAE、NVAE），或仅将 VAE 用作编码器并在 latent 上叠加扩散（这就是 Stable Diffusion 做的）。
- **β 太大，太早。** 见后验坍缩。从 β≈0.01 开始逐步升高。
- **Latent 维度太小。** 16-D 适用于 MNIST，256-D 适用于 ImageNet 256²，2048-D 适用于 ImageNet 1024²。Stable Diffusion 的 VAE 将 512×512×3 压缩到 64×64×4（空间面积 32 倍下采样，通道 32 倍）。

## Use It

2026 年的 VAE 技术栈：

| Situation | Pick |
|-----------|------|
| 扩散模型的图像 latent 编码器 | Stable Diffusion VAE (`sd-vae-ft-ema`) 或 Flux VAE |
| 音频 latent 编码器 | Encodec (Meta), SoundStream, 或 DAC (Descript) |
| 视频 latent | Sora 的时空 patches, Latte VAE, WAN VAE |
| 解纠缠表示学习 | β-VAE, FactorVAE, TCVAE |
| 离散 latent（用于 transformer 建模） | VQ-VAE, RVQ (ResidualVQ) |
| 连续 latent 用于生成 | 普通 VAE，然后在该 latent 空间中条件化 flow/diffusion 模型 |

Latent-diffusion 模型就是一个 VAE，中间夹着一个扩散模型。VAE 做粗压缩，扩散模型做重活。视频（VAE + video-diffusion DiT）和音频（Encodec + MusicGen transformer）也是同样的模式。

## Ship It

保存 `outputs/skill-vae-trainer.md`。

Skill 接收：数据集概况 + latent 维度目标 + 下游用途（重建、采样或 latent-diffusion 输入），输出：架构选择（plain/β/VQ/RVQ）、β schedule、latent 维度、解码器似然（高斯 vs 分类）、评估计划（重建 MSE、每维 KL、`q(z|x)` 与 `N(0, I)` 之间的 Fréchet 距离）。

## Exercises

1. **Easy.** 将 `code/main.py` 中的 `β` 改为 `0.01`、`0.1`、`1.0`、`5.0`。记录最终重建 MSE 和 KL。哪个 β 对你的合成数据是 Pareto 最优的？
2. **Medium.** 将高斯解码器似然替换为伯努利似然（交叉熵损失）。在同一合成数据的二值化版本上比较样本质量。
3. **Hard.** 将 `code/main.py` 扩展为一个迷你 VQ-VAE：用 K=32 个条目的码本中的最近邻查找替换连续 `z`。比较重建 MSE 并报告有多少码本条目被使用（码本坍缩是真实存在的）。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 自编码器 | Encode-decode network | `x → z → x̂`，学习 MSE。不是生成式的。 |
| VAE | AE with a sampler | 编码器输出分布，KL 惩罚塑造编码空间。 |
| ELBO | Evidence lower bound | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`；当 `q = p(z\|x)` 时紧。 |
| 重参数化 | `z = μ + σ·ε` | 将随机节点改写为确定性 + 纯噪声。使反向传播能通过采样。 |
| 先验 | `p(z)` | latent 的目标分布，通常是 `N(0, I)`。 |
| 后验坍缩 | "KL 项赢了" | 编码器忽略 `x`，输出先验；解码器必须幻觉。 |
| β-VAE | 可调 KL 权重 | `loss = recon + β·KL`。更高的 β = 更解纠缠但更模糊。 |
| VQ-VAE | 离散 latent | 用最近码本向量替换连续 `z`；使 transformer 建模成为可能。 |

## 生产笔记：VAE 是扩散服务器中最热的路径

在 Stable Diffusion / Flux / SD3 管线中，VAE 每个请求被调用两次——一次编码（如果做 img2img / inpainting）和一次解码。在 1024² 时，解码器 pass 通常是整个管线中单次最大的激活内存峰值，因为它将 `128×128×16` latent 上采样回 `1024×1024×3`。两个实际后果：

- **切片或分块解码。** `diffusers` 暴露了 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。分块以微小的接缝伪影换取 `O(tile²)` 内存而非 `O(H·W)`。在消费级 GPU 上 1024²+ 是必需的。
- **bf16 解码器，fp32 数值用于最终 resize。** SD 1.x VAE 以 fp32 发布，在 1024²+ 时转换为 fp16 会*静默产生 NaN*。SDXL 附带 `madebyollin/sdxl-vae-fp16-fix`——始终优先使用 fp16-fix 变体或使用 bf16。

## Further Reading

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — the VAE paper.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — disentangled β-VAE.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — state-of-the-art image VAE.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion; VAE as encoder.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec, the audio VAE standard.
