# 扩散模型 — 从零实现 DDPM

> Ho, Jain, Abbeel（2020）给了这个领域一个它无法放弃的配方。用噪声在一千个小步中摧毁数据。训练一个神经网络来预测噪声。在推理时反转这个过程。今天每个主流图像、视频、3D 和音乐模型都运行在这个循环上，可能加上 flow matching 或 consistency 技巧。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 8 · 02 (VAE)
**Time:** ~75 minutes

## 问题

你想要一个 `p_data(x)` 的采样器。GAN 玩一个经常发散的极小极大博弈。VAE 从高斯解码器产生模糊样本。你真正想要的是一个训练目标：(a) 单一稳定损失（没有鞍点，没有极小极大），(b) `log p(x)` 的下界（所以你有似然），(c) 样本匹配 SOTA 质量。

Sohl-Dickstein et al.（2015）有一个理论答案：定义一个马尔可夫链 `q(x_t | x_{t-1})` 逐步添加高斯噪声，训练一个反向链 `p_θ(x_{t-1} | x_t)` 来去噪。Ho, Jain, Abbeel（2020）展示了损失可以简化为一行——预测噪声——并清理了数学。2020 年这是个好奇心。2021 年它产生了 SOTA 样本。2022 年它成为 Stable Diffusion。2026 年它是基底。

## 概念

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**前向过程 `q`。** 在 `T` 个小步中添加高斯噪声。封闭形式——数学可处理的原因——是累积步也是高斯的：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)` 对于 `β_t` 的 schedule。在 T=1000 步上将 `β_t` 从 1e-4 线性增加到 0.02，`x_T` 近似为 `N(0, I)`。

**反向过程 `p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)` 预测添加的噪声。给定 `x_t`，去噪：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 是 `sqrt(β_t)` 或学习的方差。表达式很丑但只是代数——给定后验 `q(x_{t-1} | x_t, x_0)` 求解 `x_{t-1}` 并用噪声预测的估计替换 `x_0`。

**训练损失。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据采样 `x_0`，随机选一个 `t`，采样 `ε ~ N(0, I)`，通过封闭形式一步计算噪声 `x_t`，对噪声做回归。一个损失，没有极小极大，没有 KL，没有重参数化技巧。

**采样。** 从 `x_T ~ N(0, I)` 开始。从 `t = T` 到 `1` 迭代反向步。完成。

## 为什么有效

三个直觉：

1. **去噪容易；生成难。** 在 `t=T`，数据是纯噪声——网络要解决一个平凡问题。在 `t=0`，网络只需清理几个像素。在中间 `t`，问题很难但网络从每个噪声级别通过相同权重有许多梯度流过。

2. **伪装的 score matching。** Vincent（2011）证明预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即 *score*。反向 SDE 使用这个 score 沿密度梯度上行——向高概率区域的引导随机游走。

3. **ELBO 简化为简单 MSE。** 完整变分下界每个时间步有一个 KL 项。用 DDPM 的参数化，这些 KL 项简化为带特定系数的噪声预测 MSE；Ho 丢弃了系数（称之为"simple"损失），质量反而*提高了*。

## Build It

`code/main.py` 实现了一个 1-D DDPM。数据是双模混合。"网络"是一个小型 MLP，接收 `(x_t, t)` 并输出预测噪声。训练是一行损失。采样迭代反向链。

### Step 1: the forward schedule (closed form)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Step 2: sample `x_t` in one shot

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Step 3: one training step

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Step 4: reverse sampling

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于 40 个时间步和 24 单元 MLP 的 1-D 问题，这在约 200 个 epoch 内学会双模混合。

## 时间条件化

网络需要知道它在去噪哪个时间步。两个标准选项：

- **正弦嵌入。** 像 Transformer 位置编码。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过 MLP，广播到网络中。
- **FiLM / group-norm 条件化。** 将嵌入投影为每通道 scale/bias（FiLM）在每个块中。

我们的玩具代码使用正弦 → 拼接。生产 U-Net 使用 FiLM。

## Pitfalls

- **Schedule 很重要。** 线性 `β` 是 DDPM 默认但 cosine schedule（Nichol & Dhariwal, 2021）在相同计算下给出更好的 FID。如果质量停滞就切换 schedule。
- **时间步嵌入很脆弱。** 将原始 `t` 作为浮点数传递对玩具 1-D 有效但对图像失败；始终使用适当的嵌入。
- **V-prediction vs ε-prediction。** 对于窄区间（非常小或非常大的 t），`ε` 信噪比差。V-prediction（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 使用它。
- **Classifier-free guidance。** 推理时，计算条件和无条件 `ε`，然后 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，`w ≈ 3-7`。在第 08 课覆盖。
- **1000 步太多了。** 生产使用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。见第 12 课。

## Use It

| Role | Typical stack in 2026 |
|------|-----------------------|
| 图像像素空间扩散（小型，玩具） | DDPM + U-Net |
| 图像 latent diffusion | VAE encoder + U-Net 或 DiT (Lesson 07) |
| 视频 latent diffusion | Spatiotemporal DiT (Sora, Veo, WAN) |
| 音频 latent diffusion | Encodec + diffusion transformer |
| 科学（分子、蛋白质、物理） | Equivariant diffusion (EDM, RFdiffusion, AlphaFold3) |

扩散是通用生成骨干。Flow matching（第 13 课）是 2024-2026 的竞争者，通常在相同质量下推理速度更快。

## Ship It

保存 `outputs/skill-diffusion-trainer.md`。Skill 接收数据集 + 计算预算，输出：schedule（linear/cosine/sigmoid）、预测目标（ε/v/x）、步数、guidance scale、采样器家族和评估协议。

## Exercises

1. **Easy.** 在 `code/main.py` 中将 T 从 40 改为 10。样本质量（输出的视觉直方图）如何退化？在什么 T 下双模结构坍缩？
2. **Medium.** 从 ε-prediction 切换到 v-prediction。重新推导反向步。比较最终样本质量。
3. **Hard.** 添加 classifier-free guidance。条件化类别标签 `c ∈ {0, 1}`，训练时 10% 的时间丢弃它，采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。在 `w = 0, 1, 3, 7` 时测量条件模式命中率。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 前向过程 | "加噪声" | 固定马尔可夫链 `q(x_t \| x_{t-1})` 摧毁数据。 |
| 反向过程 | "去噪" | 学习的链 `p_θ(x_{t-1} \| x_t)` 重建数据。 |
| β schedule | "噪声阶梯" | 每步方差；linear、cosine 或 sigmoid。 |
| α̅ | "Alpha bar" | 累积乘积 `∏(1 - β)`；给出从 `x_0` 到 `x_t` 的封闭形式。 |
| Simple loss | "噪声上的 MSE" | `\|\|ε - ε_θ(x_t, t)\|\|²`；所有变分推导都坍缩为此。 |
| ε-prediction | "预测噪声" | 输出是添加的噪声；标准 DDPM。 |
| V-prediction | "预测速度" | 输出是 `α·ε - σ·x`；跨 t 更好的条件化。 |
| DDPM | "那篇论文" | Ho et al. 2020；linear β，1000 步，U-Net。 |
| DDIM | "确定性采样器" | 非马尔可夫采样器，20-50 步，相同训练目标。 |
| Classifier-free guidance | "CFG" | 混合条件和无条件噪声预测以放大条件化。 |

## 生产笔记：扩散推理是一个步数问题

DDPM 论文运行 T=1000 反向步。没人在生产中部署那个。每个真实推理栈选择三种策略之一——每种都清晰地映射到生产框架中"延迟从哪来"：

1. **更快的采样器，相同模型。** DDIM（20-50 步）、DPM-Solver++（10-20）、UniPC（8-16）。反向循环的即插即用替换；训练的 `ε_θ` 权重不变。延迟降低 20-50×。
2. **蒸馏。** 训练学生在更少步数中匹配教师：Progressive Distillation（2 → 1）、Consistency Models（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。延迟再降 5-10×，需要重新训练。
3. **缓存和编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散后端、`xformers`/SDPA attention、bf16 权重。每步延迟降低约 2×。与 (1) 和 (2) 叠加。

对于生产扩散服务器，预算对话与生产文献描述 LLM 的相同：延迟是 `num_steps × step_cost + VAE_decode`，吞吐量是 `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（一步）；TPOT 等价物是完整响应时间，因为从用户角度看图像生成是"一次性"的。

## Further Reading

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — the diffusion paper, ahead of its time.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM.
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM, fewer steps.
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — cosine schedule, learned variance.
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — classifier guidance.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — unified notation, cleanest recipe.
