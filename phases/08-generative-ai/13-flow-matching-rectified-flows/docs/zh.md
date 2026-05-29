# Flow Matching 与 Rectified Flows

> 扩散模型需要 20-50 个采样步因为它们走的是从噪声到数据的弯曲路径。Flow matching（Lipman et al., 2023）和 rectified flow（Liu et al., 2022）训练了直线路径。更直的路径意味着更少步数意味着更快推理。Stable Diffusion 3、Flux.1 和 AudioCraft 2 都在 2024 年切换到了 flow matching。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 06 (DDPM), Phase 1 · Calculus
**Time:** ~45 minutes

## 问题

DDPM 的反向过程是从 `N(0, I)` 回到数据分布的 1000 步随机游走。DDIM 将其压缩到 20-50 个确定性步。你想要更少步——理想情况下一步。障碍是求解反向过程的 ODE 是刚性的；路径是弯曲的。

如果你能训练模型使得从噪声到数据的路径是*直线*，从 `t=1` 到 `t=0` 的单个 Euler 步就能工作。Flow matching 直接构建这个：定义从 `x_1 ∼ N(0, I)` 到 `x_0 ∼ data` 的直线插值，训练向量场 `v_θ(x, t)` 匹配其时间导数，推理时积分。

Rectified flow（Liu 2022）更进一步：用 reflow 程序迭代地拉直路径，产生逐渐接近线性的 ODE。两次 reflow 迭代后，2 步采样器匹配 50 步 DDPM 质量。

## 概念

![Flow matching: straight-line interpolation between noise and data](../assets/flow-matching.svg)

### 直线 flow

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 `x_0 ~ data`，`x_1 ~ N(0, I)`。沿这条直线的时间导数是常数：

```
dx_t / dt = x_1 - x_0
```

定义神经向量场 `v_θ(x_t, t)` 并训练它匹配这个导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这是 **conditional flow matching** 损失（Lipman 2023）。训练无需模拟：你从不展开 ODE。只需采样 `(x_0, x_1, t)` 并回归。

### 采样

推理时，在时间上*反向*积分学到的向量场：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 `x_1 ~ N(0, I)` 开始，Euler 步降到 `t=0`。

### Rectified flow (Liu 2022)

直线 flow 有效但学到的路径*实际上不是直的*——它们弯曲因为许多 `x_0` 可以映射到同一个 `x_1`。Rectified flow 的 reflow 步骤：

1. 用随机配对训练 flow 模型 v_1。
2. 通过从 `x_1` 积分 v_1 到其着陆点 `x_0` 采样 N 对 `(x_1, x_0)`。
3. 在这些配对样本上训练 v_2。因为配对现在是"ODE 匹配的"，它们之间的直线插值真正更平。
4. 重复。

实践中 2 次 reflow 迭代就能接近线性，实现 2-4 步推理。SDXL-Turbo、SD3-Turbo、LCM 都是从 flow-matching 模型蒸馏的。

### 为什么这在 2024 年赢了图像

三个原因：

1. **无需模拟的训练** — 训练时不展开 ODE，实现简单。
2. **更好的损失几何** — 直线路径有一致的信噪比，而 DDPM ε-loss 在 schedule 边缘信噪比差。
3. **更快推理** — SDXL-Turbo 质量下 4-8 步；consistency distillation 下 1 步。

## Flow matching vs DDPM — 精确联系

带高斯条件路径的 flow matching 是扩散*加特定噪声 schedule*。选择 `x_t = α(t) x_0 + σ(t) x_1` schedule，flow matching 恢复 Stratonovich 重新表述的扩散，`v = α'·x_0 - σ'·x_1`。对于高斯路径两者代数等价。

Flow matching 添加的是：目标的*清晰度*（一个普通速度）、更干净的损失、以及实验非高斯插值的许可。

## Build It

`code/main.py` 在双模高斯混合上实现 1-D flow matching。向量场 `v_θ(x, t)` 是用直线目标训练的小型 MLP。推理时，积分 1、2、4 和 20 个 Euler 步并比较样本质量。

### Step 1: training loss

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Step 2: multi-step inference

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Step 3: compare step counts

预期 4 步采样器已经匹配 20 步质量——对延迟来说是大事。

## Pitfalls

- **时间参数化。** Flow matching 使用 `t ∈ [0, 1]`，`t=0` 在数据，`t=1` 在噪声。DDPM 使用 `t ∈ [0, T]`，`t=0` 在数据，`t=T` 在噪声。方向相同，尺度不同。论文经常搞错。
- **Schedule 选择。** Rectified flow 的直线是"那个" flow-matching schedule，但你可以用 cosine 或 logit-normal t-sampling（SD3 这样做）以获得更好的尺度覆盖。
- **Reflow 成本。** 为 reflow 生成配对数据集是每样本一次完整推理 pass。只在你真正需要 1-2 步推理时做 reflow。
- **Classifier-free guidance 仍然适用。** 只需将 ε 换成 v 在线性组合中：`v_cfg = (1+w) v_cond - w v_uncond`。

## Use It

| Use case | 2026 stack |
|----------|-----------|
| 文本到图像，最佳质量 | Flow matching: SD3, Flux.1-dev |
| 文本到图像，1-4 步 | Distilled flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| 实时推理 | Consistency distillation from a flow-matched base (LCM, PCM) |
| 音频生成 | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| 视频生成 | Flow matching mixed with diffusion (Sora, Veo, Stable Video) |
| 科学 / 物理（粒子轨迹、分子） | Flow matching + equivariant vector field |

2025-2026 年论文说"比扩散更快"时，几乎总是 flow matching + distillation。

## Ship It

保存 `outputs/skill-fm-tuner.md`。Skill 接收扩散式模型规格并转换为 flow-matching 训练配置：schedule 选择、时间采样分布（uniform / logit-normal）、优化器、reflow 计划、目标步数、评估协议。

## Exercises

1. **Easy.** 运行 `code/main.py` 并比较 1 步 vs 20 步 MSE vs 真实数据分布。
2. **Medium.** 从 uniform `t` 采样切换到 logit-normal（集中采样在中间 t）。模型质量是否提高？
3. **Hard.** 实现一次 reflow 迭代：通过积分第一个模型生成配对 (x_0, x_1)，在配对上训练第二个模型，比较 1 步样本质量。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Flow matching | "直线扩散" | 训练 `v_θ(x, t)` 沿插值匹配 `x_1 - x_0`。 |
| Rectified flow | "Reflow" | 拉直学到的 flow 的迭代程序。 |
| 速度场 | "v_θ" | 模型的输出——移动 `x_t` 的方向。 |
| 直线插值 | "路径" | `x_t = (1-t)·x_0 + t·x_1`；平凡的目标导数。 |
| Euler sampler | "一阶 ODE 求解器" | 最简单的积分器；路径直时效果好。 |
| Logit-normal t | "SD3 采样" | 将 `t` 采样集中在梯度最强的中间值。 |
| Consistency distillation | "1 步采样器" | 训练学生将任何 `x_t` 直接映射到 `x_0`。 |
| CFG with velocity | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`；相同技巧，新变量。 |

## 生产笔记：Flux.1-schnell 是 flow matching 最快的形态

Flow matching 的生产胜利是 Flux.1-schnell——一个蒸馏到 1-4 推理步同时保持 Flux-dev 级质量的 flow-matched DiT。Niels 的"在 8GB 机器上运行 Flux"notebook 是参考部署配方：T5 + CLIP 编码，量化 MMDiT 去噪（schnell 4 步 vs dev 50 步），VAE 解码。成本核算：

| Variant | Steps | Latency at 1024² on L4 | Total FLOPs (relative) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (raw) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (12× faster) |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

生产规则：**flow-matched base + distillation = 2026 年快速文本到图像的默认。** 每个主要供应商都部署这个组合：SD3-Turbo（SD3 + flow + distillation）、Flux-schnell（Flux-dev + rectified-flow straightening）、CogView-4-Flash。纯扩散基础仅存在于遗留 checkpoint。

## Further Reading

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, rectified flow at scale.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — general framework that covers FM + diffusion.
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — 1-step distillation of diffusion / flow.
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo variant.
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — flow matching in production.
