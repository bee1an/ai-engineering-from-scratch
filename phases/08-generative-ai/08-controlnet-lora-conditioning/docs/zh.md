# ControlNet、LoRA 与条件化

> 仅靠文本是一个笨拙的控制信号。ControlNet 让你克隆一个预训练扩散模型并用深度图、姿态骨架、涂鸦或边缘图来引导它。LoRA 让你通过训练 1000 万参数来微调一个 20 亿参数的模型。它们一起将 Stable Diffusion 从玩具变成了 2026 年每家机构都在部署的图像管线。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch — for LoRA foundation)
**Time:** ~75 minutes

## 问题

像"一个穿红裙的女人在繁忙街道上遛狗"这样的 prompt 没有给模型任何关于*狗在哪里*、*女人什么姿态*或*街道的透视*的信息。文本只能确定你需要指定的图像的约 10%。其余是视觉的，无法用文字高效描述。

为每个信号（姿态、深度、canny、分割）从头训练新的条件模型是不可行的。你想保持 2.6B 参数的 SDXL 骨干冻结，附加一个小型侧网络读取条件，让它微调骨干的中间特征。这就是 ControlNet。

你还想教模型新概念（你的脸、你的产品、你的风格）而不重新训练完整模型。你想要一个 100 倍更小的增量。这就是 LoRA——插入现有注意力权重的低秩适配器。

ControlNet + LoRA + text = 2026 年从业者的工具包。大多数生产图像管线在 SDXL / SD3 / Flux 基础上叠加 2-5 个 LoRA、1-3 个 ControlNet 和一个 IP-Adapter。

## 概念

![ControlNet clones the encoder; LoRA adds low-rank deltas](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

取一个预训练的 SD。*克隆* U-Net 的编码器半部分。冻结原始的。训练克隆接受额外的条件输入（边缘、深度、姿态）。通过*零卷积*跳跃连接（初始化为零的 1×1 卷积——开始时是 no-op，学习一个增量）将克隆连接回原始的解码器半部分。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

零卷积初始化意味着 ControlNet 开始时是恒等——训练前也无害。在 1M (prompt, condition, image) 三元组上用标准扩散损失训练。

每模态 ControlNet 作为小型侧模型发布（SDXL 约 360M，SD 1.5 约 70M）。推理时可以组合：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

对模型中任何线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。注意力的 rank 4-16 是标准，重度微调用 rank 64-128。新参数数量：`2 · d · r` 而非 `d²`。对于 SDXL 注意力 `d=640`，`r=16`：每个适配器 20k 参数而非 410k——20 倍缩减。整个模型：LoRA 通常 20-200MB vs 基础 5GB。

推理时可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是正常的。多个 LoRA 加性叠加（通常的警告是它们以非线性方式交互）。

### IP-Adapter (Ye et al., 2023)

一个小型适配器接受*图像*作为条件（与文本并列）。使用 CLIP 图像编码器产生图像 token，在交叉注意力中与文本 token 一起注入。每个基础模型约 20MB。让你做"以这个参考的风格生成图像"而不需要 LoRA。

## 可组合性矩阵

| Tool | What it controls | Size | When to use |
|------|------------------|------|-------------|
| ControlNet | 空间结构（姿态、深度、边缘） | 70-360MB | 精确布局、构图 |
| LoRA | 风格、主体、概念 | 20-200MB | 个性化、风格 |
| IP-Adapter | 来自参考图像的风格或主体 | 20MB | 文本无法描述的外观 |
| Textual Inversion | 单个概念作为新 token | 10KB | 遗留，大部分被 LoRA 替代 |
| DreamBooth | 对主体的完整微调 | 2-5GB | 强身份，高计算 |
| T2I-Adapter | 更轻的 ControlNet 替代 | 70MB | 边缘设备，推理预算 |

ControlNet ≈ 空间。LoRA ≈ 语义。两者都用。

## Build It

`code/main.py` 在 1-D 上模拟这两种机制：

1. **LoRA。** 一个预训练线性层 `W`。冻结它。训练低秩 `B @ A` 使得 `W + BA` 匹配目标线性层。展示 `r = 1` 足以完美学习 rank-1 修正。

2. **ControlNet-lite。** 一个"冻结基础"预测器和一个读取额外信号的"侧网络"。侧网络的输出由一个初始化为零的可学习标量门控（我们版本的零卷积）。训练并观察门逐步升高。

### Step 1: LoRA math

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Step 2: zero-init side network

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

在步骤 0 输出与 base 相同。早期训练缓慢更新 `gate`——没有灾难性漂移。

## Pitfalls

- **过度缩放 LoRA。** `α = 2` 或 `α = 3` 是常见的"让它更强"hack，会产生过度风格化/破损的输出。保持 `α ≤ 1.5`。
- **ControlNet 权重冲突。** 同时使用权重 1.0 的 Pose ControlNet 和权重 1.0 的 Depth ControlNet 通常会过冲。权重之和 ≈ 1.0 是安全默认。
- **LoRA 用在错误的基础上。** SDXL LoRA 在 SD 1.5 上静默无效因为注意力维度不匹配。Diffusers 0.30+ 会警告。
- **Textual Inversion 漂移。** 在一个 checkpoint 上训练的 token 在另一个上严重漂移。LoRA 更可移植。
- **LoRA 权重合并和存储。** 你可以将 LoRA 烘焙到基础模型权重中以加快推理（无运行时加法），但你失去了运行时缩放 `α` 的能力。保留两个版本。

## Use It

| Goal | 2026 pipeline |
|------|---------------|
| 复现品牌艺术风格 | 在约 30 张精选图像上以 rank 32 训练的 LoRA |
| 将我的脸放入生成图像 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 特定姿态 + prompt | ControlNet-Openpose + SDXL + text |
| 深度感知构图 | ControlNet-Depth + SD3 |
| 参考 + prompt | IP-Adapter + text |
| 精确布局 | ControlNet-Scribble 或 ControlNet-Canny |
| 背景替换 | ControlNet-Seg + Inpainting (Lesson 09) |
| 快速 1 步风格 | LCM-LoRA on SDXL-Turbo |

## Ship It

保存 `outputs/skill-sd-toolkit-composer.md`。Skill 接收任务（输入资产：prompt、可选参考图像、可选姿态、可选深度、可选涂鸦），输出工具栈、权重和可复现的种子协议。

## Exercises

1. **Easy.** 在 `code/main.py` 中，将 LoRA rank `r` 从 1 变到 4。在什么 rank 下 LoRA 精确匹配 rank-2 目标增量？
2. **Medium.** 在两个目标变换上训练两个独立的 LoRA。一起加载它们并展示它们的加性交互。什么时候交互打破线性？
3. **Hard.** 用 diffusers 叠加：SDXL-base + Canny-ControlNet (weight 0.8) + 风格 LoRA (α 0.8) + IP-Adapter (weight 0.6)。测量随栈权重变化的 FID-vs-prompt-adherence 权衡。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| ControlNet | "空间控制" | 克隆编码器 + 零卷积跳跃；读取条件图像。 |
| Zero convolution | "开始时是恒等" | 初始化为零的 1×1 卷积；ControlNet 开始时是 no-op。 |
| LoRA | "低秩适配器" | `W + B @ A`，`r << d`；比完整微调少 100 倍参数。 |
| rank r | "旋钮" | LoRA 压缩；4-16 典型，64+ 用于重度个性化。 |
| α | "LoRA 强度" | LoRA 增量的运行时缩放。 |
| IP-Adapter | "参考图像" | 通过 CLIP 图像 token 的小型图像条件适配器。 |
| DreamBooth | "完整主体微调" | 在约 30 张主体图像上训练完整模型。 |
| Textual Inversion | "新 token" | 仅学习新词嵌入；遗留，大部分被替代。 |

## 生产笔记：LoRA 交换、ControlNet 通道、多租户服务

真实的文本到图像 SaaS 在同一基础 checkpoint 上服务数百个 LoRA 和十几个 ControlNet。服务问题很像 LLM 多租户（生产文献在 continuous batching 和 LoRAX / S-LoRA 下覆盖 LLM 情况）：

- **热交换 LoRA，不要合并。** 将 `W' = W + α·B·A` 合并到基础中给出约 3-5% 更快的每步推理但冻结了 `α` 和基础。将 LoRA 作为 rank-r 增量保持在 VRAM 中；diffusers 暴露 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 用于每请求激活。交换成本是 `2 · d · r · num_layers` 权重——MB 级，亚秒。
- **ControlNet 作为第二注意力通道。** 克隆编码器与基础并行运行。两个权重 1.0 的 ControlNet = 每步两个额外前向传播，不是一个合并 pass。Batch-size 余量二次方下降。每个活跃 ControlNet 预算约 1.5× 步成本。
- **量化 LoRA 也行。** 如果你量化了基础（见第 07 课，8GB 上的 Flux），LoRA 增量也可以干净地量化到 8-bit 或 4-bit。QLoRA 式加载让你在 4-bit Flux 基础上叠加 5-10 个 LoRA 而不爆内存。

Flux 特定：Niels 的 Flux-on-8GB notebook 将基础量化到 4-bit；在该量化基础上叠加风格 LoRA（`pipe.load_lora_weights("user/style-lora")`）在 `weight_name="pytorch_lora_weights.safetensors"` 仍然有效。这是 2026 年大多数 SaaS 机构部署的配方。

## Further Reading

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA (originally for LLMs; ports to diffusion).
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — lighter alternative to ControlNet.
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — reference pipelines.
