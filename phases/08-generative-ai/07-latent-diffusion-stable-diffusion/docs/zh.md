# Latent Diffusion 与 Stable Diffusion

> 在 512×512 图像的像素空间做扩散是计算上的暴行。Rombach et al.（2022）注意到你不需要全部 786k 维来生成图像——你需要足够捕获语义结构的维度，再用一个单独的解码器处理其余部分。在 VAE 的 latent 空间中运行扩散。这一个想法就是 Stable Diffusion。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**Time:** ~75 minutes

## 问题

像素空间扩散在 512² 意味着 U-Net 在形状为 `[B, 3, 512, 512]` 的张量上运行。每个采样步对 500M 参数 U-Net 约 100 GFLOPS。五十步就是每张图像 5 TFLOPS。在十亿张图像上训练，计算账单是荒谬的。

大部分 FLOPs 花在将感知上不重要的细节推过网络——有损 VAE 可以压缩掉的高频纹理。Rombach 的想法：训练一次 VAE（*第一阶段*），冻结它，完全在 4 通道 64×64 latent 空间（*第二阶段*）中运行扩散。相同的 U-Net。1/16 的像素。相当质量下约 64 倍更少的 FLOPs。

这就是 Stable Diffusion 的配方。SD 1.x / 2.x 使用 860M U-Net 在 `64×64×4` latent 上，SDXL 使用 2.6B U-Net 在 `128×128×4` 上，SD3 将 U-Net 换成了带 flow matching 的 Diffusion Transformer（DiT）。Flux.1-dev（Black Forest Labs, 2024）部署了 12B 参数的 DiT-MMDiT。全部运行在相同的两阶段基底上。

## 概念

![Latent diffusion: VAE compression + diffusion in latent space](../assets/latent-diffusion.svg)

**两个阶段，分别训练。**

1. **阶段 1 — VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩：每个空间轴 8× 下采样 + 调整通道使总 latent 大小约为像素数的 1/16。损失 = 重建（L1 + LPIPS 感知）+ KL（小权重使 `z` 不被强制太高斯，因为我们不需要从 `z` 精确采样）。通常用对抗损失训练使解码图像锐利。

2. **阶段 2 — 在 `z` 上扩散。** 将 `z = E(x_real)` 视为数据。训练 U-Net（或 DiT）去噪 `z_t`。推理时：通过扩散采样 `z_0`，然后 `x = D(z_0)`。

**文本条件化。** 两个额外组件。冻结的文本编码器（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。交叉注意力注入：每个 U-Net 块取 `[Q = 图像特征, K = V = 文本 token]` 并混合。Token 是文本影响图像的唯一方式。

**损失函数与第 06 课相同。** 相同的 DDPM / flow matching 噪声 MSE。你只是换了数据域。

## 架构变体

| Model | Year | Backbone | Latent shape | Text encoder | Params |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | same | 1-4 step sampling |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 step |

趋势：用 DiT（latent patch 上的 transformer）替换 U-Net，扩大文本编码器（T5 在 prompt 遵循度上击败 CLIP），增加 latent 通道（4 → 16 给出更多细节余量）。

## Build It

`code/main.py` 在第 06 课的 DDPM 之上叠加了一个玩具 1-D "VAE"（恒等编码器 + 解码器，用于演示；真正的 VAE 是卷积网络），并添加了带 classifier-free guidance 的类条件化。它展示了相同的扩散损失无论在原始 1-D 值还是编码值上都有效——这是关键洞察。

### Step 1: encoder/decoder

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

真正的 VAE 有训练权重。为了教学，这个线性映射足以展示扩散在 `z` 上操作而不关心原始数据空间。

### Step 2: diffusion in `z`-space

与第 06 课相同的 DDPM。网络看到的数据是 `z = E(x)`。采样 `z_0` 后，用 `D(z_0)` 解码。

### Step 3: classifier-free guidance

训练时，10% 的时间丢弃类别标签（替换为 null token）。推理时，计算 `ε_cond` 和 `ε_uncond`，然后：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无 guidance（完全多样性），`w = 3` = 默认，`w = 7+` = 饱和 / 过度锐利。

### Step 4: text conditioning (concept, not code)

将类别标签替换为冻结文本编码器输出。通过交叉注意力将文本嵌入送入 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这是类条件扩散模型和 Stable Diffusion 之间唯一的实质性区别。

## Pitfalls

- **VAE 缩放不匹配。** SD 1.x VAE 在编码后应用一个缩放常数（`scaling_factor ≈ 0.18215`）。忘记这个会使 U-Net 在方差完全错误的 latent 上训练。每个 checkpoint 都附带一个。
- **文本编码器静默出错。** SD3 需要 T5-XXL 且 >=128 token，回退到仅 CLIP 是有损的。始终检查 `use_t5=True` 否则 prompt 保真度会崩溃。
- **混合 latent 空间。** SDXL、SD3、Flux 都使用不同的 VAE。在 SDXL latent 上训练的 LoRA 不会在 SD3 上工作。Hugging Face diffusers 0.30+ 拒绝加载不匹配的 checkpoint。
- **CFG 太高。** `w > 10` 产生饱和、油腻的图像，过度拟合 prompt 而牺牲多样性。最佳点是 `w = 3-7`。
- **负面 prompt 泄漏。** 空负面 prompt 变成 null token；填充的负面 prompt 变成 `ε_uncond`。这不是同一回事；某些管线静默默认为 null。

## Use It

2026 年的生产技术栈：

| Target | Recommended backbone |
|--------|----------------------|
| 窄领域，配对数据，从头训练模型 | SDXL fine-tune (LoRA / full) — 最快部署 |
| 开放领域文本到图像，开源权重 | Flux.1-dev (12B, Apache / non-commercial) 或 SD3.5-Large |
| 最快推理，开源权重 | Flux.1-schnell (1-4 step, Apache) 或 SDXL-Lightning |
| 最佳 prompt 遵循度，托管 | GPT-Image / DALL-E 3 (still), Midjourney v7, Imagen 4 |
| 编辑工作流 | Flux.1-Kontext (Dec 2024) — 原生接受图像 + 文本 |
| 研究，基线 | SD 1.5 — 古老但研究充分 |

## Ship It

保存 `outputs/skill-sd-prompter.md`。Skill 接收文本 prompt + 目标风格，输出：模型 + checkpoint、CFG scale、采样器、负面 prompt、分辨率、可选 ControlNet/IP-Adapter 组合和逐步 QA 清单。

## Exercises

1. **Easy.** 用 guidance `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。记录每类的平均样本。在什么 `w` 下类均值超过真实数据均值？
2. **Medium.** 将玩具线性编码器换成带重建损失的 tanh-MLP 编码器/解码器对。在新 latent 上重新训练扩散。样本质量是否改变？
3. **Hard.** 用 diffusers 设置真实的 Stable Diffusion 推理：加载 `sdxl-base`，运行 30 Euler 步 CFG=7，计时。然后切换到 `sdxl-turbo` 4 步 CFG=0。相同主题，不同质量——描述什么改变了以及为什么。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 第一阶段 | "The VAE" | 训练的编码器/解码器对；将 512² 压缩到 64²。 |
| 第二阶段 | "The U-Net" | latent 空间上的扩散模型。 |
| CFG | "Guidance scale" | `(1+w)·ε_cond - w·ε_uncond`；调节条件化强度。 |
| Null token | "空 prompt 嵌入" | 用于 `ε_uncond` 的无条件嵌入。 |
| Cross-attention | "文本如何进入" | 每个 U-Net 块将文本 token 作为 K 和 V 进行注意力。 |
| DiT | "Diffusion Transformer" | 用 latent patch 上的 transformer 替换 U-Net；扩展性更好。 |
| MMDiT | "Multi-modal DiT" | SD3 的架构：文本和图像流带联合注意力。 |
| VAE scaling factor | "魔法数字" | 将 latent 除以约 5.4 使扩散在单位方差空间中操作。 |

## 生产笔记：在 8GB 消费级 GPU 上运行 Flux-12B

参考 Flux 集成是典型的"我有消费级 GPU，能部署吗？"配方。技巧是生产推理文献列出的相同三旋钮配方应用于扩散 DiT：

1. **交错加载。** Flux 有三个网络永远不需要同时存在于 VRAM 中：T5-XXL 文本编码器（fp32 约 10 GB）、CLIP-L（小）、12B MMDiT 和 VAE。先编码 prompt，*删除*编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。消费级 8GB GPU 一次只能放一个阶段。
2. **通过 bitsandbytes 做 4-bit 量化。** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` 用于 T5 编码器和 DiT。内存降低 8×，文本到图像的质量下降不可感知。
3. **CPU offload。** `pipe.enable_model_cpu_offload()` 在每次前向传播推进时自动在 CPU 和 GPU 之间交换模块。增加 10-20% 延迟但使管线能运行。

内存核算：`10 GB T5 / 8 = 1.25 GB` 量化，`12B params × 0.5 bytes = ~6 GB` 量化 DiT，加上激活。用 stas00 的术语这是 TP=1 推理的极端端——没有模型并行，最大量化。生产中你会在 H100 上运行 TP=2 或 TP=4；对于单个开发笔记本，这就是配方。

## Further Reading

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 family.
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — reference implementation for every checkpoint above.
