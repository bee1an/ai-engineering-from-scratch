# Inpainting、Outpainting 与图像编辑

> 文本到图像创造新东西。Inpainting 修复旧东西。在生产中，70% 的可计费图像工作是编辑——换背景、去 logo、扩展画布、重新生成手。Inpainting 是扩散真正赚钱的地方。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**Time:** ~75 minutes

## 问题

客户发来一张完美的产品照片，背景有一个分散注意力的标志。你想擦除标志并保持其他一切像素相同。你不能从头运行文本到图像——结果会有不同的颜色、不同的光照、不同的产品角度。你想*只*重新生成遮罩区域，并且重新生成要尊重周围上下文。

这就是 inpainting。变体：

- **Inpainting。** 在遮罩内重新生成，保持外部像素。
- **Outpainting。** 在遮罩外（或画布之外）重新生成，保持内部。
- **图像编辑。** 重新生成整张图像但保持与原始的语义或结构保真度（SDEdit、InstructPix2Pix）。

2026 年每个扩散管线都附带 inpainting 模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们基于相同原理。

## 概念

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### 朴素方法（以及为什么它是错的）

用遮罩运行标准文本到图像。在每个采样步，将噪声 latent 的未遮罩区域替换为前向扩散的干净图像。它有效...效果很差。边界伪影渗透因为模型没有关于遮罩区域内容的信息。

### 正确的 inpainting 模型

训练一个修改的 U-Net，接收 9 个输入通道而非 4 个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外通道是 VAE 编码源图像的副本加上单通道遮罩。训练时，随机遮罩图像区域并训练模型只去噪遮罩区域，同时未遮罩区域作为干净条件信号给出。推理时，模型可以"看到"遮罩区域周围的内容并产生连贯的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这种 9 通道（或类似）输入。Diffusers `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit (Meng et al., 2022) — 免费编辑

给源图像加噪声到某个中间 `t`，然后用新 prompt 从 `t` 向下运行反向链到 0。无需重新训练。起始 `t` 的选择在保真度和创造自由之间权衡：

- `t/T = 0.3` → 几乎与源相同，小的风格变化
- `t/T = 0.6` → 中等编辑，保留粗结构
- `t/T = 0.9` → 从近噪声生成，最小源保留

### InstructPix2Pix (Brooks et al., 2023)

在 `(input_image, instruction, output_image)` 三元组上微调扩散模型。推理时，同时条件化输入图像和文本指令（"让它变成日落"、"加一条龙"）。两个 CFG scale：图像 scale 和文本 scale。

### RePaint (Lugmayr et al., 2022)

保持标准无条件扩散模型。在每个反向步，重采样——偶尔跳回更噪声的状态并重新生成。避免边界伪影。在没有训练过的 inpainting 模型时使用。

## Build It

`code/main.py` 在 5 维数据上实现了一个玩具 1-D inpainting 方案。我们在 5-D 混合数据上训练 DDPM，每个样本是来自两个聚类之一的 5 个浮点数。推理时，我们"遮罩"5 个维度中的 2 个，在每步注入未遮罩三个的噪声前向版本，只重新生成遮罩维度。

### Step 1: 5-D DDPM data

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Step 2: train denoiser over all 5 dims

标准 DDPM。网络为 5-D 噪声输入输出 5-D 噪声预测。

### Step 3: at inference, mask-aware reverse

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

这是朴素方法，在玩具 1-D 数据上有效。真实图像 inpainting 使用 9 通道输入因为纹理连贯性更重要。

### Step 4: outpainting

Outpainting 是遮罩反转的 inpainting：遮罩新的（之前不存在的）画布，用原始填充其余部分。相同的训练目标。

## Pitfalls

- **接缝。** 朴素方法留下可见边界因为梯度信息不跨遮罩流动。修复：将遮罩膨胀 8-16 像素，或使用正确的 inpainting 模型。
- **遮罩泄漏。** 如果条件图像的未遮罩区域质量低或有噪声，它会污染遮罩内的生成。轻微去噪或模糊。
- **CFG 与遮罩大小交互。** 小遮罩上的高 CFG = 饱和补丁。小编辑时降低 CFG。
- **SDEdit 保真度悬崖。** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能丢失主体身份。扫描并设检查点。
- **Prompt 不匹配。** Prompt 应描述*整张*图像，而不仅仅是新内容。"一只猫坐在椅子上"而不是"一只猫"。

## Use It

| Task | Pipeline |
|------|----------|
| 移除物体，小遮罩 | SD-Inpaint 或 Flux-Fill，标准 prompt |
| 替换天空 | SD-Inpaint + "blue sky at sunset" |
| 扩展画布 | SDXL outpaint mode (8px feather) 或 Flux-Fill with outpaint mask |
| 重新生成手/脸 | SD-Inpaint with prompt re-describing the subject + ControlNet-Openpose |
| 改变一个区域的风格 | SDEdit at `t/T=0.5` on masked region |
| "让它变成日落" | InstructPix2Pix 或 Flux-Kontext |
| 背景替换 | SAM mask → SD-Inpaint |
| 超高保真度 | Flux-Fill 或 GPT-Image (hosted) for hardest cases |

SAM（Meta 的 Segment Anything, 2023）+ 扩散 inpaint 是 2026 年的背景移除管线。SAM 2（2024）适用于视频。

## Ship It

保存 `outputs/skill-editing-pipeline.md`。Skill 接收原始图像 + 编辑描述 + 可选遮罩（或 SAM prompt），输出：遮罩生成方法、基础模型、CFG scales（图像 + 文本）、SDEdit-t 或 inpainting 模式和 QA 清单。

## Exercises

1. **Easy.** 在 `code/main.py` 中，将遮罩维度比例从 0.2 变到 0.8。在什么比例下 inpaint 质量（遮罩维度的残差）等于无条件生成？
2. **Medium.** 实现 RePaint：每 10 个反向步，跳回 5 步（加噪声）并重新去噪。测量它是否减少遮罩边缘的边界残差。
3. **Hard.** 用 Hugging Face diffusers 比较：SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill 在 20 个人脸重新生成任务上。分别评分姿态遵循度和身份保持。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Inpainting | "填洞" | 在遮罩内重新生成；保持外部像素。 |
| Outpainting | "扩展画布" | 在画布外重新生成；保持内部。 |
| 9-channel U-Net | "正确的 inpainting 模型" | U-Net 以 `noisy \| encoded-source \| mask` 为输入。 |
| SDEdit | "带噪声级别的 Img2img" | 加噪到时间 `t`，用新 prompt 去噪。 |
| InstructPix2Pix | "纯文本编辑" | 在 (image, instruction, output) 三元组上微调的扩散。 |
| RePaint | "无需重新训练" | 反向过程中周期性重新加噪以减少接缝。 |
| SAM | "Segment Anything" | 通过点击或框生成遮罩；与 inpaint 配对。 |
| Flux-Kontext | "带上下文的编辑" | 接受参考图像 + 指令进行编辑的 Flux 变体。 |

## 生产笔记：编辑管线对延迟敏感

用户编辑图像期望亚 5 秒的往返。30 步 SDXL-Inpaint 在 1024² 上 L4 需要 3-4 秒，加上 SAM 遮罩生成（约 200 ms）和 VAE encode/decode（合计约 500 ms）。在生产框架中，这是 TTFT 受限而非吞吐量受限——batch 1，低并发，最小化每个阶段：

- **SAM-H 是慢的那个。** SAM-H 在 1024² 约 200 ms；SAM-ViT-B 约 40 ms 质量损失轻微。SAM 2（视频）增加时间开销；不要用于单图编辑。
- **尽可能跳过编码。** `pipe.image_processor.preprocess(img)` 编码到 latent。如果你有上次生成的 latent（迭代编辑 UI 中典型），通过 `latents=...` 直接传递以跳过一次 VAE encode。
- **遮罩膨胀也影响吞吐量。** 小遮罩意味着大部分 U-Net 前向传播是浪费的（未遮罩像素反正被钳位）。`diffusers` 的 `StableDiffusionInpaintPipeline` 无论如何运行完整 U-Net；只有 9 通道正确 inpaint 变体利用遮罩计算。
- **Flux-Kontext 是 2025 年的答案。** 对 `(source_image, instruction)` 的单次前向传播——没有单独遮罩，没有 SDEdit 噪声扫描。在 H100 上约 1.5 秒完成编辑。架构教训：合并阶段。

## Further Reading

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — training-free inpainting.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — text-instruction editing.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM, the mask source.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — video SAM.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — attention-level editing.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 tooling.
