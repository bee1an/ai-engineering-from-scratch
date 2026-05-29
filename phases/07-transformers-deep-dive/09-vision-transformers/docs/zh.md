# Vision Transformer (ViT)

> 图像是 patch 的网格。句子是 token 的网格。同一个 transformer 两者通吃。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers intro)
**Time:** ~45 minutes

## 问题

2020 年之前，计算机视觉就是卷积。ImageNet、COCO 和检测基准上的每一个 SOTA 都用 CNN 骨干网络。Transformer 是语言模型的事。

Dosovitskiy et al. (2020) — "An Image is Worth 16x16 Words" — 证明你可以完全抛弃卷积。把图像切成固定大小的 patch，线性投影每个 patch 为一个 embedding，把这个序列喂给一个标准 transformer encoder。在足够大的规模下（ImageNet-21k 预训练或更大），ViT 能追平甚至超越基于 ResNet 的模型。

ViT 开启了 2026 年的一个更广泛的模式：一种架构，多种模态。Whisper 把音频 token 化。ViT 把图像 token 化。机器人用动作 token。视频用像素 token。Transformer 不在乎——给它一个序列，它就能学。

到 2026 年，ViT 及其后代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）占据了视觉领域的大部分。CNN 在边缘设备和延迟敏感任务上仍然胜出。其他一切的技术栈里都有一个 ViT。

## 概念

![Image → patches → tokens → transformer](../assets/vit.svg)

### 第 1 步 — patch 化

将一个 `H × W × C` 的图像切成 `N × (P·P·C)` 的扁平 patch 序列。典型配置：`224 × 224` 图像，`16 × 16` patch → 196 个 patch，每个 768 维。

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Patch 大小是关键杠杆。更小的 patch = 更多 token、更高分辨率、二次方注意力开销。更大的 patch = 更粗糙、更便宜。

### 第 2 步 — 线性 embedding

一个可学习矩阵将每个扁平 patch 投影到 `d_model`。等价于一个 kernel size 为 `P`、stride 为 `P` 的卷积。在 PyTorch 里就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` — 两行代码搞定。

### 第 3 步 — 前置 `[CLS]` token，加位置编码

- 前置一个可学习的 `[CLS]` token。它的最终隐藏状态就是用于分类的图像表示。
- 加可学习位置编码（ViT 原版）或 2D 正弦编码（后续变体）。
- 2024 年以后 RoPE 扩展到 2D 位置编码，有时不需要显式 embedding。

### 第 4 步 — 标准 transformer encoder

堆叠 L 个 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 块。和 BERT 完全一样。没有视觉专用层。这是论文的教学核心。

### 第 5 步 — 输出头

分类任务：取 `[CLS]` 隐藏状态 → linear → softmax。对于 DINOv2 或 SAM，丢弃 `[CLS]`，直接使用 patch embedding。

### 重要变体

| Model | Year | Change |
|-------|------|--------|
| ViT | 2020 | The original. Fixed patch size, full global attention. |
| DeiT | 2021 | Distillation; trainable on ImageNet-1k only. |
| Swin | 2021 | Hierarchical with shifted windows. Fixed sub-quadratic cost. |
| DINOv2 | 2023 | Self-supervised (no labels). Best general vision features. |
| ViT-22B | 2023 | 22B params; scaling laws apply. |
| SigLIP | 2023 | ViT + language pair, sigmoid contrastive loss. |
| SAM 3 | 2025 | Segment anything; ViT-Large + promptable mask decoder. |

### 为什么花了这么久

ViT 需要*大量*数据才能追平 CNN，因为它没有 CNN 的归纳偏置（平移不变性、局部性）。没有超过 1 亿张标注图像或强自监督预训练，CNN 在同等算力下仍然胜出。DeiT 在 2021 年用蒸馏技巧解决了这个问题；DINOv2 在 2023 年用自监督彻底解决了它。

## 动手构建

见 `code/main.py`。纯标准库实现 patch 化 + 线性 embedding + 正确性检查。不做训练——任何现实规模的 ViT 都需要 PyTorch 和数小时 GPU 时间。

### 第 1 步：伪造图像

一个 24 × 24 RGB 图像，用 `(R, G, B)` 元组的行列表表示。我们用 6×6 patch → 16 个 patch，每个 108 维 embedding 向量。

### 第 2 步：patch 化

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：行优先遍历网格。每个 ViT 都用这个顺序。

### 第 3 步：线性 embed

将每个扁平 patch 乘以一个随机 `(patch_flat_size, d_model)` 矩阵。验证输出形状在前置 `[CLS]` 后为 `(N_patches + 1, d_model)`。

### 第 4 步：计算真实 ViT 的参数量

打印 ViT-Base 的参数量：12 层、12 头、d=768、patch=16。与 ResNet-50（~25M）对比。ViT-Base 约 ~86M。ViT-Large ~307M。ViT-Huge ~632M。

## 使用方式

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 embedding 是 2026 年图像特征的默认选择。** 冻结骨干网络，训练一个小头。适用于分类、检索、检测、描述生成。Meta 的 DINOv2 checkpoint 在所有非文本视觉任务上都优于 CLIP。

**Patch 大小选择。** 小模型用 16×16（ViT-B/16）。密集预测（分割）用 8×8 或 14×14（SAM、DINOv2）。超大模型用 14×14。

## 交付产出

见 `outputs/skill-vit-configurator.md`。该 skill 根据数据集大小、分辨率和算力预算，为新视觉任务选择 ViT 变体和 patch 大小。

## 练习

1. **简单。** 运行 `code/main.py`。验证 patch 数量等于 `(H/P) * (W/P)`，扁平 patch 维度等于 `P*P*C`。
2. **中等。** 实现 2D 正弦位置编码——为每个 patch 的 `row` 和 `col` 分别生成独立的正弦编码，然后拼接。将其输入一个小型 PyTorch ViT，在 CIFAR-10 上与可学习位置编码对比准确率。
3. **困难。** 构建一个 3 层 ViT（PyTorch），在 1,000 张 MNIST 图像上用 4×4 patch 训练。测量测试准确率。然后在同样 1,000 张图像上加入 DINOv2 预训练（简化版：训练 encoder 从 masked patch 预测 patch embedding）。准确率是否提升？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Patch | "The vision-transformer token" | Flat vector of pixel values for a `P × P × C` region of the image. |
| Patchify | "Chop + flatten" | Slice image into non-overlapping patches, flatten each to a vector. |
| `[CLS]` token | "The image summary" | Prepended learnable token; its final embedding is the image representation. |
| Inductive bias | "What the model assumes" | ViT has fewer priors than CNNs; needs more data to make up the gap. |
| DINOv2 | "Self-supervised ViT" | Trained without labels using image augmentation + momentum teacher. Best general image features in 2026. |
| SigLIP | "CLIP's successor" | ViT + text encoder trained with sigmoid contrastive loss; better than CLIP on matched compute. |
| Swin | "Windowed ViT" | Hierarchical ViT with local attention + shifted windows; sub-quadratic. |
| Register tokens | "2023 trick" | A few extra learnable tokens that soak up attention sinks; improves DINOv2 features. |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — the ViT paper.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — the register-token fix for DINOv2.
