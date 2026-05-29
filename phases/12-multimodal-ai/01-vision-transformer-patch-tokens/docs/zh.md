# Vision Transformer 与 Patch-Token 原语

> 在做任何多模态工作之前，图像必须先变成 transformer 能处理的 token 序列。2020 年的 ViT 论文用 16x16 像素 patch、一个线性投影和位置编码回答了这个问题。五年后，2026 年的每个前沿模型（Claude Opus 4.7 原生支持 2576px、Gemini 3.1 Pro、Qwen3.5-Omni）仍然以此为起点——编码器从 ViT 演变到 DINOv2 再到 SigLIP 2，加入了 register tokens，位置编码方案变成了 2D-RoPE，但这个原语始终没变。本课从头到尾读通 patch-token 流水线，并用 Python 标准库实现它，为 Phase 12 后续课程建立"视觉 token"的具体心智模型。

**类型：** 学习
**语言：** Python（标准库，patch tokenizer + 几何计算器）
**前置：** Phase 7（Transformers）、Phase 4（计算机视觉）
**时间：** ~120 分钟

## 学习目标

- 将一张 HxWx3 的图像转换为带有正确位置编码的 patch token 序列。
- 给定（patch size、分辨率、hidden dim、深度），计算 ViT 的序列长度、参数量和 FLOPs。
- 说出将 ViT 从 2020 年研究推向 2026 年生产的三项升级：自监督预训练（DINO / MAE）、register tokens、原生分辨率打包。
- 针对下游任务，在 CLS pooling、mean pooling 和 register tokens 之间做出选择。

## 问题

Transformer 处理的是向量序列。文本天然就是序列（字节或 token）。图像是一个带三个颜色通道的二维像素网格——不是序列。如果把每个像素展平，一张 224x224 的 RGB 图像会变成 150,528 个 token，在这个长度上做 self-attention 根本不现实（序列长度的二次方复杂度）。

2020 年之前的方法是在前面接一个 CNN 特征提取器：ResNet 产生 7x7 的 2048 维特征图，把这 49 个 token 送给 transformer。这能用，但继承了 CNN 的偏置（平移等变性、局部感受野），也丧失了 transformer 对规模的渴望。

Dosovitskiy et al.（2020）提出了一个直截了当的问题：如果跳过 CNN 呢？把图像切成固定大小的 patch（比如 16x16 像素），对每个 patch 做线性投影得到一个向量，加上位置编码，然后送入标准 transformer。在当时这是异端——没有卷积的视觉。但有了足够的数据（JFT-300M，后来是 LAION），它在 ImageNet 上超过了 ResNet，并且持续提升。

到 2026 年，ViT 原语已是无可争议的基础。每个开源 VLM 的 vision tower 都是某种后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是"要不要用 patch"，而是"用什么 patch size、什么分辨率调度、什么预训练目标、什么位置编码"。

## 概念

### Patch 即 token

给定形状为 `(H, W, 3)` 的图像 `x` 和 patch size `P`，将图像切成 `(H/P) x (W/P)` 个不重叠的 patch。每个 patch 是一个 `P x P x 3` 的像素立方体。将每个立方体展平为 `3P^2` 维向量。应用一个共享的线性投影 `W_E`（形状 `(3P^2, D)`）将每个 patch 映射到模型的隐藏维度 `D`。

以 ViT-B/16 经典配置为例：
- 分辨率 224，patch size 16 → 网格 14x14 → 196 个 patch token。
- 每个 patch 是 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。
- 加上一个可学习的 `[CLS]` token → 序列长度 197。

Patch 投影在数学上等价于一个 kernel size 为 `P`、stride 为 `P`、输出通道数为 `D` 的 2D 卷积。生产代码实际上就是这么实现的——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。"线性投影"是概念上的说法；卷积核是高效的实现。

### 位置编码

Patch 没有固有顺序——transformer 把它们看作一个集合。早期 ViT 添加了可学习的 1D 位置编码（每个位置一个 768 维向量，共 197 个）。能用，但把模型绑定在训练分辨率上：推理时如果改变网格大小，就得插值位置表。

现代视觉骨干网络使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或分解式 2D 位置。2D-RoPE 根据 patch 的（行、列）索引旋转 query 和 key 向量，模型从旋转角度推断相对 2D 位置。不需要位置表。模型在推理时可以处理任意网格大小。

### CLS token、池化输出和 register tokens

图像级别的表示是什么？三种选择并存：

1. `[CLS]` token。在 patch 序列前面加一个可学习向量。经过所有 transformer block 后，CLS token 的隐藏状态就是图像表示。继承自 BERT。原始 ViT、CLIP 使用。
2. Mean pool。对 patch token 的输出隐藏状态取平均。SigLIP、DINOv2、大多数现代 VLM 使用。
3. Register tokens。Darcet et al.（2023）观察到，没有显式 sink token 训练的 ViT 会产生高范数的"伪影"patch，劫持 self-attention。添加 4-16 个可学习的 register tokens 可以吸收这种负载，改善密集预测质量（分割、深度）。DINOv2 和 SigLIP 2 都带有 registers。

选择对下游任务很重要。CLS 适合分类。对于将 patch token 送入 LLM 的 VLM，你完全跳过池化——每个 patch 都成为 LLM 的输入 token。Registers 在交接前被丢弃（它们是脚手架，不是内容）。

### 预训练：监督、对比、掩码、自蒸馏

2020 年的 ViT 用 JFT-300M 上的监督分类做预训练。很快被以下方法取代：

- CLIP（2021）：在 4 亿图文对上做对比学习。课程 12.02。
- MAE（2021，He et al.）：遮住 75% 的 patch，重建像素。自监督，只需纯图像。
- DINO（2021）/ DINOv2（2023）：student-teacher 自蒸馏，无标签，无描述。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉骨干，也是"密集特征"用例的默认选择。
- SigLIP / SigLIP 2（2023、2025）：CLIP 加 sigmoid loss 和 NaFlex 原生宽高比支持。2026 年开源 VLM 中占主导地位的 vision tower（Qwen、Idefics2、LLaVA-OneVision）。

预训练方式决定了骨干网络擅长什么：CLIP/SigLIP 擅长与文本的语义匹配，DINOv2 擅长密集视觉特征，MAE 适合作为下游微调的起点。

### 缩放定律

ViT 缩放（Zhai et al. 2022）确立了 ViT 的质量在模型大小、数据量和计算量上遵循可预测的规律。在固定计算量下：
- 更大的模型 + 更多数据 → 更好的质量。
- Patch size 是序列长度与保真度之间的杠杆。Patch 14（DINOv2/SigLIP SO400m 的典型配置）比 patch 16 每张图产生更多 token；对 OCR 和密集任务更好，但更慢。
- 分辨率是另一个大杠杆。从 224 到 384 到 512 几乎总是有帮助，但 FLOPs 呈二次方增长。

ViT-g/14（1B 参数，patch 14，分辨率 224 → 256 个 token）和 SigLIP SO400m/14（400M 参数，patch 14）是 2026 年开源 VLM 的两个主力编码器。

### ViT 的参数量计算

完整计算在 `code/main.py` 中。以 ViT-B/16 在 224 分辨率为例：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载 checkpoint 之前，先用这种方式估算每个 ViT 的参数量。骨干网络的大小决定了你在任何下游 VLM 中的 VRAM 下限。

### 2026 年生产配置

2026 年大多数开源 VLM 搭载的编码器是 SigLIP 2 SO400m/14，原生分辨率（NaFlex）。它具有：
- 400M 参数。
- Patch size 14，默认分辨率 384 → 每张图 729 个 patch token。
- 图像级任务用 mean pool；VQA 时所有 729 个 patch 都流入 LLM。
- 4 个 register tokens，在交给 LLM 前丢弃。
- 2D-RoPE 加图像级缩放，支持原生宽高比。

这个配置中的每个决策都可以追溯到一篇你能读到的论文。

## 动手用

`code/main.py` 是一个 patch tokenizer 和几何计算器。它接受（图像 H、W、patch P、hidden D、深度 L）并报告：

- 分 patch 后的网格形状和序列长度。
- 一张合成 8x8 像素玩具图像的 token 序列（走一遍展平 + 投影路径）。
- 按 patch embed、position embed、transformer blocks 和 head 分解的参数量。
- 目标分辨率下每次前向传播的 FLOPs。
- 跨 ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的对比表。

运行它。将参数量与公开数字对照。调整 patch size 和分辨率，感受 token 数量的代价。

## 交付物

本课产出 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT 配置（patch size、分辨率、hidden dim、深度），它会生成 token 数量、参数量和 VRAM 估算及其理由。每当你为 VLM 选择视觉骨干时使用这个技能——它能防止"token 爆炸导致 LLM 上下文被填满"的意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280x720 输入、patch size 14 时的 patch-token 序列长度。与仅用 CLS 的表示相比如何？

2. 一帧 1080p（1920x1080）在 patch 14 下产生多少 token？以 30 FPS 播放 5 分钟视频，总共多少视觉 token？池化、帧采样和 token 合并，哪个最省？

3. 用纯 Python 实现 patch token 的 mean pooling。验证对 DINOv2 输出的 196 个 token 做 mean-pool 的结果与模型 `forward` 返回的 pooled embedding 一致。

4. 阅读"Vision Transformers Need Registers"（arXiv:2309.16588）第 3 节。用两句话描述 registers 吸收了什么伪影，以及为什么这对下游密集预测很重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定一组不同分辨率的图像，产生一个打包序列和块对角注意力掩码。到课程 12.06 时进行验证。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Patch | "16x16 像素方块" | 输入图像中固定大小的不重叠区域；变成一个 token |
| Patch embedding | "线性投影" | 一个共享的学习矩阵（或 stride=P 的 Conv2d），将展平的 patch 像素映射为 D 维向量 |
| CLS token | "分类 token" | 前置的可学习向量，其最终隐藏状态代表整张图像；2026 年已是可选项 |
| Register token | "Sink token" | 额外的可学习 token，吸收 ViT 在预训练中产生的高范数注意力伪影 |
| Position embedding | "位置信息" | 使序列具有顺序感知的逐位置向量或旋转；2D-RoPE 是现代默认方案 |
| Grid | "Patch 网格" | 给定分辨率和 patch size 下的 (H/P) x (W/P) 二维 patch 数组 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 特性：单一模型无需重训即可服务多种宽高比和分辨率 |
| Backbone | "Vision tower" | 预训练的图像编码器，其 patch-token 输出送入 VLM 中的 LLM |
| Pooling | "图像级摘要" | 将 patch token 变成一个向量的策略：CLS、mean、attention pool 或基于 register |
| Patch 14 vs 16 | "更细 vs 更粗的网格" | Patch 14 每张图产生更多 token，OCR 保真度更好，但更慢；patch 16 是经典默认 |

## 延伸阅读

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，自监督预训练。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏，无标签。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register tokens 与伪影分析。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年默认 vision tower。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验缩放定律。
