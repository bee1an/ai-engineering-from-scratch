# 任意分辨率视觉：Patch-n'-Pack 与 NaFlex

> 真实图像不是 224x224 的正方形。收据是 9:16，图表是 16:9，医学扫描可能是 4096x4096，手机截图是 9:19.5。2024 年之前 VLM 的答案——把所有东西 resize 到固定正方形——丢掉了让 OCR、文档理解和高分辨率场景解析有效的信号。NaViT（Google，2023）展示了你可以用块对角掩码将可变分辨率的 patch 打包到单个 transformer batch 中。Qwen2-VL 的 M-RoPE（2024）完全抛弃了绝对位置表。LLaVA-NeXT 的 AnyRes 将高分辨率图像切成基础图 + 子图。SigLIP 2 的 NaFlex 变体（2025）现在是想用单一 checkpoint 服务所有宽高比的开源 VLM 的默认编码器。本课端到端实现 patch-n'-pack。

**类型：** 构建
**语言：** Python（标准库，patch packer + 块对角掩码）
**前置：** Phase 12 · 01（ViT patches）、Phase 12 · 05（LLaVA）
**时间：** ~120 分钟

## 学习目标

- 将一批可变分辨率图像的 patch 打包到一个序列中，并构建块对角注意力掩码。
- 针对给定任务，在 AnyRes tiling（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间做出选择。
- 在不 resize 的情况下计算 OCR、图表和摄影的 token 预算。
- 说出正方形 resize 的三种失败模式：文字被压扁、内容被裁切、padding 浪费 token。

## 问题

Transformer 期望一个序列。Batch 是一组等长序列的堆叠。如果你的图像都是 224x224，每次都得到 196 个 patch token，不需要 padding，搞定。在 224 上训练，在 224 上推理，永远不用考虑分辨率。

但现实不配合。文档是竖版（8.5x11 英寸，约 2:3）。图表截图是横版（16:9）。收据又高又窄（1:3）。医学影像是 2048x2048 或更大。手机截图是 1170x2532（0.46:1）。

2024 年之前的三种选择及其失败原因：

1. Resize 到固定正方形（224x224 或 336x336）。压缩会扭曲文字和人脸。缩小会破坏图表标签和 OCR 内容。直到 LLaVA-1.5 都是标准做法。
2. 裁切到固定宽高比。你丢掉了大部分图像，而选择裁切位置本身就是一个视觉问题。
3. Pad 到最长边。修复了扭曲但对竖版图像浪费 50%+ 的 token 在 padding 上。所有 pad token 的二次方注意力开销。

2024-2025 年的答案：让 transformer 以图像的原生分辨率吃 patch，然后想办法把异构 batch 打包到一个序列中而不浪费计算。

## 概念

### NaViT 与 patch-n'-pack

NaViT（Dehghani et al.，2023）是证明这在规模上有效的论文。思路是机械的：

1. 对 batch 中的每张图像，在选定的 patch size（比如 14）下计算其原生 patch 网格。
2. 将每张图像的 patch 展平为其自己的变长序列。
3. 将所有图像的 patch 拼接成 batch 的一个长序列。
4. 构建块对角注意力掩码，使图像 A 的 patch 只在图像 A 内部互相关注。
5. 携带逐 patch 的位置信息（2D RoPE 或分数位置编码）。

三张图像的 batch：336x336（576 token）、224x224（256 token）和 448x336（768 token）变成一个 1600-token 的序列，配一个 1600x1600 的块对角掩码。没有 padding。没有浪费的计算。Transformer 处理任意宽高比。

NaViT 还引入了训练时的分数 patch dropping——在 batch 中随机丢弃 50% 的 patch——既起正则化作用又加速训练。SigLIP 2 继承了这一点。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是务实的替代方案。给定一张高分辨率图像和一个固定编码器（CLIP 或 SigLIP 在 336），对图像做 tiling：

1. 从预定义集合中选一个网格布局——(1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2) 等——最适合图像的宽高比。
2. 将完整图像切成网格；每个 tile 变成一个 336x336 的 crop。
3. 同时产生一个缩略图：整张图 resize 到 336x336 作为全局上下文 token。
4. 将每个 tile 通过冻结的 336 编码器编码。拼接 tile token + 缩略图 token。

对于 672x672 图像在 2x2 网格加缩略图：4 * 576 + 576 = 2880 个视觉 token。昂贵但有效——LLM 同时看到局部细节和全局上下文。

AnyRes 是编码器冻结且只支持一种分辨率时的首选路线。它对大图像的 token 数会爆炸（1344x1344 图像在 4x4 网格是 9216 + 576 ≈ 9800 token，填满大部分 8k LLM 上下文）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了 Multimodal Rotary Position Embedding。不同于 NaViT 的分数位置或 AnyRes 的 tile-and-thumbnail，每个 patch 携带一个 3D 位置（temporal、height、width）。Query/key 旋转处理任意 H、W 和时间长度。

M-RoPE 无需重训即可原生支持动态分辨率。推理时你送入任意 HxW 图像，patch embedder 产生 H/14 x W/14 个 token，每个 token 获得其 (t=0, r=row, c=col) 位置，RoPE 用正确的频率旋转注意力，完成。Qwen2.5-VL 和 Qwen3-VL 延续了这一点。InternVL3 的 V2PE 是相同思想，每种模态有可变编码。

与 AnyRes 不同，M-RoPE 在原生分辨率下是 O(H x W / P^2) 个 token——没有乘法式的 tile 开销。与 NaViT 不同，它仍然期望每次前向一张图像。跨分辨率的 batching 仍需在上面加 patch-n'-pack。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 checkpoint 的 native-flex 模式。单一模型在推理时服务多种序列长度（256、729、1024 token）。内部在训练时使用 NaViT 风格的 patch-n'-pack 和逐 patch 的绝对分数位置。卖点：一个 checkpoint，根据任务在推理时选择 token 预算。

语义任务（分类、检索）用 256 token。OCR 或图表理解用 1024 token。无需重训。

### 打包掩码

块对角掩码是大多数实现容易出错的地方。对于长度为 `N_total` 的打包序列，覆盖图像 `i=0..B-1`，各自长度为 `n_i`，掩码 `M` 的形状为 `(N_total, N_total)`，当两个索引落在同一图像的块内时为 1，否则为 0。你可以从累积长度列表构建它：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

在 PyTorch 中用 `torch.block_diag` 或显式 gather 一行搞定。FlashAttention 的变长路径（`cu_seqlens`）完全跳过掩码，直接用累积长度张量在序列内做注意力——对典型 batch 比密集掩码快约 10 倍。

### Token 预算

按任务选择策略：

- OCR / 文档：1024-4096 token。SigLIP 2 NaFlex 在 1024，或 AnyRes 3x3 + 缩略图。
- 图表和 UI：729-1024 token，384-448 原生分辨率。Qwen2.5-VL 动态分辨率加 max pixels 上限。
- 自然照片：256-576 token 就够了。下游 LLM 看到的足够。在内容密度高的地方花 token。
- 视频：空间池化后每帧 64-128 token，2-8 FPS。课程 12.17 覆盖这个。

2026 年生产规则：选一个按任务的 max-pixels 上限，以原生宽高比编码到该上限，打包 batch，跳过 padding。Qwen2.5-VL 正是为这个旋钮暴露了 `min_pixels` 和 `max_pixels`。

## 动手用

`code/main.py` 为一批异构图像（整数像素坐标）实现 patch-n'-pack。它：

- 接受一个 (H, W) 图像尺寸列表。
- 在 patch size 14 下计算每张图像的 patch 序列长度。
- 将它们打包到一个总长度为 `sum(n_i)` 的序列中。
- 构建块对角注意力掩码（密集形式，为了清晰）。
- 比较打包成本 vs 正方形 resize 和 AnyRes tiling。
- 打印一个混合 batch（收据、图表、截图、照片）的 token 预算表。

运行它。掉出来的数字就是 2026 年每个开源 VLM 使用 patch-n'-pack 的原因。

## 交付物

本课产出 `outputs/skill-resolution-budget-planner.md`。给定一个混合宽高比的工作负载（OCR、图表、照片、视频帧）和总 token 预算，它选择正确的策略（NaFlex、AnyRes、M-RoPE 或固定正方形）并输出每请求配置。当你为产品确定 VLM 规格时使用这个技能——它能防止悄悄的 10 倍 token 膨胀杀死延迟预算。

## 练习

1. 一张收据是 600x1500（1:2.5）。在 patch size 14 下，原生分辨率有多少 token？正方形 resize 到 336 后呢？哪个在实践中损失更多 OCR 准确率？

2. 为四张图像（长度 256、576、729、1024）构建块对角掩码。验证注意力矩阵是 2585x2585，且恰好有 `256^2 + 576^2 + 729^2 + 1024^2` 个非零项。

3. 对于 1792x896 图像在 patch 14 下，比较：(a) 正方形 resize 到 336 然后编码，(b) AnyRes 2x1 + 缩略图，(c) M-RoPE 原生分辨率。哪个用最少 token？哪个保留最多细节？

4. 实现分数 patch dropping：给定一个打包序列，均匀随机丢弃 50% 的 token，并相应更新块对角掩码。测量掩码稀疏度的变化。

5. 阅读 Qwen2-VL 论文（arXiv:2409.12191）第 3.2 节。用两句话描述 `min_pixels` 和 `max_pixels` 控制什么，以及为什么两个边界都重要。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Patch-n'-pack | "NaViT 风格打包" | 将不同图像的变长 patch 序列拼接到一个 batch 维度中 |
| Block-diagonal mask | "打包掩码" | 限制每张图像的 patch 只关注自身、不关注打包中邻居的注意力掩码 |
| AnyRes | "LLaVA-NeXT tiling" | 将高分辨率图像切成固定大小 tile 的网格加全局缩略图；用固定编码器编码每个 tile |
| NaFlex | "SigLIP 2 native-flex" | 单一 SigLIP 2 checkpoint 在推理时无需重训即可服务 256/729/1024-token 预算 |
| M-RoPE | "Multimodal RoPE" | 3D 旋转位置编码（time、row、column），无需位置表即可处理任意 H、W、T |
| cu_seqlens | "FlashAttention 打包" | FlashAttention varlen 路径使用的累积长度张量，替代密集块对角掩码 |
| min_pixels / max_pixels | "分辨率边界" | Qwen2.5-VL 的每请求旋钮，限制极小或极大输入的 token 数 |
| Visual token budget | "每张图多少 token" | 每张图像发出的 patch token 粗略计数；决定 LLM 的 prompt 预算和注意力开销 |

## 延伸阅读

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
