# Flamingo 与 Gated Cross-Attention：Few-Shot VLM

> DeepMind 的 Flamingo（2022）做了两件别人没做过的事。它展示了单一模型可以处理任意交错的图像、视频和文本序列。它还展示了 VLM 可以 in-context 学习——给一个包含三个示例（图像，描述）对的 few-shot prompt，模型就能为新图像生成描述，无需任何梯度步骤。机制是：gated cross-attention 层，插入在冻结 LLM 的现有层之间，带有一个学习的 tanh gate，初始为零，从而在初始化时保留 LLM 的文本能力。本课走通 Flamingo 的 Perceiver resampler 和 gated cross-attention 架构——Gemini 交错输入和 Idefics2 视觉 token 的祖先。

**类型：** 学习
**语言：** Python（标准库，gated cross-attention + Perceiver resampler 演示）
**前置：** Phase 12 · 03（BLIP-2 Q-Former）
**时间：** ~120 分钟

## 学习目标

- 解释 gated cross-attention 如何通过 tanh(gate) = 0 在初始化时保留冻结 LLM 的文本能力。
- 走通 Perceiver resampler：N 个图像 patch → K 个固定"latent" query，通过 cross-attention。
- 描述 Flamingo 如何用因果掩码处理交错的图文序列，掩码尊重图像位置。
- 复现一个 few-shot 多模态 prompt 结构（3 个图像-描述示例然后一个查询图像）。

## 问题

BLIP-2 将 32 个视觉 token 送入冻结 LLM 的输入层。对单张图的 prompt 有效。但如果你想送入*多张*图像与文本交错，比如"这是图 A，描述它；这是图 B，描述它；现在这是图 C，描述它"呢？LLM 的 self-attention 需要在单一流中处理图像 token 和文本 token，而哪些位置可以关注哪些图像的问题变得棘手。

Flamingo 的答案：完全不改变 LLM 的输入流。在现有 LLM block 之间插入额外的 cross-attention 层。文本 token 仍然像往常一样流过 LLM 的 causal self-attention。每隔几个 LLM block，文本 token 还通过一个新的 gated 层 cross-attend 到图像特征。Gate（初始化为零）意味着在第零步新层是 no-op——模型表现得完全像预训练的 LLM。随着训练推进，gate 打开，视觉信息开始流入。

Flamingo 回答的第二个问题：如何处理每个 prompt 中可变数量的图像（0、1 或多张）？Perceiver resampler——一个小型 cross-attention 模块，接受任意数量的 patch 并产生固定数量的视觉 latent token。LLM 的 cross-attention 层看到的形状相同，无论 prompt 中有多少张图像。

## 概念

### 冻结的 LLM

Flamingo 从冻结的 Chinchilla 70B LLM 开始。所有 70B 权重不动。现有的文本 self-attention 和 FFN 正常运行。

### Perceiver resampler

对于 prompt 中的每张图像，ViT 产生 N 个 patch token。Perceiver resampler 有 K 个固定的可学习 latent（Flamingo 使用 K=64）。每个 resampler block 是两个子步骤：

1. Cross-attention：K 个 latent 关注 N 个 patch token（Q 来自 latent，K/V 来自 patch）。
2. Latent 内部的 self-attention + FFN。

经过 6 个 resampler block 后，输出是 K=64 个 dim 1024 的视觉 token，无论 ViT 产生了多少 patch。一张 224x224 图像（196 个 patch）和一张 480x480 图像（900 个 patch）都输出为 64 个 resampler token。

对于视频，resampler 在时间维度上应用：每帧的 patch 产生 64 个 latent，时间位置编码让模型区分 t=0 和 t=N。完整视频变成 T * 64 个视觉 token。

### Gated cross-attention

在冻结 LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的 gated cross-attention block：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个可学习标量，初始化为零。
- `tanh(0) = 0`，所以初始时 gated 分支贡献为零。
- 当 `alpha` 远离零时，cross-attention 的贡献平滑增长。
- 残差连接意味着即使 gate 完全打开也不会覆盖 LLM 的文本表示；它只是在上面添加视觉信息。

这是 Flamingo 中最重要的设计选择：视觉条件化是加性的、门控的、初始化时为零。第 0 步的 Flamingo 就是一个完美的 Chinchilla 70B 文本模型。

### 交错输入的掩码 cross-attention

在类似 "<image A> caption A <image B> caption B <image C> ?" 的 prompt 中，每个文本 token 应该只看到序列中在它之前的图像。Cross-attention 掩码强制：位置 `t` 的文本 token 只关注图像索引 `i < i_t` 的 resampler token，其中 `i_t` 是位置 `t` 之前最近的图像。"只看最近的前一张图像"或"看所有前面的图像"都是有效选择；Flamingo 选择了前者。

### In-context few-shot 学习

Flamingo 的 prompt 看起来像：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到补全模式并输出 "bird"（或 image3 显示的任何东西）。没有梯度步骤。冻结 LLM 的 in-context 学习能力通过 gated cross-attention 传递——这是论文的核心结论，也是它重要的原因。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb（M3W）：4300 万网页，包含交错的图像和文本，重建阅读顺序。
2. Image-Text Pairs（ALIGN + LTIP）：44 亿对。
3. Video-Text Pairs（VTP）：2700 万短视频片段。

OBELICS（2023）是交错网络语料库的开源复现，Idefics、Idefics2 和大多数开源"Flamingo 风格"模型在其上训练。

### OpenFlamingo 和 Otter

OpenFlamingo（2023）是开源复现。架构相同（Perceiver resampler + 冻结 LLaMA 或 MPT 上的 gated cross-attention）。3B、4B、9B 的 checkpoint。由于更小的基础 LLM 和更少的数据，质量落后于 Flamingo。

Otter（2023）在 OpenFlamingo 基础上用 MIMIC-IT（多模态指令数据集）做指令微调，证明 gated cross-attention 也适用于指令跟随。

### 后代

- Idefics / Idefics2 / Idefics3：Hugging Face 的 gated cross-attention 血统，逐步简化（Idefics2 放弃了 resampler，改用直接 patch token 加自适应池化）。
- Flamingo 到 Chameleon 的过渡：到 2024 年许多团队转向 early fusion（课程 12.11）；Flamingo 风格的 gated cross-attention 在需要冻结骨干的场景中仍在生产使用。
- Gemini 的交错输入：概念上继承了 Flamingo 的交错格式灵活性，但具体机制是专有的。

### 与 BLIP-2 的比较

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥梁 | Q-Former 在输入处一次 | 每 M 层的 gated cross-attention |
| 视觉 token | 每张图 32 个 | 每张图每个 cross-attn 层 64 个 |
| 冻结 LLM | 是 | 是 |
| Few-shot in-context | 弱 | 强——论文的核心 |
| 交错输入 | 无原生支持 | 是，设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错页面 |
| 参数量 | 188M 训练 | ~10B 训练（cross-attn 层） |
| 计算量 | 8 块 A100 几天 | 数千块 TPUv4 几周 |

单图 VQA 预算有限选 BLIP-2。交错、few-shot 或多图推理选 Flamingo/Idefics2。

## 动手用

`code/main.py` 演示：

1. 在 36 个假 patch token 上运行 Perceiver resampler，8 个可学习 latent（纯 Python cross-attention）。
2. 一个 gated cross-attention 步骤：`alpha = 0` → 输出等于输入（LLM 不变），然后 `alpha = 2.0` → 视觉贡献混入。
3. 一个交错掩码构建器，为"(image 1) (text 1) (image 2) (text 2)"序列产生 2D 注意力掩码。

## 交付物

本课产出 `outputs/skill-gated-bridge-diagnostic.md`。给定一个开源 VLM 的配置（resampler 有/无、cross-attn 频率、gate 方案），它识别 Flamingo 血统元素并解释冻结策略。用于调试为什么微调降低了文本性能（答案：gate 打开得太快太大）。

## 练习

1. 计算 Flamingo-9B 的视觉参数量：9B LLM + 1.4B gated cross-attention 层 + 64M resampler。训练参数占总参数的多少比例？

2. 用 PyTorch 实现 gated residual `y = tanh(alpha) * cross + x`。实验证明 `alpha=0` 时，`y==x` 在初始化时精确成立。

3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390），了解当 batch 中每个 prompt 有不同图像数量时如何处理。描述 padding 策略。

4. 为什么 Flamingo 的 cross-attention 掩码让文本 token 只关注*最近的前一张*图像而非所有前面的图像？阅读 Flamingo 论文第 2.4 节并解释权衡。

5. In-context few-shot：为一个新的 Flamingo 变体构造一个包含 4 个"图像 → 主要物体颜色"示例的 prompt。描述当示例数从 0 变到 8 时的预期准确率模式。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Perceiver resampler | "固定 latent cross-attention" | 从可变数量的输入 patch 产生 K 个固定 token 的模块 |
| Gated cross-attention | "Tanh 门控桥梁" | 残差层 `y = tanh(alpha)*cross + x`，可学习 alpha，初始为 0 |
| Interleaved input | "混合序列" | 图像和文本按阅读顺序自由混合的 prompt 格式 |
| Frozen LLM | "无 LLM 梯度" | 文本 LLM 的权重不更新；只有 resampler + cross-attn 层训练 |
| Few-shot | "In-context 示例" | 在 prompt 中给几个（图像，答案）对；模型无需微调即可泛化 |
| OBELICS | "交错网络语料库" | 1.41 亿网页的开放数据集，包含按阅读顺序排列的图像和文本 |
| Chinchilla | "70B 冻结基座" | Flamingo 的冻结文本 LLM，来自 DeepMind 的 Chinchilla 论文 |
| Gate schedule | "alpha 如何变化" | 训练期间 cross-attention gate 打开的速率 |
| Cross-attn frequency | "每 M 层" | gated cross-attention block 插入的频率；Flamingo 使用 M=4 |
| OpenFlamingo | "开源复现" | MosaicML/LAION 的 3-9B 开源 checkpoint；架构与 Flamingo 相同 |

## 延伸阅读

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开源复现。
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错网络语料库。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 指令微调的 Flamingo 后代。
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 方法的现代简化。
