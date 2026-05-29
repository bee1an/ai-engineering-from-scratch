# CLIP 与对比视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了一个足以驱动未来五年的单一想法：仅用嘈杂的网络图文对和对比损失，将图像编码器和文本编码器对齐到同一向量空间。零监督标签。4 亿对数据。产出的嵌入空间可以做 zero-shot 分类、图文检索，并作为 vision tower 接入 2026 年的每个 VLM。SigLIP 2（2025）用 sigmoid 替换了 softmax，以更低成本超越了 CLIP。本课从 InfoNCE 到 sigmoid pairwise loss 逐步推导数学，并用 Python 标准库实现训练步骤。

**类型：** 构建
**语言：** Python（标准库，InfoNCE + sigmoid loss 实现）
**前置：** Phase 12 · 01（ViT patches）、Phase 7（Transformers）
**时间：** ~180 分钟

## 学习目标

- 从互信息推导 InfoNCE loss，并实现一个数值稳定的向量化版本。
- 解释为什么 sigmoid pairwise loss（SigLIP）能扩展到 batch 32768+ 而不需要 softmax 所要求的 all-gather 开销。
- 通过构造文本模板（`a photo of a {class}`）并对余弦相似度取 argmax，运行 zero-shot ImageNet 分类。
- 说出 CLIP / SigLIP 预训练给你的四个杠杆：batch size、temperature、prompt template、数据质量。

## 问题

CLIP 之前的视觉是监督式的。收集标注数据集（ImageNet：120 万张图，1000 个类别），训练 CNN，上线。标注昂贵，标注偏向标注者能达成共识的内容，标注不能迁移到新任务而无需微调。

网络上的图文对有超过十亿个松散标注的配对可以免费获取。一张金毛犬的图片配上 alt text "my dog Max in the park" 就携带了监督信号——文本描述了图像。问题是：你能把这变成有用的训练吗？

CLIP 的答案：把图文对当作匹配任务。给定一个 batch 中的 N 张图和 N 条描述，学习将每张图与自己的描述匹配，对抗 N-1 个干扰项。监督信号是"这两个东西属于一起；这 N-1 个不属于"。没有类别标签。没有人工标注。只有对比损失。

产出的嵌入空间能做的事超出了 CLIP 的训练目标。ImageNet zero-shot 之所以有效，是因为 "a photo of a cat" 的嵌入靠近那些从未被显式标注为猫的猫图片。这就是催生 2026 年每个 VLM 的那个赌注。

## 概念

### 双塔编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图输出一个 D 维向量。
- 文本编码器 `g`：小型 transformer，每条描述输出一个 D 维向量。

两个塔都将输出归一化为单位长度。相似度为 `cos(f(x), g(y)) = f(x)^T g(y)`，因为两者都是单位范数。

对于一个 batch 中的 N 个（图像，描述）对，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是可学习的温度参数（CLIP 初始化为 0.07；在 log 空间中学习）。

### InfoNCE loss

CLIP 使用对行和列的对称交叉熵：

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 迫使每张图与自己的描述匹配得比 batch 中所有其他描述都好。"负样本"就是 batch 中的所有其他项。更大的 batch = 更多负样本 = 更强的信号。CLIP 在 batch 32k 下训练；规模很重要。

### Temperature

`tau` 控制 softmax 的锐度。低 tau → 尖锐分布，hard negative mining 效果。高 tau → 平滑，所有样本都有贡献。CLIP 学习 log(1/tau)，裁剪以防止坍缩。SigLIP 2 固定初始 tau，改用可学习的 bias。

### 为什么 sigmoid 扩展性更好（SigLIP）

Softmax 需要整个相似度矩阵同步。在分布式训练中，你必须将每个嵌入 all-gather 到每个副本，然后做 softmax。通信开销随 world size 呈二次方增长。

SigLIP 用逐元素 sigmoid 替换 softmax：对每个配对 `(i, j)`，损失是"这两个是否匹配"的二分类。正类标签是对角线，其他都是负类。损失为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 当 `i == j`，否则为 0。每对的损失是独立的。不需要 all-gather。每个 GPU 计算自己的本地块并求和。SigLIP 2 可以廉价地扩展到 batch 32k-512k，而 CLIP 需要成比例增加通信。

### Zero-shot 分类

给定 N 个类别名，为每个类别构建文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入你的图像。余弦相似度取 argmax = 预测类别。不需要在目标类别上训练。

Prompt template 很重要。CLIP 原始论文对每个类别使用了 80 个模板（普通、艺术、照片、绘画等）并对嵌入取平均。ImageNet 上 +3 个点。现代用法通常选一两个模板。

### Linear probe 和微调

Zero-shot 是基线。Linear probe（在冻结的 CLIP 特征上训练一个线性层用于目标类别）在域内任务上超过 zero-shot。全量微调在域内超过 linear probe，但可能损害 zero-shot 迁移。三种方案，三种权衡。

### SigLIP 2：NaFlex 和密集特征

SigLIP 2（2025）新增：
- NaFlex：单一模型处理可变宽高比和分辨率。
- 更好的密集特征，用于分割和深度估计，目标是作为 VLM 中的冻结骨干。
- 多语言：在 100+ 种语言上训练，而 CLIP 仅支持英语。
- 1B 参数规模，而 CLIP 最大到 400M。

在 2026 年的开源 VLM 中，SigLIP 2 SO400m/14 是默认 vision tower。CLIP 仍然是纯图文检索的默认选择，适用于 LAION-2B 训练分布与你的查询模式匹配的场景。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 相同的想法，18 亿对规模，90% 噪声。证明了噪声数据可以扩展。OpenCLIP（LAION）：在 LAION-400M / 2B 上的 CLIP 开源复现，多种规模，首选开源 checkpoint。EVA-CLIP：从 masked image modeling 初始化；VLM 的强骨干。BASIC：Google 的 CLIP+ALIGN 混合体。都是同一家族，不同的数据和调优。

### Zero-shot 天花板

CLIP 类模型在 ImageNet zero-shot 上大约封顶在 76%（CLIP-G、OpenCLIP-G）。超越需要更大的数据（SigLIP 2 达到 80%+）或架构变化（监督头、更多参数）。基准正在饱和；真正的价值在于下游 VLM 消费的嵌入空间。

## 动手用

`code/main.py` 实现了：

1. 一个玩具双塔编码器（基于哈希的图像特征、文本字符特征），让你无需 numpy 就能看到 InfoNCE 的形状。
2. 纯 Python 的 InfoNCE loss（通过 log-sum-exp 保证数值稳定性）。
3. Sigmoid pairwise loss 作为对比。
4. Zero-shot 分类流程：计算与一组文本 prompt 的余弦相似度，argmax 得到预测。

运行它，观察 loss 曲线。绝对数值是玩具级的；曲线形状与真实 CLIP 训练器的输出一致。

## 交付物

本课产出 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和一个目标类别列表，它用 CLIP 模板构建文本 prompt，用指定 checkpoint（如 `openai/clip-vit-large-patch14`）嵌入两侧，返回 top-1 / top-5 预测及相似度分数。该技能拒绝对不在 prompt 列表中的类别做出声明。

## 练习

1. 手动为 4 对样本实现 InfoNCE。构建 4x4 相似度矩阵，运行 softmax，取出对角线，计算交叉熵。用你的 Python 实现验证手算结果。

2. SigLIP 除了 temperature 还使用了 bias 参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当 batch 中存在严重类别不平衡（每行负样本远多于正样本）时，`b` 起什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 构建一个猫 vs 狗的 zero-shot 分类器。尝试两种 prompt template：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图上测量准确率。模板集成是否优于单一模板？

4. 计算 512-GPU 运行、batch 32k 时 softmax InfoNCE vs sigmoid pairwise 的通信开销。哪个是 O(N)，哪个是 O(N^2)？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 缩放定律论文（arXiv:2212.07143，Cherti et al.）。从图中复现他们关于数据缩放的结论：在固定模型大小下，ImageNet zero-shot 准确率与训练数据量之间的 log-linear 关系是什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| InfoNCE | "对比损失" | 对 batch 相似度矩阵的交叉熵；每个项的正样本是其配对项，负样本是其他所有 |
| Sigmoid loss | "SigLIP loss" | 逐对二元交叉熵；无 softmax，无 all-gather，分布式训练中扩展成本低 |
| Temperature | "tau" | 在 softmax/sigmoid 前缩放 logits 的标量；控制分布的锐度 |
| Zero-shot | "无微调分类" | 用文本 prompt 构造类别嵌入，通过余弦相似度分类；不在目标类别上训练 |
| Prompt template | "a photo of a ..." | 类别名周围的文本脚手架；影响 zero-shot 准确率 1-5 个点 |
| Dual encoder | "双塔" | 一个图像编码器 + 一个文本编码器，输出在共享 D 维空间中 |
| Hard negative | "困难干扰项" | 与正样本足够相似的负样本，模型必须努力才能区分 |
| Linear probe | "冻结 + 一层" | 仅在冻结特征上训练一个线性分类器；衡量特征质量 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 能力：以任意宽高比和分辨率摄入图像而无需 resize |
| Temperature scaling | "log 参数化 tau" | CLIP 参数化 `log(1/tau)` 使梯度表现良好；裁剪以防止 tau 坍缩到接近零 |

## 延伸阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 用噪声网络数据扩展。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP 缩放定律。
