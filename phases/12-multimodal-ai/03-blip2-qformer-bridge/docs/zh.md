# 从 CLIP 到 BLIP-2 — Q-Former 作为模态桥梁

> CLIP 对齐了图像和文本，但不能生成描述、回答问题或进行对话。BLIP-2（Salesforce，2023）用一个小型可训练桥梁解决了这个问题：32 个可学习的 query 向量通过 cross-attention 关注冻结 ViT 的特征，然后直接插入冻结 LLM 的输入流。188M 参数的桥梁将一个 11B LLM 连接到了 ViT-g/14。2026 年之前每个基于 adapter 的 VLM——MiniGPT-4、InstructBLIP、LLaVA 的各种变体——都是其后代。本课解读 Q-Former 的架构，解释其两阶段训练，并构建一个将视觉 token 送入冻结文本解码器的玩具版本。

**类型：** 构建
**语言：** Python（标准库，cross-attention + learnable-query 演示）
**前置：** Phase 12 · 02（CLIP）、Phase 7（Transformers）
**时间：** ~180 分钟

## 学习目标

- 解释为什么在冻结视觉编码器和冻结 LLM 之间放一个可训练瓶颈，在成本和稳定性上优于端到端微调。
- 实现一个 cross-attention block，其中一组固定的可学习 query 关注外部图像特征。
- 走通 BLIP-2 的两阶段预训练：表示学习（ITC + ITM + ITG）然后生成学习（冻结解码器上的 LM loss）。
- 比较 Q-Former 与 LLaVA 使用的更简单的 MLP projector，论证各自何时胜出。

## 问题

你有一个冻结的 ViT，每张图产生 256 个 dim 1408 的 patch token。你有一个冻结的 7B LLM，期望 dim 4096 的 token embedding。最直接的桥梁——一个从 1408 到 4096 的线性层——能用，但把所有 256 个 patch token 送入 LLM 的上下文意味着每张图多占 256 个 token。一个 batch 32 张图就是 8192 个 token 被视觉模态消耗。

BLIP-2 的问题是：能否将 256-token 的图像表示压缩到远更少的 token（比如 32 个），同时保留足够的信息让 LLM 能描述、回答问题和推理？能否在不动冻结骨干的情况下训练这个桥梁，将训练成本控制在桥梁的参数量上？

答案是：Q-Former。32 个可学习的"query"向量通过 cross-attention 关注 ViT 的 patch token，产生一个 32-token 的视觉摘要供 LLM 消费。总共 188M 参数。在接触 LLM 之前先用对比、匹配和生成目标训练。

## 概念

### 可学习 query

Q-Former 的核心技巧：不是让 LLM 的文本 token 去关注图像 patch，而是引入一组新的 32 个可学习 query 向量 `Q`，让*它们*去关注图像 patch。这些 query 是模型的参数——在训练中学习，对每张图使用相同的 32 个 query。

经过 cross-attention 后，每个 query 持有图像的压缩摘要——"描述主要物体"、"描述背景"、"数物体"等。Query 并不真的按语义标签特化；它们学习任何能让下游 loss 下降的编码。

### 架构

Q-Former 是一个小型 transformer（12 层，~100M 参数），有两条路径：

1. Query 路径：32 个 query 向量经过 self-attention（彼此之间），然后 cross-attention 关注冻结 ViT 的 patch token，然后 FFN。
2. Text 路径：一个类 BERT 的文本编码器与 query 路径共享 self-attention 和 FFN 权重。Text 路径禁用 cross-attention。

训练时两条路径都运行。Query 和文本通过共享的 self-attention 交互，这意味着 query 可以在需要时以文本为条件（ITM、ITG）。推理时用于 VLM 交接，只有 query 流过，产出 32 个视觉 token。

### 两阶段训练

BLIP-2 分两个阶段预训练：

阶段 1：表示学习（无 LLM）。三个损失：
- ITC（image-text contrastive）：CLIP 风格的对比，在池化的 query token 和文本 CLS token 之间。
- ITM（image-text matching）：二分类器——这个图文对是否匹配？使用 hard negative mining。
- ITG（image-grounded text generation）：文本上的 causal LM head，以 query 为条件。迫使 query 编码可生成文本的内容。

只有 Q-Former 训练。ViT 冻结。不涉及 LLM。

阶段 2：生成学习。接上冻结的 LLM（OPT-2.7B 或 Flan-T5-XL 等）。通过一个小线性层将 32 个 query 输出投影到 LLM 的 embedding 维度。将它们前置到文本 prompt。仅在拼接的 prompt + image + caption 序列上用 LM loss 训练线性投影和 Q-Former。

阶段 2 之后，Q-Former + 投影就是完整的视觉适配器。推理时：image → ViT → Q-Former → linear proj → 前置到文本 → 冻结 LLM 输出结果。

### 参数经济学

BLIP-2 配 ViT-g/14（1.1B，冻结）+ OPT-6.7B（6.7B，冻结）+ Q-Former（188M，训练）= 总共 8B，训练 188M。Q-Former 仅占整个栈参数的 ~2.4%。训练成本反映了这一点：几块 A100 上几天 vs 端到端需要几周。

质量：BLIP-2 在 zero-shot VQA 上匹配或超过 Flamingo-80B，同时小 50 倍。桥梁有效。

### InstructBLIP 和指令感知 Q-Former

InstructBLIP（2023）扩展了 Q-Former，增加了一个额外输入：指令文本本身。在 cross-attention 时，query 现在可以同时访问图像 patch 和指令。Query 可以按指令特化（"数车"、"描述氛围"），而不是学习单一固定摘要。在 held-out 任务上有基准提升。

### MiniGPT-4 和仅投影方法

MiniGPT-4 保留了 Q-Former，但只训练输出线性投影，冻结其他一切。便宜，但代价是质量——query 是 BLIP-2 的，不是你的。适合快速迭代，不是最佳架构。

### 为什么 LLaVA 走了更简单的路

LLaVA（2023，课程 12.05）用一个普通的 2 层 MLP 替换了 Q-Former，将每个 ViT patch token 投影到 LLM 空间——24x24 网格产生 576 个 token，全部送入 LLM。压缩更差，但让 LLM 直接关注原始 patch。当时这很有争议；到 2023 年底它已占主导，因为视觉指令数据（LLaVA-Instruct-150k）证明 MLP 可以被训练来保留足够的信号。权衡：LLaVA 的上下文填充更快，但它自然地扩展到多图和视频。

到 2026 年领域分化：Q-Former 在 token 预算受限时存活（长视频、多图）；MLP projector 在追求每 token 原始质量时占主导。

### Gated cross-attention：Flamingo，祖先

Flamingo（课程 12.04）早于 BLIP-2，使用了相同的 cross-attention 思想，但在冻结 LLM 的每一层都用，而不是作为单一桥梁。BLIP-2 证明你可以只压缩到输入层仍然有效。Gemini 和 Idefics 结合了两者：交错输入 token 加可选的 gated cross-attention 用于 in-context few-shot。

### 2026 年的后代

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4，以及大多数视频语言模型（出于 token 预算考虑）。
- Perceiver resampler：Flamingo 的变体（课程 12.04）；Idefics 家族、Eagle、OmniMAE。
- MLP projector：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- Attention pool：VILA、PaliGemma。

四种都有效。决定性问题是你受限于 token 预算还是每 token 质量。

## 动手用

`code/main.py` 用标准库构建了一个 Q-Former 风格的 cross-attention：

1. 模拟 256 个图像 patch token（dim 128）。
2. 实例化 32 个可学习 query（dim 128）。
3. 运行 scaled-dot-product cross-attention（Q 来自 query，K/V 来自 patch）。
4. 通过线性层投影到 LLM-dim（512）。
5. 输出 32 个 LLM-ready 视觉 token。

所有数学用纯 Python（向量上的嵌套循环）。玩具级但形状正确。注意力权重矩阵会打印出来，让你看到每个 query 从哪些 patch 拉取信息。

## 交付物

本课产出 `outputs/skill-modality-bridge-picker.md`。给定目标 VLM 配置（视觉编码器 token 数、LLM 上下文预算、部署约束、质量目标），它推荐 Q-Former vs MLP vs Perceiver resampler，附带简短理由和每种桥梁的参数量估算。

## 练习

1. 用 PyTorch 实现 cross-attention block。验证 32 个 query 和 256 个 key/value 时，注意力权重矩阵是 32 x 256，softmax 后每行和为 1。

2. 在 BLIP-2 阶段 1 中，Q-Former 同时运行三个损失：ITC、ITM、ITG。用伪代码写出每个的 forward 签名。哪个需要文本编码器路径激活？

3. 比较参数量：Q-Former（12 层，768 hidden）vs 2 层 MLP projector（1408 → 4096，两层）。在什么 LLM 规模下，188M Q-Former 的成本在训练效率上能回本？

4. 阅读 BLIP-2 论文（arXiv:2301.12597）第 3.2 节关于 Q-Former 如何初始化。解释为什么从 BERT-base（而非随机）初始化能加速收敛。

5. 对于一个 10 分钟视频，1 FPS 采样到 60 帧，计算每帧 token 成本：（Q-Former → 32 tokens/帧）vs（MLP projector → 576 tokens/帧）。哪个能放进 128k-token 的 LLM 上下文窗口？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Q-Former | "Querying transformer" | 带有 32 个可学习 query 向量的小型 transformer，通过 cross-attention 关注冻结 ViT 特征 |
| Learnable queries | "视觉的 soft prompt" | 一组固定参数，作为 cross-attention 的 query 侧；按模型学习，所有输入共享 |
| Cross-attention | "Q 来自这里，K/V 来自那里" | Query、key 和 value 来自不同来源的注意力；query 从 ViT patch 拉取信息的方式 |
| ITC | "Image-text contrastive" | CLIP 风格损失，应用于 Q-Former 池化 query vs 文本 CLS |
| ITM | "Image-text matching" | 在 hard-negative-mined 对上的二分类器；迫使 query 区分细粒度不匹配 |
| ITG | "Image-grounded text generation" | Causal LM loss，文本以 query 为条件生成；迫使 query 编码可解码为文本的内容 |
| 两阶段预训练 | "先表示后生成" | 阶段 1 单独训练 Q-Former（ITC/ITM/ITG）；阶段 2 接上冻结 LLM，只训练投影 + Q-Former |
| 冻结骨干 | "不微调" | 视觉编码器和 LLM 权重固定；只有桥梁训练 |
| Projection head | "到 LLM dim 的线性层" | 将 Q-Former 输出映射到 LLM embedding 维度的最终线性层 |
| Perceiver resampler | "Flamingo 的版本" | 类似的可学习 query cross-attention，Flamingo 在每层使用而非作为单一桥梁 |

## 延伸阅读

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — 核心论文。
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — 前身，带有 ITC/ITM/ITG 三件套。
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — "align before fuse"——阶段 1 训练的概念祖先。
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — 指令感知 Q-Former。
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — 仅投影方法。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 可学习 query cross-attention 的通用架构。
