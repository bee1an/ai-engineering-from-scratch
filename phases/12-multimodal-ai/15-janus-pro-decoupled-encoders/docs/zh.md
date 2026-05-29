# Janus-Pro：解耦编码器的统一多模态模型

> 统一多模态模型有一个不可避免的张力。理解需要语义特征——SigLIP 或 DINOv2 输出的向量富含概念级信息。生成需要重建友好的编码——VQ token 能组合回清晰的像素。这两个目标在单一编码器中不兼容。Janus（DeepSeek，2024 年 10 月）和 Janus-Pro（DeepSeek，2025 年 1 月）认为解决方案是停止尝试：解耦两个编码器。在任务之间共享 transformer body，但理解走 SigLIP，生成走 VQ tokenizer。在 7B 规模下，Janus-Pro 在 GenEval 上超过 DALL-E 3，同时在 MMMU 上匹配 LLaVA。本课解读为什么两个编码器在一个编码器失败的地方能成功。

**Type:** Build
**Languages:** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time:** ~120 minutes

## 学习目标

- 解释为什么单一共享编码器会在理解或生成质量上妥协。
- 描述 Janus-Pro 的路由：理解侧输入用 SigLIP 特征，生成侧输入和输出都用 VQ token。
- 追踪让 Janus-Pro 成功而 Janus 未能成功的数据混合扩展。
- 对比解耦（Janus-Pro）、耦合连续（Transfusion）和耦合离散（Show-o）架构。

## 问题

统一模型在理解和生成之间共享 transformer body。之前的尝试（Chameleon、Show-o、Transfusion）都对两个方向使用一个视觉 tokenizer。这个 tokenizer 是一个妥协：

- 为重建优化（生成）：VQ-VAE 捕获细粒度像素细节，但产生的 token 语义一致性弱。
- 为语义优化（理解）：SigLIP embedding 将"猫"图像聚集在"猫"token 附近，但不允许好的重建。

Show-o 和 Transfusion 为此在某个方向上付出了可见的质量税。Janus-Pro 问：当任务有不同需求时，为什么要求一个 tokenizer？

## 核心概念

### 解耦视觉编码

Janus-Pro 的架构分离了两个编码器：

- 理解路径。输入图像 → SigLIP-SO400m → 2 层 MLP → transformer body。
- 生成路径。输入图像（如果以现有图像为条件）→ VQ tokenizer → token ID → transformer body。
- 输出生成。Transformer 预测的图像 token → VQ decoder → 像素。

Transformer body 是共享的。body 上下游的一切都是任务专属的。

输入通过 prompt 格式消歧：`<understand>` 标签走 SigLIP；`<generate>` 走 VQ。或者路由从任务隐式推断。

### 为什么这能工作

理解损失得到 SigLIP 特征，CLIP 风格预训练已经为语义相似性调优了这些特征。模型的感知 benchmark 比 Show-o / Transfusion 提升，因为输入特征对任务更好。

生成损失得到 VQ token，tokenizer 已经为重建调优了这些 token。图像质量比 Show-o 提升，因为 VQ code 能干净地组合回像素。

共享的 transformer body 看到两种输入分布（SigLIP 和 VQ）并学会处理两者。论点是：足够的数据 + 足够的参数，body 能吸收这种切换。

### 数据扩展——Janus vs Janus-Pro

Janus（原版，arXiv 2410.13848）引入了解耦但规模小（1.3B 参数，有限数据）。Janus-Pro（arXiv 2501.17811）扩展了：

- 7B 参数（vs 1.3B）。
- Stage 1（对齐）90M 图文对，从 72M 增加。
- Stage 2（统一）72M，从 26M 增加。
- Stage 3 增加了 200k 图像生成指令样本。

结果：Janus-Pro-7B 在 MMMU 上匹配 LLaVA（60.3 vs ~58），在 GenEval 上超过 DALL-E 3（0.80 vs 0.67）。一个开源模型，在统一频谱的两侧都有竞争力。

### JanusFlow——rectified flow 变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径替换为 rectified-flow 生成路径（连续）。拆分变为 SigLIP 用于理解 + rectified-flow 用于生成。质量上限进一步提升。架构仍然是解耦编码器-共享 body。

### 共享 body 的工作

Transformer body 处理统一序列但有两种输入分布。它的工作是：

- 理解：消费 SigLIP 特征 + 文本 token → 自回归输出文本。
- 生成：消费文本 token +（可选的图像 VQ token）→ 自回归输出图像 VQ token。

Body 每个 block 没有模态专属权重。它就是你在 Qwen 或 Llama 内部会找到的文本风格 transformer，加上两个输入适配器。

有趣的是，这意味着 Janus-Pro 的 body 可以从预训练 LLM 初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这个选择很重要：LLM 贡献了从零开始训练的统一模型难以达到的推理能力。

### 与 InternVL-U 的对比

InternVL-U（Lesson 12.10）是 2026 年的后续。它结合了：

- 原生多模态预训练（InternVL3 backbone）。
- 解耦编码器路由（SigLIP 输入，VQ + diffusion head 输出）。
- 统一理解 + 生成 + 编辑。

InternVL-U 将 Janus-Pro 的架构选择纳入了一个更大的框架。解耦编码器的思路现在是大规模统一模型的默认选择。

### 局限性

解耦编码器增加了架构复杂度。两个 tokenizer 要训练，两条输入路径要维护，两组故障模式。对于不需要生成的产品，Janus-Pro 过度工程了——选一个 LLaVA 家族的理解模型。

对于不需要理解的产品，Janus-Pro 大材小用了——选一个 Stable Diffusion 3 / Flux 模型。

对于两者都需要的产品，Janus-Pro 现在是参考开源架构。

## Use It

`code/main.py` 模拟 Janus-Pro 路由：

- 两个 mock 编码器：SigLIP 风格（产出 256 维语义向量）和 VQ 风格（产出整数 code）。
- 一个 prompt 路由器，根据任务标签选择编码器。
- 一个共享 body（替身），无论哪个编码器产出的 token 序列都能处理。
- 从 stage 1（对齐）到 stage 3（指令调优）的加权采样调度切换。

打印 3 个示例的路由路径：图像 QA、T2I、图像编辑。

## Ship It

本课产出 `outputs/skill-decoupled-encoder-picker.md`。给定一个想要统一生成 + 理解且达到前沿质量的产品，在 Janus-Pro、JanusFlow 或 InternVL-U 之间选择，并给出具体的数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上超过 DALL-E 3。解释为什么一个 7B 开源模型能在生成上匹配前沿专有模型，但在理解上不能。

2. 实现一个路由函数：给定 prompt 文本，分类为 `understand` 或 `generate`。如何处理模糊的 prompt，比如"描述然后画出来"？

3. JanusFlow 用 rectified flow 替换了 VQ 路径。Transformer body 现在输出什么？损失有什么变化？

4. 提出 Janus-Pro 架构可以用一个额外解耦编码器处理的第四个任务。例如：图像分割（DINO 风格）、深度（MiDaS 风格）。

5. 阅读 Janus-Pro Section 4.2 关于数据扩展。哪个数据阶段对 T2I 质量提升贡献最大（vs Janus）？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Decoupled encoding | "两个视觉编码器" | 每个方向独立的 tokenizer 或编码器：语义用于理解，重建用于生成 |
| Shared body | "一个 transformer" | 单一 transformer 处理任一编码器的输出；无模态专属权重 |
| SigLIP for understanding | "语义特征" | CLIP 家族视觉塔提供丰富的概念特征但重建差 |
| VQ for generation | "重建编码" | 向量量化 token 能干净地解码回像素 |
| JanusFlow | "Rectified-flow 变体" | Janus-Pro 用连续 flow-matching 生成 head 替代 VQ |
| Routing tag | "任务标签" | Prompt 标记（`<understand>` / `<generate>`）选择输入编码器 |

## 延伸阅读

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
