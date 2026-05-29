# Show-o 与离散扩散统一模型

> Transfusion 混合了连续和离散表示。Show-o（Xie et al.，2024 年 8 月）走了另一条路：文本 token 用因果 next-token prediction，图像 token 用 MaskGIT 风格的 masked discrete diffusion。两者共存于一个 transformer 中，通过混合 attention mask 连接。结果是在一个 backbone、每个模态一个 tokenizer、一个损失公式（next-token 扩展为 masked prediction）上统一了 VQA、文生图、inpainting 和混合模态生成。本课讲解 Show-o 的设计——为什么 masked discrete diffusion 是一个并行、少步数的图像生成器——并与 Transfusion 和 Emu3 对比。

**Type:** Learn
**Languages:** Python (stdlib, masked-discrete-diffusion sampler)
**Prerequisites:** Phase 12 · 13 (Transfusion)
**Time:** ~120 minutes

## 学习目标

- 解释 masked discrete diffusion：均匀 mask token 然后让 transformer 恢复它们的调度策略。
- 对比并行图像解码（Show-o、MaskGIT）与自回归图像解码（Chameleon、Emu3）在速度和质量上的差异。
- 说出 Show-o 在一个 checkpoint 中处理的三个任务：T2I、VQA、image inpainting。
- 选择一个 masking schedule（cosine、linear、truncated）并推理其对样本质量的影响。

## 问题

Transfusion 的双损失训练能用，但动态更棘手——连续 diffusion loss 与离散 NTP loss 在数值尺度上不同。平衡损失权重是一个超参数搜索。架构有效但复杂。

Show-o 的答案：保持两个模态都是离散的（像 Chameleon），但通过 masked discrete diffusion 并行生成图像而非顺序生成。训练目标变成一个单一的 masked-token-prediction，自然地泛化了 next-token-prediction。

## 核心概念

### Masked discrete diffusion (MaskGIT)

Chang et al. (2022) 的 MaskGIT 技巧很优雅。从一个完全 mask 的图像开始（每个 token 都是特殊的 `<MASK>` id）。每一步并行预测所有 masked token，然后保留置信度最高的 top-K 预测，重新 mask 其余的。经过约 8-16 次迭代，所有 token 都被填充。每步 unmask 多少 token 的调度是可调的——cosine schedule 效果好。

训练很简单：从 [0, 1] 均匀采样一个 masking ratio，应用到图像的 VQ token 上，训练 transformer 恢复被 mask 的 token。和 BERT 对文本做的完全一样，只是扩展到了图像生成。

### Show-o：一个 transformer，混合 mask

Show-o 把 MaskGIT 放进了一个因果语言模型 transformer。Attention mask 是：

- 文本 token：因果（标准 LLM）。
- 图像 token：在 image block 内完全双向（这样 masked token 在预测时能看到其他所有图像 token）。
- 文本到图像：文本 attend 到之前的图像，图像 attend 到之前的文本。

训练交替进行：
1. 文本序列上的标准 NTP。
2. T2I 样本：文本 → 带 masked 图像 token 的图像，masked-token-prediction loss。
3. VQA 样本：图像 → 文本，带 masked 文本 token（实际上就是 NTP）。

统一损失是 `<MASK>` token 上的 cross-entropy，覆盖了文本 NTP（只有最后一个 token 被"mask"）和图像 masked-diffusion（随机子集被 mask）。

### 并行采样

Show-o 用约 16 步生成一张图像，而非约 1000 步（逐 token 自回归）或约 20 步（diffusion）。每一步并行预测所有 masked token；提交置信度最高的 top-K；重复。

对比：
- Chameleon / Emu3（逐 token 自回归）：N_tokens 次前向传播，每张图像通常 1024-4096 次。
- Transfusion（连续 diffusion）：约 20 步，每步一次完整 transformer pass。
- Show-o（masked discrete diffusion）：约 16 步，每步一次完整 transformer pass。

Show-o 在同规模模型下比 Chameleon 快，步数与 Transfusion 大致相当，但每步成本更低（离散 vocab logits vs 连续 MSE loss）。

### 一个 checkpoint 多任务

Show-o 在推理时支持四个任务，通过 prompt 格式选择：

- 文本生成：标准自回归文本输出。
- VQA：图像输入，文本输出。
- T2I：文本输入，通过 masked discrete diffusion 输出图像。
- Inpainting：图像中部分 token 被 mask，填充它们。

Inpainting 能力是 masked-prediction 训练免费带来的。Mask VQ-token 网格的一个区域，输入其余部分加文本 prompt，预测被 mask 的 token。

### Masking schedule

每步 unmask 多少 token 的调度影响质量。Show-o 推荐 cosine：

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

在 step 0，所有 token 被 mask（ratio 1.0）。在 step T，没有 token 被 mask。Cosine 把质量集中在中间比例，此时预测最有信息量。Linear schedule 也能用但更快饱和。

### Show-o2

Show-o2（2025 年后续，arXiv 2506.15564）扩展了 Show-o：更大的 LLM base、更好的 tokenizer、改进的 mask schedule。架构模式相同。

### Show-o 的定位

在 2026 年的分类中：

- 离散 token + NTP：Chameleon、Emu3。简单但推理慢。
- 离散 token + masked diffusion：Show-o、MaskGIT、LlamaGen、Muse。并行采样，仍受 tokenizer 有损限制。
- 连续 + diffusion：Transfusion、MMDiT、DiT。最高质量，训练更复杂。
- 连续 + flow matching in a VLM：JanusFlow、InternVL-U。最新。

按任务选择：Show-o 适合需要 T2I + inpainting + VQA 在一个开源模型中以合理速度运行的场景；Transfusion 适合质量至上且能承受双损失管道的场景。

## Use It

`code/main.py` 模拟 Show-o 采样：

- 一个 16 个 VQ token 的玩具网格。
- 一个 mock "transformer"，根据 prompt 和当前未 mask 的 token 预测 logits。
- 8 步 cosine schedule 的并行 masked 采样。
- 打印中间状态（mask 模式演变）和最终 token。

运行它，观察 mask 逐步消融。

## Ship It

本课产出 `outputs/skill-unified-gen-model-picker.md`。给定一个需要理解（VQA、captioning）和生成（T2I、inpainting）且有开源权重约束的产品，在 Show-o 家族、Transfusion/MMDiT 家族和 Emu3 / Chameleon 家族之间选择，并给出具体权衡。

## 练习

1. Masked discrete diffusion 用约 16 步采样。为什么不是 1 步？如果在 step 0 就 unmask 所有 token 会怎样？

2. Inpainting 在 masked diffusion 中是免费的。提出一个产品用例（真实或假设），其中 Show-o 的 inpainting 优于专用模型。

3. Cosine schedule vs linear schedule：对 T=8 追踪每步 unmask 的 token 数。哪个更均衡？

4. 一张 512x512 的 Show-o 图像是 1024 个 token。在 vocab K=16384 时，模型输出 1024 * log2(16384) = 14,336 bits（约 1.75 KiB）的数据。Stable Diffusion 输出 512*512*24 bits = 6,291,456 bits（约 768 KiB）的原始像素。压缩比是多少？这买到了什么质量？

5. 阅读 LlamaGen (arXiv:2406.06525)。LlamaGen 的类条件自回归图像模型与 Show-o 的 masked 方法有何不同？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Masked discrete diffusion | "MaskGIT 风格" | 训练预测被 mask 的 token；推理时迭代 unmask 置信度最高的预测 |
| Cosine schedule | "Unmask 调度" | 推理步数上 mask ratio 的衰减；将置信度增长集中在中间范围 |
| Parallel decoding | "一次预测所有 token" | 每步在一次前向传播中预测所有 masked token 的完整序列，然后提交 top-K |
| Hybrid attention | "因果 + 双向" | 文本 token 上因果、图像 block 内双向的 mask |
| Inpainting | "填充生成" | 以部分 token 被 mask 的图像为条件，预测缺失的 token；训练目标免费带来 |
| Commitment rate | "每步 Top-K" | 每次迭代声明"完成"的 token 数；控制推理速度 vs 质量的权衡 |

## 延伸阅读

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
