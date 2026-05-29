# Transfusion：一个 Transformer 同时做自回归文本 + 扩散图像

> Chameleon 和 Emu3 把赌注全押在离散 token 上。它们能用，但量化瓶颈肉眼可见——图像质量在连续空间扩散模型之下就触顶了。Transfusion（Meta，Zhou et al.，2024 年 8 月）走了相反的路：保持图像连续，完全丢弃 VQ-VAE，用两个损失训练一个 transformer。文本 token 用 next-token-prediction，图像 patch 用 flow-matching / diffusion loss。两个目标优化同一组权重。Stable Diffusion 3 底层的 MMDiT 架构是它的近亲。本课阅读 Transfusion 的核心论点，构建一个玩具级双损失训练器，并追踪让一个 transformer 同时完成两项工作的 attention mask。

**Type:** Build
**Languages:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Time:** ~180 minutes

## 学习目标

- 搭建一个在同一个 backbone 上运行两个损失（文本 token 的 NTP、图像 patch 的 diffusion MSE）的 transformer。
- 解释为什么图像 patch 之间双向注意力 + 文本 token 之间因果注意力是正确的 mask 选择。
- 对比 Transfusion 风格（连续图像，diffusion loss）与 Chameleon 风格（离散图像，NTP）在计算量、质量和代码复杂度上的差异。
- 说明 MMDiT 的贡献：每个 block 有模态专属权重，残差流上做联合注意力。

## 问题

离散 vs 连续图像 token 的争论比 LLM 还早。连续表示（原始像素、VAE latent）保留细节。离散 token（VQ 索引）适配 transformer 的原生词表，但在量化步骤丢失细节。

Chameleon / Emu3 走离散路线：一个损失、一个架构，但图像保真度被 tokenizer 质量封顶。

扩散模型走连续路线：图像质量卓越，但与 LLM 是独立模型，噪声调度工程复杂，且无法与文本生成干净集成。

Transfusion 问：能不能两者兼得？保持图像连续，仍然训练一个模型，用两个损失缝合进一个梯度步。

## 核心概念

### 双损失架构

一个 decoder-only transformer 处理包含以下内容的序列：

- 文本 token（离散，来自 BPE 词表）。
- 图像 patch（连续，16x16 像素块通过线性嵌入投射到 hidden dim——与 ViT encoder 的输入方式相同）。
- `<image>` 和 `</image>` 标签标记连续 patch 的位置。

前向传播只跑一次。损失根据每个 token 选择两个 head 之一：

- 文本 token：标准 cross-entropy，走 vocab-logits head。
- 图像 patch：连续 patch 上的 diffusion loss——预测加到每个 patch 上的噪声。

梯度流过共享的 transformer body。两个损失同时改善共享权重。

### Attention mask：因果文本 + 双向图像

文本 token 必须是因果的——不能让文本 token attend 到未来的文本，否则 teacher forcing 就失效了。但图像 patch 代表同一个快照；它们应该在同一个 image block 内双向 attend。

Mask 规则：

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

实现为训练和推理时的分块三角 mask。

### Transformer 内部的 diffusion loss

Diffusion loss 是标准的：给图像 patch 加噪声，让模型预测噪声（或等价地预测干净 patch）。Transfusion 的版本使用 flow matching——预测从噪声到干净的速度场。

训练时：
1. 对每个图像 patch x0，采样一个随机时间步 t。
2. 采样噪声 ε，计算 xt = (1-t) * x0 + t * ε（flow matching 的线性插值）。
3. Transformer 预测 v_theta(xt, t)；loss = MSE(v_theta(xt, t), ε - x0)。
4. 与同一序列中的文本 NTP loss 一起反向传播。

推理时的生成：
- 文本 token：标准自回归采样。
- 图像 patch：diffusion 采样循环（通常 10-30 步），以前面的文本 token 为条件。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3（Esser et al.，2024 年 3 月）与 Transfusion 几乎同时推出了 MMDiT（Multimodal Diffusion Transformer）。两者是兄弟架构。

MMDiT 的关键差异：

- 每个 block 有模态专属权重。每个 transformer block 对文本 token 和图像 patch 有独立的 Q、K、V 和 MLP 权重。注意力是联合的（跨模态）；其他都是模态专属的。
- Rectified flow 训练。一种特定的 flow-matching 变体，采样已知且数学比 DDPM 更简单。
- 规模。MMDiT 是 SD3 的 backbone（2B 和 8B 参数变体）。Transfusion 论文扩展到 7B。

两者收敛到同一个核心思想：一个 transformer 在文本上跑 NTP，在连续图像表示上跑 diffusion。

### 为什么比 Chameleon 风格更好

连续 diffusion 与离散 NTP 在图像生成上的质量差距是可测量的。Transfusion 论文报告：

- 在 7B 参数下，FID 比同规模 Chameleon 风格模型好 3-5 分。
- 不需要训练 tokenizer——图像编码器更简单（线性投射到 hidden，与 ViT 输入层相同）。
- 推理可以并行化图像 patch 去噪，不像自回归图像 token。

缺点：Transfusion 是双损失模型，训练动态更棘手。损失权重需要调优。NTP 和 diffusion 之间的调度不匹配可能导致一个 head 主导。

### 下游发展

Janus-Pro（Lesson 12.15）改进了 Transfusion 的思路，将理解和生成的视觉编码器解耦——SigLIP 用于理解，VQ 用于生成——同时共享 transformer body。Show-o（Lesson 12.14）用离散扩散（masked prediction）替换了 diffusion。统一生成家族在 Transfusion 之后迅速分支。

2026 年能生成图像的生产级 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎可以确定使用了这个家族的某个后代。细节是专有的。

## Use It

`code/main.py` 在一个 MNIST 级别的小问题上构建了一个玩具 Transfusion：

- 文本 caption 是描述数字（0-9）的短整数序列。
- 图像是 4x4 字节网格。
- 一对共享权重的线性投射充当 transformer 替身；文本上 NTP loss，噪声 patch 上 MSE loss。
- 训练循环交替两个损失，attention mask 是显式的。
- 生成在一次前向传播中产出文本 caption 和 4x4 图像。

Transformer 是玩具级的。双损失管道、attention mask 构造和推理循环才是真正的产出物。

## Ship It

本课产出 `outputs/skill-two-loss-trainer-designer.md`。给定一个新的多模态训练任务（文本 + 图像、文本 + 音频、文本 + 视频），它设计双损失调度（损失权重、mask 形状、共享 vs 模态专属 block）并标记实现风险。

## 练习

1. 一个 Transfusion 风格模型训练 70% 文本 token 和 30% 图像 patch。图像 diffusion loss 的量级约为文本 NTP loss 的 10 倍。什么损失权重能平衡它们？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现分块三角 mask。标记每个条目为 0 或 1。

3. MMDiT 有模态专属 QKV 权重。相比 Transfusion 的完全共享 transformer，这增加了多少参数开销？在 7B 参数下值得吗？

4. 生成：给定一个文本 prompt，模型跑 NTP 生成 50 个 token，然后遇到 `<image>`，然后在 256 个 patch 上跑 20 步去噪的 diffusion。总共多少次前向传播？

5. 阅读 SD3 论文 Section 3。描述 rectified flow 以及为什么它比 DDPM 用更少的推理步数收敛。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Two-loss training | "NTP + diffusion" | 一个 transformer 在同一个梯度步中同时优化文本 token 的 cross-entropy 和连续图像 patch 的 MSE |
| Flow matching | "Rectified flow" | 预测从噪声到干净数据的速度场的 diffusion 变体；数学比 DDPM 更简单 |
| MMDiT | "Multimodal DiT" | Stable Diffusion 3 的架构：联合注意力，模态专属 MLP 和 norm |
| Block-triangular mask | "因果文本 + 双向图像" | 文本上因果、图像区域内双向的 attention mask |
| 连续图像表示 | "No VQ" | 图像 patch 作为实值向量，而非整数 codebook 索引 |
| Velocity prediction | "v-parameterization" | 网络输出是噪声和数据之间的速度场，而非噪声本身 |

## 延伸阅读

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
