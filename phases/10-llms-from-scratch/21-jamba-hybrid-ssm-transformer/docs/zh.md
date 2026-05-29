# Jamba — 混合 SSM-Transformer

> 状态空间模型（SSM）和 Transformer 追求不同的东西。Transformer 以二次代价的 attention 换取质量。SSM 以递推获得线性时间推理和常数内存，但质量落后。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）把两者放进同一个模型：每 7 个 Mamba 层配 1 个 Transformer 层，每隔一个 block 使用 MoE，256k 上下文窗口可以装进单张 80GB GPU。Mamba-3（ICLR 2026）用复数值状态空间和 MIMO 投影收紧了 SSM 侧。本课端到端阅读两种架构，并解释为什么混合方案在三年的规模扩展中存活下来，而纯 SSM 和纯 Transformer 的长上下文尝试没有。

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**Time:** ~60 minutes

## 学习目标

- 解释 Jamba block 中的三个基本组件——Transformer 层、Mamba 层、MoE——以及 1:7:隔一个的交错方案。
- 高层次描述 SSM 的递推形式，以及为什么它能实现常数内存推理。
- 计算 Jamba 模型在 256k 上下文下的 KV cache 占用，并与纯 Transformer 模型对比。
- 说出 Mamba-3 的三项创新（exponential-trapezoidal discretization、complex-valued state update、MIMO）以及每项针对的问题。

## 问题

Attention 对序列长度是二次的。状态空间模型是线性的。这个差异会累积：在 256k token 时，Transformer 的 attention map 每个 head 有 65B 个条目；SSM 的递推状态无论序列多长都是固定大小。

纯 SSM 模型（Mamba、Mamba-2）在小规模下匹配 Transformer 的困惑度，但在状态追踪任务上落后，在某些类别的上下文内检索上失败。直觉是：SSM 将历史压缩到固定状态中，当历史很长时信息会泄漏。Attention 精确记住一切但付出二次代价。

显而易见的修复：两者都用。在需要精确回忆的地方放 Transformer 层。其他地方用 SSM 层。调节比例。Jamba 是第一个在规模化生产中发布这种混合方案的模型（总参数 52B，活跃 12B，256k 上下文，单张 80GB GPU）。Jamba 1.5 将家族扩展到总参数 398B / 活跃 94B。Mamba-3（ICLR 2026）是当前最佳的纯 SSM 基线，混合模型可以围绕它重建。

本课阅读所有三篇论文，建立"选择正确比例"的心智模型。

## 概念

### 一页纸讲清 SSM

状态空间模型通过固定大小的状态 `h` 处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每一步状态通过线性动力学 `A` 演化，接收输入 `B x_t`，输出 `C h_t`。`A, B, C` 可以学习。注意关键性质：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要更早的 `x`。内存是常数的。推理每 token 是 O(1)。

建模质量的关键在于 `A` 的结构。S4（Gu 2021）使用了高度结构化的矩阵，可以在训练时高效地作为长卷积求值。Mamba（Gu, Dao 2023）用数据依赖的 `A, B, C` 替换了固定的（即"selective"部分）。Mamba-2（2024）进一步简化了结构。Mamba-3（2026）在特定位置重新增加了复杂性。

关键性质：对于 decoder LLM，SSM 层是 attention 层的直接替代品，用固定大小的每层状态替代不断增长的 KV cache。

### Jamba block

Jamba block 根据两个数字交错排列层：

- `l`：attention 与 Mamba 的比例。Jamba 使用 `l = 8`，即每 7 个 Mamba 层配 1 个 Transformer 层（7 Mamba + 1 Attention = 每组 8 层）。
- `e`：MoE 频率。Jamba 使用 `e = 2`，即每隔一层应用 MoE。

一个 block 内的层序列：

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

每个 Jamba block 是 8 层。4 个 block 深（共 32 层），你得到 28 个 Mamba 层和 4 个 Attention 层。其中 16 层使用 MoE。

### 为什么是 1:7 比例

AI21 做了消融实验：什么 attention-to-Mamba 比例在困惑度/参数量和长上下文评估的上下文内回忆上给出最佳结果？

- 太多 attention（1:1）：质量上升但内存和速度下降。
- 太少 attention（1:15）：内存很好但上下文内检索失败。
- 最佳点：1:7 或 1:8。

直觉：Transformer 层处理精确回忆和状态追踪。Mamba 层处理廉价的大量计算。

### 位置编码

Mamba 层本身具有位置感知能力（通过递推）。原始基于 Mamba 的混合模型中的 Attention 层没有使用 RoPE——SSM 层提供了位置信息。Jamba 1.5 在 Attention 层添加了 RoPE 以获得更好的长上下文泛化，这是基于经验性长上下文评估的事后改进。

### 内存预算

对于 Jamba-1 形状（32 层：28 Mamba + 4 Attention，hidden 4096，32 个 attention head）：

- KV cache（仅 attention 层）：`2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`（256k BF16）。只有 4 个 attention 层贡献。
- SSM 状态：`28 * hidden * state_size` 每个 token 前缀，但这是每层固定大小，不随序列长度增长。典型 Mamba 状态是每特征 16，hidden 4096：`28 * 4096 * 16 * 2 = 3.7 MB` 总计。

对比纯 Transformer（32 层，相同 hidden，full MHA 32 head）：`2 * 32 * 32 * 128 * 256k * 2 = 128 GB`（256k BF16）。KV cache 减少 8 倍。即使对比大多数 2024 模型使用的 GQA(8) 基线（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 混合在 16 GB 仍然小 2 倍。

这就是 AI21 所说的"256k 上下文在单张 80GB GPU 上"的含义。纯 Transformer 的 full-MHA KV cache 装不下；即使 GQA 基线也不留空间给权重和激活值；Jamba 的可以。

### Mamba-3：2026 年的纯 SSM 基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯 SSM 侧引入三项创新：

1. **Exponential-trapezoidal discretization。** 用更具表达力的递推替换 Mamba-2 中的 Euler 方法离散化。在核心递推内部对状态-输入应用类卷积操作，而非作为 `x_t` 上的外部卷积。

2. **Complex-valued state update。** 之前的 Mamba 将状态矩阵从复数（S4）简化为实对角（Mamba）再到缩放单位矩阵（Mamba-2）。Mamba-3 重新加入复数值——等价于状态上的数据依赖旋转位置编码。这恢复了之前实值简化所损失的状态追踪能力。

3. **Multi-input multi-output (MIMO) projections。** 不再使用逐特征的标量投影，而是使用矩阵值投影。在不增加解码延迟的情况下提升建模能力和推理时的硬件利用率。

在 1.5B 参数下，Mamba-3 比 Gated DeltaNet 平均下游准确率提高 0.6 个点；MIMO 变体再加 1.2 个点，总计 1.8 个点的提升。在相同状态大小下，Mamba-3 用一半的状态匹配 Mamba-2。

Mamba-3 尚未在规模化生产混合模型中发布——但它是下一代 Jamba 级模型 SSM 侧的明显候选者。

### 何时选择混合架构

混合架构胜出的场景：

- 上下文足够长，纯 Transformer KV cache 变得痛苦（64k+）。
- 任务混合了短程结构（适合 SSM）和长程回忆（需要 Transformer）。
- 你想在单 GPU 内存预算上部署，而纯 Transformer 的 KV cache 本身就装不下。

混合架构不适合的场景：

- 上下文短（16k 以下）。SSM 开销被浪费；纯 Transformer 就够了。
- 任务需要全对全 attention（深度推理、多文档交叉引用）。混合架构中 attention 层的稀疏性会造成伤害。
- 你在扩展到万亿参数的前沿模型。纯 Transformer + MLA + MoE（DeepSeek-V3 风格）目前在能力竞赛中领先。

### 竞争格局

| Model | Family | Scale | Unique claim |
|-------|--------|------|-------------|
| Mamba-2 | pure SSM | 3B | linear time, constant memory |
| Jamba | hybrid | 52B/12B | 256k on 80GB |
| Jamba 1.5 Large | hybrid | 398B/94B | enterprise-grade long-context |
| Mamba-3 | pure SSM | 1.5B (paper) | state-tracking restored |
| DeepSeek-V3 | pure Transformer + MoE | 671B/37B | frontier capability |

2026 年格局：纯 Transformer MoE 主导前沿，但混合架构占据 256k+ 上下文的细分市场。Mamba-3 的状态追踪优势可能推动下一代混合比例更低（更多 SSM，更少 attention）。

## 使用

`code/main.py` 是混合架构的内存计算器。给定 SSM-Transformer 比例和 hidden-size / layer-count 配置，它计算：

- 目标上下文下的 KV cache。
- SSM 状态内存。
- 一系列模型形状在上下文 N 下的总内存。

计算器支持：

- 纯 Transformer 基线（KV cache 随 N 增长）。
- Jamba 风格 1:7 混合。
- 纯 SSM（完全没有 KV cache）。

数字直接来自 Jamba-1 和 Jamba-1.5 论文的公开形状，假设变体为外推。

实际部署的集成考虑：

- 大多数生产推理服务器（vLLM、SGLang）支持 Jamba 和 Mamba。检查具体版本。
- 在 256k 上下文下，Jamba 的内存优势体现在并发请求吞吐量上。在相同 VRAM 上你能装下更多 Jamba 序列而非 Transformer 序列。
- Mamba-3 作为独立模型尚未在生产中发布——1.5B 的研究预览。

## 交付

本课产出 `outputs/skill-hybrid-picker.md`。给定工作负载规格（上下文长度分布、任务组合、内存预算），它在纯 Transformer、Jamba 风格混合和纯 SSM 之间推荐，并给出关于内存和质量权衡的明确推理。

## 练习

1. 运行 `code/main.py` 计算 32 层纯 Transformer（hidden 4096，32 head）和相同形状的 Jamba-1 混合模型在 256k 上下文下的 KV cache。验证 AI21 论文声称的 ~8 倍内存减少。

2. 修改计算器以建模 1:3 混合（4 Mamba : 1 Attention）和 1:15 混合（14 Mamba : 1 Attention）。绘制 KV cache vs 比例。在什么比例下 KV cache 等于 SSM 状态内存？

3. 阅读 Jamba 论文（arXiv:2403.19887）Section 3。解释为什么 AI21 使用 Mamba-1 而非 Mamba-2，尽管 Mamba-2 更快。提示：混合消融实验部分记录了这一点。

4. 计算 Jamba 1.5 Large（总参数 398B，活跃 94B）中每隔一层 MoE 的参数开销。将活跃比例与 DeepSeek-V3（37B/671B）对比，解释为什么 Jamba 的架构推高了活跃比例。

5. 阅读 Mamba-3 论文（arXiv:2603.15569）Section 3。用三句话解释为什么复数值状态更新等价于数据依赖的旋转位置编码。将答案与 Phase 7 · Lesson 04 的 RoPE 推导联系起来。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| State space model (SSM) | "带固定状态的递推" | 具有学习递推 `h_t = A h_{t-1} + B x_t` 的层；每 token 常数内存 |
| Selective SSM | "Mamba 的技巧" | 数据依赖的 A, B, C 参数，在线性时间内给模型类似门控的选择性 |
| Attention-to-Mamba ratio | "多少 attention 层" | 在 Jamba 中，`l = 8` 意味着每 7 个 Mamba 层配 1 个 attention 层 |
| Jamba block | "8 层组" | 一个 attention + 七个 Mamba + 交替位置的 MoE |
| SSM state | "隐缓冲区" | 固定大小的每层状态，替代 Mamba 层的 KV cache |
| 256k context | "Jamba 的旗舰数字" | Jamba-1 在单张 80GB GPU 上能容纳的序列长度；纯 Transformer 在该规模下做不到 |
| Mamba-3 | "2026 纯 SSM" | 当前最佳纯 SSM 架构，具有复数状态 + MIMO；混合模型围绕它重建的基线 |
| MIMO | "Multi-input multi-output" | Mamba-3 创新，使用矩阵值投影替代逐特征标量 |
| Exponential-trapezoidal discretization | "Mamba-3 的递推" | 更具表达力的递推，包含 Mamba-2 的 Euler 方法离散化作为特例 |
| Hybrid architecture | "混合 attention 和 SSM" | 任何交错 Transformer 和 SSM 层的模型；Jamba 是生产原型 |

## 延伸阅读

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — 原始 Jamba 论文，比例消融，256k 上下文声明
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — 扩展后的家族，398B/94B 和 12B/52B 公开发布
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — Jamba 构建于其上的 selective SSM 论文
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — 简化的结构化状态空间后继者
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — 复数值状态、MIMO、2026 纯 SSM 前沿
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — S4 论文，LLM 的 SSM 谱系起点
