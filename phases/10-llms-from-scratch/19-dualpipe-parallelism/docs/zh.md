# DualPipe 并行

> DeepSeek-V3 在 2,048 块 H800 GPU 上训练，MoE expert 分散在各节点。跨节点的 expert all-to-all 通信每 1 GPU 小时的计算就要花 1 GPU 小时的通信。GPU 有一半时间在空转。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线，将前向和反向计算与它们触发的 all-to-all 通信重叠。气泡减少，吞吐量上升，而保留两份模型参数副本（"dual"名称的由来）在 Expert Parallelism 已经将 expert 分散到各 rank 的情况下代价很小。本课是一个 Learn 类型的讲解，介绍 DualPipe 实际做了什么，以及为什么 Sea AI Lab 的 DualPipeV 改进以略微更紧的气泡为代价去掉了 2 倍参数开销。

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open-model architectures and MoE)
**Time:** ~60 minutes

## 学习目标

- 说出 DualPipe 前向-反向 chunk 的四个组成部分，以及为什么每个都有自己的重叠窗口。
- 解释大规模下的流水线气泡问题，以及"无气泡"在实践中与在宣传中的含义差异。
- 手动追踪一个 8 PP rank、16 micro-batch 的 DualPipe 调度，确认前向和反向流填充了彼此的空闲槽位。
- 阐述 DualPipeV（Sea AI Lab，2025）所做的权衡：以 Expert Parallelism 不活跃时略大的气泡为代价，去掉 2 倍参数复制。

## 问题

在 2k 块 H800 GPU 上训练一个 671B MoE 模型会遇到三个叠加的瓶颈：

1. **显存压力。** 每块 GPU 持有模型的一个切片。在 128 个 head、61 层、序列长度 8k 下的激活显存是巨大的。
2. **流水线气泡。** 传统流水线并行（GPipe、1F1B）在 GPU 等待其 stage 的输入或梯度时让 GPU 空闲。在 8 个 stage 下，即使用 1F1B 调度，大约 12% 的 GPU 时间可能是气泡。
3. **跨节点 all-to-all。** 带有 expert parallelism 的 MoE 将 expert 分散到各节点。每次前向传播触发一次 all-to-all 将 token 分发到其 expert，再一次将结果合并回来。在 2k GPU 下这很容易变成 1:1 的计算通信比。

每个问题都有单独的解决方案：gradient checkpointing 解决显存，Zero Bubble（Sea AI Lab，2023）解决流水线气泡，expert-parallel 通信内核解决 all-to-all。DualPipe 做的是让它们协同工作。调度在单个前向-反向 chunk 内重叠计算和通信，从流水线两端同时注入 micro-batch，并利用由此产生的调度将 all-to-all 隐藏在计算窗口内。

报告结果：几乎消除流水线气泡，DeepSeek-V3 的 14.8T token 训练中 GPU 利用率超过 95%。

## 概念

### 流水线并行回顾

将 N 层模型分割到 P 个设备上。设备 `i` 持有层 `i * N/P .. (i+1) * N/P - 1`。一个 micro-batch 从设备 0 前向流到 P-1，然后从 P-1 反向流到 0。每个设备只有在前一个设备发送其输出后才能开始前向 stage，只有在下游设备发送上游梯度后才能开始反向。

GPipe（Huang et al., 2019）一次调度一个 micro-batch，浪费了大部分 GPU 时间。1F1B（Narayanan et al., 2021）为多个 micro-batch 交错前向和反向传播。Zero Bubble（Qi et al., 2023）将反向传播分为两部分——backward-for-input (B) 和 backward-for-weights (W)——并调度它们来填充气泡。Zero Bubble 之后，流水线几乎是紧凑的。

DualPipe 是下一步。它在此基础上添加了两个想法：

### 想法 1：chunk 分解

每个前向 chunk 被分为四个组件：

- **Attention。** Q/K/V 投影、attention、输出投影。
- **All-to-all dispatch。** 跨节点通信，将 token 发送到其 expert。
- **MLP。** MoE expert 计算。
- **All-to-all combine。** 跨节点通信，将 expert 输出带回。

反向 chunk 添加每个组件的梯度版本。DualPipe 调度它们使得 all-to-all dispatch 与下一个 chunk 的 attention 计算并行发生，all-to-all combine 与后续 chunk 的 MLP 计算并行发生。

### 想法 2：双向调度

大多数流水线调度从 stage 0 注入 micro-batch 并流向 stage P-1。DualPipe 从两端注入 micro-batch。Stage 0 看到从那里发起的前向 micro-batch；stage P-1 也看到从那里发起的前向 micro-batch。两个流在中间相遇。

为此，设备 `i` 必须同时持有早期流水线层 `i` 和晚期流水线层 `P - 1 - i`。这就是 DualPipe 的"dual"部分：每个设备保留它需要服务的模型层的两份副本（每个方向一份）。在 DeepSeek-V3 的规模下，这是 2 倍参数复制成本。这是可以承受的，因为 Expert Parallelism 已经将 MoE expert 分散得很薄，复制非 expert 层两次只是小事。

关键是，一个方向的前向流和另一个方向的反向流恰好在单向调度中气泡出现的位置重叠。气泡消失了。

### 手动追踪的调度

考虑 P = 4 rank，8 个 micro-batch，分为 4 个前向 / 4 个反向。时间从左到右；行是设备 rank。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

解读"F4/F5R"标记：rank 1 在同一时间槽中运行 micro-batch 4 的前向（在流水线中从左到右）和 micro-batch 5 的前向（从右到左）。这就是"双向"在操作上的含义。

在 rank 2 交叉流更早重叠，在 rank 0 和 P-1 最晚重叠。在调度的稳定中间阶段，每个 rank 运行 X 方向的前向与 Y 方向的反向重叠。计算一直在忙。前向传播的 all-to-all dispatch 隐藏在反向计算中。All-to-all combine 隐藏在前向计算中。气泡被挤出去了。

### 气泡核算

标准 1F1B 流水线气泡（每个 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 改进将其降低但不到零。DualPipe 在稳定阶段，如果 micro-batch 数量能被 2 倍流水线深度整除，则气泡为零。在稳定阶段之外（warmup 和 cooldown），有一些气泡但不随 micro-batch 数量增长——这是论文强调的关键特性。

宣传用语："无气泡"。技术用语：气泡不随 micro-batch 数量增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）表明完全零气泡只在 Expert Parallelism 不是瓶颈时成立；有 EP 驱动的 all-to-all 时，总有一些调度妥协。

### DualPipeV——改进版

Sea AI Lab（2025）观察到当 EP 通信重叠不是重点时，2 倍参数复制是浪费的。他们的 DualPipeV 调度将双向注入折叠为"V 形"调度，在单份参数副本上运行。气泡比 DualPipe 略大，但显存节省很可观。DeepSeek 在其开源 DualPipe 实现中采用了 DualPipeV 作为 EP-off 模式。

权衡：

| Feature | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Param copies per device | 2 | 1 | 1 | 1 |
| Bubble vs micro-batches | constant | small growth | grows | grows |
| Compute-comm overlap | full | partial | minimal | partial |
| Use when | EP-heavy MoE | dense or EP-light | baseline | any pipeline |

### 对 14.8T token 训练的意义

DeepSeek-V3 的预训练在 2,048 块 H800 GPU 上消耗了 14.8T token，大约 2.8M GPU 小时。用朴素的 1F1B，他们会损失 12-15% 给流水线气泡——340-420K GPU 小时，足以训练一个完整的 70B 模型。DualPipe 恢复了其中大部分。没有内部日志很难直接量化贡献，但论文中的声明是训练期间平均 GPU 利用率超过 95%。

对于较小的运行（1k GPU 以下），DualPipe 是杀鸡用牛刀——流水线气泡相对于总成本较小，dense 模型训练很少遇到 all-to-all 瓶颈。对于数千 GPU 规模的前沿 MoE 训练，它实际上是必需的。

### 在技术栈中的位置

- 与 **FSDP**（Phase 10 · 05）互补。FSDP 将模型参数分片到各 rank；DualPipe 调度各 rank 的计算。它们组合使用。
- 与 **ZeRO-3** 梯度分片兼容。两份副本复制的簿记需要与 ZeRO 的分片梯度协作。
- 需要针对特定集群拓扑调优的**自定义 all-to-all 内核**。DeepSeek 的开源内核是参考实现。

## 使用

`code/main.py` 是一个流水线调度模拟器。它接受 `(P, n_micro_batches, schedule)` 并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 各自的稳定阶段利用率。这是一个教学工具——数字与论文中的定性声明匹配，不是对生产实测加速的声明。

模拟器的价值：用不同的 P 和 micro-batch 数量运行它，观察气泡比例如何对 1F1B 增长但对 DualPipe 不增长。

实际训练运行的集成考虑：

- 选择一个能整除你的 micro-batch 数量的流水线并行深度。
- 确保你的 expert-parallel mesh 支持双向 all-to-all。DeepSeek 的内核是参考。
- 第一次预计要花一周调试时间在调度本身上。簿记很繁琐。
- 监控每个 rank 的 GPU 利用率，而不仅仅是聚合值。DualPipe 的收益来自收紧落后者。

## 交付

本课产出 `outputs/skill-dualpipe-planner.md`。给定训练集群规格（GPU 数量、拓扑、互连、模型形状），它推荐流水线并行策略、使用的调度算法，以及目标规模下的预期气泡比例。

## 练习

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU 利用率差异，并将其表示为每百万 token 训练恢复的 GPU 小时。

2. 手动画出 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。用 micro-batch ID 和方向标记每个时间槽。找出气泡首次消失的时间槽。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的 Figure 5。找出 DualPipe 前向 chunk 中 all-to-all dispatch 的重叠窗口。解释计算调度如何隐藏它。

4. 计算 DualPipe 对一个 70B dense 模型（P=8 pipeline stages）和一个 671B MoE 模型（P=16 pipeline stages）的 2 倍参数开销。说明为什么 MoE 情况的开销按比例更小（大部分参数是 expert，分片在大的 EP 组中）。

5. 将 DualPipe 与 Chimera（2021 年的一个竞争性双向调度器）对比。使用论文 Section 3.4 作为参考，找出 DualPipe 添加而 Chimera 没有的两个特定属性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Pipeline bubble | "Idle time per rank" | 因流水线 stage 等待输入或梯度而浪费的 GPU 周期 |
| 1F1B | "Default pipeline schedule" | 一前一后交错调度；DualPipe 所超越的 baseline |
| Zero Bubble | "Sea AI Lab 2023" | 将反向分为 B（输入梯度）和 W（权重梯度）；几乎完全收紧流水线 |
| DualPipe | "DeepSeek-V3 schedule" | 双向流水线 + 计算通信重叠；气泡不随 micro-batch 数量增长 |
| DualPipeV | "Cut-in-half" | V 形改进，以略大气泡为代价去掉 2 倍参数复制 |
| Chunk | "Unit of pipeline work" | 一个 micro-batch 通过一个 pipeline stage 的一次前向或反向传播 |
| All-to-all dispatch | "Send tokens to experts" | 跨节点通信，将 token 路由到其分配的 MoE expert |
| All-to-all combine | "Bring expert outputs back" | 跨节点通信，在 MLP 之后收集 expert 输出 |
| Expert Parallelism (EP) | "Experts across GPUs" | 将 MoE expert 分片到各 rank，使不同 GPU 持有不同 expert |
| Pipeline Parallelism (PP) | "Layers across GPUs" | 将模型层分片到各 rank；DualPipe 调度的维度 |
| Bubble fraction | "Wasted GPU time" | (bubble_time / total_time)；DualPipe 驱向零的比例 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — DualPipe 的主要参考
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — 开源参考实现，包括 DualPipeV (Cut-in-half) 模式
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble 前身
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — 影响 DeepSeek EP-off 模式的 DualPipeV 分析
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe 对比的 1F1B 调度
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — 原始流水线并行论文和气泡问题
