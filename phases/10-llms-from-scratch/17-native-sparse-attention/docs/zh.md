# Native Sparse Attention (DeepSeek NSA)

> 在 64k token 时，attention 吃掉 70-80% 的解码延迟。每个开放模型实验室都有修复它的计划。DeepSeek 的 NSA（ACL 2025 最佳论文）是留下来的那个：三个并行 attention 分支——压缩的粗粒度 token、选择性保留的细粒度 token、以及用于局部上下文的 sliding window——通过学习的门控组合。它是硬件对齐的（kernel 友好）、原生可训练的（在预训练中工作，不是推理时附加的），在 64k 解码上比 FlashAttention 更快同时匹配或超过全 attention 质量。本课端到端构建三个分支并展示为什么稀疏性是端到端可微的。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 12 (KV cache, flash-attention), Phase 7 · 15 (attention variants), Phase 10 · 16 (differential attention)
**Time:** ~60 minutes

## 学习目标

- 陈述 NSA 的三个 attention 分支以及每个捕获什么。
- 解释为什么 NSA 是"原生可训练的"而之前的 sparse-attention 方法只能在推理时使用。
- 计算 NSA 相对于全 attention 在 64k 上下文下的计算节省，作为压缩块大小和选择 top-k 的函数。
- 用 stdlib Python 在短合成序列上实现三分支组合并验证门控权重的行为。

## 问题

全 attention 在序列长度 N 时花费 `O(N^2)` 时间和每层 `O(N)` KV cache。在 64k token 时，计算和内存带宽数字是灾难性的。NSA 论文的测量理论估计：在 64k 时 attention 占总解码延迟的 70-80%。下游的一切——TTFT、tokens/sec、每百万 token 成本——都被 attention 成本主导。

Sparse attention 是显而易见的答案。之前的尝试分为两类。固定模式稀疏（sliding-window、strided、block-local）丢弃信息并在长程召回任务上失败。推理时稀疏（KV cache 剪枝、H2O、StreamingLLM）应用于在 dense attention 上预训练的模型，只能恢复潜在加速的一小部分，因为模型从未被要求通过稀疏模式路由信息。

Native Sparse Attention（Yuan et al., DeepSeek + PKU + UW, ACL 2025 最佳论文, arXiv:2502.11089）两者兼顾：一个模型在预训练期间学习的稀疏模式，实现为一个 kernel 对齐的算法，在推理时真正交付计算节省。两年后，NSA 或其直接后继是每个前沿长上下文模型的默认 attention。

## 概念

### 三个并行分支

对每个查询，NSA 运行三次 attention，针对 KV cache 的三个不同视图：

1. **压缩分支。** Token 被分组为大小 `l` 的块（通常 32 或 64）。每个块通过一个小的学习 MLP 压缩为单个摘要 token。查询在这些压缩 token 上做 attention，获得整个序列的粗粒度视图。

2. **选择分支。** 使用压缩分支的 attention 分数，识别与当前查询最相关的 top-k 块。从这些块中读取细粒度（未压缩）token，查询在所有这些 token 上做 attention。将压缩分支 attention 视为选择的路由信号。

3. **Sliding-window 分支。** 查询关注最近的 `W` 个 token（通常 512）以获取局部上下文。这个分支捕获其他两个可能遗漏的结构密集的短程模式（语法、局部共指）。

三个分支输出通过学习的逐位置门控组合：

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` 是查询上小 MLP 的门控权重。它们不必和为 1——可以独立加权各分支。

### 为什么这是"原生可训练的"

选择步骤（top-k 块）是离散的。离散操作打断梯度流。之前的 sparse-attention 工作要么跳过通过选择的反向传播（限制训练），要么使用在推理时不给出真正稀疏性的连续松弛。

NSA 绕过了这个问题：压缩分支 attention 本身就是对整个序列的可微粗粒度 attention。top-k 操作只是复用压缩分支的最高 attention 分数来选择加载哪些细粒度块。梯度通过压缩分支分数流动（它们同时影响压缩输出和选择逻辑），被选块对最终输出的贡献也是可微的。不可微的 `top_k` 操作在前向计算图上是空操作——它只控制从内存加载哪些块。

这就是为什么 NSA 可以在预训练中端到端使用。模型学会联合通过三个分支路由信息，产生一个在推理时真正交付承诺加速的稀疏模式。

### 硬件对齐的 kernel

NSA 的 kernel 为现代 GPU 内存层次设计。Kernel 按 GQA 组加载查询（外循环），获取每组对应的稀疏 KV 块（内循环），在 SRAM 上运行 attention。因为每个查询组看到相同的被选块（选择是 per-query-group 而非 per-query-head），KV 加载在组内摊销。算术强度保持高位。

论文报告 Triton kernel 在 64k 解码上比 FlashAttention 快 9x，加速比随序列长度增长。前向和反向 kernel 都提供了。

### 计算预算

设 `N` 为序列长度，`l` 为压缩块大小，`k` 为 top-k 选择数，`w` 为 sliding window，`b` 为被选块大小（通常等于 `l`）。

- 压缩分支：每查询 `O(N/l)` 个 key，总计 `O(N * N / l)`。
- 选择分支：每查询 `O(k * b)` 个 key，总计 `O(N * k * b)`。
- Sliding 分支：每查询 `O(w)` 个 key，总计 `O(N * w)`。

总计：`O(N * (N/l + k*b + w))`。

`N = 64k, l = 64, k = 16, b = 64, w = 512` 时：每查询成本为 `1000 + 1024 + 512 = 2536` 个 key。全 attention 是 `64000` 个 key。25x 计算缩减。

`N = 128k, l = 64, k = 16, b = 64, w = 512` 时：每查询成本为 `2000 + 1024 + 512 = 3536` 个 key。全 attention 是 `128000` 个 key。36x 缩减。收益随序列长度增长，这正是重点。

### 对比

| 方法 | 可微 | 真实推理加速 | 长程召回 |
|------|------|------------|---------|
| 仅 Sliding window | 是 | 是 | 失败 |
| Strided / block-sparse | 是 | 是 | 部分 |
| KV 剪枝 (H2O, StreamingLLM) | N/A（推理时） | 是 | 部分 |
| MoBA (Moonshot) | 部分 | 是 | 好 |
| NSA | 是（原生） | 是（64k 时 9x） | 匹配全 attention |

MoBA（Moonshot, arXiv:2502.13189）同期发表，采用类似的三合一方法，将 MoE 原则应用于 attention 块。NSA 和 MoBA 是 2026 年长上下文预训练需要了解的两个架构。

## 构建

`code/main.py` 在短合成序列上实现三个分支并展示：

- 压缩 MLP（为教学清晰使用简单的均值池化基线；真正的 NSA 使用学习的 MLP）。
- 由压缩分支分数驱动的 top-k 块选择。
- 最后 `w` 个 token 上的 sliding-window attention。
- 门控组合。
- 与全 attention 比较的计算量打印。

### 步骤 1：将 token 压缩为块

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### 步骤 2：压缩分支 attention

对查询与压缩 key 运行 softmax attention。压缩分支分数同时作为 top-k 选择的信号。

### 步骤 3：top-k 块选择

选择 `k` 个最高分压缩块的索引。从这些块加载原始未压缩 token 并在其上运行 attention。

### 步骤 4：sliding-window attention

取最后 `w` 个 token 并对其运行标准 attention。

### 步骤 5：门控 + 组合

查询上的小 MLP 产出三个门控权重。最终输出是三个分支输出的加权和。

### 步骤 6：计算量统计

打印每个分支每查询关注的 key 数和总计。与 `N`（全 attention）比较。在 1024-token 合成上 `l = 32, k = 4, w = 128`，NSA 每查询看到 `32 + 128 + 128 = 288` 个 key 对比全 attention 的 1024——少 3.5x。

## 使用

NSA 正在 DeepSeek 自己的长上下文预训练 pipeline 中搭载。截至 2026 年 4 月公共推理栈的集成状态：

- **DeepSeek 内部**：原生，已发布权重使用 NSA 或其后继 DSA（Deepseek Sparse Attention）。
- **vLLM**：针对 DeepSeek-V3.x 权重的实验性 NSA 支持开发中。
- **SGLang**：NSA 基准已发布；生产路径跟随 vLLM。
- **llama.cpp / CPU**：不支持；kernel 分解的开销在 CPU 吞吐下不值得。

何时使用 NSA：

- 针对 64k+ 上下文的预训练或继续训练运行，有认真的计算预算。
- DeepSeek 自己的长上下文 checkpoint 的推理。权重是 NSA 原生的。

何时不用：

- 服务现有的 dense-attention 预训练模型。你无法在不继续训练的情况下改装 NSA。
- 上下文在 16k 以下。三分支开销主导节省。
- Batch-1 交互式聊天。延迟敏感的解码受益，但仅在长上下文时。

## 交付

本课产出 `outputs/skill-nsa-integrator.md`。给定长上下文预训练运行规范，它产出 NSA 集成计划：压缩块大小、top-k、sliding window、门控 MLP 宽度、kernel 选择，以及证明架构变更合理的特定长上下文评估。

## 练习

1. 在 1024-token 合成上运行 `code/main.py`。在三个预设上扫描 `(l, k, w)` 并打印计算量。识别在 needle-in-haystack 测试上保持 95% 召回的同时实现最低每查询 key 数的预设。

2. 用一个小的学习 MLP（2 层，hidden 32）替换均值池化压缩器。在信号是块平均值的合成任务上训练它。在留出数据上测量相对均值池化基线的困惑度差距。

3. 实现门控 MLP。它以查询为输入输出三个标量。展示门控行为合理：随机查询上接近均匀加权，当查询命中远处块时选择分支权重重。

4. 计算 NSA 启用的 70B 模型在 128k 上下文下的 KV cache 内存预算。KV 头为 8，head dim 128，BF16。与全 attention 和 MLA（Phase 10 · 14 展示了 MLA 的数字）比较。识别 NSA 细粒度分支 KV cache 等于全 attention 的序列长度。

5. 阅读 NSA 论文（arXiv:2502.11089）的 Section 4，用三句话解释为什么压缩分支的 attention 分数被复用于 top-k 选择而不是计算单独的路由分数。将答案与梯度流联系起来。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Compressed branch | "粗视图" | 在块平均 key 上的 attention，以每查询 O(N/l) 个 key 提供全局上下文 |
| Selected branch | "Top-k 块" | 在压缩分支分数最高的 `k` 个块的细粒度 attention |
| Sliding window | "局部上下文" | 最后 `W` 个 token 上的 attention，用于短程模式 |
| Native trainability | "带着稀疏性预训练" | 稀疏模式在预训练期间学习，不是推理时附加的 |
| Compression block size l | "粗视图的组大小" | 多少 token 合并为一个摘要；典型 32-64 |
| Top-k | "保留的块数" | 其未压缩 token 被读取的压缩块数；典型 16 |
| Sliding window W | "局部 attention 半径" | 典型 512；更短伤害局部连贯性，更长浪费计算 |
| Branch gate | "如何混合三者" | 加权三个分支贡献的逐位置 MLP 输出 |
| Hardware alignment | "Kernel 友好的稀疏性" | 选择的稀疏模式使实际 GPU kernel 实现理论加速 |
| DSA | "NSA 的后继" | Deepseek Sparse Attention，DeepSeek 谱系中 NSA 之后的架构 |

## 延伸阅读

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — 论文
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — NSA 针对的架构家族
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — 同期工作，MoE 风格的块 attention
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — sliding-window 起源
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — NSA 改进的推理时稀疏性基线
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — NSA kernel 在 64k 时击败的全 attention 基线
