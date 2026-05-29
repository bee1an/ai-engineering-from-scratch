# Gradient Checkpointing 与激活重计算

> 反向传播保留每个中间激活值。在 70B 参数和 128K 上下文下，每个 rank 有 3 TB 的激活值。Checkpointing 用 FLOPs 换内存：重新计算而非保存。问题是丢弃哪些段，答案不是"全部"。

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## 问题

训练 transformer 时，对每一层都要存储反向传播中需要求导的每个操作的输入：attention 输入、Q/K/V 投影、softmax 输出、FFN 输入、norm 输出和残差流。对于 hidden size `d`、序列长度 `L`、batch `B` 的层，量级为每层 `12 * B * L * d` 个浮点数。

对于 `d=8192, L=8192, B=1`，BF16 下每层 800 MB。64 层模型是 51 GB 的激活值——这还没乘以 microbatch size，没加 attention-softmax 中间值（每 head `L^2`），也没算 tensor-parallel 的部分副本。

两面账单：BF16 权重加优化器状态可能装进 80GB，但激活值把你推过去了。Gradient checkpointing（又名 activation recomputation）是标准修复方案。丢弃大部分激活值；在反向传播时重做前向来恢复它们。代价：额外 FLOPs。收益：内存按 checkpoint 段数与总层数的比例下降。

朴素做法大约多花 33% 的前向 FLOPs 每步。做得好——按 Korthikanti et al. 的"智能选择"做 selective checkpointing——你用不到 5% 的 FLOP 开销节省 5 倍内存。在 FP8 矩阵乘法、FSDP offload 和 expert-parallel MoE 的场景下这真的很重要：你既承受不起内存也承受不起浪费的计算。

## 概念

### 反向传播实际需要什么

`output = layer(input)`。反向传播需要 `grad_input` 和 `grad_params`。计算它们需要：

- `input`（用于计算线性层的 `grad_params = input.T @ grad_output`）
- 一些激活导数中间值（ReLU/GELU/softmax 的导数取决于激活值本身）

前向传播在 autograd 图中自动存储这些。每个 `tensor.retain_grad()` 和每个需要其输入的操作都保留引用。

### 朴素全量 Checkpointing

将网络分成 `N` 段。前向时只存储每段的*输入*。当反向传播需要中间值时，重新运行该段的前向传播来物化它们，然后求导。

示例：32 层 transformer 分成 32 段，每段 1 层。

- 内存：32 个层输入（小）vs 32 *（每层激活量）（大）。
- 额外计算：每段 1 次额外前向，即总共多 ~33% 前向 FLOPs（因为反向是前向的 2 倍，完整步骤变成 1 + 1 + 2 = 4 个单位而非 1 + 2 = 3）。

这是 Chen et al. 2016 的原始方案：每 `sqrt(L)` 层一个 checkpoint 以平衡内存和计算。对于 L=64，就是 8 个 checkpoint。

### Selective Checkpointing（Korthikanti 2022）

不是所有激活值代价相同。Attention softmax 输出是 `B*L*L*heads`，随序列长度*二次*增长。FFN 隐层激活是 `B*L*4d`，线性增长。对于长序列，softmax 主导。

Selective checkpointing 保留存储代价低的激活值（线性投影、残差），只重计算代价高的（attention）。你付出最小的 FLOPs 来重计算，但节省了 O(L^2) 的内存。

Megatron-Core 将此实现为"selective"激活重计算。用于大多数 2024+ 前沿训练运行。

### Offload

重计算的替代方案：在前向和反向之间将激活值传到 CPU RAM。需要 PCIe 带宽；当空闲带宽超过重物化代价时有益。混合策略很常见：checkpoint 一些层，offload 其他层。

FSDP2 将 offload 作为一等选项提供。当 GPU 受内存瓶颈但 CPU-GPU 传输有余量时，offload 表现出色。

### 重计算代价模型

每步 FLOPs，朴素 checkpointing 每 `k` 层（共 `L` 层）：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

使用 selective checkpointing 只重计算 attention kernel，而非整层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活量：`A`。`L` 层的总激活内存：`L * A`。

全量 checkpoint（段大小 1）：只存储 `L * input_volume`（标准 transformer 约 `L * 1/10 A`）。节省 ~`9 * L * A * 1/10`。

每 `k` 层 checkpoint：存储 `L/k * A` 加上活跃段内 `k-1` 层的量。

当 `k = sqrt(L)` 时，内存和重计算代价都按 `sqrt(L)` 缩放——对均匀代价层的最优权衡。

### 何时不做 Checkpoint

- 流水线阶段中已在执行的最内层。它们反正要完成。
- 如果第一层和最后一层主导该阶段的计算（在 transformer 中罕见）。
- 已经使用 FlashAttention 的 attention kernel——Flash 已经快速重计算了 softmax，额外的层级 checkpointing 在此之上增益很小。

### 实现模式

1. **函数包装器：** 将一段包装在 `torch.utils.checkpoint.checkpoint(fn, input)` 中。PyTorch 只存储 `input`，反向时重计算其他所有。

2. **基于装饰器：** 标记层为可 checkpoint 的；trainer 在配置时决定哪些段被包装。

3. **手动显式重计算：** 自己写反向传播，调用自定义 `recompute_forward` 用存储的输入复制前向。

三种方式给出相同的功能结果。包装器是标准惯用法。

### 与 TP / PP / FP8 的交互

- **Tensor parallel：** checkpoint 输入在重计算时必须被 gather 或重新 scatter；处理通信代价。
- **Pipeline parallel：** 典型模式是 checkpoint 每个流水线阶段的前向，使反序 microbatch 可以复用激活内存。
- **FP8 重计算：** 重计算期间更新的 amax 历史必须匹配原始前向的，否则 FP8 scale 会漂移。大多数框架快照 scale。

## 构建

### 步骤 1：带段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 步骤 2：需要所有激活值的朴素反向

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 步骤 3：每 k 层 Checkpoint 的内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 步骤 4：代价模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 步骤 5：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 步骤 6：最优段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 步骤 7：Selective Checkpoint 决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint`——PyTorch 中的标准包装器。包装一个函数；只存储输入，反向时重计算。
- **Megatron-Core activation recomputation**：支持 `selective`、`full` 和 `block` 模式。2024+ 前沿训练的标准配置。
- **FSDP2 offload**：在 FSDP2 中使用 `module.to_empty(device="cpu")` 配合 `offload_policy` 将激活值分片到 CPU 而非重计算。
- **DeepSpeed ZeRO-Offload**：优化器状态和激活值的 CPU offload，与 checkpointing 互补。

## 交付

本课产出 `outputs/prompt-activation-recompute-policy.md`——一个接收你的模型配置（layers、hidden、seq、batch）和可用 GPU 内存，输出逐层重计算策略（none / selective / full / offload）的 prompt。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（全量激活值）vs `model_forward_checkpointed` + `model_backward_checkpointed`（分段）。参数梯度必须在机器精度内相同。

2. 扫描段大小 `k` 从 1 到 `L`。绘制 FLOP 开销和内存。找到曲线的拐点。

3. 实现 selective checkpointing：存储 attention 模块输入但不存储其中间值。测量 32 层模型在 seq=8192 下相比全层 checkpointing 的 FLOP 开销。

4. 添加 offload。将段输入保存到模拟的"CPU 缓冲区"（一个单独的列表）。将"PCIe 带宽"测量为 bytes/time，找到 offload 和重计算之间的盈亏平衡点。

5. 对真实 PyTorch transformer 进行有无 `torch.utils.checkpoint` 的基准测试。测量内存（通过 `torch.cuda.max_memory_allocated`）和步时间。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| Gradient checkpointing | "重做前向来省内存" | 只存储段输入；反向时重计算中间值以获得梯度支持张量 |
| Activation recomputation | "和 checkpointing 一样" | 同一技术的 HPC 风格名称 |
| Segment size (k) | "每个 checkpoint 多少层" | 中间值被丢弃并一起重物化的层数 |
| Selective checkpointing | "Korthikanti 的技巧" | 只重计算存储代价高的激活值（attention softmax）；保留便宜的 |
| Full checkpointing | "朴素版本" | 重计算每段中每层的中间值 |
| Block checkpointing | "粗粒度" | Checkpoint 整个 transformer block；最大粒度 |
| FLOP overhead | "计算税" | 每步额外 FLOPs = (重计算 FLOPs) / (fwd + bwd FLOPs)；朴素 33%，selective 5% |
| Activation offload | "传到 CPU" | 在 forward->backward 之间将激活值移到 CPU RAM；重计算的替代方案 |
| sqrt-L rule | "经典最优" | 对均匀代价层，最优 checkpoint 间距是 sqrt(L) 层 |
| Attention-softmax volume | "O(L^2) 问题" | L^2 * heads * batch 个浮点数；在长上下文下主导激活内存 |

## 延伸阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 形式化 gradient checkpointing 的原始论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- selective activation recomputation 和形式化代价分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 通过反向模式重物化的替代常数内存方法
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 规模化的激活 offload
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- 标准 API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- selective、full 和 block 模式
