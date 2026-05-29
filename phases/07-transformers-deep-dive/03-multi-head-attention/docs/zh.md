# Multi-Head Attention

> 一个注意力头一次学一种关系。八个头学八种。Head 是免费的。多拿几个。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 7 · 02（从零实现 Self-Attention）
**时间：** 约 75 分钟

## 问题

单个 self-attention head 计算一个注意力矩阵。这个矩阵捕获一种关系——通常是在训练信号上最小化 loss 的那种。如果你的数据中主谓一致、共指、长距离篇章关系和句法分块全部纠缠在一起，单个 head 会把它们糊成一个 softmax 分布，丢掉一半信号。

2017 年 Vaswani 论文的修复方案：并行运行多个注意力函数，每个都有自己的 Q、K、V 投影，然后拼接输出。每个 head 在维度为 `d_model / n_heads` 的更小子空间中操作。总参数量不变。表达能力提升。

Multi-head attention 是 2026 年每个 transformer 的标配。唯一的争论是关于*多少个* head，以及 key 和 value 是否共享投影（Grouped-Query Attention、Multi-Query Attention、Multi-head Latent Attention）。

## 概念

![Multi-head attention 拆分、注意、拼接](../assets/multi-head-attention.svg)

**拆分。** 取形状为 `(N, d_model)` 的 `X`。投影到 Q、K、V，每个形状为 `(N, d_model)`。Reshape 为 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。转置为 `(n_heads, N, d_head)`。

**并行注意。** 在每个 head 内运行 scaled dot-product attention。每个 head 产出 `(N, d_head)`。各 head 在 embedding 的不同子空间上操作，在注意力计算本身期间互不通信。

**拼接并投影。** 把 head 堆叠回 `(N, d_model)`，乘以学习到的输出矩阵 `W_o`，形状为 `(d_model, d_model)`。`W_o` 是 head 之间混合信息的地方。

**为什么有效。** 每个 head 可以专门化，而不需要与其他 head 竞争表示预算。2019–2024 年的探测研究显示了不同的 head 角色：位置 head、关注前一个 token 的 head、复制 head、命名实体 head、induction head（支撑 in-context learning 的核心）。

**2026 年的变体谱系：**

| 变体 | Q heads | K/V heads | 使用者 |
|------|---------|-----------|--------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA 是现代默认选择，因为它把 KV-cache 内存减少了 `N/G` 倍，同时几乎保持完整质量。MLA 更进一步，把 K/V 压缩到低秩潜空间，在计算注意力时再投影回来——花费 FLOPs，节省更多内存。

## 动手构建

### 第 1 步：从已有的单头注意力中拆分 head

取 Lesson 02 的 `SelfAttention`，用 split/concat 对包装它。见 `code/main.py` 的 numpy 实现；逻辑是：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape 加一次 transpose。没有循环。这正是 PyTorch 在 `nn.MultiheadAttention` 底层做的事。

### 第 2 步：每个 head 运行 scaled-dot-product attention

每个 head 得到自己的 Q、K、V 切片。注意力变成一个 batched matmul：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上 `Qh @ Kh.transpose(...)` 是一次 `bmm`。GPU 看到的是一个形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 的 batched matmul。增加 head 是免费的。

### 第 3 步：Grouped-Query Attention 变体

只有 key 和 value 的投影改变。Q 得到 `n_heads` 组；K 和 V 得到 `n_kv_heads < n_heads` 组，然后重复以匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

在推理时这节省了内存，因为 KV cache 中只有 `n_kv_heads` 份副本，而不是 `n_heads` 份。Llama 3 70B 使用 64 个 query head 和 8 个 KV head——cache 缩小 8 倍。

### 第 4 步：探测每个 head 学到了什么

用 4 个 head 在一个短句上运行 MHA。对每个 head，打印 `(N, N)` 注意力矩阵。你会看到不同的 head 即使在随机初始化下也能挑出不同的结构——这部分是信号，部分是子空间中的旋转对称性。

## 实际应用

在 PyTorch 中，一行版本：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**多少个 head？** 2026 年生产模型的经验法则：

| 模型规模 | d_model | n_heads | d_head |
|----------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` 几乎总是落在 64 或 128。它是一个 head 能"看到"多少的单位。低于 32，head 开始与缩放因子 `sqrt(d_head)` 冲突；高于 256，你就失去了"多个小专家"的好处。

## 交付产出

见 `outputs/skill-mha-configurator.md`。该 skill 根据参数预算、序列长度和部署目标，为新 transformer 推荐 head 数量、kv-head 数量和投影策略。

## 练习

1. **简单。** 取 `code/main.py` 中的 MHA，在 `d_model=64` 固定的情况下把 `n_heads` 从 1 改到 16。在合成复制任务上画出一个小单层模型的 loss。更多 head 是有帮助、到达平台还是有害？
2. **中等。** 实现 MQA（一个 KV head 在所有 query head 间共享）。测量相比完整 MHA 参数量减少了多少。计算在 N=2048 推理时 KV-cache 大小缩小了多少。
3. **困难。** 实现一个小型 Multi-head Latent Attention：把 K、V 压缩到秩 `r` 的潜变量，在 KV cache 中存储潜变量，在注意力计算时解压。在什么 `r` 值下 cache 内存降到完整 MHA 的 1/8 以下，同时质量保持在验证 ppl 的 1 bit 以内？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Head | "单个注意力电路" | 一个维度为 `d_head = d_model / n_heads` 的 Q/K/V 投影，有自己的注意力矩阵。 |
| d_head | "Head 维度" | 每个 head 的隐藏宽度；生产中几乎总是 64 或 128。 |
| Split / combine | "Reshape 技巧" | 注意力前后的 `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose。 |
| W_o | "输出投影" | 拼接 head 后应用的 `(d_model, d_model)` 矩阵；head 在这里混合。 |
| MQA | "一个 KV head" | Multi-Query Attention：单个共享的 K/V 投影。最小的 KV cache，有些质量损失。 |
| GQA | "Llama 2 以来的默认" | Grouped-Query Attention，`n_kv_heads < n_heads`；重复以匹配 Q。 |
| MLA | "DeepSeek 的技巧" | Multi-head Latent Attention：K、V 压缩到低秩潜变量，注意力时解压。 |
| Induction head | "in-context learning 背后的电路" | 一对 head，检测之前的出现并复制其后续内容。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 原始 multi-head 规范。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 论文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 如何在训练后把 MHA 转换为 GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA 以及为什么它在 cache 内存上打败 MHA/GQA。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 对 head 实际做什么的机制性研究。
