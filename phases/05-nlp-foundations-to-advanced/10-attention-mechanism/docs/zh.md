# 注意力机制 — 突破性进展

> Decoder 不再眯着眼看压缩摘要，而是开始查看整个源序列。此后的一切都是 attention 加工程。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 5 · 09（序列到序列模型）
**时间：** 约 45 分钟

## 问题

第 09 课以一个可量化的失败结束。在玩具复制任务上训练的 GRU encoder-decoder 从长度 5 的 89% 准确率下降到长度 80 时接近随机。原因是结构性的，不是训练 bug：encoder 获取的每一比特信息都必须塞进一个定长隐藏状态，而 decoder 永远看不到其他东西。

Bahdanau、Cho 和 Bengio 在 2014 年发表了一个三行修复。不再只给 decoder 最终 encoder 状态，而是保留每个 encoder 状态。在每个 decoder 步骤，计算 encoder 状态的加权平均，权重表示"decoder 现在需要多少关注 encoder 位置 `i`？"这个加权平均就是上下文，它在每个 decoder 步骤都变化。

这就是全部想法。Transformer 扩展了它。Self-attention 将其应用于单个序列。Multi-head attention 并行运行它。但 2014 年的版本已经打破了瓶颈，一旦你有了它，转向 transformer 就是工程问题，不是概念问题。

## 概念

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

在每个 decoder 步骤 `t`：

1. 使用前一个 decoder 隐藏状态 `s_{t-1}` 作为 **query**。
2. 对每个 encoder 隐藏状态 `h_1, ..., h_T` 打分。每个 encoder 位置一个标量。
3. 对分数做 softmax 得到注意力权重 `α_{t,1}, ..., α_{t,T}`，和为 1。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`。Encoder 状态的加权平均。
5. Decoder 接收 `c_t` 加上前一个输出 token，产生下一个 token。

加权平均是关键。当 decoder 需要将 "Je" 翻译为 "I" 时，它给 "Je" 上的 encoder 状态高权重，其他低权重。当它需要 "not" 时，它给 "pas" 高权重。上下文向量在每步重塑。

## 形状（每个人第一次都会搞错的地方）

这是每个 attention 实现第一次出错的地方。慢慢读。

| 东西 | 形状 | 备注 |
|------|------|------|
| Encoder 隐藏状态 `H` | `(T_enc, d_h)` | 如果是 BiLSTM，`d_h = 2 * d_hidden` |
| Decoder 隐藏状态 `s_{t-1}` | `(d_s,)` | 一个向量 |
| 注意力分数 `e_{t,i}` | 标量 | 每个 encoder 位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 对所有 `i` softmax 之后 |
| 上下文向量 `c_t` | `(d_h,)` | 与 encoder 状态形状相同 |

**Bahdanau（加法）分数。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 形状 `(d_s,)`，`h_i` 形状 `(d_h,)`。
- `W_a` 形状 `(d_attn, d_s)`。`U_a` 形状 `(d_attn, d_h)`。
- 它们在 tanh 内的和形状为 `(d_attn,)`。
- `v_α` 形状 `(d_attn,)`。与 `v_α` 的内积坍缩为标量。**这就是 `v_α` 的作用。** 不是魔法。它是将 attention 维度向量转为标量分数的投影。

**Luong（乘法）分数。** 三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。硬约束。如果 encoder 是双向的就跳过。
- `general`：`e_{t,i} = s_t^T * W * h_i`，`W` 形状 `(d_s, d_h)`。移除等维约束。
- `concat`：本质上是 Bahdanau 形式。很少使用因为前两个更便宜。

**一个值得说明的 Bahdanau / Luong 陷阱。** Bahdanau 使用 `s_{t-1}`（生成当前词*之前*的 decoder 状态）。Luong 使用 `s_t`（*之后*的状态）。混淆它们会产生微妙错误的梯度，极难调试。选一篇论文并坚持它的惯例。

## 动手构建

### 第 1 步：加法（Bahdanau）attention

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上面的表检查你的形状。`encoder_states` 形状 `(T_enc, d_h)`。`projected_enc` 形状 `(T_enc, d_attn)`。`projected_dec` 形状 `(d_attn,)` 并广播。`combined` 形状 `(T_enc, d_attn)`。`scores` 形状 `(T_enc,)`。`weights` 形状 `(T_enc,)`。`context` 形状 `(d_h,)`。搞定。

### 第 2 步：Luong dot 和 general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

各三行。这就是为什么 Luong 的论文成功了。大多数任务上相同准确率，代码少得多。

### 第 3 步：一个数值示例

给定三个 encoder 状态（大致对应 "cat"、"sat"、"mat"）和一个与第一个最对齐的 decoder 状态，注意力分布集中在位置 0。如果 decoder 状态移向与最后一个对齐，注意力移到位置 2。上下文向量跟踪变化。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行赢了。然后把 decoder 状态移向第三个 encoder 状态，观察权重变化。就是这样。Attention 就是显式对齐。

### 第 4 步：为什么这是通往 transformer 的桥梁

将上面的语言翻译为 Q/K/V：

- **Query** = decoder 状态 `s_{t-1}`
- **Key** = encoder 状态（我们打分的对象）
- **Value** = encoder 状态（我们加权求和的对象）

在经典 attention 中，key 和 value 是同一个东西。Self-attention 将它们分开：你可以让一个序列对自身查询，用不同的学习投影做 K 和 V。Multi-head attention 用不同的学习投影并行运行。Transformer 将整个阶段堆叠多次并丢弃 RNN。

数学是一样的。形状是一样的。从 Bahdanau attention 到 scaled dot-product attention 的教学跳跃主要是符号变化。

## 使用现成工具

PyTorch 和 TensorFlow 直接提供 attention。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 transformer attention 层。Query 批次 5 个位置，key/value 批次 10 个位置，各 128 维，8 个头。`output` 是新的上下文增强 query。`weights` 是你可以可视化的 5x10 对齐矩阵。

### 经典 attention 仍然重要的场景

- 教学。单头、单层、基于 RNN 的版本让每个概念可见。
- transformer 放不下的设备端序列任务。
- 2014-2017 年的任何论文。不了解 Bahdanau 的惯例你会读错它。
- 机器翻译中的细粒度对齐分析。原始注意力权重即使在 transformer 模型上也是可解释性工具，读懂它们需要知道它们是什么。

### 注意力权重作为解释的陷阱

注意力权重看起来可解释。它们是跨位置和为一的权重；你可以画出来；高意味着"看了这个"。审稿人喜欢它们。

它们没有看起来那么可解释。Jain and Wallace (2019) 表明注意力分布可以被置换和替换为任意替代方案而不改变某些任务的模型预测。永远不要在没有消融或反事实检查的情况下将注意力权重作为推理的证据报告。

## 交付

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习

1. **简单。** 实现 `softmax` masking 使 encoder 中的 padding token 获得零注意力权重。在变长序列的批次上测试。
2. **中等。** 给 Luong `general` 形式添加 multi-head attention。将 `d_h` 分成 `n_heads` 组，每个头运行 attention，拼接。验证单头情况与你之前的实现匹配。
3. **困难。** 在第 09 课的玩具复制任务上训练带 Bahdanau attention 的 GRU encoder-decoder。绘制准确率 vs 序列长度。与无 attention 基线对比。你应该看到差距随长度增大而扩大，确认 attention 解除了瓶颈。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Attention | 看东西 | 值序列的加权平均，权重由 query-key 相似度计算。 |
| Query, Key, Value | QKV | 三个投影：Q 提问，K 是匹配对象，V 是返回内容。 |
| 加法 attention | Bahdanau | 前馈分数：`v^T tanh(W q + U k)`。 |
| 乘法 attention | Luong dot / general | 分数是 `q^T k` 或 `q^T W k`。更便宜，大多数任务上相同准确率。 |
| 对齐矩阵 | 那张漂亮的图 | 注意力权重作为 `(T_dec, T_enc)` 网格。读它看模型关注了什么。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 那篇论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三种分数变体及其比较。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 可解释性警告。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 可运行的 PyTorch 演练。
