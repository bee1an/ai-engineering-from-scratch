# 位置编码 — Sinusoidal、RoPE、ALiBi

> Attention 是置换不变的。"The cat sat on the mat" 和 "mat the on sat cat the" 在没有位置信号时产生相同输出。三种算法修复了这个问题——每种对"位置"的含义有不同的押注。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 7 · 02（Self-Attention）、Phase 7 · 03（Multi-Head Attention）
**时间：** 约 45 分钟

## 问题

Scaled dot-product attention 对顺序无感。注意力矩阵 `softmax(Q K^T / √d) V` 由成对相似度计算得出。打乱 `X` 的行，输出的行以相同方式打乱。注意力内部没有任何东西关心位置。

对于词袋模型来说这不是 bug。但对于语言、代码、音频、视频——任何顺序承载意义的东西——这是致命的。

修复方法是以某种方式把位置注入 embedding。三个时代的答案：

1. **绝对正弦编码**（Vaswani 2017）。把位置的 `sin/cos` 加到 embedding 上。简单、无需学习、在训练长度之外外推效果差。
2. **RoPE — Rotary Position Embeddings**（Su 2021）。按与位置成比例的角度旋转 Q 和 K 向量。直接在点积中编码*相对*位置。2026 年的主导方案。
3. **ALiBi — Attention with Linear Biases**（Press 2022）。完全跳过 embedding 技巧；根据距离向注意力分数添加逐 head 的线性惩罚。出色的长度外推能力。

截至 2026 年，基本上每个前沿开源模型都使用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少数长上下文模型使用 ALiBi 或其现代变体。绝对正弦编码已成为历史。

## 概念

![正弦绝对编码 vs RoPE 旋转 vs ALiBi 距离偏置](../assets/positional-encoding.svg)

### 绝对正弦编码

预计算一个固定矩阵 `PE`，形状为 `(max_len, d_model)`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

然后在注意力之前 `X' = X + PE[:N]`。每个维度是不同频率的正弦波。模型学会从相位模式中读取位置。超过 `max_len` 就失败：没有东西告诉模型在只见过位置 0–2047 时，位置 2048 会发生什么。

### RoPE

旋转 Q 和 K 向量（不是 embedding）。对于一对维度 `(2i, 2i+1)`：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

对 key 在位置 `pos_k` 应用相同的旋转。点积 `q'_m · k'_n` 变成仅关于 `(m - n)` 的函数。也就是说：**注意力分数只取决于相对距离**，尽管旋转是基于绝对位置的。漂亮的技巧。

扩展 RoPE：`base` 可以被缩放（NTK-aware、YaRN、LongRoPE）以在不重新训练的情况下外推到更长上下文。Llama 3 就是这样从 8K 扩展到 128K 上下文的。

### ALiBi

跳过 embedding 技巧。直接偏置注意力分数：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是 head 特定的斜率（如 `1 / 2^(8·h/H)`）。更近的 token 被提升；远处的 token 被惩罚。没有训练时间开销。论文表明长度外推优于正弦编码，在原始训练长度上匹配 RoPE。

### 2026 年该选什么

| 变体 | 外推能力 | 训练开销 | 使用者 |
|------|----------|----------|--------|
| Absolute sinusoidal | 差 | 免费 | 原始 transformer、早期 BERT |
| Learned absolute | 无 | 极小 | GPT-2、GPT-3 |
| RoPE | 配合缩放效果好 | 免费 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 优秀 | fine-tune 阶段 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 优秀 | 免费 | BLOOM、MPT、Baichuan |

RoPE 胜出是因为它无需改变架构就能嵌入注意力、编码相对位置，而且它的 `base` 超参数为长上下文微调提供了一个干净的旋钮。

## 动手构建

### 第 1 步：正弦编码

见 `code/main.py`。4 行计算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前把它加到 embedding 矩阵上。

### 第 2 步：RoPE 应用于 Q、K

RoPE 原地操作 Q 和 K。对每对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

关键：对位置 `m` 的 Q 和位置 `n` 的 K 应用相同函数。它们的点积在每个坐标对上获得一个 `cos((m-n)·θ_i)` 因子。注意力免费学到了相对位置。

### 第 3 步：ALiBi 斜率和偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

把 `bias[h]` 加到 head `h` 的 `(seq_len, seq_len)` 注意力分数矩阵上，然后 softmax。

### 第 4 步：验证 RoPE 的相对距离性质

取两个随机向量 `a, b`。按 `(pos_a, pos_b)` 旋转。再按 `(pos_a + k, pos_b + k)` 旋转。两个点积必须在浮点误差内匹配。这个性质就是 RoPE 的全部要点——它对绝对偏移不变，只有相对间距重要。

## 实际应用

PyTorch 2.5+ 在 `torch.nn.functional` 中提供了 RoPE 工具。大多数生产代码使用 `flash_attn` 或 `xformers`，其中 RoPE 在注意力 kernel 内部应用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的长上下文技巧：**

- **NTK-aware interpolation。** 从 4K 扩展到 16K+ 时，把 `base` 重新缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更智能的插值，在长上下文上保持注意力熵。Llama 3.1 128K 使用它。
- **LongRoPE。** Microsoft 2024 年的方法，使用进化搜索为每个维度选择缩放因子。Phi-3-Long 使用它。
- **Position interpolation + fine-tuning。** 只需按扩展因子缩小位置，然后微调 1–5B token。效果出奇地好。

## 交付产出

见 `outputs/skill-positional-encoding-picker.md`。该 skill 根据目标上下文长度、外推需求和训练预算，为新模型选择编码策略。

## 练习

1. **简单。** 把正弦 `PE` 矩阵画成热力图，`max_len=512, d=128`。确认"条纹随维度索引增大而变宽"的模式。
2. **中等。** 实现 NTK-aware RoPE 缩放。在长度 256 的序列上训练一个小 LM，然后在长度 1024 上测试，有缩放和没缩放分别测。测量困惑度。
3. **困难。** 在同一个注意力模块中实现 ALiBi 和 RoPE。在长度 512 的复制任务上训练一个 4 层 transformer。测试时外推到 2048。比较退化程度。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Positional encoding | "告诉注意力顺序" | 任何添加到 embedding 或注意力中编码位置的信号。 |
| Sinusoidal | "最初的那个" | 几何频率的 `sin/cos` 加到 embedding 上；不能外推。 |
| RoPE | "旋转位置编码" | 按位置相关角度旋转 Q、K；点积编码相对距离。 |
| ALiBi | "线性偏置技巧" | 向注意力分数添加 `-m·|i-j|`；不需要 embedding，外推能力强。 |
| base | "RoPE 的旋钮" | RoPE 中的频率缩放器；增大它可在推理时扩展上下文。 |
| NTK-aware | "一种 RoPE 缩放技巧" | 重新缩放 `base`，使高频维度在上下文扩展时不被压缩。 |
| YaRN | "高级版" | 逐维度的插值+外推，保持注意力熵。 |
| Extrapolation | "超出训练长度仍有效" | 位置方案能否在超过训练时见过的 `max_len` 后仍给出正确输出？ |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 原始正弦编码。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 论文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 最先进的 RoPE 缩放。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 长上下文论文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Microsoft 的方法，用于 Phi-3-Long。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 每种 RoPE 缩放方案的生产级实现（default、linear、dynamic、YaRN、LongRoPE、Llama-3）。
