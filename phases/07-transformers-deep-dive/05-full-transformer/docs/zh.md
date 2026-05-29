# 完整 Transformer — Encoder + Decoder

> Attention 是主角。其他一切——残差、归一化、前馈、cross-attention——是让你能堆叠深度的脚手架。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 7 · 02（Self-Attention）、Phase 7 · 03（Multi-Head Attention）、Phase 7 · 04（位置编码）
**时间：** 约 75 分钟

## 问题

单个注意力层是特征提取器，不是模型。每层一次 matmul 对语言来说容量不够。你需要深度——而深度在没有正确管道的情况下会崩溃。

2017 年 Vaswani 论文打包了六个设计决策，把一个注意力层变成了可堆叠的 block。此后的每个 transformer——encoder-only（BERT）、decoder-only（GPT）、encoder-decoder（T5）——都继承了相同的骨架。2026 年 block 已经被改进（RMSNorm、SwiGLU、pre-norm、RoPE），但骨架完全相同。

本课就是这个骨架。后续课程将其特化——06 用于 encoder，07 用于 decoder，08 用于 encoder-decoder。

## 概念

![Encoder 和 decoder block 内部结构，连线](../assets/full-transformer.svg)

### 六个组件

1. **Embedding + 位置信号。** Token → 向量。位置通过 RoPE（现代）或正弦波（经典）注入。
2. **Self-attention。** 每个位置关注其他所有位置。在 decoder 中带 mask。
3. **前馈网络（FFN）。** 逐位置的两层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比 4×。
4. **残差连接。** `x + sublayer(x)`。没有它，梯度在约 6 层后就消失了。
5. **Layer normalization。** `LayerNorm` 或 `RMSNorm`（现代）。稳定残差流。
6. **Cross-attention（仅 decoder）。** Query 来自 decoder，key 和 value 来自 encoder 输出。

### Encoder block（用于 BERT、T5 encoder）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Encoder 是双向的。没有 masking。所有位置看到所有位置。

### Decoder block（用于 GPT、T5 decoder）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Decoder 每个 block 有三个子层。中间那个——cross-attention——是信息从 encoder 流向 decoder 的唯一通道。在纯 decoder-only 架构（GPT）中，cross-attention 被省略，只剩 masked self-attention + FFN。

### Pre-norm vs post-norm

原始论文：`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。Post-norm 在 2019 年左右失宠——没有仔细的 warmup 很难深度训练。Pre-norm（`LN` 在子层*之前*）是 2026 年的默认：Llama、Qwen、GPT-3+、Mistral 都用它。

### 2026 年的现代化 block

Vaswani 2017 发布时用的是 LayerNorm + ReLU。现代栈替换了两者。生产 block 实际长这样：

| 组件 | 2017 | 2026 |
|------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6×（SwiGLU 用三个矩阵，总参数量匹配） |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm 去掉了 LayerNorm 的均值中心化（少一次减法），节省计算且经验上至少同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 论文中一致地比 ReLU/GELU FFN 好约 0.5 点 ppl。

### 参数量

对于一个 `d_model = d`、FFN 扩展比 `r` 的 block：

- MHA：`4 · d²`（Q、K、V、O 投影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- Norms：可忽略

在 `d = 4096, r = 2.6, layers = 32`（大致是 Llama 3 8B）时，总计：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`（加上 embedding 和 head）。与公布数字吻合。

## 动手构建

### 第 1 步：基础组件

使用 Lesson 03 的小型 `Matrix` 类（复制到本文件以保持独立性）：

- `layer_norm(x, eps=1e-5)` — 减均值，除以标准差。
- `rms_norm(x, eps=1e-6)` — 除以 RMS。不减均值。
- `gelu(x)` 和 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

见 `code/main.py` 的完整连线。

### 第 2 步：连接 2 层 encoder 和 2 层 decoder

堆叠它们。把 encoder 输出传入每个 decoder 的 cross-attention。在输出投影前加一个最终 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 第 3 步：在玩具示例上运行前向传播

输入 6 个 token 的源序列和 5 个 token 的目标序列。验证输出形状是 `(5, vocab)`。不训练——本课关注的是架构，不是 loss。

### 第 4 步：换入 RMSNorm + SwiGLU

用 RMSNorm 和 SwiGLU 替换 LayerNorm 和 ReLU-FFN。确认形状仍然匹配。这就是 2026 年的现代化，只需一次函数替换。

## 实际应用

PyTorch/TF 参考实现：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但 2026 年大多数生产代码自己写 block，因为：

- Flash Attention 在注意力内部调用，不通过 `nn.MultiheadAttention`。
- GQA / MLA 不在标准库参考中。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 的默认值。

HF `transformers` 有干净的参考 block 值得阅读：`modeling_llama.py` 是 2026 年典型的 decoder-only block。约 500 行，值得通读一遍。

**Encoder vs decoder vs encoder-decoder——何时选择：**

| 需求 | 选择 | 示例 |
|------|------|------|
| 分类、embedding、文本 QA | Encoder-only | BERT, DeBERTa, ModernBERT |
| 文本生成、对话、代码、推理 | Decoder-only | GPT, Llama, Claude, Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | Encoder-decoder | T5, BART, Whisper |

Decoder-only 赢得了语言领域，因为它 scale 最干净，同时处理理解和生成。Encoder-decoder 在输入有明确"源序列"身份时仍然最好（翻译、语音识别、结构化任务）。

## 交付产出

见 `outputs/skill-transformer-block-reviewer.md`。该 skill 对照 2026 年默认值审查新的 transformer block 实现，标记缺失的部分（pre-norm、RoPE、RMSNorm、GQA、FFN 扩展比）。

## 练习

1. **简单。** 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 下计算你的 encoder_block 的参数量。通过实现 block 并使用 `sum(p.numel() for p in block.parameters())` 验证。
2. **中等。** 从 post-norm 切换到 pre-norm。初始化两者，在随机输入上测量 12 层堆叠后的激活范数。Post-norm 的激活应该爆炸；pre-norm 的应该保持有界。
3. **困难。** 在玩具复制任务（反转复制 `x`）上实现 4 层 encoder-decoder。训练 100 步。报告 loss。换入 RMSNorm + SwiGLU + RoPE——loss 是否下降？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Block | "一个 transformer 层" | norm + attention + norm + FFN 的堆叠，用残差连接包裹。 |
| Residual | "跳跃连接" | `x + f(x)` 输出；使梯度能流过深层堆叠。 |
| Pre-norm | "先归一化，不是后归一化" | 现代做法：`x + sublayer(LN(x))`。无需 warmup 体操就能训练更深。 |
| RMSNorm | "没有均值的 LayerNorm" | 除以 RMS；少一个操作，经验稳定性相同。 |
| SwiGLU | "所有人都换过去的 FFN" | `Swish(W1 x) ⊙ W3 x → W2`。在 LM ppl 上打败 ReLU/GELU。 |
| Cross-attention | "decoder 如何看到 encoder" | Q 来自 decoder、K/V 来自 encoder 输出的 MHA。 |
| FFN expansion | "中间 MLP 有多宽" | hidden-size 与 d_model 的比率，通常 4（LayerNorm）或 2.6（SwiGLU）。 |
| Bias-free | "去掉 +b 项" | 现代栈在线性层中省略偏置；ppl 略有改善，模型更小。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 block 规范。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 为什么 pre-norm 在深层上优于 post-norm。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 2026 年典型的 decoder-only block。
