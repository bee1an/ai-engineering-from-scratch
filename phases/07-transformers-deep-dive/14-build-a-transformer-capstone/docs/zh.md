# 从零构建 Transformer — Capstone 项目

> 十三节课。一个模型。没有捷径。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## 问题

你读过了每篇论文。你实现了注意力、多头拆分、位置编码、encoder 和 decoder 块、BERT 和 GPT 损失、MoE、KV cache。现在让它们在一个真实任务上协同工作。

Capstone 项目：在字符级语言建模任务上端到端训练一个小型 decoder-only transformer。它读莎士比亚。它生成新的莎士比亚。它小到可以在笔记本电脑上 10 分钟内训练完。它足够正确，换上更大的数据集和更长的训练就能得到一个真正的语言模型。

这是本课程的"nanoGPT"。它不是原创的——Karpathy 2023 年的 nanoGPT 教程是每个学生至少写一次的参考实现。我们借用其形状，围绕我们已经覆盖的内容重新组织。

## 概念

![Transformer-from-scratch block diagram](../assets/capstone.svg)

架构标注：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 我们交付什么

- `GPTConfig` — 一个地方配置所有超参数。
- `MultiHeadAttention` — 因果、批处理，可选 Flash 风格路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` — 现代 FFN。
- `Block` — pre-norm，残差包裹的 attention + FFN。
- `GPT` — embedding、堆叠块、LM head、generate()。
- 训练循环：AdamW、cosine LR、梯度裁剪。
- 莎士比亚文本上的字符级 tokenizer。

### 我们不交付什么

- RoPE — 在 Lesson 04 中概念性实现。这里为简单起见使用可学习位置编码。练习要求你换成 RoPE。
- 生成时的 KV cache — 每个生成步重新计算整个前缀的注意力。更慢但更简单。练习要求你添加 KV cache。
- Flash Attention — PyTorch 2.0+ 在输入匹配时自动调度；我们使用 `F.scaled_dot_product_attention`。
- MoE — 每个块单个 FFN。你在 Lesson 11 中见过 MoE。

### 目标指标

在 Mac M2 笔记本上，一个 4 层、4 头、d_model=128 的 GPT 在 `tinyshakespeare.txt` 上训练 2,000 步：

- 训练损失从 ~4.2（随机）收敛到 ~1.5，约 6 分钟。
- 采样输出看起来像莎士比亚：古英语词汇、换行、像"ROMEO:"这样的专有名词出现。
- 验证损失（保留的最后 10% 文本）紧跟训练损失；在这个规模/预算下没有过拟合。

## 动手构建

本课使用 PyTorch。安装 `torch`（CPU 版本即可）。见 `code/main.py`。脚本处理：

- 如果缺失则下载 `tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符 tokenizer。
- 90/10 训练/验证拆分。
- 在支持的硬件上使用 bf16 autocast 的训练循环。
- 训练完成后采样。

### 第 1 步：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个唯一字符。极小词表。适合 4 字节 vocab_size。没有 BPE，没有 tokenizer 麻烦。

### 第 2 步：模型

见 `code/main.py`。块是 Lesson 05 的教科书版——pre-norm、RMSNorm、SwiGLU、因果 MHA。4/4/128 配置的参数量：~800K。

### 第 3 步：训练循环

获取随机批次的 256 长度 token 窗口。前向。Shift-by-one 交叉熵。反向。AdamW 步。记录。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 第 4 步：采样

给定一个 prompt，反复前向、从 top-p logits 采样、追加、继续。500 token 后停止。

### 第 5 步：阅读输出

2,000 步后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚。但像莎士比亚。对于 ~800K 参数和笔记本上 6 分钟来说，这是明确的胜利。

## 使用方式

这个 capstone 是一个参考架构。三个扩展可以把它变成真正可用的东西：

1. **换 tokenizer。** 使用 BPE（如 `tiktoken.get_encoding("cl100k_base")`）。词表大小从 65 跳到 ~50,000。模型容量需要相应扩大。
2. **在更大语料上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。单张 A100 上 10B token 对 125M 参数 GPT 约需 ~24 小时。
3. **添加 RoPE + KV cache + Flash Attention。** 下面的练习逐步引导你完成。

最终你会得到一个 125M 参数的 GPT，能生成流畅英文。不是前沿模型。但同样的代码路径——只是更大——就是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年训练研究 checkpoint 所用的。

## 交付产出

见 `outputs/skill-transformer-review.md`。该 skill 审查一个从零构建的 transformer 实现，检查前 13 节课的正确性。

## 练习

1. **简单。** 运行 `code/main.py`。验证训练模型最后一步的验证损失低于 2.0。将 `max_steps` 从 2,000 改为 5,000——验证损失是否继续改善？
2. **中等。** 用 RoPE 替换可学习位置编码。在 `MultiHeadAttention` 内部对 Q 和 K 应用旋转。训练并验证验证损失至少一样低。
3. **中等。** 在采样循环中实现 KV cache。分别用和不用 cache 生成 500 个 token。笔记本上实际时间应改善 5–20 倍。
4. **困难。** 给模型添加第二个头，预测下下一个 token（MTP — DeepSeek-V3 的 Multi-Token Prediction）。联合训练。有帮助吗？
5. **困难。** 将每个块的单个 FFN 替换为 4 专家 MoE。路由器 + top-2 路由。看看在匹配激活参数下验证损失如何变化。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy's tutorial repo" | Minimal decoder-only transformer training code, ~300 LOC; the canonical reference. |
| tinyshakespeare | "The standard toy corpus" | ~1.1 MB of text; every character-LM tutorial since 2015 uses it. |
| Tied embeddings | "Share input/output matrix" | LM head weight = transpose of token embedding matrix; saves parameters, improves quality. |
| bf16 autocast | "Training precision trick" | Run forward/back in bf16, keep optimizer state in fp32; standard since 2021. |
| Gradient clipping | "Stops spikes" | Cap global grad norm at 1.0; prevents training blowups. |
| Cosine LR schedule | "The 2020+ default" | LR ramps up linearly (warmup) then decays cosine-shaped to 10% of peak. |
| MFU | "Model FLOP Utilization" | Achieved FLOPs / theoretical peak; 40% dense, 30% MoE is strong in 2026. |
| Val loss | "Held-out loss" | Cross-entropy on data the model never saw; overfit detector. |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — the classic annotated implementation.
