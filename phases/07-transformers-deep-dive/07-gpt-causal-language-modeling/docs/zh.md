# GPT — Causal Language Modeling

> BERT 看两边。GPT 只看过去。三角 mask 是现代 AI 中影响最深远的一行代码。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 7 · 02（Self-Attention）、Phase 7 · 05（完整 Transformer）、Phase 7 · 06（BERT）
**时间：** 约 75 分钟

## 问题

语言模型回答一个问题：给定前 `t-1` 个 token，token `t` 的概率分布是什么？在这个信号——next-token prediction——上训练，你就得到一个能逐 token 生成任意文本的模型。

要在整个序列上端到端并行训练，你需要每个位置的预测只依赖更早的位置。否则模型会通过偷看答案来作弊。

Causal mask 做到了这一点。它是一个上三角的 `-inf` 矩阵，在 softmax 之前加到注意力分数上。Softmax 之后，那些位置变成 0。每个位置只能关注自身和更早的位置。因为你对整个序列只应用一次，就能在一次前向传播中得到 N 个并行的 next-token 预测。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2024）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi——它们都是 decoder-only causal transformer，核心循环相同。只是更大、更好的数据、更好的 RLHF。

## 概念

![Causal mask 创建三角形注意力矩阵](../assets/causal-attention.svg)

### Mask

给定长度为 `N` 的序列，构建一个 `N × N` 矩阵：

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

在 softmax 之前把 `M` 加到原始注意力分数上。`exp(-inf) = 0`，所以被 mask 的位置贡献零权重。注意力矩阵的每一行是仅在之前位置上的概率分布。

实现代价：一次 `torch.tril()` 调用。计算时间：纳秒。对领域的影响：一切。

### 并行训练，串行推理

训练：对整个 `(N, d_model)` 序列做一次前向传播，计算 N 个交叉熵损失（每个位置一个），求和，反向传播。沿序列并行。这就是 GPT 训练能 scale 的原因——你在一次 GPU pass 中处理一个 batch 里的 1M token。

推理：你逐 token 生成。输入 `[t1, t2, t3]`，得到 `t4`。输入 `[t1, t2, t3, t4]`，得到 `t5`。输入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV cache（Lesson 12）保存 `t1…tn` 的隐藏状态，这样你不用每步重新计算。但推理时的串行深度 = 输出长度。这就是自回归税，也是每个 LLM 的延迟瓶颈所在。

### Loss — shift-by-one

给定 token `[t1, t2, t3, t4]`：

- 输入：`[t1, t2, t3]`
- 目标：`[t2, t3, t4]`

对每个位置 `i`，计算 `-log P(target_i | inputs[:i+1])`。求和。这就是整个序列的交叉熵。

你听说过的每个 transformer LM 都在这个 loss 上训练。预训练、微调、SFT——同一个 loss，不同的数据。

### 解码策略

训练之后，采样选择比人们想象的更重要。

| 方法 | 做什么 | 何时使用 |
|------|--------|----------|
| Greedy | 每步取 argmax | 确定性任务，代码补全 |
| Temperature | 把 logits 除以 T，采样 | 创意任务，T 越高多样性越大 |
| Top-k | 只从 top-k token 中采样 | 消除低概率尾部 |
| Top-p (nucleus) | 从累积概率 ≥ p 的最小集合中采样 | 2020+ 默认；适应分布形状 |
| Min-p | 保留 `p > min_p * max_p` 的 token | 2024+；比 top-p 更好地拒绝长尾 |
| Speculative decoding | 小模型提议 N 个 token，大模型验证 | 相同质量下 2–3× 延迟降低 |

2026 年，min-p + temperature 0.7 是开源模型的合理默认值。Speculative decoding 是任何生产推理栈的标配。

### "GPT 配方"为什么有效

1. **Decoder-only。** 没有 encoder 开销。每层一次 attention + FFN。
2. **Scaling。** 124M → 1.5B → 175B → 万亿。Chinchilla scaling laws（Lesson 13）告诉你如何花费算力。
3. **In-context learning。** 在 6B–13B 左右涌现。模型无需微调就能遵循 few-shot 示例。
4. **RLHF。** 在人类偏好上的后训练把原始预训练文本转变为对话助手。
5. **Pre-norm + RoPE + SwiGLU。** 大规模下的稳定训练。

核心架构自 GPT-2 以来没有太大变化。所有有趣的事情都发生在数据、规模和后训练上。

## 动手构建

### 第 1 步：causal mask

见 `code/main.py`。一行代码：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前加到注意力分数上。这就是整个机制。

### 第 2 步：2 层 GPT 风格模型

堆叠两个 decoder block（masked self-attention + FFN，没有 cross-attention）。加一个 token embedding、一个位置编码和一个 unembedding（与 token embedding 矩阵绑定——GPT-2 以来的标准技巧）。

### 第 3 步：端到端的 next-token prediction

在 20 token 的玩具词表上，在每个位置产生 logits。对 shift-by-one 目标计算交叉熵损失。不用梯度——这是前向传播的健全性检查。

### 第 4 步：采样

实现 greedy、temperature、top-k、top-p、min-p。在固定 prompt 上运行每种方法并比较输出。一个采样函数只有 10 行。

## 实际应用

PyTorch，2026 年写法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

底层，`generate()` 运行前向传播，取最后位置的 logits，采样下一个 token，追加，重复。每个生产 LLM 推理栈（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都实现相同的循环加上重度优化——batched prefill、continuous batching、KV cache paging、speculative decoding。

**GPT vs BERT，各一句话：** GPT 预测 `P(x_t | x_{<t})`。BERT 预测 `P(x_masked | x_unmasked)`。Loss 决定了模型能否生成。

## 交付产出

见 `outputs/skill-sampling-tuner.md`。该 skill 为新的生成任务选择采样参数，并在需要确定性解码时发出标记。

## 练习

1. **简单。** 运行 `code/main.py`，验证 softmax 后的 causal attention 矩阵是下三角的。抽查：第 3 行应该只在第 0–3 列有权重。
2. **中等。** 实现宽度为 4 的 beam search。在 10 个短 prompt 上比较 beam-4 和 greedy 的困惑度。Beam 总是赢吗？（提示：通常在翻译中赢，在开放式对话中不一定。）
3. **困难。** 实现 speculative decoding：用一个小型 2 层模型作为 draft，6 层模型作为 verifier。在 100 个长度 64 的补全上测量墙钟加速。确认输出与 verifier 的 greedy 匹配。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Causal mask | "三角形" | 加到注意力分数上的上三角 `-inf` 矩阵，使位置 `i` 只能看到位置 `≤ i`。 |
| Next-token prediction | "那个 loss" | 模型分布对每个位置真实下一个 token 的交叉熵。 |
| Autoregressive | "逐个生成" | 把输出反馈为输入；只在训练时并行，生成时不行。 |
| Logits | "softmax 前的分数" | LM head 在 softmax 之前的原始输出；采样在这上面进行。 |
| Temperature | "创意旋钮" | 把 logits 除以 T；T→0 = greedy，T→∞ = 均匀分布。 |
| Top-p | "Nucleus sampling" | 截断分布到累积概率 ≥p 的最小集合；从剩余中采样。 |
| Min-p | "比 top-p 更好" | 保留 `p ≥ min_p × max_p` 的 token；截断阈值适应分布的尖锐程度。 |
| Speculative decoding | "Draft + verify" | 小模型提议 N 个 token；大模型并行验证。 |
| Teacher forcing | "训练技巧" | 训练时输入真实的前一个 token，而不是模型的预测。每个 seq2seq LM 的标准做法。 |

## 延伸阅读

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 和 in-context learning。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — speculative decoding 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 典型 causal-LM 参考代码。
