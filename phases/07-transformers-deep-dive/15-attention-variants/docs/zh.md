# 注意力变体 — 滑动窗口、稀疏、差分

> 全注意力是一个圆。每个 token 看到每个 token，显存为此买单。四种变体弯曲圆的形状，收回一半成本。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**Time:** ~60 minutes

## 问题

全注意力在序列长度上花费 `O(N²)` 内存和 `O(N²)` 计算。对于 128K 上下文的 Llama 3 70B，这是每层 160 亿个注意力条目，乘以 80 层。Flash Attention（Lesson 12）隐藏了 `O(N²)` 的激活内存，但没有改变算术成本——每个 token 仍然注意到每个其他 token。

三类变体改变了注意力矩阵本身的拓扑：

1. **滑动窗口注意力（SWA）。** 每个 token 只注意固定窗口内的邻居，而非完整前缀。内存和计算降到 `O(N · W)`，其中 `W` 是窗口大小。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long。
2. **稀疏/块注意力。** 只有选定的 `(i, j)` 对被评分；其余被强制为零权重。Longformer、BigBird、OpenAI sparse transformer。
3. **差分注意力。** 用独立的 Q/K 投影计算两个注意力图，然后相减。消除将权重泄漏到前几个 token 的"注意力汇聚"问题。微软的 DIFF Transformer（2024）。

这些共存。2026 年的前沿模型通常混合使用：大多数层是 SWA-1024，每五层一个全局全注意力，还有少数差分 head 用于清理检索。Gemma 3 的 5:1 SWA-to-global 比例是当前教科书默认。

## 概念

### 滑动窗口注意力（SWA）

位置 `i` 的每个 query 只注意 `[i - W, i]`（因果 SWA）或 `[i - W/2, i + W/2]`（双向）中的位置。窗口外的 token 在分数矩阵中得到 `-inf`。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

对于 `N = 8192` 和 `W = 1024`，分数矩阵期望有 1024 × 8192 个非零行——8 倍缩减。

**KV cache 随 SWA 缩小。** 每层只需保留最后 `W` 个 token 的 K 和 V。对于 Gemma-3 风格的配置（1024 窗口，128K 上下文），KV cache 缩小 128 倍。

**质量代价。** 纯 SWA transformer 在长程检索上表现不佳。修复方法：交错 SWA 层和全注意力层。Gemma 3 使用 5:1 SWA:global。Mistral 7B 使用因果 SWA 栈，信息通过重叠窗口"向前流动"——每层将有效感受野扩展 `W`，`L` 层后模型可以回看 `L × W` 个 token。

### 稀疏/块注意力

预先选定一个 `N × N` 稀疏模式。三种经典形状：

- **局部 + 跨步（OpenAI sparse transformer）。** 注意最后 `W` 个 token 加之前每 `stride` 个 token。以 `O(N · sqrt(N))` 计算同时捕获局部和长程信息。
- **Longformer / BigBird。** 局部窗口 + 少量全局 token（如 `[CLS]`）注意所有人且被所有人注意 + 随机稀疏连接。经验上在匹配质量下 2 倍上下文。
- **Native Sparse Attention（DeepSeek, 2025）。** 学习哪些 `(Q, K)` 块重要；在 kernel 层面跳过零块。兼容 FlashAttention。

稀疏注意力是一个 kernel 工程故事。数学很简单（mask 分数矩阵）；收益来自永远不把零条目加载到 SRAM。FlashAttention-3 和 2026 年的 FlexAttention API 让自定义稀疏模式在 PyTorch 中成为一等公民。

### 差分注意力（DIFF Transformer, 2024）

常规注意力有"注意力汇聚"问题：softmax 强制每行求和为 1，所以不想注意任何特定内容的 token 会把权重倾倒到第一个 token（或前几个）。这窃取了本应分配给真实内容的容量。

差分注意力通过计算**两个**注意力图并相减来修复：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是一个可学习标量（通常 0.5–0.8）。A1 捕获真实内容权重；A2 捕获汇聚。相减消除汇聚，将权重重新分配给相关 token。

报告结果（Microsoft 2024）：困惑度降低 5–10%，在相同训练长度下有效上下文长 1.5–2 倍，needle-in-haystack 检索更锐利。

### 变体对比

| Variant | Compute | KV cache | Quality vs full | Production use |
|---------|---------|----------|-----------------|----------------|
| Full attention | O(N²) | O(N) per layer | baseline | every model's default layer |
| SWA (window 1024) | O(N·W) | O(W) per layer | -0.1 ppl, good with global layers | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | O(N·√N) | mixed | similar to SWA | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | O(N) approx | mixed | matches full at 2× context | early long-context BERT |
| Native Sparse (DeepSeek-V3.2) | O(N · active fraction) | O(N) | within 0.05 ppl | DeepSeek-V3.2, 2025 |
| Differential | O(2·N²) | O(2N) | -5 to -10% ppl | DIFF Transformer, early 2026 models |

## 动手构建

见 `code/main.py`。我们实现一个因果 mask 比较器，在一个玩具序列上并排展示全注意力、SWA、局部+跨步和差分注意力。

### 第 1 步：全因果 mask（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Lesson 07 的基线。下三角；对角线以上为零权重。

### 第 2 步：滑动窗口因果 mask

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数——`window`。当 `window >= n` 时，恢复全因果注意力。当 `window = 1` 时，每个 token 只注意自己。

### 第 3 步：局部 + 跨步稀疏 mask

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集局部窗口加回到序列开头的每 `stride` 个 token。感受野随额外层数以对数步增长。

### 第 4 步：差分注意力

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力传递，用可学习混合系数相减。代码中我们对比单注意力 vs 差分注意力的注意力汇聚热图，观察汇聚消失。

### 第 5 步：KV cache 大小

打印 `N = 131072` 时每层的 cache 大小。SWA 和稀疏变体缩小 10–100 倍。差分翻倍。有意识地支付你的显存账单。

## 使用方式

2026 年生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 的 FlexAttention 接受一个 mask 函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会编译成自定义 Triton kernel。对常见模式在 FlashAttention-3 速度的 10% 以内，且 mask 函数是一个 Python callable。

**什么时候选哪个：**

- **纯全注意力** — 每层到 ~16K 上下文，或当检索质量至关重要时。
- **SWA + 全局混合** — 长上下文（>32K），训练和推理受显存限制。32K 以上的 2026 年默认。
- **稀疏块注意力** — 自定义 kernel，自定义模式。保留给专用工作负载（检索、音频）。
- **差分注意力** — 任何注意力汇聚污染有害的工作负载（长上下文 RAG、needle-in-haystack）。

## 交付产出

见 `outputs/skill-attention-variant-picker.md`。该 skill 根据目标上下文长度、检索需求和训练/推理算力配置，为新模型选择注意力拓扑。

## 练习

1. **简单。** 运行 `code/main.py`。验证 `window=4` 的 SWA 将每行最后 4 个 token 之外的所有内容置零。验证 `window=n` 与全因果注意力 bit-identical。
2. **中等。** 在 Lesson 07 capstone 之上实现 `window=1024` 的因果 SWA。在 tinyshakespeare 上训练 1,000 步。验证损失相比全注意力回退多少？峰值内存下降多少？
3. **困难。** 在 capstone 模型中实现 Gemma-3 风格的 5:1 层混合（5 SWA，1 全局）。在匹配参数下对比纯 SWA 和纯全局基线的损失、内存和生成质量。
4. **困难。** 实现每头可学习 `λ` 的差分注意力。在合成检索任务（1 根针，2,000 个干扰项）上训练。在匹配参数下测量检索准确率 vs 单注意力基线。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Sliding window attention (SWA) | "Local attention" | Each query attends to its last `W` tokens; KV cache shrinks to `O(W)`. |
| Effective receptive field | "How far back the model sees" | In an `L`-layer SWA stack with window `W`, up to `L × W` tokens. |
| Longformer / BigBird | "Local + global + random" | Sparse patterns with a few always-attending global tokens; early long-context approach. |
| Native Sparse Attention | "DeepSeek's kernel trick" | Learn block-level sparsity; skip zero blocks at the kernel level while keeping quality. |
| Differential attention | "Two maps, one subtracts" | DIFF Transformer: subtract a learned `λ` times a second attention map from the first to cancel attention sinks. |
| Attention sink | "Weight bleeds to token 0" | Softmax normalization forces rows to sum to 1; uninformative queries dump weight on position 0. |
| FlexAttention | "Mask-as-Python" | PyTorch 2.5+ API that compiles arbitrary mask functions into FlashAttention-shape kernels. |
| Layer type mix | "5:1 SWA-to-global" | Interleave sparse and full attention layers in a stack to keep quality at lower memory. |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — the canonical sliding-window + global-token paper.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — local + global + random.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI's local+strided pattern.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — the 1:1 SWA:global mix.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — the 5:1 mix with window=1024 that's now the textbook default.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer paper.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2's learned-sparsity attention.
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — API reference for the mask-as-callable pattern in Use It.
