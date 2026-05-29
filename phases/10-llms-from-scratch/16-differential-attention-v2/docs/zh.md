# Differential Attention (V2)

> Softmax attention 会在每个不匹配的 token 上分散少量概率。超过 100k token 时，这些噪声累积起来淹没信号。Differential Transformer（Ye et al., ICLR 2025）通过计算两个 softmax 的差来修复它，减去共享的噪声底。DIFF V2（Microsoft, 2026 年 1 月）是生产栈重写：匹配基线 Transformer 的解码延迟，无需自定义 kernel，兼容 FlashAttention。本课从 V1 到 V2 端到端讲解，附带一个你可以用 stdlib Python 运行的差分运算的工作玩具实现。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**Time:** ~60 minutes

## 学习目标

- 精确陈述为什么 softmax attention 有噪声底以及为什么它随上下文长度增长。
- 推导 differential attention 公式并解释为什么减法在保留信号的同时消除了共享噪声分量。
- 走过 V1 到 V2 的 diff：什么变快了、什么变简单了、什么变稳定了，以及为什么每个变化对生产预训练是必要的。
- 用纯 Python 从零实现 differential attention，并在合成的信号加噪声查询上经验验证噪声消除特性。

## 问题

标准 softmax attention 有一个数学特性，在规模化时变成运营头痛。对于查询 `q`，attention 权重是 `softmax(qK^T / sqrt(d))`。Softmax 永远不能产生精确的零——每个不匹配的 token 都获得一些正的质量。那个残余质量是噪声，它随上下文长度缩放。在 128k token 时，即使每个不匹配 token 只获得 0.001% 的概率，127,999 个加起来贡献了总量的约 12%。模型必须学会绕过一个随上下文增长的噪声底。

经验上这表现为 attention head 干扰：长上下文 RAG 中的幻觉引用、100k-token 检索任务上的 lost-in-the-middle 失败、以及 32k 之后 needle-in-haystack 基准上的微妙精度退化。Differential Transformer 论文（arXiv:2410.05258, ICLR 2025）测量了差距：DIFF Transformer 比同等大小基线达到更低困惑度、更高长上下文精度和更少幻觉。

DIFF V1 有三个问题使其无法进入前沿预训练 pipeline。它的 value cache 每个解码步骤必须加载两次，它需要破坏 FlashAttention 兼容性的自定义 CUDA kernel，它的 per-head RMSNorm 在 70B+ 规模的长期训练中不稳定。DIFF V2（Microsoft unilm blog, 2026 年 1 月 20 日）修复了所有三个。本课讲解两个版本，构建差分算子，并在玩具查询上基准测试噪声消除。

## 概念

### Softmax 的噪声底

对于查询 `q` 和 keys `K = [k_1, ..., k_N]`，attention 权重为：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有 `w_i` 是零。如果 `k_i` 与 `q` 完全无关，分数 `q . k_i` 不是 0——它围绕零波动，方差为 `||q||^2 / d`。softmax 归一化后，每个无关 token 仍然贡献 `O(1/N)` 到加权和。无关 token 的总贡献是 `O((N-1)/N) = O(1)`——不是一个小量。

模型想要的是类似 hard top-k 的东西：匹配 token 上高权重，其他地方接近零权重。Softmax 太平滑了，无法直接做到。

### 差分思想

将每个头的 Q 和 K 投影分成两半：Q = (Q_1, Q_2) 和 K = (K_1, K_2)。计算两个 attention map：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

减法消除了两个 map 共享的任何噪声分布。如果两个 map 在 127k 无关 token 上有大致均匀的权重（在随机初始化时它们会），那些就抵消了。信号——在少数真正相关 token 上的尖峰权重——只有在两个 map 中以相同幅度出现时才会抵消，而一旦模型训练后就不会。

`lambda` 是每头的可学习标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以是负的。`lambda_init` 默认为一个小正数如 0.8。

### 为什么这匹配有源噪声消除

想象两个嘈杂的麦克风录制同一个声音。两者都拾取说话者加上相关的背景噪声。一个减去另一个，共享噪声就消失了。声音存活是因为两个信号在相位或幅度上差异足够大以防止完全抵消。per-head `lambda` 精确学习这个平衡。

### V1 vs V2：diff

V1 保持参数量等于基线 Transformer。为了每头获得两个查询，它将 head 维度减半。这牺牲了 head 表达力，更痛苦的是——将每头的 value cache 减半。解码时每步必须加载 value cache 两次（每个 softmax 分支一次）。结果：尽管匹配参数量，解码比基线慢。

V2 将 query head 数量翻倍，保持 KV head 不变（从 up-projection 借参数）。Head 维度与基线相同。减法后，额外维度被投影回来以匹配基线 Transformer 的 O_W 投影。三件事同时发生：

1. 解码速度匹配基线（KV cache 只加载一次）。
2. FlashAttention 不变运行（无自定义 kernel）。
3. 解码时算术强度上升（每从 HBM 加载一字节做更多计算）。

V2 还移除了 V1 用来稳定减法的 per-head RMSNorm。在 70B 级预训练规模下，那个 RMSNorm 在训练后期不稳定。V2 用更简单的初始化方案替代，无需额外模块即可保持训练稳定。

### 何时使用

| 工作负载 | 收益 |
|----------|------|
| 长上下文 RAG (64k+) | 更干净的 attention map，更少幻觉引用 |
| Needle-in-haystack 基准 | 32k 之后显著精度提升 |
| 多文档 QA | 更少跨文档干扰 |
| 8k 代码补全 | 边际收益，不值得架构变更 |
| 短聊天 (< 4k) | 与基线基本无法区分 |

价值随上下文长度增长。在 4k token 时噪声底足够小，标准 attention 就行。在 128k 时它在伤害你。

### 与其他 2026 旋钮的兼容性

| 特性 | 与 DIFF V2 兼容？ |
|------|-----------------|
| GQA | 是（V2 增加 Q 头，不增加 KV 头） |
| MLA (DeepSeek) | 原则上是，无已发表论文组合它们 |
| MoE | 是（attention 独立于 MLP block） |
| RoPE | 是（不变） |
| YaRN / 长上下文扩展 | 是（正是 DIFF 帮助最大的地方） |
| FlashAttention | V2 中是（V1 中不是） |
| Speculative decoding | 是（attention 变化对 spec-decode 循环不可见） |

## 构建

`code/main.py` 用纯 Python 实现 differential attention。一个具有已知信号加噪声结构的玩具查询让你直接测量噪声消除比。

### 步骤 1：标准 softmax attention

Stdlib 矩阵运算：列表的列表，手动矩阵乘法，带数值稳定性减最大值的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 步骤 2：将 Q, K 分成两半

V1 风格：将 head 维度减半。V2 风格：保持 head 维度并将头数翻倍。玩具实现使用 V1 以便教学清晰——数学相同，只是簿记不同。

### 步骤 3：两个 softmax 分支 + 减法

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以是负的。这没问题——value cache 仍然处理有符号贡献。后续的 V 投影吸收符号。

### 步骤 4：噪声消除测量

构建长度 1024 的合成序列。在已知位置放置信号 token，其余填充噪声。计算 (a) 标准 softmax attention 在信号位置的权重和 (b) differential attention 权重。测量两者的信噪比。DIFF attention 可靠地产生 3x-10x 更高的信噪比，取决于两个分支被训练到多大差异。

### 步骤 5：V1 vs V2 参数核算

给定 config（hidden=4096, heads=32, d_head=128），打印：

- 基线 Transformer：Q, K, V 各大小 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q, K 各大小 `hidden * hidden`，V 大小 `hidden * hidden`（不变），head dim 内部减半。添加 per-head `lambda` 参数（O(heads * d_head)）。
- DIFF V2：Q 大小 `2 * hidden * hidden`，K 大小 `hidden * hidden`，V 大小 `hidden * hidden`。额外维度在 O_W 之前投影回来。添加相同的 `lambda` 参数。

玩具测量 V2 的额外参数成本（每个 attention block 大约 `hidden * hidden` 额外）并打印。

## 使用

截至 2026 年 4 月，DIFF V2 尚未在每个生产推理服务器中搭载，但 vLLM 和 SGLang 中的集成正在进行。同时该模式出现在：

- Microsoft 内部长上下文生产模型。
- 多个针对 256k+ 上下文的开放模型训练运行中的研究复现。
- 在交替层上组合 DIFF attention 与 sliding-window attention 的混合架构。

2026 年你会使用它的场景：

- 从零训练一个针对 64k+ 有效上下文的新模型。从一开始就添加 differential attention；之后重训很贵。
- 微调一个 lost-in-the-middle 失败主导评估的长上下文模型。Q 投影上的 LoRA 可以近似 DIFF 结构。

你不会使用它的场景：

- 你在服务一个具有稳定长上下文性能的预训练 dense 模型。重训成本很少能在现有权重上回本。
- 你的上下文总是在 16k 以下。噪声底可忽略。

## 交付

本课产出 `outputs/skill-diff-attention-integrator.md`。给定模型架构、目标上下文长度、幻觉画像和训练预算，它产出一个将 differential attention 添加到新预训练运行或 LoRA 微调的集成计划。

## 练习

1. 运行 `code/main.py`。验证报告的 differential attention 信噪比高于合成查询上的标准 softmax attention。变化噪声幅度并展示标准 attention 变得不可用的交叉点。

2. 计算从基线到 DIFF V1 和从基线到 DIFF V2 的参数量差异，对于 7B 级模型（hidden=4096, heads=32, d_head=128, 32 层）。展示哪些组件增加了参数，哪些保持不变。

3. 阅读 DIFF V1 论文（arXiv:2410.05258）的 Section 3 和 DIFF V2 Hugging Face blog 的 Section 2。用两句话解释为什么 V1 的 per-head RMSNorm 是必要的，以及为什么 V2 能在不导致训练发散的情况下移除它。

4. 实现一个消融：计算 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完全减法）的 differential attention。在合成查询上测量信噪比如何随扫描变化。识别最大化信噪比的 `lambda`。

5. 将玩具扩展到 GQA + DIFF V2。选择 8 个 KV 头和 32 个 Q 头。展示 KV cache 大小匹配具有相同 (8, 32) 配置的基线 GQA 模型。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Differential attention | "两个 softmax 相减" | 将 Q, K 分成两半，计算两个 softmax map，从第一个减去第二个（乘以 lambda），然后乘以 V |
| Noise floor | "softmax 的非零尾巴" | Softmax 在每个无关 token 上放置的 O(1/N) 权重，在长上下文中累加到 O(1) |
| lambda | "减法缩放" | Per-head 可学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以是负的 |
| DIFF V1 | "ICLR 2025 版本" | 原始 Differential Transformer；将 head dim 减半以保持参数量，需要自定义 kernel，解码更慢 |
| DIFF V2 | "2026 年 1 月修复" | 将 Q 头翻倍保持 KV 头不变；匹配基线解码速度并兼容 FlashAttention |
| Per-head RMSNorm | "V1 稳定器" | V1 在差分后应用的额外 norm；V2 移除它以防止训练后期不稳定 |
| Signal-to-noise ratio | "多少 attention 被浪费了" | 真实信号位置上的权重与无关位置上平均权重的比率 |
| Lost in the middle | "长上下文失败模式" | 检索精度在长上下文中间文档处下降的经验现象——DIFF attention 减少这个问题 |
| Arithmetic intensity | "每加载字节的 FLOPs" | V2 通过每次 KV 加载翻倍查询来增加解码时的比率；对内存受限解码很重要 |

## 延伸阅读

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 原始论文，噪声消除理论和长上下文消融
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — 生产栈重写，匹配基线解码，兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — 为什么减法恢复预训练 attention 结构的理论分析
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — 参数共享变体
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF 从中减去的基线 Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — DIFF attention 针对的长上下文基准
