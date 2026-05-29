# Speculative Decoding — 草稿、验证、重复

> 自回归解码是串行的。每个 token 等待前一个。Speculative decoding 打破这个链条：一个便宜模型草拟 N 个 token，昂贵模型在一次前向传播中验证所有 N 个。当草稿正确时，你用一次大模型前向换来了 N 次生成。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time:** ~60 minutes

## 问题

一个 70B LLM 在 H100 上采样一个 token 需要 ~30 ms。一个 3B draft 模型需要 ~3 ms。如果我们让 3B 向前草拟 5 个 token，然后让 70B *一次*验证所有 5 个，总计 `5×3 + 30 = 45 ms` 最多得到 5 个被接受的 token——而直接生成是 `5×30 = 150 ms`。这就是 speculative decoding 的完整卖点：用少量额外 GPU 显存（draft 模型）换取 2–4 倍更低的解码延迟。

这个技巧必须保持分布不变。Speculative sampling 由 Leviathan et al.（2023）和 Chen et al. 同时提出，保证输出序列与大模型独立生成的分布**完全相同**。没有质量折衷。只是更快。

2026 年推理中有四个 draft-verifier 对的家族占主导：

1. **Vanilla speculative（Leviathan 2023）。** 独立 draft 模型（如 Llama 3 1B）+ verifier（如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** verifier 上的多个解码头并行预测位置 `t+1..t+k`。没有独立 draft 模型。
3. **EAGLE 家族（Li 2024, 2025）。** 轻量 draft 复用 verifier 的隐藏状态；接受率比 vanilla 更高；典型 3–4 倍。
4. **Lookahead decoding（Fu 2024）。** Jacobi 迭代；完全不需要 draft 模型。自推测。小众但无依赖。

2026 年每个生产推理栈都默认支持 speculative decoding。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 都至少支持 vanilla + EAGLE-2。

## 概念

### 核心算法

给定 verifier `M_q` 和更便宜的 draft `M_p`：

1. 设 `x_1..x_k` 为已解码的前缀。
2. **草拟**：用 `M_p` 自回归提议 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，draft 概率为 `p_1..p_N`。
3. **并行验证**：对 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 运行一次 `M_q`，得到位置 `k+1..k+N+1` 的 verifier 概率 `q_1..q_{N+1}`。
4. **从左到右接受/拒绝每个 draft token**：对每个 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受。
5. 在位置 `j` 首次拒绝时：从"残差"分布 `(q_j - p_j)_+` 归一化后采样 `t_j`。`j` 之后的所有 draft 被丢弃。
6. 如果所有 `N` 个都被接受：从 `q_{N+1}` 采样一个额外 token `t_{N+1}`（免费奖励 token）。

残差分布技巧是保持输出分布与 `M_q` 从头采样完全相同的数学洞察。

### 什么决定加速

设 `α` = 每个 draft token 的期望接受率。设 `c` = draft 与 verifier 的成本比。每步：

- 朴素生成每个 token 做 1 次大模型调用。
- Speculative 当 `α` 高时，每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token 做 1 次大模型调用。

`α = 0.75` 和 `N = 5` 时的典型经验法则：大模型调用减少 3 倍。Draft 成本是 5 倍便宜。总实际时间下降 ~2.5 倍。

**α 取决于：**

- Draft 对 verifier 的近似程度。同家族/同训练数据显著提升 α。
- 解码策略。贪心 draft 对贪心 verifier：高 α。温度采样：更难匹配；接受率下降。
- 任务类型。代码和结构化输出接受更多（可预测）；自由创意写作接受更少。

### Medusa — 无 draft 模型的草拟

Medusa 用 verifier 上的额外输出头替代 draft 模型。在位置 `t`：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个头输出自己的 logits。推理时从每个头采样得到候选序列，然后用一次前向传播通过 tree-attention 方案验证所有候选延续。

优点：不需要第二个模型。缺点：增加可训练参数；需要一个监督微调阶段（~1B token）；接受率比用好 draft 的 vanilla speculative 略低。

### EAGLE — 通过复用隐藏状态获得更好的 draft

EAGLE-1/2/3（Li et al., 2024–2025）让 draft 模型成为一个小型 transformer（通常 1 层），它接收 verifier 最后一层的隐藏状态。因为 draft 看到了 verifier 的特征表示，其预测与 verifier 的输出分布强相关。接受率从 ~0.6（vanilla）攀升到 0.85+。

EAGLE-3（2025）增加了候选延续的树搜索。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认 spec 路径。

### KV cache 之舞

验证在一次前向传播中将 `N` 个 draft token 送入 verifier。这将 verifier 的 KV cache 扩展 `N` 个条目。如果某些 draft 被拒绝，你必须将 cache 回滚到被接受的前缀长度。

生产实现（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）用临时 KV 缓冲区处理这个。先写入，接受时提交。概念上不难，但很琐碎。

## 动手构建

见 `code/main.py`。我们实现核心 speculative-sampling 算法（拒绝步骤 + 残差分布）：

- 一个"大模型"，是手工编码分布上的确定性 softmax（这样我们可以解析验证接受数学）。
- 一个"draft 模型"，是大模型的扰动。
- 一个接受/拒绝循环，产生与直接采样相同的边际分布。

### 第 1 步：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数。`q_prob` 是 verifier 对 draft token 的概率。`p_prob` 是 draft 模型的概率。Leviathan 定理是：这个伯努利决策，加上拒绝时从残差采样，精确保持 verifier 的分布。

### 第 2 步：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素从 `q` 减去 `p`，负值截断为零，重新归一化。任何拒绝时从此分布采样。

### 第 3 步：一个 speculative 步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个被接受 → 一个奖励 → 一次 verifier 传播产生六个 token。

### 第 4 步：测量接受率

在不同 draft 质量水平下运行 10,000 个 speculative 步骤。画出接受率 vs draft 和 verifier 分布之间的 KL 散度。你应该看到一个清晰的单调关系。

### 第 5 步：验证分布等价性

经验验证：speculative 循环产生的 token 直方图应该与直接从 verifier 采样产生的直方图匹配。这是 Leviathan 定理的实践。卡方检验在采样误差范围内确认。

## 使用方式

生产部署：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM 截至 2026 年中有最快的 Medusa 路径。`faster-whisper` 为 Whisper-large 封装了带小 draft 的 speculative decoding。

**选择 draft 策略：**

| Strategy | When to pick | Speedup |
|----------|--------------|---------|
| Vanilla draft (1B/3B Llama family) | Fast prototype, no training | 1.8–2.3× |
| Medusa heads | You can fine-tune the verifier | 2–3× |
| EAGLE-2 / 3 | Production, max speed | 3–4× |
| Lookahead | No draft, no training, no extra params | 1.3–1.6× |

**什么时候不用 spec-decode：**

- 单序列生成 1–5 个 token。开销占主导。
- 极度创意/高温度采样（α 下降）。
- 显存受限的部署（draft 模型增加 VRAM）。

## 交付产出

见 `outputs/skill-spec-decode-picker.md`。该 skill 为新推理工作负载选择 speculative decoding 策略（vanilla / Medusa / EAGLE / lookahead）和调优参数（N、draft 温度）。

## 练习

1. **简单。** 运行 `code/main.py`。确认 speculative token 分布在 50,000 个 token 上与 verifier 的直接采样分布匹配，卡方检验 p > 0.05。
2. **中等。** 画出加速（每次大模型前向的 token 数）作为 `N` 的函数，`α = 0.5, 0.7, 0.85`。找出每个 α 的最优 `N`。（提示：每次验证调用的期望 token 数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现一个小型 Medusa：取 Lesson 14 的 capstone GPT，添加 3 个额外 LM 头预测位置 t+2、t+3、t+4。在 tinyshakespeare 上用联合多头损失训练。对比接受率 vs 通过截断同一模型制作的 vanilla draft。
4. **困难。** 实现回滚：从一个 10-token 前缀 KV cache 开始，送入 5 个 draft token，模拟在位置 3 拒绝。验证你的 cache 在下一次迭代时正确读取为"前缀 + 前 2 个被接受的 draft"。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Draft model | "The cheap one" | A smaller model that proposes candidate tokens; usually 10–50× cheaper than the verifier. |
| Verifier | "The big one" | The target model whose distribution we preserve; runs once per speculative step. |
| Acceptance rate (α) | "How often the draft is right" | Per-token probability that the verifier accepts the draft. 0.7–0.9 typical. |
| Residual distribution | "The rejection fallback" | `(q - p)_+` normalized; sampling from this on rejection preserves the verifier's distribution. |
| Bonus token | "The free one" | When all N drafts accepted, sample one more from the verifier's next-step distribution. |
| Medusa | "Draft-less speculative" | Multiple LM heads on the verifier predict positions t+1..t+k in parallel. |
| EAGLE | "Hidden-state draft" | Tiny transformer draft conditioned on the verifier's last-layer hidden states. |
| Lookahead decoding | "Jacobi iteration" | Self-speculation using a fixed-point iteration; no draft model. |
| Tree attention | "Verify many candidates at once" | Branching verification that considers several draft continuations simultaneously. |
| KV rollback | "Undo rejected drafts" | Scratch KV buffer; commit on acceptance, discard on reject. |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — the core algorithm and the equivalence theorem.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — concurrent introduction; clean Bernoulli-rejection proof.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa paper; tree-attention verification.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1; hidden-state-conditioned draft.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2; dynamic tree depth.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead, no-draft approach.
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — canonical production reference with all four strategies wired up.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — the reference code for EAGLE-1/2/3.
