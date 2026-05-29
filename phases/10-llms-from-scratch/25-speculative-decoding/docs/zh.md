# Speculative Decoding 与 EAGLE

> 前沿 LLM 生成一个 token 需要对数十亿参数做一次完整前向传播。这次前向传播被大量过度配置：大多数时候一个小得多的模型就能正确猜出接下来的 3-5 个 token，大模型只需要*验证*猜测。当猜测正确时，你用一次前向的代价得到了 5 个 token。Speculative decoding（Leviathan et al. 2023）使这一过程精确，EAGLE-3（2025）将接受率推到每次验证 ~4.5 个 token——在匹配输出分布的前提下实现 4-5 倍加速。

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 minutes

## 问题

70B 级模型在 H100 上的解码吞吐量通常是 40-80 tokens/second。每个 token 需要一次完整前向传播，从 HBM 读取所有模型权重。你不能在不改变输出的情况下缩小模型。你不能把 batch size 增加到超出内存。你被卡住了——除非你能让模型每次前向传播输出多于一个 token。

自回归生成看起来本质上是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但存在并发机会。如果你有一个廉价预测器说"接下来 4 个 token 大概是 [a, b, c, d]"，你可以在**大模型的单次前向传播**中验证所有 5 个位置，并接受最长匹配前缀。

Leviathan, Kalai, Matias（2023, "Fast Inference from Transformers via Speculative Decoding"）通过一个巧妙的接受/拒绝规则使这一过程精确，保持目标模型的采样分布。相同的输出分布，快 2-4 倍。

## 概念

### 双模型设置

- **Target model** `M_p`：大的、慢的、高质量的模型，你实际想从中采样。分布：`p(x)`。
- **Draft model** `M_q`：小的、快的、低质量的模型。分布：`q(x)`。小 5-30 倍。

每步：

1. Draft model 自回归地提议 `K` 个 token：`x_1, x_2, ..., x_K ~ q`。
2. Target model 对所有 `K+1` 个位置并行运行一次前向传播，为每个提议 token 产生 `p(x_k)`。
3. 通过下面的修改拒绝采样规则从左到右接受/拒绝每个 token。接受最长匹配前缀。
4. 如果任何 token 被拒绝，从修正分布中采样替代并停止。否则从 `p(· | x_1...x_K)` 采样一个额外 token。

如果 draft 与 target 完美匹配，你每次 target 前向得到 K+1 个 token。如果 draft 在位置 1 就错了，你只得到 1 个 token。

### 精确性规则

Speculative decoding **可证明等价于从 p 采样**。拒绝规则：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差的正部。当 draft 和 target 一致时（`p ≈ q`）接受率接近 1。当它们不一致时，残差分布被构造为使整体采样仍然精确等于 `p`。

**贪心情况。** 对于 temperature=0 的采样，只需检查 `argmax(p) == x_t`。如果是，接受；如果不是，输出 `argmax(p)` 并停止。

### 期望加速

如果 draft model 的 token 级接受率是 `α`，每次 target 前向传播产生的期望 token 数是：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

当 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36` 个 token 每次前向。单次 target 前向的代价大约是 `cost_q * K + cost_p`（K 次 draft 步骤加一次 target 验证）。如果 `cost_p >> cost_q * K`，吞吐量加速比是 `3.36× / 1 = 3.36×`。

唯一真正的参数是 `α`，它完全取决于 draft-target 的对齐程度。好的 draft 就是一切。

### 训练 Draft：蒸馏

随机的小模型做不好 draft。标准方案是从 target 蒸馏：

1. 选择一个小架构（70B target 对应 ~1B，7B target 对应 ~500M）。
2. 在大文本语料上运行 target model；存储其 next-token 分布。
3. 用 KL 散度对 target 的分布（而非 ground-truth token）训练 draft。

结果：`α` 在代码上通常 0.6-0.8，在自然语言对话上 0.7-0.85。生产中加速 2-3 倍。

### EAGLE：树状 Draft + 特征复用

Li, Wei, Zhang, Zhang（2024, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"）观察到标准 speculative decoding 的两个低效：

1. Draft 做 K 次串行步骤，每次全栈。但 draft 可以复用 target 最近一次验证的特征（隐状态）——target 已经计算了丰富的表示，draft 却从头重新推导。
2. Draft 输出线性链。如果 draft 能输出候选*树*（每个节点多个猜测），target 的单次前向传播可以通过树状 attention mask 并行验证多条候选路径，并选择最长被接受的分支。

EAGLE-1 的改变：
- Draft 输入 = target 在位置 t 的最终隐状态，而非原始 token。
- Draft 架构 = 1 个 transformer decoder 层（不是独立的小模型）。
- 输出 = 每个深度 K = 4-8 个候选的树，深度 4-6。

EAGLE-2（2024）添加动态树拓扑：树在 draft 不确定的地方变宽，在确定的地方保持窄。在不增加验证代价的情况下提高 `α_effective`。

EAGLE-3（Li et al. 2025, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"）移除了固定的顶层特征依赖，用新的"test-time simulation"损失训练 draft——draft 在匹配 target 测试时分布（而非 teacher-forced 训练分布）的输出上训练。接受率从 0.75（EAGLE-2）升到 0.82（EAGLE-3），平均 token/验证从 3.0 升到 4.5。

### 树状 Attention 验证

当 draft 输出一棵树时，target model 用**树状 attention mask**在单次前向传播中验证它——一个编码树拓扑而非纯线性的因果 mask。每个 token 只 attend 到树中的祖先。验证仍然是一次前向、一次矩阵乘法；拓扑 mask 只多花几个额外的 KV 条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争的第一 token 候选，`c, d, e, f` 是第二 token 候选，所有六个位置在一次前向传播中被验证。输出是沿任何被接受路径的最长前缀。

### 何时有效，何时无效

**有效：**
- 可预测文本的对话/补全（代码、常见英语、结构化输出）。`α` 高。
- 解码期间有未使用 GPU 算力的场景（内存受限阶段）。树状 draft 利用了可用的 FLOPs。

**无效/无收益：**
- 高度随机的输出（高温度的创意写作）。`α` 降向 `1/|vocab|`。
- 高并发的批量服务——批处理已经填满了 FLOPs，没有空间做树验证。
- 非常小的 target model，draft 没有小多少。

生产环境通常报告对话 2-3 倍墙钟加速，代码生成 3-5 倍，创意写作接近零。

## 构建

`code/main.py`：

- 一个参考 `speculative_decode(target, draft, prompt, K, temperature)` 实现精确拒绝规则，并验证它保持 target 的分布（经验 KL < 0.01 vs 纯 target 采样）。
- 一个 EAGLE 风格的树状 drafter，用 top-p 分支构建深度 K 的树。
- 一个树状 attention mask 构建器，为验证器产生正确的因果模式。
- 一个接受率测试工具，在小型 LM 上运行两者（从 GPT-2-medium target 蒸馏一个 GPT-2-small）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 使用

- **vLLM** 和 **SGLang** 提供一流的 speculative decoding 支持。参数：`--speculative_model`、`--num_speculative_tokens`。EAGLE-2/3 通过 `--spec_decoding_algorithm eagle` 参数支持。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。
- **参考 draft 模型**：`Qwen/Qwen3-0.6B-spec`（为 Qwen3-32B 做 draft）、`meta-llama/Llama-3.2-1B-Instruct-spec`（为 70B 做 draft）。
- **Medusa heads**（Cai et al. 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"）：不用 draft model，而是在 target 本身上添加 K 个并行预测头。部署更简单，接受率略低于 EAGLE。

## 交付

本课产出 `outputs/skill-speculative-tuning.md`——一个分析 target model 工作负载并选择以下参数的技能：draft model、K（draft 长度）、树宽度、temperature，以及何时回退到普通解码。

## 练习

1. 实现精确拒绝规则并经验验证。通过 `speculative_decode` 运行 10K 次采样，通过纯 target 采样也运行 10K 次；计算两个输出分布之间的 TV 距离。应该 < 0.01。

2. 计算加速公式。给定固定 `α` 和 `K`，绘制每次 target 前向的期望 token 数。找到 α ∈ {0.5, 0.7, 0.9} 时的最优 K。

3. 训练一个小 draft。取 124M GPT-2 target，在 100M token 上用 KL 损失蒸馏一个 30M GPT-2 draft。在留出文本上测量 `α`。预期：0.6-0.7。

4. 实现 EAGLE 风格的树状 draft。不输出链，而是让 draft 在每个深度输出 top-3 分支。构建树状 attention mask。验证 target 接受最长正确分支。

5. 测量失败模式。在 temperature=1.5（高随机性）下运行 speculative decode。展示 α 崩溃，算法因 draft 开销比普通解码更慢。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|------------------------|
| Target model | "大模型" | 你想从中采样的慢的、高质量模型（p 分布） |
| Draft model | "推测器" | 小的、快的预测器（q 分布）；小 5-30 倍 |
| K / draft length | "前瞻" | 每次验证传播中推测的 token 数 |
| α / acceptance rate | "命中率" | draft 提议被接受的逐 token 概率 |
| Exact rejection rule | "接受测试" | r < p/q 比较，保持 target 的分布 |
| Residual distribution | "修正的 p-q" | (p - q)+ / \|\|(p - q)+\|\|_1，拒绝时从中采样的分布 |
| Tree drafting | "分支推测" | Draft 输出候选树，用树状 attention mask 在一次传播中验证 |
| Tree attention mask | "拓扑 mask" | 编码树拓扑的因果 mask，每个节点只 attend 到其祖先 |
| Medusa heads | "并行头" | target 本身上的 K 个额外预测头；无需独立 draft model |
| EAGLE feature reuse | "隐状态 draft" | Draft 输入是 target 的最后隐状态而非原始 token，缩小了 draft |
| Test-time simulation loss | "EAGLE-3 训练" | 在匹配 target 测试时分布的输出上训练 draft，而非 teacher forcing |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则和理论加速分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 的并发 speculative sampling 论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — draft model 的并行头替代方案
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特征复用和树状 draft
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树拓扑
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时测试时匹配
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/lookahead decoding，无推测器的替代方案
