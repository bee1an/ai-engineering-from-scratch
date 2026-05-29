# Speculative Decoding 与 EAGLE-3

> Phase 7 · 第 16 课证明了数学：Leviathan 拒绝规则精确保持验证器的分布。本课是 2026 年生产级 speculative decoding 的训练栈视角。EAGLE-3 将 draft 模型从一个廉价近似变成了一个专门构建的小型网络，在验证器自身的隐藏状态上训练，然后添加了一个 training-time test 循环来对齐其训练和推理分布。结果：3x 到 6.5x 端到端加速，chat 场景下每 token 接受率超过 0.9，无分布折衷。2026 年每个生产推理栈默认搭载它。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 16 (speculative decoding math), Phase 10 · 12 (inference optimization)
**Time:** ~75 minutes

## 学习目标

- 用一句话陈述 Leviathan 定理，并证明 speculative 循环产生的样本与验证器分布完全相同。
- 走过从 vanilla spec-decoding（Leviathan 2023）到 EAGLE、EAGLE-2 和 EAGLE-3 的两年演进，指出每一步移除的确切限制。
- 从接受率 `α` 和 draft-to-verifier 成本比 `c` 计算预期加速，并为每种场景选择最优 draft 长度 `N`。
- 从零实现完整的 speculative 循环：draft、verify、从残差分布拒绝采样、拒绝时回滚 KV cache、全部接受时发射 bonus token。

## 问题

70B 模型的自回归解码在 H100 上大约每秒 35 个 token。GPU 远未饱和。内存带宽是天花板：每个 token 从 HBM 加载 70B 权重，做一步算术，产出一个浮点数。计算单元大部分时间闲置。

Speculative decoding 将其转化为一个你实际可以解决的吞吐问题。一个廉价的 draft 在 `N` 次小型前向传播中提议 `N` 个 token。验证器对前缀加所有 `N` 个 draft 运行一次前向传播。如果验证器在位置 `i` 的分布与 draft 一致（在我们将精确定义的统计意义上），我们接受；否则拒绝并从残差分布采样修正。一次大模型前向产出最多 `N+1` 个被接受的 token 而非一个。

关键定理是 Leviathan, Kalman, Matias (ICML 2023)：输出分布与直接从验证器采样产出的完全相同。不是近似。完全相同。这就是 speculative decoding 在生产中可接受的全部原因——它是一个纯延迟优化，没有质量折衷。

Phase 7 · 第 16 课给你的是数学。本课给你的是训练栈。一个好的 draft 比一个廉价 draft 多值 2x 加速。EAGLE、EAGLE-2 和 EAGLE-3（Li et al., 2024-2025）将"draft = 同一模型的更小版本"变成了一门精确的工程学科。2026 年生产推理服务器默认使用 EAGLE-3。

## 概念

### 不变量：Leviathan 拒绝采样

设 `p(t)` 为 draft 在给定前缀下对下一个 token 的分布，`q(t)` 为验证器的。从 draft 采样一个 token `d ~ p`。以概率 `min(1, q(d) / p(d))` 接受。拒绝时，从残差分布 `(q - p)_+ / ||(q - p)_+||_1` 采样。结果样本按 `q` 分布。无论 `p` 有多差这都成立——它越差，你拒绝越频繁，但输出保持精确。

将 `N` 个这样的调用背靠背堆叠，使用一次验证器前向传播处理 `prefix + d_1 + ... + d_N`。验证器同时返回 `q_1, q_2, ..., q_{N+1}`。从左到右走。在位置 `j` 第一次拒绝时，从 `residual(q_j, p_j)` 采样并停止。全部接受时，从 `q_{N+1}` 采样一个 bonus token。

### 什么决定加速

设 `α` 为每个 drafted token 的预期接受率。设 `c = cost(draft) / cost(verifier)` 为成本比。每次验证器前向的预期接受 token 数为：

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

每个被接受 token 的预期总墙钟时间为 `(N * c + 1) / E[accepted]`。对 `N` 最小化就得到最优点。对于 `α = 0.8, c = 0.05`：最优 `N` 约 5-7，加速 3.2x。对于 `α = 0.95, c = 0.02`：最优 `N` 约 8-10，加速推向 5x。

最大的杠杆是 `α`。在固定 `N = 5` 下从 `α = 0.6`（vanilla draft）到 `α = 0.9`（EAGLE-3）将你从每次验证器调用 2.2 个预期接受 token 提升到 4.1。同一个验证器近 2x 更多吞吐。

### 两年演进

**Vanilla speculative（Leviathan, 2023）。** Draft 模型是同一家族独立训练的更小 LLM。容易接入，`α ≈ 0.6`，加速最多约 2x。

**EAGLE-1（Li et al., 2024）。** Draft 是一个小型 transformer——通常一两层——以验证器最后一层隐藏状态为输入直接预测下一个 token。因为 draft 看到了验证器的特征表示，其分布更接近验证器。`α` 攀升到 0.7-0.8。

**EAGLE-2（Li et al., 2024）。** 添加动态 draft 树：不是提议单个 `N` token 序列，而是提议一棵小的候选树，在一次验证器前向传播中用 tree attention 评分每个候选，走最高概率路径。Draft 长度变为每步自适应。每个被接受路径 token 的 `α` 攀升到 0.85 以上。

**EAGLE-3（Li et al., 2025, NeurIPS）。** 两个更多变化。第一，完全丢弃特征预测损失——EAGLE-1/2 训练 draft 匹配验证器的隐藏状态，这限制了数据能帮助多少。EAGLE-3 直接在 token 预测上训练。第二，training-time test（TTT）：在 draft 训练期间，将 draft 自己之前的预测作为输入反馈多步，与推理时的操作方式相同。这对齐了训练和测试分布并阻止了误差累积。测量加速：chat 上最高 6.5x，SGLang 在 H100 上 batch 64 时 38% 吞吐提升。

### KV cache 回滚

验证在一次传播中将验证器的 KV cache 扩展 `N` 个条目。如果在位置 `j` 发生拒绝，位置 `j-1` 之后的 cache 内容现在是错的。两种常见实现：写入暂存缓冲区并在接受时提交（vLLM, TensorRT-LLM），或保持物理 KV cache 加逻辑长度并在拒绝时截断。无论哪种，回滚成本是每层每头的字节数，相对前向传播成本可忽略。

对于 EAGLE-2 树搜索，验证器运行带有尊重树拓扑的非因果 mask 的 attention。工程上很繁琐但计算是标准的 flash-attention 调用加自定义 mask。

### 2026 年的 Draft 架构

| Strategy | Draft type | `α` | Speedup | Training cost |
|----------|-----------|-----|---------|---------------|
| Vanilla | Separate small LLM | 0.55-0.70 | 1.8-2.3x | None (reuse existing small model) |
| Medusa | Extra LM heads on verifier | 0.65-0.75 | 2-3x | ~1B SFT tokens |
| EAGLE-1 | 1-layer transformer on hidden states | 0.70-0.80 | 2.5-3x | ~60B tokens |
| EAGLE-2 | EAGLE-1 + dynamic draft tree | 0.80-0.88 | 3-4x | ~60B tokens |
| EAGLE-3 | Multi-layer feature fusion + TTT | 0.88-0.92 | 3.5-6.5x | ~60-200B tokens |
| Lookahead | No draft (Jacobi iteration) | N/A | 1.3-1.6x | None |

2026 年生产中：vLLM 和 SGLang 在可用时默认 EAGLE-3，否则 EAGLE-2。TensorRT-LLM 对 Meta 和 NVIDIA 公开模型有最快的 Medusa 路径。llama.cpp 为 CPU 部署搭载 vanilla draft。

## 构建

参见 `code/main.py`。这是完整的 Leviathan speculative 循环，包含所有部件：N 个 draft、验证器并行传播、逐位置拒绝、残差采样、bonus token、KV 回滚，以及输出分布匹配直接从 `q` 采样的经验验证。

### 步骤 1：拒绝规则

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 步骤 2：残差分布

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 步骤 3：完整的 speculative 步骤

`spec_step` 函数从 `p` draft `N` 个 token，然后在一次并行 `q` 评估中验证所有 token。对每个 drafted token 应用拒绝规则，在第一次拒绝时从残差采样修正。如果全部接受，从 `q_{N+1}` 发射一个 bonus token。

### 步骤 4：KV 回滚簿记

模拟器为每个 worker 跟踪逻辑 `kv_length`。接受 `k` 个 draft 时，`kv_length += k`。在位置 `j` 拒绝时，cache 已经写过 `j`，但逻辑长度设为 `prefix_length + j + 1`——修正 token 之后一位。后续读取截断到逻辑长度。

### 步骤 5：Leviathan 检验

运行 50,000 次 speculative 步骤。统计被接受 token 的经验分布。与 50,000 次直接从 `q` 采样比较。卡方统计量应远低于临界值。定理在实践中通过。

### 步骤 6：加速 vs. α

通过以不同幅度扰动 `p` 远离 `q` 来扫描 draft 质量。测量 `α`，然后绘制每次验证器调用的预期 token 数作为 `α` 和 `N` 的函数。代码打印一个表格展示 EAGLE-3 级 draft 质量（`α ≈ 0.9`）如何解锁每次验证器调用 4-5 个 token。

## 使用

生产级 `vllm serve` 配合 EAGLE-3：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang 配合 EAGLE-3 在 H100 上 batch 64：比 batch-64 vanilla decoding 大约多 1.38x 吞吐，来自 EAGLE-3 论文。

何时使用 speculative decoding：

- 任何 p50 延迟比峰值吞吐更重要的交互式聊天工作负载。
- 代码生成和结构化输出（JSON, SQL）。`α` 超过 0.9 因为目标分布高度可预测。
- 长文本生成（数千 token）。摊销加速持续回报。

何时不用：

- 非常小的模型（< 3B）。Draft 不比验证器便宜多少。
- 极小的 batch-1 CPU 部署。Draft 模型的内存开销可能不值得。
- 非常高温度的创意采样，`α` 会崩溃。

## 交付

本课产出 `outputs/skill-eagle3-tuner.md`。给定推理工作负载（模型、batch size、目标延迟、任务画像），它推荐 speculative-decoding 策略和调优参数（draft 家族、`N`、树深度、温度感知切换）。

## 练习

1. 运行 `code/main.py`。确认 50,000 样本上 Leviathan 分布检验的卡方统计量保持在 95% 临界值以下。

2. 在 `α` 固定为 0.9、`c` 固定为 0.04 时扫描 `N` 从 1 到 10。绘制每次验证器调用的预期 token 数和实际每 token 墙钟时间。找到最小化墙钟时间的 `N`。解释曲线的形状。

3. 修改代码模拟 EAGLE-2 树搜索：每步 draft 提议形状为 `[2, 2, 2]` 的树（八条候选路径）。验证器运行一次，最高概率的被接受路径获胜。计算每个叶子的 `α` 和每次验证器调用的总 token 数。与等效计算量下的线性链 spec-decoding 比较。

4. 实现一个两个并发序列的批量 KV 回滚模拟器。序列 A 所有 draft 被接受；序列 B 在位置 2 拒绝。展示正确的 `kv_length` 按序列更新且没有工作浪费。

5. 阅读 EAGLE-3 论文的 Section 4（Training-Time Test）。用两句话解释为什么没有 TTT 的朴素 draft 训练会遭受 exposure bias，以及为什么在训练期间将 draft 自己的预测反馈回去能修复它。将此与 seq2seq 中的 scheduled-sampling 文献联系起来。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Leviathan rule | "min(1, q over p)" | 以概率 `min(1, q(d)/p(d))` 的 Bernoulli 接受/拒绝，在拒绝时从残差采样精确保持验证器分布 |
| Residual distribution | "(q minus p) plus, normalized" | `(q - p)_+` 在零处截断并重新归一化——拒绝时正确的采样分布 |
| Acceptance rate α | "draft 对了多少次" | 拒绝规则下的预期每 token Bernoulli 成功概率；支配所有加速数学 |
| EAGLE-1 | "hidden-state draft" | 以验证器最后一层隐藏状态为条件的小型 transformer draft（Li et al., 2024） |
| EAGLE-2 | "dynamic draft tree" | EAGLE-1 加候选延续树，在一次验证器传播中用 tree attention 评分 |
| EAGLE-3 | "training-time test" | 丢弃特征预测损失，在 draft 训练期间将自身输出反馈，直接在 token 预测上训练 |
| Training-time test (TTT) | "exposure bias 修复" | 训练期间自回归运行 draft 使训练和测试输入分布匹配——scheduled sampling 的直接类比 |
| KV rollback | "撤销被拒绝的 draft" | 在拒绝后将验证器的 KV cache 重置到被接受前缀长度的簿记 |
| Bonus token | "免费的那个" | 当所有 `N` 个 draft 被接受时，以零额外验证器成本从 `q_{N+1}` 采样一个额外 token |
| Tree attention | "一次验证多个候选" | 带有尊重 draft 树拓扑的非因果 mask 的 attention；在一次前向传播中为树中每个节点计算 `q_i` |

## 延伸阅读

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — 基础论文和等价定理
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — 并发独立引入，证明简洁
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE-1，隐藏状态条件 draft
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 动态树搜索
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — 2026 年生产默认
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — 替代的无 draft 方法
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 标准生产参考，所有策略已接入
