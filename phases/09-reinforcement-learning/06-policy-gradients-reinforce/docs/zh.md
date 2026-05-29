# 策略梯度 — 从零实现 REINFORCE

> 别再估计价值了。直接参数化策略，计算期望回报的梯度，沿梯度上升。Williams (1992) 用一个定理写完了。PPO、GRPO 以及所有 LLM RL 循环都源于此。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 minutes

## 问题

Q-learning 和 DQN 参数化的是*价值*函数。你通过 `argmax Q` 选择动作。对于离散动作和离散状态这没问题。但当动作是连续的（对 10 维力矩取哪个 `argmax`？）或者你想要随机策略时（`argmax` 本质上是确定性的），它就不行了。

策略梯度参数化的是*策略*本身。`π_θ(a | s)` 是一个输出动作分布的神经网络。从中采样来行动。计算期望回报对 `θ` 的梯度。沿梯度上升。没有 `argmax`。没有 Bellman 递归。只是对 `J(θ) = E_{π_θ}[G]` 做梯度上升。

REINFORCE 定理（Williams 1992）告诉你这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。跑一个 episode。计算回报。在每一步乘以 `∇ log π_θ(a | s)`。取平均。梯度上升。完成。

2026 年的每一个 LLM-RL 算法——PPO、DPO、GRPO——都是 REINFORCE 的改进。把它内化到肌肉记忆中是本阶段后续内容的前提，也是 Phase 10 · 07（RLHF 实现）和 Phase 10 · 08（DPO）的前提。

## 核心概念

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**策略梯度定理。** 对于任何由 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从步骤 `t` 开始的折扣回报。期望是对从 `π_θ` 采样的完整轨迹 `τ` 取的。

**证明很短。** 对 `J(θ) = Σ_τ P(τ; θ) G(τ)` 在期望下求导。使用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（对数导数技巧）。将 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + 不依赖于 θ 的环境项` 分解。环境项消失。两行代数给出定理。

**方差缩减技巧。** 原始 REINFORCE 的方差极大——回报有噪声，`∇ log π` 有噪声，它们的乘积噪声更大。两个标准修复：

1. **基线减法。** 将 `G_t` 替换为 `G_t - b(s_t)`，其中 `b(s_t)` 是任何不依赖于 `a_t` 的基线。无偏，因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`。典型选择：`b(s_t) = V̂(s_t)` 由 critic 学习 → actor-critic（Lesson 07）。
2. **Reward-to-go。** 将 `Σ_t G_t · ∇ log π_θ(a_t | s_t)` 替换为 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`。对于给定动作只有未来回报有意义——过去的奖励贡献零均值噪声。

组合起来，你得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带基线的 REINFORCE——A2C（Lesson 07）和 PPO（Lesson 08）的直接祖先。

**Softmax 策略参数化。** 对于离散动作，标准选择：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任何输出每个动作分数的神经网络。梯度有简洁的形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即所选动作的分数减去策略下的期望值。

**连续动作的高斯策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭式解。Phase 9 · 07 的 SAC 只需要这些。

## 动手构建

### Step 1: softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对表格环境使用线性策略（每个动作一个权重向量）。对 Atari，换成 CNN 并保留 softmax 头。

### Step 2: 采样和对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Step 3: 带 log-probs 记录的 rollout

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Step 4: REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（动作 `a` 的 onehot 减去概率）是 softmax 策略梯度的核心。把它刻进肌肉记忆。

### Step 5: 基线

对近期 episode 的 `G` 取滑动平均作为基线，就足以让 4×4 GridWorld 跑起来；大约 ~500 个 episode 收敛。将基线升级为学习的 `V̂(s)` 就得到了 actor-critic。

## 常见陷阱

- **梯度爆炸。** 回报可能很大。在乘以 `∇ log π` 之前，始终将 `G` 在 batch 内归一化到 `~N(0, 1)`。
- **熵坍塌。** 策略过早收敛到近确定性动作，停止探索，陷入局部。修复：在目标中加入熵奖励 `β · H(π(·|s))`。
- **高方差。** 原始 REINFORCE 需要数千个 episode。Critic 基线（Lesson 07）或 TRPO/PPO 的信赖域（Lesson 08）是标准修复。
- **样本效率低。** On-policy 意味着每次更新后丢弃所有转移。通过重要性采样的离策略修正可以复用数据，代价是方差增加（PPO 的 ratio 就是裁剪的 IS 权重）。
- **非平稳梯度。** 100 个 episode 前的梯度使用的是旧 `π`。On-policy 方法因此每几个 rollout 就更新一次。
- **信用分配。** 没有 reward-to-go，过去的奖励贡献噪声。始终使用 reward-to-go。

## 实际应用

2026 年，REINFORCE 很少直接运行，但它的梯度公式无处不在：

| Use case | Derived method |
|----------|---------------|
| Continuous control | PPO / SAC with Gaussian policy |
| LLM RLHF | PPO with KL penalty, running on token-level policy |
| LLM reasoning (DeepSeek) | GRPO — REINFORCE with group-relative baseline, no critic |
| Multi-agent | Centralized-critic REINFORCE (MADDPG, COMA) |
| Discrete action robotics | A2C, A3C, PPO |
| Preference-only settings | DPO — REINFORCE rewritten as a preference-likelihood loss, no sampling |

当你在 2026 年的训练脚本中看到 `loss = -advantage * log_prob`，那就是带基线的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都是在这一行之上的方差缩减技巧。

## Ship It

Save as `outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **Easy.** 在 4×4 GridWorld 上用线性 softmax 策略实现 REINFORCE。训练 1,000 个 episode，不使用基线。绘制学习曲线；测量方差（回报的标准差）。
2. **Medium.** 添加滑动平均基线。再次训练。比较与原始版本的样本效率和方差。基线将收敛步数减少了多少？
3. **Hard.** 添加熵奖励 `β · H(π)`。扫描 `β ∈ {0, 0.01, 0.1, 1.0}`。绘制最终回报和策略熵。在这个任务上最佳点在哪里？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Policy gradient | "Train the policy directly" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; derived from the log-derivative trick. |
| REINFORCE | "The original PG algorithm" | Williams (1992); Monte Carlo returns multiplied by log-policy gradient. |
| Log-derivative trick | "Score function estimator" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; makes gradients of expectations tractable. |
| Baseline | "Variance reduction" | Any `b(s)` subtracted from `G`; unbiased because `E[b · ∇ log π] = 0`. |
| Reward-to-go | "Only future returns count" | `G_t^{from t}` instead of the full `G_0`; correct and lower-variance. |
| Entropy bonus | "Encourage exploration" | `+β · H(π(·\|s))` term keeps the policy from collapsing. |
| On-policy | "Train on what you just saw" | Gradient expectation is w.r.t. the current policy — cannot reuse old data directly. |
| Advantage | "How much better than average" | `A(s, a) = G(s, a) - V(s)`; the signed quantity REINFORCE-with-baseline multiplies. |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — the original REINFORCE paper.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — the modern policy-gradient theorem with function approximation.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — textbook presentation.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — clear pedagogical exposition with PyTorch code.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — variance-reduction and the natural-gradient view that connects REINFORCE to the trust-region family (TRPO, PPO).
