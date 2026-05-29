# Actor-Critic — A2C 与 A3C

> REINFORCE 噪声太大。加一个学习 `V̂(s)` 的 critic，从回报中减去它，你就得到了一个期望相同但方差低得多的 advantage。这就是 actor-critic。A2C 同步运行；A3C 跨线程运行。两者都是所有现代深度 RL 方法的思维模型。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Time:** ~75 minutes

## 问题

原始 REINFORCE 能工作，但方差很糟糕。蒙特卡洛回报 `G_t` 在不同 episode 之间可以波动 10 倍。将这个噪声乘以 `∇ log π` 再取平均，产生的梯度估计器需要数千个 episode 才能将策略移动到用少得多的 DQN 更新就能达到的距离。

方差来自使用原始回报。如果你减去一个基线 `b(s_t)`——任何状态的函数，包括学习的价值——期望不变而方差下降。最佳的可行基线是 `V̂(s_t)`。现在乘以 `∇ log π` 的量是 *advantage*：

`A(s, a) = G - V̂(s)`

一个动作好，如果它产生了高于平均的回报；差，如果低于平均。带学习 critic 的 REINFORCE 就是 *actor-critic*。Critic 给 actor 提供低方差的教学信号。2015 年之后的所有深度策略方法都是这个（A2C、A3C、PPO、SAC、IMPALA）。

## 核心概念

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**两个网络，一个共享损失：**

- **Actor** `π_θ(a | s)`：策略。采样来行动。用策略梯度训练。
- **Critic** `V_φ(s)`：估计从状态开始的期望回报。训练目标是最小化 `(V_φ(s) - target)²`。

**Advantage。** 两种标准形式：

- *MC advantage:* `A_t = G_t - V_φ(s_t)`。无偏，方差较高。
- *TD advantage:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用了 `V_φ`），方差低得多。也叫 *TD 残差* `δ_t`。

**n-step advantage。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现在 Atari 上用 `n = 5`，PPO 在 MuJoCo 上用 `n = 2048`。

**Generalized Advantage Estimation (GAE)。** Schulman et al. (2016) 提出对所有 n-step advantage 取指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差，高偏差）。`λ = 1` 是 MC（高方差，无偏）。`λ = 0.95` 是 2026 年的默认值——调节直到偏差/方差的旋钮在你想要的位置。

**A2C：同步 advantage actor-critic。** 在 `N` 个并行环境中收集 `T` 步。为每步计算 advantage。在合并的 batch 上更新 actor 和 critic。重复。A3C 更简单、更可扩展的兄弟。

**A3C：异步 advantage actor-critic。** Mnih et al. (2016)。启动 `N` 个工作线程，每个运行一个环境。每个 worker 在自己的 rollout 上本地计算梯度，然后异步地将梯度应用到共享参数服务器。不需要回放缓冲区——worker 通过运行不同轨迹来去相关化。A3C 证明了可以在 CPU 上大规模训练。2026 年，基于 GPU 的 A2C（批量并行环境）占主导，因为 GPU 需要大 batch。

**组合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：策略梯度损失、价值回归、熵奖励。`c_v ~ 0.5`，`c_e ~ 0.01` 是经典起点。

## 动手构建

### Step 1: critic

线性 critic `V_φ(s) = w · features(s)`，用 MSE 更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境上 critic 几百个 episode 就收敛。在 Atari 上，将线性 critic 替换为共享 CNN 主干 + 价值头。

### Step 2: n-step advantage

给定长度为 `T` 的 rollout 和自举的最终 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是 critic 的目标。`advantages` 是乘以 `∇ log π` 的量。

### Step 3: 组合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy，每次更新一个 rollout，actor 和 critic 使用不同的学习率。

### Step 4: 并行化（A3C vs A2C）

- **A3C：** 启动 `N` 个线程。每个运行自己的环境和前向传播。定期将梯度更新推送到共享 master。Master 上不加锁——竞争条件没关系，只是增加噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，将观测堆叠成 `[N, obs_dim]` batch，批量前向传播，批量反向传播。更高的 GPU 利用率，确定性，更容易推理。2026 年的默认选择。

我们的玩具代码为了清晰是单线程的；改写为批量 A2C 只需三行 numpy。

## 常见陷阱

- **Critic 偏差先于 actor 梯度。** 如果 critic 是随机的，它的基线没有信息量，你在纯噪声上训练。在开启策略梯度之前先预热 critic 几百步，或使用较慢的 actor 学习率。
- **Advantage 归一化。** 每个 batch 将 advantage 归一化到零均值/单位标准差。以几乎零成本大幅稳定训练。
- **共享主干。** 对图像输入，actor 和 critic 使用共享特征提取器。分离头部。共享特征从两个损失中免费获益。
- **On-policy 契约。** A2C 的数据只用于一次更新。更多次你的梯度就有偏了（重要性采样修正就是 PPO 添加的东西）。
- **熵坍塌。** 没有 `c_e > 0`，策略在几百次更新内变成近确定性的，停止探索。
- **奖励尺度。** Advantage 幅度取决于奖励尺度。归一化奖励（如除以滑动标准差）以获得跨任务一致的梯度幅度。

## 实际应用

A2C/A3C 在 2026 年很少是最终选择，但它们是后续所有方法改进的架构：

| Method | Relation to A2C |
|--------|----------------|
| PPO | A2C + clipped importance ratio for multi-epoch updates |
| IMPALA | A3C + V-trace off-policy correction |
| SAC (Phase 9 · 07) | Off-policy A2C with a soft-value critic (next lesson) |
| GRPO (Phase 9 · 12) | A2C without the critic — group-relative advantage |
| DPO | A2C collapsed into a preference-ranking loss, no sampling |
| AlphaStar / OpenAI Five | A2C with league training + imitation pre-training |

如果你在 2026 年的论文中看到"advantage"，想到 actor-critic。

## Ship It

Save as `outputs/skill-actor-critic-trainer.md`:

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **Easy.** 在 4×4 GridWorld 上用 MC advantage（`G_t - V(s_t)`）训练 actor-critic。与 Lesson 06 中带滑动平均基线的 REINFORCE 比较样本效率。
2. **Medium.** 切换到 TD 残差 advantage（`r + γ V(s') - V(s)`）。测量 advantage batch 的方差。下降了多少？
3. **Hard.** 实现 GAE(λ)。扫描 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制最终回报 vs 样本效率。在这个任务上偏差/方差的最佳点在哪里？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Actor | "The policy net" | `π_θ(a\|s)`, updated by policy gradient. |
| Critic | "The value net" | `V_φ(s)`, updated by MSE regression to returns / TD targets. |
| Advantage | "How much better than average" | `A(s, a) = Q(s, a) - V(s)` or its estimators. Multiplier for `∇ log π`. |
| TD residual | "δ" | `δ_t = r + γ V(s') - V(s)`; one-step advantage estimate. |
| GAE | "The interpolation knob" | Exponentially weighted sum of n-step advantages, parameterized by `λ`. |
| A2C | "Synchronous actor-critic" | Batched across envs; one gradient step per rollout. |
| A3C | "Async actor-critic" | Worker threads push gradients to a shared param server. Original paper; less common in 2026. |
| Bootstrap | "Use V at the horizon" | Truncate the rollout, add `γ^n V(s_{t+n})` to close the sum. |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C, the original async actor-critic paper.
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE.
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — foundations; pair this with Ch. 9 on function approximation when the critic is a neural net.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — scalable distributed actor-critic with V-trace off-policy correction.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — production A2C/PPO implementations worth reading.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — the foundational convergence result for the two-timescale actor-critic decomposition.
