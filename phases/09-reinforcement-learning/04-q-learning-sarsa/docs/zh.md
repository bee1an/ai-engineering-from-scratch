# 时序差分——Q-Learning 与 SARSA

> 蒙特卡洛等到 episode 结束。TD 在每一步之后就通过自举下一个价值估计来更新。Q-learning 是 off-policy 且乐观的；SARSA 是 on-policy 且保守的。两者都只有一行代码。两者都支撑着本 phase 中的每个深度 RL 方法。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**Time:** ~75 minutes

## 问题

蒙特卡洛方法有效，但它有两个昂贵的要求。它需要 episode 终止，而且只在最终回报出来后才更新。如果你的 episode 有 1,000 步，MC 等 1,000 步才更新任何东西。它是高方差、低偏差、实践中很慢的。

动态规划有相反的特征——零方差的自举 backup——但需要已知模型。

时序差分（TD）学习取两者之间的折中。从单个转移 `(s, a, r, s')` 出发，构造一步目标 `r + γ V(s')` 并将 `V(s)` 向它推动。不需要模型。不需要完整 episode。使用近似 `V` 在右侧带来偏差，但方差比 MC 大幅降低，而且从第一步就能在线更新。

这是整个现代 RL——DQN、A2C、PPO、SAC——转动的支点。Phase 9 的其余部分是在你将在本课中编写的一步 TD 更新之上叠加的函数逼近和技巧层。

## 概念

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

方括号中的量是 TD 误差 `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线类比。收敛需要 `α` 满足 Robbins-Monro（`Σ α = ∞`，`Σ α² < ∞`）且所有状态被无限次访问。

**Q-learning。** 一种 off-policy TD 控制方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 开始将遵循*贪心*策略，无论智能体实际采取什么动作。这种解耦使 Q-learning 在智能体通过 ε-greedy 探索的同时学习 `Q*`。Mnih et al. (2015) 将其转化为 Atari 上的 deep Q-learning（Lesson 05）。

**SARSA。** 一种 on-policy TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名字就是元组 `(s, a, r, s', a')`。SARSA 使用智能体*实际*采取的下一个动作 `a'`，而不是贪心的 `argmax`。对于当前运行的 ε-greedy `π` 收敛到 `Q^π`，在极限 `ε → 0` 时变为 `Q*`。

**悬崖行走的差异。** 在经典的悬崖行走任务（掉下悬崖 = 奖励 -100）上，Q-learning 学到沿悬崖边缘的最优路径，但在探索期间偶尔承受惩罚。SARSA 学到离悬崖一步远的更安全路径，因为它将探索噪声纳入了 Q 值。随着训练，两者在 `ε → 0` 时都达到最优。实践中这很重要：当探索确实在部署时发生时，SARSA 的行为更保守。

**Expected SARSA。** 将 `Q(s', a')` 替换为其在 `π` 下的期望值：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比 SARSA 方差更低（不需要采样 `a'`），相同的 on-policy 目标。在现代教科书中通常是默认选择。

**n-step TD 和 TD(λ)。** 通过等待 `n` 步再自举来在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 用几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 取平均。大多数深度 RL 使用 `n` 在 3 到 20 之间。

## 动手构建

### 第 1 步：ε-greedy 策略上的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行代码。与 Q-learning 的*唯一*区别在于 target 那一行。

### 第 2 步：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。那一个符号就是 on-policy 和 off-policy 的区别。

### 第 3 步：学习曲线

跟踪每 100 个 episode 的平均回报。Q-learning 在简单确定性 GridWorld 上收敛更快；SARSA 在悬崖行走上更保守。在 `code/main.py` 的 4×4 GridWorld 上，两者在约 2,000 个 episode 后（`α=0.1, ε=0.1`）接近最优。

### 第 4 步：与 DP 真值比较

运行值迭代（Lesson 02）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个健康的表格 TD 智能体在 10,000 个 episode 后在 4×4 GridWorld 上误差在 `~0.5` 以内。

## 常见陷阱

- **初始 Q 值很重要。** 乐观初始化（负奖励任务中 `Q = 0`）鼓励探索。悲观初始化可能永远困住贪心策略。
- **α 调度。** 常数 `α` 对非平稳问题没问题。衰减 `α_n = 1/n` 理论上给出收敛但实践中太慢——将 `α` 固定在 `[0.05, 0.3]` 并监控学习曲线。
- **ε 调度。** 从高开始（`ε=1.0`），衰减到 `ε=0.05`。"GLIE"（极限贪心且无限探索）是收敛条件。
- **Q-learning 中的 max 偏差。** 当 `Q` 有噪声时，`max` 算子向上偏差。导致过估计——Hasselt 的 Double Q-learning（Lesson 05 中的 DDQN 使用）用两个 Q 表修复。
- **非终止 episode。** TD 可以在没有终止的情况下学习，但你需要要么限制步数，要么在限制处正确处理自举。标准做法：将限制视为非终止，继续自举。
- **状态哈希。** 如果状态是元组/张量，使用可哈希的键（tuple 而非 list；四舍五入的 float tuple 而非原始值）。

## 实际应用

2026 年的 TD 格局：

| 任务 | 方法 | 原因 |
|------|--------|--------|
| Small tabular environments | Q-learning | 直接学习最优策略。 |
| On-policy safety-critical | SARSA / Expected SARSA | 探索期间保守。 |
| High-dimensional state | DQN (Phase 9 · 05) | 带 replay 和 target net 的神经网络 Q 函数。 |
| Continuous actions | SAC / TD3 (Phase 9 · 07) | Q 网络上的 TD 更新；策略网络输出动作。 |
| LLM RL (reward-model-based) | PPO / GRPO (Phase 9 · 08, 12) | 通过 GAE 的 TD 风格优势的 actor-critic。 |
| Offline RL | CQL / IQL (Phase 9 · 08) | 带保守正则化的 Q-learning。 |

2026 年论文中你读到的 90% 的"RL"都是 Q-learning 或 SARSA 的某种扩展。在深入阅读之前，先把表格更新刻进肌肉记忆。

## 交付

Save as `outputs/skill-td-agent.md`:

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。绘制 2,000 个 episode 的学习曲线（每 100 个 episode 的平均回报）。谁收敛更快？
2. **中等。** 构建悬崖行走环境（4×12，最后一行是悬崖，奖励 -100 并重置到起点）。比较 Q-learning 和 SARSA 的最终策略。截图各自走的路径。哪个更靠近悬崖？
3. **困难。** 实现 Double Q-learning。在噪声奖励 GridWorld（每步奖励加高斯噪声 σ=5）上，展示 Q-learning 对 `V*(0,0)` 有显著过估计，而 Double Q-learning 没有。

## 关键术语

| 术语 | 通俗说法 | 精确含义 |
|------|-----------------|-----------------------|
| TD error | "更新信号" | `δ = r + γ V(s') - V(s)`，自举残差。 |
| TD(0) | "一步 TD" | 每次转移后只使用下一个状态的估计来更新。 |
| Q-learning | "Off-policy RL 101" | 对下一状态动作取 `max` 的 TD 更新；无论行为策略如何都学习 `Q*`。 |
| SARSA | "On-policy Q-learning" | 使用实际下一个动作的 TD 更新；学习当前 ε-greedy π 的 `Q^π`。 |
| Expected SARSA | "低方差 SARSA" | 将采样的 `a'` 替换为其在 π 下的期望。 |
| GLIE | "正确的探索调度" | Greedy in the Limit with Infinite Exploration；Q-learning 收敛所需。 |
| Bootstrapping | "在目标中使用当前估计" | 区分 TD 和 MC 的关键。偏差的来源但大幅降低方差。 |
| Maximization bias | "Q-learning 过估计" | 对噪声估计取 `max` 向上偏差；由 Double Q-learning 修复。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文和收敛证明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 修复 maximization bias。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — Expected SARSA 的动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 创造 SARSA 名称的论文（当时称为"modified connectionist Q-learning"）。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 将 TD(0) 推广到 TD(n)，从 Q-learning 到 eligibility traces 再到 PPO 中 GAE 的路径。
