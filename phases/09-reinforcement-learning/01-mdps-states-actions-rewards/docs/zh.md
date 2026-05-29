# MDP、状态、动作与奖励

> 马尔可夫决策过程由五个要素组成：状态、动作、转移、奖励、折扣因子。强化学习中的一切——Q-learning、PPO、DPO、GRPO——都在这个结构上做优化。学一次，后面的强化学习内容就能一路畅通。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**Time:** ~45 minutes

## 问题

你在写一个国际象棋机器人。或者一个库存规划器。或者一个交易智能体。或者训练推理模型的 PPO 循环。四个完全不同的领域，一个令人惊讶的事实：它们全都坍缩为同一个数学对象。

监督学习给你 `(x, y)` 对，让你拟合一个函数。强化学习不给标签——只有一连串的状态、你采取的动作，以及一个标量奖励。这步棋赢了吗？补货决策省钱了吗？交易盈利了吗？LLM 刚生成的 token 是否让评判者给出了更高的奖励？

在你把这些形式化之前，你无法从这个数据流中学习。"我看到了什么"、"我做了什么"、"接下来发生了什么"、"那有多好"——每一个都必须变成你可以推理的对象。这个形式化就是马尔可夫决策过程。Phase 9 中的每一个 RL 算法，包括最后的 RLHF 和 GRPO 循环，都在这个结构上做优化。

## 概念

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**五个对象。**

- **状态** `S`。智能体做决策所需的一切信息。在 GridWorld 中是格子位置。在国际象棋中是棋盘。在 LLM 中是上下文窗口加上任何记忆。
- **动作** `A`。可选的操作。上/下/左/右移动。下一步棋。生成一个 token。
- **转移** `P(s' | s, a)`。给定状态 `s` 和动作 `a`，下一个状态的分布。国际象棋中是确定性的，库存管理中是随机的，LLM 解码中几乎是确定性的。
- **奖励** `R(s, a, s')`。标量信号。赢 = +1，输 = -1。收入减去成本。GRPO 中的 log-likelihood ratio 项。
- **折扣因子** `γ ∈ [0, 1)`。未来奖励相对于当前的权重。`γ = 0.99` 对应约 100 步的视野；`γ = 0.9` 对应约 10 步。

**马尔可夫性质** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来只取决于当前状态。如果不满足，说明状态表示不完整——这不是方法的失败，而是状态设计的失败。

**策略与回报。** 策略 `π(a | s)` 将状态映射到动作分布。回报 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来奖励的折扣和。价值 `V^π(s) = E[G_t | s_t = s]` 是在策略 `π` 下从 `s` 出发的期望回报。Q 值 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是从特定动作开始的期望回报。每个 RL 算法都在估计这两者之一，然后据此改进 `π`。

**Bellman 方程。** Phase 9 中所有算法使用的不动点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

这些方程将期望回报分解为"这一步的奖励"加上"到达下一个状态的折扣价值"。递归的。Phase 9 中的每个算法要么迭代这个方程直到收敛（动态规划），要么从中采样（蒙特卡洛方法），要么用一步自举（时序差分）。

## 动手构建

### 第 1 步：一个小型确定性 MDP

一个 4×4 的 GridWorld。智能体从左上角出发，终点在右下角，每步奖励 -1，动作集合 `{up, down, left, right}`。见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行代码。这就是整个环境。确定性转移，恒定步惩罚，吸收终止状态。

### 第 2 步：执行一个策略

策略是从状态到动作分布的函数。最简单的：均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行随机策略 1000 次。平均回报大约在 -60 到 -80 之间（4×4 棋盘）。最优回报是 -6（直线向右下走）。缩小这个差距就是 Phase 9 的全部内容。

### 第 3 步：通过 Bellman 方程精确计算 `V^π`

对于小型 MDP，Bellman 方程是一个线性系统。枚举状态，应用期望，迭代直到值不再变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这就是迭代策略评估。它是 Sutton & Barto 中的第一个算法，也是后续所有 RL 方法的理论基础。

### 第 4 步：`γ` 是一个有物理意义的超参数

有效视野大约是 `1 / (1 - γ)`。`γ = 0.9` → 10 步。`γ = 0.99` → 100 步。`γ = 0.999` → 1000 步。

太低，智能体会短视。太高，信用分配变得嘈杂，因为许多早期步骤共同承担远期奖励的责任。LLM RLHF 通常使用 `γ = 1`，因为 episode 短且有界。控制任务使用 `0.95–0.99`。长视野策略游戏使用 `0.999`。

## 常见陷阱

- **非马尔可夫状态。** 如果你需要最近三个观测才能做决策，那么"状态"不仅仅是当前观测。修复方法：堆叠帧（DQN 在 Atari 上堆叠 4 帧）或使用循环状态（LSTM/GRU 处理观测序列）。
- **稀疏奖励。** 仅有胜负奖励在大状态空间中几乎不可能学习。塑造奖励（中间信号）或用模仿学习引导（Phase 9 · 09）。
- **奖励黑客。** 优化代理奖励往往产生病态行为。OpenAI 的赛艇智能体在圈子里转圈收集道具，而不是完成比赛。始终从目标结果定义奖励，而不是代理指标。
- **折扣因子设置错误。** 在无限视野任务上 `γ = 1` 会使每个值变为无穷大。始终用有限视野或 `γ < 1` 来限制。
- **奖励尺度。** {+100, -100} 与 {+1, -1} 的奖励给出相同的最优策略，但梯度幅度差异巨大。在接入 PPO/DQN 之前归一化到 `[-1, 1]` 左右。

## 实际应用

2026 年的技术栈在写代码之前，先把每个 RL 流水线归约为一个 MDP：

| 场景 | State | Action | Reward | γ |
|-----------|-------|--------|--------|---|
| Control (locomotion, manipulation) | Joint angles + velocities | Continuous torques | Task-specific shaped | 0.99 |
| Games (chess, Go, poker) | Board + history | Legal move | Win=+1 / loss=-1 | 1.0 (finite) |
| Inventory / pricing | Stock + demand | Order qty | Revenue - cost | 0.95 |
| RLHF for LLMs | Context tokens | Next token | Reward-model score at end | 1.0 (episode ~200 tokens) |
| GRPO for reasoning | Prompt + partial response | Next token | Verifier 0/1 at end | 1.0 |

在写任何训练循环之前先写出这五元组。大多数"RL 不工作"的 bug 报告都可以追溯到一个在纸面上就有问题的 MDP 建模。

## 交付

Save as `outputs/skill-mdp-modeler.md`:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 4×4 GridWorld 和随机策略 rollout。运行 10,000 个 episode。报告回报的均值和标准差。与最优回报（-6）比较。
2. **中等。** 对均匀随机策略运行 `policy_evaluation`，`γ ∈ {0.5, 0.9, 0.99}`。将每个 `V` 打印为 4×4 网格。解释为什么靠近终点的状态值随 `γ` 增大而增长更快。
3. **困难。** 将 GridWorld 变为随机的：每个动作以概率 `p = 0.1` 滑向相邻方向。重新评估均匀策略。`V[start]` 变好还是变差了？为什么？

## 关键术语

| 术语 | 通俗说法 | 精确含义 |
|------|-----------------|-----------------------|
| MDP | "强化学习的设置" | 满足马尔可夫性质的元组 `(S, A, P, R, γ)`。 |
| State | "智能体看到的东西" | 在所选策略类下，对未来动态的充分统计量。 |
| Policy | "智能体的行为" | 条件分布 `π(a \| s)` 或确定性映射 `s → a`。 |
| Return | "总奖励" | 从当前步开始的折扣和 `Σ γ^t r_t`。 |
| Value | "一个状态有多好" | 在 `π` 下从 `s` 出发的期望回报。 |
| Q-value | "一个动作有多好" | 在 `π` 下从 `s` 出发、首先执行动作 `a` 的期望回报。 |
| Bellman equation | "动态规划递推" | 将 value / Q 分解为一步奖励加上折扣后继价值的不动点方程。 |
| Discount `γ` | "未来 vs 当前" | 远期奖励的几何权重；有效视野 `~1/(1-γ)`。 |

## 延伸阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书。第 3 章覆盖 MDP 和 Bellman 方程；第 1 章阐述了支撑后续所有课程的奖励假说。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — Bellman 方程的起源。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 从深度 RL 角度出发的简洁 MDP 入门。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 运筹学领域关于 MDP 和精确求解方法的参考书。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — 将 MDP 作为动态规划特化的最清晰推导。
