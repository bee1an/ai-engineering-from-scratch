# 蒙特卡洛方法——从完整 Episode 中学习

> 动态规划需要模型。蒙特卡洛只需要 episode。运行策略，观察回报，取平均。RL 中最简单的想法——也是解锁后续一切的那个。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75 minutes

## 问题

动态规划很优雅，但它假设你可以对每个状态和动作查询 `P(s' | s, a)`。现实世界中几乎没有什么是这样工作的。机器人无法解析地计算关节力矩后相机像素的分布。定价算法无法对每种可能的客户反应进行积分。LLM 无法枚举一个 token 之后的所有可能续写。

你需要一种只需要能从环境中*采样*的方法。运行策略。得到一条轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用它来估计价值。这就是蒙特卡洛方法。

从 DP 到 MC 的转变在哲学上很重要：我们从*已知模型 + 精确 backup* 转向*采样 rollout + 平均回报*。方差跳升了，但适用性爆炸了。这节课之后的每个 RL 算法——TD、Q-learning、REINFORCE、PPO、GRPO——本质上都是蒙特卡洛估计器，有时在上面叠加了自举。

## 概念

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**核心思想，一行话：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 是在策略 `π` 下访问 `s` 后观察到的回报。

**First-visit vs every-visit MC。** 给定一个多次访问状态 `s` 的 episode，first-visit MC 只计算第一次访问的回报；every-visit MC 计算所有访问。两者在极限下都是无偏的。First-visit 更容易分析（iid 样本）。Every-visit 每个 episode 使用更多数据，实践中通常收敛更快。

**增量均值。** 不存储所有回报，而是更新运行平均：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重新组织：`V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。将 `1/n` 换成常数步长 `α ∈ (0, 1)`，你就得到一个能跟踪 `π` 变化的非平稳 MC 估计器。这个改动就是从 MC 到 TD 再到每个现代 RL 算法的全部跳跃。

**探索现在成了问题。** DP 通过枚举触及每个状态。MC 只看到策略访问的状态。如果 `π` 是确定性的，状态空间的整个区域永远不会被采样，它们的价值估计永远停在零。三种修复方法，按历史顺序：

1. **Exploring starts。** 每个 episode 从随机的 (s, a) 对开始。保证覆盖；实践中不现实（你不能把机器人"重置"到任意状态）。
2. **ε-greedy。** 关于当前 Q 贪心行动，但以概率 `ε` 选择随机动作。所有状态-动作对渐近地都会被采样。
3. **Off-policy MC。** 在行为策略 `μ` 下收集数据，通过重要性采样学习目标策略 `π`。方差高，但它是通向 DQN 等 replay-buffer 方法的桥梁。

**Monte Carlo Control。** 评估 → 改进 → 评估，就像策略迭代一样，但评估是基于采样的：

1. 运行 `π`，得到一个 episode。
2. 从观察到的回报更新 `Q(s, a)`。
3. 令 `π` 关于 `Q` 为 ε-greedy。
4. 重复。

在温和条件下（每对被无限次访问，`α` 满足 Robbins-Monro），以概率 1 收敛到 `Q*` 和 `π*`。

## 动手构建

### 第 1 步：rollout → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

不需要模型，只需要 `env.reset()` 和 `env.step(s, a)`。与 gym 环境相同的接口，只是精简了。

### 第 2 步：计算回报（反向扫描）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一次遍历，`O(T)`。反向递推 `G_t = r_{t+1} + γ G_{t+1}` 避免了重复求和。

### 第 3 步：first-visit MC 评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行代码完成工作：标记状态为首次访问，增加计数，更新运行均值。

### 第 4 步：ε-greedy MC control（on-policy）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 第 5 步：与 DP 金标准比较

你的 MC 估计的 `V^π` 应该在 episodes → ∞ 时与 Lesson 02 的 DP 结果一致。实践中：4×4 GridWorld 上 50,000 个 episode 可以让你与 DP 答案相差 `~0.1` 以内。

## 常见陷阱

- **无限 episode。** MC 要求 episode *终止*。如果你的策略可能永远循环，设置 `max_steps` 上限并将上限视为隐式失败。随机策略在 GridWorld 上经常超时——这是正常的，只要确保正确计数。
- **方差。** MC 使用完整回报。在长 episode 上，方差巨大——末尾一个不幸的奖励会以相同幅度移动 `V(s_0)`。TD 方法（Lesson 04）通过自举来削减这个问题。
- **状态覆盖。** 在新鲜 Q 上有平局的贪心 MC 只会尝试一个动作。你*必须*探索（ε-greedy、exploring starts、UCB）。
- **非平稳策略。** 如果 `π` 在变化（如 MC control 中），旧的回报来自不同的策略。常数 α MC 能处理这个；样本平均 MC 不能。
- **Off-policy 重要性采样。** 权重 `π(a|s)/μ(a|s)` 沿轨迹相乘。方差随视野爆炸。用 per-decision weighted IS 来限制，或切换到 TD。

## 实际应用

2026 年蒙特卡洛方法的角色：

| 用例 | 为什么用 MC |
|----------|--------|
| Short-horizon games (blackjack, poker) | Episode 自然终止；回报干净。 |
| Offline evaluation of a logged policy | 对存储的轨迹取折扣回报的平均。 |
| Monte Carlo Tree Search (AlphaZero) | 从树叶节点的 MC rollout 指导选择。 |
| LLM RL evaluation | 对给定策略的采样补全计算平均奖励。 |
| PPO 中的 baseline 估计 | 优势目标 `A_t = G_t - V(s_t)` 使用 MC 的 `G_t`。 |
| 教学 RL | 最简单的实际有效的算法——去掉自举看到核心。 |

现代深度 RL 算法（PPO、SAC）通过 `n`-step returns 或 GAE 在纯 MC（完整回报）和纯 TD（一步自举）之间插值。两个端点是同一个估计器的实例。

## 交付

Save as `outputs/skill-mc-evaluator.md`:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现均匀随机策略的 first-visit MC 评估。运行 10,000 个 episode。绘制 `V(0,0)` 随 episode 数量变化的曲线，对照 DP 答案。
2. **中等。** 实现 ε-greedy MC control，`ε ∈ {0.01, 0.1, 0.3}`。比较 20,000 个 episode 后的平均回报。曲线是什么样的？偏差-方差权衡在哪里？
3. **困难。** 实现带重要性采样的 *off-policy* MC：在均匀随机策略 `μ` 下收集数据，估计确定性最优策略 `π` 的 `V^π`。比较 plain IS vs per-decision IS vs weighted IS。哪个方差最低？

## 关键术语

| 术语 | 通俗说法 | 精确含义 |
|------|-----------------|-----------------------|
| Monte Carlo | "随机采样" | 通过对分布的 iid 样本取平均来估计期望。 |
| Return `G_t` | "未来奖励" | 从步骤 `t` 到 episode 结束的折扣奖励和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| First-visit MC | "每个状态只计一次" | 只有 episode 中的首次访问贡献价值估计。 |
| Every-visit MC | "使用所有访问" | 每次访问都贡献；略有偏差但样本效率更高。 |
| ε-greedy | "探索噪声" | 以概率 `1-ε` 选贪心动作；以概率 `ε` 选随机动作。 |
| Importance sampling | "修正从错误分布采样" | 通过 `π(a\|s)/μ(a\|s)` 乘积重新加权回报，从 `μ` 数据估计 `V^π`。 |
| On-policy | "从自己的数据学习" | 目标策略 = 行为策略。Vanilla MC、PPO、SARSA。 |
| Off-policy | "从别人的数据学习" | 目标策略 ≠ 行为策略。重要性采样 MC、Q-learning、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 经典处理。
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — first-visit vs every-visit 分析。
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — off-policy MC 和方差控制。
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 现代低方差 IS 估计器。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD 自我对弈收敛到超人水平的首个大规模实证演示；本 phase 后半部分每节课的概念前身。
