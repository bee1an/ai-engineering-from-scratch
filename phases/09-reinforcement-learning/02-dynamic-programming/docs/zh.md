# 动态规划——策略迭代与值迭代

> 动态规划是"开挂的 RL"。你已经知道转移函数和奖励函数；只需迭代 Bellman 方程直到 `V` 或 `π` 不再变化。它是所有基于采样的方法试图逼近的基准。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75 minutes

## 问题

你有一个已知模型的 MDP：可以对任意状态-动作对查询 `P(s' | s, a)` 和 `R(s, a, s')`。库存管理者知道需求分布。棋盘游戏有确定性转移。一个 gridworld 只需四行 Python。你有一个*模型*。

Model-free RL（Q-learning、PPO、REINFORCE）是为没有模型的情况发明的——你只能从环境中采样。但当你确实有模型时，存在更快、更好的方法：动态规划。Bellman 在 1957 年设计了它们。它们至今仍定义着正确性：当人们说"这个 MDP 的最优策略"时，他们指的就是 DP 会返回的策略。

2026 年你需要它们有三个原因。第一，RL 研究中的每个表格环境（GridWorld、FrozenLake、CliffWalking）都用 DP 求解以产生金标准策略。第二，精确值让你能*调试*采样方法：如果 Q-learning 对 `V*(s_0)` 的估计与 DP 答案相差 30%，那你的 Q-learning 有 bug。第三，现代离线 RL 和规划方法（MCTS、AlphaZero 的搜索、Phase 9 · 10 中的 model-based RL）都在学习到的或给定的模型上迭代 Bellman backup。

## 概念

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**两个算法，都是 Bellman 上的不动点迭代。**

**策略迭代。** 交替两个步骤直到策略不再变化。

1. *评估：* 给定策略 `π`，通过反复应用 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` 计算 `V^π` 直到收敛。
2. *改进：* 给定 `V^π`，令 `π` 关于 `V^π` 贪心：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛是有保证的，因为 (a) 每次改进步骤要么保持 `π` 不变，要么严格增加某个状态的 `V^π`，(b) 确定性策略的空间是有限的。即使对于大状态空间，通常也在约 5-20 次外层迭代内收敛。

**值迭代。** 将评估和改进合并为一次扫描。应用 Bellman *最优性*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直到 `max_s |V_{new}(s) - V(s)| < ε`。最后通过取贪心动作提取策略。每次迭代严格更快——没有内层评估循环——但通常需要更多迭代才能收敛。

**广义策略迭代（GPI）。** 统一框架。价值函数和策略锁定在双向改进循环中；任何驱动两者趋向相互一致的方法（异步值迭代、修改的策略迭代、Q-learning、actor-critic、PPO）都是 GPI 的实例。

**为什么 `γ < 1` 很重要。** Bellman 算子在上确界范数下是 `γ`-压缩的：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。压缩意味着唯一不动点和几何收敛。去掉 `γ < 1` 你就失去了保证——你需要有限视野或吸收终止状态。

## 动手构建

### 第 1 步：构建 GridWorld MDP 模型

使用 Lesson 01 中相同的 4×4 GridWorld。我们添加一个随机变体：以概率 `0.1` 智能体滑向随机的垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回 `(s', r, p)` 的列表。这就是整个模型。

### 第 2 步：策略评估

给定策略 `π(s) = {action: prob}`，迭代 Bellman 方程直到 `V` 不再变化：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 第 3 步：策略改进

用关于 `V` 的贪心策略替换 `π`。如果 `π` 没有变化，返回——我们已经到达最优。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 第 4 步：将它们组合在一起

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4 上的典型收敛：4-6 次外层迭代。输出 `V*(0,0) ≈ -6` 和一个严格减少步数的策略。

### 第 5 步：值迭代（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

相同的不动点，更少的代码。

## 常见陷阱

- **忘记处理终止状态。** 如果对吸收状态应用 Bellman，它仍然会选出一个"最佳动作"但什么都不改变。用 `if s == terminal: V[s] = 0` 来保护。
- **上确界范数 vs L2 收敛。** 使用 `max |V_new - V|`，而不是平均值。理论保证是基于上确界范数的。
- **原地更新 vs 同步更新。** 原地更新 `V[s]`（Gauss-Seidel）比使用单独的 `V_new` 字典（Jacobi）收敛更快。生产代码使用原地更新。
- **策略平局。** 如果两个动作有相同的 Q 值，`argmax` 可能每次迭代打破平局的方式不同，导致"策略稳定"检查振荡。使用稳定的平局打破规则（固定顺序中的第一个动作）。
- **状态空间爆炸。** DP 每次扫描是 `O(|S| · |A|)`。适用于约 10⁷ 个状态以内。超过这个规模，你需要函数逼近（Phase 9 · 05 起）。

## 实际应用

2026 年，DP 是正确性基准和规划器的内循环：

| 用例 | 方法 |
|----------|--------|
| 精确求解小型表格 MDP | 值迭代（更简单）或策略迭代（更少外层步骤） |
| 验证 Q-learning / PPO 实现 | 在玩具环境上与 DP 最优 V* 比较 |
| Model-based RL (Phase 9 · 10) | 在学习到的转移模型上做 Bellman backup |
| AlphaZero / MuZero 中的规划 | Monte Carlo Tree Search = 异步 Bellman backup |
| Offline RL (CQL, IQL) | Conservative Q-iteration——带 OOD 动作惩罚的 DP |

每当有人说"最优价值函数"，他们指的就是"DP 不动点"。当你在论文中看到 `V*` 或 `Q*` 时，想象这个循环。

## 交付

Save as `outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行值迭代，`γ ∈ {0.9, 0.99}`。多少次扫描直到 `max |ΔV| < 1e-6`？将 `V*` 打印为 4×4 网格。
2. **中等。** 在*随机* GridWorld（滑动概率 `0.1`）上比较策略迭代和值迭代。统计：扫描次数、墙钟时间、最终 `V*(0,0)`。哪个在迭代次数上收敛更快？在墙钟时间上呢？
3. **困难。** 构建修改的策略迭代：在评估步骤中只运行 `k` 次扫描而不是到收敛。绘制 `V*(0,0)` 误差 vs `k`（`k ∈ {1, 2, 5, 10, 50}`）的曲线。这条曲线告诉你关于评估/改进权衡的什么信息？

## 关键术语

| 术语 | 通俗说法 | 精确含义 |
|------|-----------------|-----------------------|
| Policy iteration | "DP 算法" | 交替评估（`V^π`）和改进（关于 `V^π` 的贪心 `π`）直到策略不再变化。 |
| Value iteration | "更快的 DP" | 在一次扫描中应用 Bellman 最优性 backup；几何收敛到 `V*`。 |
| Bellman operator | "那个递推" | `(T V)(s) = max_a Σ P (r + γ V(s'))`；上确界范数下的 `γ`-压缩。 |
| Contraction | "为什么 DP 收敛" | 任何满足 `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` 的算子 `T` 都有唯一不动点。 |
| GPI | "一切都是 DP" | 广义策略迭代：任何驱动 `V` 和 `π` 趋向相互一致的方法。 |
| Synchronous update | "Jacobi 风格" | 整个扫描中使用旧的 `V`；分析清晰但更慢。 |
| In-place update | "Gauss-Seidel 风格" | 在更新过程中使用 `V`；实践中收敛更快。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — 策略迭代和值迭代的经典呈现。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — 压缩映射论证的严格处理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 修改的策略迭代及其收敛分析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 策略迭代的原始论文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — 从 DP 到近似 DP / 深度 RL 的桥梁，后续每节课都会用到。
