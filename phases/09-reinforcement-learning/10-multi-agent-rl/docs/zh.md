# 多智能体强化学习

> 单智能体 RL 假设环境是平稳的。把两个学习智能体放在同一个世界里，这个假设就崩了：每个智能体都是对方环境的一部分，而且双方都在变化。多智能体 RL 是一组让学习在 Markov 假设不再成立时仍能收敛的技巧。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**时间：** ~45 分钟

## 问题

一个机器人学习在房间里导航是单智能体 RL 问题。一支足球队不是。AlphaStar 对战星际争霸对手不是。一个竞价智能体市场不是。两辆车在十字路口协商不是。许多多对多的现实问题都不是。

在每个多智能体场景中，从任何一个智能体的视角看，其他智能体*就是*环境的一部分。当它们学习并改变行为时，环境变得非平稳。Markov 性质——"下一状态只取决于当前状态和我的动作"——被违反了，因为下一状态还取决于*其他*智能体选择了什么，而它们的策略是移动目标。

这打破了表格型收敛证明（Q-learning 的保证假设平稳环境）。也打破了朴素的深度 RL：智能体互相追逐形成循环，永远不收敛到稳定策略。你需要多智能体专用技术：集中训练/分散执行、反事实基线、联赛训练、自我对弈。

2026 年应用：机器人集群、交通路由、自动驾驶车队、市场模拟器、多智能体 LLM 系统（Phase 16），以及任何有多个智能玩家的游戏。

## 概念

![四种 MARL 范式：独立、集中 critic、自我对弈、联赛](../assets/marl.svg)

**形式化：Markov 博弈。** MDP 的推广：状态 `S`，联合动作 `a = (a_1, …, a_n)`，转移 `P(s' | s, a)`，以及每个智能体的奖励 `R_i(s, a, s')`。每个智能体 `i` 在自己的策略 `π_i` 下最大化自己的回报。如果奖励相同，则为**完全合作**。如果零和，则为**对抗**。如果混合，则为**一般和**。

**核心挑战：**

- **非平稳性。** 从智能体 `i` 的视角看，`P(s' | s, a_i)` 取决于 `π_{-i}`，而它在变化。
- **信用分配。** 共享奖励下，哪个智能体导致了它？
- **探索协调。** 智能体必须探索互补策略，而不是冗余地探索相同状态。
- **可扩展性。** 联合动作空间随 `n` 指数增长。
- **部分可观测性。** 每个智能体只看到自己的观测；全局状态是隐藏的。

**四种主流范式：**

**1. 独立 Q-learning / 独立 PPO（IQL, IPPO）。** 每个智能体学习自己的 Q 或策略，把其他智能体当作环境的一部分。简单，有时有效（特别是经验回放充当平滑的智能体建模技巧时）。理论收敛性：无。实践中：对松耦合任务可以，对紧耦合任务不行。

**2. 集中训练、分散执行（CTDE）。** 最常见的现代范式。每个智能体有自己的*策略* `π_i`，条件于局部观测 `o_i`——部署时标准的分散执行。*训练*期间，集中 critic `Q(s, a_1, …, a_n)` 条件于完整全局状态和联合动作。例子：
- **MADDPG**（Lowe et al. 2017）：每个智能体一个集中 critic 的 DDPG。
- **COMA**（Foerster et al. 2017）：反事实基线——问"如果我改选动作 `a'`，我的奖励会是多少？"——隔离我的贡献。
- **MAPPO** / **IPPO** with shared critic（Yu et al. 2022）：带集中价值函数的 PPO。2026 年合作 MARL 的主流。
- **QMIX**（Rashid et al. 2018）：值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，单调混合。

**3. 自我对弈。** 同一智能体的两个副本互相对弈。对手的策略*就是*我过去快照的策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。最适合零和博弈；训练信号是对称的。

**4. 联赛训练。** 自我对弈在一般和/对抗环境中的扩展：保持过去和当前策略的种群，从联赛中采样对手进行训练。添加 exploiter（专门击败当前最强）和 main exploiter（专门击败 exploiter）。AlphaStar（星际争霸 II）。当游戏存在"石头剪刀布"策略循环时需要。

**通信。** 允许智能体互相发送学习到的消息 `m_i`。在合作场景中有效。Foerster et al.（2016）展示了可微的智能体间通信可以端到端训练。今天基于 LLM 的多智能体系统（Phase 16）本质上用自然语言通信。

## 动手构建

本课使用一个 6×6 GridWorld，有两个合作智能体。它们从对角出发，必须到达共享目标。共享奖励：任一智能体还在移动时每步 `-1`，两者都到达时 `+10`。见 `code/main.py`。

### 步骤 1：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间是 `|A|² = 16`。全局状态是两个位置。

### 步骤 2：独立 Q-learning

每个智能体运行自己的 Q 表，以联合状态为键。每步：两者都选 ε-greedy 动作，收集联合转移，各自用共享奖励更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

在这个任务上有效，因为奖励是密集且对齐的。在紧耦合任务上失败（例如一个智能体必须*等待*另一个）。

### 步骤 3：集中 Q 与分解值更新

使用一个覆盖联合动作的 Q `Q(s, a_1, a_2)`。从共享奖励更新。通过边际化实现分散执行：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。用指数级联合动作空间换取*正确*的全局视角。

### 步骤 4：简单自我对弈（对抗 2 智能体）

同一智能体，两个角色。训练智能体 A 对抗智能体 B；每 `K` 个 episode 后，把 A 的权重复制到 B。对称训练，持续进步。微型版的 AlphaZero 配方。

## 常见陷阱

- **非平稳回放。** 独立智能体的经验回放比单智能体更差，因为旧转移是由现已过时的对手生成的。修复：按时间重标注或加权。
- **信用分配模糊。** 长 episode 后的共享奖励；无法明确说哪个智能体贡献了。修复：反事实基线（COMA），或每智能体奖励塑形。
- **策略漂移/追逐。** 每个智能体的最佳响应随对方的更新而变化。修复：集中 critic、慢学习率、或冻结一方轮流训练。
- **通过协调的奖励黑客。** 智能体找到设计者未预料的协调利用方式。拍卖智能体收敛到出价零。修复：仔细的奖励设计、行为约束。
- **探索冗余。** 两个智能体探索相同的状态-动作对。修复：每智能体 entropy bonus，或角色条件化。
- **联赛循环。** 纯自我对弈可能陷入优势循环。修复：使用多样化对手的联赛训练。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。用函数逼近近似；分解动作空间（每个智能体一个策略输出头）。

## 应用场景

2026 年 MARL 应用地图：

| 领域 | 方法 | 备注 |
|------|------|------|
| 合作导航/操作 | MAPPO / QMIX | CTDE；共享 critic + 分散 actor。 |
| 双人游戏（国际象棋、围棋、扑克） | Self-play with MCTS (AlphaZero) | 零和；对称训练。 |
| 复杂多人游戏（Dota、星际争霸） | League play + imitation pretraining | OpenAI Five, AlphaStar。 |
| 自动驾驶车队 | CTDE MAPPO / PPO with attention | 部分可观测；可变团队规模。 |
| 拍卖市场 | Game-theoretic equilibrium + RL | `n` → ∞ 时用 Mean-field RL。 |
| LLM 多智能体系统（Phase 16） | Natural-language comm + role conditioning | RL 循环在智能体规划层。 |

2026 年，MARL 最大的增长领域是基于 LLM 的：语言模型智能体集群在协商、辩论、构建软件。RL 表现为*轨迹级*输出上的偏好优化，而非 token 级（Phase 16 · 03）。

## 交付产出

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. **简单。** 在 2 智能体合作 GridWorld 上训练独立 Q-learning。多少个 episode 后平均 return > 0？绘制联合学习曲线。
2. **中等。** 添加一个"协调"任务：只有两个智能体在同一回合踏上目标时才算到达。独立 Q 还能收敛吗？什么会崩溃？
3. **困难。** 为 MAPPO 风格训练实现集中 critic，并在协调任务上比较与独立 PPO 的收敛速度。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Markov game | "多智能体 MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体有自己的奖励。 |
| CTDE | "集中训练、分散执行" | 训练时联合 critic；每个智能体的策略只用局部观测。 |
| IPPO | "独立 PPO" | 每个智能体单独运行 PPO。简单基线；常被低估。 |
| MAPPO | "多智能体 PPO" | 带集中价值函数（条件于全局状态）的 PPO。 |
| QMIX | "单调值分解" | `Q_tot = f_monotone(Q_1, …, Q_n)` 允许分散 argmax。 |
| COMA | "反事实多智能体" | 优势 = 我的 Q 减去对我的动作边际化后的期望 Q。 |
| Self-play | "智能体对战过去的自己" | 单一智能体，两个角色；零和博弈的标准方法。 |
| League play | "种群训练" | 缓存过去策略，从池中采样对手；处理策略循环。 |

## 延伸阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 带集中 critic 的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 用于信用分配的反事实基线。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 带单调性的值分解。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 在 MARL 中出人意料地强。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模联赛训练。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和博弈中的纯自我对弈。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书对多智能体设置和 CTDE 要解决的非平稳性问题的简短讨论。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 涵盖合作、竞争和混合 MARL 及收敛结果的综述。
