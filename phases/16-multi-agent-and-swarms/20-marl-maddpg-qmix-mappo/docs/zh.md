# 多智能体强化学习 — MADDPG, QMIX, MAPPO

> 多智能体协调的强化学习传承，在 2026 年仍然影响 LLM 智能体系统。**MADDPG**（Lowe et al., NeurIPS 2017, arXiv:1706.02275）引入了集中训练、分散执行（CTDE）：每个 critic 在训练时看到所有智能体的状态和动作；测试时只运行本地 actor。适用于合作、竞争和混合设置。**QMIX**（Rashid et al., ICML 2018, arXiv:1803.11485）是带单调混合网络的值分解；每智能体 Q 组合成联合 Q 使 `argmax` 干净地分布——在 StarCraft Multi-Agent Challenge（SMAC）上占主导。**MAPPO**（Yu et al., NeurIPS 2022, arXiv:2103.01955）是带集中值函数的 PPO；在 particle-world、SMAC、Google Research Football、Hanabi 上以最少调参"出人意料地有效"。这些支撑了必须分散行动的智能体团队的策略训练。MAPPO 是 **2026 年默认的合作 MARL 基线**。本课从小型网格世界玩具构建每种方法，在接触 LLM 智能体训练之前将三个想法植入肌肉记忆。

**Type:** Learn
**Languages:** Python (stdlib, small NumPy-free implementations)
**Prerequisites:** Phase 09 (Reinforcement Learning), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~90 minutes

## 问题

LLM 智能体系统越来越多地训练智能体间协调策略：何时让步、何时行动、调用哪个同伴。告诉你如何训练这些策略的文献是多智能体强化学习（MARL），它早于 LLM 浪潮且有一小组主导算法。

没有模式词汇表读 MARL 论文很痛苦。集中训练与分散执行（CTDE）、值分解和集中 critic 不是流行语——它们是对特定问题的特定回答：

- 独立 RL（每个智能体单独学习）从每个智能体的角度看是非平稳的。糟糕。
- 集中 RL（一个智能体控制所有）不可扩展且违反执行约束。
- CTDE 兼得两者之长：用全局信息训练，用本地策略部署。

## 概念

### 论文使用的三个环境

- **Particle World（multi-agent particle env）。** 简单 2D 物理，合作/竞争任务。MADDPG 的原始测试平台。
- **StarCraft Multi-Agent Challenge（SMAC）。** 合作微操，部分观察。QMIX 的测试平台。离散动作，连续状态。
- **Google Research Football, Hanabi, MPE。** MAPPO 基线。

不同环境有不同的动作/观察类型。算法相应选择。

### MADDPG（2017）— CTDE 模式

每个智能体 `i` 有一个 actor `mu_i(o_i)` 将自身观察映射到动作。每个智能体还有一个 critic `Q_i(x, a_1, ..., a_n)` 在训练时看到所有观察和所有动作。Actor 通过对 critic 评估的策略梯度更新。

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

为什么 CTDE：训练时我们知道所有人的动作；我们用这个来减少每个 critic 的方差。部署时每个智能体只看到 `o_i` 并调用 `mu_i(o_i)`。

失败模式：critic 随 N 个智能体增长（输入包含所有动作）。没有近似无法扩展到 ~10 个智能体以上。

### QMIX（2018）— 值分解

仅合作。全局奖励是每智能体 Q 值的单调函数之和：

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

单调性保证 `argmax_a Q_tot` 可以由每个智能体独立选择 `argmax_{a_i} Q_i` 来计算。这**恰好是你需要的分散执行属性**。训练时，混合网络从每智能体 Q 产生 `Q_tot`。

为什么 QMIX 在 SMAC 上赢：合作 StarCraft 微操有同质智能体、本地观察、全局奖励——完美适合值分解。

失败模式：单调性约束是限制性的；某些任务的奖励结构不是单调可分解的（一个智能体为团队牺牲）。扩展（QTRAN, QPLEX）放松了这一点。

### MAPPO（2022）— 被忽视的默认选择

Multi-Agent PPO：带集中值函数的 PPO。每个智能体有自己的策略；所有智能体共享（或有各自的）看到完整状态的值函数。Yu et al. 2022 在五个基准上将 MAPPO 与 MADDPG、QMIX 及其扩展进行了对比：

- MAPPO 在 particle-world、SMAC、Google Research Football、Hanabi、MPE 上匹配或击败 off-policy MARL 方法。
- 需要最少的超参数调优。
- 训练稳定；跨种子可复现。

社区在这篇论文之前低估了 on-policy MARL。在 2026 年，MAPPO 是合作 MARL 的默认基线；任何新方法都必须击败它。

### 为什么 LLM 智能体工程师应该关心

三个直接用途：

1. **路由器训练。** 一个元智能体选择哪个子智能体处理任务。这是一个有 N 个分散子智能体和一个集中路由器的 MARL 问题。MAPPO 适合。
2. **角色涌现。** 在生成式智能体模拟中，训练智能体随时间采用互补角色是一个伪装的 MARL 问题。QMIX 风格的值分解通过构造强制互补性。
3. **多智能体工具使用。** 当智能体共享工具并竞争预算时，通过 CTDE 训练它们产生尊重资源约束的可部署本地策略。

实际注意事项：在 2026 年，大多数生产 LLM 智能体系统通过提示而非训练来设定策略。MARL 在你有（a）大量交互数据、（b）清晰的奖励信号、（c）愿意投资训练基础设施时才适用。

### CTDE 作为超越 RL 的设计模式

即使不训练，CTDE 也是一个有用的架构模式：

- 在*设计*时，假设完整团队可见性。
- 在*运行时*，强制分散执行：每个智能体只看到 `o_i`。

这个模式迫使你保持每智能体状态显式，并预先考虑部分可观察性。许多生产多智能体系统默默假设到处共享状态——CTDE 纪律防止这一点。

### 非平稳性问题

当多个智能体同时学习时，每个智能体的环境（包括其他人的策略）是非平稳的。经典单智能体 RL 证明失效。本课的 MARL 算法都解决了这个问题：

- MADDPG：全局 critic 看到所有动作，所以其值估计是平稳的。
- QMIX：值分解将学习移到联合 Q 空间，最优性在那里是良定义的。
- MAPPO：集中值函数抑制来自其他人策略变化的方差。

在 LLM 智能体系统中，非平稳性表现为"我的智能体上个月还好用，现在上游那个智能体变了，我的就出问题了"。用 CTDE 训练 MARL 是原则性修复；提示级修复更快但不那么持久。

### 本课不涵盖的内容

训练实际网络是 Phase 09 的主题。本课构建脚本化策略版本，在没有梯度更新的情况下演示 CTDE、值分解和集中值模式。目标是在你拿起完整 MARL 库（PyMARL, MARLlib, RLlib multi-agent）之前内化这些模式。

## 动手构建

`code/main.py` 在一个微型 2 智能体合作网格世界上实现三种模式演示：

- 环境：4x4 网格上的 2 个智能体，一个奖励颗粒。奖励 = 1 如果任何智能体到达颗粒；任务结束。
- `IndependentAgents` — 每个智能体将其他人视为环境。基线。
- `MADDPGStyle` — 集中 critic 计算联合值；actor 策略从中更新。脚本化策略改进。
- `QMIXStyle` — 带单调混合器的值分解。
- `MAPPOStyle` — 集中值函数；策略对共享基线更新。

四种都运行相同的 episode 并报告平均到达目标步数。CTDE 变体收敛到比独立基线更短的路径。

运行：

```
python3 code/main.py
```

预期输出：独立智能体平均约 6 步；CTDE 变体收敛到约 3.5 步（4x4 网格的最优是 3）。尽管是脚本化策略，模式差异仍然显现。

## 使用方式

`outputs/skill-marl-picker.md` 是一个技能，为给定的多智能体任务选择 MARL 算法：合作 vs 竞争、同质 vs 异质、动作空间类型、规模、奖励信号。

## 上线清单

MARL 在生产中很少见。当你确实使用时：

- **从 MAPPO 开始。** 2022 年的论文确立了它作为基线；先复现它可以节省数周追逐更花哨方法的时间。
- **记录每个智能体的观察和动作流。** 没有每智能体追踪调试 MARL 是无望的。
- **分离训练代码和执行代码。** CTDE 是一种纪律；让执行路径真的只看到 `o_i`。
- **奖励塑形警告。** MARL 对奖励设计极其敏感。塑形中一个协调 bug，智能体就学会利用它。运行对抗测试。
- **对于 LLM 智能体**，先考虑提示级策略。只有当交互数据 + 奖励信号 + 基础设施都具备时才投资 MARL 训练。

## 练习

1. 运行 `code/main.py`。测量独立和 MAPPO 风格智能体之间的到达目标步数差距。在 6x6 网格上差距是增大还是缩小？
2. 实现竞争变体：两个智能体，一个颗粒，只有先到达的获得奖励。哪种模式干净地处理竞争？历史上是 MADDPG。
3. 阅读 MADDPG（arXiv:1706.02275）第 3 节。用你自己的话以伪代码符号实现精确的 critic 更新规则。
4. 阅读 MAPPO（arXiv:2103.01955）。为什么作者论证集中值 + PPO 在他们的基准上击败 off-policy MARL？列出三个最强的主张。
5. 将 CTDE 作为设计模式应用于一个假设的 LLM 智能体系统（如研究智能体 + 总结器 + 编码器）。设计时可用但运行时不可用的联合信息是什么？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| MARL | "Multi-Agent RL" | 多智能体系统的强化学习。 |
| CTDE | "集中训练，分散执行" | 用全局信息训练；用本地策略部署。 |
| MADDPG | "Multi-Agent DDPG" | CTDE，每智能体 critic 看到所有观察 + 动作。 |
| QMIX | "值分解" | 每智能体 Q 的单调混合。仅合作。 |
| MAPPO | "Multi-Agent PPO" | 带集中值函数的 PPO。2026 默认基线。 |
| Value decomposition | "个体 Q 之和" | 联合 Q 表示为每智能体 Q 的单调函数。 |
| Non-stationarity | "移动目标" | 每个智能体的环境随其他人学习而变化。核心 MARL 问题。 |
| On-policy / off-policy | "从当前/回放中学习" | PPO 是 on-policy（MAPPO）；DDPG 和 Q-learning 是 off-policy。 |
| SMAC | "StarCraft Multi-Agent Challenge" | 合作微操基准；QMIX 的主场。 |

## 延伸阅读

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG; NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX; ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO; NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — MAPPO 结果的可读框架
- [SMAC repository](https://github.com/oxwhirl/smac) — StarCraft Multi-Agent Challenge
