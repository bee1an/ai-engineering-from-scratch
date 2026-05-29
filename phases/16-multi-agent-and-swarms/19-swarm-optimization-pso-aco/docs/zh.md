# 群体优化用于 LLM（PSO, ACO）

> 生物启发优化正在 LLM 领域回归。**LMPSO**（arXiv:2504.09247）使用 PSO，其中每个粒子的速度是一个提示，LLM 生成下一个候选；在结构化序列输出（数学表达式、程序）上效果良好。**Model Swarms**（arXiv:2410.11163）将每个 LLM 专家视为模型权重流形上的 PSO 粒子，报告在 9 个数据集上相比 12 个基线有 **13.3% 平均增益**，仅用 200 个实例。**SwarmPrompt**（ICAART 2025）混合 PSO + Grey Wolf 用于提示优化。**AMRO-S**（arXiv:2603.12933）是 ACO 启发的信息素专家用于多智能体 LLM 路由——**4.7 倍加速**，可解释的路由证据，质量门控异步更新将推理与学习解耦。本课在提示参数空间上实现 PSO，在智能体路由上实现 ACO，测量为什么这些经典算法适合 LLM 时代，以及何时不适合。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 minutes

## 问题

你有一个在任务评估上得分 62% 的提示。你想改进它。朴素的做法是无梯度手动调整，扩展性差。强化学习需要奖励信号和足够的 rollout 来训练。通过提示反向传播实际上不可能——提示是离散字符串，不是可微参数。

经典生物启发优化——PSO 用于连续搜索空间，ACO 用于路径选择——正是为这种场景设计的：无梯度、基于种群、每次评估成本低。将它们与 LLM 配对用于无梯度搜索步骤，你就得到一个出人意料地实用的优化器。

同样的模式适用于多智能体系统中的智能体*路由*。ACO 风格的信息素轨迹记录哪个智能体在哪种任务类型上表现最好，让路由器利用轨迹，并衰减信息素以便重新发现路由。

## 概念

### PSO 回顾（Kennedy & Eberhart 1995）

Particle Swarm Optimization：连续搜索空间中的粒子种群。每个粒子有位置 `x_i` 和速度 `v_i`。每次迭代：

```
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
evaluate fitness(x_i)
update p_best_i if improved
update g_best if global best
```

其中 `p_best` 是粒子自身最优，`g_best` 是群体最优，`w, c1, c2` 是惯性 + 认知 + 社会权重，`r1, r2` 是随机因子。

### PSO 用于 LLM 输出 — LMPSO

arXiv:2504.09247 将 PSO 适配于 LLM 生成的结构化输出（数学表达式、程序）。每个粒子是一个候选输出。速度是一个*提示*，描述如何将当前输出修改为向个人/全局最优靠近。LLM 从速度提示生成新输出。速度的"惯性"是类似"make small incremental changes"的提示。

这在以下情况下效果好：
- 输出是结构化的（可解析、可评估）。
- 适应度是自动的（测试运行、算术评估）。
- 种群小（~10-30 个粒子）所以总 LLM 调用可控。

当适应度需要人工审查时效果不好——每次迭代成本变得过高。

### Model Swarms

arXiv:2410.11163 将 PSO 从输出层移到*模型*层。每个"粒子"是一个专家 LLM（参数）。群体通过无梯度更新将参数移向集体最优。报告：在 9 个数据集上相比 12 个基线有 13.3% 平均增益，每次迭代仅 200 个实例。

关键洞察是 LLM 专家模型已经在共享参数流形中相邻（适配器权重、LoRA 增量）。在这个低维子空间上的 PSO 既便宜又有效。

### ACO 回顾（Dorigo 1992）

Ant Colony Optimization：蚂蚁遍历图；每条路径有信息素轨迹。蚂蚁移动概率按信息素强度加权。完成任务的蚂蚁按解决方案质量成比例地沉积信息素。信息素随时间衰减。

### AMRO-S — ACO 用于智能体路由

arXiv:2603.12933 使用 ACO 进行多智能体路由。每种任务类型是一个"目的地"；每个智能体是一条可能的路由。信息素加强产生好输出的路由。关键贡献：

- **可解释的路由证据。** 信息素强度是人类可读的信号。
- **质量门控异步更新。** 信息素仅在质量检查通过后更新，将推理与学习解耦。
- 在多智能体路由基准上 **4.7 倍加速**。

质量门控很重要：没有它，快但错的智能体积累信息素，系统锁定在坏路由上。

### 何时使用 PSO / ACO 用于 LLM

**使用 PSO 当：**
- 搜索空间是连续的或映射到连续参数（提示嵌入、LoRA 权重、数值生成参数）。
- 适应度便宜且自动。
- 种群可以小（10-30）。

**使用 ACO 当：**
- 你有路由或路径选择问题。
- 决策随时间强化（相同任务类型反复出现）。
- 你需要路由决策的可解释证据。

**两者都不用当：**
- 适应度需要人工审查（每次迭代太贵）。
- 搜索空间是离散和组合的，PSO 无法覆盖（改用遗传算法）。
- 实时决策需要严格延迟（PSO/ACO 相对于单次启发式收敛慢）。

### 为什么生物启发仍然赢

基于梯度的方法需要可微信号。LLM 输出和路由决策不是平凡可微的。伪梯度方法（强化学习路由器、DPO 风格提示调优器）有效但需要昂贵的训练。

PSO 和 ACO 只需要一个*评估器*函数。如果你能对候选输出或路由决策打分，你就能在空间上优化。这使得适用性门槛低得多。

### 实际限制

- **种群预算。** N 粒子 × T 迭代 × 每次评估成本。对于 ~$0.02/次的 LLM 评估，20 粒子 PSO 运行 50 次迭代花费 ~$20。相应规划。
- **探索 vs 利用。** 信息素衰减率和 PSO 惯性权衡；衰减太快 → 忘记解；太慢 → 卡在早期局部最优。
- **灾难性漂移。** 两种算法都可能收敛后又发散，如果适应度景观变化（新数据分布）。监控最优适应度稳定性。

## 动手构建

`code/main.py` 实现：

- `LMPSO` — 在数值提示参数（temperature、top_k 权重）上的 PSO。每个粒子的"LLM 生成"模拟为脚本化适应度函数。运行算法 30 次迭代并显示 g_best 收敛。
- `AMRO_S` — ACO 风格路由。3 个智能体，4 种任务类型，信息素矩阵，100 个路由任务。打印 (task_type → agent choices) 随时间的分布以显示轨迹形成。
- 比较：相同任务流上的随机路由 vs ACO 路由。测量质量和延迟。

运行：

```
python3 code/main.py
```

预期输出：
- LMPSO：g_best 适应度从随机改善到接近最优，经过 30 次迭代。
- AMRO-S：信息素表稳定在每种任务类型的正确智能体上；ACO 路由在质量上比随机好 ~30-40%，延迟也降低（更少重试）。

## 使用方式

`outputs/skill-swarm-optimizer.md` 帮助在 PSO、ACO、遗传算法和基于梯度的优化器之间选择，用于 LLM / 智能体优化问题。

## 上线清单

- **从小开始。** 10-20 个粒子，20-50 次迭代。只有当收敛曲线显示明确增益时才扩大。
- **记录每次迭代的信息素或 g_best。** 没有轨迹调试群体优化器很痛苦。
- **质量门控更新。** 特别是 ACO 路由：快但错的智能体不能积累信息素。
- **分布变化时重置衰减。** 当你的评估分布变化时，老化的信息素是过时的；重置或临时加倍衰减率。
- **限制每次迭代成本。** 发出每次迭代成本指标。每次迭代花费 $500 且增益 0.5% 的 PSO 不可上线。

## 练习

1. 运行 `code/main.py`。观察 LMPSO 收敛。变化种群大小 5, 10, 20, 50。在什么大小时收敛时间饱和？
2. 实现"灾难性漂移"实验：在第 30 次迭代后改变适应度函数。PSO 适应多快？重置 `p_best` 有帮助吗？
3. 向 AMRO-S 添加质量门控：仅在评估分数 > 0.7 的运行上沉积信息素。这如何改变收敛 vs 无门控版本？
4. 阅读 LMPSO（arXiv:2504.09247）。将论文的"velocity as a prompt"映射回你的数值速度。模拟中丢失了什么，保留了什么？
5. 阅读 AMRO-S（arXiv:2603.12933）。实现解耦的"推理快速路径"与异步信息素更新。这如何改变持续负载下的系统延迟？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| PSO | "Particle Swarm Optimization" | Kennedy-Eberhart 1995。基于种群的无梯度优化器。 |
| ACO | "Ant Colony Optimization" | Dorigo 1992。通过信息素轨迹的路径/路由优化。 |
| LMPSO | "PSO with LLM generation" | arXiv:2504.09247。速度是提示；LLM 产生候选。 |
| Model Swarms | "PSO on expert weights" | arXiv:2410.11163。模型参数子空间上的无梯度更新。 |
| AMRO-S | "ACO for agent routing" | arXiv:2603.12933。task-type × agent 的信息素矩阵。 |
| p_best / g_best | "个人/全局最优" | 每粒子和群体范围内迄今找到的最优解。 |
| Pheromone | "路由记忆" | 边上的强度；随时间衰减；按质量沉积。 |
| Quality-gated update | "只从好的运行中学习" | 信息素沉积以质量检查为条件。 |
| Catastrophic drift | "分布变化" | 适应度景观变化；旧的 p_best 和信息素变得过时。 |

## 延伸阅读

- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — 1995 PSO 论文
- [Dorigo — Ant Colony Optimization](https://www.aco-metaheuristic.org/about.html) — 1992 ACO 基础
- [LMPSO — Language Model Particle Swarm Optimization](https://arxiv.org/abs/2504.09247) — PSO 用于结构化 LLM 输出
- [Model Swarms — gradient-free LLM expert optimization](https://arxiv.org/abs/2410.11163) — 模型权重子空间上的 PSO
- [AMRO-S — ant-colony multi-agent routing](https://arxiv.org/abs/2603.12933) — 带质量门控的信息素驱动路由
