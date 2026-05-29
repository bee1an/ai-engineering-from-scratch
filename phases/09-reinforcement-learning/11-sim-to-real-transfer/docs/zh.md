# Sim-to-Real 迁移

> 在模拟器中训练的策略如果在硬件上失败，那它只是记住了模拟器。Domain randomization、domain adaptation 和 system identification 是让学习到的控制器跨越现实差距的三个工具。

**类型：** 学习
**语言：** Python
**前置课程：** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)
**时间：** ~45 分钟

## 问题

训练真实机器人既慢、又危险、又昂贵。一个双足机器人需要数百万训练 episode 才能学会走路；真实双足机器人哪怕摔倒一次就会损坏硬件。仿真给你无限重置、确定性可复现、并行环境，且没有物理损伤。

但模拟器是错的。轴承的摩擦力比 MuJoCo 模型大。相机有模拟器未包含的镜头畸变。电机有 99% 仿真模型跳过的延迟、间隙和饱和。风、灰尘和变化的光照会破坏在无菌渲染上训练的策略。**现实差距**——仿真分布与真实分布之间的系统性差异——是部署 RL 用于机器人的核心问题。

你需要一个*对 sim-to-real 分布偏移鲁棒*的策略。三种历史方法：随机化模拟器（domain randomization）、用少量真实数据适应策略（domain adaptation / 微调）、或识别真实系统参数并匹配（system identification）。2026 年主流配方将三者结合大规模并行仿真（Isaac Sim、Isaac Lab、Mujoco MJX on GPU）。

## 概念

![三种 sim-to-real 范式：domain randomization、adaptation、system identification](../assets/sim-to-real.svg)

**Domain Randomization（DR）。** Tobin et al. 2017, Peng et al. 2018。训练期间，随机化每个可能在真实机器人上不同的仿真参数：质量、摩擦系数、电机 PD 增益、传感器噪声、相机位置、光照、纹理、接触模型。策略学习关于"今天在哪个仿真中"的条件分布，并在整个范围内泛化。如果真实机器人落在训练包络内，策略就能工作。

- **优点：** 不需要真实数据。一个配方，多个机器人。
- **缺点：** 过度随机化的训练产生"通用"但过于保守的策略。太多噪声 ≈ 太多正则化。

**System Identification（SI）。** 在训练前将模拟器参数拟合到真实世界数据。如果你能测量真实机器人的关节摩擦力，就把它插入仿真。然后训练一个期望这些值的策略。需要访问真实系统，但直接减少现实差距。

- **优点：** 精确、低噪声的训练目标。
- **缺点：** 残余模型误差对策略不可见；小的未识别效应（如电机死区）仍会破坏部署。

**Domain Adaptation。** 在仿真中训练，用少量真实数据微调。两种风格：

- **Real2Sim2Real：** 用真实 rollout 学习残差模拟器 `f(s, a, z) - f_sim(s, a)`，在修正后的仿真中训练。不需要太多真实数据就能缩小差距。
- **观测适应：** 训练一个策略，通过学习的特征提取器（如 GAN pixel-to-pixel）将真实观测映射到类仿真观测。控制器保持在仿真中。

**特权学习 / teacher-student。** Miki et al. 2022（ANYmal 四足机器人）。在仿真中训练一个*教师*，它可以访问特权信息（真实摩擦力、地形高度、IMU 漂移）。蒸馏一个只看真实传感器观测的*学生*。学生学会从历史中推断特权特征，在物理参数变化时保持鲁棒。

**大规模并行仿真。** 2024-2026。Isaac Lab、Mujoco MJX、Brax 都在单个 GPU 上运行数千个并行机器人。PPO 配合 4,096 个并行人形机器人在数小时内收集数年的经验。随着训练分布变宽，"现实差距"缩小；当这 4,096 个环境各有不同的随机化参数时，DR 几乎是免费的。

**2026 年真实世界配方（四足行走示例）：**

1. 大规模并行仿真，domain-randomized 重力、摩擦力、电机增益、载荷。
2. 教师策略用特权信息训练（地形图、身体速度真值）。
3. 学生策略从教师蒸馏，只使用本体感觉（腿部关节编码器）。
4. 可选的观测适应，通过真实 IMU 上的 autoencoder。
5. 部署。在 10+ 种环境上 zero-shot。如果失败，做几分钟带安全约束的 PPO 真实世界微调。

## 动手构建

本课的代码是在带*噪声*转移的 GridWorld 上演示 domain randomization 的微型示例。我们训练一个在"仿真"中经历随机化滑动概率的策略，并在训练期间从未见过的滑动水平的"真实"环境上评估。这个形状直接映射到 MuJoCo 到硬件的迁移。

### 步骤 1：参数化仿真

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是模拟器暴露的参数。在真实机器人中它可以是摩擦力、质量、电机增益——任何在仿真和真实之间变化的东西。

### 步骤 2：用 DR 训练

在每个 episode 开始时，采样 `slip ~ Uniform[0.0, 0.4]`。训练 PPO / Q-learning / 任何算法。做很多 episode。

### 步骤 3：在"真实"滑动上 zero-shot 评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在训练支撑内；`0.5` 和 `0.7` 在外面。DR 训练的策略应在支撑内保持接近最优，在外面优雅退化。固定 slip 训练的策略在其训练 slip 之外会很脆弱。

### 步骤 4：与窄训练比较

用 `slip = 0.0` 训练第二个策略。在相同的 `slip` 扫描上评估。你应该看到一旦真实 slip > 0 就出现灾难性下降。

## 常见陷阱

- **随机化过多。** 在 `slip ∈ [0, 0.9]` 上训练，你的策略会变得如此风险规避以至于从不尝试最优路径。匹配*预期*的真实世界分布，而不是"什么都可能发生"。
- **随机化过少。** 在薄片上训练，策略完全无法泛化。使用自适应课程（Automatic Domain Randomization），随着策略改进扩大分布。
- **参数空间识别错误。** 随机化了错误的东西（相机色调，而真实差距是电机延迟），DR 就没用。先分析真实机器人。
- **特权信息泄漏。** 教师用全局状态做动作（而不只是观测），可能产生学生无法追上的结果。确保教师的策略在给定观测历史的情况下对学生是可实现的。
- **Sim-to-sim 迁移失败。** 如果你的策略对更难的仿真变体不鲁棒，它对真实世界也不会鲁棒。部署前始终在留出的仿真变体上测试。
- **没有真实世界安全包络。** 在仿真中有效且在真实中"有效"但没有底层安全屏障的策略仍可能损坏硬件。在非学习控制器中添加速率限制、力矩限制、关节限制。

## 应用场景

2026 年 sim-to-real 技术栈：

| 领域 | 技术栈 |
|------|--------|
| 足式运动（ANYmal、Spot、人形） | Isaac Lab + DR + privileged teacher / student |
| 操作（灵巧手、抓取放置） | Isaac Lab + DR + DR-GAN for vision |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + DR + real fine-tune |
| 无人机竞速 | RotorS / Flightmare + DR + online adaptation |
| 手指/手内操作 | OpenAI Dactyl（前所未有规模的 DR） |
| 工业机械臂 | MuJoCo-Warp + SI + small real fine-tune |

对所有尺度的控制，工作流是一致的：尽可能拟合仿真，随机化你无法拟合的，训练巨大的策略，蒸馏，带安全屏障部署。

## 交付产出

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 练习

1. **简单。** 在固定 slip 的 GridWorld（slip=0.0）上训练 Q-learning 智能体。在 slip ∈ {0.0, 0.1, 0.3, 0.5} 上评估。绘制 return vs slip。
2. **中等。** 训练一个 DR Q-learning 智能体，采样 `slip ~ Uniform[0, 0.3]`。评估相同的扫描。DR 在 slip=0.5（分布外）时带来多少收益？
3. **困难。** 实现课程学习：从 slip=0.0 开始，每当策略达到最优的 90% 时扩大 DR 范围。测量达到 slip=0.3 zero-shot 所需的总环境步数，与固定 DR 基线比较。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Reality gap | "Sim-to-real 差异" | 训练和部署的物理/感知之间的分布偏移。 |
| Domain randomization (DR) | "在随机仿真中训练" | 训练期间随机化仿真参数使策略泛化。 |
| System identification (SI) | "测量真实并拟合仿真" | 估计真实物理参数；设置仿真匹配。 |
| Domain adaptation | "在真实数据上微调" | 仿真训练后的小规模真实世界微调；可能适应观测或动力学。 |
| Privileged info | "教师的真值" | 只有仿真拥有的信息；学生必须从观测历史中推断。 |
| Teacher/student | "从特权蒸馏到可观测" | 教师用捷径训练；学生学习在没有捷径的情况下模仿。 |
| ADR | "自动 Domain Randomization" | 随着策略改进扩大 DR 范围的课程。 |
| Real2Sim | "用真实数据缩小差距" | 学习残差使仿真模拟真实 rollout。 |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 原始 DR 论文（机器人视觉）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — 动力学 DR，四足运动。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl，大规模 ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal 的 teacher-student。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 驱动 2025-2026 部署的大规模并行仿真。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR 课程方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Dyna 框架（用模型做规划+rollout），支撑现代 sim-to-real 流水线。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — sim-to-real 方法分类及 benchmark 结果。
