# 近端策略优化（PPO）

> A2C 每次更新后就丢弃 rollout 数据。PPO 用裁剪的重要性比率包裹策略梯度，让你可以在同一批数据上做 10+ 轮 epoch 而不会导致策略崩溃。Schulman et al. (2017)。到 2026 年仍然是默认的策略梯度算法。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**时间：** ~75 分钟

## 问题

A2C（第 07 课）是 on-policy 的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 要求数据从*当前* `π_θ` 采样。做一次更新后 `π_θ` 就变了；你用过的数据现在是 off-policy 的。再用它，梯度就有偏了。

Rollout 很昂贵。在 Atari 上，一次 rollout 跨 8 个环境 × 128 步 = 1024 个 transition，需要十几秒的环境时间。做完一次梯度步就扔掉，太浪费了。

Trust Region Policy Optimization（TRPO，Schulman 2015）是第一个修复方案：约束每次更新使新旧策略之间的 KL 散度不超过 `δ`。理论上很优雅，但每次更新需要共轭梯度求解。2026 年没人跑 TRPO 了。

PPO（Schulman et al. 2017）用一个简单的裁剪目标替代了硬信赖域约束。多一行代码。每次 rollout 做十轮 epoch。不需要共轭梯度。理论保证够用。九年后它仍然是从 MuJoCo 到 RLHF 所有场景的默认策略梯度算法。

## 概念

![PPO 裁剪代理目标：在 1 ± ε 处裁剪比率](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略与收集数据的策略之间的似然比。`r_t = 1` 表示没有变化。`r_t = 2` 表示新策略选择 `a_t` 的概率是旧策略的两倍。

**裁剪代理目标。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项：

- 如果优势 `A_t > 0` 且比率试图超过 `1 + ε`，裁剪会压平梯度——不要把一个好动作的概率推到超过旧概率 `+ε` 以上。
- 如果优势 `A_t < 0` 且比率试图超过 `1 - ε`（意味着我们会让一个坏动作相对于其裁剪后的减少变得更可能），裁剪会封顶梯度——不要把一个坏动作推到 `-ε` 以下。

`min` 处理另一个方向：如果比率已经朝*有利*方向移动，你仍然能得到梯度（在会伤害你的那一侧不裁剪）。

典型 `ε = 0.2`。把目标函数画成 `r_t` 的函数：一个分段线性函数，"好的一侧"有平顶，"坏的一侧"有平底。

**完整 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

和 A2C 相同的 actor-critic 结构。三个系数，通常 `c_v = 0.5`，`c_e = 0.01`，`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行环境中各跑 `T` 步，收集 `N × T` 个 transition。
2. 计算优势（GAE），冻结为常量。
3. 冻结 `π_{θ_old}` 作为当前 `π_θ` 的快照。
4. 对 `K` 个 epoch，对每个 minibatch `(s, a, A, V_target, log π_old(a|s))`：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + value loss + entropy。
   - 梯度步。
5. 丢弃 rollout。回到步骤 1。

`K = 10` 和 minibatch 大小 64 是标准超参数组合。PPO 很鲁棒：具体数字在 ±50% 范围内通常不太影响结果。

**KL 惩罚变体。** 原始论文还提出了一个使用自适应 KL 惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，其中 `β` 根据观测到的 KL 调整。裁剪版本成为主流；KL 变体在 RLHF 中存活下来（因为 KL 到参考策略本身就是你始终需要的约束）。

## 动手构建

### 步骤 1：在 rollout 时捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照在 rollout 时取一次。在更新 epoch 期间不会改变。

### 步骤 2：计算 GAE 优势（第 07 课）

和 A2C 相同。在 batch 上做归一化。

### 步骤 3：裁剪代理更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"裁剪 → 零梯度"模式是 PPO 的核心。如果新策略已经在有利方向上漂移太远，更新就停止。

### 步骤 4：value 和 entropy

对 critic 目标加标准 MSE，对 actor 加 entropy bonus，和 A2C 一样。

### 步骤 5：诊断指标

每次更新要观察三件事：

- **平均 KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]`。如果超过 `0.1`，减少 `K_EPOCHS` 或 `LR`。
- **裁剪比例** — 比率落在 `[1-ε, 1+ε]` 之外的样本比例。应在 `~0.1-0.3`。如果 `~0`，裁剪从未触发 → 提高 `LR` 或 `K_EPOCHS`。如果 `~0.5+`，你在过拟合 rollout → 降低它们。
- **解释方差** `1 - Var(V_target - V_pred) / Var(V_target)`。Critic 质量指标。应随 critic 学习逐渐趋近 1。

## 常见陷阱

- **裁剪系数调错。** `ε = 0.2` 是事实标准。设为 `0.1` 更新太保守；`0.3+` 会引发不稳定。
- **epoch 太多。** `K > 20` 经常导致不稳定，因为策略偏离 `π_old` 太远。限制 epoch 数，尤其对大网络。
- **没有奖励归一化。** 大的奖励尺度会侵蚀裁剪范围。在计算优势前归一化奖励（running std）。
- **忘记优势归一化。** 每 batch 零均值/单位标准差归一化是标准做法。跳过它会在大多数 benchmark 上搞砸 PPO。
- **学习率没有衰减。** PPO 受益于线性 LR 衰减到零。恒定 LR 通常更差。
- **重要性比率数学错误。** 始终用 `exp(log_new - log_old)` 保证数值稳定，而不是 `new / old`。
- **梯度符号搞反。** 最大化代理目标 = *最小化* `-L^{CLIP}`。符号搞反是最常见的 PPO bug。

## 应用场景

PPO 是 2026 年跨越众多领域的默认 RL 算法：

| 应用场景 | PPO 变体 |
|----------|----------|
| MuJoCo / 机器人控制 | PPO with Gaussian policy, GAE(0.95) |
| Atari / 离散游戏 | PPO with categorical policy, rolling 128-step rollouts |
| LLM 的 RLHF | PPO with KL penalty to reference model, reward from RM at end of response |
| 大规模游戏智能体 | IMPALA + PPO (AlphaStar, OpenAI Five) |
| 推理 LLM | GRPO（第 12 课）— 无 critic 的 PPO 变体 |
| 仅偏好数据 | DPO — PPO+KL 的闭式折叠，无需在线采样 |

PPO 的*损失形状* — 裁剪代理 + value + entropy — 是 DPO、GRPO 和几乎所有 RLHF 流水线的脚手架。

## 交付产出

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO，`ε=0.2, K=4`。在匹配的环境步数下，比较与 A2C（每次 rollout 一个 epoch）的样本效率。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制 return vs 环境步数，并跟踪每次更新的平均 KL。在这个任务上，`K` 到多少时 KL 会爆炸？
3. **困难。** 用自适应 KL 惩罚替换裁剪代理（`KL > 2·target` 时 `β` 翻倍，`KL < target/2` 时 `β` 减半）。比较最终 return、稳定性和无裁剪程度。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Importance ratio | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`；与收集数据的策略的偏离程度。 |
| Clipped surrogate | "PPO 的核心技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有利方向超过裁剪点后梯度为零。 |
| Trust region | "TRPO / PPO 的意图" | 限制每次更新的 KL 以保证单调改进。 |
| KL penalty | "软信赖域" | PPO 的替代方案：`L - β · KL(π_θ \|\| π_old)`。自适应 `β`。 |
| Clip fraction | "裁剪触发频率" | 诊断指标 — 应在 0.1-0.3；超出范围说明调参有问题。 |
| Multi-epoch training | "数据复用" | 每次 rollout 做 K 个 epoch；用方差代价换取样本效率。 |
| On-policy-ish | "大致 on-policy" | PPO 名义上是 on-policy 的，但 K>1 epoch 安全地使用了略微 off-policy 的数据。 |
| PPO-KL | "另一种 PPO" | KL 惩罚变体；用于 RLHF，其中 KL 到参考策略本身就是约束。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 原始论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 对 PPO 每个超参数的消融实验。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；PPO-in-RLHF 的配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 清晰的现代阐述，附 PyTorch 代码。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 许多论文使用的参考单文件 PPO 实现。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 在语言模型上运行 PPO 的生产配方；配合第 09 课（RLHF）阅读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — "37 个代码级优化"论文；哪些 PPO 技巧是关键的，哪些是传说。
