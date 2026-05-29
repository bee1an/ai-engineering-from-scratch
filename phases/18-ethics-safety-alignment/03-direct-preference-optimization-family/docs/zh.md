# 直接偏好优化家族

> Rafailov et al.（2023）证明 RLHF 的最优解在偏好数据中有闭合形式，因此可以跳过显式奖励模型直接优化策略。这个洞察催生了一个家族——IPO、KTO、SimPO、ORPO、BPO——每个都修复了 DPO 的一种失败模式。到 2026 年，直接对齐算法在前沿后训练中的使用量超过了 PPO。但第 2 课的过度优化曲线仍然适用：DAA 没有逃脱 Goodhart，只是改变了它咬人的位置。

**Type:** Learn
**Languages:** Python (stdlib, six-variant preference-loss comparator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking), Phase 10 · 08 (DPO basics)
**Time:** ~75 minutes

## 学习目标

- 从 RLHF-with-KL 最优解推导 DPO 闭合形式。
- 说出 IPO、KTO、SimPO、ORPO、BPO 各自修复了 DPO 的哪种失败模式。
- 区分"隐式奖励差距"和"偏好强度"，并解释为什么 IPO 的恒等映射很重要。
- 解释为什么 Rafailov et al.（NeurIPS 2024）证明 DAA 尽管没有显式 RM 仍会过度优化。

## 问题

RLHF 目标（第 1 课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此奖励由最优策略与参考策略的比值隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将其代入 Bradley-Terry 偏好似然，配分函数 `Z(x)` 因为只依赖于 `x` 而消去。剩下的是仅含策略参数的损失——不需要奖励模型。这就是 DPO。

问题在于：推导假设最优解可达、偏好数据在分布内、参考策略是真正的模式锚点。这些都不精确成立。家族中的每个成员修复的是不同的被违反假设。

## 概念

### DPO（Rafailov et al., 2023）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出什么问题：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无界的。一个微小的偏好可以产生任意大的差距。
- 损失把选中和拒绝的 log-prob 推向相反方向。只要拒绝的下降更快，它可以把选中的绝对 log-prob 也推低。这就是退化选中响应现象。
- 分布外偏好（罕见对 vs 罕见对）产生任意的隐式奖励。

### IPO（Azar et al., 2024）

Identity Preference Optimization 用恒等映射替换 log-sigmoid 作用于偏好概率。损失变成对有界目标的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

边际被 `1/(2 beta)` 限制。偏好强度和隐式奖励差距成正比。不会爆炸。

### KTO（Ethayarajh et al., 2024）

Kahneman-Tversky Optimization 完全放弃成对结构。给定单个标注输出和一个二元"可取"或"不可取"信号，它映射到前景理论效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对收益和损失使用不同权重（损失厌恶）。好处：你可以使用非配对数据，这类数据丰富得多。

### SimPO（Meng et al., 2024）

Simple Preference Optimization 使训练信号与生成对齐。完全移除参考策略，按长度归一化对数似然：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

加一个边际 `gamma` 来稳定。长度归一化消除了利用 DPO 长度偏差失败模式的激励（更长的 `y_w` 从构造上给出更大的 log-prob 差距）。

### ORPO（Hong et al., 2024）

Odds-Ratio Preference Optimization 在标准 SFT 负对数似然上加一个偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

没有参考策略——SFT 项就是正则化器。从基础模型到对齐模型单阶段训练。不需要单独的 SFT 检查点。

### BPO（ICLR 2026 submission, OpenReview id=b97EwMUWu7）

识别了退化选中响应问题：DPO 保持排序 `y_w > y_l` 但 `y_w` 的绝对 log-prob 可能下降。BPO 加了一行修正，惩罚选中响应的下降。报告在 Llama-3.1-8B-Instruct 上数学推理比 DPO 提升 +10.1% 准确率。

### 普遍结论：DAA 仍然会过度优化

Rafailov et al. "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms"（NeurIPS 2024）用 DPO、IPO、SLiC 在多个数据集和 KL 预算上训练策略。黄金奖励 vs KL 曲线具有与 Gao et al. 相同的先升后降形状。隐式奖励在训练中查询分布外样本；KL 正则化无法稳定这一点。

DAA 没有逃脱 Goodhart。它们把咬人的表面从"奖励模型被过度优化"变成了"参考策略比值被过度优化"。通用修复——更好的数据、集成、早停——对两者都适用。

### 2026 年如何选择

- 如果你有大量配对偏好数据：DPO 配保守 beta，如果长度偏差明显则用 SimPO。
- 如果你有非配对二元反馈：KTO。
- 如果你想从基础模型单阶段流水线：ORPO。
- 如果你在 DPO 日志中看到选中 log-prob 退化：BPO。
- 如果偏好强度变化很大且 DPO 饱和：IPO。

每个实验室都在一组任务上跑全部五种然后选赢家。没有理由认为数学推理和安全的最优解是同一个。

## Use It

`code/main.py` 在一个真实偏好强度按对变化的玩具偏好数据集上比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO）。每种损失对相同的 500 对样本用小型 softmax 策略优化。绘制每种方法的最终胜率、选中 log-prob 漂移和隐式奖励分布。

## Ship It

本课产出 `outputs/skill-preference-loss-selector.md`。给定数据集统计（配对 vs 非配对、可变 vs 均匀偏好强度、长度分布）和目标（单阶段或 SFT 后接偏好），推荐一种偏好损失并报告它防护的失败模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终选中 log-prob 下降。BPO 应保持更高的选中绝对概率——验证这一点。

2. 修改偏好数据使所有对具有相等强度。六种方法中哪种最鲁棒？哪种退化？解释 IPO 在此的优势。

3. 使拒绝响应平均比选中长 2 倍。不改变其他任何东西，用数值展示 DPO 的长度利用和 SimPO 的修复。

4. Rafailov et al.（NeurIPS 2024）声称 DAA 会过度优化。复现单点版本：绘制选中减拒绝的 KL 散度，观察 DPO 在大 beta 下的过度优化。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写下 BPO 对 DPO 添加的一行修正。对照 `code/main.py` 中的实现确认。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| DPO | "不需要奖励模型的 RLHF" | 从 RLHF 闭合形式最优解推导的损失；仅含策略参数 |
| 隐式奖励 | "log 比值" | `beta * log(pi(y\|x) / pi_ref(y\|x))` — DPO 隐含的奖励 |
| IPO | "有界 DPO" | 用恒等映射替换 log-sigmoid；隐式奖励差距被 `1/(2 beta)` 限制 |
| KTO | "非配对 DPO" | 对单标签的前景理论效用，带损失厌恶 |
| SimPO | "无参考策略 DPO" | 长度归一化对数似然 + 边际；无参考策略 |
| ORPO | "单阶段 DPO" | NLL + 赔率比偏好项；从基础模型一次训练 |
| BPO | "保持选中的 DPO" | DPO 加上对选中响应绝对 log-prob 下降的惩罚 |
| 退化选中 | "选中下降了" | DPO 降低选中 log-prob，只要拒绝下降更快 |
| DAA | "直接对齐算法" | 任何跳过显式 RM 的偏好损失方法 |

## 延伸阅读

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
