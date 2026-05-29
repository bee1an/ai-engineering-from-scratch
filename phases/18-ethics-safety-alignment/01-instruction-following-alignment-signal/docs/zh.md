# 指令遵循作为对齐信号

> 后续对 RLHF 的每一项批评都是在反对这条流水线。在研究优化压力如何扭曲代理指标之前，你得先看到这个代理指标本身。InstructGPT（Ouyang et al., 2022）定义了参考架构：在指令-回复对上做监督微调，用成对偏好排序训练奖励模型，再用 PPO 对奖励模型做强化学习并加上对 SFT 策略的 KL 惩罚。一个 1.3B 的 InstructGPT 被人类评估者认为优于 175B 的 GPT-3。正是这一个结果，让 2026 年每个前沿实验室仍然在发布 RLHF 形态的后训练流水线。

**Type:** Learn
**Languages:** Python (stdlib, toy three-stage pipeline)
**Prerequisites:** Phase 10 · 06 (SFT), Phase 10 · 07 (RLHF), Phase 10 · 08 (DPO)
**Time:** ~45 minutes

## 学习目标

- 说出 InstructGPT 流水线的三个阶段及各阶段使用的损失函数。
- 解释为什么一个 1.3B 的指令微调模型在人类偏好评估中击败了原始 175B GPT-3。
- 说明第三阶段的 KL 惩罚在防止什么，以及去掉它为什么会退化为模式寻求行为。
- 描述对齐税以及 Ouyang et al. 用 PPO-ptx 来缓解它的方法。

## 问题

预训练语言模型做的是续写文本，而不是回答问题。问 GPT-3"写一个反转列表的 Python 函数"，你经常得到的是另一个提示语，因为训练分布中大部分是网页文本接着更多网页文本。模型在做它的本职工作——只是这个工作本身不对。

每个认真的实验室用来修复这个问题的代理指标是人类偏好。两个补全交给评估者；评估者选出更好的那个；奖励模型学习评估者的偏好。然后一个 RL 循环把策略推向奖励模型打高分的输出。这就是 InstructGPT 论文的全部论点，三句话说完。剩下的都是工程。

## 概念

### 第一阶段：监督微调（SFT）

收集提示-回复对，其中回复是一个善意的人类会写出的内容。Ouyang et al. 使用了来自标注员和 OpenAI API 的 13k 条提示。用标准交叉熵损失在这些数据上微调基础模型。

SFT 给你的是：模型现在会回答问题而不是续写问题。它不给你的是：当多个回答都合理时，评估者更偏好哪一个的信号。

### 第二阶段：奖励模型（RM）

对每个提示，从 SFT 模型采样 K 个补全。标注员对它们排序。训练一个奖励模型，对任意提示-回复对打分，使得当 `y_w` 被偏好于 `y_l` 时：

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

这就是 Bradley-Terry 成对偏好损失。RM 通常从 SFT 模型初始化，把语言模型头替换为标量头。

奖励模型很小：6B 就足以服务 175B 的 InstructGPT。但它们也很脆弱——论文第 5 节主要讲的是在小规模就出现的奖励黑客行为。

### 第三阶段：带 KL 惩罚的 PPO

定义目标：

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

用 PPO 最大化。KL 项防止 `pi` 偏离 SFT 策略太远。没有它，优化器会找到对抗样本——那些在 RM 下得高分的字符串，原因是 RM 从未见过它们，而不是人类真的偏好它们。

KL 系数 `beta` 是 RLHF 中最重要的单一超参数。太低：奖励黑客。太高：相比 SFT 没有改进。

### 对齐税

RLHF 之后，模型被人类偏好但在标准基准（SQuAD、HellaSwag、DROP）上退化。Ouyang et al. 称之为对齐税，并用 PPO-ptx 来修复：把预训练梯度混入 RL 目标，使模型不会忘记那些从未被奖励过的下游任务。

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx 成为了标准做法。Anthropic、DeepMind 和 Meta 都使用某种变体。

### 结果

一个 1.3B 的 InstructGPT（SFT + RM + PPO-ptx）在约 70% 的情况下被标注员认为优于 175B 的基础 GPT-3。在来自生产流量的隐藏测试提示上，差距更大。从这个数字可以读出两件事：

1. 对齐和能力是不同的轴。175B 模型有更多能力；1.3B 模型有更多对齐；标注员偏好对齐的那个。
2. 能力下限由基础模型决定。你不能通过 RLHF 让基础模型知道它从未见过的事实。

### 为什么这是 Phase 18 的参考点

后续课程中的每一项批评——奖励黑客（第 2 课）、DPO（第 3 课）、谄媚（第 4 课）、CAI（第 5 课）、sleeper agent（第 7 课）、alignment faking（第 9 课）——都在反对这条流水线的某个部分。奖励黑客攻击第二阶段。DPO 合并了第二和第三阶段。CAI 替换了人类标注员。谄媚表明标注员是有偏的信号。Alignment faking 表明策略可以完全绕过第三阶段。不把这条流水线装在脑子里，你就无法理解这些批评中的任何一个。

## Use It

`code/main.py` 在玩具偏好数据上模拟三个阶段。基础"策略"是对动作 {A, B, C} 的有偏硬币。第一阶段 SFT 在 200 个提示上模仿标注员动作。第二阶段从 500 个成对排序拟合 Bradley-Terry 奖励模型。第三阶段运行简化的 PPO 更新，带有对 SFT 策略的 KL 惩罚。你可以观察奖励攀升、KL 散度增长、策略漂移——也可以关掉 KL 项，在 50 个更新步内看到奖励黑客出现。

观察要点：

- `beta = 0.1` vs `beta = 0.0` 的奖励轨迹。
- 训练步数上的 KL(pi || pi_SFT)。
- 最终动作分布与标注员偏好的对比。

## Ship It

本课产出 `outputs/skill-instructgpt-explainer.md`。给定一个 RLHF 流水线描述或论文摘要，它识别正在修改的是三个阶段中的哪一个，每个阶段使用什么损失，以及是否存在 KL 惩罚或等效正则化器。

## 练习

1. 运行 `code/main.py`。设置 `beta = 0.0`，报告 200 个 PPO 步之后的动作分布。用一段话解释模式寻求行为。

2. 修改奖励模型，给动作 B 加上 +0.5 的偏差（模拟奖励 bug）。用 `beta = 0.1` 运行 PPO。KL 惩罚能否阻止策略利用这个偏差？在什么 `beta` 值下利用变得可见？

3. 阅读 Ouyang et al.（arXiv:2203.02155）Figure 1。通过运行 PPO 1、5、20、100 步并测量相对于 SFT 模型的偏好来复现标注员偏好曲线。

4. 论文 Section 4.3 报告 1.3B InstructGPT 约 70% 的时间击败 175B GPT-3。为什么在隐藏的生产提示上比率会高于标注员自己的提示？

5. 在相同偏好数据上用 DPO（Phase 10 · 08）替换 PPO 损失。比较最终策略漂移（到 SFT 的 KL）和最终奖励。在匹配奖励下，哪种方法漂移更远？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| SFT | "指令微调" | 第一阶段：在提示-回复对上做交叉熵微调 |
| 奖励模型 | "RM" | 在 (提示, 回复) 上的标量回归器，用 Bradley-Terry 在成对标签上训练 |
| Bradley-Terry | "成对偏好损失" | -log sigmoid(r_w - r_l)；把成对排序归约为二分类 |
| KL 惩罚 | "正则化器" | `beta * KL(pi \|\| pi_SFT)` — 让 RL 策略保持在 SFT 锚点附近 |
| PPO-ptx | "带预训练混合的 PPO" | 在 PPO 目标中加入一部分预训练对数似然以抵消对齐税 |
| 对齐税 | "RLHF 退化" | RLHF 后在 RLHF 未针对的标准基准上的性能下降 |
| 标注员偏好 | "ground truth" | 人类排序的样本；RM 是它的统计代理，不是"人类价值观"的代理 |

## 延伸阅读

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — InstructGPT 论文，后续所有 RLHF 流水线的基础
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — RLHF 用于摘要的前身
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — 基于偏好的 RL 的原始公式
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — Anthropic 对 InstructGPT 流水线的 HH 扩展
