# STaR、V-STaR、Quiet-STaR — 自我教学推理

> 最小的自我改进循环就在推理过程本身。模型生成一条思维链，保留那些得出正确答案的，然后在这些上微调。这就是 STaR。V-STaR 加了一个验证器让推理时选择更好。Quiet-STaR 把推理推到每个 token。三者都有效。但都不是魔法——循环会保留任何碰巧得出正确答案的捷径。

**Type:** Learn
**Languages:** Python (stdlib, bootstrap-loop simulator)
**Prerequisites:** Phase 13 · 01-03 (Reasoning and CoT), Phase 15 · 01 (long-horizon framing)
**Time:** ~60 minutes

## 问题

教模型推理的直接方法是收集人类编写的推理轨迹。这很贵、很慢，而且受限于人类愿意写多少高质量思维链。

STaR（Self-Taught Reasoner，Zelikman et al., 2022）问：如果模型自己写推理过程，然后用已知答案来打分呢？循环是：

1. 采样一条推理轨迹加答案。
2. 如果最终答案正确，保留该轨迹。
3. 在保留的轨迹上微调。
4. 重复。

它有效。GSM8K 和 CommonsenseQA 都在没有新人类标注的情况下提升了。但循环有一个内建偏差：任何产生正确答案的推理过程都会被保留，不管推理本身是否合理。V-STaR（Hosseini et al., 2024）用学习到的验证器修补了这个问题；Quiet-STaR（Zelikman et al., 2024）将这个想法推广到每个 token 的内部推理。

## 概念

### STaR：在有效的基础上自举

从一个有一定弱推理能力的基础模型开始。对每个训练问题，采样一条推理过程加答案。如果答案匹配标签，保留（问题，推理，答案）三元组。在保留集上微调模型。重复。

有一个关键技巧。如果模型永远无法答对某个问题，循环就无法在上面学习。STaR 加入了**合理化（rationalization）**：对于模型失败的问题，注入正确答案作为提示，重新让模型产生一条通向该答案的推理。合理化的推理被加入训练集。

原始论文的结果（Zelikman et al., 2022）：GPT-J 基础模型在 GSM8K 上通过多轮 STaR 加合理化从 5.8% 提升到 10.7%——约 5 个百分点的绝对提升。在 CommonsenseQA 上，STaR 训练的 GPT-J 6B 达到 72.5%，与在手工标注推理上微调的 GPT-3 175B（~73%）相当——一个大约 30 倍的模型。

### V-STaR：用 DPO 训练验证器

STaR 丢弃了错误的推理。Hosseini et al.（2024）观察到这些也是数据：每对（推理，"这是否正确"）都可以训练一个验证器。他们使用 Direct Preference Optimization 在正确和错误的解上构建一个排序器。推理时，采样 N 条推理并选验证器排名最高的。

报告的提升：在 GSM8K 和 MATH 上比之前的自我改进基线高 +4 到 +17 个百分点，大部分增益来自使用验证器进行推理时选择，而非额外的生成器微调。

### Quiet-STaR：每个 token 的内部推理

Zelikman et al.（2024）问：如果模型学会在每个 token 位置生成一段短的内部推理，而不仅仅是在问题和答案之间呢？Quiet-STaR 训练模型在每个预测 token 之前发出一个隐藏的"思考"，然后通过学习到的权重将思考感知的预测与基线预测混合。

结果：Mistral 7B 在 GSM8K 上获得了从 5.9% 到 10.9% 的绝对零样本提升，CommonsenseQA 从 36.3% 到 47.2%，无需任务特定微调。模型学会了"何时思考"——难的 token 获得更长的内部推理；简单的几乎没有。

### 为什么三者共享一个安全隐患

三种方法都使用最终答案作为梯度信号。一条通过有缺陷的推理到达正确答案的推理——利用捷径、猜测或使用不可泛化的模式——会被正向强化。在分布内问题上捷径有效。在分布外问题上它会静默失败。

V-STaR 的验证器通过学习排序推理来缓解，但验证器是在相同的标签集上训练的。它可能学会偏好格式良好的错误推理而非诚实的不确定性。更安全的设计是将 STaR 风格的数据与（a）过程监督奖励模型（奖励中间步骤，而非仅答案）和（b）能打破简单捷径的分布外评估结合。

### 对比

| 方法 | 训练信号 | 推理成本 | 数据浪费 | 已知失败模式 |
|---|---|---|---|---|
| STaR | 如果正确则保留（推理，答案） | 1x | 丢弃所有错误推理 | 捷径推理 |
| STaR + rationalization | 上述 + 正确答案提示重试 | 1x | 更少 | 合理化的推理可能不合理 |
| V-STaR | STaR + 从两类数据 DPO 训练验证器 | Nx (best-of-N) | 最小 | 验证器可能强化自信的错误 |
| Quiet-STaR | 每 token 推理 + 混合权重 | 1.5-3x | 最小 | 仍然是答案条件梯度 |

### 在 2026 技术栈中的位置

STaR 已经不新了。但这个模式在 2025-2026 到处重现。在可验证数学问题上的 RL（DeepSeek-R1、Kimi-k1.5、o1）就是 STaR 的答案条件梯度信号的放大版。过程奖励模型（Lightman et al., 2023；OpenAI 的 "Let's verify step by step"）是过程监督的替代方案。AlphaEvolve（Lesson 3）是代码版的 STaR，用程序评估器代替标签。Darwin Godel Machine（Lesson 4）是智能体脚手架自身的 STaR。

理解 STaR 让所有这些都串起来。它是最小可行的自我改进循环。

## Use It

`code/main.py` 在一个玩具算术任务上运行模拟的 STaR 循环。你可以观察：

- 准确率如何在自举轮次中攀升。
- 捷径如何潜入：模拟器包含一个"懒惰"推理类，40% 的时间得到正确答案但泛化很差。观察 STaR 是否保留它们。
- 验证器（V-STaR 风格）如何在推理时有帮助，但无法完全修剪训练期间引入的捷径。

## Ship It

`outputs/skill-star-loop-reviewer.md` 帮助你在训练之前审计一个提议的自我教学推理流水线。

## 练习

1. 运行模拟器。将捷径频率设为零，然后设为 0.4。两次运行的最终准确率差异有多大，即使两者在训练分布上都达到 >90%？

2. 向模拟器添加一个分布外测试。从不同分布抽取问题，在分布内和分布外集上评估自举模型。量化差距。

3. 阅读 Quiet-STaR 论文（arXiv:2403.09629）第 3 节。用各三句话解释 "end-of-thought" token 和混合权重头。

4. 比较 STaR 的正确即保留过滤器与独立奖励每个推理步骤的过程监督替代方案。找出标注成本差异和可能的质量差异。

5. 设计一个能捕捉已部署模型中捷径推理的评估。它不需要完美——它需要打破 STaR 循环会强化的最简单捷径。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| STaR | "Self-Taught Reasoner" | 在得出正确答案的模型生成推理上微调；重复 |
| Rationalization | "提示重试" | 注入正确答案并重新提示模型在失败问题上产生推理 |
| V-STaR | "Verifier STaR" | 在正确和错误推理上 DPO 训练验证器，用于推理时选择 |
| Quiet-STaR | "每 token 推理" | 在每个 token 位置生成隐藏思考；与基线预测混合 |
| Answer-conditioned gradient | "基于结果的信号" | 训练循环奖励最终答案，而非推理步骤 |
| Process reward model | "步骤级验证器" | 在每步正确性上训练的奖励模型，而非结果——与 STaR 对比 |
| Shortcut rationale | "正确答案，错误推理" | 通过不可泛化模式到达标签的推理；STaR 会保留这些 |

## 延伸阅读

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — 原始论文。
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) — 添加 DPO 验证器用于推理时选择。
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — 每 token 内部推理。
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — 过程奖励模型，替代梯度信号。
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 可验证任务上的 RL，STaR 扩展到前沿训练。
