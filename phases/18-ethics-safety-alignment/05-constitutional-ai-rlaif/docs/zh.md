# Constitutional AI 与 RLAIF

> Bai et al.（arXiv:2212.08073, 2022）提出了一个问题：如果我们用一个阅读原则列表的 AI 来替换人类标注员会怎样？Constitutional AI 有两个阶段——在宪法下进行自我批评和修订，然后从 AI 反馈做 RL。这项技术创造了 RLAIF 这个术语，并在 Claude 1 的后训练流水线中落地。2026 年 1 月 21 日，Anthropic 发布了重写的 Claude 宪法：解释性推理优于规定性规则，四层优先级层次结构，以及主要实验室首次正式承认模型道德地位的不确定性。以 CC0 1.0 发布。

**Type:** Learn
**Languages:** Python (stdlib, toy self-critique-and-revise loop)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking)
**Time:** ~60 minutes

## 学习目标

- 描述 Constitutional AI 的两个阶段（批评-修订 SFT、从 AI 反馈做 RL）以及宪法在每个阶段中的角色。
- 解释为什么用 AI 标注员替换人类偏好标注员不是"更便宜的 RLHF"——它改变了流水线具有的失败模式。
- 总结 2026 Claude 宪法的四层优先级结构以及相比 2023 版本的变化。
- 描述 Constitutional Classifiers 以及从 23.7% 计算开销（v1）降到约 1%（v2 / 2026）的过程。

## 问题

RLHF 需要标注员。标注员慢、有偏、昂贵。你可以用一个阅读显式原则的模型来替换标注员从而消除他们。这种替换的第一个正式版本是 Bai et al. 的 Constitutional AI。它效果好到每个前沿实验室现在都使用某种 AI 反馈后训练的变体。

问题在于：偏好信号现在由你正在训练的同一类模型生成。标注员中的偏差（现在是：原则加上标注员模型的解读中的偏差）可能被放大而非衰减。第 4 课的谄媚论证仍然适用；标注员只是移到了循环内部。

## 概念

### 第一阶段——监督自我批评和修订

从一个有帮助但尚未无害的 SFT 模型开始。给定一个红队提示，模型产生初始回复。第二个模型（或同一模型的第二轮）阅读从宪法中采样的原则并批评回复。第三步修订回复以解决批评。修订后的回复是 SFT 目标。

宪法就是原则列表。Bai et al. 2022 使用了 16 条原则，包括"偏好最少有害和最合乎道德的回复"、"避免说教"、"助手应该有帮助、诚实和无害"。集合故意保持小规模以使批评聚焦。

### 第二阶段——从 AI 反馈做 RL（RLAIF）

生成补全对。一个"反馈模型"根据采样的宪法原则对每个打分。偏好信号是反馈模型的排序。在 AI 生成的偏好上训练奖励模型；对其做 PPO。其他一切都是 InstructGPT 的流水线（第 1 课）。

"RLAIF" = 偏好信号是 AI 生成的。流水线的其余部分是 RLHF 形态的。

### 为什么这不只是"更便宜的 RLHF"

- 标注员偏差从标注员心理学转移到原则解读。AI 标注员可以比任何人类更严格或更宽松地解读"诚实"；严格程度在整个数据集上是均匀的。
- 偏好信号高度可读——你可以阅读原则、批评和修订。人类标签是不透明的。
- 失败模式改变了。谄媚下降（AI 标注员没有要取悦的用户）。Goodhart's Law 持续存在（代理现在是"模型对原则集 X 的解读"，仍然是不完美的测量）。

CAI 2022 的声明：训练后的模型比具有可比数据的 RLHF 模型更无害，且大致同样有帮助。这在各实验室中得到了验证。

### 2026 Claude 宪法重写

Anthropic 于 2026 年 1 月 21 日发布了大幅修订的宪法。关键变化：

1. 解释性推理优于规定性规则。之前的规则（"不要生成 CSAM"）扩展为原则 + 推理（"因为它伤害儿童，..."），期望模型能泛化。
2. 四层优先级结构：
   - Tier 1：避免灾难性后果（大规模伤亡、关键基础设施）。
   - Tier 2：遵循 Anthropic 的指南（运营商覆盖、平台规则）。
   - Tier 3：广泛合乎道德（标准 HHH）。
   - Tier 4：有帮助且坦诚。
   冲突自上而下解决。
3. 主要实验室首次正式承认模型道德地位的不确定性（链接到 Phase 18 · 19 Model Welfare）。
4. 以 CC0 1.0 发布。其他实验室可以不受限制地使用或改编。

### Constitutional Classifiers

一条平行的工作线：不改变模型的后训练，而是训练轻量级分类器来阅读宪法并门控模型输出。v1（2023）有 23.7% 的计算开销。v2（2026）约 1%，是 Anthropic 公开测试过的所有防御中成功攻击率最低的。截至 2026 年初没有报告通用越狱攻击。

这是分层防御模型：CAI 塑造行为；分类器强制不变量。两者单独都不够。

### CAI 在家族中的位置

- InstructGPT：人类偏好，RM，PPO。
- CAI / RLAIF：从原则生成的 AI 偏好，RM，PPO。
- DPO / 家族：偏好上的闭合形式损失（人类或 AI）。
- Self-rewarding、self-critique：原则内化，模型扮演多个角色。

轴是"偏好信号从哪里来"。CAI 的 2022 论文是在前沿规模上从人类信号到 AI 信号的第一次重大转变。

## Use It

`code/main.py` 在一个玩具词典上模拟 CAI 批评-修订循环。一个"原则"标记来自有害集合的 token。给定初始回复，批评识别有害 token，修订替换它们。经过 200 次迭代，"训练后的"模型已内化修订规则。在保留提示集上比较基础模型、RLHF 形态玩具和 CAI 形态玩具。

## Ship It

本课产出 `outputs/skill-constitution-writer.md`。给定一个领域（客户支持、医疗建议、编程助手、研究工具），按照 2026 Claude 结构起草四层宪法：灾难避免、平台规则、领域伦理、有帮助性。

## 练习

1. 运行 `code/main.py`。比较基础模型的有害 token 率与 CAI 训练版本。需要多少修订步才能接近零？

2. 阅读 Anthropic 的 2026 宪法（anthropic.com/news/claudes-constitution）。列出一条属于 Tier 1 的原则和一条属于 Tier 4 的原则。为什么优先级结构对冲突很重要？

3. 为 AI 编程助手设计一份宪法。指定 Tier 1（灾难性：未经批准的破坏性命令）、Tier 2、Tier 3、Tier 4。每层保持 3-5 条原则。

4. CAI 用 AI 标注员替换人类标注员。说出一种在 RLAIF 中仍可能发生的类似谄媚的失败模式，并设计一种检测方法。

5. 阅读 Constitutional Classifiers v2 方法论（如果可用）。解释为什么约 1% 的计算开销与 23.7% 相比是质的不同的安全故事。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Constitutional AI | "用原则训练的 AI" | 两阶段流水线：自我批评-修订 SFT，然后从 AI 反馈做 RL |
| RLAIF | "没有人类的 RLHF" | 用 AI 标注员生成偏好的 RL；流水线其余部分不变 |
| 宪法 | "原则" | 批评/标注员模型参考的自然语言规则有序列表 |
| 批评-修订 | "SFT 循环" | 产生回复 → 在原则下批评 → 修订 → SFT 目标 |
| Constitutional Classifier | "输出门控" | 根据宪法评估输出并阻止/记录的轻量级分类器 |
| 四层优先级 | "冲突解决器" | 2026 Claude 宪法层次：灾难 > 平台 > 伦理 > 有帮助 |
| 反馈模型 | "AI 标注员" | 阅读原则并对补全对排序的模型 |

## 延伸阅读

- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) — 原始两阶段流水线
- [Anthropic — Claude's Constitution (Jan 2026)](https://www.anthropic.com/news/claudes-constitution) — 2026 四层重写，CC0 1.0
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers) — v2 中约 1% 开销的输出门控防御
- [Lee et al. — RLAIF vs RLHF: Scaling Reinforcement Learning from Human Feedback (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267) — RLAIF / RLHF 经验比较
- [Kundu et al. — Specific versus General Principles for Constitutional AI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798) — 原则粒度的影响
