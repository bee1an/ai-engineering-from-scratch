# 心智理论与涌现协调

> Li et al.（arXiv:2310.10701）表明合作文本游戏中的 LLM 智能体展现出**涌现的高阶心智理论**（ToM）——推理另一个智能体对第三个智能体信念的信念——但由于上下文管理和幻觉在长期规划上失败。Riedl（arXiv:2510.05174）在群体中测量了高阶协同，发现**只有** ToM 提示条件产生身份关联分化和目标导向互补性；低容量 LLM 只显示虚假涌现。也就是说，协调涌现是提示条件性和模型依赖性的，不是免费的。本课实现一个最小 ToM 感知智能体，在有和没有 ToM 提示的情况下运行合作任务，并对照 Riedl 2025 协议测量协调差异。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 17 (Generative Agents)
**Time:** ~75 minutes

## 问题

多智能体协调经常看起来很神奇：智能体分工、预判彼此、避免冗余。通常这种"涌现"是提示工程的产物——有人告诉智能体"协调"。移除提示，移除协调。

Riedl 2025 的发现更严格：在受控条件下，协调只在智能体被提示推理**其他智能体的心智**（ToM）时涌现。没有 ToM 提示，即使强模型也显示出无法通过统计控制的协调模式。这对生产很重要：团队发布的"多智能体协调"功能是提示依赖的且脆弱的。

本课将 ToM 视为一种特定能力（推理关于信念的信念），构建一个最小 ToM 感知智能体，并测量真正的协调看起来像什么 vs. 提示装饰看起来像什么。

## 概念

### ToM 的含义

发展心理学：3 岁的孩子认为任何人的内心世界与自己一致。5 岁的孩子理解他人有不同的信念。7 岁的孩子推理关于信念的信念（"她认为我认为球在杯子下面"）。这些是零阶、一阶和二阶 ToM。

对于 LLM 智能体，ToM 阶数映射为：

- **零阶：** 没有他人模型。智能体仅基于自身观察行动。
- **一阶：** 智能体有每个其他智能体信念的模型。"Alice believes X."
- **二阶：** 智能体建模递归信念。"Alice believes that Bob believes X."

Li et al. 2023 发现一阶和二阶 ToM 在合作游戏中的 LLM 智能体中涌现，但在长期和不可靠通信下退化。

### Sally-Anne 测试，简述

1985 年的错误信念测试：Sally 把弹珠放在篮子 A 中，离开。Anne 把它移到篮子 B。Sally 回来时会在哪里找？有一阶 ToM 的孩子说篮子 A（Sally 的信念与现实不同）。没有的孩子说篮子 B。

GPT-4 时代的 LLM 在直接提出时通过 Sally-Anne 风格的测试。当叙述很长、场景多次变化或问题间接表述时它们失败。这是 2026 年生产 LLM 中 ToM 的实际状态。

### Riedl 的协调测量

Riedl（arXiv:2510.05174）构建了群体规模测试：N 个智能体，一个合作目标，可变提示条件。测量：

1. **身份关联分化。** 智能体是否随时间发展出稳定的角色区分？
2. **目标导向互补性。** 智能体的动作是否互补（不同子任务）而非重复？
3. **高阶协同。** 群体是否实现了任何子集都无法实现的东西的统计度量。

结果：只有在 ToM 提示条件下，三个指标都产生高于基线的信号。没有 ToM 提示，中等容量模型的指标徘徊在随机附近。大模型在没有显式 ToM 提示时显示一些协调，但效果小于显式提示。

### 协调幻觉

没有统计控制，演示中的"涌现协调"往往反映：

- 内置协调的提示工程（系统提示说"一起工作"）。
- 观察者偏差（我们看到我们期望的模式）。
- 对成功运行的事后选择。

宣传"涌现协调"但没有可测量信号的生产系统应被视为营销。先测量再声称。

### 最小 ToM 感知智能体

结构：

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions
```

`other_models` 属性是 ToM 状态。一阶 ToM 只保持一层。二阶添加 `other_models[i][other_models_of_j]`——我认为智能体 i 认为智能体 j 相信什么。

### 为什么长期会受损

Li et al. 记录：上下文限制导致智能体忘记哪个信念属于谁。幻觉向其他智能体模型添加虚假信念。两者都产生"I thought he thought X"的错误，随时间复合。

论文和 2024-2026 后续工作中记录的缓解措施：

- **提示中的显式 ToM 状态。** 结构化格式：`{agent_id: belief_list}`。强制检索保持身份-信念绑定。
- **更短的推理链。** 每轮更少的 ToM 更新减少复合幻觉。
- **外部 ToM 存储。** 在 LLM 上下文之外维护模型；每轮只注入相关部分。

### ToM 在生产中失败的场景

- **对抗性设置。** 有良好 ToM 的智能体更容易被操纵（你可以建模它们对你的建模，然后利用）。
- **异构团队。** 当模型不同时，对一个对手有效的 ToM 模型不能泛化。
- **依赖真实值的任务。** ToM 是关于信念的；如果正确性取决于事实，ToM 可能是干扰。

### 你能实际测量的协调

三个实用信号表明团队的协调是真实的而非提示装饰的：

1. **随时间的互补性。** 在多轮任务中，智能体的动作是否覆盖不相交的子任务？
2. **预判。** 智能体 A 在 T+1 轮的动作是否依赖于对 B 在 T+2 轮动作的预测，且该预测被证明正确？
3. **纠正。** 当 A 在 T 轮误读 B 的信念时，A 是否在 T+2 轮前纠正？

这些在有日志的多智能体系统中是可测量的。它们是"协调"叙事的实质版本。

## 动手构建

`code/main.py` 实现：

- `ToMAgent` — 追踪自身信念和每个其他智能体的信念模型。
- 合作任务：三个智能体必须从三个盒子中收集三个代币；每个盒子只能放一个代币。智能体不能通信；它们从彼此的动作推断意图。
- 两种配置：`zeroth_order`（无 ToM）和 `first_order`（带一层信念模型的 ToM）。
- 200 次随机试验的测量：完成率、重复率（两个智能体瞄准同一个盒子）、平均完成轮次。

运行：

```
python3 code/main.py
```

预期输出：零阶智能体以 ~35% 的比率重复努力，在 10 轮内完成 ~60% 的试验。一阶 ToM 智能体以 ~5% 重复并完成 ~95%。差异就是可测量的协调效果。

## 使用方式

`outputs/skill-tom-auditor.md` 是一个技能，审计多智能体系统的"涌现协调"声明。检查提示装饰、对照控制的统计显著性和测量的互补性。

## 上线清单

协调声明检查清单：

- **控制条件。** 没有协调提示的系统版本。两者都测量。
- **统计检验。** 系统和控制之间的差异在你的指标上是否在 `p < 0.05` 显著？
- **互补性度量。** 随时间的动作不相交性，不仅仅是最终成功。
- **失败案例日志。** 当智能体协调失败时，ToM 状态看起来怎样？
- **模型容量披露。** 如果效果在较小模型上消失，说明这一点。

## 练习

1. 运行 `code/main.py`。确认一阶 ToM 将重复率降低 ~7 倍。当你扩展到 5 个智能体和 5 个盒子时差距是否持续？
2. 实现二阶 ToM（智能体 A 建模 B 对 C 的看法）。它是否比一阶改善？在什么任务上？
3. 向 ToM 状态注入**幻觉**：每轮随机翻转一个信念。这对一阶性能退化多少？
4. 阅读 Li et al.（arXiv:2310.10701）。复现"长期退化"发现：当轮次从 10 增长到 30 时，你的一阶 ToM 性能如何变化？
5. 阅读 Riedl 2025（arXiv:2510.05174）。在你的模拟日志上实现高阶协同统计量。没有 ToM 提示条件时效果是否存在？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Theory of Mind | "理解他人的心智" | 建模另一个智能体信念的能力。按阶分级（0, 1, 2+）。 |
| Sally-Anne test | "错误信念测试" | 1985 年发展心理学；LLM 通过简单版本，复杂版本失败。 |
| First-order ToM | "A believes X" | 建模一个他人对事实的信念。 |
| Second-order ToM | "A believes B believes X" | 递归建模深一层。 |
| Identity-linked differentiation | "随时间稳定的角色" | Riedl 的指标：角色持续存在，不是随机的。 |
| Goal-directed complementarity | "不相交的动作" | 智能体瞄准不同子任务，而非相同的。 |
| Higher-order synergy | "群体超越任何子集" | Riedl 的真实协调统计度量。 |
| Coordination illusion | "看起来协调了" | 没有可测量信号的提示装饰式协调外观。 |

## 延伸阅读

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — 合作游戏中的涌现 ToM；长期失败模式
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — 群体规模测量；ToM 提示是承重条件
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — 1978 年 ToM 概念的起源
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) — Sally-Anne 论文（1985）
