# Self-Refine 与 CRITIC：迭代输出改进

> Self-Refine（Madaan et al., 2023）让一个 LLM 扮演三个角色 — 生成、反馈、精炼 — 循环执行。平均提升：7 个任务上 +20 绝对分。CRITIC（Gou et al., 2023）通过将验证路由到外部工具来强化反馈步骤。2026 年这个模式以"evaluator-optimizer"（Anthropic）或 guardrail 循环（OpenAI Agents SDK）的形式出现在每个框架中。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~60 minutes

## 学习目标

- 说出 Self-Refine 的三个 prompt（generate、feedback、refine）并解释为什么历史对 refine prompt 很重要。
- 解释 CRITIC 的核心洞察：LLM 在没有外部锚定的情况下不擅长自我验证。
- 用 stdlib 实现一个带历史和可选外部验证器的 Self-Refine 循环。
- 将这个模式映射到 Anthropic 的"evaluator-optimizer"工作流和 OpenAI Agents SDK 的 output guardrails。

## 问题

智能体产出了一个几乎正确的答案。也许一行代码有语法错误。也许摘要太长了。也许计划遗漏了一个边界情况。你想要的是：智能体批评自己的输出，然后修复它。

Self-Refine 表明这用单个模型就能工作，不需要训练数据，不需要 RL。但有一个问题：LLM 在硬事实上的自我验证很差。CRITIC 给出了解决方案 — 将验证步骤路由到外部工具（搜索、代码解释器、计算器、测试运行器）。

这两篇论文共同定义了 2026 迭代改进的默认模式：生成、验证（尽可能用外部工具）、精炼、验证器通过时停止。

## 核心概念

### Self-Refine（Madaan et al., NeurIPS 2023）

一个 LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 看到完整历史 — 所有先前的输出和批评 — 所以不会重复错误。论文做了消融实验：去掉历史，质量急剧下降。

标题数据：7 个任务（数学、代码、缩写、对话）平均 +20 绝对提升，包括 GPT-4。无训练、无外部工具、单模型。

### CRITIC（Gou et al., arXiv:2305.11738, v4 Feb 2024）

Self-Refine 的弱点：反馈步骤是 LLM 给自己打分。对于事实性声明这不可靠（幻觉对产生它的模型来说往往看起来很有说服力）。CRITIC 将 `feedback(task, output)` 替换为 `verify(task, output, tools)`，其中 `tools` 包括：

- 用于事实性声明的搜索引擎。
- 用于代码正确性的代码解释器。
- 用于算术的计算器。
- 领域特定验证器（单元测试、类型检查器、linter）。

验证器产出基于工具结果的结构化批评。精炼器然后以此批评为条件。

标题数据：CRITIC 在事实性任务上优于 Self-Refine，因为批评有锚定。在没有外部验证器的任务上（创意写作、格式化），CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形式：

1. **验证器通过。** 外部测试返回成功。有时优先使用（单元测试、类型检查器、guardrail 断言）。
2. **无反馈发出。** 模型说"输出没问题。"更便宜但不可靠；配合最大迭代次数上限使用。

2026 默认：组合使用。"如果验证器通过 OR 模型说没问题 AND 迭代次数 >= 2 OR 迭代次数 >= max_iterations 则停止。"

### Evaluator-Optimizer（Anthropic, 2024）

Anthropic 2024 年 12 月的文章将此命名为五种工作流模式之一。两个角色：

- Evaluator：对输出打分并产出批评。
- Optimizer：根据批评修改输出。

循环直到 evaluator 通过。这就是 Anthropic 框架下的 Self-Refine/CRITIC。Anthropic 补充的关键工程细节：evaluator 和 optimizer 的 prompt 应该有实质性差异，这样模型不会只是走过场。

### OpenAI Agents SDK output guardrails

OpenAI Agents SDK 将这个模式作为"output guardrails"提供。Guardrail 是在智能体产出最终输出后运行的验证器。如果 guardrail 触发（抛出 `OutputGuardrailTripwireTriggered`），输出被拒绝，智能体可以重试。Guardrails 可以调用工具（CRITIC 风格）或是纯函数（Self-Refine 风格）。

### 2026 陷阱

- **橡皮图章循环。** 同一个模型用相同 prompt 风格做生成和批评，收敛到"看起来不错"。使用结构性不同的 prompt，或用更小的廉价模型做批评。
- **过度精炼。** 每次 refine 都增加延迟和 token。预算 1-3 次；之后升级到人工审查。
- **在简单任务上用 CRITIC。** 如果没有外部验证器，CRITIC 退化为 Self-Refine；不要为一个空壳验证器付出延迟代价。

## Build It

`code/main.py` 在一个玩具任务上实现 Self-Refine 和 CRITIC：给定一个主题生成简短的要点列表。验证器检查格式（3 个要点，每个不超过 60 字符）。CRITIC 添加一个外部"事实验证器"来惩罚已知的幻觉。

组件：

- `generate` — 脚本化生成器。
- `feedback` — LLM 风格的自我批评。
- `verify_external` — CRITIC 风格的锚定验证器。
- `refine` — 根据历史重写输出。
- 停止条件 — 验证器通过或最多 4 次迭代。

运行：

```
python3 code/main.py
```

对比 Self-Refine 和 CRITIC 的运行。CRITIC 捕获了一个 Self-Refine 遗漏的事实错误，因为外部验证器有自我批评所没有的锚定。

## Use It

Anthropic 的 evaluator-optimizer 是这个模式的 Claude 友好表述。OpenAI Agents SDK 的 output guardrails 是 CRITIC 形状的（guardrails 可以调用工具）。LangGraph 提供一个读起来像 Self-Refine 的 reflection 节点。Google 的 Gemini 2.5 Computer Use 添加了一个每步安全评估器，是 CRITIC 的变体：每个动作在提交前都被验证。

## Ship It

`outputs/skill-refine-loop.md` 根据任务形态、验证器可用性和迭代预算配置一个 evaluator-optimizer 循环。输出生成器、评估器/验证器和优化器的 prompt，加上停止策略。

## 练习

1. 用 max_iterations=1 运行玩具。CRITIC 还有帮助吗？
2. 将外部验证器替换为有噪声的（随机 30% 误报）。循环会怎样？这就是 2026 大多数 guardrail 栈的现实。
3. 实现"不同模型做生成和批评"的变体：大模型生成，小模型批评。它能打败同模型吗？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。说出三个验证工具类别并各举一例。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的验证器角色。SDK 做对了什么，做错了什么？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Self-Refine | "自我修复的 LLM" | 单模型中的 Generate -> feedback -> refine 循环，带历史 |
| CRITIC | "工具锚定验证" | 用外部验证器（搜索、代码、计算器、测试）替换反馈 |
| Evaluator-Optimizer | "Anthropic 工作流模式" | 两个角色 — evaluator 打分，optimizer 修改 — 循环到收敛 |
| Output guardrail | "事后检查" | OpenAI Agents SDK 在智能体产出输出后运行的验证器 |
| 验证步骤 | "批评阶段" | 承重决策：锚定的还是自评的 |
| Refine history | "模型已经尝试过的" | 先前输出 + 批评预置到 refine prompt；去掉则质量崩溃 |
| 橡皮图章循环 | "自我认同失败" | 同 prompt 批评返回"看起来不错"；用结构性不同的 prompt 修复 |
| 停止条件 | "收敛测试" | 验证器通过 OR 无反馈 AND 迭代上限；永远不要单条件 |

## 延伸阅读

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — 原始论文
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — 工具锚定验证
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — evaluator-optimizer 工作流模式
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 作为 CRITIC 形状验证器的 output guardrails
