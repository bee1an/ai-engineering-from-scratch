# Anthropic 工作流模式：简单优于复杂

> Schluntz 和 Zhang（Anthropic，2024 年 12 月）区分了工作流（预定义路径）和智能体（动态工具使用）。五种工作流模式覆盖大多数场景。从直接 API 调用开始，只有当步骤无法预测时才引入智能体。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## 学习目标

- 列举 Anthropic 的五种工作流模式：提示链、路由、并行化、编排器-工作者、评估器-优化器。
- 解释智能体与工作流的区别，以及各自的工程成本。
- 判断何时选择工作流而非智能体（反之亦然）。
- 用 stdlib 对脚本化 LLM 实现全部五种模式。

## 问题

团队经常为一个函数调用就能解决的问题去引入多智能体框架。代价是真实的：框架增加了层次，遮蔽了提示词，隐藏了控制流，并引入了过早的复杂性。Schluntz 和 Zhang 2024 年 12 月的文章是业界引用最多的反对声音：从简单开始，只有当复杂性能带来回报时才增加它。

## 概念

### 工作流 vs 智能体

- **工作流。** LLM 和工具通过预定义的代码路径编排。工程师掌控图。
- **智能体。** LLM 动态指挥自己的工具并自主决定步骤。模型掌控图。

两者各有用武之地。工作流更便宜、更快、更容易调试。智能体能解锁开放式问题，但失败模式更难推理。

### 增强型 LLM

五种模式的基础：一个 LLM 接入三种能力——搜索（检索）、工具（动作）、记忆（持久化）。任何 API 调用都可以使用这些能力。

### 五种模式

1. **提示链。** 调用 1 的输出是调用 2 的输入。适用于任务有清晰线性分解的场景。步骤之间可选程序化门控。

2. **路由。** 一个分类器 LLM 选择调用哪个下游 LLM 或工具。适用于分类上不同的输入需要不同处理的场景（一线支持 vs 退款 vs Bug vs 销售）。

3. **并行化。** 并发运行 N 个 LLM 调用，聚合结果。两种形态：分段（不同块）和投票（相同提示词，N 次运行，多数/综合）。

4. **编排器-工作者。** 一个编排器 LLM 动态决定运行哪些工作者（也是 LLM）并综合它们的输出。类似智能体循环，但编排器不会无限循环。

5. **评估器-优化器。** 一个 LLM 提出答案，另一个 LLM 评估它。迭代直到评估器通过。这是 Self-Refine（Lesson 05）的泛化。

### 工作流胜出的场景

- **可预测的任务。** 如果你能枚举步骤，就应该这样做。
- **成本受限的任务。** 工作流有有界的步骤数；智能体可能失控。
- **合规受限的任务。** 审计人员想读图，而不是从轨迹中推断。

### 智能体胜出的场景

- **开放式研究。** 当下一步取决于上一步返回了什么。
- **可变长度任务。** 几分钟到几小时的工作，步骤数未知。
- **新领域。** 当你还不知道正确的工作流时——先探索，后固化。

### 上下文工程伴侣

"Effective context engineering for AI agents"（Anthropic 2025）形式化了相邻学科：200k 窗口是预算，不是容器。包含什么、何时压缩、何时让上下文增长。在 Phase 14 上下文压缩课程中详细讲解（本课程重编号前的 Phase 14 第 06 课）。

## Build It

`code/main.py` 对 `ScriptedLLM` 实现了全部五种工作流模式：

- `prompt_chain(input, steps)` — 顺序执行。
- `route(input, classifier, handlers)` — 分类 + 分发。
- `parallel_vote(prompt, n, aggregator)` — N 次运行，聚合。
- `orchestrator_workers(task, workers)` — 编排器选择工作者。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — 循环直到通过。

运行：

```
python3 code/main.py
```

每种模式打印其执行轨迹。每种模式的代码总行数约 10-15 行；框架的成本以千行计。

## Use It

- 大多数任务直接 API 调用。
- 只有当模式确实需要持久状态（LangGraph）、Actor 模型并发（AutoGen v0.4）或角色模板（CrewAI）时才用框架。
- 当你想要 Claude Code 的 harness 形态而不想重建时，使用 Claude Agent SDK。

## Ship It

`outputs/skill-workflow-picker.md` 为给定任务描述选择正确的模式，包括决策理由和当工作流不够用时重构为智能体的路径。

## 练习

1. 实现带置信度阈值的路由。低于阈值 -> 升级到人工。对于一线支持场景，阈值落在哪里？
2. 给 `parallel_vote` 添加超时。当一个调用挂起时会发生什么？如何在缺少投票的情况下聚合？
3. 将 `evaluator_optimizer` 改造为 bandit：跨迭代保留 top-2 输出，这样一个晚到的好结果不会被晚到的坏结果覆盖。
4. 将提示链与路由组合：路由器选择三条链之一。测量 token 成本 vs 单个大提示词方案。
5. 选择你的一个生产功能。画出工作流图。数步骤。智能体在这里真的更好吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 工作流 | "预定义流程" | 工程师掌控的 LLM 和工具调用图 |
| 智能体 | "自主 AI" | 模型掌控的图；动态工具指挥 |
| 增强型 LLM | "带工具的 LLM" | LLM + 搜索 + 工具 + 记忆；原子单元 |
| 提示链 | "顺序调用" | 调用 N 的输出是调用 N+1 的输入 |
| 路由 | "分类器分发" | 选择哪条链/模型处理输入 |
| 并行化 | "扇出" | N 个并发调用；通过分段或投票聚合 |
| 编排器-工作者 | "调度智能体" | 编排器 LLM 动态选择专家 LLM |
| 评估器-优化器 | "提议者 + 评判者" | 迭代直到评估器通过；Self-Refine 的泛化 |

## 延伸阅读

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 五种工作流模式
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 伴侣学科
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图何时值得其成本
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 编排器-工作者模式的产品化
