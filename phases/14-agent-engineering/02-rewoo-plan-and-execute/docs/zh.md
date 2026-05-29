# ReWOO 与规划执行：解耦规划

> ReAct 在一个流中交替思考和行动。ReWOO 将它们分离：先做一个完整的计划，然后执行。token 减少 5 倍，HotpotQA 准确率 +4%，而且你可以把规划器蒸馏到 7B 模型中。Plan-and-Execute 将其泛化；Plan-and-Act 将其扩展到网页导航。

**类型：** 构建
**语言：** Python (stdlib)
**前置知识：** Phase 14 · 01 (Agent Loop)
**时间：** ~60 分钟

## 学习目标

- 解释为什么 ReWOO 的 Planner / Worker / Solver 分离比 ReAct 的交错循环节省 token 并提高鲁棒性。
- 实现一个计划 DAG、一个依赖排序的执行器和一个组合 worker 输出的 solver——全部用标准库。
- 使用 2026 年"五种工作流模式"框架（Anthropic）来决定任务应该用先规划后执行还是交错 ReAct。
- 识别何时需要 Plan-and-Act 的合成计划数据来处理长时间跨度的网页或移动端任务。

## 问题

ReAct 的交错 thought-action-observation 循环简单灵活，但每次工具调用都必须携带完整的先前上下文——包括之前的每一个 thought。token 使用量随深度二次增长。更糟的是：当工具在循环中途失败时，模型必须从错误观察中重新推导整个计划。

ReWOO (Xu et al., arXiv:2305.18323, May 2023) 注意到了这一点并做了一个赌注：预先规划好所有事情，并行获取证据，最后组合答案。一次 LLM 调用做规划，N 次工具调用获取证据（可以并行），一次 LLM 调用做求解。代价是灵活性降低（计划是静态的），换来更好的 token 效率和更清晰的故障模式。

## 概念

### 三个角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 生成一个 DAG。每个节点指定一个工具、它的参数以及它依赖哪些更早的节点（引用如 `#E1`、`#E2`）。Workers 按拓扑顺序执行节点。Solver 将所有内容拼接在一起。

### 为什么减少 5 倍 token

ReAct 的 prompt 长度随步数线性增长。在第 10 步，prompt 包含 thought 1 加 action 1 加 observation 1 加 thought 2 加 action 2 加 observation 2，依此类推。每个中间步骤还冗余地包含原始 prompt。

ReWOO 支付一个 planner prompt（大的），N 个小的 worker prompt（每个只是工具调用，没有链），和一个 solver prompt。在 HotpotQA 上论文测量到约 5 倍更少的 token，同时准确率绝对值 +4。

### 为什么更鲁棒

如果 worker 3 在 ReAct 中失败，循环必须在流中推理出错误。在 ReWOO 中，worker 3 返回一个错误字符串；solver 在原始计划的上下文中看到它，可以优雅降级。故障定位是按节点的，而不是按步骤的。

### Planner 蒸馏

论文的第二个结果：因为 planner 不看 observations，你可以用 175B 教师模型的 planner 输出来微调一个 7B 模型。小模型处理规划；推理时不需要大模型。这现在是标准做法——许多 2026 年的生产智能体使用小 planner 和大 executor，或反过来。

### Plan-and-Execute (LangChain, 2023)

LangChain 团队 2023 年 8 月的文章将 ReWOO 泛化为一个模式名称：Plan-and-Execute。前置 planner 输出一个步骤列表，executor 运行每个步骤，一个可选的 replanner 可以在观察结果后修订。这比 ReWOO 更接近 ReAct（replanner 将 observations 带回规划中），但保留了 token 节省。

### Plan-and-Act (Erdogan et al., arXiv:2503.09572, ICML 2025)

Plan-and-Act 将该模式扩展到长时间跨度的网页和移动端智能体。关键贡献是合成计划数据：一个标注的轨迹生成器产生训练数据，其中计划是显式的。用于微调 planner 模型，使其在 WebArena 类任务上超过 30–50 步后仍能工作，而单一 ReAct 轨迹在这些任务上会失去连贯性。

### 何时选择哪个

| 模式 | 适用场景 |
|------|---------|
| ReAct | 短任务、未知环境、需要响应式异常处理 |
| ReWOO | 结构化任务、已知工具、token 敏感、可并行化的证据 |
| Plan-and-Execute | 类似 ReWOO 但在部分执行后有重新规划 |
| Plan-and-Act | 长时间跨度（>30 步）、网页/移动端/computer-use |
| Tree of Thoughts | 搜索值得付出代价时（Lesson 04） |

Anthropic 2024 年 12 月的指导：从最简单的开始。如果任务是一次工具调用加一个摘要，不要构建 ReWOO。如果任务是一个 40 步的研究任务，不要单独用 ReAct。

## 构建

`code/main.py` 实现了一个 toy ReWOO：

- `Planner` — 一个脚本化策略，从 prompt 生成计划 DAG。
- `Worker` — 通过注册表分发每个节点的工具调用。
- `Solver` — 脚本化组合，读取证据并产生最终答案。
- 依赖解析 — 像 `#E1` 这样的引用被替换为更早的 worker 输出。

演示回答"法国首都的人口是多少，四舍五入到百万？"使用两步计划：(1) 查找首都，(2) 查找人口，然后求解。

运行：

```
python3 code/main.py
```

轨迹先显示完整计划，然后是 worker 结果，然后是 solver 组合。将 token 计数（我们打印一个粗略的字符计数）与 ReAct 风格的交错运行进行比较——ReWOO 在这类结构化任务上胜出。

## 使用

LangGraph 将 Plan-and-Execute 作为配方提供（`create_react_agent` 用于 ReAct，自定义图用于 plan-execute）。CrewAI 的 Flows 直接编码该模式：你预先定义任务，Flow DAG 执行它们。Plan-and-Act 的合成数据方法仍主要是研究性的；运行时模式（显式计划 DAG）通过 LangGraph 和 CrewAI Flows 在生产中使用。

## 交付

`outputs/skill-rewoo-planner.md` 根据用户请求和工具目录生成一个 ReWOO 计划 DAG。它在交给执行器之前验证计划（无环、每个引用已解析、每个工具存在）。

## 练习

1. 对独立的计划节点并行化 worker 执行。在一个有 2 个并行组的 6 节点 DAG 上，这能带来什么？
2. 添加一个 replanner 节点，当任何 worker 返回错误时触发。将 ReWOO 变成 Plan-and-Execute 的最小改动是什么？
3. 用小模型（7B 级别）替换 `Planner`，保持 `Solver` 在前沿模型上。比较端到端质量——分离在哪里失败？
4. 阅读 ReWOO 论文第 4 节关于 planner 蒸馏的内容。概念性地复现 175B -> 7B 的结果：你需要什么训练数据，如何评分计划质量？
5. 将 toy 移植到 Plan-and-Act 的轨迹形态：计划是序列而不是 DAG。哪些权衡发生了变化？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| ReWOO | "无观察推理" | 先规划，然后并行获取证据，然后求解——规划 prompt 中没有 observations |
| Plan-and-Execute | "LangChain 的规划执行模式" | ReWOO 加上执行后的可选 replanner 节点 |
| Plan-and-Act | "扩展的规划执行" | 显式 planner/executor 分离，带合成计划训练数据，用于长时间跨度任务 |
| Evidence reference | "#E1, #E2, ..." | 计划节点占位符，在分发时被替换为先前 worker 的输出 |
| Planner distillation | "小 planner，大 executor" | 用大教师模型的 planner 轨迹微调小模型 |
| Token efficiency | "更少的往返" | 论文中在 HotpotQA 上比 ReAct 少 5 倍 token |
| DAG executor | "拓扑分发器" | 按依赖顺序运行计划节点；每层可并行 |

## 延伸阅读

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — 经典论文
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — 带合成计划的扩展 planner-executor
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) — 框架配方
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 选择最简单的有效模式
