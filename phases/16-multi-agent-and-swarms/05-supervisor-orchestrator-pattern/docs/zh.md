# 监督者 / 编排器-工作者模式

> 一个主导 agent 规划和委派；专业化工作者在并行上下文中执行并汇报。这是 Anthropic Research 系统背后的模式（Claude Opus 4 作为主导，Sonnet 4 作为子 agent），在内部研究评估中比单 agent Opus 4 高出 +90.2%。Anthropic 的工程博文报告称 BrowseComp 上 80% 的方差仅由 token 使用量解释——多 agent 主要因为每个子 agent 获得全新的上下文窗口而获胜。本课从原语构建监督者模式，并涵盖生产部署的 2026 工程经验。

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## 问题

研究是单 agent 系统失败的典型任务。你问"2023 到 2026 年多智能体系统有什么变化？"单个 agent 顺序读五篇论文，用它们的文本填满一半上下文，然后必须同时推理所有论文。它读到第五篇时已经忘了第一篇。它无法并行化。

监督者模式修复了这个问题：一个主导 agent 规划搜索，将每个子问题委派给一个工作者，然后综合。每个工作者为一个窄问题获得自己的 200k-token 窗口。主导永远不看原始论文——只看工作者摘要。

Anthropic 的生产 Research 系统报告在内部研究评估中比单个 Opus 4 高出 +90.2%。同一篇博文指出 BrowseComp 方差的 80% 由*token 使用量*单独解释。每个子 agent 的全新上下文是主要机制。

## 概念

### 模式

```
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

主导永远不读原始材料。工作者在主导综合之前看不到彼此的工作。每个箭头是一个带窄制品的 handoff。

### 为什么它赢

三个机制：

1. **每个子 agent 的全新上下文。** 探索"FIPA-ACL 遗产"的工作者不携带主导花在规划上的 40k token。它为一个问题获得 200k 窗口。
2. **通过提示专业化。** 主导的提示是"分解和综合"，不是"研究"。每个工作者的提示是窄的："找出 X 中有什么变化。"聚焦的提示产出聚焦的输出。
3. **并行性。** 工作者并发运行。挂钟时间大约是 `max(worker_times) + plan + synthesis`，而不是 `sum(worker_times)`。

### 工程经验（Anthropic 2025）

Anthropic 博文列出了几个在 2026 年仍然相关的生产经验：

- **根据查询复杂度调整投入。** 简单查询：一个 agent，3-10 次工具调用。复杂查询：10+ 个 agent。主导必须估计这一点，而不是调用者。
- **先广后窄。** 先分解为广泛的子问题，然后如果答案需要深度则为每个子问题生成更多工作者。
- **彩虹部署。** Agent 是长时间运行且有状态的。传统蓝绿部署不适用。Anthropic 使用彩虹：新版本逐步推出，旧版本排空。
- **Token 使用量主导。** 多 agent 是单 agent 的约 15 倍 token。只在任务价值证明成本合理时运行。

### LangGraph 的转变

LangGraph 最初附带了一个 `langgraph-supervisor` 库，带有高级 `create_supervisor` 辅助函数。2025 年 LangChain 将推荐改为直接通过工具调用实现监督者模式，因为工具调用对*监督者看到什么*（上下文工程）给予更多控制。库仍然可用；文档现在推荐工具调用形式。

### 失败模式

- **主导幻觉计划。** 如果主导生成的子问题没有分解真正的问题，工作者对错误目标做精确研究。
- **工作者过度探索。** 没有显式范围边界，工作者偏离其分配的子问题并污染综合步骤。
- **综合冲突。** 两个工作者返回矛盾的事实。主导必须要么重新询问（增加一轮）要么明确注明分歧。默默选择一方是最糟糕的失败：用户永远不知道发生了分歧。

### 何时监督者是错误的

- **顺序任务。** 如果步骤 2 确实需要步骤 1 的输出，并行性没有收益。使用流水线（CrewAI Sequential、LangGraph 线性图）。
- **简单查询。** 单 agent 处理更快更便宜。在生成工作者之前使用主导的"调整投入"检查。
- **严格确定性。** 监督者使用 LLM 选择的委派。当审计/重放比适应性更重要时，静态图更好。

## Build It

`code/main.py` 使用 `threading` 实现了三个并行工作者的监督者。主导将查询分解为子问题，工作者在每个子问题上并发运行，主导综合。没有真正的 LLM——工作者是脚本化的，模拟获取和摘要。

关键结构：

- `Lead.plan(query)` 将查询拆分为 3 个子问题。
- `Worker.run(sub_q)` 返回一个假摘要（在生产中可以是任何使用工具的 agent）。
- `Lead.run(query)` 在线程中启动工作者，join，然后综合。

运行：

```
python3 code/main.py
```

输出显示计划、带开始/结束时间戳的并行工作者追踪，以及最终综合。你可以看到挂钟收益：三个 0.3 秒的工作者在约 0.35 秒内运行，而不是 0.9 秒。

## Use It

`outputs/skill-supervisor-designer.md` 接受用户查询并产出监督者模式设计：主导系统提示、工作者角色、子问题分解规则和综合模板。在构建新的研究风格 agent 系统之前使用。

## Ship It

部署监督者模式前的检查清单：

- **模型配对。** 主导用推理级模型（Opus 级、`o3` 级）。工作者用更快更便宜的模型（Sonnet、`o4-mini`）。
- **工作者超时。** 任何超过中位运行时间 2 倍的工作者被终止；主导要么用更窄范围重新生成，要么不带它继续。
- **每工作者 token 上限。** 硬限制（比如预期综合输入的 10 倍）防止失控工作者炸掉预算。
- **可观测性。** 追踪主导的计划、每个工作者的工具调用和综合。这是任何事后调试的基础。
- **彩虹推出。** 有状态的长时间运行 agent 需要渐进版本过渡，而不是热切换。

## 练习

1. 运行 `code/main.py`，然后修改主导生成 5 个工作者而不是 3 个。观察挂钟效果。在这个演示中，工作者数量到多少时生成开销超过并行节省？
2. 实现工作者超时：终止任何运行超过 0.5 秒的工作者，让主导综合剩余结果。你需要什么可观测性来知道一个工作者被切断了？
3. 在主导的综合中添加冲突检测步骤：如果两个工作者返回矛盾答案，主导注明分歧而不是选择一个。如何在不调用 LLM 的情况下检测矛盾？
4. 阅读 Anthropic 的 Research 系统工程博文。列出三个这个玩具演示需要采用才能在生产中运行的实践。
5. 比较 LangGraph 的 `create_supervisor`（遗留）与新的工具调用推荐。哪个给你更好的控制监督者看到什么？为什么 Anthropic 明确只将子答案而不是原始工作者上下文传入综合？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 监督者 | "主导 agent" | 规划、委派和综合的编排器 agent。不自己做工作。 |
| 工作者 | "子 agent" | 由监督者调用的聚焦 agent，范围窄，有自己的上下文窗口。 |
| 编排器-工作者 | "监督者模式" | 同一件事，不同名称。2026 文献两者都用。 |
| 全新上下文 | "干净窗口" | 工作者的上下文从其系统提示和分配的问题开始，不是主导的历史。 |
| 彩虹部署 | "渐进推出" | 长时间运行的有状态 agent 需要版本化的排空替换，而不是蓝绿。 |
| Token 主导 | "上下文是变量" | 根据 Anthropic，研究评估方差的 80% 来自总 token 使用量，而不是模型选择。 |
| 调整投入 | "匹配 agent 数量与复杂度" | 主导估计查询难度，相应生成 1 vs 10+ 个工作者。 |
| 综合冲突 | "工作者不同意" | 两个工作者返回矛盾事实；主导必须表面化分歧，而不是默默选一个。 |

## 延伸阅读

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 监督者模式的生产参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 工具调用监督者现在是推荐形式
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — 遗留辅助函数，2026 生产中仍在使用
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 基于 handoff 的监督者变体
