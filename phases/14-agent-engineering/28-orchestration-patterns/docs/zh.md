# 编排模式：Supervisor、Swarm、Hierarchical

> 2026 年的框架中反复出现四种编排模式：supervisor-worker、swarm / peer-to-peer、hierarchical、debate。Anthropic 的指导原则："关键是为你的需求构建正确的系统。"从简单开始；只有当单个 agent 加五种 workflow 模式不够用时，才引入拓扑。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 25 (Multi-Agent Debate)
**Time:** ~60 minutes

## 学习目标

- 说出四种反复出现的编排模式，以及各自适用的场景。
- 描述 2026 年 LangChain 的建议：基于 tool call 的 supervision vs supervisor 库。
- 解释 Anthropic 的"构建正确的系统"原则，以及它如何决定拓扑选择。
- 用 stdlib 对着一个公共脚本化 LLM 实现全部四种模式。

## 问题

团队在真正需要之前就急着上"多智能体"。四种模式在各框架中反复出现；一旦你能叫出它们的名字，就能选对——或者干脆跳过拓扑。

## 概念

### Supervisor-worker

- 一个中央路由 LLM 将任务分派给专家 agent。
- 决策：回到自身、交给专家、终止。
- 专家之间不互相通信；所有路由都经过 supervisor。

框架：LangGraph `create_supervisor`、Anthropic orchestrator-workers、CrewAI Hierarchical Process。

**2026 年 LangChain 建议：** 通过直接 tool call 实现 supervision，而不是用 `create_supervisor`。这样能更精细地控制 context engineering——你决定每个专家看到什么。

### Swarm / peer-to-peer

- Agent 通过共享的 tool surface 直接交接。
- 没有中央路由器。
- 比 supervisor 延迟更低（跳数更少）。
- 更难推理（没有单一控制点）。

框架：LangGraph swarm topology、OpenAI Agents SDK handoffs（当所有 agent 都能互相交接时）。

### Hierarchical

- Supervisor 管理 sub-supervisor，sub-supervisor 管理 worker。
- 在 LangGraph 中实现为嵌套子图；在 CrewAI 中实现为嵌套 crew。
- 以运维复杂度为代价，扩展到大规模 agent 群体。

何时需要：当单个 supervisor 的 context budget 无法容纳所有专家的描述时。

### Debate

- 并行提议者 + 迭代交叉批评（Lesson 25）。
- 严格来说不算编排——更像验证——但在框架中作为拓扑选项出现。

### CrewAI Crew vs Flow

CrewAI 形式化了两种部署模式：

- **Flow** 用于确定性事件驱动自动化（推荐的生产起点）。
- **Crew** 用于自主的基于角色的协作。

这与上面四种模式正交，但可以映射到拓扑：Flow 通常是 supervisor 或 hierarchical；Crew 通常是带 LLM 路由器的 supervisor。

### Anthropic 的指导

"在 LLM 领域取得成功，不是要构建最复杂的系统。而是要为你的需求构建正确的系统。"

决策顺序：

1. 单个 agent + workflow 模式（Lesson 12）——从这里开始。
2. Supervisor-worker——当你有 2-4 个专家时。
3. Swarm——当延迟比推理清晰度更重要时。
4. Hierarchical——只在 supervisor context budget 不够时。
5. Debate——当准确性比成本更重要时。

### 这种模式在哪里出问题

- **拓扑优先思维。** 还没搞清楚多智能体解决什么问题，就说"我们需要多智能体"。
- **Swarm 中的弹跳交接。** A -> B -> A -> B。用跳数计数器。
- **假层级。** 因为"企业级"搞三层；实际只有两个团队。压缩掉。

## Build It

`code/main.py` 用 stdlib 对着脚本化 LLM 实现全部四种模式：

- `Supervisor` — 中央路由器。
- `Swarm` — peer-to-peer 直接交接。
- `Hierarchical` — supervisor 的 supervisor。
- `Debate` — 并行提议者 + 批评。

每种模式处理相同的三意图任务（退款 / bug / 销售）。Trace 形状不同。

运行：

```
python3 code/main.py
```

输出：每种模式的 trace + 操作计数。Supervisor 最干净；swarm 最短；hierarchical 最深；debate 最贵。

## Use It

- **LangGraph** 用于 supervisor 和 hierarchical（嵌套子图）。
- **OpenAI Agents SDK** 用于 handoffs-as-tools（supervisor 形状）。
- **CrewAI Flow** 用于生产环境的确定性流程。
- **Custom** 用于 debate 或需要精确控制时。

## Ship It

`outputs/skill-orchestration-picker.md` 选择一种拓扑并实现它。

## 练习

1. 把一个 supervisor-worker 转换成 swarm，去掉路由器。什么坏了？什么改善了？
2. 给 swarm 加一个跳数计数器：3 次交接后拒绝。能抓住 A->B->A 的弹跳吗？
3. 为一个 12 专家领域构建两层 hierarchical 系统。没有嵌套时 context budget 在哪里失败？
4. 在生产形状的工作负载上对四种模式做性能分析。哪种在哪个指标上胜出（延迟、成本、准确性、可调试性）？
5. 阅读 Anthropic 的"Building Effective Agents"文章。把你的每个生产流程映射到四种模式之一。有没有映射不干净的？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Supervisor-worker | "路由器 + 专家" | 中央 LLM 分派给专家；专家之间不互相通信 |
| Swarm | "Peer-to-peer" | 通过共享 tool 直接交接；没有中央路由器 |
| Hierarchical | "Supervisor 的 supervisor" | 用于大规模群体的嵌套子图 |
| Debate | "提议者 + 批评" | 并行提议者，交叉批评（Lesson 25） |
| Tool-call-based supervision | "不用库的 supervisor" | 用直接 tool call 实现 supervisor 以控制 context |
| Crew | "自主团队" | CrewAI 的基于角色的协作模式 |
| Flow | "确定性工作流" | CrewAI 的事件驱动生产模式 |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — five patterns + agent vs workflow
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor, swarm, hierarchical
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — debate pattern
