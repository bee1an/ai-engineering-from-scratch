# CrewAI：基于角色的 Crew 与 Flow

> CrewAI 是 2026 年基于角色的多智能体框架。四个原语：Agent、Task、Crew、Process。两种顶层形态：Crew（自主的、基于角色的协作）和 Flow（事件驱动的、确定性的）。文档直言不讳："对于任何生产就绪的应用，从 Flow 开始。"

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## 学习目标

- 列举 CrewAI 的四个原语（Agent、Task、Crew、Process）及各自的职责。
- 区分 Sequential、Hierarchical 和计划中的 Consensus process；为每种工作负载选择一个。
- 区分 Crew（自主的基于角色）和 Flow（事件驱动的确定性），并解释文档的生产建议。
- 用 `@tool` 装饰器和 `BaseTool` 子类接入工具；推理结构化输出 vs 自由文本。
- 列举 CrewAI 的四种记忆类型及各自的适用场景。
- 用 stdlib 实现一个三智能体 crew（researcher、writer、editor）来产出简报。
- 识别 CrewAI 的三种失败模式：提示膨胀、manager-LLM 税、脆弱交接。

## 问题

采用多智能体框架的团队撞上同一堵墙。"自主协作"在演示中听起来很棒。然后客户提了一个 bug，你需要确定性重放。或者财务问每次运行 LLM 路由的 crew 花多少钱。或者值班人员需要知道凌晨 3 点哪个智能体卡住了。

自由形式的 LLM 路由 crew 无法干净地回答这些问题。纯 DAG 全部能回答，但失去了头脑风暴智能体需要的探索性形态。

CrewAI 的拆分对这个权衡是诚实的。Crew 用于协作的、基于角色的、探索性的工作。Flow 用于事件驱动的、代码掌控的、可审计的生产。同一个框架，两种形态，按场景选择。

## 概念

### 四个原语

CrewAI 的表面很小。记住这些，其余都是配置。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。backstory 是承重的。它塑造语气、判断力、智能体何时停止。Tools 是智能体可以调用的函数（下文详述）。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。可复用的工作单元。`expected_output` 是契约。`context` 列出上游任务，其输出会被传入。`output_pydantic` 强制结构化形状。
- **Crew。** 容器。拥有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory` + `verbose` + `manager_llm` 设置。
- **Process。** 执行策略。Sequential、Hierarchical、Consensus（计划中）。决定运行的形状。

Agent 之间不直接看到彼此。Task 引用 Agent。Crew 排列 Task。Process 决定谁选择下一个 Task。这就是全部心智模型。

> **验证版本** CrewAI 0.86（2026-05）。更新版本可能重命名或合并 process 类型；在依赖特定形状前请查看 [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential。** Task 按声明顺序运行。Task N 的输出作为 `context` 对 Task N+1 可用。最低成本。最可预测。当顺序固定时使用。
- **Hierarchical。** 一个 manager Agent（独立的 LLM 调用）在专家之间路由。CrewAI 从你的 `manager_llm` 配置或默认值生成 manager。Manager 每轮选择下一个 task，可以拒绝或重新路由。当你有四个或更多专家且顺序确实依赖于先前输出时使用。
- **Consensus。** 计划中，尚未在公开 API 中实现。文档为未来基于投票的 process 保留了这个名称。今天不要依赖它。

Hierarchical 在每个专家调用之上增加一个每轮 LLM 调用（manager）。五步运行的 token 成本可能翻三倍。只有当你需要路由时才为此付费。

### Crew vs Flow

这是文档在 2026 年开篇的框架。

- **Crew。** LLM 驱动的自主性。框架在运行时选择形状。适合：研究、头脑风暴、初稿、路径本身就是答案的场景。难以重放。难以测试。原型开发便宜。
- **Flow。** 你掌控的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记当另一步发出该 topic 时触发的步骤。每步是纯 Python（内部可以调用 Crew）。适合：生产。可观测。可测试。确定性。

文档的 2026 生产建议：从 Flow 开始。当自主性能带来回报时，将 Crew 作为 Flow 步骤内的 `Crew.kickoff()` 调用折入。Flow 给你审计轨迹，Crew 给你探索。组合，不要二选一。

### 工具集成

三种方式给 Agent 一个工具。选择最简单的那个。

1. **`@tool` 装饰器。** 纯函数变成工具。签名是 schema；docstring 是 LLM 看到的描述。适合一次性辅助函数。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 基于类的工具，带显式 args schema、async 支持、重试。当工具有状态（客户端、缓存）或需要结构化参数时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 提供第一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一个 import 即可接入。

结构化输出使用 Pydantic。在 Task 上传入 `output_pydantic=MyModel`。CrewAI 对照模型验证 LLM 响应，要么强制转换要么重试。配合紧凑的 `expected_output` 字符串使用。自由文本输出适合草稿；结构化输出是下游 Flow 能消费的。

### 记忆钩子

CrewAI 开箱即用提供四种记忆类型。它们可以组合：一个 Crew 可以同时启用全部四种。

> **验证版本** CrewAI 0.86（2026-05）。近期版本通过统一的 `Memory` 系统路由一切，包装这四个存储。下面的概念模型仍然成立，但公开类表面可能在更新版本中收敛为单一 `Memory` 入口点；请查看 [CrewAI memory docs](https://docs.crewai.com/concepts/memory) 获取当前 API。

- **短期。** 单次运行内的对话缓冲。运行结束时清除。
- **长期。** 跨运行持久化。存储在向量数据库中（默认 Chroma，可替换）。按与当前任务的相似度检索。
- **实体。** 每实体事实。"客户 X 在企业计划上。"按实体键控，不按相似度。跨运行存活。
- **上下文。** 组装时检索。在 Agent 需要时拉取相关记忆，不是预加载。

在 Crew 上用 `memory=True` 或按类型配置启用。由你配置的 embeddings 提供者支持（默认 OpenAI，可替换为本地）。记忆是 CrewAI 相对于更薄框架证明其价值的地方之一；纯 LangGraph 需要你自己接入每一种。

### CrewAI 适合的场景

- 三到六个有命名角色和协作工作流的智能体。起草、审查、规划、头脑风暴。
- LLM 对下一步的判断本身就是价值的路由（Hierarchical）。
- 团队更乐于阅读 `role + goal + backstory` 而非图定义的场景。

### CrewAI 不适合的场景

- 严格排序的确定性 DAG。使用 LangGraph（Lesson 13）。图形状是正确的抽象；CrewAI 的角色框架是摩擦。
- 亚秒级延迟预算。Hierarchical 增加往返。即使 Sequential 也序列化包含 backstory 和先前输出的提示词。
- 单智能体循环。跳过框架；一个 agent loop（Lesson 1）加工具注册表更短。

Lesson 17（Agent Framework Tradeoffs）在矩阵中展示了这一点。简短版本：CrewAI 位于"协作式基于角色"的角落。

### 依赖形状

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。Star 数：见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（2026-05 快照）。AWS Bedrock 集成有文档；供应商基准测试报告在 QA 工作负载上相对 LangGraph 有显著加速，但方法论（数据集、硬件、评估指标）未公开，所以将框架供应商数字仅作为方向性参考。

### 这种模式出错的地方

- **backstory 导致的提示膨胀。** 每个智能体 2000 字的 backstory 加五个智能体的 crew，在第一次工具调用前就烧光了上下文预算。保持 backstory 在 200 字以内。跨智能体复用短语；不要重复五次 house style。
- **Manager-LLM token 税。** Hierarchical process 在每个专家调用前增加一个 manager LLM 调用。五个 task 的 crew 是六次 LLM 调用而非五次，且 manager 调用携带完整任务列表加先前输出。除非路由依赖输出，否则切换到 Sequential。
- **脆弱交接。** Task N 的 `expected_output` 是"一个大纲"。Task N+1 将其作为 `context` 读取并尝试解析三个部分。LLM 产出了四个。下游 Agent 即兴发挥。用 Task N 上的 `output_pydantic` 修复，使 Task N+1 读取类型化对象而非自由文本。
- **Crew 直接上生产。** 自由形式的 Crew 在没有 Flow 包装的情况下发布到生产。输出变异性高；重放不可能；值班人员无法 diff 一次坏运行与好运行。用 Flow 包装。

## Build It

`code/main.py` 实现了两种形态的 stdlib 版本加一个三智能体 crew。

形状：

- `Agent`、`Task` dataclass，匹配 CrewAI 的表面。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行 task，将输出作为 `context` 串联。
- `HierarchicalCrew.kickoff(topic)` 添加一个 manager Agent 每轮选择下一个专家，在"done"时停止。
- `Flow` 带 `@start` 和 `@listen(topic)` 装饰器、一个小型事件循环和轨迹。
- `tool(name)` 装饰器，镜像 CrewAI 的 `@tool` 形状。
- `Memory` 带 `short_term`、`long_term`、`entity` 存储；模拟相似度使用 numpy。
- Mock LLM 响应是基于 role 加输入前缀的硬编码字符串。无网络。确定性。

具体演示：researcher、writer、editor crew 产出关于"agent engineering 2026"的简报。Researcher 拉取（模拟的）来源。Writer 起草。Editor 精炼。同一个 crew 通过 Flow 运行以展示确定性形态。

运行：

```bash
python3 code/main.py
```

轨迹覆盖：sequential crew 通过 `context` 串联输出、hierarchical crew 的 manager 选择（researcher、writer、editor，然后"done"）、flow 用显式 topic（`researched`、`drafted`、`edited`）运行相同三步、通过 `@tool` 路由的工具调用、以及长期记忆在两次 kickoff 间存活。

Crew 轨迹是流动的；manager 原则上可以重新排序。Flow 轨迹是固定的。这个选择就是本课的要点。

## Use It

- **CrewAI Flow** 用于生产。即使 Flow 只有一步调用 `Crew.kickoff()`。Flow 给出审计边界。
- **CrewAI Crew (Sequential)** 用于顺序明确的协作工作，特别是初稿和审查循环。
- **CrewAI Crew (Hierarchical)** 当路由依赖输出且你有四个或更多专家时。
- **LangGraph**（Lesson 13）用于显式状态机、持久恢复、严格排序。
- **AutoGen v0.4**（Lesson 14）用于 actor 模型并发和故障隔离。
- **OpenAI Agents SDK**（Lesson 16）用于 OpenAI 优先的产品，带 handoff 和 guardrail。
- **Claude Agent SDK**（Lesson 17）用于 Claude 优先的产品，带子智能体和 session store。

## Ship It

`outputs/skill-crew-or-flow.md` 为任务选择 Crew vs Flow 并搭建最小实现。硬拒绝：没有 backstory 的 Crew、没有显式 topic 的 Flow、少于三个专家的 Hierarchical。

## 陷阱

- **Backstory 当调味料。** 它塑造输出。每个智能体测试三个变体；变异是真实的。选一个，冻结它。
- **跳过 `expected_output`。** 没有每个 task 的契约，下游 task 接收 LLM 产出的任何东西。Crew 运行了；审计失败了。
- **记忆始终开启。** 长期记忆每次运行都写入。向量数据库增长。检索变得嘈杂。将写入范围限定到事实是持久的 task。
- **Manager 提示漂移。** Hierarchical 的 manager 提示是隐式的。如果路由变得奇怪，在 verbose 模式下 dump 它并阅读。
- **Crew 中的工具副作用。** Crew 可能比预期更多次调用工具。POST、DELETE、支付属于 Flow 步骤，永远不要放在 Crew 工具中。

## 练习

1. 将 Sequential crew 转换为 Flow。数变异性下降的触点。注意可读性下降的地方。
2. 给 crew 添加实体记忆：关于客户的事实跨 kickoff 持久化。验证检索拉取了正确的实体。
3. 实现一个 Hierarchical process，其中 manager 拒绝路由到 editor，直到 writer 的输出至少有三段。追踪重试。
4. 为（模拟的）web 搜索接入一个 `BaseTool` 子类。比较轨迹形状 vs `@tool` 装饰器版本。
5. 给 editor task 添加 `output_pydantic=Brief`，其中 `Brief` 有 `title`、`summary`、`sections`。让 writer task 输出一次格式错误的 JSON；验证 CrewAI 在轨迹中的重试行为。
6. 阅读 CrewAI 的文档介绍。将玩具移植到真正的 `crewai` API。stdlib 版本跳过了哪些保证？
7. 将 AgentOps 或 Langfuse（Lesson 24）接入真实运行。你在 stdlib 版本中遗漏了哪些轨迹？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Agent | "角色" | Role + goal + backstory + tools |
| Task | "工作单元" | Description + expected output + assignee + 可选结构化输出 |
| Crew | "智能体团队" | Agent + Task + Process 的容器 |
| Process | "执行策略" | Sequential / Hierarchical / Consensus（计划中） |
| Flow | "确定性工作流" | 事件驱动的、代码掌控的、可测试的 |
| Backstory | "角色提示" | Agent 的语气和判断力塑造器 |
| `@tool` | "函数工具" | 将函数变成 Agent 可调用工具的装饰器 |
| `BaseTool` | "类工具" | 带 args schema、重试、async 支持的基于类的工具 |
| 实体记忆 | "每实体事实" | 范围限定到客户/账户/问题的记忆 |
| 长期记忆 | "跨运行记忆" | 在 kickoff 之间存活的向量支持记忆 |
| 上下文记忆 | "即时检索" | 在 Agent 需要时拉取的记忆 |
| Manager LLM | "路由智能体" | Hierarchical process 中选择下一个 task 的额外 LLM |
| `expected_output` | "Task 契约" | 告诉 Agent（和审计）返回什么形状的字符串 |

## 延伸阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：概念和推荐的生产路径
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)：事件驱动形态，`@start`，`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)：`@tool`，`BaseTool`，内置工具包
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：多智能体何时有帮助，何时没有
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：状态机替代方案
