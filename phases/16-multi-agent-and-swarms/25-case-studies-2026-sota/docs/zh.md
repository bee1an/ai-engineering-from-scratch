# Case Studies and the 2026 State of the Art

> 三个值得从头到尾研读的生产级参考案例，每一个都展示了 multi-agent 工程的不同侧面。**Anthropic 的 Research system**（orchestrator-worker、15 倍 token 用量、相比单 agent Opus 4 +90.2%、rainbow deployment）是 supervisor 模式的标杆案例。**MetaGPT / ChatDev**（用 SOP 编码软件工程的角色专精；ChatDev 的 "communicative dehallucination"；MacNet 通过 DAG 把规模扩展到 >1000 个 agent，arXiv:2406.07155）是角色分解模式的标杆案例。**OpenClaw / Moltbook**（最初是 Peter Steinberger 在 2025 年 11 月发布的 Clawdbot；改名两次；2026 年 3 月达到 247k GitHub stars；本地 ReAct-loop agents；Moltbook 是仅供 agent 使用的社交网络，上线数日内就有约 230 万 agent 账号，2026-03-10 被 Meta 收购）展示了 population scale 上会发生什么：涌现式经济活动、prompt-injection 风险、国家级监管（2026 年 3 月中国限制政府电脑使用 OpenClaw）。**2026 年 4 月的框架格局：** LangGraph 与 CrewAI 是生产领跑者；AG2 是社区维护的 AutoGen 延续版；Microsoft AutoGen 进入维护模式（已并入 Microsoft Agent Framework，2026 年 2 月发布 RC）；OpenAI Agents SDK 是 Swarm 的生产级继任者；Google ADK（2025 年 4 月）是 A2A-native 的新晋者。所有主流框架现在都支持 MCP，大多数也支持 A2A。本课会从头到尾过一遍每个案例，提炼共通模式，让你能为下一套生产系统挑出对的参考。

**Type:** Learn (capstone)
**Languages:** —
**Prerequisites:** all of Phase 16 (Lessons 01-24)
**Time:** ~90 minutes

## Problem

Multi-agent 工程是一门年轻的学科。生产级参考案例不多，每个又都覆盖了空间中不同的一块。一个一个读有用；当作一组对照来读，更有用。本课把 2026 年的三个标杆案例当作一份从头到尾的阅读清单，钉死共通模式，并梳理框架格局，让你能基于知识而不是营销话术来选型。

## Concept

### Anthropic Research system

生产级 supervisor-worker 案例。Claude Opus 4 负责规划与综合；Claude Sonnet 4 子 agent 并行调研。已发布的工程博客：https://www.anthropic.com/engineering/multi-agent-research-system 。

关键实测结果：

- 在内部研究类评测上相对单 agent Opus 4 提升 **+90.2%**。
- BrowseComp 上 **80% 的方差** 仅由 **token 用量** 解释 —— multi-agent 之所以胜出，很大程度上是因为每个子 agent 都拿到一个全新的上下文窗口。
- 每次查询消耗 **15 倍 token**，相比单 agent。
- 因为 agent 是长时运行且有状态的，所以采用 **Rainbow deployment**。

被固化下来的设计经验：

1. **按查询复杂度匹配投入。** 简单 → 1 个 agent，3-10 次工具调用。中等 → 3 个 agent。复杂研究 → 10 个以上子 agent。
2. **先广后深。** 子 agent 先做广撒网式搜索；lead 进行综合；后续子 agent 再做有针对性的深挖。
3. **Rainbow deploys。** 老版本 runtime 要继续存活，直到其上还在跑的 agent 跑完。
4. **Verification 不是可选项。** 没有显式的 verifier 角色时，系统会被观察到产生幻觉。

这是 supervisor-worker 拓扑（Phase 16 · 05）在生产规模下的参考案例。

### MetaGPT / ChatDev

生产级 SOP 角色分解案例。覆盖 arXiv:2308.00352（MetaGPT）与 arXiv:2307.07924（ChatDev）。

MetaGPT 把软件工程的 SOP 编码成角色 prompt：Product Manager、Architect、Project Manager、Engineer、QA Engineer。论文的核心论述：`Code = SOP(Team)`。每个角色都有一个窄而专的 prompt；角色之间的交接传递结构化产物（PRD 文档、架构文档、代码）。

ChatDev 的贡献：**communicative dehallucination**。Agent 在回答前会主动追问细节 —— designer agent 会先问 programmer 用的是哪种语言，再去画 UI，而不是靠猜。论文报告这种做法在 multi-agent 流水线上能可量化地降低幻觉。

MacNet（arXiv:2406.07155）把 ChatDev 扩展到 **>1000 个 agent，借助 DAG**。每个 DAG 节点是一个角色专精；边编码交接契约。之所以能撑到这个规模，是因为路由是显式的、可离线计算的。

设计经验：

1. **结构比规模更重要。** 一个 5 角色的紧凑 SOP 团队，比一个 50 agent 的无结构组群更强。
2. **交接契约要落到纸面。** 角色之间传递的产物遵循 schema。
3. **Communicative dehallucination** 是一种廉价但承重的模式。
4. **DAG 比 chat 扩得更远。** 当流程是可知的，就把它编码下来。

这是角色专精（Phase 16 · 08）与结构化拓扑（Phase 16 · 15）的参考案例。

### OpenClaw / Moltbook ecosystem

生产级 population-scale 案例。时间线：

- **2025 年 11 月：** Clawdbot（Peter Steinberger 的本地 ReAct-loop 编码 agent）发布。
- **2025 年 12 月 – 2026 年 3 月：** 两次改名（Clawdbot → OpenClaw → 继续以 OpenClaw 运营）。
- **2026 年 2 月：** Moltbook 在同一组原语之上上线，作为仅供 agent 使用的社交网络；上线数日内约 230 万 agent 账号。
- **2026 年 3 月（2026-03-10）：** Meta 收购 Moltbook。
- **2026 年 3 月：** 中国限制政府电脑使用 OpenClaw。
- **2026 年 3 月：** OpenClaw 突破 247k GitHub stars。

当你把数百万 agent 放到一个共享底座上，multi-agent 看起来就是这样：

- **涌现式经济活动。** Agent 之间用 token 支付互相买卖、互相提供服务。
- **Population scale 下的 prompt-injection 风险。** 一个病毒式 agent profile 里的恶意 prompt，能在数小时内传播到数千次 agent-to-agent 交互。
- **国家级的监管反应。** 上线数周内，监管就抵达了这套生态。

本案例的设计经验一半技术、一半治理：

1. **Population scale 下的 multi-agent 是一种新形态。** 单系统层面的最佳实践（verification、角色清晰）依然适用，但不再够用。
2. **Prompt injection 是新的 XSS。** 默认把 agent profile 与跨 agent 消息当作不可信输入处理。
3. **监管比设计周期跑得更快。** 提前规划。
4. **开源 + 病毒式扩张会复利叠加。** 4 个月达到 247k stars 是不寻常的；要为部署期的爆发式负载做设计。

生态细节见 [OpenClaw Wikipedia](https://en.wikipedia.org/wiki/OpenClaw) 以及 CNBC / Palo Alto Networks 的报道。技术底层方面，Clawdbot / OpenClaw 仓库展示了本地 ReAct loop；Moltbook 的公开帖子揭示了上层的 social-graph 架构。

### Framework landscape April 2026

| Framework | Status | Best for | Notes |
|---|---|---|---|
| **LangGraph** (LangChain) | Production leader | structured graph + checkpointing + human-in-the-loop | recommended default for production |
| **CrewAI** | Production leader | role-based crews with Sequential/Hierarchical processes | strong for role decomposition |
| **AG2** | Community maintained | GroupChat + speaker selection | AutoGen v0.2 continuation |
| **Microsoft AutoGen** | Maintenance mode (Feb 2026) | — | merged into Microsoft Agent Framework RC |
| **Microsoft Agent Framework** | RC (Feb 2026) | orchestration patterns + enterprise integration | new entrant; watch |
| **OpenAI Agents SDK** | Production | Swarm successor | tool-return handoff pattern |
| **Google ADK** | Production (April 2025) | A2A-native | Google Cloud integration |
| **Anthropic Claude Agent SDK** | Production | single-agent + Research extension | see the Research system post |

所有主流框架现在都支持 **MCP**，大多数也支持 **A2A**。协议兼容性已经不再是差异点。

### The common patterns across all three cases

1. **Orchestrator + workers**（Anthropic 显式 supervisor、MetaGPT 中 PM 充当 supervisor、OpenClaw 个体 agent + 网络效应）。
2. **结构化交接契约**（Anthropic 的子 agent 任务描述、MetaGPT 的 PRD/架构文档、OpenClaw 的 A2A artifact）。
3. **Verification 作为一等公民角色**（Anthropic 的 verifier、MetaGPT 的 QA Engineer、OpenClaw 网络内的 validator）。
4. **扩展靠的是拓扑 + 底座，而不是单纯加 agent**（rainbow deploys、MacNet DAG、population-scale 底座）。
5. **成本是实打实的，并且公开披露**（15 倍 token、MetaGPT 的 per-role 预算、Moltbook 的 per-interaction 计价）。
6. **安全姿态是显式的**（Anthropic 的 sandboxing、MetaGPT 的角色权限限制、OpenClaw 把 prompt-injection 当作已知攻击面）。

### Choosing a reference for your next project

- **生产级研究 / 知识型任务 → Anthropic Research。** Fresh-context 子 agent 占优。
- **工程 / 工具链工作流 → MetaGPT / ChatDev。** Roles + SOPs + 交接契约。
- **网络效应型社交产品 → OpenClaw / Moltbook。** 底座 + 涌现式经济。
- **经典企业自动化 → CrewAI 或 LangGraph**（生产领跑者，runtime 稳定）。

### The 2026 state-of-the-art summary

2026 年 4 月时这个领域的状态：

- **框架在收敛。** MCP + A2A 支持已成标配。剩下的设计选择是 handoff 语义。
- **评测正在硬化。** SWE-bench Pro、MARBLE、STRATUS mitigation benchmark。Pro 是当下抗污染的真实校验。
- **生产级失败率是可量化的**（Cemri 2025 MAST；真实 MAS 上 41-86.7%）。这个领域已经走出 "demo 看起来很棒" 的时代了。
- **成本是工程上的核心约束。** 每任务 token 成本、每交互墙钟时间、rainbow-deploy 开销。Multi-agent 在准确率上胜出但在成本上落败 —— 这个权衡就是商业决策。
- **监管是近在眼前的输入，不是背景噪音。** 各司法辖区的动作比单个部署周期更快。

## Use It

`outputs/skill-case-study-mapper.md` 是一个 skill，它会读取一份提议中的 multi-agent 系统设计，并把它映射到最接近的案例研究上，浮现出该案例已经验证过的设计决策。

## Ship It

2026 年生产级 multi-agent 的入门规则：

- **从案例出发，不要从零开始。** 在 Anthropic Research / MetaGPT / OpenClaw 中挑最接近的，再做适配。
- **采用 MCP + A2A。** 跨框架的可迁移性是有价值的；协议支持是免费的。
- **用 SWE-bench Pro 或你内部的 Pro-equivalent 做对照。** Verified 已经被污染了。
- **付 verification 税。** 一个独立 verifier 大约消耗你 token 预算的 20-30%，换来可量化的正确性提升。
- **对长时运行的 agent 做 rainbow deploy。** 准备好把数小时的 agent run 当作常态。
- **读 WMAC 2026 与 MAST 的后续工作。** 这个学科推进得很快。

## Exercises

1. 把 Anthropic Research system 那篇博客从头到尾读一遍。指出三个：如果把 Opus 4 换成更小的模型（比如 Haiku 4），会改变的设计决策。
2. 读 MetaGPT 第 3-4 节（arXiv:2308.00352）。把你自己领域（不是软件）的一个 SOP 编码成角色 prompt。这个 SOP 暗含多少个角色？
3. 读 ChatDev（arXiv:2307.07924）。指出 "communicative dehallucination" 的机制。在你现有的某个 multi-agent 系统中实现它。
4. 读关于 OpenClaw 与 Moltbook 的资料。挑一个在 population scale 才会冒出、5-agent 系统中不会出现的具体失败模式。你会如何在工程上对抗它？
5. 挑出你当前的 multi-agent 项目。三个案例中哪一个是最接近的参考？该案例的哪些设计决策你 **还没有** 采纳？写下你这季度会采纳的一个。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Anthropic Research | "The supervisor reference" | Claude Opus 4 + Sonnet 4 子 agent；15 倍 token；相比单 agent +90.2%。 |
| MetaGPT | "SOP as prompts" | 软件工程的角色分解；`Code = SOP(Team)`。 |
| ChatDev | "Agents as roles" | Designer / programmer / reviewer / tester；communicative dehallucination。 |
| MacNet | "Scale ChatDev via DAG" | arXiv:2406.07155；通过显式 DAG 路由扩展到 1000+ agent。 |
| OpenClaw | "Local ReAct-loop agents" | Steinberger 的项目；2026 年 3 月达到 247k stars。 |
| Moltbook | "Agent-only social network" | 230 万 agent 账号；2026 年 3 月被 Meta 收购。 |
| Rainbow deploy | "Multiple versions concurrent" | 让老版本 runtime 持续存活，服务上面还在跑的长时 agent。 |
| Communicative dehallucination | "Ask before answering" | Agent 向同伴追问细节，而不是靠猜。 |
| WMAC 2026 | "The AAAI workshop" | 2026 年 4 月 multi-agent 协调方向的社区焦点。 |

## Further Reading

- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — supervisor-worker 的生产参考
- [MetaGPT — Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352) — SOP 角色分解
- [ChatDev — Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924) — communicative dehallucination
- [MacNet — scaling role-based agents to 1000+](https://arxiv.org/abs/2406.07155) — 基于 DAG 的扩展
- [OpenClaw on Wikipedia](https://en.wikipedia.org/wiki/OpenClaw) — 生态总览
- [WMAC 2026](https://multiagents.org/2026/) — AAAI 2026 Bridge Program Workshop on Multi-Agent Coordination
- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 生产领跑者
- [CrewAI docs](https://docs.crewai.com/en/introduction) — 角色驱动框架
