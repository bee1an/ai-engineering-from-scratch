# 投票、自一致性与辩论拓扑

> 最廉价的聚合方式：采样 N 个独立智能体，多数投票。Wang et al. 2022 的自一致性用一个模型采样 N 次实现了这一点。多智能体将其扩展为**异构**智能体以逃离单一文化——不同模型、不同提示、不同温度、不同上下文。超越多数投票，辩论拓扑很重要：MultiAgentBench（arXiv:2503.01935, ACL 2025）评估了 star / chain / tree / graph 协调方式，发现**graph 最适合研究任务**，且超过 ~4 个智能体后出现"协调税"。AgentVerse（ICLR 2024）记录了两种涌现模式——志愿行为和从众行为——从众既是特性（达成共识）也是风险（群体思维，Lesson 24）。本课映射拓扑空间，构建每种变体，并测量协调税。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 minutes

## 问题

辩论可以提高准确率（Du et al., arXiv:2305.14325）。也可以降低准确率。辩论是否有帮助取决于四个结构性选择：

1. 谁和谁对话（拓扑）。
2. 多少轮（Du 2023：轮次和智能体数量独立起作用）。
3. 智能体是否异构（不同基础模型打破单一文化）。
4. 是否存在对抗性声音（钢人论证 vs. 稻草人论证）。

团队在任务上简单加上"跑 5 个智能体然后投票"往往比单个智能体退步。这些失败不是随机的，它们跟踪拓扑和异构性。本课就是拓扑地图。

## 概念

### 自一致性，单模型基线

Wang et al. 2022（"Self-Consistency Improves Chain of Thought Reasoning"）在 temperature > 0 下对同一模型采样 N 次，并对推理路径答案进行多数投票。在 GSM8K 上的结果：N=40 次采样相比单次贪心解码有显著提升。自一致性是多智能体投票的单智能体前身。

局限：自一致性使用一个基础模型。错误在构造上是相关的。如果模型有系统性偏差，所有 N 个样本都共享它。

### 多智能体投票，异构扩展

将 N 个样本替换为 N 个*不同*的智能体。不同基础模型（Claude、GPT、Llama），不同提示，不同工具访问。好处：不相关错误。代价：不同智能体成本不同；协调它们增加开销。

2026 年异构辩论的规范名称是 **A-HMAD** — Adversarial Heterogeneous Multi-Agent Debate。尚未被普遍采用，但论文用这个术语表示"不同模型辩论，减少单一文化崩溃带来的相关错误"。

### 四种拓扑

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

Star：一个中心，其他所有人只与中心对话。等同于没有反向通道的 supervisor-worker。
Chain：线性，每个智能体看到前一个的输出。类似流水线。
Tree：层次化，用于层次化智能体系统（Lesson 06）。
Graph：任意对任意。包括全连接团和任意 DAG。

### 协调税（MultiAgentBench）

MultiAgentBench（MARBLE, ACL 2025, arXiv:2503.01935）在包含研究、编码和规划的任务集上对 star、chain、tree、graph 进行了基准测试。关键测量结果：

- **Graph** 拓扑在研究任务上获胜。信息任意流动；智能体可以互相批评。
- **Star** 在快速回答事实性任务上获胜。中心过滤和整合。
- **Chain** 在逐步流水线（分阶段精炼）上获胜。
- **协调税**在 graph 拓扑超过 ~4 个智能体后出现。时钟时间和 token 成本增长快于质量。

4 智能体上限是经验性的，不是根本性的。它反映了 2026 年 LLM 上下文容量：每个智能体的上下文被同伴的输出填满，添加第 N+1 个智能体的边际价值在所有人都能看到所有人后下降。

### 多智能体辩论策略（"Should we be going MAD?"）

arXiv:2311.17371 是 2023 年 MAD 策略的综述。被他人复现的关键发现：与自一致性*结构相似*的 MAD 变体（独立采样 + 聚合）在使用相同预算时往往不如自一致性。MAD 在智能体真正异构且辩论具有对抗结构（一个智能体反对）时帮助最大。

### AgentVerse 涌现模式

AgentVerse（ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf）记录了两种即使没有显式设计也会从多智能体辩论中涌现的行为：

- **志愿。** 一个智能体主动提供帮助（"I can take the next step"）。有用：它将工作分配给最适合子任务的智能体。
- **从众。** 一个智能体调整立场以匹配批评者，即使批评者是错的。这是辩论版的谄媚（Lesson 14）。

从众是为什么"辩论直到达成一致"会奖励霸凌者。有界轮次加独立裁判可以缓解。

### 异构性：真正移动准确率的旋钮

2024-2026 年实践文献中的一个模式：将你的 N 个智能体中的一个换成不同的基础模型，比将 N 增加 1 带来更大的准确率提升。直觉是单一文化——每个新的独立错误源比额外的相关样本更有价值。

在极限情况下，异构性胜过数量。三个不同模型在大多数有清晰真实答案的任务上击败五个相同模型的副本。

### 陪审团方法

Sibyl 框架（在 Minsky-LLM 文献中被引用）形式化了"陪审团"——一小组专业化智能体在每个阶段通过投票精炼答案。与简单多数投票不同，陪审团有角色：一个智能体交叉质询，一个提供上下文，一个评分合理性。陪审团方法是简单投票（便宜，易受单一文化影响）和完整 MAD（昂贵，易受从众影响）之间的中间点。

### 投票+辩论占优的场景

- 问题有真实答案（事实、数学、代码行为）。投票收敛有意义。
- 智能体可以访问不同的来源或工具（异构性可用）。
- 轮次有界（通常 2-3 轮）且有独立裁判或验证器。
- 预算允许 3-5 个智能体。在 graph 拓扑上超过 5-7 个，协调税占主导。

### 投票+辩论有害的场景

- 问题是观点性的。智能体收敛到看起来最自信的答案，而非最正确的。
- 所有智能体共享一个基础模型。单一文化使共识无意义。
- 轮次无界。从众每次都赢。
- 任务简单。单个智能体在 N=5 时的自一致性更便宜且同样准确。

## 动手构建

`code/main.py` 实现：

- `run_star(agents, hub, question)` — 中心轮询每个工作者，聚合。
- `run_chain(agents, question)` — 顺序精炼。
- `run_tree(root, children, question)` — 深度为 2 的层次聚合。
- `run_graph(agents, question, rounds)` — 全对全辩论，有界轮次。
- 脚本化异构性旋钮：每个智能体有一个 `error_bias` 表示其系统性错误。
- 测量框架在 N=3, 5, 7 下运行每种拓扑并报告 (accuracy, total_tokens, wallclock_simulated)。

运行：

```
python3 code/main.py
```

预期输出：topology × N → (accuracy, tokens, latency) 的表格。Graph 在 N=3-5 的研究型任务上获胜；star 在快速事实任务上获胜；graph 在 N=7 时显示协调税（延迟膨胀快于准确率）。

## 使用方式

`outputs/skill-topology-picker.md` 是一个技能，读取任务描述并推荐拓扑（star / chain / tree / graph）、N（智能体数量）、异构性配置（使用的基础模型）和轮次上限。

## 上线清单

对于任何集成：

- 从**单一强基础模型的 N=5 自一致性**开始。这是便宜的基线。
- 如果准确率重要，升级到 **N=3 的异构投票**。测量差异。
- 只有当任务有结构（研究、多步骤）且有界轮次可行时，才升级到**辩论拓扑**。
- 始终记录少数簇。当少数派持续正确时，你有一个多样性信号。
- 在准确率旁边基准测试时钟时间和 token。"10 倍成本换更好准确率"是商业决策。

## 练习

1. 运行 `code/main.py`。绘制 graph 拓扑的协调税曲线：accuracy vs N，tokens vs N。在什么 N 处曲线拐点？
2. 实现 A-HMAD：三个具有故意不同偏差的智能体。全相同偏差基线与 A-HMAD 在 Lesson 14 的单一文化攻击上相比如何？
3. 向 graph 拓扑添加一个"裁判"角色，不投票，只对最终共识打分。这是否改变了涌现的从众行为？
4. 阅读 AgentVerse 论文（ICLR 2024）。识别你的实现最强烈展现的涌现行为。你能通过提示变更引出相反的行为吗？
5. 阅读 MultiAgentBench（arXiv:2503.01935）第 4 节（拓扑实验）。使用你的框架在论文的一个任务上复现"graph-wins-research"结果。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Self-consistency | "采样 N 次，投票" | Wang 2022。单模型，N 个 temperature>0 采样，对推理路径多数投票。 |
| Heterogeneity | "不同模型" | 不同基础模型或提示族的集成。打破单一文化。 |
| MAD | "多智能体辩论" | 智能体在多轮中交换批评的通用术语。见 Du 2023。 |
| A-HMAD | "对抗性异构 MAD" | 强调不同模型 + 对抗结构的 MAD 变体。 |
| Topology | "谁和谁对话" | Star、chain、tree、graph。决定信息流。 |
| Coordination tax | "收益递减" | 在 graph 上超过 ~4 个智能体，成本增长快于质量。 |
| Volunteer behavior | "主动帮助" | AgentVerse 涌现模式：智能体主动承担一个步骤。 |
| Conformity behavior | "压力下的一致" | AgentVerse 涌现模式：智能体与批评者对齐。 |
| Jury | "小型专业化面板" | Sibyl 风格集成，有角色（质询者、上下文提供者、评分者）。 |

## 延伸阅读

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — 单模型基线
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) — 智能体数量和轮次独立起作用
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — 拓扑基准，显示 graph 最适合研究，chain 适合流水线
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) — MAD 策略综述；发现 MAD 在相同预算下常输给自一致性
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — 志愿和从众涌现模式
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — 参考基准实现
