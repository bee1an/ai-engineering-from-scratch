# 多智能体辩论与协作

> Du et al.（ICML 2024，"Society of Minds"）运行 N 个模型实例独立提出答案，然后经过 R 轮迭代互相批评以收敛。提升事实性、规则遵循和推理能力。稀疏拓扑在 token 成本上优于全连接。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 05 (Self-Refine and CRITIC)
**Time:** ~60 minutes

## 学习目标

- 解释辩论协议：N 个提议者，R 轮，收敛到共享答案。
- 描述为什么辩论能提升事实性、规则遵循和推理能力。
- 解释稀疏拓扑：不是每个辩论者都需要看到其他所有人。
- 用 stdlib 实现基于脚本化 LLM 的辩论，包含全连接和稀疏变体；衡量 token 成本 vs 准确率。

## 问题

Self-Refine（Lesson 05）是一个模型自我批评 — 有群体思维风险。CRITIC（Lesson 05）将批评建立在外部工具上 — 并非总是可用。辩论引入第三种模式：多个实例、交叉批评、通过分歧收敛。

## 概念

### Society of Minds (Du et al., ICML 2024)

- N 个模型实例独立对同一问题提出答案。
- 经过 R 轮，每个模型阅读其他人的提案并批评。
- 模型根据批评更新自己的答案。
- R 轮后返回收敛答案。

原始实验因成本使用 N=3, R=2。在困难问题（MMLU、GSM8K、Chess Move Validity、传记生成）上，更多 agent 和更多轮次能提升准确率。

跨模型组合优于单模型辩论：ChatGPT + Bard 一起 > 任一单独使用。

### 稀疏拓扑

"Improving Multi-Agent Debate with Sparse Communication Topology"（arXiv:2406.11776, 2024-2025）表明全连接辩论并非总是最优。稀疏拓扑（星形、环形、hub-and-spoke）可以在更低 token 成本下匹配准确率。每个辩论者只看到一部分同伴。

影响：

- 全连接 N=5, R=3 = 5 × 3 = 15 个提案，每个读 4 个同伴 = 60 次批评操作。
- 星形 N=5, R=3（一个 hub + 4 个 spoke）= 15 个提案，spoke 只读 hub = 12 次批评操作。

### 辩论有帮助的场景

- **事实性。** N 个独立提案，交叉检查减少幻觉。
- **规则遵循。** 国际象棋走法合法性 — 一个模型漏掉规则，其他人抓住。
- **开放式推理。** 多种框架逐步缩小到正确答案。

### 辩论有害的场景

- **延迟敏感的 UX。** N × R 串行轮次是你可能承受不起的延迟。
- **成本敏感的规模。** 每个问题 N × R token。
- **简单事实查询。** 一次查询比五次辩论便宜。

### 2026 年实际应用

- **Anthropic orchestrator-workers**（Lesson 12）— 辩论的一种变体，带综合步骤。
- **LangGraph supervisor**（Lesson 13）— 中央路由器 + 专家 agent 可以将辩论实现为一个节点。
- **OpenAI Agents SDK**（Lesson 16）— agent 来回 handoff 进行迭代批评。
- **Multi-agent evals** — 辩论 + evaluator-optimizer 配对获取 eval 信号。

### 这个模式容易出错的地方

- **收敛坍塌。** 所有 agent 收敛到第一个错误答案。用强制分歧轮次缓解。
- **Hub 失败。** 在星形拓扑中，坏的 hub 会污染所有人。轮换或使用多个 hub。
- **Prompt 同质化。** 所有 agent 使用相同 prompt；产出相同答案。使用多样化的 prompt 和/或模型。

## Build It

`code/main.py` 实现 stdlib 辩论：

- `Debater` 类（脚本化 LLM，带 per-debater 意见漂移）。
- `FullMeshDebate` 和 `SparseDebate` 运行器。
- 三个问题：一个事实性、一个规则性、一个推理性。
- 指标：收敛答案、收敛轮数、总批评操作数。

运行：

```
python3 code/main.py
```

输出：per-protocol 准确率和成本；稀疏在 2/3 问题上以更低成本匹配全连接。

## Use It

- **Anthropic orchestrator-workers** 用于简单 2-3 worker 辩论。
- **LangGraph** 用于带 checkpointing 的有状态多轮辩论。
- **Custom** 用于研究或专门的正确性保证。

## Ship It

`outputs/skill-debate.md` 搭建一个多智能体辩论，可配置拓扑、N、R 和收敛规则。

## 练习

1. 实现"强制分歧"规则：第 1 轮每个辩论者必须产出不同的提案。衡量对收敛速度的影响。
2. 添加置信度加权聚合：辩论者返回 (answer, confidence)；聚合器按置信度加权。有帮助吗？
3. 将一个"agent"替换为具有不同意见的脚本化 LLM。异质性是否提升准确率？
4. 衡量全连接 vs 稀疏在你的 3 个问题上的 token 成本。绘制成本 vs 准确率图。
5. 阅读 Society of Minds 论文。将你的玩具移植到 N=5, R=3。什么坏了？什么变好了？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Debate | "Multi-agent critique" | N 个提议者，R 轮交叉批评，收敛 |
| Full mesh | "Everyone reads everyone" | 每个辩论者每轮读所有同伴 |
| Sparse topology | "Limited peer view" | 辩论者只读一部分同伴 |
| Hub-and-spoke | "Star topology" | 一个中心辩论者，N-1 个 spoke 只读 hub |
| Convergence | "Agreement" | 辩论者收敛到共享答案 |
| Society of Minds | "Du et al. debate paper" | ICML 2024 多智能体辩论方法 |

## 延伸阅读

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — canonical multi-agent debate
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — sparse topology results
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — orchestrator-workers as a debate variant
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — single-model self-critique counterpart
