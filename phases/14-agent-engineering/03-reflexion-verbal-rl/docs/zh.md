# Reflexion：语言强化学习

> 基于梯度的 RL 需要数千次试验和 GPU 集群来修复一个失败模式。Reflexion（Shinn et al., NeurIPS 2023）用自然语言完成同样的事：每次失败后，智能体写一段反思，存入情景记忆，下一次试验以此为条件。这就是 Letta sleep-time compute、Claude Code 的 CLAUDE.md learnings 和 pro-workflow learn-rule 背后的模式。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 02 (ReWOO)
**Time:** ~60 minutes

## 学习目标

- 说出 Reflexion 的三个组件（Actor、Evaluator、Self-Reflector）以及情景记忆的作用。
- 用 stdlib 实现一个 Reflexion 循环，包含二元评估器、反思缓冲区和全新重试。
- 根据任务特征选择标量、启发式和自评估反馈源。
- 解释为什么语言强化能捕获梯度 RL 需要数千次试验才能修复的错误。

## 问题

智能体执行任务失败了。在标准 RL 中你需要再跑数千次试验、计算梯度、更新权重。昂贵、缓慢，而且大多数生产智能体没有为每次失败准备训练预算。

Reflexion（Shinn et al., arXiv:2303.11366）提出了一个不同的问题：如果智能体只是想想为什么失败，然后把这个想法放进 prompt 再试一次呢？不更新权重。不计算梯度。只是在试验之间存储自然语言。

结果：在 ALFWorld 上超越 ReAct 和其他非微调基线。在 HotpotQA 上优于 ReAct。在代码生成（HumanEval/MBPP）上达到当时的 SOTA。全部不需要一步梯度。

## 核心概念

### 三个组件

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

加上一个数据结构：

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

一次试验运行 Actor。Evaluator 打分。如果分数低，Self-Reflector 生成一段反思（"我选错了工具，因为我把问题误读为问 X，实际上是问 Y"）。反思进入情景记忆。下一次试验从头开始，但能看到反思。

### 三种评估器类型

1. **标量型** — 外部二元信号。ALFWorld 成功或失败。HumanEval 测试通过或失败。最简单，信号最强。
2. **启发式** — 预定义的失败特征。"如果智能体连续两次产生相同动作，标记为卡住。""如果轨迹超过 50 步，标记为低效。"
3. **自评估** — LLM 给自己的轨迹打分。在没有 ground truth 时使用。信号较弱；与工具验证配合效果好（Lesson 05 — CRITIC）。

2026 默认做法是混合使用：有标量信号时用标量，没有时用自评估，启发式作为安全护栏。

### 为什么这个模式能泛化

Reflexion 与其说是一个新算法，不如说是一个命名模式。几乎每个生产级"自愈"智能体都在运行某种变体：

- Letta 的 sleep-time compute（Lesson 08）：一个独立智能体反思过去的对话并写入 memory blocks。
- Claude Code 的 `CLAUDE.md` / "save memory" 模式：反思被捕获为 learnings，预置到未来会话中。
- pro-workflow 的 `/learn-rule` 命令：修正被捕获为显式规则。
- LangGraph 的 reflection nodes：一个节点对输出打分，如果需要则路由到 refine。

所有这些都源于同一个洞察：自然语言是一个足够丰富的媒介，可以在运行之间传递"我从失败中学到了什么"。

### 什么时候有效，什么时候无效

Reflexion 有效的条件：

- 有明确的失败信号（测试失败、工具错误、答案错误）。
- 任务类别可复现（同类问题可以再次被问到）。
- 反思有改进轨迹的空间（足够的动作预算）。

Reflexion 无效的情况：

- 智能体第一次就成功了。
- 失败是外部原因（网络断了、工具坏了）— 反思"网络断了"对未来运行没帮助。
- 反思变成迷信 — 存储了关于一次偶发故障的叙事。

2026 陷阱：记忆腐化。反思不断积累；有些过时或错误；随着情景缓冲区增长，重跑变慢。缓解方案：定期压缩（Lesson 06）、反思 TTL、或独立的 sleep-time 清理智能体（Letta）。

## Build It

`code/main.py` 在一个玩具谜题上实现 Reflexion：生成一个 3 元素列表使其和等于目标值。Actor 输出候选列表；Evaluator 检查和；Self-Reflector 写一行关于哪里出错的诊断。反思进入情景记忆供下一次试验使用。

组件：

- `Actor` — 一个脚本化策略，看到反思后会改进。
- `Evaluator.binary()` — 对目标和的通过/失败判定。
- `SelfReflector` — 生成一行失败诊断。
- `EpisodicMemory` — 带 TTL 语义的有界列表。

运行：

```
python3 code/main.py
```

trace 展示三次试验。试验 1 失败，存储一条反思，试验 2 看到反思后改进但仍然失败，试验 3 成功。与基线运行（无反思）对比 — 它一直卡在试验 1 的答案上。

## Use It

LangGraph 将 reflection 作为节点模式提供。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 将情景缓冲区外化为 markdown 文件。Letta 的 sleep-time compute 在空闲时运行 Self-Reflector，使主智能体保持低延迟。OpenAI Agents SDK 不直接提供 Reflexion；你需要用自定义 Guardrail（按分数拒绝轨迹）和跨运行存活的 memory `Session` 来构建。

## Ship It

`outputs/skill-reflexion-buffer.md` 创建并维护一个情景缓冲区，包含反思捕获、TTL 和去重。给定一个任务类别和一次失败，它输出一条真正有助于下次试验的反思（而不是泛泛的"更仔细一点"）。

## 练习

1. 从二元评估器切换到返回距离度量（离目标多远）的标量评估器。收敛更快吗？
2. 给反思加 10 次试验的 TTL。超过这个时间后，旧反思是有害还是有益？
3. 实现启发式评估器：如果相同动作重复则标记为卡住。这与 Self-Reflector 如何交互？
4. 用一个对抗性 Actor（忽略反思）运行 Reflexion。强制 Actor 注意到反思所需的最小 prompt 工程是什么？
5. 阅读 Reflexion 论文第 4 节关于 AlfWorld 的内容。从概念上复现 130% 成功率提升：相对于 vanilla ReAct 的关键差异是什么？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Reflexion | "自我修正" | Shinn et al. 2023 — Actor、Evaluator、Self-Reflector 加情景记忆 |
| Verbal reinforcement | "无梯度学习" | 自然语言反思预置到下一次试验的 prompt 中 |
| 情景记忆 | "每任务反思" | 一个任务类别的先前反思的有界缓冲区 |
| 标量评估器 | "二元成功信号" | 来自 ground truth 的通过/失败或数值分数 |
| 启发式评估器 | "基于模式的检测器" | 预定义的失败特征（如卡住循环、步数过多） |
| 自评估器 | "LLM 对自身 trace 的判断" | 无 ground truth 时的低信号后备方案 — 与工具验证配合使用 |
| 记忆腐化 | "过时反思" | 情景缓冲区充满过时条目；用压缩/TTL 修复 |
| Sleep-time reflection | "异步自我反思" | 在非关键路径上运行 Self-Reflector，使主智能体保持快速 |

## 延伸阅读

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — 原始论文
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — 生产中的异步反思
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 将情景缓冲区作为上下文的一部分管理
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — reflection 节点模式
