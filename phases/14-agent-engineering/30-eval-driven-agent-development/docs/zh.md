# 评估驱动的 Agent 开发

> Anthropic 的指导："从简单的 prompt 开始，用全面的评估来优化它们，只在需要时才添加多步骤 agentic 系统。"评估不是最后一步。它是驱动 Phase 14 中所有其他选择的外层循环。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** All of Phase 14.
**Time:** ~60 minutes

## 学习目标

- 说出三个评估层——静态 benchmark、自定义离线评估、在线生产评估——以及各自的用途。
- 解释 evaluator-optimizer 紧密循环。
- 描述 2026 年最佳实践：eval 与代码并存，在 CI 中运行，gate PR。
- 将 Phase 14 的每一课连接到它生成的 eval case。

## 问题

Agent 能通过 demo。它们在生产中以 demo 无法预测的方式失败。Benchmark 回答的是"这个模型是否广泛有能力？"而不是"这个 agent 是否在为我的产品交付正确的 patch？"答案是：三层评估，持续运行，每个 guardrail 和学到的规则都映射到一个 eval case。

## 概念

### 三个评估层

1. **静态 benchmark** — SWE-bench Verified 用于代码（Lesson 19）、WebArena/OSWorld 用于浏览/桌面（Lesson 20）、GAIA 用于通用（Lesson 19）、BFCL V4 用于 tool use（Lesson 06）。用于跨模型比较和回归 gating。数据污染是真实的：SWE-bench+ 发现 32.67% 的解决方案泄漏。始终报告 Verified / +-audited 分数。

2. **自定义离线 eval** — 你产品的形状：
   - LLM-as-judge（Langfuse、Phoenix、Opik — Lesson 24）。
   - 基于执行的（运行 patch，检查测试）。
   - 基于轨迹的（将动作序列与 gold 对比；OSWorld-Human 显示顶级 agent 是 gold 的 1.4-2.7 倍）。

3. **在线 eval** — 生产环境：
   - Session replay（Langfuse）。
   - Guardrail 触发的告警（Lesson 16、21）。
   - Per-step 成本/延迟追踪（Lesson 23 OTel span）。

### Evaluator-optimizer（Anthropic）

紧密循环：

1. Proposer 生成输出。
2. Evaluator 判断。
3. 精炼直到 evaluator 通过。

这是 Self-Refine（Lesson 05）的泛化。任何你关心的 agent 流程都可以包裹在 evaluator-optimizer 中以提高可靠性。

### 2026 年最佳实践

- Eval 与代码并存。
- 在每个 PR 的 CI 中运行。
- 基于 eval 分数 gate merge（例如"相对 main 回归不超过 5%"）。
- 每个 guardrail 映射到一个 eval case。
- 每个学到的规则（Reflexion、pro-workflow learn-rule）映射到一个 failure case。

### 串联 Phase 14

Phase 14 的每一课都生成 eval case：

| Lesson | 它生成的 eval case |
|--------|-------------------|
| 01 Agent Loop | Budget 耗尽、无限循环守卫 |
| 02 ReWOO | 工具失败时 planner 正确重新规划 |
| 03 Reflexion | 学到的反思在重试时应用 |
| 05 Self-Refine/CRITIC | Judge 通过精炼后的输出 |
| 06 Tool Use | 参数强制转换有效；未知工具被拒绝 |
| 07-10 Memory | 检索引用匹配来源；过时事实失效 |
| 12 Workflow Patterns | 每种模式产生正确输出 |
| 13 LangGraph | Resume 精确复现状态 |
| 14 AutoGen Actors | DLQ 捕获崩溃的 handler |
| 16 OpenAI Agents SDK | Guardrail 在正确输入上触发 |
| 17 Claude Agent SDK | Subagent 结果返回给 orchestrator |
| 19-20 Benchmarks | SWE-bench Verified 分数、WebArena 成功率、OSWorld 效率 |
| 21 Computer Use | Per-step safety 捕获注入的 DOM |
| 23 OTel | Span 发出必需属性 |
| 26 Failure Modes | Detector 标记已知故障 |
| 27 Prompt Injection | PVE 拒绝被投毒的检索 |
| 28 Orchestration | Supervisor 路由到正确的专家 |
| 29 Runtime Shapes | DLQ 处理 N% 的故障 |

如果你的 eval suite 对每一项都有 case，你就覆盖了 Phase 14。

### 评估驱动开发在哪里失败

- **没有 baseline。** 没有 last-known-good 的 eval 无法解读。存储 baseline。
- **LLM-judge 没有 grounding。** Judge 也会幻觉。CRITIC 模式（Lesson 05）——judge 基于外部工具 grounding。
- **过拟合 eval。** 为 eval 优化会偏离生产有用性。轮换 case。
- **Flaky eval。** 非确定性 case 导致误报。固定 seed，快照 state。

## Build It

`code/main.py` 是一个 stdlib eval harness：

- Case registry，带类别（benchmark、custom、online）。
- 一个被测试的脚本化 agent。
- Evaluator-optimizer 循环：propose、judge、refine 直到通过或达到最大轮数。
- CI gate：聚合通过率 + 相对 baseline 的回归。

运行：

```
python3 code/main.py
```

输出：per-case pass/fail、回归标志、CI gate 判定。

## Use It

- 在与 agent 代码相同的 repo 中编写 eval case。
- 在每个 PR 的 CI 中运行。
- 回归时 fail the build。
- 追踪通过率随时间的变化。
- 将每个生产故障关联到一个新 case。

## Ship It

`outputs/skill-eval-suite.md` 为 agent 产品构建三层 eval suite，带 CI gate 和回归追踪。

## 练习

1. 取一个你的生产故障。写一个能复现它的 eval case。你的 agent 现在能通过吗？
2. 为你的领域构建一个 LLM-judge rubric，包含三个维度（事实性、语气、范围）。对 50 个 session 评分。
3. 将 eval suite 接入 CI。回归 >=5% 时 fail the build。
4. 添加一个轨迹效率指标：agent 走了多少步 vs gold 轨迹？
5. 将 Phase 14 的每一课映射到你 suite 中的一个 eval case。有缺失的吗？那就是要补的缺口。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Static benchmark | "现成的 eval" | SWE-bench、GAIA、AgentBench、WebArena、OSWorld |
| Custom offline eval | "领域 eval" | 在你产品形状上的 LLM-as-judge / exec / trajectory |
| Online eval | "生产 eval" | Session replay、guardrail 告警、成本/延迟追踪 |
| Evaluator-optimizer | "Propose-judge-refine" | 迭代直到 judge 通过 |
| CI gate | "Merge blocker" | Eval 回归时 fail the build |
| Baseline | "Last-known-good" | 用于检测回归的参考分数 |
| Trajectory efficiency | "Steps over gold" | Agent 步数除以人类专家最小步数 |

## 延伸阅读

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — "start simple, optimize with evals"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — the curated benchmark
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — tool-use benchmark
- [Langfuse docs](https://langfuse.com/) — evals + session replay in practice
