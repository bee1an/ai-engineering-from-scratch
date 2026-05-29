# 规划：HTN 与进化搜索

> 符号规划处理计划可证明正确的情况。进化代码搜索处理适应度函数可机器检查的情况。ChatHTN（2025）和 AlphaEvolve（2025）展示了各自与 LLM 配对时能解锁什么。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 02 (ReWOO and Plan-and-Execute)
**Time:** ~75 minutes

## 学习目标

- 解释层次任务网络：任务、方法、算子、前置条件、效果。
- 描述 ChatHTN 的混合循环 — 符号搜索加 LLM 后备分解。
- 解释 AlphaEvolve 的进化循环以及为什么它只在有程序化评估器时才有效。
- 用 stdlib 实现一个玩具 HTN 规划器加一个玩具进化搜索。

## 问题

ReWOO（Lesson 02）、Plan-and-Execute 和 ReAct 覆盖了大多数智能体规划。两种情况它们处理不好：

1. **需要可证明正确性的计划。** 调度、航线规划、合规工作流 — 计划必须在构造上就是正确的。一个偶尔幻觉出步骤的流畅 LLM 计划是不可接受的。
2. **有机器可检查适应度函数的优化。** 矩阵乘法、调度启发式、编译器 pass — 目标不是"一个正确的计划"而是"最好的计划"。

HTN 规划和 AlphaEvolve 解决两个不同的问题。两者都将 LLM 用作放大器，而非替代品。

## 核心概念

### 层次任务网络

HTN 是：

- **Tasks** — 复合的（待分解）和原始的（直接可执行）。
- **Methods** — 将复合任务分解为子任务的方式，带前置条件。
- **Operators** — 带前置条件和效果的原始动作。
- **State** — 一组事实。

规划：给定一个目标任务和初始状态，找到一个分解为原始算子的序列，其前置条件依次满足。

HTN 比 LLM 更古老，至今仍是可证明正确计划的参考。

### ChatHTN（Gopalakrishnan et al., 2025）

ChatHTN（arXiv:2505.11814）交替使用符号 HTN 和 LLM 查询：

1. 尝试用现有方法分解当前复合任务。
2. 如果没有方法适用，问 LLM："你会如何在状态 `s` 中分解 `task`？"
3. 将 LLM 响应翻译为候选子任务。
4. 对照算子 schema 验证；拒绝无效分解。
5. 递归。

论文的核心主张：每个产出的计划都是可证明正确的，因为 LLM 建议只作为候选分解进入，永远不作为直接的计划编辑。符号层拥有正确性；LLM 扩展方法库。

在线方法学习（OpenReview `gwYEDY9j2x`，2025 后续工作）添加了一个学习器，通过回归泛化 LLM 产出的分解 — 将 LLM 查询频率降低最多 75%。

### AlphaEvolve（Novikov et al., 2025）

AlphaEvolve（arXiv:2506.13131, DeepMind, June 2025）是另一种东西：由 Gemini 2.0 Flash/Pro 集成编排的进化代码搜索。

循环：

1. 从种子程序 + 程序化评估器（返回适应度分数）开始。
2. LLM 集成提出变异。
3. 通过评估器运行变异。
4. 保留最好的；再次变异。

已发表的成果：

- 56 年来首次改进 4x4 复数矩阵乘法的 Strassen 算法（48 次标量乘法）。
- 通过 Borg 调度启发式回收 0.7% 的 Google 算力。
- 在前沿工作负载上 FlashAttention 加速 32%。

硬约束：适应度函数必须是机器可检查的。对散文答案的进化搜索不会收敛。

### 何时用哪个

| 问题类别 | 使用 | 原因 |
|---------|------|------|
| 有硬约束的调度 | HTN + ChatHTN | 可证明的正确性 |
| 编译器优化 | AlphaEvolve | 机器可检查的适应度 |
| 多步任务执行 | ReAct / ReWOO | LLM 在循环中，无形式保证 |
| 有测试的代码改进 | AlphaEvolve | 测试就是评估器 |
| 策略约束的自动化 | HTN | 前置条件编码策略 |

### 这个模式哪里会出错

- **没有算子的 HTN。** 没有前置条件/效果 schema，正确性主张就崩溃了。ChatHTN 的"LLM 建议分解"需要 schema 来拒绝无效动作。
- **没有真实评估器的 AlphaEvolve。** "问 LLM 代码是否更好"不是适应度函数。评估器必须是确定性的且快速的。
- **过度工程。** 大多数智能体任务不需要这两者。先用 ReAct 或 ReWOO。

## Build It

`code/main.py` 实现两个玩具：

- 一个 stdlib HTN 规划器，带算子、方法、前置条件、效果和一个 `LLMFallback`，在没有方法匹配复合任务时启动。"LLM"是一个脚本化分解器，所以规划器可以离线运行。
- 一个 stdlib 进化搜索，在算术程序上进行：生长表达式使其输出在测试集上最小化 `|f(x) - target|`。评估器是确定性的。

运行：

```
python3 code/main.py
```

trace 展示 HTN 规划器分解一个复合任务（中途有 LLM 后备）和进化循环收敛到目标表达式。

## Use It

- **HTN 规划器** — `pyhop`、`SHOP3`，或为领域特定的策略执行自建。
- **ChatHTN** — 研究代码；模式（符号 + LLM 后备）可以干净地移植到任何 HTN 规划器。
- **AlphaEvolve** — DeepMind 论文；模式（集成 + 评估器）可复现。OpenEvolve 和类似的开源分支正在涌现。
- **智能体框架** — 目前没有一个提供一等的 HTN 或 AlphaEvolve。将其构建为子智能体或后台 worker。

## Ship It

`outputs/skill-hybrid-planner.md` 生成一个混合规划器脚手架（HTN 或进化），LLM 角色被显式限定。

## 练习

1. 用回溯扩展 HTN 规划器：当算子的后置条件在运行时失败时，回滚并尝试下一个方法。
2. 给 ChatHTN 添加 LLM 方法缓存：当 LLM 在状态模式 `P` 中分解任务 `T` 时，存储结果。下次调用时先检查方法库。
3. 将进化搜索评估器换成真实测试套件。进化一个通过 20 个测试用例的排序函数；报告收敛所需的代数。
4. 阅读 AlphaEvolve 的评估器设计笔记。为你关心的领域设计一个评估器（SQL 查询优化、测试套件最小化、部署 YAML）。
5. 组合：用 HTN 将复合任务分解为子任务，然后对每个子任务的原始算子使用进化搜索。哪里出彩，哪里过度工程？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| HTN | "层次规划器" | 带算子、前置条件、效果的任务分解 |
| Method | "分解规则" | 将复合任务拆分为子任务的方式 |
| Operator | "原始动作" | 带前置条件和效果的具体步骤 |
| ChatHTN | "LLM + HTN" | 符号规划器在没有方法匹配时询问 LLM |
| AlphaEvolve | "进化代码搜索" | LLM 集成变异代码；确定性评估器选择 |
| Fitness function | "评估器" | 确定性的、机器可检查的输出分数 |
| Online method learning | "缓存的 LLM 分解" | 存储 + 泛化 LLM 计划以降低查询成本 |

## 延伸阅读

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — 符号 + LLM 混合规划器
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — 带 LLM 变异的进化代码搜索
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 何时用规划器 vs 简单循环
