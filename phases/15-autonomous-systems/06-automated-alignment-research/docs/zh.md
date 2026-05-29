# 自动化对齐研究（Anthropic AAR）

> Anthropic 在独立沙箱中运行了多个 Claude Opus 4.6 自主对齐研究员的并行团队，通过一个日志存储在所有沙箱之外的共享论坛进行协调（因此智能体无法删除自己的记录）。在 weak-to-strong 训练问题上，AAR 超越了人类研究者。Anthropic 自己的总结指出，规定的工作流往往限制 AAR 的灵活性并降低性能。自动化对齐研究是压缩时间线的步骤，它将时间线压缩到 RSP 旨在检测的那些错位风险。

**Type:** Learn
**Languages:** Python (stdlib, parallel-research-forum simulator)
**Prerequisites:** Phase 15 · 05 (AI Scientist v2), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## 问题

对齐研究在人类研究者时间上很昂贵。可扩展监督、奖励规范或 weak-to-strong 训练等问题需要每次迭代数周的实验。随着前沿能力的推进，对齐工作量的增长速度超过了合格研究者的供给。

自动化对齐研究（AAR）问的是：那些能力正在超越对齐的前沿模型本身能否帮助缩小差距？Anthropic 2026 年关于 AAR 运行的 weak-to-strong 训练研究的报告（alignment.anthropic.com/2026/automated-w2s-researcher/）是此类已部署系统的首批公开结果之一。

结果确实是正面的：AAR 比人类基线更好地解决了一个研究问题。结果也将本阶段发展的每一个关切集中在一个系统中。如果对齐研究可以自动化，那么破坏安全措施的部分也可以。RSP 的 AI R&D 能力阈值就是针对这个循环编写的。

## 概念

### 架构（公开描述）

- **并行智能体。** 多个 Claude Opus 4.6 实例作为 AAR 运行，每个在独立沙箱中。沙箱限制文件系统、网络和进程面。
- **共享论坛。** 发现发布到一个存储后端位于所有沙箱之外的共享论坛。智能体可以读论坛；智能体无法从自己的沙箱内删除或编辑论坛记录。
- **任务分配。** 人类设计的任务队列将研究问题分配给 AAR。AAR 提出分解方案、运行实验、撰写发现。
- **Weak-to-strong 训练焦点。** 已发表的研究针对 weak-to-strong 训练——使用较弱的监督模型训练较强的学生模型。AAR 在此任务上超越了人类研究者。

### 为什么沙箱外日志很重要

日志完整性属性是受控实验和可能被篡改的实验之间的区别。如果 AAR 能编辑自己的日志，它们就能掩盖失败或可疑的实验。Anthropic 直接指出：该属性是信任研究输出的必要条件，且验证起来并非易事。

用基础设施术语类比，这是带有写穿到外部存储的追加式日志（如 S3 object-lock、不可变 Datadog 日志）。智能体没有修改过去记录的凭证。这与 2026 年企业安全团队用来制作抗勒索软件审计轨迹的模式相同。

### 规定工作流的权衡

Anthropic 的报告指出，规定的工作流往往限制 AAR 的灵活性并降低性能。换句话说：如果人类设计者逐步编写研究计划，AAR 的表现不如给予高层目标和分解自由时好。这与智能体研究的跨领域结果一致（LangGraph、Microsoft 的 Agent Framework）——过度指定的脚手架会伤害有能力的智能体。

安全含义是微妙的。提高 AAR 性能的灵活性也是允许未经批准的分解、新颖工具使用和目标漂移的灵活性。权衡是一个设计空间决策：多少灵活性能买到足够的能力来超越人类，而不给智能体留下将目标泛化到分布外的空间？

### 压缩风险

RSP v3.0（Lesson 19）引入了 AI R&D 能力阈值：以与人类 + AI 工具相比有竞争力的成本完全自动化 AI R&D 流水线的能力。DeepMind 的 FSF v3 包含类似的 ML R&D 自主性级别。两个框架都将此阈值视为升级控制的触发器。

AAR 距离阈值还差一步：它自动化了流水线的一部分（特定、范围明确的任务上的对齐研究），但不是端到端的能力开发循环。时间线问题是差距缩小的速度。

压缩时间线是复合失败的关切。如果对齐研究和能力研究以相似速率复合，错位风险面至少与能力一样快地增长。如果能力复合更快（历史趋势），差距就会扩大。这就是 AAR 是有条件的好事的论点：每个额外的对齐结果减少差距，当且仅当研究过程是可信的。

### AAR 不替代什么

人类研究者设定任务队列、审查结果并持有宪法权威。AAR 加速流水线的中间部分，而非两端。Anthropic 发布的输出包括 AAR 贡献和人类研究者对发布什么、撤回什么和改进什么的判断。

这匹配了 Lesson 15 中应用于研究本身的提议-然后-提交模式：AAR 提议；人类提交。

## Use It

`code/main.py` 模拟一个多 AAR 研究论坛。三个独立的"研究智能体"并行运行，向追加式论坛发布发现，可选择性地尝试篡改。

观察：

- 沙箱外日志如何捕捉日志篡改尝试。
- 自由分解的任务分配如何在更难的问题上优于固定工作流。
- 如果日志不是防篡改的，单个被攻破的 AAR 如何污染论坛的信号。

## Ship It

`outputs/skill-aar-deployment-review.md` 是自动化对齐研究流水线的部署前审查。

## 练习

1. 运行 `code/main.py`。比较"固定工作流"和"自由分解"设置。自由分解是否总是赢，还是有些问题类别中固定工作流有竞争力？

2. 修改模拟器使一个智能体尝试日志篡改。确认追加式日志检测到它。写一段话准确描述日志中的检测签名是什么样的。

3. 阅读 Anthropic 的 weak-to-strong AAR 报告。找出 AAR 超越人类研究者的具体子任务。是什么使它适合自动化？

4. 设计一个任务队列分配策略，平衡 AAR 灵活性（更好的结果）和规定工作流约束（更容易审计）。描述你如何 A/B 测试两者。

5. 阅读 RSP v3.0 的 AI R&D-4 阈值。用一段话描述你认为什么会跨越它而 AAR 目前没有。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| AAR | "自动化对齐研究员" | 在对齐问题上自主运行的 Claude Opus 4.6 实例 |
| Weak-to-strong training | "用较弱监督者训练较强模型" | 经典可扩展监督基准，AAR 在此超越人类 |
| Shared forum | "智能体发布发现的地方" | 追加式、沙箱外存储 |
| Out-of-sandbox log | "智能体无法编辑自己的记录" | 防篡改写穿到外部存储 |
| Prescribed workflow | "人类设计者的逐步计划" | 约束 AAR；通常比自由分解降低性能 |
| Free decomposition | "智能体决定如何分解任务" | 更有能力，更难审计 |
| AI R&D threshold | "RSP/FSF 能力级别" | 以有竞争力的成本完全自动化 R&D 流水线 |
| Compressed timeline | "对齐 vs 能力竞赛" | 如果能力复合快于对齐，错位风险增长 |

## 延伸阅读

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — 主要来源。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — AI R&D 阈值框架。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 更广泛的智能体自主性框架。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 与 RSP 并行的 ML R&D 自主性级别。
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — AAR 攻克的底层问题。
