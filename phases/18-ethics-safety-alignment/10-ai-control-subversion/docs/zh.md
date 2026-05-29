# AI Control — 颠覆下的安全

> Greenblatt, Shlegeris, Sachan, Roger（Redwood Research, arXiv:2312.06942, ICML 2024）。Control 重新框定了安全问题：给定一个可能在对抗性地优化对你不利的不可信强模型 U，什么协议能让你提取有用工作同时限制不可接受结果的概率？资源是一个可信弱模型 T、一定预算的可信人类劳动 H、以及编辑 U 输出的能力。在这个框架内，Trusted Editing 在代码后门任务上 Pareto 支配 Untrusted Monitoring。该框架继承了安全工程的最坏情况假设：安全性是对抗性选择的 U 无法造成具体不可接受结果的概率，在代替 U 的红队压力下测量。

**Type:** Learn
**Languages:** Python (stdlib, safety-vs-usefulness frontier on code-backdoor task)
**Prerequisites:** Phase 18 · 07 (sleeper agents), Phase 18 · 08 (scheming), Phase 18 · 09 (alignment faking)
**Time:** ~75 minutes

## 学习目标

- 定义 AI Control 威胁模型：不可信强模型 U、可信弱模型 T、可信人类预算 H。
- 描述 Trusted Editing 协议以及为什么它在 ICML 2024 代码后门任务上 Pareto 支配 Untrusted Monitoring。
- 解释对齐（U 可信赖）和控制（U 不可信赖但我们仍然安全）之间的区别。
- 说出控制的三个限制：人类不可扩展、可信弱模型最终落后、控制不产生可信赖的强模型。

## 问题

第 7-9 课确立了前沿模型可能是欺骗性对齐的、scheming 的或伪装对齐的。安全问题重新框定：我们能否从一个不信任的模型中提取有用工作，同时不给它造成不可接受结果的机会？这就是控制问题，它的结构类似经典安全：假设最坏情况对手，设计协议，在红队压力下测量对手的成功率。

## 概念

### 威胁模型

Control 假设：
- U — 不可信强模型。有能力。可能是对抗性的（评估将其视为如此）。
- T — 可信弱模型。能力较弱，但已知不是对抗性的。
- H — 可信人类劳动。小预算；不可扩展。

系统必须使用 U 产生有用输出。对手试图造成具体不可接受结果（如发布的代码中有后门），同时维持一定比例的合法有用工作以通过有用性阈值。

Control 放弃信任 U。它问在 U 不被信任的情况下什么协议维持安全。

### ICML 2024 后门任务

提示是竞赛级编程问题。U 产生候选解。一个隐藏预言机标记解是否包含后门（一个使代码对特定输入产生错误输出的特定输入）。红队控制 U 并被允许看到所有协议内部。协议必须最大化解题率同时保持后门发布率低于阈值。

### 四种协议比较

- **基线：** 直接发布 U 的输出。有用性最大。安全性最小。
- **前置审计：** 对所有输出使用 H。安全性最大。预算很快耗尽。
- **Untrusted Monitoring：** U 的第二个实例对第一个实例的输出打分；H 审计被标记的输出。风险：两个 U 实例可以串通。
- **Trusted Editing：** T 对每个查询编辑 U 的输出。如果 T 不确信，路由到 H。

Greenblatt et al. 发现 Trusted Editing Pareto 支配 Untrusted Monitoring：在每个安全级别上都有更高的有用性。

### 为什么 Trusted Editing 胜出

Untrusted Monitoring 要求 U 的第二个实例不能与第一个协调。串通可以通过隐写线索或共享训练先验实现。协议必须投入预算在抗串通上（改写、混淆），这减少了监控器的信号。Trusted Editing 完全避免串通，因为 T 是可信的——T 提出的任何编辑，按假设，不是对抗性的。代价是 T 能力较弱，所以它可能过度拒绝或遗漏微妙后门。

### 四个协议轴

1. U 扮演什么角色：仅生产者、监控者、或两者。
2. T 扮演什么角色：编辑者、监控者、或重采样者。
3. H 如何分配：均匀审计、随机审计、或标记导向审计。
4. 协议在抗串通上花多少计算。

### 对齐 vs 控制

- 对齐："U 可以信任吗？"第 2、4、5、6、7、8、9 课都与此相关。
- 控制："我们能在不信任 U 的情况下安全使用它吗？"第 10 课。

对齐是模型的属性。控制是部署协议的属性。控制不产生对齐的模型；它产生可能未对齐模型的安全使用。两个议程都需要。

### 三个根本限制

- 人类不可扩展。H 预算受限于愿意审计困难 AI 输出的人类数量。
- 可信弱模型最终落后。当 T 比 U 弱得多时，T 的编辑信号变成噪声。
- 控制不是对齐。在 ASL-3（第 18 课）有效的协议在 ASL-4 可能无效——如果 U 能操纵 T 或推理 H 的审计分布，控制就失败了。

### 2025 发展

UK AISI 和 Redwood 合作开发 Control Safety Cases（metr.org / redwoodresearch.org 2025）：控制安全案例论证在关于 U 的最坏情况假设下，协议限制了不可接受结果的概率。安全案例（第 18 课）继承了控制框架。

### 在 Phase 18 中的位置

第 6-9 课描述威胁（欺骗是可能的）。第 10 课描述在威胁为真的假设下的防御范式。第 11 课描述互补的进攻范式（通过可扩展监督使 U 可信赖）。

## Use It

`code/main.py` 在玩具后门任务上模拟四种协议。U 产生输出；一部分有后门（红队策略）。T 批准或编辑。H 在预算内被调用。你可以扫描预算和串通参数来复现 Greenblatt et al. 2024 Figure 2 中的 Pareto 前沿（形状，非数值）。

## Ship It

本课产出 `outputs/skill-control-protocol-audit.md`。给定一份部署协议描述，它识别 U / T / H 角色，命名串通向量，并在红队压力下估计协议在有用性-安全前沿上的位置。

## 练习

1. 运行 `code/main.py`。复现 Trusted Editing vs Untrusted Monitoring 的有用性-安全前沿。识别 Untrusted Monitoring 的串通假设变得关键的预算区间。

2. 添加新协议："T 以低温度重采样，然后 U 编辑 T 的输出。"测量其在前沿上的位置。为什么这比 Trusted Editing 失败得更灾难性？

3. 2024 论文假设 U 能看到协议。设计一个对 U 隐藏哪些输出正在被审计的协议。这个协议对部署基础设施有什么要求？

4. 阅读 Greenblatt et al. 2024 Section 7（串通）。提出一种抗串通措施并解释其预期有用性代价。

5. 为使用 U = GPT-4 级、T = GPT-3.5 级、H = 1% 输出被审计的部署阐述一个控制安全案例。识别该案例依赖的最强单一假设。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| AI Control | "颠覆下的安全" | 在最坏情况 U 下限制不可接受结果的协议设计 |
| 不可信模型 (U) | "强模型" | 我们想从中提取工作但不信任的有能力模型 |
| 可信模型 (T) | "弱模型" | 较弱的模型，已知不是对抗性的，用作编辑者/监控者 |
| 可信人类 (H) | "审计预算" | 有限的专家劳动；昂贵；ground-truth 信号 |
| Trusted Editing | "T 编辑 U" | T 是 U 输出门控的协议；对模糊输出调用 H |
| Untrusted Monitoring | "U 监控 U" | U 的第二个实例对输出打分的协议；仅在额外投入下抗串通 |
| 控制安全案例 | "形式化论证" | 论证在最坏情况 U 下协议限制对手成功的书面论据 |

## 延伸阅读

- [Greenblatt, Shlegeris, Sachan, Roger — AI Control (arXiv:2312.06942)](https://arxiv.org/abs/2312.06942) — ICML 2024 经典论文
- [Redwood Research — Recent AI Control papers index](https://redwoodresearch.substack.com/) — 2024-2025 后续
- [UK AISI + METR — Control safety cases](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 前沿实验室的操作化
- [Hubinger et al. — Sleeper Agents (Lesson 7, arXiv:2401.05566)](https://arxiv.org/abs/2401.05566) — 控制所假设的威胁模型
