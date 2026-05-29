# AI Scientist v2 — Workshop 级别的自主研究

> Sakana 的 AI Scientist v2（Yamada et al., arXiv:2504.08066）运行完整的研究循环：假设、代码、实验、图表、论文撰写、投稿。它是第一个生成论文通过 ICLR 2025 workshop 同行评审的系统。独立评估（Beel et al.）发现 42% 的实验因编码错误失败，文献综述频繁将已有概念错误标记为新颖。Sakana 自己的文档警告该代码库执行 LLM 编写的代码并建议 Docker 隔离。这幅图景的两面都是重点。

**Type:** Learn
**Languages:** Python (stdlib, research-loop state-machine toy)
**Prerequisites:** Phase 15 · 03 (AlphaEvolve), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## 问题

研究是一个开放式任务。与 AlphaEvolve 的算法搜索或 DGM 的基准约束自我修改不同，研究结果没有机器可验证的正确性标准。论文由审稿人评判，而非单元测试。这使得循环更难闭合——但如果闭合则更有价值，因为研究是复合进步所在之处。

AI Scientist v1（Sakana, 2024）通过从人类编写的模板开始来闭合循环。LLM 在固定脚手架内填充实验。AI Scientist v2（Yamada et al., 2025）通过使用带有视觉语言模型评审循环的智能体树搜索移除了模板要求。系统生成想法、实现实验、产生图表、撰写论文，并根据审稿反馈迭代。

同行评审结论：一篇 v2 生成的论文被 ICLR 2025 workshop 接收（有披露）。独立评估结论：系统远非可靠。两者都是事实。

## 概念

### 架构

1. **想法生成。** LLM 基于主题和先前文献提出研究想法。v1 使用模板；v2 使用在假设空间上的智能体搜索。
2. **新颖性检查。** 文献检索步骤检查想法是否已发表。这是 Beel et al. 评估发现错误标记的步骤——已有方法频繁被分类为新颖。
3. **实验计划。** 智能体起草实验方案并编写代码。
4. **执行。** 代码在沙箱中运行。失败被反馈到重试循环。在 Beel et al. 的测量中，42% 的实验在此阶段因编码错误失败。
5. **图表生成。** 视觉语言模型读取生成的图表并重写以提高清晰度。这是 v2 的关键技术新增。
6. **撰写。** LLM 起草论文，与内部审稿人迭代。
7. **可选：投稿。** 论文提交到会议。

### Workshop 接收结果意味着什么

一篇 v2 生成的论文通过了 ICLR 2025 workshop 的同行评审。作者向程序委员会披露了论文的来源。接收是一个数据点；它不是声称系统"做研究"的许可证。

重要背景：workshop 论文的门槛低于主会议论文。同行评审有噪声；任何一天只有一小部分投稿被接收。一次成功是概念验证，不是可靠性声明。Nature 2026 论文记录了端到端循环，本身由人类研究者共同撰写；它不是"系统写了一篇 Nature 论文"。

### 独立评估发现了什么

Beel et al.（arXiv:2502.14297）进行了外部评估。主要发现：

- **实验失败。** 42% 的实验因编码错误失败（错误的 import、shape 不匹配、未定义变量）。重试循环捕捉了一些，但不是全部。
- **新颖性错误标记。** 文献检索步骤频繁将已有概念标记为新颖。这是研究中的幻觉等价物。
- **展示质量差距。** 视觉语言模型的图表评审产生了出版级别的视觉效果，掩盖了底层实验弱点。

最后一个发现对本阶段最重要。一个产生令人信服的输出但没有做令人信服的研究的系统，比一个明显失败的系统更危险而非更安全。评估必须触及底层声明，而非停留在图表。

### 沙箱逃逸问题

Sakana 自己的仓库 README 警告：

> Due to the nature of this software, which executes LLM-generated code, we cannot guarantee safety. There are risks of dangerous packages, uncontrolled web access, and spawning of unintended processes. Use at your own risk and consider Docker isolation.

这是未验证领域中自主性的操作形态。LLM 写代码；代码运行；代码可以做进程被允许做的任何事。没有硬限制文件系统、网络和进程操作的沙箱，任何自主研究智能体都可以泄露数据、烧算力或重写自己。

AlphaEvolve 的沙箱故事更简单，因为它的评估器很紧。AI Scientist v2 的循环运行开放式代码和开放式目标。这就是为什么它需要更强的隔离（Docker 最低要求；seccomp / gVisor 更佳）以及在每次投稿离开系统之前的人工审查。

### v2 在前沿技术栈中的位置

| 系统 | 目标 | 输出类型 | 评估器 | 已知失败 |
|---|---|---|---|---|
| AlphaEvolve | 算法 | 代码 | 单元测试 + 基准 | 受限于评估器严格性 |
| DGM | 智能体脚手架 | 代码 | SWE-bench | 奖励黑客 |
| AI Scientist v2 | 研究论文 | 文本 + 代码 + 图表 | 同行评审（弱） | 实验失败、错误标记、润色掩盖弱点 |

v2 有三者中最弱的自动评估器、最宽的输出面和最短的公开产出路径。操作控制（沙箱、审查、披露）承担了大部分安全工作。

## Use It

`code/main.py` 将 v2 循环模拟为状态机：想法 → 新颖性检查 → 实验 → 图表 → 撰写 → 审稿 → 接收或迭代。每个状态有一个从 Beel et al. 发现中提取的可配置失败概率。运行模拟器 N 次循环并计数：

- 多少想法到达投稿。
- 多少投稿会有一个被润色论文隐藏的关键实验缺陷。
- 重试预算如何在质量和产出之间权衡。

## Ship It

`outputs/skill-ai-scientist-sandbox-review.md` 是研究循环智能体产出在离开沙箱之前的双门审查清单。

## 练习

1. 使用默认参数运行 `code/main.py`。多少比例的循环运行产生"干净"论文？多少比例产生有实验失败缺陷但被图表评审润色过的论文？

2. 默认已使用 Beel et al. 的 42% / 25%。用 `--experiment-failure 0.20 --novelty-mislabel 0.10` 重新运行，然后用 `--experiment-failure 0.60 --novelty-mislabel 0.40`。两次运行之间润色但有缺陷的比例如何变化？

3. 阅读 Sakana 的 AI Scientist v2 仓库 README 关于沙箱要求的部分。列出两个你会为多天自主运行额外应用的限制（Docker 之外）。

4. 阅读 Beel et al. 第 4 节关于展示质量差距的内容。设计一个额外的评估器来捕捉看起来精美但实验有缺陷的论文。

5. 提出一个比"一个博士读每篇论文"更可扩展的研究智能体输出人工审查协议。找出瓶颈并围绕它设计。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| AI Scientist v1 | "Sakana 的模板化研究智能体" | 在固定脚手架中填充实验 |
| AI Scientist v2 | "无模板研究智能体" | 带 VLM 图表评审的智能体树搜索 |
| Agentic tree search | "分支研究智能体" | 并行展开多个实验计划；由内部评审修剪 |
| Vision-language critique | "VLM 图表润色" | 多模态模型读取图表并重写以提高清晰度 |
| Literature retrieval | "新颖性检查" | 搜索先前工作以确认想法新颖性——已记录会错误标记 |
| Polish masking | "漂亮论文，破碎研究" | 展示质量超过实验质量；隐藏弱点 |
| Sandbox escape | "LLM 代码逃逸" | 智能体执行的代码做了循环设计者未预期的事 |

## 延伸阅读

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) — 论文。
- [Sakana blog on the Nature 2026 publication](https://sakana.ai/ai-scientist-nature/) — 厂商摘要及同行评审背景。
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) — 外部评估数据。
- [Sakana AI Scientist v1 paper](https://arxiv.org/abs/2408.06292) — 模板化前身。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 开放式研究智能体的更广泛框架。
