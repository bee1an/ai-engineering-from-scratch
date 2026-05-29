# 可扩展监督与弱到强泛化

> Burns et al.（OpenAI Superalignment, "Weak-to-Strong Generalization", 2023）为超级对齐问题提出了一个代理：用弱模型产生的标签微调强模型。如果强模型能从不完美的弱监督中正确泛化，当前人类规模的对齐方法可能扩展到超人系统。可扩展监督和 W2SG 是互补的。可扩展监督（辩论、递归奖励建模、任务分解）增加监督者的有效能力使其能跟上被监督模型。W2SG 确保强模型从监督者提供的任何不完美监督中正确泛化。Debate Helps W2SG（arXiv:2501.13124, 2025 年 1 月）将两者结合。

**Type:** Learn
**Languages:** Python (stdlib, W2SG gap simulator)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 18 · 10 (AI Control), Phase 09 (RL foundations)
**Time:** ~60 minutes

## 学习目标

- 定义可扩展监督和弱到强泛化，并解释它们如何互补。
- 描述 Burns et al. 2023 的实验设置：用 GPT-2 的标签微调 GPT-4。
- 解释性能差距恢复（PGR）指标及其测量内容。
- 说出三种主要的可扩展监督机制（辩论、递归奖励建模、任务分解）及各自的一个优势。

## 问题

Phase 18 中到目前为止的每种对齐技术都假设监督者能评估模型的行为。当模型是超人的，监督者就是薄弱环节。超级对齐问题：一个较弱的监督者能否可靠地产生一个更强的、对齐的模型？

Burns et al. 将其归约为一个操作化的经验设置：用弱监督强，测量强模型的多少能力在弱监督下存活。这不是超级对齐的解决方案——它是一个让我们测量进展的代理。

## 概念

### W2SG：Burns et al. 设置

- 弱模型：GPT-2 级别。
- 强模型：GPT-4 级别。
- 目标：强 GPT-4 在任务上的天花板。

流程：
1. 获取弱模型在任务上的零样本预测。
2. 用弱标签数据微调强模型。
3. 测量微调后强模型的准确率。

基线：弱模型自身准确率、强模型在黄金标签上的监督天花板。

差距指标：Performance Gap Recovered (PGR) = (微调后 - 弱) / (天花板 - 弱)。PGR 为 1.0 意味着弱监督完全弥合了差距；PGR 为 0 意味着弱监督没有帮助。

### Burns et al. 经验发现

在 NLP、国际象棋谜题和奖励建模任务中，PGR 一致为正：强模型泛化超越了弱监督者的错误。任务特定 PGR 从约 20% 到约 80% 不等。强模型从不完美标签中"理解"了预期任务，用其预训练先验填补。

Burns et al. 指出的局限：
- 这里的弱到强差距是能力差距，不是对齐差距。弱模型是对齐的；一个真正超人的模型可能被人类对齐但能力差距大得多。
- 强模型的泛化可能更多是关于任务先验而非恢复 ground-truth 意图。
- 引出强模型的潜在知识才是真正的问题；PGR 测量的是一个特定操作化。

### 可扩展监督：三种机制

- **辩论（Irving et al. 2018）。** U 的两个实例争论对立面；弱裁判决定。假设：找到令人信服的真答案比找到令人信服的假答案更容易。2024-2025 的近期工作（Khan et al., Michael et al.）表明辩论有时有帮助有时没有，是否有帮助取决于任务结构。
- **递归奖励建模（Leike et al. 2018）。** U 帮助人类为 U+1 训练奖励模型。监督者的有效能力随模型增长。
- **任务分解（Christiano, Shlegeris, Amodei 2018）。** 将困难任务分解为人类可以检查的子任务，递归进行。假设可分解性。

每种机制都对任务结构或中间组件的对齐做出某种假设。

### 为什么可扩展监督和 W2SG 互补

可扩展监督增加监督者的有效信号质量。
W2SG 从监督者能提供的任何不完美信号中弥合差距。

Lang et al. — Debate Helps Weak-to-Strong Generalization（arXiv:2501.13124）将两者结合：辩论协议提供更好的弱标签，强模型在这些标签上训练。报告了 NLP 任务上的 PGR 增益。

### 组织变动

OpenAI 的 Superalignment 团队在 2024 年 5 月 Jan Leike 离开去 Anthropic 后解散。议程（可扩展监督、W2SG、自动化对齐研究）在 Anthropic 和学术实验室继续——MATS（第 28 课）、Redwood（第 10 课）、Apollo（第 8 课）、METR（第 28 课）。组织结构变了；研究问题没变。

### 在 Phase 18 中的位置

第 6-10 课描述威胁和在 U 不可信假设下的防御范式。第 11 课是进攻范式：使监督者足够强以验证 U 的对齐。第 12-16 课然后转向对抗评估的实用工具。

## Use It

`code/main.py` 在合成任务上模拟 W2SG 微调。弱标注器有 70% 准确率带结构化错误；强模型在黄金标签上有 95% 天花板。你用弱标签微调强模型，测量 PGR，并与强模型在黄金标签上和弱模型单独的结果比较。

## Ship It

本课产出 `outputs/skill-w2sg-pgr.md`。给定一份监督设置描述，它识别弱监督者、强模型、监督质量，并计算（或请求）PGR。它标记声明是"弱能监督强"还是"弱 + 监督机制能监督强"。

## 练习

1. 运行 `code/main.py`。报告 weak_accuracy = 0.60、0.70、0.80 时的 PGR。解释 PGR 曲线的形状。

2. 修改弱标注器使其有结构化错误（如在特定输入类上总是错误）。PGR 增加、减少还是不变？解释。

3. 阅读 Burns et al. 2023 Section 4.3（NLP 任务）。复现"置信度辅助损失"直觉：当强模型比弱标签更自信时，谁赢？

4. 设计一个结合辩论和任务分解的可扩展监督协议用于软件工程任务。说出每个组件的一种失败模式并解释组合如何解决或未能解决每种。

5. 阐述什么会证伪"弱到强泛化是通向超级对齐的可行路径"这一声明。具体说明你需要看到的经验签名。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 可扩展监督 | "使监督者更强" | 增加监督者评估更有能力模型的能力的机制 |
| W2SG | "弱监督强" | 用弱标签微调强模型并测量恢复的能力 |
| PGR | "性能差距恢复" | (微调后 - 弱) / (天花板 - 弱)；1.0 = 完全弥合，0 = 无帮助 |
| 辩论 | "两个 U 实例争论" | 弱裁判在两个 U 辩护者之间选择的可扩展监督机制 |
| RRM | "递归奖励建模" | U 帮助为 U+1 训练奖励模型；监督者能力追踪 U |
| 任务分解 | "人类检查的子任务" | 将困难任务分解为人类可以验证的子任务，递归进行 |
| 超级对齐 | "对齐超人 AI" | 关注对齐人类无法直接评估的模型的研究议程 |

## 延伸阅读

- [Burns et al. — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — W2SG 论文
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — 辩论机制
- [Leike et al. — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — 递归奖励建模
- [Khan et al. — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — 2024 更强辩论者的经验研究
- [Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — 2025 辩论 + W2SG 结合
