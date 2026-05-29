# LLM 功能的 A/B 测试 — GrowthBook、Statsig 与"凭感觉"问题

> 传统 A/B 测试不是为非确定性 LLM 设计的。关键区分：eval 回答"模型能不能做这个任务？"A/B 测试回答"用户在不在乎？"两者都需要；凭感觉上线的时代结束了。2026 年要测什么：prompt 工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs 开源；准确率 vs 成本 vs 延迟）、生成参数（temperature、top-p）。真实案例：一个聊天机器人 reward-model 变体带来 +70% 对话长度和 +30% 留存；Nextdoor AI 邮件标题实验在 reward-function 优化后带来 +1% CTR；Khan Academy Khanmigo 在延迟-vs-数学准确率轴上迭代。平台分化：**Statsig**（2025 年 9 月被 OpenAI 以 $1.1B 收购）— 序贯检验、CUPED、一体化。**GrowthBook** — 开源、数据仓库原生、贝叶斯 + 频率派 + 序贯引擎、CUPED、SRM 检查、Benjamini-Hochberg + Bonferroni 校正。你根据数据仓库-SQL 偏好以及"被 OpenAI 收购"对你组织是否重要来选择。

**Type:** Learn
**Languages:** Python (stdlib, toy sequential test simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 20 (Progressive Deployment)
**Time:** ~60 minutes

## 学习目标

- 区分 eval（"模型能不能做这个任务"）和 A/B 测试（"用户在不在乎"）。
- 列举三个可测试轴（prompt、模型、参数）并为每个选择指标。
- 解释 CUPED、序贯检验和 Benjamini-Hochberg 多重比较校正。
- 根据数据仓库-SQL 姿态和企业收购立场选择 Statsig 或 GrowthBook。

## 问题

你手动调优了一个 system prompt。感觉更好了。你上线了。转化率在噪声中变化。你怪指标。或者你上了一个新模型，转化率没动——是模型退化了还是变化太小检测不到？你不知道，因为你没做 A/B。

Eval 回答模型能否在标注集上完成任务。它们不回答用户是否偏好输出。只有受控在线实验能回答这个问题，而且前提是实验有足够的统计功效、控制了非确定性、并校正了多重比较。

## 核心概念

### Eval vs A/B 测试

**Eval** — 离线，标注集，评判（rubric 或 LLM-as-judge 或人工）。回答："在这个固定分布上，输出是否正确/有帮助/安全？"

**A/B 测试** — 在线，真实用户，随机化。回答："新变体是否移动了重要的用户级指标？"

两者都需要。Eval 在暴露前捕捉退化；A/B 在暴露后确认产品影响。

### 测什么

1. **Prompt 工程** — 措辞、system-prompt 结构、示例。指标：任务成功率、用户留存、每请求成本。
2. **模型选择** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：准确率（任务）+ 每请求成本 + 延迟 P99。多目标。
3. **生成参数** — temperature、top-p、max_tokens。指标：任务特定（输出多样性 vs 确定性）。

### CUPED — 方差缩减

Controlled-experiments Using Pre-Experiment Data。在比较实验后期之前回归掉实验前期方差。典型方差缩减：30-70%。有效样本量免费增加。

实现：Statsig 和 GrowthBook 都实现了。

### 序贯检验

经典 A/B 假设固定样本量。序贯检验（"看一眼就决定"）在重复查看下控制假阳性率。始终有效的序贯程序（mSPRT、Howard 的置信序列）让你在明确赢家时提前停止。

### 多重比较校正

在 95% 置信度下运行 20 个 A/B 测试，偶然会产生一个假阳性。Bonferroni 校正收紧每个测试的 α；Benjamini-Hochberg 控制错误发现率。GrowthBook 两者都实现了。

### SRM — 样本比例不匹配

分配哈希将用户随机分到变体。如果 50/50 分配实际交付 47/53，说明有问题——SRM 检查会标记它。两个平台都实现了。

### Statsig vs GrowthBook

**Statsig**：
- 2025 年 9 月被 OpenAI 以 $1.1B 收购。托管，SaaS。
- 序贯检验、CUPED、保留人群。
- 一体化：feature flags + 实验 + 可观测性。
- 最适合：团队想要捆绑产品，不在意 OpenAI 所有权。

**GrowthBook**：
- 开源（MIT）；数据仓库原生（直接从 Snowflake/BigQuery/Redshift 读取）。
- 多引擎：贝叶斯、频率派、序贯。
- CUPED、SRM、Bonferroni、BH 校正。
- 自托管或托管云。
- 最适合：数据仓库-SQL 团队，数据团队控制指标层，想要开源。

### 非确定性使统计功效复杂化

相同 prompt 产生不同输出。传统功效计算假设 IID 观测。LLM 非确定性下，有效样本量低于名义值。将所需样本量乘以约 1.3-1.5 倍作为安全边际。

### 真实案例结果

- 聊天机器人 reward model 变体：+70% 对话长度，+30% 留存。
- Nextdoor 邮件标题：reward-function 优化后 +1% CTR。
- Khan Academy Khanmigo：迭代延迟-vs-数学准确率权衡。

### 反模式：凭感觉上线

每个资深工程师都能说出一个因为"感觉更好"而没做 A/B 就上线的功能。大多数都退化了产品指标，团队几个月都没注意到。A/B 是强制机制。

### 需要记住的数字

- Statsig 被 OpenAI 收购：$1.1B，2025 年 9 月。
- GrowthBook：开源 MIT；贝叶斯 + 频率派 + 序贯。
- CUPED 方差缩减：30-70%。
- LLM 非确定性 → +30-50% 样本量缓冲。

## Use It

`code/main.py` 模拟带固定和序贯边界的序贯 A/B 测试。展示序贯如何让你提前停止。

## Ship It

本课产出 `outputs/skill-ab-plan.md`。给定功能变更、工作负载、基线，选择平台、门控、样本量。

## 练习

1. 运行 `code/main.py`。对于预期 5% 提升、基线 3% 转化率，达到 80% 功效需要多少样本量？
2. 为一个医疗合规的本地部署客户选择 Statsig 或 GrowthBook。
3. 设计一个测试 GPT-4 vs GPT-3.5 在每解决工单成本上的 A/B。主要指标、护栏指标、次要指标分别是什么？
4. 你的灰度通过了但 A/B 显示 -1.2% 转化率。该上线吗？写出升级标准。
5. 对一个前期方差为后期 60% 的场景应用 CUPED。计算有效样本量提升。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| Eval | "离线测试" | 标注集上的模型能力评估 |
| A/B test | "实验" | 对用户的线上随机对比 |
| CUPED | "方差缩减" | 用前期回归减少方差 |
| Sequential test | "可以偷看的测试" | 允许提前停止的始终有效程序 |
| Multiple comparison | "族错误" | 运行多个测试膨胀假阳性 |
| Bonferroni | "严格校正" | 将 α 除以测试数量 |
| Benjamini-Hochberg | "BH FDR" | 错误发现率控制，不那么保守 |
| SRM | "坏分割" | 样本比例不匹配；分配 bug |
| Statsig | "OpenAI 旗下" | 商业一体化，2025 年被收购 |
| GrowthBook | "开源那个" | MIT 数据仓库原生平台 |
| mSPRT | "序贯概率比检验" | 经典序贯程序 |

## 延伸阅读

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
