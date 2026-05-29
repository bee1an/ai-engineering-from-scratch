# 影子流量、灰度发布与 LLM 渐进式部署

> LLM 发布结合了软件部署中最难的部分：没有单元测试、故障模式分散、信号延迟。流程是 (1) 影子模式——将生产请求复制到候选模型，记录日志，对比，零用户影响；能捕捉明显的分布问题但不是质量保证；(2) 灰度发布——渐进式流量切换 10% → 25% → 50% → 75% → 100%，每步设门控；跟踪延迟百分位、每请求成本、错误/拒绝率、输出长度分布、用户反馈率；(3) 稳定性确认后对不同方案做 A/B 测试。非确定性不可消除——由于 GPU 浮点非结合性加批大小差异，相同输入跨运行可达 15% 准确率变化。成本是变量而非常量——一个好 20% 的模型可能每次调用贵 3 倍。回滚速度是决定性的：如果回滚需要重新部署，你太慢了。策略在配置/flag 中；模型在 registry 中用固定 digest；回滚 = 翻转策略 + 恢复阈值 + 固定旧模型，秒级完成。

**Type:** Learn
**Languages:** Python (stdlib, toy canary-progression simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 21 (A/B Testing)
**Time:** ~60 minutes

## 学习目标

- 区分影子模式（零影响对比）、灰度（线上流量渐进）和 A/B（稳定性确认后的对比）。
- 列举五个 LLM 特有的灰度指标（延迟、每请求成本、错误/拒绝、输出长度分布、用户反馈）。
- 解释为什么 LLM 非确定性（高达 15%）改变了发布中"稳定"的含义。
- 设计一个秒级（策略翻转）而非小时级（重新部署）的回滚路径。

## 问题

你上线了一个新模型。离线评估显示 3% 准确率提升。你在生产中打开它。24 小时内，成本上升 40%，用户差评上升 8%，三个客户工单报告"奇怪的回答"。你回滚。重新部署花了 3 小时。你的周末毁了。

这里每一步都是可以避免的。影子模式会在任何用户看到之前捕捉到 40% 的成本飙升。灰度会在 10% 时因差评变动而停止。策略 flag 回滚只需 30 秒。这个纪律填补了"离线评估看起来不错"和"真实用户满意"之间的鸿沟。

## 核心概念

### 影子模式

候选模型接收与生产相同的请求；输出被记录但不返回给用户。零用户影响。记录：

- 输出内容（与生产对比 diff）。
- Token 数量（成本差异）。
- 延迟。
- 拒绝和错误。

能捕捉：成本暴涨、长度退化、明显的拒绝变化、硬错误。不能捕捉：用户会感知到的质量差异。影子是冒烟测试，不是质量测试。

### 灰度发布

带门控的渐进式流量切换。典型进度：1% → 10% → 25% → 50% → 75% → 100%。每步对 5 个指标设门控：

1. **延迟百分位** — P50、P95、P99。违规：灰度 P99 > 基线 1.5 倍。
2. **每请求成本** — 混合 $。违规：超过基线 20%。
3. **错误/拒绝率** — 5xx 加显式拒绝。违规：基线 2 倍。
4. **输出长度分布** — 均值 + P99。违规：分布偏移。
5. **用户反馈率** — 差评/工单。违规：基线 1.5 倍。

### 非确定性是新的方差

相同输入产生不同输出。原因：

- GPU 浮点非结合性（浮点归约顺序随 batch 变化）。
- 批大小差异（同一 prompt 在 128 的 batch 中 vs 16 的 batch 中）。
- 采样（temperature > 0）。

测量：相同评估集上跨运行高达 15% 准确率变化。发布中的"稳定"意味着指标在预期方差内，而非与基线完全相同。将门控设在噪声底线之上。

### 成本是变量

一个好 20% 的模型可能每次调用贵 3 倍。每请求成本是五个门控之一。上线一个"更好"但破坏单位经济的模型是回滚场景。

### 回滚是武器

- 策略 flag（feature flag 系统）：在配置中翻转百分比；秒级。
- 模型固定（registry digest）：固定的模型不会自动升级。
- 回滚 = 恢复 flag + 设置固定 digest 为上一版本。秒级，不是小时级。

如果你的栈需要重新部署才能回滚，先修这个再做发布。

### 工具

**Argo Rollouts** / **Flagger** — Kubernetes 渐进式交付控制器。与 Istio/Linkerd 加权路由集成。

**Istio weighted routing** — 服务网格级别的流量分割。

**KServe / Seldon Core** — 内置灰度的模型 serving。

**Feature flags** — LaunchDarkly、Flagsmith、Unleash。策略级翻转，无需重新部署。

### 指标节奏

灰度门控每 5-15 分钟检查一次，取决于流量。1% 流量 10 req/min 给出每窗口 50-150 个数据点——对延迟够用但对用户反馈有噪声。10% 给出约 10 倍更多。每步应暂停足够长时间以积累足够样本。

### A/B 步骤是可选的

如果新模型明显不同（不同行为、不同成本曲线、不同语气），在灰度通过后以 50% 做 A/B 测试。如果只是改进版本，灰度门控通过后直接推到 100%。

### 需要记住的数字

- 灰度进度：1% → 10% → 25% → 50% → 75% → 100%。
- 非确定性上限：相同输入跨运行高达 15% 方差。
- 五个灰度指标：延迟、成本、错误/拒绝、输出长度、用户反馈。
- 成本门控：超过基线 20% 即违规。
- 回滚：秒级，不是小时级。

## Use It

`code/main.py` 模拟带注入退化的灰度发布。报告发布在哪个阶段停止以及哪个门控触发。

## Ship It

本课产出 `outputs/skill-rollout-runbook.md`。给定候选模型、基线和风险容忍度，设计影子→灰度→100% 计划。

## 练习

1. 运行 `code/main.py`。注入 25% 成本退化。灰度在哪个阶段停止？
2. 你的新模型离线准确率提升 3% 但每请求成本 +18%。该上线吗？取决于策略——写出两条路径。
3. 设计一个端到端 60 秒内完成的回滚。列出所需基础设施。
4. 非确定性在你的评估上显示 ±7%。设置灰度门控使其不误报。你用什么倍数？
5. 影子模式在灰度之前捕捉到 40% 成本飙升。写出在影子中触发的告警规则。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| Shadow mode | "复制到新模型" | 零影响发送到候选模型用于记录 |
| Canary | "渐进式流量" | 带门控的逐步用户暴露发布 |
| Gates | "发布检查" | 阻止推进的指标阈值 |
| Non-determinism | "LLM 方差" | 不可消除的跨运行差异 |
| Policy flag | "flag 翻转回滚" | 配置级回滚，秒级而非小时级 |
| Model pin | "registry digest" | 模型版本的不可变引用 |
| Argo Rollouts | "K8s 渐进式" | Kubernetes 原生灰度/回滚控制器 |
| KServe | "推理 K8s" | 带灰度原语的模型 serving |
| Istio weighted | "网格分割" | 服务网格流量分割器 |

## 延伸阅读

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)
