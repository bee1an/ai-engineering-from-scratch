# Batch API — 50% 折扣已成行业标准

> 所有主流供应商都提供异步 batch API，统一 50% 折扣、约 24 小时交付。OpenAI、Anthropic、Google 以及大多数推理平台（Fireworks batch tier、Together batch）都实现了相同模式。将 batch 与 prompt caching 叠加，夜间流水线成本可降至同步无缓存的约 10%。规则极其简单：不是交互式的，就该走 batch。内容生成流水线、文档分类、数据提取、报告生成、批量标注、目录打标——任何能容忍 24 小时延迟的工作负载，不走 batch 就是在白白烧钱。2026 年的生产模式是将每个新 LLM 工作负载分流到三条车道：交互式（同步 + 缓存）、半交互式（异步队列 + 降级）、batch（夜间 + 缓存叠加）。那些假装是交互式但实际能容忍分钟级延迟的工作负载，浪费最多。

**Type:** Learn
**Languages:** Python (stdlib, toy batch-vs-sync cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt & Semantic Caching)
**Time:** ~45 minutes

## 学习目标

- 列举三大供应商的 batch API（OpenAI、Anthropic、Google）及其共同的 50% 折扣 + 24h 交付保证。
- 计算 batch + 缓存叠加在夜间分类工作负载上的成本，并与同步无缓存基线对比。
- 将工作负载分流为交互式 / 半交互式 / batch，并说明理由。
- 指出两个陷阱：部分交互性（用户期望比 24h 更快）和输出格式差异（各供应商 batch 文件格式不同）。

## 问题

你的团队有一条夜间报告生成流水线。50,000 篇文档，逐一摘要，聚类摘要，起草执行简报。同步运行需要 4 小时，每晚 $2,000。你听说了 batch API。

Batch 给你 50% 折扣。你还对 system prompt（所有 50k 调用共享）启用了 prompt caching。叠加后，账单降到每晚 $180——约为基线的 9%。同一条流水线，三处配置改动。

Batch 是 LLM 成本工具箱中最便宜但没人拉的杠杆。原因主要是组织性的：团队默认"实时"，而 SLA 实际上是"明早之前"。这节课就是关于不要把 90% 的账单白白留在桌上。

## 核心概念

### 三大 batch API

**OpenAI Batch API**：JSONL 文件上传，包含请求列表。承诺 24 小时交付（实际通常 2-8 小时）。输入和输出 token 均享 50% 折扣。`/v1/batches` 端点。符合缓存条件的输入还能叠加 cached-input 定价。

**Anthropic Message Batches**：JSONL 上传。24 小时交付。50% 折扣。支持 `cache_control`——cache write 是显式的，read 在 batch 内自动发生。

**Google Vertex AI Batch Prediction**：BigQuery 或 GCS 输入。Gemini 同样享受约 50% 折扣。与 Vertex pipelines 集成。

### 语义：异步，不是慢

Batch 是"我承诺 24 小时内返回"——不是"这会花 24 小时"。典型 P50 是 2-6 小时。供应商在 GPU 利用率低的非高峰时段调度你的 batch。

### 与缓存叠加

一个 50k 文档摘要任务，共享 4K-token system prompt：

- 同步无缓存：50000 × ($input × 4000 + $output × 200)，全价。
- 同步有缓存：system prompt 首次写入后缓存；剩余 49999 次获得 10 倍更便宜的输入。
- Batch + 缓存：以上全部再加 50% 折扣。

叠加效果：batch + cache = 同步无缓存账单的约 10%。任何夜间运行且有共享 system prompt 的工作负载都应该用这个。

### 工作负载分流

**交互式** — 用户等待响应。TTFT 很重要。同步调用 + prompt caching。不能走 batch。

**半交互式** — 用户提交任务，几分钟后回来查看。异步队列，batch 不可用时降级为同步。比如中等量级的 RAG 索引。

**Batch** — 用户期望"明早"或"下一小时"出结果。内容流水线、大规模分类、离线分析。永远走 batch，永远叠加缓存。

常见错误：因为流水线是"生产环境"就把所有东西归为交互式。生产环境不是延迟规格——SLA 才是。

### 部分交互性陷阱

有些功能看起来是交互式的，但能容忍 5-10 分钟。例如：一个夜间客户健康报告带"刷新"按钮。用户点刷新；等 10 分钟完全没问题。团队却按同步方式上线。50 个并发刷新的成本是 batch + 邮件推送方式的 10 倍。

要问的问题是："24 小时对这个用户意味着什么？"如果答案是"他们不会注意到"，那就 batch 它。

### 输出格式陷阱

各供应商的 batch 文件格式不同：

- OpenAI：JSONL，每行一个请求。
- Anthropic：JSONL，每行一条消息；响应格式内嵌。
- Vertex：BigQuery 表或 GCS 前缀 + TFRecord。

写"一个跨供应商的 batch 客户端"意味着每个供应商都需要适配代码。宣传多供应商 batch 的网关（Portkey、LiteLLM 部分层级）仍然只是薄封装原始格式。

### 需要记住的数字

- 各供应商 batch 折扣：输入 + 输出统一 50%。
- 交付 SLA：保证 24 小时，典型 P50 为 2-6 小时。
- Batch + 缓存叠加：约为同步无缓存成本的 10%。
- 工作负载分流规则：如果 24h 延迟可接受，永远走 batch。

## Use It

`code/main.py` 计算 50k 文档工作负载在 sync、sync+cache、batch、batch+cache 四种模式下的成本。报告节省的美元数和百分比。

## Ship It

本课产出 `outputs/skill-batch-triager.md`。给定工作负载特征，分流为 interactive/semi/batch 并估算节省。

## 练习

1. 运行 `code/main.py`。对于 100k 文档流水线（3K-token system prompt、500-token 输出），计算全栈（batch + cache）相对于 sync 基线的节省。
2. 选择你熟悉的一个真实产品中的三个功能。将每个分流为 interactive/semi/batch。
3. 用户抱怨报告花了 3 小时。这是 batch 误分流还是合理的交互式？写出判断标准。
4. 你的 batch API 返回 SLA 是 24h，但 P99 是 20 小时。你如何向用户传达——边缘情况下下游系统的行为是什么？
5. 计算盈亏平衡点：共享前缀长度达到多少时，batch + cache 比在自有预留 GPU 上跑夜间任务更便宜？

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| Batch API | "异步折扣" | 50% 折扣 + 24h 交付 |
| JSONL | "batch 格式" | 每行一个 JSON 请求；OpenAI/Anthropic 标准 |
| Message Batches | "Anthropic batch" | Anthropic 的 batch API 产品名 |
| Batch prediction | "Vertex batch" | Vertex AI 的 batch API 产品 |
| Turnaround SLA | "24h 承诺" | 保证值，非典型值；典型为 2-6h |
| 工作负载分流 | "交互性决策" | Interactive / semi / batch 路由决策 |
| Output schema | "响应格式" | 各供应商 JSONL 布局；不可移植 |
| 叠加折扣 | "batch + cache" | 两者叠加时约为无缓存同步账单的 10% |

## 延伸阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL 格式和 `/v1/batches` 语义。
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — batch 格式和 `cache_control` 交互。
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini batch 语义。
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
