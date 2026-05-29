# LLM 路由层 — LiteLLM、OpenRouter、Portkey

> 供应商锁定代价高昂。不同的 tool-calling 工作负载适合不同的模型。路由网关提供统一的 API 接口、重试、故障转移、成本追踪和 guardrails。2026 年三种主流架构：LiteLLM（开源自托管）、OpenRouter（托管 SaaS）、Portkey（生产级，2026 年 3 月开源）。本课讲解决策标准，并实现一个 stdlib 路由网关。

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02 (function calling), Phase 13 · 17 (gateways)
**Time:** ~45 minutes

## 学习目标

- 区分自托管、托管和生产级路由方案。
- 实现一个按优先级顺序在供应商故障时重试的 fallback chain。
- 跨供应商追踪每次请求的成本和 token 用量。
- 根据具体生产约束在 LiteLLM、OpenRouter 和 Portkey 之间做出选择。

## 问题

供应商路由重要的场景：

1. **成本。** Claude Sonnet 的价格是 Haiku 的 3 倍。对于分类任务，Haiku 就够了；对于综合任务，Sonnet 值得。按请求路由。

2. **故障转移。** OpenAI 出了一小时故障，所有请求失败。你希望自动回退到 Anthropic，无需重新部署。

3. **延迟。** 实时聊天 UI 需要快速的 time-to-first-token。批量摘要器不需要。按延迟 SLA 路由。

4. **合规。** 欧盟用户必须留在欧盟区域。按区域路由。

5. **实验。** 在同一工作负载上 A/B 测试两个模型。按测试分桶路由。

为每个集成手动编写所有这些逻辑是重复劳动。路由网关提供一个 OpenAI 兼容的 API，处理其余一切。

## 概念

### OpenAI 兼容的代理形态

所有人都说 OpenAI 格式。路由网关暴露 `/v1/chat/completions`，接受 OpenAI schema，内部代理到 Anthropic / Gemini / Cohere / Ollama / 任何后端。客户端无需关心。

### 模型别名

代码中不写 `claude-3-5-sonnet-20251022`，而是写 `our_smart_model`。网关将别名映射到真实模型。当 Anthropic 发布 Claude 4 时，你在服务端改别名；代码不动。

### Fallback chains

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

网关在配置中定义这些。重试计入预算，防止 fallback 级联导致成本爆炸。

### 语义缓存

相同或近似相同的 prompt 命中缓存而非供应商。在重复的 agent 循环中节省 30% 到 60%。key 基于 embedding；近似 prompt 共享缓存槽。

### Guardrails

网关级别：

- **PII 脱敏。** 发送 prompt 前用正则或 ML 方式过滤。
- **策略违规。** 拒绝包含禁止内容的 prompt。
- **输出过滤。** 清洗 completion 中的泄露。

Portkey 和 Kong 都内置了 guardrails。LiteLLM 将其作为可选项。

### 按 key 限速

一个 API key = 一个团队。按 key 预算防止一个团队消耗共享配额。大多数网关支持此功能。

### 自托管 vs 托管的权衡

| 因素 | LiteLLM（自托管） | OpenRouter（托管） | Portkey（生产级） |
|--------|----------------------|----------------------|----------------------|
| 代码 | 开源，Python | 托管 SaaS | 开源（2026 年 3 月）+ 托管 |
| 部署 | 部署代理 | 注册即用 | 两者皆可 |
| 供应商 | 100+ | 300+ | 100+ |
| 计费 | 使用自己的 key | OpenRouter 积分 | 使用自己的 key |
| 可观测性 | OpenTelemetry | Dashboard | 完整 OTel + PII 脱敏 |
| 最适合 | 需要完全控制的团队 | 快速原型 | 需要合规的生产环境 |

LiteLLM 适合有 SRE 团队且需要数据主权的场景。OpenRouter 适合只想要一个订阅、不想管基础设施的场景。Portkey 适合需要开箱即用的 guardrails 和合规的场景。

### 成本追踪

每个请求携带 `provider`、`model`、`input_tokens`、`output_tokens`。乘以每模型每 token 价格（从网关维护的价格表中获取）。按用户 / 团队 / 项目聚合。

### MCP 加路由

网关可以同时路由 LLM 调用和 MCP sampling 请求。当 sampling 请求的 modelPreferences 偏好特定模型时，网关转换到正确的后端。这就是 Phase 13 · 17（MCP gateway）和本课路由网关有时合并为一个服务的地方。

### 路由策略

- **静态优先级。** 列表中第一个；出错时回退。
- **负载均衡。** Round-robin 或加权。
- **成本感知。** 选择满足延迟/质量要求的最便宜模型。
- **延迟感知。** 选择最近 N 分钟内最快的模型。
- **任务感知。** Prompt 分类器将编码路由到一个模型，摘要路由到另一个。

## Use It

`code/main.py` 用约 150 行实现了一个路由网关：接受 OpenAI 格式的请求，转换为每供应商的 stub，运行优先级 fallback chain，追踪每请求成本，并对输入应用 PII 脱敏。用三个场景运行：正常请求、主供应商故障触发 fallback、PII 泄露被脱敏捕获。

关注点：

- `ROUTES` 字典：别名 -> 优先级排序的具体供应商列表。
- Fallback 循环在 5xx 时重试。
- 成本追踪器将 token 用量乘以每模型费率。
- PII 脱敏器在转发前清洗 SSN 格式的模式。

## Ship It

本课产出 `outputs/skill-routing-config-designer.md`。给定工作负载特征（延迟、成本、合规），该 skill 选择 LiteLLM / OpenRouter / Portkey 并生成路由配置。

## 练习

1. 运行 `code/main.py`。触发故障场景；确认 fallback 落到第二个供应商且成本归属正确。

2. 添加语义缓存：prompt 的 SHA256 作为查找 key；缓存命中时立即返回。测量重复调用的成本节省。

3. 添加 prompt 分类器，将 "code ..." prompt 路由到偏好智能的别名，将 "summarize ..." prompt 路由到偏好速度的别名。

4. 设计按团队预算：每个团队有月度消费上限；达到上限后网关拒绝请求。选择执行粒度（按请求或按窗口）。

5. 并排阅读 LiteLLM、OpenRouter 和 Portkey 文档。列出每个产品独有的一个功能（另外两个没有的）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| Routing gateway | "LLM 代理" | 在多个供应商前面的统一 API 层 |
| OpenAI-compatible | "说 OpenAI 格式" | 接受 `/v1/chat/completions` 格式，转换到任何后端 |
| Model alias | "our_smart_model" | 代码中的名称，网关映射到具体模型 |
| Fallback chain | "重试列表" | 故障时按顺序尝试的供应商列表 |
| Semantic caching | "Prompt embedding 缓存" | key 是 prompt 的 embedding；近似重复共享缓存命中 |
| Guardrails | "输入/输出过滤" | 脱敏 PII，拒绝策略违规 |
| Per-key rate limit | "团队预算" | 限定到某个 API key 的配额 |
| Cost tracking | "每请求花费" | 聚合 token 用量 x 每模型价格 |
| LiteLLM | "开源代理" | 可自托管的 OSS 路由网关 |
| OpenRouter | "托管 SaaS" | 基于积分计费的托管网关 |
| Portkey | "生产级选项" | 开源 + 托管，内置 guardrails |

## 延伸阅读

- [LiteLLM — docs](https://docs.litellm.ai/) — 自托管路由网关
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — 托管路由 SaaS
- [Portkey — docs](https://portkey.ai/docs) — 带 guardrails 的生产级路由
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — 决策指南
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — 供应商调研
