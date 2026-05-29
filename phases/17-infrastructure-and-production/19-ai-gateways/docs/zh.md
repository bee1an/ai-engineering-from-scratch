# AI Gateway — LiteLLM、Portkey、Kong AI Gateway、Bifrost

> Gateway 位于你的应用和模型供应商之间。核心功能是供应商路由、降级、重试、限流、密钥引用、可观测性、护栏。2026 年市场格局：**LiteLLM** 是 MIT 开源，支持 100+ 供应商，OpenAI 兼容，但在约 2000 RPS 时崩溃（8 GB 内存，已发布基准测试中出现级联故障）；最适合 Python、<500 RPS、开发/原型。**Portkey** 定位为控制平面（护栏、PII 脱敏、越狱检测、审计追踪），2026 年 3 月转为 Apache 2.0 开源，20-40 ms 延迟开销，$49/月生产层。**Kong AI Gateway** 基于 Kong Gateway 构建——Kong 自己在相同 12 CPU 上的基准测试：比 Portkey 快 228%，比 LiteLLM 快 859%；$100/模型/月定价（Plus 层最多 5 个）；如果你已经在用 Kong 则适合企业。**Bifrost**（Maxim AI）— 可配置退避的自动重试，OpenAI 429 时降级到 Anthropic。**Cloudflare / Vercel AI Gateway** — 托管、零运维、基础重试。数据驻留驱动自托管决策；Portkey 和 Kong 处于中间位置，有开源 + 可选托管。

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 minutes

## 学习目标

- 列举六大网关核心功能（路由、降级、重试、限流、密钥、可观测性、护栏）。
- 将四个 2026 网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到规模上限和使用场景。
- 引用 Kong 基准测试（比 Portkey 快 228%，比 LiteLLM 快 859%）并解释为什么这对 >500 RPS 很重要。
- 根据数据驻留和运维预算选择自托管 vs 托管。

## 问题

你的产品调用 OpenAI、Anthropic 和自托管的 Llama。每个供应商有不同的 SDK、错误模型、限流和认证方案。你想要故障转移（如果 OpenAI 429，试 Anthropic）、统一凭证存储、统一可观测性和按租户限流。

在应用层重新发明这些会将每个服务耦合到每个供应商。网关层将其整合为一个进程、一个 API（通常 OpenAI 兼容），扇出到各供应商。

## 核心概念

### 六大核心功能

1. **供应商路由** — OpenAI、Anthropic、Gemini、自托管等统一在一个 API 后面。
2. **降级** — 遇到 429、5xx 或质量故障时，重试其他供应商。
3. **重试** — 指数退避，有界尝试次数。
4. **限流** — 按租户、按 key、按模型。
5. **密钥引用** — 运行时从 vault 拉取凭证（永远不在应用中）。
6. **可观测性** — OTel + GenAI 属性（Phase 17 · 13）+ 成本归因。
7. **护栏** — PII 脱敏、越狱检测、允许话题过滤。

### LiteLLM — MIT 开源，Python

- 100+ 供应商，OpenAI 兼容，router 配置，降级，基础可观测性。
- 在 Kong 基准测试中约 2000 RPS 时崩溃；8 GB 内存占用，持续负载下级联故障。
- 最适合：Python 应用，<500 RPS，开发/预发布网关，实验性路由。
- 成本：开源 $0；有云端免费层。

### Portkey — 控制平面定位

- 2026 年 3 月起 Apache 2.0 开源。护栏、PII 脱敏、越狱检测、审计追踪。
- 每请求 20-40 ms 延迟开销。
- $49/月生产层，含留存 + SLA。
- 最适合：需要护栏 + 可观测性捆绑的受监管行业。

### Kong AI Gateway — 规模方案

- 基于 Kong Gateway 构建（成熟的 API 网关产品，lua+OpenResty）。
- Kong 自己在 12-CPU 等效环境的基准测试：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：$100/模型/月，Plus 层最多 5 个。
- 最适合：已在用 Kong；>1000 RPS；愿意付费许可。

### Bifrost (Maxim AI)

- 可配置退避的自动重试。
- OpenAI 429 时降级到 Anthropic 是经典配方。
- 较新进入者；商业。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管，零运维。基础重试和可观测性。
- 最适合：在 Cloudflare/Vercel 上的边缘 JavaScript 应用。
- 在护栏和限流方面不如 Kong/Portkey。

### 自托管 vs 托管

数据驻留是决定性因素。医疗和金融默认自托管（LiteLLM 或 Portkey 开源或 Kong）。消费产品默认托管（Cloudflare AI Gateway）或中间层（Portkey 托管）。混合：受监管租户自托管，其他托管。

### 延迟预算

- LiteLLM：典型 5-15 ms 开销。
- Portkey：20-40 ms 开销。
- Kong：3-8 ms 开销。
- Cloudflare/Vercel：1-3 ms 开销（边缘优势）。

网关延迟直接叠加到 TTFT。对于 TTFT P99 < 100 ms SLA，选 Kong 或 Cloudflare。对于 P99 < 500 ms，任何都行。

### 限流语义很重要

简单令牌桶在中等规模下可用。多租户需要滑动窗口 + 突发允许 + 按租户分层。LiteLLM 提供令牌桶；Kong 提供滑动窗口；Portkey 提供分层。

### 网关 + 可观测性 + 路由组合

Phase 17 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产中是同一层。选一个覆盖全部三者的工具，或仔细串联：大多数 2026 部署组合 Helicone（可观测性）或 Portkey（护栏）与 Kong（规模）来分担角色。

### 需要记住的数字

- LiteLLM：约 2000 RPS 崩溃，8 GB 内存。
- Portkey：20-40 ms 开销；2026 年 3 月起 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：$100/模型/月，Plus 层最多 5 个。
- Cloudflare/Vercel：边缘 1-3 ms 开销。

## Use It

`code/main.py` 模拟在 429/5xx 注入下跨 3 个供应商的网关路由和降级。报告延迟、重试率和降级命中率。

## Ship It

本课产出 `outputs/skill-gateway-picker.md`。给定规模、运维姿态、合规、延迟预算，选择网关。

## 练习

1. 运行 `code/main.py`。配置 OpenAI→Anthropic→自托管的降级链。在 5% 供应商错误率下预期命中率是多少？
2. 你的 SLA 是 TTFT P99 < 200 ms，基线 300 ms。哪些网关在预算内？
3. 一个医疗客户要求自托管 + PII 脱敏 + 审计。选 Portkey 开源还是 Kong。
4. 比较 LiteLLM vs Kong：在什么 RPS 上限时团队应该迁移？
5. 为多租户 SaaS 设计限流策略：免费层、试用层、付费层。令牌桶还是滑动窗口？

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| Gateway | "API 代理" | 位于应用和供应商之间的进程 |
| LiteLLM | "MIT 那个" | Python 开源，100+ 供应商，2K RPS 崩溃 |
| Portkey | "护栏网关" | 控制平面 + 可观测性，Apache 2.0 |
| Kong AI Gateway | "规模那个" | 基于 Kong Gateway，基准测试领先 |
| Bifrost | "Maxim 的网关" | 重试 + Anthropic 降级配方 |
| Cloudflare AI Gateway | "边缘托管" | 边缘部署的托管网关，零运维 |
| PII redaction | "数据清洗" | 发送到模型前用 Regex + NER 遮蔽 |
| Jailbreak detection | "prompt 注入防护" | 对用户输入的分类器 |
| Audit trail | "合规日志" | 每次 LLM 调用的不可变记录 |
| Token-bucket | "简单限流" | 基于补充的限流器 |
| Sliding-window | "精确限流" | 时间窗口限流器；更公平 |

## 延伸阅读

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
