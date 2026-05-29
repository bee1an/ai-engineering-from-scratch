# LLM 可观测性技术栈选型

> 2026 年的可观测性市场分为两大类。开发平台（LangSmith、Langfuse、Comet Opik）将监控与 evals、prompt 管理、session replay 打包在一起。网关/instrumentation 工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注于遥测。Langfuse 核心为 MIT 许可，OSS 平衡性强（云端免费层 50K events/月）。Phoenix 原生支持 OpenTelemetry，采用 Elastic License 2.0 — 在 drift/RAG 可视化方面表现出色，但不是持久化的生产后端。Arize AX 使用 zero-copy Iceberg/Parquet 集成，声称比单体可观测性便宜 100 倍。LangSmith 在 LangChain/LangGraph 场景领先，$39/用户/月，仅 Enterprise 版支持自托管。Helicone 基于代理模式，15-30 分钟即可部署，100K req/月免费，但在 agent trace 深度上不足。常见生产模式：Gateway（Helicone/Portkey）+ eval 平台（Phoenix/TruLens）通过 OpenTelemetry 粘合。

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## 学习目标

- 区分开发平台（打包式：evals + prompts + sessions）和网关/遥测工具（仅 traces + metrics）。
- 将六大工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到其许可证、定价和最佳适用场景。
- 解释 OpenTelemetry 粘合模式如何让你将网关工具与独立的 eval 平台组合使用。
- 说出 2026 年的成本差异化因素（Arize AX 的 zero-copy 方案 vs 单体 ingest）及大约 100 倍的成本差距。

## 问题

你上线了一个 LLM 功能。它能跑。但你对 prompt 失败、tool 循环、延迟回归、成本飙升或 prompt-cache 命中率完全没有可见性。你搜索 "LLM observability"，得到八个工具，都声称在三个不同价位解决同一个问题。

它们解决的不是同一个问题。LangSmith 回答的是 "为什么这个 LangGraph run 失败了？" Phoenix 回答的是 "我的 RAG pipeline 是否在漂移？" Helicone 回答的是 "哪个应用在烧 token？" Langfuse 回答的是 "我能不能自托管整套系统？" 不同的工具，不同的受众。

选型涉及四个维度：技术栈（LangChain？原生 SDK？多供应商？）、许可证容忍度（仅 MIT？Elastic 可接受？商业许可也行？）、预算（免费层？$100/月？$1000/月？）、以及自托管（必须？有则更好？绝不？）。

## 概念

### 两大类别

**开发平台** 将可观测性与 evals、prompt 管理、dataset 版本控制、session replay 打包。你可以运行实验、查看哪个 prompt 有效、对新 prompt 做 dataset-regression 对比历史最优。LangSmith、Langfuse、Comet Opik。

**网关/遥测工具** 对推理调用做 instrumentation — prompt、response、tokens、延迟、模型、成本。Helicone、SigNoz、OpenLLMetry、Phoenix。极简主义。可通过 OpenTelemetry 与独立的 eval 工具组合。

### Langfuse — OSS 平衡

- 核心 Apache / MIT 许可；通过 Docker 自托管。
- 云端免费层：50K events/月。付费：$29/月（团队版）。
- Evals、prompt 管理、traces、datasets。四项开发平台功能覆盖合理。
- 最佳场景：你想要 LangSmith 级别的功能但必须自托管或保持 OSS 许可。

### Phoenix (Arize) — 遥测优先，OpenTelemetry 原生

- Elastic License 2.0；自托管极简。
- RAG 和 drift 可视化出色。Embedding 空间散点图作为一等功能提供。
- 不是为持久化生产后端设计的 — 主要用于开发时可观测性。
- 最佳场景：RAG pipeline 开发、drift 调试，搭配独立网关用于生产。

### Arize AX — 规模化方案

- 商业许可。通过 Iceberg/Parquet 实现 zero-copy 数据湖集成。
- 声称在规模化场景下比单体可观测性（Datadog 级别）便宜约 100 倍。原理：你将 traces 存储在自己 S3 上的 Parquet 中；Arize 直接读取。
- 最佳场景：>10M traces/天，已有数据湖，想要 LLM 专用 dashboard 但不想付 Datadog 的价格。

### LangSmith — LangChain/LangGraph 优先

- 商业许可，$39/用户/月。仅 Enterprise 版支持自托管。
- 对 LangChain 和 LangGraph 技术栈是同类最佳。如果你不用这两者，吸引力会下降。
- 最佳场景：团队已投入 LangChain，愿意付费。

### Helicone — 基于代理的最小可行方案

- 15-30 分钟部署，只需将 `OPENAI_API_BASE` 切换到 Helicone 代理。
- MIT 许可；100K req/月免费，付费 $20/月起。
- 包含 failover、caching、rate limits — 同时充当网关。
- 在 agent / 多步 trace 深度上不足。
- 最佳场景：快速启动、单技术栈应用、需要网关 + 可观测性合一。

### Opik (Comet) — OSS 开发平台

- Apache 2.0，完全开源。
- 功能集与 Langfuse 类似，继承 Comet 血统。
- 最佳场景：ML 团队已在使用 Comet，想在同一面板中获得 LLM 可观测性。

### SigNoz — OpenTelemetry 优先的全栈 APM

- Apache 2.0。通过 OpenTelemetry 同时处理通用 APM 和 LLM。
- 最佳场景：跨服务和 LLM 调用的统一可观测性。

### 粘合层：OpenTelemetry + GenAI semantic conventions

OpenTelemetry 在 2025 年底发布了 GenAI semantic conventions（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。消费 OTel 的工具可以互操作。正在形成的生产模式：

1. 在每次 LLM 调用中发射带 GenAI conventions 的 OTel。
2. 路由到网关（Helicone / Portkey）用于日常监控。
3. 双发到 eval 平台（Phoenix / Langfuse）用于回归检测。
4. 归档到数据湖（Iceberg）用于通过 Arize AX 或 DuckDB 做长期分析。

### 陷阱：在错误的层做 instrumentation

在 agent 框架内部做 instrumentation（例如添加 LangSmith traces）会将你耦合到该框架。在 HTTP/OpenAI-SDK 层做 instrumentation（通过 OpenLLMetry 或你的网关）是可移植的。

### 采样 — 你不可能保留所有数据

在 >1M requests/天时，全量 trace 保留的成本超过 LLM 调用本身。按规则采样：100% 错误、100% 高成本、5% 成功。始终保留聚合数据；对长尾保留原始数据。

### 你应该记住的数字

- Langfuse 免费云：50K events/月。
- LangSmith：$39/用户/月。
- Helicone 免费：100K req/月。
- Arize AX 声称：在规模化场景下比单体方案便宜约 100 倍。
- OpenTelemetry GenAI conventions：2025 年发布，2026 年广泛采用。

## 动手试试

`code/main.py` 模拟一天 1M trace 在不同保留策略（100% ingest、采样、采样 + 错误）下的表现。报告每种策略的存储成本和丢失的内容。

## 交付产出

本课程产出 `outputs/skill-observability-stack.md`。给定技术栈、规模、预算、许可证立场，选择工具。

## 练习

1. 你的团队使用 LangChain，想要 OSS 自托管可观测性。选择 Langfuse 或 Opik 并说明理由。
2. 在 5M traces/天、Datadog 报价 $150K/月的情况下，计算 Arize AX 的盈亏平衡点。
3. 设计一组 OpenTelemetry GenAI attribute，作为你组织的规范要求每次 LLM 调用必须携带。
4. 论证 Phoenix 单独是否足以支撑生产。什么时候它不够用？
5. Helicone 有 20ms 代理开销。在 P99 TTFT 300 ms 时，这可以接受吗？如果 SLA 是 100 ms 呢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| OpenLLMetry | "OTel for LLMs" | 面向 LLM 的开源 OpenTelemetry instrumentation |
| GenAI conventions | "OTel attributes" | LLM 调用的标准 OTel attribute 名称 |
| LangSmith | "LangChain observability" | 与 LangChain 生态绑定的商业平台 |
| Langfuse | "OSS LangSmith" | 功能集类似的 MIT 开源方案 |
| Phoenix | "Arize dev tool" | OpenTelemetry 原生的开发/eval 平台 |
| Arize AX | "scale observability" | 商业 zero-copy Iceberg/Parquet 可观测性 |
| Helicone | "proxy observability" | 收集 LLM 遥测的 HTTP 代理 + 网关功能 |
| Opik | "Comet LLM" | 来自 Comet 的 Apache 2.0 OSS 开发平台 |
| Session replay | "trace rerun" | 重放包含 tool calls 的完整 agent session |
| Eval | "offline test" | 在标注数据集上运行候选模型/prompt |

## 延伸阅读

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
