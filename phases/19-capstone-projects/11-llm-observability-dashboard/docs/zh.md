# 毕业项目 11 — LLM 可观测性与评估仪表板

> Langfuse 走向 open-core。Arize Phoenix 发布了 2026 GenAI semconv 映射。Helicone 和 Braintrust 都加倍投入每用户成本归因。Traceloop 的 OpenLLMetry 成为事实上的 SDK 插桩标准。生产形态是 ClickHouse 存 trace、Postgres 存元数据、Next.js 做 UI，加一组评估任务（DeepEval、RAGAS、LLM-judge）在采样 trace 上运行。构建一个自托管的，从至少四个 SDK 家族摄入，并演示在五分钟内捕获注入的回归。

**类型：** 毕业项目
**语言：** TypeScript（UI），Python / TypeScript（摄入 + 评估），SQL（ClickHouse）
**前置要求：** Phase 11（LLM 工程）、Phase 13（工具）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段：** P11 · P13 · P17 · P18
**时间：** 25 小时

## 问题

2026 年每个运行生产流量的 AI 团队都在模型旁边维护一个可观测性平面。成本归因。幻觉检测。漂移监控。越狱信号。SLO 仪表板。PII 泄露告警。开源参考——Langfuse、Phoenix、OpenLLMetry——收敛到 OpenTelemetry GenAI 语义约定作为摄入 schema。你现在可以用一个 SDK 插桩 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM，发送兼容的 span。

你将构建一个自托管仪表板，从至少四个 SDK 家族摄入，在采样 trace 上运行一组评估任务，检测漂移并告警。衡量标准：给定一个故意注入的回归（一个开始产出 PII 的提示），仪表板在五分钟内捕获并触发告警。

## 概念

摄入是 OTLP HTTP。SDK 产出 GenAI-semconv span：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Span 落入 ClickHouse 做列式分析；元数据（用户、会话、应用）落入 Postgres。

评估作为批处理任务在采样 trace 上运行。DeepEval 评分 faithfulness、toxicity 和 answer relevance。RAGAS 在 trace 携带检索上下文时评分检索指标。自定义 LLM-judge 运行领域特定检查（PII 泄露、违规响应）。评估运行写回同一 ClickHouse 作为链接到父 trace 的 eval span。

漂移检测监控嵌入空间分布随时间的变化（prompt 嵌入上的 PSI 或 KL 散度）加评估分数趋势。告警送入 Prometheus Alertmanager 然后 Slack / PagerDuty。UI 是 Next.js 15 带 Recharts。

## 架构

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- 摄入：OpenTelemetry SDK + GenAI 语义约定；OTLP HTTP 传输
- Collector：OpenTelemetry Collector 带 tail-sampling 处理器（用于成本控制）
- 存储：ClickHouse 存 span，Postgres 存元数据，S3 存原始事件归档
- 评估：DeepEval、RAGAS 0.2、Arize Phoenix evaluator pack、自定义 LLM-judge
- 漂移：每周对池化 prompt 嵌入（sentence-transformers）计算 PSI / KL
- 告警：Prometheus Alertmanager -> Slack / PagerDuty
- UI：Next.js 15 App Router + Recharts + server actions
- 开箱支持的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 构建步骤

1. **Collector 配置。** OpenTelemetry Collector 带 OTLP HTTP receiver、tail-sampler 保留 100% 错误 trace 和 10% 成功 trace，以及导出到 ClickHouse 和 S3 的 exporter。

2. **ClickHouse schema。** 表 `spans` 带镜像 GenAI semconv 的列：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，加 JSON bag 用于长 payload。按 user_id 和 app_id 添加二级索引。

3. **SDK 覆盖测试。** 用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）编写小客户端应用，带 OpenLLMetry 自动插桩。验证每个产出标准 GenAI span 并落入 ClickHouse。

4. **评估任务。** 定时任务读取最近 15 分钟采样 trace 并运行 DeepEval faithfulness、toxicity 和 answer relevance。输出是链接到父 trace 的 eval span。

5. **自定义 LLM-judge。** PII 泄露 judge：给定一个响应，调用 guard LLM 评分 PII 泄露可能性。高分响应进入分诊队列。

6. **漂移检测。** 每周任务计算本周池化 prompt 嵌入与过去 4 周基线之间的 PSI。如果 PSI 超过阈值，告警。

7. **仪表板。** Next.js 15 带页面：概览（spans/sec、cost/user、p95 延迟）、traces（搜索 + 瀑布图）、evals（faithfulness 趋势、toxicity）、drift（PSI 随时间）、alerts。

8. **告警链。** Prometheus exporter 读取评估分数聚合和延迟百分位；Alertmanager 将警告路由到 Slack，严重违规路由到 PagerDuty。

9. **回归探测。** 注入一个 bug：被评估的聊天机器人 1% 的时间开始泄露假 SSN。衡量 MTTR：从 bug 部署到 Slack 告警。

## 使用示例

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付标准

`outputs/skill-llm-observability.md` 是交付物。给定一个 LLM 应用，仪表板摄入其 trace，运行评估，对漂移告警，并在 Next.js 中展示 cost/user 分解。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | Trace-schema 覆盖 | 产出标准 GenAI span 的 SDK 家族数量（目标：6+） |
| 20 | 评估正确性 | DeepEval / RAGAS 分数 vs 手工标注集 |
| 20 | 仪表板体验 | 注入回归上的 MTTR（目标低于 5 分钟） |
| 20 | 成本/规模 | 1k spans/sec 持续摄入无积压 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链端到端验证 |
| **100** | | |

## 练习

1. 为 Haystack 框架添加自定义插桩。验证标准 span 带忠实的 `gen_ai.*` 属性落入 ClickHouse。

2. 在相同 trace 上将 DeepEval 换成 Phoenix evaluators。衡量两个评估引擎之间的分数漂移。

3. 细化漂移检测器：按 app-id 而非全局计算 PSI。展示每应用漂移轨迹。

4. 添加"用户影响"页面：cost-per-user 和 failure-rate-per-user 带 sparklines。

5. 构建 tail-sampling 策略：保留 100% toxicity > 0.5 的 trace 加其余的 10% 分层采样。衡量引入的采样偏差。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GenAI semconv | "OTel LLM 属性" | 2025 OpenTelemetry 规范，用于 LLM span 属性（system、model、tokens） |
| Tail sampling | "后 trace 采样" | Collector 在 trace 完成后决定保留或丢弃（可以窥探错误） |
| PSI | "Population stability index" | 比较两个分布的漂移指标；> 0.2 通常表示有意义的漂移 |
| LLM-judge | "评估即模型" | 一个 LLM 按评分标准对另一个 LLM 的输出打分（faithfulness、toxicity、PII） |
| Tail-sampling 策略 | "保留规则" | 决定哪些 trace 持久化 vs 丢弃的规则；错误 + 采样率 |
| Eval span | "链接的评估 trace" | 携带评估分数的子 span，链接到原始 LLM 调用 span |
| Cost per user | "单位经济" | 在一个时间窗口内归因到 user_id 的美元成本；关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 参考 open-core 可观测性平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 备选参考，漂移支持强
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — 自动插桩 SDK 家族
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 摄入 schema
- [Helicone](https://www.helicone.ai) — 备选托管可观测性
- [Braintrust](https://www.braintrust.dev) — 备选评估优先平台
- [ClickHouse documentation](https://clickhouse.com/docs) — 列式 span 存储
- [DeepEval](https://github.com/confident-ai/deepeval) — 评估器库
