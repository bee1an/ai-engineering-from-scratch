# Agent 可观测性：Langfuse、Phoenix、Opik

> 2026 年三个开源 agent 可观测性平台占据主导。Langfuse（MIT）— 月安装量 6M+，tracing + prompt 管理 + evals + session replay。Arize Phoenix（Elastic 2.0）— 深度 agent 专用 eval、RAG 相关性、OpenInference 自动 instrumentation。Comet Opik（Apache 2.0）— 自动化 prompt 优化、guardrails、LLM-judge 幻觉检测。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 23 (OTel GenAI)
**Time:** ~45 minutes

## 学习目标

- 列举三大开源 agent 可观测性平台及其许可证。
- 区分各平台的强项：Langfuse（prompt 管理 + sessions）、Phoenix（RAG + 自动 instrumentation）、Opik（优化 + guardrails）。
- 解释为什么到 2026 年 89% 的组织报告已部署 agent 可观测性。
- 实现一个 stdlib trace-to-dashboard 流水线，带 LLM-judge 评估。

## 问题

OTel GenAI（Lesson 23）给了你 schema。你仍然需要一个平台来摄入 span、运行评估、存储 prompt 版本并暴露回归。三个竞争者各自强调生命周期的不同部分。

## 概念

### Langfuse (MIT)

- 月 SDK 安装量 6M+，GitHub stars 19k+。
- 功能：tracing、带版本控制 + playground 的 prompt 管理、评估（LLM-as-judge、用户反馈、自定义）、session replay。
- 2025 年 6 月：原商业模块（LLM-as-a-judge、annotation queues、prompt experiments、Playground）在 MIT 下开源。
- 最强项：端到端可观测性 + 紧密的 prompt 管理闭环。

### Arize Phoenix (Elastic License 2.0)

- 更深的 agent 专用评估：trace 聚类、异常检测、RAG 检索相关性。
- 原生 OpenInference 自动 instrumentation。
- 搭配托管 Arize AX 用于生产。
- 无 prompt 版本控制 — 定位为漂移/行为回归工具，与更广泛的平台配合使用。
- 最强项：RAG 相关性、行为漂移、异常检测。

### Comet Opik (Apache 2.0)

- 通过 A/B 实验进行自动化 prompt 优化。
- Guardrails（PII 脱敏、主题约束）。
- LLM-judge 幻觉检测。
- Comet 自测基准：Opik 日志 + eval 23.44s vs Langfuse 327.15s（~14x 差距）— 厂商基准仅作方向性参考。
- 最强项：优化闭环、自动化实验、guardrail 执行。

### 行业数据

据 Maxim（2026 年现场分析）：89% 的组织已部署 agent 可观测性；质量问题是首要生产障碍（32% 的受访者提及）。

### 如何选择

| 需求 | 选择 |
|------|------|
| 一体化 + prompt 管理 | Langfuse |
| 深度 RAG 评估 + 漂移检测 | Phoenix |
| 自动化优化 + guardrails | Opik |
| 开放许可，无 ELv2 | Langfuse (MIT) 或 Opik (Apache 2.0) |
| Datadog / New Relic 集成 | 任意 — 都导出 OTel |

### 这个模式容易出错的地方

- **没有 eval 策略。** 没有评估的 tracing 只是昂贵的日志。
- **自建 LLM-judge 但没有 grounding。** CRITIC 模式（Lesson 05）适用 — judge 需要外部工具做事实验证。
- **Prompt 版本未关联 trace。** 生产回归时无法二分定位到导致问题的 prompt。

## Build It

`code/main.py` 实现了一个 stdlib trace 收集器 + LLM-judge 评估器：

- 摄入 GenAI 形态的 span。
- 按 session 分组，标记失败运行（guardrail 触发、低置信度 eval）。
- 一个脚本化 LLM-judge，按评分标准对 agent 响应打分。
- 类 dashboard 摘要：失败率、top 失败原因、eval 分数分布。

运行：

```
python3 code/main.py
```

输出：per-session eval 分数和失败分类，匹配 Langfuse/Phoenix/Opik 会展示的内容。

## Use It

- **Langfuse** 自托管或云端；通过 OTel 或其 SDK 接入。
- **Arize Phoenix** 自托管；自动 instrument OpenInference。
- **Comet Opik** 自托管或云端；自动化优化闭环。
- **Datadog LLM Observability** 适合已使用 Datadog 的混合 ops+ML 团队。

## Ship It

`outputs/skill-obs-platform-wiring.md` 选择一个平台并将 trace + eval + prompt 版本接入现有 agent。

## 练习

1. 将一周的 OTel trace 导出到 Langfuse cloud（免费层）。哪些 session 失败了？为什么？
2. 为你的领域编写 LLM-judge 评分标准（事实正确性、语气、范围遵守）。在 50 条 trace 上测试。
3. 比较 Langfuse prompt 版本控制与 Phoenix 的 trace 聚类。哪个能更快告诉你什么坏了？
4. 阅读 Opik 的 guardrail 文档。将 PII 脱敏 guardrail 接入你的一个 agent 运行。
5. 在你的语料上对三者做基准测试。忽略厂商发布的数字；测量你自己的。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Tracing | "Spans collector" | 摄入 OTel / SDK span；按 session 索引 |
| Prompt management | "Prompt CMS" | 关联 trace 的版本化 prompt |
| LLM-as-judge | "Automated eval" | 独立 LLM 按评分标准对 agent 输出打分 |
| Session replay | "Trace playback" | 逐步回放历史运行用于调试 |
| RAG relevancy | "Retrieval quality" | 检索到的上下文是否匹配查询 |
| Trace clustering | "Behavioral grouping" | 聚类相似运行用于漂移检测 |
| Guardrail enforcement | "Policy at log time" | 对日志内容做 PII/毒性/范围检查 |

## 延伸阅读

- [Langfuse docs](https://langfuse.com/) — tracing, evals, prompt mgmt
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — auto-instrumentation, drift
- [Comet Opik](https://www.comet.com/site/products/opik/) — optimization + guardrails
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the schema all three consume
