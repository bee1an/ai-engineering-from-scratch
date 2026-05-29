# LLM FinOps — 单位经济与多租户归因

> 传统 FinOps 在 LLM 支出上失效。成本是 token 交易，不是资源运行时间。标签无法映射——API 调用是交易，不是资产。工程决策（prompt 设计、上下文窗口、输出长度）就是财务决策。2026 年的 playbook 有三个归因维度需要在第一天就埋点：per-user（`user_id`）用于座位定价和扩展，per-task（`task_id` + `route`）用于产品表面成本和优先级，per-tenant（`tenant_id`）用于单位经济和续约。四个 token 层——prompt、tool、memory、response——一个桶隐藏支出。多租户产品的执行阶梯：按租户限流（2-3 倍预期峰值，清晰的 429 + retry-after）；日支出上限（1.5-3 倍合同上限；触发限流收紧 + 告警）；支出 z-score > 4 时的熔断开关（自动暂停 + 呼叫值班）。归因模式：tag-and-aggregate、telemetry-joiner（trace-ID → 计费；最高精度）、sampling-and-extrapolation、model-based allocation、event-sourced、real-time streaming。单位指标：每解决查询成本、每生成产物成本——不是 $/M tokens。事后打标总是遗漏；在请求创建时埋点。

**Type:** Learn
**Languages:** Python (stdlib, toy cost-attribution simulator with kill switch)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 14 (Caching)
**Time:** ~60 minutes

## 学习目标

- 解释为什么传统 FinOps（标签 + 层级）在 LLM 支出上失效，并说明三个新归因维度。
- 列举四个 token 层（prompt、tool、memory、response）以及为什么单桶计费隐藏成本。
- 为多租户产品设计执行阶梯（限流 → 支出上限 → 熔断开关）。
- 选择单位指标（每解决查询成本/产物成本）而非 $/M tokens。

## 问题

你的账单显示 $40,000。你不知道：
- 哪个租户花的。
- 哪个产品功能驱动的。
- 是否有个别用户在滥用。
- 是 prompt 膨胀、tool 调用还是 memory 放大导致的。

供应商侧的 tag-and-aggregate 对云资源（EC2、S3）有效，因为标签传播到行项目。LLM API 调用不会自动打标——你必须在调用点打上 user/task/tenant 并贯穿全程。事后归因总是遗漏边缘情况。

## 核心概念

### 三个归因维度

**Per-user**（`user_id`）：谁花了多少。驱动座位定价、扩展对话、识别重度用户。

**Per-task**（`task_id` + `route`）：哪个产品表面花了多少。驱动功能优先级、砍掉昂贵功能的决策。

**Per-tenant**（`tenant_id`）：哪个客户是盈利的。驱动单位经济、续约定价、层级阈值。

第一天就在调用点埋点所有三个。事后总是更差。

### 四个 token 层

| 层 | 示例 | 典型占比 |
|---|------|---------|
| Prompt | system + 用户输入 | 40-60% |
| Tool | tool-call 结果回传 | 20-40%（agent 工作负载） |
| Memory | 历史对话/检索文档 | 10-30% |
| Response | 模型输出 | 10-30% |

将四者放在一个桶里使优化盲目。在归因 schema 中拆分它们。

### 执行阶梯

1. **限流** 按租户。2-3 倍预期峰值。返回 429 + `Retry-After`。租户感受到摩擦；无意外账单。

2. **日支出上限** 按租户。1.5-3 倍合同上限。触发：收紧限流 + 告警客户成功团队。

3. **熔断开关** 支出 z-score > 4（相对于租户基线）。自动暂停租户；呼叫值班；升级到运维 + CS。

### 归因模式

- **Tag-and-aggregate**：打元数据 header；后续聚合。简单；粗略。
- **Telemetry joiner**：通过 trace ID 将 trace 关联到计费。最高精度。成熟团队的做法。
- **Sampling + extrapolation**：采样 5-10%，乘以倍数。对粗略支出有效；遗漏尾部。
- **Model-based allocation**：回归推断成本驱动因素。用于没有标签的历史数据。
- **Event-sourced**：成本作为流中的事件（Kafka / Kinesis）。实时。
- **Real-time streaming**：仪表板亚秒更新。

### 每 X 成本是单位指标

$/M tokens 是供应商话术。产品指标：

- 每解决支持工单成本。
- 每生成文章成本。
- 每成功 agent 任务成本。
- 每用户会话分钟成本。

将成本绑定到产品结果。否则优化没有锚点。

### 成本归因 trace 形状

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

每次调用发射。存储在数据湖。按维度聚合。Phase 17 · 13 可观测性栈是这些数据的归宿。

### 复合节省栈

栈：cache + batch + route + gateway。四者全开时：
- Cache L2（Phase 17 · 14）：输入约便宜 10 倍。
- Batch（Phase 17 · 15）：50% 折扣。
- 路由到廉价模型（Phase 17 · 16）：60% 成本降低。
- Gateway 效率（Phase 17 · 19）：冗余 + 重试。

最佳情况叠加：约为朴素基线的 5-10%。大多数团队启用了 2-3 个杠杆；很少有人全部叠加。

### 需要记住的数字

- 归因维度：per-user、per-task、per-tenant。
- 四个 token 层：prompt、tool、memory、response。
- 熔断开关：支出 z-score > 4。
- 单位指标：每解决查询成本，不是 $/M tokens。
- 叠加优化：可达基线的约 5-10%。

## Use It

`code/main.py` 模拟带三级执行阶梯的多租户 LLM 服务。注入一个滥用租户并演示熔断开关触发。

## Ship It

本课产出 `outputs/skill-finops-plan.md`。给定产品和规模，设计归因 schema 和执行阶梯。

## 练习

1. 运行 `code/main.py`。在什么 z-score 时熔断开关触发？你如何选择阈值？
2. 设计一个 per-tenant、per-task 成本仪表板。你首先构建的 5 个视图是什么？
3. 你最大的租户单位经济为负。提出三个按客户影响排序的干预措施。
4. 计算支持产品的每解决工单成本：3M tokens/工单，约 800 工单/天，GPT-5 缓存价格。
5. 论证事后打标是否可行。什么时候可以接受？

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| Per-user attribution | "用户级成本" | 每次调用打上 `user_id` |
| Per-task attribution | "功能成本" | `task_id` + `route` 标识产品表面 |
| Per-tenant attribution | "客户成本" | `tenant_id`；驱动单位经济 |
| Four token layers | "成本层" | prompt + tool + memory + response |
| Rate limit | "429 护栏" | 在网关执行的按租户上限 |
| Daily spend cap | "日上限" | 租户范围预算 + 告警 |
| Kill switch | "自动暂停" | 支出 z-score > 4 触发自动暂停 |
| Cost per resolved | "产品单位指标" | 成本绑定到产品结果，不是 token |
| Telemetry joiner | "trace-to-billing" | 最高精度归因模式 |
| Stacked optimization | "cache+batch+route+gateway" | 复合节省至基线的约 5-10% |

## 延伸阅读

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
