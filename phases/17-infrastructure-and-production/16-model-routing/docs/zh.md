# 模型路由作为成本优化原语

> 一个动态代理评估每个请求（任务类型、token 长度、embedding 相似度、置信度），将简单查询发送到廉价模型，复杂查询升级到前沿模型。也叫模型级联（model cascading）。生产案例显示在美国/英国/欧盟部署中，等质量条件下可实现 20-60% 的成本降低；高流量 SaaS 上 30% 的路由效率提升可转化为六位数的年度节省。2026 年的背景是 LLM 推理价格每年下降约 10 倍——GPT-4 级别的 token 从 2022 年底的 $20/M 降到 2026 年的约 $0.40/M。大部分降幅来自更好的 serving 栈（Phase 17 · 04-09），而非硬件。路由是你在不产生产品退化的前提下将价格下降转化为利润的方式。失败模式是廉价模型漂移：路由将 40% 推向弱模型，推理任务质量下降 3-5%，一个季度内没人注意到。用在线质量指标来把关路由，而不仅仅是离线评估集。

**Type:** Learn
**Languages:** Python (stdlib, toy cascading router simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**Time:** ~60 minutes

## 学习目标

- 解释模型级联：先走廉价模型 + 置信度检查，低置信度时升级。
- 列举四种路由信号（任务分类、prompt 长度、与已知困难集的 embedding 相似度、首次推理的自信度）。
- 计算目标路由分配和质量损失容忍度下的预期混合成本。
- 指出捕捉廉价模型蠕变的漂移监控指标（在线质量门控）。

## 问题

你的服务每月在 GPT-5 上花费 $80k。分析显示 70% 的查询是简单的："巴黎现在几点？""改写这句话。"一个 Haiku 级别的模型能以 3% 的成本完美处理这些。30% 需要 GPT-5 的推理能力——编程、数学、多步规划。

如果你把 70% 路由到廉价模型、30% 到昂贵模型，账单在相同产品质量下降低约 65%。这就是路由。难点在于构建代理时不让质量退化。

## 核心概念

### 四种路由信号

1. **任务分类**：simple/complex/codegen/math/chat。可以是基于规则的分类器、小型 LLM（Haiku 级别 $0.25/M）、或与标注桶的 embedding 相似度。输出：route = cheap / balanced / frontier。

2. **Prompt 长度**：>4K token 的 prompt 通常需要前沿模型保证连贯性。<500 token 的通常不需要。

3. **与已知困难集的 embedding 相似度**：如果查询与已知困难桶接近（cosine > 0.88），直接升级到前沿模型。

4. **首次推理的自信度**：先发给廉价模型；如果模型的 log-probs 显示低置信度、或拒绝回答、或输出模棱两可的语言，则重试前沿模型。约 10% 的流量增加 P95 延迟，但其余 90% 节省 50%+。

### 三种模式

**Pre-route**（前置分类器）：增加约 5-10ms 延迟；整体最快。

**Cascade**（先走廉价，低置信度时升级）：中位延迟约 1.2 倍（廉价运行 + 验证），升级时约 2 倍。质量下限最好。

**Ensemble route**（并行运行廉价和前沿模型的样本，reward-model 选择）：最高质量，最高成本；仅用于关键 A/B 测试。

### 实现

AI gateway（Phase 17 · 19）暴露路由功能。LiteLLM 有 `router` 配置支持 fallback 和 cost-routing。Portkey 有 guards + routing。Kong AI Gateway 有基于插件的路由。OpenRouter 的模型市场暴露推荐 API。

开源：RouteLLM (LMSYS)、Not Diamond (商业)、Prompt Mule。

### 2026 价格曲线

| 模型级别 | 2022 年底 | 2026 | 变化 |
|---------|----------|------|------|
| GPT-4 级别质量 | ~$20/M | ~$0.40/M | 便宜 50 倍 |
| 前沿 (GPT-5, Claude 4) | — | ~$3-10/M | 新层级 |

大部分改善来自 serving 效率——Phase 17 · 04-09 的核心课程转化为供应商侧的成本下降。路由让你在应用层捕获这些收益，而不是等所有用户迁移到廉价层级。

### 漂移才是真正的风险

你的路由将 40% 发送到廉价模型。六个月后，任务分布发生变化（用户变得更复杂，问更长的问题）。路由器没有注意到，因为它的分类器是在 Q1 数据上训练的。质量悄悄下降。没人抱怨得足够大声。你在竞品基准测试中发现自己落后了。

用在线质量指标把关路由：

- 每条路由的用户点赞/点踩。
- 对每条路由的留出样本（5%）进行自动 LLM-judge 评估。
- 升级率：如果 cascade 向上路由超过 30%，说明廉价模型被过度路由了。
- 每条路由的拒绝率。

### 需要记住的数字

- 2026 年等质量路由节省：20-60% 案例。
- LLM 价格下降 2022-2026：每年约 10 倍。
- GPT-4 级别 2022 vs 2026：~$20/M → ~$0.40/M。
- Cascade 延迟影响：中位约 1.2 倍，升级时约 2 倍（约 10% 流量）。

## Use It

`code/main.py` 模拟混合工作负载上的 pre-route、cascade 和 ensemble。报告混合成本、质量损失和升级率。

## Ship It

本课产出 `outputs/skill-router-plan.md`。给定工作负载和质量预算，选择路由模式和信号。

## 练习

1. 运行 `code/main.py`。在什么准确率下限时 cascade 优于 pre-route？
2. 你的用户群是 30% 企业用户（复杂查询）、70% 免费层（简单查询）。设计路由分配。用什么在线指标把关？
3. 一条路由质量下降 2% 但节省 40%。该上线吗？取决于产品——论证两种立场。
4. 使用 OpenAI / Anthropic API 的 logprobs 实现置信度检查。你从什么阈值开始？
5. 六个月内，升级率从 8% 攀升到 22%。诊断三个原因及各自的修复方案。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| Model routing | "成本代理" | 每个请求动态选择模型 |
| Model cascade | "先廉价后升级" | 先走廉价模型，低置信度时升级到前沿 |
| Pre-route | "先分类" | 前置分类器；不重跑 |
| Ensemble route | "并行选择" | 运行多个，reward-model 选最佳 |
| Escalation rate | "升级率" | cascade 中升级到上层的请求比例 |
| RouteLLM | "LMSYS 路由器" | 开源路由库 |
| Not Diamond | "商业路由器" | SaaS 模型路由产品 |
| Drift | "廉价蠕变" | 分布漂移但路由器未察觉 |
| Online quality gate | "在线检查" | 自动 LLM-judge 采样线上流量 |

## 延伸阅读

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — multi-model gateway with routing primitives.
