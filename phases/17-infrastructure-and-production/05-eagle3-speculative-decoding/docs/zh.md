# EAGLE-3 投机解码的生产实践

> 投机解码将一个快速的 draft 模型与目标模型配对。Draft 提出 K 个 token；目标模型在一次 forward 中验证；被接受的 token 相当于免费获得。2026 年，EAGLE-3 是生产级变体——它在目标模型的隐藏状态上训练 draft head，而非在原始 token 上训练，将接受率 alpha 推到通用对话场景下的 0.6-0.8 区间。正确的问题不是"draft 有多快"，而是"alpha 在我的流量上是多少？"如果 alpha 低于 ~0.55，在高并发下投机解码是净负收益，因为每个被拒绝的 draft 都要付出第二次目标模型 forward 的代价。这节课教你先测量 alpha，再开启开关。

**Type:** Learn
**Languages:** Python (stdlib, toy acceptance-rate simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 18 (Multi-Token Prediction)
**Time:** ~60 minutes

## 学习目标

- 说出投机解码的三代演进，解释 EAGLE-3 相对于 EAGLE-2 和经典 draft 模型的变化。
- 定义接受率 alpha，根据 alpha 和 K（draft 长度）计算预期加速比，识别目标并发下的盈亏平衡 alpha。
- 解释为什么投机解码在 vLLM 2026 中是 opt-in（非默认），以及为什么不测量 alpha 就开启是生产反模式。
- 编写测量计划：用哪个 benchmark、哪种 prompt 分布、哪个并发点、以哪个指标作为门控。

## 问题

Decode 是内存带宽瓶颈。在 H100 上运行 Llama 3.3 70B FP8 时，每个解码 token 读取 ~140 GB/s 的权重并输出一个 token。GPU 算力在 decode 阶段几乎空闲——瓶颈是 HBM 带宽，不是矩阵乘法吞吐。

投机解码利用了这个空隙。用一个廉价的 draft 模型生成 K 个候选 token，然后让目标模型在一次 forward pass 中验证全部 K 个。每个被验证通过的 token 实际上是免费的（摊销到目标模型本来就要做的 batch-of-K forward 中）。

经典 draft 模型方法使用同系列的小模型（Llama 3.2 1B 为 Llama 3.3 70B 做 draft）。能用，但接受率平庸——小模型的分布与目标模型有偏差。EAGLE、EAGLE-2、再到 EAGLE-3，直接在目标模型的内部状态上训练轻量 draft head，使 draft 的分布更紧密地跟踪目标。这就是为什么 alpha 从 draft 模型的 0.4 提升到 EAGLE-3 的 0.6-0.8。

关键点：EAGLE-3 在 vLLM 2026 中是 opt-in 的。必须显式设置 `speculative_config`。不设置就没有加速。团队在没有测量真实流量 alpha 的情况下开启，往往看到尾部延迟变差而非变好。

## 核心概念

### 投机解码实际带来什么

没有 spec decode 时，每个 token 的代价是一次目标模型 forward。有 spec decode 时，draft 长度为 K、接受率为 alpha，每次目标模型 forward 的期望产出 token 数为 `1 + K * alpha`。加速比为 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是 draft 加验证的开销。K=5, alpha=0.7 时：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。实际数字集中在 2-3x，因为生产流量上 alpha 很少那么高，且 epsilon 在大 batch 下会增长。

### 为什么 alpha 是唯一重要的指标

被拒绝的 token 不会消失——它们强制对第一个被拒 token 做第二次目标模型 forward。在 alpha 降到 0.4 的负载上，你要付出 draft 开销加验证加重新生成的代价。在高并发（比如 256 并发）下，decode batch 已经足够大，"纯目标模型"和"目标模型加验证"之间的内存带宽差距缩小。在 2026 大多数硬件上，alpha 低于 0.55 时 spec decode 是净负收益。

Alpha 因负载而异。在 ShareGPT 风格的通用对话上，基于 ShareGPT 训练的 EAGLE-3 达到 0.6-0.8。在领域特定流量（代码、医疗、法律）上，基于通用数据训练的 draft head 降到 0.4-0.6。训练领域特定的 draft head 可以恢复 alpha——相比目标模型微调，这是一个轻量快速的训练任务。

### EAGLE 各代一览

- **经典 draft 模型**：同系列小模型。Alpha 0.3-0.5。基础设施简单——加载两个模型，draft 每次目标 forward 运行 K 次。
- **EAGLE-1 (2024)**：在目标隐藏状态（最后一层）上训练单个 draft head。Alpha ~0.5-0.6。在目标模型之上有少量参数开销。
- **EAGLE-2 (2025)**：自适应 draft 长度和树状 draft（在一次目标 pass 中验证多个分支）。Alpha ~0.6-0.7。更复杂的 draft 调度器。
- **EAGLE-3 (2025-2026)**：在目标模型多层（不仅是最后一层）上训练 draft head，更好的对齐。通用对话上 Alpha ~0.6-0.8。

### 2026 生产配方

1. 先部署纯目标模型。在目标并发下测量基线 TTFT、ITL、吞吐。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 draft。重新跑 benchmark。
3. 记录接受率 alpha。vLLM V1 将其报告为 `spec_decode_metrics.accepted_tokens_per_request`。除以请求的 draft 长度得到 alpha。
4. 如果生产流量分布上 alpha < 0.55，禁用 spec decode 或训练领域特定的 EAGLE-3 draft。
5. 在生产并发下重新运行。确认 P99 ITL 没有变差。

### 生产陷阱：P99 尾部

Mean ITL 在 spec decode 下会降低。如果不调优，P99 可能变差。被拒绝的 draft 触发两次 pass 序列（draft + 验证失败 + 重新生成）。在满 batch 下，这两次 pass 串行化。关注 P99 ITL，而非 P50。

### EAGLE-3 已部署的场景

Google 在 2025 年将投机解码部署到 AI Overviews（相同质量，更快响应）。vLLM V1 以 `speculative_config` 作为文档化接口；V1 中的 N-gram GPU 投机解码是与 chunked prefill 兼容的变体。SGLang 支持 EAGLE-3 作为前缀密集型负载的推荐 draft 路径。

### 一行公式算盈亏平衡

期望加速比：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1` 解出 alpha：`alpha_breakeven = verify_overhead / K`。典型 verify_overhead ~0.15，K=5 时：`alpha_breakeven = 0.03`。但这是原始 decode 数学。在高并发下验证开销上升，且 decode batch 已经在多个序列间摊销内存读取，所以实际 alpha_breakeven 在实践中攀升到 ~0.45-0.55。

### 什么时候不该用投机解码

- Batch-1 离线生成，延迟无所谓。用纯目标模型。
- 非常短的输出（50 token 以下）。Draft 开销和验证代价占主导。
- 没有领域训练 draft head 的专业领域。Alpha 太低。
- vLLM v0.18.0 加 draft-model spec decode 加 `--enable-chunked-prefill`。这个组合无法编译。文档中的例外是 V1 中的 N-gram GPU spec decode。

## Use It

`code/main.py` 模拟有无投机解码的 decode 循环，覆盖一系列 alpha 值和 draft 长度 K。打印盈亏平衡 alpha、测量加速比和尾部行为。在多个 (alpha, K) 组合上运行，精确看到投机解码在哪里不再划算。

## Ship It

本课产出 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和并发目标，产出分阶段的 EAGLE-3 上线计划——benchmark 基线、启用配置、测量 alpha、以 alpha >= 0.55 为门控、监控 P99 ITL。

## 练习

1. 运行 `code/main.py`。K=5 时，2x 加速需要多少 alpha？3x 呢？对 verify_overhead 有多敏感？
2. 假设生产流量 70% 通用对话、30% 代码。通用对话用 ShareGPT 训练的 EAGLE-3 达到 alpha 0.7；代码达到 alpha 0.4。混合 alpha 是多少？spec decode 是否净正收益？
3. 阅读 vLLM `speculative_config` 文档。说出三种模式（draft model、EAGLE、N-gram）以及哪种与 chunked prefill 兼容。
4. 启用 EAGLE-3 后 mean ITL 降了 25%，但 P99 ITL 升了 15%。诊断并提出缓解方案。
5. 计算 Llama 3.3 70B 的 EAGLE-3 draft head 的内存开销。与运行 Llama 3.2 1B 作为经典 draft 相比如何？

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| Speculative decoding | "draft plus verify" | 用廉价模型提出 K 个 token，在一次目标 forward 中验证全部 K 个 |
| Acceptance rate alpha | "spec accept rate" | 被目标模型接受的 draft token 比例；唯一重要的指标 |
| Draft length K | "spec k" | Draft 每次目标 forward 提出多少 token；典型 4-8 |
| Verify overhead epsilon | "spec overhead" | 验证加重新生成相对于纯目标 forward 的额外代价；随 batch 增长 |
| EAGLE-3 | "latest EAGLE" | 2025-2026 变体；在目标模型多层上训练 draft head；通用对话 alpha 0.6-0.8 |
| `speculative_config` | "vLLM spec config" | vLLM V1 中的显式 opt-in；不设置就没有加速 |
| N-gram spec decode | "N-gram draft" | GPU 端使用 prompt 中 N-gram 查找的 draft；与 chunked-prefill 兼容 |
| Break-even alpha | "no-op alpha" | Spec decode 加速为零的 alpha；在生产并发下关注此值 |
| Rejected-draft two-pass | "reroll cost" | Draft 被拒时的两次目标 forward；驱动 P99 尾部 |

## 延伸阅读

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — `speculative_config` 和 V1 中 chunked-prefill 兼容性的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 精确字段集。
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — 原始 EAGLE draft-head 设计。
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — 自适应 draft 和树结构。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 基于投机解码的高效 LLM 系统。
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) — 生产上线清单。
