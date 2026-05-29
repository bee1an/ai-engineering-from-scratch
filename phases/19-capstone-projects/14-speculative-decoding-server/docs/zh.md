# Capstone 14 — Speculative-Decoding 推理服务器

> EAGLE-3 在 vLLM 0.7 中实现了真实流量 2.5-3x 吞吐量提升。P-EAGLE（AWS 2026）将并行推测进一步推进。SGLang 的 SpecForge 大规模训练 draft heads。Red Hat 的 Speculators hub 发布了常见开源模型的对齐 drafts。TensorRT-LLM 将 speculative decoding 作为 NVIDIA 上的一等功能。2026 年的生产 serving 技术栈是 vLLM 或 SGLang 配合 EAGLE 系列 drafts、FP8 或 INT4 量化、以及基于 queue-wait 的 HPA。本 capstone 的目标是以 2.5x+ 基线吞吐量服务两个开源模型，并提供完整的尾延迟报告。

**Type:** Capstone
**Languages:** Python (serving), C++ / CUDA (kernel inspection), YAML (configs)
**Prerequisites:** Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 17 (infrastructure)
**Phases exercised:** P3 · P7 · P10 · P17
**Time:** 30 hours

## 问题

Speculative decoding 在 2026 年成为了标配。EAGLE-3 draft heads 在 target 模型的 hidden states 上训练，预测前方 N 个 token；target 模型在单次 pass 中验证。60-80% 的 acceptance rate 转化为 2-3x 端到端吞吐量。vLLM 0.7 原生集成了这一功能。SGLang + SpecForge 提供训练 pipeline。Red Hat 的 Speculators 发布了 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 的对齐 drafts。

核心技艺在于 serving 运维，而非模型本身。Acceptance rate 会随流量分布漂移（ShareGPT vs 代码 vs 领域数据）。拒绝时的尾延迟比不做 speculation 更差 — 你必须在多个 batch size 下报告 p99，而不仅仅是稳态 tokens/sec。每 1M tokens 的成本对比 Anthropic / OpenAI API 是可信度杠杆。

## 概念

Speculative decoding 有两层。**Draft** 模型（EAGLE-3 head、ngram、或更小的 target 对齐模型）每步提议 k 个候选 token。**Target** 模型在一次 pass 中验证所有 k 个；任何被接受的前缀替换 greedy 路径。Acceptance rate 取决于 draft-target 对齐度和输入分布。

EAGLE-3 在大多数流量上优于 ngram drafts。P-EAGLE 运行并行推测以获得更深的 draft tree。权衡：拒绝时的 P99 延迟更高，因为 verify pass 更大。Serving 配置必须报告按 batch-size 分桶的延迟以暴露这一点。

部署在 Kubernetes 上。vLLM 0.7 每 GPU 或 tensor-parallel shard 运行一个 replica。HPA 基于 queue-wait 而非 CPU 进行自动扩缩。FP8（Marlin）和 INT4（AWQ）量化将 GPU 内存控制在 H100 / H200 范围内。端到端报告包括吞吐量、acceptance rate、batch 1/8/32 下的 p50/p99、以及 $/1M tokens。

## 架构

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- Serving：vLLM 0.7 或 SGLang 0.4
- Speculative 方法：EAGLE-3 draft heads、P-EAGLE parallel speculation、ngram fallback
- Draft 训练：SpecForge (SGLang) 或 Red Hat Speculators
- Target 模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化：FP8 (Marlin)、INT4 AWQ
- 部署：Kubernetes + NVIDIA device plugin；基于 queue-wait metric 的 HPA
- 评估：ShareGPT、MT-Bench-v2、GSM8K、HumanEval 用于跨领域 acceptance 测量
- 参考：TensorRT-LLM speculative decoding 作为供应商基线

## 构建步骤

1. **Target 模型准备。** 选择 Llama 3.3 70B。通过 Marlin 量化为 FP8。在 1xH100（或 2x tensor-parallel）上用 vLLM 0.7 部署。

2. **Draft 来源。** 从 Red Hat Speculators 拉取对齐的 EAGLE-3 draft head（或通过 SpecForge 训练一个）。加载到 vLLM 的 speculative-decoding 配置中。

3. **基线数据。** 在启用 speculation 之前：batch 1/8/32 下的 tokens/s、p50/p99 延迟、GPU 利用率。发布。

4. **启用 EAGLE-3。** 切换配置；重新运行相同 benchmark。报告加速比、acceptance rate、p99 尾延迟变化。

5. **P-EAGLE。** 启用并行推测；测量更深的 draft tree vs 串行 EAGLE-3。报告 P-EAGLE 帮助 vs 伤害的拐点。

6. **领域流量。** 通过同一服务器运行 ShareGPT vs HumanEval vs 领域特定流量。测量每种分布的 acceptance rate。识别 drafts 何时漂移。

7. **第二个 target 模型。** 在 Qwen3-Coder-30B MoE 上运行相同 pipeline。Draft 更棘手（MoE routing noise）。报告。

8. **K8s HPA。** 在 K8s 下部署，HPA 跟踪 `queue_wait_ms`。演示负载增加三倍时的 scale-out。

9. **成本对比。** 在相同 eval 上计算 $/1M tokens vs Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4。发布。

## 使用示例

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 交付产出

`outputs/skill-inference-server.md` 描述交付物。一个经过测量的 speculative decoding serving 技术栈、完整的 benchmark 报告、以及 K8s 部署。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 相对基线的测量加速比 | 两个模型上 2.5x+ 吞吐量且质量匹配 |
| 20 | 真实流量上的 acceptance rate | 按分布的 acceptance-rate 报告 |
| 20 | P99 尾延迟纪律 | 有无 speculation 时 batch 1/8/32 的 p99 |
| 20 | 运维 | K8s 部署、基于 queue-wait 的 HPA、平滑 rollout |
| 15 | 文档和方法论 | 清晰解释改了什么以及为什么 |
| **100** | | |

## 练习

1. 测量当 draft 比 target 落后一个版本时的 acceptance-rate 退化（例如 Llama 3.3 -> 3.4 drift）。构建监控告警。

2. 实现 ngram-fallback：如果 EAGLE-3 acceptance 降到阈值以下，切换到 ngram drafts。报告可靠性改善。

3. 运行受控 MoE 实验：相同的 Qwen3-Coder-30B，注入 routing noise vs 不注入。测量 draft acceptance 敏感度。

4. 扩展到 H200（141 GB）。报告每 replica 模型大小余量的增加，以及是否可以服务未量化的 Llama 3.3 70B。

5. 在相同 H100 硬件上 benchmark TensorRT-LLM speculative decoding。报告它在哪些方面优于 vLLM。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Draft model | "Speculator" | 为 target 验证提议 N 个 token 的小模型 |
| EAGLE-3 | "2026 draft architecture" | 在 target hidden states 上训练的 draft head；约 75% acceptance |
| P-EAGLE | "Parallel speculation" | 在一次 target pass 中验证的 draft 分支树 |
| Acceptance rate | "Hit rate" | 被接受而无需 resampling 的 drafted tokens 比例 |
| Quantization | "FP8 / INT4" | 低精度权重以在 GPU 内存中容纳更大模型 |
| Queue wait | "HPA metric" | 请求在推理开始前在 pending 队列中等待的时间 |
| Speculators hub | "Aligned drafts" | Red Hat Neural Magic 发布的常见开源模型 EAGLE drafts hub |

## 延伸阅读

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 参考 serving 技术栈
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行 speculative decoding 论文 + 集成
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — draft-head 训练 pipeline
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 对齐 draft hub
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — 供应商替代方案
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — 商业参考
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — 方法论文
- [vLLM repository](https://github.com/vllm-project/vllm) — 代码和 benchmarks
