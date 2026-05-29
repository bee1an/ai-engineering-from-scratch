# 推理指标 — TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定推理部署是否正常工作。TTFT 是 prefill 加队列加网络。TPOT（等同于 ITL）是内存带宽受限的每 token decode 代价。端到端延迟是 TTFT 加 TPOT 乘以输出长度。吞吐是全集群聚合的每秒 token 数。但对产品真正重要的是 goodput——同时满足所有 SLO 的请求比例。高吞吐低 goodput 意味着你在处理永远无法按时到达用户的 token。Llama-3.1-8B-Instruct 在 TRT-LLM 上的 2026 参考数字：mean TTFT 162 ms，mean TPOT 7.33 ms，mean E2E 1,093 ms。始终报告 P50、P90、P99——永远不要只报 mean。还要注意测量陷阱：GenAI-Perf 在 ITL 计算中排除 TTFT，LLMPerf 包含它；两个工具对同一次运行的 TPOT 不一致。

**Type:** Learn
**Languages:** Python (stdlib, toy percentile calculator and goodput reporter)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、吞吐和 goodput，说出每个指标测量的组件。
- 解释为什么 mean 是 LLM serving 的错误统计量，以及如何读 P50/P90/P99。
- 构造一个 SLO 多约束（如 TTFT<500 ms AND TPOT<15 ms AND E2E<2 s）并据此计算 goodput。
- 说出两个对同一次运行 TPOT 不一致的 benchmark 工具并解释原因。

## 问题

"我们的吞吐是每秒 15,000 token。"然后呢？如果 40% 的请求端到端超过 2 秒，用户已经放弃了会话。吞吐本身不能告诉你产品是否正常工作。

推理有多个延迟维度，每个维度的失败方式不同。Prefill 是计算受限的，随 prompt 长度扩展。Decode 是内存带宽受限的，随 batch size 扩展。排队延迟是运维问题。网络是物理距离问题。你需要每个维度的独立指标，需要百分位数，还需要一个单一复合指标来回答"用户是否得到了预期的体验"——那就是 goodput。

## 核心概念

### TTFT — 首 token 时间

`TTFT = queue_time + network_request + prefill_time`

Prompt 长时 prefill 占主导。在 H100 上 Llama-3.3-70B FP8，32k prompt 纯 prefill 约 ~800 ms。Queue time 是负载下的调度器行为。Network request 是包含 TLS 的线路时间。TTFT 是用户在任何内容流回之前看到的延迟。

### TPOT / ITL — token 间延迟

同一个量的多个名字。`TPOT`（time per output token）、`ITL`（inter-token latency）、`decode latency per token`——都一样。它是首 token 之后连续流式 token 之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同一个 Llama-3.3-70B H100 栈上配合 chunked prefill，TPOT mean ~7 ms。没有 chunked prefill 时，在相邻序列的长 prefill 期间，TPOT 可能飙到 50 ms。关注 P99，不是 mean。

### E2E 延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

长输出（>500 token）时 E2E 由 TPOT 主导。短输出长 prompt 时 E2E 由 TTFT 主导。报告按输出长度条件化的 E2E。

### 吞吐

`throughput = total_output_tokens / elapsed_time`

聚合指标。告诉你集群效率。不告诉你单个请求的健康状况。

### Goodput — 你真正关心的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是多约束的。一个请求只有在每个约束都满足时才算"好"。Goodput 是这个比例。高吞吐 60% goodput 是失败。低吞吐 99% goodput 才是目标。

2026 年，goodput 是 MLPerf Inference v6.0 提交和 AI 平台提供商内部 SLA 跟踪中使用的指标。

### 为什么 mean 是错误的统计量

LLM 延迟分布是右偏的。一个 decode batch 中有一个长 prefill 邻居，可能 500 个 token 的 TPOT ~7 ms，20 个 token 的 TPOT ~60 ms。Mean TPOT 是 9 ms。P99 TPOT 是 65 ms。用户经常命中 P99——这就是他们离开的原因。

始终报告三元组（P50, P90, P99）。对用户体验来说，P99 是你要优化的。

### 参考数字 — Llama-3.1-8B-Instruct on TRT-LLM, 2026

- mean TTFT: 162 ms
- mean TPOT: 7.33 ms
- mean E2E: 1,093 ms
- P99 TPOT: 10-25 ms，取决于 chunked-prefill 配置。

这些是 NVIDIA 发布的参考点。它们随模型大小（70B 会是 3-5x）、硬件（H100 vs B200 ~3x）和负载变化。

### 测量陷阱

2026 年最常用的两个 benchmark 工具对同一次运行的 TPOT 不一致：

- **NVIDIA GenAI-Perf**：从 ITL 计算中排除 TTFT。ITL 从第 2 个 token 开始。
- **LLMPerf**：包含 TTFT。ITL 从第 1 个 token 开始。

对于一个 TTFT 500 ms、100 个输出 token 在 700 ms 总 decode 时间的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具选择改变数字。

始终声明用的哪个工具。始终公布定义。

### 构造 SLO

2026 年面向消费者的 70B 对话模型的合理 SLO：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- E2E P99 <= 3 s（<300-token 输出）。
- Goodput 目标 >= 99%。

企业 SLO 收紧 TTFT（200-400 ms）放宽 E2E。关键是写下来、测量全部三个、并将 goodput 作为单一复合指标跟踪。

### 如何测量

- 运行真实流量或逼真的合成流量（LLMPerf 配合 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- benchmark 运行目标为 2x 峰值并发。
- 运行 30-50 次迭代，取合并样本的百分位数。
- 发布时附带工具名、工具版本、模型、硬件、并发、prompt 分布。

## Use It

`code/main.py` 是一个 toy goodput 计算器。生成合成延迟分布，应用 SLO，计算 goodput。还展示了 GenAI-Perf vs LLMPerf 在同一 trace 上的 TPOT 差异。

## Ship It

本课产出 `outputs/skill-slo-goodput-gate.md`。给定负载和 SLO，产出一个 CI/CD 就绪的 benchmark 配方，以 goodput 而非吞吐作为部署门控。

## 练习

1. 运行 `code/main.py`。生成一个 1% 尾部尖峰的分布。当你将 P99 TPOT 从 30 ms 收紧到 15 ms 时，goodput 如何变化？
2. 一个供应商报价"Llama 3.3 70B H100 上 15,000 tok/s"。在信任之前说出三个要问的问题。
3. 为什么 chunked prefill 保护 P99 TPOT 但不保护 mean TPOT？
4. 为语音助手构造一个消费者 SLO（首 token 是被听到的，不是被读到的）。哪个指标对用户最可见？
5. 阅读 LLMPerf README 和 GenAI-Perf 文档。找出另外三个两个工具不一致的指标。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| TTFT | "time to first token" | 队列 + 网络 + prefill；长 prompt 时由 prefill 主导 |
| TPOT | "time per output token" | 首 token 之后的内存带宽受限每 token decode 代价 |
| ITL | "inter-token latency" | 在大多数工具中与 TPOT 相同（不是全部——见 GenAI-Perf） |
| E2E | "end to end" | TTFT + TPOT * output_len；加上响应侧网络 |
| Throughput | "tok/s" | 集群效率；没有延迟百分位数就没用 |
| Goodput | "SLO-met rate" | 同时满足所有 SLO 约束的请求比例 |
| P99 | "tail" | 百分之一最差延迟；用户体验指标 |
| SLO multi-constraint | "the joint" | 所有三个延迟边界的 AND；任何一个违反请求就失败 |
| GenAI-Perf vs LLMPerf | "the tool trap" | 工具对 ITL 是否包含 TTFT 不一致 |

## 延伸阅读

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的权威定义。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 替代定义和测量配方。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 真实部署上的应用测量。
- [LLMPerf](https://github.com/ray-project/llmperf) — Ray 基础的开源 benchmark。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 的 benchmark 工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 行业认可的基于 goodput 的 benchmark。
