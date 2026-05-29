# LLM API 负载测试 — 为什么 k6 和 Locust 会骗你

> 传统负载测试工具不是为流式响应、可变输出长度、token 级指标或 GPU 饱和设计的。两个陷阱坑了大多数团队。GIL 陷阱：Locust 的 token 级测量在 Python GIL 下运行分词，与高并发下的请求生成竞争；分词积压膨胀了报告的 inter-token 延迟——你的客户端是瓶颈，不是服务器。Prompt 均匀性陷阱：循环中的相同 prompt 只测试 token 分布上的一个点；真实流量有可变长度和多样的前缀匹配。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 修复了这个问题。2026 年工具映射：LLM 专用（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于 token 级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**— 流式感知，通过 TestRun/PrivateLoadZone CRD 实现 Kubernetes 原生分布式，最适合 CI/CD 门控；Vegeta 用于 Go 恒定速率饱和；Locust 2.43.3 仅在使用 LLM-Locust 扩展时适用于流式。负载模式：稳态、爬坡、尖峰（自动扩缩测试）、浸泡（内存泄漏）。

**Type:** Build
**Languages:** Python (stdlib, toy realistic-prompt generator + latency collector)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~75 minutes

## 学习目标

- 解释两个反模式（GIL 陷阱、prompt 均匀性陷阱），它们使通用负载测试工具对 LLM API 撒谎。
- 根据目的选择工具：LLMPerf（基准测试运行）、k6 + 流式扩展（CI 门控）、guidellm（大规模合成）、GenAI-Perf（NVIDIA 参考）。
- 设计四种负载模式（稳态、爬坡、尖峰、浸泡）并说明每种捕捉的故障模式。
- 使用 input token 的均值 + 标准差构建真实 prompt 分布，而非固定长度。

## 问题

你用 k6 在 500 并发用户下测试了 LLM 端点。它撑住了。你上线了。在 200 个真实用户下服务崩了——P99 TTFT 爆炸，GPU 打满。

发生了两件事。第一，k6 发送了 500 个相同的 prompt——你的请求合并和前缀缓存让它看起来像在处理 500 个并发 decode，实际上只处理了一个。第二，k6 不像人眼体验那样跟踪流式响应的 inter-token 延迟；它看到一个 HTTP 连接，而不是 500 个以不同间隔到达的 token。

LLM 的负载测试是一门独立学科。

## 核心概念

### GIL 陷阱（Locust）

Locust 使用 Python，在 GIL 下运行客户端分词。高并发下分词器排在请求生成后面。报告的 inter-token 延迟包含了客户端分词积压。你以为服务器慢；其实是测试工具慢。

修复：LLM-Locust 扩展将分词移到独立进程，或使用编译语言工具（k6、LLMPerf 使用 tokenizers.rs）。

### Prompt 均匀性陷阱

所有已知负载测试工具都允许你配置一个 prompt。在 10,000 次迭代的循环测试中每次发送完全相同的 prompt。服务器每次看到相同前缀——前缀缓存命中接近 100%，吞吐看起来很好。

修复：从 prompt 分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`——多样的长度，多样的内容。

### 四种负载模式

1. **稳态** — 恒定 RPS 持续 30-60 分钟。捕捉：基线性能退化。
2. **爬坡** — 15 分钟内从 0 线性增加到目标 RPS。捕捉：容量断点、预热异常。
3. **尖峰** — 突然 3-10 倍 RPS 持续 2 分钟然后恢复。捕捉：自动扩缩延迟、队列饱和、冷启动影响。
4. **浸泡** — 稳态持续 4-8 小时。捕捉：内存泄漏、连接池漂移、可观测性溢出。

### 2026 年工具映射

**LLMPerf**（Anyscale）— Python 但 Rust 后端分词。均值/标准差 prompt。流式感知。性能测试的最佳默认选择。

**NVIDIA GenAI-Perf** — NVIDIA 的参考工具。使用 Triton 客户端；全面的指标覆盖。注意其 ITL 不包含 TTFT；LLMPerf 的包含。两个工具对同一服务器产生不同的 TPOT。

**LLM-Locust**（TrueFoundry）— 修复 GIL 陷阱的 Locust 扩展。熟悉的 Locust DSL + 流式指标。

**guidellm** — 大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go，编译，无 GIL）增加了流式感知指标。
- k6 Operator 使用 TestRun / PrivateLoadZone CRD 实现 Kubernetes 原生分布式测试。
- 最适合 CI/CD 门控和 SLA 测试。

**Vegeta** — Go，比 k6 简单。恒定速率 HTTP 饱和。非 LLM 感知但适合网关/限流测试。

**Locust 2.43.3 原版** — 对 LLM 有 GIL 陷阱。仅在使用 LLM-Locust 扩展时可用。

### CI 中的 SLA 门控

在 PR 上运行 k6：

- 基线 RPS 下各 30-50 次迭代。
- 门控：P50/P95 TTFT、5xx < 5%、TPOT 低于阈值。
- 违规时中断构建。

### 真实 prompt 分布

从真实流量样本构建（如果有的话）或从已发布分布构建（如 ShareGPT prompt 用于聊天、HumanEval 用于代码）。将均值 + 标准差输入 LLMPerf。不惜一切代价避免单 prompt 循环。

### 需要记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：流式感知指标。
- 典型 LLMPerf 运行：并发 X 下 100-1000 请求。
- 典型 CI 门控：每 PR 30-50 次迭代。
- 四种模式：稳态、爬坡、尖峰、浸泡。

## Use It

`code/main.py` 模拟带真实 prompt 分布的负载测试，测量有效 TPOT，并演示均匀 prompt 陷阱。

## Ship It

本课产出 `outputs/skill-load-test-plan.md`。给定工作负载和 SLA，选择工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。比较均匀 vs 真实分布——差距在哪里？
2. 写 k6 脚本用于 CI 门控：100 并发下 TTFT P95 < 800 ms，运行 5 分钟。
3. 你的浸泡测试显示内存每小时增长 50 MB。列出三个原因和区分它们的检测手段。
4. 尖峰测试从 10 RPS 到 100 RPS。如果 Karpenter + vLLM production-stack 就位（Phase 17 · 03 + 18），预期恢复时间是多少？
5. GenAI-Perf 报告 TPOT=6ms；LLMPerf 报告 TPOT=11ms，同一服务器。解释原因。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| LLMPerf | "LLM 测试工具" | Anyscale 基准测试工具，流式感知 |
| GenAI-Perf | "NVIDIA 工具" | NVIDIA 参考测试工具 |
| LLM-Locust | "LLM 版 Locust" | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | "合成基准" | 大规模合成工具 |
| k6 Operator | "K8s k6" | 基于 CRD 的分布式 k6 |
| GIL trap | "Python 客户端开销" | 分词积压膨胀报告延迟 |
| Prompt-uniformity trap | "单 prompt 谎言" | 相同 prompt 循环命中缓存，膨胀吞吐 |
| Steady-state | "恒定负载" | 恒定 RPS 持续 N 分钟 |
| Ramp | "线性上升" | 在持续时间内从 0 到目标 |
| Spike | "突发测试" | 突然倍增然后恢复 |
| Soak | "长时间测试" | 数小时用于泄漏检测 |

## 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
