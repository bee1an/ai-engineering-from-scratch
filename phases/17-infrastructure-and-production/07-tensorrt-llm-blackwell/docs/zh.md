# TensorRT-LLM 在 Blackwell 上的 FP8 与 NVFP4

> TensorRT-LLM 是 NVIDIA 专属的，但它在 Blackwell 上赢了。在 GB200 NVL72 配合 Dynamo 编排下，SemiAnalysis InferenceX 在 2026 年 Q1-Q2 测得 120B 模型每百万 token $0.012，对比 H100 + vLLM 的 $0.09/M——7x 的经济差距。这个栈是三种浮点精度的叠加：FP8 对 KV cache 和 attention kernel 仍然关键，因为它们需要动态范围；NVFP4（4-bit microscaling）处理权重和激活；multi-token prediction (MTP) 和分离式 prefill/decode 在此基础上再加 2-3x。Day-0 模型支持直接加载 FP4 权重，无需训后转换。2026 年工程团队面临的取舍：TRT-LLM 是封闭的 NVIDIA 栈，采用它意味着用可移植性换吞吐。在承诺之前，对你的模型和硬件组合算清楚。

**Type:** Learn
**Languages:** Python (stdlib, toy FP8/NVFP4 memory and cost calculator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 13 (Quantization)
**Time:** ~75 minutes

## 学习目标

- 解释为什么即使权重在 NVFP4，FP8 对 KV cache 和 attention 仍然关键。
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 占用，并推理节省来自哪里。
- 说出 TRT-LLM 利用的 Blackwell 特有功能（day-0 FP4、MTP、分离式 serving、all-to-all 原语）。
- 判断 TRT-LLM 的 NVIDIA 锁定何时值得 7x 成本差距（对比 Hopper 上的 vLLM）。

## 问题

2026 年推理经济学的前沿是"每美元多少 token"。答案取决于四个叠加选择：硬件代际（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、serving 引擎（vLLM vs SGLang vs TRT-LLM）、以及编排（普通 vs 分离式 vs Dynamo）。

在 Hopper 上用 vLLM，一个 120B MoE 运行在 ~$0.09/百万 token。在 Blackwell 上用 TRT-LLM + Dynamo，同一模型运行在 ~$0.012——便宜 7x。部分差距来自硬件（Blackwell 每 GPU LLM 吞吐是 Hopper 的 11-15x）。部分来自栈：FP4 权重、MTP draft、分离式 prefill/decode、以及 NVLink 5 all-to-all 用于 MoE expert 通信。

你无法在 NVIDIA 栈之外复制这个。这就是取舍——用可移植性换经济性。理解哪些栈选择贡献了差距的哪部分，是这节课的重点。

## 核心概念

### 为什么 FP8 仍是 KV cache 的底线

2026 年的常见错误：假设 NVFP4 到处适用。并非如此。KV cache 需要 FP8（8-bit 浮点），因为它存储的 attention key 和 value 跨越很宽的动态范围。将 KV 量化到 FP4 会导致灾难性精度损失——分布的尾部被截断，attention score 崩塌。FP8 的指数位给了 KV cache 所需的范围。

NVFP4（2025-2026）适用于权重和激活。Microscaling：每个权重块有自己的 scale factor，使小块可以跨越不同动态范围而不丢失 per-tensor scale。对于激活，FP4 能撑住，因为激活在层内范围较小。

典型的 Blackwell 配置：

- 权重：NVFP4（4-bit microscaling）。
- 激活：NVFP4。
- KV cache：FP8。
- Attention 累加器：FP32（softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 特有原语

- **Day-0 FP4 权重**：模型提供方直接发布 FP4 权重；TRT-LLM 无需训后转换即可加载。FP4 不需要 AWQ / GPTQ 步骤。
- **Multi-token prediction (MTP)**：与 EAGLE（Phase 17 · 05）相同的思路，但集成在 TRT-LLM 构建中。
- **分离式 serving**：prefill 和 decode 在不同 GPU 池上，KV cache 通过 NVLink 或 InfiniBand 传输。与 Dynamo（Phase 17 · 20）相同的思路。
- **All-to-all 通信原语**：NVLink 5 将 MoE expert 通信延迟降低 3x（对比 Hopper）。TRT-LLM 的 MoE kernel 针对此优化。
- **NVFP4 + MXFP8 microscaling**：Blackwell Tensor Core 上硬件加速的 scale-factor 处理。

### 你应该记住的数字

- HGX B200 通过 TRT-LLM 在 GPT-OSS-120B 上 $0.02/M tokens。
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）$0.012/M tokens。
- H100 + vLLM ≈ $0.09/M tokens（可比负载）。
- TRT-LLM 三个月更新带来 2.8x 吞吐提升（2026）。
- 每 GPU LLM 吞吐 Blackwell vs Hopper：11-15x。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在每个提交任务上领先。

### FP4 在质量上的实际代价

NVFP4 是激进的。在推理密集型负载（chain-of-thought、数学、长上下文代码生成）上，FP4 权重的退化肉眼可见。Per-block 校准能缓解但不能消除。发布推理模型的团队通常使用 FP8 权重 + FP4 激活作为折中，或者坚持在 H200 上全程 FP8。

规则：在承诺 NVFP4 权重之前，始终在你的评估集上验证任务质量。

### 为什么这是 NVIDIA 锁定决策

TRT-LLM 是 C++ + CUDA + 闭源 kernel。模型需要为特定 GPU SKU 编译。不支持 AMD、Intel、ARM。如果你的基础设施策略是多供应商，TRT-LLM 对 TRT-LLM 服务层来说是不可行的——你仍然可以在混合硬件上用 vLLM 服务。如果你是纯 NVIDIA，7x 的差距值得锁定。

### 2026 实用配方

对于年推理账单 $100M+ 的场景，在 Hopper + vLLM 上运行意味着浪费 7-10x。将成本主导的负载迁移到 Blackwell + TRT-LLM + Dynamo。保留实验层在 H100 + vLLM 上以获得模型迭代速度。在每个 NVFP4 转换的模型上线前验证质量。

### 分离式 serving 的加成

TRT-LLM 的分离式 serving（独立的 prefill 和 decode 池）在 Phase 17 · 20 中深入讲解。在 Blackwell 上，乘数叠加：FP4 权重 × MTP 加速 × 分离式放置 × cache-aware 路由。7x 的数字假设了这个完整栈。

## Use It

`code/main.py` 计算模型在三个栈上的 HBM 占用、decode 吞吐（内存带宽受限区间）和 $/M-tokens：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它来看叠加效应以及每个变化贡献的差距份额。

## Ship It

本课产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定负载、模型大小和年 token 量，判断 Blackwell + TRT-LLM 栈是否值得 NVIDIA 锁定。

## 练习

1. 运行 `code/main.py`。在 120B MoE（30% 活跃参数）上，计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 的内存带宽受限 decode 吞吐。最大跳跃来自哪里？
2. 一个客户每年在 H100 + vLLM 上花 $2M。给定 7x 经济差距，他们需要购买多少 Blackwell GPU 才能在 12 个月内摊销迁移到 TRT-LLM 的成本？
3. NVFP4 权重转换后 MATH 上精度下降 3 分。说出两条恢复路径：一条质量优先（保持 FP8 权重），一条成本优先（用领域内数据校准）。
4. 阅读 MLPerf v6.0 推理结果。哪个任务的 Blackwell-over-Hopper 差距最小？为什么？
5. 计算 405B 模型在 NVFP4 权重 + FP8 KV cache、128k 上下文下所需的 HBM。能放进单个 GB200 NVL72 节点吗？

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| FP8 | "eight-bit float" | 8-bit 浮点；因动态范围需求用于 KV cache 和 attention |
| NVFP4 | "four-bit micro" | NVIDIA 的 4-bit microscaling FP 格式；Blackwell 上用于权重和激活 |
| MXFP8 | "MX eight" | Microscaling FP8 变体；Blackwell Tensor Core 上硬件加速 |
| Day-0 FP4 | "ship FP4 weights" | 模型提供方直接发布 FP4 权重；无需训后转换步骤 |
| MTP | "multi-token prediction" | TRT-LLM 集成的投机解码 draft（Phase 17 · 05） |
| Disaggregated serving | "split prefill/decode" | Prefill 和 decode 在不同 GPU 池上；KV 通过 NVLink/IB 传输 |
| All-to-all | "MoE expert comm" | 将 token 路由到 expert GPU 的通信模式；NVLink 5 降低 3x |
| InferenceX | "SemiAnalysis inference bench" | 2026 年行业认可的每 token 成本 benchmark |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all 和 MoE kernel。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方引擎文档。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 之上的分离式编排。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 数字的 benchmark 套件。
