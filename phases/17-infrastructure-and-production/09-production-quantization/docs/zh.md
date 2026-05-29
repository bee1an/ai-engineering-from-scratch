# 生产环境量化 — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

> 量化格式不是一个通用选择——它是硬件、推理引擎和工作负载的函数。GGUF Q4_K_M 或 Q5_K_M 主导 CPU 和边缘场景，通过 llama.cpp 和 Ollama 交付。GPTQ 在 vLLM 中需要多 LoRA 共享同一基座模型时胜出。AWQ 配合 Marlin-AWQ kernel 在 7B 级模型上达到约 741 tok/s，且在 INT4 中 Pass@1 最优——这是 2026 年数据中心生产的默认选择。FP8 在 Hopper、Ada 和 Blackwell 上保持中间地带——近乎无损且广泛支持。NVFP4 和 MXFP4（Blackwell microscaling）激进且需要逐 block 验证。两个陷阱常坑团队：校准数据集必须匹配部署领域，以及 KV cache 与权重量化是分开的——AWQ 那句"我的模型现在只有 4 GB"忽略了生产 batch size 下 10-30 GB 的 KV cache。

**Type:** Learn
**Languages:** Python (stdlib, toy memory and throughput comparison across formats)
**Prerequisites:** Phase 10 · 13 (Quantization foundations), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## 学习目标

- 列举 2026 年六种生产量化格式及其各自的最佳适用场景。
- 根据硬件（CPU vs GPU、Hopper vs Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（日常对话、推理、多 LoRA）选择格式。
- 计算所选格式节省的权重内存以及未被压缩的 KV cache 大小。
- 说明校准数据集陷阱如何导致量化模型在领域流量上质量下降。

## 问题

量化减少内存和 HBM 带宽，这正是 decode 阶段所需要的。一个 FP16 的 70B 模型权重为 140 GB。将权重量化到 INT4（AWQ 或 GPTQ），模型变为 35 GB——可以放进一块 H100 并留出 KV cache 空间，这很重要，因为在 128 并发序列、2k 上下文的情况下，仅 KV cache 就需要 20-30 GB。

但量化不是免费的。激进量化会降低质量，尤其在推理密集型任务上。不同格式适配不同引擎。不同硬件原生支持不同精度。2026 年的格式动物园是真实存在的，你不能照搬别人的选择——必须根据自己的技术栈来选。

## 概念

### 六种格式

| 格式 | 位数 | 最佳场景 | 引擎 |
|------|------|----------|------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘、笔记本 | llama.cpp, Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA | vLLM, TGI |
| AWQ | 4 | 数据中心 GPU 生产 | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell 多用户 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户 | TRT-LLM |

### GGUF — CPU/边缘默认选择

GGUF 是一种文件格式，本身不是量化方案——它将 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包在一个容器中。Q4_K_M 和 Q5_K_M 是生产默认值——在 4-5 bit 下接近 BF16 质量。CPU 或边缘推理的最佳选择，因为 llama.cpp 是目前最快的 CPU 推理引擎。

在 vLLM 中的吞吐量损失：7B 模型约 93 tok/s——该格式未针对 GPU kernel 优化。部署目标是 CPU/边缘时用 GGUF，否则不用。

### GPTQ — vLLM 中的多 LoRA

GPTQ 是一种带校准步骤的训练后量化算法。Marlin kernel 使其在 GPU 上很快（相比非 Marlin GPTQ 有 2.6x 加速）。7B 模型约 712 tok/s。

独特优势：GPTQ-Int4 在 vLLM 中支持 LoRA adapter。如果你在服务一个基座模型加 10-50 个微调变体（每个作为 LoRA），GPTQ 是你的路径。截至 2026 年初，NVFP4 尚不支持 LoRA。

### AWQ — 数据中心 GPU 默认选择

Activation-aware Weight Quantization。在量化过程中保护约 1% 最显著的权重。Marlin-AWQ kernel：相比朴素实现 10.9x 加速。7B 模型约 741 tok/s，INT4 格式中 Pass@1 最优。

新的 GPU 推理服务选 AWQ，除非你需要多 LoRA（选 GPTQ）或激进的 Blackwell FP4（选 NVFP4）。

### FP8 — 可靠的中间选择

8-bit 浮点。近乎无损。广泛支持。Hopper Tensor Core 原生加速 FP8。Blackwell 继承。当质量不可妥协时（推理、医疗、代码生成），FP8 是 2026 年的安全默认选择。内存节省是 INT4 的一半，但质量风险远低。

### MXFP4 / NVFP4 — Blackwell 激进选择

Microscaling FP4。每个权重 block 有自己的 scale factor。激进但在 Blackwell Tensor Core 上有硬件加速。相比 FP8 每 token 字节数减半——Phase 17 · 07 中的经济优势。

注意事项：
- 截至 2026 年初尚不支持 LoRA。
- 在推理密集型工作负载上质量下降明显。
- 需要在你的 eval set 上逐模型验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集——通常是 C4 或 WikiText。对于领域模型（代码、医疗、法律），用通用网页文本校准会让算法在保护哪些权重上做出错误决策。HumanEval 上的 Pass@1 可能下降数个百分点。

修复方法：用领域内数据校准。几百条领域样本通常就够了。上线前在 eval set 上测试。

### KV cache 陷阱

AWQ 将权重压缩到 4 bit。KV cache 是独立的，保持在 FP16/FP8。对于使用 AWQ 的 70B 模型：

- 权重：约 35 GB（从 140 GB 的 INT4）。
- KV cache（128 并发 × 2k 上下文）：约 20 GB。
- 激活值：约 5 GB。
- 总计：约 60 GB——可以放进 H100 80GB。

天真地说"我把模型量化到 4 GB 了"忽略了另外 30-50 GB。需要整体规划 HBM 预算。

另外，KV cache 量化（FP8 KV 或 INT8 KV）是与权重量化不同的选择，有自己的权衡——它直接影响 attention 精度，不是免费的优化。

### AWQ INT4 对推理任务有害

Chain-of-thought、数学、长上下文代码生成——这些在激进量化下质量明显下降。AWQ INT4 在 MATH 上损失约 3-5 分。对于推理密集型工作负载，用 FP8 或 BF16；接受内存成本。

### 2026 选择指南

- CPU/边缘服务：GGUF Q4_K_M。结束。
- GPU 服务，日常对话，无 LoRA：AWQ。
- GPU 服务，多 LoRA：GPTQ with Marlin。
- 推理工作负载：FP8。
- Blackwell 数据中心，质量已验证：NVFP4 + FP8 KV。
- 不确定：在每个候选格式上跑 1,000 样本 eval。

## Use It

`code/main.py` 计算不同模型大小下六种格式的内存占用（权重 + KV + 激活值）和相对吞吐量。展示 KV cache 在哪里占主导、权重压缩在哪里有收益、以及 FP8 在哪里是安全选择。

## Ship It

本课程产出 `outputs/skill-quantization-picker.md`。给定硬件、模型大小、工作负载类型和质量容忍度，选择格式并生成校准/验证计划。

## 练习

1. 运行 `code/main.py`。对于 70B 模型在 128 并发、2k 上下文下，计算每种格式的总 HBM 占用。哪种格式能放进一块 H100 80GB？
2. 你有一个 7B 代码模型。选择一种格式并说明理由。如果你对质量容忍度判断错误，恢复路径是什么？
3. 计算为医疗领域模型校准 AWQ 所需的校准数据集大小。为什么更多数据不一定更好？
4. 阅读 Marlin-AWQ kernel 论文或发布说明。用三句话解释为什么 AWQ 在 7B 上达到 741 tok/s 而原始 GPTQ 约 712 tok/s。
5. 什么时候将 AWQ 权重与 FP8 KV cache 组合使用比保持 KV 在 BF16 更合理？

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|----------|----------|
| GGUF | "llama.cpp 格式" | 打包 K-quant 变体的文件格式；CPU/边缘默认 |
| Q4_K_M | "Q4 K M" | 4-bit K-quant medium；生产 GGUF 默认值 |
| GPTQ | "gee pee tee q" | 带校准的训练后 INT4；在 vLLM 中支持 LoRA |
| AWQ | "a w q" | Activation-aware INT4；Marlin kernel；INT4 中 Pass@1 最优 |
| Marlin kernels | "快速 INT4 kernel" | Hopper 上 INT4 的定制 CUDA kernel；10x 加速 |
| FP8 | "八位浮点" | Hopper/Ada/Blackwell 上的安全精度默认值 |
| MXFP4 / NVFP4 | "microscaling four" | Blackwell 4-bit FP，带逐 block scale factor |
| Calibration dataset | "校准数据" | 用于选择量化参数的输入文本；必须匹配领域 |
| KV cache quantization | "KV INT8" | 与权重分开的选择；影响 attention 精度 |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 对比基准测试。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 各格式吞吐量数据。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 逐格式选择指南。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式和参数。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — AWQ 原始论文。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — GPTQ 原始论文。
