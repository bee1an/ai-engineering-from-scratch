# 自托管推理引擎选型 — llama.cpp、Ollama、TGI、vLLM、SGLang

> 2026 年四个引擎主导自托管推理。根据硬件、规模和生态系统选择。**llama.cpp** 在 CPU 上最快——最广泛的模型支持，对量化和线程的完全控制。**Ollama** 是开发笔记本一键安装，比 llama.cpp 慢约 15-30%（Go + CGo + HTTP 序列化），生产级负载下 3 倍吞吐差距。**TGI 于 2025 年 12 月 11 日进入维护模式**——仅修 bug，原始吞吐比 vLLM 慢约 10% 但历史上可观测性和 HF 生态集成最佳。维护状态使其成为长期风险——SGLang 或 vLLM 是新项目更安全的默认选择。**vLLM** 是通用生产默认——v0.15.1（2026 年 2 月）增加 PyTorch 2.10、RTX Blackwell SM120、H200 优化。**SGLang** 是 agentic 多轮/前缀密集型专家——400,000+ GPU 在生产中（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬件约束：仅 CPU → 只有 llama.cpp。AMD / 非 NVIDIA → 只有 vLLM（TRT-LLM 锁定 NVIDIA）。2026 年流水线模式：dev = Ollama，staging = llama.cpp，prod = vLLM 或 SGLang。全程使用相同 GGUF/HF 权重。

**Type:** Learn
**Languages:** Python (stdlib, engine-decision tree walker)
**Prerequisites:** All Phase 17 lessons covering engines (04, 06, 07, 09, 18)
**Time:** ~45 minutes

## 学习目标

- 根据硬件（CPU / AMD / NVIDIA Hopper / Blackwell）、规模（1 用户 / 100 / 10,000）和工作负载（通用聊天 / agent / 长上下文）选择引擎。
- 说明 2026 年 TGI 维护模式状态（2025 年 12 月 11 日）以及为什么它使新项目偏向 vLLM 或 SGLang。
- 描述使用相同 GGUF 或 HF 权重贯穿的 dev/staging/prod 流水线。
- 解释为什么"仅 CPU"强制 llama.cpp 而"AMD"排除 TRT-LLM。

## 问题

你的团队启动一个新的自托管 LLM 项目。一个工程师说 Ollama，另一个说 vLLM，第三个说"TGI 不是开箱即用吗？"三个在不同上下文中都对。没有一个对所有场景都对。

在 2026 年选择树很重要：硬件第一，规模第二，工作负载第三。还有一个特定的 2025 年事件——TGI 于 12 月 11 日进入维护模式——改变了新项目的默认选择。

## 核心概念

### 五个引擎

| 引擎 | 最适合 | 备注 |
|------|--------|------|
| **llama.cpp** | CPU / 边缘 / 最少依赖 / 最广模型支持 | CPU 上最快，完全控制 |
| **Ollama** | 开发笔记本，单用户，一键安装 | 比 llama.cpp 慢 15-30%；生产吞吐差距 3 倍 |
| **TGI** | HF 生态，受监管行业 | **2025 年 12 月 11 日维护模式** |
| **vLLM** | 通用生产，100+ 用户 | 广泛的生产默认；v0.15.1 2026 年 2 月 |
| **SGLang** | Agentic 多轮，前缀密集型工作负载 | 400,000+ GPU 在生产中 |

### 硬件优先决策

**仅 CPU** → llama.cpp。Ollama 也行但更慢。没有其他引擎在 CPU 上有竞争力。

**AMD GPU** → vLLM（AMD ROCm 支持）。SGLang 也行。TRT-LLM 锁定 NVIDIA，排除。

**NVIDIA Hopper (H100 / H200)** → vLLM 或 SGLang 或 TRT-LLM。三者都是顶级。

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM 是吞吐领先者（Phase 17 · 07）。vLLM 和 SGLang 紧随其后。

**Apple Silicon (M 系列)** → llama.cpp (Metal)。Ollama 封装了这个。

### 规模第二决策

**1 用户 / 本地开发** → Ollama。一条命令，秒级首 token。

**10-100 用户 / 小团队** → vLLM 单 GPU。

**100-10k 用户 / 生产** → vLLM production-stack（Phase 17 · 18）或 SGLang。

**10k+ 用户 / 企业** → vLLM production-stack + 分离架构（Phase 17 · 17）+ LMCache（Phase 17 · 18）。

### 工作负载第三决策

**通用聊天 / Q&A** → vLLM 作为广泛默认胜出。

**Agentic 多轮（工具、规划、记忆）** → SGLang 的 RadixAttention（Phase 17 · 06）主导。

**RAG 高前缀复用** → SGLang。

**代码生成** → vLLM 可以；SGLang 在缓存上略好。

**长上下文 (128K+)** → vLLM + chunked prefill；SGLang + tiered KV。

### TGI 维护陷阱

Hugging Face TGI 于 2025 年 12 月 11 日进入维护模式——此后仅修 bug。历史上：顶级可观测性，最佳 HF 生态集成（model cards、安全工具），原始吞吐略落后于 vLLM。

2026 年新项目：默认远离 TGI。现有 TGI 部署可以继续但应最终迁移。SGLang 和 vLLM 是更安全的默认选择。

### 流水线模式

Dev (Ollama) → staging (llama.cpp) → prod (vLLM)。全程使用相同 GGUF 或 HF 权重。工程师在笔记本上快速迭代；staging 镜像生产量化；prod 是 serving 目标。

### Ollama 注意事项

Ollama 适合开发。不适合共享生产：Go HTTP 序列化增加开销，并发管理比 vLLM 简单，OpenTelemetry 支持滞后。在 Ollama 擅长的地方用它——一个用户，一条命令——共享时切换到 vLLM。

### 自托管 vs 托管是独立决策

Phase 17 · 01（托管超大规模）、· 02（推理平台）覆盖托管。本课假设你已经决定自托管。自托管的理由：数据驻留、自定义微调、规模化总拥有成本、托管平台上没有的领域模型。

### 需要记住的数字

- TGI 维护模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 支持。
- SGLang 生产规模：400,000+ GPU。
- Ollama 吞吐差距 vs llama.cpp：慢 15-30%；生产负载下 3 倍。

## Use It

`code/main.py` 是一个决策树遍历器：给定硬件 + 规模 + 工作负载，选择引擎并解释原因。

## Ship It

本课产出 `outputs/skill-engine-picker.md`。给定约束，选择引擎并写出迁移计划。

## 练习

1. 用你的硬件/规模/工作负载运行 `code/main.py`。输出与你的直觉一致吗？
2. 你的基础设施是 12 块 H100 和 8 块 MI300X AMD。什么引擎？为什么 TRT-LLM 不行？
3. 一个团队想在 2026 年用 TGI 因为"这是我们熟悉的"。论证迁移理由。
4. Ollama 开发到 vLLM 生产：量化、配置和可观测性有什么变化？
5. RAG 产品，P99 前缀长度 8K，跨租户高复用。选择引擎并与 Phase 17 · 11 + 18 叠加。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| llama.cpp | "CPU 那个" | 最广模型支持，CPU 上最快 |
| Ollama | "笔记本那个" | 一键安装，开发级吞吐 |
| TGI | "HF 的 serving" | 2025 年 12 月起维护模式 |
| vLLM | "默认选择" | 2026 年广泛的生产基线 |
| SGLang | "agentic 那个" | 前缀密集型，RadixAttention |
| TRT-LLM | "NVIDIA 锁定" | Blackwell 吞吐领先者，仅 NVIDIA |
| GGUF | "llama.cpp 格式" | 捆绑 K-quant 变体 |
| Production-stack | "vLLM K8s" | Phase 17 · 18 参考部署 |
| Pipeline pattern | "dev→stage→prod" | Ollama → llama.cpp → vLLM 使用相同权重 |

## 延伸阅读

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — release notes.
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
