# Edge Inference — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> 边缘推理的核心瓶颈是内存带宽，而非算力。移动端 DRAM 带宽为 50-90 GB/s；数据中心 HBM3 可达 2-3 TB/s——差距 30-50 倍。Decode 阶段受内存带宽限制，因此这个差距是决定性的。2026 年的格局分为四条路线。Apple M4/A18 Neural Engine 峰值 38 TOPS，统一内存（无 CPU↔NPU 拷贝）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达到 45 TOPS。WebGPU + WebLLM 在 M3 Max 上运行 Llama 3.1 8B (Q4) 约 41 tok/s（约为原生性能的 70-80%）；GitHub 17.6k stars，OpenAI 兼容 API，移动端覆盖率约 70-75%。NVIDIA Jetson Orin Nano Super (8GB) 可运行 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 运行 gpt-oss-20b 约 40 tok/s；Jetson T4000 (JetPack 7.1) 性能为 AGX Orin 的 2 倍。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、chunked prefill——在 CES 2026 上由 Bosch、ThunderSoft、MediaTek 展示。

**类型：** 学习
**语言：** Python（标准库，简易带宽受限 decode 模拟器）
**前置课程：** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 09 (Production Quantization)
**时间：** 约 60 分钟

## 学习目标

- 解释为什么移动端 LLM 推理受内存带宽限制，算力是次要因素。
- 列举四个边缘目标平台（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson）并将每个匹配到对应的使用场景。
- 说出 2026 年 WebGPU 的覆盖缺口（Firefox Android 仍在追赶）以及 Safari iOS 26 的正式发布。
- 为每个目标平台选择量化格式（ANE 用 Core ML INT4 + FP16，Hexagon 用 QNN INT8/INT4，浏览器用 WebGPU Q4，Jetson Thor 用 NVFP4）。

## 问题

客户想要一个端侧聊天机器人：语音优先、默认隐私保护、支持离线。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 运行约 55 tok/s——没问题。在 iPhone 16 Pro 上，同一模型只有 3 tok/s——不行。在搭载 Snapdragon 8 Gen 3 的中端 Android 上，7 tok/s。在 Chrome Android v121+ 上通过 WebGPU 运行，4-8 tok/s 取决于设备。

吞吐量的差异不是移植问题。它等于带宽差距乘以量化格式乘以 NPU 是否可从用户空间访问。2026 年的边缘推理是四个不同的问题，对应四种不同的解决方案。

## 概念

### 带宽才是真正的天花板

Decode 阶段每生成一个 token 都要读取全部权重。一个 7B 模型 Q4 量化后约 3.5 GB。以 50 GB/s 读取 3.5 GB 需要 70 ms——理论上限约 14 tok/s。以 90 GB/s（高端移动 DRAM）上限提升到约 25 tok/s。在这个数字以下，再多的算力也无济于事。

数据中心 HBM3 以 3 TB/s 读取同样的 3.5 GB 只需 1.2 ms——上限为 830 tok/s。同样的模型，同样的权重，不同的内存子系统。

### Apple Neural Engine (M4 / A18)

- 最高 38 TOPS。统一内存（CPU 和 ANE 共享同一内存池）——无拷贝开销。
- 通过 Core ML + `.mlmodel` 编译模型访问，或通过 PyTorch 的 Metal Performance Shaders (MPS) 访问。
- Llama.cpp Metal 后端使用 MPS，不直接使用 ANE；原生 ANE 需要 Core ML 转换。
- 2026 年 iOS 应用的最佳实践路径：Core ML + INT4 权重 + FP16 激活值。

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- 最高 45 TOPS。与 CPU 和 GPU 集成在同一 SoC 中，但内存域独立。
- QNN (Qualcomm Neural Network) SDK 和 AI Hub 提供从 PyTorch/ONNX 的转换。
- Chat templates、Llama 3.2、Phi-3 均作为 AI Hub 上的一等公民提供。

### Intel / AMD NPU (Lunar Lake, Ryzen AI 300)

- 40-50 TOPS。软件生态落后于 Apple/Qualcomm；OpenVINO 在改进但仍属小众。
- 最适合 Windows ARM copilot 应用；在 AMD/Intel 桌面端适合 local-first 场景。

### WebGPU + WebLLM

- 通过 WebGPU compute shaders 在浏览器中运行模型；无需安装。
- Llama 3.1 8B Q4 在 M3 Max 上约 41 tok/s——约为同后端原生性能的 70-80%。
- WebLLM GitHub 17.6k stars；OpenAI 兼容 JS API；Apache 2.0 许可。
- 2026 年覆盖率：Chrome Android v121+、Safari iOS 26 GA、Firefox Android 仍在追赶。整体移动端覆盖率约 70-75%。

### NVIDIA Jetson 系列

- Orin Nano Super (8GB)：可运行 Llama 3.2 3B、Phi-3，吞吐量不错。
- AGX Orin：通过 vLLM 运行 gpt-oss-20b 约 40 tok/s。
- Thor / T4000 (JetPack 7.1)：性能为 AGX Orin 的 2 倍，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM (2026) 支持 EAGLE-3 投机解码、NVFP4 权重、chunked prefill——将数据中心优化移植到边缘。

### 各目标平台的量化选择

| 目标平台 | 格式 | 备注 |
|--------|--------|-------|
| Apple ANE | INT4 weights + FP16 activations | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | 使用 `mlc_llm convert_weight` + 编译后的 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 受内存带宽限制 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘端的长上下文陷阱

Llama 3.1 的 128K 上下文是数据中心特性。在 8 GB RAM 的手机上，4 GB 模型 + 32K token 的 2 GB KV cache + 系统开销 = OOM。边缘部署将上下文控制在 4K-8K，除非接受激进的 KV 量化（Q4 KV）。

### 语音是杀手级应用

语音 agent 对延迟敏感（首 token < 500 ms）。本地推理完全消除网络延迟。结合语音转文字（Whisper Turbo 变体可在边缘运行），边缘推理就构成了生产级的语音闭环。

### 需要记住的数字

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 约 41 tok/s。
- AGX Orin：通过 vLLM 运行 gpt-oss-20b 约 40 tok/s。
- 数据中心与边缘的带宽差距：30-50 倍。
- WebGPU 移动端覆盖率：约 70-75%（Firefox Android 落后）。

## 动手实践

`code/main.py` 基于带宽受限的数学模型，计算各边缘目标平台的理论 decode 吞吐量上限。与实测基准对比，突出带宽（而非算力）才是瓶颈。

## 交付产出

本课程产出 `outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/浏览器/Jetson）、模型和延迟/内存预算，选择量化格式和转换流水线。

## 练习

1. 运行 `code/main.py`。对于 Snapdragon 8 Gen 3（带宽约 77 GB/s）上的 7B Q4 模型，计算 decode 上限。与实测 6-8 tok/s 对比——运行时效率如何？
2. Android 上的 WebGPU 需要 Chrome v121+。为旧版浏览器设计一个回退方案——通过相同的 OpenAI 兼容 API 走服务端。
3. 你的 iOS 应用需要 4K 上下文流式输出。哪种模型/格式组合能让 iPhone 16 的活跃内存保持在 4 GB 以下？
4. Jetson AGX Orin 运行 gpt-oss-20b 达 40 tok/s。Jetson Nano 只能放 3B 模型。如果你的产品同时面向两者，如何统一推理栈？
5. 论证"WebLLM 在 2026 年是否已达到生产就绪"。引用覆盖率、性能和 Firefox Android 缺口。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| ANE | "Apple neural engine" | M 系列和 A 系列中的端侧 NPU；统一内存 |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | "浏览器 GPU" | W3C 标准化的浏览器 GPU API；Chrome/Safari 2026 |
| WebLLM | "浏览器 LLM 运行时" | MLC-LLM 项目；Apache 2.0；OpenAI 兼容 JS |
| Jetson | "NVIDIA edge" | Orin Nano / AGX / Thor / T4000 系列 |
| TRT Edge-LLM | "edge TensorRT" | 2026 年 TensorRT-LLM 的边缘移植版；EAGLE-3 + NVFP4 |
| Unified memory | "共享内存池" | CPU 和 NPU 访问同一 RAM；无拷贝开销 |
| Bandwidth-bound | "内存受限" | Decode 受读取权重的字节/秒限制 |
| Core ML | "Apple 转换框架" | Apple 的 ANE 原生模型框架 |
| QNN | "Qualcomm 技术栈" | Qualcomm Neural Network SDK |

## 延伸阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 全景与基准测试。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 年边缘移植公告。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 设计与基准测试。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE 原生转换。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Hexagon 预转换模型。
