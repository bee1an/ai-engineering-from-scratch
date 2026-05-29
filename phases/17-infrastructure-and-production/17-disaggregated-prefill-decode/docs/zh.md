# 预填充/解码分离 — NVIDIA Dynamo 与 llm-d

> Prefill 是计算密集型；decode 是内存带宽密集型。在同一块 GPU 上运行两者会浪费其中一种资源。分离架构将它们拆分到独立的 GPU 池，通过 NIXL（RDMA/InfiniBand 或 TCP 回退）在池间传输 KV cache。NVIDIA Dynamo（GTC 2025 发布，1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上——其 Planner Profiler + SLA Planner 自动匹配 prefill:decode 比例以满足 SLO。NVIDIA 公布的吞吐量提升大致在这个范围——developer.nvidia.com (2025-06) 显示 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 上中等延迟区间约 6 倍提升，Dynamo 产品页（developer.nvidia.com，未标注日期）宣传 GB300 NVL72 + Dynamo 相比 Hopper 可达 50 倍 MoE 吞吐。"30 倍"数字是社区对完整 Blackwell + Dynamo + DeepSeek-R1 报告的汇总；我们未找到单一主要来源明确声称 30 倍，因此将其视为方向性声明。llm-d（Red Hat + AWS）是 Kubernetes 原生的：prefill / decode / router 作为独立 Service，带按角色的 HPA。llm-d 0.5 新增分层 KV offloading、缓存感知 LoRA 路由、UCCL 网络、缩容到零。经济性：多个客户披露的内部汇总表明，在恒定 SLA 下从共置 serving 切换到 Dynamo 分离架构，$2M 级推理支出可节省 30-40%（即 $600-800K/年）；具体的 $2M→$600-800K 数字是内部综合值，非单一已发布案例——将其作为数量级锚点而非引用来源。短 prompt（<512 token，短输出）不值得传输开销。

**Type:** Learn
**Languages:** Python (stdlib, toy disaggregated-vs-colocated simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 08 (Inference Metrics)
**Time:** ~75 minutes

## 学习目标

- 解释为什么 prefill 和 decode 有不同的最优 GPU 分配，并量化共置下的浪费。
- 画出分离架构图：prefill 池、decode 池、通过 NIXL 的 KV 传输、路由器。
- 指出分离不划算的条件（短 prompt、短输出）。
- 区分 NVIDIA Dynamo（栈上层编排器）和 llm-d（Kubernetes 原生），并将各自匹配到运维场景。

## 问题

你在 8 块 H100 上运行 Llama 3.3 70B。在混合工作负载（长 prompt + 短输出）下，GPU 在 decode 阶段空闲，因为大部分计算花在了 prefill 上。在另一种工作负载（短 prompt + 长输出）下，情况相反。共置 prefill + decode 意味着两者都过度配置。

预算影响：20-40% 的 GPU 时间浪费在错误的资源上。你在用 H100 的计算能力跑内存带宽密集的 decode，或者用 H100 的 HBM 带宽跑计算密集的 prefill。两者都是昂贵的浪费。

分离架构将 prefill 和 decode 拆分到各自按瓶颈调整大小的独立池。KV cache 通过高带宽互联从 prefill 池传输到 decode 池。

## 核心概念

### 为什么瓶颈不同

**Prefill** — 在一次前向传播中对完整输入 prompt 运行 transformer。矩阵乘法主导；计算密集型。H100 FP8 提供约 2000 TFLOPS 的有效吞吐。批处理效率好——一次前向处理多个 token。

**Decode** — 每次生成一个 token，每次迭代读取完整权重。内存带宽密集型。HBM3 提供约 3 TB/s。批处理效率仅在高并发时好——权重读取在 batch 中摊销。

共置：你买的 GPU 需要同时擅长两者。H100 两者都行但成本一样。在规模化时，你希望 prefill 池用 H100 / 计算密集型；decode 池用 H200 / 内存密集型，或使用激进量化。

### 架构

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的节点间传输。有 RDMA/InfiniBand 时使用，否则 TCP 回退。传输延迟是真实的——70B FP8 模型 4K-token prompt 的 KV cache 传输通常需要 20-80 ms。这就是为什么短 prompt 不值得分离：传输税超过了节省。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA）：
- 位于 vLLM、SGLang、TRT-LLM 之上作为编排器。
- Planner Profiler 测量工作负载，SLA Planner 自动配置 prefill:decode 比例。
- Rust 核心，Python 可扩展。
- 吞吐量提升：NVIDIA 报告 DeepSeek-R1 MoE 在 GB200 NVL72 + Dynamo 上中等延迟区间 6 倍（developer.nvidia.com, 2025-06）；社区关于完整 Blackwell + Dynamo + DeepSeek-R1 栈"高达 30 倍"的报告缺乏单一主要来源，应视为方向性。
- GB300 NVL72 + Dynamo：相比 Hopper 高达 50 倍 MoE 吞吐（developer.nvidia.com，未标注日期）。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- Prefill / decode / router 作为独立 Kubernetes Service。
- 按角色 HPA，信号为队列深度（prefill）/ KV 利用率（decode）。
- `topologyConstraint packDomain: rack` 将 prefill+decode 集群打包在同一机架以实现高带宽 KV 传输。
- llm-d 0.5 (2026)：分层 KV offloading、缓存感知 LoRA 路由、UCCL 网络、缩容到零。

如果你想要托管的栈上层编排器，用 Dynamo。如果你想要 Kubernetes 原生原语并且投入 CNCF 生态，用 llm-d。

### 经济性

内部综合值（非单一已发布案例——数量级锚点）：

- $2M/年推理支出，共置 serving。
- 切换到 Dynamo 分离架构。
- 相同请求量，相同 P99 延迟 SLA。
- 报告节省：$600K-$800K/年（30-40% 降低）。
- 无新硬件。

我们从多个客户披露中综合此数字，而非单一可引用案例；最接近的已发布数据点是 Baseten 的 2 倍更快 TTFT / 61% 更高吞吐（baseten.co, 2025-10），以及 VAST + CoreWeave 在 40-60% KV 命中率下预测的 60-130% 更多 tokens/$（vastdata.com, 2025-12）。节省来自对每个池的正确调整；prefill 密集型工作负载（RAG 8K+ 前缀）比均衡工作负载受益更多。

### 何时不该分离

- Prompt < 512 token 且输出 < 200 token：传输税超过收益。
- 小集群（< 4 GPU）：池多样性不足。
- 团队无法运维两个带按角色扩缩的 GPU 池：Dynamo 有帮助但并非简单。
- 无 RDMA 网络：TCP 传输税更重。

### 路由器与 Phase 17 · 11 集成

分离架构的路由器是 KV-cache 感知的（Phase 17 · 11）。请求落在持有其前缀的 decode 池上——如果没有匹配，则走 prefill → decode 流程。命中率和分离架构复合——缓存感知路由器决定是否需要新的 prefill。

### MoE 在 Blackwell 上才是真正的数字

GB300 NVL72 + Dynamo 显示相比 Hopper 基线 50 倍 MoE 吞吐。MoE 专家路由在 prefill 时计算密集但在 decode 时内存密集（专家缓存），所以分离是双赢。2026 年前沿模型 serving 以 MoE 为主（DeepSeek-V3，未来 GPT-5 变体）。

### 需要记住的数字

基准数字会漂移——NVIDIA 和推理栈每季度发布更新结果。引用前请重新核实。

- DeepSeek-R1 在 GB200 NVL72 + Dynamo 上：中等延迟区间约 6 倍吞吐（developer.nvidia.com, 2025-06）；社区"高达 30 倍"声明是方向性汇总，无单一主要来源。
- GB300 NVL72 + Dynamo：相比 Hopper 高达 50 倍 MoE 吞吐（developer.nvidia.com，未标注日期）。
- 节省锚点（内部综合值，非单一案例）：$2M 年度支出在恒定 SLA 下节省 $600-800K/年。
- 分离阈值：prompt >512 token + 输出 >200 token。
- 通过 NIXL 的 KV 传输：70B FP8 4K-prompt KV 需 20-80 ms。

## Use It

`code/main.py` 模拟共置 vs 分离 serving。报告吞吐量、每请求成本和 prompt 长度交叉点。

## Ship It

本课产出 `outputs/skill-disaggregation-decider.md`。给定工作负载和集群，决定是否分离。

## 练习

1. 运行 `code/main.py`。在什么 prompt 长度时分离优于共置？
2. 为一个 RAG 服务设计 prefill 池和 decode 池，P99 前缀长度 8K，输出 300。
3. Dynamo vs llm-d：为一个纯 Kubernetes 团队（无 Python 运行时偏好）选择一个。
4. 计算 KV 传输成本：70B FP8 4K prefill = ~500 MB KV。RDMA 100 GB/s 下传输 = 5 ms。TCP 10 GB/s = 50 ms。哪个对你的 SLA 重要？
5. MoE 专家路由改变了 KV 访问模式。分离架构在每个 token 激活不同专家的 MoE 下如何表现？

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| Disaggregated serving | "拆分 prefill/decode" | 每个阶段使用独立 GPU 池 |
| NIXL | "NVIDIA 传输" | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | "编排器" | 位于 vLLM/SGLang/TRT-LLM 之上的协调器 |
| llm-d | "Kubernetes 原生" | Red Hat + AWS K8s 分离栈 |
| Planner Profiler | "Dynamo 自动配置" | 测量工作负载，配置池比例 |
| SLA Planner | "Dynamo 策略" | 自动匹配 prefill:decode 以满足 SLO |
| `packDomain: rack` | "llm-d 拓扑" | 将 prefill+decode 打包在同一机架以加速 KV |
| UCCL | "统一集合通信" | llm-d 0.5 网络层，支持缩容到零 |
| MoE expert routing | "每 token 专家" | DeepSeek-V3 模式；分离有帮助 |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
