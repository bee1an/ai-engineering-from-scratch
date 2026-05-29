# Serverless LLM 冷启动缓解

> 一个 20 GB 的模型镜像从冷启动到可服务需要 5-10 分钟（7B）到 20+ 分钟（70B）。在真正的 serverless 世界里，这不是预热——这是故障。缓解措施在五个层面运作：预置节点镜像（AWS 上的 Bottlerocket，双卷架构）、模型流式加载（NVIDIA Run:ai Model Streamer，vLLM 原生集成）、GPU 内存快照（Modal checkpoint，最高 10x 更快重启）、warm pool（`min_workers=1`）、分层加载（ServerlessLLM 的 NVMe→DRAM→HBM 管线，10-200x 延迟降低），以及迁移输入 token（KB）而非 KV cache（GB）的热迁移。Modal 公布 2-4s 冷启动作为下限；Baseten 默认 5-10s，预热后亚秒级。本课程教你测量、预算和叠加这五个层面。

**Type:** Learn
**Languages:** Python (stdlib, toy cold-start path simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~60 minutes

## 学习目标

- 列举冷启动缓解的五个层面，并在每个层面命名一个工具或模式。
- 将 70B 模型的总冷启动时间计算为（节点供给）+（权重下载）+（权重加载到 HBM）+（引擎初始化）之和。
- 解释为什么热迁移传输输入 token（KB）而非 KV cache（GB），以及代价是什么（重新计算）。
- 说明 warm pool 的权衡（为空闲 GPU 付费还是接受冷启动尾延迟），以及 `min_workers > 0` 变为强制的 SLA 阈值。

## 问题

你的 serverless LLM endpoint 在夜间缩容到零。早上 8 点流量激增。第一个请求等待期间：

1. Karpenter 供给一个 GPU 节点：45-60s。
2. 容器拉取一个 30 GB 的带权重镜像：120-300s。
3. 引擎将权重加载到 HBM：45-120s，取决于模型大小和存储速度。
4. vLLM 或 TRT-LLM 初始化 CUDA graph、KV cache pool、tokenizer：10-30s。

总计：220-510s（大约 3-8 分钟）才能返回第一个 token。你的 SLA 是 2s。你部署了 warm pool（`min_workers=1`），问题似乎消失了——但现在你为一个空闲 GPU 24x7 付费。如果你的服务有 5 个产品各保持一个 warm 副本，那就是 5 × 24 × 30 = 3,600 GPU-hours/月，无论是否有用户调用。

冷启动缓解就是如何保持 serverless 经济性的同时逼近 always-on 的延迟。

## 概念

### Layer 1 — 预置节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的双卷架构将 OS 与数据分离。对数据卷做快照，其中容器镜像已预拉取；在 `EC2NodeClass` 中引用快照 ID。新节点启动时权重已在本地 NVMe 上——步骤 2 和部分步骤 3 消失。与 Karpenter 原生配合。典型节省：大模型每次冷启动 2-4 分钟。

GCP 等价方案：预烘焙容器层的自定义 VM 镜像。Azure：相同模式的托管磁盘快照。

### Layer 2 — 模型流式加载（Run:ai Model Streamer）

不再等整个文件加载完才响应第一个请求，而是逐层将权重流式传入 GPU 内存，第一个 transformer block 驻留后即开始处理。NVIDIA Run:ai Model Streamer 在 vLLM 2026 中原生集成。支持 S3、GCS 和本地 NVMe。通过将 I/O 与计算设置重叠，大模型的权重加载时间大约减半。

### Layer 3 — GPU 内存快照（Modal）

Modal 在首次加载后对 GPU 状态（权重、CUDA graph、KV cache 区域）做 checkpoint。后续重启直接反序列化到 HBM——比重新初始化快 10x。这是最接近"2 秒启动一个 warm GPU"的方案。权衡：快照绑定 GPU 拓扑，如果 Karpenter 将你迁移到不同 SKU，需要重新做 checkpoint。

### Layer 4 — warm pool（min_workers=1）

最简单的缓解：始终保持一个副本就绪。成本是一个 GPU 的小时费率 24x7。对小模型来说算术很残酷（你付 $0.85-$1.50/hr 来避免 30s 冷启动），对大模型则划算（付 $4/hr 来避免 5 分钟冷启动）。warm pool 变为强制的 SLA 阈值：通常是 70B+ 模型的 TTFT P99 < 60s。

### Layer 5 — 分层加载（ServerlessLLM）

ServerlessLLM 将存储视为层级：NVMe（快但大）、DRAM（中等但分层）、HBM（小但即时）。权重预加载到 DRAM；按需加载到 HBM。论文报告相比朴素 disk-to-HBM，冷加载延迟降低 10-200x。生产采用尚早但已有 vLLM 集成。

### Layer 6 — 热迁移（附加模式）

当节点不可用时（spot 回收、节点排空），传统模式是冷启动另一个副本并排空请求队列。热迁移将输入 token（KB 级）移动到已加载模型的目标节点，在目标节点重新计算 KV cache。重新计算比通过网络传输 GB 级 KV cache 更便宜。适用于解耦部署。

### warm pool 数学

对于 P99 TTFT SLA 为 2s 的服务，问题不是"要不要 warm pool"，而是"多少个 warm 副本，哪些路径需要"。

- 高价值交互路径（实时聊天、语音 agent）：`min_workers=1-2`。
- 后台批处理路径（夜间分类）：接受缩容到零，5-10 分钟冷启动可容忍。
- 高级套餐：每租户 `min_workers` 配专用容量。

### 优化前先测量

70B 模型在全新节点上的冷启动解剖（示意）：

| 阶段 | 时间 | 缓解措施 |
|------|------|----------|
| 节点供给 | 50s | Bottlerocket + 预置镜像，warm pool |
| 镜像拉取 | 180s | 预置数据卷（消除） |
| 权重到 HBM | 75s | Model streamer（减半）；GPU 快照（消除） |
| 引擎初始化 | 20s | 持久化 CUDA graph cache |
| 首次前向 | 3s | 最小固有延迟 |
| **总冷启动** | **328s** | |
| **叠加缓解后** | **~15s** | 22x 降低 |

### 你应该记住的数字

- Modal 冷启动：2-4s（使用 GPU 快照）。
- Baseten 默认冷启动：5-10s；预热后亚秒级。
- 原始 70B 冷启动：3-8 分钟。
- Run:ai Model Streamer：约 2x 权重加载加速。
- ServerlessLLM 分层加载：10-200x 延迟降低（论文数据）。

## Use It

`code/main.py` 模拟有无各项缓解措施的冷启动路径。报告总冷启动时间、warm pool 成本，以及 warm pool 自身回本的盈亏平衡请求率。

## Ship It

本课程产出 `outputs/skill-cold-start-planner.md`。给定 SLA、模型大小和流量形态，选择要叠加哪些缓解措施。

## 练习

1. 运行 `code/main.py`。计算 warm 副本比在 SLO 处因冷启动导致额外请求丢弃更便宜的盈亏平衡请求率。
2. 你部署一个 13B 模型，P99 TTFT SLA 为 3s。选择能达标的最小缓解栈（最少层数）。
3. Bottlerocket 预置消除了镜像拉取，但权重仍需从快照加载到 HBM。如果快照支持的 NVMe 读取速度为 7 GB/s，计算 70B 模型的实际耗时。
4. 你的 serverless 提供商提供 GPU 快照（Modal），但你的团队拒绝因为"快照泄露 PII"。论证双方——实际风险是什么，缓解措施是什么（临时快照、加密、命名空间隔离）？
5. 设计一个分层 warm pool 策略：付费用户、试用用户和批处理工作负载各需要多少 warm 副本？展示计算过程。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|----------|----------|
| Cold start | "大停顿" | 从请求到全新副本返回第一个 token 的时间 |
| Warm pool | "always-on 最小值" | `min_workers >= 1` 保持至少一个副本就绪 |
| Pre-seeded image | "烘焙好的 AMI" | 容器权重已预驻留的节点镜像 |
| Bottlerocket | "AWS 节点 OS" | AWS 容器优化 OS，支持双卷快照 |
| Model streamer | "流式加载" | 权重 I/O 与计算设置重叠 |
| GPU snapshot | "checkpoint 到 HBM" | 序列化加载后的 GPU 状态；重启时反序列化 |
| Tiered loading | "NVMe + DRAM + HBM" | 存储层级；按需加载 |
| Live migration | "移动 token" | 传输输入（KB），在目标节点重新计算 KV |
| `min_workers` | "warm 副本数" | Serverless 最小保活数量 |
| Scale-to-zero | "完全 serverless" | 空闲时零成本；接受完整冷启动代价 |

## 延伸阅读

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Modal 公布的基准测试和 checkpoint 架构。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — 预置数据卷快照模式。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 权重加载与计算设置重叠。
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — 预热手册。
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — 分层加载设计。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — 解耦部署的热迁移。
