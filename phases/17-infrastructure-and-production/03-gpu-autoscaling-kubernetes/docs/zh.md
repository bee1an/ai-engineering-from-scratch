# Kubernetes 上的 GPU 自动扩缩容 — Karpenter、KAI Scheduler、Gang Scheduling

> 三层，不是一层。Karpenter 动态供应节点（一分钟内，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理 gang scheduling、拓扑感知和分层队列 — 它防止 7-of-8 部分分配陷阱，即七个节点等待并烧钱等一个缺失的 GPU。应用层 autoscaler（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）基于推理特定信号扩缩 — 队列深度、KV cache 利用率 — 而非 CPU/DCGM 占空比。经典 HPA 陷阱是 `DCGM_FI_DEV_GPU_UTIL` 是占空比度量：100% 可能是 10 个请求也可能是 100 个。vLLM 预分配 KV cache 内存，所以内存永远不会触发缩容。这节课教你组合三层并避免默认 Karpenter `WhenEmptyOrUnderutilized` 策略在推理中途终止运行中的 GPU 任务。

**Type:** Learn
**Languages:** Python (stdlib, toy queue-depth autoscaler simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## 学习目标

- 画出三层自动扩缩容（节点供应、gang scheduling、应用层）并说出每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 是 vLLM 的错误 HPA 信号，并说出两个替代（队列深度、KV cache 利用率）。
- 描述 gang scheduling 以及 KAI Scheduler 防止的部分分配故障模式（7 of 8 GPU 空闲）。
- 说出会终止运行中 GPU 任务的 Karpenter consolidation 策略（`WhenEmptyOrUnderutilized`）并说明 2026 年的安全替代。

## 问题

你的团队在 Kubernetes 上部署了 LLM 服务。你用 `DCGM_FI_DEV_GPU_UTIL` 作为信号设置了 HPA。服务在工作时间钉在 100% 利用率。HPA 永远不扩容 — 它已经认为你满了。你手动加一个副本；TTFT 下降。HPA 仍然不扩。信号在骗你。

另外，你用 Cluster Autoscaler 管节点。凌晨 2 点来了一个 1M token 的 prompt；集群花 3 分钟供应节点，请求超时了。

再另外，你部署一个 70B 模型需要跨 2 个节点的 8 个 GPU。集群有 7 个空闲 GPU 分散在 3 个节点上。Cluster Autoscaler 为缺的 1 个 GPU 供应一个节点。七个节点等 4 分钟烧钱等 Kubernetes 把最后一个 GPU 拉起来。

三层，三种不同的故障模式。2026 年的 GPU 感知自动扩缩容不是"打开 HPA"。而是组合节点供应、gang scheduling 和应用信号扩缩。

## 概念

### 第一层 — 节点供应（Karpenter）

Karpenter 监视 pending pod 并在约 45-60 秒内供应节点（Cluster Autoscaler 对 GPU 节点通常需要 90-120 秒）。它根据 `NodePool` 约束动态选择实例类型 — 如果你的 pod 需要 8 个 H100 而集群没有匹配节点，Karpenter 直接供应一个而不是扩展现有组。

**consolidation 陷阱**：Karpenter 的默认 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池很危险。它会终止运行中的 GPU 节点以将 pod 迁移到更便宜的合适实例。对推理工作负载这意味着驱逐运行中的请求并在新节点上重新加载 70B 模型。损失是数分钟的容量加请求失败。

GPU 池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

让 Karpenter 在一小时后合并真正空的节点，但永远不驱逐运行中的任务。

### 第二层 — gang scheduling（KAI Scheduler）

KAI Scheduler（项目曾叫 "Karp" 后改名）处理默认 kube-scheduler 做不到的事：

**Gang scheduling** — 全有或全无调度。一个需要 8 个 GPU 的分布式推理 pod 要么 8 个全部一起启动，要么一个都不启动。没有这个，你就会遇到部分分配陷阱：7 of 8 pod 启动了，无限等待，烧钱。

**拓扑感知** — 知道哪些 GPU 共享 NVLink，哪些在同一机架，哪些之间有 InfiniBand。据此放置 pod。DeepSeek-V3 67B tensor-parallel 工作负载必须留在一个 NVLink 域内；KAI Scheduler 尊重这一点。

**分层队列** — 多个团队竞争同一 GPU 池，有优先级和配额。Team A 的生产紧急只有在优先级规则允许时才会被 Team B 的训练任务抢占。

KAI 作为 kube-scheduler 的辅助调度器部署；你给工作负载打注解来使用它。Ray 和 vLLM 生产栈都有集成。

### 第三层 — 应用层信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是占空比指标 — 它测量 GPU 在每个采样间隔是否在做工作。100% 利用率可能意味着 10 个并发请求或 100 个；GPU 反正都在忙。基于占空比扩缩就是盲目扩缩。

更糟的是，vLLM 和类似引擎预分配 KV cache 内存（最高到 `--gpu-memory-utilization`）。即使只有一个请求，内存使用也接近 90%。基于内存的 HPA 永远不会缩容。

**2026 年替代信号**：

- 队列深度（等待 prefill 的请求数）。
- KV cache 利用率（分配给活跃序列的 block 比例）。
- 每副本 P99 TTFT（你的 SLA 信号）。
- Goodput（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d Workload Variant Autoscaler 消费这些信号并扩缩副本。它们完全替代 HPA 用于 LLM serving。

### 什么时候用什么

| 扩缩决策 | 工具 |
|---------|------|
| 增减节点 | Karpenter |
| 调度多 GPU 任务 | KAI Scheduler |
| 增减副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级 | KAI Scheduler 队列 |

### 分离式 prefill/decode 让一切更复杂

如果你运行分离式 prefill/decode（Phase 17 · 17），你有两类 pod 有不同的扩缩触发器：prefill pod 基于队列深度扩缩，decode pod 基于 KV cache 压力扩缩。llm-d 将它们暴露为独立的 `Services`，每个角色有自己的 HPA。不要试图在两者前面放一个 HPA。

### 冷启动在这里也很重要

冷启动缓解（Phase 17 · 10）是节点供应时间变得用户可见的地方。Karpenter 的 45-60 秒预热加 20GB 模型加载加引擎初始化意味着从零开始的请求需要 2-5 分钟。为 SLO 关键路径保持一个 warm pool（`min_workers=1`），或在应用层使用 Modal 风格的 checkpointing。

### 你应该记住的数字

- Karpenter 节点供应：约 45-60s vs Cluster Autoscaler 约 90-120s（GPU 节点）。
- KAI Scheduler 防止部分分配浪费 — 7-of-8 陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：坏了；用队列深度或 KV 利用率。
- Karpenter `WhenEmptyOrUnderutilized`：终止运行中的 GPU 任务。用 `WhenEmpty + consolidateAfter: 1h` 做推理。

## Use It

`code/main.py` 在突发 GPU 工作负载上模拟三层 autoscaler。比较朴素 HPA（占空比）、队列深度 HPA 和 KAI-gang-scheduled 扩缩。报告未满足请求数、空闲 GPU 分钟数和综合评分。

## Ship It

本课产出 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形状和 SLO，设计三层自动扩缩容方案。

## 练习

1. 运行 `code/main.py`。在突发工作负载下，朴素占空比 HPA 丢了多少请求是队列深度 HPA 能接住的？差异来自哪里？
2. 为一个在 H100 SXM5 上服务 Llama 3.3 70B FP8 的集群设计 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个让非 GPU 工作负载远离这些节点的 taint。
3. 你的团队报告部署卡在 Pending 因为"GPU 可用但 pod 调度不上"。诊断 — 是 Karpenter、kube-scheduler 还是 KAI Scheduler？哪些指标能确认？
4. 为分离式 prefill pod 选一个扩缩信号，为 decode pod 选另一个。论证两者。
5. 计算 `WhenEmptyOrUnderutilized` consolidation 陷阱在一个 24x7 生产服务上的成本，该服务平均每天 60 次请求丢弃事件且 P99 TTFT > 10s。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| Karpenter | "节点供应器" | Kubernetes 节点 autoscaler；亚分钟级供应 |
| Cluster Autoscaler | "老 scaler" | Kubernetes 节点 autoscaler 前身；更慢，基于组 |
| KAI Scheduler | "GPU 调度器" | 用于 gang + 拓扑 + 队列的辅助调度器 |
| Gang scheduling | "全有或全无" | 原子调度 N 个 pod 或全部推迟 |
| Topology awareness | "机架感知" | 基于 NVLink/IB/机架位置放置 pod |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU 利用率" | 占空比指标；不是 LLM 的扩缩信号 |
| Queue depth | "等待请求数" | prefill 受限扩缩的正确 HPA 信号 |
| KV cache utilization | "内存压力" | decode 受限扩缩的正确 HPA 信号 |
| Consolidation | "Karpenter 合并" | 终止节点迁移到更便宜的实例类型 |
| `WhenEmpty + 1h` | "安全合并" | 不驱逐运行中 GPU 任务的策略 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档和配置示例。
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — consolidation 策略语义和 GPU 安全默认值。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 扩缩信号。
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 集成模式。
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 托管 Kubernetes 特定指导。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler 设计。
