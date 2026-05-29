# vLLM Production Stack 与 LMCache KV Offloading

> vLLM 的 production-stack 是参考 Kubernetes 部署——路由器、引擎和可观测性连接在一起。LMCache 是 KV offloading 层，将 KV cache 从 GPU 显存中提取出来，跨查询和引擎复用（CPU DRAM，然后是磁盘/Ceph）。vLLM 0.11.0 KV Offloading Connector（2026 年 1 月）使其异步化并通过 Connector API（v0.9.0+）可插拔。Offload 延迟不面向用户。即使没有共享前缀，LMCache 也有价值——当 GPU 的 KV 槽位用完时，被抢占的请求可以从 CPU 恢复而不是重新计算 prefill。已发布的基准测试在 16x H100（80GB HBM）跨 4 台 a3-highgpu-4g 上：当 KV cache 超过 HBM 时，原生 CPU offload 和 LMCache 都显著提升吞吐；在低 KV 占用时，所有配置与基线匹配，仅有少量开销。

**Type:** Learn
**Languages:** Python (stdlib, toy KV-spill simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang/RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 画出 vLLM production-stack 各层：路由器、引擎、KV offload、可观测性。
- 解释 KV Offloading Connector API（v0.9.0+）以及 0.11.0 异步路径如何隐藏 offload 延迟。
- 量化 LMCache CPU-DRAM 何时有帮助（KV > HBM）vs 何时增加开销（KV 足够放入 HBM）。
- 根据部署约束在原生 vLLM CPU offload 和 LMCache connector 之间选择。

## 问题

你的 vLLM serving 显示 GPU HBM 100% 占用，并发上升时出现抢占事件。请求被驱逐、重新排队，你在一分钟内对同一个 2K-token prompt 重新 prefill 了四次。GPU 计算花在了冗余 prefill 上；goodput 远低于原始吞吐。

增加更多 GPU 成本线性增长。增加更多 HBM 不可能。但 CPU DRAM 便宜——一个 socket 有 512 GB+，延迟比 HBM 差几个数量级，但对"暂时温热"的 KV cache 来说足够了。

LMCache 将 KV cache 提取到 CPU DRAM，使被抢占的请求快速恢复，并且跨引擎共享重复前缀的缓存，无需每个引擎重新 prefill。

## 核心概念

### vLLM production-stack

`github.com/vllm-project/production-stack` 是参考 Kubernetes 部署：

- **Router** — 缓存感知（Phase 17 · 11）。消费 KV 事件。
- **Engines** — vLLM worker。每 GPU 一个或每 TP/PP 组一个。
- **KV cache offload** — LMCache 部署或原生 connector。
- **Observability** — Prometheus 抓取、Grafana 仪表板、OTel traces。
- **Control plane** — 服务发现、配置、滚动更新。

以 Helm chart + operator 形式交付。

### KV Offloading Connector API (v0.9.0+)

vLLM 0.9.0 引入了 Connector API，用于可插拔的 KV cache 后端。你的引擎将 block offload 到 connector；connector 存储它们（RAM、磁盘、对象存储、LMCache）。请求需要某个 block 时，connector 加载回来。

vLLM 0.11.0（2026 年 1 月）增加了异步 offload 路径——offload 可以在后台进行，常见情况下引擎不会阻塞。端到端延迟和吞吐仍取决于工作负载形态、KV cache 命中率和系统压力；vLLM 自己的说明指出自定义内核 offload 在低命中率时可能降低吞吐，异步调度与投机解码存在已知交互问题。

### 原生 CPU offload vs LMCache

**原生 vLLM CPU offload**：引擎本地。将 KV block 存储在主机 RAM 中。实现快，零网络跳。不跨引擎。

**LMCache connector**：集群级别。将 block 存储在共享 LMCache 服务器中（CPU DRAM + Ceph/S3 层）。任何引擎都可访问 block。已发布 16x H100 基准测试。

单个引擎有 HBM 压力时选原生。多个引擎共享前缀时选 LMCache（RAG 共享 system prompt、多租户共享模板）。

### 基准测试表现

16x H100（80 GB HBM）跨 4 台 a3-highgpu-4g 测试：

- 低 KV 占用（短 prompt、低并发）：所有配置与基线匹配，LMCache 增加约 3-5% 开销。
- 中等占用：LMCache 开始在跨引擎前缀复用上有帮助。
- KV 超过 HBM：原生 CPU offload 和 LMCache 都显著提升吞吐；LMCache 收益更大，因为跨引擎共享。

### LMCache 决定性场景

- 多租户 serving，system prompt 跨租户共享。
- RAG，文档块跨查询重复。
- 同一基础模型的微调变体（LoRA），基础模型 KV 复用减少冗余工作。
- 抢占密集型工作负载：从 CPU 恢复比重新 prefill 便宜。

### 何时不该启用

- HBM 压力小——你付出开销却没有收益。
- 短上下文（<1K token）——传输时间 > 重新 prefill。
- 单租户单 prompt 工作负载——没有可捕获的复用。

### 与分离 serving 的集成

Phase 17 · 17 分离 serving + LMCache 复合：从 prefill 池到 decode 池的 KV 传输如果未使用则落入 LMCache；后续查询从 LMCache 拉取。Phase 17 · 11 缓存感知路由器可以路由到本地或 LMCache 共享缓存匹配的引擎。

### 需要记住的数字

- vLLM 0.9.0：Connector API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步 offload 路径；端到端延迟影响取决于工作负载、KV 命中率和系统压力（非绝对保证）。
- 16x H100 基准测试：KV 占用超过 HBM 时 LMCache 有帮助。
- HBM 压力小时：3-5% 开销无收益。

## Use It

`code/main.py` 模拟有无 LMCache 的抢占密集型工作负载。报告避免的重新 prefill 次数、吞吐提升和盈亏平衡 HBM 利用率。

## Ship It

本课产出 `outputs/skill-vllm-stack-decider.md`。给定工作负载形态和 vLLM 部署，决定原生 vs LMCache vs 都不用。

## 练习

1. 运行 `code/main.py`。在什么 HBM 利用率时 LMCache 开始有回报？
2. 一个租户跨 200 查询/小时共享 6K-token system prompt。计算每租户的预期 LMCache 节省。
3. LMCache 服务器是单点故障。设计高可用策略（副本、降级到原生）。
4. LMCache 存储到 Ceph 机械硬盘。对于 70B FP8 的 4K-token KV（500 MB），读取时间 vs 重新 prefill 是多少？
5. 论证 vLLM 0.11.0 异步路径是否"免费"——开销隐藏在哪里？

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| Production-stack | "参考部署" | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | "KV 后端接口" | vLLM 0.9.0+ 可插拔 KV 存储接口 |
| Native CPU offload | "引擎本地溢出" | 将 KV 存储在同一引擎的主机 RAM 中 |
| LMCache | "集群 KV cache" | 跨引擎 KV cache 服务器，CPU DRAM + 磁盘 |
| 0.11.0 async | "非阻塞 offload" | Offload 隐藏在引擎流之后 |
| Preemption | "驱逐腾空间" | HBM 满时的 KV cache 洗牌 |
| Prefix reuse | "相同 system prompt" | 多个查询共享开头；缓存命中 |
| Ceph tier | "磁盘层" | 缓存层次中 DRAM 之下的持久存储 |

## 延伸阅读

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator.
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector implementation.
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — asynchronous path details.
