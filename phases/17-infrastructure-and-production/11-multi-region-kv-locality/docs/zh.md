# 多区域 LLM 服务与 KV Cache 局部性

> Round-robin 负载均衡对缓存 LLM 推理是有害的。请求如果没有落在持有其 prefix 的节点上，就要付出完整 prefill 成本——长 prompt 的 P50 约 800 ms，而 cache 命中时约 80 ms。2026 年的生产模式是 cache-aware router（Rust 实现的 vLLM Router、llm-d router），它消费 KV cache 事件并基于 prefix hash 匹配进行路由。最新研究（GORGO）将跨区域网络延迟作为路由目标中的显式项。商业"跨区域推理"产品（Bedrock cross-region inference、GKE multi-cluster gateway）将推理视为黑盒——它们处理可用性，而非 TTFT。JPMorgan 和 Mayo Clinic 在 2024 年 11 月的 us-east-1 故障转移中耗时约 22 分钟。DR 现实：32% 的 LLM DR 失败是因为团队备份了权重但忘了 tokenizer 文件或量化配置。

**Type:** Learn
**Languages:** Python (stdlib, toy prefix-cache-aware router simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 解释为什么 round-robin 负载均衡会破坏缓存推理，并量化 TTFT 损失。
- 画出 cache-aware router 的架构：输入（KV cache 事件）、算法（prefix hash 匹配）、平局决策（GPU 利用率）。
- 说明 LLM 32% DR 失败的驱动因素（缺失 tokenizer 文件/量化配置），并陈述三文件 DR 清单。
- 区分商业跨区域产品（Bedrock CRI、GKE Multi-Cluster Gateway）与 KV-aware 路由。

## 问题

你的服务运行在 us-east-1、us-west-2 和 eu-west-1。你在前面放了一个 ALB 做 round-robin。生产环境中 prefix cache 命中率降到 8%。TTFT P50 翻了三倍。你的 vLLM 日志显示每个请求都在付完整 prefill 成本。

Round-robin 对无状态服务是最优的。LLM 推理在设计上是有状态的——KV cache 编码了模型已见过的所有内容。盲目路由就是路由到错误的 cache。

另外，你的团队有 DR 计划。你把模型权重跨区域备份到 S3。区域故障发生；你尝试故障转移；副本拒绝启动。你忘了 tokenizer.json、量化配置和 RoPE scaling 配置在一个你没有同步的单独 bucket 里。

多区域 LLM 服务是一个 cache 问题、一个路由问题和一个 DR 卫生问题——不是一个负载均衡器问题。

## 概念

### Cache-aware 路由

请求带着 prompt 到达。Router 对 prefix 做 hash（比如前 512 个 token）；它询问每个副本"你缓存了这个 prefix 吗？"。副本在分配和驱逐 block 时通过 pub/sub 通道发布 KV cache 事件。Router 选择匹配的副本，如果没有匹配则回退到基于 GPU 利用率的平局决策。

**vLLM Router**（Rust，2026 production stack）：订阅 `kv.cache.block_added` 事件，维护 prefix hash → 副本索引，O(1) 查找路由。无匹配时回退到最短队列深度。

**llm-d router**：相同模式，Kubernetes 原生。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（Phase 17 · 06）是副本内的等价物。跨副本路由严格在上游。

### 数据

TTFT P50，2K token prompt，Llama 3.3 70B FP8，H100：
- Cache 命中（同一副本，prefix 驻留）：约 80 ms。
- Cache 未命中（冷 prefill）：约 800 ms。

10x 差距。如果你的 router 在跨副本中达到 60-80% 的 prefix cache 命中率，你就能在 N 副本容量下逼近单副本性能。如果只有 10%，你就逼近朴素扩展。

### 跨区域有新约束——网络延迟

区域间 RTT：
- us-east-1 ↔ us-west-2：约 65 ms。
- us-east-1 ↔ eu-west-1：约 75 ms。
- us-east-1 ↔ ap-southeast-1：约 220 ms。

如果路由将请求从 us-east-1 发送到 ap-southeast-1 的热 prefix，节省的 prefill（800 → 80 ms）被 440 ms 往返延迟淹没。GORGO（2026 研究）将此显式化——联合最小化 `prefill_time + network_latency`，而非仅最小化 prefill。通常答案是保持区域内路由，除非在 prefill 占主导的超大 multi-MB prefix 场景。

### 商业"跨区域推理"在这里帮不上忙

AWS Bedrock cross-region inference 在容量压力时自动将请求路由到其他区域。它优化可用性，而非 TTFT，并将推理视为黑盒。GKE Multi-Cluster Gateway 同理——服务级故障转移，不感知 KV cache。

即使使用这些产品，你仍然需要应用层的 cache-aware router。它们处理"us-east-1 着火了"的场景。Cache-aware 路由处理 TTFT 场景。

### DR 卫生——32% 缺失文件问题

2026 年广泛引用的统计：32% 的 LLM DR 失败发生在团队备份了权重但忘了：

- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ scales、GPTQ zero-points）
- 模型特定配置（RoPE scaling、attention masks、chat templates）
- 引擎配置（`vllm_config.yaml`、sampling defaults、LoRA adapter manifests）

修复方法是三文件最小 DR manifest：

1. HF 模型仓库下的所有文件（权重 + 配置 + tokenizer）。
2. 引擎特定的服务配置。
3. 部署 manifest（K8s YAML、Dockerfile、依赖锁文件）。

另外：每季度做一次 DR 演练。JPMorgan 的 us-east-1 演练在 2024 年 11 月达到 22 分钟恢复，仅仅因为 playbook 经过了排练。

### 数据驻留是正交的

EU 客户的 PHI 不能离开 EU。如果你的 cache-aware router 为了 prefix 匹配将巴黎发起的请求发送到 us-east-1，无论 TTFT 收益如何你都违反了 GDPR。在优化 cache 之前，先按驻留边界分区 router。

### 你应该记住的数字

- Cache 命中 vs 未命中 TTFT 差距：约 10x（2K prompt 上 80 ms vs 800 ms）。
- 区域间 RTT US-EU：约 75 ms。
- DR 失败：32% 缺失 tokenizer/量化配置。
- JPMorgan us-east-1 故障转移 2024 年 11 月：22 分钟（30 分钟 SLA）。

## Use It

`code/main.py` 模拟三种路由策略（round-robin、cache-aware 区域内、cache-aware 全局）在多区域工作负载上的表现。报告 cache 命中率、TTFT P50/P99 和跨区域费用。

## Ship It

本课程产出 `outputs/skill-multi-region-router.md`。给定区域、驻留约束和 SLA，设计路由方案。

## 练习

1. 运行 `code/main.py`。在 75 ms RTT 下，prompt 长度达到多少时跨区域路由优于仅本地路由？
2. 你的 cache 命中率从 70% 降到 12%。诊断三个可能原因以及确认每个原因的可观测指标。
3. 为一个在 vLLM 中服务的 70B AWQ 量化模型（带 5 个 LoRA adapter）设计 DR manifest。列出每个文件和配置。
4. 论证 Bedrock cross-region inference 对于有严格 TTFT SLO 的金融科技公司是否"足够"。引用具体行为。
5. 一个巴黎发起的请求匹配了 us-east-1 中的 prefix。你路由它吗？写出策略。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|----------|----------|
| Cache-aware routing | "智能 LB" | 基于 prefix hash 匹配路由到持有 KV cache 的副本 |
| KV cache events | "cache pub-sub" | 副本发布 block 添加/驱逐事件；router 索引 |
| Prefix hash | "cache key" | 前 N 个 token 的 hash，用作 router 查找 |
| GORGO | "跨区域路由研究" | arXiv 2602.11688；网络延迟作为显式项 |
| Cross-region inference | "Bedrock CRI" | AWS 产品；可用性故障转移，非 TTFT 感知 |
| DR manifest | "备份清单" | 恢复所需的每个文件——不仅是权重 |
| Data residency | "GDPR 边界" | 哪个区域可以看到用户数据的法律约束 |
| RTT | "往返时间" | 网络延迟；US-EU 75 ms，US-APAC 220 ms |
| LLM-aware LB | "cache 命中 LB" | Cache-aware router 作为产品类别 |

## 延伸阅读

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 带网络延迟项的跨区域 KV cache 复用。
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — 可用性故障转移文档。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — cache-aware router 源码。
