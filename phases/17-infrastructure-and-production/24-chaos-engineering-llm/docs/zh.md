# LLM 生产环境的混沌工程

> 2026 年 LLM 的混沌工程已成为独立学科。在生产环境运行实验的前提条件：已定义 SLI/SLO、trace+metric+log 可观测性、自动回滚、runbook、值班。架构有四个平面：控制（实验调度器）、目标（服务、基础设施、数据存储）、安全（护栏 + 中止 + 流量过滤）、可观测性（指标 + trace + 日志）、反馈（回馈到 SLO 调整）。护栏是强制性的：burn-rate 告警在日错误预算消耗 > 2 倍预期时暂停实验；抑制窗口 + trace-ID 关联去重告警噪声。节奏：每周小型灰度 + SLO 审查；每月 game day + 事后分析；每季度跨团队韧性审计 + 依赖映射。LLM 特有实验：内存过载、网络故障、供应商中断、畸形 prompt、KV cache 驱逐风暴。工具：Harness Chaos Engineering（LLM 衍生推荐、爆炸半径缩小、MCP 工具集成）；LitmusChaos（CNCF）；Chaos Mesh（CNCF Kubernetes 原生）。

**Type:** Learn
**Languages:** Python (stdlib, toy chaos experiment runner)
**Prerequisites:** Phase 17 · 23 (SRE for AI), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## 学习目标

- 列出五个混沌工程前提条件（SLI/SLO、可观测性、回滚、runbook、值班）并解释跳过任何一个为什么会破坏实践。
- 画出四个平面（控制、目标、安全、可观测性）和反馈到 SLO 的循环。
- 列举五个 LLM 特有实验（内存过载、网络故障、供应商中断、畸形 prompt、KV 驱逐风暴）。
- 根据技术栈选择工具——Harness、LitmusChaos、Chaos Mesh。

## 问题

传统栈的混沌测试已经成熟。LLM 栈增加了新的故障模式。一个带毒字符的 4K-token prompt 使分词器停滞 12 秒。上游供应商 429；你的网关重试；你的服务因重试放大的并发而 OOM。突发负载下的 KV cache 驱逐风暴导致重新 prefill 级联，饱和计算。

这些都不会在单元测试中出现。混沌工程是你在用户之前发现它们的方式。

## 核心概念

### 前提条件

不要在没有以下条件的情况下在生产环境运行混沌：

1. **SLI/SLO** — 已定义的服务级别指标和目标。
2. **可观测性** — trace、指标、日志，连接到仪表板。
3. **自动回滚** — Phase 17 · 20 策略 flag 回滚。
4. **Runbook** — 结构化的，Phase 17 · 23。
5. **值班** — 有人响应。

缺少任何一个意味着混沌变成真实事故。

### 四个平面 + 反馈

**控制平面** — 实验调度器（Litmus workflow、Chaos Mesh schedule、Harness UI）。

**目标平面** — 服务、pod、节点、负载均衡器、数据存储。

**安全平面** — 终止开关、抑制窗口、爆炸半径限制、错误预算门控。

**可观测性平面** — 正常指标 + trace-ID 关联以区分混沌引起的和自然故障。

**反馈循环** — 发现回馈到 SLO 调整、runbook 更新、代码修复。

### 护栏是强制性的

- **Burn-rate 告警**：日错误预算消耗超过 2 倍预期时暂停实验。
- **抑制窗口**：实验期间在爆炸半径内静默非实验告警。
- **Trace-ID 关联**：所有实验引起的错误携带标签，值班可以去重。

### 五个 LLM 特有实验

1. **内存过载** — 通过高并发发送长上下文请求强制 KV cache 抢占风暴。观察：服务是优雅降载还是崩溃？

2. **网络故障** — 切断推理网关和供应商之间的连接。观察：降级是否在 SLA 内启动？（Phase 17 · 19）

3. **供应商中断模拟** — OpenAI 100% 429。观察：路由是否故障转移到 Anthropic？（Phase 17 · 16, 19）

4. **畸形 prompt** — 注入使分词器停滞的载荷（如深度嵌套 unicode、巨大 UTF-8 码点）。观察：单个请求是否锁住一个 worker？

5. **KV 驱逐风暴** — 通过饱和 vLLM block 预算强制驱逐。观察：LMCache 是否恢复还是服务降级？

### 节奏

- **每周** — 在预发布环境做小型灰度实验，可能 5% 生产。
- **每月** — 针对特定场景的计划 game day；跨团队参与；事后分析。
- **每季度** — 跨团队韧性审计；依赖映射更新。

### 工具

- **Harness Chaos Engineering** — 商业；AI 衍生的实验推荐；爆炸半径缩小；MCP 工具集成。
- **LitmusChaos** — CNCF 毕业；Kubernetes workflow 式。
- **Chaos Mesh** — CNCF sandbox；Kubernetes 原生 CRD 风格。
- **Gremlin** — 商业；广泛支持。
- **AWS FIS** / **Azure Chaos Studio** — 托管云产品。

### 从小开始

第一个实验：在稳定流量下 pod-kill 一个 decode 副本。观察重路由和恢复。如果这个可行且看起来安全，升级到网络混沌。

第一个 LLM 特有实验：注入一个供应商 429 持续 5 分钟。观察降级。大多数团队发现他们的降级没有被完全测试过。

### 需要记住的数字

- 四个平面：控制、目标、安全、可观测性。
- Burn-rate 暂停：2 倍预期日预算消耗。
- 节奏：每周灰度，每月 game day，每季度审计。
- 五个 LLM 实验：内存、网络、供应商、畸形 prompt、KV 风暴。

## Use It

`code/main.py` 模拟三个带安全平面门控的混沌实验。报告哪些实验会触发 burn-rate 中止。

## Ship It

本课产出 `outputs/skill-chaos-plan.md`。给定技术栈和成熟度，选择前三个实验和工具。

## 练习

1. 运行 `code/main.py`。哪个实验触发了 burn-rate 门控？为什么？
2. 为基于 vLLM 的 RAG 服务设计前五个混沌实验。包含成功标准。
3. 你的 burn-rate 告警暂停了一个实验。如何确定根因——混沌还是自然？
4. 论证混沌应该在生产还是仅在预发布运行。什么时候生产是正确答案？
5. 列出三个通用网络混沌无法复现的 LLM 特有故障模式。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| SLI / SLO | "服务目标" | 指标 + 目标；必需前提 |
| Blast radius | "范围" | 受实验影响的服务/用户集合 |
| Burn-rate alert | "预算门控" | 错误预算消耗率 > 2 倍预期时触发 |
| Game day | "月度演练" | 计划的跨团队混沌演练 |
| LitmusChaos | "CNCF workflow" | CNCF 毕业的 Kubernetes 混沌工具 |
| Chaos Mesh | "CNCF CRD" | CNCF sandbox Kubernetes 原生混沌 |
| Harness CE | "商业 AI 辅助" | 带 AI 推荐的 Harness 混沌 |
| Malformed prompt | "分词器炸弹" | 使分词停滞的输入 |
| KV eviction storm | "抢占级联" | 大规模驱逐触发重新 prefill |

## 延伸阅读

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
