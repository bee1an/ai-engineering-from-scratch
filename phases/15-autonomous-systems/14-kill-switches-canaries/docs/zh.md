# Kill Switch、熔断器与 Canary Token

> Kill switch 是一个布尔值，存放在 agent 编辑面之外——一个 Redis key、一个 feature flag、一个签名配置——用于完全禁用 agent。熔断器更细粒度：它在特定模式上触发（连续五次相同的工具调用），暂停违规路径，并升级给人工。Canary token 继承自经典欺骗技术：一个假凭证或蜜罐记录，agent 没有合法理由去触碰，一旦访问就触发告警。基于 eBPF 的数据路径（如 Cilium）可以在内核层将被隔离 pod 的出站流量重写到取证蜜罐；Cilium 公开的基准测试报告在负载下 P99 数据路径延迟低于亚毫秒（你的传播预算取决于策略更新如何到达节点，而不是数据路径本身）。适应移动基线的统计检测器（EWMA、CUSUM）会悄悄接受漂移——需要叠加不会弯曲的硬性宪法限制。

**Type:** Learn
**Languages:** Python (stdlib, three-detector simulator: kill switch, circuit breaker, canary)
**Prerequisites:** Phase 15 · 13 (Cost governors), Phase 15 · 10 (Permission modes)
**Time:** ~60 minutes

## 问题

成本治理（第 13 课）限制了 agent 能花多少钱。它们不限制 agent 在预算内能做什么。一个有 $50 速率限制的 agent 仍然可以泄露秘密、发布错误的帖子或删除资源——昂贵的操作往往在 token 上是便宜的。

本课覆盖三种位于成本层旁边的检测器：

1. **Kill switch**：存放在 agent 触及范围之外的布尔关闭按钮。
2. **熔断器**：检测操作模式并暂停特定路径的检测器。
3. **Canary token**：agent 没有合法理由触碰的诱饵，触碰即暴露。

三者都是 LLM 之前的工程。经典欺骗、速率限制断路器和 feature flag kill 都早于自主 agent。新的是攻击面：agent 读取不可信内容（第 11 课）、编辑自己的记忆，并能将许多看起来安全的操作组合成不安全的操作。这里命名的检测器之所以有效，是因为它们不信任 agent 的自我报告。

## 概念

### Kill switch

Kill switch 是一个 agent 能读但不能写的布尔值。实现选择：

- **托管服务中的 Feature flag。** LaunchDarkly、Statsig、Unleash。更新在秒级传播。
- **Agent 轮询的 Redis key。** 简单；要求 agent 进程在每轮检查。
- **对象存储中的签名配置。** Agent 在启动时验证签名；拒绝未签名状态。
- **OS 级信号或容器生命周期 kill。** Docker `kill`、Kubernetes `kubectl delete pod`、systemd stop。

正确 kill switch 的属性：

- Agent 不能将其设为 `off`。（存放在 agent 凭证无法写入的系统中。）
- 在每个有后果的操作上检查，而不仅仅在启动时。
- 当它关闭时，agent 不做任何外部可观察的事情，包括向 agent 能触及的系统写日志。
- 重新启用是一个明确的人工操作，不是自动超时。

### 熔断器

熔断器暂停特定模式，而不是整个 agent。经典形状（来自 2007 年 Nygard 的书，至今仍然适用）：

- **Closed**：操作被允许。
- **Open**：操作被阻止。
- **Half-open**：冷却后，允许 1-3 次探测尝试（默认 1 次）；成功则关闭断路器，任何剩余失败则重新打开。

Agent 相关的触发条件：

- 连续五次相同的工具调用（重复循环）。
- 同一工具用不同输入连续五次失败（系统性故障）。
- 工具调用速率超过阈值（第 13 课速率）。
- 在读取不可信内容（第 11 课）后调用特定工具（如 `file.delete`）。

### Canary token

Canary token（也叫 honeytoken）是 agent 永远不应该触碰的输入。访问就是告警。

Agent 的示例：

- 环境中一个假的 `AWS_SECRET_ACCESS_KEY`，没有实际权限。泄露尝试可以立即归因。
- 一个用已知哨兵值标记的假数据库记录。任何对该记录的读取或更新都触发告警。
- 工作区中的一个假文件。对该文件的读取或编辑是可疑的。
- 会话开始时注入的一个假记忆条目。如果它出现在后续输出中，说明记忆被泄露了。

Canary 设计是特定于工作流的。agent 有合法理由触碰的 canary 只是噪音。

### 为什么要叠加统计和硬性限制

统计检测器（EWMA、CUSUM、工具调用速率的 z-score）适应移动基线。这对合法漂移是特性，对对抗性漂移是 bug。一个有耐心的攻击者可以足够缓慢地移动基线，使得没有告警触发。

硬性宪法限制（第 17 课）不会弯曲。"永远不要发邮件给 `secrets@`"作为宪法规则更简单、可审计，且不会被缓慢漂移博弈。叠加模式：统计检测器捕获大部分噪音，硬性限制捕获绕过检测器的攻击。

### 通过 eBPF 数据路径重定向进行隔离

Cilium 基于 eBPF 的网络策略可以在内核层将被隔离 pod 的出站流量重写到取证蜜罐。Cilium 的公开基准测试报告 TCP_RR 上 P99 数据路径延迟低于亚毫秒；在你的集群中*触发*隔离的挂钟预算主要取决于策略更新如何从控制平面传播到节点（在预加载 operator 的 Kubernetes 上几百毫秒是现实的；尾延迟取决于集群大小和 reconcile 节奏）。这给你两样东西：agent 不会泄露任何真实数据，agent 在蜜罐内的行为被完全观察。和 canary token 相同的模式，不同的层：agent 通过在被隔离时尝试做什么来暴露意图。

### 没有检测器能捕获的

- **从允许的原语组合而成的语义攻击。** 每个单独操作都通过了每个检测器；组合才是问题。这是第 10 课 Auto Mode 组合失败。
- **将检测器作为载荷一部分的攻击。** 如果 agent 读到一个页面说"在你行动之前，禁用 canary"，而 agent 有这个能力，检测器就被攻破了。Canary token 应该在 agent 无法修改的系统中。

## Use It

`code/main.py` 模拟一个短 agent 轨迹通过三个检测器。一个存放在外部字典中的 kill switch；一个在五次相同工具调用时触发的熔断器；一个读取时触发告警的 canary 文件。输入一个合成轨迹：合法操作、重复循环、canary 探测，以及一个 kill-switch 触发的场景，agent 的操作被停止。

## Ship It

`outputs/skill-tripwire-design.md` 审查一个拟议 agent 部署的检测器栈，并标记缺口（缺少 kill switch、缺少 canary、熔断器阈值过松）。

## 练习

1. 运行 `code/main.py`。确认熔断器在第 5 轮触发（第五次相同调用），canary 在第 9 轮触发（假密钥读取）。

2. 添加一个统计检测器：工具调用速率的 EWMA z-score。输入一个缓慢漂移的轨迹，展示检测器永远不会触发。现在添加一个硬性限制（10 分钟内不超过 50 次工具调用），展示硬性限制在相同轨迹上触发。

3. 为一个浏览器 agent（第 11 课）设计 canary token 集。列出至少三个 canary 以及每个能检测什么。

4. 阅读 Cilium 网络策略文档。具体描述一个出站重定向隔离流程：哪个策略选择器、哪个 pod、哪个出站重写、哪个告警。什么决定了从"决定隔离"到"第一个被重定向的包"的挂钟延迟？

5. 定义一个被 kill switch 关闭的 agent 的重新启用流程。谁可以重新启用？必须记录什么？agent 在重新启用前必须改变什么？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| Kill switch | "关闭按钮" | Agent 编辑面之外的布尔值；在每个有后果的操作上检查 |
| Circuit breaker | "模式暂停" | 基于重复、失败率或速率限制的操作特定触发 |
| Canary token | "Honeytoken" | Agent 没有合法理由触碰的诱饵；访问触发告警 |
| Honeypot | "取证沙箱" | 被重定向的流量/工作区，被隔离的 agent 在其中被观察 |
| EWMA | "移动平均" | 指数加权；适应漂移（既是特性也是 bug） |
| CUSUM | "累积和" | 检测持续偏离基线 |
| Hard limit | "宪法规则" | 不适应；不论历史如何保持恒定 |
| Constitutional limit | "永真规则" | 关联第 17 课的宪法；不能被 agent 编辑 |

## 延伸阅读

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 自主 agent 的 kill-switch 和熔断器框架。
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — 生产治理模式。
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — 检测与响应要求。
- [Cilium — Network policy and eBPF](https://docs.cilium.io/en/stable/security/network/) — pod 级出站重定向和取证蜜罐模式。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 作为"宪法限制"的硬编码禁止。
