# AI SRE — 多 Agent 事件响应、Runbook、预测性检测

> AI SRE 使用通过 RAG 接入基础设施数据（日志、runbook、服务拓扑）的 LLM 来自动化调查、文档和协调阶段。2026 年的架构模式是多 agent 编排——专业 agent（日志、指标、runbook）由 supervisor 协调；AI 提出假设和查询，人类批准判断性决策。Datadog Bits AI 和 Azure SRE Agent 以托管产品形式提供。Runbook 在进化：NeuBird Hawkeye 使用对抗性评估（两个模型分析同一事件；一致 = 高置信度，不一致 = 不确定性）；运维记忆跨团队变动持久化。自动修复保持谨慎：AI 建议，人类批准。完全自主操作范围很窄（重启 pod、回滚特定部署），有严格护栏——任何卖"设好就忘"的都在过度承诺。新兴前沿：事前预测。MIT 研究报告一个在历史日志 + GPU 温度 + API 错误模式上训练的 LLM 提前 10-15 分钟预测了 89% 的故障。预测：到 2026 年底 95% 的企业 LLM 将有自动故障转移。

**Type:** Learn
**Languages:** Python (stdlib, toy multi-agent incident triage simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 24 (Chaos Engineering)
**Time:** ~60 minutes

## 学习目标

- 画出多 agent AI SRE 架构图：supervisor + 专业 agent（日志、指标、runbook）+ 人类审批门控。
- 解释为什么自动修复范围窄（重启 pod、回滚部署）而非宽（重新架构服务）。
- 说明对抗性评估模式（NeuBird Hawkeye）：两个模型一致 = 高置信度；不一致 = 升级。
- 引用 MIT 89% 早期检测结果及运维约束：没有执行能力的预测只是仪表板。

## 问题

一个值班工程师凌晨 3 点被叫醒。"结账服务错误率高。"他们查 Datadog、Loki、三个 runbook、部署日志。30 分钟后发现根因是 vLLM 因 KV cache 飙升导致 OOM。他们重启 pod；错误消除。

在 2026 年，调查的前 20 分钟是可自动化的。按服务分组日志、关联最近部署、匹配 runbook——都是 RAG + tool-use。一个受监督的 agent 可以在人类打开 Datadog 之前完成初步分诊并呈现假设。

完全自主修复是另一个问题。重启 pod：安全。扩展 GPU 池：如果策略允许则安全。重新架构服务：绝对不行。纪律在于画出那条窄线。

## 核心概念

### 多 agent 架构

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Supervisor 将事件拆分为子查询。专业 agent 有工具访问权限（日志搜索、PromQL、文档检索）。Supervisor 综合后向人类呈现假设 + 证据。人类批准或重定向。

### 自动修复范围

**安全（窄）**：重启 pod、回滚特定部署、在预批准范围内扩展池、启用预批准的 feature flag。

**不安全（宽）**：更改服务拓扑、修改资源限制、部署新代码、更改 IAM、修改数据库。

任何卖"设好就忘"的都在过度承诺。安全集合随 AI SRE 成熟而增长，但边界是真实的。

### 对抗性评估（NeuBird Hawkeye）

两个模型独立分析同一事件。如果它们对根因一致，置信度高。如果不一致，升级给人类，两个假设都可见。简单模式，有效过滤幻觉根因。

### 运维记忆

团队流动是传统 SRE 的隐形杀手——部落知识随人离开。AI SRE 将 runbook + 事后分析存储在向量数据库中；agent 在每个新事件上检索。新工程师加入时，AI 有完整历史。

### 事前预测

MIT 2025 研究：在历史日志、GPU 温度、API 错误模式上训练的 LLM 在测试集上提前 10-15 分钟预测了 89% 的故障。

现实检查：没有执行能力的预测就是仪表板。运维问题是"当我们预测到时，我们做什么？"预防性排空？呼叫？自动扩缩？答案是策略特定的。

### 2026 年产品

- **Datadog Bits AI** — Datadog 内的托管 SRE 副驾驶。
- **Azure SRE Agent** — Azure 原生。
- **NeuBird Hawkeye** — 对抗性评估 + 运维记忆。
- **PagerDuty AIOps** — 分诊 + 去重。
- **Incident.io Autopilot** — 事件指挥官 + 协调。

### Runbook 即代码

Runbook 从 Confluence 页面进化为带结构化章节（症状、假设、验证、操作）的版本化 markdown。结构化 runbook 提供更好的 RAG 检索。任何 AI-SRE 推出都应从将非结构化 runbook 转为结构化开始。

### 需要记住的数字

- MIT 早期检测：89% 的故障，10-15 分钟提前量。
- 多 agent 分诊：supervisor +（日志、指标、runbook）+ 人类。
- 安全自动修复集：重启 pod、回滚部署、在范围内扩缩。
- 对抗性评估：两个模型独立；一致 = 高置信度。

## Use It

`code/main.py` 模拟多 agent 分诊：日志 agent 发现错误，指标 agent 发现 CPU 飙升，runbook agent 匹配到已知问题。Supervisor 排列假设。

## Ship It

本课产出 `outputs/skill-ai-sre-plan.md`。给定当前值班、事件量、团队成熟度，设计 AI SRE 推出方案。

## 练习

1. 运行 `code/main.py`。如果日志和指标 agent 不一致怎么办？Supervisor 如何解决？
2. 为你的服务定义三个"安全"自动修复操作。论证每个。
3. 写一个结构化 runbook 模板：章节、必填字段、验证命令。
4. 预测性检测在 12 分钟提前量时触发。你的策略是什么——呼叫、预排空、还是两者？
5. 论证一个 3 人团队是否应该在 2026 年采用 AI SRE 还是等待。考虑成熟度、量、风险。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| AI SRE | "值班 agent" | LLM 支持的事件调查 + 协调 |
| Supervisor agent | "编排器" | 将事件拆分为子查询的顶层 agent |
| Specialized agent | "领域 agent" | 有工具访问权限的子 agent（日志、指标、runbook） |
| Auto-remediation | "AI 修复" | 窄范围预批准操作；不是宽泛的重新架构 |
| Operational memory | "向量 runbook" | 事后分析 + runbook 在向量数据库中供 RAG |
| Adversarial eval | "双模型检查" | 独立分析；一致 = 高置信度 |
| NeuBird Hawkeye | "对抗性那个" | 带对抗性评估 + 记忆模式的产品 |
| Bits AI | "Datadog 的 SRE agent" | Datadog 托管 AI SRE |
| Pre-incident prediction | "早期检测" | 故障预测 10-15 分钟提前量 |

## 延伸阅读

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
