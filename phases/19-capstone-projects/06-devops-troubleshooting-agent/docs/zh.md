# 毕业项目 06 — Kubernetes DevOps排障智能体

> AWS 的 DevOps Agent 正式发布，Resolve AI 公开了其 K8s playbook，NeuBird 演示了语义监控，Metoro 将 AI SRE 与每服务 SLO 绑定。生产形态已经定型：告警 webhook 触发，智能体读取遥测数据，遍历 K8s 对象图，排序根因假设，并在 Slack 中发布带审批按钮的简报。默认只读。每个修复操作都由人工审批。这个毕业项目就是构建这样一个智能体，在 20 个合成事件上评估，并与 AWS Agent 在三个共享案例上对比。

**类型：** 毕业项目
**语言：** Python（智能体），TypeScript（Slack 集成）
**前置要求：** Phase 11（LLM 工程）、Phase 13（工具与 MCP）、Phase 14（智能体）、Phase 15（自主系统）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段：** P11 · P13 · P14 · P15 · P17 · P18
**时间：** 30 小时

## 问题

2025-2026 年的 SRE 叙事变成了："AI 智能体分诊事件，人类审批修复。"AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产中发布了这种形态。智能体读取 Prometheus 指标、Loki 日志、Tempo trace、kube-state-metrics 和一个 K8s 对象知识图谱。它在五分钟内产出带遥测引用的排序根因假设。它永远不会在没有通过 Slack 获得明确人工审批的情况下执行破坏性命令。

大部分难点在于范围界定和安全，而非推理。智能体需要一个默认只读的 RBAC 面、一个加固的 MCP 工具服务器，以及每个考虑过 vs 执行过的命令的审计日志。它需要知道何时超出能力范围并升级。而且它必须足够便宜运行，OOM-kill 级联不会产生 $5k 的智能体账单。

## 概念

智能体在一个知识图谱上操作。节点是 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC）加遥测源（Prometheus series、Loki streams、Tempo traces）。边编码所有权（Pod -> ReplicaSet -> Deployment）、调度（Pod -> Node）和观测（Pod -> Prometheus series）。图谱通过 kube-state-metrics 同步保持新鲜，每次告警时重新采样。

当告警触发时，智能体从受影响对象开始根因分析。它遍历边，拉取相关遥测切片（最近 15 分钟），并起草假设。假设按证据排序：有多少遥测引用支持它、多近期、多具体。前 3 个假设带图路径可视化和修复操作审批按钮发送到 Slack。

修复是有门控的。默认允许的操作是只读的。破坏性操作（缩容、回滚、删除 Pod）需要 Slack 审批；ArgoCD 回滚 hook 需要智能体永远不持有的 auth token。审计日志记录智能体*考虑过*的每个命令——不仅仅是执行过的——这样审查过程能捕获险些发生的事故。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 技术栈

- 可观测性源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱：Neo4j（托管）或 kuzu（嵌入式），存储 K8s 对象 + 遥测边
- 智能体：LangGraph 带每工具允许列表，默认只读
- 工具传输：FastMCP over StreamableHTTP；破坏性工具在审批门后的独立服务器
- 模型：Claude Sonnet 4.7 用于根因推理，Gemini 2.5 Flash 用于日志摘要
- 修复：ArgoCD 回滚 webhook、PagerDuty 升级、Slack 审批卡片
- 审计：append-only 结构化日志（considered、executed、approved、outcome）
- 部署：K8s deployment 带自己的窄 RBAC 角色；独立 namespace

## 构建步骤

1. **图摄入。** 每 30 秒将 kube-state-metrics 同步到 Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测叠加边：OBSERVED_BY（一个 Pod 被一个 Prometheus series 观测）。

2. **告警接收器。** FastAPI 端点接受 PagerDuty 或 Alertmanager webhook。提取受影响对象和 SLO 违规。

3. **只读工具面。** 通过 FastMCP 封装 kubectl、Prometheus query、Loki logql、Tempo traceql。每个工具有窄 RBAC 动词（"get"、"list"、"describe"）。默认服务器中没有"delete"、"exec"、"scale"。

4. **根因智能体。** LangGraph 三节点：`sample` 拉取最近 15 分钟遥测切片，`walk` 查询图中相邻对象，`hypothesize` 起草带遥测引用的排序根因候选。

5. **证据评分。** 每个假设的评分 = 近期性 * 具体性 * 图路径长度倒数 * 引用数量。返回前 3 个。

6. **Slack 简报。** 发布一个附件，包含假设、图路径可视化（服务端渲染的子图图像）和最多一个修复操作的审批按钮。

7. **修复门控。** 破坏性工具（缩容、回滚、删除）在审批 token 后的第二个 MCP 服务器上。智能体只有在 Slack 卡片被人工审批后才能调用它们。

8. **审计日志。** Append-only JSONL：对每个候选命令，记录是否被考虑、是否被执行、谁审批了。每天发送到 S3。

9. **合成事件套件。** 构建 20 个场景：OOMKill 级联、DNS 抖动、HPA 抖动、PVC 满、噪声邻居、故障 sidecar、错误 ConfigMap 发布、证书轮换、image-pull backoff 等。对智能体的根因准确率和假设时间评分。

## 使用示例

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 交付标准

`outputs/skill-devops-agent.md` 是交付物。给定一个 K8s 集群和告警源，智能体产出排序根因假设和 Slack 门控的修复流程。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 场景套件上的 RCA 准确率 | 20 个合成事件中 ≥80% 正确根因 |
| 20 | 安全性 | 审计日志中破坏性操作守卫从未在无 Slack 审批时触发 |
| 20 | 假设时间 | 从告警到 Slack 简报 p50 低于 5 分钟 |
| 20 | 可解释性 | 每个假设都有图路径和遥测引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端工作 |
| **100** | | |

## 练习

1. 在 AWS DevOps Agent 演示的同三个事件上运行你的智能体。发布并排对比。报告智能体在哪里产生分歧。

2. 添加"险些发生"审计，标记智能体*考虑过*的任何在无审批情况下会是破坏性的命令。衡量一周内的险些发生率。

3. 将假设模型从 Claude Sonnet 4.7 换成自托管的 Llama 3.3 70B。衡量 RCA 准确率差异和每事件美元。

4. 构建因果过滤器：区分相关遥测尖峰和真正的根因。在 20 场景标签上训练一个小分类器。

5. 添加回滚 dry-run：对具有相同 manifest 的 staging 集群执行 ArgoCD 回滚。在 Slack 审批按钮之前在活集群中验证回滚计划。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| K8s 知识图谱 | "集群图" | 节点 = K8s 对象 + 遥测 series；边 = 所有权、调度、观测 |
| 默认只读 | "范围化 RBAC" | 智能体的 service account 只有 get/list/describe 动词；破坏性动词在审批后的独立服务器中 |
| 审计日志 | "考虑过 vs 执行过" | 每个候选命令的 append-only 记录，是否运行了，谁审批了 |
| 假设排序 | "证据评分" | 近期性 × 具体性 × 图路径长度倒数 × 引用数量 |
| Slack 审批卡片 | "HITL 门" | 带修复按钮的交互式 Slack 消息；人工点击前智能体无法继续 |
| 遥测引用 | "证据指针" | 支持声明的 Prometheus 查询、Loki 选择器或 Tempo trace URL |
| MTTR | "解决时间" | 从告警触发到 SLO 恢复的挂钟时间 |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 年的标准参考
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞品参考
- [NeuBird semantic monitoring](https://www.neubird.ai) — 语义图方法
- [Metoro AI SRE](https://metoro.io) — SLO 优先的生产框架
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考智能体编排器
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 门控修复目标
