# 托管 LLM 平台 — Bedrock、Vertex AI、Azure OpenAI

> 三大云厂商，三种截然不同的策略。AWS Bedrock 是模型市场 — Claude、Llama、Titan、Stability、Cohere 统一在一个 API 后面。Azure OpenAI 是与 OpenAI 的独家合作加上 Provisioned Throughput Units (PTUs) 提供专属算力。Vertex AI 以 Gemini 为核心，拥有最好的长上下文和多模态能力。2026 年 Artificial Analysis 的测量显示，Azure OpenAI 在 Llama 3.1 405B 等效部署上中位延迟约 50 ms，Bedrock 约 75 ms — PTU 解释了这个差距，因为专属算力胜过共享按需。决策规则不是"哪个最快"，而是"哪个模型目录和 FinOps 界面匹配我的产品"。这节课教你把权衡写下来再做选择，而不是凭感觉。

**Type:** Learn
**Languages:** Python (stdlib, toy cost-and-latency comparator)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols)
**Time:** ~60 minutes

## 学习目标

- 说出三种平台策略（市场型 vs 独家型 vs Gemini 优先型），并将每种匹配到一个产品用例。
- 解释 Provisioned Throughput Units (PTUs) 在 Azure OpenAI 中买到了什么，以及为什么按需 Bedrock 在 405B 规模上通常慢约 25 ms。
- 画出每个平台的 FinOps 归因面（Bedrock Application Inference Profiles vs Vertex project-per-team vs Azure scopes + PTU reservations）。
- 写出"双供应商最低标准"策略，并解释为什么单一厂商锁定是 2026 年最贵的错误。

## 问题

你为产品选了 Claude 3.7 Sonnet。现在需要部署它。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者走网关。直接 API 最简单；Bedrock 增加了 BAA、VPC 端点、IAM 和 CloudWatch 归因。网关增加了故障转移、统一计费和跨供应商限流。

更深层的问题是目录。如果你的产品同时需要 Claude、Llama 和 Gemini，你无法从一个地方全部买到，除非那个地方是 Bedrock 加 Vertex 加 Azure OpenAI 同时使用。云厂商不可互换 — 它们各自在谁拥有模型层上做了不同的赌注。

这节课梳理三个赌注、延迟差距、FinOps 差距和锁定风险。

## 概念

### 三种策略

**AWS Bedrock** — 市场型。Claude (Anthropic)、Llama (Meta)、Titan (AWS 自研)、Stability (图像)、Cohere (嵌入)、Mistral，加上图像和嵌入子目录。一个 API、一个 IAM 面、一个 CloudWatch 导出。Bedrock 的赌注是客户更想要选择权而非单一模型。

**Azure OpenAI** — 独家合作型。你得到 GPT-4 / 4o / 5 / o-series、DALL·E、Whisper，以及在 Azure 数据中心对 OpenAI 模型的微调。"Azure OpenAI Service" 目录中没有非 OpenAI 模型 — 那些在 Azure AI Foundry（独立产品）。Azure 的赌注是 OpenAI 保持前沿，客户想要对这个特定关系的企业级管控。

**Vertex AI** — Gemini 优先，其他次之。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，加上 Model Garden（第三方）。Vertex 的赌注是多模态长上下文 — 1M token 的 Gemini 上下文是差异化优势。

### 规模化延迟差距

Artificial Analysis 持续运行基准测试。在等效的 Llama 3.1 405B 部署（共享按需）上，Azure OpenAI 中位首 token 延迟约 50 ms；Bedrock 约 75 ms。这个差距不是 AWS 的失败 — 而是容量模型的差异。Azure 卖 PTU（Provisioned Throughput Units），为你的租户预留 GPU 算力。Bedrock 的等效产品（Provisioned Throughput）存在但起步约 $21/小时每单位，大多数客户留在共享按需上。

共享按需容量与其他所有客户的流量竞争。专属容量不会。如果你的产品 SLA 是 TTFT < 100 ms at P99，你要么在 Azure 上买 PTU，要么买 Bedrock Provisioned Throughput，要么接受默认方差。

### Provisioned Throughput 经济学

Azure PTU：一块预留的推理算力。对可预测工作负载最高可节省约 70%。无论流量如何，每小时固定费用 — 即使空闲也要付费。盈亏平衡通常在 40-60% 持续利用率左右。

Bedrock Provisioned Throughput：$21-$50/小时，取决于模型和区域。类似的数学 — 盈亏平衡在峰值利用率的一半左右。需要月度承诺。

Vertex 预留容量按 Gemini SKU 出售；定价因模型和区域而异，公开程度较低。

### FinOps 面 — 真正的差异化

**Bedrock Application Inference Profiles** 是市场中最干净的归因。给 profile 打上 `team`、`product`、`feature` 标签；将所有模型调用路由通过它；CloudWatch 按 profile 拆分成本，无需后处理。2025 年添加，仍是最细粒度的云厂商原生方案。

**Vertex** 归因是 project-per-team 加上到处打标签。你把每个团队建模为一个 GCP project，给每个资源打标签，用 BigQuery Billing Export + DataStudio 做汇总。工作量更大，但 BigQuery 让你对成本数据做任意 SQL。

**Azure** 依赖 subscription/resource-group scopes 加标签，PTU 预留作为一等成本对象。标签从 resource group 继承，不是从请求继承，所以按请求归因需要 Application Insights 自定义指标或一个打 header 的网关。

模式：Bedrock 原生最干净，Vertex 通过 BigQuery 最灵活，Azure 最不透明除非你做了埋点。

### 锁定是 2026 年的风险

单一云厂商承诺在一个模型主导时还行。2026 年前沿每月都在变 — 这个季度 Claude 3.7，下个季度 Gemini 2.5，再下个季度 GPT-5。锁定一个平台就锁死了三分之二的前沿。

团队采用的模式：对任何产品关键 LLM 调用实行双供应商最低标准。Bedrock 加 Azure OpenAI 是常见组合 — 一个出 Claude，另一个出 GPT，两者之间故障转移，同一个网关。成本增加可忽略因为网关路由最优；宕机期间的可用性提升（如 Azure OpenAI 2025 年 1 月事件、AWS us-east-1 宕机）是决定性的。

### 数据驻留、BAA 和受监管行业

Bedrock：大多数区域有 BAA；VPC 端点；guardrails。常见的金融科技默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；EU 数据驻留；企业受监管的默认选择。
Vertex：HIPAA、GDPR、按区域数据驻留；Google Cloud 的合规栈。

三者都满足基本合规要求。差异在数据保留策略、日志处理方式，以及滥用监控是否读取你的流量（大多数默认开启；企业可选退出）。

### 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等效上的中位 TTFT：约 50 ms（使用 PTU）。
- Bedrock 按需中位 TTFT：约 75 ms。
- Bedrock Provisioned Throughput：$21-$50/hr 每单位。
- Azure PTU 盈亏平衡：约 40-60% 持续利用率。
- PTU 在高利用率下相比按需的节省：最高 70%。

## Use It

`code/main.py` 在合成工作负载上比较三个平台 — 建模按需 vs PTU 经济学、TTFT 方差和成本归因保真度。运行它看看 PTU 在哪里划算，以及市场的模型广度在哪里胜过 TTFT 差距。

## Ship It

本课产出 `outputs/skill-managed-platform-picker.md`。给定工作负载画像（所需模型、TTFT SLA、日均量、合规要求），推荐主平台、备选平台和 FinOps 埋点方案。

## 练习

1. 运行 `code/main.py`。在什么持续利用率下 Azure PTU 对 70B 级模型胜过按需？计算盈亏平衡并与宣传的 40-60% 区间对比。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个双供应商部署 — 哪个放哪个云厂商，前面放什么网关，故障转移策略是什么？
3. 一个受监管的医疗客户要求 BAA、US-East 数据驻留和 sub-100ms P99 TTFT。选一个平台并用三个具体特性论证。
4. 你发现 Bedrock 账单本月涨了 4 倍但流量没变。没有 Application Inference Profiles 时你怎么找到元凶？有 profiles 时需要多久？
5. 阅读 Azure OpenAI 和 Bedrock 定价页。对于每月 100M token 的 Claude 工作负载，哪个更便宜 — 直接 Anthropic API、Bedrock 按需、还是 Bedrock Provisioned Throughput？

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| Bedrock | "AWS LLM 服务" | 跨 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | "Azure 的 ChatGPT" | Azure 数据中心中的独家 OpenAI 模型加企业管控 |
| Vertex AI | "Google 的 LLM" | Gemini 优先平台加 Model Garden 第三方模型 |
| PTU | "专属算力" | Provisioned Throughput Unit — 预留推理 GPU，按小时计费 |
| Application Inference Profile | "Bedrock 标签" | 按产品的成本/用量 profile，带标签，CloudWatch 原生 |
| Model Garden | "Vertex 目录" | Vertex AI 的第三方模型区，与 Gemini 分开 |
| Two-provider minimum | "LLM 冗余" | 每条关键 LLM 路径跨 ≥2 个云厂商运行的策略 |
| BAA | "HIPAA 文书" | Business Associate Agreement；PHI 必需；三家都提供 |
| Abuse monitoring | "日志监控" | 供应商侧对 prompt/output 的安全扫描；企业可选退出 |

## 延伸阅读

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — 权威费率卡和 Provisioned Throughput 定价。
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU 经济学和费率卡。
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 层级和 Model Garden 附加费。
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — 跨供应商的持续延迟和吞吐量基准测试。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 归因机制对比。
