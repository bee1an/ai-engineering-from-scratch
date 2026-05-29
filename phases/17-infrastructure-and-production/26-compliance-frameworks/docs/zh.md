# 合规框架 — SOC 2、HIPAA、GDPR、PCI-DSS、EU AI Act、ISO 42001

> 多框架覆盖是 2026 年企业交易的基本门槛。**EU AI Act**：2024 年 8 月 1 日生效。大多数高风险要求 2026 年 8 月 2 日执行。高风险系统义务罚款高达 1500 万欧元或全球年营业额 3%（Art. 99(4)）；禁止 AI 实践罚款高达 3500 万欧元或 7%（Art. 99(3)）。服务欧盟用户则全球适用。**Colorado AI Act**：2026 年 6 月 30 日生效（由 SB25B-004 从 2026 年 2 月推迟）——高风险系统影响评估，AI 决策申诉权。Virginia 对信用/就业/住房/教育类似。**SOC 2 Type II**：事实上的 B2B AI 要求（Type II，不是 Type I，对金融科技）。**GDPR**：最大的已记录 AI 特定罚款是 3050 万欧元对 Clearview AI（荷兰 DPA，2024 年 9 月）；意大利 Garante 对 OpenAI 开出 1500 万欧元（2024 年 12 月，2026 年 3 月上诉推翻）。推理时实时 PII 脱敏是可辩护标准；事后处理清理不够。**HIPAA**：医疗约束——没有 BAA 不能将 PHI 发送到外部 AI 服务。**PCI-DSS**：AI 交互层覆盖需要配置 + 合同协议，非自动。**ISO 42001**：新兴 AI 治理标准，与 ISO 27001 并列的采购要求增长中。参考档案：OpenAI 维护 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA (BAA)/FERPA、PCI-DSS（ChatGPT 支付组件）。跨框架映射减少审计疲劳：访问控制映射到 ISO 27001 A.5.15-5.18、GDPR Art. 32、HIPAA §164.312(a)。

**Type:** Learn
**Languages:** (Python optional — compliance is policy + process, not code)
**Prerequisites:** Phase 17 · 25 (Security), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## 学习目标

- 列举与 LLM 产品相关的七个 2026 框架，并将每个匹配到客户群体。
- 引用 EU AI Act 执行时间线（2024 年 8 月生效；2026 年 8 月高风险执行）和两级罚款上限（1500 万欧元 / 3% 高风险义务，3500 万欧元 / 7% 禁止实践）。
- 解释为什么事后 PII 清理对 GDPR 不够，并指出推理层实时脱敏为可辩护标准。
- 描述跨框架控制映射（如访问控制映射到 ISO 27001 A.5.15-5.18 + GDPR Art. 32 + HIPAA §164.312(a)）。

## 问题

一个企业客户的采购要求 SOC 2 Type II、GDPR、HIPAA BAA、ISO 27001 和"EU AI Act 合规声明"。你的团队有 SOC 2 Type I。距离 Type II 还有六个月，GDPR Article 30 记录还没开始。

多框架覆盖不是 LLM 问题——是企业 SaaS 问题，加上 LLM 特定叠加。2026 年的采购团队想要一个矩阵，每行一个框架、每列一个控制，而不是一个 PDF。

## 核心概念

### 七个框架

| 框架 | 范围 | LLM 特定要求 |
|------|------|-------------|
| SOC 2 Type II | B2B SaaS 基线 | 6-12 个月审计的流程控制 |
| HIPAA | 美国医疗 | 需要 BAA；PHI 未签协议不能离开基础设施 |
| GDPR | 欧盟用户 | 实时 PII 脱敏；数据主体权利；Article 30 记录 |
| PCI-DSS | 支付数据 | AI 接触支付需配置 + 合同 |
| EU AI Act | 服务欧盟用户 | 风险层级分类；高风险系统：合规评估、文档、日志 |
| Colorado AI Act | 服务科罗拉多州居民 | 影响评估；申诉权 |
| ISO 42001 | AI 治理 | 新兴；与 ISO 27001 配对 |

### EU AI Act 时间线

- 2024 年 8 月 1 日：生效。
- 2025 年 2 月 2 日：禁止 AI 实践执行。
- 2026 年 8 月 2 日：高风险系统执行（合规评估、文档、日志）。
- 2027 年 8 月：协调立法下产品中的高风险系统。

风险层级：不可接受（禁止）、高风险（合规 + 日志）、有限风险（透明度）、最小风险（无约束）。大多数 B2B LLM SaaS 是有限风险；高风险在就业、信用、教育、执法、移民、基本服务时触发。

罚款（Article 99）：高风险系统义务违规高达 1500 万欧元或全球年营业额 3%（Art. 99(4)）；禁止 AI 实践高达 3500 万欧元或 7%（Art. 99(3)）；取较高者。

### GDPR — 实时脱敏是标准

事后处理清理（LLM 看到数据后再脱敏 PII）不是可辩护姿态——模型已经看到了数据。推理层实时脱敏是 2026 年标准：

- LLM 调用前的实体识别。
- 一致性 tokenization（Mesh 方法）保留语义。
- 仅存储脱敏 prompt + 同意的 opt-in 原始数据。

近期执法：3050 万欧元对 Clearview AI（荷兰 DPA，2024 年 9 月）是迄今最大的已记录 AI 特定 GDPR 罚款；1500 万欧元对 OpenAI（意大利 Garante，2024 年 12 月）是最大的 LLM 特定罚款，但 2026 年 3 月上诉推翻，裁决仍在进一步审查中。事后处理声明在审计中失败。

### HIPAA — BAA 不是可选的

没有签署 Business Associate Agreement 不能将 PHI 发送到外部 AI 服务。三大超大规模 LLM 平台（Bedrock、Azure OpenAI、Vertex）都提供 BAA。OpenAI 直接 API 提供 BAA。Anthropic 直接 API 提供 BAA。发送 PHI 前确认。

### SOC 2 Type II

Type I：控制已设计和记录。
Type II：控制在 6-12 个月内有效运行。

2026 年 B2B 采购默认 Type II。Type I 是起步；Type II 是门槛。

常见审计驱动因素：访问日志（谁看了什么）、变更管理（如何部署的）、风险评估（季度）、事件响应（测试过吗？）。Phase 17 · 25 的审计日志可直接复用。

### 跨框架映射

一个访问控制策略满足多个框架控制：

| 控制 | 框架 |
|------|------|
| 访问日志 | ISO 27001 A.5.15-5.18, GDPR Art. 32, HIPAA §164.312(a) |
| 变更管理 | ISO 27001 A.8.32, PCI DSS Req. 6, HIPAA breach-notification scope |
| 传输加密 | ISO 27001 A.8.24, GDPR Art. 32, HIPAA §164.312(e) |
| 密钥管理 | ISO 27001 A.8.19, PCI DSS Req. 8, SOC 2 CC6.1 |

合规工具（Drata、Vanta、Secureframe）自动化此映射。规模化时值得投入。

### ISO 42001 — 新兴

2023 年底发布。与 ISO 27001 并列的采购要求增长中。AI 治理框架，包括风险管理、数据质量、透明度、人类监督。

### OpenAI 的参考档案

OpenAI 维护 SOC 2 Type 2、ISO/IEC 27001:2022、ISO/IEC 27701:2019、GDPR/CCPA/HIPAA (BAA)/FERPA、PCI-DSS（ChatGPT 支付组件）。这大致是 2026 年的企业基本门槛。

### 需要记住的数字

- EU AI Act 罚款：高达 1500 万欧元 / 3%（高风险义务，Art. 99(4)）；高达 3500 万欧元 / 7%（禁止实践，Art. 99(3)）。
- EU AI Act 高风险执行：2026 年 8 月 2 日。
- 最大已记录 AI 特定 GDPR 罚款：3050 万欧元，Clearview AI（荷兰 DPA，2024 年 9 月）。
- 最大 LLM 特定 GDPR 罚款：1500 万欧元，OpenAI（意大利 Garante，2024 年 12 月；2026 年 3 月上诉推翻）。
- SOC 2 Type II 窗口：6-12 个月运行的控制。
- Colorado AI Act 生效日期：2026 年 6 月 30 日（由 SB25B-004 从 2026 年 2 月推迟）。

## Use It

`code/main.py` 是一个 Python 合规映射表——给定一个控制，列出它满足的框架。

## Ship It

本课产出 `outputs/skill-compliance-matrix.md`。给定客户群体和地理位置，指定所需框架和控制。

## 练习

1. 你的第一个企业客户要求 SOC 2 Type II、HIPAA BAA、EU AI Act 声明。赢得交易的最小可行合规姿态是什么？
2. 将三个假设的 LLM 产品按 EU AI Act 风险层级分类。高风险时有什么变化？
3. 你意外将 PHI 发送到没有 BAA 的供应商。走一遍事件响应流程。
4. 论证 ISO 42001 对 2026 年的中型 AI 供应商是否"必要"。
5. 将你的 LLM 审计日志字段（Phase 17 · 25）映射到至少三个框架控制。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| SOC 2 Type II | "审计过的控制" | 6-12 个月运行的控制，独立证明 |
| HIPAA BAA | "医疗合同" | Business Associate Agreement；PHI 必需 |
| GDPR | "欧盟隐私" | 实时 PII 脱敏是 2026 年可辩护标准 |
| EU AI Act | "欧盟 AI 规则" | 2026 年 8 月高风险执行；1500 万欧元 / 3%（高风险义务）— 3500 万欧元 / 7%（禁止实践） |
| Colorado AI Act | "美国 AI 州法" | 2026 年 6 月 30 日生效（SB25B-004 推迟）；影响评估 |
| ISO 42001 | "AI 治理" | AI 风险 + 透明度的新兴框架 |
| ISO 27001 | "安全 ISMS" | 信息安全管理体系基线 |
| Conformity assessment | "EU AI 文档包" | 高风险要求：文档、测试、日志 |
| Cross-framework mapping | "一个控制，多个框架" | 单一策略满足多个框架控制 |

## 延伸阅读

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — reference compliance profile.
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — primary source.
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — primary source.
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI management system standard.
