# 安全 — 密钥管理、API Key 轮换、审计日志、护栏

> 通过集中式 vault（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除密钥散落。永远不要将凭证存储在配置文件、VCS 中的 env 文件、电子表格中。使用 IAM 角色而非静态密钥；CI/CD 使用 OIDC。AI gateway 模式是 2026 年的解决方案：应用 → 网关 → 模型供应商，网关在运行时从 vault 拉取凭证。在 vault 中轮换，所有应用几分钟内获取新密钥——无需重新部署，无需 Slack "谁有新 key"消息。轮换策略 ≤90 天；每次提交用 TruffleHog / GitGuardian / Gitleaks 扫描。零信任：MFA、SSO、RBAC/ABAC、短期 token、设备姿态。PII 清洗使用实体识别在转发前遮蔽 PHI/PII；一致性 tokenization（Mesh 方法）将敏感值映射到稳定占位符，使 LLM 保留代码/关系语义。网络出口：LLM 服务在专用 VPC/VNet 子网中，白名单仅 `api.openai.com`、`api.anthropic.com` 等；阻止所有其他出站。2026 年事件驱动因素：Vercel 供应链攻击通过被入侵的 CI/CD 凭证窃取了数千客户部署的环境变量。

**Type:** Learn
**Languages:** Python (stdlib, toy PII-scrubber + audit-log writer)
**Prerequisites:** Phase 17 · 19 (AI Gateways), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## 学习目标

- 列举四种密钥管理反模式（VCS 中的配置文件、硬编码 env、电子表格、静态密钥）并说明其替代方案。
- 解释 AI-gateway-从-vault-拉取模式作为 2026 年生产标准。
- 实现带一致性 tokenization（相同值 → 相同占位符）的 PII 清洗器，使语义得以保留。
- 说明 2026 年 Vercel 供应链事件及其对 CI/CD 凭证卫生的教训。

## 问题

一个实习生提交了带 API key 的 `.env`。他们很快删除了。但 key 已经在 git 历史中——GitGuardian 扫描捕获了它，你的轮换流程是"Slack 通知团队，更新 40 个配置文件，重新部署所有服务。"8 小时后，一半服务已上线，另一半在等部署窗口。

另外，用户 prompt 包含"My SSN is 123-45-6789。"Prompt 发送到 OpenAI。你有 BAA 但内部策略是转发前遮蔽 PII。你没做到。

另外，你的 EKS 集群的 LLM pod 可以访问任何互联网主机。有人通过 DNS 查询到攻击者控制的域名窃取数据。没有任何东西阻止它。

LLM 服务的安全必须解决所有三个向量。Vault 支持的凭证。PII 清洗。网络出口过滤。审计日志。

## 核心概念

### 集中式 vault + IAM 角色拉取

**Vault**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。单一事实来源。

**IAM 角色**：应用/网关通过其 IAM 身份认证，而非静态密钥。Vault 在 token 生命周期内返回密钥。

**AI gateway 模式**：网关在请求时从 vault 拉取 `OPENAI_API_KEY`。在 vault 中轮换；下一个请求获取新密钥。无需重新部署。

### 轮换策略 ≤ 90 天

所有 API key、vault root token、CI/CD 凭证。尽可能自动轮换。手动轮换需记录和跟踪。

### 密钥扫描

- **TruffleHog** — 对提交做正则 + 熵检测。
- **GitGuardian** — 商业，高准确率。
- **Gitleaks** — 开源，在 CI 中运行。

每次提交运行。检测到新密钥时阻止 PR。

### 零信任姿态

- 所有账户要求 MFA。
- 通过 SAML/OIDC 的 SSO。
- RBAC（基于角色）或 ABAC（基于属性）的细粒度访问。
- 短期 token（小时级，非天级）。
- 设备姿态——仅允许带磁盘加密的公司设备。

### PII / PHI 清洗

在 prompt 离开你的基础设施之前：

1. 实体识别（spaCy NER、Presidio、商业方案）。
2. 遮蔽匹配实体：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. 一致性 tokenization（Mesh 方法）：相同值映射到相同占位符，使 LLM 保留关系。
4. 可选的 LLM 响应反向映射。

静态正则过滤器捕捉基本模式；NER 捕捉更多。两者都用。

### 输入 + 输出护栏

输入：阻止已知越狱、禁止话题；按用户限流。

输出：正则清洗泄露的密钥（API key 模式、拒绝上下文中的邮箱模式），策略违规分类器。

### 网络出口白名单

LLM 服务在专用子网中：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、vault 端点。
- 其他所有：丢弃。
- DNS 通过仅允许列表解析器（避免 DNS 隧道窃取）。

### 审计日志

每次 LLM 调用的不可变日志：
- 时间戳。
- 用户/租户。
- Prompt 哈希（非原始 prompt，保护隐私）。
- 模型 + 版本。
- Token 数量。
- 成本。
- 响应哈希。
- 任何护栏触发。

按监管要求保留（SOC 2 1 年，HIPAA 6 年）。

### 2026 年 Vercel 事件

供应链攻击：被入侵的 CI/CD 凭证窃取了数千客户部署的环境变量。教训：CI/CD 凭证等同于生产凭证。存储在 vault 中。范围尽量窄。积极轮换。

### 需要记住的数字

- 轮换策略：≤ 90 天。
- 每次提交扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD 凭证被入侵 → 数千客户环境变量泄露。
- 审计日志保留：SOC 2 = 1 年，HIPAA = 6 年。

## Use It

`code/main.py` 实现一个带一致性 tokenization 的玩具 PII 清洗器和一个追加式审计日志。

## Ship It

本课产出 `outputs/skill-llm-security-plan.md`。给定监管范围和当前状态，规划 vault 迁移、清洗器、出口、审计日志。

## 练习

1. 运行 `code/main.py`。发送两个引用相同 SSN 的 prompt。确认两者获得相同占位符。
2. 为调用 OpenAI + Anthropic + Weaviate 的 vLLM-on-EKS 部署设计网络出口策略。
3. 你发现 git 历史中有一个密钥（2 年前的）。正确响应是什么——轮换密钥、清洗历史、还是两者？论证。
4. 你的审计日志每天增长 10 GB。设计保留层级（热 30 天、温 12 月、冷 6 年）。
5. 论证反向 tokenization（将真实值替换回 LLM 响应）是否值得复杂性，vs 保持占位符可见。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| Vault | "密钥存储" | 集中式凭证管理服务 |
| IAM role | "基于身份的认证" | 应用承担的角色；返回短期凭证 |
| OIDC for CI/CD | "云签发 token" | CI 中无静态密钥——通过 OIDC 获取身份 |
| TruffleHog / GitGuardian / Gitleaks | "密钥扫描器" | 提交时密钥检测 |
| RBAC / ABAC | "访问控制" | 基于角色 vs 基于属性 |
| PII scrubbing | "数据遮蔽" | 移除或 tokenize 敏感实体 |
| Consistent tokenization | "稳定占位符" | 相同值 → 每次相同 token |
| Mesh approach | "Mesh tokenization" | 保留语义的 tokenization 模式 |
| Egress whitelist | "出站允许列表" | 仅允许的域名可达 |
| Audit log | "不可变历史" | 用于合规的追加式记录 |

## 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII detection and anonymization.
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
