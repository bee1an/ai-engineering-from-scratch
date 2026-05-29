# CAIS、CAISI 与社会级风险

> Center for AI Safety（CAIS，旧金山，2022 年由 Hendrycks 和 Zhang 创立）发布了四风险框架——恶意使用、AI 竞赛、组织风险、失控 AI——以及 2023 年 5 月由数百位教授和公司领导人签署的灭绝风险声明。CAIS 2026 年发布：AI Dashboard 用于前沿模型评估、Remote Labor Index（与 Scale AI 合作）、Superintelligence Strategy Paper、AI Frontiers newsletter。一个不同的实体：NIST Center for AI Standards and Innovation (CAISI)——面向美国政府的自愿协议和非机密能力评估，聚焦 cyber、bio 和化学武器风险。CAIS 将组织风险标记为四大顶级风险之一：安全文化、严格审计、多层防御和信息安全是基础性的，但经常被牺牲以换取部署速度。加州 SB-53 如果签署，将成为美国首个州级灾难性风险法规。

**Type:** Learn
**Languages:** Python (stdlib, four-risk inventory and mitigation matcher)
**Prerequisites:** Phase 15 · 19 (RSP), Phase 15 · 20 (PF + FSF)
**Time:** ~45 minutes

## 问题

Lessons 19 和 20 覆盖了实验室内部的 scaling policies。Lesson 21 覆盖了独立的能力评估。本课覆盖第三个视角：塑造公共讨论和灾难性 AI 风险监管基线的民间社会和政府组织。

两个不同的实体很重要。CAIS 是一个非营利研究组织，发布 AI 风险思考框架并协调公开声明。CAISI 是 NIST 内的美国政府中心，与实验室运行自愿协议并发布非机密能力评估。名字押韵；使命不重叠。从业者应该了解两者。

实用内容：CAIS 的四风险框架是文献中被引用最多的社会级风险分类法。安全文化和组织风险是四者之一，也是从业者最能直接控制的。SB-53（加州）如果签署将成为美国首个州级灾难性风险法规；该法案的框架很重要，因为州级法规在美国科技政策中历来引领联邦行动。

## 概念

### CAIS — Center for AI Safety

- 成立：2022 年，旧金山，由 Dan Hendrycks 和同事创立（"Zhang"指早期合作者，非当前联合创始人；见 CAIS 网站了解当前领导层）。
- 状态：501(c)(3) 非营利组织。
- 2023 年重要产出：灭绝风险声明，由数百位研究者和 CEO 联署。声明："减轻 AI 带来的灭绝风险应成为与大流行病和核战争并列的全球优先事项。"
- 2026 年产出：AI Dashboard 用于前沿模型评估、Remote Labor Index（与 Scale AI 联合）、Superintelligence Strategy Paper、AI Frontiers newsletter。

### 四风险框架

CAIS 的框架将灾难性 AI 风险分为四个顶级类别：

1. **恶意使用**：恶意行为者使用 AI 造成伤害（生物武器合成、虚假信息、网络攻击）。
2. **AI 竞赛**：实验室、公司或国家之间的竞争压力将部署推过安全点。
3. **组织风险**：内部实验室动态（安全文化失败、审计不足、安全资源不足）导致糟糕的部署。
4. **失控 AI**：足够有能力的 AI 追求与人类福祉冲突的目标。

这不是唯一的分类法；它是被引用最多的。类别不互斥——一个由在竞赛中以审计换速度的组织产生的失控 AI 同时属于四个类别。

### 组织风险的位置

四个类别中，组织风险对从业者最可操作。实验室的安全文化、审计严格性、防御分层和信息安全决定了其模型是否带着 Lessons 10–18 的控制措施实际到位地发布，还是那些控制措施只是没人验证的清单项。

具体的组织风险杠杆：

- **安全文化**：团队成员是否能在没有职业代价的情况下升级关切？CAIS 调查发现这是其他杠杆的强预测因子。
- **严格审计**：外部和内部。仅内部审计产生乐观报告。
- **多层防御**：没有单一层是充分的（Phase 15 的贯穿主题）。
- **信息安全**：模型权重泄露、评估数据泄露、监控绕过技术泄露。Lesson 19 中的 RAND SL-4 是一个具体标准。

### CAISI — Center for AI Standards and Innovation

- 在 NIST 内运营。
- 与前沿实验室运行自愿协议。
- 发布聚焦 cyber、bio 和化学武器风险的非机密能力评估。
- 与 CAIS 不同；缩写碰撞；检查 URL（nist.gov）确认你在读哪个。

CAISI 的角色是 METR 私有实验室合作（Lesson 21）的公共、面向政府的对应物。CAISI 报告是非机密的；METR 报告通常受 NDA 限制。同时阅读两者的从业者能获得更完整的图景。

### 加州 SB-53

加州参议院法案（2025–2026 会期）针对前沿模型的灾难性风险。草案中的关键条款：

- 触发州级义务的具体能力阈值。
- AI 实验室员工的举报人保护。
- 灾难性故障的事件报告要求。

如果签署，将成为美国首个州级灾难性风险法规。无论是否签署，该法案的框架都影响其他州立法机构如何处理这个问题。加州的从业者应追踪法案状态；其他地方的从业者应阅读它以了解美国州级法规可能的样子。

### 社会级风险不是单层问题

Phase 15 的贯穿主题——纵深防御——也适用于社会层面。没有单一组织、法规或框架能关闭灾难性风险。生态系统只有在以下条件下才能运作：

- 实验室发布 scaling policies（Lessons 19、20）。
- 外部评估者产出测量（Lesson 21）。
- 民间社会追踪和公开（CAIS）。
- 政府运行自愿项目和基线法规（CAISI、SB-53）。
- 从业者构建多层控制（Lessons 10–18）。

这是本 phase 的最终综合：每个前面的课程都是一个栈中的一层，其完整性比任何单层的强度更重要。

## Use It

`code/main.py` 实现了一个小型风险清单工具。给定一个拟议部署，它将部署标记到四风险类别并返回缓解清单。这是框架的阅读辅助工具，不是人类判断的替代品。

## Ship It

`outputs/skill-societal-risk-review.md` 审查部署的社会级风险态势：它触及四个类别中的哪些，有哪些缓解措施到位，组织风险暴露是什么。

## 练习

1. 运行 `code/main.py`。输入三个不同规模的合成部署。确认四风险标签与你的预期匹配；识别一个工具标记不足或过度的案例。

2. 完整阅读 CAIS 四风险论文。选择一个风险类别，写两段关于你认为该类别中 2026 年最重要发展的内容。

3. 阅读加州 SB-53 的当前草案。识别一个你认为加强灾难性风险态势的条款和一个你认为削弱它的条款。论证两者。

4. 选择一个你了解的生产 AI 部署（你自己的或已发布的）。按组织风险子杠杆评分：安全文化、审计严格性、多层防御、信息安全。哪个最弱？将其提升到标准需要什么成本？

5. 勾画一个 2028 版四风险框架，反映一年的额外能力和一年的额外部署经验。你会添加、移除或重组什么？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| CAIS | "Center for AI Safety" | 非营利；四风险框架；2023 年灭绝声明 |
| CAISI | "美国政府 AI 安全" | NIST 中心；自愿协议；非机密评估 |
| Four-risk framework | "CAIS 的分类法" | 恶意使用、AI 竞赛、组织风险、失控 AI |
| Malicious use | "恶意行为者使用 AI" | 生物武器、虚假信息、网络攻击 |
| AI races | "竞争压力" | 实验室/公司/国家将部署推过安全点 |
| Organizational risk | "实验室内部失败" | 安全文化、审计、防御、信息安全 |
| Rogue AI | "失对齐的 agent" | 有能力的 AI 追求与人类福祉冲突的目标 |
| California SB-53 | "州级法规" | 2025–2026 法案；如签署则为美国首个州级灾难性风险法规 |

## 延伸阅读

- [Center for AI Safety](https://safe.ai/) — 四风险框架的机构主页。
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) — 四风险论文。
- [CAIS — May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk) — 简短联合声明。
- [NIST CAISI](https://www.nist.gov/caisi) — 面向政府的 AI 标准与创新中心。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将实验室级承诺与社会级框架连接。
