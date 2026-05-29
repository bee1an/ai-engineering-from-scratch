# Constitutional AI 与规则覆盖

> Anthropic 2026 年 1 月 22 日发布的 Claude Constitution 长达 79 页，采用 CC0 许可。它从基于规则的对齐转向基于推理的对齐，并建立了四层优先级层次：(1) 安全与支持人类监督，(2) 伦理，(3) Anthropic 指南，(4) 有用性。行为分为硬编码禁止项（生物武器提升、CSAM）——运营商和用户都无法覆盖——以及软编码默认值——运营商可在定义的范围内调整。2022 年的原始版本（Bai et al.）通过自我批评和基于宪法的 RLAIF 训练无害性。诚实的注脚：基于推理的对齐依赖模型将原则泛化到未预见的情况。Anthropic 自己 2023 年的参与式实验显示公众来源和企业原则之间有约 50% 的分歧；2026 年版本没有纳入这些发现。

**Type:** Learn
**Languages:** Python (stdlib, 四层优先级解析器)
**Prerequisites:** Phase 15 · 06 (Automated alignment research), Phase 15 · 10 (Permission modes)
**Time:** ~60 minutes

## 问题

一个部署中的 agent 会看到其设计者从未见过的输入。没有任何规则列表长到能覆盖所有情况。没有任何规则列表短到能在计算压力下快速应用。实际问题是：如何让 agent 对齐到既能覆盖长尾案例又能快速推理的原则？

基于规则的对齐（RBA）：列出每个不允许的事情。检查快、审计容易、不可能保持最新、经常对未预见的近似情况过度拒绝。基于推理的对齐（2026 Claude Constitution）：编码原则，让模型推理。能扩展到未见案例，更难审计，失败模式是原则误用而非遗漏规则。

2026 Constitution 采取了明确的中间立场。硬编码禁止项——其错误性不依赖于上下文的事物（生物武器提升、CSAM）——是 RBA：永远不行，无论运营商或用户指令如何。其他一切都是在四层层次结构内基于推理的：安全与支持人类监督优先；伦理第二；Anthropic 声明的指南第三；有用性最后。运营商可以在软编码区域内调整默认值，但不能触碰硬编码禁止项。

## 概念

### 四层优先级层次

1. **安全与支持人类监督。** 最高。模型优先不破坏人类和 Anthropic 监督和纠正 AI 的能力。这不是"要谨慎"；而是具体的"不要以使人类监督更困难的方式行事"。
2. **伦理。** 诚实、避免对人的伤害、不欺骗、不操纵。当与 Anthropic 指南冲突时优先于后者。
3. **Anthropic 指南。** Anthropic 决定重要的操作规范：产品范围、交互模式、何时使用什么工具。
4. **有用性。** 最低。在更高优先级内尽可能有用。

当层级冲突时，更高的获胜。这和 Unix 优先级或网络 QoS 是同一个形状——这个框架旨在产生可预测的解决方案，而不一定是任何单一轴上的最佳行为。

### 硬编码禁止项 vs 软编码默认值

**硬编码：**
- 生物武器 / CBRN 提升
- CSAM
- 对关键基础设施的攻击
- 当被直接询问时欺骗用户关于模型身份

运营商不能覆盖这些。用户不能覆盖这些。它们在可能的情况下在模型权重层面执行（RLHF / Constitutional AI 训练），在不可能的情况下在推理层执行。

**软编码默认值（运营商可调整）：**
- 响应长度默认值
- 主题范围（模型可以拒绝运营商部署范围外的主题）
- 风格（正式 vs 随意）
- 工具使用模式

运营商调整发生在声明的范围内。运营商不能通过重命名来移除硬编码禁止项。

### 2022 CAI 训练

原始的 Constitutional AI（Bai et al., 2022）训练无害性：

1. 对一组提示生成响应。
2. 让模型根据宪法（显式原则）批评每个响应。
3. 基于批评修改响应。
4. 对修改后的配对进行 RLAIF（来自 AI 反馈的强化学习）。

结果：一个用有原则的解释拒绝有害请求的模型，而不是一刀切的拒绝。2026 Constitution 使用这种训练的后代加上对显式层级层次的额外后训练。

### 基于推理的对齐能捕获和遗漏什么

**能捕获：**
- 允许的原语的未预见组合，其中原则明确适用。
- 与被禁止请求相近的新颖请求。
- 依赖"你没说 X 是不允许的"的社会工程攻击。

**会遗漏：**
- 利用原则模糊性的攻击（"用户要求了这个所以有用性说可以"）。
- 两个原则以未预见的方式冲突，且层级顺序模糊的场景。
- 训练周期中原则解释的缓慢漂移（重新解释）。

### 2023 参与式实验

Anthropic 在 2023 年进行了一个实验，比较企业编写的宪法和通过公众输入（约 1,000 名美国受访者）生成的宪法。两个版本在约 50% 的原则上一致。在分歧之处，公众来源版本在某些问题上更严格（政治内容处理），在其他问题上更宽松（AI 身份的自我披露）。2026 Constitution 没有纳入公众来源的发现。这是该方法中一个有据可查的张力。

### 为什么硬编码禁止项是必要的

仅靠基于推理的对齐无法封闭尾部。一个能让模型接受前提的攻击者（例如"我们是一个持牌生物武器研究实验室"）通常可以绕过依赖案例推理的原则。硬编码禁止项不会因前提框架而弯曲。它们是对齐层面的第 14 课"硬宪法限制"。

### Constitution 在栈中的位置

Constitution 不是第 14 课的 kill switch。它存在于模型层：模型权重被训练偏好什么。Kill switch 和 canary token 存在于运行时层：运行时允许什么。两者都是必需的。一个因为模型权重过于宽容而触发所有错误动作的运行时是运行时问题。一个因为运行时过于严格而拒绝所有正确动作的模型是运行时问题。不同层覆盖不同类别。

## Use It

`code/main.py` 实现了一个最小的四层优先级解析器。解析器接受一个提议的动作和一组原则评估（安全、伦理、指南、有用性），返回该动作、一个拒绝或一个修改后的动作。驱动程序运行一个小案例集：明确允许、明确禁止、硬编码禁止项、跨层级的模糊案例。

## Ship It

`outputs/skill-constitution-review.md` 审计一个部署的宪法层：什么是硬编码的、什么是软编码的、运营商可以在哪里调整、四层层次是否确实是解决顺序。

## 练习

1. 运行 `code/main.py`。确认即使有用性很高，硬编码禁止项也会触发。修改解析器使有用性权重高于伦理；观察失败模式。

2. 阅读 Claude Constitution（公开，79 页，CC0）。找出一个你认为定义不足的原则。写两段话解释具体的模糊性并提出更紧凑的表述。

3. 为客服 agent 设计一组软编码默认值。运营商调整什么？运营商不能触碰什么？论证每个边界。

4. 阅读 Bai et al. 2022 CAI 论文。描述一个 Constitutional AI 的批评-修改循环会产生比一刀切规则更差结果的案例。识别该类别。

5. Anthropic 2023 年的参与式实验发现公众和企业原则之间有约 50% 的分歧。选择一个对生产部署重要的类别（例如政治中立性）。提出一个设计，让运营商表达自己的价值观，同时硬编码禁止项保持不变。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|---|---|---|
| Constitutional AI | "Anthropic 的对齐方法" | 自我批评 + 基于书面宪法的 RLAIF |
| Reason-based alignment | "原则而非规则" | 模型基于原则推理以处理未见案例 |
| Hardcoded prohibition | "永远不做 X" | 基于规则的禁止项，运营商或用户都无法覆盖 |
| Soft-coded default | "运营商可调整" | 在声明范围内的行为，运营商控制 |
| Four-tier hierarchy | "优先级顺序" | 安全 > 伦理 > 指南 > 有用性 |
| RLAIF | "AI 反馈 RL" | 奖励来自模型生成的批评的强化学习 |
| Participatory constitution | "公众来源的原则" | 2023 Anthropic 实验；与企业版约 50% 分歧 |
| Principle drift | "解释滑移" | 模型对固定原则文本的解读缓慢变化 |

## 延伸阅读

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 79 页 CC0 文档。
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — 2022 原始论文。
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — 参与式实验。
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — Constitution 在 RSP 栈中的位置。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Constitution 在长时域部署中的角色。
