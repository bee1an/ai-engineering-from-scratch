# 毕业项目 15 — 宪法式安全护栏 + 红队靶场

> Anthropic 的 Constitutional Classifiers、Meta 的 Llama Guard 4、Google 的 ShieldGemma-2、NVIDIA 的 Nemotron 3 Content Safety，以及覆盖多语言的 X-Guard，定义了 2026 年的安全分类器技术栈。garak、PyRIT、NVIDIA Aegis 和 promptfoo 成为标准的对抗性评估工具。NeMo Guardrails v0.12 将它们串联成生产流水线。本毕业项目把这一切组装起来：围绕目标应用构建分层安全护栏，运行覆盖 6+ 攻击族的自主红队智能体，并执行一次宪法式自我批评训练，产出可量化的无害性提升。

**类型：** 毕业项目
**语言：** Python（安全流水线、红队）、YAML（策略配置）
**前置课程：** Phase 10（从零构建 LLM）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 18（伦理、安全、对齐）
**覆盖阶段：** P10 · P11 · P13 · P14 · P18
**时间：** 25 小时

## 问题

2026 年 LLM 安全的前沿不在于分类器是否有效（大致有效），而在于如何正确地将它们组合到生产应用周围，既不过度拒绝，也不留下明显漏洞。Llama Guard 4 处理英文策略违规。X-Guard（132 种语言）处理多语言越狱。ShieldGemma-2 捕获基于图像的 prompt injection。NVIDIA Nemotron 3 Content Safety 覆盖企业类别。Anthropic 的 Constitutional Classifiers 是一种在训练阶段而非服务阶段使用的不同方法。

攻击演化同样重要。PAIR 和 TAP 自动化越狱发现。GCG 运行基于梯度的后缀攻击。多轮和语码转换攻击利用智能体记忆。任何部署的 LLM 都需要一个红队靶场——garak 和 PyRIT 是标准驱动器——加上文档化的缓解措施和 CVSS 评分的发现。

你将加固一个目标应用（8B 指令微调模型或其他毕业项目中的 RAG 聊天机器人），对其运行 6+ 攻击族，并产出前后无害性对比测量。

## 概念

安全流水线分五层。**输入净化**：剥离零宽字符、解码 base64/rot13、规范化 Unicode。**策略层**：NeMo Guardrails v0.12 rails（离题、毒性、PII 提取）。**分类器门控**：输入端 Llama Guard 4、非英文用 X-Guard、图像输入用 ShieldGemma-2。**模型**：目标 LLM。**输出过滤**：输出端 Llama Guard 4、Presidio PII 脱敏、适用时的引用强制。**人工审核层**：被标记为高风险的输出进入 Slack 队列。

红队靶场按调度运行。PAIR 和 TAP 自主发现越狱。GCG 运行基于梯度的后缀攻击。ASCII / base64 / rot13 编码攻击。多轮攻击（角色扮演、记忆利用）。语码转换攻击（英文混合斯瓦希里语或泰语）。每次运行产出带 CVSS 评分和披露时间线的结构化发现文件。

宪法式自我批评是训练时干预。取 1k 有害尝试 prompt，让模型起草回复，对照书面宪法（不伤害规则）进行批评，然后在批评循环上重新训练。在留出的评估集上测量前后无害性差异。

## 架构

```
request (text / image / multilingual)
      |
      v
input sanitize (strip zero-width, decode, normalize)
      |
      v
NeMo Guardrails v0.12 rails (off-domain, policy)
      |
      v
classifier gate:
  Llama Guard 4 (English)
  X-Guard (multilingual, 132 langs)
  ShieldGemma-2 (image prompts)
  Nemotron 3 Content Safety (enterprise)
      |
      v (allowed)
target LLM
      |
      v
output filter: Llama Guard 4 + Presidio PII + citation check
      |
      v
HITL tier for flagged outputs

parallel:
  red-team scheduler
    -> garak (classic attacks)
    -> PyRIT (orchestrated red team)
    -> autonomous jailbreak agent (PAIR + TAP)
    -> GCG suffix attacks
    -> multilingual / code-switch
    -> multi-turn persona adoption

output: CVSS-scored findings + disclosure timeline + before/after harmlessness delta
```

## 技术栈

- 安全分类器：Llama Guard 4、ShieldGemma-2、NVIDIA Nemotron 3 Content Safety、X-Guard
- 护栏框架：NeMo Guardrails v0.12 + OPA
- 红队驱动器：garak（NVIDIA）、PyRIT（Microsoft Azure）、NVIDIA Aegis、promptfoo
- 越狱智能体：PAIR（Chao et al., 2023）、Tree-of-Attacks（TAP）、GCG suffix
- 宪法式训练：Anthropic 风格自我批评循环 + SFT on critiques
- PII 脱敏：Presidio
- 目标：8B 指令微调模型或其他毕业项目的 RAG 聊天机器人

## 构建步骤

1. **目标搭建。** 在 vLLM 上部署一个 8B 指令微调模型（或复用其他毕业项目的 RAG 聊天机器人）。这是被测应用。

2. **安全流水线包装。** 将五层流水线接入目标。验证每层可独立观测（Langfuse 中每层一个 span）。

3. **分类器覆盖。** 加载 Llama Guard 4、X-Guard（多语言）、ShieldGemma-2（图像）。在小型标注集上运行，建立基线。

4. **红队调度器。** 调度 garak、PyRIT、一个 PAIR 智能体、一个 TAP 智能体、一个 GCG 运行器、一个多轮攻击器和一个语码转换攻击器。各自运行在独立队列上。

5. **攻击套件。** 六个攻击族：(1) PAIR 自动越狱，(2) TAP 攻击树，(3) GCG 梯度后缀，(4) ASCII / base64 / rot13 编码，(5) 多轮角色扮演，(6) 多语言语码转换。报告每族成功率。

6. **宪法式自我批评。** 策划 1k 有害尝试 prompt。对每个，目标起草回复。批评者 LLM 对照书面宪法评分（"不伤害"、"引用证据"、"拒绝非法请求"）。批评者反对的 prompt 被改写；目标在批评改进的配对上微调。在留出评估集上测量前后无害性。

7. **过度拒绝测量。** 在良性 prompt 套件（如 XSTest）上跟踪误报率。目标在良性问题上必须保持有用。

8. **CVSS 评分。** 对每个成功的越狱，按 CVSS 4.0 评分（攻击向量、复杂度、影响）。产出披露时间线和缓解计划。

9. **靶场自动化。** 以上所有按 cron 运行；发现写入队列；过度拒绝回归告警发送到 Slack。

## 使用示例

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR agent running on target
[attack]     attempt 1/50: disguise query as academic research ... blocked
[attack]     attempt 2/50: appeal to roleplay ... blocked
[attack]     attempt 3/50: chain-of-thought coax ... SUCCEEDED
[finding]    CVSS 4.8 medium: roleplay bypass on target
[range]      7 successes out of 50 (14% success rate)
```

## 交付标准

`outputs/skill-safety-harness.md` 是交付物。一个生产级分层安全流水线加上可复现的红队靶场，附带前后无害性差异。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 攻击面覆盖 | 6+ 攻击族已执行，2+ 语言 |
| 20 | 真阳性/假阳性权衡 | 攻击拦截率 vs XSTest 良性通过率 |
| 20 | 自我批评差异 | 留出评估集上的前后无害性 |
| 20 | 文档和披露 | 带时间线的 CVSS 评分发现 |
| 15 | 自动化和可重复性 | 一切按 cron 运行并带告警 |
| **100** | | |

## 练习

1. 在 RAG 聊天机器人上运行 garak 的 prompt-injection 插件，比较有无输出过滤层时的攻击成功率。

2. 添加第七个攻击族：通过检索文档的间接 prompt injection。测量所需的额外防御。

3. 实现"拒绝并帮助"模式：当护栏拦截时，目标提供一个更安全的相关回答而非直接拒绝。测量 XSTest 差异。

4. 多语言覆盖缺口：找到 X-Guard 表现不佳的语言。提出针对性的微调数据集。

5. 在 30B 模型上运行宪法式自我批评，测量差异是否随规模变化。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 分层安全 | "纵深防御" | 在输入、门控、输出、人工审核多层设置护栏 |
| Llama Guard 4 | "Meta 的安全分类器" | 2026 年参考级输入/输出内容分类器 |
| PAIR | "越狱智能体" | 论文（Chao et al.）关于 LLM 驱动的越狱发现 |
| TAP | "攻击树" | PAIR 的树搜索变体 |
| GCG | "贪心坐标梯度" | 基于梯度的对抗后缀攻击 |
| 宪法式自我批评 | "Anthropic 风格训练" | 目标起草 -> 批评者评分 -> 改写 -> 重新训练 |
| XSTest | "良性探测集" | 过度拒绝回归的基准测试 |
| CVSS 4.0 | "严重性评分" | 安全发现的标准漏洞评分 |

## 延伸阅读

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 训练时参考
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年输入/输出分类器
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 图像 + 多模态安全
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 企业参考
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132 语言多语言安全
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 红队工具包
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 红队框架
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — rail 框架
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 越狱智能体论文
