# 毕业项目 17 — 个人AI导师（自适应、多模态、带记忆）

> Khanmigo（可汗学院）、Duolingo Max、Google LearnLM / Gemini for Education、Quizlet Q-Chat 和 Synthesis Tutor 都在 2026 年大规模交付了自适应多模态辅导。共同形态是苏格拉底式策略（永远不直接给答案）、每次交互后更新的学习者模型（贝叶斯知识追踪风格）、语音 + 文本 + 拍照数学输入、课程图谱检索、间隔重复调度，以及严格的年龄适当内容安全过滤。本毕业项目要交付一个学科专用导师（K-12 代数或 Python 入门），运行为期两周的 10 名学习者效果研究，并通过内容安全审计。

**类型：** 毕业项目
**语言：** Python（后端、学习者模型）、TypeScript（Web 应用）、SQL（通过 Postgres + Neo4j 的课程图谱）
**前置课程：** Phase 5（NLP）、Phase 6（语音）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 14（智能体）、Phase 17（基础设施）、Phase 18（安全）
**覆盖阶段：** P5 · P6 · P11 · P12 · P14 · P17 · P18
**时间：** 30 小时

## 问题

自适应辅导曾经是教育科技的研究小众。到 2026 年它已是消费级产品。Khanmigo 部署在美国大多数学区。Duolingo Max 达到数千万 MAU。Google 的 LearnLM / Gemini for Education 驱动 Google Classroom 中的辅导。Quizlet Q-Chat 与闪卡并列。Synthesis Tutor 以"好奇孩子的导师"走红。共同要素：多模态输入（打字、说话、拍照方程）、苏格拉底式教学法（先问后讲）、每次交互后更新的学习者模型，以及严格的年龄适当安全。

你将为特定群体构建其中之一。衡量标准是实际效果研究：10 名学习者两周内的前测和后测分数。语音循环必须感觉自然（毕业项目 03 子栈）。记忆必须尊重隐私。安全过滤必须通过面向 K-12 的 COPPA 感知红队测试。

## 概念

四个组件。**导师策略**是苏格拉底式循环：当学习者要答案时，策略问一个引导性问题；当他们答对时，进入下一个概念；当他们卡住时，提供脚手架式提示。**学习者模型**是贝叶斯知识追踪（或简单变体），在每次交互后更新每个课程节点的掌握概率。**课程图谱**是 Neo4j 中带前置依赖边的概念图；策略遍历图谱来选择下一个概念。**记忆**是情景 + 语义存储（agentmemory 风格），保存过去的交互、错误和偏好。

UX 是多模态的。文本输入用于打字回答。语音输入通过 LiveKit + Whisper（复用毕业项目 03）。拍照输入用于数学题，通过 dots.ocr 或 PaliGemma 2。语音输出通过 Cartesia Sonic-2。安全使用 Llama Guard 4 加年龄适当过滤器（屏蔽成人内容、暴力、自残）和 COPPA 感知的记忆保留策略。

效果研究是交付物。10 名学习者，前测和后测，两周。报告学习增益差异和置信区间。与非自适应基线（相同内容线性交付，无导师策略）对比。

## 架构

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## 技术栈

- 学科选择：K-12 代数或 Python 入门（选一个深入）
- 导师策略：LangGraph over Claude Sonnet 4.7（带 prompt caching）
- 学习者模型：贝叶斯知识追踪（经典）或 FSRS 用于间隔
- 课程图谱：Neo4j 概念 + 前置依赖边 + OER 内容
- 记忆：agentmemory 风格持久化向量 + 情景 + 语义存储
- 语音：LiveKit Agents 1.0 + Cartesia Sonic-2（复用毕业项目 03 子栈）
- 拍照数学：dots.ocr 或 PaliGemma 2 用于方程识别
- 安全：Llama Guard 4 + 自定义年龄适当过滤器
- 评估：Bloom 层级问题生成、前/后测框架、效果研究工具

## 构建步骤

1. **课程图谱。** 构建一个 50-150 个概念节点的 Neo4j（例如 K-12 代数从"数轴"到"二次公式"），带前置依赖边。每个节点附加 OER 内容（Open Textbook、OpenStax）。

2. **学习者模型。** 用先验初始化贝叶斯知识追踪：猜测率、失误率、学习率。每次交互后更新每概念掌握度。按学习者持久化。

3. **导师策略。** LangGraph 节点：`read_signal`（学习者的回答是正确/部分/卡住？）、`select_concept`（遍历课程图谱选择最高优先级概念）、`scaffold`（苏格拉底式提示）、`update_mastery`。

4. **记忆。** 每次交互写入情景存储。错误和偏好提升到语义记忆。COPPA 感知保留策略：1 年后自动删除，家长可访问。

5. **语音路径。** LiveKit Agents worker 接入导师策略。ASR 通过 Whisper-v3-turbo。TTS 通过 Cartesia Sonic-2。支持打断（复用毕业项目 03 机制）。

6. **拍照数学路径。** 上传或拍摄图像；运行 dots.ocr 或 PaliGemma 2 识别方程；作为结构化输入传给导师。

7. **安全。** 每个模型输出通过 Llama Guard 4 + 年龄适当过滤器（屏蔽自残、成人内容、暴力）。记忆访问按学习者 ID 范围限定；家长可访问删除界面。

8. **效果研究。** 10 名学习者，前测（标准化 30 题基线），两周导师交互（每周 3 次），后测。与 10 名学习者的非自适应基线组在相同内容上对比。

9. **每周进度报告。** 按学习者自动生成 PDF 摘要：探索的主题、掌握轨迹和建议的下一步。

## 使用示例

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## 交付标准

`outputs/skill-ai-tutor.md` 是交付物。一个学科专用自适应导师，具有多模态输入、学习者模型、记忆、安全和经过测量的效果。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 学习增益差异 | 10 名学习者两周研究中的前/后测差异 |
| 20 | 苏格拉底式忠实度 | 对话样本的评分标准得分 |
| 20 | 多模态 UX | 语音 + 拍照 + 文本端到端一致性 |
| 20 | 安全 + 隐私姿态 | Llama Guard 4 通过率 + COPPA 感知保留 |
| 15 | 课程广度和图谱质量 | 概念覆盖 + 前置依赖图一致性 |
| **100** | | |

## 练习

1. 分别运行有无自适应学习者模型（随机概念顺序）的效果研究。报告差异。预期自适应会赢，但差异大小才是有趣的数字。

2. 添加多模态探测：同一概念问题分别以文本、语音和拍照交付。测量学习者是否在偏好的模态上更快收敛。

3. 构建家长仪表板：练习的主题、掌握轨迹、即将到来的概念、安全事件（任何护栏触发）。符合 COPPA。

4. 添加语言切换模式：导师接受西班牙语输入并用西班牙语教学。测量 X-Guard 覆盖。

5. 压力测试记忆隐私：验证学习者 A 即使通过语音片段重新摄入攻击也无法看到学习者 B 的数据。记录尝试访问并告警。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 苏格拉底式策略 | "问，不灌" | 导师问引导性问题而非直接给答案 |
| 贝叶斯知识追踪 | "BKT" | 经典学习者模型方程，计算每概念掌握概率 |
| FSRS | "Free Spaced Repetition Scheduler" | 2024 年间隔重复调度器，优于 SM-2 |
| 课程图谱 | "概念 DAG" | Neo4j 中带前置依赖边的概念图 |
| 情景记忆 | "每次交互日志" | 每次交互存储以供后续检索 |
| 语义记忆 | "学习模式存储" | 从情景记忆提升的压缩错误和偏好 |
| COPPA | "儿童隐私法" | 限制收集 13 岁以下儿童数据的美国法律 |

## 延伸阅读

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — 参考消费级 K-12 导师
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — 参考语言学习导师
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — 托管参考模型
- [Quizlet Q-Chat](https://quizlet.com) — 替代参考
- [Synthesis Tutor](https://www.synthesis.com) — 创业公司参考
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — 间隔重复调度器
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — 学习者模型经典
- [LiveKit Agents](https://github.com/livekit/agents) — 语音栈
