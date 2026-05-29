# 共指消解

> "She called him. He did not answer. The doctor was at lunch." 三个指代，两个人，没有一个被点名。共指消解搞清楚谁是谁。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time:** ~60 minutes

## 问题

从一篇 300 词的文章中提取所有对 Apple Inc. 的提及。当文章说"Apple"时很简单。当它说"the company"、"they"、"Cupertino's technology giant"或"Jobs's firm"时就难了。如果不把这些提及解析到同一个实体，你的 NER 管线会漏掉 60-80% 的提及。

共指消解将所有指向同一个现实世界实体的表达链接到一个簇中。它是表层 NLP（NER、句法分析）和下游语义（信息抽取、QA、摘要、知识图谱）之间的粘合剂。

为什么在 2026 年仍然重要：

- 摘要："The CEO announced..." vs "Tim Cook announced..."——摘要应该点名 CEO。
- 问答："Who did she call?" 需要解析 "she"。
- 信息抽取：知识图谱中 "PER1 founded Apple" 和 "Jobs founded Apple" 作为两个独立条目是错误的。
- 跨文档信息抽取：合并多篇关于同一事件的文章中的提及就是跨文档共指消解。

## 概念

![Coreference clustering: mentions → entities](../assets/coref.svg)

**任务。** 输入：一个文档。输出：提及（span）的聚类，每个簇指向一个实体。

**提及类型。**

- **命名实体。** "Tim Cook"
- **名词性。** "the CEO", "the company"
- **代词性。** "he", "she", "they", "it"
- **同位语。** "Tim Cook, Apple's CEO,"

**架构。**

1. **基于规则（Hobbs, 1978）。** 基于句法树的代词消解，使用语法规则。好的基线。在代词上出奇地难以超越。
2. **Mention-pair 分类器。** 对每对提及 (m_i, m_j)，预测它们是否共指。通过传递闭包聚类。2016 年前的标准。
3. **Mention-ranking。** 对每个提及，对候选先行词（包括"无先行词"）排序。取最高的。
4. **基于 span 的端到端（Lee et al., 2017）。** Transformer 编码器。枚举所有长度上限内的候选 span。预测提及分数。预测每个 span 的先行词概率。贪心聚类。现代默认方法。
5. **生成式（2024+）。** 提示 LLM："列出文本中每个代词及其先行词。"在简单情况下效果好，在长文档和罕见指代物上表现差。

**评估指标。** 五个标准指标（MUC、B³、CEAF、BLANC、LEA），因为没有单一指标能捕获聚类质量。报告前三个的平均值作为 CoNLL F1。2026 年在 CoNLL-2012 上的 SOTA：~83 F1。

**已知难点。**

- 定指描述指向数页之前引入的实体。
- 桥接回指（"the wheels" → 之前提到的一辆车）。
- 中文和日文中的零回指。
- 前指（代词在指代物之前）："When **she** walked in, Mary smiled."

## 动手构建

### 第 1 步：预训练神经共指（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在更长的文档上，你会得到类似：
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

### 第 2 步：基于规则的代词消解器（教学用）

参见 `code/main.py` 的纯标准库实现：

1. 提取提及：命名实体（大写 span）、代词（字典查找）、定指描述（"the X"）。
2. 对每个代词，查看前 K 个提及并按以下标准打分：
   - 性别/数量一致性（启发式）
   - 近因性（越近越好）
   - 句法角色（主语优先）
3. 链接得分最高的先行词。

无法与神经模型竞争。但它展示了搜索空间和端到端模型必须做出的决策。

### 第 3 步：使用 LLM 做共指消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

两个需要注意的失败模式。第一，LLM 过度合并（"him" 和 "her" 指向两个不同的人）。第二，LLM 在长文档中静默丢弃提及。始终用 span-offset 检查来验证。

### 第 4 步：评估

标准 conll-2012 脚本计算 MUC、B³、CEAF-φ4 并报告平均值。对于内部评估，从你标注的测试集上的 span 级精确率和召回率开始，然后加上 mention-linking F1。

## 常见陷阱

- **单例爆炸。** 有些系统将每个提及报告为自己的簇。B³ 对此宽容。MUC 会惩罚。始终检查所有三个指标。
- **长上下文中的代词。** 在超过 2,000 token 的文档上性能下降 ~15 F1。谨慎分块。
- **性别假设。** 硬编码的性别规则在非二元指代物、组织、动物上会失效。使用学习模型或中性打分。
- **LLM 在长文档上的漂移。** 单次 API 调用无法可靠地聚类 50+ 段落中的提及。使用滑动窗口 + 合并。

## 实际应用

2026 技术栈：

| 场景 | 选择 |
|-----------|------|
| 英文，单文档 | `en_coreference_web_trf` (spaCy-experimental) 或 AllenNLP neural coref |
| 多语言 | SpanBERT / XLM-R trained on OntoNotes or Multilingual CoNLL |
| 跨文档事件共指 | 专用端到端模型（2025–26 SOTA） |
| 快速 LLM 基线 | GPT-4o / Claude with structured-output coref prompt |
| 生产对话系统 | 基于规则的后备 + 神经主力 + 关键 slot 的人工审核 |

2026 年的集成模式：先跑 NER，再跑共指消解，将共指簇合并到 NER 实体中。下游任务看到的是每个簇一个实体，而不是每个提及一个实体。

## 交付

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习

1. **简单。** 在 5 个手工构造的段落上运行 `code/main.py` 中的基于规则的消解器。对照 ground truth 测量 mention-link 准确率。
2. **中等。** 在一篇新闻文章上使用预训练神经共指模型。将簇与你自己的人工标注对比。它在哪里失败了？
3. **困难。** 构建一个共指增强的 NER 管线：先 NER，然后通过共指簇合并。在 100 篇文章上测量相对于纯 NER 的实体覆盖率提升。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| Mention | 一个指代 | 指向一个实体的文本 span（名称、代词、名词短语）。 |
| Antecedent | "it" 指的是什么 | 后面的提及与之共指的更早的提及。 |
| Cluster | 实体的所有提及 | 所有指向同一现实世界实体的提及的集合。 |
| Anaphora | 回指 | 后面的提及指向前面的（"he" → "John"）。 |
| Cataphora | 前指 | 前面的提及指向后面的（"When he arrived, John..."）。 |
| Bridging | 隐式指代 | "I bought a car. The wheels were bad."（那辆车的轮子。） |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 F1 分数的平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — canonical textbook chapter.
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — span-based end-to-end.
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — pretraining that improves coref.
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — the benchmark.
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — the rule-based classic.
