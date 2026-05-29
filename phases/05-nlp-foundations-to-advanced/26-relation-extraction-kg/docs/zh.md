# 关系抽取与知识图谱构建

> NER 找到了实体。实体链接锚定了它们。关系抽取找到它们之间的边。知识图谱是节点、边及其溯源的总和。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time:** ~60 minutes

## 问题

一位分析师读到："Tim Cook became CEO of Apple in 2011." 四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（RE）将自由文本转化为结构化三元组 `(subject, relation, object)`。在语料上聚合，你就有了知识图谱。聚合并查询，你就有了 RAG、分析或合规审计的推理基底。

2026 年的问题：LLM 抽取关系很积极。太积极了。它们会幻觉出源文本不支持的三元组。没有溯源，你无法区分真实三元组和看似合理的虚构。2026 年的答案是 AEVS 风格的锚定-验证管线。

## 概念

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**三元组形式。** `(subject_entity, relation_type, object_entity)`。关系来自封闭本体（Wikidata 属性、FIBO、UMLS）或开放集合（OpenIE 风格，什么都行）。

**三种抽取方法。**

1. **规则 / 模式匹配。** Hearst 模式："X such as Y" → `(Y, isA, X)`。加上手工正则。脆弱，精确，可解释。
2. **有监督分类器。** 给定句子中的两个实体提及，从固定集合中预测关系。在 TACRED、ACE、KBP 上训练。2015–2022 的标准。
3. **生成式 LLM。** 提示模型输出三元组。开箱即用。需要溯源，否则会幻觉出看似合理的垃圾。

**AEVS（Anchor-Extraction-Verification-Supplement, 2026）。** 当前的幻觉缓解框架：

- **Anchor。** 识别每个实体 span 和关系短语 span 的精确位置。
- **Extract。** 生成链接到锚定 span 的三元组。
- **Verify。** 将每个三元组元素匹配回源文本；拒绝任何无支撑的内容。
- **Supplement。** 覆盖检查确保没有锚定 span 被遗漏。

幻觉大幅减少。需要更多计算但可审计。

**开放 vs 封闭的权衡。**

- **封闭本体。** 固定属性列表（如 Wikidata 的 11,000+ 属性）。可预测。可查询。难以凭空发明。
- **Open IE。** 任何动词短语都成为关系。高召回。低精度。查询混乱。

生产 KG 通常混合使用：Open IE 用于发现，然后将关系规范化到封闭本体上，再合并到主图中。

## 动手构建

### 第 1 步：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

参见 `code/main.py` 的完整玩具抽取器。Hearst 模式仍然在领域特定管线中使用，因为它们可调试。

### 第 2 步：有监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个 seq2seq 关系抽取器：文本输入，三元组输出，已经是 Wikidata 属性 id。在远程监督数据上微调。标准开源基线。

### 第 3 步：带锚定的 LLM 提示抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

验证每个返回的 span 是否与源文本一致。拒绝任何 `text[start:end] != triple_entity` 的结果。这就是 AEVS "verify" 步骤的最小形式。

### 第 4 步：规范化到封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

规范化通常占工程工作的 60-80%。为此做好预算。

### 第 5 步：构建小图并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个 RAG-over-KG 系统的原子。用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强图存储来扩展。

## 常见陷阱

- **RE 之前要做共指消解。** "He founded Apple"——RE 需要知道 "he" 是谁。先跑 coref（第 24 课）。
- **实体规范化。** "Apple Inc" 和 "Apple" 必须解析到同一个节点。先做实体链接（第 25 课）。
- **幻觉三元组。** LLM 输出源文本不支持的三元组。强制 span 验证。
- **关系规范化漂移。** Open IE 关系不一致（"was born in"、"came from"、"is a native of"）。折叠到规范 id，否则图不可查询。
- **时间错误。** "Tim Cook is CEO of Apple"——现在为真，2005 年为假。许多关系是有时间边界的。使用限定符（Wikidata 中的 `P580` 开始时间、`P582` 结束时间）。
- **领域不匹配。** REBEL 在 Wikipedia 上训练。法律、医学和科学文本通常需要领域微调的 RE 模型。

## 实际应用

2026 技术栈：

| 场景 | 选择 |
|-----------|------|
| 快速生产，通用领域 | REBEL 或 LlamaPred with Wikidata canonicalization |
| 领域特定（生物医学、法律） | SciREX-style domain fine-tune + custom ontology |
| LLM 提示，审计输出 | AEVS pipeline: anchor → extract → verify → supplement |
| 高吞吐新闻 IE | Pattern-based + supervised hybrid |
| 从零构建 KG | Open IE + manual canonicalization pass |
| 时序 KG | Extract with qualifiers (start/end time, point in time) |

集成模式：NER → coref → entity linking → relation extraction → ontology mapping → graph load。每个阶段都是潜在的质量门。

## 交付

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习

1. **简单。** 在 5 个新闻文章句子上运行 `code/main.py` 中的模式抽取器。手动检查精确率。
2. **中等。** 在相同句子上使用 REBEL（或小型 LLM）。比较三元组。哪个抽取器精确率更高？召回率更高？
3. **困难。** 构建 AEVS 管线：用 LLM 抽取 + 验证 span 是否与源文本一致。在 50 个 Wikipedia 风格的句子上测量验证步骤前后的幻觉率。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| Triple | 主-关系-宾 | `(s, r, o)` 元组，KG 的原子单元。 |
| Open IE | 抽取一切 | 开放词汇的关系短语；高召回，低精度。 |
| Closed ontology | 固定 schema | 有界的关系类型集合（Wikidata、UMLS、FIBO）。 |
| Canonicalization | 全部规范化 | 将表面名称 / 关系映射到规范 id。 |
| AEVS | 有据抽取 | Anchor-Extraction-Verification-Supplement 管线（2026）。 |
| Provenance | 溯源链接 | 每个三元组携带 doc id + char-span 指向其来源。 |
| Distant supervision | 廉价标签 | 将文本与现有 KG 对齐来创建训练数据。 |

## 延伸阅读

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — the distant-supervision paper.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE workhorse.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — joint IE.
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 hallucination-mitigation design.
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — canonical graph queries.
