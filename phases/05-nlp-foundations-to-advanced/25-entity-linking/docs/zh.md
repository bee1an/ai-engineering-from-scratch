# 实体链接与消歧

> NER 找到了"Paris"。实体链接来决定：法国巴黎？Paris Hilton？德克萨斯州 Paris？特洛伊王子 Paris？没有链接，你的知识图谱就是模糊的。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 minutes

## 问题

一个句子写道："Jordan beat the press." 你的 NER 将 "Jordan" 标记为 PERSON。很好。但*哪个* Jordan？

- Michael Jordan（篮球）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（Berkeley ML 教授——是的，这个混淆在 ML 论文中真实存在）？
- Jordan（国家）？
- Jordan（希伯来语名字）？

实体链接（EL）将每个提及解析到知识库中的唯一条目：Wikidata、Wikipedia、DBpedia 或你的领域 KB。两个子任务：

1. **候选生成。** 给定 "Jordan"，哪些 KB 条目是合理的？
2. **消歧。** 给定上下文，哪个候选是正确的？

两个步骤都是可学习的。两个都有基准。组合管线已经稳定了十年——变化的是消歧器的质量。

## 概念

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**候选生成。** 给定提及的表面形式（"Jordan"），在别名索引中查找候选。Wikipedia 别名字典覆盖了大多数命名实体："JFK" → John F. Kennedy、Jacqueline Kennedy、JFK 机场、JFK（电影）。典型索引每个提及返回 10-30 个候选。

**消歧：三种方法。**

1. **Prior + context（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好，快，不需要训练。
2. **基于 embedding（ESS / REL / Blink）。** 编码提及 + 上下文。编码每个候选的描述。取最大余弦。2020-2024 的默认方法。
3. **生成式（GENRE, 2021; LLM-based, 2023+）。** 逐 token 解码实体的规范名称。约束在有效实体名称的 trie 上，保证输出是有效的 KB id。

**端到端 vs 管线。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）在一次前向传播中完成 NER + 候选生成 + 消歧。管线系统在生产中仍然占主导，因为你可以替换组件。

### 两个度量

- **Mention recall（候选生成）。** 正确 KB 条目出现在候选列表中的 gold mention 比例。整个管线的下限。
- **消歧准确率 / F1。** 给定正确候选，top-1 正确的频率。

始终报告两者。一个消歧准确率 99% 但候选召回只有 80% 的系统，是一个 80% 的管线。

## 动手构建

### 第 1 步：从 Wikipedia 重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 别名数据：~18M (alias, entity) 对。从 Wikidata dumps 下载。存储为倒排索引。

### 第 2 步：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠是个玩具。替换为 embedding 上的余弦相似度（参见 `code/main.py` step-2 的 transformer 版本）。

### 第 3 步：基于 embedding（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

索引时，对每个 KB 实体编码一次。查询时，对提及 + 上下文编码一次，与候选池做点积，取最大值。

### 第 4 步：生成式实体链接（概念）

GENRE 逐字符解码实体的 Wikipedia 标题。约束解码（见第 20 课）确保只能输出有效标题。与 KB 支持的 trie 紧密集成。现代后继者是 REL-GEN 和带结构化输出的 LLM-prompted EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice`），这是 2026 年最简单的可部署 EL 管线。

### 第 5 步：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准 EL 基准：1,393 篇路透社文章，34k 提及，Wikipedia 实体。报告 in-KB 准确率（`P@1`）和 out-of-KB NIL 检测率。

## 常见陷阱

- **NIL 处理。** 有些提及不在 KB 中（新兴实体、不知名的人）。系统必须预测 NIL 而不是猜一个错误的实体。单独度量。
- **提及边界错误。** 上游 NER 漏掉部分 span（"Bank of America" 只标记了 "Bank"）。EL 召回下降。
- **流行度偏差。** 训练过的系统过度预测高频实体。ML 论文中提到 "Michael I. Jordan" 经常被链接到篮球 Jordan。
- **跨语言 EL。** 将中文文本中的提及映射到英文 Wikipedia 实体。需要多语言编码器或翻译步骤。
- **KB 过时。** 新公司、事件、人物不在去年的 Wikipedia dump 中。生产管线需要刷新循环。

## 实际应用

2026 技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英文 + Wikipedia | BLINK 或 REL |
| 跨语言，KB = Wikipedia | mGENRE |
| LLM 友好，每天少量提及 | Prompt Claude/GPT-4 with candidate list + constrained JSON |
| 领域特定 KB（医学、法律） | Custom BERT with KB-aware retrieval + fine-tune on domain AIDA-style set |
| 极低延迟 | Exact-match prior only (Milne-Witten baseline) |
| 研究 SOTA | GENRE / ExtEnD / generative LLM-EL |

2026 年的生产模式：NER → coref → 对每个提及做 EL → 将簇折叠为每个簇一个规范实体。输出：文档中每个实体一个 KB id，而不是每个提及一个。

## 交付

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习

1. **简单。** 在 `code/main.py` 中对 10 个歧义提及（Paris、Jordan、Apple）实现 prior+context 消歧器。手动标注正确实体。测量准确率。
2. **中等。** 用 sentence transformer 编码 50 个歧义提及。对每个候选的描述做 embedding。比较基于 embedding 的消歧与 Jaccard 上下文重叠。
3. **困难。** 构建一个 1k 实体的领域 KB（如公司的员工 + 产品）。端到端实现 NER + EL。在 100 个留出句子上测量精确率和召回率。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| Entity linking (EL) | 链接到 Wikipedia | 将提及映射到唯一的 KB 条目。 |
| Candidate generation | 可能是谁？ | 为一个提及返回合理 KB 条目的短列表。 |
| Disambiguation | 选对的那个 | 用上下文对候选打分，选出赢家。 |
| Alias index | 查找表 | 从表面形式 → 候选实体的映射。 |
| NIL | 不在 KB 中 | 显式预测没有 KB 条目匹配。 |
| KB | 知识库 | Wikidata、Wikipedia、DBpedia 或你的领域 KB。 |
| AIDA-CoNLL | 那个基准 | 1,393 篇路透社文章，带有 gold entity links。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — the foundational prior+context approach.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — the embedding-based workhorse.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — generative EL with constrained decoding.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — the benchmark paper.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — the open production stack.
