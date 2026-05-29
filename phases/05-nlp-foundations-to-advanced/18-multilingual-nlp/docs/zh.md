# 多语言NLP

> 一个模型，100+ 种语言，大多数语言零训练数据。跨语言迁移是 2020 年代的实用奇迹。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**Time:** ~45 minutes

## 问题

英语有数十亿标注样本。乌尔都语有数千。迈蒂利语几乎没有。任何服务全球受众的实用 NLP 系统都必须在任务特定训练数据不存在的长尾语言上工作。

多语言模型通过同时在多种语言上训练一个模型来解决这个问题。共享表示让模型将在高资源语言中学到的技能迁移到低资源语言。在英语情感分析上微调模型，它就能开箱即用地在乌尔都语上产生出人意料好的情感预测。这就是 zero-shot 跨语言迁移，它重塑了 NLP 向世界交付的方式。

本课命名权衡、经典模型，以及让多语言新手团队犯错的那个决策：选择迁移的源语言。

## 概念

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**共享词表。** 多语言模型使用在所有目标语言文本上训练的 SentencePiece 或 WordPiece 分词器。词表是共享的：同一个子词单元在相关语言中代表同一个语素。英语和意大利语中的 `anti-` 得到同一个 token。

**共享表示。** 在多种语言上通过 masked language modeling 预训练的 transformer 学会了：不同语言中语义相似的句子产生相似的隐藏状态。mBERT、XLM-R 和 NLLB 都展现了这一点。英语 "cat" 的嵌入与法语 "chat" 和西班牙语 "gato" 聚在一起，完整句子嵌入也是如此。

**Zero-shot 迁移。** 在一种语言（通常是英语）的标注数据上微调模型。推理时在模型支持的任何其他语言上运行。不需要目标语言标签。对类型学相近的语言效果强，对距离远的语言效果弱。

**Few-shot 微调。** 添加 100-500 个目标语言的标注样本。分类任务上准确率跳升到英语基线的 95-98%。这是多语言NLP中性价比最高的杠杆。

## 模型

| 模型 | 年份 | 覆盖 | 备注 |
|-------|------|----------|-------|
| mBERT | 2018 | 104 种语言 | 在 Wikipedia 上训练。第一个实用的多语言 LM。低资源表现弱。 |
| XLM-R | 2019 | 100 种语言 | 在 CommonCrawl 上训练（比 Wikipedia 大得多）。设定跨语言基线。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 种语言 | XLM-R + 1M token 词表（vs 250k）。低资源表现更好。 |
| mT5 | 2020 | 101 种语言 | 多语言生成的 T5 架构。 |
| NLLB-200 | 2022 | 200 种语言 | Meta 的翻译模型；包含 55 种低资源语言。 |
| BLOOM | 2022 | 46 种语言 + 13 种编程语言 | 开放的 176B 多语言 LLM。 |
| Aya-23 | 2024 | 23 种语言 | Cohere 的多语言 LLM。阿拉伯语、印地语、斯瓦希里语表现强。 |

按用例选择。分类任务用 XLM-R-base 作为合理默认值效果好。生成任务根据翻译还是开放生成选择 mT5 或 NLLB。LLM 风格的工作搭配 Aya-23 或 Claude 使用显式多语言提示。

## 源语言决策（2026 研究）

大多数团队默认用英语作为微调源。最近的研究（2026）表明这通常是错的。

语言相似性比原始语料库大小更好地预测迁移质量。对于斯拉夫语目标，德语或俄语通常优于英语。对于印度语目标，印地语通常优于英语。**qWALS** 相似性指标（2026，基于 World Atlas of Language Structures 特征）量化了这一点。**LANGRANK**（Lin et al., ACL 2019）是一个独立的、更早的方法，从语言相似性、语料库大小和谱系亲缘关系的组合中排列候选源语言。

实用规则：如果你的目标语言有一个类型学上接近的高资源亲属，先尝试在那个语言上微调，然后与英语微调比较。

## 动手构建

### 第 1 步：zero-shot 跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，同一个 API。在 NLI 数据上训练的 XLM-R 通过 entailment 技巧很好地迁移到分类。

### 第 2 步：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译在嵌入空间中距离近。不同的英语句子距离远。这就是跨语言检索、聚类和相似度工作的原理。

### 第 3 步：few-shot 微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 个目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全默认值。更高的学习率会导致多语言对齐崩溃，你会得到一个仅英语的模型。

## 真正有效的评估

- **每种语言在留出集上的准确率。** 不要聚合。聚合会隐藏长尾。
- **与单语基线对比。** 对于有足够数据的语言，从头训练的单语模型有时会击败多语言模型。测试。
- **实体级测试。** 目标语言中的命名实体。多语言模型对远离拉丁文字的脚本通常分词较弱。
- **跨语言一致性。** 两种语言中相同含义应该产生相同预测。测量差距。

## 实际应用

2026 年技术栈：

| 任务 | 推荐 |
|-----|-------------|
| 分类，100 种语言 | XLM-R-base (~270M) 微调 |
| Zero-shot 文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，200 种语言 | `facebook/nllb-200-distilled-600M`（见第 11 课） |
| 生成式多语言 | Claude, GPT-4, Aya-23, mT5-XXL |
| 低资源语言 NLP | XLM-V 或在相关高资源语言上的领域特定微调 |

如果性能重要，始终为目标语言的微调留出预算。Zero-shot 是起点，不是最终答案。

### 分词税（低资源语言出什么问题）

多语言模型在所有语言间共享一个分词器。该词表在以英语、法语、西班牙语、中文、德语为主的语料上训练。对于主导集之外的任何语言，三种税悄悄叠加：

- **生育率税。** 低资源语言文本每词分词成远多于英语的 token。一个印地语句子可能需要等价英语句子 3-5 倍的 token。这 3-5 倍吃掉你的上下文窗口、训练效率和延迟。
- **变体恢复税。** 每个拼写错误、变音符号变体、Unicode 归一化不匹配或大小写变化都变成嵌入空间中的冷启动无关序列。模型无法学习母语者认为显而易见的正字法对应关系。
- **容量溢出税。** 税 1 和税 2 消耗上下文位置、层深度和嵌入维度。留给实际推理的系统性地少于高资源语言从同一模型获得的。

实际症状：你的模型在印地语上正常训练，loss 曲线看起来对，eval 困惑度看起来合理，而生产输出微妙地错误。形态在句子中间崩溃。罕见屈折变化无法恢复。**你无法通过扩大数据规模来解决一个坏分词器。**

缓解：选择对目标语言有良好覆盖的分词器（XLM-V 的 1M token 词表是直接修复）；在训练前验证留出目标文本上的分词生育率；对真正长尾的文字使用字节级回退（SentencePiece `byte_fallback=True`，GPT-2 风格的字节级 BPE），这样就永远不会有 OOV。

## 交付

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习

1. **简单。** 在英语、法语、印地语和阿拉伯语各 10 个句子上运行 zero-shot 分类流水线。报告每种语言的准确率。你应该看到法语强、印地语不错、阿拉伯语不稳定。
2. **中等。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在小型混合语言语料库上构建跨语言检索器。用英语查询，检索任何语言的文档。测量 recall@5。
3. **困难。** 比较英语源和印地语源微调对印地语分类任务的效果。在两种方案下使用 500 个目标语言样本进行 few-shot 微调。报告哪个源产生更好的印地语准确率以及差多少。这是 LANGRANK 论点的微缩版。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| Multilingual model | 一个模型，多种语言 | 跨语言共享词表和参数。 |
| Cross-lingual transfer | 在一种语言上训练，在另一种上运行 | 在源语言上微调，在目标语言上评估，无需目标语言标签。 |
| Zero-shot | 无目标语言标签 | 不在目标语言上微调就迁移。 |
| Few-shot | 少量目标标签 | 100-500 个目标语言样本用于微调。 |
| mBERT | 第一个多语言 LM | 在 Wikipedia 上预训练的 104 种语言 BERT。 |
| XLM-R | 标准跨语言基线 | 在 CommonCrawl 上预训练的 100 种语言 RoBERTa。 |
| NLLB | Meta 的 200 种语言 MT | No Language Left Behind。包含 55 种低资源语言。 |

## 延伸阅读

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 开启跨语言迁移研究线的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere 的多语言 LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK 源语言论文。
