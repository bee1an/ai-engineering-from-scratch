# 文本处理 — 分词、词干提取、词形还原

> 语言是连续的，模型是离散的。预处理就是连接两者的桥梁。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 2 · 14 (Naive Bayes)
**时间：** 约 45 分钟

## 问题

模型无法阅读 "The cats were running."，它只能读整数。

每个 NLP 系统都从同样的三个问题开始：一个词从哪里开始？词的词根是什么？我们如何在需要时把 "run"、"running"、"ran" 当作同一个东西，又在需要时把它们当作不同的东西？

分词搞错了，模型就是在从垃圾中学习。如果你的 tokenizer 把 `don't` 当作一个 token，但把 `do n't` 当作两个，训练分布就会分裂。如果你的词干提取器把 `organization` 和 `organ` 归为同一个词干，主题建模就完了。如果你的词形还原器需要词性上下文但你没传入，动词就会被当作名词处理。

本课从零构建这三个预处理步骤，然后展示 NLTK 和 spaCy 如何完成同样的工作，让你看到其中的取舍。

## 概念

三个操作，每个都有自己的职责和失败模式。

**分词（Tokenization）** 将字符串拆分为 token。"Token" 这个词故意模糊，因为合适的粒度取决于任务。经典 NLP 用词级别，transformer 用子词级别，没有空格的语言用字符级别。

**词干提取（Stemming）** 用规则砍掉后缀。快速、激进、笨拙。`running -> run`。`organization -> organ`。第二个就是失败模式。

**词形还原（Lemmatization）** 利用语法知识将词还原为词典形式。更慢、更准确，需要查找表或形态分析器。`ran -> run`（需要知道 "ran" 是 "run" 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则：当速度重要且能容忍噪声时用词干提取（搜索索引、粗略分类）。当语义重要时用词形还原（问答、语义搜索、任何用户会看到的内容）。

## 动手构建

### 第 1 步：正则表达式分词器

最简单实用的 tokenizer 按非字母数字字符分割，同时保留标点作为独立 token。不完美，不是最终方案，但一行就能跑。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级排列：带可选内部撇号的单词（`don't`、`it's`）；纯数字；任何单个非空白非字母数字字符作为独立 token（标点）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失败模式：`3pm` 被拆成 `['3', 'pm']`，因为我们在字母序列和数字序列之间交替。对大多数任务够用了。URL、邮箱、hashtag 都会出问题。生产环境中，在通用模式之前添加专用模式。

### 第 2 步：Porter 词干提取器（仅 step 1a）

完整的 Porter 算法有五个阶段的规则。仅 Step 1a 就覆盖了最常见的英语后缀，足以教会你这个模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

从上往下读规则。`ies -> i` 规则是 `ponies -> poni` 而不是 `pony` 的原因。真正的 Porter 有 step 1b 会修正这个。规则之间竞争，靠前的规则优先。顺序比任何单条规则都重要。

### 第 3 步：基于查找表的词形还原器

真正的词形还原需要形态学。一个可教学的版本使用小型词元表加回退策略。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个案例是关键教学点。`watched` 不在我们的表中，回退策略只处理 `ing`。真正的词形还原覆盖 `ed`、不规则动词、比较级形容词、带音变的复数（`children -> child`）。这就是为什么生产系统使用 WordNet、spaCy 的形态分析器或完整的形态分析器。

### 第 4 步：串联起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的部分是词性标注器。Phase 5 · 07（词性标注）会构建一个。目前默认所有词为 `NOUN`，并承认这个局限性。

## 使用现成工具

NLTK 和 spaCy 提供了生产版本，各只需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 处理缩写、Unicode 和你的正则遗漏的边界情况。`PorterStemmer` 运行全部五个阶段。`WordNetLemmatizer` 需要将 NLTK 的 Penn Treebank 标签方案翻译为 WordNet 的缩写集。上面的翻译代码就是大多数教程跳过的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 把整个流水线隐藏在 `nlp(text)` 背后。分词、词性标注和词形还原全部运行。规模化时比 NLTK 更快，开箱即用更准确。代价是你不能轻松替换单个组件。

### 如何选择

| 场景 | 选择 |
|------|------|
| 教学、研究、需要替换组件 | NLTK |
| 生产环境、多语言、速度重要 | spaCy |
| Transformer 流水线（反正会用模型自带的 tokenizer） | 用 `tokenizers` / `transformers`，跳过经典预处理 |

### 没人警告你的两个失败模式

大多数教程教完算法就停了。两件事会咬到真实的预处理流水线，而且几乎从不被提及。

**可复现性漂移。** NLTK 和 spaCy 在版本之间会改变分词和词形还原行为。spaCy 2.x 中产生 `['do', "n't"]` 的代码在 3.x 中可能产生 `["don't"]`。你的模型在一个分布上训练，推理现在运行在另一个分布上。准确率悄悄下降，没人知道为什么。在 `requirements.txt` 中锁定库版本。写一个预处理回归测试，冻结 20 个样本句子的预期分词结果。每次升级都运行它。

**训练/推理不匹配。** 训练时用激进的预处理（小写化、停用词移除、词干提取），部署时用原始用户输入，然后看着性能崩塌。这是最常见的生产 NLP 故障。如果训练时做了预处理，推理时必须运行完全相同的函数。把预处理作为模型包内的函数发布，而不是让服务团队重写的 notebook cell。

## 交付

一个可复用的 prompt，帮助工程师选择预处理策略，不用读三本教科书。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习

1. **简单。** 扩展 `tokenize` 使其将 URL 保留为单个 token。测试：`tokenize("Visit https://example.com today.")` 应产生一个 URL token。
2. **中等。** 实现 Porter step 1b。如果一个词包含元音且以 `ed` 或 `ing` 结尾，移除它。处理双辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. **困难。** 构建一个词形还原器，使用 WordNet 作为查找表，当 WordNet 没有条目时回退到你的 Porter 词干提取器。在标注语料库上对比纯 WordNet 和纯 Porter 的准确率。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Token | 一个词 | 模型消费的任何单元。可以是词、子词、字符或字节。 |
| Stem（词干） | 词的词根 | 基于规则的后缀剥离结果。不一定是真实的词。 |
| Lemma（词元） | 词典形式 | 你会去查的那个形式。需要语法上下文才能正确计算。 |
| POS tag（词性标签） | 词性 | NOUN、VERB、ADJ 等类别。准确词形还原所必需。 |
| Morphology（形态学） | 词形变化规则 | 词如何根据时态、数、格改变形式。词形还原依赖于此。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，五页，仍然是最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实流水线是如何连接的。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你还没想到的分词边界情况。
