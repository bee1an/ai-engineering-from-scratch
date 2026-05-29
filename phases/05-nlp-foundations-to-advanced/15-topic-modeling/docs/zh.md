# 主题模型 — LDA 和 BERTopic

> LDA：文档是主题的混合，主题是词的分布。BERTopic：文档在嵌入空间中聚类，聚类就是主题。目标相同，分解方式不同。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45 minutes

## 问题

你有 10,000 条客服工单、50,000 篇新闻文章或 200,000 条推文。你需要在不阅读的情况下知道这个集合在讲什么。你没有标注好的类别。你甚至不知道有多少个类别。

主题模型在无监督的情况下回答这个问题。给它一个语料库，返回一小组连贯的主题，以及每个文档在这些主题上的分布。

两个算法族占主导。LDA（2003）将每个文档视为潜在主题的混合，每个主题是词的分布。推断是贝叶斯的。在需要混合成员主题分配和可解释的词级概率分布的场景中，它仍然在生产中使用。

BERTopic（2020）用 BERT 编码文档，用 UMAP 降维，用 HDBSCAN 聚类，通过 class-based TF-IDF 提取主题词。它在短文本、社交媒体以及语义相似性比词重叠更重要的任何场景中胜出。一个文档只得到一个主题，这对长文本内容是一个限制。

本课为两者建立直觉，并指出在给定语料库下该选哪个。

## 概念

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**LDA 生成故事。** 每个主题是词的分布。每个文档是主题的混合。要在文档中生成一个词，先从文档的混合中采样一个主题，然后从该主题的分布中采样一个词。推断反转这个过程：给定观察到的词，推断每个文档的主题分布和每个主题的词分布。Collapsed Gibbs sampling 或变分贝叶斯完成数学计算。

LDA 的关键输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行和为 1（文档的主题混合）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行和为 1（主题的词分布）。

**BERTopic 流水线。**

1. 用 sentence transformer（如 `all-MiniLM-L6-v2`）编码每个文档。384 维向量。
2. 用 UMAP 降维到约 5 维。BERT 嵌入维度太高，不适合直接聚类。
3. 用 HDBSCAN 聚类。基于密度，产生可变大小的聚类和一个"离群"标签。
4. 对每个聚类，在该聚类的文档上计算 class-based TF-IDF 来提取主题词。

输出是每个文档一个主题（加上 -1 离群标签）。可选地，通过 HDBSCAN 的概率向量获得软成员关系。

## 动手构建

### 第 1 步：用 scikit-learn 做 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：停用词已移除，min_df 和 max_df 过滤稀有和普遍的词，使用 CountVectorizer（不是 TfidfVectorizer）因为 LDA 期望原始计数。

### 第 2 步：BERTopic（生产用）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

过滤 `Topic != -1` 去掉 BERTopic 的离群桶（HDBSCAN 无法聚类的文档）。`min_topic_size` 控制 HDBSCAN 的最小聚类大小；BERTopic 库默认值是 10。本示例为课程规模显式设为 15。对于超过 10,000 文档的语料库，增加到 50 或 100。

### 第 3 步：评估

两种方法都输出主题词。问题是这些词是否连贯。

- **主题连贯性（c_v）。** 结合 top 词对在滑动窗口上下文中的 NPMI（归一化逐点互信息），将分数聚合为主题向量，并通过余弦相似度比较这些向量。越高越好。使用 `gensim.models.CoherenceModel`，设置 `coherence="c_v"`。
- **主题多样性。** 所有主题 top 词中唯一词的比例。越高越好（主题不重叠）。
- **定性检查。** 阅读每个主题的 top 词。它们是否命名了一个真实的事物？人类判断仍然是最后的防线。

## 何时选哪个

| 场景 | 选择 |
|-----------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 有主题混合的长文档 | LDA |
| 无 GPU / 有限计算资源 | LDA 或 NMF |
| 需要文档级多主题分布 | LDA |
| LLM 集成做主题标注 | BERTopic（直接支持） |
| 资源受限的边缘部署 | LDA |
| 最大语义连贯性 | BERTopic |

最大的实际考量是文档长度。BERT 嵌入会截断；LDA 计数对任何长度都有效。对于超过嵌入模型上下文长度的文档，要么分块 + 聚合，要么用 LDA。

## 实际应用

2026 年技术栈：

- **BERTopic。** 短文本和语义重要的任何场景的默认选择。
- **`gensim.models.LdaModel`。** 经典 LDA 用于生产，成熟、久经考验。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 实验用的简易 LDA。
- **NMF。** 非负矩阵分解。LDA 的快速替代，短文本上质量相当。
- **Top2Vec。** 与 BERTopic 设计类似。社区较小但在某些基准上表现好。
- **FASTopic。** 更新，在非常大的语料库上比 BERTopic 快。
- **基于 LLM 的标注。** 运行任何聚类，然后提示模型为每个聚类命名。

## 交付

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 练习

1. **简单。** 在 20 Newsgroups 数据集上用 5 个主题拟合 LDA。打印每个主题的 top 10 词。手动标注每个主题。算法找到了真实类别吗？
2. **中等。** 在同一个 20 Newsgroups 子集上拟合 BERTopic。比较发现的主题数量、top 词和定性连贯性与 LDA 的差异。哪个更清晰地呈现了真实类别？
3. **困难。** 在你的语料库上计算 LDA 和 BERTopic 的 c_v 连贯性。分别用 5、10、20、50 个主题运行。绘制连贯性 vs 主题数量的图。报告哪种方法在不同主题数量下更稳定。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| Topic | 语料库讲的一个事 | 词的概率分布（LDA）或相似文档的聚类（BERTopic）。 |
| Mixed membership | 文档属于多个主题 | LDA 为每个文档分配所有主题上的分布。 |
| UMAP | 降维 | 保留局部结构的流形学习；用于 BERTopic。 |
| HDBSCAN | 密度聚类 | 找到可变大小的聚类；为离群点产生"噪声"标签（-1）。 |
| c_v coherence | 主题质量指标 | 滑动窗口内 top 主题词的平均逐点互信息。 |

## 延伸阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入 c_v 等指标的论文。
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 生产参考。优秀的示例。
