# 问答系统

> 三种系统塑造了现代 QA。抽取式找到文本片段。检索增强式将其锚定在文档中。生成式产出答案。每个现代 AI 助手都是三者的混合。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Time:** ~75 minutes

## 问题

用户输入"第一代 iPhone 什么时候发布的？"，期望得到"2007 年 6 月 29 日"。不是"苹果的历史悠久而丰富"。不是孤零零的"2007"没有上下文。一个直接的、有据可查的、正确的答案。

过去十年，三种架构主导了 QA：

- **抽取式 QA。** 给定一个问题和一段已知包含答案的文本，找到答案片段在文本中的起止索引。SQuAD 是经典基准。
- **开放域 QA。** 文本不是给定的。先检索相关段落，然后抽取或生成答案。这是今天每个 RAG 流水线的基石。
- **生成式 / 闭卷 QA。** 大语言模型从其参数记忆中回答。无检索。推理最快，事实可靠性最低。

2026 年的趋势是混合式：检索最佳的几段文本，然后提示生成模型基于这些段落回答。这就是 RAG，第 14 课深入讲解检索部分。本课构建 QA 部分。

## 概念

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**抽取式。** 用 transformer（BERT 系列）一起编码问题和段落。训练两个头来预测答案的起始和结束 token 索引。损失是有效位置上的交叉熵。输出是段落中的一个片段。构造上不会产生幻觉，构造上也无法处理段落无法回答的问题。

**检索增强式（RAG）。** 两个阶段。首先，检索器从语料库中找到 top-`k` 段落。然后，阅读器（抽取式或生成式）使用这些段落产出答案。检索器-阅读器的分离让每个部分可以独立训练和评估。现代 RAG 通常在两者之间加一个 reranker。

**生成式。** 一个 decoder-only LLM（GPT, Claude, Llama）从学到的权重中回答。无检索步骤。对常见知识表现优秀，对罕见或最新事实则是灾难性的。幻觉率与事实在预训练数据中的出现频率成反比。

## 动手构建

### 第 1 步：用预训练模型做抽取式 QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，包含无法回答的问题。默认情况下，`question-answering` pipeline 即使模型的 null score 胜出也会返回最高分片段——它*不会*自动返回空答案。要获得显式的"无答案"行为，在 pipeline 调用中传入 `handle_impossible_answer=True`：此时 pipeline 仅在 null score 超过所有片段分数时才返回空答案。无论如何都要检查 `score` 字段。

### 第 2 步：检索增强流水线（草图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段流水线。稠密检索器（Sentence-BERT）通过语义相似度找到相关段落。抽取式阅读器（RoBERTa-SQuAD）从合并的 top 段落中提取答案片段。适用于小语料库。对于百万文档级语料库，使用 FAISS 或向量数据库。

### 第 3 步：生成式 RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

Prompt 模式很重要。明确告诉模型基于上下文回答，并在上下文不足时返回"我不知道"，相比朴素提示可以将幻觉率降低 40-60%。更精细的模式会添加引用、置信度分数和结构化提取。

### 第 4 步：反映真实世界的评估

SQuAD 使用 **Exact Match (EM)** 和 **token 级 F1**。EM 是标准化后的严格匹配（小写、去标点、去冠词）——预测要么完全匹配得 1 分，要么得 0 分。F1 基于预测和参考之间的 token 重叠计算，给予部分分数。两者都低估改述："June 29, 2007" vs "June 29th, 2007" 通常 EM 为 0（序数词破坏了标准化），但由于重叠 token 仍能获得可观的 F1。

对于生产 QA：

- **答案准确率**（LLM 判断或人工判断，因为指标无法捕捉语义等价）。
- **引用准确率。** 引用的段落是否真的支持答案？通过生成引用与检索段落之间的字符串匹配可以自动检查。
- **拒答校准。** 当答案不在检索段落中时，系统是否正确地说"我不知道"？测量虚假自信率。
- **检索召回率。** 在评估阅读器之前，测量检索器是否将正确段落放入了 top-`k`。阅读器无法修复缺失的段落。

### RAGAS：2026 生产评估框架

`RAGAS` 专为 RAG 系统构建，是 2026 年的生产默认值。它在不需要黄金参考的情况下对四个维度打分：

- **忠实度。** 答案中的每个声明是否来自检索到的上下文？通过基于 NLI 的 textual entailment 测量。你的主要幻觉指标。
- **答案相关性。** 答案是否回应了问题？通过从答案生成假设问题并与真实问题比较来测量。
- **上下文精确率。** 检索到的块中，有多少比例实际相关？低精确率 = prompt 中有噪声。
- **上下文召回率。** 检索集是否包含了所有需要的信息？低召回率 = 阅读器无法成功。

无参考评分让你可以在实时生产流量上评估，无需精心策划的黄金答案。在精确匹配指标无用的开放式问题上，叠加 LLM-as-judge。

`pip install ragas`。接入你的检索器 + 阅读器。每个查询获得四个标量。对回归发出警报。

## 实际应用

2026 年技术栈：

| 用例 | 推荐 |
|---------|-------------|
| 给定段落，找答案片段 | `deepset/roberta-base-squad2` |
| 在固定语料库上，闭卷不可接受 | RAG：稠密检索器 + LLM 阅读器 |
| 在文档存储上实时查询 | RAG + 混合（BM25 + 稠密）检索器 + reranker（第 14 课） |
| 对话式 QA（追问） | LLM + 对话历史 + 每轮 RAG |
| 高度事实性、受监管领域 | 在权威语料库上做抽取式；绝不单独使用生成式 |

抽取式 QA 在 2026 年不太流行，因为 RAG + LLM 能处理更多场景。但在需要逐字引用的场景中仍然在用：法律研究、合规监管、审计工具。

## 交付

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习

1. **简单。** 在 10 段维基百科文本上搭建上述 SQuAD 抽取式流水线。手工编写 10 个问题。测量答案正确的频率。如果段落和问题干净，你应该看到 7-9 个正确。
2. **中等。** 添加拒答分类器。当最高检索分数低于阈值（比如 0.3 余弦相似度）时，返回"我不知道"而不是调用阅读器。在留出集上调整阈值。
3. **困难。** 在你选择的 10,000 文档语料库上构建 RAG 流水线。实现混合检索（BM25 + 稠密）+ RRF 融合（见第 14 课）。测量有无混合步骤时的答案准确率。记录哪些问题类型受益最大。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| Extractive QA | 找答案片段 | 预测给定段落中答案的起止索引。 |
| Open-domain QA | 在语料库上做 QA | 没有给定段落；必须先检索再回答。 |
| RAG | 检索后生成 | Retrieval-augmented generation。检索器 + 阅读器流水线。 |
| SQuAD | 经典基准 | Stanford Question Answering Dataset。EM + F1 指标。 |
| Hallucination | 编造答案 | 阅读器输出不被检索上下文支持。 |
| Refusal calibration | 知道何时闭嘴 | 系统在无法回答时正确地说"我不知道"。 |

## 延伸阅读

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，QA 的经典稠密检索器。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名 RAG 的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 全面的 RAG 综述。
