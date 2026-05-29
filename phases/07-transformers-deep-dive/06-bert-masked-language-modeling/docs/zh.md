# BERT — Masked Language Modeling

> GPT 预测下一个词。BERT 预测缺失的词。一句话的差异——以及半个十年里所有 embedding 相关的东西。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 7 · 05（完整 Transformer）、Phase 5 · 02（文本表示）
**时间：** 约 45 分钟

## 问题

2018 年，每个 NLP 任务——情感分析、NER、QA、蕴含——都在自己的标注数据上从头训练自己的模型。没有预训练的"理解英语"检查点可以微调。ELMo（2018）展示了你可以用双向 LSTM 预训练上下文 embedding；它有帮助但没有泛化。

BERT（Devlin et al. 2018）问了一个问题：如果我们拿一个 transformer encoder，在互联网上的每个句子上训练它，强迫它从两侧的上下文预测缺失的词呢？然后你在下游任务上微调一个 head。参数效率是一个启示。

结果：18 个月内 BERT 及其变体（RoBERTa、ALBERT、ELECTRA）统治了所有存在的 NLP 排行榜。到 2020 年，地球上每个搜索引擎、内容审核管道和语义搜索系统内部都有一个 BERT。

2026 年，encoder-only 模型仍然是分类、检索和结构化抽取的正确工具——它们每 token 比 decoder 快 5–10 倍，其 embedding 是每个现代检索栈的骨干。ModernBERT（2024 年 12 月）用 Flash Attention + RoPE + GeGLU 把架构推到了 8K 上下文。

## 概念

![Masked language modeling：选择 token，mask 它们，预测原始值](../assets/bert-mlm.svg)

### 训练信号

取一个句子：`the quick brown fox jumps over the lazy dog`。

随机 mask 15% 的 token：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型预测 masked 位置的原始 token。因为 encoder 是双向的，预测位置 1 的 `[MASK]` 可以使用位置 2+ 的 `brown fox jumps`。这是 GPT 做不到的事。

### BERT 的 mask 规则

在被选中预测的 15% token 中：

- 80% 被替换为 `[MASK]`。
- 10% 被替换为随机 token。
- 10% 保持不变。

为什么不总是 `[MASK]`？因为 `[MASK]` 在推理时从不出现。如果训练模型在 100% 的 masked 位置都期望 `[MASK]`，会在预训练和微调之间产生分布偏移。10% 随机 + 10% 不变让模型保持诚实。

### Next Sentence Prediction (NSP) — 以及为什么被丢弃

原始 BERT 还在 NSP 上训练：给定两个句子 A 和 B，预测 B 是否跟在 A 后面。RoBERTa（2019）做了消融实验，表明 NSP 有害无益。现代 encoder 跳过它。

### 2026 年的变化：ModernBERT

2024 年的 ModernBERT 论文用 2026 年的原语重建了 block：

| 组件 | Original BERT (2018) | ModernBERT (2024) |
|------|----------------------|-------------------|
| Positional | Learned absolute | RoPE |
| Activation | GELU | GeGLU |
| Normalization | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | Alternating local (128) + global |
| Context length | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

而且不像 2018 年的栈，它原生支持 Flash-Attention。在序列长度 8K 时推理比 DeBERTa-v3 快 2–3 倍，GLUE 分数更好。

### 2026 年仍然选择 encoder 的场景

| 任务 | 为什么 encoder 优于 decoder |
|------|----------------------------|
| 检索 / 语义搜索 embedding | 双向上下文 = 每 token 更好的 embedding 质量 |
| 分类（情感、意图、毒性） | 一次前向传播；没有生成开销 |
| NER / token 标注 | 逐位置输出，天然双向 |
| 零样本蕴含（NLI） | Encoder 上面加分类器 head |
| RAG 的 reranker | Cross-encoder 打分，比 LLM reranker 快 10 倍 |

## 动手构建

### 第 1 步：masking 逻辑

见 `code/main.py`。函数 `create_mlm_batch` 接收 token ID 列表、词表大小和 mask 概率。返回 input ID（应用了 mask）和 label（只在 masked 位置有值，其他位置为 -100——PyTorch 的 ignore index 约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 第 2 步：在小语料上运行 MLM 预测

在 20 词词表、200 个句子上训练 2 层 encoder + MLM head。不用梯度——我们做前向传播的健全性检查。完整训练需要 PyTorch。

### 第 3 步：比较 mask 类型

展示三路规则如何让模型在没有 `[MASK]` 时仍然可用。在未 mask 的句子和 masked 句子上预测。两者都应该产生合理的 token 分布，因为模型在训练中见过两种模式。

### 第 4 步：微调 head

用分类 head 替换 MLM head，在玩具情感数据集上训练。只有 head 训练；encoder 冻结。这是每个 BERT 应用遵循的模式。

## 实际应用

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding 模型就是微调过的 BERT。** `sentence-transformers` 模型如 `all-MiniLM-L6-v2` 是用对比损失训练的 BERT。Encoder 相同。Loss 变了。

**Cross-encoder reranker 也是微调过的 BERT。** 在 `[CLS] query [SEP] doc [SEP]` 上做配对分类。query 和 doc 之间的双向注意力正是 cross-encoder 相比 biencoder 的质量优势所在。

**2026 年什么时候不选 BERT。** 任何生成任务。Encoder 没有合理的方式自回归地产生 token。还有：1B 参数以下的场景，小 decoder 可以用更大灵活性匹配质量（Phi-3-Mini、Qwen2-1.5B）。

## 交付产出

见 `outputs/skill-bert-finetuner.md`。该 skill 为新的分类或抽取任务确定 BERT 微调范围（backbone 选择、head 规格、数据、评估、停止条件）。

## 练习

1. **简单。** 运行 `code/main.py`，打印 10,000 个 token 上的 mask 分布。确认约 15% 被选中，其中约 80% 变成 `[MASK]`。
2. **中等。** 实现全词 masking：如果一个词被分词为子词，要么全部 mask 要么全部不 mask。在 500 句语料上测量这是否提高了 MLM 准确率。
3. **困难。** 在公开数据集的 10,000 个句子上训练一个小型（2 层，d=64）BERT。在 SST-2 情感上微调 `[CLS]` token。与参数量匹配的 decoder-only 基线比较——谁赢？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| MLM | "Masked language modeling" | 训练信号：随机替换 15% 的 token 为 `[MASK]`，预测原始值。 |
| Bidirectional | "两边都看" | Encoder 注意力没有 causal mask——每个位置看到其他所有位置。 |
| `[CLS]` | "Pooler token" | 预置在每个序列前的特殊 token；其最终 embedding 用作句子级表示。 |
| `[SEP]` | "段落分隔符" | 分隔配对序列（如 query/doc、句子 A/B）。 |
| NSP | "Next sentence prediction" | BERT 的第二个预训练任务；在 RoBERTa 中被证明无用，2019 年后被丢弃。 |
| Fine-tuning | "适配到任务" | 保持 encoder 大部分冻结；在上面训练一个小 head 用于下游任务。 |
| Cross-encoder | "Reranker" | 同时接收 query 和 doc 作为输入、输出相关性分数的 BERT。 |
| ModernBERT | "2024 年刷新" | 用 RoPE、RMSNorm、GeGLU、交替 local/global attention、8K 上下文重建的 encoder。 |

## 延伸阅读

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始论文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正确训练 BERT；干掉了 NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 替换 token 检测在匹配计算量下打败 MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 典型 encoder 参考。
