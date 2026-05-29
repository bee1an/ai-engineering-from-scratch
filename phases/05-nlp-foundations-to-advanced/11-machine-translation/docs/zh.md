# 机器翻译

> 翻译是养活了 NLP 研究三十年、至今仍在买单的任务。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## 问题

模型读入一种语言的句子，输出另一种语言的句子。长度不同，词序不同，有些源语言词对应多个目标词，反之亦然。习语拒绝逐词映射。英语 "I miss you" 在法语里是 "tu me manques"——字面意思是"你对我而言是缺失的"。没有任何词级对齐能撑过这种转换。

机器翻译是迫使 NLP 发明 encoder-decoder、attention、transformer，乃至整个 LLM 范式的任务。每一次进步都源于翻译质量可量化，而人机差距又极其顽固。

本课跳过历史回顾，直接教 2026 年的工作流水线：预训练多语言 encoder-decoder（NLLB-200 或 mBART）、子词分词、beam search、BLEU 和 chrF 评估，以及那些仍然悄悄上线的失败模式。

## 概念

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

现代机器翻译是在平行语料上训练的 transformer encoder-decoder。编码器以源语言的分词方式读入源文本，解码器通过 cross-attention（第 10 课）利用编码器输出，逐子词生成目标文本。解码使用 beam search 来避免贪心解码陷阱。输出经过去分词、去大小写还原，然后与参考译文打分。

三个操作层面的选择决定了实际翻译质量：

- **分词器。** 在混合语言语料上训练的 SentencePiece BPE。跨语言共享词表是 NLLB 实现 zero-shot 语言对的关键。
- **模型规模。** NLLB-200 蒸馏版 600M 可以在笔记本上跑。NLLB-200 3.3B 是公开的生产默认值。54.5B 是研究上限。
- **解码。** 通用内容用 beam width 4-5。Length penalty 防止输出过短。需要术语一致性时使用 constrained decoding。

## 动手构建

### 第 1 步：调用预训练 MT 模型

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三件事很重要。`src_lang` 告诉分词器使用哪种文字和分割方式。`forced_bos_token_id` 告诉解码器生成哪种语言。两者都是 NLLB 特有的技巧；mBART 和 M2M-100 有各自的约定，不能互换。

### 第 2 步：BLEU 和 chrF

BLEU 衡量输出与参考之间的 n-gram 重叠。四种参考 n-gram 大小（1-4），精度的几何平均值，对过短输出施加 brevity penalty。分数范围 [0, 100]。常用但解读起来令人沮丧：30 BLEU 是"可用"；40 是"好"；50 是"优秀"；差异小于 1 BLEU 就是噪声。

chrF 衡量字符级 F-score。对形态丰富的语言更敏感，因为 BLEU 会低估匹配。通常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

务必使用 `sacrebleu`。它会标准化分词，使得分数在不同论文间可比。自己实现 BLEU 计算是产生误导性基准的根源。

### 三层评估体系（2026）

现代 MT 评估使用三个互补的指标族。至少用两个。

- **启发式**（BLEU, chrF）。快速、基于参考、可解释、对改述不敏感。用于历史对比和回归检测。
- **学习型**（COMET, BLEURT, BERTScore）。在人类判断上训练的神经模型；比较翻译与源文本和参考之间的语义相似度。COMET 自 2023 年以来与 MT 研究的相关性最高，是 2026 年质量要求高时的生产默认值。
- **LLM-as-judge**（无参考）。提示大模型对翻译的流畅度、充分性、语气、文化适当性打分。设计好评分标准时，GPT-4-as-judge 与人类一致性约 80%。用于没有参考译文的开放内容。

2026 实用技术栈：`sacrebleu` 算 BLEU 和 chrF，`unbabel-comet` 算 COMET，加一个提示 LLM 作为最终面向用户的信号。在信任生产数据上的指标之前，先用 50-100 个人工标注样本校准每个指标。

无参考指标（COMET-QE, BLEURT-QE, LLM-as-judge）让你无需参考译文即可评估翻译，这对没有参考译文的长尾语言对很重要。

### 第 3 步：生产中会出什么问题

上面的工作流水线 80% 的时间翻译流畅，剩下 20% 会静默失败。命名的失败模式：

- **幻觉。** 模型编造源文本中没有的内容。在不熟悉的领域词汇中常见。症状：输出流畅但声称了源文本未陈述的事实。缓解：对领域术语使用 constrained decoding，对受监管内容进行人工审核，监控输出远长于输入的情况。
- **目标语言错误。** 模型翻译成了错误的语言。NLLB 在罕见语言对上出奇地容易出这个问题。缓解：验证 `forced_bos_token_id`，始终用语言识别模型检查输出。
- **术语漂移。** "Sign up" 在文档 1 中变成 "s'inscrire"，在文档 2 中变成 "créer un compte"。对 UI 文本和面向用户的字符串，一致性比原始质量更重要。缓解：术语表约束解码或后编辑词典。
- **正式度不匹配。** 法语 "tu" vs "vous"，日语敬语级别。模型选择训练中更常见的形式。对面向客户的内容，这通常是错的。缓解：如果模型支持，用正式度 token 作为 prompt 前缀，或在仅正式语料上微调小模型。
- **短输入的长度爆炸。** 非常短的输入句子经常产生过长的翻译，因为 length penalty 在源 token 少于约 5 个时会失效。缓解：按源长度比例设置硬性最大长度上限。

### 第 4 步：领域微调

预训练模型是通才。法律、医学或游戏对话翻译从领域平行数据微调中可以获得可衡量的提升。方法并不特殊：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千条高质量平行样本胜过几十万条嘈杂的网络爬取数据。训练数据质量是生产中最大的杠杆。

## 实际应用

2026 年 MT 生产技术栈：

| 用例 | 推荐起点 |
|---------|---------------------------|
| 任意语言对，200 种语言 | `facebook/nllb-200-distilled-600M`（笔记本）或 `nllb-200-3.3B`（生产） |
| 以英语为中心，高质量，50 种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短文本，低成本推理，英法/德/西 | Helsinki-NLP / Marian 模型 |
| 延迟敏感的浏览器端 | ONNX 量化 Marian（~50 MB） |
| 最高质量，愿意付费 | GPT-4 / Claude / Gemini 配翻译提示 |

截至 2026 年，LLM 在多个语言对上已经超越专用 MT 模型，尤其是在习语内容和长上下文方面。代价是每 token 成本和延迟。当上下文长度、风格一致性或通过提示进行领域适配比吞吐量更重要时，选择 LLM。

## 交付

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习

1. **简单。** 用 `nllb-200-distilled-600M` 将一段 5 句英文翻译成法语再翻回英语。衡量往返结果与原文的接近程度。你应该看到语义保留但用词有漂移。
2. **中等。** 使用 `fasttext lid.176` 或 `langdetect` 对翻译输出实现语言识别检查。集成到 MT 调用中，使目标语言错误在返回前被捕获。
3. **困难。** 在你选择的 5,000 对领域语料上微调 `nllb-200-distilled-600M`。在留出集上测量微调前后的 BLEU。报告哪些类型的句子改善了，哪些退步了。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| BLEU | 翻译分数 | 带 brevity penalty 的 N-gram 精度。[0, 100]。 |
| chrF | 字符 F-score | 字符级 F-score。对形态丰富的语言更敏感。 |
| NMT | 神经机器翻译 | 在平行文本上训练的 transformer encoder-decoder。2017+ 的默认方案。 |
| NLLB | No Language Left Behind | Meta 的 200 种语言 MT 模型族。 |
| Constrained decoding | 受控输出 | 强制特定 token 或 n-gram 出现/不出现在输出中。 |
| Hallucination | 编造内容 | 模型输出中源文本不支持的内容。 |

## 延伸阅读

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 为什么 `sacrebleu` 是报告 BLEU 的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 论文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 实用微调教程。
