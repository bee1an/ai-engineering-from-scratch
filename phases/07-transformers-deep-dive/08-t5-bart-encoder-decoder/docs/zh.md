# T5、BART — Encoder-Decoder 模型

> Encoder 理解。Decoder 生成。把它们重新组合，你就得到一个为输入 → 输出任务而生的模型：翻译、摘要、改写、转录。

**类型：** 学习
**语言：** Python
**前置课程：** Phase 7 · 05（完整 Transformer）、Phase 7 · 06（BERT）、Phase 7 · 07（GPT）
**时间：** 约 45 分钟

## 问题

Decoder-only 的 GPT 和 encoder-only 的 BERT 各自为不同目标精简了 2017 年的架构。但很多任务天然是输入-输出的：

- 翻译：英语 → 法语。
- 摘要：5,000 token 文章 → 200 token 摘要。
- 语音识别：音频 token → 文本 token。
- 结构化抽取：散文 → JSON。

对于这些任务，encoder-decoder 是最干净的匹配。Encoder 产生源的稠密表示。Decoder 生成输出，每一步都 cross-attend 到那个表示。训练是输出侧的 shift-by-one。和 GPT 相同的 loss，只是以 encoder 输出为条件。

两篇论文定义了现代 playbook：

1. **T5**（Raffel et al. 2019）。"Text-to-Text Transfer Transformer。" 每个 NLP 任务重新表述为 text-in、text-out。单一架构、单一词表、单一 loss。在 masked span prediction 上预训练（破坏输入中的 span，在输出中解码它们）。
2. **BART**（Lewis et al. 2019）。"Bidirectional and Auto-Regressive Transformer。" 去噪自编码器：以多种方式破坏输入（打乱、mask、删除、旋转），让 decoder 重建原始内容。

2026 年 encoder-decoder 格式在输入结构重要的地方继续存在：

- Whisper（语音 → 文本）。
- Google 的翻译栈。
- 一些有明确上下文-编辑结构的代码补全/修复模型。
- Flan-T5 及其变体用于结构化推理任务。

Decoder-only 赢得了聚光灯，但 encoder-decoder 从未消失。

## 概念

![带 cross-attention 的 encoder-decoder](../assets/encoder-decoder.svg)

### 前向循环

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键是，encoder 对每个输入只运行一次。Decoder 自回归运行，但每一步都 cross-attend 到*相同的* encoder 输出。缓存 encoder 输出对长输入是免费的加速。

### T5 预训练 — span corruption

随机选取输入的 span（平均长度 3 token，总共 15%）。用唯一的 sentinel 替换每个 span：`<extra_id_0>`、`<extra_id_1>` 等。Decoder 只输出被破坏的 span 及其 sentinel 前缀：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

比预测整个序列更便宜的信号。在 T5 论文的消融中与 MLM（BERT）和 prefix-LM（UniLM）竞争力相当。

### BART 预训练 — 多噪声去噪

BART 尝试五种噪声函数：

1. Token masking。
2. Token deletion。
3. Text infilling（mask 一个 span，decoder 插入正确长度）。
4. Sentence permutation。
5. Document rotation。

组合 text infilling + sentence permutation 产生了最好的下游数字。Decoder 总是重建原始内容。BART 的输出是完整序列，不只是被破坏的 span——所以预训练计算量比 T5 高。

### 推理

和 GPT 相同的自回归生成。Greedy / beam / top-p 采样都适用。Beam search（宽度 4–5）是翻译和摘要的标准，因为输出分布比对话更窄。

### 2026 年何时选择哪个变体

| 任务 | Encoder-decoder？ | 为什么 |
|------|-------------------|--------|
| 翻译 | 是，通常 | 明确的源序列；固定的输出分布；beam search 有效 |
| 语音转文本 | 是（Whisper） | 输入模态与输出不同；encoder 塑造音频特征 |
| 对话 / 推理 | 否，decoder-only | 没有持久的"输入"——对话就是序列 |
| 代码补全 | 通常否 | 长上下文的 decoder-only 赢；代码模型如 Qwen 2.5 Coder 是 decoder-only |
| 摘要 | 都行 | BART、PEGASUS 打败了早期 decoder-only 基线；现代 decoder-only LLM 匹配它们 |
| 结构化抽取 | 都行 | T5 很干净，因为"text → text"吸收任何输出格式 |

自约 2022 年以来的趋势：decoder-only 接管了 encoder-decoder 曾经拥有的任务，因为 (a) 指令微调的 decoder-only LLM 通过 prompting 泛化到任何事情，(b) 一个架构比两个更容易 scale，(c) RLHF 假设有一个 decoder。Encoder-decoder 在输入模态不同（语音、图像）或 beam search 质量重要的地方坚守。

## 动手构建

见 `code/main.py`。我们为玩具语料实现 T5 风格的 span corruption——本课最有用的单个部分，因为它出现在此后的每个 encoder-decoder 预训练配方中。

### 第 1 步：span corruption

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式是 T5 约定：`<sent0> span0 <sent1> span1 ...`。被破坏的输入在 span 位置交错放置未改变的 token 和 sentinel token。

### 第 2 步：验证往返

给定被破坏的输入和目标，重建原始句子。如果你的 corruption 是可逆的，前向传播就是良定义的。这是健全性检查——真正的训练从不这样做，但测试很便宜，能捕获 span 记账中的 off-by-one bug。

### 第 3 步：BART 噪声

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 实际应用

HuggingFace 参考：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的技巧：任务名称放入输入文本。同一个模型处理几十个任务，因为每个任务都是 text-in、text-out。2026 年这个模式已被指令微调的 decoder-only 模型泛化，但 T5 最先将其编纂。

## 交付产出

见 `outputs/skill-seq2seq-picker.md`。该 skill 根据输入-输出结构、延迟和质量目标，在 encoder-decoder 和 decoder-only 之间选择。

## 练习

1. **简单。** 运行 `code/main.py`，对 30 token 的句子应用 span corruption，验证拼接非 sentinel 源 token 和解码的目标 span 能重现原始内容。
2. **中等。** 实现 BART 的 `text_infill` 噪声：用单个 `<mask>` token 替换随机 span，decoder 必须推断正确的 span 长度加内容。展示一个例子。
3. **困难。** 在小型英语 → pig-Latin 语料（200 对）上微调 `flan-t5-small`。在 50 对留出集上测量 BLEU。与在相同数据和相同计算量下微调 `Llama-3.2-1B` 比较。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Encoder-decoder | "Seq2seq transformer" | 两个栈：双向 encoder 处理输入，带 cross-attention 的 causal decoder 处理输出。 |
| Cross-attention | "源和目标对话的地方" | Decoder 的 Q × encoder 的 K/V。Encoder 信息进入 decoder 的唯一通道。 |
| Span corruption | "T5 的预训练技巧" | 用 sentinel token 替换随机 span；decoder 输出这些 span。 |
| Denoising objective | "BART 的游戏" | 对输入应用噪声函数，训练 decoder 重建干净序列。 |
| Sentinel token | "`<extra_id_N>` 占位符" | 在源中标记被破坏 span、在目标中重新标记它们的特殊 token。 |
| Flan | "指令微调的 T5" | 在 >1,800 个任务上微调的 T5；让 encoder-decoder 在指令遵循上有竞争力。 |
| Beam search | "解码策略" | 每步保留 top-k 个部分序列；翻译/摘要的标准。 |
| Teacher forcing | "训练时的输入" | 训练时向 decoder 输入真实的前一个输出 token，而不是采样的那个。 |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年典型的 encoder-decoder。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 参考实现。
