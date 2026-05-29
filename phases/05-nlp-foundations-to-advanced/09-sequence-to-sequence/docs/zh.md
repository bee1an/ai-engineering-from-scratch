# 序列到序列模型

> 两个 RNN 假装是翻译器。它们遇到的瓶颈就是 attention 存在的原因。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 5 · 08（CNN + RNN 用于文本）、Phase 3 · 11（PyTorch 入门）
**时间：** 约 75 分钟

## 问题

分类将变长序列映射到单个标签。翻译将变长序列映射到另一个变长序列。输入和输出在不同的词汇表中，可能是不同的语言，没有长度对等的保证。

Seq2seq 架构（Sutskever, Vinyals, Le, 2014）用一个刻意简单的配方破解了这个问题。两个 RNN。一个读取源句子并产生定长上下文向量。另一个读取该向量并逐 token 生成目标句子。和你在第 08 课写的代码一样，只是拼接方式不同。

这值得学习有两个原因。第一，上下文向量瓶颈是 NLP 中最有教学价值的失败。它驱动了 attention 和 transformer 擅长的一切。第二，训练配方（teacher forcing、scheduled sampling、推理时的 beam search）仍然适用于每个现代生成系统，包括 LLM。

## 概念

**Encoder。** 一个读取源句子的 RNN。它的最终隐藏状态就是**上下文向量** — 整个输入的定长摘要。理论上不丢失任何源信息。

**Decoder。** 另一个从上下文向量初始化的 RNN。每一步它接收之前生成的 token 作为输入，产生目标词汇表上的分布。采样或 argmax 选择下一个 token。反馈回去。重复直到产生 `<EOS>` token 或达到最大长度。

**训练：** 每个 decoder 步骤的交叉熵损失，在序列上求和。通过两个网络的标准时间反向传播。

**Teacher forcing。** 训练期间，decoder 在步骤 `t` 的输入是位置 `t-1` 的*真实* token，而不是 decoder 自己之前的预测。这稳定了训练；没有它，早期错误会级联，模型永远学不会。推理时你必须使用模型自己的预测，所以总是存在训练/推理分布差距。这个差距叫做**暴露偏差（exposure bias）**。

**瓶颈。** Encoder 学到的关于源的一切都必须挤进那一个上下文向量。长句子丢失细节。稀有词被模糊。重排序（chat noir vs. black cat）必须被记忆，而不是被计算。

Attention（第 10 课）通过让 decoder 查看*每个* encoder 隐藏状态而不仅仅是最后一个来修复这个问题。这就是全部卖点。

## 动手构建

### 第 1 步：一个 encoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 形状为 `[batch, seq_len, hidden_dim]` — 每个输入位置一个隐藏状态。`hidden` 形状为 `[1, batch, hidden_dim]` — 最后一步。第 08 课说"对 outputs 池化做分类"。这里我们保留最后隐藏状态作为上下文向量，忽略逐步输出。

### 第 2 步：一个 decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder 每次调用一步。输入：一批单个 token 和当前隐藏状态。输出：下一个 token 的词汇表 logits 和更新后的隐藏状态。

### 第 3 步：带 teacher forcing 的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

两个值得说明的参数。`ignore_index=0` 跳过 padding token 的损失。`teacher_forcing_ratio` 是每步使用真实 token vs. 模型预测的概率。从 1.0（完全 teacher forcing）开始，在训练过程中退火到约 0.5 以缩小暴露偏差差距。

### 第 4 步：推理循环（贪心）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码在每步选择最高概率的 token。它可能偏离：一旦你提交了一个 token，就无法收回。**Beam search** 保持前 `k` 个部分序列存活，最后选择得分最高的完整序列。Beam 宽度 3-5 是标准。

### 第 5 步：瓶颈演示

在玩具复制任务上训练模型：源 `[a, b, c, d, e]`，目标 `[a, b, c, d, e]`。增加序列长度。观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU 隐藏状态无法无损记忆 40 个 token 的输入。信息在每个 encoder 步骤都在，但 decoder 只看到最后状态。Attention 直接修复这个问题。

## 使用现成工具

PyTorch 有 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库提供在数十亿 token 上训练的完整 encoder-decoder 模型（BART、T5、mBART、NLLB）。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代 encoder-decoder 用 transformer 替代了 RNN。高层形状（encoder、decoder、逐 token 生成）与 2014 年的 seq2seq 论文完全相同。每个块内部的机制不同。

### 何时仍然选择基于 RNN 的 seq2seq

几乎从不，对于新项目。特定例外：

- 流式翻译，你逐个 token 消费输入且内存有界。
- 设备端文本生成，transformer 内存成本过高。
- 教学。理解 encoder-decoder 瓶颈是理解为什么 transformer 赢了的最快路径。

### 暴露偏差及其缓解

- **Scheduled sampling。** 在训练过程中退火 teacher forcing 比率，让模型学会从自己的错误中恢复。
- **最小风险训练。** 在句子级 BLEU 分数上训练而不是 token 级交叉熵。更接近你实际想要的。
- **强化学习微调。** 用指标奖励序列生成器。用于现代 LLM RLHF。

三者仍然适用于基于 transformer 的生成。

## 交付

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习

1. **简单。** 实现玩具复制任务。在输入-输出对（目标等于源）上训练 GRU seq2seq。在长度 5、10、20 上测量准确率。复现瓶颈。
2. **中等。** 添加 beam width 为 3 的 beam search 解码。在小型平行语料库上对比贪心测量 BLEU。记录 beam search 在哪里胜出（通常是最后几个 token）以及在哪里没有区别。
3. **困难。** 在 10k 对释义数据集上微调 `facebook/bart-base`。将微调模型的 beam-4 输出与基础模型在留出输入上的输出对比。报告 BLEU 并选 10 个定性示例。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Encoder | 输入 RNN | 读取源。产生逐步隐藏状态和最终上下文向量。 |
| Decoder | 输出 RNN | 从上下文向量初始化。逐个生成目标 token。 |
| 上下文向量 | 摘要 | Encoder 最终隐藏状态。定长。Attention 解决的瓶颈。 |
| Teacher forcing | 使用真实 token | 训练时喂入真实的前一个 token。稳定学习。 |
| 暴露偏差 | 训练/测试差距 | 在真实 token 上训练的模型从未练习过从自己的错误中恢复。 |
| Beam search | 更好的解码 | 每步保持前 k 个部分序列存活，而不是贪心提交。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原始 seq2seq 论文。四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入了 GRU 和 encoder-decoder 框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — attention 论文。本课之后立即阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可构建的 seq2seq + attention 代码。
