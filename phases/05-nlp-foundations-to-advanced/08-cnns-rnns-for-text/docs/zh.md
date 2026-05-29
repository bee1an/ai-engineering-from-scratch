# CNN 和 RNN 用于文本

> 卷积学习 n-gram。循环记住上下文。两者都被 attention 取代了。两者在受限硬件上仍然重要。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 3 · 11（PyTorch 入门）、Phase 5 · 03（词嵌入）、Phase 4 · 02（从零实现卷积）
**时间：** 约 75 分钟

## 问题

TF-IDF 和 Word2Vec 产生的平坦向量忽略了词序。基于它们构建的分类器无法区分 `dog bites man` 和 `man bites dog`。词序有时承载信号。

在 transformer 到来之前，两个架构家族填补了这个空白。

**文本卷积网络（TextCNN）。** 在词嵌入序列上应用一维卷积。宽度为 3 的滤波器是一个可学习的 trigram 检测器：它跨越三个词并输出一个分数。堆叠不同宽度（2、3、4、5）来检测多尺度模式。最大池化到定长表示。扁平、并行、快速。

**循环网络（RNN、LSTM、GRU）。** 逐个处理 token，维护一个向前传递信息的隐藏状态。顺序的、有记忆的、灵活的输入长度。从 2014 到 2017 年主导序列建模，然后 attention 出现了。

本课构建两者，然后指出驱动 attention 发明的失败模式。

## 概念

**TextCNN**（Kim, 2014）。Token 被嵌入。宽度为 `k` 的一维卷积在连续 `k`-gram 的嵌入上滑动滤波器，产生特征图。全局最大池化选取最强激活。拼接多个滤波器宽度的最大池化输出。送入分类头。

为什么有效。一个滤波器就是一个可学习的 n-gram。最大池化是位置不变的，所以 "not good" 在评论开头或中间触发相同的特征。三种滤波器宽度各 100 个滤波器给你 300 个学习到的 n-gram 检测器。训练是并行的；没有顺序依赖。

**RNN。** 在每个时间步 `t`，隐藏状态 `h_t = f(W * x_t + U * h_{t-1} + b)`。`W`、`U`、`b` 跨时间共享。时间 `T` 的隐藏状态是整个前缀的摘要。分类时，在 `h_1 ... h_T` 上池化（max、mean 或 last）。

普通 RNN 有梯度消失问题。**LSTM** 添加门控来决定遗忘什么、存储什么、输出什么，稳定长序列的梯度。**GRU** 将 LSTM 简化为两个门；性能相似，参数更少。

**双向 RNN** 正向运行一个 RNN，反向运行另一个，拼接隐藏状态。每个 token 的表示同时看到左右上下文。对标注任务至关重要。

## 动手构建

### 第 1 步：PyTorch 中的 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将 `[batch, seq_len, embed_dim]` 变形为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间轴视为通道。池化输出是定长的，与输入长度无关。

### 第 2 步：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

在序列上做最大池化，而不是取最后状态。对于分类，最大池化通常优于取最后隐藏状态，因为长序列末尾的信息往往主导最后状态。

### 第 3 步：梯度消失演示（直觉）

没有门控的普通 RNN 无法学习长程依赖。考虑一个玩具任务：预测 token `A` 是否出现在序列中的任何位置。如果 `A` 在位置 1 而序列有 100 个 token 长，损失的梯度必须通过 99 次循环权重的乘法回流。如果权重小于 1，梯度消失。如果大于 1，梯度爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM 通过**细胞状态**修复这个问题，细胞状态在网络中只有加法交互（遗忘门乘法缩放它，但梯度仍沿"高速公路"流动）。GRU 用更少的参数做类似的事。两者都能在 100+ 步序列上稳定训练。

### 第 4 步：为什么这仍然不够

即使有 LSTM，三个问题仍然存在。

1. **顺序瓶颈。** 在长度 1000 的序列上训练 RNN 需要 1000 次串行前向/反向步骤。无法跨时间并行化。
2. **encoder-decoder 设置中的定长上下文向量。** Decoder 只看到 encoder 的最终隐藏状态，压缩了整个输入。长输入丢失细节。第 09 课直接覆盖这个。
3. **远程依赖准确率天花板。** LSTM 优于普通 RNN，但仍然难以在 200+ 步上传播特定信息。

Attention 解决了这三个问题。Transformer 完全丢弃了循环。第 10 课是转折点。

## 使用现成工具

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 是生产就绪的。训练代码是标准的。

Hugging Face 提供预训练 embedding 作为输入层：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

适用场景清单：

- **边缘/设备端推理。** TextCNN 加 GloVe embedding 比 transformer 小 10-100 倍。如果部署目标是手机，这就是技术栈。
- **流式/在线分类。** RNN 逐个处理 token；transformer 需要完整序列。对于实时输入文本，LSTM 仍然胜出。
- **小模型做基线。** 在新任务上快速迭代。CPU 上 5 分钟训练一个 TextCNN。
- **有限数据的序列标注。** BiLSTM-CRF（第 06 课）对于 1k-10k 标注句子仍然是生产级 NER 架构。

其他一切都交给 transformer。

## 交付

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习

1. **简单。** 在 3 类玩具数据集上训练 TextCNN（你自己造数据）。验证滤波器宽度 (2, 3, 4) 在平均 F1 上优于单一宽度 (3)。
2. **中等。** 为 LSTM 分类器实现 max-pool、mean-pool 和 last-state pooling。在小数据集上比较；记录哪种池化胜出并假设原因。
3. **困难。** 构建 BiLSTM-CRF NER 标注器（结合第 06 课和本课）。在 CoNLL-2003 上训练。与第 06 课的纯 CRF 基线和 BERT 微调对比。报告训练时间、内存和 F1。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| TextCNN | 文本 CNN | 词嵌入上的一维卷积堆叠加全局最大池化。Kim (2014)。 |
| RNN | 循环网络 | 每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控 RNN | 添加输入/遗忘/输出门 + 细胞状态。长序列上稳定训练。 |
| GRU | 更简单的 LSTM | 两个门代替三个。准确率相似，参数更少。 |
| 双向 | 两个方向 | 正向 + 反向 RNN 拼接。每个 token 看到两侧上下文。 |
| 梯度消失 | 训练信号消亡 | 普通 RNN 中 <1 权重的重复乘法使早期步骤的梯度实际为零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 论文。八页。可读。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 论文。出乎意料地清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让 LSTM 对所有人变得可理解的图解。
