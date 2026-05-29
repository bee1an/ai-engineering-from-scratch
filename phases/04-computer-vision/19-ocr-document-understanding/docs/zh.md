# OCR 与文档理解

> OCR 是一个三阶段流水线 — 检测文本框、识别字符、然后排版。每个现代 OCR 系统都在重新排列或合并这些阶段。

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (Detection), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~45 minutes

## 学习目标

- 梳理经典 OCR 流水线（检测 -> 识别 -> 排版）和现代端到端替代方案（Donut, Qwen-VL-OCR）
- 实现 CTC（Connectionist Temporal Classification）loss 用于序列到序列的 OCR 训练
- 使用 PaddleOCR 或 EasyOCR 进行生产级文档解析，无需训练
- 区分 OCR、版面分析和文档理解 — 并为每个任务选择正确的工具

## 问题

包含文字的图像无处不在：收据、发票、身份证、扫描书籍、表单、白板、标牌、截图。从中提取结构化数据 — 不只是字符，而是"这是总金额" — 是最高价值的应用视觉问题之一。

这个领域分为三个技能层：

1. **OCR 本身**：把像素变成文字。
2. **版面分析**：把 OCR 输出分组为区域（标题、正文、表格、页眉）。
3. **文档理解**：从版面中提取结构化字段（"invoice_total = $42.50"）。

每一层都有经典和现代方法，而"我想从图片中获取文字"和"我需要这张收据上的总金额"之间的差距比大多数团队意识到的要大。

## 概念

### 经典流水线

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **文本检测**产出逐行或逐词的四边形。
- **识别**将每个区域裁剪到固定高度，运行 CNN + BiLSTM + CTC 产出字符序列。
- **排版**重建阅读顺序（拉丁文从上到下、从左到右；阿拉伯文、日文不同）。

### 一段话讲清 CTC

OCR 识别从固定长度的特征图产出变长序列。CTC（Graves et al., 2006）让你无需字符级对齐就能训练。模型在每个时间步输出（词表 + blank）上的分布；CTC loss 对所有能在合并重复和移除 blank 后还原为目标文本的对齐方式求边际化。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC 是 CRNN 在 2015 年能工作的原因，2026 年仍然训练着大多数生产级 OCR 模型。

### 现代端到端模型

- **Donut**（Kim et al., 2022）— ViT 编码器 + 文本解码器；读入图像直接输出 JSON。没有文本检测器，没有排版模块。
- **TrOCR** — ViT + transformer 解码器，用于行级 OCR。
- **Qwen-VL-OCR / InternVL** — 为 OCR 任务微调的完整视觉语言模型；2026 年在复杂文档上精度最高。
- **PaddleOCR** — 经典 DB + CRNN 流水线，成熟的生产级包；仍然是开源主力。

端到端模型需要更多数据和算力，但跳过了多阶段流水线的误差累积。

### 版面分析

对于结构化文档，运行版面检测器（LayoutLMv3, DocLayNet）标注每个区域：标题、段落、图表、表格、脚注。阅读顺序就变成了"按版面顺序遍历区域，拼接"。

对于表单，使用 **Key-Value 提取**模型（Donut 用于视觉丰富的文档，LayoutLMv3 用于普通扫描件）。它们接收图像 + 检测到的文本 + 位置，预测结构化的键值对。

### 评估指标

- **字符错误率（CER）** — 编辑距离 / 参考长度。越低越好。生产目标：清晰扫描件 < 2%。
- **词错误率（WER）** — 词级别的同样计算。
- **结构化字段 F1** — 用于键值任务；衡量 `{invoice_total: 42.50}` 是否正确出现。
- **JSON 编辑距离** — 用于端到端文档解析；Donut 论文引入了归一化树编辑距离。

## 动手构建

### Step 1: CTC loss + 贪心解码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 在可用时使用高效的 CuDNN 实现。贪心解码器比 beam search 简单，通常 CER 差距在 1% 以内。

### Step 2: 微型 CRNN 识别器

最小化的 CNN + BiLSTM 用于行级 OCR。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

固定高度输入（CNN 的 max-pool 把高度降到 1）。宽度是 CTC 的时间维度。

### Step 3: 合成 OCR 数据

生成黑底白字的数字字符串用于端到端冒烟测试。

```python
import numpy as np

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        shade = 0.0 if c.isalnum() else 0.5
        img[6:height - 6, x + 2:x + char_width - 2] = shade
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真实 OCR 数据集会加入字体、噪声、旋转、模糊和颜色。流水线结构完全相同。

### Step 4: 训练草图

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在这个简单的合成数据上，loss 应该从 ~3 降到 ~0.2，经过 200 步。

## 实际应用

三条生产路径：

- **PaddleOCR** — 成熟、快速、多语言。一行调用：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR** — Python 原生、多语言、PyTorch backbone。
- **Tesseract** — 经典方案；在模型表现不佳的旧扫描文档上仍然有用。

端到端文档解析用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

对于收据、发票和有重复结构的表单，微调 Donut。对于任意文档或需要推理的 OCR，Qwen-VL-OCR 这样的 VLM 是当前默认选择。

## 交付产出

本课产出：

- `outputs/prompt-ocr-stack-picker.md` — 一个 prompt，根据文档类型、语言和结构选择 Tesseract / PaddleOCR / Donut / VLM-OCR。
- `outputs/skill-ctc-decoder.md` — 一个 skill，从零编写贪心和 beam-search CTC 解码器，包含长度归一化。

## 练习

1. **（简单）** 在 5 位随机数字字符串上训练 TinyCRNN 500 步。报告 held-out 集上的 CER。
2. **（中等）** 用 beam search（beam_width=5）替换贪心解码。报告 CER 差异。在哪些输入上 beam search 更优？
3. **（困难）** 在 20 张收据上使用 PaddleOCR，提取行项目，对 {item_name, price} 对计算 F1，与手工标注的 ground truth 对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| OCR | "从像素中提取文字" | 将图像区域转换为字符序列 |
| CTC | "无对齐 loss" | 训练序列模型无需逐时间步标签的 loss；对所有对齐方式求边际化 |
| CRNN | "经典 OCR 模型" | Conv 特征提取器 + BiLSTM + CTC；2015 年的基线，至今仍在生产中使用 |
| Donut | "端到端 OCR" | ViT 编码器 + 文本解码器；直接从图像输出 JSON |
| Layout parsing | "找区域" | 检测并标注文档中的标题/表格/图表/段落区域 |
| Reading order | "文本序列" | 将识别出的区域排列成句子的顺序；拉丁文简单，混合版面则不简单 |
| CER / WER | "错误率" | 字符或词粒度的编辑距离 / 参考长度 |
| VLM-OCR | "会读的 LLM" | 为 OCR 任务训练或 prompt 的视觉语言模型；复杂文档上的当前 SOTA |

## 延伸阅读

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717) — 原始 CNN+RNN+CTC 架构
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — 原始 CTC 论文；算法思想密集
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664) — 无 OCR 的文档理解 transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — 开源生产级 OCR 技术栈
