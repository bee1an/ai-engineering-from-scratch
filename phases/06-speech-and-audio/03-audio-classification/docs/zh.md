# 音频分类 — 从 MFCC 上的 k-NN 到 AST 和 BEATs

> 从"狗叫 vs 警笛"到"这是什么语言"，都是音频分类。特征是 mels。架构每十年换一次。评估指标始终是 AUC、F1 和 per-class recall。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNNs), Phase 5 · 08 (CNNs & RNNs for Text)
**Time:** ~75 minutes

## 问题

你拿到一段 10 秒的音频。你想知道："这是什么？"城市声音（警笛、电钻、狗叫）、语音命令（yes/no/stop）、语言识别（en/es/ar）、说话人情绪（愤怒/中性），或环境声音（室内/室外、嘈杂人声）。所有这些都是 *音频分类*，2026 年的基线架构已经成熟：log-mel → CNN 或 Transformer → softmax。

核心难点不在网络，而在数据。音频数据集有严重的类别不平衡、强烈的域偏移（干净 vs 噪声），以及标签噪声（谁决定了"城市嘈杂"和"餐厅噪声"的区别？）。80% 的问题在于数据整理、增强和评估，而不是把 CNN 换成 Transformer。

## 概念

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**MFCC 上的 k-NN（1990 年代基线）。** 将每段音频的 MFCC 展平，计算与标注库的余弦相似度，返回 top K 的多数投票。在干净的小数据集（Speech Commands、ESC-50）上出奇地强。不需要 GPU。

**Log-mel 上的 2D CNN（2015-2019）。** 把 `(T, n_mels)` 的 log-mel 当作图像。用 ResNet-18 或 VGG 风格网络。对时间轴做全局平均池化。Softmax 分类。在 2026 年大多数 Kaggle 比赛中仍是基线。

**Audio Spectrogram Transformer, AST（2021-2024）。** 将 log-mel 切成 patch（如 16×16），加位置编码，送入 ViT。在 AudioSet 上的有监督学习 SOTA（mAP 0.485）。

**BEATs 和 WavLM-base（2024-2026）。** 在数百万小时音频上自监督预训练。用你原本需要的 1-10% 有监督数据在你的任务上微调。2026 年这是非语音音频的默认起点。BEATs-iter3 在 AudioSet 上比 AST 高 1-2 mAP，同时只用 1/4 的计算量。

**Whisper encoder 作为冻结 backbone（2024）。** 取 Whisper 的 encoder，丢掉 decoder，接一个线性分类器。在语言识别和简单事件分类上接近 SOTA，零音频增强。"免费午餐"基线。

### 类别不平衡才是真正的挑战

ESC-50：50 类，每类 40 段 — 平衡，简单。UrbanSound8K：10 类，不平衡 10:1。AudioSet：632 类，长尾比 100,000:1。有效的技术：

- 训练时平衡采样（评估时不要）。
- Mixup：线性插值两段音频（及其标签）作为增强。
- SpecAugment：遮蔽随机的时间和频率带。简单但关键。

### 评估

- 多类互斥（Speech Commands）：top-1 accuracy、top-5 accuracy。
- 多类多标签（AudioSet、UrbanSound 风格）：mean average precision (mAP)。
- 严重不平衡：per-class recall + macro F1。

2026 年你应该知道的数字：

| Benchmark | Baseline | SOTA 2026 | Source |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

## 动手构建

### Step 1：特征提取

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Step 2：定长摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但有效：对时间轴取均值 + 方差，为 13 系数 MFCC 给出 26 维定长 embedding。瞬间运行。直到 2017 年还能在 ESC-50 上击败 SOTA 神经网络基线。

### Step 3：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Step 4：升级到 log-mel 上的 CNN

PyTorch 实现：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M 参数。在单张 RTX 4090 上 ESC-50 训练约 10 分钟。80%+ 准确率。

### Step 5：2026 年的默认方案 — 微调 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

对于 BEATs，通过 `beats` 库使用 `microsoft/BEATs-base`；transformers API 形状相同。

## 实际使用

2026 年技术栈：

| Situation | Start with |
|-----------|-----------|
| 小数据集（<1000 段） | MFCC 均值上的 k-NN（你的基线）+ 音频增强 |
| 中等数据集（1K–100K） | BEATs 或 AST 微调 |
| 大数据集（>100K） | 从头训练或微调 Whisper-encoder |
| 实时、边缘设备 | 40-MFCC CNN，量化到 int8（KWS 风格） |
| 多标签（AudioSet） | BEATs-iter3 + BCE loss + mixup + SpecAugment |
| 语言识别 | MMS-LID、SpeechBrain VoxLingua107 基线 |

决策规则：**从冻结的 backbone 开始，而不是从头训练模型**。微调 BEATs 的 head 能在几小时内达到 SOTA 的 95%，而不是几周。

## 交付

保存为 `outputs/skill-classifier-designer.md`。为给定的音频分类任务选择架构、增强方法、类别平衡策略和评估指标。

## 练习

1. **Easy.** 运行 `code/main.py`。它在一个 4 类合成数据集（不同音高的纯音）上训练 k-NN MFCC 基线。报告混淆矩阵。
2. **Medium.** 将 `summarize` 替换为 [mean, var, skew, kurtosis]。4 阶矩池化在同一合成数据集上是否优于 mean+var？
3. **Hard.** 使用 `torchaudio`，在 ESC-50 fold 1 上训练 2D CNN。报告 5-fold 交叉验证准确率。加入 SpecAugment（time mask = 20, freq mask = 10）并报告差异。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| AudioSet | 音频界的 ImageNet | Google 的 2M 片段、632 类弱标注 YouTube 数据集。 |
| ESC-50 | 小型分类基准 | 50 类 × 40 段环境声音。 |
| AST | Audio Spectrogram Transformer | log-mel patch 上的 ViT；2021 SOTA。 |
| BEATs | 自监督音频 | 微软模型，iter3 在 2026 年领跑 AudioSet。 |
| Mixup | 配对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于遮蔽的增强 | 将 spectrogram 的随机时间和频率带置零。 |
| mAP | 主要多标签指标 | 跨类别和阈值的平均精度均值。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021–2024 年的标准架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024+ 的默认选择。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 主导的音频增强方法。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 至今仍在使用的 50 类基准。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632 类 YouTube 分类体系；仍是金标准。
