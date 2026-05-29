# 声纹识别与验证

> ASR 问的是"他们说了什么？"声纹识别问的是"谁在说？"数学看起来一样 — embedding 加余弦 — 但每个生产决策都取决于一个 EER 数字。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 minutes

## 问题

一个用户说了一句口令。你想知道：这是不是他们声称的那个人（*验证*，1:1），还是你注册库中的第一个人（*辨识*，1:N）？或者都不是 — 这是一个未知说话人（*开集*）？

2018 年前：GMM-UBM + i-vectors。EER 尚可但对信道偏移（手机 vs 笔记本）和情绪脆弱。2018–2022：x-vectors（TDNN backbone + angular margin 训练）。2022+：ECAPA-TDNN 和 WavLM-large embeddings。到 2026 年，这个领域由三个模型和一个指标主导。

这个指标就是 **EER** — Equal Error Rate。设定决策阈值使 False Accept Rate = False Reject Rate。交叉点就是 EER。每篇论文、每个排行榜、每次采购评估都用它。

## 概念

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**流水线。** 注册：录制目标说话人 5–30 秒音频；计算定维 embedding（ECAPA-TDNN 为 192 维，WavLM-large 为 256 维）。验证：获取测试语音的 embedding；计算余弦相似度；与阈值比较。

**ECAPA-TDNN（2020，2026 年仍主导）。** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network。1D 卷积块 + squeeze-excitation + 多头注意力池化，接线性层到 192 维。在 VoxCeleb 1+2（2,700 说话人，1.1M 语音）上用 Additive Angular Margin loss（AAM-softmax）训练。

**WavLM-SV（2022+）。** 在预训练的 WavLM-large SSL backbone 上用 AAM loss 微调。质量更高但更慢 — 300+ MB vs 15 MB。

**x-vector（基线）。** TDNN + 统计池化。经典；在 CPU / 边缘设备上仍有用。

**AAM-softmax。** 标准 softmax 在角度空间加入 margin `m`：对正确类别用 `cos(θ + m)`。强制类间角度分离。典型 `m=0.2`，scale `s=30`。

### 打分

- **余弦** 注册和测试 embedding 之间。基于阈值决策。
- **PLDA（概率线性判别分析）。** 将 embedding 投影到潜空间，同说话人 vs 不同说话人有闭式似然比。在余弦之上加 PLDA 可降低 10–20% EER。2020 年前的标准；现在只在闭集场景使用。
- **分数归一化。** `S-norm` 或 `AS-norm`：对每个分数用冒充者群体的均值和标准差归一化。跨域评估必备。

### 2026 年你应该知道的数字

| Model | VoxCeleb1-O EER | Params | Throughput (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### 说话人分离（Diarization）

"多说话人音频中谁在什么时候说话"。流水线：VAD → 分段 → 对每段提取 embedding → 聚类（凝聚或谱聚类）→ 平滑边界。现代方案：`pyannote.audio` 3.1，将说话人分割 + embedding + 聚类封装在一次调用中。2026 年 AMI 上的 SOTA DER 约 15%（2022 年为 23%）。

## 动手构建

### Step 1：基于 MFCC 统计量的简易 embedding

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

远非 SOTA — 仅用于教学。`code/main.py` 在合成说话人数据上用它做概念验证。

### Step 2：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Step 3：从相似度对计算 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回 (eer, threshold_at_eer)。两个都要报告。

### Step 4：用 SpeechBrain 生产部署

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### Step 5：用 pyannote 做说话人分离

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 实际使用

2026 年技术栈：

| Situation | Pick |
|-----------|------|
| 闭集 1:1 验证，边缘设备 | ECAPA-TDNN + 余弦阈值 |
| 开集验证，云端 | WavLM-SV + AS-norm |
| 说话人分离（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 反欺骗（重放 / deepfake 检测） | AASIST 或 RawNet2 |
| 微型嵌入式（KWS + 注册） | Titanet-Small (NeMo) |

## 常见坑

- **信道不匹配。** 在 VoxCeleb（网络视频）上训练的模型 ≠ 电话音频。务必在目标信道上评估。
- **短语音。** 测试音频低于 3 秒时 EER 急剧恶化。
- **带噪注册。** 一条噪声注册会毒化锚点。用 ≥3 条干净样本取平均。
- **跨条件固定阈值。** 务必在目标域的 held-out dev 集上调阈值。
- **未归一化 embedding 上的余弦。** 先做 L2 归一化；否则幅度会主导。

## 交付

保存为 `outputs/skill-speaker-verifier.md`。选择模型、注册协议、阈值调优方案和防欺诈措施。

## 练习

1. **Easy.** 运行 `code/main.py`。构建合成"说话人"（不同音调配置），注册，在 100 对试验列表上计算 EER。
2. **Medium.** 用 SpeechBrain ECAPA 在 30 条 VoxCeleb1 语音（5 说话人 × 6 条）上计算余弦 vs PLDA 的 EER。
3. **Hard.** 用 `pyannote.audio` 构建完整的注册 → 分离 → 验证流水线。在 AMI dev 集上评估 DER。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| EER | 核心指标 | False Accept = False Reject 时的阈值。 |
| Verification | 1:1 | "这是 Alice 吗？" |
| Identification | 1:N | "谁在说话？" |
| Open-set | 可能有未知人 | 测试集可能包含未注册的说话人。 |
| Enrollment | 注册 | 计算说话人的参考 embedding。 |
| AAM-softmax | 那个 loss | 带加性角度 margin 的 softmax；强制聚类分离。 |
| PLDA | 经典打分 | 概率 LDA；在 embedding 之上做似然比打分。 |
| DER | 分离指标 | Diarization Error Rate — miss + false alarm + confusion。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 经典深度 embedding 论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020–2026 年的主导架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — 用于声纹验证和分离的 SSL backbone。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — 生产级分离 + embedding 方案。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 各模型当前 EER 排名。
