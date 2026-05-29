# 语音 Anti-Spoofing 与 Audio Watermarking — ASVspoof 5、AudioSeal、WaveVerify

> Voice cloning 的部署速度超过了防御手段。2026 年的生产语音系统需要两样东西：一个检测器（AASIST、RawNet2）分类真实 vs 伪造语音，以及一个水印（AudioSeal）能在压缩和编辑后存活。两者都部署，否则不要上线 voice cloning。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 minutes

## 问题

三种相关的防御手段：

1. **Anti-spoofing / deepfake 检测。** 给定一段音频，它是合成的还是真实的？ASVspoof benchmark（ASVspoof 2019 → 2021 → 5）是金标准。
2. **Audio watermarking。** 在生成的音频中嵌入不可感知的信号，检测器之后可以提取。AudioSeal（Meta）和 WavMark 是开源选项。
3. **认证溯源。** 音频文件的加密签名 + 元数据。C2PA / Content Authenticity Initiative。

检测处理不合作的对手。Watermarking 处理合规——AI 生成的音频应该可以被识别为 AI 生成。2026 年两者都是必需的。

## 概念

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024-2025 benchmark

与之前版本的最大变化：

- **众包数据**（不是录音棚干净音频）——真实条件。
- **约 2000 个说话人**（之前约 100 个）。
- **32 种攻击算法。** TTS + voice conversion + 对抗扰动。
- **两个赛道。** Countermeasure (CM) 独立检测；Spoofing-robust ASV (SASV) 用于生物识别系统。

ASVspoof 5 上的 SOTA：约 7.23% EER。在更早的 ASVspoof 2019 LA 上：0.42% EER。真实部署：在野外音频上预期 5-10% EER。

### AASIST 和 RawNet2 — 检测模型家族

**AASIST**（2021，持续更新到 2026）。频谱特征上的 graph-attention。当前 ASVspoof 5 countermeasure 任务的 SOTA。

**RawNet2。** 原始波形上的卷积前端 + TDNN backbone。更简单的 baseline；微调后仍有竞争力。

**NeXt-TDNN + SSL features。** 2025 变体：ECAPA 风格 + WavLM 特征 + focal loss。在 ASVspoof 2019 LA 上达到 0.42% EER。

### AudioSeal — 2024 年的水印默认方案

Meta 的 **AudioSeal**（2024 年 1 月，v0.2 2024 年 12 月）。核心设计：

- **局部化。** 在 16 kHz 采样分辨率（1/16000 s）上逐帧检测水印。
- **Generator + detector 联合训练。** Generator 学习嵌入不可听信号；detector 学习通过数据增强找到它。
- **鲁棒。** 能在 MP3 / AAC 压缩、EQ、±10% 变速、+10 dB SNR 噪声混合下存活。
- **快速。** Detector 以 485 倍实时速度运行；比 WavMark 快 1000 倍。
- **容量。** 16-bit payload（可编码模型 ID、生成时间戳、用户 ID）可嵌入每段话语。

### WavMark

AudioSeal 之前的开源 baseline。可逆神经网络，32 bits/sec。问题：

- 同步暴力搜索很慢。
- 可被高斯噪声或 MP3 压缩移除。
- 不适合实时。

### WaveVerify（2025 年 7 月）

解决 AudioSeal 的弱点——特别是时间域操作（反转、变速）。使用 FiLM-based generator + Mixture-of-Experts detector。在标准攻击上与 AudioSeal 竞争；能处理时间域编辑。

### 对手利用的缺口

来自 AudioMarkBench："在 pitch shift 下，所有水印的 Bit Recovery Accuracy 低于 0.6，表明几乎完全被移除。" **Pitch-shift 是通用攻击。** 2026 年没有水印能完全抵抗激进的 pitch 修改。这就是为什么你需要检测（AASIST）配合 watermarking。

### C2PA / Content Authenticity Initiative

不是 ML 技术——是一种 manifest 格式。音频文件携带关于创建工具、作者、日期的加密签名元数据。Audobox / Seamless 使用它。对溯源有用；如果坏人重新编码并剥离元数据则无效。

## 动手构建

### Step 1：简单的频谱特征检测器（toy）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音通常有异常平坦的高频能量。生产检测器使用 AASIST，不是这个。但直觉是对的。

### Step 2：AudioSeal 嵌入 + 检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### Step 3：评估 — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Step 4：生产集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次生成都附带：(1) 水印，(2) 签名 manifest，(3) 符合保留策略的审计日志。

## 使用指南

| Use case | Defense |
|----------|---------|
| Shipping TTS / voice cloning | AudioSeal embed on every output (non-negotiable) |
| Biometric voice unlock | AASIST + ECAPA ensemble; liveness challenge |
| Call-center fraud detection | AASIST on 20% sample of incoming calls |
| Podcast authenticity | C2PA signing on upload, AudioSeal if AI-generated |
| Research / training detectors | ASVspoof 5 train/dev/eval sets |

## 常见陷阱

- **嵌入水印但从不运行检测器。** 毫无意义。把检测器放进你的 CI。
- **检测无校准。** 在 ASVspoof LA 上训练的 AASIST 会过拟合；真实世界准确率下降。在你的领域上校准。
- **Pitch-shift 缺口。** 激进的 pitch shift 能移除大多数水印。要有检测后备方案。
- **元数据剥离重新托管。** C2PA 可以通过重新编码轻易绕过。始终同时添加加密 + 感知（水印）防御。
- **活体检测作为检测手段。** 让用户说一个随机短语。防止重放攻击但不防实时克隆。

## 交付

保存为 `outputs/skill-spoof-defender.md`。为语音生成部署选择检测模型、水印、溯源 manifest 和运营手册。

## 练习

1. **Easy。** 运行 `code/main.py`。Toy 检测器 + toy 水印嵌入/检测，作用于合成音频。
2. **Medium。** 安装 `audioseal`，在 TTS 输出中嵌入 16-bit payload，重新解码。用噪声损坏音频并测量 Bit Recovery Accuracy。
3. **Hard。** 在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST。测量 EER。在 F5-TTS 生成的留出集上测试——观察 OOD 检测如何退化。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| ASVspoof | The benchmark | Biennial challenge; 2024 = ASVspoof 5. |
| CM (countermeasure) | Detector | Classifier: real speech vs synthetic / converted. |
| SASV | Speaker verif + CM | Integrated biometric + spoof detection. |
| AudioSeal | Meta watermark | Localized, 16-bit payload, 485× faster than WavMark. |
| Bit Recovery Accuracy | Watermark survival | Fraction of payload bits recovered after attack. |
| C2PA | Provenance manifest | Cryptographic metadata about creation / authorship. |
| AASIST | Detector family | Graph-attention-based anti-spoofing SOTA. |

## 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — the current benchmark.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — the watermark default.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — MoE detector for temporal attacks.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — the SOTA detection backbone.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — robustness evaluation.
- [C2PA specification](https://c2pa.org/specifications/specifications/) — provenance manifest format.
