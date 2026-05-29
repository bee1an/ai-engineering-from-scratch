# Spectrogram、Mel 尺度与音频特征

> 神经网络不擅长直接消费原始波形。它们消费 spectrogram。消费 mel spectrogram 效果更好。2026 年每个 ASR、TTS 和音频分类器的成败都取决于这一个预处理选择。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 01 (Audio Fundamentals)
**Time:** ~45 minutes

## 问题

取一段 10 秒、16 kHz 的音频。那是 160,000 个浮点数，全在 `[-1, 1]` 范围内，与"狗叫"或"cat 这个词"的标签几乎完全不相关。原始波形包含信息，但形式上模型很难提取。两个相同的音素相隔 100 ms 发出，原始采样完全不同。

Spectrogram 解决了这个问题。它在人类感知忽略的地方（微秒级抖动）压缩时间细节，在感知关注的地方（哪些频率有能量，在 ~10–25 ms 的时间窗口内）保留结构。

Mel spectrogram 更进一步。人类对音高的感知是对数的：100 Hz 到 200 Hz 听起来"距离相同"于 1000 Hz 到 2000 Hz。Mel 尺度将频率轴扭曲以匹配这种感知。Mel 尺度的 spectrogram 是 2010 到 2026 年语音 ML 中最重要的单一特征。

## 概念

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT（短时傅里叶变换）。** 将波形切成重叠帧（典型：25 ms 窗长，10 ms 步进 = 16 kHz 下 400 采样 / 160 采样）。每帧乘以窗函数（Hann 是默认选择；Hamming 有略微不同的权衡）。对每帧做 FFT。将幅度谱堆叠成 `(n_frames, n_freq_bins)` 形状的矩阵。这就是你的 spectrogram。

**对数幅度。** 原始幅度跨越 5-6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩动态范围。所有生产流水线都用对数幅度，而非原始幅度。

**Mel 尺度。** 频率 `f`（Hz）映射到 mel `m`：`m = 2595 * log10(1 + f / 700)`。这个映射在 1 kHz 以下大致线性，1 kHz 以上大致对数。80 个 mel bin 覆盖 0–8 kHz 是标准 ASR 输入。

**Mel filterbank。** 一组在 mel 尺度上等间距的三角滤波器。每个滤波器是相邻 FFT bin 的加权和。将 STFT 幅度乘以 filterbank 矩阵，一次矩阵乘法就得到 mel spectrogram。

**Log-mel spectrogram。** `log(mel_spec + 1e-10)`。Whisper 的输入。Parakeet 的输入。SeamlessM4T 的输入。2026 年通用的音频前端。

**MFCC。** 取 log-mel spectrogram，应用 DCT（type II），保留前 13 个系数。去相关并进一步压缩。在 2015 年左右 CNN/Transformer 直接处理 log-mel 追上来之前，MFCC 是主导特征。在声纹识别（x-vectors、ECAPA）中仍在使用。

**分辨率权衡。** 更大的 FFT = 更好的频率分辨率但更差的时间分辨率。25 ms / 10 ms 是音频 ML 的默认值；音乐用 50 ms / 12.5 ms；瞬态检测（鼓点、爆破音）用 5 ms / 2 ms。

## 动手构建

### Step 1：对波形分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一段 10 秒 16 kHz 的音频，`frame_len=400, hop=160` 产生 998 帧。

### Step 2：Hann 窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在 FFT 前逐元素相乘。消除因在非零端点截断而产生的频谱泄漏。

### Step 3：STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境用 `torch.stft` 或 `librosa.stft`（FFT 支持，向量化）。这里的循环是教学用的；在 `code/main.py` 中对短音频运行。

### Step 4：mel filterbank

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 个 mel 覆盖 0–8 kHz，`n_fft=400` 给出 `(80, 201)` 矩阵。将 `(n_frames, 201)` 的 STFT 幅度乘以其转置，得到 `(n_frames, 80)` 的 mel spectrogram。

### Step 5：log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代方案：`librosa.power_to_db`（参考归一化 dB）、`10 * log10(power + eps)`。Whisper 使用更复杂的 clip + normalize 流程（参见 Whisper 的 `log_mel_spectrogram`）。

### Step 6：MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每个 log-mel 帧应用 DCT，保留前 13 个系数。这就是你的 MFCC 矩阵。第一个系数通常被丢弃（它编码的是整体能量）。

## 实际使用

2026 年技术栈：

| Task | Features |
|------|----------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 log-mels，10 ms hop，25 ms window |
| TTS 声学模型（VITS、F5-TTS、Kokoro） | 80 mels，5–12 ms hop 以获得精细时间控制 |
| 音频分类（AST、PANNs、BEATs） | 128 log-mels，10 ms hop |
| 说话人 embedding（ECAPA-TDNN、WavLM） | 80 log-mels 或原始波形 SSL |
| 音乐（MusicGen、Stable Audio 2） | EnCodec 离散 token（不是 mels） |
| 关键词检测 | 40 MFCC，用于微型设备 |

经验法则：**如果你不是在做音乐，从 80 log-mels 开始。** 任何偏离都需要举证。

## 2026 年仍在出现的坑

- **Mel 数量不匹配。** 训练用 80 mels，推理用 128 mels。静默失败。在两端都记录特征形状。
- **上游采样率不匹配。** 22.05 kHz 计算的 mels 和 16 kHz 的看起来不同。在特征化 *之前* 修正采样率。
- **dB vs log。** Whisper 期望 log-mel，不是 dB-mel。某些 HF pipeline 会自动检测；你的自定义代码不会。
- **归一化漂移。** 训练时按 utterance 归一化，推理时全局归一化。这个生产 bug 会让 WER 翻倍。
- **padding 泄漏。** 对音频末尾零填充会在尾部帧产生平坦频谱。用对称填充或复制填充。

## 交付

保存为 `outputs/skill-feature-extractor.md`。这个 skill 为给定的目标模型选择特征类型、mel 数量、帧长/步进和归一化方式。

## 练习

1. **Easy.** 运行 `code/main.py`。它合成一个 chirp（频率从 200 扫到 4000 Hz）并打印每帧的 argmax mel bin。绘图（可选）并确认它与扫频匹配。
2. **Medium.** 用 `n_mels` 取 `{40, 80, 128}` 和 `frame_len` 取 `{200, 400, 800}` 重新运行。测量时间轴上的尖峰带宽。哪个组合对 chirp 的分辨率最好？
3. **Hard.** 实现 `power_to_db`，在 AudioMNIST 上用一个小 CNN 分类器比较 (a) 原始 log-mel、(b) `ref=max` 的 dB-mel、(c) MFCC-13 + delta + delta-delta 的 ASR 准确率。报告 top-1 accuracy。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Frame | 一个切片 | 送入一次 FFT 的 25 ms 波形块。 |
| Hop | 步幅 | 连续帧之间的采样数；ASR 默认 10 ms。 |
| Window | Hann/Hamming 那个东西 | 将帧边缘渐变到零的逐点乘子。 |
| STFT | Spectrogram 生成器 | 分帧 + 加窗的 FFT；产生时间 × 频率矩阵。 |
| Mel | 扭曲的频率 | 对数感知尺度；`m = 2595·log10(1 + f/700)`。 |
| Filterbank | 那个矩阵 | 将 STFT 投影到 mel bin 的三角滤波器组。 |
| Log-mel | Whisper 的输入 | `log(mel_spec + eps)`；2026 年的标准。 |
| MFCC | 老派特征 | log-mel 的 DCT；13 个系数，去相关。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 原始 mel 尺度。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 阅读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram` 和 hop/window 的参考。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 模型的生产级流水线。
