# 音频基础 — 波形、采样、傅里叶变换

> 波形是原始信号，spectrogram 是表示形式，mel 特征是对 ML 友好的形态。每个现代 ASR 和 TTS 流水线都沿着这个阶梯往上走，而第一级台阶就是理解采样和傅里叶变换。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Vectors & Matrices), Phase 1 · 14 (Probability Distributions)
**Time:** ~45 minutes

## 问题

麦克风产生一个压力-时间信号。你的神经网络消费张量。两者之间是一堆约定，一旦违反就会产生静默 bug：模型训练正常但 WER 翻倍，或者 TTS 输出嘶嘶声，或者声音克隆系统记住了麦克风而不是说话人。

语音系统中的每个 bug 都可以追溯到三个问题之一：

1. 数据是以什么采样率录制的，模型期望什么采样率？
2. 信号是否发生了 aliasing？
3. 你操作的是原始采样还是频率表示？

搞对这三点，Phase 6 的其余部分就是可控的。搞错的话，即使 Whisper-Large-v4 也会输出垃圾。

## 概念

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**波形。** 一个一维浮点数组，值域 `[-1.0, 1.0]`，按采样编号索引。转换为秒：`t = n / sr`。一段 10 秒、16 kHz 的音频就是一个 160,000 个浮点数的数组。

**采样率 (sr)。** 每秒多少个采样。2026 年常见的采样率：

| Rate | Use |
|------|-----|
| 8 kHz | 电话、传统 VOIP。Nyquist 在 4 kHz 会丢失辅音。ASR 不要用。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都消费 16 kHz。 |
| 22.05 kHz | 旧模型的 TTS vocoder 训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**Nyquist-Shannon 定理。** 采样率为 `sr` 时，能无歧义表示的最高频率为 `sr/2`。这个 `sr/2` 边界就是 *Nyquist 频率*。超过 Nyquist 的能量会被 *aliased* — 折叠到低频 — 从而破坏信号。下采样前务必先做低通滤波。

**位深度。** 16-bit PCM（有符号 int16，范围 ±32,767）是通用交换格式。音乐用 24-bit，内部 DSP 用 32-bit float。`soundfile` 等库读取 int16 但暴露 `[-1, 1]` 范围的 float32 数组。

**傅里叶变换。** 任何有限信号都是不同频率正弦波的叠加。离散傅里叶变换（DFT）对 `N` 个采样计算 `N` 个复数系数 — 每个频率 bin 一个。`bin k` 对应频率 `k · sr / N` Hz。幅度是该频率的振幅，相角是相位。

**FFT。** 快速傅里叶变换：当 `N` 为 2 的幂时，DFT 的 `O(N log N)` 算法。所有音频库底层都用 FFT。16 kHz 下 1024 点 FFT 给出 512 个可用频率 bin，覆盖 0–8 kHz，分辨率 15.6 Hz。

**分帧 + 窗函数。** 我们不对整段音频做 FFT。我们把它切成重叠的 *帧*（通常 25 ms 窗长、10 ms 步进），每帧乘以窗函数（Hann、Hamming）来消除边缘不连续性，然后对每帧做 FFT。这就是短时傅里叶变换（STFT）。第 02 课从这里接续。

## 动手构建

### Step 1：读取音频并绘制波形

`code/main.py` 只使用标准库的 `wave` 模块以保持零依赖。生产环境中你会用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### Step 2：从零合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

16 kHz 下 1 秒的 440 Hz 正弦波（标准 A 音）是 16,000 个浮点数。用 `wave.open(..., "wb")` 以 16-bit PCM 编码写入。

### Step 3：手动计算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — 对 `N=256` 验证正确性没问题，对真实音频没用。实际代码调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### Step 4：找到主频率

幅度峰值索引 `k_star` 对应频率 `k_star * sr / N`。对 440 Hz 正弦波运行应该在 bin `440 * N / sr` 处出现峰值。

### Step 5：演示 aliasing

以 10 kHz 采样一个 7 kHz 正弦波（Nyquist = 5 kHz）。7 kHz 超过了 Nyquist，折叠到 `10 − 7 = 3 kHz`。FFT 峰值出现在 3 kHz。这就是经典的 aliasing 演示，也是每个 DAC/ADC 都配备砖墙低通滤波器的原因。

## 实际使用

2026 年你实际会用到的技术栈：

| Task | Library | Why |
|------|---------|-----|
| 读写 WAV/FLAC/OGG | `soundfile`（libsndfile 封装） | 最快、稳定、返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠。 |
| STFT / Mel | `torchaudio` 或 `librosa` | GPU 友好；PyTorch 生态。 |
| 实时流式处理 | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio 绑定。 |
| 检查文件信息 | `ffprobe` 或 `soxi` | CLI，快速，报告 sr/通道数/编码格式。 |

决策规则：**先匹配采样率，再做其他任何事**。Whisper 期望 16 kHz 单声道 float32。传入 44.1 kHz 立体声，你会得到看起来像模型 bug 的垃圾输出。

## 交付

保存为 `outputs/skill-audio-loader.md`。这个 skill 帮你检查音频输入是否匹配下游模型的期望，并在不匹配时正确重采样。

## 练习

1. **Easy.** 合成一段 1 秒的 220 Hz + 440 Hz + 880 Hz 混合信号，采样率 16 kHz。运行 DFT。确认在预期的 bin 位置出现三个峰值。
2. **Medium.** 录制一段 3 秒、48 kHz 的语音 WAV。用 `torchaudio.transforms.Resample`（带抗混叠）下采样到 16 kHz，再用朴素抽取（每三个采样取一个）下采样到 16 kHz。对两者做 FFT。aliasing 出现在哪里？
3. **Hard.** 仅使用 `math` 和 Step 3 的 DFT 从零构建 STFT。帧长 400，步进 160，Hann 窗。用 `matplotlib.pyplot.imshow` 绘制幅度图。这就是第 02 课的 spectrogram。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Sample rate | 每秒多少个采样 | ADC 测量信号的频率（Hz）。 |
| Nyquist | 能表示的最高频率 | `sr/2`；超过它的能量会 alias 回来。 |
| Bit depth | 每个采样的分辨率 | `int16` = 65,536 级；`float32` = `[-1, 1]` 中 24-bit 精度。 |
| DFT | 序列的傅里叶变换 | `N` 个采样 → `N` 个复数频率系数。 |
| FFT | 快速 DFT | 要求 `N` 为 2 的幂的 `O(N log N)` 算法。 |
| Bin | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | Spectrogram 的底层 | 分帧 + 加窗的 FFT，沿时间轴展开。 |
| Aliasing | 奇怪的频率幽灵 | 超过 Nyquist 的能量镜像折叠到低频 bin。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 采样定理背后的论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费的经典 DSP 教材。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — 带代码的实用入门。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 解释为什么真实世界的音频不是干净正弦波。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10 分钟搞清频率 bin 的直觉。
