# Neural Audio Codecs — EnCodec、SNAC、Mimi、DAC 与 Semantic-Acoustic 分离

> 2026 年的音频生成几乎全是 token。EnCodec、SNAC、Mimi 和 DAC 将连续波形转化为 transformer 可以预测的离散序列。Semantic vs acoustic token 的分离——第一个 codebook 作为 semantic，其余作为 acoustic——是自 Transformer 以来音频领域最重要的架构变革。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## 问题

语言模型处理离散 token。音频是连续的。如果你想为语音/音乐构建 LLM 风格的模型——MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus——你首先需要一个 **neural audio codec**：一个学习到的编码器，将音频离散化为小词表的 token，以及一个匹配的解码器来重建波形。

两个家族已经形成：

1. **重建优先的 codec** — EnCodec、DAC。优化感知音频质量。Token 是"acoustic"的——它们捕获一切，包括说话人身份、音色、背景噪声。
2. **语义优先的 codec** — Mimi（Kyutai）、SpeechTokenizer。强制第一个 codebook 编码语言/音素内容（通常通过从 WavLM 蒸馏）。后续 codebook 是声学细节。

2024-2026 的洞察：**纯重建 codec 在从文本生成时会产生模糊的语音。** 基于 codec token 的 LLM 必须在同一个 codebook 中同时学习语言结构和声学结构，这无法扩展。将它们分开——semantic codebook 0、acoustic codebooks 1-N——正是 Moshi 和 Sesame CSM 能工作的原因。

## 概念

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### 核心技巧：Residual Vector Quantization (RVQ)

与其用一个大 codebook（需要数百万个码才能获得好质量），所有现代音频 codec 都使用 **RVQ**：一系列小 codebook 的级联。第一个 codebook 量化编码器输出；第二个量化残差；以此类推。每个 codebook 有 1024 个码。8 个 codebook = 有效词表 1024^8 = 10^24。

推理时，解码器将每帧所有选中的码求和来重建。

### 2026 年重要的四个 codec

**EnCodec（Meta, 2022）。** 基线。波形上的 encoder-decoder，RVQ 瓶颈。24 kHz，最多 32 个 codebook，默认 4 codebook @ 1.5 kbps。使用 `1D conv + transformer + 1D conv` 架构。被 MusicGen 使用。

**DAC（Descript, 2023）。** RVQ + L2 归一化 codebook、周期性激活函数、改进的损失函数。所有开源 codec 中重建保真度最高——12 个 codebook 时有时与原始语音无法区分。44.1 kHz 全频带。

**SNAC（Hubert Siuzdak, 2024）。** 多尺度 RVQ——粗粒度 codebook 以比细粒度更低的帧率运行。有效地分层建模音频：~12 Hz 的粗略"草图"加上 50 Hz 的细节。被 Orpheus-3B 使用，因为层次结构很好地映射到基于 LM 的生成。

**Mimi（Kyutai, 2024）。** 2026 年的 game-changer。12.5 Hz 帧率（极低），8 codebook @ 4.4 kbps。Codebook 0 **从 WavLM 蒸馏**——训练来预测 WavLM 的语音内容特征。Codebook 1-7 是声学残差。这种分离驱动了 Moshi（Lesson 15）和 Sesame CSM。

### 帧率对语言建模很重要

更低的帧率 = 更短的序列 = 更快的 LM。

| Codec | Frame rate | 1 s = N frames | Good for |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | music, general audio |
| DAC-44.1k | 86 Hz | 86 | high-fidelity music |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM efficient |
| Mimi | 12.5 Hz | 12.5 | streaming speech |

在 12.5 Hz 下，一段 10 秒的语音只有 125 个 codec 帧——transformer 可以轻松预测。

### Semantic vs acoustic tokens

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Semantic token（Mimi 中的 codebook 0）。** 编码说了什么——音素、词、内容。通过辅助预测损失从 WavLM 蒸馏。
- **Acoustic tokens（codebook 1-7）。** 编码音色、说话人身份、韵律、背景噪声、精细细节。

AR LM 先预测 semantic token（以文本为条件），然后预测 acoustic tokens（以 semantic + 说话人参考为条件）。这种分解是现代 TTS 能零样本克隆声音的原因：semantic 模型处理内容；acoustic 模型处理音色。

### 2026 重建质量（比特率越低越好）

| Codec | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

传统 codec 如 Opus 在每比特感知质量上仍然胜出。Neural codec 的优势在于**离散 token**（Opus 不产生）和**生成模型质量**（LM 能用这些 token 做什么）。

## 动手构建

### Step 1：用 EnCodec 编码

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

6 kbps 时 `n_codebooks=8`。每个码是 0-1023（10-bit）。

### Step 2：解码并测量重建质量

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Step 3：semantic-acoustic 分离（Mimi 风格）

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Semantic codebook 0 与 WavLM 对齐。你可以训练一个 text-to-semantic transformer——词表比直接生成音频小得多。然后一个独立的 acoustic-to-waveform decoder 以说话人参考为条件。

### Step 4：为什么 AR LM over codec tokens 能工作

对于 Mimi 12.5 Hz × 8 codebook 下的 10 秒语音片段：

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 个 token 对 transformer 来说是微不足道的上下文。一个 256M 参数的 transformer 可以在现代 GPU 上毫秒级生成 10 秒语音。

## 使用指南

问题 → codec 映射：

| Task | Codec |
|------|-------|
| General music generation | EnCodec-24k |
| Highest-fidelity reconstruction | DAC-44.1k |
| AR LM over speech (TTS) | SNAC or Mimi |
| Streaming full-duplex speech | Mimi (12.5 Hz) |
| Sound-effect library with text | EnCodec + T5 condition |
| Fine-grained audio editing | DAC + inpainting |

经验法则：**如果你在构建生成模型，从 Mimi 或 SNAC 开始。如果你在构建压缩管线，用 Opus。**

## 常见陷阱

- **Codebook 太多。** 增加 codebook 线性提升保真度，但也线性增加 LM 序列长度。停在 8-12。
- **帧率不匹配。** 在 12.5 Hz Mimi 上训练 LM 然后在 50 Hz EnCodec 上微调会静默失败。
- **假设所有 codebook 等价。** 在 Mimi 中，codebook 0 承载内容；丢失它会摧毁可懂度。丢失 codebook 7 几乎察觉不到。
- **只用重建质量作为指标。** 一个 codec 可以有很好的重建质量，但如果语义结构不好，对基于 LM 的生成毫无用处。

## 交付

保存为 `outputs/skill-codec-picker.md`。为给定的生成或压缩任务选择 codec。

## 练习

1. **Easy。** 运行 `code/main.py`。它实现了一个 toy scalar + residual quantizer，并测量随着 codebook 增加重建误差的变化。
2. **Medium。** 安装 `encodec`，在一个留出的语音片段上比较 1、4、8、32 个 codebook。绘制 PESQ 或 MSE vs bitrate 的图。
3. **Hard。** 加载 Mimi。编码一个片段。将 codebook 0 替换为随机整数；解码。然后类似地替换 codebook 7。比较两种损坏——codebook 0 损坏应该摧毁可懂度；codebook 7 损坏应该几乎没有变化。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| RVQ | Residual quantization | Cascade of small codebooks; each quantizes the previous residual. |
| Frame rate | Codec speed | How many token-frames per second. Lower = faster LM. |
| Semantic codebook | Codebook 0 (Mimi) | Codebook distilled from SSL features; encodes content. |
| Acoustic codebooks | Everything else | Timbre, prosody, noise, fine detail. |
| PESQ / ViSQOL | Perceptual quality | Objective metrics correlating with MOS. |
| EnCodec | Meta codec | The RVQ baseline; used by MusicGen. |
| Mimi | Kyutai codec | 12.5 Hz frame rate; semantic-acoustic split; powers Moshi. |

## 延伸阅读

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — the RVQ baseline.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — highest-fidelity open.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — multi-scale RVQ.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — semantic-acoustic split, WavLM distillation.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — the two-stage semantic/acoustic paradigm.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — the original streamable RVQ codec.
