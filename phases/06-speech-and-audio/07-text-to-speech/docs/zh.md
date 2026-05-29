# 语音合成 (TTS) — 从 Tacotron 到 F5 和 Kokoro

> ASR 把语音反转为文本；TTS 把文本反转为语音。2026 年的技术栈分三部分：text → tokens，tokens → mel，mel → waveform。每部分都有一个能在笔记本上跑的默认模型。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## 问题

你有一个字符串："Please remind me to water the plants at 6 pm." 你需要一段 3 秒的音频，听起来自然，韵律正确（停顿、重音），"plants" 的元音发音正确，并且在 CPU 上 300 ms 内完成以支持实时语音助手。你还需要切换声音、处理混合语言输入（"remind me at 6 pm, daijoubu?"），以及不在人名上出丑。

现代 TTS 流水线长这样：

1. **文本前端。** 文本归一化（日期、数字、邮箱），转换为音素或 subword token，预测韵律特征。
2. **声学模型。** Text → mel spectrogram。Tacotron 2 (2017)、FastSpeech 2 (2020)、VITS (2021)、F5-TTS (2024)、Kokoro (2024)。
3. **Vocoder。** Mel → waveform。WaveNet (2016)、WaveRNN、HiFi-GAN (2020)、BigVGAN (2022)、2024+ 的 neural codec vocoder。

2026 年声学模型 + vocoder 的分界随着端到端 diffusion 和 flow-matching 模型变得模糊。但三部分的心智模型在调试时仍然成立。

## 概念

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2 (2017)。** Seq2seq：char-embedding → BiLSTM encoder → location-sensitive attention → 自回归 LSTM decoder 输出 mel 帧。慢（AR），长文本不稳定。仍被引用为基线。

**FastSpeech 2 (2020)。** 非自回归。Duration predictor 输出每个音素占多少 mel 帧。一次前向，比 Tacotron 快 10 倍。损失一些自然度（单调对齐）但到处在用。

**VITS (2021)。** 端到端联合训练 encoder + flow-based duration + HiFi-GAN vocoder，使用变分推断。高质量，单模型。2022–2024 年主导的开源 TTS。变体：YourTTS（多说话人零样本）、XTTS v2（2024, Coqui）。

**F5-TTS (2024)。** 基于 flow matching 的 diffusion transformer。自然韵律，5 秒参考音频即可零样本声音克隆。2026 年开源 TTS 排行榜榜首。335M 参数。

**Kokoro (2024)。** 小型（82M），CPU 可运行，实时使用中最佳的英语 TTS。封闭词表、仅英语，Apache-2.0。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商业 SOTA。ElevenLabs v2.5 的情感标签（"[whispered]"、"[laughing]"）和角色声音在 2026 年主导有声书制作。

### Vocoder 演进

| Era | Vocoder | Latency | Quality |
|-----|---------|---------|---------|
| 2016 | WaveNet | offline only | SOTA at release |
| 2018 | WaveRNN | ~realtime | good |
| 2020 | HiFi-GAN | 100× realtime | near-human |
| 2022 | BigVGAN | 50× realtime | generalizes across speakers/langs |
| 2024 | SNAC, DAC (neural codecs) | integrated with AR models | discrete tokens, bit-efficient |

到 2026 年大多数"TTS"模型是端到端从文本到波形的；mel spectrogram 是内部表示。

### 评估

- **MOS (Mean Opinion Score)。** 1–5 分，众包评分。仍是金标准；非常慢。
- **CMOS (Comparative MOS)。** A vs B 偏好。每次标注的置信区间更紧。
- **UTMOS、DNSMOS。** 无参考的神经 MOS 预测器。用于排行榜。
- **CER (Character Error Rate) via ASR。** 将 TTS 输出送入 Whisper，计算与输入文本的 CER。可懂度的代理指标。
- **SECS (Speaker Embedding Cosine Similarity)。** 声音克隆质量。

2026 年 LibriTTS test-clean 上的数字：

| Model | UTMOS | CER (via Whisper) | Size |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 动手构建

### Step 1：音素化输入

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用桥梁。不要把原始文本直接喂给 VITS 级别以下的模型。

### Step 2：运行 Kokoro（2026 CPU 默认）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

离线运行，单文件，82M 参数。

### Step 3：用 F5-TTS 做声音克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传入 5 秒参考音频 + 其转录文本；F5 克隆韵律和音色。

### Step 4：从零构建 HiFi-GAN vocoder

太大放不进教程脚本，但结构是：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

训练：对抗式（短窗口上的判别器）+ mel-spectrogram 重建 loss + feature-matching loss。已商品化 — 使用 `hifi-gan` 仓库或 nvidia-NeMo 的预训练 checkpoint。

### Step 5：完整流水线（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 实际使用

2026 年技术栈：

| Situation | Pick |
|-----------|------|
| 实时英语语音助手 | Kokoro (CPU) 或 XTTS v2 (GPU) |
| 5 秒参考的声音克隆 | F5-TTS |
| 商业角色声音 | ElevenLabs v2.5 |
| 有声书朗读 | ElevenLabs v2.5 或 XTTS v2 + 微调 |
| 低资源语言 | 在 5–20 小时目标语言数据上训练 VITS |
| 表现力 / 情感标签 | ElevenLabs v2.5 或 StyleTTS 2 微调 |

2026 年开源领先者：**F5-TTS 主打质量，Kokoro 主打效率**。除非你是历史学家，否则不要用 Tacotron。

## 常见坑

- **没有文本归一化器。** "Dr. Smith" 读成 "Doctor" 还是 "Drive"？"2026" 读成 "twenty twenty six" 还是 "two zero two six"？在 phonemizer 之前做归一化。
- **OOV 专有名词。** "Ghumare" → "ghyu-mair"？为未知 token 配备一个 fallback grapheme-to-phoneme 模型。
- **削波。** Vocoder 输出很少削波，但推理时 mel 缩放不匹配可能超出 ±1.0。始终 `np.clip(wav, -1, 1)`。
- **采样率不匹配。** Kokoro 输出 24 kHz；你的下游流水线期望 16 kHz → 重采样，否则会 aliasing。

## 交付

保存为 `outputs/skill-tts-designer.md`。为给定的声音、延迟和语言目标设计 TTS 流水线。

## 练习

1. **Easy.** 运行 `code/main.py`。从一个玩具词表构建音素字典，估计每个音素的时长，打印一个假的"mel"调度。
2. **Medium.** 安装 Kokoro，用 `af_bella` 和 `am_adam` 两个声音合成同一句话。比较音频时长和主观质量。
3. **Hard.** 录制一段 5 秒的自己的参考音频。用 F5-TTS 克隆它。报告参考和克隆输出之间的 SECS。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Phoneme | 声音单元 | 抽象的声音类别；英语有 39 个（ARPABet）。 |
| Duration predictor | 每个音素持续多久 | 非 AR 模型输出；每个音素的整数帧数。 |
| Vocoder | Mel → waveform | 将 mel-spec 映射到原始采样的神经网络。 |
| HiFi-GAN | 标准 vocoder | 基于 GAN；2020–2024 年主导。 |
| MOS | 主观质量 | 人类评分者的 1–5 平均意见分。 |
| SECS | 声音克隆指标 | 目标和输出说话人 embedding 之间的余弦相似度。 |
| F5-TTS | 2024 开源 SOTA | Flow-matching diffusion；零样本克隆。 |
| Kokoro | CPU 英语领先者 | 82M 参数模型，Apache 2.0。 |

## 延伸阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seq 基线。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 端到端 flow-based。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前开源 SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026 年仍在使用的 vocoder。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 CPU 友好的英语 TTS。
