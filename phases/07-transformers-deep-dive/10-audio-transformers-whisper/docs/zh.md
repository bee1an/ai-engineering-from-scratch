# 音频 Transformer — Whisper 架构

> 音频是频率随时间变化的图像。Whisper 是一个吃 mel 频谱图、吐文字的 ViT。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time:** ~45 minutes

## 问题

在 Whisper（OpenAI, Radford et al. 2022）之前，最先进的自动语音识别（ASR）意味着 wav2vec 2.0 和 HuBERT——自监督特征提取器加微调头。质量高，但数据管线昂贵、领域脆弱。多语言语音识别需要每个语系单独的模型。

Whisper 做了三个赌注：

1. **在所有数据上训练。** 从互联网抓取的 680,000 小时弱标注音频，覆盖 97 种语言。没有干净的学术语料库。没有音素标签。
2. **多任务单模型。** 一个 decoder 联合训练转录、翻译、语音活动检测、语言识别和时间戳，通过 task token 控制。
3. **标准 encoder-decoder transformer。** Encoder 消费 log-mel 频谱图。Decoder 自回归生成文本 token。没有 vocoder，没有 CTC，没有 HMM。

结果：Whisper large-v3 在各种口音、噪声和零标注数据的语言上都很鲁棒。它是 2026 年每个开源语音助手和大多数商业语音助手的默认语音前端。

## 概念

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### 第 1 步 — 重采样 + 加窗

音频 16 kHz。裁剪/填充到 30 秒。计算 log-mel 频谱图：80 个 mel bin，10 ms 步长 → ~3,000 帧 × 80 特征。这就是 Whisper 看到的"输入图像"。

### 第 2 步 — 卷积 stem

两个 Conv1D 层，kernel 3，stride 2，将 3,000 帧降到 1,500。序列长度减半，参数量增加不多。

### 第 3 步 — encoder

一个 24 层（large 版本）transformer encoder，处理 1,500 个时间步。正弦位置编码、self-attention、GELU FFN。输出 1,500 × 1,280 隐藏状态。

### 第 4 步 — decoder

一个 24 层 transformer decoder。自回归生成 token，词表是 GPT-2 的超集，加了一些音频专用特殊 token。

### 第 5 步 — task token

Decoder 的 prompt 以控制 token 开头，告诉模型要做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型在这个约定上训练。你通过前缀控制任务。这是 2026 年版的 instruction-tuning，但应用于语音。

### 第 6 步 — 输出

Beam search（宽度 5）加 log-prob 阈值。当 `<|notimestamps|>` token 不存在时，每 0.02 秒预测一个时间戳。

### Whisper 尺寸

| Model | Params | Layers | d_model | Heads | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-layer decoder) |

Large-v3-turbo（2024）把 decoder 从 32 层砍到 4 层。解码速度提升 8 倍，WER 回退不到 1 个点。这个解码速度的突破是 Whisper-turbo 成为 2026 年实时语音 agent 默认选择的原因。

### Whisper 不做什么

- 不做说话人分离（谁在说话）。需要配合 pyannote。
- 原生不支持实时流式——30 秒窗口是固定的。现代封装（`faster-whisper`、`WhisperX`）通过 VAD + 重叠实现流式。
- 超过 30 秒没有长程上下文，需要外部分块。实际效果不错，因为人类语音的转录很少需要长程上下文。

### 2026 年格局

| Task | Model | Notes |
|------|-------|-------|
| English ASR | Whisper-turbo, Moonshine | Moonshine is 4× faster on edge |
| Multilingual ASR | Whisper-large-v3 | 97 languages |
| Streaming ASR | faster-whisper + VAD | 150 ms latency targets achievable |
| TTS | Piper, XTTS-v2, Kokoro | Encoder-decoder pattern, but Whisper-shaped |
| Audio + language | AudioLM, SeamlessM4T | Text tokens + audio tokens in one transformer |

## 动手构建

见 `code/main.py`。我们不训练 Whisper——我们构建 log-mel 频谱图管线 + task-token prompt 格式化器。这些是你在生产中实际会接触的部分。

### 第 1 步：合成音频

生成一个 1 秒的 440 Hz 正弦波，采样率 16 kHz。16,000 个采样点。

### 第 2 步：log-mel 频谱图（简化版）

完整的 mel 频谱图需要 FFT。我们做一个简化的分帧 + 逐帧能量版本，展示管线而不需要 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧 = 25 ms，hop = 10 ms。与 Whisper 的加窗一致。逐帧能量代替 mel bin 用于教学。

### 第 3 步：填充到 30 秒

Whisper 总是处理 30 秒的块。将频谱图填充（或裁剪）到 3,000 帧。

### 第 4 步：构建 prompt token

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是整个任务控制面。一个 4-token 前缀。

## 使用方式

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快的 OpenAI 兼容方式：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年什么时候选 Whisper：**

- 一个模型搞定多语言 ASR。
- 对噪声大、多样化音频的鲁棒转录。
- 研究/原型 ASR——最快的起步点。

**什么时候选别的：**

- 边缘设备上的超低延迟流式——Moonshine 在同等质量下比 Whisper 快 4 倍。
- 需要 <200 ms 的实时对话 AI——专用流式 ASR。
- 说话人分离——Whisper 不做这个；配合 pyannote。

## 交付产出

见 `outputs/skill-asr-configurator.md`。该 skill 为新语音应用选择 ASR 模型、解码参数和预处理管线。

## 练习

1. **简单。** 运行 `code/main.py`。确认 1 秒信号在 16 kHz、10 ms hop 下的帧数为 ~100 帧。30 秒为 ~3,000 帧。
2. **中等。** 使用 `numpy.fft` 构建完整的 log-mel 频谱图。验证 80 个 mel bin 与 `librosa.feature.melspectrogram(n_mels=80)` 在数值误差范围内一致。
3. **困难。** 实现流式推理：将音频切成 10 秒窗口、2 秒重叠，对每个块运行 Whisper，合并转录结果。在一个 5 分钟的播客样本上测量词错误率（WER）与单次处理的对比。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Mel spectrogram | "Audio image" | 2D representation: frequency bins on one axis, time frames on the other; log-scaled energy per cell. |
| Log-mel | "What Whisper sees" | Mel spectrogram passed through log; approximates human perception of loudness. |
| Frame | "One time slice" | A 25 ms window of samples; overlapping at 10 ms stride. |
| Task token | "Prompt prefix for speech" | Special tokens like `<\|transcribe\|>` / `<\|translate\|>` in the decoder prompt. |
| Voice activity detection (VAD) | "Find the speech" | Gate that removes silence before ASR; cuts cost massively. |
| CTC | "Connectionist Temporal Classification" | Classic ASR loss for alignment-free training; Whisper does NOT use it. |
| Whisper-turbo | "Small decoder, full encoder" | large-v3 encoder + 4-layer decoder; 8× faster decoding. |
| Faster-whisper | "The production wrapper" | CTranslate2 reimplementation; int8 quantization; 4× faster than OpenAI's reference. |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper paper.
- [OpenAI Whisper repo](https://github.com/openai/whisper) — reference code + model weights. Read `whisper/model.py` to see the Conv1D stem + encoder + decoder top-to-bottom in ~400 lines.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — the beam-search + task-token logic described in Steps 5–6 is here; 500 lines, fully readable.
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — precursor; still SOTA features in some settings.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — production wrapper, 4× faster than reference.
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 edge-friendly ASR, Whisper-shaped but smaller.
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — canonical fine-tuning recipe including mel spectrogram preprocessor and token-timestamp handling.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — full implementation (encoder, decoder, cross-attention, generation) that mirrors the lesson's architecture diagram.
