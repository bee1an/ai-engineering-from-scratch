# Voice Activity Detection 与 Turn-Taking — Silero、Cobra 与 Flush Trick

> 每个语音 agent 的生死取决于两个决策：用户现在在说话吗？他们说完了吗？VAD 回答第一个。Turn-detection（VAD + silence-hangover + 语义端点模型）回答第二个。任何一个搞错，你的助手要么打断用户，要么永远不闭嘴。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time:** ~45 minutes

## 问题

语音 agent 在每个 20 ms chunk 上做三个不同的决策：

1. **这一帧是语音吗？** — VAD。逐帧二分类。
2. **用户开始了一段新的话语吗？** — onset detection。
3. **用户说完了吗？** — end-pointing（轮次结束）。

朴素的答案（能量阈值）在任何噪声下都会失败——交通、键盘、人群嘈杂。2026 年的答案：Silero VAD（开源、深度学习）+ turn-detection 模型（语义端点检测）+ VAD 校准的 silence hangover。

## 概念

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### 三层 VAD 级联

**Tier 1：能量门控。** 最便宜。RMS 阈值设在 -40 dBFS。过滤明显的静音，但对任何超过阈值的噪声都会触发。

**Tier 2：Silero VAD**（2020-2026, MIT）。100 万参数。在 6000+ 种语言上训练。单 CPU 线程上每 30 ms chunk 约 1 ms。87.7% TPR @ 5% FPR。开源默认选择。

**Tier 3：语义 turn detector。** LiveKit 的 turn-detection 模型（2024-2026）或你自己的小分类器。区分"句中停顿"和"说完了"。使用语言上下文（语调 + 最近的词），而不仅仅是静音。

### 关键参数及其默认值

- **阈值。** Silero 输出概率；在 > 0.5（默认）或 > 0.3（敏感）时分类为语音。更低的阈值 = 更少的首词截断，更多的误触发。
- **最小语音时长。** 拒绝短于 250 ms 的语音——通常是咳嗽或椅子噪声。
- **Silence hangover（end-pointing）。** VAD 回到 0 后，等待 500-800 ms 再宣布轮次结束。太短 → 打断用户。太长 → 感觉迟钝。
- **Pre-roll buffer。** 保留 VAD 触发前 300-500 ms 的音频。防止"hey"被截断。

### Flush trick（Kyutai 2025）

Streaming STT 模型有前瞻延迟（Kyutai STT-1B 为 500 ms，STT-2.6B 为 2.5 s）。正常情况下你需要在语音结束后等那么久才能拿到转录。Flush trick：当 VAD 触发语音结束时，**向 STT 发送 flush 信号**强制立即输出。STT 以约 4 倍实时速度处理，所以 500 ms 的缓冲区在约 125 ms 内完成。

端到端：125 ms VAD + flush STT = 对话级延迟。

### 2026 VAD 对比

| VAD | TPR @ 5% FPR | Latency | License |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | commercial |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero 是正确的默认选择。Cobra 是合规/精度升级。纯能量 VAD 在 2026 年的生产环境中没有位置。

## 动手构建

### Step 1：能量门控

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Step 2：Python 中的 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Step 3：轮次结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Step 4：flush trick 骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush 才能使用。Whisper streaming 不支持——它是基于块的，总是等待 chunk。

## 使用指南

| Situation | VAD choice |
|-----------|-----------|
| Open, fast, general | Silero VAD |
| Commercial call center | Cobra VAD |
| On-device (phone) | Silero VAD ONNX |
| Research / diarization | pyannote segmentation |
| Zero-dependency fallback | WebRTC VAD (legacy) |
| Need turn-ending quality | Silero + LiveKit turn-detector layered |

经验法则：除非真的没有其他选择，否则永远不要上线纯能量 VAD。

## 常见陷阱

- **固定阈值。** 安静环境下有效，噪声环境下失败。要么在设备上校准，要么切换到 Silero。
- **Silence hangover 太短。** Agent 在句中打断用户。500-800 ms 是对话语音的最佳区间。
- **Hangover 太长。** 感觉迟钝。和目标用户做 A/B 测试。
- **没有 pre-roll buffer。** 用户音频的前 200-300 ms 丢失。始终保持滚动 pre-roll。
- **忽略语义端点检测。** "Hmm, let me think..." 包含长停顿。用户讨厌在思考中途被打断。使用 LiveKit 的 turn-detector 或类似方案。

## 交付

保存为 `outputs/skill-vad-tuner.md`。为工作负载选择 VAD 模型、阈值、hangover、pre-roll 和 turn-detection 策略。

## 练习

1. **Easy。** 运行 `code/main.py`。它模拟一段语音 + 静音 + 语音 + 咳嗽的序列，并测试三层 VAD。
2. **Medium。** 安装 `silero-vad`，处理一段 5 分钟的录音，调整阈值以最小化首词截断和误触发。报告 precision/recall。
3. **Hard。** 构建一个迷你 turn-detector：Silero VAD + 一个 3 层 MLP，输入最近 10 个词的 embedding（使用 sentence-transformers）。在手工标注的 turn-end 数据集上训练。比 Silero-only 提升 10% F1。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| VAD | Voice detector | Binary per-frame: is this speech? |
| Turn detection | End-pointing | VAD + silence-hangover + semantic endpoint. |
| Silence hangover | Wait-after-speech | Time to wait before declaring turn end; 500-800 ms. |
| Pre-roll | Pre-speech buffer | Keep 300-500 ms audio before VAD fires. |
| Flush trick | Kyutai hack | VAD → flush-STT → 125 ms instead of 500 ms delay. |
| Semantic endpoint | "Did they mean to stop?" | ML classifier that looks at words, not just silence. |
| TPR @ FPR 5% | ROC point | Standard VAD benchmark; 87.7% for Silero, 50% WebRTC. |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — the reference open VAD.
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — commercial accuracy leader.
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — the sub-200 ms engineering trick.
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — semantic endpointing in production.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — the legacy baseline.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — diarization-grade segmentation.
