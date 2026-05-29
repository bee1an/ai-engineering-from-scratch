# 实时音频处理

> 批处理管线处理一个文件。实时管线在下一个 20 毫秒到来之前处理完当前的 20 毫秒。每一个对话式 AI、广播工作室和电话机器人的生死都取决于这个延迟预算。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## 问题

你想要一个感觉"活着"的语音助手。人类对话轮次切换的延迟约 230 ms（从沉默到回应）。超过 500 ms 感觉像机器人；超过 1500 ms 感觉坏了。2026 年完整的 **听 → 理解 → 回应 → 说** 循环的预算是：

| Stage | Budget |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi（Kyutai, 2024）实现了 200 ms full-duplex。GPT-4o-realtime（2024）约 320 ms。2022 年的级联管线延迟高达 2500 ms。10 倍的改进来自三项技术：(1) 全链路 streaming，(2) 基于部分结果的异步流水线，(3) 可中断生成。

## 概念

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**帧 / chunk / 窗口。** 实时音频以固定大小的块流动。常见选择：20 ms（16 kHz 下 320 个采样点）。下游所有环节必须跟上这个节奏。

**Ring buffer。** 固定大小的环形缓冲区。生产者线程写入新帧，消费者线程读取。避免热路径上的内存分配。大小 ≈ 最大延迟 × 采样率；2 秒 16 kHz ring = 32,000 个采样点。

**VAD (Voice Activity Detection)。** 在没人说话时门控下游工作。Silero VAD 4.0（2024）在 CPU 上每 30 ms 帧 <1 ms。`webrtcvad` 是更老的替代方案。

**Streaming ASR。** 音频到达时就输出部分转录的模型。Parakeet-CTC-0.6B streaming 模式（NeMo, 2024）在 320 ms 延迟下达到 2-5% WER。Whisper-Streaming（Macháček et al., 2023）将 Whisper 分块处理，实现约 2 s 延迟的近实时。

**中断。** 当用户在助手说话时开口，你必须 (a) 检测 barge-in，(b) 停止 TTS，(c) 丢弃剩余的 LLM 输出。全部在 100 ms 内完成，否则用户会觉得助手"聋了"。

**WebRTC Opus 传输。** 20 ms 帧，48 kHz，自适应码率 8-128 kbps。浏览器和移动端的标准。LiveKit、Daily.co、Pion 是 2026 年构建语音应用的主流技术栈。

**Jitter buffer。** 网络包乱序 / 延迟到达。Jitter buffer 重排序并平滑；太小 → 可听到的断裂，太大 → 延迟。典型值 60-80 ms。

### 常见坑

- **线程竞争。** Python 的 GIL + 重模型会饿死音频线程。使用 C 回调音频库（sounddevice, PortAudio），让 Python 远离热路径。
- **采样率转换延迟。** 管线内重采样增加 5-20 ms。要么提前重采样，要么使用零延迟重采样器（PolyPhase, `soxr_hq`）。
- **TTS 预热。** 即使是快速的 TTS 如 Kokoro，首次请求也有 100-200 ms 的预热。缓存模型 + 在第一个真实轮次前用 dummy 请求预热。
- **回声消除。** 没有 AEC，TTS 输出会重新进入麦克风，触发 ASR 识别机器人自己的声音。WebRTC AEC3 是开源默认方案。

## 动手构建

### Step 1：ring buffer

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。16 kHz 下 32,000 个采样点 = 2 s。

### Step 2：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产环境替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Step 3：streaming ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Step 4：中断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

关键在于异步 I/O 和可取消的 TTS streaming。WebRTC peerconnection.stop() 作用于 audio track 是标准做法。

## 使用指南

2026 技术栈：

| Layer | Pick |
|-------|------|
| Transport | LiveKit (WebRTC) or Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B or Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro or ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API or Moshi |

## 常见陷阱

- **缓冲 500 ms 以求安全。** 缓冲区*就是*你的延迟下限。缩小它。
- **不固定线程。** 音频回调在优先级低于 UI 的线程上 = 负载下出现卡顿。
- **TTS chunk 太小。** 小于 200 ms 的 chunk 会让 vocoder 伪影可听。320 ms chunk 是最佳平衡点。
- **没有 jitter buffer。** 真实网络有抖动；不做平滑就会有爆音。
- **一次性错误处理。** 音频管线必须防崩溃。一个异常就会杀死整个会话。

## 交付

保存为 `outputs/skill-realtime-designer.md`。设计一个实时音频管线，为每个阶段给出具体的延迟预算。

## 练习

1. **Easy。** 运行 `code/main.py`。模拟一个 ring buffer + energy VAD；为一个假的 10 秒流打印各阶段延迟。
2. **Medium。** 使用 `sounddevice`，构建一个直通循环，以 20 ms 帧处理你的麦克风输入，并在每帧打印 VAD 状态。
3. **Hard。** 用 `aiortc` 构建一个 full duplex 回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。用 1 kHz 脉冲测量端到端延迟。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Ring buffer | The circular queue | Fixed-size, lock-free (or SPSC-locked) FIFO for audio frames. |
| VAD | Silence gate | Model or heuristic marking speech vs non-speech. |
| Streaming ASR | Real-time STT | Emits partial text as audio arrives; bounded lookahead. |
| Jitter buffer | Network smoother | Queue reordering out-of-order packets; 60–80 ms typical. |
| AEC | Echo cancellation | Subtracts speaker-to-mic feedback path. |
| Barge-in | User interrupt | System detects user speech mid-TTS; must cancel playback. |
| Full duplex | Simultaneous both ways | User and bot can talk at the same time; Moshi is full duplex. |

## 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — chunked near-streaming Whisper.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — full-duplex 200 ms latency.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — production audio agent orchestration.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — sub-1 ms VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — echo cancellation under open source.
