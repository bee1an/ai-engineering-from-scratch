# 构建语音助手管线 — Phase 6 综合项目

> 把 Lesson 01-11 的所有内容缝合在一起。构建一个能听、能思考、能回话的语音助手。在 2026 年这是一个已解决的工程问题，不是研究问题——但集成细节决定了它能否上线。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 minutes

## 问题

构建一个端到端助手：

1. 采集麦克风输入（16 kHz mono）。
2. 检测用户语音的开始/结束。
3. 流式转录。
4. 将转录文本传给能调用工具（定时器、天气、日历）的 LLM。
5. 将 LLM 文本流式传给 TTS。
6. 将音频播放给用户。
7. 如果用户在回应过程中打断，则停止。

延迟目标：用户说完话后 800 ms 内输出第一个 TTS 音频字节（笔记本 CPU 上）。质量目标：不漏词、静音时不产生幻觉字幕、不泄露 voice cloning、不被 prompt injection 攻破。

## 概念

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### 七个组件

1. **音频采集。** Mic → 16 kHz mono → 20 ms chunks。Python 中通常用 `sounddevice`，生产环境用原生 AudioUnit/ALSA/WASAPI。
2. **VAD（Lesson 11）。** Silero VAD @ threshold 0.5，最小语音 250 ms，静音挂起 500 ms。发出"开始"和"结束"信号。
3. **Streaming STT（Lesson 4-5）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。部分 + 最终转录。
4. **带工具调用的 LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。工具用 JSON schema 定义。流式输出 token。
5. **Streaming TTS（Lesson 7）。** Kokoro-82M（最快的开源方案）或 Cartesia Sonic（商业）。在 LLM 输出 20 个 token 后启动 TTS。
6. **播放。** 扬声器输出；低带宽网络用 opus 编码。
7. **中断处理器。** 如果 VAD 在 TTS 播放期间触发，停止播放、取消 LLM、重启 STT。

### 你会遇到的三种失败模式

1. **首词截断。** VAD 启动晚了一拍。用户的"hey"丢失了。把起始阈值设为 0.3 而不是 0.5。
2. **回应中途打断混乱。** 用户打断后 LLM 继续生成；助手和用户抢话。把 VAD → cancel-LLM 连起来。
3. **静音幻觉。** Whisper 在静音预热帧上输出"Thanks for watching"。始终用 VAD 做门控。

### 2026 生产参考技术栈

| Stack | Latency | License | Notes |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | commercial API | Industry default 2026 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | mostly open | DIY-friendly |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | Single-model; different architecture, lesson 15 |
| Vapi / Retell (managed) | 300-500 ms | commercial | Fastest to launch; limited customization |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | open | Privacy / edge |

## 动手构建

### Step 1：麦克风采集与分块（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Step 2：VAD 门控的轮次捕获

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Step 3：streaming STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Step 4：LLM 循环中的工具调用

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Step 5：中断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使用指南

参见 `code/main.py`，它用 stub 模型连接了所有七个组件的可运行模拟，即使没有硬件也能看到管线的形状。要做真实实现，把 stub 替换为：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（`gpt-4o`）或 `anthropic`
- `kokoro` 或 `cartesia`
- `sounddevice` 做 I/O

## 常见陷阱

- **永久记录 PII。** 完整轮次音频在大多数司法管辖区属于 PII。30 天保留期，静态加密。
- **不支持 barge-in。** 用户会打断。你的助手必须停止说话。
- **TTS 阻塞。** 同步 TTS 阻塞事件循环。使用 async 或独立线程。
- **工具调用无错误处理。** 工具会失败。LLM 必须收到错误 + 重试一次，然后优雅降级。
- **过度激进的幻觉过滤。** 过度过滤，助手反复说"I can't help with that"。过滤不足，它什么都说。在留出的测试集上校准。
- **没有唤醒词选项。** 始终监听是隐私隐患。加一个唤醒词门控（Porcupine 或 openWakeWord）。

## 交付

保存为 `outputs/skill-voice-assistant-architect.md`。给定预算 + 规模 + 语言 + 合规约束，产出完整的技术栈规格。

## 练习

1. **Easy。** 运行 `code/main.py`。它用 stub 模块模拟一个完整轮次的端到端流程，并打印各阶段延迟。
2. **Medium。** 用真实的 Whisper 模型替换 STT stub，处理一个预录的 `.wav`。测量 WER 和端到端延迟。
3. **Hard。** 添加工具调用：实现 `get_weather`（任意 API）和 `set_timer`。让 LLM 通过工具路由，验证当用户说"set a 5 minute timer"时正确的函数被触发，且语音回复确认了操作。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Turn | A user + assistant round-trip | One VAD-bounded user speech + one LLM-TTS response. |
| Barge-in | Interruption | User speaks while assistant talks; assistant stops. |
| Wake word | "Hey assistant" | Short keyword detector; Porcupine, Snowboy, openWakeWord. |
| End-pointing | Turn ending | VAD + min-silence decision that user has finished. |
| Pre-roll | Pre-speech buffer | Keep 200-400 ms of audio before VAD fires to avoid first-word clip. |
| Tool call | Function invocation | LLM emits JSON; runtime dispatches; result feeds back in-loop. |

## 延伸阅读

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — production-grade reference.
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — DIY-friendly framework.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — the managed voice-native path.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — full-duplex reference (Lesson 15).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — wake-word gating.
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — LLM function calling.
