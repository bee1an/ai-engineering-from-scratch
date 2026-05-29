# 毕业项目 03 — 实时语音助手（ASR → LLM → TTS）

> 一个体验良好的语音智能体端到端延迟低于 800ms，知道你何时停止说话，处理打断，并能在不卡顿的情况下调用工具。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都达到了这个标准。它们用相同的形态实现：流式 ASR、轮次检测器、流式 LLM 和流式 TTS，全部通过 WebRTC 串联，每一跳都有激进的延迟预算。构建一个，衡量 WER、MOS 和误截断率，并在丢包条件下运行。

**类型：** 毕业项目
**语言：** Python（智能体 + 管道），TypeScript（Web 客户端）
**前置要求：** Phase 6（语音与音频）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 17（基础设施）
**涉及阶段：** P6 · P7 · P11 · P13 · P14 · P17
**时间：** 30 小时

## 问题

语音是 2025-2026 年发展最快的 AI 用户体验类别。技术天花板每个季度都在降低。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都将亚 800ms 首音频输出带入了可及范围。标准不仅仅是延迟。而是交互感受：不打断用户、不被打断、从句中打断中恢复、对话中调用工具而不卡住音频、在抖动的移动网络上存活。

你无法通过拼接三个 REST 调用来实现。架构必须是端到端的流水线流式处理。构建它，失败模式就会显现：一个为电话音频调优的 VAD 在背景电视声上误触发、一个等待永远不来的标点的轮次检测器、一个在发出前缓冲 400ms 的 TTS。这个毕业项目就是在负载下逐一修复这些问题，并发布延迟与质量报告。

## 概念

管道有五个流式阶段：**音频输入**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转录）、**轮次检测**（VAD 加一个小型轮次检测模型，读取部分转录中的完成线索）、**LLM**（轮次判定完成后立即流式输出 token）、**TTS**（在第一个 LLM token 后约 200ms 内流式输出音频）。

三个横切关注点。**打断（Barge-in）**：当用户在智能体说话时开始说话，TTS 取消，ASR 立即接管。**工具使用**：对话中的函数调用（天气、日历）必须在侧通道运行而不卡住音频；如果延迟超过 300ms，智能体预填一个确认 token（"稍等一下..."）。**背压**：在丢包情况下，部分转录被暂存，VAD 提高语音门限阈值，智能体避免在未确认的消息上说话。

衡量标准是量化的。15 dB SNR 下 Hamming VAD 基准上 WER 低于 8%。100 次录音通话中首音频输出 p50 低于 800ms。误截断率低于 3%。TTS MOS 高于 4.2。单台 g5.xlarge 上 50 路并发通话。这些数字就是交付物。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- 传输：LiveKit Agents 1.0（WebRTC）加 Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- ASR：Deepgram Nova-3（流式，亚 300ms 首个部分结果）或自托管的 faster-whisper Whisper-v3-turbo
- VAD：Silero VAD v5 加 LiveKit 轮次检测器（读取部分转录的小型 transformer）
- LLM：OpenAI GPT-4o-realtime 用于紧密集成，Gemini 2.5 Flash Live，或级联 Claude Haiku 4.5（流式补全，独立音频路径）
- TTS：Cartesia Sonic-2（最低首字节延迟），ElevenLabs Flash v3，或开源 Orpheus 用于自托管
- 工具：FastMCP 侧通道用于天气/日历/预约；工具耗时超过 300ms 时智能体预发填充语
- 可观测性：OpenTelemetry 语音 span，Langfuse 语音 trace 带音频回放
- 部署：单台 g5.xlarge（24GB VRAM）用于自托管 Whisper + Orpheus；托管 API 用于最低延迟

## 构建步骤

1. **WebRTC 会话。** 搭建一个 LiveKit room 和一个流式传输麦克风音频的 Web 客户端。在服务端，附加一个加入 room 的 agent worker。

2. **ASR 流式处理。** 将 20ms PCM 帧送入 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分和最终转录。记录每个部分结果的延迟。

3. **VAD 和轮次检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件时，对最新的部分转录触发 LiveKit 轮次检测器。只有当 VAD 检测到 500ms 静音且轮次检测器完成度评分 > 0.6 时，才提交"轮次完成"。

4. **LLM 流。** 轮次完成时，用运行中的对话加最终转录启动 LLM 调用。流式输出 token。在第一个 token 时交接给 TTS。

5. **TTS 流。** Cartesia Sonic-2 流式返回音频块。第一个块必须在第一个 LLM token 后 200ms 内离开服务器。将块发送到 LiveKit room；客户端通过 WebRTC jitter buffer 播放。

6. **打断处理。** 当 VAD 在 TTS 播放时检测到新的用户语音，立即取消 TTS 流，丢弃剩余 LLM 输出，重新启动 ASR。发布一个 `tts_canceled` span。

7. **工具侧通道。** 将天气和日历注册为函数调用工具。调用时并发触发；如果 300ms 内未返回，让 LLM 发出"稍等，让我查一下"作为填充语；工具返回后继续。

8. **评估框架。** 录制 100 次通话。计算 WER（对照保留转录）、误截断率（用户说话中途 TTS 被取消）、首音频输出 p50、TTS MOS（人工或 NISQA），以及抖动丢包测试（丢弃 3% 的包）。

9. **负载测试。** 在单台 g5.xlarge 上用合成呼叫者驱动 50 路并发通话。衡量持续的首音频输出 p95。

## 使用示例

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付标准

`outputs/skill-voice-agent.md` 是交付物。给定一个领域（客服、排程或自助终端），它搭建一个 LiveKit 智能体，ASR/VAD/LLM/TTS 管道调优到衡量标准。评分标准：

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 端到端延迟 | 100 次录音通话中首音频输出 p50 低于 800ms |
| 20 | 轮次交接质量 | Hamming VAD 基准上误截断率低于 3% |
| 20 | 工具使用正确性 | 对话中工具调用返回正确数据且不卡住音频 |
| 20 | 丢包下的可靠性 | 注入 3% 丢包时的 WER 和轮次交接稳定性 |
| 15 | 评估框架完整性 | 可复现的测量结果，配置公开 |
| **100** | | |

## 练习

1. 将 Deepgram Nova-3 换成 g5.xlarge 上的 faster-whisper v3 turbo。衡量延迟和 WER 差距。识别 CPU vs GPU 决策在哪里重要。

2. 添加打断仲裁策略：用户在工具调用期间打断时智能体怎么办？对比三种策略（硬取消、完成工具后停止、排队下一轮）。

3. 运行对抗性轮次检测器测试：让用户在句中长时间停顿。调优 VAD 静音阈值和轮次检测器评分阈值，在不超过 900ms 的前提下实现最低误截断。

4. 通过 Twilio 在 PSTN 上部署同一智能体。对比 PSTN 首音频输出与 WebRTC。解释 jitter buffer 和编解码器差异。

5. 为非英语语言（日语、西班牙语）添加语音活动检测。衡量 Silero VAD v5 的误触发率与语言特定微调的对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 轮次检测 | "话语结束" | 分类器，给定 VAD 静音和部分转录，判断用户是否说完了 |
| Barge-in | "打断处理" | 当 VAD 检测到新的用户语音时取消正在播放的 TTS |
| 首音频输出 | "延迟" | 从用户停止说话到第一个音频包离开服务器的时间 |
| VAD | "语音门" | 将音频帧分类为语音 vs 静音的模型；Silero VAD v5 是 2026 默认选择 |
| Jitter buffer | "音频平滑" | 客户端缓冲区，短暂持有数据包以吸收网络抖动 |
| 填充语 | "确认 token" | 工具较慢时智能体发出的短语以避免沉默 |
| MOS | "平均意见分" | 感知语音质量评分；NISQA 是自动化代理 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考 WebRTC 智能体框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 备选 Python 优先流式智能体框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型参考
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI architecture](https://docs.retellai.com) — 生产语音智能体架构
- [Vapi.ai production stack](https://docs.vapi.ai) — 备选生产参考
