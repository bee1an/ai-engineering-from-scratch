# 语音 Agent：Pipecat 与 LiveKit

> 语音 agent 在 2026 年已是一线生产级品类。Pipecat 提供基于 Python 的帧流水线（VAD → STT → LLM → TTS → transport）。LiveKit Agents 通过 WebRTC 将 AI 模型桥接到用户。高端方案的端到端延迟目标落在 450–600ms。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~60 minutes

## 学习目标

- 描述 Pipecat 的帧流水线：DOWNSTREAM（source→sink）和 UPSTREAM（control）。
- 列举标准语音流水线阶段及 Pipecat 支持的 transport。
- 解释 LiveKit Agents 的两种语音 agent 类（MultimodalAgent、VoicePipelineAgent）及各自适用场景。
- 总结 2026 年生产延迟预期及其对架构选择的影响。

## 问题

语音 agent 不是"文本循环加个 TTS"。延迟预算极其苛刻（~600ms），部分音频是常态，轮次检测本身就是一个模型，transport 从电话 SIP 到 WebRTC 都有。要么你构建帧流水线（Pipecat），要么依赖平台（LiveKit）。

## 概念

### Pipecat (pipecat-ai/pipecat)

- Python 帧流水线框架。
- `Frame` → `FrameProcessor` 链。
- 两个流向：
  - **DOWNSTREAM** — source → sink（音频输入，TTS 输出）。
  - **UPSTREAM** — 反馈与控制（取消、指标、barge-in）。
- `PipelineTask` 通过事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）和 observer 管理生命周期，支持 metrics/tracing/RTVI。

典型流水线：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Transports: Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp。

Pipecat Flows 增加了结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents (livekit/agents)

- 通过 WebRTC 将 AI 模型桥接到用户。
- 核心概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两种语音 agent 类：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或等效方案直接处理音频。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；提供文本级控制。
- 基于 transformer 模型的语义轮次检测。
- 原生 MCP 集成。
- 通过 SIP 支持电话。
- 通过 LiveKit Inference 免 API key 使用 50+ 模型；通过插件支持 200+ 更多模型。

### 商业平台

Vapi（优化高端方案 ~450–600ms）和 Retell（180 次测试通话端到端 ~600ms）构建在这些之上。当你想要托管语音方案而不需要 WebRTC 团队时，选择平台。

### 这个模式容易出错的地方

- **没有 barge-in 处理。** 用户打断了，agent 还在说。Pipecat 中需要 UPSTREAM cancel frame，LiveKit 中有等效机制。
- **忽略 STT 置信度。** 低置信度转录被当作真理喂给 LLM。应该基于置信度做门控或请求确认。
- **TTS 句中截断。** 流水线取消时 TTS 需要知道或切断音频。
- **忽略延迟预算。** 每个组件增加 50–200ms。上线前先把链路延迟加起来。

### 2026 年典型延迟

- VAD: 20–60ms
- STT partial: 100–250ms
- LLM first token: 150–400ms
- TTS first audio: 100–200ms
- Transport RTT: 30–80ms

端到端 450–600ms 是高端水平。800–1200ms 是常见水平。超过 1500ms 体验就崩了。

## Build It

`code/main.py` 是一个基于帧的玩具流水线：

- `Frame` 类型（audio、transcript、text、tts_audio、control）。
- `Processor` 接口，带 `process(frame)`。
- 五阶段流水线（VAD → STT → LLM → TTS → transport），用脚本化 processor 实现。
- 一个 UPSTREAM cancel frame 演示 barge-in。

运行：

```
python3 code/main.py
```

trace 展示正常流程和一次 barge-in cancel 中断 TTS 的过程。

## Use It

- **Pipecat** 用于完全控制 — 自定义 processor、Python 优先、可插拔 provider。
- **LiveKit Agents** 用于 WebRTC 优先的部署和电话场景。
- **Vapi / Retell** 用于不需要 WebRTC 团队的托管语音 agent。
- **OpenAI Realtime / Gemini Live** 用于直接音频输入/输出（MultimodalAgent）。

## Ship It

`outputs/skill-voice-pipeline.md` 搭建一个 Pipecat 形态的语音流水线，包含 VAD + STT + LLM + TTS + transport 以及 barge-in 处理。

## 练习

1. 给你的玩具流水线加一个 metrics observer：统计每阶段每秒帧数。延迟在哪里累积？
2. 实现置信度门控 STT：低于阈值时请求"能再说一遍吗？"
3. 加入语义轮次检测：简单规则 — 如果转录以"？"结尾，视为轮次结束。
4. 阅读 Pipecat 的 transport 文档。将 stdlib transport 替换为 SmallWebRTCTransport 配置（stub）。
5. 对同一查询测量 OpenAI Realtime vs STT+LLM+TTS 级联。文本级控制带来多少延迟代价？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Frame | "Event" | 流水线中的类型化数据单元（audio、transcript、text、control） |
| Processor | "Pipeline stage" | 带 process(frame) 的处理器 |
| DOWNSTREAM | "Forward flow" | Source 到 sink：音频输入，语音输出 |
| UPSTREAM | "Feedback flow" | 控制：cancel、metrics、barge-in |
| VAD | "Voice activity detection" | 检测用户是否在说话 |
| Semantic turn detection | "Smart end-of-turn" | 基于模型判断用户是否说完 |
| MultimodalAgent | "Direct audio agent" | 音频输入，音频输出；中间没有文本 |
| VoicePipelineAgent | "Cascade agent" | STT + LLM + TTS；文本级控制 |

## 延伸阅读

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — frame-based pipeline, processors, transports
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC + voice primitives
- [Vapi](https://vapi.ai/) — managed voice platform
- [Retell AI](https://www.retellai.com/) — managed voice, latency-benchmarked
