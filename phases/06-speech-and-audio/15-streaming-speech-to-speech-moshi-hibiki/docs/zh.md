# Streaming Speech-to-Speech — Moshi、Hibiki 与 Full-Duplex 对话

> 2024-2026 重新定义了语音 AI。Moshi 用单一模型同时听和说，延迟 200 ms。Hibiki 逐块做 speech-to-speech 翻译。两者都抛弃了 ASR → LLM → TTS 管线，转向基于 Mimi codec token 的统一 full-duplex 架构。这是新的参考设计。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 13 (Neural Audio Codecs), Phase 6 · 11 (Real-Time Audio), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## 问题

用 Lesson 11 + 12 构建的每个语音 agent 都有一个根本性的延迟下限，约 300-500 ms：VAD 触发、STT 处理、LLM 推理、TTS 生成。每个阶段都有自己的最小延迟。你可以调优和并行化，但管线形状限制了你。

Moshi（Kyutai, 2024-2026）问了一个不同的问题：如果没有管线呢？如果一个模型直接、持续地接收音频并输出音频，文本只是一个中间的"内心独白"而不是必经阶段呢？

答案是 **full-duplex speech-to-speech**。理论延迟 160 ms（80 ms Mimi 帧 + 80 ms 声学延迟）。实际延迟在单张 L4 GPU 上 200 ms。这是最佳管线式语音 agent 的一半。

## 概念

![Moshi architecture: two parallel Mimi streams + inner-monologue text](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两个 Mimi codec 流，都是 12.5 Hz × 8 codebooks：

- Stream 1：用户音频（Mimi 编码，持续到达）
- Stream 2：Moshi 自己的音频（由 Moshi 生成）

**Transformer。** 一个 7B 参数的 Temporal Transformer 处理两个流和一个文本"内心独白"流。在每个 80 ms 步骤中，它：

1. 消费最新的用户 Mimi tokens（8 codebooks）。
2. 消费最近的 Moshi Mimi tokens（8 codebooks，已生成的）。
3. 生成下一个 Moshi 文本 token（内心独白）。
4. 生成下一个 Moshi Mimi tokens（8 codebooks，通过一个小的 Depth Transformer）。

三个流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 可以在说话时听到用户；可以在用户打断时中断自己；可以在不打断主要话语的情况下做 back-channel（"mhm"）。

**Depth transformer。** 在一帧内，8 个 codebook 不是并行预测的——它们有 codebook 间的依赖关系。一个小的 2 层"depth transformer"在 80 ms 内顺序预测它们。这是 AR codec LM 的标准分解方式（VALL-E、VibeVoice 也使用）。

### 为什么内心独白文本有帮助

没有显式文本，模型必须在声学流中隐式建模语言。Moshi 的洞察：强制它在音频旁边输出文本 token。文本流本质上是 Moshi 所说内容的转录。这改善了语义连贯性，使得更容易替换语言模型头，并且免费给你转录文本。

### Hibiki：streaming speech-to-speech 翻译

相同的架构，在翻译对上训练。源语言音频输入，目标语言音频持续输出。Hibiki-Zero（2026 年 2 月）消除了对词级对齐训练数据的需求——使用句子级数据 + GRPO 强化学习来优化延迟。

初始支持四个语言对；可以用约 1000 小时数据适配新语言。

### 更广泛的 Kyutai 技术栈（2026）

- **Moshi** — full-duplex 对话（法语优先，英语良好支持）
- **Hibiki / Hibiki-Zero** — 同步语音翻译
- **Kyutai STT** — streaming ASR（500 ms 或 2.5 s 前瞻）
- **Kyutai Pocket TTS** — 1 亿参数 TTS，可在 CPU 上运行（2026 年 1 月）
- **Unmute** — 将这些组合在公共服务器上的完整管线

L40S GPU 上的吞吐量：64 个并发会话，3 倍实时。

### Sesame CSM — 近亲

Sesame CSM（2025）使用类似的思路——Llama-3 backbone + Mimi codec head。但 CSM 是单向的（接收上下文 + 文本，产出语音）而非 full-duplex。它是市场上最好的"voice presence" TTS；与 Moshi 的 full-duplex 能力不完全相同。

### 2026 性能数据

| Model | Latency | Use case | License |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | full-duplex English / French dialogue | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | French ↔ English streaming translation | CC-BY 4.0 |
| Hibiki-Zero | same | 5 language-pairs, no aligned data | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | context-conditioned TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | closed, OpenAI API | commercial |
| Gemini 2.5 Live | ~350 ms | closed, Google API | commercial |

## 动手构建

### Step 1：接口

Moshi 暴露一个 WebSocket 服务器，接收 80 ms 的 Mimi 编码音频块并返回 80 ms 的 Mimi 编码音频块。双向。持续不断。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Step 2：full-duplex 循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准传输方式。

### Step 3：训练目标（概念性）

对于每个 80 ms 帧 `t`：

- Input: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- Predict: `moshi_text[t]`, then `moshi_mimi[t, codebook_0..7]`

文本在音频之前预测（内心独白）；音频在 depth transformer 内按 codebook 顺序预测。

### Step 4：Moshi 的优势和劣势

Moshi 的优势：

- 在便宜硬件上 sub-250 ms 端到端。
- 自然的 back-channel 和中断。
- 没有管线胶水代码。

Moshi 的劣势：

- 工具调用（没有为此训练；你需要单独的 LLM 路径）。
- 长推理（Moshi 是一个约 8B 的对话模型，不是 Claude/GPT-4）。
- 小众话题的事实准确性。
- 大多数生产企业用例（2026 年仍然使用管线）。

## 使用指南

| Situation | Pick |
|-----------|------|
| Lowest-latency voice companion | Moshi |
| Live translation call | Hibiki |
| Voice demo / research | Moshi, CSM |
| Enterprise agent with tools | Pipeline (Lesson 12), not Moshi |
| Custom-voice TTS in context | Sesame CSM |
| Speech-to-speech, any languages | GPT-4o Realtime or Gemini 2.5 Live (commercial) |

## 常见陷阱

- **有限的工具调用。** Moshi 是对话模型，不是 agent 框架。工具需要结合管线。
- **特定声音条件化。** Moshi 使用单一训练的 persona；克隆是单独的训练过程。
- **语言覆盖。** 法语 + 英语表现优秀；其他语言有限。Hibiki-Zero 有帮助，但你仍然需要训练数据。
- **资源成本。** 一个完整的 Moshi 会话占用一个 GPU 槽位；不是便宜的共享租户部署模式。

## 交付

保存为 `outputs/skill-duplex-pipeline.md`。为语音 agent 工作负载选择管线 vs full-duplex 架构，并给出理由。

## 练习

1. **Easy。** 运行 `code/main.py`。它象征性地模拟双流 + 内心独白架构。
2. **Medium。** 从 HuggingFace 拉取 Moshi，运行服务器，测试一次对话。测量从用户语音结束到 Moshi 开始回应的 wall-clock 延迟。
3. **Hard。** 拿你 Lesson 12 的管线 agent，与 Moshi 在 20 个匹配测试话语上比较 P50 延迟。写出管线在什么情况下架构上仍然胜出。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Full-duplex | Hear-and-speak at once | Two audio streams active simultaneously on the same model. |
| Inner monologue | Model's text stream | Moshi emits text tokens alongside its audio output. |
| Depth transformer | Inter-codebook predictor | Small transformer that predicts 8 codebooks within one 80 ms frame. |
| Mimi | Kyutai's codec | 12.5 Hz × 8 codebooks; semantic+acoustic; powers Moshi. |
| Streaming S2S | Audio → audio live | Chunk-by-chunk translation/dialogue, no pipeline stages. |
| Back-channeling | "Mhm" reactions | Moshi can emit small acknowledgments without breaking its turn. |

## 延伸阅读

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — the paper.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — streaming translation without aligned data.
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM spec.
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — install + server.
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — closed commercial peer.
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — the STT/TTS framework under the hood.
