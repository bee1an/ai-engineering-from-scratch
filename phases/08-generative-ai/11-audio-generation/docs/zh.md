# 音频生成

> 音频是 16-48 kHz 的 1-D 信号。五秒片段是 80-240k 个采样点。没有 transformer 能直接对这个序列做注意力。2026 年每个生产音频模型的解决方案都相同：神经编解码器（Encodec、SoundStream、DAC）将音频压缩为 50-75 Hz 的离散 token，然后 transformer 或扩散模型生成 token。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## 问题

三个音频生成任务：

1. **文本到语音。** 给定文本，产生语音。干净语音是窄带的且有强音素结构——transformer-over-tokens 解决得很好。VALL-E (Microsoft)、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定 prompt（文本、旋律、和弦进行、流派），产生音乐。分布宽得多。MusicGen (Meta)、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效 / 声音设计。** 给定 prompt，产生环境音或拟音。AudioGen、AudioLDM 2、Stable Audio Open。

三者都运行在相同基底上：神经音频编解码器 + token-AR 或扩散生成器。

## 概念

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec (Meta, 2022)、SoundStream (Google, 2021)、Descript Audio Codec (DAC, 2023)。卷积编码器将波形压缩为逐时间步向量；残差向量量化（RVQ）将每个向量转换为 K 个码本索引的级联。解码器反转它。24 kHz 音频在 2 kbps 使用 8 个 RVQ 码本在 75 Hz = 600 tokens/sec。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 两种生成范式

**Token-autoregressive。** 将 RVQ token 展平为序列，运行 decoder-only transformer。MusicGen 使用"delayed parallel"以逐流偏移并行发射 K 个码本流。VALL-E 从文本 prompt + 3 秒语音样本生成语音 token。

**Latent diffusion。** 将 codec token 打包为连续 latent 或用分类扩散建模。Stable Audio 2.5 在连续音频 latent 上使用 flow matching。AudioLDM 2 使用 text-to-mel-to-audio 扩散。

2024-2026 趋势：flow matching 在音乐上胜出（更快推理，更干净样本），而 token-AR 仍主导语音因为它天然因果且流式传输好。

## 生产格局

| System | Task | Backbone | Latency |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms first token |
| OpenAI GPT-4o audio | Full-duplex speech | End-to-end multimodal AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | Non-streaming |
| Stable Audio 2.5 | Music / SFX | DiT + flow matching on audio latents | ~10s for 1-minute clip |
| Suno v4 | Full songs | Undisclosed; token-AR suspected | ~30s per song |
| Udio v1.5 | Full songs | Undisclosed | ~30s per song |
| MusicGen 3.3B | Music | Token-AR on Encodec 32kHz | Real-time |
| AudioCraft 2 | Music + SFX | Flow matching | ~5s for 5s clip |
| Riffusion v2 | Music | Spectrogram diffusion | ~10s |

## Build It

`code/main.py` 模拟核心思想：在从两种不同"风格"生成的合成"音频 token"序列上训练一个小型 next-token transformer（风格 A 交替高低 token，风格 B 单调递增）。条件化风格并采样。

### Step 1: synthetic audio tokens

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### Step 2: train a tiny token predictor

条件化风格的 bigram 式预测器。重点是模式：codec tokens → 交叉熵训练 → autoregressive 采样。

### Step 3: sample conditionally

给定风格 token 和起始 token，从预测分布中采样下一个 token。继续 20-40 个 token。

## Pitfalls

- **编解码器质量限制输出质量。** 如果编解码器不能忠实表示一个声音，再多的生成器质量也没用。DAC 是当前最佳开源。
- **RVQ 误差累积。** 每个 RVQ 层建模前一层的残差。第 1 层的错误会传播。对更高层用 temperature 0 采样有帮助。
- **音乐结构。** 30 秒 token 在 75 Hz 是 20k+ token。对 transformer 很难。MusicGen 使用滑动窗口 + prompt 续写；Stable Audio 使用更短片段 + 交叉淡入淡出。
- **边界伪影。** 生成片段之间的交叉淡入淡出需要仔细的 overlap-add。
- **干净数据需求。** 音乐生成器需要数万小时的授权音乐。Suno / Udio RIAA 诉讼（2024）将此浮出水面。
- **声音克隆伦理。** 3 秒样本加文本 prompt 就足以让 VALL-E / XTTS / ElevenLabs 克隆声音。每个生产模型都需要滥用检测 + 退出列表。

## Use It

| Task | 2026 stack |
|------|------------|
| 商业 TTS | ElevenLabs, OpenAI TTS, 或 Azure Neural |
| 声音克隆（已验证同意） | XTTS v2 (open) 或 ElevenLabs Pro |
| 背景音乐，快速 | Stable Audio 2.5 API, Suno, 或 Udio |
| 带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效 / 拟音 | AudioCraft 2, ElevenLabs SFX, 或 Stable Audio Open |
| 实时语音代理 | GPT-4o realtime 或 Gemini Live |
| 开源权重音乐研究 | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| 配音 / 翻译 | HeyGen, ElevenLabs Dubbing |

## Ship It

保存 `outputs/skill-audio-brief.md`。Skill 接收音频简报（任务、时长、风格、声音、许可），输出：模型 + 托管、prompt 格式（流派标签、风格描述符、结构标记）、codec + generator + vocoder 链、种子协议和评估计划（MOS / CLAP score / CER for TTS / 用户 A/B）。

## Exercises

1. **Easy.** 运行 `code/main.py` 并显式设置风格。验证生成序列匹配风格的模式。
2. **Medium.** 添加 delayed parallel 解码：模拟 2 个必须保持 1 步偏移的 token 流。训练联合预测器。
3. **Hard.** 用 HuggingFace transformers 本地运行 MusicGen-small。用三个不同 prompt 生成 10 秒片段；A/B 测试风格遵循度。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Codec | "神经压缩" | 音频的编码器/解码器；典型输出是 50-75 Hz token。 |
| RVQ | "Residual VQ" | K 个量化器的级联；每个建模前一个的残差。 |
| Token | "一个 codec 符号" | 码本中的离散索引；1024 或 2048 典型。 |
| Delayed parallel | "偏移码本" | 以交错偏移发射 K 个 token 流以减少序列长度。 |
| Flow matching | "2024 年音频的胜利" | 比扩散更直路径的替代；更快采样。 |
| Voice prompt | "3 秒样本" | 引导克隆声音的说话人嵌入或 token 前缀。 |
| Mel spectrogram | "可视化" | 对数幅度感知频谱图；许多 TTS 系统使用。 |
| Vocoder | "Mel 到波形" | 将 mel 频谱图转换回音频的神经组件。 |

## 生产笔记：音频是流式问题

音频是用户期望*边生成边到达*的唯一输出模态，而非一次性全部。在生产术语中这意味着 TPOT 很重要（Time Per Output Token），因为用户的听速是目标吞吐量——不是阅读速度。对于 16kHz 音频以约 75 tokens/second（Encodec）tokenize，服务器必须每用户生成 ≥75 tokens/sec 以保持播放流畅。

两个架构后果：

- **Flow-matching 音频模型不能简单流式传输。** Stable Audio 2.5 和 AudioCraft 2 在一次 pass 中渲染固定片段长度。要流式传输，你分块片段并重叠边界——想象滑动窗口扩散——相比 codec AR 模型增加 100-300ms 延迟开销。

如果产品是"实时语音聊天"或"实时音乐续写"，选 codec AR 路径。如果是"提交后渲染 30 秒片段"，flow-matching 在质量和总延迟上胜出。

## Further Reading

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — the codec standard.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — the first widely used neural audio codec.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2.
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 2025 text-to-music with flow matching.
