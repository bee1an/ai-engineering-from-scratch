# 音乐生成 — MusicGen、Stable Audio、Suno，以及版权地震

> 2026 年音乐生成：Suno v5 和 Udio v4 主导商业领域；MusicGen、Stable Audio Open 和 ACE-Step 领跑开源。技术问题基本解决。法律问题（Warner Music 5 亿美元和解、UMG 和解）在 2025-2026 年重塑了整个领域。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time:** ~75 minutes

## 问题

文本 → 一段 30 秒到 4 分钟的音乐片段，带歌词、人声和结构。三个子问题：

1. **纯器乐生成。** "lo-fi hip-hop drums with warm keys" 这样的文本 → 音频。MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（带人声 + 歌词）。** "Country song about rainy Texas nights" → 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **条件化 / 可控生成。** 延续现有片段、重新生成桥段、切换风格、分离音轨、或 inpainting。Udio 的 inpainting + 音轨分离是 2026 年需要追赶的功能。

## 概念

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### Neural-codec token 上的 Token LM

Meta 的 **MusicGen**（2023, MIT）及众多衍生：以文本/旋律 embedding 为条件，自回归预测 EnCodec token（32 kHz，4 个 codebook），用 EnCodec 解码。300M - 3.3B 参数。强基线；超过 30 秒后表现挣扎。

**ACE-Step**（开源，4B XL 于 2026 年 4 月发布）将此扩展到歌词条件的完整歌曲生成。开源社区最接近 Suno 的方案。

### Mel 或潜空间上的 Diffusion

**Stable Audio (2023)** 和 **Stable Audio Open (2024)**：压缩音频上的 latent diffusion。擅长循环、音效设计、氛围纹理。不擅长有结构的完整歌曲。

**AudioLDM / AudioLDM2**：通过 T2I 风格的 latent diffusion 做 text-to-audio，泛化到音乐、音效、语音。

### 混合方案（生产级）— Suno、Udio、Lyria

闭源权重。可能是 AR codec LM + diffusion-based vocoder，带专门的人声 / 鼓 / 旋律 head。Suno v5（2026）是 ELO 1293 的质量领先者。Udio v4 增加了 inpainting + 音轨分离（bass、drums、vocals 分别下载）。

### 评估

- **FAD (Frechet Audio Distance)。** 使用 VGGish 或 PANNs 特征计算生成与真实音频分布之间的 embedding 级距离。越低越好。MusicGen small：MusicCaps 上 4.5 FAD；SOTA 约 3.0。
- **音乐性（主观）。** 人类偏好。Suno v5 ELO 1293 领先。
- **文本-音频对齐。** prompt 和输出之间的 CLAP 分数。
- **音乐性瑕疵。** 节拍不准的过渡、人声短语漂移、超过 30 秒后结构丧失。

## 2026 年模型图谱

| Model | Params | Length | Vocals | License |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | no | MIT |
| Stable Audio Open | 1.2B | 47 s | no | Stability non-commercial |
| ACE-Step XL (Apr 2026) | 4B | > 2 min | yes | Apache-2.0 |
| YuE | 7B | > 2 min | yes, multilingual | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | yes, ELO 1293 | commercial |
| Udio v4 (closed) | ? | 4 min | yes + stems | commercial |
| Google Lyria 3 (closed) | ? | real-time | yes | commercial |
| MiniMax Music 2.5 | ? | 4 min | yes | commercial API |

## 法律格局（2025-2026）

- **Warner Music vs Suno 和解。** 5 亿美元。WMG 现在对 Suno 上的 AI 肖像权、音乐版权和用户生成曲目有监督权。Udio 有类似的 UMG 和解。
- **EU AI Act** + **California SB 942**：AI 生成的音乐必须披露。
- **Riffusion / MusicGen** 在 MIT 下没有合规负担，但也没有商业人声。

安全可发布的模式：

1. 只生成器乐（MusicGen、Stable Audio Open、MIT/CC0 输出）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music）并获取每次生成的许可。
3. 在自有或授权目录上训练（大多数企业最终走这条路）。
4. 为生成内容打水印 + 元数据标签。

## 动手构建

### Step 1：用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三个尺寸：`small`（300M，快）、`medium`（1.5B）、`large`（3.3B）。Small 足以验证"这个想法是否成立"。

### Step 2：旋律条件化

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 接收 chromagram 并保留旋律，同时替换音色。适用于"把这段旋律变成弦乐四重奏"。

### Step 3：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish-embedding 距离。适用于风格级别的回归测试；不能替代人类听众。

### Step 4：加入 LLM-音乐工作流

结合第 7-8 课的思路：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 实际使用

| Goal | Stack |
|------|-------|
| 器乐音效设计 | Stable Audio Open |
| 游戏 / 自适应音乐 | Google Lyria RealTime (closed) |
| 带人声的完整歌曲（商业） | Suno v5 或 Udio v4 + 明确许可 |
| 带人声的完整歌曲（开源） | ACE-Step XL 或 YuE |
| 短广告配乐 | MusicGen melody-conditioned + 哼唱参考 |
| 音乐视频背景 | MusicGen + Stable Video Diffusion |

## 2026 年仍在出现的坑

- **版权洗白 prompt。** "Song in the style of Taylor Swift" — 商业 Suno/Udio 现在会过滤这些，开源模型不会。加你自己的过滤列表。
- **超过 30 秒的重复 / 漂移。** AR 模型会循环。交叉淡入多次生成，或用 ACE-Step 获得结构连贯性。
- **节奏漂移。** 模型会偏离 BPM。在 prompt 中使用 BPM 标签，并用 librosa 的 `beat_track` 后处理过滤。
- **人声可懂度。** Suno 很好；开源模型的歌词往往含糊。如果歌词重要，用商业 API 或微调。
- **单声道输出。** 开源模型生成单声道或假立体声。用适当的立体声重建升级（ezst、Cartesia 的 stereo diffusion）。

## 交付

保存为 `outputs/skill-music-designer.md`。为音乐生成部署选择模型、许可策略、长度/结构规划和披露元数据。

## 练习

1. **Easy.** 运行 `code/main.py`。它以 ASCII 符号生成一个"生成式"和弦进行 + 鼓型 — 音乐生成的卡通版。如果你想的话可以用任何 MIDI 渲染器回放。
2. **Medium.** 安装 `audiocraft`，用 MusicGen-small 在 4 个风格 prompt 上生成 10 秒片段，对比参考风格集测量 FAD。
3. **Hard.** 使用 ACE-Step（或 MusicGen-melody），用不同音色 prompt 生成同一旋律的三个变体。计算与 prompt 的 CLAP 相似度以验证对齐。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| FAD | 音频版 FID | 真实与生成 embedding 分布之间的 Frechet 距离。 |
| Chromagram | 旋律的音高表示 | 12 维逐帧向量；旋律条件化的输入。 |
| Stems | 乐器轨道 | 分离的 bass / drums / vocals / melody WAV。 |
| Inpainting | 重新生成一段 | 遮蔽一个时间窗口；模型只重新生成那部分。 |
| CLAP | 文本-音频版 CLIP | 对比式音频-文本 embedding；评估文本-音频对齐。 |
| EnCodec | 音乐 codec | Meta 的 neural codec，MusicGen 使用；32 kHz，4 个 codebook。 |

## 延伸阅读

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 开源自回归基准。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 音效设计的默认选择。
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 开源 4B 完整歌曲生成器，2026 年 4 月。
- [Suno v5 platform docs](https://suno.com) — 商业质量领先者。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 音乐 + 音效的 latent diffusion。
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025 年 11 月的先例。
