# 声音克隆与声音转换

> 声音克隆用别人的声音朗读你的文本。声音转换把你的声音改写成别人的，同时保留你说的内容。两者都依赖同一个分解：将说话人身份与内容分离。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## 问题

2026 年，一段 5 秒的音频就足以用消费级 GPU 生成任何人声音的高质量克隆。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都提供零样本或少样本克隆。这项技术既是福音（无障碍 TTS、配音、辅助语音）也是武器（诈骗电话、政治 deepfake、知识产权盗窃）。

两个密切相关的任务：

- **声音克隆（TTS 侧）：** 文本 + 5 秒参考声音 → 用那个声音的音频。
- **声音转换（语音侧）：** 源音频（A 说 X）+ B 的参考声音 → B 说 X 的音频。

两者都将波形分解为（内容、说话人、韵律），然后用一个来源的内容与另一个来源的说话人重新组合。

2026 年你必须遵守的关键约束：**水印和同意门控在欧盟（AI Act，2026 年 8 月生效）和加州（AB 2905，2025 年生效）已是法律要求**。你的流水线必须嵌入不可听水印并拒绝未经同意的克隆。

## 概念

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**零样本克隆。** 将 5 秒音频传给一个在数千说话人上训练过的模型。Speaker encoder 将音频映射为 speaker embedding；TTS decoder 以该 embedding 加文本为条件生成。

使用者：F5-TTS (2024)、YourTTS (2022)、XTTS v2 (2024)、OpenVoice v2 (2024)。

**少样本微调。** 录制目标声音 5-30 分钟。对基础模型做 LoRA 微调一小时。质量从"还行"跃升到"难以区分"。Coqui 和 ElevenLabs 都支持这种模式；社区用 F5-TTS 做同样的事。

**声音转换 (VC)。** 两个家族：

- **识别-合成。** 运行类 ASR 模型提取内容表示（如 soft phoneme posteriors、PPG），然后用目标 speaker embedding 重新合成。对语言和口音鲁棒。KNN-VC (2023)、Diff-HierVC (2023) 使用。
- **解耦。** 训练一个自编码器，在瓶颈处的潜空间中分离内容、说话人和韵律。推理时替换 speaker embedding。质量较低但更快。AutoVC (2019)、VITS-VC 变体使用。

**基于 neural codec 的克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox — 将音频视为 SoundStream / EnCodec 的离散 token，训练大型自回归或 flow-matching 模型。短 prompt 上质量可比 ElevenLabs。

### 伦理部分，不是附加项

**水印。** PerTh (Perth) 和 SilentCipher (2024) 在音频中不可感知地嵌入约 16-32 bit ID。经受重编码、流式传输和常见编辑后仍可检测。生产就绪的开源方案。

**同意门控。** 每个克隆输出必须配对一条可验证的同意记录。"我，Rohit，于 2026-04-22，授权此声音用于 X 目的。"存储在防篡改日志中。

**检测。** AASIST、RawNet2 和 Wav2Vec2-AASIST 作为检测器发布。ASVspoof 2025 挑战赛公布了 SOTA 检测器对 ElevenLabs、VALL-E 2 和 Bark 输出的 EER 为 0.8–2.3%。

### 数字（2026）

| Model | Zero-shot? | SECS (target sim) | WER (intel.) | Params |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Yes | 0.72 | 2.1% | 335M |
| XTTS v2 | Yes | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Yes | 0.70 | 2.8% | 220M |
| VALL-E 2 | Yes | 0.77 | 2.4% | 370M |
| VoiceBox | Yes | 0.78 | 2.1% | 330M |

SECS > 0.70 对大多数听众来说与目标声音难以区分。

## 动手构建

### Step 1：用识别-合成分解（main.py 中的代码演示）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念上简单；实现的主体在 `tts_model` 和 speaker encoder 中。

### Step 2：用 F5-TTS 零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转录文本必须与参考音频完全匹配；不匹配会破坏对齐。

### Step 3：用 KNN-VC 做声音转换

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 用 WavLM 提取源和目标池的逐帧 embedding，然后将每个源帧替换为池中最近邻。非参数化，一分钟目标语音即可工作。

### Step 4：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

约 32 bit payload，MP3 重编码和轻度噪声后仍可检测。

### Step 5：同意门控

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 实际使用

2026 年技术栈：

| Situation | Pick |
|-----------|------|
| 5 秒零样本克隆，开源 | F5-TTS 或 OpenVoice v2 |
| 商业生产克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 声音转换（改写） | KNN-VC 或 Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + speaker adapter |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| Deepfake 检测 | Wav2Vec2-AASIST |

## 常见坑

- **参考转录不对齐。** F5-TTS 等要求参考文本与参考音频完全匹配，包括标点。
- **混响参考。** 回声会杀死克隆效果。录制干声、近距离拾音。
- **情绪不匹配。** 训练参考"欢快"会让所有克隆都欢快。参考情绪要匹配目标用途。
- **语言泄漏。** 克隆英语说话人然后让模型说法语，往往会带着英语口音；使用跨语言模型（XTTS、VALL-E X）。
- **没有水印。** 2026 年 8 月起在欧盟不可合法发布。

## 交付

保存为 `outputs/skill-voice-cloner.md`。设计一个带同意门控 + 水印 + 质量目标的克隆或转换流水线。

## 练习

1. **Easy.** 运行 `code/main.py`。通过计算两个"说话人"交换前后的余弦来演示 speaker-embedding 交换。
2. **Medium.** 用 OpenVoice v2 克隆你自己的声音。测量参考和克隆之间的 SECS。通过 Whisper 测量 CER。
3. **Hard.** 对 20 个克隆应用 SilentCipher 水印，通过 128 kbps MP3 编解码，检测 payload。报告 bit 准确率。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Zero-shot clone | 5 秒就够 | 预训练模型 + speaker embedding；无需训练。 |
| PPG | Phonetic posteriorgram | 逐帧 ASR 后验概率，用作语言无关的内容表示。 |
| KNN-VC | 最近邻转换 | 将每个源帧替换为目标池中最近邻帧。 |
| Neural codec TTS | VALL-E 风格 | 在 EnCodec/SoundStream token 上的 AR 模型。 |
| Watermark | 不可听签名 | 嵌入音频中的 bit，经受重编码。 |
| SECS | 克隆保真度 | 目标和克隆 speaker embedding 之间的余弦。 |
| AASIST | Deepfake 检测器 | 反欺骗模型；检测合成语音。 |

## 延伸阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 开源 SOTA 零样本克隆。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) and [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — neural-codec TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 基于解耦的声音转换。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 基于检索的 VC。
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — 生产就绪的 32-bit 音频水印。
- [ASVspoof 2025 results](https://www.asvspoof.org/) — 检测器 vs 合成器军备竞赛，2026 年更新。
