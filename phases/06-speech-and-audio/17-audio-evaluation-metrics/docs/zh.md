# 音频评估指标 — WER、MOS、UTMOS、MMAU、FAD 与开放排行榜

> 无法度量就无法上线。本课列出 2026 年每个音频任务的指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、WER-on-ASR-round-trip）、audio-language（MMAU、LongAudioBench）、音乐（FAD、CLAP）和说话人（EER）。加上你用来比较的排行榜。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 minutes

## 问题

每个音频任务都有多个指标，各自衡量不同的维度。用错指标就是你上线一个在仪表盘上看起来很好、在生产中表现糟糕的模型的方式。2026 年的标准列表：

| Task | Primary | Secondary |
|------|---------|-----------|
| ASR | WER | CER · RTFx · first-token latency |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| Voice cloning | SECS (ECAPA cosine) | MOS · CER |
| Speaker verification | EER | minDCF · FAR / FRR at operating point |
| Diarization | DER | JER · speaker confusion |
| Audio classification | top-1 · mAP | macro F1 · per-class recall |
| Music generation | FAD | CLAP · listening panel MOS |
| Audio language model | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S | latency P50/P95 | WER · MOS |

## 概念

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### ASR 指标

**WER (Word Error Rate)。** `(S + D + I) / N`。评分前先小写、去标点、数字归一化。使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。< 5% = 朗读语音的人类水平。

**CER (Character Error Rate)。** 同样的公式，字符级别。用于声调语言（普通话、粤语），因为分词有歧义。

**RTFx（逆实时因子）。** 每 wall-clock 秒处理的音频秒数。越高越好。Parakeet-TDT 达到 3380×。Whisper-large-v3 约 30×。

**First-token latency。** 从音频输入到第一个转录 token 的 wall-clock 时间。对 streaming 至关重要。Deepgram Nova-3：约 150 ms。

### TTS 指标

**MOS (Mean Opinion Score)。** 1-5 人类评分。金标准但慢。每个样本收集 20+ 听众，每个模型 100+ 样本。

**UTMOS（2022-2026）。** 学习到的 MOS 预测器。在标准 benchmark 上与人类 MOS 相关性约 0.9。F5-TTS：UTMOS 3.95；ground truth：4.08。

**SECS (Speaker Encoder Cosine Similarity)。** 用于 voice cloning。参考音频和克隆输出之间的 ECAPA embedding 余弦相似度。> 0.75 = 可识别的克隆。

**WER-on-ASR-round-trip。** 对 TTS 输出运行 Whisper，计算与输入文本的 WER。捕获可懂度回归。2026 SOTA：< 2% CER。

**TTFA (time-to-first-audio)。** Wall-clock 延迟。Kokoro-82M：约 100 ms；F5-TTS：约 1 s。

### Voice cloning 专用

**SECS + MOS + CER** 作为三元组。SECS 高但 MOS 低意味着音色对了但不自然；反过来意味着声音自然但说话人不对。

### 说话人验证

**EER (Equal Error Rate)。** False Accept Rate 等于 False Reject Rate 的阈值。ECAPA 在 VoxCeleb1-O 上：0.87%。

**minDCF (min Detection Cost)。** 在选定操作点（通常 FAR=0.01）的加权代价。比 EER 更贴近生产。

### Diarization

**DER (Diarization Error Rate)。** `(FA + Miss + Confusion) / total_speaker_time`。漏检语音 + 误报语音 + 说话人混淆，各自作为比例。AMI 会议：DER 约 10-20% 是现实的。pyannote 3.1 + Precision-2 商业版：录音质量好时 <10% DER。

**JER (Jaccard Error Rate)。** DER 的替代方案，对短片段偏差更鲁棒。

### 音频分类

多标签：**mAP (mean Average Precision)** 覆盖所有类别。AudioSet：BEATs-iter3 达到 0.548 mAP。

多类互斥：**top-1, top-5 accuracy**。Speech Commands v2：99.0% top-1（Audio-MAE）。

不平衡：**macro F1** + **per-class recall**。逐类报告——聚合准确率会掩盖哪些类别失败。

### 音乐生成

**FAD (Fréchet Audio Distance)。** 真实 vs 生成音频的 VGGish embedding 分布之间的距离。MusicGen-small 在 MusicCaps 上：4.5。MusicLM：4.0。越低越好。

**CLAP Score。** 使用 CLAP embedding 的文本-音频对齐分数。> 0.3 = 合理的对齐。

**听众面板 MOS。** 仍然是消费级音乐的最终裁判。Suno v5 在 TTS Arena 上 ELO 1293（来自配对人类偏好）。

### Audio-language benchmarks

**MMAU (Massive Multi-Audio Understanding)。** 10k 音频 QA 对。

**MMAU-Pro。** 1800 个困难项目，四个类别：speech / sound / music / multi-audio。4 选 1 随机猜测 25%。Gemini 2.5 Pro 总体约 60%；multi-audio 所有模型约 22%。

**LongAudioBench。** 多分钟片段配语义查询。Audio Flamingo Next 超越 Gemini 2.5 Pro。

**AudioCaps / Clotho。** Captioning benchmark。SPICE、CIDEr、FENSE 指标。

### Streaming speech-to-speech

**Latency P50 / P95 / P99。** 从用户语音结束到第一个可听回应的 wall-clock 时间。Moshi：200 ms；GPT-4o Realtime：300 ms。

**WER / MOS** 作用于输出。

**Barge-in 响应性。** 从用户打断到助手静音的时间。目标 < 150 ms。

### 2026 排行榜

| Leaderboard | Tracks | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | English + multilingual + long-form | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO from paired votes | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM reasoning | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | Speaker recognition | `voxsrc.github.io` |
| MMAU music subset | Music LALM | (within MMAU) |
| HEAR benchmark | Self-supervised audio | `hearbenchmark.com` |

## 动手构建

### Step 1：带归一化的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Step 2：TTS round-trip WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Step 3：voice cloning 的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Step 4：音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Step 5：说话人验证的 EER（与 Lesson 6 相同代码）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 使用指南

每次部署都配一个固定的 eval harness，在每次模型更新时运行。三条基本规则：

1. **评分前归一化。** 小写、去标点、数字展开。报告归一化规则。
2. **报告分布，不是平均值。** 延迟用 P50/P95/P99。分类用 per-class recall。MMAU 用 per-category。
3. **跑一个标准公开 benchmark。** 即使你的生产数据不同，在 Open ASR / TTS Arena / MMAU 上报告让评审者能做苹果对苹果的比较。

## 常见陷阱

- **UTMOS 外推。** 在 VCTK 风格的干净语音上训练；对噪声 / 克隆 / 情感音频评分不准。
- **MOS 面板偏差。** 20 个 Amazon Mechanical Turk 工人 ≠ 20 个目标用户。如果风险高，花钱请领域面板。
- **FAD 依赖参考集。** 跨模型比较时使用相同的参考分布。
- **聚合 WER。** 总体 5% WER 可能隐藏了口音语音上 30% 的 WER。按人口统计切片报告。
- **公开 benchmark 饱和。** 大多数前沿模型在标准 benchmark 上接近天花板。构建反映你流量的内部留出集。

## 交付

保存为 `outputs/skill-audio-evaluator.md`。为任何音频模型发布选择指标、benchmark 和报告格式。

## 练习

1. **Easy。** 运行 `code/main.py`。在 toy 输入上计算 WER / CER / EER / SECS / FAD-ish / MMAU-ish。
2. **Medium。** 构建一个 TTS round-trip WER harness。将你的 Kokoro 或 F5-TTS 输出通过 Whisper。在 50 个 prompt 上计算 WER。标记 WER > 10% 的 prompt。
3. **Hard。** 在 MMAU-Pro speech + multi-audio 子集（各 50 项）上评测你 Lesson 10 选择的 LALM。报告 per-category accuracy 并与发表的数字对比。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| WER | ASR score | `(S+D+I)/N` at word level after normalization. |
| CER | Character WER | For tone languages or char-level systems. |
| MOS | Human opinion | 1-5 rating; 20+ listeners × 100 samples. |
| UTMOS | ML MOS predictor | Learned model; correlates ~0.9 with human MOS. |
| SECS | Voice-clone similarity | ECAPA cosine between reference and clone. |
| EER | Speaker verif score | Threshold where FAR = FRR. |
| DER | Diarization score | (FA + Miss + Confusion) / total. |
| FAD | Music-gen quality | Fréchet distance on VGGish embeddings. |
| RTFx | Throughput | Audio seconds per wall-clock second. |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — WER/CER library with normalization utilities.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — learned MOS predictor.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — the music-gen standard.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 live rankings.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — human-vote TTS leaderboard.
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM reasoning leaderboard.
- [HEAR benchmark](https://hearbenchmark.com/) — audio SSL benchmarks.
