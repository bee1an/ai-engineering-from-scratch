# 语音识别 (ASR) — CTC、RNN-T、Attention

> 语音识别就是在每个时间步做音频分类，再用一个懂英语和静音的序列模型把它们粘在一起。CTC、RNN-T 和 attention 是三种做法。选一种，理解为什么。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**Time:** ~45 minutes

## 问题

你有一段 10 秒、16 kHz 的音频。你想要一个字符串："turn on the kitchen lights"。挑战是结构性的：音频帧和字符之间不是一一对应的。"okay" 这个词可能占 200 ms 也可能占 1200 ms。静音穿插在语句中。有些音素比其他的长。输出 token 的数量事先未知。

三种方案解决这个问题：

1. **CTC (Connectionist Temporal Classification)。** 输出每帧的 token 概率，包含一个特殊的 *blank*。解码时合并重复和 blank。非自回归，快速。wav2vec 2.0、MMS 使用。
2. **RNN-T (Recurrent Neural Network Transducer)。** 联合网络根据 encoder 帧和之前的 token 预测下一个 token。可流式。Google 的设备端 ASR、NVIDIA Parakeet 使用。
3. **Attention encoder-decoder。** Encoder 将音频压缩为隐状态，decoder 通过 cross-attention 自回归生成 token。Whisper、SeamlessM4T 使用。

2026 年，LibriSpeech test-clean 上的 SOTA WER 是 1.4%（Parakeet-TDT-1.1B, NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。差异很小；部署差异很大。

## 概念

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC 直觉。** 让 encoder 输出 `T` 个帧级分布，覆盖 `V+1` 个 token（V 个字符 + blank）。对于长度为 `U < T` 的目标字符串 `y`，任何能合并为 `y` 的帧对齐都算数。CTC loss 对所有这样的对齐求和。推理：逐帧 argmax，合并重复，移除 blank。

优势：非自回归、可流式、零前瞻。缺点：*条件独立假设* — 每帧的预测独立于其他帧，因此没有内部语言模型。通过 beam search 或 shallow fusion 加外部 LM 来修补。

**RNN-T 直觉。** 增加一个 *predictor* 网络来编码 token 历史，以及一个 *joiner* 将 predictor 状态与 encoder 帧组合成 `V+1` 上的联合分布（`+1` 是 null / 不输出）。显式建模了 CTC 忽略的条件依赖。可流式，因为每步只依赖过去的帧和过去的 token。

优势：可流式 + 内部 LM。缺点：训练更复杂、更耗内存（3D loss lattice）；RNN-T loss kernel 本身就是一个库类别。

**Attention encoder-decoder。** Encoder（6-32 层 transformer）处理 log-mel 帧。Decoder（6-32 层 transformer）通过 cross-attention 关注 encoder 输出，自回归生成 token。没有对齐约束 — attention 可以看音频的任何位置。除非限制 attention（chunked Whisper-Streaming, 2024），否则不可流式。

优势：离线 ASR 质量最高，用标准 seq2seq 工具训练简单。缺点：自回归延迟与输出长度成正比；不做工程改造无法流式。

### WER：那一个数字

**Word Error Rate** = `(S + D + I) / N`，其中 S=替换、D=删除、I=插入、N=参考词数。等价于词级别的 Levenshtein 编辑距离。越低越好。WER 超过 20% 通常不可用；低于 5% 对朗读语音来说是人类水平。2026 年标准基准上的数字：

| Model | LibriSpeech test-clean | LibriSpeech test-other | Size |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

这些都是 encoder-decoder 或 RNN-T 架构。纯 CTC 系统（wav2vec 2.0）在 test-clean 上大约 1.8–2.1%。

## 动手构建

### Step 1：贪心 CTC 解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：合并连续重复，丢弃 blank。示例：`a a _ _ a b b _ c` → `a a b c`。

### Step 2：beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境用前缀树 beam search 加 LM fusion；这里是概念骨架。

### Step 3：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Step 4：用 Whisper 推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026 年最强通用 ASR 的一行代码。在 24 GB GPU 上以约 20 倍实时速度运行。

### Step 5：用 Parakeet 或 wav2vec 2.0 流式处理

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块 encoder attention 和状态延续；使用支持它的库（NeMo 用于 Parakeet，`transformers` pipeline 配合 `chunk_length_s`）。

## 实际使用

2026 年技术栈：

| Situation | Pick |
|-----------|------|
| 英语、离线、最高质量 | Whisper-large-v3-turbo |
| 多语言、鲁棒 | SeamlessM4T v2 |
| 流式、低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘、移动端、<500 ms 延迟 | Whisper-Tiny 量化 或 Moonshine (2024) |
| 长音频 | Whisper + 基于 VAD 的分块（WhisperX） |
| 领域特定（医疗、法律） | 微调 wav2vec 2.0 + 领域 LM fusion |

## 2026 年仍在出现的坑

- **没有 VAD。** 对静音运行 Whisper 会产生幻觉（"Thanks for watching!"）。务必用 VAD 门控。
- **字符 vs 词 vs subword WER。** 报告归一化后（小写、去标点）的词级 WER。
- **语言 ID 漂移。** Whisper 的自动 LID 会把噪声片段误判为日语或威尔士语；确定语言时强制 `language="en"`。
- **长音频不分块。** Whisper 有 30 秒窗口。超过这个长度用 `chunk_length_s=30, stride=5`。

## 交付

保存为 `outputs/skill-asr-picker.md`。为给定的部署目标选择模型、解码策略、分块方式和 LM fusion。

## 练习

1. **Easy.** 运行 `code/main.py`。它对手工构造的 CTC 输出做贪心解码，并计算与参考的 WER。
2. **Medium.** 正确实现 Step 2 中的前缀树 beam search（处理 blank 合并规则）。在 10 个合成样本上与贪心解码比较。
3. **Hard.** 用 `whisper-large-v3-turbo` 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上运行。计算前 100 条 utterance 的 WER。与公布数字比较。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| CTC | blank-token loss | 对所有帧到 token 对齐的边际化；非自回归。 |
| RNN-T | 流式 loss | CTC + next-token predictor；处理词序。 |
| Attention enc-dec | Whisper 风格 | Encoder + cross-attending decoder；最佳离线质量。 |
| WER | 你报告的那个数字 | 词级别的 `(S+D+I)/N`。 |
| Blank | 空白 | CTC 中表示"这一帧不输出"的特殊 token。 |
| LM fusion | 外部语言模型 | 在 beam search 中加入加权的 LM log-probs。 |
| VAD | 静音门控 | Voice activity detector；裁剪非语音部分。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022 年的经典论文；v3-turbo 扩展于 2024 年。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 Open ASR Leaderboard 领先者。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 25+ 模型的实时基准。
