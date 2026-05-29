# Whisper — 架构与微调

> Whisper 是一个 30 秒窗口的 transformer encoder-decoder，在 680k 小时多语言弱监督音频-文本对上训练。一个架构，多个任务，跨 99 种语言鲁棒。2026 年的参考 ASR。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 5 · 10 (Attention), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## 问题

Whisper 由 OpenAI 于 2022 年 9 月发布，是第一个作为商品化产品交付的 ASR 模型：输入音频，得到文本，99 种语言，对噪声鲁棒，笔记本电脑就能跑。到 2024 年 OpenAI 发布了 Large-v3 和 Turbo 变体；到 2026 年，Whisper 是从播客转录到语音助手到 YouTube 字幕的默认基线。

但 Whisper 不是一个你能永远当黑盒用的流水线。域偏移会杀死它 — 技术术语、说话人口音、专有名词、短片段、静音。你需要知道：

1. 它内部到底是什么。
2. 如何正确地给它分块、流式或长音频输入。
3. 什么时候微调以及怎么微调。

## 概念

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**架构。** 标准 transformer encoder-decoder。

- 输入：30 秒 log-mel spectrogram，80 mels，10 ms hop → 3000 帧。短于 30 秒的零填充，长于 30 秒的分块。
- Encoder：conv 下采样（stride 2）+ `N` 个 transformer block。Large-v3：32 层，1280 维，20 头。
- Decoder：`N` 个 transformer block，带因果 self-attn + 对 encoder 输出的 cross-attn。与 encoder 同尺寸。
- 输出：51,865 token 词表上的 BPE token。

Large-v3 有 1.55B 参数。Turbo 使用 4 层 decoder（从 32 层减少），延迟降低 8 倍，WER 损失 <1%。

**Prompt 格式。** Whisper 是一个多任务模型，通过 decoder prompt 中的特殊 token 来引导：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 语言标签；控制翻译 vs 转录行为。
- `<|transcribe|>` 或 `<|translate|>` — 从任意语言输入翻译为英语输出，或逐字转录。
- `<|notimestamps|>` — 跳过词级时间戳（更快）。

Prompt 让一个模型能做多个任务。把 `<|en|>` 改成 `<|fr|>` 就转录法语。

**30 秒窗口。** 一切都锁定在 30 秒。更长的音频需要分块；更短的零填充。窗口不能原生流式 — 这就是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**Log-mel 归一化。** `(log_mel - mean) / std`，统计量来自 Whisper 自己的训练语料。你 *必须* 使用 Whisper 的预处理（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### 2026 年的变体

| Variant | Params | Latency (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | streaming | 2.0% |

### 微调

2026 年的标准工作流：

1. 收集 10–100 小时目标领域音频及对齐的转录文本。
2. 用 `transformers.Seq2SeqTrainer` 配合 `generate_with_loss` callback 运行。
3. 参数高效：对 attention 层的 `q_proj`、`k_proj`、`v_proj` 加 LoRA，GPU 内存减少 4 倍，WER 损失 <0.3。
4. 如果数据 <10 小时，冻结 encoder，只调 decoder。
5. 使用 Whisper 自己的 tokenizer 和 prompt 格式；永远不要换 tokenizer。

社区结果：在 20 小时医疗听写上微调 Medium，医疗词汇的 WER 从 12% 降到 4.5%。在 4 小时冰岛语上微调 Turbo，WER 从 18% 降到 6%。

## 动手构建

### Step 1：开箱即用运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

你应该始终覆盖的关键默认值：`temperature=0.0`（采样默认 0.0 → 0.2 → 0.4 … 回退链）、`condition_on_previous_text=False`（防止级联幻觉问题）、`no_speech_threshold=0.6`（静音检测）。

### Step 2：分块长音频

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 增加了 (1) Silero VAD 门控，(2) 通过 wav2vec 2.0 的词级对齐，(3) 通过 `pyannote.audio` 的说话人分离。2026 年生产转录的主力工具。

### Step 3：用 LoRA 微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

然后是标准 Trainer 循环。每 1000 步保存 checkpoint。用 held-out 集上的 WER 评估。

### Step 4：检查每层学到了什么

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

用热力图可视化 — 你会看到对角线对齐，decoder 步骤扫过 encoder 帧。那条对角线就是 Whisper 对词时间戳的理解。

## 实际使用

2026 年技术栈：

| Situation | Pick |
|-----------|------|
| 通用英语、离线 | Large-v3-turbo via `whisperx` |
| 移动端 / 边缘 | Whisper-Tiny 量化 (int8) 或 Moonshine |
| 多语言长音频 | Large-v3 via `whisperx` + 说话人分离 |
| 低资源语言 | 用 LoRA 微调 Medium 或 Turbo |
| 流式（2 秒延迟） | Whisper-Streaming 或 Parakeet-TDT |
| 词级时间戳 | WhisperX（通过 wav2vec 2.0 强制对齐） |

`faster-whisper`（CTranslate2 后端）是 2026 年最快的 CPU+GPU 推理运行时 — 比原版快 4 倍，输出完全相同。

## 2026 年仍在出现的坑

- **静音上的幻觉文本。** Whisper 在字幕上训练，会产生 "Thanks for watching!"、"Subscribe!"、歌词。调用前务必用 VAD 门控。
- **`condition_on_previous_text` 级联。** 一次幻觉污染后续窗口。除非需要跨块流畅性，否则设为 `False`。
- **短片段填充。** 2 秒片段填充到 30 秒可能在尾部静音中产生幻觉。使用 `pad=False` 或 VAD 门控。
- **错误的 mel 统计量。** 用 librosa 的 mels 代替 Whisper 的会产生近乎随机的输出。使用 `whisper.audio.log_mel_spectrogram`。

## 交付

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计 Whisper 微调或推理流水线。

## 练习

1. **Easy.** 运行 `code/main.py`。它对 Whisper 风格的 prompt 做 tokenize，计算解码形状预算，并打印 10 分钟音频的分块调度。
2. **Medium.** 安装 `faster-whisper`，转录一段 10 分钟播客，与人工转录比较 WER。尝试 `language="auto"` vs 强制 `language="en"`。
3. **Hard.** 使用 HF `datasets`，选一种 Whisper 表现不好的语言（如乌尔都语），用 LoRA 在 2 小时数据上微调 Medium 2 个 epoch，报告 WER 变化。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 30-sec window | Whisper 的限制 | 硬性输入上限；更长音频需分块。 |
| SOT | Start-of-transcript | `<\|startoftranscript\|>` 启动 decoder prompt。 |
| Timestamps token | 时间对齐 | 每 0.02 秒偏移是 51k 词表中的一个特殊 token。 |
| Turbo | 快速变体 | 4 层 decoder，快 8 倍，WER 回退 <1%。 |
| WhisperX | 长音频封装 | VAD + Whisper + wav2vec 对齐 + 说话人分离。 |
| LoRA fine-tune | 高效微调 | 在 attention 上加低秩适配器；训练约 0.3% 的参数。 |
| Hallucination | 静默失败 | Whisper 从噪声/静音中产生流畅英语。 |

## 延伸阅读

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — 原始架构和训练方案。
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4 层 decoder，8 倍加速。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 长音频、词对齐、说话人分离。
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 支持，快 4 倍。
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — 标准 LoRA / 全量微调教程。
