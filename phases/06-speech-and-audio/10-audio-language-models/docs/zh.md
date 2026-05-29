# Audio-Language Models — Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的 audio-language models 能对语音、环境声和音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上追平 GPT-4o Audio。Audio Flamingo Next 在 LongAudioBench 上超越 Gemini 2.5 Pro。开源与闭源的差距基本消失——除了多音频任务，所有模型都接近随机水平。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time:** ~45 minutes

## 问题

你有 5 秒音频：狗叫、有人喊"stop!"、然后沉默。有用的问题横跨多个维度：

- **转录。** "说了什么？"——ASR 的领域。
- **语义推理。** "这个人有危险吗？"——需要联合理解狗叫 + 喊声 + 沉默。
- **音乐推理。** "旋律中用了什么乐器？"
- **长音频检索。** "这段 90 分钟的讲座中，讲师在哪里讲解了梯度下降？"

一个模型用一条 prompt 回答所有这些问题，就是 **audio-language model**（LALM / ALM）。它和纯 ASR 不同：LALM 生成自由形式的自然语言回答，而不仅仅是转录文本。

## 概念

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### 三组件模板

2026 年每个 LALM 都有相同的骨架：

1. **Audio encoder。** Whisper encoder · BEATs · CLAP · WavLM · 或各模型自定义的 encoder。
2. **Projector。** Linear 或 MLP，将 audio encoder 的特征桥接到 LLM 的 token embedding 空间。
3. **LLM。** 基于 Llama / Qwen / Gemma 的 decoder。接收交错的文本 + 音频 token；生成文本。

训练：

- **Stage 1。** 冻结 encoder + LLM；仅在 ASR / captioning 数据上训练 projector。
- **Stage 2。** 全量 / LoRA 微调，用于指令跟随的音频任务（QA、推理、音乐理解）。
- **Stage 3（可选）。** Voice-in / voice-out 添加语音 decoder。Qwen2.5-Omni 和 AF3-Chat 做了这一步。

### 2026 模型全景

| Model | Backbone | Audio encoder | Output modality | Access |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### Benchmark 现实检验（2026）

**MMAU-Pro。** 1800 个 QA 对，覆盖语音 / 声音 / 音乐 / 混合。包含多音频子集。

| Model | Overall | Speech | Sound | Music | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA on LongAudioBench | — | — | — | — |

**多音频列是所有模型的致命弱点。** 4 选 1 随机猜测 = 25%；大多数模型的得分就在这附近。LALM 仍然难以比较两段音频。

### 2026 年 LALM 的实用场景

- **呼叫中心录音合规审计。** "客服是否提到了必须的免责声明？"
- **无障碍。** 为听障用户描述声音事件（不仅仅是转录）。
- **内容审核。** 检测暴力语言 + 威胁语气 + 背景上下文。
- **播客 / 会议分章。** 语义摘要，而不仅仅是说话人轮次。
- **音乐目录分析。** "找出所有 B 段有转调的曲目。"

### 目前还不实用的场景

- 细粒度乐理分析（和弦级别以下）。
- 长对话中的说话人归因推理（超过 10 分钟后退化）。
- 多音频比较（22-26% 几乎等于随机）。
- 实时流式推理（大多数是离线批处理推理）。

## 动手构建

### Step 1：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Step 2：projector 模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就这么简单。Projector 通常是 1-3 层线性层。在 ASR 对（audio → transcript）上训练它就是 Stage-1 的预训练任务。

### Step 3：在 MMAU / LongAudioBench 上评测

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

分类别（speech / sound / music / multi-audio）单独报告。聚合数字会掩盖模型的薄弱环节。

## 使用指南

| Task | 2026 pick |
|------|-----------|
| Free-form audio QA (open) | Qwen2.5-Omni-7B |
| Best open on long audio | Audio Flamingo Next |
| Best closed | Gemini 2.5 Pro |
| Voice-in / voice-out agent | Qwen2.5-Omni or GPT-4o Audio |
| Music reasoning | Audio Flamingo 3 or 2 (music-specialized AF-CLAP) |
| Call-center audit | Gemini 2.5 Pro via API, with RAG over your policy docs |

## 常见陷阱

- **过度信任多音频能力。** 如果你的任务需要"哪段音频包含 X"，随机水平的表现是真实的。
- **长音频退化。** 超过 10 分钟后，大多数模型的说话人归因会崩溃。先做 diarization（Lesson 6），再做摘要。
- **静音幻觉。** 使用 Whisper encoder 的 LALM 继承了同样的问题。用 VAD 做门控。
- **Benchmark 挑数据。** 厂商博客会突出最好的类别。自己跑 MMAU-Pro 多音频子集。

## 交付

保存为 `outputs/skill-alm-picker.md`。为给定的音频理解任务选择 LALM + benchmark 子集 + 输出模态（text vs speech）。

## 练习

1. **Easy。** 运行 `code/main.py`，查看一个 toy projector 模式 + 假 LALM 路由 (audio-embedding, text-tokens) → output tokens。
2. **Medium。** 在 100 个 MMAU-Pro speech 项目上评测 Qwen2.5-Omni-7B。与论文报告的数字对比。
3. **Hard。** 构建一个最小的 audio-captioning baseline：BEATs encoder + 2 层 projector + 冻结的 Llama-3.2-1B。仅微调 projector，数据用 AudioCaps。在 Clotho-AQA 上与 SALMONN 对比。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | Audio encoder + projector + LLM decoder. |
| Projector | Adapter | Small MLP mapping audio features into LLM embedding space. |
| MMAU | The benchmark | 10k audio-QA pairs across speech, sound, music. |
| MMAU-Pro | Harder MMAU | 1800 multi-audio / reasoning-heavy questions. |
| LongAudioBench | Long-form eval | Multi-minute clips with semantic queries. |
| Voice-in / voice-out | Speech-native | Model ingests speech and emits speech without text detour. |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — reference architecture.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — speech-in-speech-out.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — the open long-audio leader.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — dual-encoder pioneer.
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — live 2026 rankings.
