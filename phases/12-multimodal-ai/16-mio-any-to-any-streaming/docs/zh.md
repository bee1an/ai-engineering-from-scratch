# MIO 与 Any-to-Any 流式多模态模型

> GPT-4o 交付了一个大多数开源模型无法复制的产品：一个能听到语音、看到视频、并实时回话的 agent。开源生态到 2024 年底的答案是 MIO（Wang et al.，2024 年 9 月）。MIO 将文本、图像、语音和音乐 tokenize，在交错序列上训练一个因果 transformer，并生成任意模态到任意模态。AnyGPT（Zhan et al.，2024 年 2 月）是概念验证；MIO 是规模化版本；Unified-IO 2（Allen AI，2023 年 12 月）是带视觉 + 动作 grounding 的表亲。本课阅读 any-to-any 模式——四个 tokenizer、一个 transformer、流式友好的解码。

**Type:** Learn
**Languages:** Python (stdlib, four-modality token allocator + streaming decode loop)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 6 (Speech and Audio)
**Time:** ~120 minutes

## 学习目标

- 设计一个承载文本、图像、语音和音乐 token 且无冲突的共享词表。
- 对比 SEED-Tokenizer（图像）和 SpeechTokenizer residual-VQ（语音）在压缩 + 重建上的权衡。
- 解释构建 any-to-any 生成的四阶段课程。
- 说出三个开源 any-to-any 方案及其主要权衡：MIO、AnyGPT、Unified-IO 2。

## 问题

统一多模态模型容易声称但难以大规模构建。2024 年之前大多数"any-to-any"系统是流水线式的：视觉模型 → 文本表示 → 语音模型 → 音频。每一跳都丢失信息、增加延迟、复杂化训练。GPT-4o 的演示视频展示了一个亚秒响应的单模型替代方案；开源系统落后了数月。

工程挑战：

- 每个模态都必须有 tokenizer，压缩到足够无损以支持重建，并以 transformer 能消费的速率产出 token。
- 单一词表必须为文本（32k+）、图像（16k+）、语音（4k+）、音乐（8k+）分配空间。最少四万多条目。
- 训练数据必须覆盖每个输入-输出对（文本→图像、图像→语音、语音→图像等），否则模型必须组合。
- 推理必须足够快地流式输出 token 以满足对话延迟（<500ms time-to-first-audio-byte）。

## 核心概念

### 四个模态的四个 tokenizer

MIO 的 tokenizer 栈：

- 文本：标准 BPE，vocab ~32000。
- 图像：SEED-Tokenizer (2023)——量化 VAE 带离散 codebook，4096 条目，每张图像 32x32 token。
- 语音：SpeechTokenizer residual-VQ (2023)——将 16kHz 波形编码为 8 个层级 codebook；第一层是粗粒度内容，后续层添加韵律和说话人身份。
- 音乐：类似的 residual-VQ（Meta 的 MusicGen / Encodec 家族），4-8 个 codebook。

每个模态产出整数 token。这些 token 在共享词表中获得不相交的 ID 范围：

```
text:   0..31999
image:  32000..36095  (4096 image tokens)
speech: 36096..40191  (4096 speech base tokens, plus residual layers)
music:  40192..48383  (8192 music tokens)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, etc.)
```

总计：约 48k 词表。输入 embedding 和输出 projection 覆盖全部。

### 流式解码

语音生成使用 residual-VQ。Transformer 预测基础（layer 0）语音 token；一个并行解码的残差量化器预测后续层。每个 layer 0 token 大约是 16kHz 下 50ms 的音频。

流式模式：

1. 用户对着麦克风说话；实时音频 tokenizer 每 50ms 输出语音 token。
2. MIO 在 token 到达时消费它们（prompt prefill + 增量前向）。
3. 输出 token 在生成时流出；并行语音解码器以约 50-150ms 延迟将它们转换为音频样本。
4. Time-to-first-audio-byte：MIO 论文中约 300-500ms，接近 GPT-4o 的约 250ms。

Mini-Omni (arXiv:2408.16725)、GLM-4-Voice (arXiv:2412.02612) 和 Moshi (arXiv:2410.00037) 是互补的流式语音-LLM 设计。Moshi 尤其在单 GPU 上实现了 160ms 往返。

### 四阶段课程

MIO 的训练课程：

1. Stage 1——对齐。大规模模态对语料：文本-图像、文本-语音、文本-音乐。每对使用自己的 token 词表段。训练共享词表。
2. Stage 2——交错。多模态交错文档（带图像 + 视频的博客、带转录的播客等）。训练跨模态上下文。
3. Stage 3——语音增强。额外音频数据提升语音质量而不丢失文本能力。
4. Stage 4——SFT。跨模态指令调优：VQA、captioning、叙述、语音对语音对话。

跳过某个阶段会降低特定能力：跳过 stage 2 模型丢失跨模态上下文；跳过 stage 3 语音质量差。

### Chain-of-visual-thought

MIO 引入了 chain-of-visual-thought：模型输出中间图像 token 作为推理步骤。对于"猫在爬树吗？"模型：

1. 输出 `<image>` token 渲染场景（来自输入图像或草图）。
2. 输出文本分析草图。
3. 输出最终答案。

渲染的中间图像充当草稿纸。空间推理任务的 benchmark 提升。这个想法类似于文本推理的 chain-of-thought。

### Any-to-any 的竞争者

- AnyGPT (arXiv:2402.12226)：4 个模态（文本、图像、语音、音乐），类似设计。
- Unified-IO 2 (arXiv:2312.17172)：增加视觉动作输出、深度、法线。更多任务多样性，更小规模。
- NExT-GPT (arXiv:2309.05519)：LLM + 模态专属 diffusion 解码器。不是单模型方案。
- CoDi (arXiv:2305.11846)：可组合 diffusion；通过共享 latent 实现 any-to-any。

MIO 最接近纯 token any-to-any。AnyGPT 是它的概念祖先。

### 延迟预算

对于对话产品，每个组件的延迟都重要：

- 麦克风到音频 token：约 50ms。
- Prefill（音频 token + 历史）：8B 模型上约 100ms。
- 第一个输出 token：约 50ms。
- 并行 residual-VQ + 语音解码器：约 100-150ms。

总 time-to-first-audio-byte：最少约 300ms。GPT-4o 声称约 250ms。Moshi 声称 160ms。MIO/AnyGPT 在公开 benchmark 中为 400-600ms 范围。

### 为什么 any-to-any 仍然很难

即使在 2026 年，开源 any-to-any 模型在两个轴上落后于闭源：

- 语音质量。Residual-VQ tokenizer 有损；对话语音听起来比 ElevenLabs 级别的声音更机械。
- 跨模态推理。让模型"唱出你看到的"仍然比纯视觉任务更容易失败。

这些是开放研究问题。Qwen3-Omni（Lesson 12.20）是 2025 年最先进的开源尝试。

## Use It

`code/main.py`：

- 定义四模态词表分配并打印。
- 将多模态输入列表（文本、图像、音频片段、音乐）通过 tokenizer 路由器。
- 模拟文本到语音响应的流式解码，带延迟计数。
- 给定编码器、prefill 和解码器延迟，计算预期的 time-to-first-audio-byte。

## Ship It

本课产出 `outputs/skill-any-to-any-pipeline-auditor.md`。给定一个对话产品规格（输入模态、输出模态、延迟目标），审计 MIO 家族的设计选择并计算延迟预算。

## 练习

1. 你的产品接受语音输入并返回语音输出。端到端延迟预算目标是什么？列出花费时间的组件。

2. SpeechTokenizer residual-VQ 使用 8 个 codebook。提出为什么并行解码残差层是必要的（vs 顺序）以及它带来什么延迟节省。

3. 你的词表有 32k 文本 + 4k 图像 + 4k 语音。添加 8k 音乐和约 10 个分隔符。在 hidden dim 4096 下，embedding 矩阵的参数成本是多少？

4. Chain-of-visual-thought 输出中间图像。什么类型的问题受益？什么类型的问题被额外 token 拖累？

5. 阅读 Moshi (arXiv:2410.00037)。描述其"inner monologue"技术并与 MIO 的 chain-of-visual-thought 对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Any-to-any | "多模态输入/输出" | 单一模型在任意方向接受和输出文本、图像、语音和音乐 |
| Residual-VQ | "语音 tokenizer 栈" | 多 codebook tokenization，每层添加信息；基础层是内容，后续层是韵律 |
| SEED-Tokenizer | "图像编码" | MIO 使用的 4096 条目 codebook 离散图像 tokenizer |
| Chain-of-visual-thought | "视觉草稿纸" | 模型在最终答案前生成中间图像作为推理步骤 |
| Time-to-first-audio-byte | "TTFAB" | 从用户语音到第一个音频输出的延迟；<500ms 才有对话感 |
| 四阶段课程 | "训练配方" | 对齐 -> 交错 -> 语音增强 -> SFT，按此顺序 |

## 延伸阅读

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)
