# Omni 模型：Qwen2.5-Omni 与 Thinker-Talker 拆分

> GPT-4o 在 2024 年 5 月的产品演示之所以颠覆性，不是因为底层模型，而是因为产品形态——一个语音界面，你说话，模型看到摄像头看到的，然后在 250ms 内回话。开源生态在 2024 和 2025 年的剩余时间里竞相达到这个产品表面。Qwen2.5-Omni（2025 年 3 月）是参考开源设计：一个 Thinker（大型文本生成 transformer）加一个 Talker（并行语音生成 transformer），通过流式语音 token 连接。Mini-Omni 简化了它，Moshi 匹配了它的延迟，GLM-4-Voice 将其扩展到中文。本课阅读 Thinker-Talker 架构和使流式实时对话工作的延迟预算。

**Type:** Build
**Languages:** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**Prerequisites:** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**Time:** ~180 minutes

## 学习目标

- 将推理流水线拆分为 Thinker（文本推理）和 Talker（语音合成），解释为什么并行流式有效。
- 逐组件计算对话交互的 time-to-first-audio-byte (TTFAB) 预算。
- 描述 TMRoPE 在 Thinker 内跨视觉、音频和文本的时间对齐位置编码。
- 说出三种实时对话模式：半双工、轮流、全双工。

## 问题

实时语音助手需要做很多事，而且要快：

1. 听用户说话。实时语音 tokenization，语音活动检测（VAD）知道他们何时说完。
2. 可选地看。摄像头输入 2-4 FPS，与音频一起流入 Thinker。
3. 思考。根据对话历史组合响应。
4. 说话。合成音频 token，解码为波形，流式传输到用户扬声器。

每一步都增加延迟。对话感需要总往返 < 500ms——低于此用户不再注意到延迟。GPT-4o 声称约 250ms。Moshi 约 160ms。Qwen2.5-Omni 约 350-500ms。

每个组件都需要流式。没有什么可以"批处理所有然后解码"。

## 核心概念

### Thinker 和 Talker

Qwen2.5-Omni 的分解：

- Thinker：7B-80B 文本生成 transformer。消费交错的文本 + 图像 + 音频 token。输出代表要说什么的文本 token。
- Talker：较小的语音生成 transformer（200M-1B）。消费 Thinker 的文本输出 token 加最近的语音上下文 token。输出离散语音 token（residual-VQ 索引）。
- 语音解码器：流式波形解码器（SNAC、MoVQGAN 家族），实时将语音 token 转为音频样本。

这种分离很重要。Thinker 必须大才能有好的推理。Talker 可以小因为它的工作是局部的——将文本转为语音 token。更大的 Talker 不会更有表现力；只会更慢。

并行运行两者：

1. Thinker 输出文本 token t_i。
2. Talker 消费 t_i（通过流式）并输出语音 token s_i, s_{i+1}, ..., s_{i+k}。
3. 语音解码器在语音 token 到来时消费它们并输出音频样本。
4. 当 Thinker 到达文本 token t_{i+3} 时，Talker 已经为 t_0..t_{i+2} 流式输出了音频。

### TMRoPE——时间对齐的多模态位置

Thinker 需要整合图像帧（比如 4 FPS 到达）、音频帧（50 帧/秒到达）和对话历史中的文本。朴素的序列顺序（所有图像，然后所有音频，然后文本）丢失了时间对齐。

TMRoPE 为每个 token 分配绝对时间戳。视觉 token 在 t=2.3s。音频 token 在 t=2.32s。用户的文本 token "停"在 t=2.35s。RoPE 按时间戳旋转注意力；模型将它们视为时间上并发的。

这是让"他一边挥手一边说你好"能工作的基础设施——模型在同一个概念时刻看到视频帧和音频。

### 流式语音合成

语音 token 必须流式。Mini-Omni（Xie & Wu，2024）引入了"语言模型可以在流式中听、说、同时思考"：Thinker 输出 token 和 Talker 输出 token 在同一序列中交错。Talker 在 Thinker 提交下一个文本 token 后立即触发。没有批处理边界。

Moshi（Défossez et al.，2024 年 10 月）是最快的开源实现。单 A100 上 160ms TTFAB。架构：单一 7B transformer 在交替位置输出文本和语音 token，带"inner monologue"将思考流与说话流分离。这实际上是 Thinker + Talker 融合为一个模型并精心训练。

### VAD 和轮流

语音活动检测在输入侧运行。两种模式：

- 半双工：用户说话，模型听。模型说话，用户听。通过 VAD 静音检测（约 200ms）清晰交接。
- 全双工：双方可以同时说话。模型可以反馈（"嗯嗯"）或打断。难得多。Moshi 支持这个。

Qwen2.5-Omni 默认支持半双工，通过静音阈值轮流。全双工需要应用层处理。

### Qwen3-Omni（2025 年 11 月）

后继者。Qwen3-80B Thinker，更大的 Talker，改进的 TMRoPE-v2。延迟接近 GPT-4o 的 250ms。开源权重。OmniBench 上的 benchmark 与 Gemini 2.0 Live 有竞争力。

### 生产延迟预算

典型流式交互：

- 麦克风 -> 音频 token：40-80ms。
- Prefill（prompt + 历史）：7B 下 100-200ms，70B 下多得多。
- 第一个 Thinker 文本 token：40ms。
- Talker 处理第一个文本 token：20ms。
- 第一个语音 token 提交：40ms。
- Residual-VQ 解码：30ms。
- 语音波形解码：50-80ms。

总 TTFAB：7B 下 320-510ms，70B 下 600-900ms。前沿质量通常意味着 70B+；因此有前沿延迟差距。

### Token 速率数学

16kHz 语音在 50 Hz 基础语音 token 下，你需要每秒 50 个语音 token 的输出。Talker 必须输出 ≥50 tok/s 才能跟上。在 H100 上典型 LLM 吞吐量 30-80 tok/s 下，小型（200-300M）Talker 足够快；7B Talker 会跟不上。

这就是为什么存在小型专用 Talker 模型而不是"直接用主模型"。

## Use It

`code/main.py`：

- 用 mock token 输出速率模拟 Thinker-Talker 流水线。
- 为可配置的模型大小和麦克风采样率计算 TTFAB。
- 演示带 VAD 静音阈值的半双工轮流。

## Ship It

本课产出 `outputs/skill-omni-streaming-budget.md`。给定实时语音产品的目标 TTFAB 和功能集（视觉输入、双语、全双工），选择 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni 并确定 Thinker/Talker 大小。

## 练习

1. 你的目标 TTFAB 是 300ms。在 7B Thinker 和 300M Talker 上，写出每个组件的延迟。

2. Qwen2.5-Omni 使用 TMRoPE。描述当用户在 t=1s 开始说话、摄像头在 t=1.2s 捕捉到一个手势时，模型看到什么。

3. 全双工支持要求模型在听的同时输出音频。提出一种教会这个的训练数据格式。

4. 阅读 Moshi 论文 Section 4。描述"inner monologue"分离以及为什么它避免了 Thinker-Talker 拆分。

5. 计算吞吐量预算：Talker 必须多快输出 token 才能跟上 16kHz 语音在 50 基础层 token/sec 下的速度？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Thinker | "推理大脑" | 产出要说什么的大型文本生成 transformer |
| Talker | "语音生成嘴巴" | 从 Thinker 的文本产出离散语音 token 的小型 transformer |
| TTFAB | "延迟预算" | Time-to-first-audio-byte：从用户语音结束到第一个音频样本输出 |
| TMRoPE | "时间对齐 RoPE" | 跨视觉、音频、文本使用绝对时间戳的位置编码 |
| 半双工 | "轮流" | 用户和模型交替；VAD 静音检测用户说完 |
| 全双工 | "同时" | 模型可以同时说和听；能做反馈 |
| Inner monologue | "Moshi 分离" | 单模型设计，思考流和说话流交错 |

## 延伸阅读

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
