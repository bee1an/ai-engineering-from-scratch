# 音频语言模型：从 Whisper 到 Audio Flamingo 3 的演进

> Whisper（Radford et al.，2022 年 12 月）解决了语音识别——680k 小时弱监督多语言语音，一个简单的 encoder-decoder transformer，一个让此后每个 ASR 发布都引用它的 benchmark。但识别不是推理。问"这段录音里有什么乐器"或"说话人表达了什么情绪"或"第 3 分钟发生了什么"需要音频理解，而非转录。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3，2025 年 7 月）逐步构建了这个栈：保留 Whisper 级编码器，接上 Q-former，在音频-文本指令数据上训练，添加 chain-of-thought 推理。本课走过这段演进。

**Type:** Build
**Languages:** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Prerequisites:** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**Time:** ~180 minutes

## 学习目标

- 从波形计算 log-Mel 频谱图：加窗、FFT、滤波器组、对数变换。
- 对比编码器选项：Whisper encoder、BEATs、AF-Whisper 混合。各自何时胜出。
- 构建一个 audio Q-former：N 个可学习查询对频谱图 patch 做 cross-attention。
- 解释级联（Whisper-then-LLM）vs 端到端 audio-LLM 训练：为什么端到端在推理上扩展更好。

## 问题

语音识别被 Whisper 解决了。音频的 OCR 是商品化的。但"商品化"止步于转录。如果模型不能对听到的内容进行推理——时序、说话人、情绪、音乐结构、环境声音——仅靠转录无法驱动产品功能。

三条明显路线：

1. 级联：Whisper 转录，LLM 对转录文本推理。适用于纯语音场景。对音乐、环境音频、多说话人重叠、情绪失败。

2. 端到端 audio-LLM：音频编码器将音频 token 直接喂入 LLM，跳过转录。保留声学信息（情绪、说话人、环境）。需要新的训练数据。

3. 混合：音频编码器 + 文本解码器，既能转录又能推理。Qwen-Audio 和 Audio Flamingo 选择这条路。

## 核心概念

### Log-Mel 频谱图：输入特征

每个音频编码器都从同一个特征开始：log-Mel 频谱图。

1. 重采样到 16 kHz。
2. 短时傅里叶变换，25ms 窗口，10ms 步进。
3. 取 FFT 结果的幅度。
4. 应用 Mel 滤波器组（通常 80 个滤波器，对数间隔 0-8000 Hz）将频率映射到感知尺度。
5. 对数压缩（log(1 + x)）用于动态范围。

结果：形状为 (T, 80) 的 2D 数组，T 是时间帧数。30 秒 clip 在 100 Hz 帧率下：(3000, 80)。

### Whisper 的编码器

Whisper 的编码器是一个 12 层 ViT 风格 transformer，将 log-Mel 频谱图作为时间帧序列处理。输出：每个时间帧一个 hidden-state 向量。

对于 ASR，Whisper 的解码器是一个 cross-attention transformer，根据编码器输出生成文本 token。标准 encoder-decoder。

对于 ALM（audio-LLM），你想要编码器输出作为另一个 LLM 的输入。模式：Whisper encoder 冻结，Q-former 可训练，LLM 冻结或微调。

### BEATs 和音频专属编码器

Whisper 在语音主导数据上训练。它在音乐和环境音频上较弱。

BEATs（Chen et al.，2022）是在 AudioSet 上训练的自监督 transformer。在相同参数量下比 Whisper 更好地捕获音乐和环境声音。

AF-Whisper（Audio Flamingo 3 的混合方案）：拼接 Whisper + BEATs 特征作为音频输入。Whisper 携带语言信号，BEATs 携带声学信号。

### Audio Q-former

与 BLIP-2 的 visual Q-former 相同模式。固定数量的可学习查询（通常 32 或 64）对音频编码器的输出帧做 cross-attention。查询成为 LLM 消费的音频 token。

训练对齐阶段：仅 Q-former，在音频-文本对（AudioCaps、Clotho）上用对比 + captioning loss。指令阶段：端到端，解冻 LLM，在指令数据上训练。

### 演进弧线——SALMONN、Qwen-Audio、AF3

SALMONN（Tang et al.，2023）：Whisper + BEATs + Q-former + LLaMA。第一个有严肃推理能力的开源 audio-LLM。MMAU benchmark 约 0.55 综合分。

Qwen-Audio（Chu et al.，2023）：类似架构，在更丰富的数据集上训练，为多轮对话调优。MMAU 约 0.60。

LTU——Listen, Think, Understand（Gong et al.，2023）：显式推理数据，专注于音频片段上的 chain-of-thought。更小但更聚焦。

Audio Flamingo 3（Goel et al.，2025 年 7 月）：当前开源 SOTA。8B LLM backbone（Qwen2 7B），Whisper-large encoder 拼接 BEATs，64 查询 Q-former，在 1M+ 音频-文本指令对上训练。MMAU 0.72，在某些子任务上匹配专有前沿。

AF3 还引入了按需 chain-of-thought 用于音频：模型可以选择性地在最终答案前输出思考 token（"让我先识别乐器：..."）。复杂推理任务的准确率在启用思考时提升 3-5 个点。

### 级联 vs 端到端

级联流水线：

1. Whisper 将音频转录为文本。
2. LLM 对文本推理。

对"总结这个播客"完美适用。对以下场景失败：
- "这首歌的情绪是什么？"——情绪在声音中，不在文字中。
- "谁在说话，Alice 还是 Bob？"——需要说话人识别。
- "爆炸在第几秒发生？"——temporal grounding 在文本中丢失。
- "这是真实还是生成的音频？"——深度伪造检测需要声学特征。

端到端保留声学信号。Qwen-Audio 和 AF3 原生处理音乐、环境和情绪。

### 2026 生产配方

对于新的音频理解产品：

- 级联适用于：转录是目标，无音乐，无情绪推断。
- AF3 / Qwen-Audio 家族适用于：音乐、情绪、多说话人或复杂音频推理。

级联更便宜更简单。端到端更有能力。

### MMAU——音频推理 benchmark

MMAU（Massive Multimodal Audio Understanding）是 2024-2025 年的音频推理 benchmark：

- 10,000 个音频-文本 QA 对，跨语音、音乐、环境声音。
- 覆盖分类、时间推理、因果推理、开放式 QA。
- 测试级联流水线系统性遗漏的内容。

开源 SOTA（AF3）0.72；专有前沿约 0.78（Gemini 2.5 Pro、Claude Opus 4.7）。差距比 VideoMME 的开源-闭源差距更小，表明 audio-LLM 正在成熟。

## Use It

`code/main.py`：

- 用 stdlib 实现 log-Mel 频谱图计算：加窗、朴素 DFT、Mel 滤波器组。
- Audio Q-former 骨架：给定编码器输出帧，计算 Q、K、V、attention，输出 N 个 token。
- 级联 vs 端到端在玩具任务上的对比。

## Ship It

本课产出 `outputs/skill-audio-llm-pipeline-picker.md`。给定一个音频任务（转录、音乐标注、情绪推断、多说话人分离、环境分类），选择级联、端到端 AF3 或混合方案。

## 练习

1. 计算 30 秒 clip 在 16kHz、25ms 窗口、10ms 步进、80 Mel bins 下的 log-Mel 频谱图维度。在 48kHz 下如何变化？

2. 为什么 Whisper 在音乐上表现不佳？BEATs 捕获了 Whisper 没有的什么音频特征？

3. Audio Q-former 64 查询 vs 32：在什么任务复杂度下 64 值得？32 为什么节省计算？

4. 阅读 AF3 Section 4 关于按需思考。提出三个 chain-of-thought 帮助最大的音频任务。

5. 用 AF3 的输出实现一个最小化的说话人分离流水线。如何标记说话人切换？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Log-Mel 频谱图 | "Mel 特征" | 经 Mel 滤波器组后的 2D（时间，频率）对数幅度值数组 |
| Audio Q-former | "Audio Perceiver" | 从音频编码器输出到固定长度查询的 cross-attention 瓶颈，喂给 LLM |
| 级联 | "ASR-then-LLM" | Whisper 转录然后文本 LLM 推理的流水线；丢失声学信息 |
| 端到端 | "Audio-LLM" | 音频特征通过 Q-former 直接进入 LLM；保留声学信号 |
| BEATs | "Audio AudioSet encoder" | 在 AudioSet 上训练的 SSL transformer；擅长音乐 + 环境声音 |
| MMAU | "音频推理 bench" | 10k QA 对，跨语音、音乐、环境；2024 评估标准 |
| 按需思考 | "Audio CoT" | 模型可选择性地在最终答案前输出推理 token，提升准确率 3-5 个点 |

## 延伸阅读

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
