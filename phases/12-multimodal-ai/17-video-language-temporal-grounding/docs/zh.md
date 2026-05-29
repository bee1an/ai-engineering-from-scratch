# 视频语言模型：Temporal Tokens 与 Grounding

> 视频不是一叠照片。一个 5 秒的片段有因果顺序、动作动词和事件时序，这些是图像模型无法表示的。Video-LLaMA（Zhang et al.，2023 年 6 月）发布了第一个带音视频 grounding 的开源视频-LLM。VideoChat 和 Video-LLaVA 扩展了这个模式。到 2025 年 Qwen2.5-VL 的 TMRoPE 缩小了与前沿专有模型的差距。每个系统以不同方式解决 temporal tokens——Q-former per clip、concat-pool per frame、TMRoPE per token。本课阅读这些模式，构建均匀 vs 动态帧采样器，并在 temporal grounding 任务上评估。

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 minutes

## 学习目标

- 解释为什么时间位置编码独立于视觉编码器改变视频 VLM 性能。
- 对比均匀、动态 FPS 和事件驱动帧采样在 tokens-per-second vs grounding 精度上的表现。
- 描述 Q-former-per-clip（Video-LLaMA）vs pooled-per-frame（Video-LLaVA）vs M-RoPE-per-token（Qwen2.5-VL）设计。
- 说出四个视频 benchmark：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题

一分钟视频在 30 FPS 下是 1800 帧。每帧 196 个视觉 token（ViT-B at 224），就是 352k token——超过任何 2024 年代 LLM 的上下文。

三种缩减策略：

1. 子采样帧（根据内容 1-8 FPS）。
2. 激进池化每帧的 patch token（3x3 或 4x4 双线性池化）。
3. 通过 Q-former 压缩，取 16 帧 clip 输出 64 个 token。

每种权衡不同。子采样丢失时间细节。池化丢失空间细节。Q-former 两者都丢一点但节省 token。

时间位置编码是另一个轴：模型怎么知道第 5 帧在第 6 帧之前？选项包括简单的 1D temporal RoPE（Video-LLaMA）、学习的时间嵌入（Video-LLaVA）和 TMRoPE（Qwen2.5-VL，完整 3D）。

## 核心概念

### Video-LLaMA：Q-former per clip + 音频分支

Video-LLaMA (2023) 是第一个开源视频-LLM。架构：

- 16 帧 clip，2 FPS（即 8 秒）。
- 每帧 ViT 特征 -> Video Q-former 对所有 16 帧做 cross-attention -> 32 个学习查询 -> LLM。
- 并行音频分支：波形 -> ImageBind 音频编码器 -> Audio Q-former -> 32 个查询 -> LLM。

优势：音视频联合推理。劣势：固定 clip 长度，无法任意时间 grounding。

### VideoChat 和 Video-LLaVA

VideoChat 保留了 Video-LLaMA 的思路但去掉了音频并简化。Video-LLaVA（Lin et al.，2023）在图像和视频帧上训练单一视觉编码器（"alignment before projection"），给出统一表示。两者都是 frozen-CLIP-encoder + MLP + LLM。

两者都不处理长视频。都是 8-16 帧系统。

### Qwen2.5-VL 和 TMRoPE

Qwen2.5-VL 引入了 TMRoPE——Temporal-Modality Rotary Position Embedding。每个 patch token 携带一个 (t, h, w) 位置，其中 t 是实际时间戳（不是帧索引）。

与简单时间嵌入的关键差异：

- 绝对时间，不是索引。模型看到"在 4.2 秒"而不是"在第 15 帧"。
- Per-token 旋转，不是 per-clip。每个视觉 token 根据其时间戳独立旋转。
- 兼容动态 FPS。如果这里采样 2 FPS 那里采样 4 FPS，TMRoPE 原生处理不均匀间距。

TMRoPE 使"猫在第几秒跳起来？"这类查询成为可能。模型可以输出"在 4.2 秒"。Video-LLaMA 只能说"在片段早期"。

### 帧采样策略

均匀：在时长上均匀采样 N 帧。简单，丢失运动峰值。

动态 FPS：根据运动强度自适应采样。光流或帧差分选择高运动段进行更密集采样。Qwen2.5-VL 在此上训练。

事件驱动：运行轻量检测器，在动作发生处采样更多。VideoAgent 使用。

关键帧 + 上下文：在镜头边界采样 + 几个相邻帧。用于电影内容。

### 每帧池化

在 1 FPS 和每帧 576 token 下，5 分钟 clip 是 172,800 token。Qwen2.5-VL-72B 的 128k 上下文可以做到但昂贵。

3x3 双线性池化减少到每帧 64 token -> 5 分钟 19,200 token。大多数任务的甜蜜点。

更激进的池化（6x6 -> 每帧 16 token）用于空间细节不太重要的 agent 工作流。

### 四个视频 benchmark

- VideoMME：综合视频理解，短 + 中 + 长。
- TempCompass：细粒度时间推理，"之前"/"之后"问题。
- EgoSchema：长时间第一人称视频。
- Video-MMMU：多模态多学科视频问题。

完整的视频-VLM 评估要覆盖全部四个。它们压力测试不同轴——TempCompass 全是排序，EgoSchema 是 3+ 分钟推理，VideoMME 跨时长。

### Grounding 输出格式

Temporal grounding 的输出格式：

- 自由文本："猫大约在第 4 秒跳起来。"容易解析但不精确。
- 结构化 JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 训练这个。
- 基于 token：特殊的 `<time>4.1</time>` token 穿插在答案中。Qwen2.5-VL 的内部格式。

基于 token 对下游使用最准确。Qwen2.5-VL 的 JSON 输出格式可以直接解析。

### 2026 最佳实践

2026 年视频 VLM 的最佳实践：

- 编码器：SigLIP 2 配 M-RoPE 或 TMRoPE（Qwen2.5-VL）。
- 帧采样：动态 FPS（根据运动 1-4）带最大帧数上限。
- 每帧池化：3x3 双线性。
- 输出：带时间 + 事件字段的结构化 JSON。
- Benchmark：VideoMME + TempCompass 用于通用；EgoSchema 用于长时间。

## Use It

`code/main.py` 包含：

- 均匀和动态 FPS 帧采样器。
- 一个玩具 temporal-grounding 评估器：给定时间 T 的"ground truth"事件和模型输出，带容差评分精度。
- 跨 Video-LLaMA（16 帧，Q-former）、Video-LLaVA（8 帧，MLP）、Qwen2.5-VL（动态 FPS + TMRoPE）的对比。

## Ship It

本课产出 `outputs/skill-video-vlm-frame-planner.md`。给定一个视频任务（监控、动作识别、temporal grounding、摘要），选择帧采样器、池化因子、输出格式和预期精度层级。

## 练习

1. 对于一个 3 分钟的烹饪演示，选择均匀 vs 动态 FPS。用 token 数量证明。

2. TMRoPE 具体添加了什么是简单时间嵌入表做不到的？

3. 写一个 VLM 可以学会输出的 temporal grounding JSON schema。包含错误情况。

4. 阅读 Video-LLaVA 的 Section 3 关于"Alignment Before Projection"。为什么这比训练独立的图像和视频编码器更好？

5. 根据 VideoMME 排行榜，2026 年顶级开源模型和顶级专有模型之间的差距是多少？这个差距中有多少归因于时间编码 vs 基础 LLM 规模？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Temporal grounding | "时间定位答案" | VLM 输出事件发生的具体时间戳范围 |
| TMRoPE | "Time-Multimodal RoPE" | 带绝对时间戳的 3D 旋转位置编码，Qwen2.5-VL 使用 |
| Dynamic FPS | "运动感知采样" | 高运动段采样更多帧，静态段更少 |
| Frame pooling | "每帧空间压缩" | 在 LLM 之前用双线性插值减少每帧的 patch 数 |
| Video Q-former | "Clip 压缩器" | 将 N 帧映射到 K 个学习查询的 cross-attention 瓶颈 |
| VideoMME | "视频 bench" | 综合短/中/长视频 benchmark，2500+ 样本 |

## 延伸阅读

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
