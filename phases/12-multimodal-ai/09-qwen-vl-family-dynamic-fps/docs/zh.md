# Qwen-VL 家族与动态 FPS 视频

> Qwen-VL 家族——Qwen-VL（2023）、Qwen2-VL（2024）、Qwen2.5-VL（2025）、Qwen3-VL（2025）——是 2026 年最具影响力的开源视觉语言模型血统。每一代都做了一个决定性的架构押注，其余开源生态在十二个月内跟进：通过 M-RoPE 实现原生动态分辨率、带绝对时间对齐的动态 FPS 采样、ViT 中的窗口注意力、以及结构化 agent 输出格式。到 Qwen3-VL，方案已经稳定：一个带原生宽高比输入的 2D-RoPE-ViT 编码器，一个 MLP projector 接入大型 Qwen3 语言基座，以及将 OCR、定位和 agent 行为作为一等目标的训练阶段。本课按时间顺序解读这个家族，让你理解每个旋钮为什么在那个位置。

**类型：** 学习
**语言：** Python（标准库，M-RoPE 编码器 + 动态 FPS 采样器）
**前置：** Phase 12 · 06（patch-n'-pack）
**时间：** ~120 分钟

## 学习目标

- 计算 M-RoPE 的三轴旋转（temporal、height、width）并解释为什么三个都需要。
- 为视频选择动态 FPS 采样策略，并推理 tokens-per-second vs 事件检测准确率。
- 按顺序说出 Qwen-VL 四代升级及各自启用了什么。
- 接入 Qwen2.5-VL 风格的 JSON agent 输出格式，并从 VLM 响应中解析结构化工具调用。

## 问题

Qwen-VL 于 2023 年 8 月发布，直接回应 LLaVA-1.5 和 BLIP-2。Qwen 团队瞄准的差距有三个：分辨率、视频和结构化输出。

分辨率：LLaVA-1.5 跑在 336x336。对照片够用，对中文发票或密集电子表格截图没用。Qwen-VL 的第一个创新是 448x448 加定位边界框输出，让模型能指向东西。

视频：Video-LLaMA 堆叠逐帧编码器送入 LLM。对短片段有效，对时间轴是信号的多分钟视频无效。Qwen 团队想要一个理解时间的单一编码器。

结构化输出：LLaVA 输出自由文本。Agent 需要 JSON。Qwen-VL 在显式 JSON 输出格式上训练，包括边界框坐标作为文本。

每一代 Qwen-VL 都扩展这三个轴之一。

## 概念

### Qwen-VL（2023 年 8 月）

第一代：OpenCLIP ViT-bigG/14 作为编码器（2.5B 参数），LLama 兼容的 Q-Former（1 步 256 query），Qwen-7B 基座。贡献：

- 448x448 分辨率（当时开源 VLM 的 SOTA）。
- 定位：在带显式坐标 token 输出的图文对上训练。"The cat is at <box>(112, 204), (280, 344)</box>"。
- 从一开始就是中英双语训练。

当时的基准：英语上与 GPT-4V 竞争，中文上占主导。定位监督是真正的头条。

### Qwen2-VL（2024 年 9 月）——M-RoPE 与原生分辨率

Qwen2-VL 用原生动态分辨率 ViT 编码器替换了固定分辨率 + Q-Former 栈。关键变化：

- 原生动态分辨率。ViT 接受任何能被 28 整除的 HxW（patch 14 加 2x 空间合并）。1120x672 的图像（40x24 合并 patch）产生 960 个视觉 token。无 resize，无 tiling，无缩略图。
- M-RoPE（Multimodal RoPE）。每个 token 携带 3D 位置 (t, h, w) 而非 1D。图像 t=0，视频 t = frame_index。RoPE 按每轴频率旋转 query/key 向量。无位置编码表。
- MLP projector。丢弃 Q-Former；在合并的 patch token 上用 2 层 MLP。
- 动态 FPS 视频。视频默认以 1-2 FPS 采样，但模型接受任意帧数。

结果：Qwen2-VL-7B 在多个多模态基准上匹配 GPT-4o，在 DocVQA 上超过它（94.5 vs 88.4）。架构变化是决定性的一步。

### Qwen2.5-VL（2025 年 2 月）——动态 FPS + 绝对时间

Qwen2.5-VL 的大转变是视频。动态 FPS 不只是"需要时采更多帧"。论文形式化了：

- 绝对时间 token。不用位置索引（帧 0、1、2...），而用实际时间戳。"At 0:04, the cat jumps." 模型看到 `<time>0.04</time>` token 穿插在帧 token 之间。
- 动态 FPS。慢镜头 1 FPS 采样，动作场景 4+ FPS。用户或训练者选择；M-RoPE 适应。
- ViT 中的窗口注意力。空间注意力是窗口化的（块内局部）以提高吞吐量；每隔几层做全局注意力。
- 显式 JSON 输出格式。在工具调用数据上训练："{\"tool\": \"click\", \"coords\": [380, 220]}"。开箱即用的 agent 能力。
- MRoPE-v2 缩放。位置随最大输入大小缩放，使 10 分钟视频不会耗尽频率范围。

基准：Qwen2.5-VL-72B 在大多数视频基准上超过 GPT-4o，在文档上匹配 Gemini 2.0，并在 GUI 定位上创下开源模型 SOTA（ScreenSpot：84% 准确率 vs GPT-4o 的 38%）。

### Qwen3-VL（2025 年 11 月）

Qwen3-VL 是巩固而非重新发明的增量升级：更大的 LLM 骨干（Qwen3-72B）、扩展的训练数据、改进的 OCR、通过 Qwen3"思考模式"增强的推理。ViT 和 M-RoPE 保持不变。论文聚焦于数据和训练改进而非架构。

血统要点：到 2025 年 Qwen-VL 架构已经稳定。后续代际扩展计算和数据，而非原语。

### M-RoPE 数学

经典 RoPE 用位置 `m` 旋转维度为 `d` 的 query `q`，使用配对坐标：

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE 将隐藏维度分成三个频带。比如 `d = 96`。分配 32 维给 temporal，32 给 height，32 给 width。每个频带按自己的轴位置旋转。位于 (t=5, h=10, w=20) 的 patch 在其三个频带上分别获得旋转 `R_t(5)`、`R_h(10)`、`R_w(20)`。

文本 token 使用 `t = text_index, h = 0, w = 0`（或归一化选择），保持兼容性。视频帧使用 `t = frame_time, h = row, w = col`。单张图像使用 `t = 0`。

好处：一种位置编码处理文本、图像和视频，无需分支代码或不同的位置表。

### 动态 FPS 采样逻辑

给定时长 `T` 秒的视频和目标 token 预算 `B`：

1. 计算你能负担的最大 FPS：`fps_max = B / (T * tokens_per_frame)`。
2. 从 `{1, 2, 4, 8}` 中选一个满足 `fps <= fps_max` 的目标 FPS。
3. 如果运动剧烈（光流启发式或用户显式请求），选更高 FPS。如果运动平缓，选更低。
4. 以选定 FPS 均匀采样；在帧之间插入 `<time>t</time>` token。

Qwen2.5-VL 隐式训练了这个逻辑；推理时用户通过 `fps` 参数控制。60 秒动作序列在 4 FPS、每帧 81 token = 19440 token，在 32k 上下文中可管理。

### 结构化 agent 输出

Qwen2.5-VL 的 agent 训练显式针对结构化工具调用：

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是确定性的：对模型输出做 JSON.parse。对比自由文本 "click at (1024, 512)" 需要正则和歧义处理。这个转变是 Qwen2.5-VL 的 ScreenSpot 分数从 Qwen2-VL 的 55% 跳到 84% 的原因。

## 动手用

`code/main.py` 实现了：

- 为混合文本、图像 patch 和视频帧的打包序列计算 M-RoPE 位置。
- 动态 FPS 采样器：给定（时长、预算、运动级别），选择 FPS 并输出帧时间戳。
- 一个玩具 Qwen2.5-VL JSON 输出解析器，处理带坐标字段的工具调用响应。

运行它，然后感受在 5 分钟视频上从固定 FPS 换到动态 FPS 的差异。

## 交付物

本课产出 `outputs/skill-qwen-vl-pipeline-designer.md`。给定视频任务（监控、agent、动作识别、无障碍），它输出 Qwen2.5-VL 配置（帧预算、FPS 策略、窗口注意力标志、agent 输出模式）和延迟估算。每当你为视频产品部署 Qwen-VL 家族模型时使用。

## 练习

1. 计算位于 (t=3, h=5, w=7) 的 patch 的 M-RoPE 旋转，hidden 48（每频带 16，base theta 10000）。展示每个频带前三对的旋转角度。

2. 10 分钟安防摄像头录像在 1 FPS 下产生多少帧？在 384 分辨率加 3x 池化下，总共多少 token？Qwen2.5-VL 默认的 32k 上下文能处理吗？

3. 为 30 秒网球对打 vs 30 秒食谱演示 vs 30 秒 UI-agent 录制选择 FPS。用动态 FPS 逻辑证明每个选择。

4. Qwen2.5-VL 完全丢弃了 Q-Former。为什么简单 MLP 在 2025 年有效但 2023 年不行？（提示：数据规模和编码器质量。）

5. 将三个 Qwen2.5-VL JSON 工具调用输出解析为 Python dict。格式错误的 JSON 会怎样失败？Qwen cookbook 推荐什么恢复策略？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| M-RoPE | "Multimodal RoPE" | 隐藏维度中带 temporal、height 和 width 频带的 3D 旋转位置编码 |
| Dynamic FPS | "智能采样" | 根据运动、时长和 token 预算为每个视频选择的帧采样率 |
| Absolute time token | "时间戳 token" | 序列中穿插的 `<time>t</time>`，使模型看到实际秒数而非帧索引 |
| Window attention | "局部注意力" | 限制在小窗口内的空间 self-attention 以提速；周期性添加全局注意力 |
| Structured agent output | "JSON 模式" | 训练数据监督教 VLM 输出可解析的 JSON，带坐标和工具名 |
| min_pixels / max_pixels | "分辨率边界" | Qwen2.5-VL 的每请求控制，限制总像素数从而限制 token 数 |
| Grounding | "指向它" | 将边界框坐标作为文本 token 输出；自 Qwen-VL v1 起使用 |

## 延伸阅读

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
