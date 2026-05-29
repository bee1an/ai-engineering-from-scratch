# 视频生成

> 图像是 2-D 张量。视频是 3-D 的。理论相同；计算难 10-100 倍。OpenAI 的 Sora（2024 年 2 月）证明了可行性。到 2026 年 Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 以 1080p 从文本生成生产级视频——开源权重栈（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）落后 12 个月。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## 问题

一个 10 秒 1080p 24fps 视频是 240 帧 1920×1080×3 像素。约 1.5 GB 原始数据每个片段。像素空间扩散不可行。你需要：

1. **时空压缩。** 一个编码视频（而非帧）为时空 patch 序列的 VAE。
2. **时间连贯性。** 帧需要在数秒内共享内容、光照和物体身份。网络必须建模运动。
3. **计算预算。** 视频训练比相同模型大小的图像贵 10-100 倍。
4. **条件化。** 文本、图像（首帧）、音频或另一个视频。大多数生产模型接受全部四种。

解决这个问题的架构是应用于时空 patch 的 **Diffusion Transformer (DiT)**，在大规模 (prompt, caption, video) 数据集上训练。与第 06 课相同的扩散损失。

## 概念

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### Patchify

用 3D VAE（学习的时空压缩）编码视频。Latent 形状为 `[T_latent, H_latent, W_latent, C_latent]`。分割为大小 `[t_p, h_p, w_p]` 的 patch。对于 Sora 式模型，`t_p = 1`（逐帧 patch）或 `t_p = 2`（每两帧）。10 秒 1080p 视频压缩为约 20,000-100,000 个 patch。

### 时空 DiT

Transformer 处理 patch 的扁平序列。每个 patch 有 3D 位置嵌入（时间 + y + x）。注意力通常是分解的：

- **空间注意力** 在每帧的 patch 内。
- **时间注意力** 跨帧在相同空间位置。
- **完整 3D 注意力** 贵 16-100 倍；仅在低分辨率或研究中使用。

### 文本条件化

与大型文本编码器的交叉注意力（Sora 用 T5-XXL，CogVideoX-5B 用 T5-XXL）。长 prompt 很重要——Sora 的训练集有 GPT 生成的密集重描述，平均每个片段 200 token。

### 训练

标准扩散损失（ε 或 v prediction）在时空 latent 上。数据：网络视频 + 约 1 亿精选片段 + 合成文本描述。计算：即使小型研究运行也需要 10,000+ GPU 小时；Sora 规模是 100,000+。

## 2026 年生产格局

| Model | Date | Max duration | Max res | Open weights? | Notable |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | No | 第一个展示大规模世界模拟器属性的模型 |
| Sora Turbo | 2024-12 | 20s | 1080p | No | 生产 Sora，推理快 5 倍 |
| Veo 2 (Google) | 2024-12 | 8s | 4K | No | 2025 年最高质量 + 物理 |
| Veo 3 | 2025 Q3 | 15s | 4K | No | 原生音频和更强相机控制 |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | No | 2025 Q1 最佳人体运动 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | No | 专业视频工具叠加 |
| Pika 2.0 | 2024-10 | 5s | 1080p | No | 最强角色一致性 |
| CogVideoX (THUDM) | 2024 | 10s | 720p | Yes (2B, 5B) | 第一个开源 5B 规模视频 |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | Yes (13B) | 2024 年末开源 SOTA |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | Yes (10B) | 最宽松许可 |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | Yes | 2025 年中最强开源模型 |

开源权重比图像领域更快地缩小差距：HunyuanVideo + WAN 2.2 LoRA 到 2026 年中已驱动大多数开源工作流。

## Build It

`code/main.py` 模拟核心时空 DiT 思想：将小型合成视频 patchify，添加逐 patch 位置嵌入，用 transformer 式注意力在 patch 上去噪整个序列。无 numpy；纯 Python。我们展示即使在 1-D 中，当相邻帧 patch 共享去噪器和位置嵌入时，时间连贯性也会涌现。

### Step 1: patchify a synthetic 1-D "video"

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Step 2: position embedding per frame

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Step 3: denoiser sees the whole sequence

不是独立去噪每帧，我们的小网络拼接所有帧值 + 它们的位置嵌入，联合预测所有帧的噪声。

### Step 4: temporal coherence test

训练后，采样一个视频。测量帧间 delta。如果模型学到了时间结构，delta 比独立采样每帧时更小。

## Pitfalls

- **独立逐帧采样 = 闪烁。** 如果你对每帧独立运行图像扩散，输出会闪烁因为每帧的噪声是独立的。视频扩散通过注意力或共享噪声耦合帧来修复。
- **朴素 3D 注意力 = OOM。** 10 秒 1080p latent 上的完整 3D 注意力是数千亿次操作。分解为空间 + 时间。
- **数据描述比规模更重要。** Sora 相对先前工作的主要升级是在约 10 倍更详细的描述上训练（GPT-4 重新标注片段）。OpenAI 的技术报告对此很明确。
- **首帧条件化。** 大多数生产模型也接受图像作为首帧。这是"图像到视频"模式；训练包含此变体。
- **物理漂移。** 长片段（>10s）累积微妙的不一致。滑动窗口生成 + 关键帧锚定有帮助。

## Use It

| Use case | 2026 pick |
|----------|-----------|
| 最高质量文本到视频，托管 | Veo 3 或 Sora |
| 相机控制的电影级 | Runway Gen-3 with motion brushes |
| 跨片段角色一致性 | Pika 2.0 或 Kling 2.1 |
| 开源权重，快速微调 | WAN 2.2 + LoRA |
| 图像到视频 | WAN 2.2-I2V, Kling 2.1 I2V, 或 Runway |
| 音频到视频唇同步 | Veo 3（原生音频）或专用唇同步模型 |
| 视频编辑 | Runway Act-Two, Kling Motion Brush, Flux-Kontext（静帧） |

2024 到 2026 年间，同等质量每秒视频的成本下降了 20 倍。

## Ship It

保存 `outputs/skill-video-brief.md`。Skill 接收视频简报（时长、宽高比、风格、相机计划、主体一致性、音频），输出：模型 + 托管、prompt 脚手架（相机语言、主体描述、运动描述符）、种子 + 可复现协议和帧级 QA 清单。

## Exercises

1. **Easy.** 在 `code/main.py` 中，比较 (a) 独立逐帧采样和 (b) 联合序列采样的帧间 delta。报告 delta 的均值和方差。
2. **Medium.** 添加首帧条件：将帧 0 固定为给定值并采样其余。测量固定值如何传播。
3. **Hard.** 用 HuggingFace diffusers 在本地 GPU 上运行 CogVideoX-2B。对 6 秒片段在 720p 计时 20 推理步。分析时空注意力以识别瓶颈。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Video VAE | "3-D VAE" | 将 `(T, H, W, C)` 压缩为时空 latent 的编码器。 |
| Patches | "Token" | latent 的固定大小 3-D 块；DiT 的输入。 |
| 分解注意力 | "空间 + 时间" | 先在空间上做注意力，再在时间上；跳过完整 3-D 注意力。 |
| Image-to-video (I2V) | "让这张照片动起来" | 模型接收图像 + 文本，输出从该图像开始的视频。 |
| 关键帧条件化 | "锚定帧" | 固定特定帧以控制视频的弧线。 |
| Motion brush | "方向提示" | 用户在图像上绘制运动向量的 UI 输入。 |
| Re-captioning | "密集描述" | 使用 LLM 用详细 prompt 重新标注训练片段。 |
| 闪烁 | "时间伪影" | 帧间不一致；通过耦合去噪修复。 |

## 生产笔记：视频 latent 是内存带宽问题

10 秒 1080p 24 fps 片段是 240 帧 × 1920 × 1080 × 3 ≈ 1.5 GB 原始像素。经过 4× video VAE 压缩（`2 × spatial × 2 × temporal`）latent 约 100 MB 每请求。在 batch 1 下通过时空 DiT 运行 30 步，你每步移动约 3 GB 通过 HBM——内存带宽而非 FLOPs 是瓶颈。

三个生产旋钮，都直接来自生产推理文献：

- **DiT 上的 TP。** 文本到视频模型通常 ≥10B 参数。4 个 H100 上 TP=4 是标准；405B 级模型用 PP=2 × TP=2。每步延迟随 TP 大致线性下降直到 all-reduce 墙。
- **帧 batching = continuous batching。** 生成时，视频概念上是通过注意力链接的帧 batch。Continuous batching（在途调度）适用：如果模型架构允许滑动窗口生成，在帧 `t-1` 返回时开始渲染帧 `t+1`。
- **片段级 prefill cache。** 对于图像到视频，首帧条件化类似于 LLM 的 prompt prefill：计算一次，跨时间解码器 pass 重用。这实际上是视频的 KV-cache。

## Further Reading

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora technical report.
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX.
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo.
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1.
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — open SOTA mid-2025.
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — the seminal video diffusion paper.
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion's ancestor.
