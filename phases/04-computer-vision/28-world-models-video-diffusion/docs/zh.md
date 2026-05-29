# 世界模型与视频扩散

> 一个预测场景未来几秒的视频模型就是一个世界模拟器。将预测条件化在动作上，你就有了一个学到的游戏引擎。

**类型：** 学习 + 构建
**语言：** Python
**前置课程：** Phase 4 Lesson 10（Diffusion）、Phase 4 Lesson 12（视频理解）、Phase 4 Lesson 23（DiT + Rectified Flow）
**时长：** 约 75 分钟

## 学习目标

- 解释纯视频生成模型（Sora 2）与动作条件世界模型（Genie 3、DreamerV3）之间的区别
- 描述 video DiT：时空 patch、3D 位置编码、跨 (T, H, W) token 的联合注意力
- 追溯世界模型如何接入机器人：VLM 规划 → 视频模型模拟 → 逆动力学输出动作
- 根据用例（创意视频、交互式仿真、自动驾驶合成），在 Sora 2、Genie 3、Runway GWM-1 Worlds、Wan-Video 和 HunyuanVideo 之间选择

## 问题背景

视频生成和世界建模在 2026 年融合了。一个能生成连贯一分钟视频的模型，在某种意义上已经学会了世界如何运动：物体持久性、重力、因果关系、风格。如果你将预测条件化在动作上（向左走、开门），视频模型就变成了一个可学习的模拟器，可以替代游戏引擎、驾驶模拟器或机器人环境。

利害关系是具体的。Genie 3 从单张图像生成可玩环境。Runway GWM-1 Worlds 合成无限可探索场景。Sora 2 产出带同步音频和建模物理的分钟级视频。NVIDIA Cosmos-Drive、Wayve Gaia-2 和 Tesla DrivingWorld 为自动驾驶训练数据生成逼真的驾驶视频。世界模型范式正在悄然接管机器人的 sim-to-real。

本课是 Phase 4 的"全景"课。它将图像生成、视频理解和 agent 推理连接到主流研究正在走向的架构模式中。

## 核心概念

### 世界建模的三个家族

```mermaid
flowchart LR
    subgraph GEN["Pure video generation"]
        G1["Text / image prompt"] --> G2["Video DiT"] --> G3["Video frames"]
    end
    subgraph ACTION["Action-conditioned world model"]
        A1["Past frames + action"] --> A2["Latent-action video DiT"] --> A3["Next frames"]
        A3 --> A1
    end
    subgraph RL["World models for RL (DreamerV3)"]
        R1["State + action"] --> R2["Latent transition model"] --> R3["Next latent + reward"]
        R3 --> R1
    end

    style GEN fill:#dbeafe,stroke:#2563eb
    style ACTION fill:#fef3c7,stroke:#d97706
    style RL fill:#dcfce7,stroke:#16a34a
```

- **Sora 2** 是纯视频生成，条件化在 prompt 上。没有动作接口。你无法在生成过程中"操控"它。
- **Genie 3**、**GWM-1 Worlds**、**Mirage / Magica** 是动作条件世界模型。从观察到的视频推断潜在动作，然后将未来帧预测条件化在动作上。交互式——你按键或移动相机，场景就会响应。
- **DreamerV3** 和经典 RL 世界模型家族在潜在空间中预测，带显式动作条件，在奖励信号上训练。视觉性较弱；对样本高效 RL 更有用。

### Video DiT 架构

```
Video latent:          (C, T, H, W)
Patchify (spatial):    grid of P_h x P_w patches per frame
Patchify (temporal):   group P_t frames into a temporal patch
Resulting tokens:      (T / P_t) * (H / P_h) * (W / P_w) tokens
```

位置编码是 3D 的：每个 (t, h, w) 坐标一个旋转或学到的嵌入。注意力可以是：

- **完全联合** —— 所有 token 关注所有 token。O(N^2)，N 为 token 数。对长视频不可行。
- **分离式** —— 交替时间注意力（同一空间位置，跨时间：`(H*W) * T^2`）和空间注意力（同一时间步，跨空间：`T * (H*W)^2`）。TimeSformer 和大多数 video DiT 使用。
- **窗口式** —— (t, h, w) 中的局部窗口。Video Swin 使用。

2026 年每个视频扩散模型都使用这三种模式之一，加上 AdaLN 条件注入（Lesson 23）和 rectified flow。

### 动作条件：潜在动作模型

Genie 通过判别式预测连续帧对之间的动作来学习每帧的**潜在动作**。模型的解码器然后条件化在推断的潜在动作上——而非显式的键盘按键。推理时，用户可以指定一个潜在动作（或从新先验中采样），模型生成与该动作一致的下一帧。

Sora 完全跳过动作接口。其解码器从过去的时空 token 预测下一个时空 token。Prompt 条件化开始；生成过程中没有东西操控它。

### 物理合理性

Sora 2 的 2026 年发布明确宣传了**物理合理性**：重量、平衡、物体持久性、因果关系。团队通过手动评分的合理性分数来衡量；模型在掉落物体、角色碰撞和故意失败（跳跃失误）上相比 Sora 1 有明显改善。

合理性仍然是主要的失败模式。2024-2025 年人们吃意面或从杯子喝水的视频暴露了模型缺乏持久物体表示。2026 年模型（Sora 2、Runway Gen-5、HunyuanVideo）减少但未消除这些问题。

### 自动驾驶世界模型

驾驶世界模型生成条件化在轨迹、边界框或导航地图上的逼真道路场景。用途：

- **Cosmos-Drive-Dreams**（NVIDIA）—— 为 RL 训练生成分钟级驾驶视频。
- **Gaia-2**（Wayve）—— 轨迹条件场景合成，用于策略评估。
- **DrivingWorld**（Tesla）—— 模拟多种天气、时间、交通条件。
- **Vista**（ByteDance）—— 反应式驾驶场景合成。

它们替代了昂贵的真实世界数据采集，用于边角案例——行人夜间横穿马路、结冰路口、不常见车辆类型——否则需要数百万英里的驾驶。

### 机器人栈：VLM + 视频模型 + 逆动力学

新兴的三组件机器人循环：

1. **VLM** 解析目标（"pick up the red cup"），规划高层动作序列。
2. **视频生成模型** 模拟执行每个动作会是什么样子——预测未来 N 帧的观察。
3. **逆动力学模型** 提取产生这些观察的具体电机命令。

这替代了奖励塑形和样本密集的 RL。世界模型做想象；逆动力学闭环执行。Genie Envisioner 是一个实例；许多研究组正在收敛到这个结构。

### 评估

- **视觉质量** —— FVD（Fréchet Video Distance）、用户研究。
- **Prompt 对齐** —— 逐帧 CLIPScore、VQA 风格评估。
- **物理合理性** —— 在 benchmark 套件上手动评分（Sora 2 的内部 benchmark、VBench）。
- **可控性**（交互式世界模型）—— 动作 → 观察一致性；能否回到之前的状态？

### 2026 年模型格局

| 模型 | 用途 | 参数量 | 输出 | 许可证 |
|------|------|--------|------|--------|
| Sora 2 | text-to-video, audio | — | 1 分钟 1080p + 音频 | API only |
| Runway Gen-5 | text/image-to-video | — | 10s 片段 | API |
| Runway GWM-1 Worlds | interactive world | — | 无限 3D rollout | API |
| Genie 3 | 从图像生成交互世界 | 11B+ | 可玩帧 | research preview |
| Wan-Video 2.1 | 开放 text-to-video | 14B | 高质量片段 | non-commercial |
| HunyuanVideo | 开放 text-to-video | 13B | 10s 片段 | permissive |
| Cosmos / Cosmos-Drive | 自动驾驶仿真 | 7-14B | 驾驶场景 | NVIDIA open |
| Magica / Mirage 2 | AI 原生游戏引擎 | — | 可修改世界 | product |

## 动手构建

### Step 1：视频的 3D patchify

```python
import torch
import torch.nn as nn


class VideoPatch3D(nn.Module):
    def __init__(self, in_channels=4, dim=64, patch_t=2, patch_h=2, patch_w=2):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, dim,
            kernel_size=(patch_t, patch_h, patch_w),
            stride=(patch_t, patch_h, patch_w),
        )
        self.patch_t = patch_t
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, x):
        # x: (N, C, T, H, W)
        x = self.proj(x)
        n, c, t, h, w = x.shape
        tokens = x.reshape(n, c, t * h * w).transpose(1, 2)
        return tokens, (t, h, w)
```

一个 stride 等于 kernel 的 3D 卷积充当时空 patchifier。`(T, H, W) -> (T/2, H/2, W/2)` 的 token 网格。

### Step 2：3D 旋转位置编码

Rotary Position Embeddings（RoPE）分别沿 `t`、`h`、`w` 轴应用：

```python
def rope_3d(tokens, t_dim, h_dim, w_dim, grid):
    """
    tokens: (N, T*H*W, D)
    grid: (T, H, W) sizes
    t_dim + h_dim + w_dim == D
    """
    T, H, W = grid
    n, seq, d = tokens.shape
    if t_dim + h_dim + w_dim != d:
        raise ValueError(f"t_dim+h_dim+w_dim ({t_dim}+{h_dim}+{w_dim}) must equal D={d}")
    assert seq == T * H * W
    t_idx = torch.arange(T, device=tokens.device).repeat_interleave(H * W)
    h_idx = torch.arange(H, device=tokens.device).repeat_interleave(W).repeat(T)
    w_idx = torch.arange(W, device=tokens.device).repeat(T * H)
    # Simplified: just scale channels by frequencies. Real RoPE rotates pairs.
    freqs_t = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(t_dim // 2, device=tokens.device) / (t_dim // 2))
    freqs_h = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(h_dim // 2, device=tokens.device) / (h_dim // 2))
    freqs_w = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(w_dim // 2, device=tokens.device) / (w_dim // 2))
    emb_t = torch.cat([torch.sin(t_idx[:, None] * freqs_t), torch.cos(t_idx[:, None] * freqs_t)], dim=-1)
    emb_h = torch.cat([torch.sin(h_idx[:, None] * freqs_h), torch.cos(h_idx[:, None] * freqs_h)], dim=-1)
    emb_w = torch.cat([torch.sin(w_idx[:, None] * freqs_w), torch.cos(w_idx[:, None] * freqs_w)], dim=-1)
    return tokens + torch.cat([emb_t, emb_h, emb_w], dim=-1)
```

简化的加法形式。真正的 RoPE 在频率处旋转配对通道；位置信息是相同的。

### Step 3：分离式注意力 block

```python
class DividedAttentionBlock(nn.Module):
    def __init__(self, dim=64, heads=2):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.space_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, grid):
        T, H, W = grid
        n, seq, d = x.shape
        # time attention: same (h, w), across t
        xt = x.view(n, T, H * W, d).permute(0, 2, 1, 3).reshape(n * H * W, T, d)
        a, _ = self.time_attn(self.ln1(xt), self.ln1(xt), self.ln1(xt), need_weights=False)
        xt = (xt + a).reshape(n, H * W, T, d).permute(0, 2, 1, 3).reshape(n, seq, d)
        # space attention: same t, across (h, w)
        xs = xt.view(n, T, H * W, d).reshape(n * T, H * W, d)
        a, _ = self.space_attn(self.ln2(xs), self.ln2(xs), self.ln2(xs), need_weights=False)
        xs = (xs + a).reshape(n, T, H * W, d).reshape(n, seq, d)
        xs = xs + self.mlp(self.ln3(xs))
        return xs
```

时间注意力在每个空间位置内跨时间关注；空间注意力在每帧内跨位置关注。两个 O(T^2 + (HW)^2) 操作代替一个 O((THW)^2)。这是 TimeSformer 和每个现代 video DiT 的核心。

### Step 4：组合一个小型 video DiT

```python
class TinyVideoDiT(nn.Module):
    def __init__(self, in_channels=4, dim=64, depth=2, heads=2):
        super().__init__()
        self.patch = VideoPatch3D(in_channels=in_channels, dim=dim, patch_t=2, patch_h=2, patch_w=2)
        self.blocks = nn.ModuleList([DividedAttentionBlock(dim, heads) for _ in range(depth)])
        self.out = nn.Linear(dim, in_channels * 2 * 2 * 2)

    def forward(self, x):
        tokens, grid = self.patch(x)
        for blk in self.blocks:
            tokens = blk(tokens, grid)
        return self.out(tokens), grid
```

不是一个能工作的视频生成器；是一个结构演示，展示每个部件的形状正确。

### Step 5：检查形状

```python
vid = torch.randn(1, 4, 8, 16, 16)  # (N, C, T, H, W)
model = TinyVideoDiT()
out, grid = model(vid)
print(f"input  {tuple(vid.shape)}")
print(f"tokens grid {grid}")
print(f"output {tuple(out.shape)}")
```

期望 `grid = (4, 8, 8)` 和 `out = (1, 256, 32)`（patchify 后）；head 然后投影到逐 token 的时空 patch，准备被 un-patchify 回视频。

## 实际使用

2026 年的生产访问模式：

- **Sora 2 API**（OpenAI）—— text-to-video，同步音频。高端定价。
- **Runway Gen-5 / GWM-1**（Runway）—— image-to-video，交互式世界。
- **Wan-Video 2.1 / HunyuanVideo** —— 开源自托管。
- **Cosmos / Cosmos-Drive**（NVIDIA）—— 驾驶仿真开放权重。
- **Genie 3** —— 研究预览，申请访问。

构建交互式世界模型 demo：从 Wan-Video 开始获得质量，叠加潜在动作适配器实现交互性。自动驾驶仿真：Cosmos-Drive 是 2026 年的开放参考。

对于机器人，实际使用的栈：

1. 语言目标 -> VLM（Qwen3-VL）-> 高层计划。
2. 计划 -> 潜在动作视频模型 -> 想象的 rollout。
3. Rollout -> 逆动力学模型 -> 低层动作。
4. 动作执行 -> 观察反馈到步骤 1。

## 交付产出

本课产出：

- `outputs/prompt-video-model-picker.md` —— 根据任务、许可证和延迟，在 Sora 2 / Runway / Wan / HunyuanVideo / Cosmos 之间选择。
- `outputs/skill-physical-plausibility-checks.md` —— 一个 skill，定义自动化检查（物体持久性、重力、连续性），在发布前对任何生成视频运行。

## 练习

1. **（简单）** 计算 5 秒 360p 视频在 patch-t=2、patch-h=8、patch-w=8 时的 token 数量。推理此规模下注意力的内存需求。
2. **（中等）** 将上面的分离式注意力 block 替换为完全联合注意力 block，测量形状和参数量。解释为什么分离式注意力对真实视频模型是必要的。
3. **（困难）** 构建一个最小的潜在动作视频模型：取一个 (frame_t, action_t, frame_{t+1}) 三元组数据集（任何简单 2D 游戏），训练一个条件化在动作嵌入上的小型 video DiT，展示不同动作产生不同的下一帧。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| World model | "学到的模拟器" | 给定状态和动作，预测未来观察的模型 |
| Video DiT | "时空 transformer" | 带 3D patchification 和分离式注意力的扩散 transformer |
| Latent action | "推断的控制" | 从帧对推断的离散或连续动作潜变量；用于条件化下一帧生成 |
| Divided attention | "先时间后空间" | 每个 block 两个注意力操作——跨时间然后跨空间——以保持 O(N^2) 可控 |
| Object permanence | "物体保持真实" | 视频模型必须学习的场景属性；食物、玻璃器皿上的经典失败模式 |
| FVD | "Fréchet Video Distance" | FID 的视频等价物；主要视觉质量指标 |
| Inverse dynamics model | "观察到动作" | 给定 (state, next state)，输出连接它们的动作；闭环机器人循环 |
| Cosmos-Drive | "NVIDIA 驾驶仿真" | 用于 RL 和评估的开放权重自动驾驶世界模型 |

## 延伸阅读

- [Sora technical report (OpenAI)](https://openai.com/index/video-generation-models-as-world-simulators/)
- [Genie: Generative Interactive Environments (Bruce et al., 2024)](https://arxiv.org/abs/2402.15391) — 潜在动作世界模型
- [TimeSformer (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095) — 视频 transformer 的分离式注意力
- [DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104) — RL 的世界模型
- [Cosmos-Drive-Dreams (NVIDIA, 2025)](https://research.nvidia.com/labs/toronto-ai/cosmos-drive-dreams/) — 驾驶世界模型
- [Top 10 Video Generation Models 2026 (DataCamp)](https://www.datacamp.com/blog/top-video-generation-models)
- [From Video Generation to World Model — survey repo](https://github.com/ziqihuangg/Awesome-From-Video-Generation-to-World-Model/)
