# 视频理解 — 时序建模

> 视频是一系列图像加上连接它们的物理规律。每个视频模型要么把时间当作额外的轴（3D 卷积），要么当作要 attend 的序列（transformer），要么当作提取一次然后池化的特征（2D+pool）。

**Type:** Learn + Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 03 (CNNs), Phase 4 Lesson 04 (Image Classification)
**Time:** ~45 minutes

## 学习目标

- 区分三种主要的视频建模方法（2D+pool、3D conv、时空 transformer），并预测它们的计算成本和精度权衡
- 用 PyTorch 实现帧采样、时序池化和 2D+pool baseline 分类器
- 解释为什么 I3D 的"膨胀"3D 卷积核能很好地从 ImageNet 权重迁移，以及分解式 (2+1)D 卷积有何不同
- 了解标准动作识别数据集和指标：Kinetics-400/600、UCF101、Something-Something V2；clip 级和 video 级的 top-1 准确率

## 问题背景

一段 30 秒、30 fps 的视频是 900 张图像。朴素地说，视频分类就是图像分类运行 900 次再做某种聚合。当动作在几乎每一帧都可见时（运动、烹饪、健身视频）这行得通，但当动作由运动本身定义时就彻底失败："从左向右推某物"在每一帧中看起来都只是两个静止物体。

每个视频架构的核心问题是：时序结构在何时、如何被建模？答案决定了一切——计算成本、预训练策略、能否复用 ImageNet 权重、模型在什么数据集上训练。

本课刻意比静态图像课程短。核心图像机制已经就位，视频理解主要是关于时序的故事：采样、建模和聚合。

## 核心概念

### 三大架构家族

```mermaid
flowchart LR
    V["Video clip<br/>(T frames)"] --> A1["2D + pool<br/>run 2D CNN per frame,<br/>average over time"]
    V --> A2["3D conv<br/>convolve over<br/>T x H x W"]
    V --> A3["Spatio-temporal<br/>transformer<br/>attention over<br/>(t, h, w) tokens"]

    A1 --> C["Logits"]
    A2 --> C
    A3 --> C

    style A1 fill:#dbeafe,stroke:#2563eb
    style A2 fill:#fef3c7,stroke:#d97706
    style A3 fill:#dcfce7,stroke:#16a34a
```

### 2D + pool

取一个 2D CNN（ResNet、EfficientNet、ViT）。在每个采样帧上独立运行。对逐帧 embedding 做平均（或 max-pool、或 attention-pool）。将池化后的向量送入分类器。

优点：
- ImageNet 预训练直接迁移。
- 实现最简单。
- 便宜：T 帧 * 单图推理成本。

缺点：
- 无法建模运动。动作 = 外观的聚合。
- 时序池化是顺序无关的；"开门"和"关门"看起来一样。

适用场景：外观主导的任务、小视频数据集上的迁移学习、初始 baseline。

### 3D 卷积

将 2D (H, W) 卷积核替换为 3D (T, H, W) 卷积核。网络同时在空间和时间上卷积。早期家族：C3D、I3D、SlowFast。

I3D 技巧：取一个预训练的 2D ImageNet 模型，将每个 2D 卷积核沿新的时间轴复制来"膨胀"。一个 3x3 的 2D 卷积变成 3x3x3 的 3D 卷积。这给了 3D 模型强大的预训练权重，而非从零训练。

优点：
- 直接建模运动。
- I3D 膨胀提供免费的迁移学习。

缺点：
- 比 2D 对应物多 T/8 的 FLOPs（对于时间核为 3 堆叠 3 次的情况）。
- 时间核很小；长程运动需要金字塔或双流方法。

适用场景：运动是信号的动作识别（Something-Something V2、Kinetics 中运动密集的类别）。

### 时空 transformer

将视频 tokenize 为时空 patch 的网格，在所有 token 上做 attention。TimeSformer、ViViT、Video Swin、VideoMAE。

重要的 attention 模式：
- **Joint** — 在 (t, h, w) 上做一个大 attention。对 `T*H*W` 是二次的；昂贵。
- **Divided** — 每个 block 两个 attention：一个在时间上，一个在空间上。近似线性扩展。
- **Factorised** — 时间 attention 和空间 attention 在 block 间交替。

优点：
- 在每个主要 benchmark 上达到 SOTA 精度。
- 从图像 transformer（ViT）通过 patch 膨胀迁移。
- 通过稀疏 attention 支持长上下文视频。

缺点：
- 计算密集。
- 需要仔细选择 attention 模式，否则运行时间暴涨。

适用场景：大数据集、高保真视频理解、多模态视频+文本任务。

### 帧采样

一个 10 秒、30 fps 的片段是 300 帧；将全部 300 帧送入任何模型都是浪费。标准策略：

- **均匀采样** — 在片段中均匀选取 T 帧。2D+pool 的默认选择。
- **密集采样** — 随机连续 T 帧窗口。3D 卷积常用，因为运动需要相邻帧。
- **Multi-clip** — 从同一视频采样多个 T 帧窗口，分别分类，测试时平均预测。

T 通常是 8、16、32 或 64。更高的 T = 更多时序信号但更多计算。

### 评估

两个层级：
- **Clip 级准确率** — 模型看到一个 T 帧 clip，报告 top-k。
- **Video 级准确率** — 对每个视频的多个 clip 级预测取平均；更高更稳定。

始终报告两者。一个模型 78% clip / 82% video 说明它严重依赖测试时平均；80% / 81% 的模型每个 clip 更鲁棒。

### 你会遇到的数据集

- **Kinetics-400 / 600 / 700** — 通用动作数据集。40 万片段；YouTube URL（很多已失效）。
- **Something-Something V2** — 运动定义的动作（"将 X 从左移到右"）。2D+pool 无法解决。
- **UCF-101**、**HMDB-51** — 更老、更小，仍在报告。
- **AVA** — 时空动作*定位*；比分类更难。

## 动手构建

### Step 1: 帧采样器

均匀和密集采样器，作用于帧列表（或视频张量）。

```python
import numpy as np

def sample_uniform(num_frames_total, T):
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    step = num_frames_total / T
    return [int(i * step) for i in range(T)]


def sample_dense(num_frames_total, T, rng=None):
    rng = rng or np.random.default_rng()
    if num_frames_total <= T:
        return list(range(num_frames_total)) + [num_frames_total - 1] * (T - num_frames_total)
    start = int(rng.integers(0, num_frames_total - T + 1))
    return list(range(start, start + T))
```

两者都返回 `T` 个索引，用于切片视频张量。

### Step 2: 2D+pool baseline

在每帧上运行 2D ResNet-18，平均池化特征，分类。

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FramePool(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))  # global avg pool kept
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (N, T, 3, H, W)
        N, T = x.shape[:2]
        x = x.view(N * T, *x.shape[2:])
        feats = self.features(x).view(N, T, -1)
        pooled = feats.mean(dim=1)
        return self.head(pooled)

model = FramePool(num_classes=10)
x = torch.randn(2, 8, 3, 224, 224)
print(f"output: {model(x).shape}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")
```

1100 万参数，ImageNet 预训练，逐帧运行，平均，分类。这个 baseline 在外观主导的任务上通常与正规 3D 模型相差 5-10 个点——有时更好，因为它复用了更强的 ImageNet backbone。

### Step 3: I3D 风格的膨胀 3D 卷积

将单个 2D 卷积沿新的时间轴复制权重，变成 3D 卷积。

```python
def inflate_2d_to_3d(conv2d, time_kernel=3):
    out_c, in_c, kh, kw = conv2d.weight.shape
    weight_3d = conv2d.weight.data.unsqueeze(2)  # (out, in, 1, kh, kw)
    weight_3d = weight_3d.repeat(1, 1, time_kernel, 1, 1) / time_kernel
    conv3d = nn.Conv3d(in_c, out_c, kernel_size=(time_kernel, kh, kw),
                        padding=(time_kernel // 2, conv2d.padding[0], conv2d.padding[1]),
                        stride=(1, conv2d.stride[0], conv2d.stride[1]),
                        bias=False)
    conv3d.weight.data = weight_3d
    return conv3d

conv2d = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
conv3d = inflate_2d_to_3d(conv2d, time_kernel=3)
print(f"2D weight shape:  {tuple(conv2d.weight.shape)}")
print(f"3D weight shape:  {tuple(conv3d.weight.shape)}")
x = torch.randn(1, 3, 8, 56, 56)
print(f"3D output shape:  {tuple(conv3d(x).shape)}")
```

除以 `time_kernel` 保持激活幅度大致不变——这对于第一次前向传播时不破坏 batch-norm 统计量很重要。

### Step 4: 分解式 (2+1)D 卷积

将 3D 卷积拆分为 2D（空间）和 1D（时间）卷积。相同感受野，更少参数，在某些 benchmark 上精度更好。

```python
class Conv2Plus1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        mid_c = (in_c * out_c * kernel_size * kernel_size * kernel_size) \
                // (in_c * kernel_size * kernel_size + out_c * kernel_size)
        self.spatial = nn.Conv3d(in_c, mid_c, kernel_size=(1, kernel_size, kernel_size),
                                 padding=(0, kernel_size // 2, kernel_size // 2), bias=False)
        self.bn = nn.BatchNorm3d(mid_c)
        self.act = nn.ReLU(inplace=True)
        self.temporal = nn.Conv3d(mid_c, out_c, kernel_size=(kernel_size, 1, 1),
                                  padding=(kernel_size // 2, 0, 0), bias=False)

    def forward(self, x):
        return self.temporal(self.act(self.bn(self.spatial(x))))

c = Conv2Plus1D(3, 64)
x = torch.randn(1, 3, 8, 56, 56)
print(f"(2+1)D output: {tuple(c(x).shape)}")
```

完整的 R(2+1)D 网络就是将 ResNet-18 中每个 3x3 卷积替换为 `Conv2Plus1D`。

## 实际使用

两个库覆盖生产视频工作：

- `torchvision.models.video` — R(2+1)D、MViT、Swin3D，带预训练 Kinetics 权重。与图像模型相同的 API。
- `pytorchvideo`（Meta）— 模型库、Kinetics / SSv2 / AVA 的数据加载器、标准变换。

对于视觉-语言视频模型（视频字幕、视频 QA），使用 `transformers`（`VideoMAE`、`VideoLLaMA`、`InternVideo`）。

## 交付产出

本课产出：

- `outputs/prompt-video-architecture-picker.md` — 一个 prompt，根据外观 vs 运动、数据集大小和计算预算选择 2D+pool / I3D / (2+1)D / transformer。
- `outputs/skill-frame-sampler-auditor.md` — 一个 skill，检查视频 pipeline 的采样器并标记常见 bug：off-by-one 索引、`num_frames < T` 时的不均匀采样、缺少保持宽高比的裁剪等。

## 练习

1. **（简单）** 计算 FramePool（T=8）与 I3D 风格 3D ResNet（T=8）的近似 FLOPs。解释为什么 2D+pool 便宜 3-5 倍。
2. **（中等）** 生成一个合成视频数据集：随机球体沿随机方向运动，按运动方向标注（"从左到右"、"从右到左"、"对角向上"）。在上面训练 FramePool。展示它达到接近随机的准确率，证明仅靠外观不足以完成运动任务。
3. **（困难）** 通过将 ResNet-18 中每个 Conv2d 替换为 `Conv2Plus1D` 来构建 R(2+1)D-18。从 ImageNet 预训练的 ResNet-18 膨胀第一个卷积的权重。在练习 2 的运动数据集上训练并击败 FramePool。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 2D + pool | "逐帧分类器" | 在每个采样帧上运行 2D CNN，跨时间平均池化特征，分类 |
| 3D convolution | "时空卷积核" | 在 (T, H, W) 上卷积的核；能原生建模运动 |
| Inflation | "将 2D 权重提升到 3D" | 将 2D 卷积权重沿新时间轴复制来初始化 3D 卷积权重，然后除以 kernel_T 以保持激活尺度 |
| (2+1)D | "分解卷积" | 将 3D 拆分为 2D 空间 + 1D 时间；更少参数，中间多一个非线性 |
| Divided attention | "先时间后空间" | 每层两个 attention 的 transformer block：一个在同一帧的 token 上，一个在同一位置的 token 上 |
| Clip | "T 帧窗口" | 采样的 T 帧子序列；视频模型消费的单位 |
| Clip vs video accuracy | "两种评估设置" | Clip = 每个视频一个样本，video = 多个采样 clip 的平均 |
| Kinetics | "视频的 ImageNet" | 400-700 个动作类别，30 万+ YouTube 片段，标准视频预训练语料 |

## 延伸阅读

- [I3D: Quo Vadis, Action Recognition (Carreira & Zisserman, 2017)](https://arxiv.org/abs/1705.07750) — 引入膨胀和 Kinetics 数据集
- [R(2+1)D: A Closer Look at Spatiotemporal Convolutions (Tran et al., 2018)](https://arxiv.org/abs/1711.11248) — 分解卷积，仍是强 baseline
- [TimeSformer: Is Space-Time Attention All You Need? (Bertasius et al., 2021)](https://arxiv.org/abs/2102.05095) — 第一个强视频 transformer
- [VideoMAE (Tong et al., 2022)](https://arxiv.org/abs/2203.12602) — 视频的 masked autoencoder 预训练；当前主流预训练方案
