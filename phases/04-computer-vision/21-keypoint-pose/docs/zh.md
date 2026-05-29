# 关键点检测与姿态估计

> Pose 是一组有序关键点。关键点检测器是一个热力图回归器。其他都是记账工作。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (Detection), Phase 4 Lesson 07 (U-Net)
**Time:** ~45 minutes

## 学习目标

- 区分 top-down 和 bottom-up 姿态估计，说明各自的适用场景
- 用 Gaussian-per-keypoint 目标回归 K 个关键点的热力图，并在推理时提取关键点坐标
- 解释 Part Affinity Fields（PAFs）以及 bottom-up 流水线如何将关键点关联到实例
- 使用 MediaPipe Pose 或 MMPose 进行生产级关键点估计，理解其输出格式

## 问题

关键点任务隐藏在很多名字下：人体姿态（17 个身体关节）、人脸 landmarks（68 或 478 个点）、手部（21 个点）、动物姿态、机器人物体姿态、医学解剖 landmarks。它们都共享相同的结构：检测物体上的 K 个离散点并输出其 (x, y) 坐标。

姿态估计是动作捕捉、健身应用、体育分析、手势控制、动画、AR 试穿和机器人抓取的基础。2D 情况已经成熟；3D pose（从单目相机估计世界坐标中的关节位置）是当前的研究前沿。

工程问题在于规模。单图、单人 pose 是 20ms 的问题。人群中 30 fps 的多人 pose 是一个不同的问题，需要不同的架构。

## 概念

### Top-down vs bottom-up

```mermaid
flowchart LR
    subgraph TD["Top-down pipeline"]
        A1["Detect person boxes"] --> A2["Crop each box"]
        A2 --> A3["Per-box keypoint model<br/>(HRNet, ViTPose)"]
    end
    subgraph BU["Bottom-up pipeline"]
        B1["One pass over image"] --> B2["All keypoint heatmaps<br/>+ association field"]
        B2 --> B3["Group keypoints into<br/>instances (greedy matching)"]
    end

    style TD fill:#dbeafe,stroke:#2563eb
    style BU fill:#fef3c7,stroke:#d97706
```

- **Top-down** — 先检测人，然后对每个裁剪运行逐人关键点模型。精度最高；随人数线性扩展。
- **Bottom-up** — 一次前向传播预测所有关键点加关联场；然后分组。无论人群多大，时间恒定。

Top-down（HRNet, ViTPose）是精度领先者；bottom-up（OpenPose, HigherHRNet）是拥挤场景的吞吐量领先者。

### 热力图回归

不直接回归 `(x, y)`，而是为每个关键点预测一个 `H x W` 的热力图，以真实位置为中心放一个高斯 blob。

```
target[k, y, x] = exp(-((x - cx_k)^2 + (y - cy_k)^2) / (2 sigma^2))
```

推理时，每个热力图的 argmax 就是预测的关键点位置。

为什么热力图比直接回归效果好：网络的空间结构（conv feature map）与空间输出天然对齐。高斯目标还起到正则化作用 — 小的定位误差产生小的 loss，而不是零。

### 亚像素定位

Argmax 给出整数坐标。要获得亚像素精度，可以对 argmax 及其邻居拟合抛物线，或使用经典的偏移 `(dx, dy) = 0.25 * (heatmap[y, x+1] - heatmap[y, x-1], ...)` 方向。

### Part Affinity Fields（PAFs）

OpenPose 用于 bottom-up 关联的技巧。对每对相连的关键点（如左肩到左肘），预测一个 2 通道场，编码从一个指向另一个的单位向量。要关联一个肩膀和它的肘部，沿连接候选对的线积分 PAF；积分最高的对被匹配。

```
For each connection (limb):
  PAF channels: 2 (unit vector x, y)
  Line integral: sum over sample points of (PAF . line_direction)
  Higher integral = stronger match
```

优雅且可扩展到任意人群规模，无需逐人裁剪。

### COCO keypoints

标准人体姿态数据集：每人 17 个关键点，PCK（Percentage of Correct Keypoints）和 OKS（Object Keypoint Similarity）作为指标。OKS 是关键点版的 IoU，COCO mAP@OKS 报告的就是它。

### 2D vs 3D

- **2D pose** — 图像坐标；已达到生产质量（MediaPipe, HRNet, ViTPose）。
- **3D pose** — 世界/相机坐标；仍在活跃研究中。常见方法：
  - 用小 MLP 将 2D 预测提升到 3D（VideoPose3D）。
  - 从图像直接 3D 回归（PyMAF, MHFormer）。
  - 多视角设置（CMU Panoptic）获取 ground truth。

## 动手构建

### Step 1: 高斯热力图目标

```python
import numpy as np
import torch

def gaussian_heatmap(size, cx, cy, sigma=2.0):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)

hm = gaussian_heatmap(64, 32, 32, sigma=2.0)
print(f"peak: {hm.max():.3f} at ({hm.argmax() % 64}, {hm.argmax() // 64})")
```

逐关键点热力图沿通道轴堆叠得到完整的目标张量。

### Step 2: 微型关键点 head

一个 U-Net 风格的模型，输出 K 个热力图通道。

```python
import torch.nn as nn
import torch.nn.functional as F

class TinyKeypointNet(nn.Module):
    def __init__(self, num_keypoints=4, base=16):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, base, 3, 2, 1), nn.ReLU(inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(base, base * 2, 3, 2, 1), nn.ReLU(inplace=True))
        self.mid = nn.Sequential(nn.Conv2d(base * 2, base * 2, 3, 1, 1), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.up2 = nn.ConvTranspose2d(base, num_keypoints, 2, 2)

    def forward(self, x):
        h1 = self.down1(x)
        h2 = self.down2(h1)
        h3 = self.mid(h2)
        u1 = self.up1(h3)
        return self.up2(u1)
```

输入 `(N, 3, H, W)`，输出 `(N, K, H, W)`。Loss 是对高斯目标的逐像素 MSE。

### Step 3: 推理 — 提取关键点坐标

```python
def heatmap_to_coords(heatmaps):
    """
    heatmaps: (N, K, H, W)
    returns:  (N, K, 2) float coordinates in image pixels
    """
    N, K, H, W = heatmaps.shape
    hm = heatmaps.reshape(N, K, -1)
    idx = hm.argmax(dim=-1)
    ys = (idx // W).float()
    xs = (idx % W).float()
    return torch.stack([xs, ys], dim=-1)

coords = heatmap_to_coords(torch.randn(2, 4, 32, 32))
print(f"coords: {coords.shape}")  # (2, 4, 2)
```

推理时一行搞定。要亚像素精度，在 argmax 周围插值。

### Step 4: 合成关键点数据集

简单：在白色画布上画四个点，学习预测它们。

```python
def make_synthetic_sample(size=64):
    img = np.ones((3, size, size), dtype=np.float32)
    rng = np.random.default_rng()
    kps = rng.integers(8, size - 8, size=(4, 2))
    for cx, cy in kps:
        img[:, cy - 2:cy + 2, cx - 2:cx + 2] = 0.0
    hms = np.stack([gaussian_heatmap(size, cx, cy) for cx, cy in kps])
    return img, hms, kps
```

简单到一个微型模型一分钟就能学会。

### Step 5: 训练

```python
model = TinyKeypointNet(num_keypoints=4)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(200):
    batch = [make_synthetic_sample() for _ in range(16)]
    imgs = torch.from_numpy(np.stack([b[0] for b in batch]))
    hms = torch.from_numpy(np.stack([b[1] for b in batch]))
    pred = model(imgs)
    # Upsample pred to full resolution
    pred = F.interpolate(pred, size=hms.shape[-2:], mode="bilinear", align_corners=False)
    loss = F.mse_loss(pred, hms)
    opt.zero_grad(); loss.backward(); opt.step()
```

## 实际应用

- **MediaPipe Pose** — Google 的生产级姿态估计器；提供 WebGL + 移动端运行时，延迟低于 10ms。
- **MMPose**（OpenMMLab）— 全面的研究代码库；每个 SOTA 架构都有预训练权重。
- **YOLOv8-pose** — 单次前向传播最快的实时多人 pose。
- **transformers HumanDPT / PoseAnything** — 更新的视觉语言方法，用于开放词汇 pose（任意物体、任意关键点集）。

## 交付产出

本课产出：

- `outputs/prompt-pose-stack-picker.md` — 一个 prompt，根据延迟、人群规模和 2D vs 3D 需求选择 MediaPipe / YOLOv8-pose / HRNet / ViTPose。
- `outputs/skill-heatmap-to-coords.md` — 一个 skill，编写每个生产级 pose 模型使用的亚像素热力图到坐标转换例程。

## 练习

1. **（简单）** 在合成 4 点数据集上训练微型关键点模型。报告 200 步后预测和真实关键点之间的平均 L2 误差。
2. **（中等）** 添加亚像素精化：给定 argmax 位置，沿 x 和 y 从相邻像素拟合 1D 抛物线。报告相比整数 argmax 的精度提升。
3. **（困难）** 构建一个 2 人合成数据集，每张图片显示两个 4 关键点模式的实例。训练一个带 PAFs 的 bottom-up 流水线，预测哪个关键点属于哪个实例，并评估 OKS。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Keypoint | "Landmark" | 物体上的一个特定有序点（关节、角点、特征） |
| Pose | "骨架" | 属于一个实例的有序关键点集合 |
| Top-down | "先检测再估计 pose" | 两阶段流水线：人体检测器 + 逐裁剪关键点模型；精度最高 |
| Bottom-up | "先估计 pose 再分组" | 单次全关键点预测 + 分组；时间不随人群规模增长 |
| Heatmap | "高斯目标" | 每个关键点的 H x W 张量，峰值在真实位置；首选的回归目标 |
| PAF | "Part Affinity Field" | 编码肢体方向的 2 通道单位向量场；用于将关键点分组到实例 |
| OKS | "关键点 IoU" | Object Keypoint Similarity；COCO 的 pose 指标 |
| HRNet | "High-Resolution Net" | 主流的 top-down 关键点架构；全程保持高分辨率特征 |

## 延伸阅读

- [OpenPose (Cao et al., 2017)](https://arxiv.org/abs/1812.08008) — 带 PAFs 的 bottom-up 方法；至今仍是该方法最好的文字说明
- [HRNet (Sun et al., 2019)](https://arxiv.org/abs/1902.09212) — top-down 参考架构
- [ViTPose (Xu et al., 2022)](https://arxiv.org/abs/2204.12484) — 纯 ViT 作为 pose backbone；多个基准上的当前 SOTA
- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) — 生产级实时 pose；2026 年部署最快的技术栈
