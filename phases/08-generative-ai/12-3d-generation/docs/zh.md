# 三维生成

> 3D 是 2D-to-3D 杠杆最强的模态。2023 年的突破是 3D Gaussian Splatting。2024-2026 的生成推进在其上叠加多视角扩散 + 3D 重建，从单个 prompt 或照片生成物体和场景。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 4 (Vision), Phase 8 · 07 (Latent Diffusion)
**Time:** ~45 minutes

## 问题

3D 内容很痛苦：

- **表示。** Mesh、点云、体素网格、有符号距离场（SDF）、神经辐射场（NeRF）、3D Gaussians。每种都有权衡。
- **数据稀缺。** ImageNet 有 1400 万张图像。最大的干净 3D 数据集（Objaverse-XL, 2023）有约 1000 万个物体，大多质量低。
- **内存。** 512³ 体素网格是 1.28 亿体素；有用的场景 NeRF 需要每条光线 100 万次采样。生成比重建更难。
- **监督。** 对于 2D 图像你有像素。对于 3D 你通常只有少量 2D 视角，必须提升到 3D。

2026 年的技术栈将两个问题分开。首先，用扩散模型生成 *2D 多视角图像*。其次，将 *3D 表示*（通常是 Gaussian splatting）拟合到这些图像。

## 概念

![3D generation: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### 表示：3D Gaussian Splatting (Kerbl et al., 2023)

将场景表示为约 100 万个 3D Gaussians 的云。每个有 59 个参数：位置 (3)、协方差 (6，或四元数 4 + 缩放 3)、不透明度 (1)、球谐颜色 (degree 3 时 48，degree 0 时 3)。

渲染 = 投影 + alpha 合成。快（4090 上 1080p 约 100 fps）。可微分。通过梯度下降对真实照片拟合。一个场景在消费级 GPU 上 5-30 分钟拟合完成。

2023-2024 的两个创新：
- **生成式 Gaussian splats。** LGM、LRM、InstantMesh 等模型直接从一张或几张图像预测 Gaussian 云。
- **4D Gaussian Splatting。** 带逐帧偏移的 Gaussians 用于动态场景。

### 多视角扩散

微调预训练图像扩散模型以从文本 prompt 或单张图像生成同一物体的多个一致视角。Zero123 (Liu et al., 2023)、MVDream (Shi et al., 2023)、SV3D (Stability, 2024)、CAT3D (Google, 2024)。通常输出物体周围 4-16 个视角，通过 Gaussian splatting 或 NeRF 提升到 3D。

### Text-to-3D 管线

| Model | Input | Output | Time |
|-------|-------|--------|------|
| DreamFusion (2022) | text | NeRF via SDS | ~1 hour per asset |
| Magic3D | text | mesh + texture | ~40 min |
| Shap-E (OpenAI, 2023) | text | implicit 3D | ~1 min |
| SJC / ProlificDreamer | text | NeRF / mesh | ~30 min |
| LRM (Meta, 2023) | image | triplane | ~5 s |
| InstantMesh (2024) | image | mesh | ~10 s |
| SV3D (Stability, 2024) | image | novel views | ~2 min |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | ~1 min |
| TripoSR (2024) | image | mesh | ~1 s |
| Meshy 4 (2025) | text + image | PBR mesh | ~30 s |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | ~60 s |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | ~30 s |

2025-2026 方向：直接 text-to-mesh 模型带 PBR 材质，适合游戏引擎。多视角扩散中间步骤仍然是通用物体的最佳配方。

### NeRF（作为背景）

Neural Radiance Field (Mildenhall et al., 2020)。一个小型 MLP 接收 `(x, y, z, view direction)` 并输出 `(color, density)`。通过沿光线积分渲染。在质量上击败基于 mesh 的新视角合成但渲染慢 100-1000 倍。在大多数实时用途中被 Gaussian splatting 取代但在研究中仍占主导。

## Build It

`code/main.py` 实现了一个玩具 2D "Gaussian splatting" 拟合：将合成目标图像（平滑渐变）表示为 2D Gaussian splats 的和。通过梯度下降优化位置、颜色和协方差以匹配目标。你看到两个核心操作：前向渲染（splat + alpha 合成）和梯度下降拟合。

### Step 1: 2D Gaussian splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Step 2: render by summing splats

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实 3D Gaussian splatting 按深度排序 Gaussians 并按顺序 alpha 合成。我们的 2D 玩具只是求和。

### Step 3: fit by gradient descent

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Pitfalls

- **视角不一致。** 如果你独立生成 4 个视角且它们对物体结构不一致，3D 拟合会模糊。修复：带共享注意力的多视角扩散。
- **背面幻觉。** 单图像 → 3D 必须发明看不见的一面。质量差异很大。
- **Gaussian splat 爆炸。** 无约束训练增长到 1000 万 splats 并过拟合。致密化 + 剪枝启发式（来自 3D-GS 原始论文）是必需的。
- **拓扑问题。** 从隐式场（SDF）提取的 mesh 经常有孔或自交叉。部署前运行 remesher（如 blender 的 voxel remesh）。
- **训练数据许可。** Objaverse 有混合许可；商业使用因模型而异。

## Use It

| Task | 2026 pick |
|------|-----------|
| 从照片重建场景 | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| 游戏用 text-to-3D 物体 | Meshy 4 或 Rodin Gen-1.5 (PBR output) |
| Image-to-3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| 少量图像的新视角合成 | CAT3D, SV3D |
| 动态场景重建 | 4D Gaussian Splatting |
| Avatar / 穿衣人体 | Gaussian Avatar, HUGS |
| 研究 / SOTA | 上周刚出的那个 |

在游戏或电商管线中部署生产 3D：Meshy 4 或 Rodin Gen-1.5 输出 PBR mesh 可直接进入 Unity / Unreal。

## Ship It

保存 `outputs/skill-3d-pipeline.md`。Skill 接收 3D 简报（输入：文本 / 一张图像 / 几张图像；输出：mesh / splat / NeRF；用途：渲染 / 游戏 / VR），输出：管线（多视角扩散 + 拟合，或直接 mesh 模型）、基础模型、迭代预算、拓扑后处理、所需材质通道。

## Exercises

1. **Easy.** 用 4、16、64 个 Gaussians 运行 `code/main.py`。报告最终 MSE vs 目标。
2. **Medium.** 扩展为彩色 Gaussians (RGB)。确认重建匹配目标颜色模式。
3. **Hard.** 使用 gsplat 或 Nerfstudio，从 50 张照片捕获重建真实物体。报告拟合时间和保留视角上的最终 SSIM。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | "3DGS" | 场景作为 3D Gaussians 云；可微分 alpha 合成渲染。 |
| NeRF | "神经辐射场" | 在 3D 点输出颜色 + 密度的 MLP；通过光线积分渲染。 |
| Triplane | "三个 2-D 平面" | 将 3D 分解为三个 2-D 轴对齐特征网格；比体积便宜。 |
| SDS | "Score distillation sampling" | 用 2D 扩散 score 作为伪梯度训练 3D 模型。 |
| 多视角扩散 | "一次多个视角" | 输出一批一致相机视角的扩散模型。 |
| PBR | "基于物理的渲染" | 带 albedo、roughness、metallic、normal 通道的材质。 |
| Densification | "增长 splats" | 3DGS 训练启发式：在高梯度区域分裂/克隆 splats。 |

## 生产笔记：3D 还没有共享基底

不像图像（latent diffusion + DiT）和视频（时空 DiT），3D 在 2026 年没有单一主导运行时。生产决策树在表示上分叉：

- **NeRF / triplane。** 推理是光线行进 + 每采样点一次 MLP 前向。512² 渲染需要数百万次 MLP 前向。积极批处理光线采样；SDPA/xformers 适用。
- **多视角扩散 + LRM 重建。** 两阶段管线。阶段 1（多视角 DiT）是与第 07 课相同的扩散服务器。阶段 2（LRM transformer）是对视角的一次性前向传播。整体延迟特征是"扩散 + 一次性"——相应选择每阶段服务原语。
- **SDS / DreamFusion。** 逐资产优化，不是推理。构建作业，不是请求处理器。

对于大多数 2026 年产品，正确答案是"按请求运行多视角扩散模型，异步重建为 3DGS，为实时查看服务 3DGS"。这将工作负载干净地分为 GPU 推理服务器（快）和离线优化器（慢）。

## Further Reading

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — multi-view diffusion.
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D.
