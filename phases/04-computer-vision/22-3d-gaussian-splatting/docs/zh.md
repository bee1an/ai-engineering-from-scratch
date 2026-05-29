# 从零实现 3D Gaussian Splatting

> 一个场景就是数百万个 3D 高斯的点云。每个高斯有位置、朝向、尺度、不透明度，以及随观察方向变化的颜色。光栅化它们，通过光栅化反向传播，搞定。

**类型：** 动手构建
**语言：** Python
**前置课程：** Phase 4 Lesson 13（3D 视觉与 NeRF）、Phase 1 Lesson 12（张量运算）、Phase 4 Lesson 10（扩散模型基础，可选）
**时长：** 约 90 分钟

## 学习目标

- 解释为什么 3D Gaussian Splatting 在 2026 年取代 NeRF 成为照片级真实感 3D 重建的生产默认方案
- 列出每个高斯的六类参数（位置、旋转四元数、尺度、不透明度、球谐颜色、可选特征），以及每类贡献多少浮点数
- 从零实现一个 2D Gaussian Splatting 光栅器（使用 alpha 合成），然后展示 3D 情况如何投影到同一循环
- 使用 `nerfstudio`、`gsplat` 或 `SuperSplat` 从 20-50 张照片重建场景，并导出为 `KHR_gaussian_splatting` glTF 扩展或 OpenUSD 26.03 的 `UsdVolParticleField3DGaussianSplat` schema

## 问题背景

NeRF 将场景存储为 MLP 的权重。每渲染一个像素需要沿射线进行数百次 MLP 查询。训练需要数小时，渲染需要数秒，而且权重无法编辑——如果你想移动场景中的一把椅子，就得重新训练。

3D Gaussian Splatting（Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023）取代了这一切。场景是一组显式的 3D 高斯。渲染是 GPU 光栅化，帧率 100+ fps。训练只需几分钟。编辑是直接的：平移一部分高斯就等于移动了椅子。到 2026 年，Khronos Group 已经批准了高斯 splat 的 glTF 扩展，OpenUSD 26.03 内置了高斯 splat schema，Zillow 和 Apartments.com 用它们渲染房产，大多数新的 3D 重建研究论文都是核心 3DGS 思想的变体。

心智模型很简单，但数学有足够多的活动部件，以至于大多数介绍从光栅化开始，跳过了投影和球谐函数。本课从头构建整个流程——先做 2D 版本，再扩展到 3D。

## 核心概念

### 一个高斯携带什么

一个 3D 高斯是空间中的参数化 blob，具有以下属性：

```
position         mu         (3,)    世界坐标中的中心
rotation         q          (4,)    编码朝向的单位四元数
scale            s          (3,)    每轴的 log 尺度（渲染时取指数）
opacity          alpha      (1,)    经 sigmoid 后的不透明度 [0, 1]
SH coefficients  c_lm       (3 * (L+1)^2,)   视角相关的颜色
```

旋转 + 尺度构建一个 3x3 协方差矩阵：`Sigma = R S S^T R^T`。这就是高斯在 3D 中的形状。球谐函数让颜色随观察方向变化——镜面高光、微妙光泽、视角相关的辉光——而无需存储逐视角纹理。使用 SH degree 3 时，每个颜色通道有 16 个系数，每个高斯仅颜色就需要 48 个浮点数。

一个场景通常有 1-5 百万个高斯。每个存储大约 60 个浮点数（3 + 4 + 3 + 1 + 48 + 其他）。五百万高斯的场景约 240 MB——远小于等效的带逐点纹理的点云，比 NeRF 的 MLP 权重在高分辨率下重新渲染小一个数量级。

### 光栅化，而非光线步进

```mermaid
flowchart LR
    SCENE["Millions of 3D Gaussians<br/>(position, rotation, scale,<br/>opacity, SH colour)"] --> PROJ["Project to 2D<br/>(camera extrinsics + intrinsics)"]
    PROJ --> TILES["Assign to tiles<br/>(16x16 screen-space)"]
    TILES --> SORT["Depth-sort<br/>per tile"]
    SORT --> ALPHA["Alpha-composite<br/>front-to-back"]
    ALPHA --> PIX["Pixel colour"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

五个步骤，全部对 GPU 友好。无需逐像素 MLP 查询。单张 RTX 3080 Ti 可以 147 fps 渲染 600 万个 splat。

### 投影步骤

世界位置 `mu`、3D 协方差 `Sigma` 的 3D 高斯投影为屏幕位置 `mu'`、2D 协方差 `Sigma'` 的 2D 高斯：

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = viewing transform (rotation + translation of camera)
J = Jacobian of the perspective projection at mu'
```

2D 高斯的覆盖范围是一个椭圆，其轴是 `Sigma'` 的特征向量。椭圆内的每个像素都会收到该高斯的贡献，权重为 `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`。

### Alpha 合成规则

对于一个像素，覆盖它的高斯按从后到前排序（或等价地，用反转公式从前到后）。颜色的合成方程与 1980 年代以来所有半透明光栅器相同：

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       transmittance up to i
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   local contribution
c_i = eval_SH(SH_i, view_direction)    view-dependent colour
```

这与 **NeRF 的体渲染方程完全相同**，只是作用于显式稀疏的高斯集合，而非沿射线的密集采样。这个等价性就是为什么渲染质量能匹配 NeRF——两者都在积分同一个辐射场方程。

### 为什么这是可微的

每一步——投影、tile 分配、alpha 合成、SH 求值——都对高斯参数可微。给定真实图像，计算渲染像素损失，通过光栅器反向传播，用梯度下降更新所有 `(mu, q, s, alpha, c_lm)`。经过约 30,000 次迭代，高斯会找到正确的位置、尺度和颜色。

### 致密化与剪枝

固定数量的高斯无法覆盖复杂场景。训练包含两种自适应机制：

- **Clone**（克隆）——当梯度幅度大但尺度小时，在当前位置克隆一个高斯——重建在此处需要更多细节。
- **Split**（分裂）——当梯度大时，将一个大尺度高斯分裂为两个更小的——一个大高斯太平滑，无法拟合该区域。
- **Prune**（剪枝）——移除不透明度降到阈值以下的高斯——它们没有贡献。

致密化每 N 次迭代运行一次。场景通常从约 100k 初始高斯（由 SfM 点播种）增长到训练结束时的 1-5M。

### 一段话讲清球谐函数

视角相关颜色是单位球上的函数 `c(direction)`。球谐函数是球面的傅里叶基。截断到 degree `L` 时有 `(L+1)^2` 个基函数每通道。为新视角计算颜色就是学到的 SH 系数与在观察方向上求值的基之间的点积。Degree 0 = 一个系数 = 常数颜色。Degree 3 = 16 个系数 = 足以捕捉 Lambertian 着色、镜面反射和轻微反射。3DGS 论文默认使用 degree 3。

### 2026 年的生产技术栈

```
1. Capture         smartphone / DJI drone / handheld scanner
2. SfM / MVS       COLMAP or GLOMAP derives camera poses + sparse points
3. Train 3DGS      nerfstudio / gsplat / inria official / PostShot (~10-30 min on RTX 4090)
4. Edit            SuperSplat / SplatForge (clean floaters, segment)
5. Export          .ply -> glTF KHR_gaussian_splatting or .usd (OpenUSD 26.03)
6. View            Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D 与生成式变体

- **4D Gaussian Splatting** —— 高斯是时间的函数；用于体积视频（Superman 2026、A$AP Rocky 的 "Helicopter"）。
- **生成式 splat** —— text-to-splat 模型（World Labs 的 Marble）可以凭空生成整个场景。
- **3D Gaussian Unscented Transform** —— NVIDIA NuRec 的变体，用于自动驾驶仿真。

## 动手构建

### Step 1：2D 高斯

我们先构建一个 2D 光栅器。3D 情况在投影后归结为它。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      centres
    covs:   (G, 2, 2)   covariance matrices
    points: (H, W, 2)   pixel coordinates
    returns: (G, H, W)  density at every pixel for every Gaussian
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum` 对每个 (高斯, 像素) 对执行二次型 `diff^T Sigma^-1 diff`。

### Step 2：2D splatting 光栅器

从前到后的 alpha 合成。在 2D 中深度没有意义，所以我们用一个可学习的逐高斯标量来排序。

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     in [0, 1]
    depths:    (G,)     per-Gaussian scalar used for ordering
    image_size: (H, W)
    returns:   (H, W, 3) rendered image
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

不快——真正的实现使用基于 tile 的 CUDA kernel——但数学完全正确且完全可微。

### Step 3：可训练的 2D splat 场景

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # single angle in 2D
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`、`opacity_logit` 和 `colour_logits` 都是无约束参数，在渲染时通过正确的激活函数映射。这是每个 3DGS 实现的标准模式。

### Step 4：将 2D 高斯拟合到目标图像

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Red circle
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # Blue square
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

经过 200 步，64 个高斯会收敛到两个形状。这就是全部思想——对显式几何基元做梯度下降。

### Step 5：从 2D 到 3D

3D 扩展保持相同的循环。新增部分：

1. 每个高斯的旋转是四元数而非单个角度。
2. 协方差是 `R S S^T R^T`，其中 `R` 由四元数构建，`S = diag(exp(log_scale))`。
3. 投影 `(mu, Sigma) -> (mu', Sigma')` 使用相机外参和透视投影在 `mu` 处的雅可比矩阵。
4. 颜色变为球谐展开；在观察方向上求值。
5. 深度排序使用实际的相机空间 z 值，而非可学习标量。

每个生产级实现（`gsplat`、`inria/gaussian-splatting`、`nerfstudio`）都在 GPU 上用基于 tile 的 CUDA kernel 做完全相同的事情。

### Step 6：球谐函数求值

SH 基到 degree 3 有 16 项每通道。求值：

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   last dim is RGB channels
    dirs:      (..., 3)       unit vectors
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # degree 3 terms omitted here for brevity; full 16-coefficient version in the code file
    return result
```

学到的 `sh_coeffs` 存储了该高斯"每个方向的颜色"。渲染时对当前观察方向求值，得到一个 3 维 RGB 向量。

## 实际使用

对于真正的 3DGS 工作，使用 `gsplat`（Meta）或 `nerfstudio`：

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto` 是 nerfstudio 的 3DGS 训练器。典型场景在 RTX 4090 上运行 10-30 分钟。

2026 年重要的导出选项：

- `.ply` —— 原始高斯点云（可移植，文件最大）。
- `.splat` —— PlayCanvas / SuperSplat 量化格式。
- glTF `KHR_gaussian_splatting` —— Khronos 标准，跨查看器可移植（2026 年 2 月 RC）。
- OpenUSD `UsdVolParticleField3DGaussianSplat` —— USD 原生，用于 NVIDIA Omniverse 和 Vision Pro 管线。

对于 4D / 动态场景，`4DGS` 和 `Deformable-3DGS` 用时变的均值和不透明度扩展了同一套机制。

## 交付产出

本课产出：

- `outputs/prompt-3dgs-capture-planner.md` —— 一个 prompt，为给定场景类型规划拍摄方案（照片数量、相机路径、光照）。
- `outputs/skill-3dgs-export-router.md` —— 一个 skill，根据下游查看器或引擎选择正确的导出格式（`.ply` / `.splat` / glTF / USD）。

## 练习

1. **（简单）** 在不同的合成图像上运行上面的 2D splat 训练器。将 `num_splats` 设为 `[16, 64, 256]`，绘制每种情况下 MSE 随步数的变化曲线。找出收益递减的拐点。
2. **（中等）** 扩展 2D 光栅器，支持逐高斯 RGB 颜色通过 degree-2 谐波依赖于标量"观察角度"。在一对目标图像上训练，验证模型能重建两者。
3. **（困难）** 克隆 `nerfstudio`，在你手边任何场景（桌面、植物、人脸、房间）的 20 张照片上训练 `splatfacto`。导出为 glTF `KHR_gaussian_splatting` 并在查看器中打开（Three.js `GaussianSplats3D`、SuperSplat、Babylon.js V9）。报告训练时间、高斯数量和渲染帧率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 3DGS | "Gaussian splats" | 显式场景表示，由数百万个 3D 高斯组成，每个高斯有位置、旋转、尺度、不透明度、SH 颜色 |
| Covariance | "高斯的形状" | `Sigma = R S S^T R^T`；一个高斯的朝向和各向异性尺度 |
| Alpha compositing | "从后到前混合" | 与 NeRF 体渲染相同的方程，现在作用于显式稀疏集合 |
| Densification | "克隆与分裂" | 在重建欠拟合处自适应添加新高斯 |
| Pruning | "删除低不透明度" | 移除训练中不透明度降到接近零的高斯 |
| Spherical harmonics | "视角相关颜色" | 球面上的傅里叶基；将颜色存储为观察方向的函数 |
| Splatfacto | "nerfstudio 的 3DGS" | 2026 年训练 3DGS 最简单的路径 |
| `KHR_gaussian_splatting` | "glTF 标准" | Khronos 2026 扩展，使 3DGS 在查看器和引擎间可移植 |

## 延伸阅读

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — 原始论文
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) — 生产级 CUDA 光栅器
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — 参考训练方案
- [Khronos KHR_gaussian_splatting extension](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — 2026 可移植格式
- [OpenUSD 26.03 release notes](https://openusd.org/release/) — `UsdVolParticleField3DGaussianSplat` schema
- [THE FUTURE 3D State of Gaussian Splatting 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — 行业概览
