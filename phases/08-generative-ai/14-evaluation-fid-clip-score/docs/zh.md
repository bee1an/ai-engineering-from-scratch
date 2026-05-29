# 评估 — FID、CLIP Score、人类偏好

> 每一份生成模型的排行榜都会引用 FID、CLIP score，以及来自人类偏好竞技场的胜率。每个数字都有一种失败模式，被有心的研究者拿来"刷榜"。如果你不了解这些失败模式，就无法分辨哪些是真正的改进，哪些只是刷榜的结果。

**类型：** Build
**语言：** Python
**前置：** Phase 8 · 01（分类体系）、Phase 2 · 04（评估指标）
**用时：** 约 45 分钟

## 问题

生成模型要在两件事上接受评判：*样本质量* 和 *条件遵从度*。两者都没有闭式的度量方式。你的模型要渲染 1 万张图片；某个东西得给这些图片打分；你还得相信这些数字在不同模型族、不同分辨率、不同架构之间是可比的。从 2014 年到 2026 年，活下来的指标只有三个：

- **FID（Fréchet Inception Distance）。** 在 Inception 网络的特征空间里，衡量真实分布与生成分布之间的距离。越低越好。
- **CLIP score。** 生成图片的 CLIP-image 嵌入与提示词的 CLIP-text 嵌入之间的余弦相似度。越高越好，衡量提示词遵从度。
- **人类偏好。** 让两个模型在同一条 prompt 上正面对决，让人类（或一个 GPT-4 级别的模型）挑出更好的那张，再聚合成 Elo 分数。

你还会看到：IS（inception score，基本退役）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每一个都修正了上一代的某个失败模式。

## 概念

![FID、CLIP 和偏好：三条轴，不同的失败模式](../assets/evaluation.svg)

### FID — 样本质量

Heusel 等人（2017）。步骤如下：

1. 对 N 张真实图片和 N 张生成图片提取 Inception-v3 特征（2048 维）。
2. 给每个样本池拟合一个高斯分布：计算均值 `μ_r, μ_g` 和协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：特征空间里两个多元高斯分布之间的 Fréchet 距离。越低 = 两个分布越相似。

失败模式：
- **小 N 下有偏。** FID 是在特征分布上做均方度量 — N 太小会低估协方差，给出虚低的 FID。永远使用 N ≥ 10,000。
- **依赖 Inception。** Inception-v3 是在 ImageNet 上训练的。离 ImageNet 较远的领域（人脸、艺术、文字图像）算出来的 FID 没有意义。要用领域特定的特征提取器。
- **可被刷榜。** 过拟合 Inception 的先验能拿到很低的 FID，但视觉质量并没有改进。用下面的 CMMD 来对抗。

### CLIP score — 提示词遵从度

Radford 等人（2021）。对于一张生成图片 + prompt：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

在 3 万张生成图片上取均值 → 得到一个可在不同模型间比较的标量。

失败模式：
- **CLIP 自身的盲区。** CLIP 的组合推理能力很弱（"红色立方体放在蓝色球体上"经常失败）。模型可以在 CLIP score 上排名很高，却并未真正遵循复杂 prompt。
- **短 prompt 偏置。** 短 prompt 在野外有更多 CLIP-image 匹配。长 prompt 在机制上 CLIP score 就更低。
- **prompt 刷榜。** 在 prompt 里加上 "high quality, 4k, masterpiece" 会推高 CLIP score，但并没有改善图文绑定。

CMMD（Jayasumana 等人，2024）修了一些上述问题：用 CLIP 特征代替 Inception，用最大均值差异（maximum-mean discrepancy）代替 Fréchet。在察觉细微质量差异上更敏锐。

### 人类偏好 — 真正的 ground truth

挑一组 prompt。用模型 A 和模型 B 各生成一份。把成对结果展示给人类（或一个强大的 LLM 评委）。把胜负聚合成 Elo 或 Bradley-Terry 分数。常用 benchmark：

- **PartiPrompts（Google）**：1,600 条多样化 prompt，12 个类别。
- **HPSv2**：10.7 万条人类标注，被广泛当作自动化代理。
- **ImageReward**：13.7 万对 prompt-图像偏好对，MIT 许可证。
- **PickScore**：在 Pick-a-Pic 的 260 万条偏好上训练。
- **Chatbot-Arena 风格的图像竞技场**：https://imagearena.ai/ 等。

失败模式：
- **评委方差。** 非专家的偏好和专家不同。两者都用。
- **prompt 分布。** 精心挑选的 prompt 偏向某个模型族。一定要把 prompt 集合记录在案。
- **LLM 评委被奖励黑客。** GPT-4 评委会被"漂亮但错误"的输出蒙骗。要和人类评委做三角验证。

## 三者合用

一份生产级的评估报告应当包含：

1. 在 1–3 万样本上、对照一份留出真实分布算 FID（样本质量）。
2. 在同一批样本上对照其 prompt 算 CLIP score / CMMD（遵从度）。
3. 与上一代模型在盲评竞技场上的胜率（整体偏好）。
4. 失败模式分析：随机抽 50 张输出，对照已知问题打标（手部解剖、文字渲染、物体计数一致性）。

任何单一指标都是谎言。三个相互印证的指标 + 定性审阅，才能算一个声明。

## 动手实现

`code/main.py` 在合成的"特征向量"上实现了 FID、类 CLIP-score 和 Elo 聚合（我们用 4 维向量来代替 Inception 特征）。你会看到：

- 在小 N 和大 N 上分别计算 FID — 偏差就在这里。
- 把"CLIP score"实现为特征池之间的余弦相似度。
- 在合成偏好流上的 Elo 更新规则。

### 第 1 步：四行写出 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 第 2 步：CLIP 风格的余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 第 3 步：Elo 聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 陷阱

- **N=1000 的 FID。** 在 N=10k 以下，启发式不可靠。报告低 N FID 的论文都在刷榜。
- **跨分辨率比较 FID。** Inception 的 299×299 缩放会改变特征分布。只能在分辨率匹配的条件下比较。
- **只跑一个 seed。** 至少跑 3 个 seed，并报告标准差。
- **通过负向 prompt 拉高 CLIP score。** 有些 pipeline 通过过拟合 prompt 来推高 CLIP，要检查是否出现视觉饱和。
- **prompt 重叠造成的 Elo 偏置。** 如果两个模型都在训练时见过某条 benchmark prompt，那么 Elo 毫无意义。要用留出的 prompt 集合。
- **付费众包人评偏置。** Prolific 和 MTurk 标注员偏年轻、偏 tech-friendly。要混入招募的艺术 / 设计专家。

## 实战使用

2026 年的生产评估协议：

| 维度 | 最低要求 | 推荐配置 |
|--------|---------|-------------|
| 样本质量 | 在 10k 上对照留出真实集算 FID | + 5k 上的 CMMD + 各类别子集上的 FID |
| 提示词遵从 | 30k 上的 CLIP score | + HPSv2 + ImageReward + VQA 式问答 |
| 偏好 | 与 baseline 做 200 对盲评 | + 2000 对人评 + LLM 评委 + Chatbot Arena |
| 失败分析 | 50 张人工标注 | 500 张人工标注 + 自动安全分类器 |

四个支柱都齐 = 声明。任一支柱单干 = 营销话术。

## 交付

保存 `outputs/skill-eval-report.md`。Skill 接收一个新模型 checkpoint + 一个 baseline，输出一份完整的评估方案：样本量、指标、失败模式探针、放行标准。

## 练习

1. **简单。** 运行 `code/main.py`。在同一组合成分布上，比较 N=100 与 N=1000 时的 FID。报告偏差量级。
2. **中等。** 用合成的 CLIP 风格特征实现 CMMD（公式见 Jayasumana 等人，2024）。比较它对质量差异的敏感度，相对 FID 如何。
3. **困难。** 复现 HPSv2 的设置：从 Pick-a-Pic 的子集取 1000 对图像-prompt，在偏好上微调一个小型 CLIP 评分器，并在留出集上测量它与人类的吻合度。

## 关键术语

| 术语 | 大家挂在嘴边的说法 | 实际含义 |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | 真实和生成 Inception 特征做高斯拟合后的 Fréchet 距离。 |
| CLIP score | "图文相似度" | CLIP image 嵌入和 text 嵌入之间的余弦相似度。 |
| CMMD | "FID 的接班人" | 用 CLIP 特征做 MMD；偏差更小，不假设高斯分布。 |
| IS | "Inception score" | Exp KL(p(y|x) \|\| p(y))；在现代模型上相关性差，已退役。 |
| HPSv2 / ImageReward / PickScore | "学到的偏好代理" | 在人类偏好上训练的小模型，被用作自动化评委。 |
| Elo | "国际象棋评分" | 对两两胜负做 Bradley-Terry 聚合。 |
| PartiPrompts | "标准 benchmark prompt 集" | Google 整理的 1,600 条 prompt，覆盖 12 类。 |
| FD-DINO | "自监督版替代品" | 用 DINOv2 特征算 FD；在非 ImageNet 领域更靠谱。 |

## 生产备注：评估也是一种推理负载

在 1 万样本上跑 FID 意味着要生成 1 万张图。对于在单卡 L4 上、1024² 分辨率、50 步的 SDXL base，这是大约 11 小时的单请求推理。评估预算是真实存在的，而它的画面正是离线推理场景（最大化吞吐，忽略 TTFT）：

- **死命堆 batch，别管延迟。** 离线评估 = 用内存允许的最大 batch 做 static batching。`pipe(...).images` 在 80GB H100 上设 `num_images_per_prompt=8`，墙钟时间比单请求快 4-6 倍。
- **缓存真实特征。** 真实参考集上的 Inception（FID）或 CLIP（CLIP-score、CMMD）特征提取只跑一次，存为 `.npz`。不要每次评估都重算。

对于 CI / 回归门禁：每个 PR 在 500 样本子集上跑 FID + CLIP score（约 30 分钟）；夜间跑完整的 10k FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失败模式综述。
