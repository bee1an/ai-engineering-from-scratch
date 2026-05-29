# 缩放定律

> 2020 年 Kaplan 论文说：模型越大，损失越低。2022 年 Hoffmann 论文说：你训练不足了。算力分配到两个桶——参数和 token——而分配比例并不显然。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## 问题

当你有 C FLOPs 的训练算力并想要最好的模型时，你面对两个旋钮：

1. **多少参数（N）？** 更大的模型，更高的容量。
2. **多少训练 token（D）？** 更多数据，更好地利用容量。

FLOPs 大约按 `6 × N × D` 缩放。你可以把 N 推高、D 降低，或者 D 推高、N 降低。哪个更好？

2022 年之前，答案是"使劲推 N"。GPT-3（2020）是 175B 参数训练了 ~300B token。比例约 1.7 token/参数。Kaplan 缩放定律支持这个结论。

Hoffmann et al.（2022）训练了一系列叫 Chinchilla 的模型，发现了不同的结论：最优比例接近**每参数 20 个 token**。GPT-3 训练不足 10 倍。Chinchilla（70B 参数，1.4T token）在每个基准上都击败 GPT-3（175B，300B token），推理成本还低 2.5 倍。

2026 年是 Chinchilla 的世界——但有一个重要转折。Llama 3 8B 在 15 万亿 token 上训练，比例为每参数 1,875 个 token。超过 Chinchilla 最优 94 倍。对于大规模使用的模型，推理成本比训练成本更重要，所以过度训练（超过 Chinchilla）以获得更小的可部署模型是 2026 年的默认做法。

## 概念

![Chinchilla curves: loss vs compute at various N/D ratios](../assets/scaling-laws.svg)

### Hoffmann 定律

来自 Chinchilla 论文，损失遵循：

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = 参数（非 embedding）。
- `D` = 训练 token。
- `α ≈ 0.34`，`β ≈ 0.28`（大致对称）。
- `E ≈ 1.69`，不可约损失上限。
- `A ≈ 406`，`B ≈ 411`。

两项在扩展时相互权衡。在固定算力（C = 6ND）下对 `N` 求导并求解：

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

算力最优：每参数 20 个 token。

### 为什么还要过度训练

Chinchilla 最优最小化每训练 FLOP 的训练损失。但训练成本只付一次；推理成本永远在付。

对于一个每月服务一万亿 token 的聊天机器人，推理主导总成本。Llama 的方法：训练更小的模型，训练更久。8B 在 15T token 上是深度推理优化的：

- 能跑在消费级 GPU 上。
- 延迟是 70B Chinchilla 最优的几分之一。
- 质量对大多数任务足够接近。

DeepMind 2024 年的论文（"Over-training is the new optimal"）形式化了这一点。对于推理主导的工作负载，正确比例接近每参数 100–500 个 token，取决于服务量。

### 涌现 vs 平滑

说法：某些能力（算术、多步推理、思维链跟随）在某个规模"突然涌现"。

Schaeffer et al.（2023）论证这是测量伪影：涌现指标使用不连续评分（精确匹配、阈值准确率），掩盖了底层 logits 的平滑改进。连续指标（交叉熵）显示平滑曲线。

2026 年的共识是：通过连续损失的预测是可靠的。基准跳跃通常是评分器伪影。根据连续指标规划预算。

### 2026 年的图景

缩放定律仍然有效，但：

| Factor | Changed how |
|--------|-------------|
| Data quality | Curating "good" tokens (Phi-style) shifts curves by >2× effective compute |
| MoE | Total params decouple from active FLOPs; scaling laws per-active-FLOP |
| Post-training | Some capabilities (instruction following, code) shift with SFT+RLHF more than pretraining |
| Multimodality | Image + text tokens scale together; separate curves per modality |
| Synthetic data | Models generate training data; effective compute can compound |

Muon 优化器（Kimi Moonlight, 2024）在匹配数据下展示了相比 AdamW ~2 倍的有效算力增益。一些 2026 年的训练运行默认使用 Muon。改变缩放定律中的绝对常数，不改变其形状。

## 动手构建

见 `code/main.py`。我们实现 Chinchilla 损失方程，并在多个算力预算下求解算力最优的 `(N, D)`。

### 第 1 步：Chinchilla 损失

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

在固定 `C = 6ND` 下将 `L` 画成 `(N, D)` 的等高线图。找到最小值。

### 第 2 步：算力最优前沿

对于从 `1e17` 到 `1e25` FLOPs 的算力预算，找到在 `6ND = C` 约束下最小化损失的 `(N, D)`。验证比例 `D/N ≈ 20`。

### 第 3 步：过度训练的代价

计算训练一个 10 倍小的模型（最优 N 的 1/10，最优 D 的 10 倍）所付出的额外损失。报告推理 FLOP 节省（与 N 成比例）作为交换。

### 第 4 步：与真实模型对比

代入已知的 `(N, D)` 对：GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（激活参数），对比预测损失与报告损失。

## 使用方式

你不太可能自己训练前沿模型。但缩放定律告诉你：

1. **你的微调数据是否足够。** 如果你的任务特定数据低于基础模型每参数 20 个 token，预期会在某个损失下限饱和。
2. **是否该选更大的基础模型。** 如果你的预算全花在推理上，优先选更小、训练更久的模型。
3. **收益递减在哪里。** 超过 1000 倍 Chinchilla 最优后，log-loss 变化变成噪声。

**2026 年的研究方向：**

- **数据受限区间。** 网络上高质量 token 数量有限（过滤后 ~5–10 万亿英文）。前沿预训练正在接近这个天花板。合成数据、多语言、多模态和 RLHF 规模化微调是下一个杠杆。
- **算力倍增技巧。** Muon 优化器、MoE、更好的数据筛选——每个都偏移绝对常数，不改变渐近线。
- **RL 的缩放定律。** 开放问题。早期证据表明 RL 样本中存在幂律，但指数与预训练非常不同。

## 交付产出

见 `outputs/skill-training-budget-estimator.md`。该 skill 根据算力预算、部署约束和目标损失，为新训练运行选择 `(N, D, hours, GPU)`。

## 练习

1. **简单。** 运行 `code/main.py`。打印算力预算 `1e20`、`1e22`、`1e24` 的 Chinchilla 最优 `(N, D)`。与真实模型表对比。
2. **中等。** 实现 Hoffmann 损失-算力曲线。画出算力最优前沿的 loss vs `log10(C)`。找出定律预测需要 `>10^28` FLOPs 才能再降 0.1 交叉熵的点。
3. **困难。** 在同一数据集上训练 5 个小模型（100K 到 10M 参数），拟合你自己的缩放定律。估计 `α` 和 `E`。你的指数与已发表的匹配程度如何？

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Parameters (N) | "Model size" | Non-embedding weight count; determines capacity. |
| Tokens (D) | "Training data" | Number of training tokens seen; determines how well the parameters get used. |
| Compute (C) | "FLOPs spent" | Approximately `6 × N × D` for a standard transformer. |
| Chinchilla-optimal | "D/N ≈ 20" | Ratio that minimizes loss per FLOP of pretraining. |
| Over-training | "Past Chinchilla" | Spend extra training FLOPs to save inference FLOPs; D/N >> 20. |
| Irreducible loss | "The floor" | The `E` term in the scaling law; the entropy of the data itself. |
| Emergent capability | "Sudden jumps at scale" | Often a scorer artifact; continuous loss is smooth. |
| Effective compute | "Training-efficiency multiplier" | Better data / optimizer / architecture multiplies how far a FLOP goes. |

## 延伸阅读

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — the first scaling law paper; undertrained.
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla.
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — emergence as measurement artifact.
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — why Llama's over-training is right for its workload.
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 2× compute multiplier.
