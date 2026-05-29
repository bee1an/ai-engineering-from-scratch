# Multi-Token Prediction (MTP)

> 从 GPT-2 到 Llama 3，所有自回归 LLM 在每个位置只训练一个损失：预测下一个 token。DeepSeek-V3 在每个位置加了第二个损失：预测再下一个 token。这额外的 14B 参数（在 671B 模型之上）通过梯度流蒸馏回主模型，训练好的 MTP head 在推理时被复用为 speculative decoding 的 drafter，接受率超过 80%。1.8 倍的生成吞吐量几乎是白送的。本课从 DeepSeek 技术报告出发，构建顺序式 MTP 模块，计算损失和共享 head 的参数布局，并解释为什么 MTP 保持了因果链，而 Gloeckle 等人最初的并行 MTP 打破了它。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 04 (pre-training a mini GPT), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## 学习目标

- 阐述 MTP 训练目标，推导跨预测深度的联合损失。
- 解释 Gloeckle 等人的并行 MTP heads（2024）与 DeepSeek-V3 的顺序式 MTP 模块之间的区别，以及为什么顺序设计保持了因果链。
- 计算在预训练中添加 MTP 模块带来的参数和显存开销。
- 从零实现一个 MTP 模块：共享 embedding、逐深度 transformer block、投影矩阵和共享输出 head。

## 问题

Next-token prediction 是标准的 LLM 训练目标。每个 hidden state 只被监督预测一件事：紧接着的下一个 token。这其实是一个相当弱的信号。序列中的大部分信息都超越了单个 token 的范围——结构、连贯性、事实性、算术推理流。模型必须通过在数万亿 token 上积累大量单 token 信号来学习这些。

MTP 问的是：如果每个 hidden state 同时被监督预测多个未来 token 呢？Gloeckle 等人（Meta，2024）证明了这有帮助。他们的实现在 backbone 顶部放了几个独立的输出 head，每个预测不同的偏移量。并行、简单，但这些 head 看到的是相同的 hidden state，没有任何层次化的精炼——而且预测之间没有因果链，所以不能用于 speculative decoding。

DeepSeek-V3（2024 年 12 月）将 MTP 重新设计为顺序模块，在每个预测深度保持因果链。模型从 `h_i^(0)` 预测 `t+1`，然后从一个新的 hidden state `h_i^(1)`（结合了 `h_i^(0)` 和 `E(t+1)` embedding）预测 `t+2`，以此类推。每个深度都有自己的小型 transformer block。共享 embedding 和共享输出 head 使参数开销保持适度。在 DeepSeek-V3 的规模下，MTP 模块在 671B 主模型权重之上额外增加了 14B 参数。这 2% 的开销换来了更密集的训练信号和推理时现成的 speculative decoding draft。

本课从零构建一个 MTP 模块和 D 深度损失。数学很简洁，实现只有 150 行。

## 概念

### 顺序式 MTP 方案

DeepSeek-V3 在主模型之上添加 `D` 个 MTP 模块。每个模块 `k`（`k = 1..D`）预测深度 `k` 处的 token——即给定前缀到位置 `i` 时预测 `t_{i+k}`。

模块 `k` 由以下部分组成：

- 一个 transformer block `T_k`，有自己的 attention 和 MLP。
- 一个投影矩阵 `M_k`，将前一深度的 hidden state 与下一深度 ground-truth token 的 embedding 结合。
- 共享 embedding `E`（与主模型相同）。
- 共享输出 head `Out`（与主模型相同）。

训练时，对于前缀到位置 `i`，逐深度的 hidden state 为：

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

逐深度的预测为：

```
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

逐深度的损失是对 ground-truth `t_{i+k}` 的交叉熵：

```
L_k = CE(logits_{i+k}, t_{i+k})
```

跨深度的联合损失：

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` 是一个小的权重因子——DeepSeek-V3 在训练前 10% 使用 0.3，之后使用 0.1。总训练损失为 `L_main + L_MTP`。

### 为什么是顺序式而非并行式

Gloeckle 最初的并行 MTP 有 D 个输出 head，每个直接作用于 `h_i^(0)`。每个 head 从相同的 backbone hidden state 预测 `t_{i+k}`。训练没问题，但预测之间没有条件依赖。你不能用 `head_1` 的输出来帮助 `head_2`——这些 head 是并行触发的。

DeepSeek-V3 的顺序设计从 `h_i^(k-1)` 加上实际的 next-token embedding `E(t_{i+k})` 构建 `h_i^(k)`。这保持了因果链：要预测 `t_{i+k+1}`，深度 `k+1` 的模块看到了 `t_{i+k}` 处的内容。这在结构上与自回归 decoder 消费自身输出的方式完全相同——使得 MTP 模块可以直接用作 speculative decoding 的 drafter。

推理时：将 `h_i^(k-1)` 和 drafted 的 `t_{i+k}` 送入模块 `k+1`，得到对 `t_{i+k+1}` 的预测。重复。这正是 EAGLE 风格的 draft，使用训练好的 MTP 模块作为 draft 网络。DeepSeek-V3 报告第一个 MTP 模块的接受率超过 80%，加速约 1.8 倍。

### 参数核算

对于 hidden 维度为 `h`、词表大小为 `V` 的模型：

- 主模型：数十亿参数，加上一个大小为 `V * h` 的输出 head。
- 共享输出 head：复用主模型的 head。无额外参数。
- 共享 embedding：复用主模型的 embedding。无额外参数。
- 每个 MTP 模块：
  - 投影 `M_k`：`(2h) * h = 2h^2`。
  - Transformer block `T_k`：attention（MHA 为 `4h^2`）加 MLP（SwiGLU 比例 8/3 时通常为 `8h^2`）。每个 block 约 `12h^2`。

每个模块的总额外参数：`~14h^2`。对于 DeepSeek-V3 的 `h = 7168`，D = 1 个模块：`~14 * 7168^2 = ~720M` 参数（纸面计算）。DeepSeek-V3 报告 14B——差异主要来自 MTP 模块中的 expert 层也是 MoE 结构。

### Speculative decoding 的回报

预训练期间，MTP 模块使训练速度降低约 10%（更多前向计算、额外损失）。回报是双重的：

1. 更密集的训练信号。每个 hidden state 看到 D+1 个监督目标。在 MMLU、GSM8K、MATH、HumanEval 上的测量效果：DeepSeek-V3 的消融实验中一致有几个百分点的提升。

2. 推理时免费的 speculative decoding draft。MTP 模块已经被训练来预测接下来的几个 token。复用为 draft 网络，它提供 80% 以上的接受率。在这个水平上，N=3 或 N=5 的 spec decoding 给出 1.8 倍吞吐量。10% 的训练时间成本在第一次运行推理时就收回了。

### 与 EAGLE 的关系

EAGLE 在预训练之后单独训练一个小型 draft 模型。MTP 将 draft 烘焙进预训练。两种方法在接受率上趋于相似，但通过不同的流程：

| 维度 | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| When trained | Post-pre-training | During pre-training |
| Backward-compatible with existing weights | Yes | No (need to re-train) |
| Draft params | 1-2 transformer layers | 1 transformer block + projection |
| Acceptance rate | 0.88-0.92 | 0.80+ at depth 1 |
| Benefit beyond speedup | Speculative decoding only | Denser training signal + speedup |

## 构建

`code/main.py` 端到端构建一个 MTP 模块：共享 embedding、投影、transformer block、共享输出 head。然后在一个短的合成序列上计算逐深度交叉熵损失，并按组件打印参数数量。使用 32 个 token 的玩具词表使数字易读。

### Step 1：共享 embedding 表

一个 `vocab_size x hidden` 的表被主模型和每个 MTP 模块在每个深度共用。不是第二份拷贝——字面上是同一个 tensor。

### Step 2：逐深度的组合

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

真实的 DeepSeek-V3 将两个 RMSNorm 后的向量拼接为 `[2h]`，然后用一个 `h x 2h` 的矩阵投影。玩具版本为了 stdlib 简洁性使用向量加法。

### Step 3：深度 k 的 transformer block

Self-attention 加 MLP。在玩具中，一个单层线性 attention block 和一个 SwiGLU MLP 保持结构可见，无需 numpy。

### Step 4：共享输出 head

复用主模型的输出投影。词表上的 logits。

### Step 5：逐深度损失

softmax(logits) 对偏移 `k` 处 ground-truth token 的交叉熵。用 `lambda / D` 缩放因子跨深度聚合。

### Step 6：参数核算

打印总参数数量、共享（embedding、head）数量和每模块额外数量。展示 MTP 额外参数与主模型大小的比率。

## 使用

MTP 已集成到 DeepSeek-V3（2024 年 12 月）和 DeepSeek-R1 系列中。推理时：

- DeepSeek 自己的 serving 栈开箱即用地将 MTP 模块作为 speculative decoder 消费。
- vLLM 和 SGLang 截至 2026 年 4 月已有 DeepSeek-V3 MTP 的集成路径。
- AMD 的 ROCm SGLang 教程展示了一个特定的 MTP speculative decoding 配置，在 V3 checkpoint 上测得 1.8 倍加速。

何时在新的预训练中使用 MTP：

- 你控制完整的预训练流程，想要积累更密集的训练信号。
- 你知道将大规模服务该模型，想要免费的 speculative decoding。
- 你的 hidden size 至少为 4096。在 1B 规模下，开销带来的伤害大于收益。

何时不用：

- 微调现有的预训练 dense 模型。MTP 模块未经训练。
- 研究模型中你想要一个干净的 baseline 来对比。MTP 改变了架构。

## 交付

本课产出 `outputs/skill-mtp-planner.md`。给定预训练运行规格（模型大小、数据、算力），它返回集成 MTP 的计划：深度数 D、`lambda` 调度、显存开销，以及推理时的 speculative decoding 接线方式。

## 练习

1. 运行 `code/main.py`。展示随着合成信号增强，逐深度损失单调递减。修改合成数据使用固定模式，验证 depth-1 和 depth-2 损失都收敛。

2. 计算一个 dense 70B 模型（hidden 8192，80 层）加 D=1 MTP 模块的参数开销。与 DeepSeek-V3 报告的 14B 开销对比。解释为什么 DeepSeek 的数字更高：MTP transformer block 继承了相同的 MoE 结构，膨胀了每模块参数数量。

3. 在玩具中实现 D=2：添加第二个 MTP 模块，接收 h^(1) 并预测 `t_{i+2}`。验证联合损失和参数核算与 DeepSeek 论文的公式 19-21 匹配。

4. 将玩具切换为并行 MTP（Gloeckle 风格）：在主 hidden state 之上添加 D 个输出 head，每个预测不同的偏移。在相同合成信号上测量逐深度损失与顺序版本的对比。顺序版本在 k > 1 时应产生更低的 depth-k 损失，因为它以中间预测为条件。

5. 将训练好的 MTP 模块用作 EAGLE 风格的 draft：调用模块 k 在推理时提议 `t_{i+k}`。在 held-out 序列上测量这些 draft token 相对于主模型预测的接受率。如果在玩具上达到 50% 以上，你就复现了经验性的 MTP-as-draft 特性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| MTP module | "Extra loss block" | 一个小型 transformer block 加投影，预测主模型前方 `k` 个位置的 token |
| Prediction depth | "Which offset" | 整数 `k`，使得模块 `k` 从前缀到位置 `i` 预测 `t_{i+k}` |
| Parallel MTP | "Gloeckle-style" | D 个独立 head 作用于相同的 backbone hidden state，无条件链 |
| Sequential MTP | "DeepSeek-V3 style" | 每个模块以前一深度的 hidden state 加 next-token embedding 为条件；保持因果链 |
| Shared output head | "Reuse the main head" | MTP 模块调用主模型的 LM head，而非单独的输出投影 |
| Shared embedding | "Reuse the main table" | 相同的词表 embedding 表到处使用；无重复参数 |
| Projection matrix M_k | "Combine hidden + next-token" | 一个 `h x 2h` 线性层，将前一 hidden state 和目标 token embedding 折叠为下一深度的输入 |
| Joint loss L_MTP | "Averaged extra losses" | 逐深度交叉熵损失的算术平均，乘以 `lambda` 缩放 |
| Acceptance rate at depth 1 | "How often MTP draft is right" | D=1 MTP 模块的 top-1 预测等于主模型 top-1 预测的比率；DeepSeek-V3 上 80%+ |
| Lambda weighting | "Extra-loss importance" | 逐深度缩放因子；DeepSeek-V3 训练初期 0.3，后期 0.1 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的顺序式 MTP 描述（Section 2.2），包括联合损失公式和推理时 1.8 倍加速
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) — DeepSeek 设计所改进的并行 MTP baseline
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 685B 总量（671B 主模型 + 14B MTP），部署说明
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — MTP 所适配的 speculative decoding 框架
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — EAGLE 的 2025 draft 架构，MTP 的竞争对手
