# LLaVA 与 Visual Instruction Tuning

> LLaVA（2023 年 4 月）是地球上被复制最多的多模态架构。它用 2 层 MLP 替换了 BLIP-2 的 Q-Former，用朴素的 token 拼接替换了 Flamingo 的 gated cross-attention，并在 GPT-4 从纯文本描述生成的 158k 视觉指令对话上训练。2023 到 2026 年间任何构建 VLM 的从业者都构建了某种 LLaVA 变体。LLaVA-1.5 加了 AnyRes。LLaVA-NeXT 提升了分辨率。LLaVA-OneVision 将图像、多图和视频统一到一个方案中。本课解读这个方案，实现 projector，并解释为什么"更简单的赢了"。

**类型：** 构建
**语言：** Python（标准库，projector + 指令模板构建器）
**前置：** Phase 12 · 02（CLIP）、Phase 11（LLM 工程——指令微调）
**时间：** ~180 分钟

## 学习目标

- 构建一个 2 层 MLP projector，将 ViT patch embedding（dim 1024）映射到 LLM 的 embedding dim（dim 4096）。
- 走通 LLaVA 两阶段方案：(1) 在 558k 描述对上做 projector 对齐，(2) 在 158k GPT-4 生成的对话上做 visual instruction tuning。
- 构造一个 LLaVA 格式的 prompt，包含图像 token 占位符、system prompt 和 user/assistant 轮次。
- 解释为什么社区从 Q-Former 转向了 MLP，尽管 Q-Former 在 token 预算上有优势。

## 问题

BLIP-2 的 Q-Former（课程 12.03）将图像压缩到 32 个 token。干净、高效、基准表现好。但它有两个问题。

第一，Q-Former 是可训练的，但它的损失不是最终任务。阶段 1 训练 ITC+ITM+ITG。阶段 2 训练 LM loss。Query 学到的是某种中间表示，LLM 然后必须解码。信息在瓶颈中丢失。

第二，Q-Former 有 188M 参数，在 LLaVA 2023 年的规模下你必须针对目标 LLM 联合设计它。换 LLM，重训 Q-Former。换视觉编码器，重训。每种组合都是一个独立的研发项目。

LLaVA 的答案简单得令人尴尬：取 ViT 的 576 个 patch token，每个通过 2 层 MLP（`1024 → 4096 → 4096`），然后把所有 576 个倒入 LLM 的输入序列。没有瓶颈。没有阶段 1 的奇怪目标预训练。直接用 LM loss 训练 MLP。

数据从哪来？LLaVA 的第二个洞察：用 GPT-4（纯文本）生成指令数据。把 COCO 图像的描述和边界框数据喂给 GPT-4，让它产生对话、描述和复杂推理问题。免费获得 158k 指令-回复对。无需人工标注。

结果：一个在 8 块 A100 上训练一天的 VLM，在 MMMU 上超过 Flamingo，并发布了社区可以扩展的开源 checkpoint。到 2023 年底它已经衍生出 50+ 个分支。

## 概念

### 架构

LLaVA-1.5 13B 版：
- 视觉编码器：CLIP ViT-L/14 @ 336（阶段 1 冻结，阶段 2 可选解冻）。
- Projector：2 层 MLP，GELU 激活，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来是 Llama-3.1-8B）。

图像 + 文本 prompt 的前向传播：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占据 LLM 上下文的 576 个 token。在 2048 上下文下，剩余 1472 个 token 给文本。在 32k 上下文下，这是舍入误差。

### 阶段 1：Projector 对齐

冻结 ViT。冻结 LLM。只训练 2 层 MLP。数据集：558k 图文对（LAION-CC-SBU）。损失：以投影后的图像 token 为条件，对描述做语言建模。

在 batch 128 下单个 epoch 几小时就完成。Projector 学会将 ViT 空间映射到 LLM 空间。无任务特定监督。

### 阶段 2：Visual instruction tuning

解冻 projector（仍可训练）。解冻 LLM（通常全量，有时 LoRA）。在 158k 视觉指令对话上训练。

指令数据是关键。Liu et al. 的生成方式：
1. 取一张 COCO 图像。
2. 提取文本描述（5 条人工描述 + 边界框列表）。
3. 发送给 GPT-4，使用三种 prompt 模板：
   - 对话："Generate a back-and-forth dialogue between a user and assistant about this image."
   - 详细描述："Give a rich, detailed description of the image."
   - 复杂推理："Ask a question that requires reasoning about the image, then answer it."
4. 将 GPT-4 的输出解析为（指令，回复）对。

这些都不直接接触图像——只有文本描述。GPT-4 幻想出合理的图像内容。有些噪声，但有效：158k 对话就足以解锁对话能力。

### 为什么社区复制了这个

- 没有阶段 1 特定的损失需要调。全程 LM loss。
- Projector 几小时就训完，不需要几天。
- LLM 可以替换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重训 projector。
- 视觉指令数据流水线用 GPT-4，为新领域重新生成成本低。

### LLaVA-1.5 和 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）新增：
- 学术任务数据（VQA、OKVQA、RefCOCO）混入指令微调。
- 更好的 system prompt。
- 2048 → 32k 上下文。

LLaVA-NeXT（2024 年 1 月）新增：
- AnyRes：将高分辨率图像切成 2x2 或 1x3 的 336x336 crop 网格，加一个全局低分辨率缩略图。每个 crop 变成 576 个 token；每张图总共约 2880 个视觉 token。OCR 和图表任务大幅提升。
- 更好的指令数据混合，使用 ShareGPT4V（高质量 GPT-4V 描述）。
- 更强的基础 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

课程 12.08 详细介绍 OneVision。简短版本：相同的 projector，但用课程学习训练，在一个模型中覆盖单图、多图和视频，共享视觉 token 预算。

### 与 Q-Former 的比较

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| 每张图视觉 token | 32 | 576（基础）或 2880（AnyRes） |
| 可训练参数 | 188M + LM | 40M + LM |
| 阶段 1 损失 | ITC+ITM+ITG | 仅 LM |
| LLM 即插即用 | 需要重训 | 最小重训即可替换 |
| 多图 | 别扭 | 自然（拼接） |
| 视频 | 别扭 | 自然（逐帧拼接） |
| Token 预算 | 小 | 大 |

MLP 在简洁性和 token 灵活性上胜出。Q-Former 在 token 预算上胜出。到 2023 年底 token 预算不再是约束（LLM 上下文增长到 32k-128k+），简洁性占了主导。

### Prompt 格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是占位符 token。在 tokenization 之前，它被替换为 576 个视觉 token（或 AnyRes 的 2880 个）。Tokenizer 看到的序列比训练时稍长，但 LLM 能处理这个新输入，因为阶段 1 教会了它。

### 参数经济学

LLaVA-1.5-7B 分解：
- CLIP ViT-L/14 @ 336：303M（阶段 1 冻结，阶段 2 通常解冻）。
- Projector（2x linear）：~22M 可训练。
- Llama-7B：7B。
- 总计：7.3B 参数。阶段 2 可训练：完整 7B + 22M projector。

阶段 2 训练成本：8xA100 约 20 小时。这是关键数字——一天，一个节点，可复现。这就是 LLaVA 传播的原因。

## 动手用

`code/main.py` 实现了：

1. 2 层 MLP projector（玩具规模 dim 16 → 32 → 32），纯 Python。
2. Prompt 构建流水线：system prompt + `<image>` 替换为 N 个投影 token + user 轮次 + assistant 生成占位符。
3. 576-token 视觉块在 LLM 上下文中的可视化（占 2k / 32k / 128k 上下文的百分比）。

## 交付物

本课产出 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA 家族 checkpoint，它运行一个 10-prompt vibes-eval 套件（3 个描述、3 个 VQA、2 个推理、2 个拒绝）并报告人类可读的评分卡。不是基准；是确认 projector 和 LLM 连接良好的冒烟测试。

## 练习

1. 计算 `1024 → 4096 → 4096` 的 2 层 MLP projector 的可训练参数量。带 GELU 和 bias，它占 LLaVA-13B 的多少比例？

2. 为一个"拒绝"案例构造 LLaVA prompt——图像包含一个私人个体。写出预期的 assistant 回复。为什么 LLaVA 应该 zero-shot 拒绝这个，需要什么训练数据来强化拒绝？

3. 阅读 LLaVA-NeXT 博客的 AnyRes 部分。计算 1344x672 图像在 AnyRes 下的视觉 token 数。与 336x336 基础的 576 个 token 比较。

4. LLaVA 阶段 1 的 projector 用描述上的 LM loss 训练。如果跳过阶段 1 直接进入阶段 2（visual instruction tuning）会怎样？引用 Prismatic VLMs 消融实验（arXiv:2402.07865）的答案。

5. LLaVA-Instruct-150k 用 GPT-4 和 COCO 描述生成指令。对于一个新领域（医学 X 光、卫星图像），描述生成领域指令的四步数据流水线。每步可能出什么问题？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Projector | "MLP 桥梁" | 带 GELU 的 2 层 MLP，将 ViT dim 映射到 LLM dim |
| Image token | "<image> 占位符" | 推理前被 N 个投影视觉 token 替换的 prompt 标记 |
| Visual instruction tuning | "LLaVA 阶段 2" | 在 GPT-4 生成的（图像，指令，回复）三元组上训练 |
| Stage 1 alignment | "Projector 预训练" | 冻结 ViT 和 LLM，用描述上的 LM loss 训练 projector |
| AnyRes | "多 crop 切片" | 将高分辨率图像切成 tile 网格并拼接每个 tile 的视觉 token |
| LLaVA-Instruct | "GPT-4 生成的" | 从 COCO 描述 + GPT-4 合成的 158k 指令-回复对 |
| Vision encoder freeze | "骨干锁定" | CLIP 权重在阶段 1 不更新，有时阶段 2 也不更新 |
| ShareGPT4V | "更好的描述" | GPT-4V 生成的 100 万条密集描述，用于更高质量的对齐 |
| VQA | "视觉问答" | 回答关于图像的自由形式问题的任务 |
| Prismatic VLMs | "设计空间论文" | Karamcheti 2024 消融实验，系统测试 projector 和数据选择 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 密集描述数据集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 设计空间消融。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一单图、多图、视频。
