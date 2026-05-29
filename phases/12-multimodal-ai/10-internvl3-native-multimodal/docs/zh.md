# InternVL3：原生多模态预训练

> 在 InternVL3 之前，所有开源 VLM 都遵循同一套三步流程：拿一个在数万亿文本 token 上训练好的文本 LLM，接上一个视觉编码器，然后微调接缝处。这种方法可行，但存在对齐负债——文本 LLM 把全部预训练预算花在了纯文本上，并不原生理解视觉 token。当你事后加入视觉时，LLM 必须重新学习如何将视觉输入与其文本推理关联起来，同时还不能遗忘文本能力。InternVL3（Zhu et al., 2025 年 4 月）拒绝了这种事后方案：一次预训练，文本和多模态从第一步就交织在一起。结果是在 78B 参数的开源模型上匹配了 Gemini 2.5 Pro 在 MMMU-Pro 上的表现。本课阅读原生预训练的论证，以及当你采用这种方式时会发生什么变化。

**Type:** Learn
**Languages:** Python (stdlib, training-corpus mixer)
**Prerequisites:** Phase 12 · 05, Phase 12 · 07 (recipes)
**Time:** ~120 minutes

## 学习目标

- 解释为什么事后 VLM 训练会积累对齐负债，引用三个可测量的症状（灾难性遗忘、回答漂移、视觉-文本不一致）。
- 描述 InternVL3 的原生预训练语料配比，以及为什么 text : interleaved : caption 的比例很重要。
- 比较 V2PE（可变视觉位置编码）与 Qwen2-VL 的 M-RoPE。
- 说出 Visual Resolution Router (ViR) 和 Decoupled Vision-Language (DvD) 部署优化方案。

## 问题

事后 VLM 训练是默认做法。LLaVA、BLIP-2、Qwen-VL、Idefics——都是拿一个已经预训练好的 LLM（Llama、Vicuna、Qwen、Mistral）然后加上视觉。训练阶段通常如下：

1. 冻结 LLM + 冻结视觉编码器 + 可训练投影层，在 caption 对上训练以对齐 embedding。
2. 解冻 LLM，在指令数据上训练（LLaVA-Instruct、ShareGPT4V）。
3. 可选的任务特定微调。

对齐负债的三个症状：

- 灾难性遗忘。事后 VLM 会遗忘纯文本能力。GSM8K 分数下降 5-10 分。Hellaswag 分数下降。纯文本 agent 退化。
- 回答漂移。同一个视觉问题的微小措辞变化会得到不同答案。视觉编码器与 LLM 的连接比 LLM 自身 token 之间的绑定更弱。
- 视觉-文本不一致。VLM 可以正确描述一张图片，然后在回答问题时与自己的描述矛盾。视觉 token 不像文本那样参与 LLM 的内部一致性检查。

这些症状有充分文档记录。MM1.5 第 4 节量化了它们。LLaVA-OneVision 的消融实验暗示了它们。原生预训练是答案。

## 概念

### 原生多模态预训练

InternVL3 从头开始在一个从第一步就是原生多模态的语料上训练。配比为：

- 40% 纯文本数据（FineWeb、Proof-Pile-2 等）
- 35% 交织图文数据（OBELICS、MMC4 风格）
- 20% 配对图文 caption 数据
- 5% 视频-文本数据

视觉 token、文本 token 和跨模态交互从第一个梯度步就参与同一个 loss。没有对齐预训练，没有投影层冻结阶段，没有需要恢复的灾难性遗忘。

基础模型的训练是单阶段的。指令微调随后进行，但基础模型已经将视觉 token 视为一等公民。

### V2PE（可变视觉位置编码）

Qwen2-VL 使用固定轴分配的 M-RoPE。InternVL3 引入了 V2PE：位置编码按模态类型（文本、图像、视频）变化，带有可学习的缩放。实践中：

- 文本 token 获得 1D 位置（文本索引）。
- 图像 patch 获得 2D 位置（行、列）。
- 视频帧获得 3D 位置（时间、行、列）。

三者共享相同的 RoPE 频率基数，但每个频带的 hidden-dim 分配是可学习参数而非固定分割。在预训练期间可以自由权衡时间分辨率与空间频率分辨率。

V2PE 的消融结论：在相同计算量下，视频 benchmark 上比 M-RoPE 高 1-2 分。不是革命性的，但更干净。

### Visual Resolution Router (ViR)

部署优化。不是所有图像都需要全分辨率编码。一张只有一个低细节物体的照片在 1280px 原生分辨率编码时浪费 token。ViR 是一个小型分类器，在编码之前预测回答问题所需的最低分辨率。

路由有三个层级：低分辨率（256 token）、中等（576）、高（2048+）。在生产流量中，60% 的查询用低或中等就够了。净效果：在相同质量下吞吐量提升 2-3 倍。

### Decoupled Vision-Language 部署 (DvD)

当你部署一个大型 VLM 时，视觉编码器每张图像只运行一次，但 LLM 对每个输出 token 都要自回归运行。两个组件有不同的瓶颈（视觉 = GPU 内存带宽用于 conv + attention；LLM = KV cache）。DvD 将它们分到不同 GPU 上，中间用流式传输连接。

对于 8B + 400M 编码器模型，DvD 相比共置部署大约将每节点吞吐量翻倍。

### 单阶段 vs 多阶段质量

InternVL3 的主要 benchmark 声明：在 78B 参数下匹配 Gemini 2.5 Pro 的 MMMU-Pro。在 38B 下匹配 GPT-4o。在 8B 下领先开源 8B 排行榜。全部基于单阶段预训练 + 指令微调的方案。

对齐负债假说是可测量的：InternVL3-8B 每单位视觉 benchmark 增益损失的文本 benchmark 分数（MMLU、GSM8K）比 Qwen2.5-VL-7B 更少。模型更通用，因为训练是一体的，而非两段式。

### InternVL3.5 和 InternVL-U

InternVL3.5（2025 年 8 月）扩展了这个方案。相同的原生预训练方法，更多数据，更多参数。MMMU 改进是渐进式的。

InternVL-U（2026）增加了统一生成——在同一个 backbone 上通过 MMDiT head 实现图像输出。"U" 代表 "Understanding + generation"，追赶 Transfusion 风格的统一模型（Lesson 12.13）。同一个原生预训练 backbone 同时支持理解和生成 head。

### 原生预训练的权衡

原生预训练不是免费的：

- 计算量。从头训练一个新 VLM 的成本与训练一个文本 LLM 相同——数百万 GPU 小时。事后适配复用已有 LLM 权重，节省大部分成本。
- 数据。大规模交织图文语料稀缺。OBELICS 是 1.41 亿文档；MMC4 是 5.71 亿。纯文本可达 15T token。多模态预训练数据稀缺是硬约束。
- 基础 LLM 复用。原生预训练放弃了以后换入新 LLM 的选项。事后方案允许你只重训 adapter 就能把 Llama-3.1 换成 Llama-4。

InternVL3 的赌注：对齐负债比复用损失更严重。Benchmark 支持这个论断。生产成本阻止了未来实验室廉价复制。事后 VLM 会继续存在，因为对大多数项目来说它们仍然更便宜。

## Use It

`code/main.py` 是一个训练语料混合器和 ViR 路由模拟器。它：

- 接受目标语料配比（%text、%interleaved、%caption、%video）并计算每种模态的预期步数。
- 在一批查询上模拟 ViR 路由（分布：50% 低细节、30% 中等、20% 高细节）并报告平均 token 数。
- 给出编码器 vs LLM FLOPs 下的 DvD 吞吐量估算。
- 打印事后 vs 原生预训练在参数、计算量、数据和预期对齐负债症状上的对比。

## Ship It

本课产出 `outputs/skill-native-vs-posthoc-auditor.md`。给定一个拟议的 VLM 训练计划，它审计应该走原生还是事后路线，标记对齐负债风险，并推荐语料配比。当你在规划一个新的开源 VLM 项目并需要选择训练策略时使用它。

## 练习

1. 估算 InternVL3-8B（原生预训练）和 LLaVA-OneVision-7B（事后）之间的计算量差异。GPU 小时的大致比例是多少？什么解释了这个差距？

2. InternVL3 报告 40% text / 35% interleaved / 20% caption / 5% video。如果你的目标任务是视频密集型的，提出一个新比例并论证为什么基础模型仍然需要大量文本和 caption 数据。

3. 阅读 MM1.5 第 4 节关于遗忘的内容。指出事后训练显示最大退化的确切 benchmark。退化代价有多大？

4. ViR 将 60% 的流量路由到低分辨率编码。哪些类型的查询会被误路由（需要高分辨率时被送到低分辨率）？提出三种路由失败模式。

5. DvD 将视觉和 LLM 分到不同 GPU 上。在什么流量模式下 DvD 反而会降低吞吐量？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Native multimodal pretraining | "从头一起训" | 文本 + 图像 + 视频 token 从第 1 步就参与 loss，而非事后接入 |
| Alignment debt | "事后惩罚" | 将视觉接到冻结 LLM 上带来的可测量的文本能力退化和回答一致性下降 |
| V2PE | "可变视觉位置编码" | 按模态可学习的位置编码分配；InternVL3 的 M-RoPE 后继者 |
| ViR | "分辨率路由器" | 在编码前为每个查询选择所需最低分辨率的小型分类器，节省推理 token |
| DvD | "解耦部署" | 视觉编码器在一个 GPU 上，LLM 在另一个上，流式交接；大型 VLM 吞吐量翻倍 |
| InternVL-U | "统一理解 + 生成" | 2026 年后续工作，在原生预训练 backbone 上增加图像生成 head |
| Interleaved corpus | "OBELICS / MMC4" | 文本和图像按自然阅读顺序排列的文档；原生预训练的原材料 |

## 延伸阅读

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)
