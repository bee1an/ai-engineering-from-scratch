# 开源 VLM 方案：什么真正重要

> 2024-2026 年的开源 VLM 文献是一片消融表的森林。Apple 的 MM1 测试了 13 种图像编码器、连接器和数据混合的组合。Allen AI 的 Molmo 证明了详细人工描述优于 GPT-4V 蒸馏。Cambrian-1 跑了 20+ 种编码器对比。Idefics2 形式化了五轴设计空间。Prismatic VLMs 在受控基准上比较了 27 种训练方案。从所有这些噪声中，一小组结论跨论文成立：图像编码器比连接器架构重要，数据混合比两者都重要，详细人工描述优于蒸馏合成数据。本课替你读完那些表格。

**类型：** 学习 + 实验
**语言：** Python（标准库，消融表解析器 + 方案选择器）
**前置：** Phase 12 · 05（LLaVA 基线）
**时间：** ~180 分钟

## 学习目标

- 说出五轴 VLM 设计空间：图像编码器、连接器、LLM、数据混合、分辨率调度。
- 读懂 MM1 / Idefics2 / Cambrian-1 消融表，预测哪个旋钮移动给定基准。
- 给定计算预算和任务组合，为新 VLM 选择方案（编码器、连接器、数据、分辨率）。
- 解释为什么在相同 token 数下，详细人工描述优于 GPT-4V 蒸馏。

## 问题

数百个开源 VLM 存在。"好"和"最先进"之间的大部分差距不是架构。是数据、分辨率调度和编码器选择。知道模型表现不佳时先调哪个旋钮，能帮你避免 500 万 GPU 小时的错误。

2023 年浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）跑在描述对预训练 + LLaVA-Instruct-150k 上。好的基线。MMMU 封顶约 35%。

2024 年浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）跑了详尽的消融。结果出人意料且实用。

## 概念

### 五轴设计空间

Idefics2（Laurençon et al.，2024）命名了这些轴：

1. 图像编码器。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器在 patch size、分辨率和预训练目标上不同。
2. 连接器。MLP（2-4 层）、Q-Former（32 query + cross-attn）、Perceiver Resampler（64 query）、C-Abstractor（卷积 + 双线性池化）。
3. 语言模型。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 大小是主要参数成本。
4. 训练数据。描述对（CC3M、LAION）、交错（OBELICS、MMC4）、指令（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. 分辨率调度。固定 224/336/448、AnyRes、原生动态。训练中递增或恒定。

每个生产 VLM 在每个轴上做出选择。MMMU 分数的大部分方差由轴 1、4 和 5 解释——不是你选了哪个连接器。

### 轴 1：编码器 > 连接器

MM1 第 3.2 节显示：从 CLIP ViT-L/14 换到 SigLIP SO400m/14 加了 3+ 个 MMMU 点。连接器从 MLP 换到 Perceiver Resampler 加了不到 1 个点。Idefics2 复现了：SigLIP > CLIP，Q-Former ≈ MLP ≈ Perceiver 在相同 token 数下。

Cambrian-1 的"Cambrian Vision Encoders Match-Up"（Tong et al.，2024）在视觉中心基准（CV-Bench）上跑了 20+ 种编码器。排行榜顶部是 DINOv2 和 SigLIP 的混合；CLIP 在中间；ImageBind 和 ViT-MAE 更低。从 CLIP ViT-L 到 DINOv2 ViT-g/14 的差距在 CV-Bench 上约 5-7 个点。

2026 年开源 VLM 的默认编码器是 SigLIP 2 SO400m/14（语义 + 密集特征），有时与 DINOv2 ViT-g/14 特征拼接（Cambrian 的"Spatial Vision Aggregator"就是这么做的）。

### 轴 2：连接器设计差异不大

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出相同结论：在固定视觉 token 数下，连接器架构几乎不重要。在 mean-pooled patch 上的 2 层 MLP 与相同 token 预算下的 32-query Q-Former 性能差距在 1 个点以内。

重要的是 token 数。更多视觉 token = 更多 LLM 计算 = 更好性能，到某个点后收益递减。每张图 64 token 对 OCR 太少。576-1024 token 是大多数开源 VLM 的甜蜜点。2048+ 只对文档和图表有帮助。

Q-Former vs MLP 是成本问题，不是质量问题：Q-Former 无论图像分辨率如何都将 token 限制在 32-64；MLP 发出所有 patch token。对高分辨率输入，Q-Former 节省 LLM 上下文；对低分辨率，差异是噪声。

### 轴 3：LLM 大小决定天花板

LLM 从 7B 翻倍到 13B 在每篇 VLM 论文中都可靠地在 MMMU 上加 2-4 个点。到 70B 你饱和大多数基准。VLM 的多模态推理天花板就是 LLM 的文本推理天花板——视觉编码器只能喂它，不能替它推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 碾压 MMMU-Pro 和 ScreenSpot-Pro：语言大脑巨大。7B VLM 不能通过巧妙的连接器设计替代 70B VLM。

### 轴 4：数据——详细人工描述优于蒸馏

Molmo + PixMo（Deitke et al.，2024）是 2024 年每个人都应该读的结果。Allen AI 让人工标注者用 1-3 分钟的密集语音转文本描述图像，产出 712K 张密集描述的图像。训练数据中没有任何 GPT-4V 蒸馏。

Molmo-72B 在 11 个基准中的 11 个上超过了 Llama-3.2-90B-Vision。差距不是架构——是描述质量。详细人工描述每张图包含的信息比短网络描述多 5-10 倍，且保持事实基础，而 GPT-4V 蒸馏会产生幻觉。

ShareGPT4V（Chen et al.，2023）和 Cauldron（Idefics2）用混合人工 + GPT-4V 描述遵循了相同路线。趋势很清楚：对于 2026 年前沿，描述密度 > 描述数量 > 蒸馏便利性。

### 轴 5：分辨率及其调度

Idefics2 的消融：384 -> 448 加 1-2 个点。448 -> 980 加图像分割（AnyRes）在 OCR 基准上再加 3-5 个点。固定分辨率训练在中等准确率处平台；分辨率递增（从 224 开始，到 448 或原生结束）训练更快且最终更高。

Cambrian-1 跑了分辨率 vs token 的权衡：在固定计算下，你可以选择更低分辨率更多 token 或更高分辨率更少 token。更高分辨率对 OCR 胜出；更低分辨率更多 token 对通用场景理解胜出。

2026 年生产方案：阶段 1 在 384 固定训练，阶段 2 用动态分辨率到 1280（OCR 密集任务）。

### Prismatic 受控对比

Prismatic VLMs（Karamcheti et al.，2024）是控制了所有轴的论文。相同 13B LLM，相同指令数据，相同评估——每次只变一个轴。结果：

- 每张图视觉 token 数解释约 60% 的方差。
- 编码器选择解释约 20%。
- 连接器架构解释约 5%。
- 其他一切（数据混合、调度器、LR）剩余约 15%。

这是粗略分解，但它是文献中对"我应该先消融什么"最干净的回答。

### 2026 年方案选择器

基于证据，2026 年新项目的默认开源 VLM 方案：

- 编码器：SigLIP 2 SO400m/14 原生分辨率加 NaFlex，如果需要分割/定位则拼接 DINOv2 ViT-g/14 密集特征。
- 连接器：patch token 上的 2 层 MLP。除非 token 受限否则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，7B 求成本，70B 求质量，按目标延迟选择。
- 数据：PixMo + ShareGPT4V + Cauldron，加上任务特定指令数据。
- 分辨率：动态（长边最小 256，最大 1280 像素）。
- 调度：阶段 1 对齐（仅 projector），阶段 2 全量微调，阶段 3 任务特定微调。

这些默认值中的每一个都可以追溯到本课末尾引用的论文中的一个测量消融。

## 动手用

`code/main.py` 是一个消融表解析器和方案选择器。它编码了 MM1 和 Idefics2 消融表（精简版）并让你查询：

- "给定预算 X 和任务 Y，什么方案胜出？"
- "如果我在 7B Llama 上把 SigLIP 换成 CLIP，预期 MMMU 变化多少？"
- "我应该先消融哪个轴以获得 80% 置信度的答案？"

输出是排序的方案列表，附带预期基准变化和"先消融"建议。

## 交付物

本课产出 `outputs/skill-vlm-recipe-picker.md`。给定目标任务组合、计算预算和延迟目标，它输出完整方案（编码器、连接器、LLM、数据混合、分辨率调度），附带引用支持每个选择的消融。阻止工程师每次新 VLM 项目启动时重新发明 Idefics2 消融表。

## 练习

1. 阅读 MM1 第 3.2 节。对于固定 2B LLM、预算 5000 万张图，哪个编码器胜出？在 13B LLM 下答案会翻转吗？为什么？

2. Cambrian-1 发现拼接 DINOv2 + SigLIP 在视觉中心基准上优于单独使用任一个，但在 MMMU 上没有增益。预测哪些基准提升，哪些持平。

3. 你的目标是 2B LLM 上的移动 UI agent。选择编码器、连接器、分辨率和数据混合。用具体消融表证明每个选择。

4. Molmo 发布了 4B 和 72B 模型。4B 与闭源 7B VLM 竞争力相当；72B 在 11/11 基准上超过 Llama-3.2-90B-Vision。这对 LLM 大小平台假说说明了什么？

5. 设计一个消融表来隔离数据混合质量与编码器质量对 7B VLM 的影响。最少需要多少训练运行？提出四个轴设置。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Ablation | "调一个旋钮" | 训练多次运行，恰好在一个设计空间轴上不同，其他一切保持不变 |
| Connector | "桥梁" / "projector" | 将视觉编码器输出映射到 LLM token 空间的可训练模块（MLP、Q-Former、Perceiver） |
| Detailed human caption | "密集描述" | 多句人工撰写的描述（通常 80-300 token），比网络 alt text 丰富得多 |
| Distillation | "GPT-4V 描述" | 由更强的专有 VLM 生成的训练数据；方便但容易继承幻觉 |
| AnyRes / dynamic res | "高分辨率路径" | 通过 tiling 或 M-RoPE 将大于编码器原生分辨率的图像送入的策略 |
| Resolution ramp | "课程学习" | 从低分辨率开始逐步增加的训练调度，加速对齐学习 |
| Vision-centric bench | "CV-Bench / BLINK" | 强调细粒度视觉感知而非语言密集推理的评估 |
| PixMo | "Molmo 的数据" | Allen AI 的 712K 密集描述图像数据集；人工语音转录为密集描述 |

## 延伸阅读

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
