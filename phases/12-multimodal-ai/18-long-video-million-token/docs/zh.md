# 百万 Token 上下文的长视频理解

> 一小时 4K 视频在 24 FPS 下，经过 patch 和 embedding，产生约 6000 万 token。一个 2 小时播客转录是 30,000 token。一部完整蓝光电影，即使用激进池化压缩，也是数十万 token。Google 的 Gemini 1.5（2024 年 3 月）以 1000 万 token 上下文开启了这个时代，在一小时视频上做到了可靠的 needle-in-a-haystack 召回。LWM（Liu et al.，2024 年 2 月）展示了 ring attention 的扩展路径。LongVILA 和 Video-XL 进一步扩展了摄入能力。VideoAgent 用 agentic retrieval 替代了原始上下文。每种方法是计算、召回和工程复杂度之间的不同权衡。本课并排阅读它们。

**Type:** Build
**Languages:** Python (stdlib, needle-in-haystack simulator + agentic-retrieval router)
**Prerequisites:** Phase 12 · 17 (video temporal tokens)
**Time:** ~180 minutes

## 学习目标

- 计算不同 FPS 和池化下长视频的总视觉 token 数。
- 解释三条扩展路径：暴力上下文（Gemini 1.5）、ring attention（LWM）、token 压缩（LongVILA / Video-XL）。
- 对比原始上下文视频 VLM vs agentic-retrieval 视频 VLM（VideoAgent）在精度和延迟上的表现。
- 为 30 分钟视频设计 needle-in-a-haystack 测试并测量特定分钟的召回率。

## 问题

Qwen2.5-VL 尺寸 patch 在 384 原生分辨率下的单帧约 729 token。3x3 池化后是每帧 81 token。30 分钟 clip 在 1 FPS = 1800 帧 = 145,800 token。2025 年开源 VLM 可以做到，但紧张。在 2 FPS 下，291,600 token——只有最大的上下文能装下。

2 小时电影在 1 FPS 下是 583k token。超出大多数 2026 年开源模型；需要 Gemini 2.5 Pro 或更激进的池化。

三条扩展路径出现了。

## 核心概念

### 路径 1：暴力上下文（Gemini 1.5、Claude Opus）

用硬件砸问题。将上下文扩展到百万 token，一次前向传播处理所有内容。

Gemini 1.5 Pro 以 1M token 发布；Gemini 1.5 Ultra 到 10M；2026 年的 Gemini 2.5 Pro 可靠地处理数小时视频。论文（arXiv:2403.05530）记录了在约 9.5M token 下 99.7% 的 needle-in-a-haystack 召回率。

工程：自定义注意力实现，带内存层级（local + global + sparse）加 MoE 专家路由以实现长上下文效率。未完整公开细节。非开源。

### 路径 2：Ring attention（LWM、LongVILA）

Ring attention 将长序列分布在设备上形成"环"，每个设备持有一个 chunk。跨全序列的注意力通过每个设备将其 chunk 发送给环中下一个设备、计算部分注意力并聚合来实现。

LWM（Liu et al.，2024）以此方式训练了 1M token 上下文模型。训练计算随上下文线性扩展，而非二次——注意力的二次开销在环的设备间摊销。

LongVILA（arXiv:2408.10188）将该模式适配到 VLM。1400 帧视频，每帧 192 token = 268k 上下文，用 8 路并行的 ring attention 训练。

### 路径 3：Token 压缩（Video-XL、LongVA）

比暴力上下文更便宜：在 LLM 看到序列之前激进压缩。

Video-XL（arXiv:2409.14485）使用视觉摘要 token：每个 N 帧 clip 产出一个"摘要"token，该 token attend 所有 N 帧。推理时 LLM 每个 clip 只看到一个摘要 token，大幅缩小上下文。

LongVA 用"长上下文迁移"技术将 LLM 上下文从 200k 扩展到 2M。在长上下文文本上训练，通过共享表示迁移到长上下文视频。

Token 压缩用特定时间戳的召回换取可扩展性。模型大致知道发生了什么但有时会错过精确帧。

### 路径 4：Agentic retrieval（VideoAgent）

不要把完整视频喂给 LLM。而是把视频当作数据库，用 LLM 来查询它。

VideoAgent（arXiv:2403.10517）：

1. LLM 读取问题。
2. LLM 向检索工具请求相关片段（"给我看有猫的片段"）。
3. 工具返回匹配的 clip 时间戳。
4. LLM 通过 VLM 读取这些 clip。
5. LLM 组合答案或提出后续查询。

这是 LLM-as-agent 模式应用于长视频。更便宜的推理（只编码相关 clip），更难的工程（检索质量成为瓶颈）。

### Needle-in-a-haystack benchmark

标准长上下文测试：在视频的随机位置插入一个独特的视觉或文本标记，然后提出需要回忆它的查询。

指标：跨视频长度和标记位置的 Recall@k。

Gemini 2.5 Pro 在 90 分钟视频上得分 >99% 召回。开源 72B 模型（Qwen2.5-VL-72B、InternVL3-78B）在 30 分钟得分约 85-90%，超过 60 分钟后下降。

VideoAgent 在 2+ 小时可以匹配或超过原始上下文模型，因为如果工具好的话检索能命中 needle。

### 如何选择路径

15 分钟 clip 要前沿精度：开源 72B + 原生上下文通常够用。选 Qwen2.5-VL-72B。

30 分钟到 1 小时内容：开源选 LongVILA 或 Video-XL；闭源选 Gemini 2.5 Pro。质量标准很重要——前沿走闭源。

2+ 小时内容：VideoAgent 或类似检索模式。或者分块摘要并喂层级摘要。

### 2026 生产模式

实践中，生产级长视频流水线是混合的：

1. 对整个视频运行动态 FPS 采样 + 激进池化（得到 100k token 的全局表示）。
2. 传给 72B VLM 做全局摘要。
3. 如果用户问详细问题，用摘要作为索引运行 agentic retrieval。

这结合了暴力上下文的全局理解和检索的局部细节。

## Use It

`code/main.py`：

- 计算 1 分钟到 3 小时视频在不同 FPS + 池化下的 token 预算。
- 模拟 needle-in-a-haystack 运行：在随机时间戳注入标记，提问，评分召回。
- 包含一个 agentic-retrieval 路由器模拟器，选择特定 clip 喂给下游 VLM。

运行预算表，感受规模差距。

## Ship It

本课产出 `outputs/skill-long-video-strategy-planner.md`。给定视频时长和查询复杂度，在暴力上下文、压缩和 agentic retrieval 之间选择，并计算延迟 + 质量预期。

## 练习

1. 45 分钟讲座，1 FPS，每帧 81 token。总 token 数？能装进哪些模型的上下文？

2. 设计一个 needle-in-a-haystack 测试：在第几分钟注入标记？确切的查询格式是什么？

3. 对比暴力上下文 Qwen2.5-VL-72B（80k 上下文）和 VideoAgent（Claude 3.5 + 检索）在 1 小时视频上的表现。哪个在召回上赢？哪个在延迟上赢？

4. Ring attention 的内存成本在序列长度上线性、在设备数上线性。解释为什么，以及如果去掉 ring-rotation 阶段会怎样。

5. 阅读 Gemini 1.5 Section 5 关于 needle-in-a-haystack。论文在 1M vs 10M token 边界上发现了什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Brute context | "就是更多 token" | 将 LLM 上下文扩展到百万 token；一次 pass 处理所有 |
| Ring attention | "LWM 风格并行" | 分布式注意力模式，每个设备持有一个 chunk 并旋转 |
| Token compression | "摘要 token" | 通过学习的压缩器在 LLM 之前减少每个 clip 的 token |
| Needle-in-haystack | "NIH 测试" | 在随机位置插入独特标记，测试时让模型回忆 |
| Agentic retrieval | "LLM 作为查询规划器" | LLM 向检索工具请求相关 clip，通过 VLM 读取，组合答案 |
| VideoAgent | "视频检索模式" | 典型的 agentic-retrieval 设计：问题 -> 工具 -> clip -> 答案 |

## 延伸阅读

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
