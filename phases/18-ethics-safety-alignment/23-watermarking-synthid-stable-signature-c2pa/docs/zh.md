# 水印 — SynthID、Stable Signature、C2PA

> 三项技术构成了 2026 年 AI 生成内容溯源的结构。SynthID（Google DeepMind）— 2023 年 8 月推出图像水印，2024 年 5 月扩展到文本+视频（Gemini + Veo），2024 年 10 月通过 Responsible GenAI Toolkit 开源文本水印，2025 年 11 月随 Gemini 3 Pro 推出统一多媒体检测器。文本水印不可感知地调整下一 token 采样概率；图像/视频水印能经受压缩、裁剪、滤镜、帧率变化。Stable Signature（Fernandez et al., ICCV 2023, arXiv:2303.15435）— 微调潜在扩散解码器使每个输出包含固定消息；裁剪到 10% 内容的生成图像在 FPR<1e-6 时检测率 >90%。后续 "Stable Signature is Unstable"（arXiv:2405.07145, 2024 年 5 月）— 微调可在保持质量的同时移除水印。C2PA — 加密签名的防篡改元数据标准（C2PA 2.2 Explainer 2025）。水印和 C2PA 互补：元数据可被剥离但携带更丰富的溯源信息；水印在转码中持续存在但携带的信息更少。

**Type:** Build
**Languages:** Python (stdlib, token-watermark embed + detect)
**Prerequisites:** Phase 10 · 04 (sampling), Phase 01 · 09 (information theory)
**Time:** ~75 minutes

## 学习目标

- 描述 token 级水印（SynthID-text 风格）及其可检测的机制。
- 描述 Stable Signature 和 2024 年破解它的移除攻击。
- 说明 C2PA 的角色以及为什么它与水印互补。
- 描述关键局限：模型特定信号、改写下的鲁棒性，以及保义攻击（arXiv:2508.20228）。

## 问题

2023-2024 年，深度伪造和 AI 生成内容大规模进入政治和消费场景。水印是提议的技术溯源信号：在创建时标记生成内容，之后检测它们。2025 年的证据：没有水印是无条件鲁棒的，但与 C2PA 元数据分层结合后，组合提供了可用的溯源方案。

## 概念

### 文本水印（SynthID-text 风格）

Kirchenbauer et al. 2023 的机制，由 Google 产品化：

1. 在每个解码步骤，对前 K 个 token 进行哈希，产生词汇表到"绿色"和"红色"集合的伪随机划分。
2. 通过向绿色 logits 添加 δ 来偏向绿色集合采样。
3. 生成内容包含比随机更多的绿色 token。

检测：对每个前缀重新哈希，计算生成内容中的绿色 token 数量，计算 z-score。水印文本的 z-score >0，人类文本约为 0。

性质：
- 对读者不可感知（δ 足够小，质量损失很小）。
- 在有词汇划分函数访问权限时可检测。
- 对改写不鲁棒 — 重写文本会破坏信号。

SynthID-text 于 2024 年 10 月通过 Google 的 Responsible GenAI Toolkit 开源。

### Stable Signature（图像）

Fernandez et al. ICCV 2023。微调潜在扩散解码器，使每个生成的图像在潜在表示中包含固定的二进制消息。检测通过神经解码器从潜在空间解码。裁剪到 10% 内容的图像在 FPR<1e-6 时检测率 >90%。

2024 年 5 月 "Stable Signature is Unstable"（arXiv:2405.07145）：微调解码器可在保持图像质量的同时移除水印。对抗性的生成后微调成本低；水印的对抗鲁棒性有限。

### SynthID 统一检测器（2025 年 11 月）

随 Gemini 3 Pro 推出：一个多媒体检测器，通过一个 API 读取文本、图像、音频和视频中的 SynthID 信号。统一了 Google 的溯源技术栈。

### C2PA

Coalition for Content Provenance and Authenticity。加密签名的防篡改元数据标准。C2PA 2.2 Explainer（2025）。C2PA 清单记录溯源声明（谁创建、何时、什么转换），由创建者的密钥签名。

与水印互补：
- 元数据可被剥离；水印不能（轻易地）。
- 元数据丰富（完整溯源链）；水印携带比特。
- C2PA 依赖平台采用；水印自动嵌入。

Google 在搜索、广告和"关于此图片"中集成了两者。

### 局限性

- **模型特定。** SynthID 为启用了 SynthID 的模型的生成内容加水印。来自未启用 SynthID 模型的生成内容没有水印，因此"无 SynthID 信号"不是真实性的证明。
- **改写。** 文本水印不能经受保义改写。
- **变换攻击。** arXiv:2508.20228（2025）展示了破坏文本水印和许多图像水印的保义攻击。
- **微调移除。** 根据 "Stable Signature is Unstable"，生成后微调可移除嵌入的水印。

### EU AI Act 第 50 条

AI 生成内容标注的透明度守则（2025 年 12 月首稿，2026 年 3 月第二稿，预计 2026 年 6 月最终版，参见 [European Commission status page](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)）。截至 2026 年 4 月该守则仍为草案，时间线可能变化。这是要求技术层的监管层。深度伪造必须被标注。

### 在 Phase 18 中的位置

Lessons 22-23 关于模型输出的内容（私有数据、溯源信号）。Lesson 27 覆盖训练数据治理。Lesson 24 是要求这些技术措施的监管框架。

## Use It

`code/main.py` 构建了一个玩具文本水印。Token 是整数 0..N-1；水印采样偏向哈希定义的绿色集合。检测器计算绿色 token 的 z-score。你可以观察 1000-token 生成的检测效果，观察改写如何破坏信号，并测量人类文本上的误报率。

## Ship It

本课产出 `outputs/skill-provenance-audit.md`。给定一个带有溯源声明的内容部署，它审计：水印机制（如有）、C2PA 签名链（如有）、各自的对抗鲁棒性，以及每种模态的覆盖。

## 练习

1. 运行 `code/main.py`。报告水印 1000-token 生成 vs 人类撰写文本的 z-score。找出 95% 置信阈值下的误报率。

2. 实现一个改写攻击，替换 30% 的 token 为同义词。重新测量 z-score。

3. 阅读 Kirchenbauer et al. 2023 第 6 节关于鲁棒性的内容。为什么文本水印在改写下失败但图像水印能经受裁剪？

4. 设计一个使用 SynthID-text + C2PA 元数据的部署。描述消费者看到的溯源链。找出每个组件的一个失败模式。

5. 2024 年 "Stable Signature is Unstable" 的结果表明微调可移除图像水印。设计一个限制此攻击的部署控制 — 例如，要求微调检查点的签名发布。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| SynthID | "Google 的水印" | 跨模态溯源信号；文本、图像、音频、视频 |
| Token 水印 | "Kirchenbauer 风格" | 偏向采样的文本水印，通过绿色 token z-score 检测 |
| Stable Signature | "图像水印" | 微调解码器的水印；ICCV 2023 |
| C2PA | "元数据标准" | 加密签名的防篡改溯源元数据 |
| 改写鲁棒性 | "改写会破坏吗" | 文本水印属性；目前有限 |
| 微调移除 | "对抗性去水印" | 通过解码器微调移除图像水印的攻击 |
| 跨模态检测器 | "统一 SynthID" | 2025 年 11 月跨模态统一 API |

## 延伸阅读

- [Kirchenbauer et al. — A Watermark for Large Language Models (ICML 2023, arXiv:2301.10226)](https://arxiv.org/abs/2301.10226) — token 水印机制
- [Fernandez et al. — Stable Signature (ICCV 2023, arXiv:2303.15435)](https://arxiv.org/abs/2303.15435) — 图像水印论文
- ["Stable Signature is Unstable" (arXiv:2405.07145)](https://arxiv.org/abs/2405.07145) — 移除攻击
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — 跨模态水印
- [C2PA 2.2 Explainer (2025)](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — 元数据标准
