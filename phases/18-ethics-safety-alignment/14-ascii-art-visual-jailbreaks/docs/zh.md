# ASCII Art 与视觉越狱攻击

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs"（ACL 2024, arXiv:2402.11753）。遮蔽有害请求中的安全相关 token，用相同字母的 ASCII art 渲染替换，发送伪装提示。GPT-3.5、GPT-4、Gemini、Claude、Llama-2 都无法鲁棒地识别 ASCII art token。攻击绕过 PPL（困惑度过滤器）、改写防御和重新分词。相关：ViTC 基准测量非语义视觉提示的识别；StructuralSleight 泛化到非常见文本编码结构（树、图、嵌套 JSON）作为编码攻击家族。

**Type:** Build
**Languages:** Python (stdlib, ArtPrompt token-masking harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Time:** ~60 minutes

## 学习目标

- 描述 ArtPrompt 攻击：词识别步骤、ASCII art 替换、最终伪装提示。
- 解释为什么标准防御（PPL、改写、重新分词）对 ArtPrompt 失败。
- 定义 ViTC 并描述它测量什么。
- 描述 StructuralSleight 作为对任意非常见文本编码结构的泛化。

## 问题

通过改写和角色扮演的攻击（第 12 课）和通过长上下文的攻击（第 13 课）在文本级模式上操作。ArtPrompt 在识别级操作：模型不解析被禁止的 token。它解析的是用字符渲染的图像。安全过滤器看到无害的标点符号。模型看到一个词。

## 概念

### ArtPrompt，两步

步骤 1. 词识别。给定有害请求，攻击者使用 LLM 识别安全相关词（如"how to make a bomb"中的"bomb"）。

步骤 2. 伪装提示生成。用 ASCII art 渲染（形成字母形状的 7x5 或 7x7 字符块）替换每个识别的词。模型接收到一个标点和空格的网格，足够有能力的模型可以将其识别为该词；安全过滤器只看到网格。

结果：GPT-4、Gemini、Claude、Llama-2、GPT-3.5 全部失败。在其基准子集上攻击成功率超过 75%。

### 为什么标准防御失败

- **PPL（困惑度过滤器）。** ASCII art 有高困惑度——但所有新颖输入也是如此。阻止 ArtPrompt 的阈值选择也会阻止合法的结构化输入。
- **改写。** 改写提示会破坏 ASCII art。实践中，改写 LLM 经常保留或重建 art。
- **重新分词。** 不同地分割 token 不改变模型的视觉在识别字母形状这一事实。

根本问题是安全过滤器是 token 级或语义级的；ArtPrompt 在视觉识别级操作。

### ViTC 基准

非语义视觉提示的识别。测量模型阅读 ASCII art、wingdings 和其他非文本语义视觉内容的能力。ArtPrompt 的有效性与 ViTC 准确率相关：模型越善于阅读视觉文本，ArtPrompt 对它越有效。这是能力-安全权衡。

### StructuralSleight

泛化 ArtPrompt：非常见文本编码结构（UTES）。树、图、嵌套 JSON、CSV-in-JSON、diff 风格代码块。如果一种结构在训练安全数据中罕见但模型可解析，它就能隐藏有害内容。

防御含义：安全必须泛化到模型能解析的结构化表示。这个集合很大且在增长。

### 图像模态类比

视觉 LLM（GPT-5.2、Gemini 3 Pro、Claude Opus 4.5、Grok 4.1）扩展了攻击面。用实际图像的 ArtPrompt 风格攻击比 ASCII art 类比更强，因为图像编码器产生更丰富的信号。

### 在 Phase 18 中的位置

第 12-14 课描述三种正交攻击向量：迭代精炼（PAIR）、上下文长度（MSJ）和编码（ArtPrompt/StructuralSleight）。第 15 课从以模型为中心的攻击转向系统边界攻击（间接注入）。第 16 课描述防御工具响应。

## Use It

`code/main.py` 构建一个玩具 ArtPrompt。你可以用 ASCII art 字形伪装有害查询中的特定词，验证伪装字符串通过关键词过滤器，并（可选地）使用简单识别器将伪装字符串解码回来。

## Ship It

本课产出 `outputs/skill-encoding-audit.md`。给定一份越狱防御报告，它枚举覆盖的编码攻击家族（ASCII art、base64、leet-speak、UTF-8 同形字、UTES）以及捕获每种的防御层。

## 练习

1. 运行 `code/main.py`。验证伪装字符串通过简单关键词过滤器。报告所需的字符级变化。

2. 实现第二种编码：对相同目标词的 base64。比较对 ArtPrompt 的过滤器绕过率和恢复难度。

3. 阅读 Jiang et al. 2024 Section 4.3（五模型结果）。提出 Claude 的 ArtPrompt 抵抗力高于 Gemini 在相同基准上的一个原因。

4. 设计一种生成前防御，检测提示中 ASCII art 形状的区域。测量对合法代码、表格和数学符号的误报率。

5. StructuralSleight 列出 10 种编码结构。草拟一种处理所有 10 种的通用防御并估计每个防御提示的计算成本。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| ArtPrompt | "ASCII art 攻击" | 用 ASCII art 渲染遮蔽安全词的两步越狱 |
| 伪装 | "隐藏词" | 用模型能读但过滤器不能读的视觉表示替换被禁 token |
| UTES | "非常见结构" | 非常见文本编码结构——树、图、嵌套 JSON 等用于走私内容 |
| ViTC | "视觉文本能力" | 模型阅读非语义视觉编码能力的基准 |
| 困惑度过滤器 | "PPL 防御" | 拒绝高困惑度提示；失败因为合法结构化输入也得分高 |
| 重新分词 | "分词器偏移防御" | 用不同分词器预处理提示；失败因为识别是视觉的 |
| 同形字 | "外观相似字符" | 看起来与拉丁字母相同的 Unicode 字符；绕过子串检查 |

## 延伸阅读

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — ASCII art 越狱论文
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — UTES 泛化
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 互补迭代攻击
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — 互补长度攻击
