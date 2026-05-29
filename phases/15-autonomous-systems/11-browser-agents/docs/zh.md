# 浏览器智能体与长期规划 Web 任务

> ChatGPT agent（2025 年 7 月）将 Operator 和 deep research 合并为一个浏览器/终端智能体，在 BrowseComp 上创下 68.9% 的 SOTA。OpenAI 于 2025 年 8 月 31 日关闭了 Operator——产品层的整合。Anthropic 收购 Vercept 将 Claude Sonnet 在 OSWorld 上从不到 15% 提升到 72.5%。WebArena-Verified（ServiceNow, ICLR 2026）修复了原始 WebArena 中 11.3 个百分点的假阴性率，并发布了 258 任务的 Hard 子集。数字是真实的。攻击面也是：OpenAI 的准备负责人公开表示，浏览器智能体中的间接提示注入"不是一个可以完全修补的 bug"。已记录的 2025-2026 攻击：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）和 Perplexity Comet 中的一键劫持。

**Type:** Learn
**Languages:** Python (stdlib, indirect prompt-injection attack surface model)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 minutes

## 问题

浏览器智能体是一个读取不可信内容并执行有后果动作的长期规划智能体。智能体访问的每个页面都是用户没有编写的输入。每个页面上的每个表单都是潜在的命令通道。2025-2026 攻击语料库表明这不是假设性的：Tainted Memories 让攻击者通过精心制作的页面将恶意指令绑定到智能体的记忆；HashJack 将命令隐藏在智能体访问的 URL 片段中；Perplexity Comet 劫持只需一次点击。

防御图景令人不安。OpenAI 的准备负责人说出了安静的部分：间接提示注入"不是一个可以完全修补的 bug"。这是因为攻击存在于智能体的读取-行动边界，这在架构上是模糊的——模型读取的每个 token 原则上都可以被读作指令。

本课命名攻击面，命名基准全景（BrowseComp、OSWorld、WebArena-Verified），并建模一个最小的间接提示注入场景，以便你能在 Lessons 14 和 18 中推理真实防御。

## 概念

### 2026 年全景，每个系统一段话

**ChatGPT agent (OpenAI).** 2025 年 7 月发布。统一了 Operator（浏览）和 Deep Research（多小时研究）。2025 年 8 月 31 日关闭了独立的 Operator。BrowseComp SOTA 68.9%；OSWorld 和 WebArena-Verified 上数字强劲。

**Claude Sonnet + Vercept (Anthropic).** Anthropic 收购 Vercept 聚焦于计算机使用能力。将 Claude Sonnet 在 OSWorld 上从 <15% 提升到 72.5%。Claude Computer Use 作为工具 API 发布。

**Gemini 3 Pro with Browser Use (DeepMind).** Browser Use 集成发布计算机使用控制；FSF v3（2026 年 4 月，Lesson 20）专门跟踪 ML R&D 领域的自主性。

**WebArena-Verified (ServiceNow, ICLR 2026).** 修复了一个有据可查的问题：原始 WebArena 有约 11.3% 的假阴性率（标记为失败但实际已解决的任务）。Verified 版本用人工策划的成功标准重新评分，并添加了 258 任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| 基准 | 测量什么 | 时间跨度 |
|---|---|---|
| BrowseComp | 在时间压力下在开放网络上查找特定事实 | 分钟 |
| OSWorld | 智能体操作完整桌面（鼠标、键盘、shell） | 数十分钟 |
| WebArena-Verified | 模拟站点中的事务性 Web 任务 | 分钟 |
| Hard subset | 带多页面状态转换的 WebArena-Verified 任务 | 数十分钟 |

不同的轴。高 BrowseComp 分数说明智能体能找到事实；它不说明智能体能订机票。OSWorld 分数更接近"它在我的桌面上能用吗"。WebArena-Verified 更接近"它能完成一个流程吗"。任何生产决策都需要匹配任务分布的基准。

### 攻击面，命名

1. **间接提示注入。** 不可信页面内容包含指令。智能体读取它们。智能体执行它们。公开例子：2024 Kai Greshake et al.，2025 Tainted Memories 论文，2026 HashJack（Cato Networks）。
2. **URL 片段/查询注入。** 爬取 URL 的 `#fragment` 或查询字符串包含命令。从不可见渲染；仍在智能体的上下文中。
3. **记忆绑定攻击。** 页面指示智能体写入持久记忆（Lesson 12 涵盖持久状态）。下一个会话，记忆在没有可见触发器的情况下触发载荷。
4. **对已认证会话的 CSRF 形状攻击。** Tainted Memories 类：智能体在某处已登录；攻击者的页面发出智能体用用户 cookie 执行的状态变更请求。
5. **一键劫持。** 一个视觉上无害的按钮携带智能体跟随的载荷。Comet 类。
6. **智能体宿主面的 Content-Security-Policy 漏洞。** 渲染和工具层本身可以是攻击向量；浏览器中的浏览器智能体栈很宽。

### 为什么"不可完全修补"

攻击与智能体的能力同构。智能体必须读取不可信内容才能完成工作。智能体读取的任何内容都可能包含指令。智能体遵循的任何指令都可能与用户的实际请求不一致。防御（信任边界、分类器、工具允许列表、对有后果动作的 HITL）提高了攻击成本并减少了爆炸半径。它们不关闭这个类别。

这与 Lob 定理（Lesson 8）的推理模式相同：智能体无法证明下一个 token 是安全的；它只能建立一个使不安全 token 更可检测的系统。

### 实际发布的防御姿态

- **读/写边界。** 读取永远没有后果。写入（提交表单、发布内容、调用有副作用的工具）如果发起内容来自信任边界之外则需要新的人类批准。
- **每任务工具允许列表。** 智能体可以浏览；除非该工具被明确为任务启用，否则不能发起电汇。Lesson 13 涵盖预算。
- **会话隔离。** 浏览器智能体会话仅使用范围凭证运行。无生产认证，无个人邮箱。每个 HTTP 请求的日志保留用于审计。
- **内容清理器。** 获取的 HTML 在连接到模型上下文之前剥离已知恶意模式。（减少简单攻击；不阻止复杂载荷。）
- **对有后果动作的 HITL。** 提议-然后-提交模式（Lesson 15）。
- **记忆上的金丝雀令牌。** 如果记忆条目触发，用户看到它（Lesson 14）。

## Use It

`code/main.py` 建模一个小型浏览器智能体运行，针对三个合成页面。一个页面是良性的，一个在可见文本中有直接提示注入块，一个有 URL 片段注入（不可见但在智能体上下文中）。脚本展示（a）朴素智能体会做什么，（b）读/写边界捕捉什么，（c）清理器捕捉什么，（d）两者都捕捉不到什么。

## Ship It

`outputs/skill-browser-agent-trust-boundary.md` 界定一个提议的浏览器智能体部署：它触及哪些信任区域，它被授权写什么，以及在首次运行前必须到位哪些防御。

## 练习

1. 运行 `code/main.py`。识别清理器捕捉但读/写边界不捕捉的攻击，以及只有读/写边界捕捉的攻击。

2. 扩展清理器以检测一类 HashJack 风格的 URL 片段注入。测量带有合法片段的良性 URL 上的误报率。

3. 选一个你了解的真实浏览器智能体工作流（如"订机票"）。列出每次读取和每次写入。标记哪些写入需要 HITL 以及为什么。

4. 阅读 WebArena-Verified ICLR 2026 论文。找出原始 WebArena 评分不可靠的一个任务类别，并解释 Verified 子集如何解决它。

5. 为浏览器智能体设置设计一个记忆金丝雀。你会存储什么，在哪里，什么触发警报？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Indirect prompt injection | "恶意页面文本" | 智能体读取的页面中的不可信内容包含智能体执行的指令 |
| Tainted Memories | "记忆攻击" | 智能体将攻击者提供的指令写入持久记忆；下一会话触发 |
| HashJack | "URL 片段攻击" | 隐藏在 URL 片段/查询字符串中的载荷在智能体上下文中但不可见渲染 |
| One-click hijack | "恶意按钮" | 可见的交互元素携带智能体执行的后续载荷 |
| BrowseComp | "Web 搜索基准" | 在开放网络上查找特定事实；分钟级时间跨度 |
| OSWorld | "桌面基准" | 完整 OS 控制；多步 GUI 任务 |
| WebArena-Verified | "修复的 Web 任务基准" | ServiceNow 重新评分的 WebArena 及 Hard 子集 |
| Read/write boundary | "副作用门" | 读取永远没有后果；如果内容在信任之外则写入需要新批准 |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 和 deep research 的合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator 谱系和成为 ChatGPT agent 的架构。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始基准。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 修复子集论文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含计算机使用智能体的攻击面讨论。
