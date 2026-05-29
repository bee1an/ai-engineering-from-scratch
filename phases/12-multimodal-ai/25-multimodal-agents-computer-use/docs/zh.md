# 多模态 Agent 与 Computer-Use（Capstone）

> 2026 年的前沿产品是一个多模态 agent，它能读取截图、点击按钮、导航 Web UI、填写表单，并端到端完成工作流。SeeClick 和 CogAgent（2024）证明了 GUI grounding 原语。Ferret-UI 增加了移动端。ChartAgent 引入了图表的视觉工具使用。VisualWebArena 和 AgentVista（2026）是前沿追逐的 benchmark——即使 Gemini 3 Pro 和 Claude Opus 4.7 在 AgentVista 的困难任务上也只得到约 30%。这个 capstone 汇集了 Phase 12 的所有线索：感知（高分辨率 VLM）、推理（带工具使用的 LLM）、grounding（坐标输出）、长程记忆和评估。

**Type:** Capstone
**Languages:** Python (stdlib, action schema + agent loop skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 09 (Qwen-VL JSON), Phase 14 (Agent Engineering)
**Time:** ~240 minutes

## 学习目标

- 设计多模态 agent 循环：感知 → 推理 → 行动 → 观察 → 重复。
- 构建 VLM 可以输出为 JSON 的 GUI grounding 输出 schema（点击坐标、输入文本、滚动、拖拽）。
- 比较纯截图 agent vs accessibility-tree agent vs 混合 agent。
- 在一个小型 VisualWebArena 切片上搭建多模态 agent benchmark 评估。

## 问题

一个订票网站工作流："帮我找一张 4 月 15 日去东京的航班，靠过道座位，800 美元以下，订好它。"

一个多模态 agent 需要：

1. 对浏览器截图。
2. 将截图 + URL + 目标解析为计划。
3. 输出结构化动作：click（在 x,y 处）、type "Tokyo"（在元素 E 处）、scroll down、select（单选按钮）。
4. 将动作应用到浏览器。
5. 观察新状态（下一张截图）。
6. 重复直到任务完成。

每一步都是一次多模态 VLM 调用。VLM 输出必须是可解析的 JSON。错误跨步骤累积，因此恢复能力很重要。

## 概念

### GUI grounding——原语

GUI grounding 是：给定一张截图和一条自然语言指令，输出要点击的 (x, y) 坐标（或其他动作）。

SeeClick (arXiv:2401.10935) 是第一个大规模开源结果：在合成 + 真实 GUI 数据上微调 VLM，将坐标作为纯文本 token 输出。有效。

CogAgent (arXiv:2312.08914) 增加了 1120x1120 高分辨率编码用于密集 UI。得分：Web 导航约 84%。

Ferret-UI (arXiv:2404.05719) 专注于移动端 UI，集成 iOS accessibility 数据。

输出格式通常是 JSON：

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` 有助于恢复：如果坐标在截图之间漂移，语义提示让系统可以重新 grounding。

### Action schema

典型的 action schema 有 6-10 种动作类型：

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

Agent 每步输出一个动作。浏览器包装器执行并返回新状态。

### 纯截图 vs accessibility-tree

两种输入模式：

- 纯截图：完整图像，无结构信息。最通用；适用于任何应用。
- Accessibility tree：结构化 DOM / iOS accessibility 信息。grounding 更可靠；在 tree 可用的地方有效。
- 混合：两者兼用，tree 作为原子动作的可靠 grounder，截图提供语义上下文。

生产 agent 在可能时使用混合模式。浏览器自动化（Selenium + accessibility）总是有 tree；桌面应用有时有。

### 长程记忆

一个 20 步工作流生成 20 张截图。VLM 的 context 很快填满。三种压缩策略：

- Summary-chain：每 5 步后总结发生了什么，丢弃旧截图。
- Skip-frame：保留第一张、最后一张和每第 3 张截图。
- 工具记录日志：执行动作，保留做了什么的文本日志；不回看旧截图。

Claude 的 computer-use API 使用日志模式。更简单，更可靠。

### 视觉工具使用

ChartAgent (arXiv:2510.04514) 引入了图表理解的视觉工具使用：裁剪、缩放、OCR、调用外部检测。Agent 可以输出 "crop to region (100, 200, 300, 400) then call OCR" 作为工具调用。工具返回文本；VLM 继续推理。

这个模式可以泛化：set-of-mark prompting、区域标注和外部检测工具都适合同一个"输出工具调用，接收结构化响应"的 schema。

### 2026 年的 benchmark

- ScreenSpot-Pro。约 1k 张 Web 截图上的 GUI grounding。开源 SOTA Qwen2.5-VL-72B 约 85%。前沿约 90%。
- VisualWebArena。端到端 Web 任务（购物、论坛、分类信息）。开源 SOTA 约 20%。Gemini 3 Pro 约 27%。
- AgentVista (arXiv:2602.23166)。2026 年最难的 benchmark。跨 12 个领域的真实工作流。前沿模型得分 27-40%；开源模型 10-20%。
- WebArena / WebShop。较老的 benchmark；已被前沿饱和。

### 为什么仍然很难

Agent 性能瓶颈：

1. 细粒度视觉 grounding。"点击那个小 X" 在移动端分辨率下经常失败。
2. 长程规划。10 个动作后，agent 偏离目标。
3. 错误恢复。当点击失败（点错按钮）时，检测 + 恢复很少有训练数据。
4. 跨页面上下文。在标签页之间跳转或长表单会丢失状态。

研究方向：记忆架构、显式重规划、多模态验证（截图匹配以确认动作成功）。

### Capstone 构建任务

Capstone 任务：构建一个 computer-use agent，它：

1. 读取订票网站模拟页面的 HTML + 截图。
2. 规划多步序列：搜索 → 选择 → 填写表单 → 提交。
3. 输出匹配 action schema 的 JSON 动作。
4. 在固定的 10 个任务切片上评估。

本课提供的脚手架代码易于扩展到真实浏览器。

## Use It

`code/main.py` 是 capstone 脚手架：

- Action schema JSON 定义（10 种动作）。
- Mock 浏览器状态作为 dict。
- Agent 循环骨架：接收状态，输出动作，应用，循环。
- 10 个任务的 mini-benchmark（合成页面）用于测量端到端成功率。
- 动作失败时的错误恢复 hook。

## Ship It

本课产出 `outputs/skill-multimodal-agent-designer.md`。给定一个 computer-use 产品（领域、动作集、评估目标），设计完整的 agent 循环、记忆策略、grounding 模式和预期 benchmark 分数。

## 练习

1. 用 `screenshot_region` 工具（裁剪 + 缩放）扩展 action schema。哪些任务受益？

2. 阅读 AgentVista (arXiv:2602.23166)。描述最难的任务类别以及为什么前沿模型仍然失败。

3. 长程记忆压缩：设计一个 summary-chain，保持 ≤4 张截图活跃，任意数量记录在日志中。

4. 构建错误恢复 hook：当动作失败（按钮未找到）时，agent 下一步做什么？

5. 比较纯截图的 Claude 4.7 和混合截图 + accessibility-tree 的 Qwen2.5-VL 在 10 个 Web 任务上的表现。哪个在哪些任务上赢？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| GUI grounding | "点击坐标" | 模型在截图上为指令目标输出 (x,y) |
| Action schema | "工具定义" | 有效动作（click、type、scroll、drag）的 JSON 描述 |
| Accessibility tree | "结构化 DOM" | 来自浏览器/iOS API 的机器可读 UI 层次结构 |
| Hybrid agent | "截图 + tree" | 同时使用图像和结构信息；比单独使用任一种更可靠 |
| Visual tool use | "缩放/裁剪/检测" | Agent 在计划中途调用外部视觉工具（OCR、检测） |
| Summary-chain | "记忆压缩" | 定期文本摘要替代长截图历史 |
| VisualWebArena | "端到端 Web benchmark" | 2024 年端到端 Web 任务 benchmark |
| AgentVista | "2026 年困难 benchmark" | 12 个领域的真实工作流；即使 Gemini 3 Pro 也只得约 30% |

## 延伸阅读

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
