# 基准测试：WebArena 和 OSWorld

> WebArena 跨四个自托管应用测试 web 智能体能力。OSWorld 跨 Ubuntu、Windows、macOS 测试桌面智能体能力。发布时（2023–2024）两者都显示了最佳智能体与人类之间的巨大差距。差距在缩小；失败模式没有变。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 19 (SWE-bench, GAIA)
**Time:** ~60 minutes

## 学习目标

- 描述 WebArena 的四个自托管应用以及为什么基于执行的评估很重要。
- 解释为什么 OSWorld 使用真实 OS 截图而非无障碍 API。
- 列举 OSWorld 的两个主要失败模式：GUI 定位和操作知识。
- 总结 OSWorld-G 和 OSWorld-Human 在基础基准测试之上添加了什么。

## 问题

通用智能体可以调用工具。它们能驱动浏览器跨 20 次点击完成购物结账吗？它们能仅用键盘和鼠标配置一台 Linux 机器吗？这些是 WebArena 和 OSWorld 回答的问题。

## 概念

### WebArena（Zhou et al., ICLR 2024）

- 跨四个自托管 web 应用的 812 个长时间任务：一个购物网站、一个论坛、一个类 GitLab 的开发工具、一个商业 CMS。
- 加上工具：地图、计算器、草稿本。
- 评估是通过 gym API 基于执行的——订单是否下了、issue 是否关闭了、CMS 页面是否更新了？
- 发布时：最佳 GPT-4 智能体达到 14.41% 成功率 vs 人类 78.24%。

自托管的框架很重要——基准测试不会因为目标应用是固定和可复现的而不稳定。

### 扩展

- **VisualWebArena** — 视觉定位任务，成功取决于解释图像（截图作为一等观察）。
- **TheAgentCompany**（2024 年 12 月）— 添加终端 + 编码；更像真实的远程工作环境。

### OSWorld（Xie et al., NeurIPS 2024）

- 跨 Ubuntu、Windows、macOS 的 369 个真实计算机任务。
- 对真实应用的自由形式键盘和鼠标控制。
- 1920×1080 截图作为观察。
- 发布时：最佳模型 12.24% vs 人类 72.36%。

### 主要失败模式

1. **GUI 定位。** 像素 → 元素映射。模型难以在 1920×1080 中可靠地定位 UI 元素。
2. **操作知识。** 哪个菜单有设置、哪个键盘快捷键、哪个偏好设置面板。人类多年积累的知识长尾。

### 后续工作

- **OSWorld-G** — 564 样本定位套件 + Jedi 训练集。将定位与规划分解，使你可以分别测量。
- **OSWorld-Human** — 手动策划的黄金动作轨迹。显示顶级智能体使用了比必要多 1.4-2.7 倍的步骤（轨迹效率差距）。

### 为什么这很重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use（Lesson 21）都在 WebArena 和 OSWorld 塑造的工作负载上训练。基准测试是目标；生产模型是发布的答案。

### 基准测试出错的地方

- **仅截图评估。** OSWorld 是截图驱动的；在 OSWorld 上评估使用 DOM 或无障碍 API 的智能体会错过定位挑战。
- **忽略轨迹长度。** 只评分成功率会错过 OSWorld-Human 揭示的 1.4-2.7 倍步骤低效。
- **过时的自托管应用。** WebArena 的应用固定特定版本；不重新策划就更新会破坏可比性。

## Build It

`code/main.py` 实现了一个玩具 web 智能体 harness：

- 一个最小的"购物应用"状态机：list_items、add_to_cart、checkout。
- 3 个任务的黄金轨迹。
- 一个脚本化智能体尝试每个任务。
- 基于执行的评估器（状态检查）和轨迹效率指标（步骤 vs 黄金）。

运行：

```
python3 code/main.py
```

输出：每任务成功率和轨迹效率，镜像 OSWorld-Human 的方法论。

## Use It

- **WebArena Verified** 自托管在内部集群上用于持续评估。
- **OSWorld** 在 VM 集群中用于桌面智能体。
- **Computer-use 智能体**（Lesson 21）— Claude、OpenAI CUA、Gemini — 都在类似这些的工作负载上训练。
- **你自己的产品流程** — 为你的 top 20 任务捕获黄金轨迹；每周对它们运行智能体。

## Ship It

`outputs/skill-web-desktop-harness.md` 构建一个 web/桌面智能体 harness，带基于执行的 eval 和轨迹效率指标。

## 练习

1. 用第二个应用（论坛）扩展玩具 harness。编写 3 个任务加黄金轨迹。
2. 添加每任务的轨迹效率报告。在你的玩具上，智能体是黄金的 1x、2x 还是 3x？
3. 实现一个"干扰"工具——黄金轨迹从不使用的工具。脚本化智能体会被诱惑吗？
4. 阅读 OSWorld-G。你如何在自己的 eval 中将定位失败与规划失败分离？
5. 阅读 WebArena 的应用 README。当你升级一个固定的应用版本时会破坏什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| WebArena | "Web 智能体基准测试" | 跨 4 个自托管应用的 812 个任务；gym 风格评估 |
| VisualWebArena | "视觉 WebArena" | 视觉定位的 WebArena；截图是观察 |
| OSWorld | "桌面智能体基准测试" | 真实 Ubuntu/Windows/macOS 上的 369 个任务 |
| GUI 定位 | "像素到元素映射" | 模型在 1920x1080 中定位 UI 元素 |
| 操作知识 | "OS 经验" | 哪个菜单、哪个快捷键、哪个偏好设置面板 |
| OSWorld-G | "定位套件" | 564 个仅定位样本 + 训练集 |
| OSWorld-Human | "黄金轨迹" | 手动专家动作序列用于测量效率 |
| 轨迹效率 | "步骤超出黄金" | 智能体步骤数除以人类最小值 |

## 延伸阅读

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — 四应用 web 基准测试
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — 跨 OS 桌面基准测试
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 的基准测试形状能力
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld 和 WebArena 数字
