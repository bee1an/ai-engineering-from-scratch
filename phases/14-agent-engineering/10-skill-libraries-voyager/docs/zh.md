# 技能库与终身学习（Voyager）

> Voyager（Wang et al., TMLR 2024）将可执行代码视为技能。技能是命名的、可检索的、可组合的，并通过环境反馈精炼。这是 Claude Agent SDK skills、skillkit 和 2026 技能库模式的参考架构。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## 学习目标

- 说出 Voyager 的三个组件 — automatic curriculum、skill library、iterative prompting — 及各自的作用。
- 解释为什么 Voyager 将动作空间设为代码而非原始命令。
- 用 stdlib 实现一个技能库，带注册、检索、组合和失败驱动的精炼。
- 将 Voyager 的模式映射到 2026 的 Claude Agent SDK skills 和 skillkit 生态。

## 问题

每次会话都从零重建所有能力的智能体有三个问题：

1. **浪费 token。** 每个任务重新引出相同的推理。
2. **丢失进展。** 会话 A 中学到的修正不会转移到会话 B。
3. **长程组合失败。** 复杂任务需要能力层级；one-shot prompt 无法表达。

Voyager 的答案：将每个可复用能力视为存储在库中的命名代码块，按相似度可检索，可与其他技能组合，并通过执行反馈精炼。

## 核心概念

### 三个组件

Voyager（arXiv:2305.16291）围绕以下结构组织智能体：

1. **Automatic curriculum。** 一个好奇心驱动的提议者根据智能体当前的技能集和环境状态选择下一个任务。探索是自底向上的。
2. **Skill library。** 每个技能是可执行代码。任务成功时添加新技能。技能通过查询到描述的相似度检索。
3. **Iterative prompting mechanism。** 失败时，智能体收到执行错误、环境反馈和自验证输出，然后精炼技能。

Minecraft 评估（Wang et al., 2024）：相比基线，3.3 倍更多独特物品、8.5 倍更快的石器工具、6.4 倍更快的铁器工具、2.3 倍更长的地图遍历。数据是 Minecraft 特定的，但模式可迁移。

### 动作空间 = 代码

大多数智能体输出原始命令。Voyager 输出 JavaScript 函数。一个技能是：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

由子技能组合而成。以描述和 embedding 为键存储。作为程序检索，而非 prompt。

这就是 2026 的 Claude Agent SDK skill：一个命名的、可检索的代码块加指令，智能体按需加载。

### 技能检索

新任务"制作一把钻石镐。"智能体：

1. 对任务描述做 embedding。
2. 查询技能库获取 top-k 相似技能。
3. 检索到 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 从检索到的原语 + 新逻辑组合新技能。

这就是 MCP resources（Phase 13）和 Agent SDK skills 实现的模式：在知识/代码面上检索，范围限定到当前任务。

### 迭代精炼

Voyager 的反馈循环：

1. 智能体写一个技能。
2. 技能在环境中运行。
3. 三种信号之一返回：`success`、`error`（带堆栈跟踪）、`self-verification failure`。
4. 智能体用信号作为上下文重写技能。
5. 循环直到成功或达到最大轮数。

这是 Self-Refine（Lesson 05）应用于代码生成，带环境锚定验证。CRITIC（Lesson 05）是同一模式，用外部工具作为验证器。

### Curriculum 与探索

Voyager 的 curriculum 模块基于智能体已有的和尚未做过的事情提出任务，如"在湖边建一个庇护所"。提议者使用环境状态 + 技能清单来选择刚好超出当前能力的任务 — 探索的甜蜜点。

对于生产智能体，这转化为一个"缺什么"算子：给定当前技能库和一个领域，我们还没覆盖哪些技能？团队通常将此作为 curriculum review 手动实现。

### 这个模式哪里会出错

- **技能库腐化。** 同一技能以略有不同的描述被添加 10 次。写入时添加去重；检索只返回一个。
- **组合技能漂移。** 父技能依赖一个被精炼过的子技能。对技能做版本控制；固定在 v1 的父技能不会自动获取 v3。
- **检索质量。** 对技能描述的向量检索在库超过几百个后退化。用标签过滤和硬约束补充（"只要 `category=tooling` 的技能"）。

## Build It

`code/main.py` 用 stdlib 实现一个技能库：

- `Skill` — name、description、code（字符串）、version、tags、dependencies。
- `SkillLibrary` — register、search（token 重叠）、compose（依赖的拓扑排序）和 refine（更新时版本递增）。
- 一个脚本化智能体注册三个原始技能、组合第四个、遇到失败、然后精炼。

运行：

```
python3 code/main.py
```

trace 展示库写入、检索、组合、一次失败的执行和一次 v2 精炼 — Voyager 的端到端循环。

## Use It

- **Claude Agent SDK skills**（Anthropic）— 2026 参考：每个 skill 有描述、代码和指令；在智能体会话中按需加载。
- **skillkit**（npm: skillkit）— 32+ AI 编码智能体的跨智能体技能管理。
- **自定义技能库** — 领域特定（数据智能体的 SQL 技能、基础设施智能体的 Terraform 技能）。Voyager 模式可以缩小规模。
- **OpenAI Agents SDK `tools`** — 在低端；每个 tool 是一个轻量级技能。

## Ship It

`outputs/skill-skill-library.md` 为任何目标运行时生成一个 Voyager 形状的技能库，带注册、检索、版本控制和精炼。

## 练习

1. 给 `compose()` 添加依赖循环检测器。当技能 A 依赖 B 而 B 依赖 A 时会怎样？报错还是警告？
2. 实现每技能版本固定。当父技能组合子技能 `crafting@1` 时，对 `crafting@2` 的精炼不能静默升级父技能。
3. 将 token 重叠检索替换为 sentence-transformers embeddings（或 BM25 stdlib 实现）。在 50 技能的玩具库上衡量 retrieval@5。
4. 添加一个"curriculum"智能体：给定当前库和领域描述，提出 5 个缺失技能。每周调用一次。
5. 阅读 Anthropic 的 Claude Agent SDK skill 文档。将玩具库移植到 SDK 的 skill schema。可发现性有什么变化？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Skill | "可复用能力" | 命名的代码块 + 描述，按相似度可检索 |
| Skill library | "智能体的 how-to 记忆" | 技能的持久存储，可搜索可组合 |
| Curriculum | "任务提议者" | 由当前能力差距驱动的自底向上目标生成器 |
| Composition | "技能 DAG" | 技能调用技能；执行时拓扑排序 |
| Iterative refinement | "自修正循环" | 环境反馈 + 错误 + 自验证折叠回下一版本 |
| Action-space-as-code | "程序化动作" | 输出函数而非原始命令，用于时间扩展行为 |
| Dedup on write | "技能折叠" | 近重复描述折叠为一个规范技能 |

## 延伸阅读

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — 原始技能库论文
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — skills 作为 2026 产品化
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — skills 和 subagents 实践
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — Voyager 底层的精炼循环
