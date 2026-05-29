# 毕业项目 01 — 终端原生编码智能体

> 到 2026 年，编码智能体的形态已经定型。一个 TUI 外壳、一个有状态的计划、一个沙箱化的工具面、一个规划-执行-观察-恢复的循环。Claude Code、Cursor 3 和 OpenCode 从远处看都长一个样。这个毕业项目要求你端到端地构建一个——CLI 输入，Pull Request 输出——并在 SWE-bench Pro 上与 mini-swe-agent 和 Live-SWE-agent 进行对比。你会发现，难点不在模型调用，而在工具循环、沙箱和 50 轮运行的成本上限。

**类型：** 毕业项目
**语言：** TypeScript / Bun（外壳），Python（评估脚本）
**前置要求：** Phase 11（LLM 工程）、Phase 13（工具与协议）、Phase 14（智能体）、Phase 15（自主系统）、Phase 17（基础设施）
**涉及阶段：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**时间：** 35 小时

## 问题

编码智能体在 2026 年成为最主要的 AI 应用类别。Claude Code（Anthropic）、Cursor 3 的 Composer 2 和 Agent Tabs（Cursor）、Amp（Sourcegraph）、OpenCode（112k stars）、Factory Droids 和 Google Jules 都在同一架构上做变体：终端外壳、权限化的工具面、沙箱，以及围绕前沿模型构建的规划-执行-观察循环。前沿很窄——Live-SWE-agent 用 Opus 4.5 在 SWE-bench Verified 上达到了 79.2%——但工程技艺很宽。大多数失败模式不是模型犯错，而是工具循环不稳定、上下文污染、token 成本失控和破坏性文件系统操作。

你无法从外部推理这些智能体。你必须亲手构建一个，看着循环在第 47 轮崩溃——因为 ripgrep 返回了 8MB 的匹配结果——然后重建截断层。这就是这个毕业项目的意义。

## 概念

外壳有四个面。**Plan** 维护一个 TodoWrite 风格的状态对象，模型每轮重写它。**Act** 分发工具调用（read、edit、run、search、git）。**Observe** 捕获 stdout / stderr / 退出码，截断后将摘要反馈回去。**Recover** 处理工具错误，既不撑爆上下文窗口也不无限循环。2026 年的形态还多了一样东西：**Hooks**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact`——可配置的扩展点，运营方在此注入策略、遥测和护栏。

沙箱是 E2B 或 Daytona。每个任务在一个全新的 devcontainer 中运行，挂载一个可读写的 git worktree。外壳永远不碰宿主文件系统。worktree 在成功或失败后被销毁。成本控制在三层执行：每轮 token 上限、每会话美元预算、硬性轮次限制（通常 50 轮）。可观测性层是带有 GenAI 语义约定的 OpenTelemetry span，发送到自托管的 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- 外壳运行时：Bun 1.2 + Ink 5（React-in-terminal）
- 模型访问：OpenRouter 统一 API，支持 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最难的任务）
- 工具传输：Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙箱：E2B sandboxes（JS SDK）或 Daytona devcontainers
- 代码搜索：ripgrep 子进程，17 种语言的 tree-sitter 解析器（预编译）
- 隔离：每个任务 `git worktree add`，成功/失败后清理
- 评估框架：SWE-bench Pro（verified 子集）+ Terminal-Bench 2.0 + 你自己的 30 任务保留集
- 可观测性：OpenTelemetry SDK 带 `gen_ai.*` semconv → 自托管 Langfuse
- PR 发布：GitHub App 带细粒度 token，scope 限定到目标仓库

## 构建步骤

1. **TUI 和命令循环。** 用 Ink 搭建一个 Bun 项目。接受 `agent run <repo> "<task>"`。打印分屏视图：计划面板（上）、工具调用流（中）、token 预算（下）。Ctrl-C 取消时触发 `SessionEnd` hook 后退出。

2. **计划状态。** 定义一个类型化的 TodoWrite schema（pending / in_progress / done 条目带备注）。模型每轮以工具调用的形式重写完整状态——不要让它增量修改。将计划持久化到 `.agent/state.json`，这样崩溃后可以恢复。

3. **工具面。** 定义六个工具：`read_file`、`edit_file`（带 diff 预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（status / diff / commit / push）。通过 MCP StreamableHTTP 暴露，使外壳与传输无关。每个工具返回截断后的输出（每次调用上限 4k tokens）。

4. **沙箱封装。** 每个任务启动一个 E2B 沙箱。`git worktree add -b agent/$TASK_ID` 创建一个新分支。所有工具调用在沙箱内执行。宿主文件系统不可达。

5. **Hooks。** 实现全部八种 2026 hook 类型。至少接入四个用户编写的 hook：(a) `PreToolUse` 破坏性命令守卫，阻止 worktree 外的 `rm -rf`；(b) `PostToolUse` token 计费；(c) `SessionStart` 预算初始化；(d) `Stop` 写入最终 trace bundle。

6. **评估循环。** 克隆 SWE-bench Pro Python 的 30 个 issue 子集。对每个运行你的外壳。与 mini-swe-agent（最小基线）在 pass@1、每任务轮次和每任务美元上进行对比。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬性截断：50 轮、200k 上下文、每任务 $5。`PreCompact` hook 在 150k 标记处将旧轮次总结为先验状态块，为新观察腾出空间而不丢失计划。

8. **PR 发布。** 成功后，最后一步是 `git push` + 一个 GitHub API 调用，打开一个 PR，body 中包含计划和 diff 摘要。

## 使用示例

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付标准

交付技能文件位于 `outputs/skill-terminal-coding-agent.md`。给定一个仓库路径和任务描述，它在沙箱中运行完整的规划-执行-观察循环，返回一个 PR URL 加一个 trace bundle。评分标准：

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 对比基线 | 你的外壳 vs mini-swe-agent，30 个匹配的 Python 任务 |
| 20 | 架构清晰度 | Plan/act/observe 分离、hook 面、工具 schema——对照 Live-SWE-agent 布局评审 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示、破坏性命令守卫通过红队测试 |
| 20 | 可观测性 | Trace 完整性（100% 工具调用有 span）、每轮 token 计费 |
| 15 | 开发者体验 | 冷启动 < 2s、崩溃恢复能续接计划、Ctrl-C 能在工具执行中干净取消 |
| **100** | | |

## 练习

1. 将后端模型从 Claude Sonnet 4.7 换成在 vLLM 上部署的 Qwen3-Coder-30B。对比 pass@1 和每任务美元。报告开源模型在哪些地方表现不佳。

2. 添加一个 `reviewer` 子智能体，在 PR 发布前阅读 diff 并可以请求修订循环。衡量误报审查是否将 SWE-bench 通过率拉低到单智能体基线以下（提示：通常会）。

3. 压力测试沙箱：编写一个尝试 `curl` 外部 URL 的任务和一个尝试写入 worktree 外部的任务。确认两者都被 PreToolUse hook 阻止。记录尝试。

4. 用更小的模型（Haiku 4.5）实现 `PreCompact` 摘要。衡量 3 倍压缩后计划保真度损失了多少。

5. 将 MCP StreamableHTTP 传输换成 stdio。基准测试冷启动和每次调用延迟。为纯本地使用场景选出优胜者。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Harness | "智能体循环" | 围绕模型的代码，负责分发工具、维护计划状态、执行预算 |
| Hook | "智能体事件监听器" | 用户编写的脚本，在外壳的八个生命周期事件之一上运行 |
| Worktree | "Git 沙箱" | 在独立路径上的链接式 git checkout；可丢弃而不影响主克隆 |
| TodoWrite | "计划状态" | 一个类型化的 pending/in-progress/done 列表，模型每轮重写 |
| StreamableHTTP | "MCP 传输" | 2026 MCP 修订版：长连接 HTTP 双向流；取代 SSE |
| Token ceiling | "上下文预算" | 每轮或每会话的 input+output token 上限；触发压缩或终止 |
| pass@1 | "单次通过率" | SWE-bench 任务在首次运行中解决的比例，无重试、无测试集窥探 |

## 延伸阅读

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 的参考外壳
- [Cursor 3 changelog](https://cursor.com/changelog) — Agent Tabs 和 Composer 2 产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — SWE-bench 外壳对比的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 用 Opus 4.5 达到 79.2% SWE-bench Verified
- [OpenCode](https://opencode.ai) — 开源外壳，112k stars
- [SWE-bench Pro leaderboard](https://www.swebench.com) — 本毕业项目的评估目标
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 工具调用和 token 使用的 span schema
