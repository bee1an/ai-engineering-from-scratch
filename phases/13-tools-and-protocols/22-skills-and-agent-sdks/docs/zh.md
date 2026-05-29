# Skills 与 Agent SDK — Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说的是"有哪些工具"。Skills 说的是"如何完成一个任务"。2026 年的技术栈将两者分层组合。Anthropic 的 Agent Skills（开放标准，2025 年 12 月）以 SKILL.md 形式发布，支持渐进式披露。OpenAI 的 Apps SDK 是 MCP 加 widget 元数据。AGENTS.md（已被 60,000+ 仓库采用）放在仓库根目录，作为项目级 agent 上下文。本课讲解每层覆盖的内容，并构建一个可跨 agent 使用的最小 SKILL.md + AGENTS.md 包。

**Type:** Learn
**Languages:** Python (stdlib, SKILL.md parser and loader)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## 学习目标

- 区分三个层次：AGENTS.md（项目上下文）、SKILL.md（可复用知识）、MCP（工具）。
- 编写带 YAML frontmatter 和渐进式披露的 SKILL.md。
- 以文件系统方式将 skills 加载到 agent 运行时。
- 将 skill 与 MCP server 和 AGENTS.md 组合，使一个包在 Claude Code、Cursor 和 Codex 中都能工作。

## 问题

一位工程师将发布说明编写工作流提炼为多步 prompt："读取最新合并的 PR。按领域分组。逐个摘要。按团队风格写 changelog 条目。发到 Slack 草稿。"他们把它放在 Notion 文档中供团队使用。

现在他们想从 Claude Code、Cursor 和 Codex CLI 使用这个工作流。每个 agent 加载指令的方式不同：Claude Code 斜杠命令、Cursor rules、Codex `.codex.md`。工程师复制了三份工作流，维护三份副本。

AGENTS.md 和 SKILL.md 一起解决这个问题：

- **AGENTS.md** 放在仓库根目录。每个兼容的 agent 在会话开始时读取它。"这个项目怎么运作？有什么约定？哪个命令跑测试？"
- **SKILL.md** 是一个可移植的包：YAML frontmatter（name、description）+ markdown 正文 + 可选资源。支持 skills 的 agent 按名称按需加载。
- **MCP**（Phase 13 · 06-14）处理 skill 需要调用的工具。

三层，一个可移植的制品。

## 概念

### AGENTS.md (agents.md)

2025 年底推出，到 2026 年 4 月已被 60,000+ 仓库采用。仓库根目录一个文件。格式：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

Agent 在会话开始时读取此文件，用它来校准对该项目的行为。2026 年所有编码 agent 都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（2025 年 12 月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter 声明 skill 的身份。正文是 skill 加载时展示给模型的 prompt。

### 渐进式披露

Skills 可以引用子资源，agent 仅在需要时才获取。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 说"参见 style-guide.md 了解风格规则。"Agent 仅在 skill 实际运行时才拉取 style-guide.md。这避免了用模型可能不需要的细节膨胀 prompt。

### 文件系统发现

Agent 运行时扫描已知目录中的 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

按文件夹名和 frontmatter `name` 加载。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨 agent）都遵循此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话开始时加载 skills，将其作为运行时内可调用的 "agents" 暴露。Agent 循环在用户调用时分派到对应 skill。

### OpenAI Apps SDK

2025 年 10 月推出；直接构建在 MCP 之上。将 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一到单一开发者界面。一个 Apps SDK 应用是：

- 一个 MCP server（tools、resources、prompts）。
- 加上 ChatGPT UI 的 widget 元数据。
- 加上可选的 MCP Apps `ui://` resource 用于交互界面。

同一协议，更丰富的 UX。

### 通过 SkillKit 实现跨 agent 可移植性

SkillKit 等跨 agent 分发层将单个 SKILL.md 翻译为 32+ AI agent（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）的原生格式。一个事实来源；多个消费者。

### 三层栈

| 层 | 文件 | 加载时机 | 用途 |
|-------|------|-------------|---------|
| AGENTS.md | 仓库根目录 | 会话开始 | 项目级约定 |
| SKILL.md | skills 目录 | skill 被调用 | 可复用工作流 |
| MCP server | 外部进程 | 需要工具时 | 可调用的操作 |

三者组合：agent 在会话开始时读取 AGENTS.md，用户调用一个 skill，skill 的指令包含 MCP tool 调用，agent 通过 MCP client 分派。

## Use It

`code/main.py` 提供了一个 stdlib SKILL.md 解析器和加载器。它发现 `./skills/` 下的 skills，解析 YAML frontmatter 和 markdown 正文，生成按 skill 名称为 key 的字典。然后模拟一个 agent 循环，按名称调用 `release-notes-writer`。

关注点：

- YAML frontmatter 用最小 stdlib 解析器解析（无 `pyyaml` 依赖）。
- Skill 正文原样存储；agent 在调用时将其前置到 system prompt。
- 渐进式披露通过 `read_subresource` 函数演示，按需拉取引用的文件。

## Ship It

本课产出 `outputs/skill-agent-bundle.md`。给定一个工作流，该 skill 生成组合的 SKILL.md + AGENTS.md + MCP-server-blueprint 包，可跨 agent 移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个 skill，确认加载器能发现它。

2. 为本课程仓库编写一个 AGENTS.md。包含测试命令、风格约定和 Phase 13 的心智模型。

3. 将团队内部文档中的一个多步工作流移植为 SKILL.md。验证它能在 Claude Code 中加载。

4. 手动将该 skill 翻译为 Cursor 和 Codex 的原生规则格式。计算格式间的差异——这就是 SkillKit 自动化的翻译面。

5. 阅读 Anthropic Agent Skills 博客文章。找出 Claude Agent SDK 中本课加载器未覆盖的一个功能。（提示：agent 子调用。）

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| SKILL.md | "Skill 文件" | YAML frontmatter 加 markdown 正文，由 agent 运行时加载 |
| AGENTS.md | "仓库根目录 agent 上下文" | 会话开始时读取的项目级约定文件 |
| Progressive disclosure | "延迟加载子资源" | Skill 正文引用的文件仅在需要时拉取 |
| Frontmatter | "顶部 YAML 块" | `---` 分隔符中的元数据（name、description） |
| Claude Agent SDK | "Anthropic 的 skill 运行时" | `@anthropic-ai/claude-agent-sdk`，加载 skills 并路由 |
| OpenAI Apps SDK | "MCP + widget 元数据" | OpenAI 基于 MCP 加 ChatGPT UI hooks 的开发者界面 |
| Skill discovery | "文件系统扫描" | 遍历已知目录查找 SKILL.md，按名称索引 |
| Cross-agent portability | "一个 skill 多个 agent" | 通过 SkillKit 类工具将一个 SKILL.md 翻译到 32+ agent |
| Agent Skill | "可移植的知识" | MCP tool 概念之外的可复用任务模板 |
| Apps SDK | "MCP 加 ChatGPT UI" | Connectors 和 Custom GPTs 统一到 MCP 上 |

## 延伸阅读

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 基于 MCP 的 ChatGPT 开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式和采用列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方 skill 示例
