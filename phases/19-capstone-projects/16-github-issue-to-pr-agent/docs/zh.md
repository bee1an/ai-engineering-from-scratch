# 毕业项目 16 — GitHub Issue 到 PR 的自主智能体

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 都在 2026 年交付了同一产品形态：给 issue 打标签，拿到 PR。在云端沙箱中运行智能体，验证测试通过，然后发布一个附带理由说明的可审查 PR。难点在于自动复现仓库的构建环境、防止凭证泄露、执行每仓库预算，以及确保智能体不能 force-push。本毕业项目构建自托管版本，并在成本和通过率上与托管替代方案进行对比。

**类型：** 毕业项目
**语言：** Python（智能体）、TypeScript（GitHub App）、YAML（Actions）
**前置课程：** Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 15（自主）、Phase 17（基础设施）
**覆盖阶段：** P11 · P13 · P14 · P15 · P17
**时间：** 30 小时

## 问题

异步云端编码智能体是与交互式编码智能体（毕业项目 01）不同的产品类别。UX 是一个 GitHub 标签。你给 issue 打上 `@agent fix this` 标签，一个 worker 在云端沙箱中启动，克隆仓库，运行测试，编辑文件，验证，然后打开一个 PR 并在正文中附上智能体的理由说明。没有交互循环，没有终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 都收敛到了这个形态。

工程挑战是具体的：环境复现（智能体必须从零构建仓库，没有缓存的开发镜像）、不稳定测试（必须重跑或隔离）、凭证范围控制（具有最小细粒度权限的 GitHub App）、每仓库每天的预算执行，以及禁止 force-push 策略。本毕业项目衡量通过率、成本和安全性，并与托管替代方案对比。

## 概念

触发器是 GitHub webhook（issue 标签或 PR 评论）。调度器将工作入队到 ECS Fargate 或 Lambda。Worker 将仓库拉入 Daytona 或 E2B 沙箱，使用从仓库推断的通用 Dockerfile（语言、框架）。智能体运行 mini-swe-agent 或 SWE-agent v2 循环，对接 Claude Opus 4.7 或 GPT-5.4-Codex。它迭代：读代码、提出修复、应用补丁、运行测试。

验证是门控步骤。完整 CI 必须在沙箱内通过才能打开 PR。计算覆盖率差异；如果超过阈值为负，PR 仍然打开但标记为 `needs-review`。智能体将理由说明作为 PR 描述发布，加上一个 `@agent` 线程供审查者 ping 后续问题。

安全通过两个不同的 GitHub 表面来限定范围：App 提供短期安装令牌，具有 `workflows: read` 和窄范围的仓库内容/PR 权限；分支保护（而非 app 权限）强制"不能直接写入 `main`"和"不能 force-push"——app 永远不在绕过列表中。路径范围的只读访问 `.github/workflows` 不是 GitHub App 的真实原语，所以智能体对文件编辑的允许列表必须在 worker 端强制执行。每仓库每天的预算上限在调度器端执行（例如每仓库每天最多 5 个 PR，每个 PR $20）。

## 架构

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技术栈

- 触发器：具有细粒度令牌的 GitHub App；通过 Lambda 或 Fly.io 接收 webhook
- Worker：ECS Fargate 任务（或 GitHub Actions 自托管 runner）
- 沙箱：每任务一个 Daytona devcontainer 或 E2B sandbox
- 智能体循环：mini-swe-agent 基线或 SWE-agent v2，对接 Claude Opus 4.7 / GPT-5.4-Codex
- 检索：tree-sitter repo-map + ripgrep
- 验证：沙箱内完整 CI + 覆盖率差异门控
- 可观测性：Langfuse，每 PR trace 归档并从 PR 正文链接
- 预算：每仓库每天美元上限；每仓库每天最大 PR 数

## 构建步骤

1. **GitHub App。** 细粒度安装令牌：issues read+write、pull_requests write、contents read+write、workflows read。分支保护（唯一能做到这一点的表面）强制"不能直接 push 到 `main`"和"不能 force-push"；app 不在绕过列表中。Worker 强制"不能写入 `.github/workflows` 下的文件"作为对提议 diff 的允许列表检查，因为 GitHub App 权限不是路径范围的。

2. **Webhook 接收器。** Lambda 函数接受 issue 标签 / PR 评论 webhook。按标签 `@agent fix this` 过滤。入队到 SQS。

3. **调度器。** 从 SQS 弹出任务。执行每仓库每天预算。启动 ECS Fargate 任务，传入仓库 URL、issue 正文和一个新的 Daytona 沙箱。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）和包管理器（uv、pnpm、go mod、cargo）。如果不存在则动态生成 Dockerfile。

5. **智能体循环。** mini-swe-agent 或 SWE-agent v2，使用 Claude Opus 4.7。工具：ripgrep、tree-sitter repo-map、read_file、edit_file、run_tests、git。硬限制：$20 成本、30 分钟墙钟时间、30 个智能体轮次。

6. **验证。** 循环结束后，在沙箱内运行完整测试套件。通过 jacoco / coverage.py 计算覆盖率差异。如果 CI 红色：停止，不打开 PR。如果覆盖率下降超过 2%：打开 PR 并标记 `needs-review`。

7. **PR 发布。** Push 智能体分支。通过 GitHub API 打开 PR，包含：标题、理由说明、diff 摘要、trace URL、成本、轮次。

8. **凭证卫生。** Worker 使用短期 GitHub App 安装令牌运行。日志在归档前清洗密钥。

9. **评估。** 30 个不同难度的预设内部 issue。衡量通过率、PR 质量（diff 大小、风格、覆盖率）、成本、延迟。与 Cursor Background Agents 和 AWS Remote SWE Agents 在相同 issue 上对比。

## 使用示例

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 交付标准

`outputs/skill-issue-to-pr.md` 是交付物。一个 GitHub App + 异步云端 worker，将打标签的 issue 转化为可审查的 PR，具有有限成本和范围化凭证。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 30 个 issue 的通过率 | 端到端成功（CI 绿色 + 覆盖率 OK） |
| 20 | PR 质量 | Diff 大小、覆盖率差异、风格一致性 |
| 20 | 每个已解决 issue 的成本和延迟 | 每 PR 的美元和墙钟时间 |
| 20 | 安全性 | 范围化令牌、每仓库预算、禁止 force-push、凭证卫生 |
| 15 | 运维 UX | 理由说明评论、重试能力、@-mention 后续 |
| **100** | | |

## 练习

1. 添加"修复不稳定测试"模式：标签 `@agent stabilize-flake TestX` 在沙箱内运行测试 50 次，并提出稳定它的最小改动。

2. 在三个共享 issue 上比较与 Cursor Background Agents 的成本。报告哪个工具在哪里胜出。

3. 实现预算仪表板：每仓库每天成本、每用户成本。异常告警。

4. 构建"试运行"模式：打开 draft PR 而不运行 CI，让审查者低成本地检查计划。

5. 添加保留策略：超过 7 天未合并的 PR 分支自动删除。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| GitHub App | "范围化机器人身份" | 具有细粒度权限 + 短期安装令牌的 App |
| 异步云端智能体 | "后台智能体" | 在云端沙箱中运行的非交互式 worker，不是终端 |
| 环境推断 | "Dockerfile 合成" | 检测语言 + 包管理器，缺失时生成 Dockerfile |
| 验证 | "沙箱内 CI" | 在打开 PR 之前在 worker 内运行完整测试套件 |
| 覆盖率差异 | "覆盖率保持" | 从基线到智能体分支的测试覆盖率百分比变化 |
| 每仓库预算 | "每日上限" | 在调度器端执行的美元和 PR 数量上限 |
| 理由说明 | "PR 正文解释" | 智能体对改了什么以及为什么改的总结；PR 正文中必须包含 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 标准异步云端智能体参考
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 参考
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业替代方案
- [OpenAI Codex (cloud)](https://openai.com/codex) — 托管竞品
- [Google Jules](https://jules.google) — Google 的托管版本
- [Factory Droids](https://www.factory.ai) — 另一商业参考
- [GitHub App documentation](https://docs.github.com/en/apps) — 范围化机器人身份
- [Daytona cloud sandboxes](https://daytona.io) — 参考沙箱
