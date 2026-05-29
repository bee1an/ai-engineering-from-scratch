# 毕业项目 10 — 多智能体软件工程团队

> SWE-AF 的工厂架构、MetaGPT 的角色提示、AutoGen 0.4 的类型化 actor 图、Cognition 的 Devin 和 Factory 的 Droids 都在 2026 年收敛到同一形态：架构师规划，N 个编码者在并行 worktree 中工作，审查者把关，测试者验证。并行 worktree 将挂钟时间转化为吞吐量。共享状态和交接协议成为失败面。这个毕业项目要求构建团队，在 SWE-bench Pro 上评估，并报告哪些交接断裂以及频率。

**类型：** 毕业项目
**语言：** Python / TypeScript（智能体），Shell（worktree 脚本）
**前置要求：** Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 15（自主系统）、Phase 16（多智能体）、Phase 17（基础设施）
**涉及阶段：** P11 · P13 · P14 · P15 · P16 · P17
**时间：** 40 小时

## 问题

单智能体编码外壳在大型任务上遇到天花板。不是因为任何单个智能体弱，而是因为 200k-token 上下文无法同时容纳架构计划加四个并行代码库切片加审查者评论加测试输出。多智能体工厂拆分问题：架构师拥有计划，编码者在并行 worktree 中拥有实现，审查者把关，测试者验证。SWE-AF 的"工厂"架构、MetaGPT 的角色、AutoGen 的类型化 actor 图——三种框架描述的是同一形态。

失败面在交接。架构师规划了编码者无法实现的东西。编码者产出冲突的 diff。审查者批准了幻觉修复。测试者与仍在编写的编码者竞争。你将构建这样一个团队，在 50 个 SWE-bench Pro issue 上运行，跟踪每次交接，并发布事后分析。

## 概念

角色是类型化的智能体。**架构师**（Claude Opus 4.7）阅读 issue，编写计划，将其分解为带显式接口的子任务。**编码者**（Claude Sonnet 4.7，N 个并行实例，每个在 `git worktree` + Daytona 沙箱中）独立实现子任务。**审查者**（GPT-5.4）阅读合并后的 diff，批准或请求具体修改。**测试者**（Gemini 2.5 Pro）在隔离环境中运行测试套件，报告通过/失败及产物。

通信通过共享任务板（文件或 Redis）。每个角色消费它被允许处理的任务。交接是 A2A 协议类型化消息。协调关注点：合并冲突解决（协调者角色或自动三方合并）、共享状态同步（编码者开始后计划冻结；重新规划是独立事件）、审查者把关（审查者不能批准自己的变更或自己提出的变更）。

Token 放大是隐藏成本。每个角色边界增加摘要提示和交接上下文。40 轮的单智能体运行变成跨四个角色的 160 总轮次。评分标准特别衡量 token 效率 vs 单智能体基线，因为问题不是"多智能体是否有效"而是"它是否在每美元上胜出"。

## 架构

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技术栈

- 编排：LangGraph 带共享状态 + 每智能体子图
- 消息传递：A2A 协议（Google 2025）用于类型化智能体间消息
- 模型：Opus 4.7（架构师）、Sonnet 4.7（编码者）、GPT-5.4（审查者）、Gemini 2.5 Pro（测试者）
- Worktree 隔离：每编码者 `git worktree add` + Daytona 沙箱
- 合并协调者：自定义三方合并 + LLM 介导的冲突解决
- 评估：SWE-bench Pro（50 issues）、SWE-AF 场景、HumanEval++ 用于单元测试
- 可观测性：Langfuse 带角色标签 span，每智能体 token 计费
- 部署：K8s 每角色一个独立 Deployment + HPA 基于积压

## 构建步骤

1. **任务板。** 文件支持的 JSONL 带类型化消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。智能体订阅标签。

2. **架构师。** 阅读 GitHub issue，用计划模板运行 Opus 4.7，要求显式子任务接口（涉及文件、公开函数、测试影响）。发出一个带子任务 DAG 的 `plan_request`。

3. **编码者。** N 个并行 worker，每个从板上认领一个子任务。每个启动一个新的 `git worktree add` 分支加 Daytona 沙箱。实现子任务。发出 `diff_ready` 带 patch + 测试增量。

4. **合并协调者。** 所有编码者完成后，将 N 个分支三方合并到 staging 分支。仅在文件级重叠时使用 LLM 介导的冲突解决。

5. **审查者。** GPT-5.4 阅读合并后的 diff。不能批准自己编写的 diff。发出 `approved`（无操作）或 `review_feedback` 带路由回相关编码者的具体修改请求。

6. **测试者。** Gemini 2.5 Pro 在干净沙箱中运行测试套件。捕获产物。发出 `test_passed` 或 `test_failed` 带堆栈跟踪。失败测试循环回拥有失败子任务的编码者。

7. **交接计费。** 每条跨角色边界的消息在 Langfuse 中获得一个 span，带 payload 大小和使用的模型。计算每子任务 token 放大（coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens）。

8. **评估。** 在 50 个 SWE-bench Pro issue 上运行。与单智能体基线（一个 Sonnet 4.7 在单个 worktree 中）对比 pass@1 和每解决 issue 的美元。

9. **事后分析。** 对每个失败 issue，识别断裂的交接（计划太模糊、合并冲突、审查者误批准、测试者 flake）。产出交接失败直方图。

## 使用示例

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 交付标准

`outputs/skill-multi-agent-team.md` 是交付物。给定一个 issue URL 和并行度，团队产出一个可合并的 PR，带每角色 token 计费。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 匹配的 50-issue 子集，pass@1 |
| 20 | 并行加速 | 挂钟时间 vs 单智能体基线 |
| 20 | 审查质量 | 注入 bug 探测上的误批准率 |
| 20 | Token 效率 | 每解决 issue 的总 token vs 单智能体 |
| 15 | 协调工程 | 合并冲突解决、交接失败直方图 |
| **100** | | |

## 练习

1. 在运行中向 diff 注入一个明显 bug（在主体前额外加 `return None`）。衡量审查者的误批准率。调优审查者提示直到误批准低于 5%。

2. 减少到两个编码者（架构师 + 编码者 + 审查者 + 测试者，编码者顺序运行两个子任务）。对比挂钟时间和通过率。

3. 用单写者约束替换合并协调者（子任务触及不相交的文件集）。衡量架构师的规划负担。

4. 将审查者从 GPT-5.4 换成 Claude Opus 4.7。衡量误批准率和 token 成本差异。

5. 添加第五个角色：文档员（Haiku 4.5）。审查后产出 changelog 条目。衡量文档质量是否值得额外的 token 花费。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 并行 worktree | "隔离分支" | `git worktree add` 为每个编码者产出一个新的工作树 |
| 任务板 | "共享消息总线" | 智能体订阅的类型化消息的文件或 Redis 存储 |
| 交接 | "角色边界" | 从一个角色上下文跨越到另一个的任何消息 |
| Token 放大 | "多智能体开销" | 跨角色总 token / 同一任务的单智能体 token |
| A2A 协议 | "Agent-to-agent" | Google 2025 年的类型化智能体间消息规范 |
| 合并协调者 | "集成者" | 运行三方合并并介导冲突的组件 |
| 误批准 | "审查者幻觉" | 审查者批准了带已知 bug 的 diff |

## 延伸阅读

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 参考 2026 多智能体工厂
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 基于角色的多智能体框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft 的类型化 actor 框架
- [Cognition AI (Devin)](https://cognition.ai) — 参考产品
- [Factory Droids](https://www.factory.ai) — 备选参考产品
- [Google A2A protocol](https://developers.google.com/agent-to-agent) — 智能体间消息规范
- [git worktree documentation](https://git-scm.com/docs/git-worktree) — 隔离基底
- [SWE-bench Pro](https://www.swebench.com) — 评估目标
