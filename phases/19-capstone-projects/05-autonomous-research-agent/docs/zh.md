# 毕业项目 05 — 自主研究智能体（AI-Scientist 级别）

> Sakana 的 AI-Scientist-v2 发表了完整论文。Agent Laboratory 运行了实验。Allen AI 分享了 trace。2026 年的形态是：对实验进行 plan-execute-verify 树搜索、预算化成本、沙箱化代码执行、视觉反馈的 LaTeX 写作器，以及自动化的 NeurIPS 风格审稿人集成。这个毕业项目要求你构建一个，端到端运行，每篇论文控制在 $30 以内，并通过 Sakana 记录的沙箱逃逸红队测试。

**类型：** 毕业项目
**语言：** Python（智能体 + 沙箱），LaTeX（输出）
**前置要求：** Phase 2（ML）、Phase 3（深度学习）、Phase 7（Transformer）、Phase 10（从零构建 LLM）、Phase 14（智能体）、Phase 15（自主系统）、Phase 16（多智能体）、Phase 18（安全）
**涉及阶段：** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**时间：** 40 小时

## 问题

自主研究智能体在 2026 年跨过了一个门槛。Sakana AI 的 AI-Scientist-v2 发表在 Nature 上，生成的论文通过了 workshop 同行评审。ShinkaEvolve（ICLR 2026）将路线扩展到进化假设。AMD 的 Agent Laboratory 发布了可复现的 trace。这些智能体不是魔法——它们是在候选实验树上运行的 plan-execute-verify 循环，带有成本上限、种子绑定的沙箱和自动化审查。技艺在于循环、预算和安全故事。

你通过在一个窄领域（例如，100M 参数 transformer 上的注意力稀疏性消融）对一个种子想法实现循环来学习。价值不在于第一次运行就发现新东西。价值在于基础设施：树搜索、实验沙箱、写作-审稿循环、红队报告。Sakana 团队记录了沙箱逃逸失败；你的智能体必须通过同样的红队测试。

## 概念

智能体是一个最佳优先树搜索。节点是实验规格：（假设、配置、代码、预期结果）。展开步骤提出带小编辑的子节点（换优化器、调 batch size、消融一个组件）。每个子节点在一个有硬资源上限的全新沙箱中运行。结果反馈到一个评分函数，按（新颖性 × 质量 × 剩余预算）排序节点。树增长直到预算耗尽，然后最佳分支被写成论文。

写作器是多模态的。它生成 LaTeX 草稿，编译，渲染图表，将渲染的 PDF 反馈给 Claude Opus 4.7 的视觉模式进行布局、图表可读性和声明-证据对齐的批评。五个 LLM 评委的审稿人集成发出 NeurIPS 风格评分（新颖性、严谨性、清晰度、可复现性、影响力）；如果平均分低于阈值，论文带着批评返回写作器。

安全是承重的。每个实验在 E2B 或 Daytona 沙箱中运行，无网络出口、有限挂钟时间和固定资源限制。智能体的代码生成步骤通过一个策略层，阻止逃逸沙箱的系统调用。红队报告复现了 Sakana 记录的攻击面（fork bomb、文件系统逃逸、LLM 编写的网络调用）。

## 架构

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- 编排：LangGraph 带 checkpointing 和人工审批门
- 树搜索：自定义最佳优先搜索，覆盖实验节点（Sakana v2 的 AB-MCTS 风格）
- 沙箱：每个实验一个 E2B，Docker-in-Docker 备选；通过 cgroups 限制资源
- 文献：Semantic Scholar Graph API + OpenAlex + 本地 FAISS 摘要缓存
- 写作器：LaTeX 模板 + Claude Opus 4.7（视觉模式）用于图表批评和布局
- 审稿人：5 个评委集成（Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max）带加权聚合
- 实验框架：PyTorch 2.5 用于物理实验，W&B 用于日志
- 可观测性：Langfuse 用于智能体 trace，每篇论文 $30 硬预算

## 构建步骤

1. **种子和领域界定。** 取一个种子想法（例如，"研究 sub-1B transformer 中注意力图的稀疏性模式"）。定义搜索空间：模型、数据集、计算预算。

2. **文献检索。** 查询 Semantic Scholar + OpenAlex 获取 50 篇最高引用的相关论文；本地缓存摘要；生成 1 页领域摘要。

3. **树脚手架。** 用种子假设初始化根节点。实现 `expand(node) -> children`，带小编辑提案（每个子节点一个配置变更）。实现 `score(node)` 为加权的新颖性 × 质量 × 预算项。

4. **沙箱封装。** 每个实验运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等效的 E2B 策略）。种子写入沙箱；输出以只读方式挂载回来。

5. **Plan-execute-verify 循环。** `plan` 提出子节点。`execute` 运行沙箱，捕获日志和指标。`verify` 对指标运行单元检查（loss 是否下降？消融是否隔离了效果？）。失败节点在树上存储失败原因。

6. **写作器。** 预算用完后，选择最佳分支。用 matplotlib 渲染图表。通过 Claude Opus 4.7 生成 LaTeX 草稿，上下文中包含分支 trace。编译。将编译后的 PDF 反馈给 Opus 4.7 视觉进行批评。迭代。

7. **审稿人集成。** 五个评委按（新颖性、严谨性、清晰度、可复现性、影响力）对草稿评分，使用 NeurIPS 风格评分标准。如果均分 < 4.0/5，带批评返回写作器。3 次重写后硬停止。

8. **红队。** 构建或集成一组针对沙箱的对抗性任务：fork bomb、网络渗出尝试、文件系统逃逸、LLM 编写的 shell 元字符。确认全部被阻止。写出发现。

9. **可复现性。** 每篇论文附带其树搜索 trace JSON、种子、W&B run 链接、沙箱配置，以及一个端到端复现的 README。

## 使用示例

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 交付标准

`outputs/skill-ai-scientist.md` 是交付物。给定一个种子想法 + 一个领域 + $30 预算，它运行完整管道并输出一篇可审阅的论文加可复现性 bundle。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 论文质量 | 与已发表 workshop 论文的盲审评分对比 |
| 20 | 实验严谨性 | 基线、种子、消融；每个声明都有结果表中的单元格支撑 |
| 20 | 成本和计算纪律 | $30/论文上限强制执行，Langfuse 追踪 |
| 20 | 安全性 | 沙箱红队测试通过；网络策略和 kill-switch 验证 |
| 15 | 可复现性 | 一条命令用相同种子重跑复现论文 |
| **100** | | |

## 练习

1. 在同一领域对三个不同种子想法运行管道。对比树搜索的哪些部分重叠。识别重复浪费的计算。

2. 在估计超过 $5 的节点执行前添加人工审批门。衡量总成本下降了多少。

3. 将审稿人集成换成单个评委。在一组已知差论文的保留集上衡量误接受率。

4. 引入网络渗出红队测试：智能体编写尝试 `curl` 外部地址的代码。确认 `--network=none` 策略阻止了它。记录尝试。

5. 将你的树搜索与平坦随机基线（相同预算，无展开策略）对比。报告新颖性 × 质量增益。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 树搜索 | "AB-MCTS 风格展开" | 在实验节点上的最佳优先探索，带新颖性×质量×预算评分 |
| 沙箱 | "实验隔离" | 无网络、有限 CPU/内存、固定种子、只读输入的容器 |
| 视觉批评 | "渲染后阅读" | 将论文编译为 PDF，将 PDF 反馈给 VLM 进行布局和声明-证据批评 |
| 审稿人集成 | "自动化同行评审" | 多个 LLM 评委用 NeurIPS 评分标准对论文打分；加权聚合控制管道 |
| 新颖性评分 | "这是新的吗？" | 惩罚与 50 篇文献缓存接近度的启发式 |
| 成本上限 | "$ 预算" | 每篇论文总花费的硬上限；Langfuse 计数器 + 预运行估算 |
| 红队 | "沙箱逃逸审计" | 如果策略有误就会逃逸沙箱的对抗性任务 |

## 延伸阅读

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — 参考生产研究智能体
- [Sakana AI-Scientist-v1 paper (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — 原始方法论
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — 进化扩展
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — 多角色研究实验室框架
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — 参考编排层
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — 文献搜索
- [E2B sandboxes](https://e2b.dev) — 参考实验隔离
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — 审稿人集成编码的评分标准
