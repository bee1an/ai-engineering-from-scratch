# Tree of Thoughts 与 LATS：审慎搜索

> 单条 chain-of-thought 轨迹没有回溯的余地。ToT（Yao et al., 2023）将推理变成一棵树，每个节点都有自评估。LATS（Zhou et al., 2024）将 ToT、ReAct 和 Reflexion 统一在蒙特卡洛树搜索之下。Game of 24 从 4%（CoT）提升到 74%（ToT）；LATS 在 HumanEval 上达到 92.7% pass@1。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~75 minutes

## 学习目标

- 将推理框架化为搜索：节点是"思考"，边是"展开"，值是"有多大前景"。
- 用 stdlib 实现一个 ToT 风格的 BFS 树搜索，带自评估打分。
- 扩展为一个玩具 LATS MCTS 循环，包含 select / expand / simulate / backpropagate。
- 判断什么时候搜索值得付出 token 倍增的代价（Game of 24、代码生成），什么时候单条轨迹就够了（简单问答）。

## 问题

Chain-of-thought 是一条线性路径。如果第一步错了，后续每一步都建立在错误前提上。在 Game of 24（用四个数字和 + − × ÷ 凑出 24）上，GPT-4 CoT 只有 4% 准确率。模型在早期选错了子表达式，无法恢复。

推理需要的是：提出多个候选方案、评估它们、选择有前景的、在遇到死胡同时回溯。这就是搜索。Tree of Thoughts 和 LATS 是两种经典表述。

## 核心概念

### Tree of Thoughts（Yao et al., NeurIPS 2023）

每个节点是一个连贯的中间步骤（"一个思考"）。每个节点可以展开为 K 个子思考。LLM 用打分 prompt 对每个节点进行自评估。搜索探索这棵树 — BFS、DFS 或 beam。

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

自评估是承重部件。论文展示了三种变体：`sure / likely / impossible` 分类、`1..10` 数值分数、以及候选方案投票。三种都在 Game of 24 上大幅超越 CoT（4% -> 74%，GPT-4）。

### LATS（Zhou et al., ICML 2024）

LATS 将 ToT、ReAct 和 Reflexion 统一在 MCTS 之下。LLM 扮演三个角色：

- **Policy**：提出候选下一步动作（ReAct 风格）。
- **Value function**：对部分轨迹打分（ToT 风格自评估）。
- **Self-reflector**：失败时写一段自然语言反思（Reflexion 风格），用于重新播种未来的 rollout。

环境反馈（observations）混入 value function，使搜索基于真实工具结果而非仅模型意见。论文发表时的结果：GPT-4 在 HumanEval pass@1 达到 92.7%（SOTA），GPT-3.5 在 WebShop 平均 75.9（接近基于梯度的微调）。

### MCTS 极简版

每次迭代四个阶段：

1. **Select** — 用 UCT（树的置信上界）从根走到叶节点。
2. **Expand** — 通过 policy 生成 K 个子节点。
3. **Simulate** — 从子节点用 policy 做 rollout，用 value function（或环境奖励）对叶节点打分。
4. **Backpropagate** — 沿路径向上更新访问计数和价值估计。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是利用；第二项是探索。按任务调节 `c`。

### 成本现实

搜索会让 token 爆炸。ToT 在 Game of 24 上使用 CoT 的 100-1000 倍 token。LATS 类似。这不是免费的；将搜索保留给：

- 单条轨迹明显不够的任务（Game of 24、复杂代码）。
- 正确性比响应时间更重要的任务。
- 有廉价、可靠 value function 的任务（代码的单元测试、数学的明确目标）。

如果你的任务只有一个正确答案且评估器有噪声，搜索往往会让事情更糟 — 它会找到一个"高分"的错误答案。

### 2026 定位

大多数生产智能体不运行 LATS。它们运行带工具验证的 ReAct（CRITIC，Lesson 05）。搜索出现在专门的场景中：

- 将测试作为 value function 的编码智能体（HumanEval 风格）。
- 探索多条查询路径的深度研究智能体。
- LangGraph 子图中的重规划工作流。

AlphaEvolve（Lesson 11）是 2025 的极端案例：对代码做进化搜索，机器可检查的适应度，前沿突破（56 年来首次改进 4x4 矩阵乘法）。

## Build It

`code/main.py` 实现了：

- 一个在风格化"选择算术运算"任务上的小型 ToT BFS。
- 一个在同一任务上的玩具 LATS MCTS 循环（Select / Expand / Simulate / Backpropagate），带 UCT 选择。
- 一个组合符号分数和自评估分数的 value function。

运行：

```
python3 code/main.py
```

trace 展示 ToT 用 BFS 每个节点展开三个候选，对比 LATS 通过 MCTS 收敛到最佳 rollout。两者都打印 token 计数。

## Use It

LangGraph 将 ToT 风格的探索作为子图模式提供；LangChain 团队关于 LATS 的博客（2024 年 5 月）是参考教程。LlamaIndex 提供 `TreeOfThoughts` agent。对于大多数 2026 生产智能体，这个模式藏在 `if task_complexity > threshold: use_search()` 门控后面 — 参见 Lesson 05 的 evaluator-optimizer 模式。

## Ship It

`outputs/skill-search-policy.md` 根据任务形态、预算和评估器保真度，在线性 ReAct、ToT、LATS 和进化搜索之间做选择。

## 练习

1. 用 UCT c=0.1 vs c=2.0 运行玩具 LATS。trace 中有什么变化？
2. 将 value function 换成更有噪声的打分器（加随机抖动）。MCTS 还能找到最佳叶节点吗？它能容忍的最小信噪比是多少？
3. 实现 beam-search ToT（每层保留 top-k）并与 BFS 对比。在紧张的 token 预算下哪个更好？
4. 阅读 LATS 第 5.1 节。复现 HumanEval 轨迹计数：需要多少次 rollout 才能达到报告的 pass@1？
5. 阅读 LATS 论文关于"LATS 帮助较少的情况"的讨论。写一段决策规则，将任务形态映射到搜索策略。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Tree of Thoughts | "分支 CoT" | Yao et al. — 带自评估的思考节点树 |
| LATS | "LLM 的 MCTS" | Zhou et al. — 将 ToT + ReAct + Reflexion 统一在 MCTS 下 |
| UCT | "置信上界" | 平衡利用（Q）和探索（ln N / n）的选择公式 |
| Value function | "这个状态有多好" | Prompted LLM 分数或环境奖励；供 backprop 使用 |
| Policy | "动作提议者" | ReAct 风格生成器；输出候选下一步思考/动作 |
| Rollout | "模拟轨迹" | 从节点用 policy 走到叶节点，用 value 打分 |
| Backpropagate | "更新祖先" | 将叶节点的奖励沿路径向上推，更新访问计数和 Q |
| 搜索成本 | "Token 爆炸" | Game of 24 上是 CoT 的 100-1000 倍；采用前先做预算 |

## 延伸阅读

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — 原始论文
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — 带 Reflexion 反馈的 MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 搜索的子图模式
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — 带程序化评估器的进化搜索
