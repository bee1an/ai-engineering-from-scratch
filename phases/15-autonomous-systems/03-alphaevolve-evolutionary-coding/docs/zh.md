# AlphaEvolve — 进化式编码智能体

> 将前沿编码模型与进化循环和机器可验证评估器配对。让循环运行足够长时间。它发现了一个使用 48 次标量乘法的 4x4 复数矩阵乘法过程——56 年来首次超越 Strassen。它还找到了一个 Google 全局 Borg 调度启发式算法，在生产中回收了约 0.7% 的集群算力。架构故意设计得很无聊。胜利来自评估器的严格性。

**Type:** Learn
**Languages:** Python (stdlib, evolutionary-loop toy)
**Prerequisites:** Phase 15 · 01 (long-horizon framing), Phase 15 · 02 (self-taught reasoning)
**Time:** ~60 minutes

## 问题

大语言模型能写代码。进化算法能在代码上搜索。两者已经分别尝试了几十年；两者都碰到了天花板。LLM 的天花板是幻觉：模型写出看似合理但实际上不做它声称之事的代码。进化的天花板是搜索成本：对语法的随机变异很少产生可编译的程序，更别说更好的了。

AlphaEvolve（Novikov et al., DeepMind, arXiv:2506.13131, 2025 年 6 月）将两者结合。LLM 对程序数据库提出有针对性的编辑；自动评估器对每个变体打分；高分变体成为下一代的父代。LLM 处理写出合理代码这个昂贵步骤；评估器捕捉幻觉。循环运行数小时到数周。

报告的结果：48 次标量乘法的 4x4 复数矩阵乘法（Strassen 1969 年的界是 49），Google 生产中的 Borg 调度启发式，32.5% 的 FlashAttention 内核加速，Gemini 训练吞吐量改进。

这个架构之所以有效，是因为评估器是机器可验证的。在评估器不可验证的地方它就不行。这种不对称性就是本课的核心。

## 概念

### 循环

1. 从一个正确但次优的种子程序 `P_0` 开始。
2. 维护一个变体程序数据库，每个都由评估器打分。
3. 从数据库中采样一个或多个父代（MAP-elites 风格或岛模型）。
4. 提示 LLM（Gemini Flash 用于大量候选，Gemini Pro 用于难题）生成父代的修改变体。
5. 编译、运行并在留出评估器上评估变体。
6. 按分数和特征向量为键插入数据库。
7. 重复。

两个细节很重要。第一，LLM 的提示不仅包含父程序——通常还有数据库中的几个顶级变体，加上评估器签名，加上简短的任务描述。模型的工作是提出一个可能提高分数的有针对性的修改。第二，数据库是结构化的（MAP-elites 网格，岛模型），所以循环探索多样性，而不仅仅是当前领先者。

### 为什么评估器不可妥协

AlphaEvolve 的所有胜利都来自评估器快速、确定性且难以博弈的领域：

- **矩阵乘法算法**：一个单元测试，乘矩阵并逐位检查相等性。
- **Borg 调度启发式**：一个生产级模拟器，重放历史集群负载并测量浪费的算力。
- **FlashAttention 内核**：正确性测试加上真实硬件上的挂钟基准。
- **Gemini 训练吞吐量**：测量的 GPU 秒/步。

在每种情况下，评估器都捕捉了否则会主导的 LLM 错误类别：虚构的正确性声明、在硬件上消失的性能声明、以及边缘情况失败。移除评估器，循环就会优化漂亮的代码。

### 奖励黑客是同一陈述的另一面

进化优化评估器测量的任何东西。如果评估器不完美，循环会找到不完美之处。在未验证的领域中，循环会优化表面特征，而非预期行为。DeepMind 在论文中明确指出：AlphaEvolve 的成功仅转移到评估器严格性匹配搜索雄心的领域。

2025-2026 年代码搜索循环中奖励黑客的具体例子：

- 奖励"完成时间"的优化目标奖励了提交空解。
- 奖励测试下正确性的基准分数奖励了记忆测试和过拟合。
- "代码质量"代理奖励了删除注释和重写变量名，没有语义变化。

AlphaEvolve 的修复：发布一个 LLM 从未见过的留出评估器，输入在评估时生成。即便如此，DeepMind 仍建议对任何提议的部署进行强审查。

### 为什么 LLM + 搜索胜过单独使用任一

LLM 能产生可编译的、语义上合理的修改。对 2000 行 Python 文件的随机变异 GA 几乎总是产生语法错误。LLM 还将搜索集中在合理的邻域（改变一个函数，而非随机字节），这大大减少了浪费的评估器调用。

评估器反过来捕捉 LLM 的幻觉。LLM 会自信地声称一个函数"在极限下是 O(n log n)"而实际上是 O(n^2)；挂钟基准让问题尘埃落定。

### AlphaEvolve 在前沿技术栈中的位置

| 系统 | 生成器 | 评估器 | 领域 | 示例成果 |
|---|---|---|---|---|
| AlphaEvolve | Gemini | 正确性 + 基准 | 算法、内核、调度器 | 48-mul 4x4 matmul |
| FunSearch (DeepMind, 2023) | PaLM / Codey | 正确性 | 组合数学 | cap-set 下界 |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM 评审 + 实验 | ML 研究 | ICLR workshop 论文 |
| Darwin Godel Machine (L4) | 智能体脚手架 | SWE-bench / Polyglot | 智能体代码 | 20% → 50% SWE-bench |

四者都是同一配方的变体：生成器加评估器，循环。区别在于评估器评什么以及它有多严格。

## Use It

`code/main.py` 在一个玩具符号回归问题上实现了一个最小的 AlphaEvolve 风格循环。"LLM"是一个 stdlib 代理，对计算目标函数的程序提出小的语法变异。"评估器"在留出测试点上测量均方误差。

观察：

- 最佳分数如何随代数改进。
- MAP-elites 网格如何保持多样解存活，使循环不会收敛到局部最小值。
- 移除留出测试（仅训练评估器）如何让循环壮观地过拟合。

## Ship It

`outputs/skill-evaluator-rigor-audit.md` 是在新领域考虑 AlphaEvolve 风格循环的前提条件：你的评估器是否真的能捕捉你关心的失败？

## 练习

1. 运行 `code/main.py`。记录最佳分数轨迹。禁用留出评估器（标志 `--no-holdout`）并重新运行。量化过拟合。

2. 阅读 AlphaEvolve 论文第 3 节关于 MAP-elites 网格的内容。为一个新问题（如编译器优化 pass）设计一个特征向量描述符，使搜索保持多样性。

3. 48 次乘法的 4x4 结果在 56 年后改进了 Strassen 的 49 次乘法界。阅读论文附录 F，用三句话解释为什么这个问题的评估器特别容易做对，以及为什么大多数领域不是这样。

4. 提出一个 AlphaEvolve 会失败的领域。准确指出评估器在哪里崩溃以及为什么。

5. 对你了解的一个领域，写出你会使用的评估器签名。包括（a）正确性条件，（b）性能指标，（c）留出输入生成规则，（d）至少一个反奖励黑客检查。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|---|---|---|
| AlphaEvolve | "DeepMind 的进化编码智能体" | Gemini + 程序数据库 + 机器可验证评估器 |
| MAP-elites | "保持多样性的存档" | 按特征向量为键的网格；每个单元格保存具有该描述符的最佳变体 |
| Island model | "并行进化子种群" | 独立种群定期迁移；防止过早收敛 |
| Machine-checkable evaluator | "确定性预言机" | LLM 无法伪造的单元测试、模拟器或基准——此循环的前提条件 |
| Reward hacking | "优化度量而非目标" | 循环找到一种方法最大化分数而不做预期任务 |
| Seed program | "起点" | 循环从中进化的初始正确但次优程序 |
| Held-out evaluator | "LLM 从未见过的评估数据" | 在评估时生成的输入以防止记忆 |

## 延伸阅读

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — 完整论文。
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — 厂商文章及结果。
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results) — 发现的算法，包括 48-mul 4x4 matmul。
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — 前身系统。
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 将评估器约束的自主性定义为关键研究方向。
