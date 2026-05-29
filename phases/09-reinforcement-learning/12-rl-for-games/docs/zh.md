# 游戏 RL — AlphaZero、MuZero 与 LLM 推理时代

> 1992 年：TD-Gammon 用纯 TD 击败人类西洋双陆棋冠军。2016 年：AlphaGo 击败李世石。2017 年：AlphaZero 从零开始统治国际象棋、将棋和围棋。2024 年：DeepSeek-R1 证明了同样的配方（用 GRPO 替代 PPO）在推理上也有效。游戏是驱动本阶段每一个突破的 benchmark。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**时间：** ~120 分钟

## 问题

游戏拥有 RL 想要的一切。干净的奖励（胜/负）。无限 episode（自我对弈重置）。完美仿真（游戏*本身就是*模拟器）。离散或小连续动作空间。多智能体结构迫使对抗鲁棒性。

而且游戏是每个重大 RL 突破的测试场。TD-Gammon（西洋双陆棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（星际争霸 II，2019）。MuZero（学习模型，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学推理，2025）——最新证明游戏 RL 技术在文本上也有效的案例。

这个综合课程通过一个统一视角审视三个里程碑架构——AlphaZero、MuZero 和 GRPO：**自我对弈 + 搜索 + 策略改进**。每个都是前一个的推广；GRPO 尤其是 AlphaZero 的配方应用于 LLM 推理，token 作为动作，数学验证作为胜利信号。

## 概念

![AlphaZero ↔ MuZero ↔ GRPO：相同循环，不同环境](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero（2017）。** Silver et al. 给定一个规则已知的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：一个塔 `f_θ(s) → (p, v)`。`p` 是合法走法上的先验。`v` 是期望游戏结果。
- 蒙特卡洛树搜索（MCTS）：每步棋展开一棵可能后续的树。用 `(p, v)` 作为先验 + bootstrap。通过 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自我对弈：智能体对智能体下棋。在第 `t` 步，MCTS 访问分布 `π_t` 成为策略训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是游戏结果（+1 / 0 / -1）。

零人类知识。零手工启发式。一个配方在各自数千万局自我对弈后掌握了国际象棋、将棋和围棋。

**MuZero（2019）。** Schrittwieser et al. 去除了规则已知的要求。

- 不用固定环境，而是学习一个*隐空间动力学模型* `(h, g, f)`：
  - `h(s)`：将观测编码为隐状态。
  - `g(s_latent, a)`：预测下一个隐状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在*学习的隐空间*中运行。相同的搜索，相同的训练循环。
- 在围棋、国际象棋、将棋*和* Atari 上都有效——一个算法，无需规则知识。

**Stochastic MuZero（2022）。** 添加随机动力学和机会节点；扩展到西洋双陆棋类游戏。

**Muesli, Gumbel MuZero（2022-2024）。** 样本效率和确定性搜索的改进。

**GRPO（2024-2025）。** DeepSeek-R1 配方。相同的 AlphaZero 形状循环，应用于语言模型推理：

- "游戏"：回答数学/编程/推理问题。"胜利" = 验证器（测试用例通过，数值答案匹配）返回 1。
- 策略：LLM。动作：token。状态：prompt + 已生成的回复。
- 无 critic（PPO 风格的 V_φ）。相反，对每个 prompt，从策略中采样 `G` 个补全。计算每个的奖励。使用 **group-relative advantage** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- KL 惩罚到参考策略以防止漂移（类似 RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无奖励模型，无 critic，无 MCTS。Group-relative baseline 替代了这三者。在推理 benchmark 上匹配或超过 PPO-RLHF 质量，计算量只是一小部分。

**完整的 R1 配方。** DeepSeek-R1（DeepSeek 2025）是一篇论文中的两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基础模型开始。无 SFT。直接应用 GRPO，两个奖励组件：*准确性奖励*（基于规则——最终答案是否解析为正确数字/代码是否通过单元测试）和*格式奖励*（补全是否将思维链包裹在 `<think>…</think>` 标签中）。经过数千步，平均回复长度从 ~100 增长到 ~10,000 token，数学 benchmark 分数攀升到接近 o1-preview 水平。模型从零学会推理。缺点：其思维链通常不可读，混合语言，缺乏风格打磨。
- **R1。** 用四阶段流水线修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集几千个格式清晰的长 CoT 示范。在其上监督微调基础模型。这给出一个可读的起点。
  2. **面向推理的 GRPO。** 应用 GRPO，使用准确性+格式奖励加上*语言一致性*奖励以防止语码转换。
  3. **拒绝采样 + 第二轮 SFT。** 从 RL checkpoint 采样 ~600K 推理轨迹，只保留最终答案正确且 CoT 可读的，与 ~200K 非推理 SFT 样本（写作、QA、自我认知）合并。再次微调基础模型。
  4. **全谱 GRPO。** 再做一轮 RL，覆盖推理（基于规则的奖励）和通用对齐（有帮助性/无害性基于偏好的奖励）。

结果在 AIME 和 MATH-500 上匹配 o1，开放权重，且小到可以蒸馏。同一论文还发布了六个蒸馏的 dense 模型（Qwen-1.5B 到 Llama-70B），通过在 R1 的推理轨迹上 SFT——学生端无 RL。强 RL 教师的蒸馏在学生规模上始终优于从零开始的 RL。

**为什么推理用 GRPO 而非 PPO。** DeepSeekMath 论文（2024 年 2 月）给出三个原因：(1) 无需训练 value network，内存减半；(2) group baseline 自然处理推理任务产生的稀疏 end-of-trajectory 奖励；(3) per-prompt 归一化使不同难度问题的优势可比较，而 PPO 的单一 critic 做不到。

**无搜索 vs 基于搜索。** 游戏已经分化：

- *完全信息长期博弈*（围棋、国际象棋）：仍然基于搜索。AlphaZero / MuZero 主导。
- *LLM 推理*：生产中还没有 MCTS；GRPO 在完整 rollout 上，best-of-N 用于推理计算。过程奖励模型（PRM）暗示步级搜索正在被加回来。

## 动手构建

`code/main.py` 中的代码实现了**微型 GRPO**——一个带多组样本的 bandit。算法与 LLM 上的相同；只是策略和环境更简单。它教授*损失*和 *group-relative advantage*，这是 2025 年的创新。

### 步骤 1：微型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实 GRPO 中验证器运行单元测试或检查数学等式。

### 步骤 2：策略：每个 prompt 上 K 个答案 token 的 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于 LLM 条件于 prompt 时的最后一层输出。

### 步骤 3：组采样和 group-relative advantage

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Group-relative advantage 是 2024 年 DeepSeek 的技巧。不需要 critic。"基线"是组均值，归一化使用组标准差。

### 步骤 4：与 REINFORCE baseline（无 value）比较

相同设置，相同计算量，普通 REINFORCE。GRPO 收敛更快更稳定。

### 步骤 5：观察 entropy 和 KL

与 RLHF 相同的诊断：到参考的平均 KL、策略 entropy、奖励随时间变化。一旦这些稳定，训练就完成了。

## 常见陷阱

- **通过验证器博弈的奖励黑客。** GRPO 继承了 RLHF 的风险：如果验证器是错的或可利用的，LLM 会找到利用方式。鲁棒的验证器（多个测试用例、形式化证明）很重要。
- **组大小太小。** 组基线的方差按 `1/√G` 缩放。低于 `G = 4`，优势信号很嘈杂；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的 LLM 补全有不同的 log-probability。按 token 数归一化，或使用序列级 log-prob，或截断到最大长度。
- **纯自我对弈循环。** AlphaZero 风格训练在一般和博弈上可能陷入优势循环。通过多样化对手池缓解（联赛训练，第 10 课）。
- **搜索-策略不匹配。** AlphaZero 训练策略模仿搜索输出。如果策略网络太小无法表示搜索的分布，训练会停滞。
- **计算下限。** MuZero / AlphaZero 需要大量计算。单次消融实验通常需要数百 GPU 小时。存在微型演示（如 Connect Four 上的 AlphaZero）用于学习。
- **验证器覆盖率。** 对有 bug 的解通过的单元测试会强化 bug。设计能捕获边界情况的验证器。

## 应用场景

2026 年游戏 RL 格局，按领域：

| 领域 | 主流方法 |
|------|----------|
| 双人零和棋盘游戏（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完全信息卡牌游戏（扑克） | CFR + deep learning (DeepStack, Libratus, Pluribus) |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略（Dota、星际争霸） | PPO + self-play + league (OpenAI Five, AlphaStar) |
| LLM 数学/代码推理 | GRPO (DeepSeek-R1, Qwen-RL, open replications) |
| LLM 对齐 | DPO / RLHF-PPO（非 GRPO；验证器是偏好而非可验证的） |
| 机器人 | PPO + DR（非游戏 RL，但使用相同的策略梯度工具） |
| 组合问题 | AlphaZero variants (AlphaTensor, AlphaDev) |

*配方*——自我对弈、搜索增强改进、策略蒸馏——跨越文本、像素和物理控制。GRPO 是最年轻的实例；更多正在到来。

## 交付产出

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 GRPO bandit。在 2 个 prompt × 4 个答案 token 上训练。用 `G=8` 在 < 1,000 次更新内收敛。
2. **中等。** 接入 PPO（裁剪版）和 vanilla REINFORCE。在相同 bandit 上比较与 GRPO 的样本效率和奖励方差。
3. **困难。** 扩展到长度为 2 的"推理链"：智能体发出两个 token，验证器奖励这对。测量 GRPO 如何处理两步序列的信用分配。（提示：按*完整序列*计算 group advantage，传播到两个 token 位置。）

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| MCTS | "带学习网络的树搜索" | 蒙特卡洛树搜索；UCB1/PUCT 选择配合学习的 `(p, v)` 先验。 |
| AlphaZero | "自我对弈 + MCTS" | 策略-价值网络训练匹配 MCTS 访问次数和游戏结果。 |
| MuZero | "学习模型的 AlphaZero" | 相同循环但在通过学习动力学的隐空间中。 |
| GRPO | "无 critic 的 PPO" | Group Relative Policy Optimization；REINFORCE with group-mean baseline + KL。 |
| PUCT | "AlphaZero 的 UCB" | `Q + c · p · √N / (1 + N_a)` — 平衡价值估计与先验。 |
| Self-play | "智能体对战过去的自己" | 零和博弈的标准方法；对称训练信号。 |
| League play | "基于种群的自我对弈" | 过去 + 当前 + exploiter 作为对手采样。 |
| Verifier reward | "可验证 RL" | 奖励来自确定性检查器（测试通过，答案匹配）。 |
| Process reward | "PRM" | 对每个推理步骤评分，而非仅最终答案。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 引入 GRPO 和 group-relative baseline 的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整的四阶段 R1 配方加 R1-Zero 消融。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 大规模 CFR + deep-learning。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 开启一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 使用自定义奖励函数应用 GRPO 的生产参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 多规模的 R1 配方开源复现。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书对自我对弈、搜索和"设计奖励"的框架，R1 在 LLM 规模上实例化了它。
