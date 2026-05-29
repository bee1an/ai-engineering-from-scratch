# Deep Q-Networks (DQN)

> 2013 年：Mnih 用一个 Q-learning 网络直接在原始像素上训练，在七款 Atari 游戏上击败了所有经典 RL 智能体。2015 年：扩展到 49 款游戏，发表在 Nature 上，开启了深度强化学习时代。DQN 就是 Q-learning 加上三个让函数逼近稳定的技巧。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75 minutes

## 问题

表格式 Q-learning 需要为每个 (state, action) 对维护一个独立的 Q 值。国际象棋棋盘有 ~10⁴³ 种状态。一帧 Atari 画面是 210×160×3 = 100,800 个特征。表格式 RL 在几千个状态时就撑不住了，更别说几十亿。

修复方案事后看来很明显：用神经网络 `Q(s, a; θ)` 替换 Q 表。但"事后看来明显"花了几十年。朴素的函数逼近加 Q-learning 会在"致命三角"下发散——函数逼近 + 自举 + 离策略学习。Mnih et al. (2013, 2015) 找到了三个稳定训练的工程技巧：

1. **经验回放** 打破转移之间的相关性。
2. **目标网络** 冻结自举目标。
3. **奖励裁剪** 归一化梯度幅度。

DQN 在 Atari 上是第一次用单一架构、单一超参数集从原始像素解决数十个控制问题。此后所有"深度 RL"方法——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都建立在这三个技巧之上。

## 核心概念

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**目标函数。** DQN 最小化神经 Q 函数上的单步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，每步通过梯度下降更新。`θ^-` = 目标网络，定期从 `θ` 复制（大约每 10,000 步）。`D` = 过去转移的回放缓冲区。

**三个技巧，按重要性排序：**

**经验回放。** 一个容量约 `10⁶` 的环形缓冲区。每个训练步从中均匀随机采样一个 minibatch。这打破了时间相关性（连续帧几乎相同），让网络能多次从稀有的高奖励转移中学习，并去相关化连续的梯度更新。没有它，在线 TD 加神经网络在 Atari 上会发散。

**目标网络。** 在 Bellman 方程两侧使用同一个网络 `Q(·; θ)` 会让目标每次更新都在移动——"追自己的尾巴"。修复方法：保留第二个权重冻结的网络 `Q(·; θ^-)`。每 `C` 步，将 `θ → θ^-` 复制一次。这让回归目标在数千个梯度步内保持稳定。软更新 `θ^- ← τ θ + (1-τ) θ^-`（用于 DDPG、SAC）是更平滑的变体。

**奖励裁剪。** Atari 奖励幅度从 1 到 1000+ 不等。裁剪到 `{-1, 0, +1}` 防止任何单个游戏主导梯度。当奖励幅度本身有意义时这是错的；对于 Atari 只关心符号的场景没问题。

**Double DQN。** Hasselt (2016) 修复了最大化偏差：用在线网络*选择*动作，用目标网络*评估*它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

即插即用，效果一致更好。默认使用它。

**其他改进（Rainbow, 2017）：** 优先回放（更多采样高 TD-error 的转移）、对决架构（分离 `V(s)` 和 advantage 头）、噪声网络（学习的探索）、n-step 回报、分布式 Q（C51/QR-DQN）、多步自举。每个加几个百分点；增益大致可叠加。

## 动手构建

这里的代码是纯标准库、无 numpy 的——我们在一个小型连续 GridWorld 上使用手写的单隐藏层 MLP，每个训练步只需微秒。算法与大规模 Atari DQN 完全相同。

### Step 1: 回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 用 ~50,000 容量；我们的玩具环境 5,000 就够了。

### Step 2: 一个小型 Q 网络（手写 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性 → ReLU → 线性。就这么简单。

### Step 3: DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

结构和 Lesson 04 的 Q-learning 一样，有两个区别：(a) 我们通过可微的 `Q(·; θ)` 反向传播而不是索引表格，(b) 目标使用 `Q(·; θ^-)`。

### Step 4: 外层循环

每个 episode 中，对 `Q(·; θ)` 执行 ε-greedy 动作选择，将转移推入缓冲区，采样 minibatch，执行梯度步，定期同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们的小型 GridWorld（16 维 one-hot 状态）上，智能体在 ~500 个 episode 内学到近最优策略。在 Atari 上，将其扩展到 2 亿帧并加上 CNN 特征提取器。

## 常见陷阱

- **致命三角。** 函数逼近 + 离策略 + 自举可能发散。DQN 用目标网络 + 回放来缓解；不要移除任何一个。
- **探索。** ε 必须衰减，通常从 1.0 衰减到 0.01，在训练前 ~10% 完成。没有足够的早期探索，Q 网络会收敛到局部盆地。
- **过高估计。** 对噪声 Q 取 `max` 有向上偏差。生产环境中始终使用 Double DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度幅度与奖励幅度成正比。
- **回放缓冲区冷启动。** 缓冲区有几千个转移之前不要训练。早期在 ~20 个样本上的梯度会过拟合。
- **目标同步频率。** 太频繁 ≈ 没有目标网络；太不频繁 ≈ 过时的目标。Atari DQN 使用 10,000 环境步。经验法则：每 ~1/100 训练周期同步一次。
- **观测预处理。** Atari DQN 堆叠 4 帧使状态满足马尔可夫性。任何包含速度信息的环境都需要帧堆叠或循环状态。

## 实际应用

2026 年，DQN 很少是最先进的，但仍然是离策略算法的参考基准：

| Task | Method of choice | Why not DQN? |
|------|------------------|--------------|
| Discrete-action Atari-like | Rainbow DQN or Muesli | Same framework, more tricks. |
| Continuous control | SAC / TD3 (Phase 9 · 07) | DQN has no policy network. |
| On-policy / high-throughput | PPO (Phase 9 · 08) | No replay buffer; easier to scale. |
| Offline RL | CQL / IQL / Decision Transformer | Conservative Q targets, no bootstrapping blowups. |
| Large discrete action spaces (recommender) | DQN with action embedding, or IMPALA | Fine; decoration matters. |
| LLM RL | PPO / GRPO | Sequence-level, not step-level; different loss. |

这些经验仍然适用。回放和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自博弈缓冲区以及所有离线 RL 方法中。奖励裁剪以 PPO 中的 advantage 归一化形式延续。这个架构就是蓝图。

## Ship It

Save as `outputs/skill-dqn-trainer.md`:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **Easy.** 运行 `code/main.py`。绘制每 episode 回报曲线。多少个 episode 后滑动平均超过 -10？
2. **Medium.** 禁用目标网络（Bellman 目标两侧都用在线网络）。测量训练不稳定性——回报是否振荡或发散？
3. **Hard.** 添加 Double DQN：用在线网络选择 `argmax a'`，目标网络评估。在噪声奖励 GridWorld 上训练 1,000 个 episode 后，比较有无 Double DQN 时 `Q(s_0, best_a)` 与真实 `V*(s_0)` 的偏差。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| DQN | "Deep Q-learning" | Q-learning with a neural Q-function, replay buffer, and target network. |
| Experience replay | "Shuffled transitions" | Ring buffer sampled uniformly each gradient step; decorrelates data. |
| Target network | "Frozen bootstrap" | Periodic copy of Q used in the Bellman target; stabilizes training. |
| Deadly triad | "Why RL diverges" | Function approximation + bootstrapping + off-policy = no convergence guarantee. |
| Double DQN | "Fix for maximization bias" | Online net selects action, target net evaluates it. |
| Dueling DQN | "V and A heads" | Decompose Q = V + A - mean(A); same output, better gradient flow. |
| Rainbow | "All the tricks" | DDQN + PER + dueling + n-step + noisy + distributional in one. |
| PER | "Prioritized Replay" | Sample transitions proportional to TD-error magnitude. |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — the 2013 NeurIPS workshop paper that kicked off deep RL.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — the Nature paper, 49-game DQN.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — the stacked-tricks paper.
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — clear modern exposition.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — the textbook treatment of the "deadly triad" (function approximation + bootstrapping + off-policy) that DQN's target network and replay buffer are designed to tame.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — reference single-file DQN used in ablation studies; good to read alongside this lesson's from-scratch version.
