# 奖励建模与 RLHF

> 人类写不出"好的助手回复"的奖励函数，但他们可以比较两个回复并选出更好的那个。用这些比较拟合一个奖励模型，然后用 RL 优化语言模型。Christiano 2017。InstructGPT 2022。把 GPT-3 变成 ChatGPT 的配方。2026 年它正在被 DPO 取代——但心智模型不变。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**时间：** ~45 分钟

## 问题

你用 next-token-prediction 目标训练了一个语言模型。它能写出语法正确的英文。但它也会撒谎、啰嗦、该拒绝时不拒绝。你无法通过更多预训练来修复——网络文本本身就是问题，不是解药。

你想要一个*标量奖励*来表达"对于指令 X，回复 A 比回复 B 更好"。手写这个奖励函数是不可能的。"有帮助性"不是 token 上的闭式表达式。但人类可以比较两个输出并标记偏好。这在规模上收集成本很低。

RLHF（Christiano et al. 2017; Ouyang et al. 2022）将偏好转化为奖励模型，然后通过 PPO 优化 LM。三个步骤：SFT → RM → PPO。这是 ChatGPT、Claude、Gemini 以及 2023-2025 年所有对齐 LLM 的配方。

2026 年 PPO 步骤大多被 DPO（Phase 10 · 08）取代，因为它更便宜且对齐调优效果几乎一样好。但*奖励模型*部分仍然是每个 Best-of-N 采样器、每个可验证奖励 RL 流水线、以及每个使用过程奖励模型的推理模型的基础。理解 RLHF 就理解了整个对齐技术栈。

## 概念

![三阶段 RLHF：SFT、基于成对偏好的 RM 训练、带 KL 惩罚的 PPO](../assets/rlhf.svg)

**阶段 1：监督微调（SFT）。** 从预训练基础模型开始。在人类编写的目标行为示范（遵循指令的回复、有帮助的回答等）上微调。结果：一个*偏向好行为*但动作空间仍然无界的模型 `π_SFT`。

**阶段 2：奖励模型训练。**

- 收集对提示 `x` 的回复对 `(y_+, y_-)`，由人类标注为"y_+ 优于 y_-"。
- 训练奖励模型 `R_φ(x, y)` 使其对 `y_+` 给出更高分数。
- 损失：**Bradley-Terry 成对 logistic**：

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ 是 sigmoid。奖励差值隐含了偏好的对数几率。BT 自 1952 年（Bradley-Terry）以来一直是标准，也是现代 RLHF 的主流选择。

- `R_φ` 通常从 SFT 模型初始化，顶部加一个标量头。相同的 transformer 骨干；一个线性层输出奖励。

**阶段 3：带 KL 惩罚的 PPO 对抗 RM。**

- 从 `π_SFT` 初始化可训练策略 `π_θ`。保留一个冻结的*参考* `π_ref = π_SFT`。
- 回复 `y` 结束时的奖励：

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 惩罚防止 `π_θ` 任意偏离 `π_SFT` — 它是*正则化器*，不是硬信赖域。`β` 通常 `0.01`-`0.05`。
- 用这个奖励运行 PPO（第 08 课）。优势在 token 级轨迹上计算，但 RM 只对完整回复评分。

**为什么需要 KL？** 没有它，PPO 会愉快地找到奖励黑客策略——RM 只在分布内的补全上训练过。分布外的回复可能比任何人类写的都得分更高。KL 让 `π_θ` 保持在 RM 训练时的流形附近。它是 RLHF 中最重要的旋钮。

**2026 年现状：**

- **DPO**（Rafailov 2023）：闭式代数将阶段 2+3 折叠为偏好数据上的单一监督损失。无 RM，无 PPO。在对齐 benchmark 上质量相同，计算量只是一小部分。见 Phase 10 · 08。
- **GRPO**（DeepSeek 2024-2025）：PPO 用 group-relative baseline 替代 critic，奖励来自*验证器*（代码运行通过 / 数学答案匹配）而非人类训练的 RM。推理模型的主流方法。见 Phase 9 · 12。
- **过程奖励模型（PRM）：** 对部分解（每个推理步骤）评分，用于 RLHF 和 GRPO 的推理变体。
- **Constitutional AI / RLAIF：** 用对齐的 LLM 生成偏好而非人类。扩展偏好预算。

## 动手构建

本课使用微型合成"提示"和"回复"（字符串表示）。RM 是基于 bag-of-tokens 表示的线性评分器。没有真正的 LLM——*流水线的形状*才重要，不是规模。见 `code/main.py`。

### 步骤 1：合成偏好数据

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真实 RLHF 中这由人类标注员替代。形状——`(prompt, preferred_response, rejected_response)`——完全相同。

### 步骤 2：Bradley-Terry 奖励模型

线性评分：`R(x, y) = w · bag(y)`。训练最小化 BT 成对 log-loss：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

几百次更新后，`w` 对好词 token 赋正权重，对坏词赋负权重。

### 步骤 3：在 RM 之上的类 PPO 策略

我们的玩具策略从词汇表中产生单个 token。我们在 RM 下对 token 评分，计算 `log π_θ(token | prompt)`，加上 KL-to-reference 惩罚，并应用裁剪的 PPO 代理。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### 步骤 4：监控 KL

每次更新跟踪平均 `KL(π_θ || π_ref)`。如果它爬过 `~5-10`，策略已经偏离 `π_SFT` 太远——`β` 太低或奖励黑客正在开始。这是真实 RLHF 中最重要的诊断指标。

### 步骤 5：使用 TRL 的生产配方

理解了玩具流水线后，这是真实库用户写的同一循环。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是参考实现——`RewardTrainer` 用于阶段 2，`PPOTrainer`（内置 KL-to-reference）用于阶段 3。

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

库为你做的三件事。`adap_kl_ctrl=True` 实现了自适应 β 调度：如果观测到的 KL 超过 `target_kl`，β 翻倍；如果低于一半，β 减半。参考模型按惯例冻结——你不能意外地与 `policy` 共享参数。value head 和策略在同一骨干上（`AutoModelForCausalLMWithValueHead` 附加了一个标量 MLP 头），这就是为什么 TRL 分别报告 `policy/kl` 和 `value/loss`。

## 常见陷阱

- **过度优化 / 奖励黑客。** RM 是不完美的；`π_θ` 会找到得分高但实际很差的对抗性补全。症状：奖励无限攀升而人类评估分数停滞或下降。修复：提前停止，提高 `β`，扩大 RM 训练数据。
- **长度黑客。** 在有帮助的回复上训练的 RM 往往隐式奖励长度。策略学会填充回复。补救：长度归一化奖励，或使用长度感知的 RLAIF RM。
- **RM 太小。** RM 需要至少和策略一样大。太小的 RM 无法忠实地对策略输出评分。
- **KL 调参。** β 太低 → 漂移和奖励黑客。β 太高 → 策略几乎不变。标准技巧是*自适应* β，目标是每步固定的 KL。
- **偏好数据噪声。** ~30% 的人类标签是有噪声或模糊的。通过在一致性过滤的数据上训练 RM 来校准，或在 BT 上使用温度。
- **Off-policy 问题。** PPO 数据在第一个 epoch 后就略微 off-policy。像第 08 课一样监控 clip fraction。

## 应用场景

2026 年的 RLHF 是分层的：

| 层级 | 目标 | 方法 |
|------|------|------|
| 指令遵循、有帮助性、无害性 | 对齐 | DPO（Phase 10 · 08）优于 RLHF-PPO。 |
| 推理正确性（数学、代码） | 能力 | GRPO with verifier reward（Phase 9 · 12）。 |
| 长期多步任务 | 智能体 | PPO / GRPO with process reward models over steps。 |
| 安全 / 拒绝行为 | 安全 | RLHF-PPO with separate safety RM，或 Constitutional AI。 |
| 推理时 Best-of-N | 快速对齐 | 在解码时使用 RM；无需策略训练。 |
| 奖励蒸馏 | 推理计算 | 在冻结 LM 上训练小型"reward head"。 |

RLHF 在 2022-2024 年是*唯一*方法。2026 年，生产对齐流水线是 DPO 优先，只在 RM 密集型或安全关键步骤使用 PPO。

## 交付产出

保存为 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 练习

1. **简单。** 在 `code/main.py` 中用 500 个合成偏好对训练 Bradley-Terry 奖励模型。在留出的 100 对上测量成对准确率。应超过 90%。
2. **中等。** 用 `β ∈ {0.0, 0.1, 1.0}` 运行玩具 PPO-RLHF 循环。对每个值，绘制 RM 分数 vs KL-to-reference 随更新的变化。哪些运行出现了奖励黑客？
3. **困难。** 在相同偏好数据上实现 DPO（闭式偏好似然损失），并与 RLHF-PPO 流水线在计算量和最终 RM 分数上进行比较。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| RLHF | "对齐 RL" | 三阶段 SFT + RM + PPO 流水线（Christiano 2017, Ouyang 2022）。 |
| Reward Model (RM) | "评分网络" | 通过 Bradley-Terry 拟合成对偏好的学习标量函数。 |
| Bradley-Terry | "成对 logistic 损失" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`；标准 RM 目标。 |
| KL penalty | "保持接近参考" | 奖励中的 `β · KL(π_θ \|\| π_ref)`；反奖励黑客正则化器。 |
| Reward hacking | "Goodhart 定律" | 策略利用 RM 缺陷；症状：奖励上升，人类评估持平。 |
| RLAIF | "AI 标注偏好" | 标签来自另一个 LM 而非人类的 RLHF。 |
| PRM | "过程奖励模型" | 对部分推理步骤评分；用于推理流水线。 |
| Constitutional AI | "Anthropic 的方法" | 由显式规则引导的 AI 生成偏好。 |

## 延伸阅读

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — 开启 RLHF 的论文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 背后的配方。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 更早的摘要 RLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026 年后 RLHF 时代的默认方法。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 和自我批评循环。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 论文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — 生产级 `RewardTrainer` 和 `PPOTrainer`。阅读 trainer 源码了解自适应 KL 和 value-head 细节。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — 三阶段流水线的经典图解讲解。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 库本身；`examples/` 有 Llama、Mistral 和 Qwen 的端到端 RLHF 脚本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — 奖励假说视角；思考奖励黑客的必要前置知识。
