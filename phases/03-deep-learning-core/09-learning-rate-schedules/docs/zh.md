# Learning Rate Schedule 与 Warmup

> Learning rate 是最重要的超参数。不是架构，不是数据集大小，不是激活函数。是 learning rate。如果你只调一个东西，就调它。

**类型：** 构建
**语言：** Python
**前置课程：** Lesson 03.06（优化器）、Lesson 03.08（Weight Initialization）
**时间：** 约 90 分钟

## 学习目标

- 从零实现 constant、step decay、cosine annealing、warmup + cosine 和 1cycle learning rate schedule
- 演示 learning rate 选择的三种失败模式：发散（太高）、停滞（太低）和振荡（无衰减）
- 解释为什么 warmup 对基于 Adam 的优化器是必要的，以及它如何稳定早期训练
- 在相同任务上比较所有五种 schedule 的收敛速度，并为给定的训练预算选择合适的 schedule

## 问题

把 learning rate 设为 0.1。训练发散——loss 在 3 步内跳到无穷大。设为 0.0001。训练爬行——100 个 epoch 后，模型几乎没有从随机状态移动。设为 0.01。训练在 50 个 epoch 内有效，然后 loss 在一个永远无法到达的最小值附近振荡，因为步长太大了。

最优 learning rate 不是常数。它在训练过程中变化。早期，你需要大步前进以快速覆盖地形。训练后期，你需要小步来稳定地落入一个尖锐的最小值。90% 准确率的模型和 95% 准确率的模型之间的差异，往往只是 schedule 的不同。

过去三年发表的每个主要模型都使用了 learning rate schedule。Llama 3 使用 peak lr=3e-4，2000 步 warmup，cosine 衰减到 3e-5。GPT-3 使用 lr=6e-4，在 3.75 亿 token 上 warmup。这些不是随意的选择，而是花费数百万美元进行大规模超参数搜索的结果。

你需要理解 schedule，因为默认值不会适用于你的问题。当你微调预训练模型时，正确的 schedule 与从头训练不同。当你增加 batch size 时，warmup 周期需要改变。当训练在第 10,000 步崩溃时，你需要知道这是 schedule 问题还是其他问题。

## 概念

### Constant Learning Rate

最简单的方法。选一个数字，每步都用它。

```
lr(t) = lr_0
```

很少是最优的。它要么对训练末期太高（在最小值附近振荡），要么对训练初期太低（在小步上浪费算力）。对小模型和调试来说够用。对任何训练超过一小时的任务来说是糟糕的选择。

### Step Decay

ResNet 时代的老派方法。在固定 epoch 处将 learning rate 除以一个因子（通常是 10 倍）。

```
lr(t) = lr_0 * gamma^(floor(epoch / step_size))
```

其中 gamma = 0.1，step_size = 30 意味着：lr 每 30 个 epoch 降低 10 倍。ResNet-50 就是这样用的——lr=0.1，在 epoch 30、60 和 90 时降低 10 倍。

问题是：最优的衰减点取决于数据集和架构。换一个问题就需要重新调整何时降低。过渡是突然的——当 rate 突然变化时 loss 可能会跳升。

### Cosine Annealing

从最大 learning rate 平滑衰减到最小值，遵循余弦曲线：

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
```

其中 t 是当前步数，T 是总步数。

在 t=0 时，余弦项为 1，所以 lr = lr_max。在 t=T 时，余弦项为 -1，所以 lr = lr_min。衰减开始时平缓，中间加速，接近结束时又变平缓。

这是大多数现代训练的默认选择。除了 lr_max 和 lr_min 之外没有需要调的超参数。余弦形状符合经验观察：大部分学习发生在训练中期——你需要在那个关键时期保持合理的步长。

### Warmup：为什么要从小开始

Adam 和其他自适应优化器维护梯度均值和方差的运行估计。在第 0 步，这些估计被初始化为零。最初几步的梯度更新基于垃圾统计量。如果此时 learning rate 很大，模型会迈出巨大的、方向错误的步伐。

Warmup 解决了这个问题。从一个很小的 learning rate 开始（通常是 lr_max / warmup_steps 甚至零），在前 N 步内线性增加到 lr_max。当你达到完整的 learning rate 时，Adam 的统计量已经稳定了。

```
lr(t) = lr_max * (t / warmup_steps)     for t < warmup_steps
```

典型的 warmup：总训练步数的 1-5%。Llama 3 训练了约 1.8 万亿 token，warmup 了 2000 步。GPT-3 在 3.75 亿 token 上 warmup。

### Linear Warmup + Cosine Decay

现代默认方案。线性上升，然后余弦衰减：

```
if t < warmup_steps:
    lr(t) = lr_max * (t / warmup_steps)
else:
    progress = (t - warmup_steps) / (total_steps - warmup_steps)
    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
```

这是 Llama、GPT、PaLM 和大多数现代 transformer 使用的方案。Warmup 防止早期不稳定。Cosine decay 让模型稳定地落入一个好的最小值。

### 1cycle Policy

Leslie Smith 的发现（2018）：在训练前半段将 learning rate 从低值升到高值，然后在后半段降回来。反直觉——为什么要在训练中途*增加* learning rate？

理论是：高 learning rate 通过给优化轨迹添加噪声来起到正则化作用。模型在上升阶段探索更多的 loss landscape，找到更好的盆地。下降阶段则在找到的最佳盆地内精细调整。

```
Phase 1 (0 to T/2):    lr ramps from lr_max/25 to lr_max
Phase 2 (T/2 to T):    lr ramps from lr_max to lr_max/10000
```

在固定计算预算下，1cycle 通常比 cosine annealing 训练更快。代价是：你必须提前知道总步数。

### Schedule 形状

```mermaid
graph LR
    subgraph "Constant"
        C1["lr"] --- C2["lr"] --- C3["lr"]
    end

    subgraph "Step Decay"
        S1["0.1"] --- S2["0.1"] --- S3["0.01"] --- S4["0.001"]
    end

    subgraph "Cosine Annealing"
        CS1["lr_max"] --> CS2["gradual"] --> CS3["steep"] --> CS4["lr_min"]
    end

    subgraph "Warmup + Cosine"
        WC1["0"] --> WC2["lr_max"] --> WC3["cosine"] --> WC4["lr_min"]
    end
```

### 决策流程图

```mermaid
flowchart TD
    Start["Choosing a LR schedule"] --> Know{"Know total<br/>training steps?"}

    Know -->|"Yes"| Budget{"Compute budget?"}
    Know -->|"No"| Constant["Use constant LR<br/>with manual decay"]

    Budget -->|"Large (days/weeks)"| WarmCos["Warmup + Cosine Decay<br/>(Llama/GPT default)"]
    Budget -->|"Small (hours)"| OneCycle["1cycle Policy<br/>(fastest convergence)"]
    Budget -->|"Moderate"| Cosine["Cosine Annealing<br/>(safe default)"]

    WarmCos --> Warmup["Warmup = 1-5% of steps"]
    OneCycle --> FindLR["Find lr_max with LR range test"]
    Cosine --> MinLR["Set lr_min = lr_max / 10"]
```

### 已发表模型的实际数值

```mermaid
graph TD
    subgraph "Published LR Configs"
        L3["Llama 3 (405B)<br/>Peak: 3e-4<br/>Warmup: 2000 steps<br/>Schedule: Cosine to 3e-5"]
        G3["GPT-3 (175B)<br/>Peak: 6e-4<br/>Warmup: 375M tokens<br/>Schedule: Cosine to 0"]
        R50["ResNet-50<br/>Peak: 0.1<br/>Warmup: none<br/>Schedule: Step decay x0.1 at 30,60,90"]
        B["BERT (340M)<br/>Peak: 1e-4<br/>Warmup: 10K steps<br/>Schedule: Linear decay"]
    end
```

## 动手构建

### 第 1 步：Schedule 函数

每个函数接收当前步数，返回该步的 learning rate。

```python
import math


def constant_schedule(step, lr=0.01, **kwargs):
    return lr


def step_decay_schedule(step, lr=0.1, step_size=100, gamma=0.1, **kwargs):
    return lr * (gamma ** (step // step_size))


def cosine_schedule(step, lr=0.01, total_steps=1000, lr_min=1e-5, **kwargs):
    if step >= total_steps:
        return lr_min
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * step / total_steps))


def warmup_cosine_schedule(step, lr=0.01, total_steps=1000, warmup_steps=100, lr_min=1e-5, **kwargs):
    if total_steps <= warmup_steps:
        return lr * (step / max(warmup_steps, 1))
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))


def one_cycle_schedule(step, lr=0.01, total_steps=1000, **kwargs):
    mid = max(total_steps // 2, 1)
    if step < mid:
        return (lr / 25) + (lr - lr / 25) * step / mid
    else:
        progress = (step - mid) / max(total_steps - mid, 1)
        return lr * (1 - progress) + (lr / 10000) * progress
```

### 第 2 步：可视化所有 Schedule

打印文本图表，展示每种 schedule 在训练过程中的变化。

```python
def visualize_schedule(name, schedule_fn, total_steps=500, **kwargs):
    steps = list(range(0, total_steps, total_steps // 20))
    if total_steps - 1 not in steps:
        steps.append(total_steps - 1)

    lrs = [schedule_fn(s, total_steps=total_steps, **kwargs) for s in steps]
    max_lr = max(lrs) if max(lrs) > 0 else 1.0

    print(f"\n{name}:")
    for s, lr_val in zip(steps, lrs):
        bar_len = int(lr_val / max_lr * 40)
        bar = "#" * bar_len
        print(f"  Step {s:4d}: lr={lr_val:.6f} {bar}")
```

### 第 3 步：训练网络

一个简单的两层网络在圆形数据集上训练，和之前的课程一样，但现在我们改变 schedule。

```python
import random


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def relu(x):
    return max(0.0, x)


def relu_deriv(x):
    return 1.0 if x > 0 else 0.0


def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


def train_with_schedule(schedule_fn, schedule_name, data, epochs=300, base_lr=0.05, **kwargs):
    random.seed(0)
    hidden_size = 8
    total_steps = epochs * len(data)

    std = math.sqrt(2.0 / 2)
    w1 = [[random.gauss(0, std) for _ in range(2)] for _ in range(hidden_size)]
    b1 = [0.0] * hidden_size
    w2 = [random.gauss(0, std) for _ in range(hidden_size)]
    b2 = 0.0

    step = 0
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for x, target in data:
            lr = schedule_fn(step, lr=base_lr, total_steps=total_steps, **kwargs)

            z1 = []
            h = []
            for i in range(hidden_size):
                z = w1[i][0] * x[0] + w1[i][1] * x[1] + b1[i]
                z1.append(z)
                h.append(relu(z))

            z2 = sum(w2[i] * h[i] for i in range(hidden_size)) + b2
            out = sigmoid(z2)

            error = out - target
            d_out = error * out * (1 - out)

            for i in range(hidden_size):
                d_h = d_out * w2[i] * relu_deriv(z1[i])
                w2[i] -= lr * d_out * h[i]
                for j in range(2):
                    w1[i][j] -= lr * d_h * x[j]
                b1[i] -= lr * d_h
            b2 -= lr * d_out

            total_loss += (out - target) ** 2
            if (out >= 0.5) == (target >= 0.5):
                correct += 1
            step += 1

        avg_loss = total_loss / len(data)
        accuracy = correct / len(data) * 100
        epoch_losses.append(avg_loss)

    return epoch_losses
```

### 第 4 步：比较所有 Schedule

用每种 schedule 训练相同的网络，比较最终 loss 和收敛行为。

```python
def compare_schedules(data):
    configs = [
        ("Constant", constant_schedule, {}),
        ("Step Decay", step_decay_schedule, {"step_size": 15000, "gamma": 0.1}),
        ("Cosine", cosine_schedule, {"lr_min": 1e-5}),
        ("Warmup+Cosine", warmup_cosine_schedule, {"warmup_steps": 3000, "lr_min": 1e-5}),
        ("1cycle", one_cycle_schedule, {}),
    ]

    print(f"\n{'Schedule':<20} {'Start Loss':>12} {'Mid Loss':>12} {'End Loss':>12} {'Best Loss':>12}")
    print("-" * 70)

    for name, schedule_fn, extra_kwargs in configs:
        losses = train_with_schedule(schedule_fn, name, data, epochs=300, base_lr=0.05, **extra_kwargs)
        mid_idx = len(losses) // 2
        best = min(losses)
        print(f"{name:<20} {losses[0]:>12.6f} {losses[mid_idx]:>12.6f} {losses[-1]:>12.6f} {best:>12.6f}")
```

### 第 5 步：LR 太高 vs 太低

演示三种失败模式：太高（发散）、太低（爬行）、刚好。

```python
def lr_sensitivity(data):
    learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]

    print("\nLR Sensitivity (constant schedule, 100 epochs):")
    print(f"  {'LR':>10} {'Start Loss':>12} {'End Loss':>12} {'Status':>15}")
    print("  " + "-" * 52)

    for lr in learning_rates:
        losses = train_with_schedule(constant_schedule, f"lr={lr}", data, epochs=100, base_lr=lr)
        start = losses[0]
        end = losses[-1]

        if end > start or math.isnan(end) or end > 1.0:
            status = "DIVERGED"
        elif end > start * 0.9:
            status = "BARELY MOVED"
        elif end < 0.15:
            status = "CONVERGED"
        else:
            status = "LEARNING"

        end_str = f"{end:.6f}" if not math.isnan(end) else "NaN"
        print(f"  {lr:>10.4f} {start:>12.6f} {end_str:>12} {status:>15}")
```

## 实际使用

PyTorch 在 `torch.optim.lr_scheduler` 中提供了 scheduler：

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = optim.Adam(model.parameters(), lr=3e-4)

scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)

for step in range(1000):
    loss = train_step(model, optimizer)
    scheduler.step()
```

对于 warmup + cosine，使用 lambda scheduler 或 HuggingFace 的 `get_cosine_schedule_with_warmup`：

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=100000,
)
```

HuggingFace 的这个函数是大多数 Llama 和 GPT 微调脚本使用的。如果不确定，就用 warmup + cosine，warmup = 总步数的 3-5%。它几乎适用于所有情况。

## 交付产出

本课产出：
- `outputs/prompt-lr-schedule-advisor.md` -- 一个为你的训练设置推荐正确 learning rate schedule 和超参数的 prompt

## 练习

1. 实现指数衰减：lr(t) = lr_0 * gamma^t，其中 gamma = 0.999。在圆形数据集上与 cosine annealing 对比。

2. 实现 learning rate range test（Leslie Smith）：训练几百步，同时将 LR 从 1e-7 指数增加到 1。绘制 loss vs LR。最优 max LR 是 loss 开始上升之前的值。

3. 用 warmup + cosine 训练，但改变 warmup 长度：总步数的 0%、1%、5%、10%、20%。找到训练最稳定的最佳点。

4. 实现带热重启的 cosine annealing（SGDR）：每 T 步将 learning rate 重置为 lr_max 并再次衰减。在更长的训练中与标准 cosine 对比。

5. 构建一个"schedule 外科医生"，监控训练 loss，当 loss 稳定时自动从 warmup 切换到 cosine，当 loss 长时间停滞时降低 lr。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Learning rate | "模型学习的速度" | 乘以梯度来确定参数更新大小的标量 |
| Schedule | "随时间改变 LR" | 将训练步数映射到 learning rate 的函数，旨在优化收敛 |
| Warmup | "从小 LR 开始" | 在前 N 步内将 LR 从接近零线性增加到目标值，以稳定优化器统计量 |
| Cosine annealing | "平滑 LR 衰减" | 沿余弦曲线将 LR 从 lr_max 降低到 lr_min |
| Step decay | "在里程碑处降低 LR" | 在固定 epoch 间隔将 LR 乘以一个因子（通常是 0.1） |
| 1cycle policy | "先升后降" | Leslie Smith 的方法，在单个周期内先升后降 LR 以加速收敛 |
| LR range test | "找到最佳 learning rate" | 短暂训练同时增加 LR，找到 loss 开始发散前的值 |
| Cosine with warm restarts | "重置并重复" | 周期性地将 LR 重置为 lr_max 并再次衰减（SGDR） |
| Eta min | "LR 的下限" | Schedule 衰减到的最小 learning rate |
| Peak learning rate | "最大 LR" | 训练中达到的最高 LR，通常在 warmup 之后 |

## 延伸阅读

- Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (2017) -- 引入了 cosine annealing 和热重启
- Smith, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates" (2018) -- 1cycle policy 论文
- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023) -- 记录了大规模使用的 warmup + cosine schedule
- Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017) -- 大 batch 训练的线性缩放规则和 warmup
