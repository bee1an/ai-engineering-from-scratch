# Weight Initialization 与训练稳定性

> 初始化错了，训练永远不会开始。初始化对了，50 层网络训练起来和 3 层一样顺畅。

**类型：** 构建
**语言：** Python
**前置课程：** Lesson 03.04（激活函数）、Lesson 03.07（正则化）
**时间：** 约 90 分钟

## 学习目标

- 实现 zero、random、Xavier/Glorot 和 Kaiming/He 初始化策略，并测量它们对 50 层网络中激活值幅度的影响
- 推导 Xavier init 为何使用 Var(w) = 2/(fan_in + fan_out)，Kaiming 为何使用 Var(w) = 2/fan_in
- 演示零初始化的对称性问题，解释为什么仅靠随机缩放是不够的
- 将正确的初始化策略与激活函数匹配：Xavier 用于 sigmoid/tanh，Kaiming 用于 ReLU/GELU

## 问题

把所有权重初始化为零。什么都学不到。每个神经元计算相同的函数，接收相同的梯度，更新也完全一样。训练 10,000 个 epoch 后，你的 512 个隐藏神经元仍然是同一个神经元的 512 份拷贝。你花了 512 个参数的代价，只得到了 1 个。

初始化太大。激活值在网络中爆炸。到第 10 层，数值达到 1e15。到第 20 层，溢出到无穷大。梯度在反向传播中走同样的轨迹。

从标准正态分布随机初始化。3 层网络能用。到 50 层时，信号要么坍缩为零，要么爆炸到无穷大——取决于随机缩放是略微偏小还是略微偏大。"能用"和"崩溃"之间的边界极其狭窄。

权重初始化是深度学习中最被低估的决策。架构能发论文，优化器能写博客，初始化只能得到一个脚注。但如果搞错了，其他一切都不重要——你的网络在训练开始之前就已经死了。

## 概念

### 对称性问题

一层中的每个神经元结构相同：输入乘以权重，加偏置，应用激活函数。如果所有权重从相同的值开始（零是极端情况），每个神经元计算相同的输出。反向传播时，每个神经元接收相同的梯度。更新时，每个神经元变化量相同。

你被卡住了。网络有数百个参数，但它们全部同步移动。这叫做对称性，随机初始化是打破它的暴力方法。每个神经元从权重空间的不同点出发，因此各自学习不同的特征。

但"随机"还不够。随机性的*尺度*决定了网络能否训练。

### 方差在层间的传播

考虑一个有 fan_in 个输入的单层：

```
z = w1*x1 + w2*x2 + ... + w_n*x_n
```

如果每个权重 wi 来自方差为 Var(w) 的分布，每个输入 xi 的方差为 Var(x)，则输出方差为：

```
Var(z) = fan_in * Var(w) * Var(x)
```

如果 Var(w) = 1 且 fan_in = 512，输出方差是输入方差的 512 倍。经过 10 层：512^10 = 1.2e27。信号爆炸了。

如果 Var(w) = 0.001，输出方差每层缩小 0.001 * 512 = 0.512 倍。经过 10 层：0.512^10 = 0.00013。信号消失了。

目标：选择 Var(w) 使得 Var(z) = Var(x)。信号幅度在各层间保持恒定。

### Xavier/Glorot Initialization

Glorot 和 Bengio（2010）为 sigmoid 和 tanh 激活函数推导出了解决方案。为了在前向和反向传播中都保持方差恒定：

```
Var(w) = 2 / (fan_in + fan_out)
```

实际操作中，权重从以下分布采样：

```
w ~ Uniform(-limit, limit)  where limit = sqrt(6 / (fan_in + fan_out))
```

或：

```
w ~ Normal(0, sqrt(2 / (fan_in + fan_out)))
```

这之所以有效，是因为 sigmoid 和 tanh 在零附近近似线性，而正确初始化的激活值正好落在这个区域。方差能在数十层中保持稳定。

### Kaiming/He Initialization

ReLU 会杀死一半的输出（所有负值变为零）。有效的 fan_in 减半，因为平均有一半的输入被置零。Xavier init 没有考虑这一点——它低估了所需的方差。

He et al.（2015）调整了公式：

```
Var(w) = 2 / fan_in
```

权重从以下分布采样：

```
w ~ Normal(0, sqrt(2 / fan_in))
```

因子 2 补偿了 ReLU 将一半激活值置零的效果。没有它，信号每层缩小约 0.5 倍。50 层后：0.5^50 = 8.8e-16。Kaiming init 防止了这种情况。

### Transformer 的初始化

GPT-2 引入了一种不同的模式。残差连接将每个子层的输出加到其输入上：

```
x = x + sublayer(x)
```

每次相加都会增加方差。有 N 个残差层时，方差与 N 成正比增长。GPT-2 将残差层的权重缩放 1/sqrt(2N)，其中 N 是层数。这使累积的信号幅度保持稳定。

Llama 3（405B 参数，126 层）使用类似的方案。没有这种缩放，残差流会在 126 层 attention 和 feedforward 块中无限增长。

```mermaid
flowchart TD
    subgraph "Zero Init"
        Z1["Layer 1<br/>All weights = 0"] --> Z2["Layer 2<br/>All neurons identical"]
        Z2 --> Z3["Layer 3<br/>Still identical"]
        Z3 --> ZR["Result: 1 effective neuron<br/>regardless of width"]
    end

    subgraph "Xavier Init"
        X1["Layer 1<br/>Var = 2/(fan_in+fan_out)"] --> X2["Layer 2<br/>Signal stable"]
        X2 --> X3["Layer 50<br/>Signal stable"]
        X3 --> XR["Result: Trains with<br/>sigmoid/tanh"]
    end

    subgraph "Kaiming Init"
        K1["Layer 1<br/>Var = 2/fan_in"] --> K2["Layer 2<br/>Signal stable"]
        K2 --> K3["Layer 50<br/>Signal stable"]
        K3 --> KR["Result: Trains with<br/>ReLU/GELU"]
    end
```

### 50 层中的激活值幅度

```mermaid
graph LR
    subgraph "Mean Activation Magnitude"
        direction LR
        L1["Layer 1"] --> L10["Layer 10"] --> L25["Layer 25"] --> L50["Layer 50"]
    end

    subgraph "Results"
        R1["Random N(0,1): EXPLODES by layer 5"]
        R2["Random N(0,0.01): Vanishes by layer 10"]
        R3["Xavier + Sigmoid: ~1.0 at layer 50"]
        R4["Kaiming + ReLU: ~1.0 at layer 50"]
    end
```

### 选择正确的初始化

```mermaid
flowchart TD
    Start["What activation?"] --> Act{"Activation type?"}

    Act -->|"Sigmoid / Tanh"| Xavier["Xavier/Glorot<br/>Var = 2/(fan_in + fan_out)"]
    Act -->|"ReLU / Leaky ReLU"| Kaiming["Kaiming/He<br/>Var = 2/fan_in"]
    Act -->|"GELU / Swish"| Kaiming2["Kaiming/He<br/>(same as ReLU)"]
    Act -->|"Transformer residual"| GPT["Scale by 1/sqrt(2N)<br/>N = num layers"]

    Xavier --> Check["Verify: activation magnitudes<br/>stay between 0.5 and 2.0<br/>through all layers"]
    Kaiming --> Check
    Kaiming2 --> Check
    GPT --> Check
```

## 动手构建

### 第 1 步：初始化策略

四种初始化权重矩阵的方法。每种返回一个列表的列表（2D 矩阵），有 fan_in 列和 fan_out 行。

```python
import math
import random


def zero_init(fan_in, fan_out):
    return [[0.0 for _ in range(fan_in)] for _ in range(fan_out)]


def random_init(fan_in, fan_out, scale=1.0):
    return [[random.gauss(0, scale) for _ in range(fan_in)] for _ in range(fan_out)]


def xavier_init(fan_in, fan_out):
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


def kaiming_init(fan_in, fan_out):
    std = math.sqrt(2.0 / fan_in)
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
```

### 第 2 步：激活函数

我们需要 sigmoid、tanh 和 ReLU 来测试每种初始化策略与其对应激活函数的配合。

```python
def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def tanh_act(x):
    return math.tanh(x)


def relu(x):
    return max(0.0, x)
```

### 第 3 步：通过 50 层的前向传播

将随机数据通过深层网络，测量每层的平均激活值幅度。

```python
def forward_deep(init_fn, activation_fn, n_layers=50, width=64, n_samples=100):
    random.seed(42)
    layer_magnitudes = []

    inputs = [[random.gauss(0, 1) for _ in range(width)] for _ in range(n_samples)]

    for layer_idx in range(n_layers):
        weights = init_fn(width, width)
        biases = [0.0] * width

        new_inputs = []
        for sample in inputs:
            output = []
            for neuron_idx in range(width):
                z = sum(weights[neuron_idx][j] * sample[j] for j in range(width)) + biases[neuron_idx]
                output.append(activation_fn(z))
            new_inputs.append(output)
        inputs = new_inputs

        magnitudes = []
        for sample in inputs:
            magnitudes.append(sum(abs(v) for v in sample) / width)
        mean_mag = sum(magnitudes) / len(magnitudes)
        layer_magnitudes.append(mean_mag)

    return layer_magnitudes
```

### 第 4 步：实验

运行所有组合：zero init、random N(0,1)、random N(0,0.01)、Xavier + sigmoid、Xavier + tanh、Kaiming + ReLU。打印关键层的幅度。

```python
def run_experiment():
    configs = [
        ("Zero init + Sigmoid", lambda fi, fo: zero_init(fi, fo), sigmoid),
        ("Random N(0,1) + ReLU", lambda fi, fo: random_init(fi, fo, 1.0), relu),
        ("Random N(0,0.01) + ReLU", lambda fi, fo: random_init(fi, fo, 0.01), relu),
        ("Xavier + Sigmoid", xavier_init, sigmoid),
        ("Xavier + Tanh", xavier_init, tanh_act),
        ("Kaiming + ReLU", kaiming_init, relu),
    ]

    print(f"{'Strategy':<30} {'L1':>10} {'L5':>10} {'L10':>10} {'L25':>10} {'L50':>10}")
    print("-" * 80)

    for name, init_fn, act_fn in configs:
        mags = forward_deep(init_fn, act_fn)
        row = f"{name:<30}"
        for idx in [0, 4, 9, 24, 49]:
            val = mags[idx]
            if val > 1e6:
                row += f" {'EXPLODED':>10}"
            elif val < 1e-6:
                row += f" {'VANISHED':>10}"
            else:
                row += f" {val:>10.4f}"
        print(row)
```

### 第 5 步：对称性演示

展示零初始化产生完全相同的神经元。

```python
def symmetry_demo():
    random.seed(42)
    weights = zero_init(2, 4)
    biases = [0.0] * 4

    inputs = [0.5, -0.3]
    outputs = []
    for neuron_idx in range(4):
        z = sum(weights[neuron_idx][j] * inputs[j] for j in range(2)) + biases[neuron_idx]
        outputs.append(sigmoid(z))

    print("\nSymmetry Demo (4 neurons, zero init):")
    for i, out in enumerate(outputs):
        print(f"  Neuron {i}: output = {out:.6f}")
    all_same = all(abs(outputs[i] - outputs[0]) < 1e-10 for i in range(len(outputs)))
    print(f"  All identical: {all_same}")
    print(f"  Effective parameters: 1 (not {len(weights) * len(weights[0])})")
```

### 第 6 步：逐层幅度报告

打印 50 层中激活值幅度的文本柱状图。

```python
def magnitude_report(name, magnitudes):
    print(f"\n{name}:")
    for i, mag in enumerate(magnitudes):
        if i % 5 == 0 or i == len(magnitudes) - 1:
            if mag > 1e6:
                bar = "X" * 50 + " EXPLODED"
            elif mag < 1e-6:
                bar = "." + " VANISHED"
            else:
                bar_len = min(50, max(1, int(mag * 10)))
                bar = "#" * bar_len
            print(f"  Layer {i+1:3d}: {bar} ({mag:.6f})")
```

## 实际使用

PyTorch 提供了内置函数：

```python
import torch
import torch.nn as nn

layer = nn.Linear(512, 256)

nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)

nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

nn.init.zeros_(layer.bias)
```

当你调用 `nn.Linear(512, 256)` 时，PyTorch 默认使用 Kaiming uniform 初始化。这就是为什么大多数简单网络"开箱即用"——PyTorch 已经做了正确的选择。但当你构建自定义架构或深度超过 20 层时，你需要理解底层发生了什么，并可能需要覆盖默认值。

对于 transformer，HuggingFace 模型通常在 `_init_weights` 方法中处理初始化。GPT-2 的实现将残差投影缩放 1/sqrt(N)。如果你从头构建 transformer，需要自己添加这个。

## 交付产出

本课产出：
- `outputs/prompt-init-strategy.md` -- 一个诊断权重初始化问题并推荐正确策略的 prompt

## 练习

1. 添加 LeCun 初始化（Var = 1/fan_in，为 SELU 激活设计）。用 LeCun init + tanh 运行 50 层实验，与 Xavier + tanh 对比。

2. 实现 GPT-2 残差缩放：在加到残差流之前，将每层输出乘以 1/sqrt(2*N)。分别运行有缩放和无缩放的 50 层，测量残差幅度增长速度。

3. 创建一个"初始化健康检查"函数，输入网络的层维度和激活类型，推荐正确的初始化方式，并在当前初始化会导致问题时发出警告。

4. 用 fan_in = 16 和 fan_in = 1024 分别运行实验。Xavier 和 Kaiming 会适应 fan_in，但 random init 不会。展示随着层变大，"能用"和"崩溃"之间的差距如何扩大。

5. 实现正交初始化（生成随机矩阵，计算其 SVD，使用正交矩阵 U）。在 50 层 ReLU 网络上与 Kaiming 对比。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Weight initialization | "随机设置初始权重" | 选择初始权重值的策略，决定了网络能否训练 |
| Symmetry breaking | "让神经元不同" | 使用随机初始化确保神经元学习不同的特征，而非计算相同的函数 |
| Fan-in | "神经元的输入数量" | 传入连接的数量，决定了输入方差如何在加权求和中累积 |
| Fan-out | "神经元的输出数量" | 传出连接的数量，与反向传播中维持梯度方差相关 |
| Xavier/Glorot init | "sigmoid 的初始化" | Var(w) = 2/(fan_in + fan_out)，设计用于保持 sigmoid 和 tanh 激活中的方差 |
| Kaiming/He init | "ReLU 的初始化" | Var(w) = 2/fan_in，补偿 ReLU 将一半激活值置零的效果 |
| Variance propagation | "信号如何在层间增长或缩小" | 基于权重尺度，分析激活值方差如何逐层变化的数学方法 |
| Residual scaling | "GPT-2 的初始化技巧" | 将残差连接权重缩放 1/sqrt(2N)，防止方差在 N 个 transformer 层中增长 |
| Dead network | "什么都训练不了" | 由于初始化不当导致所有梯度为零或所有激活值饱和的网络 |
| Exploding activations | "数值变成无穷大" | 权重方差过高时，激活值幅度在各层间指数增长 |

## 延伸阅读

- Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010) -- Xavier 初始化的原始论文，包含方差分析
- He et al., "Delving Deep into Rectifiers" (2015) -- 为 ReLU 网络引入了 Kaiming 初始化
- Radford et al., "Language Models are Unsupervised Multitask Learners" (2019) -- GPT-2 论文，包含残差缩放初始化
- Mishkin & Matas, "All You Need is a Good Init" (2016) -- 逐层单位方差初始化，一种分析公式的经验替代方案
