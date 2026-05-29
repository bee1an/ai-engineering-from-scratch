# 概率与分布

> 概率是 AI 用来表达不确定性的语言。

**类型：** 学习
**语言：** Python
**前置课程：** Phase 1, Lessons 01-04
**时间：** 约 75 分钟

## 学习目标

- 从零实现 Bernoulli、categorical、Poisson、uniform、normal 分布的 PMF 和 PDF
- 计算期望值、方差，并用中心极限定理解释为什么 Gaussian 分布无处不在
- 用数值稳定技巧（减去最大 logit）构建 softmax 和 log-softmax 函数
- 从 logits 计算 cross-entropy 损失，并把它和负对数似然联系起来

## 问题

一个分类器输出 `[0.03, 0.91, 0.06]`。一个语言模型从 50,000 个候选中挑下一个词。一个扩散模型通过从学到的分布中采样来生成图像。这些都是概率在起作用。

模型的每一个预测都是一个概率分布。每一个损失函数都在衡量预测分布与真实分布的距离。每一次训练都在调整参数，让一个分布更像另一个分布。没有概率，你读不懂任何 ML 论文，调不了任何模型，也搞不清楚训练损失为什么变成 NaN。

## 概念

### 事件、样本空间与概率

样本空间 S 是所有可能结果的集合。事件是样本空间的子集。概率把事件映射到 0 到 1 之间的数。

```
抛硬币：
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

掷一颗骰子：
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了整个概率论：
1. 对任意事件 A，P(A) >= 0
2. P(S) = 1（必有事发生）
3. 当 A 和 B 不能同时发生时，P(A or B) = P(A) + P(B)

其他一切（Bayes 定理、期望、分布）都从这三条规则推出来。

### 条件概率与独立性

P(A|B) 是在 B 发生条件下 A 发生的概率。

```
P(A|B) = P(A and B) / P(B)

例子：一副扑克牌
  P(K | 花牌) = P(K and 花牌) / P(花牌)
              = (4/52) / (12/52)
              = 4/12 = 1/3
```

两个事件独立，意味着知道一个对另一个没任何信息：

```
独立：         P(A|B) = P(A)
等价于：       P(A and B) = P(A) * P(B)
```

抛硬币是独立的。不放回地抽牌不是。

### PMF vs PDF（概率质量函数 vs 概率密度函数）

离散随机变量有概率质量函数（PMF）。每个结果有一个具体的概率，可以直接读出来。

```
PMF: P(X = k)

均匀骰子：
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  所有概率之和 = 1
```

连续随机变量有概率密度函数（PDF）。某一点的密度并不是概率。概率是密度在区间上的积分。

```
PDF: f(x)

P(a <= X <= b) = f(x) 在 a 到 b 上的积分

f(x) 可以大于 1（密度，不是概率）
f(x) 在 -inf 到 +inf 上的积分 = 1
```

这个区分在 ML 中很重要。分类输出是 PMF（离散选择）。VAE 隐空间用 PDF（连续）。

### 常见分布

**Bernoulli：** 一次试验，两种结果。建模二分类。

```
P(X = 1) = p
P(X = 0) = 1 - p
均值 = p,  方差 = p(1-p)
```

**Categorical：** 一次试验，k 种结果。建模多分类（softmax 输出）。

```
P(X = i) = p_i,  其中 p_i 之和为 1
例：P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform：** 所有结果等可能。用于随机初始化。

```
离散：P(X = k) = 1/n，k ∈ {1, ..., n}
连续：f(x) = 1/(b-a)，x ∈ [a, b]
```

**Normal（Gaussian）：** 钟形曲线。由均值（mu）和方差（sigma^2）参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

标准正态：mu = 0, sigma = 1
  68% 的数据在 1 个 sigma 内
  95% 在 2 个 sigma 内
  99.7% 在 3 个 sigma 内
```

**Poisson：** 固定区间内稀有事件的计数。建模事件率。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
均值 = lambda,  方差 = lambda
```

### 期望值与方差

期望值是按概率加权的平均结果。

```
离散：    E[X] = sum of x_i * P(X = x_i)
连续：    E[X] = integral of x * f(x) dx
```

方差衡量围绕均值的分散程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
标准差 = sqrt(Var(X))
```

ML 中，期望值出现在损失函数（数据分布上的平均损失）。方差告诉你模型的稳定性。梯度方差大意味着训练有噪声。

### 联合分布与边际分布

联合分布 P(X, Y) 描述两个随机变量一起的情况。

联合 PMF 例子（X = 天气，Y = 是否带伞）：

| | Y=0（没带伞） | Y=1（带伞） | 边际 P(X) |
|---|---|---|---|
| X=0（晴） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边际 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边际分布把另一个变量加和掉：

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

上面表格的行和列总和就是边际。

### 为什么正态分布无处不在

中心极限定理：许多独立随机变量的和（或平均）会收敛到正态分布，无论原始分布是什么。

```
掷 1 颗骰子：均匀分布（平的）
2 颗骰子的均值：三角形（有峰）
30 颗骰子的均值：几乎完美的钟形曲线

对任意起始分布都成立。
```

这就是为什么：
- 测量误差近似正态（很多小的独立来源）
- 神经网络的权重初始化用正态分布
- SGD 的梯度噪声近似正态（许多样本梯度的和）
- 给定均值和方差，正态分布是最大熵分布

### 对数概率

原始概率会带来数值问题。把许多小概率相乘很快就会下溢到零。

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0（约 30 项后下溢）
```

对数概率解决这个问题。乘法变加法。

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> 有限数（不会下溢）
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率始终 <= 0（因为 0 < P <= 1）
- 越负越不可能
- Cross-entropy 损失就是正确类别的负对数概率

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax 把它们转成有效的概率分布。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

性质：
  - 所有输出在 (0, 1) 之间
  - 所有输出加起来等于 1
  - 保留输入的相对顺序
  - exp() 放大 logits 之间的差距
```

Softmax 技巧：在做指数之前减去最大 logit，防止溢出。

```
z = [100, 101, 102]
exp(102) = 溢出

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1（安全）

结果一样，没溢出。
```

Log-softmax 把 softmax 和 log 合起来以保证数值稳定。PyTorch 在 cross-entropy 损失里就这么做。

### 采样

采样意味着从分布中抽取随机值。在 ML 中：
- Dropout 随机采样要置零的神经元
- 数据增强采样随机变换
- 语言模型从预测分布中采样下一个 token
- 扩散模型采样噪声并逐步去噪

从任意分布采样需要 inverse transform sampling、rejection sampling 或重参数化技巧（VAE 用的）等技术。

## 动手构建

### Step 1：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Step 2：从零实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Step 3：期望值与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Step 4：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Step 5：Softmax 与对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Step 6：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Step 7：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整实现和所有可视化在 `code/probability.py`。

## 用起来

用 NumPy 和 SciPy，上面的一切都是一行的事：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你已经从零搭过了，现在你知道库函数在做什么。

## 练习

1. 为 exponential 分布实现 inverse transform sampling。采样 10,000 个值，把直方图与真实 PDF 对比来验证。

2. 为两颗有偏骰子构建联合分布表。计算边际分布，并检查骰子是否独立。

3. 计算一个 5 类分类器的 cross-entropy 损失，logits 为 `[2.0, 0.5, -1.0, 3.0, 0.1]`，正确类别是 index 3。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证。

4. 写一个函数，输入一组 log 概率，返回最可能序列、总 log 概率，以及对应的原始概率。用 50 个词每个概率 0.01 的句子测试。

## 关键术语

| 术语 | 大家怎么说 | 它实际上是什么 |
|------|----------------|----------------------|
| Sample space | "所有可能性" | 一个实验所有可能结果的集合 S |
| PMF | "概率函数" | 给出每个离散结果精确概率的函数，加和为 1 |
| PDF | "概率曲线" | 连续变量的密度函数。在区间上积分得到概率 |
| Conditional probability | "给定条件下的概率" | P(A\|B) = P(A and B) / P(B)。Bayesian 思维和 Bayes 定理的基础 |
| Independence | "互不影响" | P(A and B) = P(A) * P(B)。知道一个事件对另一个毫无信息 |
| Expected value | "平均值" | 按概率加权的所有结果之和。损失函数就是一个期望值 |
| Variance | "分散程度" | 与均值偏差的平方的期望。方差大 = 估计有噪声、不稳定 |
| Normal distribution | "钟形曲线" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。因为 CLT 而无处不在 |
| Central Limit Theorem | "平均值会变正态" | 许多独立样本的均值收敛到正态分布，与原分布无关 |
| Joint distribution | "两个变量一起" | P(X, Y) 描述 X 和 Y 每种组合的概率 |
| Marginal distribution | "把另一个变量加和掉" | P(X) = sum_y P(X, Y)。从联合分布中恢复一个变量的分布 |
| Log probability | "概率的对数" | log P(x)。把乘法变加法，防止长序列的数值下溢 |
| Softmax | "把分数变概率" | softmax(z_i) = exp(z_i) / sum(exp(z_j))。把实数 logits 映射到有效概率分布 |
| Cross-entropy | "损失函数" | -sum(p_true * log(p_predicted))。衡量两个分布有多不同。越低越好 |
| Logits | "模型原始输出" | softmax 之前未归一化的分数。名字来自 logistic 函数 |
| Sampling | "抽取随机值" | 按概率分布生成值。模型生成输出的方式 |

## 延伸阅读

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 为什么平均值会变正态的可视化证明
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 涵盖这里全部内容及更多的简明参考
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 数值稳定性为什么重要以及如何实现
