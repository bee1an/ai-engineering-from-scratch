# Numerical Stability

> 浮点数是一种会漏的抽象。它会在你训练时咬你一口，而你毫无察觉。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## Learning Objectives

- 使用 max-subtraction 技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的 overflow、underflow 以及 catastrophic cancellation
- 使用 centered finite differences 验证解析梯度与数值梯度的一致性
- 解释为什么训练中 bfloat16 优于 float16，以及 loss scaling 如何防止梯度 underflow

## The Problem

你的模型训练了三个小时，然后 loss 变成了 NaN。你加了一句 print。在第 9,000 步时 logits 还正常，第 9,001 步它们变成 `inf`，到第 9,002 步每个梯度都是 `nan`，训练彻底完蛋。

或者：你的模型训练完成了，但准确率比论文声称的低 2%。你检查了所有东西。架构对得上，超参数对得上，数据也对得上。问题在于论文用的是 float32，而你用的是 float16 且没有正确的 scaling。三十二位累积的 rounding error 悄悄吃掉了你的准确率。

或者：你从零实现了 cross-entropy loss。在小 logits 上工作正常。当 logits 超过 100 时，它返回 `inf`。softmax 发生了 overflow，因为 `exp(100)` 大于 float32 能表示的范围。每个 ML 框架都用一个两行的技巧处理这件事，而你不知道这个技巧的存在。

数值稳定性不是一个理论问题。它是训练成功与悄无声息地失败之间的差别。你将要 debug 的每一个严重 ML bug，最终都会归结到浮点数上。

## The Concept

### IEEE 754: How Computers Store Real Numbers

计算机按照 IEEE 754 标准把实数存储为 floating point 值。一个 float 有三部分：sign bit、exponent 和 mantissa（significand）。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

mantissa 决定精度（多少有效数字），exponent 决定范围（数能多大或多小）。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 大约提供 7 位十进制精度。这意味着它能区分 1.0000001 和 1.0000002，但区分不了 1.00000001 和 1.00000002。超过 7 位之后，全是 rounding noise。

float16 大约只有 3 位精度。它能表示的最大数是 65,504。对于 logits、gradients 和 activations 经常超过这个数量级的 ML 来说，这小得令人不安。

bfloat16 是 Google 对 float16 范围问题的回答。它有和 float32 相同的 8 位 exponent（相同的范围，最大 3.4e38），但只有 7 位 mantissa（比 float16 精度更低）。对于训练神经网络来说，range 比 precision 更重要，所以 bfloat16 通常胜出。

### Why 0.1 + 0.2 != 0.3

数字 0.1 在二进制 floating point 中无法被精确表示。在二进制下，它是一个无限循环小数：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

float32 把这个截断到 23 位 mantissa。存储的值约为 0.100000001490116。同样地，0.2 存储为约 0.200000002980232。它们的和是 0.300000004470348，不是 0.3。

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这件事对 ML 重要，因为：

1. 像 `if loss < threshold` 这样的 loss 比较可能给出错误答案
2. 累加许多小值（成千上万步的 gradient updates）会偏离真实和
3. 如果用 `==` 比较 float，checksum 和可重复性测试会失败

修复办法：永远不要用 `==` 比较 float。用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### Catastrophic Cancellation

当你做两个几乎相等的 floating point 数的相减时，有效数字相互抵消，剩下的是被提到首位的 rounding noise。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

仅一次相减就有 19% 的相对误差。在 ML 中，这会发生在你：

- 计算均值很大的数据的方差时：`E[x^2] - E[x]^2` 当 E[x] 很大
- 相减几乎相等的 log-probabilities
- 用太小的 epsilon 计算 finite-difference gradients

修复办法：重新整理公式以避免相减大的、几乎相等的数。对于方差，使用 Welford algorithm 或先把数据中心化。对于 log-probabilities，全程在 log-space 中工作。

### Overflow and Underflow

Overflow 发生在结果太大无法表示时。Underflow 发生在结果太小（比最小可表示的正数更接近零）时。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 函数是 ML 中 overflow 的主要来源：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 函数则在另一个方向出问题：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在 cross-entropy、log-likelihoods 和 KL divergence 中。组合 `log(exp(x))` 是个雷区，没有正确技巧的话很危险。

### The Log-Sum-Exp Trick

直接计算 `log(sum(exp(x_i)))` 在数值上很危险。如果某个 `x_i` 很大，`exp(x_i)` 会 overflow。如果所有 `x_i` 都非常负，每个 `exp(x_i)` 都会 underflow 到零，`log(0)` 是 `-inf`。

技巧是：在指数化之前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么有效：减去 `max(x)` 之后，最大的指数是 `exp(0) = 1`。不可能 overflow。和中至少有一项是 1，所以和至少为 1，且 `log(1) = 0`。不可能 underflow 到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

令 `c = max(x)`，overflow 就消除了。

这个技巧在 ML 中无处不在：
- Softmax normalization
- Cross-entropy loss computation
- 序列模型中的 log-probability summation
- Mixture of Gaussians
- Variational inference

### Why Softmax Needs the Max-Subtraction Trick

Softmax 把 logits 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

不用这个技巧时，[100, 101, 102] 的 logits 会引发 overflow：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

用这个技巧，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率结果完全相同，但计算是安全的。这不是优化，而是正确性的要求。

### NaN and Inf: Detection and Prevention

`nan`（Not a Number）和 `inf`（infinity）会像病毒一样在计算中传播。一次梯度更新中的一个 `nan` 会让权重变成 `nan`，进而让所有后续输出变成 `nan`。训练在一步之内就死掉了。

`inf` 如何出现：
- 大正数的 `exp()`
- 除以零：`1.0 / 0.0`
- 累加中的 `float32` overflow

`nan` 如何出现：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 负数的 `sqrt()`
- 负数的 `log()`
- 涉及已有 `nan` 的任何运算

检测：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

预防策略：

1. 钳制 `exp()` 的输入：`exp(clamp(x, -80, 80))`
2. 给分母加 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 内部加 epsilon：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp，stable softmax）
5. 用 gradient clipping 防止权重爆炸
6. 在 debugging 时每次 forward pass 后检查 `nan`/`inf`

### Numerical Gradient Checking

解析梯度（来自 backpropagation）可能有 bug。Numerical gradient checking 通过 finite differences 计算梯度来验证它们。

centered difference 公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是 O(h^2) 精度，比只是 O(h) 的 forward difference `(f(x+h) - f(x)) / h` 好得多。

选 h：太大近似就不准；太小则 catastrophic cancellation 会破坏答案。`h = 1e-5` 到 `1e-7` 是典型值。

检查：计算解析梯度和数值梯度的相对差。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可以接受，可能正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错了

实现新 layer 或 loss function 时一定要检查梯度。PyTorch 提供了 `torch.autograd.gradcheck()` 来做这件事。

### Mixed Precision Training

现代 GPU 有专门硬件（Tensor Cores），计算 float16 矩阵乘法比 float32 快 2-8 倍。Mixed precision training 利用了这一点：

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯 float16 训练的问题：梯度通常很小（1e-8 或更小）。float16 把任何低于 ~6e-8 的值 underflow 到零。你的模型停止学习，因为所有梯度更新都是零。

修复办法是 loss scaling：

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Dynamic loss scaling 自动调整 scale factor。从一个大值开始（65536）。如果梯度 overflow 到 `inf`，把它减半。如果 N 步没有 overflow，把它加倍。

### bfloat16 vs float16: Why bfloat16 Wins for Training

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 精度更高（10 位 mantissa 对 7 位）但范围有限（最大 ~65,504）。bfloat16 精度更低，但范围与 float32 相同（最大 ~3.4e38）。

对于训练神经网络：

- Activations 和 logits 在训练 spike 时经常超过 65,504。float16 会 overflow；bfloat16 能处理。
- Loss scaling 在 float16 下是必需的，但在 bfloat16 下通常没必要，因为它的范围已经覆盖了梯度幅度的频谱。
- bfloat16 是 float32 的简单截断：丢掉 mantissa 的低 16 位。转换简单，且 exponent 部分无损。

float16 在 inference 中更受青睐，因为那里的值有界且精度更重要。bfloat16 在 training 中更受青睐，因为那里 range 更重要。这就是为什么 TPU 和现代 NVIDIA GPU（A100、H100）原生支持 bfloat16。

### Gradient Clipping

Exploding gradients 在梯度通过许多层指数级增长时发生（在 RNN、深层网络和 transformer 中常见）。一个大梯度就能在一步内毁掉所有权重。

两种 clipping：

**Clip by value：** 独立钳制每个梯度元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单，但可能改变梯度向量的方向。

**Clip by norm：** 缩放整个梯度向量，使其 norm 不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留梯度方向。这就是 `torch.nn.utils.clip_grad_norm_()` 所做的，是标准选择。

典型值：transformer 用 `max_norm=1.0`，RL 用 `max_norm=0.5`，简单网络用 `max_norm=5.0`。

Gradient clipping 不是 hack，而是一种安全机制。没有它，一个 outlier batch 就能产生足以毁掉数周训练的大梯度。

### Normalization Layers as Numerical Stabilizers

Batch normalization、layer normalization 和 RMS normalization 通常被介绍为帮助训练收敛的 regularizer。它们也是数值稳定器。

没有 normalization 时，activations 可以在层间指数级增长或缩小：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Normalization 在每一层重新中心化和重新缩放 activations：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常是 1e-5）防止当所有 activations 都相同时除以零。可学习的参数 `gamma` 和 `beta` 让网络可以恢复任何它需要的尺度。

这让值在整个网络中保持在数值安全范围，既防止 forward pass 中的 overflow，也防止 backward pass 中的梯度爆炸。

### Common ML Numerical Bugs

**Bug：几个 epoch 之后 loss 是 NaN。**
原因：logits 长得太大，softmax overflow 了。或者 learning rate 太高，权重发散了。
修复：使用 stable softmax（max subtraction），降低 learning rate，加 gradient clipping。

**Bug：loss 卡在 log(num_classes)。**
原因：模型输出接近均匀概率。通常意味着梯度消失或模型根本没在学。
修复：检查数据 label 是否正确，验证 loss function，检查是否有 dead ReLU。

**Bug：验证准确率比预期低 1-3%。**
原因：mixed precision 没有正确的 loss scaling。Gradient underflow 悄悄把小更新清零了。
修复：启用 dynamic loss scaling，或换成 bfloat16。

**Bug：某些层的 gradient norm 是 0.0。**
原因：dead ReLU 神经元（所有输入都为负），或 float16 underflow。
修复：用 LeakyReLU 或 GELU，使用 gradient scaling，检查权重初始化。

**Bug：模型在一个 GPU 上工作，但在另一个 GPU 上给出不同结果。**
原因：非确定性的浮点累加顺序。GPU 并行 reduction 在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
修复：接受小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度损失。

**Bug：loss 计算中 `exp()` 返回 `inf`。**
原因：原始 logits 没经 max-subtraction 技巧就传给了 `exp()`。
修复：使用 `torch.nn.functional.log_softmax()`，它内部实现了 log-sum-exp。

**Bug：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示低于 6e-8 的梯度幅度或高于 65,504 的 activations。
修复：使用带 loss scaling 的 mixed precision（AMP），或改用 bfloat16。

## Build It

### Step 1: Demonstrate floating point precision limits

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Step 2: Implement naive vs stable softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Step 3: Implement stable log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Step 4: Implement stable cross-entropy

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Step 5: Gradient checking

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Use It

### Mixed precision simulation

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient clipping

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf detection

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现及所有 edge case 演示见 `code/numerical.py`。

## Ship It

本节产出：
- `code/numerical.py`，包含 stable softmax、log-sum-exp、cross-entropy、gradient checking 和 mixed precision simulation
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现会在 Phase 3 构建训练循环、Phase 4 实现 attention 机制时再次出现。

## Exercises

1. **Catastrophic cancellation。** 用 float32 和 naive 公式 `E[x^2] - E[x]^2` 计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后用 Welford 在线算法计算它。把误差与真实方差（0.6667）做比较。

2. **Precision hunt。** 找到使得 Python 中 `1.0 + x == 1.0` 的最小正 float32 值 `x`。这就是 machine epsilon。验证它与 `numpy.finfo(numpy.float32).eps` 一致。

3. **Log-sum-exp edge cases。** 用以下输入测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其他，(c) 所有值都非常负（-1000）。验证它在 naive 版本失败的地方仍能给出正确结果。

4. **Gradient checking a neural network layer。** 实现一个单线性层 `y = Wx + b` 及其解析的 backward pass。用 `numerical_gradient` 为一个 3x2 权重矩阵验证正确性。

5. **Loss scaling experiment。** 模拟 float16 训练：在 [1e-9, 1e-3] 范围内创建随机梯度，转换为 float16，测量变成零的比例。然后应用 loss scaling（乘以 1024），转换为 float16，再缩放回去，再次测量零的比例。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| IEEE 754 | "The float standard" | 定义二进制 floating point 格式、舍入规则和特殊值（inf、nan）的国际标准。每个现代 CPU 和 GPU 都实现了它。 |
| Machine epsilon | "The precision limit" | 在给定 float 格式下使得 1.0 + e != 1.0 的最小值 e。对 float32 来说约为 1.19e-7。 |
| Catastrophic cancellation | "Precision loss from subtraction" | 相减几乎相等的 floating point 数时，有效数字相互抵消，rounding noise 主导了结果。 |
| Overflow | "Number too big" | 结果超过最大可表示值，变成 inf。exp(89) 在 float32 中 overflow。 |
| Underflow | "Number too small" | 结果比最小可表示的正数更接近零，变成 0.0。exp(-104) 在 float32 中 underflow。 |
| Log-sum-exp trick | "Subtract the max first" | 通过提取 exp(max(x)) 来计算 log(sum(exp(x)))，从而防止 overflow 和 underflow。用于 softmax、cross-entropy 和 log-probability 数学。 |
| Stable softmax | "Softmax that does not explode" | 在指数化之前减去 max(logits)。结果在数值上完全相同，且不可能 overflow。 |
| Gradient checking | "Verify your backprop" | 把 backpropagation 得到的解析梯度与 finite differences 得到的数值梯度做比较，以发现实现 bug。 |
| Mixed precision | "Float16 forward, float32 backward" | 在速度关键的运算上使用更低精度的 float，在数值敏感的运算上使用更高精度的 float。典型加速 2-3 倍。 |
| Loss scaling | "Prevent gradient underflow" | 在 backprop 之前把 loss 乘以一个大常数，使梯度保持在 float16 可表示范围内，然后在权重更新前再除以同一常数。 |
| bfloat16 | "Brain floating point" | Google 的 16 位格式，有 8 位 exponent（与 float32 同范围）和 7 位 mantissa（精度低于 float16）。训练中更受青睐。 |
| Gradient clipping | "Cap the gradient norm" | 缩放梯度向量使其 norm 不超过阈值，防止 exploding gradients 毁掉权重。 |
| NaN | "Not a Number" | 来自未定义运算（0/0、inf-inf、sqrt(-1)）的特殊 float 值。会传播到所有后续算术中。 |
| Inf | "Infinity" | 来自 overflow 或除以零的特殊 float 值。可组合产生 NaN（inf - inf、inf * 0）。 |
| Numerical gradient | "Brute force derivative" | 通过计算 f(x+h) 和 f(x-h) 然后除以 2h 来近似导数。慢但可靠，适合验证。 |

## Further Reading

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考，密集但全面
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- 引入 float16 训练 loss scaling 的 NVIDIA 论文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch mixed precision 实战指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- 为什么 Google 为 TPU 选择这个格式
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 减少浮点求和 rounding error 的算法
