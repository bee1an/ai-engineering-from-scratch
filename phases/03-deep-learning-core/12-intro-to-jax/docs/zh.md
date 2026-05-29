# JAX 入门

> PyTorch 修改 tensor。TensorFlow 构建图。JAX 编译纯函数。最后这个会改变你对深度学习的思考方式。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 03 Lessons 01-10，基础 NumPy
**时间：** 约 90 分钟

## 学习目标

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数神经网络代码
- 解释 PyTorch 的 eager mutation 和 JAX 的函数式编译模型之间的关键设计差异
- 应用 jit 编译和 vmap 向量化来加速训练循环，对比朴素 Python
- 在 JAX 中训练一个简单网络，对比显式状态管理与 PyTorch 面向对象方法的差异

## 问题

你知道如何在 PyTorch 中构建神经网络。定义一个 `nn.Module`，调用 `.backward()`，step 优化器。它能用。数百万人在用。

但 PyTorch 有一个刻在 DNA 里的约束：它在 Python 中逐个操作地 eager trace。每个 `tensor + tensor` 是一次单独的 kernel 启动。每个训练步骤重新解释相同的 Python 代码。这在你需要跨 2,048 个 TPU 训练 5400 亿参数模型之前都没问题。然后开销就会杀死你。

Google DeepMind 用 JAX 训练 Gemini。Anthropic 用 JAX 训练 Claude。这些不是小操作——它们是地球上最大的神经网络训练任务。他们选择 JAX 是因为它把你的训练循环当作一个可编译的程序，而不是一系列 Python 调用。

JAX 是带三个超能力的 NumPy：自动微分、JIT 编译到 XLA、自动向量化。你写一个处理单个样本的函数。JAX 给你一个处理批次、计算梯度、编译为机器码、跨多设备运行的函数。全部不需要修改原始函数。

## 概念

### JAX 的哲学

JAX 是一个函数式框架。没有类，没有可变状态，没有 `.backward()` 方法。取而代之的是：

| PyTorch | JAX |
|---------|-----|
| `nn.Module` class with state | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | JIT compilation via XLA |
| `for x in batch:` manual loop | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | Immutable pytree of arrays |

这不是风格偏好，而是编译器约束。JIT 编译要求纯函数——相同输入总是产生相同输出，没有副作用。这个限制正是使 100 倍加速成为可能的原因。

### jax.numpy：熟悉的表面

JAX 在加速器上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同的函数名。相同的广播规则。相同的切片语义。但数组存在于 GPU/TPU 上，每个操作都可被编译器追踪。

一个关键区别：JAX 数组是不可变的。不能 `a[0] = 5`。而是：`a = a.at[0].set(5)`。这在第一周感觉别扭，然后你会顿悟——不可变性正是使 `grad`、`jit` 和 `vmap` 等变换可组合的原因。

### jax.grad：函数式自动微分

PyTorch 将梯度附加到 tensor 上（`.grad`）。JAX 将梯度附加到函数上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接收一个函数，返回一个计算梯度的新函数。没有 `.backward()` 调用。没有存储在 tensor 上的计算图。梯度只是另一个你可以调用、组合或 JIT 编译的函数。

这可以任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数。三阶导数。Jacobian。Hessian。全部通过组合 `grad`。PyTorch 也能做到（`torch.autograd.functional.hessian`），但那是后加的。在 JAX 中，这是基础。

约束：`grad` 只对纯函数有效。内部不能有 print 语句（它们在 tracing 时运行，不是执行时）。不能修改外部状态。不能在没有显式 key 管理的情况下生成随机数。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

第一次调用时，JAX trace 函数——它记录发生了哪些操作，但不执行它们。然后将该 trace 交给 XLA（Accelerated Linear Algebra），Google 为 TPU 和 GPU 开发的编译器。XLA 融合操作，消除冗余内存拷贝，生成优化的机器码。

后续调用完全跳过 Python。编译后的代码以 C++ 速度在加速器上运行。

JIT 有帮助的场景：
- 训练步骤（相同计算重复数千次）
- 推理（相同模型，不同输入）
- 任何用相似 shape 输入调用多次的函数

JIT 有害的场景：
- 有依赖值的 Python 控制流的函数（`if x > 0`，其中 x 是被 trace 的数组）
- 一次性计算（编译开销超过运行时间）
- 调试（tracing 隐藏了实际执行）

控制流限制是真实的。`jax.lax.cond` 替代 `if/else`。`jax.lax.scan` 替代 `for` 循环。这些不是可选的——它们是编译的代价。

### vmap：自动向量化

你写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 将它提升为处理批次：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 意味着：不在 `params` 上批处理（共享），在 `x` 的 axis 0 上批处理。没有手动 `for` 循环。没有 reshape。没有批次维度穿线。JAX 自动找出批次维度并向量化整个计算。

这不是语法糖。`vmap` 生成融合的向量化代码，比 Python 循环快 10-100 倍。而且它与 `jit` 和 `grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐样本梯度。一行代码。这在 PyTorch 中几乎不可能不用 hack 实现。

### pmap：跨设备数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 将函数复制到所有可用设备（GPU/TPU）并分割批次。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 跨设备同步梯度。

Google 使用 `pmap`（及其后继 `shard_map`）在数千个 TPU v5e 芯片上训练 Gemini。编程模型：写单设备版本，用 `pmap` 包装，完成。

### Pytree：通用数据结构

JAX 操作"pytree"——列表、元组、字典和数组的嵌套组合。你的模型参数就是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX 变换——`grad`、`jit`、`vmap`——都知道如何遍历 pytree。`jax.tree.map(f, tree)` 对每个叶子应用 `f`。这就是优化器一次更新所有参数的方式：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` 方法。没有参数注册。树结构就是模型。

### 函数式 vs 面向对象

PyTorch 将状态存储在对象内部：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用带显式状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Params 被传入。什么都不存储。什么都不修改。这使每个函数都可测试、可组合、可编译。这也意味着你自己管理 params——或者使用 Flax 或 Equinox 这样的库。

### JAX 生态系统

JAX 给你原语。库给你人体工程学：

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | Neural network layers | `nn.Module` with explicit state |
| **Equinox** (Patrick Kidger) | Neural network layers | Pytree-based, Pythonic |
| **Optax** (DeepMind) | Optimizers + LR schedules | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | Save/restore pytrees |
| **CLU** (Google) | Metrics + logging | Training loop utilities |

Optax 是标准优化器库。它将梯度变换（Adam、SGD、clipping）与参数更新分离，使组合变得简单：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时用 JAX vs PyTorch

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU support | First-class (Google built both) | Community-maintained (torch_xla) |
| GPU support | Good (CUDA via XLA) | Best-in-class (native CUDA) |
| Debugging | Hard (tracing + compilation) | Easy (eager, line-by-line) |
| Ecosystem | Research-focused (Flax, Equinox) | Massive (HuggingFace, torchvision, etc.) |
| Hiring | Niche (Google/DeepMind/Anthropic) | Mainstream (everywhere) |
| Large-scale training | Superior (XLA, pmap, mesh) | Good (FSDP, DeepSpeed) |
| Prototyping speed | Slower (functional overhead) | Faster (mutate and go) |
| Production inference | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Who uses it | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

诚实的答案：除非你有特定理由使用 JAX，否则用 PyTorch。那些理由是——TPU 访问、需要逐样本梯度、大规模多设备训练，或者在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每个随机操作都需要一个显式的 PRNG key：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

这一开始很烦人。但它保证了跨设备和编译的可复现性——这是 PyTorch 的 `torch.manual_seed` 在多 GPU 设置中无法保证的属性。

## 动手构建

### 第 1 步：设置和数据

我们将使用 JAX 和 Optax 在 MNIST 上训练一个 3 层 MLP。784 个输入，两个隐藏层分别有 256 和 128 个神经元，10 个输出类别。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 第 2 步：初始化参数

没有类。只是一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He initialization，手动完成。三个 PRNG key 从一个种子分裂出来。每个权重是嵌套字典中的不可变数组。

### 第 3 步：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。Params 进，预测出。没有 `self`，没有存储状态。`loss_fn` 从头计算 cross-entropy——softmax、log、负均值。

### 第 4 步：JIT 编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 在一次传递中返回 loss 值和梯度。`@jax.jit` 装饰器将两个函数编译到 XLA。第一次调用后，每个训练步骤不再接触 Python。

### 第 5 步：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 个 epoch。约 97% 测试准确率。第一个 epoch 慢（JIT 编译）。第 2-10 个 epoch 快。

注意缺少了什么：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新是一个组合的函数调用。梯度被计算、被 Adam 变换、被应用到参数——全部在 `train_step` 内部。

## 实际使用

### Flax：Google 标准

Flax 是最常见的 JAX 神经网络库。它加回了 `nn.Module`，但带有显式状态管理：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构与 PyTorch 相同，但 `params` 与模型分离。`model.init()` 创建 params。`model.apply(params, x)` 运行前向传播。模型对象没有状态。

### Equinox：Pythonic 替代方案

Equinox（Patrick Kidger 开发）将模型表示为 pytree：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身就是一个 pytree。不需要 `.apply()`。参数就是模型的叶子。这更接近 JAX 的思维方式。

### Optax：可组合的优化器

Optax 将梯度变换与更新解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、learning rate warmup、weight decay——全部组合为变换链。每个变换看到梯度，修改它们，传递给下一个。没有单体优化器类。

## 交付产出

**安装：**

```bash
pip install jax jaxlib optax flax
```

GPU 支持：

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能陷阱：**

- 第一次 JIT 调用慢（编译）。基准测试前先预热。
- 避免在 JIT 内部对 JAX 数组使用 Python 循环。使用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 在 JIT 内部有效。普通 `print()` 无效。
- 用 `jax.profiler` 或 TensorBoard 做性能分析。XLA 编译可能隐藏瓶颈。
- JAX 默认预分配 75% 的 GPU 内存。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 来禁用。

**Checkpointing：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产出：**
- `outputs/prompt-jax-optimizer.md` -- 一个选择正确 JAX 优化器配置的 prompt
- `outputs/skill-jax-patterns.md` -- 涵盖 JAX 函数式模式的技能

## 练习

1. 给 MLP 添加 dropout。在 JAX 中，dropout 需要一个 PRNG key——在前向传播中穿线一个 key，并为每个 dropout 层分裂它。对比有无 dropout 的测试准确率。

2. 使用 `jax.vmap` 计算 32 张 MNIST 图片批次的逐样本梯度。计算每个样本的梯度范数。哪些样本的梯度最大，为什么？

3. 用一个通用的 `mlp_forward(params, x)` 替换手动前向函数，使其适用于任意层数。使用 `jax.tree.leaves` 自动确定深度。

4. 对比有无 `@jax.jit` 的训练步骤基准。计时 100 步。在你的硬件上加速有多大？第一次调用的编译开销是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现梯度裁剪。有无裁剪分别训练。绘制训练过程中的梯度范数以观察效果。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| XLA | "让 JAX 快的东西" | Accelerated Linear Algebra——一个从计算图融合操作并生成优化 GPU/TPU kernel 的编译器 |
| JIT | "即时编译" | JAX 在第一次调用时 trace 函数，编译到 XLA，然后在后续调用中运行编译版本 |
| Pure function | "无副作用" | 输出仅依赖输入的函数——没有全局状态，没有修改，没有不带显式 key 的随机性 |
| vmap | "自动批处理" | 将处理单个样本的函数变换为处理批次的函数，无需重写 |
| pmap | "自动并行" | 将函数复制到多个设备并分割输入批次 |
| Pytree | "嵌套的数组字典" | 列表、元组、字典和数组的任意嵌套结构，JAX 可以遍历和变换 |
| Tracing | "记录计算" | JAX 用抽象值执行函数以构建计算图，不计算实际结果 |
| Functional autodiff | "函数的 grad" | 通过变换函数来计算导数，而不是将梯度存储附加到 tensor |
| Optax | "JAX 的优化器库" | 可组合的梯度变换库——Adam、SGD、clipping、scheduling——链式组合 |
| Flax | "JAX 的 nn.Module" | Google 为 JAX 开发的神经网络库，添加层抽象同时保持状态显式 |

## 延伸阅读

- JAX documentation: https://jax.readthedocs.io/ -- 官方文档，有关于 grad、jit 和 vmap 的优秀教程
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- 解释设计哲学的原始论文
- Flax documentation: https://flax.readthedocs.io/ -- Google 为 JAX 开发的神经网络库
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- Flax 的 Pythonic 替代方案
- DeepMind, "Optax: composable gradient transformation and optimisation" -- 标准优化器库
- "You Don't Know JAX" (Colin Raffel, 2020) -- JAX 陷阱和模式的实用指南，来自 T5 作者之一
