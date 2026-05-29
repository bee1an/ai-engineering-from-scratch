# 为什么选择 Transformer — RNN 的致命缺陷

> RNN 逐个处理 token。Transformer 一次处理所有 token。这一个架构决策改变了 2017 年之后深度学习的每一条 scaling 曲线。

**类型：** 学习
**语言：** Python
**前置课程：** Phase 3（深度学习核心）、Phase 5 · 09（Sequence-to-Sequence）、Phase 5 · 10（注意力机制）
**时间：** 约 45 分钟

## 问题

2017 年之前，地球上每一个最先进的序列模型——语言、翻译、语音——都是循环神经网络。LSTM 和 GRU 在翻译基准测试上称霸了整整五年。它们是唯一的工具。

它们有三个致命弱点。串行计算意味着你无法沿时间轴并行化：token `t+1` 需要 token `t` 的隐藏状态。一个 1,024 token 的序列意味着在一个每周期能做 1,000,000 次浮点运算的 GPU 上执行 1,024 个串行步骤。训练的墙钟时间随序列长度线性增长——而硬件本身是为并行设计的。

梯度消失意味着 50 个 token 之前的信息已经被压缩过 50 次非线性变换。门控循环单元（LSTM、GRU）缓解了这种压缩，但从未消除它。长距离依赖——"the book I read last summer on a plane to Kyoto was…"——经常失败。

固定宽度的隐藏状态意味着 encoder 在 decoder 看到任何东西之前，就把整个源序列压缩成一个向量。不管源序列是 5 个 token 还是 500 个，瓶颈的形状都一样。

2017 年的论文 "Attention Is All You Need" 提出了一个激进的方案：完全抛弃循环。让每个位置并行地关注其他所有位置。用一次大矩阵乘法来训练，而不是 1,024 次串行运算。

到 2026 年，这个结果主导了每一个模态。语言（GPT-5、Claude 4、Llama 4）、视觉（ViT、DINOv2、SAM 3）、音频（Whisper）、生物学（AlphaFold 3）、机器人（RT-2）。同一个 block，不同的输入。

## 概念

![RNN 串行计算 vs Transformer 并行注意力](../assets/rnn-vs-transformer.svg)

**循环作为瓶颈。** RNN 计算 `h_t = f(h_{t-1}, x_t)`。每一步都依赖前一步。你无法在 `h_4` 之前计算 `h_5`。在拥有 10,000+ 并行核心的现代 GPU 上，长序列会浪费 99% 的算力。

**注意力作为广播。** Self-attention 对每一对 `(i, j)` 同时计算 `output_i = sum_j(a_ij * v_j)`。整个 N×N 注意力矩阵在一次 batched matmul 中填满。没有步骤依赖另一个步骤。GPU 喜欢这个。

**加速不是一个常数。** 它是 `O(N)` 串行深度和 `O(1)` 串行深度之间的差异。实际上，在 N=512 时，transformer 在相同硬件上每个 epoch 训练快 5–10 倍，而且差距随序列长度增大——直到你撞上注意力的 `O(N²)` 内存墙（Flash Attention 后来解决了这个问题——见 Lesson 12）。

**Transformer 的代价。** 注意力内存按 `O(N²)` 增长。2K 上下文没问题。128K 上下文就需要滑动窗口、RoPE 外推、Flash Attention tiling 或线性注意力变体。循环在时间和内存上都是 `O(N)`；transformer 用内存换时间，然后通过并行把时间赢回来。

**归纳偏置的转变。** RNN 假设局部性和近因性。Transformer 什么都不假设——每一对都是注意力的候选。这就是为什么 transformer 需要更多数据才能训练好，但一旦有了数据就能 scale 得更远。Chinchilla（2022）形式化了这一点：给定足够的 token，transformer 总是打败同等参数量的 RNN。

## 动手构建

这里没有神经网络——我们用数值模拟核心瓶颈，让你在笔记本电脑上感受差距。

### 第 1 步：测量串行深度

见 `code/main.py`。我们构建两个函数。一个把序列编码为加法链（串行，像 RNN）。一个编码为并行归约（广播，像注意力）。相同的数学，不同的依赖图。

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

我们对长度最多 100,000 的序列计时。RNN 版本是 O(N) 的单 CPU 流水线。即使在纯 Python 中，attention 风格的归约在长度 ≥ 1,000 时就能胜出，因为 Python 的 `sum()` 是用 C 实现的，迭代时没有每步的解释器开销。

### 第 2 步：计算理论操作数

两种算法都做 N 次加法。区别在于*依赖深度*：在下一个操作开始之前，必须串行完成多少操作。RNN 深度 = N。注意力深度 = log(N)（树归约）或 1（并行扫描）。决定 GPU 时间的是深度，不是操作数。

### 第 3 步：长序列上的经验 scaling

我们打印一个计时表，让 O(N) 的差距可见。在 2026 年的 Mac 笔记本上，1,000 以下的序列太快无法测量。100,000 的序列展示出清晰的线性扫描。把这个 scale 到一个 16,384-token 的 transformer 对比 12 层 LSTM，你就能看到为什么训练墙钟时间在 2016 年是个阻碍。

## 实际应用

2026 年什么时候还会选 RNN：

| 场景 | 选择 |
|------|------|
| 流式推理，逐 token，恒定内存 | RNN 或状态空间模型（Mamba、RWKV） |
| 超长序列（>1M token），注意力内存爆炸 | 线性注意力、Mamba 2、Hyena |
| 没有矩阵乘法加速器的边缘设备 | Depthwise-separable RNN 在 FLOPs/watt 上仍然赢 |
| 其他所有情况（训练、批量推理、上下文到 128K） | Transformer |

状态空间模型（SSM）如 Mamba 本质上是带有结构化参数化的 RNN，兼具两者优势：`O(N)` 扫描内存、通过 selective scan 并行训练。它们恢复了 transformer 90% 的质量，同时有更好的长上下文 scaling。2026 年大多数前沿实验室训练混合 SSM+transformer 模型（如 Jamba、Samba）——循环没有死，它是一个组件。

## 交付产出

见 `outputs/skill-architecture-picker.md`。该 skill 根据长度、吞吐量和训练预算约束，为新的序列问题选择架构。它应该始终拒绝在超过 1B token 的训练中推荐纯 RNN，除非说明了 trade-off。

## 练习

1. **简单。** 取 `code/main.py` 中的 `rnn_style`，把标量隐藏状态替换为长度 64 的向量。重新测量。串行开销随隐藏状态维度增长了多少？
2. **中等。** 用纯 Python 实现并行前缀和（Hillis-Steele scan）。验证它在长度 1024 上产生与串行扫描相同的数值输出。计算深度。
3. **困难。** 把 attention 风格的归约移植到 GPU 上的 PyTorch。在序列长度从 64 到 65,536 的范围内计时。画图并解释曲线形状。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Recurrence | "RNN 是串行的" | 步骤 `t` 依赖步骤 `t-1` 的计算，强制沿时间轴串行执行。 |
| Serial depth | "图有多深" | 最长的依赖操作链；即使在无限硬件上也限制墙钟时间。 |
| Attention | "让 token 互相看" | 加权求和 `sum_j a_ij v_j`，其中 `a_ij` 来自位置 i 和 j 之间的相似度分数。 |
| Context window | "模型能看多少" | 注意力层能接受的位置数量；二次内存开销在这里 scale。 |
| Inductive bias | "架构中内置的假设" | 关于数据长什么样的先验；CNN 假设平移不变性，RNN 假设近因性。 |
| State-space model | "有代数支撑的 RNN" | 通过结构化状态空间矩阵参数化的循环，可并行训练。 |
| Quadratic bottleneck | "为什么上下文这么贵" | 注意力内存 = `O(N²)` 随序列长度增长；Flash Attention 隐藏了常数，但没改变 scaling。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 在主流 NLP 中终结循环的论文。
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 注意力诞生的地方，嫁接在 RNN 上。
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — 原始 LSTM 论文，留作记录。
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — 现代循环对 transformer 的回应。
