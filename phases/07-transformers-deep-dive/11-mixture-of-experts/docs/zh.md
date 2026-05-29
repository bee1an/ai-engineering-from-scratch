# Mixture of Experts (MoE)

> 一个稠密的 70B transformer 对每个 token 激活所有参数。一个 671B 的 MoE 每个 token 只激活 37B，却在每个基准上都赢了。稀疏性是这十年最重要的扩展思想。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## 问题

稠密 transformer 推理时的 FLOPs 等于其参数量（乘以 2 做前向传播）。放大稠密模型，每个 token 都要付全部账单。到 2024 年，前沿模型撞上了算力墙：要显著变聪明，你需要指数级增长的每 token FLOPs。

Mixture of Experts 打破了这个绑定。把每个 FFN 替换为 `E` 个独立专家 + 一个路由器，每个 token 选 `k` 个专家。总参数 = `E × FFN_size`。每 token 激活参数 = `k × FFN_size`。2026 年典型配置：`E=256`，`k=8`。存储随 `E` 扩展，计算随 `k` 扩展。

2026 年的前沿几乎全是 MoE：DeepSeek-V3（671B 总量 / 37B 激活）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，前 10 名开源模型全是 MoE。

## 概念

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### FFN 替换

稠密 transformer 块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家是一个独立的 FFN（通常是 SwiGLU）。路由器是一个线性层。每个 token 选自己的 `k` 个专家，得到它们输出的门控混合。

### 负载均衡问题

如果路由器把 90% 的 token 送到专家 3，其他专家就饿死了。三种修复方案：

1. **辅助负载均衡损失**（Switch Transformer、Mixtral）。添加一个与专家使用方差成比例的惩罚。有效，但增加了超参数和第二个梯度信号。
2. **专家容量 + token 丢弃**（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；溢出 token 跳过该层。损害质量。
3. **无辅助损失均衡**（DeepSeek-V3）。添加一个可学习的逐专家偏置，偏移路由器的 top-k 选择。偏置在训练损失之外更新。对主目标没有惩罚。2024 年的重大突破。

DeepSeek-V3 的方法：每个训练步之后，对每个专家检查其使用量是高于还是低于目标。将偏置微调 `±γ`。选择使用 `scores + bias`。用于门控的专家概率是原始 `scores` 不变。将路由与表达解耦。

### 共享专家

DeepSeek-V2/V3 还将专家分为*共享*和*路由*两类。每个 token 都经过所有共享专家。路由专家通过 top-k 选择。共享专家捕获通用知识；路由专家做专业化。V3 运行 1 个共享专家加 256 个路由专家中的 top-8。

### 细粒度专家

经典 MoE（GShard、Switch）：每个专家和完整 FFN 一样宽。`E` 小（8–64），`k` 小（1–2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（1/8 FFN 大小）。`E` 大（256+），`k` 更大（8+）。总参数相同，但组合数扩展快得多。`C(256, 8) = 400 万亿`种可能的"专家组合"每 token。质量上升，延迟不变。

### 成本概况

每 token，每层：

| Config | Active params / token | Total params |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 在几乎每个基准上都击败 Llama 3 70B（稠密），同时**每 token 激活 FLOPs 更少**。更多参数 = 更多知识。更多激活 FLOPs = 每 token 更多计算。MoE 将两者解耦。

### 代价：显存

所有专家都驻留在 GPU 上，不管哪些被激活。一个 671B 模型的 fp16 权重需要 ~1.3 TB VRAM。前沿 MoE 部署需要专家并行——将专家分片到不同 GPU，跨网络路由 token。延迟由 all-to-all 通信主导，而非矩阵乘法。

## 动手构建

见 `code/main.py`。一个紧凑的纯标准库 MoE 层：

- `n_experts=8` 个 SwiGLU 风格专家（每个一个线性层，用于演示）
- top-k=2 路由
- softmax 归一化的门控权重
- 通过逐专家偏置实现无辅助损失均衡

### 第 1 步：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，不影响门控权重。这就是 DeepSeek-V3 的技巧——偏置纠正负载不均衡，但不干预模型的预测。

### 第 2 步：让 100 个 token 通过路由器

追踪哪些专家被激活了多少次。没有偏置时，使用量是倾斜的。加上偏置更新循环（过度使用的专家 `-γ`，使用不足的 `+γ`），使用量在几次迭代后收敛到均匀分布。

### 第 3 步：参数量对比

打印一个 MoE 配置的"稠密等价"。DeepSeek-V3 形状：256 路由 + 1 共享，8 激活，d_model=7168。总参数量令人咋舌。激活参数量是稠密 Llama 3 70B 的七分之一。

## 使用方式

HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年生产推理：vLLM 原生支持 MoE 路由。SGLang 有最快的专家并行路径。两者都自动处理 top-k 选择和专家并行。

**什么时候选 MoE：**
- 你想要前沿质量但每 token 推理成本更低。
- 你有 VRAM / 专家并行基础设施。
- 你的工作负载是 token 密集型（聊天、代码）而非上下文密集型（长文档）。

**什么时候不选 MoE：**
- 边缘部署——你为任何激活 FLOP 都要付全部存储成本。
- 延迟敏感的单用户服务——专家路由增加开销。
- 小模型（<7B）——MoE 的质量优势只在超过某个算力阈值（~6B 激活参数）后才出现。

## 交付产出

见 `outputs/skill-moe-configurator.md`。该 skill 根据参数预算、训练 token 数和部署目标，为新 MoE 选择 E、k 和共享专家布局。

## 练习

1. **简单。** 运行 `code/main.py`。观察无辅助损失偏置更新如何在 50 次迭代中均衡专家使用量。
2. **中等。** 将学习型路由器替换为基于哈希的路由器（确定性，无学习）。对比质量和均衡性。为什么学习型路由器更好？
3. **困难。** 实现 GRPO 风格的"rollout-matched routing"（DeepSeek-V3.2 技巧）：记录推理时哪些专家被激活，在梯度计算时强制相同路由。在一个玩具策略梯度设置上测量效果。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Expert | "One FFN among many" | An independent feed-forward network; parameters dedicated to a sparse slice of the FFN computation. |
| Router | "The gate" | A tiny linear layer that scores each token against each expert; top-k selection. |
| Top-k routing | "k active experts per token" | Each token's FFN computation goes through exactly k experts, weighted by gate. |
| Auxiliary loss | "Load-balance penalty" | Extra loss term that penalizes skewed expert usage. |
| Auxiliary-loss-free | "DeepSeek-V3's trick" | Balance via per-expert bias on the router's selection only; no extra gradient. |
| Shared expert | "Always on" | Extra expert through which every token passes; captures common knowledge. |
| Expert parallelism | "Shard by expert" | Distribute different experts to different GPUs; route tokens across the network. |
| Sparsity | "Active params < total params" | The ratio `k × expert_size / (E × expert_size)`; 37/671 ≈ 5.5% for DeepSeek-V3. |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — the idea.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch, the classic MoE.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + auxiliary-loss-free MoE + MTP.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — the bias-based balancing paper.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — the fine-grained + shared-expert split this lesson's router uses.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — original shared-expert paper.
