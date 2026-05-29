# KV Cache、Flash Attention 与推理优化

> 训练是并行的、算力瓶颈。推理是串行的、显存瓶颈。不同的瓶颈，不同的技巧。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~75 minutes

## 问题

一个朴素的自回归 decoder 生成 `N` 个 token 需要 `O(N²)` 的工作量：每一步都重新计算整个前缀的注意力。对于一个 4K token 的回复，这是 1600 万次注意力运算，其中大部分是冗余的。前缀 token 的每个隐藏状态一旦计算出来就是确定的——你只需要用新 token 的 query 去查询之前所有 token 缓存的 key 和 value。

除此之外，注意力本身移动大量数据。标准注意力会物化一个 N×N 的分数矩阵、N×d 的 softmax 输出、N×d 的最终输出——对 HBM 的读写太多了。当 N≥2K 时，注意力在成为算力瓶颈之前就先成为显存瓶颈了。经典注意力 kernel 对现代 GPU 的利用率只有 4–10 分之一。

两个优化，都来自 Dao et al.，把前沿推理从"慢"推到了"快"：

1. **KV cache。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力就是一个 query 对缓存 key 的查询。推理从每步 `O(N²)` 降到 `O(N)`。
2. **Flash Attention。** 将注意力计算分块，使完整的 N×N 矩阵永远不碰 HBM。所有 softmax + 矩阵乘法都在 SRAM 中完成。A100 上 2–4 倍实际加速；H100 上用 FP8 可达 5–10 倍。

到 2026 年两者都是标配。每个生产推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都假设它们存在。每个前沿模型都启用了 Flash Attention。

## 概念

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV cache 数学

每个 decoder 层，每个 token，每个 head：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

对于一个 7B 模型，32 层，32 头，d_head=128，fp16：

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

对于 Llama 3 70B（80 层，d_head=128，GQA 8 个 KV head）：

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

这 10 GB 就是为什么 Llama 3 70B 在 128K 上下文时，batch size 1 光 KV cache 就需要 40 GB A100 的大部分空间。

**GQA 是 KV cache 的胜利。** MHA 64 头会是 32 GB。MLA 压缩得更多。

### Flash Attention — 分块技巧

标准注意力：

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

三次 HBM 往返。H100 上 HBM 带宽 3 TB/s；SRAM 是 30 TB/s。每次 HBM 往返相比片上计算慢 10 倍。

Flash Attention：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每个 tile 一次 HBM 往返。总内存占用从 `O(N²)` 降到 `O(N)`。反向传播从前向传播重新计算一些值而不是存储它们——又一个内存节省。

**数值技巧。** Running softmax 跨 tile 维护 `(max, sum)`，所以最终归一化是精确的。不是近似——Flash Attention 计算出与标准注意力 bit-identical 的输出（模 fp16 非结合性）。

**版本演进：**

| Version | Year | Key change | Speedup on reference hardware |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | Tiled SRAM kernel | 2× on A100 |
| Flash 2 | 2023 | Better parallelism, causal-first ordering | 3× on A100 |
| Flash 3 | 2024 | Hopper asynchrony, FP8 | 1.5–2× on H100 (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5-stage pipeline, software exp2 | Inference-first (forward only initially) |

Flash 4 发布时只支持前向传播。训练仍用 Flash 3。Flash 4 的 GQA 和 varlen 支持待定（2026 年中）。

### Speculative decoding — 另一个延迟优化

便宜模型提议 N 个 token。大模型并行验证所有 N 个。如果验证接受了 k 个 token，你只付了 1 次大模型前向传播换来 k 次生成。代码和散文上典型 k=3–5。

2026 年默认方案：
- **EAGLE 2 / Medusa。** 集成 draft head，共享验证器的隐藏状态。2–3 倍加速，无质量损失。
- **带 draft 模型的 speculative decoding。** 消费级硬件上 2–4 倍加速。
- **Lookahead decoding。** Jacobi 迭代；不需要 draft 模型。小众但免费。

### 连续批处理

经典批推理：等最慢的序列完成，然后开始新批次。短回复先完成时浪费 GPU。

连续批处理（首先在 Orca 中实现，现在在 vLLM、TensorRT-LLM、SGLang 中）：旧请求完成后立即将新请求换入批次。典型聊天工作负载 5–10 倍吞吐量提升。

### PagedAttention — KV cache 作为虚拟内存

vLLM 的核心特性。KV cache 以 16-token 块分配；页表将逻辑位置映射到物理块。让你可以跨并行采样（beam search、并行采样）共享 KV，热交换前缀用于 prompt 缓存，以及内存碎片整理。比朴素连续分配 4 倍吞吐量提升。

## 动手构建

见 `code/main.py`。我们实现：

1. 一个朴素的 `O(N²)` 增量 decoder。
2. 一个 `O(N)` 的 KV cache decoder。
3. 一个模拟 Flash Attention running-max 算法的分块 softmax。

### 第 1 步：KV cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：逐 token 增长每层、每头的 K、V 向量列表。

### 第 2 步：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

与 `softmax(qK) V` 一次性计算 bit-identical 的输出，但任何时刻工作集只是一个 `tile × d_head` 块，而非完整的 `N × d_head`。

### 第 3 步：对比朴素 vs 缓存解码（100 token 生成）

计算注意力操作次数。朴素：`O(N²)` = 5050。缓存：`O(N)` = 100。代码会打印两者。

## 使用方式

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM 生产部署：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的前缀缓存是 2026 年的一大优势——相同的系统 prompt、few-shot 示例或长上下文文档跨调用复用 KV。对于带重复工具 prompt 的 agent 工作负载，前缀缓存通常带来 5 倍吞吐量提升。

## 交付产出

见 `outputs/skill-inference-optimizer.md`。该 skill 为新推理部署选择注意力实现、KV cache 策略、量化和 speculative decoding。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素和缓存 decoder 产生相同输出；注意操作次数差异。
2. **中等。** 实现前缀缓存：给定一个 prompt P 和多个补全，对 P 运行一次前向传播填充 KV cache，然后按补全分支。测量相比为每个补全重新编码 P 的加速。
3. **困难。** 实现一个玩具 PagedAttention：KV cache 用固定 16-token 块加空闲列表。当一个序列完成时，将其块归还池。模拟 1,000 次不同长度的聊天补全。对比与连续分配的内存碎片化。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| KV cache | "The trick that makes decoding fast" | Stored K and V from every prefix token; new queries attend to them instead of recomputing. |
| HBM | "GPU main memory" | High Bandwidth Memory; 80 GB on H100, 192 GB on B200. ~3 TB/s bandwidth. |
| SRAM | "On-chip memory" | Per-SM fast memory, ~256 KB per SM on H100. ~30 TB/s bandwidth. |
| Flash Attention | "Tiled attention kernel" | Computes attention without materializing N×N in HBM. |
| Continuous batching | "No-wait batching" | Swap finished sequences out, new ones in, without draining the batch. |
| PagedAttention | "vLLM's headline" | KV cache allocated in fixed blocks with a page table; eliminates fragmentation. |
| Prefix caching | "Reuse long prompts" | Cache KV for a shared prefix across requests; major cost cut for agents. |
| Speculative decoding | "Draft + verify" | Cheap draft model proposes tokens; big model verifies k in one pass. |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5-stage pipeline and the software-exp2 trick; read the repo README for the forward-only launch caveats this lesson mentions.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM paper.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — spec decoding.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1/2 paper for the integrated-draft approach the lesson cites.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — the Medusa approach referenced alongside EAGLE.
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — the canonical deep dive on the 16-token block and page-table design.
