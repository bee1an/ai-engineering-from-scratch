# vLLM Serving 内部机制：PagedAttention、Continuous Batching、Chunked Prefill

> vLLM 在 2026 年的主导地位建立在三个叠加的默认机制上，而非单一技巧。PagedAttention 始终开启。Continuous batching 在 decode 迭代之间将新请求注入活跃 batch。Chunked prefill 切片长 prompt 使 decode token 永远不会饿死。三者全开，一块 H100 SXM5 上的 Llama 3.3 70B FP8 在 128 并发下推到 2,200-2,400 tok/s — 大约比 vLLM 自身默认高 25%，比朴素 PyTorch 循环高 3-4x。这节课在你能画图的层面读调度器和 attention kernel，最后在 `code/main.py` 中实现一个 toy continuous batcher，按 vLLM 的方式调度 prefill 和 decode。

**Type:** Learn
**Languages:** Python (stdlib, toy continuous batching scheduler)
**Prerequisites:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Time:** ~75 minutes

## 学习目标

- 将 PagedAttention 解释为 KV cache 分配器：block、block table，以及为什么碎片在生产负载下保持在 4% 以下。
- 在迭代级别画出 continuous batching：完成的序列如何离开 batch，新序列如何加入而不排空。
- 用一句话描述 chunked prefill 并说出它保护的延迟指标（提示：是 TTFT 尾部，不是平均吞吐量）。
- 说出 2026 年 vLLM v0.18.0 的坑，它会坑到同时开启所有优化的团队。

## 问题

朴素 PyTorch serve 循环一次运行一个请求：tokenize、prefill、decode 直到 EOS、返回。一个用户时没问题。一百个用户时就是一队耐心的人。显而易见的修复 — static batching — 将每个请求 pad 到窗口中最长 prompt，将每个 decode pad 到最长预期输出，整个 batch 在最慢序列上停滞。你为从未使用的 padding 付费，快请求等慢请求。

vLLM 同时解决三个问题。PagedAttention 阻止 KV cache 碎片吃掉 60-80% GPU 内存（经典连续分配就是这样）。Continuous batching 让请求在每次 decode 迭代之间加入和离开 batch，所以 batch 总是充满真实工作。Chunked prefill 将 32k token 的 prompt 切成约 512 token 的片段与 decode 交错，这样长 prompt 不会冻结 GPU 上的每个 decode token。

2026 年的生产默认是三者全开。你需要理解每个做什么，因为故障模式都在调度器上，不在模型上。

## 概念

### PagedAttention 作为虚拟内存系统

KV cache 是每个序列 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。对于 Llama 3.3 70B 在 8192 token 时，BF16 下每个序列大约 1.25 GB。如果你为每个请求预留 8192 个槽位但平均请求只用 1500 token，你浪费了约 82% 预留的 HBM。经典 batching 付出这个浪费。

PagedAttention 借鉴了操作系统虚拟内存的思想。KV cache 不是每个序列连续的。它以固定大小的 block（默认 16 token）分配。每个序列有一个 block table 将其逻辑 token 位置映射到物理 block ID。当序列增长超过已分配的 block 时，再加一个 block。当它完成时，其 block 返回池中。

碎片从 60-80%（经典）降到 4% 以下（PagedAttention）。你不需要用 flag 启用 PagedAttention — 它是 vLLM 唯一的分配器。旋钮是 `--gpu-memory-utilization`（默认 0.9），告诉 vLLM 在加载权重和激活后为 KV block 预留多少 HBM。

### 迭代级别的 Continuous batching

旧的"dynamic batching"等一个窗口（比如 10 ms）填满 batch，然后运行 prefill + decode + decode + decode 直到每个序列完成。快序列提前离开然后空闲等 GPU 完成慢序列。

Continuous batching 在每个 decode 步骤之间操作。将运行中的序列集合称为 `RUNNING` 列表。每次迭代：

1. `RUNNING` 中刚命中 EOS 或 max_tokens 的序列被移除。
2. 调度器查看等待队列。如果有空闲 KV block，它接纳新序列（prefill 或恢复的）。
3. forward pass 在当前 `RUNNING` 中的所有内容上运行，每个序列发出一个新 token。

batch 大小永远不会 pad 到固定数字。不同输出位置的序列共享一次融合 forward。在 2026 年 vLLM 中这叫 `V1 scheduler`。关键不变量：调度器每次 decode 迭代运行一次，不是每个请求运行一次。

### Chunked prefill 保护 TTFT 尾部

Prefill 是计算受限的。一个 32k token 的 prompt 在一块 H100 上的 Llama 3.3 70B 需要约 800 ms 纯 prefill。Prefill 运行时，batch 中其他所有序列的 decode token 都在等。在 serving 循环中，一个长 prompt 的首 token 延迟（TTFT）变成了其他几十个用户的 inter-token 延迟（ITL）抖动。

Chunked prefill 将 prefill 切成固定大小的 chunk（默认 512 token）并将每个 chunk 作为一个单元调度。chunk 之间调度器可以推进 decode 序列一个 token。你用一点绝对 prefill 延迟增加（每个 chunk 几 ms）换来更低的 decode 时间抖动。混合负载下 P99 ITL 从约 50 ms 降到约 15 ms（已发布基准测试）。

### 三个默认机制相互作用

三个特性互相假设对方存在。PagedAttention 给调度器一个细粒度的 KV 资源来交易。Continuous batching 需要那个细粒度资源，这样接纳新序列不会强制全局重排。Chunked prefill 是调度器在同一个 `RUNNING` 列表上做的决策 — 它是又一个调度器策略，不是独立系统。

你不需要知道每个 flag。你需要知道调度器优化什么：在 KV block 预算下的 goodput，受 chunked prefill 切片约束。

### 2026 年 v0.18.0 的坑

在 vLLM v0.18.0 中你不能将 `--enable-chunked-prefill` 与 draft-model speculative decoding（`--speculative-model`）组合。文档中的例外是 V1 scheduler 中的 N-gram GPU speculative decoding。不看 release notes 就把每个 flag 都打开的团队会在启动时得到运行时错误，而不是软回归。如果你的 speculative 收益值得为之启用 chunked prefill，重新审视选择 — 2026 年的正确答案通常是 EAGLE-3 不带 chunked prefill，而不是 draft model 加 chunked prefill 但编译不过。

### 你应该记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三者全开：2,200-2,400 tok/s。
- 同模型，默认 vLLM（无 chunked prefill）：约 1,800 tok/s。
- 同模型，朴素 PyTorch forward 循环：约 600 tok/s。
- PagedAttention 在生产负载下的 KV 碎片浪费：<4%。
- 混合负载下 P99 ITL：有 chunked prefill 约 15 ms，无约 50 ms。

### 调度器长什么样

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 就是这个循环的 stdlib Python 版本，用假 token 计数和假 forward 延迟。运行它看 chunked prefill 如何在长 prefill 期间保持 decode 序列存活。

## Use It

`code/main.py` 模拟一个可切换特性的 vLLM 风格调度器。运行它看：

- `NAIVE` 模式：一次一个请求，无 batching。
- `STATIC` 模式：pad 并等待，经典 batching。
- `CONTINUOUS` 模式：迭代级接纳和释放。
- `CONTINUOUS + CHUNKED` 模式：prefill 切片与 decode 交错。

输出显示总吞吐量（每虚拟秒 token 数）、TTFT 均值和 P99 ITL。`CONTINUOUS + CHUNKED` 行应该在混合流量上占优。

## Ship It

本课产出 `outputs/skill-vllm-scheduler-reader.md`。给定 serving 配置（batch size、KV memory utilization、chunked prefill size、speculative config），产出调度器诊断，指出三个默认机制中哪个是瓶颈以及调什么。

## 练习

1. 运行 `code/main.py`。在混合短长请求的工作负载上比较 `STATIC` 和 `CONTINUOUS`。吞吐量差距来自哪里 — prefill 效率、decode 效率还是尾部延迟？
2. 修改 toy 调度器添加 `--max-num-batched-tokens`。对于运行 Llama 3.3 70B FP8 的 H100，正确值是多少？（提示：它是 KV block 大小和空闲 block 数的函数，不是原始 HBM。）
3. 重读 vLLM v0.18.0 release notes。哪些 flag 组合是互斥的？列出它们。
4. 计算 1,000 个请求的 trace 的 KV cache 碎片浪费，平均 1,500 输出 token，标准差 600 token，在 (a) 8192 max 的连续按请求分配 (b) 16 token block 的 PagedAttention 下。
5. 用一段话解释为什么 chunked prefill 帮助 P99 ITL 但单独不帮助吞吐量。实践中吞吐量提升来自哪里？

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| PagedAttention | "KV 技巧" | KV cache 的固定大小 block 分配器；碎片 <4% |
| Block table | "页表" | 每序列从逻辑 token 位置到物理 KV block 的映射 |
| Continuous batching | "动态 batching，但做对了" | 每次 decode 迭代做接纳/释放决策 |
| Chunked prefill | "prefill 切片" | 将长 prefill 切成 512 token 片段与 decode 交错 |
| TTFT | "首 token 时间" | Prefill + 队列 + 网络；长 prompt 时由 prefill 主导 |
| ITL | "inter-token 延迟" | 连续 decode token 之间的时间；由 batch size 主导 |
| Goodput | "满足 SLO 的吞吐量" | 每个请求仍命中 TTFT 和 ITL 目标的 tokens/sec |
| V1 scheduler | "新调度器" | vLLM 的 2026 调度器；N-gram spec decode 是 chunked-prefill 兼容路径 |
| `--gpu-memory-utilization` | "内存旋钮" | 权重和激活之后为 KV block 预留的 HBM 比例 |

## 延伸阅读

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — chunked-prefill 和 speculative-decoding 兼容性的官方来源。
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 发布节奏和版本特定行为。
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 仍然定义如何思考分配器的原始文章。
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — 碎片分析和调度器设计。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm) — 带火焰图的详细 V1 调度器走读。
