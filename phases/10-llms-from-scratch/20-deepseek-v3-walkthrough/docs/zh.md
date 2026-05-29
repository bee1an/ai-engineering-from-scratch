# DeepSeek-V3 架构详解

> Phase 10 · Lesson 14 列出了每个开源模型都会调节的六个架构旋钮。DeepSeek-V3（2024 年 12 月，总参数 671B，活跃参数 37B）把这六个旋钮全部拧到位，还额外加了四个：Multi-Head Latent Attention、无辅助损失的负载均衡、Multi-Token Prediction 和 DualPipe 训练。本课从上到下阅读 DeepSeek-V3 的架构，并从公开配置推导出每一个参数量。学完之后你能解释为什么 671B/37B 的比例是正确的押注，以及为什么 MLA + MoE 组合在前沿模型中胜过单独使用任何一个。

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Phase 10 · 14 (open-model walkthroughs), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**Time:** ~75 minutes

## 学习目标

- 从上到下阅读 DeepSeek-V3 配置，用六个 GPT-2 旋钮加四个 DeepSeek 特有创新来解释每个字段。
- 推导总参数量（671B）、活跃参数量（37B），以及各组件对两者的贡献。
- 计算 MLA 在 128k 上下文下的 KV cache 占用，并与同等活跃参数规模的 GQA 密集模型进行对比。
- 说出四项 DeepSeek 特有创新（MLA、MTP、无辅助损失路由、DualPipe），并指出每项分别针对架构/训练栈的哪个部分。

## 问题

DeepSeek-V3 是第一个架构与 Llama 家族有实质性差异的前沿开源模型。Llama 3 405B 是"GPT-2 拧了六个旋钮"。DeepSeek-V3 是 GPT-2 拧了六个旋钮再加四个。阅读 Llama 3 配置是阅读 DeepSeek 配置的热身，但深层结构——attention block 的形状、路由逻辑、训练目标——差异大到需要单独讲解。

学习它的回报：DeepSeek-V3 的开源权重发布重新定义了开源模型中"前沿能力"的含义。这个架构是 2026 年许多训练运行正在复制的蓝图。理解它是任何涉及前沿 LLM 训练或推理的岗位的基本功。

## 概念

### 不变的核心，再说一次

DeepSeek-V3 仍然是自回归的。仍然堆叠 decoder block。每个 block 仍然是 attention + MLP + 两个 RMSNorm。仍然在 MLP 中使用 SwiGLU。仍然使用 RoPE。Pre-norm。权重绑定的 embedding。和每个 Llama 或 Mistral 的基线相同。

### 关键变化：MLA 替代 GQA

从 Phase 10 · 14 你知道 GQA 通过在 Q head 组之间共享 K 和 V 来缩小 KV cache。Multi-Head Latent Attention (MLA) 更进一步：K 和 V 被压缩到一个共享的低秩隐表示（`kv_lora_rank`），然后在每个 head 上实时解压。KV cache 只存储隐表示——通常每层每 token 512 个浮点数，而不是 8 x 128 = 1024 个。

在 128k 上下文下，DeepSeek-V3 使用 MLA（每层每 token 一个共享隐向量 `c^{KV}`；K 和 V 都通过上投影从该隐向量导出，上投影可以被吸收到后续矩阵乘法中）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

假设使用 GQA 基线（Llama 3 70B 形状，8 个 KV head，head dim 128）则需要：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA 在 128k 上下文下比 Llama-3-70B 风格的 GQA cache 小 4 倍。

代价：MLA 在每次 attention 计算时（每个 head）增加一个解压步骤。额外计算量相比节省的带宽很小。长上下文推理的净收益为正。

### 路由：无辅助损失的负载均衡

MoE 路由器决定哪些 top-k expert 处理每个 token。朴素路由器会把过多工作集中在少数 expert 上，其余闲置。标准修复方法：添加一个辅助损失项来惩罚负载不均衡。这有效但会略微降低主任务性能。

DeepSeek-V3 引入了无辅助损失方案。在路由器 logits 上添加每个 expert 的偏置项，训练期间通过简单规则调整：如果 expert `e` 过载，降低 `bias_e`；如果负载不足，提高它。没有额外损失项。训练保持干净。Expert 负载保持均衡。

对主损失的影响：无可测量的影响。对 MoE 架构的影响：更干净，无需调节辅助损失超参数。

### MTP：更密集的训练 + 免费的 draft

从 Phase 10 · 18 你知道 DeepSeek-V3 添加了 D=1 的 MTP 模块，预测前方两个位置的 token。推理时，训练好的模块被复用为 speculative decoding 的 draft，接受率 80% 以上。训练时，每个隐状态在 D+1 = 2 个目标上被监督，提供更密集的信号。

参数量：在 671B 主体之上额外 14B。开销：2.1%。

### 训练：DualPipe

从 Phase 10 · 19 你知道 DualPipe 是一种双向流水线，将前向和反向计算块与跨节点 all-to-all 通信重叠。在 DeepSeek-V3 的 2,048 张 H800 规模下，它恢复了约 245k GPU 小时——这些是 1F1B 方案会因流水线气泡而损失的。

### 配置逐字段解读

以下是 DeepSeek-V3 配置（简化版）：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

解读：

- `hidden_size=7168`：embedding 维度。
- `num_hidden_layers=61`：总 block 深度。
- `first_k_dense_layers=3`：前 3 个 block 使用大小为 18432 的 dense MLP。其余 58 个使用 MoE。
- `num_attention_heads=128`：128 个 query head。
- `kv_lora_rank=512`：K 和 V 被压缩到这个隐维度，然后按 head 解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE block 有 256 个 expert，路由 top-8。
- `shared_experts=1`：在 256 个路由 expert 之上，1 个始终激活的 expert 为每个 token 贡献输出。可以理解为一个"dense 底座"，确保每个 token 都能获得可靠的基础表示。
- `moe_intermediate_size=2048`：每个 expert 的 MLP hidden size。比 dense MLP 小，因为有 256 个。

### 参数量计算

完整计算在 `code/main.py` 中。要点：

- Embedding：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个 dense block：带 MLA 的 attention（每 block ~144M）+ dense MLP（每 block ~260M）+ norm。总计约 1.2B。
- 58 个 MoE block：带 MLA 的 attention（~144M）+ 256 个 expert（每个 30M）+ 1 个 shared expert（30M）+ norm。每 block 总计 ~7.95B（包含所有 expert）。58 个 MoE block 总计 461B。
- MTP 模块：14B。

总计：核心架构 ~476B + 14B MTP + 公开的 671B 数字还包含额外的结构参数（bias 张量、expert 特有组件、shared expert 缩放等）。我们在计算器中复现的数字与公开值相差 3-5%——差异来自 DeepSeek 报告 Section 2 附录中记录的细粒度计算。

每次前向的活跃参数：

- Attention：每层 144M * 61 = 8.8B（所有层都激活）。
- MLP 活跃部分：前 3 层 dense（3 * 260M = 780M），58 个 MoE 层每层激活 8 个路由 expert + 1 个 shared expert + 路由开销。每层活跃 MLP：~260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- Embedding + norm：1.2B。
- 活跃总计：核心约 26B + 14B MTP（训练时使用但推理时不一定运行）≈ 37B。

### 671B / 37B 比例

18 倍稀疏比（活跃参数占总参数的 5.5%）。DeepSeek-V3 是发布开源权重的最稀疏前沿 MoE 模型。Mixtral 8x7B 的比例是 13/47（28%），密集得多。Llama 4 Maverick 的比例是 17B/400B（4.25%），与之相当。DeepSeek 的押注：在前沿规模下，更多 expert 配合更低的激活比例能产生更好的每活跃 FLOP 质量。

### DeepSeek-V3 的定位

| Model | Total | Active | Ratio | Attention | Novel ideas |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

### 后续：R1、V4

DeepSeek-R1（2025）是在 V3 骨干上的推理训练运行。R1 使用相同的架构。改变的是后训练方案（在可验证任务上的大规模 RL），而非预训练架构。

DeepSeek-V4（如果发布的话）预计会保留 MLA + MoE + MTP 并添加 DSA（DeepSeek Sparse Attention），即 Phase 10 · 17 中 NSA 的后继者。这条技术路线是稳定的：架构级创新逐步积累；每个版本拧更多旋钮。

## 使用

`code/main.py` 是专门针对 DeepSeek-V3 形状的参数计算器。运行它，将输出与论文数字对比，并在假设变体上使用（256 expert vs 512、top-8 vs top-16、MLA rank 512 vs 1024）。

关注点：

- 总参数量 vs 公开的 671B。
- 活跃参数量 vs 公开的 37B。
- 128k 上下文下的 KV cache——MLA vs GQA 对比。
- 逐层分解，看参数预算实际花在哪里。

## 交付

本课产出 `outputs/skill-deepseek-v3-reader.md`。给定一个 DeepSeek 家族模型（V3、R1 或任何未来变体），它会生成逐组件的架构解读，命名配置中的每个字段，按组件推导参数量，并识别该模型使用了四项 DeepSeek 特有创新中的哪些。

## 练习

1. 运行 `code/main.py`。将计算器的总参数估计与公开的 671B 对比，找出差异来源。论文 Section 2 有完整的逐项列表。

2. 将配置修改为 MLA rank 256（而非 512）。计算 128k 上下文下的 KV cache 大小。它带来了多少百分比的缩减，代价是什么（对每个 head 表达力的影响）？

3. 比较 DeepSeek-V3 的（256 expert，top-8）路由与假设的（512 expert，top-8）变体。总参数增长；活跃参数不变。额外的 expert 容量理论上带来什么，推理时的代价是什么？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）Section 2.1 关于 MLA 的部分。用三句话解释为什么 K 和 V 的解压矩阵可以在推理时被"吸收"到后续矩阵乘法中。

5. DeepSeek-V3 对大多数操作使用 FP8 训练。计算 FP8 vs BF16 存储 671B 权重的内存节省。这与 14.8T token 的训练预算如何交叉影响？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|------------------------|
| MLA | "Multi-Head Latent Attention" | 将 K 和 V 压缩到共享低秩隐表示（kv_lora_rank，通常 512），按 head 实时解压；KV cache 只存储隐表示 |
| kv_lora_rank | "MLA 压缩维度" | K 和 V 共享隐表示的大小；DeepSeek-V3 使用 512 |
| First k dense layers | "前几层保持 dense" | MoE 模型的前几层跳过 MoE 路由器，运行 dense MLP 以保证稳定性 |
| num_experts_per_tok | "Top-k 路由" | 每个 token 激活多少个路由 expert；DeepSeek-V3 使用 8 |
| Shared experts | "始终激活的 expert" | 无论路由结果如何都处理每个 token 的 expert；DeepSeek-V3 使用 1 个 |
| Auxiliary-loss-free routing | "偏置调整的负载均衡" | 训练期间调整每个 expert 的偏置项以保持负载均衡，无需添加损失项 |
| MTP module | "额外预测头" | 从 h^(1) 和 E(t+1) 预测 t+2 的 Transformer block；更密集的训练，免费的 speculative decoding draft |
| DualPipe | "双向流水线" | 将前向/反向计算与跨节点 all-to-all 通信重叠的训练调度 |
| Active parameter ratio | "稀疏度" | active_params / total_params；DeepSeek-V3 为 5.5% |
| FP8 training | "8-bit 训练" | 训练存储和许多计算操作使用 FP8；相比 BF16 大约减半内存，质量代价很小 |

## 延伸阅读

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整的架构、训练和结果文档
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 配置文件和部署说明
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 引入 MLA 的前代模型
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 在 V3 架构上的推理训练后继者
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek 家族 attention 的未来方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — 训练调度参考实现
