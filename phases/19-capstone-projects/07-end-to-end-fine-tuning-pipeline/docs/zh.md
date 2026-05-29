# 毕业项目 07 — 端到端 Fine-Tuning Pipeline（数据 → SFT → DPO → 部署）

> 一个用你自己数据训练的 8B 模型，用你自己偏好做 DPO 对齐，量化，speculative decoding，以可衡量的 $/1M tokens 提供服务。2026 年的开源栈是 Axolotl v0.8、TRL 0.15、Unsloth 用于迭代、GPTQ/AWQ/GGUF 用于量化、vLLM 0.7 带 EAGLE-3 用于服务。这个毕业项目要求你可复现地运行整个管道——YAML 输入，服务端点输出——并在 2026 Model Openness Framework 下发布模型卡。

**类型：** 毕业项目
**语言：** Python（管道），YAML（配置），Bash（脚本）
**前置要求：** Phase 2（ML）、Phase 3（DL）、Phase 7（Transformer）、Phase 10（从零构建 LLM）、Phase 11（LLM 工程）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段：** P2 · P3 · P7 · P10 · P11 · P17 · P18
**时间：** 35 小时

## 问题

2026 年每个认真的 AI 团队都备有一个 fine-tuning pipeline。不是因为他们发布前沿基础模型，而是因为下游适配——领域 SFT、对标注偏好做 DPO、为 speculative decoding 蒸馏 draft、用 EAGLE-3 服务——才是可衡量收益所在。Axolotl v0.8 处理多 GPU SFT 配置。TRL 0.15 处理 DPO 和 GRPO。Unsloth 让你快速单 GPU 迭代。vLLM 0.7 带 EAGLE-3 将解码吞吐量提升 2-3 倍而不损失质量。工具好用；技艺在于 YAML、数据卫生和评估纪律。

你将把一个 8B 基础模型（Llama 3.3、Qwen3 或 Gemma 3）通过 SFT 然后 DPO 在任务特定数据上训练，量化用于服务，并对照 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 和 MMLU-Pro 衡量增益。你将在 2026 Model Openness Framework 下产出模型卡。重点是可复现性——一条命令端到端重跑整个管道。

## 概念

管道有五个阶段。**数据**：去重（MinHash / Datatrove）、质量过滤（Nemotron-CC 风格分类器）、PII 清洗、对公开基准的 split 卫生检查。**SFT**：Axolotl YAML，8xH100 上 ZeRO-3，cosine schedule，packed sequences，2-3 epochs。**DPO 或 GRPO**：TRL 配置，1 epoch，偏好对来自人工标注或模型判断，beta 调优。**量化**：GPTQ + AWQ + GGUF 用于部署灵活性。**服务**：vLLM 0.7 带 EAGLE-3 speculative heads（或 SGLang 带 SpecForge），K8s 部署，HPA 基于 queue-wait。

消融实验是交付物：SFT-only vs SFT+DPO vs SFT+GRPO 在三个任务特定基准上。服务指标：batch 1 / 8 / 32 时的 tokens/s、EAGLE-3 接受率、$/1M tokens。安全评估：Llama Guard 4 通过率。模型卡：偏见评估、可复现性种子、数据许可。

## 架构

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技术栈

- 数据：Datatrove 用于去重，Nemotron-CC 分类器用于质量，Presidio 用于 PII
- 基础模型：Llama 3.3 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8 带 ZeRO-3、Flash Attention 3、packed sequences
- 偏好调优：TRL 0.15 用于 DPO 或 GRPO；Unsloth 用于单 GPU 迭代
- 量化：GPTQ（Marlin）、AWQ、GGUF via llama.cpp
- 服务：vLLM 0.7 带 EAGLE-3 speculative decoding（或 SGLang 0.4 + SpecForge）
- 评估：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全评估：Llama Guard 4、ShieldGemma-2
- 基础设施：Kubernetes + NVIDIA device plugin，HPA 基于 queue-wait 指标
- 可观测性：W&B 用于训练，Langfuse 用于推理

## 构建步骤

1. **数据管道。** 对原始语料运行 Datatrove 去重。应用 Nemotron-CC 风格质量分类器。Presidio 清洗 PII。用显式种子写入 train/val 分割。

2. **污染检查。** 对每个验证分割，计算与 MMLU-Pro、MT-Bench-v2、RewardBench-2 测试集的 MinHash。拒绝任何重叠。

3. **Axolotl SFT。** YAML 带 ZeRO-3、FA3、sequence packing。8xH100 上 2-3 epochs。日志记录到 W&B。

4. **TRL DPO / GRPO。** 取 SFT checkpoint，在偏好对上运行一个 epoch 的 DPO（或带可验证奖励的 GRPO 用于数学/代码）。扫描 beta。

5. **量化。** 产出三种量化：GPTQ-INT4-Marlin、AWQ-INT4、GGUF-Q4_K_M 用于 llama.cpp。记录大小和标称吞吐量。

6. **带 speculative decoding 服务。** vLLM 0.7 配置带通过 Red Hat Speculators 训练的 EAGLE-3 draft heads。衡量 batch 1 / 8 / 32 时的接受率和尾延迟。报告与 Anthropic / OpenAI 在相同评估上的 $/1M tokens 对比。

7. **评估矩阵。** 在 base、SFT-only、SFT+DPO、SFT+GRPO 上运行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。产出表格。

8. **安全评估。** dev set 上的 Llama Guard 4 通过率。ShieldGemma-2 输出过滤器。

9. **模型卡。** MOF 2026 模板：数据、训练、评估、安全、许可、带 YAML 和 commit SHA 的可复现性部分。

## 使用示例

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 交付标准

`outputs/skill-finetuning-pipeline.md` 描述交付物。一条命令将数据通过 SFT 通过 DPO 通过量化通过服务通过评估，输出模型卡 + 服务端点。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 对比基础模型的评估增量 | 目标任务上的衡量增益（MMLU-Pro、MT-Bench-v2、任务特定） |
| 20 | 管道可复现性 | 一条命令用相同种子端到端重跑 |
| 20 | 数据卫生 | 去重率、PII 清洗覆盖率、污染检查通过 |
| 20 | 服务效率 | bs=1/8/32 时的 tokens/s、EAGLE-3 接受率、$/1M tokens |
| 15 | 模型卡 + 安全评估 | 2026 MOF 完整性 + Llama Guard 4 通过率 |
| **100** | | |

## 练习

1. 在同一任务特定基准上运行 SFT-only vs SFT+DPO vs SFT+GRPO。报告哪种偏好方法胜出以及胜出多少。

2. 将 Llama 3.3 8B 换成 Qwen3 14B。在匹配质量下衡量 $/1M tokens。

3. 衡量 EAGLE-3 在领域数据 vs 通用 ShareGPT 上的接受率。报告差异及其对延迟预算的意义。

4. 注入 1% 污染（将 MMLU-Pro 答案泄露到训练数据中）并重跑评估。观察 MMLU-Pro 准确率不切实际地跳升。构建一个能捕获此问题的污染检查 CI 门。

5. 添加 LoRA SFT 作为全量 fine-tune 的替代。在 10 倍更低内存下衡量质量差距。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Axolotl | "SFT 训练器" | 统一的 YAML 驱动训练器，用于 SFT、DPO 和蒸馏 |
| TRL | "偏好调优器" | Hugging Face 库，用于 LLM 上的 DPO、GRPO、PPO |
| GRPO | "Group-relative policy optimization" | DeepSeek R1 的 RL 方案，带可验证奖励 |
| EAGLE-3 | "Speculative decoding draft" | 在目标模型隐藏状态上训练的 draft heads，预测 N 个 token；vLLM 用目标模型验证 |
| MOF | "Model Openness Framework" | 2026 年对模型发布在数据、代码、许可方面评级的标准 |
| 污染检查 | "Split 卫生" | 基于 MinHash 的测试集泄露到训练中的检测 |
| 接受率 | "EAGLE / MTP 指标" | 目标模型接受的 draft token 比例 |

## 延伸阅读

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — 参考 SFT / DPO 训练器
- [TRL documentation](https://huggingface.co/docs/trl) — DPO 和 GRPO 参考实现
- [Unsloth](https://github.com/unslothai/unsloth) — 单 GPU 迭代参考
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO 方法论
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — 参考服务栈
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 备选 speculative-decoding 训练器
- [Model Openness Framework 2026](https://isocpp.org/) — 开放发布评级标准
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 标准评估运行器
