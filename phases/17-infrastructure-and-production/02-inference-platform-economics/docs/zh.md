# 推理平台经济学 — Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的推理市场不再是 GPU 时间租赁。它分化为定制芯片（Groq、Cerebras、SambaNova）、GPU 平台（Baseten、Together、Fireworks、Modal）和 API 优先市场（Replicate、DeepInfra）。Fireworks 在 2026 年 5 月 1 日每 GPU 涨价 $1/hr，$4B 估值加上每天 10T+ token 处理量说明走量模式行得通。Baseten 在 2026 年 1 月以 $5B 估值完成 $300M E 轮。竞争定位规则很简单：Fireworks 优化延迟，Together 优化目录广度，Baseten 优化企业精致度，Modal 优化 Python 原生开发体验，Replicate 优化多模态覆盖，Anyscale 优化分布式 Python。这节课给你一个可以直接交给创始人的矩阵。

**Type:** Learn
**Languages:** Python (stdlib, toy per-call economics comparator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## 学习目标

- 说出三个市场细分（定制芯片、GPU 平台、API 优先）并将每个供应商映射到对应细分。
- 解释为什么"按 token"API 定价模型向推理引擎的成本曲线压缩，而非硬件的。
- 计算至少三个供应商的有效每请求成本，并解释什么时候按分钟（Baseten、Modal）胜过按 token。
- 识别给定工作负载（serverless 突发、稳定高吞吐、微调变体、多模态）的正确默认平台。

## 问题

你评估了托管云厂商平台。你决定需要一个更窄、更快的供应商 — Fireworks 要延迟，Together 要广度，Baseten 要微调自定义模型。现在你有六个真实选择，定价页面对不上。Fireworks 显示 $/M tokens；Baseten 显示 $/minute；Modal 显示 $/second；Replicate 显示 $/prediction。不建模工作负载就无法直接对比。

更糟的是，每个定价页面背后的商业模式不同。Fireworks 在共享 GPU 上运行自己的定制引擎（FireAttention）；按 token 费率反映的是他们的利用率曲线。Baseten 给你 Truss + 专属 GPU；按分钟反映的是独占性。Modal 是真正的 Python serverless — 按秒计费加亚秒级冷启动。同样的输出（一个 LLM 响应），三种不同的成本函数。

这节课建模六家并告诉你每家什么时候赢。

## 概念

### 三个细分

**定制芯片** — Groq (LPU)、Cerebras (WSE)、SambaNova (RDU)。在同一模型上通常比 GPU 集群快 5-10x decode。更高的按 token 价格（Groq 在 2025 年底 Llama-70B 上约 $0.99/M）但在延迟敏感场景无可匹敌。Groq 是语音 agent 和实时翻译的生产首选。

**GPU 平台** — Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（H100、H200、2026 年的 B200）或有时 AMD 上。介于"裸 GPU 租赁"（RunPod、Lambda）和"云厂商托管服务"（Bedrock）之间的经济层。

**API 优先市场** — Replicate、DeepInfra、OpenRouter、Fal。广泛目录，按预测或按秒付费，强调首次调用时间。

### Fireworks — 延迟优化的 GPU 平台

- FireAttention 引擎（定制）；宣称在等效配置上比 vLLM 低 4x 延迟。
- Batch tier 约为 serverless 费率的 50%，用于非交互工作负载。
- 微调模型以基础模型相同费率服务 — 相比对 LoRA 收取溢价的供应商，这是真正的差异化。
- 2026 年中：按需 GPU 租赁自 2026 年 5 月 1 日起涨 $1/hour。大规模可协商量价。
- 财务信号：$4B 估值，每天处理 10T+ token。

### Together — 广度优化

- 200+ 模型，包括上游发布后数天内的开源版本。
- 在等效 LLM 模型上比 Replicate 便宜 50-70% — "AI Native Cloud" 定位是走量和目录。
- 推理 + 微调 + 训练在一个 API 中。

### Baseten — 企业精致度优化

- Truss 框架：模型打包含依赖、密钥、serving 配置在一个 manifest 中。
- GPU 范围从 T4 到 B200。按分钟计费加合理的冷启动缓解。
- SOC 2 Type II、HIPAA-ready。常见的金融科技和医疗选择。
- $5B 估值，2026 年 1 月 E 轮（$300M，来自 CapitalG、IVP、NVIDIA）。

### Modal — Python 原生优化

- 纯 Python 的基础设施即代码。用 `@modal.function(gpu="A100")` 装饰一个函数，一条命令部署。
- 按秒计费。冷启动 2-4s 带预热；小模型 <1s。
- $87M B 轮，$1.1B 估值（2025）。独立调查中开发者体验评分最高。

### Replicate — 多模态广度

- 按预测付费。图像、视频和音频模型的默认平台。
- 集成生态（Zapier、Vercel、CMS 插件）。
- LLM 按 token 费率竞争力较弱但在多模态多样性上胜出。

### Anyscale — Ray 原生

- 基于 Ray 构建；RayTurbo 是 Anyscale 的专有推理引擎（与 vLLM 竞争）。
- 最适合分布式 Python 工作负载，其中推理步骤是更大图中的一个节点。
- 托管 Ray 集群；与 Ray AIR 和 Ray Serve 紧密集成。

### 按 token vs 按分钟 — 各自什么时候赢

按 token 在工作负载延迟不敏感且突发时有意义 — 你只为使用的付费。按分钟在利用率高且可预测时有意义 — 一旦你把 GPU 跑满就能胜过按 token。

粗略规则：对于超过约 30% 持续利用率的专属 GPU 工作负载，按分钟（Baseten、Modal）开始胜过按 token（Fireworks、Together）。低于此值，按 token 赢因为你避免了为空闲付费。

### 定制引擎是真正的护城河

上面每个平台都在 vLLM 和 SGLang 之上声称有定制引擎。FireAttention、RayTurbo、Baseten 的推理栈。定制引擎声称偏营销 — 诚实的说法是 vLLM + SGLang 代表了约 80% 的生产开源推理，平台层的差异化在于开发体验、归因和 SLA。

### 你应该记住的数字

- Fireworks GPU 租赁：2026 年 5 月 1 日起涨 $1/hr。
- Fireworks 声称：在等效配置上比 vLLM 低 4x 延迟。
- Together：在 LLM 上比 Replicate 便宜 50-70%。
- Baseten 估值：$5B（E 轮，2026 年 1 月，$300M 轮次）。
- Modal 估值：$1.1B（B 轮，2025）。
- 按分钟在约 30% 持续利用率以上胜过按 token。

## Use It

`code/main.py` 在合成工作负载上跨定价模型比较六家供应商。报告 $/day 和有效 $/M tokens。运行它找到按 token 和按分钟之间的盈亏平衡点。

## Ship It

本课产出 `outputs/skill-inference-platform-picker.md`。给定工作负载画像、SLA 和预算，选出主推理平台并指出备选。

## 练习

1. 运行 `code/main.py`。在什么持续利用率下 Baseten（按分钟）对一台 H100 上的 70B 模型胜过 Fireworks（按 token）？自己推导交叉点并与经验法则对比。
2. 你的产品服务图像生成加聊天加语音转文字。为每种模态选平台并说出统一它们的网关模式。
3. Fireworks 对你的主模型涨价 $1/hr。如果 40% 流量转到 batch tier（5 折），建模混合成本影响。
4. 一个受监管客户要求 SOC 2 Type II + HIPAA + 专属 GPU。哪三个平台可行，哪个在 FinOps 上胜出？
5. 比较 Llama 3.1 70B 在 Fireworks serverless、Together 按需、Baseten 专属和 Replicate API 上每 1,000 次预测的成本。10 次/天时哪个最便宜？10,000 次时呢？

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|---------|---------|
| Custom silicon | "非 GPU 芯片" | Groq LPU、Cerebras WSE、SambaNova RDU — 为 decode 优化 |
| FireAttention | "Fireworks 引擎" | 定制 attention kernel；宣称比 vLLM 低 4x 延迟 |
| Truss | "Baseten 的格式" | 模型打包 manifest；依赖 + 密钥 + serving 配置 |
| Per-token | "API 定价" | 按消耗 token 收费；不为空闲付费 |
| Per-minute | "专属定价" | 按 GPU 挂钟时间收费；高利用率时胜出 |
| Per-prediction | "Replicate 定价" | 按模型调用收费；常见于图像/视频 |
| RayTurbo | "Anyscale 引擎" | Ray 上的专有推理；在 Ray 集群上与 vLLM 竞争 |
| Batch tier | "5 折" | 非交互队列以降低费率；常见于 Fireworks、OpenAI |
| Fine-tuned at base rate | "Fireworks LoRA" | LoRA 服务的请求按基础模型费率收费（差异化） |

## 延伸阅读

- [Fireworks Pricing](https://fireworks.ai/pricing) — 按 token 费率、batch tier、GPU 租赁。
- [Baseten Pricing](https://www.baseten.co/pricing/) — 按分钟费率、承诺容量、企业层级。
- [Modal Pricing](https://modal.com/pricing) — 按秒 GPU 费率和免费层。
- [Together AI Pricing](https://www.together.ai/pricing) — 模型目录和按 token 费率。
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo 和托管 Ray 定价。
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 对比评估。
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商全景。
