# 智能体经济、Token 激励与声誉

> 长期自主智能体（METR 的 1 小时到 8 小时工作曲线）需要经济代理能力。新兴的 **5 层栈**是：**DePIN**（物理计算）→ **Identity**（W3C DIDs + 声誉资本）→ **Cognition**（RAG + MCP）→ **Settlement**（账户抽象）→ **Governance**（Agentic DAOs）。生产级智能体激励网络包括 **Bittensor**（TAO 子网奖励任务特定模型）、**Fetch.ai / ASI Alliance**（ASI-1 Mini LLM + FET token）和 **Gonka**（基于 transformer 的 PoW，将计算重新分配到生产性 AI 任务）。学术工作：AAMAS 2025 的去中心化 LaMAS 使用 **Shapley 值信用归因**公平奖励贡献智能体；Google Research "Mechanism design for large language models" 提出在单调聚合下使用第二价格支付的 **token 拍卖**。本课构建一个最小智能体市场，对多智能体流水线应用 Shapley 值信用归因，并运行第二价格 token 拍卖，使博弈论机制具体落地。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 16 (Negotiation and Bargaining), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## 问题

当智能体联合产出价值但需要单独奖励时，多智能体系统变得复杂。经典机制——平均分配、最后贡献者全拿——不公平或可被博弈。通过 Shapley 值的联盟奖励在构造上是公平的，但计算昂贵。2025-2026 年的文献推动了有用的近似：Shapley 采样、单调聚合拍卖和从确认贡献中积累的链上声誉。

超越信用归因，该领域已转向实际的经济智能体：Bittensor TAO 奖励挖矿计算来微调子网特定模型，Fetch.ai/ASI 用 FET token 奖励 ASI-1 Mini LLM 使用，Gonka 将 transformer 工作量证明重新分配到生产性 AI 任务。自主交易的智能体今天已经存在；问题是如何对齐激励。

本课将智能体经济视为一个特定问题族——信用归因、机制设计和声誉——并用最少的数学构建每个，使想法扎根。

## 概念

### 5 层智能体经济栈

1. **DePIN（物理计算）。** 出租 GPU、存储、带宽的去中心化基础设施。Bittensor 子网、Render Network、Akash。不是智能体特定的；智能体使用它。
2. **Identity。** W3C 去中心化标识符（DIDs）给每个智能体一个独立于任何平台的持久 ID。声誉积累到 DID。Agent Network Protocol（ANP）使用 DID 作为发现层。
3. **Cognition。** 智能体的推理循环：LLM + RAG + MCP。这是其他阶段构建的。
4. **Settlement。** 账户抽象（ERC-4337）让智能体从自己的余额支付 gas 而无需持有 ETH。智能体可以为服务、彼此或计算付费。
5. **Governance。** Agentic DAOs：人类*和*智能体对协议变更投票的治理结构，投票权与声誉挂钩。

不是每个生产系统都使用全部五层。Bittensor 使用 1、2、部分 3、部分 4、不用 5。OpenAI 智能体除了 3 什么都不用。这个栈是参考地图，不是要求。

### Bittensor, Fetch.ai, Gonka — 运行中的系统

**Bittensor（TAO）。** 子网是专业化任务（语言建模、图像生成、预测）。矿工提交模型输出。验证者排名；质押加权评分分配 TAO 奖励。每个子网有自己的评估。经济教训：为任务特定的输出质量付费，而非使用的计算。

**Fetch.ai / ASI Alliance。** ASI-1 Mini LLM 在 Fetch.ai 网络上运行；用户用 FET token 支付推理。智能体即对等体的叙事在这里更强：Fetch 上的智能体可以调用另一个执行任务并用 FET 支付。

**Gonka。** Transformer 工作量证明："工作"是 transformer 的前向传播。矿工通过运行有已知正确输出（来自训练数据）的推理任务来赚取。资源生产性 PoW 而非基于哈希的 PoW。

截至 2026 年 4 月，三者都是生产级的。收益分配不同。Bittensor 奖励相对于子网验证者的质量；Fetch 奖励由付费用户衡量的效用；Gonka 奖励可验证的推理工作。

### Shapley 值信用归因

三个智能体协作完成一个任务。输出得分 0.8。谁贡献了什么？

Shapley 值：满足四个公理（效率、对称、线性、空值）的唯一信用分配。对于智能体 `i`：

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

其中 `S_i_O` 是排列 `O` 中 `i` 之前的智能体集合。实践中：枚举所有排列，记录每个智能体在每个排列中的边际贡献，取平均。

对于 N=3 个智能体，有 6 个排列。对于 N=10，360 万——所以实践中采样排列而非枚举。

### 第二价格拍卖用于聚合

Google Research（"Mechanism design for large language models"）提出用于聚合 LLM 输出的第二价格 token 拍卖。设置：N 个智能体各提出一个补全；每个对被选中有私有价值。拍卖师选择最高价值提案并支付*第二高*价值。在单调聚合下（价值取决于选择哪个提案，而非出价了多少），这是真实的——智能体出价其真实价值。

为什么这对 LLM 系统重要：你可以将补全任务外包给多个不同定价的智能体；拍卖选出最好的 + 公平支付，智能体没有动机虚报。

### 声誉资本

绑定到 DID 的声誉分数从确认的贡献中积累。简单更新规则：

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

衰减因子 `alpha` 接近 1。声誉：

- 对路由决策读取成本低（"将困难任务发送给高声誉智能体"）。
- 伪造成本高（随时间积累，绑定到 DID）。
- 可以被削减：未通过验证的贡献扣分。

### AAMAS 2025 去中心化 LaMAS

LaMAS 提案（AAMAS 2025）结合：DID 身份、Shapley 值信用归因和简单拍卖机制。关键主张：去中心化信用归因步骤使系统可审计且免疫单点操纵。

### 经济学崩溃的场景

- **价格预言机操纵。** 如果信用函数可以被博弈，智能体就会博弈它。每个机制都需要对抗测试。
- **Sybil 攻击。** 一个运营者启动 N 个假智能体来膨胀自己的贡献。DIDs 减缓但不阻止这一点；声誉伪造成本是缓解措施。
- **验证成本。** 信用归因只有验证者公平才公平。如果验证便宜（小 LLM），可以被博弈；如果昂贵（人类面板），系统不可扩展。
- **监管悬而未决。** 智能体经济与金融监管交叉。截至 2026 年，Bittensor、Fetch 和 Gonka 在某些司法管辖区都在法律灰色地带运营。

### 智能体经济有意义的场景

- **有异构运营者的开放网络。** 没有单一团队控制所有智能体。
- **可验证的输出。** 没有验证，信用归因是猜测。
- **长期工作流。** 一次性任务不受益于声誉积累。
- **代币化支付在你的司法管辖区合法可行。**

在封闭的企业系统中，经济学让位于更简单的分配（管理者分配工作，指标是内部的）。经济学文献主要适用于开放网络。

## 动手构建

`code/main.py` 实现：

- `shapley(value_fn, agents)` — 对小 N 通过枚举的精确 Shapley 计算。
- `second_price_auction(bids)` — 真实机制；赢家支付第二高出价。
- `Reputation` — 绑定 DID 的声誉，带指数衰减和削减。
- 演示 1：三个智能体协作，精确 Shapley 归因信用。
- 演示 2：五个智能体竞标一个任务槽位；第二价格拍卖选出赢家 + 支付。
- 演示 3：100 轮任务分配给异构声誉的智能体；声誉加权路由在预热后比随机好。

运行：

```
python3 code/main.py
```

预期输出：每个智能体的 Shapley 值；拍卖结果显示真实出价均衡；声誉加权路由在预热后显示比随机高 10-20% 的质量增益。

## 使用方式

`outputs/skill-economy-designer.md` 设计一个最小智能体经济：身份层选择、信用归因机制、支付机制、声誉规则。

## 上线清单

2026 年运行智能体经济：

- **从声誉开始，不是 token。** 声誉实现便宜且单独有价值；token 增加法律和经济复杂性。
- **奖励前先验证。** 永远不要在没有独立验证步骤的情况下分配信用。自报质量积累 sybil 博弈。
- **Shapley 采样，不是 Shapley 精确。** 采样 100-1000 个排列；精确枚举不可扩展。
- **限制衰减因子并设声誉下限。** 无界衰减抹去合法贡献者；太慢的衰减奖励过时的高声誉智能体。
- **对抗性审计机制。** 在开放网络前运行红队场景。每个机制都有博弈论；你想找到漏洞，而不是攻击者。

## 练习

1. 运行 `code/main.py`。确认 Shapley 值之和等于总价值（效率公理）。改变价值函数；Shapley 分配是否按预期方向变化？
2. 实现 Shapley *采样*（对 K 个排列的蒙特卡洛）。K 如何影响近似精度？与 N=4 的精确值比较。
3. 在拍卖前实现联盟形成步骤：智能体可以合并为团队并作为一个单位出价。哪些联盟形成？结果是否比个体出价帕累托更优？
4. 阅读 Google Research 机制设计文章。识别一个如果被违反就破坏真实性的假设。在 LLM 设置中那个失败模式看起来怎样？
5. 阅读 AAMAS 2025 去中心化 LaMAS 论文。在合成任务上对 10 个智能体实现他们的 Shapley 步骤。精确计算需要多长时间？100 次抽样能接近到什么程度？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| DePIN | "去中心化物理基础设施" | Token 激励的计算/存储/带宽。Bittensor, Akash, Render。 |
| DID | "去中心化标识符" | W3C 规范的可移植 ID。智能体声誉绑定到 DID，而非平台。 |
| ERC-4337 | "账户抽象" | 可以赞助 gas 的合约账户，使智能体支付成为可能。 |
| Shapley value | "公平信用归因" | 满足效率、对称、线性、空值的唯一分配。 |
| Second-price auction | "Vickrey 拍卖" | 真实机制：赢家支付第二高出价。与单调聚合兼容。 |
| Reputation capital | "累积质量分数" | 从确认贡献中绑定 DID 的分数；随时间衰减。 |
| Agentic DAO | "智能体 + 人类治理" | 智能体投票者作为一等公民的 DAO，投票权与声誉挂钩。 |
| TAO / FET / GPU credits | "Token 面额" | Bittensor TAO, Fetch.ai FET, 各种 DePIN token。 |

## 延伸阅读

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 2026 年 5 层智能体经济栈综述
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — 带单调聚合的 token 拍卖
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Shapley 值信用归因
- [Bittensor TAO documentation](https://docs.bittensor.com/) — 子网结构和奖励分配
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM 和 FET token
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — 身份基础
