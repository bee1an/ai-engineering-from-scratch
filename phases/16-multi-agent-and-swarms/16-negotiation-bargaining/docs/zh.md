# 谈判与博弈

> 智能体就资源、价格、任务分配和条款进行谈判。2026 年的基准集很明确：NegotiationArena（arXiv:2402.05863）显示 LLM 可以通过人设操纵（"绝望"）将收益提高 ~20%；"Measuring Bargaining Abilities"（arXiv:2402.15813）显示买方比卖方更难且规模无济于事——他们的 **OG-Narrator**（确定性报价生成器 + LLM 叙述者）将成交率从 26.67% 推到 88.88%；大规模自主谈判竞赛（arXiv:2503.06416）运行了 ~180k 次谈判，发现**隐藏思维链**的智能体通过对对手隐藏推理而获胜；Bhattacharya et al. 2025 在哈佛谈判项目指标上将 Llama-3 排为最有效、Claude-3 最激进、GPT-4 最公平。本课实现 Contract Net Protocol（FIPA 前身，Lesson 02），连接 LLM 风格的买方/卖方，运行 OG-Narrator 风格的分解，并测量每种结构选择如何改变成交率。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 02 (FIPA-ACL Heritage), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## 问题

两个智能体需要就价格达成一致。仅靠纯语言提示，2024-2026 年的 LLM 以惊人的低成交率达成交易（在 arXiv:2402.15813 的严格参数化博弈中约 27%）。规模无法修复：GPT-4 在博弈上并不比 GPT-3.5 结构性地更好；它只是在博弈的*语言*上更好。

根本问题是 LLM 混淆了两项工作——决定报价和叙述报价。OG-Narrator 将它们分开：确定性报价生成器计算数值动作；LLM 只负责叙述。成交率跳升到 ~89%。

这反映了一个经典的多智能体发现：将机制与通信层解耦就能赢。Contract Net Protocol（FIPA, 1996; Smith, 1980）是参考任务市场机制。将 LLM 插入叙述槽位，你就得到一个现代 LLM 驱动的任务市场。

## 概念

### Contract Net，一段话

Smith 1980 年的 Contract Net Protocol：一个 **manager** 广播一个 **call for proposals (cfp)**；**bidders** 用包含报价的 **propose** 消息回应；manager 选出赢家并向赢家发送 **accept-proposal**，向输家发送 **reject-proposal**。赢家执行工作。可选消息：**refuse**（bidder 拒绝提案）。FIPA 将其编纂为 `fipa-contract-net` 交互协议。

### 为什么 OG-Narrator 赢

"Measuring Bargaining Abilities of Language Models"（arXiv:2402.15813）观察到：

- LLM 经常违反博弈规则（以荒谬价格报价，忽略对方的 ZOPA）。
- 它们锚定效果差（接受糟糕的首次报价；以象征性而非策略性金额还价）。
- 规模本身无法修复这些。更大的模型产生更合理的语言但有类似的策略错误。

OG-Narrator 分解：

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

报价生成器是经典谈判策略：Rubinstein 博弈模型、Zeuthen 策略，或简单的价格 tit-for-tat。LLM 负责叙述。消息包含确定性价格和自然语言框架。

成交率跳升因为：
- 价格保持在博弈区间内。
- 锚定是策略性的，不是情绪化的。
- LLM 做它擅长的事：写作。

### NegotiationArena 发现

arXiv:2402.05863 提供了规范基准。标题发现：

- LLM 可以通过采用人设（"I am desperate to sell this by Friday"）将收益提高 ~20%——人设操纵是真实的策略。
- 公平/合作型智能体被对抗型智能体利用；防御需要显式的反姿态。
- 对称配对在约 40% 的基准场景中收敛到不公平结果。

这不是"LLM 是糟糕的谈判者"。而是"LLM 谈判太像人类了，包括可被利用的部分"。

### 思维链隐藏

大规模自主谈判竞赛（arXiv:2503.06416）在许多 LLM 策略间运行了 ~180k 次谈判。赢家对对手隐藏了推理：

- 如果一个智能体在公开可见的草稿本中打印"I will only go to $75; my reservation price is $70"，对手会读到它。
- 赢家私下计算策略；输出通道只包含报价和最少必要叙述。

这是经典博弈论的 2026 年回响（Aumann 1976 关于理性与信息）：暴露你的私人估值会损失收益。LLM 不会直觉到这一点，会愉快地在对对手可见的推理痕迹中输入它们的保留价。

工程要点：将私有草稿本上下文与公开消息上下文分开。不是可选的。

### Bhattacharya et al. 2025 — 模型排名

在哈佛谈判项目指标（原则性谈判、BATNA 尊重、利益互惠）上：

- **Llama-3** 在达成交易方面最有效（成交率 + 收益）。
- **Claude-3** 是最激进的谈判者（高锚定，晚让步）。
- **GPT-4** 是最公平的（配对间收益方差最小）。

这是 2025 年的快照。重点不是哪个模型在 2026 年 4 月获胜——而是不同基础模型有持久的谈判风格。异构集成（Lesson 15）将此作为多样性来源。

### 通过 Contract Net + LLM 进行任务分配

Contract Net 在 LLM 多智能体中的现代复用：

1. Manager 智能体将任务分解为单元。
2. 向工作智能体广播带任务描述的 `cfp`。
3. 每个工作者返回报价：`(price, eta, confidence)`，其中 price 可以是 token、计算单元或美元。
4. Manager 选出赢家（单个或多个，取决于任务）并授予。
5. 被拒绝的工作者可以自由竞标其他任务。

这在超过 100 个工作者时扩展良好，因为协调是广播-响应式的，不是同步聊天。生产中使用：Microsoft Agent Framework 的编排模式，一些 LangGraph 实现。

### LLM-Stakeholders 交互式谈判

NeurIPS 2024（https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf）引入了带**秘密分数**和**最低接受阈值**的多方可评分博弈。每个利益相关者有私有效用；LLM 必须从消息中推断它们。这是两方博弈到 N 方联盟形成的推广。与具有异构工作者能力的生产任务市场相关。

### 叙述 vs 机制规则

在所有 2024-2026 年谈判基准中，一致的工程规则是：

> 让 LLM 叙述。不要让 LLM 计算报价。

如果报价需要是数字（价格、ETA、数量），从谈判状态确定性地生成它，让 LLM 产出框架。如果报价需要是提案结构（任务分解、角色分配），让 LLM 起草它，但在发送前对照 schema 和约束检查进行验证。

## 动手构建

`code/main.py` 实现：

- `ContractNetManager`, `ContractNetTask`, `Bid` — manager + bidders，广播 cfp，收集提案，授予。
- `og_narrator_bargain(state, rng)` — OG-Narrator 买方：确定性 Zeuthen 风格向中点让步。
- `seller_response(state, rng)` — 确定性卖方还价策略（两种风格的结构性真实值）。
- `naive_llm_bargain(state, rng)` — 模拟全 LLM 博弈者：以高方差选择价格，经常超出 ZOPA。
- 测量：1000 次试验的成交率，每次试验采样新的保留价。

运行：

```
python3 code/main.py
```

预期输出：naive-LLM 成交率 ~65-75%；OG-Narrator 成交率 ~85-95%；15-25 个百分点的差距是将报价生成与叙述分解的结构性优势。加上一个三个 bidder 和一个任务的 Contract Net 任务市场分配示例。

## 使用方式

`outputs/skill-bargainer-designer.md` 设计一个博弈协议：谁生成报价（确定性或 LLM），谁叙述，私有草稿本如何与公开消息分离，以及如何监控成交率。

## 上线清单

生产博弈检查清单：

- **分离草稿本。** 私有状态永远不到达对手的上下文。这是不可谈判的。
- **确定性报价生成。** 价格、数量、ETA：计算，不要提示。
- **验证所有传入报价**对照 schema。在协议边界拒绝超出 ZOPA 的报价。
- **限制轮次。** 最多 3-5 轮；僵局时升级到调解人。
- **持续测量成交率和收益方差。** 成交率下降是症状——通常是提示漂移或对手侧攻击。
- **记录所有被拒绝的提案**及确定性理由。对于 Contract Net manager，失败的 bidder 需要理解原因。

## 练习

1. 运行 `code/main.py`。确认 OG-Narrator 在成交率上击败 naive-LLM。差多少？
2. 实现**基于人设的收益改善**（arXiv:2402.05863）——买方仅在叙述中采用"desperate to buy this week"人设，报价生成器不变。成交率或收益是否变化？
3. 实现思维链**隐藏**：维护一个不传递给对手的私有草稿本字符串。如果你意外泄露它会怎样（通过交换通道模拟）？
4. 将 Contract Net 扩展为带保留价的 N-bidder 拍卖。当所有出价都超过保留价时，manager 如何在最低价和最高质量之间决定？你选择哪种授予规则，为什么？
5. 阅读 Bhattacharya et al. 2025 关于哈佛谈判项目指标。实现两个不同风格的博弈者（激进 vs 公平）。测量对称和非对称配对下的收益方差。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Contract Net | "任务市场" | Smith 1980, FIPA 1996。cfp + propose + accept/reject。规范任务市场。 |
| ZOPA | "可能达成协议的区间" | 买方最高价和卖方最低价的重叠。超出它的报价无法成交。 |
| BATNA | "谈判协议的最佳替代方案" | 如果这笔交易失败你的后备方案。设定你的保留价。 |
| OG-Narrator | "报价生成器 + 叙述者" | 分解：确定性报价，LLM 叙述。 |
| Zeuthen strategy | "风险最小化让步" | 基于风险限制让步的经典报价生成器。 |
| Rubinstein bargaining | "交替报价均衡" | 带折扣的无限期博弈的博弈论模型。 |
| CoT concealment | "隐藏你的推理" | arXiv:2503.06416 中的赢家保持私有草稿本；公开通道只显示报价。 |
| Persona manipulation | "情绪姿态" | arXiv:2402.05863：绝望/紧迫人设带来 ~20% 收益增益。 |

## 延伸阅读

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — 基准；人设操纵和利用发现
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) — OG-Narrator 和买方比卖方更难的结果
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) — ~180k 次谈判；思维链隐藏获胜
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — 带秘密效用的多方可评分博弈
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — 经典机制，IEEE Transactions on Computers
