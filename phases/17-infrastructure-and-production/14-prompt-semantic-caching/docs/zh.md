# Prompt Caching 与 Semantic Caching 经济学

> **定价快照日期 2026-04。** 以下数字来自本课程发布时捕获的供应商价目表；在下游引用前请对照链接文档验证。

> Caching 发生在两层。L2（provider 级别）prompt/prefix caching 复用重复前缀的 attention KV — Anthropic 的 prompt-caching 文档宣传长 prompt 可降低最高 90% 成本和 85% 延迟；Claude 3.5 Sonnet 的 cache read 为 $0.30/M，而 fresh 为 $3.00/M，5 分钟 TTL，1 小时 TTL 选项有 2x write premium（docs.anthropic.com, 2026-04）。OpenAI prompt caching 对 ≥1024 tokens 的 prompt 自动生效，cached input 定价约为 fresh 的 90% 折扣（platform.openai.com, 2026-04）；具体每模型 cached 费率取决于实时价目表。L1（应用级别）semantic caching 在 embedding 相似度命中时完全跳过 LLM。供应商所称的 "95% accuracy" 指的是匹配正确率，而非命中率 — 报告的生产命中率从 10%（开放式聊天）到 70%（结构化 FAQ）不等；两家 provider 都未发布官方基线，因此应将这些视为社区遥测数据而非保证。生产陷阱：并行化会杀死 caching（N 个并行请求在第一次 cache write 之前发出，可能使开支膨胀数倍），前缀中的动态内容会完全阻止 cache 命中。ProjectDiscovery 报告通过将动态文本移出可缓存前缀，命中率从 7% 提升到 74%（2025-11）。

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## 学习目标

- 区分 L2 prompt/prefix caching（provider 端 KV 复用）和 L1 semantic caching（相似 prompt 时绕过 LLM）。
- 解释 Anthropic 的 `cache_control` 显式标记及两种 TTL 选项（5 分钟 vs 1 小时）及其价格倍率。
- 给定命中率、prompt/response 比例和 token 价格，计算预期月度节省。
- 说出使账单膨胀 5-10 倍的并行化反模式和使命中率崩塌的动态内容反模式。

## 问题

你给 RAG 服务加了 prompt caching。账单纹丝不动。你测量命中率：7%。你的 prompt 看起来是静态的但实际不是 — system prompt 包含精确到分钟的当前日期、一个 request ID、以及为多样性随机重排的示例。每个请求写入一条新 cache entry，读取为零。

另外，你的 agent 每个用户问题并行发起十个 tool call。十个请求在第一次 cache write 完成之前全部到达 provider。十次写入，零次读取。你的账单是 "有 caching" 预期成本的 5-10 倍。

Caching 是一个协议，不是一个开关。两层，两种不同的失败模式。

## 概念

### L2 — provider prompt/prefix caching

Provider 存储可缓存前缀的 attention KV，并在下一个匹配该前缀的请求中复用。你只付一次 write 成本，read 几乎免费。

**Anthropic (Claude 3.5 / 3.7 / 4 系列)**：请求中的显式 `cache_control` 标记。你标注哪些 block 可缓存。TTL：5 分钟（write 成本 1.25x base）或 1 小时（write 成本 2x base）。Cache read：Claude 3.5 Sonnet 上 $0.30/M vs fresh $3.00/M — 便宜 10 倍（docs.anthropic.com, as of 2026-04）。不同模型（Opus/Haiku）费率单独发布；务必交叉核对实时定价页面。

**OpenAI**：对 ≥1024 tokens 的 prompt 自动缓存（platform.openai.com, 2026-04）。无需显式标记。当前 gpt-4o/gpt-5 价目表上 cached input 约比 fresh 便宜 10 倍。文档和 release notes 均未发布官方命中率基线；社区报告在精心设计 prompt 时集中在 30-60%。监控 `usage.cached_tokens` 来测量你自己的数据。

**Google (Gemini)**：通过显式 API 实现 context caching；1M-token context 意味着 caching 收益更大。

**自托管 (vLLM, SGLang)**：Phase 17 · 06 涵盖 RadixAttention — 在你自己的算力上实现相同模式。

### L1 — 应用级 semantic caching

在调用 LLM 之前，对 prompt 做 hash、embedding，然后查找相似的已缓存请求（cosine similarity 超过阈值，通常 0.95+）。命中则返回缓存响应。未命中则调用 LLM 并缓存结果。

开源方案：Redis Vector Similarity、GPTCache、Qdrant。商业方案：Portkey Cache、Helicone Cache。

供应商的 accuracy 声称指的是返回的缓存响应在语义上是否恰当 — 而非命中频率。生产命中率：

- 开放式聊天：10-15%。
- 结构化 FAQ / 客服：40-70%。
- 代码问题：20-30%（细微变体会杀死命中）。
- 语音 agent 重复 prompt：50-80%（语音归一化固定集合）。

### 并行化反模式

你的 agent 并行发起 10 个 tool call。10 个都有相同的 4K-token system prompt。Anthropic 的 cache write 是 per-request 的；第一次 cache-write 在 provider 收到 prompt 后约 300 ms 完成。请求 2-10 在同一毫秒窗口到达，每个都看到 cache miss。你付了 10 次 write premium，0 次 read discount。

修复方案：先顺序发一个 — 单独发请求 1，等请求 1 的 cache 填充完成后再发 2-10。给第一个 tool call 增加 300 ms；节省 5-10 倍账单。

### 动态内容反模式

你的 system prompt 长这样：

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每个请求都是唯一的。每个请求都写入。零命中。

修复方案：将真正静态的内容放入可缓存前缀；将动态内容追加到 cache 边界之后：

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 通过这种方式将 cache 命中率从 7% 提升到 74%，并发布了详细分析。

### 叠加 batch + cache 用于过夜工作负载

Batch API（Phase 17 · 15）在 24 小时周转时给 50% 折扣。在此基础上叠加 cached input 再获得约 10 倍优惠。过夜分类、标注和报告生成工作负载可以通过叠加降至同步-无缓存成本的约 10%。

### 你应该记住的数字

定价数据捕获于 2026-04，来自链接的供应商文档，每隔几个月会变动 — 依赖前请重新核实。

- Anthropic cached read：Claude 3.5 Sonnet 上 $0.30/M，约比 fresh input 便宜 10 倍（docs.anthropic.com）。
- Anthropic cache write premium：1.25x（5 分钟 TTL）或 2x（1 小时 TTL）。
- OpenAI auto-cache：适用于 ≥1024 tokens 的 prompt；cached input 定价约为当前价目表 fresh input 的 10%（platform.openai.com）。
- Semantic cache 命中率（社区报告）：开放聊天约 10%；结构化 FAQ 最高约 70%。非供应商官方基线。
- ProjectDiscovery：通过将动态内容移出前缀，命中率从 7% 提升到 74%（项目博客, 2025-11）。
- 并行化反模式：N 个并行请求 miss 第一次 cache write 时，典型报告为 5-10 倍账单膨胀。

## 动手试试

`code/main.py` 模拟混合工作负载上的 L1 + L2 caching。报告命中率、账单，并展示并行化惩罚。

## 交付产出

本课程产出 `outputs/skill-cache-auditor.md`。给定 prompt 模板和流量，审计可缓存性并推荐重构方案。

## 练习

1. 运行 `code/main.py`。切换 parallelization flag。账单变化多少？
2. 你的 system prompt 包含日期。将其移出。展示前后命中率计算。
3. 给定你的请求到达率，计算 1 小时 TTL（2x write）vs 5 分钟 TTL（1.25x write）的盈亏平衡点。
4. Semantic cache 在 0.95 阈值时命中 20%。在 0.85 时命中 50% 但你看到不正确的缓存响应。选择正确的阈值并说明理由。
5. 你每个用户问题 batch 10 个并行子查询。在不增加端到端延迟的前提下重写为 cache 友好的方式。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| L2 prompt cache | "prefix cache" | Provider 存储重复前缀的 KV |
| `cache_control` | "Anthropic cache marker" | 标记可缓存 block 的显式 attribute |
| Cache write premium | "write tax" | 首次 miss-to-cache 的额外成本（1.25x 或 2x） |
| L1 semantic cache | "embedding cache" | 调用 LLM 前的应用级 hash-and-embed |
| GPTCache | "LLM caching lib" | 流行的 OSS L1 cache 库 |
| Cache hit rate | "hits / total" | 从 cache 提供服务的请求比例 |
| Parallelization anti-pattern | "the N-write trap" | N 个并行请求 miss cache N 次 |
| Dynamic content trap | "the time-in-prompt trap" | 前缀中的动态字节杀死命中率 |
| RadixAttention | "intra-replica cache" | SGLang 的 prefix-cache 实现 |

## 延伸阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 官方 `cache_control` 语义和 TTL。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动缓存行为和资格条件。
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
