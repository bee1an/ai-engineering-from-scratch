# Prompt Caching 与 Context Caching

> 你的 system prompt 有 4,000 token。你的 RAG 上下文有 20,000 token。每次请求你都发送两者。你也为两者付费 — 每一次。Prompt caching 让 provider 在他们那边保持该前缀热缓存，复用时只收正常费率的 10%。正确使用时，它能将推理成本降低 50-90%，首 token 延迟降低 40-85%。

**类型：** 构建
**语言：** Python
**前置课程：** Phase 11 · 01 (Prompt Engineering), Phase 11 · 05 (Context Engineering), Phase 11 · 11 (Caching and Cost)
**时长：** ~60 分钟

## 问题

一个编码 agent 在对话的每一轮都向 Claude 发送相同的 15,000 token system prompt。二十轮对话以 $3/M input tokens 计算，仅输入成本就是 $0.90 — 还没算用户的实际消息。乘以每天 10,000 个对话，账单达到 $9,000/天，全是为了从不改变的文本。

你不能缩短 prompt 而不损害质量。你不能不发送它 — 模型每轮都需要它。唯一的办法是停止为 provider 已经见过的前缀付全价。

这个办法就是 prompt caching。Anthropic 在 2024 年 8 月推出（2025 年增加了 1 小时扩展 TTL 变体），OpenAI 在同年晚些时候自动化了它，Google 随 Gemini 1.5 推出了显式 context caching，三家现在都将其作为前沿模型的一等功能提供。

## 概念

![Prompt caching：写一次，读便宜](../assets/prompt-caching.svg)

**机制。** 当一个请求的前缀与最近请求的前缀匹配时，provider 从上次运行的 KV-cache 提供服务，而不是重新编码 token。你第一次付一个小的写入溢价，之后每次都获得大的读取折扣。

**2026 年三家 provider 的风格。**

| Provider | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存量 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 内容块上的显式 `cache_control` 标记 | 输入 90% 折扣 | 25% 附加费 | 5 分钟（可扩展到 1 小时） | 1,024 tokens (Sonnet/Opus), 2,048 (Haiku) |
| OpenAI | 自动前缀检测 | 输入 50% 折扣 | 无 | 最长 1 小时（尽力而为） | 1,024 tokens |
| Google (Gemini) | 显式 `CachedContent` API | 按存储计费；读取约为正常的 25% | 按 token·小时收存储费 | 用户设置（默认 1 小时） | 4,096 tokens (Flash), 32,768 (Pro) |

**不变量。** 三家都只缓存前缀。如果请求之间有任何 token 不同，第一个不同 token 之后的所有内容都是 miss。把*稳定*部分放在顶部，*可变*部分放在底部。

### 缓存友好的布局

```
[system prompt]          <-- 缓存这个
[tool definitions]       <-- 缓存这个
[few-shot examples]      <-- 缓存这个
[retrieved documents]    <-- 如果复用则缓存，否则不缓存
[conversation history]   <-- 缓存到上一轮
[current user message]   <-- 永远不缓存（每次都不同）
```

违反顺序 — 把用户消息放在 system prompt 上面，在 few-shots 之间穿插动态检索 — 缓存永远不会命中。

### 盈亏平衡计算

Anthropic 的 25% 写入溢价意味着一个缓存块至少要被读取两次才能净省钱。1 次写入 + 1 次读取平均每请求 0.675x 成本（省 32%）；1 次写入 + 10 次读取平均 0.205x（省 80%）。经验法则：缓存任何你预期在 TTL 内至少复用 3 次的内容。

## 构建

### 步骤 1：Anthropic prompt caching 与显式标记

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 存储该块 5 分钟。在该窗口内复用命中；过期后复用则重新写入。

**响应 usage 字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

在 CI 中检查两个字段 — 如果 `cache_read_input_tokens` 在请求间一直为零，你的缓存键在漂移。

### 步骤 2：一小时扩展 TTL

对于长时间运行的批处理任务，5 分钟默认值在任务间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价是 2x（比基线高 50% 而非 25%），但对任何复用前缀超过 5 次的批处理很快回本。

### 步骤 3：OpenAI 自动缓存

OpenAI 不需要你配置任何东西。任何超过 1,024 token 且匹配最近请求的前缀自动获得 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

同样的缓存友好布局规则适用。两件事会杀死 OpenAI 的缓存但不会杀死 Anthropic 的：改变 `user` 字段（用作缓存键组件）和重排 tools。

### 步骤 4：Gemini 显式 context caching

Gemini 把缓存当作你创建和命名的一等对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 按缓存存活期间的 token·小时收存储费，读取按正常输入费率的约 25%。当你跨多天多会话复用同一个巨大 prompt 时，这是正确的形状。

### 步骤 5：在生产中测量命中率

见 `code/main.py` 中模拟的三 provider 会计，跟踪写入/读取/miss 计数并计算每 1K 请求的混合成本。以目标命中率作为部署门控 — 大多数生产 Anthropic 设置在预热后应该看到 >80% 的读取比例。

## 2026 年仍在上线的陷阱

- **顶部的动态时间戳。** system prompt 顶部的 `"Current time: 2026-04-22 15:30:02"`。每个请求都 miss。把时间戳移到缓存断点下方。
- **Tool 重排。** 以稳定顺序序列化 tools — 部署间的 dict 重排会破坏每次命中。
- **自由文本近似重复。** "You are helpful." vs "You are a helpful assistant." — 一个字节的差异 = 完全 miss。
- **太小的块。** Anthropic 强制 1,024 token 下限（Haiku 为 2,048）。更小的块静默不缓存。
- **盲目的成本仪表盘。** 把"输入 tokens"拆分为已缓存 vs 未缓存。否则流量下降看起来像缓存胜利。

## 使用

2026 年的缓存栈：

| 场景 | 选择 |
|-----------|------|
| 带稳定 10k+ system prompt 的 agent，多轮对话 | Anthropic `cache_control` 配 5 分钟 TTL |
| 复用前缀 30+ 分钟的批处理任务 | Anthropic 配 `ttl: "1h"` |
| GPT-5 上的 serverless 端点，无自定义基础设施 | OpenAI 自动（只需让前缀稳定且足够长） |
| 跨多天复用巨大代码/文档语料库 | Gemini 显式 `CachedContent` |
| 跨 provider 降级 | 保持可缓存前缀布局在各 provider 间一致，这样任何命中都有效 |

结合语义缓存（Phase 11 · 11）用于用户消息层：prompt caching 处理 *token 完全相同* 的复用，语义缓存处理 *含义相同* 的复用。

## 交付

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. **简单。** 用一个 5,000 token system prompt 对 Claude 进行 10 轮对话。分别在不带 `cache_control` 和带 `cache_control` 的情况下运行。报告各自的输入 token 账单。
2. **中等。** 编写一个测试 harness，给定一个 prompt 模板和请求日志，计算每个 provider（Anthropic 5m、Anthropic 1h、OpenAI 自动、Gemini 显式）的预期命中率和美元节省。
3. **困难。** 构建一个布局优化器：给定一个 prompt 和一个标记了 `stable=True/False` 的字段列表，重写 prompt 把单个缓存断点放在最大缓存友好位置而不丢失信息。在真实 Anthropic 端点上验证。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Prompt caching | "让长 prompt 变便宜" | 复用 provider 侧的 KV-cache 用于匹配前缀；重复输入 token 50-90% 折扣。 |
| `cache_control` | "Anthropic 的标记" | 内容块属性，声明"到这里为止都是可缓存的"；`{"type": "ephemeral"}`。 |
| Cache write | "付溢价" | 填充缓存的第一个请求；Anthropic 按约 1.25x 输入费率计费，OpenAI 免费。 |
| Cache read | "折扣" | 匹配前缀的后续请求；Anthropic 按 10% 计费，OpenAI 50%，Gemini 约 25%。 |
| TTL | "它活多久" | 缓存保持热的秒数；Anthropic 默认 5 分钟（可扩展 1h），OpenAI 尽力最长 1h，Gemini 用户设置。 |
| Extended TTL | "1 小时 Anthropic 缓存" | `{"type": "ephemeral", "ttl": "1h"}`；2x 写入溢价但对批处理复用值得。 |
| Prefix match | "为什么我的缓存 miss 了" | 缓存只在从开头到断点的每个 token 都字节相同时才命中。 |
| Context caching (Gemini) | "显式的那个" | Google 的命名、按存储计费的缓存对象；最适合跨多天复用大型语料库。 |

## 延伸阅读

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1h TTL、盈亏平衡表。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 和存储定价。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — 原始发布文章，含延迟数据。
- Phase 11 · 05 (Context Engineering) — 在哪里切分 prompt 让缓存能落地。
- Phase 11 · 11 (Caching and Cost) — 将 prompt caching 与用户消息上的语义缓存配对。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — prompt caching 暴露给用户的 KV-cache 内存模型；解释为什么缓存前缀重读比重算便宜约 10 倍。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — prefill 是 prompt caching 跳过的阶段；这篇论文解释为什么缓存命中时 TTFT 大幅下降而 TPOT 不受影响。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — prompt caching 与 speculative decoding、Flash Attention 和 MQA/GQA 并列为弯曲推理成本曲线的杠杆；阅读这篇了解其他三个。
