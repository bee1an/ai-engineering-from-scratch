# 对话状态追踪

> "I want a cheap restaurant in the north... actually make it moderate... and add Italian." 三轮对话，三次状态更新。DST 保持 slot-value 字典同步，这样预订才能成功。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time:** ~75 minutes

## 问题

在任务导向对话系统中，用户的目标被编码为一组 slot-value 对：`{cuisine: italian, area: north, price: moderate}`。每一轮用户发言都可能添加、修改或删除一个 slot。系统必须读取整个对话并正确输出当前状态。

搞错一个 slot，系统就会订错餐厅、排错航班或扣错卡。DST 是用户所说的话和后端执行之间的铰链。

为什么在 2026 年尽管有 LLM 它仍然重要：

- 合规敏感领域（银行、医疗、航空订票）需要确定性的 slot 值，而不是自由形式的生成。
- Tool-use agent 在调用 API 之前仍然需要 slot 解析。
- 多轮修正比看起来更难："actually no, make it Thursday."

现代管线：经典 DST 概念 + LLM 抽取器 + structured-output 护栏。

## 概念

![DST: dialog history → slot-value state](../assets/dst.svg)

**任务结构。** Schema 定义领域（restaurant、hotel、taxi）及其 slot（cuisine、area、price、people）。每个 slot 可以为空、填入封闭集合中的值（price: {cheap, moderate, expensive}），或自由形式的值（name: "The Copper Kettle"）。

**两种 DST 形式。**

- **分类。** 对每个 (slot, candidate_value) 对，预测 yes/no。适用于封闭词汇 slot。2020 年前的标准。
- **生成。** 给定对话，以自由文本生成 slot 值。适用于开放词汇 slot。现代默认。

**指标。** Joint Goal Accuracy (JGA)——*每个* slot 都正确的轮次比例。全有或全无。MultiWOZ 2.4 排行榜在 2026 年顶部约 83%。

**架构。**

1. **基于规则（slot regex + keyword）。** 窄领域的强基线。可调试。
2. **TripPy / BERT-DST。** 基于复制的生成 + BERT 编码。LLM 之前的标准。
3. **LDST（LLaMA + LoRA）。** 指令微调 LLM + domain-slot prompting。在 MultiWOZ 2.4 上达到 ChatGPT 级质量。
4. **Ontology-free（2024–26）。** 跳过 schema；直接生成 slot 名称和值。处理开放领域。
5. **Prompt + structured output（2024–26）。** LLM + Pydantic schema + constrained decoding。5 行代码，生产就绪。

### 经典失败模式

- **跨轮共指。** "Let's stay with the first option." 需要解析哪个选项。
- **覆盖 vs 追加。** 用户说 "add Italian." 是替换 cuisine 还是追加？
- **隐式确认。** "OK cool"——这是接受了提供的预订吗？
- **修正。** "Actually make it 7 pm." 必须更新时间而不清除其他 slot。
- **对前一个系统发言的共指。** "Yes, that one." 哪个 "that"？

## 动手构建

### 第 1 步：基于规则的 slot 抽取器

参见 `code/main.py`。Regex + 同义词字典在窄领域覆盖 70% 的规范发言：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在规范词汇之外很脆弱。适用于确定性 slot 确认。

### 第 2 步：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 永远不要重置用户没有触及的 slot。
- 显式否定（"never mind the cuisine"）必须清除。
- 用户修正（"actually..."）必须覆盖，不是追加。

### 第 3 步：LLM 驱动的 DST + structured output

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic 保证有效的状态对象。没有 regex，没有 schema 不匹配，没有幻觉 slot。

### 第 4 步：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统在多少比例的轮次中所有 slot 都正确？MultiWOZ 2.4 上 2026 年顶级系统：80-83%。你的领域内系统在你的窄词汇上应该超过这个数字，否则 LLM 基线就赢了。

### 第 5 步：处理修正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

检测到修正时，覆盖最后更新的 slot 而不是追加。没有 LLM 帮助很难做对。现代模式：始终让 LLM 从历史重新生成整个状态，而不是增量更新——这自然处理了修正。

## 常见陷阱

- **全历史重新生成成本。** 让 LLM 每轮重新生成状态总共花费 O(n²) token。限制历史或总结较早的轮次。
- **Schema 漂移。** 事后添加新 slot 会破坏旧训练数据。给 schema 加版本。
- **大小写敏感。** "Italian" vs "italian" vs "ITALIAN"——到处做归一化。
- **隐式继承。** 如果用户之前指定了 "for 4 people"，对不同时间的新请求不应该清除 people。始终传递完整历史。
- **自由形式 vs 封闭集合。** 名称、时间和地址需要自由形式 slot；cuisine 和 area 是封闭的。在 schema 中混合两者。

## 实际应用

2026 技术栈：

| 场景 | 方法 |
|-----------|----------|
| 窄领域（一两个意图） | Rule-based + regex |
| 宽领域，有标注数据 | LDST (LLaMA + LoRA on MultiWOZ-style data) |
| 宽领域，无标签，生产就绪 | LLM + Instructor + Pydantic schema |
| 语音 | ASR + normalizer + LLM-DST |
| 多领域预订流程 | Schema-guided LLM with per-domain Pydantic models |
| 合规敏感 | Rule-based primary, LLM fallback with confirmation flow |

## 交付

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习

1. **简单。** 在 `code/main.py` 中为 3 个 slot（cuisine、area、price）构建基于规则的状态追踪器。在 10 个手工构造的对话上测试。测量 JGA。
2. **中等。** 同一数据集用 Instructor + Pydantic + 小型 LLM。比较 JGA。检查最难的轮次。
3. **困难。** 实现两者并路由：基于规则为主，当规则只输出 <2 个有置信度的 slot 时 LLM 兜底。测量组合 JGA 和每轮推理成本。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| DST | 对话状态追踪 | 跨对话轮次维护 slot-value 字典。 |
| Slot | 用户意图的单元 | 后端需要的命名参数（cuisine、date）。 |
| Domain | 任务领域 | Restaurant、hotel、taxi——slot 的集合。 |
| JGA | Joint Goal Accuracy | 每个 slot 都正确的轮次比例。全有或全无。 |
| MultiWOZ | 那个基准 | 多领域 WOZ 数据集；标准 DST 评估。 |
| Ontology-free DST | 无 schema | 直接生成 slot 名称和值，没有固定列表。 |
| Correction | "Actually..." | 覆盖之前已填充 slot 的轮次。 |

## 延伸阅读

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — the canonical benchmark.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — LLaMA + LoRA instruction tuning for DST.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — the copy-based DST workhorse.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — EM-based unsupervised TOD.
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — canonical DST results.
