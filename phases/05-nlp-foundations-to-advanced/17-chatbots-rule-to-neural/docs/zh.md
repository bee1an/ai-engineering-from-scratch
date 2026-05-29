# 对话机器人 — 从规则到神经网络到 LLM Agent

> ELIZA 用模式匹配回复。DialogFlow 映射意图。GPT 从权重中回答。Claude 运行工具并验证。每个时代解决了前一个时代最严重的失败。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## 问题

用户说"我想改签航班"。系统必须弄清楚他们想要什么、缺少什么信息、如何获取、以及如何完成操作。然后用户说"等等，如果我取消呢？"系统必须记住上下文、切换任务并保持状态。

对话对 ML 系统来说很难。输入是开放式的。输出必须在多轮中保持连贯。系统可能需要对世界采取行动（改签航班、扣款）。每一步错误对用户都是可见的。

对话机器人架构经历了四个范式的循环，每个范式的引入都是因为前一个失败得太明显。本课按顺序讲解。2026 年的生产格局是后两者的混合。

## 概念

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

**基于规则（ELIZA, AIML, DialogFlow）。** 手工编写的模式匹配用户输入并产生回复。意图分类器路由到预定义流程。槽填充状态机收集所需信息。在其设计的狭窄范围内工作出色。超出范围立即失败。在不容忍幻觉的安全关键领域（银行认证、航空订票）仍然在用。

**基于检索。** FAQ 风格的系统。编码每对（话语，回复）。运行时编码用户消息并检索最近的存储回复。想想 Zendesk 经典的"相似文章"功能。比规则更好地处理改述。没有生成，所以没有幻觉。

**神经网络（seq2seq）。** 在对话日志上训练的 encoder-decoder。从头生成回复。流畅但容易产生通用输出（"我不知道"）和事实漂移。从不可靠地切题。这就是 Google、Facebook 和 Microsoft 在 2016-2019 年都有令人失望的聊天机器人的原因。

**LLM agent。** 包裹在循环中的语言模型，进行规划、调用工具并验证结果。不是带长 prompt 的聊天机器人。而是一个 agent 循环：规划 → 调用工具 → 观察结果 → 决定下一步。检索优先的锚定（RAG）防止幻觉。工具调用让它真正能做事。这是 2026 年的架构。

四个范式不是顺序替代。2026 年的生产聊天机器人路由通过所有四个：规则用于认证和破坏性操作，检索用于 FAQ，神经生成用于自然措辞，LLM agent 用于模糊的开放式查询。

## 动手构建

### 第 1 步：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20 行的 ELIZA。反射技巧（"I feel sad" → "Why do you feel sad"）是 Weizenbaum 1966 年经典的心理治疗师演示。至今仍有教学价值。

### 第 2 步：基于检索（FAQ）

这个示例代码需要 `pip install sentence-transformers`（会拉入 torch）。本课的可运行 `code/main.py` 使用标准库的 Jaccard 相似度代替，因此课程无需外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒答是关键设计选择。如果最佳匹配不够接近，返回 `None` 让系统升级处理。

### 第 3 步：神经生成（基线）

使用小型指令微调的 encoder-decoder（FLAN-T5）或微调的对话模型。2026 年单独使用在生产中不可行（矛盾、跑题漂移、事实胡说），但在混合系统中用于自然措辞。DialoGPT 风格的 decoder-only 模型需要显式的轮次分隔符和 EOS 处理才能产生连贯回复；FLAN-T5 text2text pipeline 作为教学示例开箱即用。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 第 4 步：LLM agent 循环

2026 年的生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

三件事需要说明。工具是 LLM 可以调用的可执行函数。循环在 LLM 返回最终答案而非工具调用时终止。步骤预算防止在模糊任务上的无限循环。

真正的生产还需要：检索优先的锚定（每次 LLM 调用前注入相关文档）、护栏（未经确认拒绝破坏性操作）、可观测性（记录每一步）、以及评估（自动检查 agent 行为是否符合规范）。

### 第 5 步：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式：确定性规则处理任何破坏性操作，检索处理固定 FAQ，LLM agent 处理其他一切。这就是 2026 年客服系统的实际部署方式。

## 实际应用

2026 年技术栈：

| 用例 | 架构 |
|---------|---------------|
| 订票、支付、认证 | 基于规则的状态机 + 槽填充 |
| 客服 FAQ | 在策划好的答案上检索 |
| 开放式帮助聊天 | LLM agent + RAG + 工具调用 |
| 内部工具 / IDE 助手 | LLM agent + 工具调用（搜索、读取、写入） |
| 陪伴 / 角色聊天机器人 | 带人设 system prompt 的微调 LLM，知识检索 |

生产中始终使用混合路由。没有单一架构能很好地处理每个请求。路由层本身通常是一个小型意图分类器。

## 仍然在上线的失败模式

- **自信的编造。** LLM agent 声称完成了一个它没有完成的操作。缓解：验证结果，记录工具调用，绝不让 LLM 在没有成功工具返回的情况下声称做了某事。
- **Prompt 注入。** 用户插入覆盖 system prompt 的文本。在 OWASP Top 10 for LLM Applications 2025 中排名 LLM01。两种形式：直接注入（粘贴到聊天中）和间接注入（隐藏在 agent 读取的文档、邮件或工具输出中）。

  攻击成功率因场景而异。在通用工具使用和编码基准中，前沿模型的测量成功率约 0.5-8.5%。特定高风险设置（针对 AI 编码 agent 的自适应攻击、脆弱的编排）已达到约 84%。生产 CVE 包括 EchoLeak（CVE-2025-32711, CVSS 9.3）——Microsoft 365 Copilot 中由攻击者控制的邮件触发的零点击数据泄露漏洞。

  缓解：在整个循环中将用户输入视为不可信；工具调用前进行清理；将工具输出与主 prompt 隔离；使用 Plan-Verify-Execute (PVE) 模式，agent 先规划，然后在执行前验证每个操作是否符合计划（这阻止工具结果注入新的计划外操作）；破坏性操作需要用户确认；对工具范围应用最小权限。

  再多的 prompt 工程也无法完全消除这个风险。需要外部运行时防御层（LLM Guard、白名单验证、语义异常检测）。
- **范围蔓延。** Agent 跑题，因为工具调用返回了切线相关的信息。缓解：收窄工具契约；保持 system prompt 聚焦；添加跑题率评估。
- **无限循环。** Agent 持续调用同一个工具。缓解：步骤预算、工具调用去重、LLM 判断"我们是否在取得进展"。
- **上下文窗口耗尽。** 长对话将最早的轮次推出上下文。缓解：总结旧轮次，按相似度检索相关的过去轮次，或使用长上下文模型。

## 交付

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习

1. **简单。** 用 10 个模式为咖啡店点单机器人实现上述基于规则的回复。测试边界情况：重复下单、修改、取消、意图不明。
2. **中等。** 构建混合 FAQ + LLM 兜底。50 条 SaaS 产品的固定 FAQ 条目，LLM 兜底带文档站检索。在 100 个真实客服问题上测量拒答率和准确率。
3. **困难。** 用三个工具（搜索、读取用户数据、发送邮件）实现上述 agent 循环。用 50 个测试场景运行评估，包括 prompt 注入尝试。报告跑题率、任务失败率和任何注入成功。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|-----------------|-----------------------|
| Intent | 用户想要什么 | 分类标签（book_flight, reset_password）。路由到处理器。 |
| Slot | 一条信息 | 机器人需要的参数（日期、目的地）。槽填充是一系列询问。 |
| RAG | 检索加生成 | 检索相关文档，然后锚定 LLM 的回复。 |
| Tool call | 函数调用 | LLM 发出带名称 + 参数的结构化调用。运行时执行，返回结果。 |
| Agent loop | 规划、行动、验证 | 控制器运行 LLM 调用与工具调用交替，直到任务完成。 |
| Prompt injection | 用户攻击 prompt | 试图覆盖 system prompt 的恶意输入。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 原始基于规则的聊天机器人论文。
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — Google 晚期的神经聊天机器人论文，就在 LLM agent 接管之前。
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — 命名 agent 循环模式的论文。
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2024 年的生产指南，2026 年仍然适用。
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — prompt 注入论文。
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 使 prompt 注入成为头号安全关注的排名。
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 实用编排层防御，包括 Plan-Verify-Execute 和用户确认流程。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 间接 prompt 注入的经典零点击数据泄露 CVE。说明为什么有写权限的 agent 需要运行时防御的参考案例。
