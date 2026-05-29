# FIPA-ACL 与言语行为的遗产

> 在 MCP 之前，在 A2A 之前，有 FIPA-ACL。2000 年 IEEE 智能物理代理基金会批准了一种 agent 通信语言，包含二十个施为动词、两种内容语言和一组交互协议——合同网、订阅/通知、条件请求。它因本体论开销对 web 来说太重而从工业界消退，但 LLM 驱动的多智能体系统复兴正在悄悄重新实现相同的思想，只是没有形式语义：JSON 契约代替了施为动词，自然语言代替了本体论。本课认真研读 FIPA-ACL，让你看清 2026 年的协议决策中哪些是重新发明、哪些是真正的创新，以及当前浪潮将在哪里重新发现 2000 年代已经解决的问题。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 minutes

## 问题

2026 年的 agent 协议版图很拥挤：MCP 用于工具、A2A 用于 agent、ACP 用于企业审计、ANP 用于去中心化信任、NLIP 用于自然语言内容，加上 CA-MCP 和二十多个研究提案。每个规范都宣称自己是基础性的。

诚实的解读是：它们大多在重新发现一棵非常具体的、二十年前的决策树。Austin（1962）和 Searle（1969）的言语行为理论给了我们"话语即行动"。KQML（1993）将其变成了线路协议。FIPA-ACL（2000 年批准）产出了参考标准化：二十个施为动词、内容语言 SL0/SL1、合同网和订阅通知的交互协议。JADE 和 JACK 是 Java 参考平台。这项工作在 2010 年左右消退，因为本体论开销太重，web 正在胜出。

当你看到 MCP 的 `tools/call`、A2A 的任务生命周期或 CA-MCP 的共享上下文存储时，你看到的是 FIPA 决策的更软、JSON 原生的翻版。了解这段遗产告诉你两件事：哪些新"创新"实际上是重新发明，以及哪些旧的失败模式新规范将重新发现。

## 概念

### 言语行为，一段话

Austin 注意到有些句子不是描述世界——而是改变世界。"我承诺。""我请求。""我宣布。"他称之为施为性话语。Searle 形式化了五个类别：断言式、指令式、承诺式、表达式、宣告式。KQML（Finin 等，1993）将其操作化为软件 agent：一条消息是一个施为动词（动作）加上内容（动作的对象）。FIPA-ACL 清理了 KQML 的缺陷，围绕二十个施为动词进行了标准化。

### 二十个 FIPA 施为动词（部分列表）

| Performative | 意图 |
|---|---|
| `inform` | "我告诉你 P 为真" |
| `request` | "我请求你做 X" |
| `query-if` | "P 为真吗？" |
| `query-ref` | "X 的值是什么？" |
| `propose` | "我提议我们做 X" |
| `accept-proposal` | "我接受提议" |
| `reject-proposal` | "我拒绝提议" |
| `agree` | "我同意做 X" |
| `refuse` | "我拒绝做 X" |
| `confirm` | "我确认 P 为真" |
| `disconfirm` | "我否认 P" |
| `not-understood` | "你的消息无法解析" |
| `cfp` | "对 X 征集提案" |
| `subscribe` | "当 X 变化时通知我" |
| `cancel` | "取消正在进行的 X" |
| `failure` | "我尝试了 X 但失败了" |

完整列表在 `fipa00037.pdf`（FIPA ACL Message Structure）中。重点不是记住它——重点是其中每一个都对应着 LLM 协议最终会重新添加的一个原语。

### 规范的 FIPA-ACL 消息

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段承载协议信封；一个字段（`content`）承载有效载荷。其余字段正是你每次在 JSON 协议上加装重试、线程和本体论时重新发明的东西。

### 两个遗留平台

**JADE**（Java Agent DEvelopment framework，1999–2020s）是使用最广泛的 FIPA 兼容运行时。Agent 继承基类，交换 ACL 消息，在容器中运行，使用"行为"进行协调。交互协议库附带了合同网、订阅通知、条件请求和提议接受。

**JACK**（Agent Oriented Software，商业）强调在 FIPA 消息之上的 BDI（信念-愿望-意图）推理。更形式化，采用更少。

两者都在 web 技术栈吃掉多 agent 用例后衰落。MCP 和 A2A 是 2026 年的运行时"容器"。

### 为什么 FIPA 消退了

- **本体论开销。** FIPA 要求共享本体论来解析 `content`。就本体论达成一致是一个持续数年的标准化过程。Web 直接用了 HTTP + JSON。
- **没人用的形式语义。** SL（语义语言）给出了严格的真值条件，但大多数生产系统使用自由格式内容并忽略了形式主义。
- **工具锁定。** JADE 只支持 Java；JACK 是商业的。多语言团队绕过了两者。
- **互联网赢了技术栈。** REST，然后 JSON-RPC，然后 gRPC 取代了 ACL 的传输层。

### LLM 复兴是 FIPA-lite

比较 FIPA `request` 和 MCP `tools/call`：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

相同的信封，不同的语法。两者都携带：谁、给谁、意图、有效载荷、关联 ID。两者都不是对另一个的革命——它们是同一设计上的不同权衡。

Liu 等人 2025 年的综述（"A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP"，arXiv:2505.02279）明确指出了这一谱系：MCP 对应工具使用言语行为，A2A 对应 agent 对等言语行为，ACP 对应审计追踪言语行为，ANP 对应去中心化身份扩展。新规范是 ACL 的后代，带有 JSON 语法和更松散的语义。

### 权衡，直白地说

**FIPA 给你的而现代规范丢弃的：**

- 形式语义——你可以证明 `inform` 意味着发送者相信内容。
- 施为动词的规范目录——你不必重新争论"我们是否应该有 `cancel`？"
- 数十年的交互协议模式——合同网、订阅通知、提议接受——具有已知的正确性属性。

**现代规范给你的而 FIPA 没有的：**

- JSON 原生有效载荷，兼容每个现代工具。
- 自然语言内容，LLM 无需手工编码的本体论即可解释。
- Web 技术栈传输（HTTP、SSE、WebSocket）。
- 通过自描述文档进行能力发现（MCP `listTools`、A2A Agent Card）。

用更松散的意图语义换取更容易的实现。这就是确切的交易。

### 值得移植的交互协议

FIPA 附带了约 15 个交互协议。三个值得带入 LLM 多智能体系统：

1. **合同网协议 (CNP)。** 管理者发出 `cfp`（征集提案）；投标者用 `propose` 响应；管理者接受/拒绝。这是规范的任务市场模式（Phase 16 · 16 谈判）。
2. **订阅/通知。** 订阅者发送 `subscribe`；发布者在主题变化时发送 `inform`。这就是 2026 年的每个事件总线。
3. **条件请求。** "当条件 Y 成立时做 X。"带前置条件的延迟动作。2026 年的类比是持久工作流引擎中的延迟任务（Phase 16 · 22 生产扩展）。

每个都能干净地映射到现代消息队列、HTTP + 轮询或 SSE 流。

### 丢弃本体论后会出什么问题

没有共享本体论，agent 从自然语言内容推断含义。2026 年记录在案的失败模式是**语义漂移**：两个 agent 对同一个词（`"customer"`）使用微妙不同的概念，接收方的 agent 基于错误解释行动，没有 schema 验证器能捕获它。FIPA 的本体论要求会在解析时拒绝该消息。

不走全本体论路线的缓解措施：

- 对 `content` 使用 JSON Schema——在线路层拒绝结构错误。
- 类型化制品（A2A）——拒绝错误的模态。
- 信封中的显式施为动词——即使内容是自然语言也能使意图明确。

### 2026 年规范，映射到言语行为遗产

| 现代规范 | FIPA 类比 | 保留了什么 | 丢弃了什么 |
|---|---|---|---|
| MCP `tools/call` | `request` | 显式意图，关联 ID | 形式语义，本体论 |
| MCP `resources/read` | `query-ref` | 显式意图，关联 ID | 形式语义 |
| A2A Task lifecycle | contract-net + request-when | 异步生命周期，状态转换 | 形式完备性保证 |
| A2A streaming events | subscribe/notify | 异步推送 | 类型化谓词订阅 |
| CA-MCP shared context | blackboard (Hayes-Roth 1985) | 多写者共享记忆 | 逻辑一致性模型 |
| NLIP | natural-language content | LLM 原生 | schema |

从上到下读这个表，模式是：保留结构原语，丢弃形式主义，让 LLM 掩盖歧义。

## Build It

`code/main.py` 实现了一个纯 stdlib 的 FIPA-ACL 翻译器。它编码和解码规范的 ACL 信封，展示每个 MCP / A2A 消息形状如何归结为相同的七个字段。演示：

- 将五条 MCP 风格和 A2A 风格的消息编码为 FIPA-ACL。
- 将 FIPA-ACL 解码回现代等价物。
- 使用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 在一个管理者和三个投标者之间运行一个玩具合同网谈判。

运行：

```
python3 code/main.py
```

输出是一个并排追踪，显示每条现代消息的 2026 JSON 形式和 FIPA-ACL 形式，然后是合同网投标的往返。相同的协议原语在往返中存活；只有语法不同。

## Use It

`outputs/skill-fipa-mapper.md` 是一个技能，读取任何 agent 协议规范并产出 FIPA-ACL 映射。在采用新协议之前使用它来回答："这是真正的新东西，还是带 JSON 语法的 `inform`？"

## Ship It

不要把 FIPA-ACL 带回来。带回它的检查清单：

- 每条消息的意图原语（施为动词）是什么？
- 是否有用于请求-响应和取消的关联 ID？
- 是否有显式的内容语言（JSON-RPC、纯文本、结构化类型制品）？
- 交互协议是一等公民，还是你在从头重新实现合同网？
- 当两个 agent 对内容含义产生分歧时会发生什么（语义漂移）？

在将任何新协议投入生产之前，记录这五个问题的答案。

## 练习

1. 运行 `code/main.py`。观察往返编码。识别哪个 FIPA 施为动词对应 `tools/call`、`resources/read` 和 A2A 任务创建。
2. 用 `cancel` 施为动词扩展合同网演示，让管理者可以在投标中途撤回任务。`cancel` 解决了什么重试无法解决的失败场景？
3. 阅读 FIPA ACL Message Structure (http://www.fipa.org/specs/fipa00037/) 第 4.1–4.3 节。选一个本课未涵盖的施为动词，描述其现代 JSON-RPC 类比。
4. 阅读 Liu 等人，arXiv:2505.02279。对于 MCP、A2A、ACP、ANP 各自列出它们保留和丢弃的 FIPA 施为动词族。
5. 为你自己系统中 `request` 施为动词的 `content` 字段设计一个最小 JSON-Schema。该 schema 给你什么纯自然语言给不了的，代价是什么？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 言语行为 | "做事的话语" | Austin/Searle：话语即行动。ACL 的理论母体。 |
| FIPA | "那个老 XML 东西" | IEEE 智能物理代理基金会。2000 年标准化了 ACL。 |
| ACL | "Agent 通信语言" | FIPA 的信封格式：施为动词 + 内容 + 元数据。 |
| Performative | "动词" | 消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | "FIPA 的前身" | 知识查询与操作语言（1993）。更简单，更窄。 |
| 本体论 | "共享词汇表" | 内容语言所讨论概念的形式定义。 |
| SL0 / SL1 | "FIPA 内容语言" | 语义语言级别 0 和 1——形式内容语言族。 |
| 合同网 | "任务市场" | 管理者发出 cfp；投标者提议；管理者接受。规范的交互协议。 |
| 交互协议 | "消息模式" | 具有已知正确性的施为动词序列：条件请求、订阅通知等。 |

## 延伸阅读

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 连接现代规范与 FIPA 遗产的 2025 规范综述
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 2000 年批准的信封格式
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 完整施为动词目录
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — `request`/`query-ref` 的现代工具使用等价物
- [A2A specification](https://a2a-protocol.org/latest/specification/) — 合同网和订阅通知的现代 agent 对等等价物
