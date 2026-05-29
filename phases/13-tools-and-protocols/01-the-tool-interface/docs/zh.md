# 工具接口 — 为什么 Agent 需要结构化 I/O

> 语言模型产出 token，程序执行动作。两者之间的鸿沟就是工具接口：一份让模型请求动作、宿主执行动作的契约。2026 年的每个技术栈 — OpenAI、Anthropic、Gemini 的 function calling；MCP 的 `tools/call`；A2A 的 task parts — 都是同一个四步循环的不同编码。本课命名这个循环，并展示运行它所需的最小机制。

**类型：** 学习
**语言：** Python（标准库，无 LLM）
**前置课程：** Phase 11（LLM completion API）
**时长：** 约 45 分钟

## 学习目标

- 解释为什么一个只能生成文本的 LLM 无法独立对真实世界执行动作。
- 画出四步工具调用循环（describe → decide → execute → observe），并说明每一步由谁负责。
- 将工具描述写成三部分：名称、JSON Schema 输入、确定性执行器函数。
- 区分纯工具和有副作用的工具，并说明这种区分对安全性的意义。

## 问题

LLM 输出的是下一个 token 的概率分布，这就是它全部的输出面。如果你问一个聊天模型"班加罗尔现在天气怎么样"，它能写出一句看似合理的话，但它无法调用天气 API。那句话可能碰巧正确，也可能已经过时三天了。

弥合这个鸿沟就是工具接口的目的。宿主程序 — 你的 agent 运行时、Claude Desktop、ChatGPT、Cursor 或自定义脚本 — 向模型广播一组可调用的工具。当模型判断需要执行动作时，它会发出一个结构化载荷，指定工具名称和参数。宿主解析该载荷，真正运行工具，并将结果回传。循环持续进行，直到模型决定不再需要更多调用。

这份契约的第一个版本于 2023 年 6 月以 OpenAI 的 "functions" 参数形式发布。Anthropic 随后在 Claude 2.1 中加入了 `tool_use` 块。Gemini 几个月后添加了 `functionDeclarations`。现在每个提供商都暴露相同的形状：输入是 JSON-Schema 类型的工具列表，输出是 JSON 载荷的工具调用。Model Context Protocol（2024 年 11 月）将这份契约泛化，使一个工具注册表服务于所有模型。A2A（2026 年 4 月，v1.0）在同一原语之上叠加了 agent 间委托。

四步循环是所有这些之下的不变量。Phase 13 的其余内容都是对它的展开。

## 概念

### 第一步：describe（描述）

宿主用三个字段声明每个工具。

- **名称。** 稳定的、机器可读的标识符。`get_weather`，而不是"天气那个东西"。
- **描述。** 一段自然语言简介。"当用户询问某个城市的当前天气时使用。不要用于历史数据。"
- **输入 schema。** 一个 JSON Schema 对象（draft 2020-12），描述工具的参数。

模型接收这个列表。现代提供商使用特定于提供商的模板将这些声明序列化到 system prompt 中，所以作为调用者你只需处理结构化形式。

### 第二步：decide（决策）

给定用户消息和可用工具，模型选择三种行为之一。

1. **直接用文本回答。** 不调用工具。
2. **调用一个或多个工具。** 发出结构化调用对象。在 `parallel_tool_calls: true`（OpenAI 和 Gemini 默认开启，Anthropic 需手动开启）下，模型可以在一个回合中发出多个调用。
3. **拒绝。** Strict mode 结构化输出可以产生一个类型化的 `refusal` 块而非调用。

工具调用载荷有三个稳定字段：调用 `id`、工具 `name` 和 JSON `arguments` 对象。id 的存在是为了让宿主能将后续结果与特定调用关联起来，这在并行调用乱序返回时很重要。

### 第三步：execute（执行）

宿主接收调用，根据声明的 schema 验证参数，然后运行执行器。无效参数意味着模型幻觉了一个字段或使用了错误类型 — 这在弱模型上是非常常见的失败模式。生产环境的宿主对无效参数做三件事之一：快速失败并将错误暴露给模型、用受约束的解析器修复 JSON、或将验证错误包含在 prompt 中重试模型。

执行器本身是普通代码。Python、TypeScript、shell 命令、数据库查询。它产生一个结果，通常是字符串，但可以是任何 JSON 值或结构化内容块（MCP 中的文本、图像或资源引用）。结果必须可序列化。

### 第四步：observe（观察）

宿主将工具结果追加到对话中（作为带有匹配 `id` 的 `tool` 角色消息），然后重新调用模型。模型现在在上下文中有了工具输出，可以产生最终答案或请求更多调用。这个过程持续到模型停止发出调用或宿主达到迭代次数的安全限制。

### 信任分界

工具分为两种对安全性有影响的类型。

- **纯工具。** 只读、确定性、无副作用。`get_weather`、`search_docs`、`get_current_time`。可以安全地推测性调用。
- **有后果的工具。** 修改状态、花费金钱、触及用户数据。`send_email`、`delete_file`、`execute_trade`。必须设置门控。

Meta 2026 年的 agent 安全"二选二规则"说：单个回合最多组合以下三者中的两个：不可信输入、敏感数据、有后果的动作。工具接口就是你执行这条规则的地方 — 通过拒绝调用、要求用户确认或升级权限范围。详见 Phase 13 · 15 的完整安全章节和 Phase 14 · 09 的 agent 级权限策略。

### 循环在哪里运行

| 场景 | 谁描述 | 谁决策 | 谁执行 |
|------|--------|--------|--------|
| 单轮 function calling（OpenAI/Anthropic/Gemini） | 应用开发者 | LLM | 应用开发者 |
| MCP | MCP server | LLM（通过 MCP client） | MCP server |
| A2A | Agent Card 发布者 | 调用方 agent | 被调用 agent |
| Web 浏览器（function-calling agent） | 浏览器扩展 / WebMCP | LLM | 浏览器运行时 |

到处都是相同的四步。列名变了；结构没变。

### 为什么不直接让模型输出 JSON？

"让模型以 JSON 格式回复"是 function calling 之前的模式。它在前沿模型上大约 5% 到 15% 的时间会失败，在小模型上失败率更高。失败模式包括缺少花括号、尾随逗号、幻觉字段和错误类型。然后你需要一个 JSON 修复步骤、一次重试或一个受约束的解码器。

原生 function calling 更好，原因有三。第一，提供商端到端地训练模型以匹配精确的调用形状，所以有效 JSON 率在 strict mode 下攀升到 98% 到 99%。第二，调用载荷位于自己的协议槽中，不在自由文本内 — 所以工具调用永远不会泄漏到用户可见的回复中。第三，提供商通过受约束解码强制 schema 合规（OpenAI 的 strict mode、Anthropic 的 `tool_use`、Gemini 的 `responseSchema`）。输出保证通过验证。

Phase 13 · 02 并排展示三个提供商的 API。Phase 13 · 04 深入结构化输出。

### 熔断器

循环在模型停止发出调用或宿主达到最大回合数时终止。生产环境的宿主将此设置为 5 到 20 个回合。超过这个数，你几乎可以确定处于模型无法退出的循环中。Claude Code 默认 20；OpenAI Assistants 默认 10；Cursor 的 agent 模式默认 25。

另一种选择 — 无界循环 — 每六个月就会以"agent 一夜之间花了 400 美元 API 费用"的事后分析形式出现。不要在没有上限的情况下上线。

Phase 14 · 12 深入讲解错误恢复和自愈；Phase 17 讲解生产环境的速率限制。

### Phase 13 接下来的内容

- 第 02 到 05 课打磨提供商级别的工具调用表面。
- 第 06 到 14 课将循环泛化为 MCP。
- 第 15 到 18 课防御循环免受恶意服务器、对抗性用户和未认证远程认证面的攻击。
- 第 19 到 22 课将模式扩展到 agent 间协作、可观测性、路由和打包。
- 第 23 课使用所有原语交付一个完整的生态系统。

后续每一课都是这个四步循环的展开。把它作为不变量记在心里。

## 动手试试

`code/main.py` 在没有 LLM 的情况下运行四步循环。一个假的"决策器"函数通过模式匹配用户消息来模拟模型；执行器、schema 验证器和观察步骤的框架是真实的。运行它可以看到完整的请求/响应编排和可打印的中间状态，然后在后续课程中用任何真实提供商替换假决策器。

关注点：

- 工具注册表为每个工具保存三个字段：名称、描述、schema 和一个执行器引用。
- 验证器是一个最小的 JSON Schema 子集（types、required、enum、min/max），仅使用标准库编写。Phase 13 · 04 提供更完整的版本。
- 循环将迭代次数限制在五次。生产环境的 agent 正需要这种熔断器。

## 交付物

本课产出 `outputs/skill-tool-interface-reviewer.md`。给定一个工具定义草案（名称 + 描述 + schema + 执行器概要），该技能审计其循环适配性：名称是否机器稳定、描述是否是完整的使用简介、schema 是否正确使用 JSON Schema 2020-12、纯工具与有后果工具的分类是否明确。

## 练习

1. 在 `code/main.py` 中添加第四个工具 `get_stock_price(ticker)`。将其描述写为"当用户通过股票代码询问当前股价时使用。不要用于历史价格或市场摘要。"运行框架并确认假决策器将提到股票代码的查询路由到新工具。

2. 破坏 schema 验证器。传入一个 `arguments` 对象缺少必填字段的调用，确认宿主在执行前拒绝它。然后传入一个带有额外未知字段的调用。决定：宿主应该拒绝还是忽略？用安全性论证来证明你的选择。

3. 将框架中的每个工具分类为纯工具或有后果的工具。在需要的注册表条目中添加 `consequential: true` 标志，并修改循环使其在选择有后果的工具时打印"would confirm with user"行。这就是每个生产环境宿主需要的确认门控的形状。

4. 在纸上画出四步循环，用上面的提供商列表填入你最喜欢的客户端（Claude Desktop、Cursor、ChatGPT 或自定义栈）。与 Phase 13 · 06 中 MCP 特定的变体交叉参考。

5. 从头到尾阅读 OpenAI 的 function-calling 指南。找出一个存在于请求中但不在本课呈现的四步循环中的字段。解释它添加了什么以及为什么它是方便的而非必要的。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Tool（工具） | "模型能调用的东西" | 名称 + JSON-Schema 类型输入 + 执行器函数的三元组 |
| Function calling | "原生工具使用" | 提供商级别的 API 支持，用于发出结构化工具调用而非散文 |
| Tool call（工具调用） | "模型的动作请求" | 模型发出的带有 `id`、`name`、`arguments` 的 JSON 载荷 |
| Tool result（工具结果） | "工具返回了什么" | 执行器的输出，包装在带有匹配 id 的 `tool` 角色消息中 |
| Parallel tool calls（并行工具调用） | "一次多个调用" | 一个模型回合中的多个调用对象，独立且可通过 id 排序 |
| Strict mode | "保证的 JSON" | 受约束解码，强制模型输出通过声明的 schema 验证 |
| 纯工具 | "只读工具" | 无副作用；可安全重新运行 |
| 有后果的工具 | "动作工具" | 修改外部状态；需要门控、审计或用户确认 |
| 四步循环 | "工具调用周期" | describe → decide → execute → observe |
| Host（宿主） | "Agent 运行时" | 持有工具注册表、调用模型并运行执行器的程序 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — OpenAI 风格工具声明和调用形状的权威参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — Claude 的 `tool_use` / `tool_result` 块格式
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 中的 `functionDeclarations` 和并行调用语义
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 工具接口的提供商无关泛化
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 每个现代工具 API 使用的 schema 方言
