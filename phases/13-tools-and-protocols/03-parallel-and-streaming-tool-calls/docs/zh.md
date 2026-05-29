# 并行工具调用与工具 Streaming

> 三个独立的天气查询串行执行就是三次往返。并行运行它们，总时间坍缩为最慢的单次调用。现在每个前沿提供商都能在单个回合中发出多个工具调用。收益是实实在在的；管道工作是微妙的。本课讲解两个方面：并行扇出和流式参数重组，重点是 id 关联陷阱。

**类型：** 构建
**语言：** Python（标准库，线程池 + streaming 框架）
**前置课程：** Phase 13 · 02（function calling 深入）
**时长：** 约 75 分钟

## 学习目标

- 解释 `parallel_tool_calls: true` 为什么存在以及何时应该禁用它。
- 在并行扇出期间将流式参数块关联到正确的工具调用 id。
- 将部分 `arguments` 字符串重组为完整 JSON，而不提前解析。
- 运行一个三城市天气基准测试，展示串行与并行的延迟差异。

## 问题

没有并行调用时，一个回答"班加罗尔、东京和苏黎世的天气怎么样"的 agent 会这样做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM 往返，每次还要加上执行器延迟。大约是理想挂钟时间的 4 倍。

有并行调用时：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

一次 LLM 往返。执行器时间是三者的最大值，而非总和。OpenAI、Anthropic 和 Gemini 上的生产基准测试显示扇出工作负载的挂钟时间减少 60% 到 70%。

代价是关联复杂性。当三个调用乱序完成时，你的结果必须携带匹配的 `tool_call_id`，这样模型才能对齐它们。当结果流式传输时，你必须将部分参数片段组装成完整 JSON 后才能执行。Gemini 3 添加唯一 id 部分是为了解决两个对同一工具的并行调用无法区分的实际问题。

## 概念

### 启用并行

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设为 `false` 强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false`（Claude 3.5 及以上默认）启用并行。设为 `true` 强制串行。
- **Gemini。** 始终具备并行能力；`tool_config.function_calling_config.mode = "AUTO"` 让模型决定。

在以下情况禁用并行：工具有顺序依赖（`create_file` 然后 `write_file`）、一个调用的输出是另一个的输入、或速率限制器无法处理扇出。

### Id 关联

模型发出的每个调用都有一个 `id`。宿主返回的每个结果必须包含相同的 id。没有这个，结果就是模糊的。

- **OpenAI。** 每条 tool 角色消息上的 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块上的 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上；Gemini 2 按名称匹配，对同名并行调用会出错）。

### 并发运行调用

宿主在自己的线程、协程或远程 worker 上运行每个调用的执行器。最简单的框架使用线程池；生产环境使用 asyncio 的 `asyncio.gather` 或结构化并发。完成顺序不可预测 — id 是标识符。

一个常见 bug：按调用列表顺序而非完成顺序回复结果。这通常能工作，因为模型只关心 `tool_call_id`，但如果结果被丢弃或重复，乱序提交会使调试更困难。优先按完成顺序回复，带上明确的 id。

### 流式工具调用

当模型流式输出时，`arguments` 分片到达。三个并行调用的三个独立块流在线上交错。你需要每个 id 一个累加器。

各提供商的形状：

- **OpenAI。** 每个块是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。块携带 `index`（在调用列表中的位置）。你按 index 累积，在 `id` 首次出现时读取它，在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流事件是 `message_start`，然后每个块一个 `content_block_start`，类型为 `tool_use`（包含 id、name、空 input）。`content_block_delta` 事件携带 `input_json_delta` 块。`content_block_stop` 关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）发出带有 `functionCallId` 的块，使调用可以干净地交错。Gemini 3 之前，streaming 一次返回一个完整调用。

### 部分 JSON 和提前解析陷阱

你不能在 `arguments` 完成之前解析它。部分 JSON 如 `{"city": "Beng` 是无效的，会抛出异常。正确的门控是提供商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`、或 Gemini 的流结束事件。只有在那时才尝试 `json.loads`。更健壮的方法使用增量 JSON 解析器，在结构完成时产出事件；OpenAI 的 streaming 指南推荐这种方式用于显示实时"思考中"指示器的 UX。花括号计数作为完整性测试是不可靠的（引号字符串或转义内容中的花括号会导致误报），只应作为非正式的调试启发式使用。

### 乱序完成

```
call_A: 快速 API，最先返回
call_B: 慢速 API，第二个返回
call_C: 中等 API，第三个返回
```

宿主回复仍必须引用 id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

回复中的顺序对 OpenAI 或 Anthropic 的正确性无关紧要。Gemini 接受任何顺序，只要 id 匹配。

### 基准测试：串行 vs 并行

`code/main.py` 中的框架模拟三个执行器，延迟分别为 400、600 和 800 毫秒。串行运行总计 1800 毫秒。并行运行为 max(400, 600, 800) = 800 毫秒。差异是常数而非比例，所以节省随工具数量增长。

实际注意事项：并行调用会给下游 API 带来压力。对限速服务的 10 路扇出会失败。Phase 13 · 17 讲解网关级背压；重试语义计划在未来的 phase 中讲解。

### Streaming 扇出挂钟时间

如果模型本身是流式的，你可以在一个调用的参数完成后立即开始执行，而不是等待所有调用完成。这是 OpenAI 记录但并非所有 SDK 都暴露的优化。本课的框架就是这样做的：一旦模拟流产出一个完整的参数对象，宿主就启动该调用。

## 动手试试

`code/main.py` 有两个部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 串行和并行运行三个模拟天气调用，并打印挂钟时间。第二部分重放一个假的 streaming 响应 — 三个并行调用的 `arguments` 块交错在一个流上 — 并用 `StreamAccumulator` 按 id 重组它们。没有 LLM，没有网络，只有重组逻辑。

关注点：

- 串行计时器达到 1.8 秒。并行计时器在相同的假延迟下达到 0.8 秒。
- 累加器通过按 id 缓冲来处理乱序到达的块，只在每个调用的 JSON 完整时才解析。
- 执行器在一个 id 的参数完成后立即启动，而不是在所有流结束后。

## 交付物

本课产出 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，该技能审计哪些工具可以安全并行化、哪些有顺序依赖、哪些会压垮下游速率限制 — 返回一个修订后的注册表，带有每个工具的 `parallel_safe` 标志。

## 练习

1. 运行 `code/main.py` 并改变模拟延迟。确认并行与串行的比率大约是 `max/sum`（实际运行因线程调度、序列化和框架开销而略有偏差）。在什么延迟分布下并行不再重要？

2. 扩展累加器以处理"调用在流中途被取消"的情况，丢弃其缓冲区并发出 `cancelled` 事件。哪个提供商明确记录了这种情况？检查 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池。对两者进行基准测试。你应该看到 async 的小幅优势，因为上下文切换成本更低，但只有在执行器做真实 I/O 时才明显。

4. 选择两个不应该并行化的工具（例如 `create_file` 然后 `write_file`）。在注册表中添加一个 `ordering_dependency` 图，并在该图上门控并行扇出。这是依赖感知调度的最小机制，未来的 agent 工程 phase 会正式化。

5. 阅读 OpenAI 的并行 function-calling 部分和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 推荐禁用并行的一种实际工具类型。（提示：对同一资源的有后果的变更。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Parallel tool calls（并行工具调用） | "一个回合中的扇出" | 模型在单条 assistant 消息中发出多个工具调用 |
| `parallel_tool_calls` | "OpenAI 的标志" | 启用或禁用多调用发出 |
| `disable_parallel_tool_use` | "Anthropic 的反向标志" | 退出标志；默认启用并行 |
| Tool call id | "关联句柄" | 结果消息必须回显的每调用标识符 |
| Accumulator（累加器） | "流缓冲区" | 用于部分 `arguments` 块的每 id 字符串缓冲区 |
| Out-of-order completion（乱序完成） | "最快的先到" | 并行调用以不可预测的顺序完成；id 是粘合剂 |
| Dependency graph（依赖图） | "顺序约束" | 输出馈入其他工具输入的工具；不能并行化 |
| Parse-early trap（提前解析陷阱） | "JSON.parse 爆炸了" | 尝试解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | "Gemini 3 特性" | 带有每调用唯一 id 的流式参数块 |
| Completion-order reply（完成顺序回复） | "不要等所有的" | 结果到达时按 id 回复 |

## 延伸阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 默认行为和退出标志
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 和结果批处理
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 的 id 关联并行调用
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 流的块参数重组
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — 带 `input_json_delta` 的 `content_block_delta`
