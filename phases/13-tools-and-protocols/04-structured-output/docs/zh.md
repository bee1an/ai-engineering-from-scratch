# 结构化输出 — JSON Schema、Pydantic、Zod、受约束解码

> "好好让模型返回 JSON"在前沿模型上也有 5% 到 15% 的失败率。结构化输出通过受约束解码弥合了这个差距：模型被从字面上阻止发出违反 schema 的 token。OpenAI 的 strict mode、Anthropic 的 schema 类型化 tool use、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type`、以及 Zod 的 `.parse` 是同一思想的五种表面形式。本课构建 schema 验证器和 strict-mode 契约，学习者将在每个生产提取管道中使用它们。

**类型：** 构建
**语言：** Python（标准库，JSON Schema 2020-12 子集）
**前置课程：** Phase 13 · 02（function calling 深入）
**时长：** 约 75 分钟

## 学习目标

- 使用正确的约束（enum、min/max、required、pattern）为提取目标编写 JSON Schema 2020-12。
- 解释为什么 strict mode 和受约束解码提供的保证不同于"生成后验证"。
- 区分三种失败模式：解析错误、schema 违规、模型拒绝。
- 交付一个带有类型化修复和类型化拒绝处理的提取管道。

## 问题

一个读取采购订单邮件的 agent 需要将自由文本转换为 `{customer, line_items, total_usd}`。三种方法。

**方法一：提示输出 JSON。** "以 JSON 格式回复，包含 customer、line_items、total_usd 字段。"在前沿模型上 85% 到 95% 的时间有效。以六种方式失败：缺少花括号、尾随逗号、错误类型、幻觉字段、在 token 限制处截断、泄漏散文如"这是你的 JSON："。

**方法二：生成后验证。** 自由生成，解析，根据 schema 验证，失败时重试。可靠但昂贵 — 每次重试都要付费，截断 bug 每次出现都多花一个回合。

**方法三：受约束解码。** 提供商在解码时强制 schema。无效 token 从采样分布中被屏蔽。输出保证可解析且保证通过验证。失败坍缩为一种模式：拒绝（模型判断输入不适合 schema）。

2026 年每个前沿提供商都提供某种形式的方法三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}` 加上模型拒绝时响应中的 `refusal`。
- **Anthropic。** 对 `tool_use` 输入的 schema 强制；`stop_reason: "refusal"` 不存在，但 `end_turn` 且无工具调用是信号。
- **Gemini。** 请求级别的 `responseSchema`；2026 年 Gemini 为选定类型提供 token 级语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 发出类型化为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。** 运行时解析器，根据 Zod schema 验证提供商输出；与 OpenAI 的 `beta.chat.completions.parse` 配对。

共同线索：声明一次 schema，端到端强制。

## 概念

### JSON Schema 2020-12 — 通用语言

每个提供商都接受 JSON Schema 2020-12。你最常用的构造：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到子 schema 的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的封闭集合。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子 schema。
- `additionalProperties`：`false` 禁止额外字段（默认值因模式而异）。

OpenAI strict mode 添加三个要求：每个属性必须列在 `required` 中、到处都是 `additionalProperties: false`、没有未解析的 `$ref`。如果你违反这些，API 在请求时返回 400。

### Pydantic，Python 绑定

Pydantic v2 通过 `model_json_schema()` 从 dataclass 形状的模型生成 JSON Schema。Pydantic AI 包装了这一点，所以你写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

agent 框架在边缘将 schema 翻译为 OpenAI strict mode、Anthropic `input_schema` 或 Gemini `responseSchema`。模型的输出作为类型化的 `Invoice` 实例返回。验证错误抛出带有类型化错误路径的 `ValidationError`。

### Zod，TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TS 等价物。OpenAI 的 Node SDK 暴露 `zodResponseFormat(Invoice)`，将其翻译为 API 的 JSON Schema 载荷。

### 拒绝

Strict mode 不能强制模型回答。如果输入不适合 schema（"邮件是一首诗，不是发票"），模型发出包含原因的 `refusal` 字段。你的代码必须将此作为一等结果处理，而非失败。拒绝也可用作安全信号：被要求从受保护内容邮件中提取信用卡号的模型会返回附带安全原因的拒绝。

### 开放权重中的受约束解码

开放权重实现使用三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从 schema 构建确定性有限自动机；在每一步，屏蔽会违反 FSM 的 token 的 logits。
2. **带 JSON 解析器的 logit 屏蔽**：与模型同步运行流式 JSON 解析器；在每一步，计算有效的下一个 token 集合。
3. **带验证器的推测解码**：廉价的草稿模型提议 token，验证器强制 schema。

商业提供商在幕后选择其中之一。2026 年的技术水平：对于短结构化输出比普通生成更快，对于长输出大致相同速度。

### 三种失败模式

1. **解析错误。** 输出不是有效 JSON。在 strict mode 下不可能发生。在非 strict 提供商上仍可能发生。
2. **Schema 违规。** 输出可解析但违反 schema。在 strict mode 下不可能发生。在其外很常见。
3. **拒绝。** 模型拒绝。必须作为类型化结果处理。

### 重试策略

当你在 strict mode 之外（Anthropic tool use、非 strict OpenAI、旧版 Gemini）时，恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

一次重试通常就够了。三次重试捕获弱模型的偶发问题。超过三次是 schema 有问题的信号：模型对某些输入无法满足它，prompt 或 schema 需要修复。

### 小模型支持

受约束解码在小模型上有效。一个 3B 参数的开放模型加语法强制在结构化任务上优于 70B 参数模型的原始提示。这是结构化输出对生产重要的主要原因：它将可靠性与模型大小解耦。

## 动手试试

`code/main.py` 提供一个最小的标准库 JSON Schema 2020-12 验证器（types、required、enum、min/max、pattern、items、additionalProperties）。它包装一个 `Invoice` schema 并通过验证器运行假 LLM 输出，演示解析错误、schema 违规和拒绝路径。在生产中用任何提供商的真实响应替换假输出。

关注点：

- 验证器返回带有路径和消息的类型化 `[ValidationError]` 列表。这就是你想暴露给重试 prompt 的形状。
- 拒绝分支不重试。它记录日志并返回类型化拒绝。Phase 14 · 09 将拒绝用作安全信号。
- `additionalProperties: false` 检查在对抗性测试输入上触发，展示为什么 strict mode 关闭了幻觉字段的大门。

## 交付物

本课产出 `outputs/skill-structured-output-designer.md`。给定一个自由文本提取目标（发票、支持工单、简历等），该技能产出一个 strict-mode 兼容的 JSON Schema 2020-12 和一个镜像它的 Pydantic 模型，带有类型化拒绝和重试处理的桩代码。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，其 `total_usd` 是负数。确认验证器以 `minimum` 约束路径拒绝它。

2. 扩展验证器以支持带鉴别器的 `oneOf`。常见情况：`line_item` 要么是产品要么是服务，由 `kind` 标记。Strict mode 在这里有微妙的规则；查看 OpenAI 的结构化输出指南。

3. 将相同的 Invoice schema 写成 Pydantic BaseModel，并将 `model_json_schema()` 输出与你手写的 schema 比较。找出 Pydantic 默认设置而手写版本遗漏的那个字段。

4. 测量拒绝率。构造十个不应该可提取的输入（一首歌词、一个数学证明、一封空邮件），通过真实提供商的 strict mode 运行它们。计算拒绝数与幻觉输出数。这是你拒绝感知重试的基准真相。

5. 从头到尾阅读 OpenAI 的结构化输出指南。找出它在 strict mode 中明确禁止但普通 JSON Schema 允许的那个构造。然后设计一个非本质地使用该禁止构造的 schema，并将其重构为 strict 兼容的。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| JSON Schema 2020-12 | "Schema 规范" | 每个现代提供商使用的 IETF 草案 schema 方言 |
| Strict mode | "保证的 schema" | OpenAI 标志，通过受约束解码强制 schema |
| Constrained decoding（受约束解码） | "Logit 屏蔽" | 解码时强制，屏蔽无效的下一个 token |
| Refusal（拒绝） | "模型拒绝" | 输入不适合 schema 时的类型化结果 |
| Parse error（解析错误） | "无效 JSON" | 输出未能解析为 JSON；strict 下不可能 |
| Schema violation（Schema 违规） | "形状错误" | 已解析但违反 types / required / enum / range |
| `additionalProperties: false` | "不允许额外字段" | 禁止未知字段；OpenAI strict 中必需 |
| Pydantic BaseModel | "类型化输出" | 发出和验证 JSON Schema 的 Python 类 |
| Zod schema | "TypeScript 输出类型" | 用于提供商输出验证的 TS 运行时 schema |
| Grammar enforcement（语法强制） | "开放权重受约束解码" | 基于 FSM 的 logit 屏蔽，如 outlines / guidance |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict mode、拒绝和 schema 要求
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月发布文章，解释解码保证
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) — 序列化到每个提供商的类型化 output_type 绑定
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 权威规范
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明和 strict-mode 注意事项
