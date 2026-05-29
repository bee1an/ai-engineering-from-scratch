# 工具使用与 Function Calling

> Toolformer（Schick et al., 2023）开创了自监督工具标注。Berkeley Function Calling Leaderboard V4（Patil et al., 2025）设定了 2026 的标准：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination。单轮已经解决。记忆、动态决策和长程工具链还没有。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## 学习目标

- 解释 Toolformer 的自监督训练信号：只保留执行后能降低下一 token loss 的工具标注。
- 说出 BFCL V4 的五个评估类别及各自衡量的内容。
- 用 stdlib 实现一个带 schema 验证、参数强制转换和执行沙箱的工具注册表。
- 诊断 2026 三个开放问题：长程工具链、动态决策和记忆。

## 问题

早期工具使用问的是：模型能预测一个正确的函数调用吗？现代工具使用问的是：模型能跨 40 步链式调用工具吗？带记忆、带部分可观测性、能从工具失败中恢复、不幻觉出不存在的工具？

Toolformer 建立了基线：模型可以通过自监督学习何时调用工具。BFCL V4 定义了 2026 的评估目标。两者之间的差距就是生产智能体所处的空间。

## 核心概念

### Toolformer（Schick et al., NeurIPS 2023）

思路：让模型在自己的预训练语料上标注候选 API 调用。对每个候选执行它。只有当包含工具结果能降低下一 token 的 loss 时才保留标注。在过滤后的语料上微调。

覆盖的工具：计算器、QA 系统、搜索引擎、翻译器、日历。自监督信号纯粹关于工具是否有助于预测文本 — 无人工标签。

规模效应：工具使用在规模上涌现。小模型因工具标注而受损；大模型则获益。这就是为什么 2026 前沿模型内置了强大的工具使用能力，而大多数 7B 模型需要显式的工具使用微调才能可靠。

### Berkeley Function Calling Leaderboard V4（Patil et al., ICML 2025）

BFCL 是 2026 事实上的评估标准。V4 组成：

- **Agentic（40%）** — 完整智能体轨迹：记忆、多轮、动态决策。
- **Multi-Turn（30%）** — 带工具链的交互式对话。
- **Live（10%）** — 用户提交的真实 prompt（更难的分布）。
- **Non-Live（10%）** — 合成测试用例。
- **Hallucination（10%）** — 检测何时不应调用任何工具。

V3 引入了基于状态的评估：在工具序列之后，检查 API 的实际状态（如"文件是否已创建？"）而不是匹配工具调用的 AST。V4 增加了 web 搜索、记忆和格式敏感性类别。

2026 关键发现：单轮 function calling 接近解决。失败集中在记忆（跨轮携带上下文）、动态决策（基于先前结果选择工具）、长程链（20+ 步后的漂移）和幻觉检测（没有合适工具时拒绝调用）。

### 工具 schema

每个提供商都有 schema。细节不同但形状相同：

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic 直接使用 `input_schema`。OpenAI 使用 `function.parameters`。两者都接受 JSON Schema。描述是承重部件 — 模型读它们来选择正确的工具。糟糕的工具描述是选错工具的第一大根因。

### 参数验证

不要信任任何工具调用。验证：

1. **类型强制转换。** 模型可能在 schema 说 int 的地方返回字符串 "5"。无歧义时强制转换；有歧义时拒绝。
2. **枚举验证。** 如果 schema 说 `status in {"open", "closed"}` 而模型输出 `"in_progress"`，用描述性错误拒绝。
3. **必填字段。** 缺少必填字段 -> 立即将错误观察返回给模型，而不是崩溃。
4. **格式验证。** 日期、邮箱、URL — 用具体的解析器验证，不用正则。

每个验证失败都应返回结构化观察，使模型可以用正确的形状重试。

### 并行工具调用

现代提供商支持在一个 assistant turn 中并行调用工具。流程：

1. 模型输出 3 个带不同 `tool_use_id` 的工具调用。
2. 运行时执行它们（如果独立则并行）。
3. 每个结果作为 `tool_result` block 通过 `tool_use_id` 关联返回。

工程规则：将关联 ID 视为承重部件。交换它们就会导致错误的工具对应错误的结果。

### 沙箱

工具执行是沙箱边界。详见 Lesson 09。简短版本：每个工具应指定读/写面、网络访问、超时、内存上限。通用的 `run_shell(cmd)` 是红旗；具体的 `git_status()` 更安全。

## Build It

`code/main.py` 实现了一个生产形态的工具注册表：

- JSON Schema 子集验证器（仅 stdlib）。
- 带描述、输入 schema、超时和执行器的工具注册。
- 参数强制转换和枚举验证。
- 带关联 ID 的并行工具分发。
- 结构化字符串形式的错误观察。

运行：

```
python3 code/main.py
```

trace 展示一个迷你智能体在一个 turn 中调用三个工具，其中一个故意格式错误的调用被拒绝并返回模型可以据此行动的描述性错误。

## Use It

每个提供商都有自己的工具 schema — Anthropic、OpenAI、Gemini、Bedrock。如果需要多提供商支持，使用翻译层（OpenAI Agents SDK、Vercel AI SDK、LangChain tool adapter）。BFCL 是参考基准 — 如果工具使用是产品的核心，发布前对你的智能体运行它。

## Ship It

`outputs/skill-tool-registry.md` 为给定任务领域生成工具目录、schema 和注册表。包含描述质量检查（每个工具的描述是否告诉模型何时使用它？）。

## 练习

1. 添加一个"no-op"工具，让模型可以显式拒绝使用任何其他工具。在类 BFCL 的幻觉测试上衡量效果。
2. 实现 int-as-string 和 float-as-string 的参数强制转换。强制转换从什么时候开始隐藏真正的 bug？
3. 添加每工具超时和熔断器（连续 3 次失败后拒绝该工具 60 秒）。这对模型的恢复方式有什么改变？
4. 阅读 BFCL V4 描述。选一个类别（如"multi-turn"）并通过你的智能体运行 10 个示例 prompt。报告通过率。
5. 将 stdlib 验证器移植到 Pydantic 或 Zod。Pydantic/Zod 捕获了哪些玩具版本遗漏的问题？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Function calling | "工具使用" | 带验证 schema 的结构化输出工具调用 |
| Toolformer | "自监督工具标注" | Schick 2023 — 保留结果能降低下一 token loss 的工具调用 |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 基准：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination |
| Tool schema | "模型的函数签名" | name、description、参数的 JSON Schema |
| tool_use_id | "关联 ID" | 将工具调用与其结果绑定；对并行分发至关重要 |
| Hallucination detection | "知道何时不调用" | V4 类别：没有合适工具时拒绝调用 |
| 参数强制转换 | "String-to-int 修复" | 对可预测的 schema 不匹配做窄修复；有歧义时拒绝 |
| 沙箱 | "工具执行边界" | 每工具的读/写面、网络、超时、内存上限 |

## 延伸阅读

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — 自监督工具标注
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — 2026 评估基准
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Agent SDK 中的生产工具 schema
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — function tool 类型和 Guardrails
