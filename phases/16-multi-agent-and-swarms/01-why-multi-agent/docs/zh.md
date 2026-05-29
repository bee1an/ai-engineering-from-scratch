# 为什么需要多智能体？

> 单个 agent 会撞墙。聪明的做法不是造一个更大的 agent，而是用更多 agent。

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## 学习目标

- 识别单 agent 天花板（上下文溢出、混合专业知识、串行瓶颈），并解释何时拆分为多 agent 是正确选择
- 比较编排模式（流水线、并行扇出、监督者、层级），并为给定任务结构选择合适的模式
- 设计一个具有清晰角色边界、共享状态和通信契约的多智能体系统
- 分析多 agent 复杂性（延迟、成本、调试难度）与单 agent 简洁性之间的权衡

## 问题

你在 Phase 14 构建了一个单 agent。它能工作——能读文件、跑命令、调 API、推理结果。然后你把它指向一个真实代码库：200 个文件、三种语言、依赖基础设施的测试，还要求在写代码之前先调研外部 API。

Agent 卡住了。不是因为 LLM 笨，而是因为任务超出了单个 agent 循环的处理能力。上下文窗口被文件内容塞满。Agent 忘了 40 次工具调用之前读过什么。它试图同时当研究员、程序员和审查员，结果三样都做得很差。

这就是单 agent 天花板。每当任务需要以下条件时你就会撞上它：

- **超出单个窗口容量的上下文** — 读 50 个文件就超过 200k token
- **不同阶段需要不同专业知识** — 研究和代码生成需要不同的提示策略
- **可以并行完成的工作** — 为什么要串行读三个文件，而不是同时读？

## 概念

### 单 Agent 天花板

单个 agent 就是一个循环、一个上下文窗口、一个系统提示。想象一下：

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

三个问题会出现：

1. **上下文饱和** — 工具结果不断堆积。到第 30 轮时，agent 已经消耗了 150k token 的文件内容、命令输出和先前推理。第 5 轮的关键细节已经丢失。

2. **角色混乱** — 一个系统提示说"你是研究员、程序员、审查员和测试员"，产出的 agent 半研究、半写码，审查永远做不完。

3. **串行瓶颈** — agent 读文件 A，然后文件 B，然后文件 C。三次串行 LLM 调用。三次串行工具执行。没有并行。

### 多 Agent 解决方案

拆分工作。给每个 agent 一个任务、一个上下文窗口、一个针对该任务调优的系统提示：

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个 agent 拥有：
- 聚焦的系统提示（"你是代码审查员。你唯一的工作是找 bug。"）
- 自己的上下文窗口（不被其他 agent 的工作污染）
- 清晰的输入/输出契约（接收研究笔记，输出代码）

### 真实系统案例

**Claude Code subagents** — 当 Claude Code 用 `Task` 生成子 agent 时，它创建了一个有限定任务的子 agent。父 agent 保持上下文干净。子 agent 做聚焦工作并返回摘要。

**Devin** — 运行一个规划 agent、一个编码 agent 和一个浏览器 agent。规划器把工作拆成步骤。编码器写代码。浏览器研究文档。每个都有独立上下文。

**多 agent 编码团队 (SWE-bench)** — SWE-bench 上表现最好的系统使用一个研究员读代码库、一个规划器设计修复方案、一个编码器实现。单 agent 系统得分更低。

**ChatGPT Deep Research** — 并行生成多个搜索 agent，每个探索不同角度，然后综合结果。

### 复杂度光谱

多 agent 不是二元的。它是一个光谱：

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**单 agent** — 一个循环，一个提示。适合简单任务。

**子 agent** — 父 agent 为聚焦子任务生成子 agent。父 agent 维护计划。子 agent 汇报。这就是 Claude Code 的做法。

**流水线** — agent 按顺序运行。Agent A 的输出成为 Agent B 的输入。适合分阶段工作流：研究 → 编码 → 审查 → 测试。

**团队** — agent 并行运行，共享消息总线。每个有角色。编排器协调。适合同时需要不同技能的场景。

**Swarm** — 许多相同或近似的 agent 共享状态。没有固定编排器。Agent 从队列中取工作。适合高吞吐并行任务。

### 四种多 Agent 模式

#### 模式 1：流水线

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个 agent 转换数据并向前传递。推理简单。一个阶段失败会阻塞后续。

#### 模式 2：扇出 / 扇入

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

将工作分散到并行 agent，然后合并结果。适合可分解为独立子任务的任务。

#### 模式 3：编排器-工作者

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

一个智能编排器决定做什么，委派给工作者，综合结果。编排器本身也是一个 agent，拥有生成工作者的工具。

#### 模式 4：对等 Swarm

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

没有中央编排器。Agent 点对点通信。决策从交互中涌现。更难调试，但可扩展到多个 agent。

### 何时不该用多 Agent

多 agent 增加复杂性。agent 之间的每条消息都是潜在故障点。调试从"读一个对话"变成"追踪五个 agent 之间的消息"。

**保持单 agent 的情况：**
- 任务适合一个上下文窗口（工作数据低于 ~100k token）
- 不同阶段不需要不同的系统提示
- 串行执行足够快
- 任务简单到拆分带来的开销大于价值

**复杂性成本：**
- 每个 agent 边界都是有损压缩步骤：agent A 的完整上下文被压缩成给 agent B 的一条消息
- 协调逻辑（谁做什么、何时、什么顺序）本身就是 bug 来源
- 延迟增加：N 个 agent 意味着至少 N 次串行 LLM 调用，如果需要来回对话则更多
- 成本倍增：每个 agent 独立消耗 token

经验法则：如果任务少于 20 次工具调用且适合 100k token，保持单 agent。

## Build It

### Step 1: 过载的单 Agent

这是一个试图做所有事情的单 agent。它有一个巨大的系统提示和一个包含研究、代码和审查的上下文窗口：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的问题：
- 上下文窗口随每个阶段增长。到审查步骤时，它包含研究笔记、代码和先前推理。
- 系统提示是通用的。无法为每个阶段调优。
- 没有并行。

### Step 2: 专家 Agent

现在拆分。每个 agent 做一件事：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个专家有聚焦的提示。每个获得干净的上下文窗口，只包含它需要的输入。

### Step 3: 通过消息协调

用显式消息传递将专家连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个 agent 只接收发给它的消息。没有上下文污染。研究员读文档的 50k token 永远不会进入审查员的上下文。

### Step 4: 对比

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

多 agent 版本使用更多总 token（三个 agent，三次独立 LLM 调用），但每个 agent 的上下文保持干净。每个阶段的质量提升，因为系统提示是专门化的。

## Use It

本课产出一个可复用的提示，用于决定何时使用多 agent。见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个专家：一个"测试员" agent，接收编码器的代码和审查员的反馈，然后编写测试
2. 修改流水线，让审查员可以将反馈发回给编码器进行修订循环（最多 2 轮）
3. 将串行流水线转换为扇出：并行运行研究员和"需求分析员" agent，然后合并输出再传给编码器

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| Swarm | "AI agent 蜂群" | 一组对等 agent，共享状态，没有固定领导者。行为从局部交互中涌现。 |
| 编排器 | "老板 agent" | 一个 agent，其工具包括生成和管理其他 agent。它规划和委派，但可能不做实际工作。 |
| 协调器 | "交通警察" | 一个非 agent 组件（通常只是代码，不是 LLM），根据规则在 agent 之间路由消息。 |
| 共识 | "agent 们达成一致" | 多个 agent 必须在继续之前达成一致的协议。用于需要解决冲突输出的场景。 |
| 涌现行为 | "agent 自己想出来的" | 从 agent 交互中产生但未被显式编程的系统级模式。可能有用也可能有害。 |
| 扇出 / 扇入 | "agent 版 Map-Reduce" | 将任务分散到并行 agent（扇出），然后合并结果（扇入）。 |
| 消息传递 | "agent 互相对话" | agent 之间的通信机制：从一个 agent 发送到另一个的结构化数据，替代共享上下文窗口。 |

## 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 多 agent 模式综述
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - 微软的多 agent 对话框架
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code 如何用 Task 委派
- [CrewAI documentation](https://docs.crewai.com/) - 基于角色的多 agent 框架
