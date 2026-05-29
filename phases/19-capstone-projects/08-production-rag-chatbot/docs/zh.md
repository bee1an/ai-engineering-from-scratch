# 毕业项目 08 — 面向受监管行业的生产级 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年都运行相同的生产形态。用 docling 或 Unstructured 加 ColPali 处理视觉内容进行摄入。混合搜索。用 bge-reranker-v2-gemma 重排序。用 Claude Sonnet 4.7 带 prompt caching 在 60-80% 命中率下合成。用 Llama Guard 4 和 NeMo Guardrails 守护。用 Langfuse 和 Phoenix 监控。用 RAGAS 在 200 问黄金集上评分。在一个受监管领域（法律、临床、保险）中构建一个，毕业项目就是通过黄金集、红队测试和漂移仪表板。

**类型：** 毕业项目
**语言：** Python（管道 + API），TypeScript（聊天 UI）
**前置要求：** Phase 5（NLP）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段：** P5 · P7 · P11 · P12 · P17 · P18
**时间：** 30 小时

## 问题

受监管领域的 RAG（法律合同、临床试验方案、保险条款）是 2026 年最多部署的生产形态，因为 ROI 显而易见且风险具体。Harvey（Allen & Overy）为法律构建了它。Mendable 发布了开发者文档版本。Glean 覆盖企业搜索。模式是：高保真摄入，混合检索加重排序，带引用强制和 prompt caching 的合成，多层安全守护，持续监控漂移。

难点不在模型。而在于管辖区感知的合规（HIPAA、GDPR、SOC2）、引用级可审计性、成本控制（prompt caching 在命中率高时买到 60-90% 折扣）、通过 RAGAS faithfulness 检测幻觉，以及源文档更新而索引未跟上时的漂移检测。这个毕业项目要求你在 200 问黄金集上交付所有这些，并附带红队套件。

## 概念

管道有两侧。**摄入**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富的文档；分块获得摘要、标签和基于角色的访问标签。向量进入 pgvector + pgvectorscale（5000 万向量以下）或 Qdrant Cloud；稀疏 BM25 并行运行。**对话**：LangGraph 处理记忆和多轮；每个查询运行混合检索，用 bge-reranker-v2-gemma-2b 重排序，用 Claude Sonnet 4.7（prompt-cached）合成，输出通过 Llama Guard 4 和 NeMo Guardrails，发出带引用锚点的响应。

评估栈有四层。**黄金集**（200 个带引用的标注 Q/A）用于正确性。**红队**（越狱、PII 提取尝试、领域外问题）用于安全。**RAGAS** 用于每轮自动评估 faithfulness / answer relevance / context precision。**漂移仪表板**（Arize Phoenix）每周监控检索质量和幻觉评分。

Prompt caching 是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存 system prompt + 检索上下文。在 60-80% 命中率下，每查询成本降低 3-5 倍。管道必须为稳定前缀（system prompt + 重排序上下文在前）而设计，以实现高缓存命中率。

## 架构

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- 摄入：Unstructured.io 或 docling 用于结构化文档；ColPali 用于视觉丰富的 PDF
- 向量数据库：pgvector + pgvectorscale（5000 万向量以下）；否则 Qdrant Cloud
- 稀疏：Tantivy BM25 带字段权重
- 编排：LlamaIndex Workflows（摄入）+ LangGraph（对话）
- 重排序器：自托管 bge-reranker-v2-gemma-2b 或托管 Voyage rerank-2
- LLM：Claude Sonnet 4.7 带 prompt caching；备选自托管 Llama 3.3 70B
- 评估：RAGAS 0.2 在线，DeepEval 用于幻觉和越狱套件
- 可观测性：自托管 Langfuse 带标注队列；Arize Phoenix 用于漂移
- 护栏：Llama Guard 4 输入/输出分类器，NeMo Guardrails v0.12 策略，Presidio PII 清洗
- 合规：分块上的基于角色访问标签；GDPR/HIPAA 管辖区标签

## 构建步骤

1. **摄入。** 用 Unstructured 或 docling 解析你的语料（认真构建需要 1000-10000 个文档）。对扫描/视觉密集页面，路由到 ColPali。产出带摘要、角色标签、管辖区标签的分块。

2. **索引。** 稠密嵌入（Voyage-3 或 Nomic-embed-v2）进入 pgvector + pgvectorscale。BM25 侧索引通过 Tantivy。角色和管辖区过滤器作为 payload。

3. **混合检索。** 先按角色+管辖区过滤；然后并行 dense + BM25；用 reciprocal rank fusion 合并；top-20 送重排序器；top-5 送合成。

4. **带 prompt caching 合成。** System prompt + 静态策略在缓存头中；重排序上下文作为缓存扩展；用户问题作为未缓存后缀。目标稳态 60-80% 缓存命中率。

5. **护栏。** Llama Guard 4 对输入；NeMo Guardrails rails 阻止领域外问题或策略禁止的话题；Presidio 清洗输出中意外的 PII；引用强制后置过滤器。

6. **黄金集。** 200 个由领域专家标注的 Q/A 对，带（答案、引用）。对智能体评分：精确引用匹配、答案正确性、faithfulness（RAGAS）。

7. **红队。** 50 个对抗性提示：越狱（PAIR、TAP）、PII 提取尝试、领域外、跨管辖区泄露。用通过/失败和严重性评分。

8. **漂移仪表板。** Arize Phoenix 每周跟踪检索质量（nDCG、引用 faithfulness）。下降 5% 时告警。

9. **成本报告。** Langfuse：prompt-caching 命中率、每查询 token、按阶段的 $/query 分解。

## 使用示例

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付标准

`outputs/skill-production-rag.md` 描述交付物。一个带合规标签的受监管领域聊天机器人，通过评分标准，带实时漂移监控。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | RAGAS faithfulness + answer relevance | 黄金集（200 Q/A）上的在线评分 |
| 20 | 引用正确性 | 有可验证源锚点的回答比例 |
| 20 | 护栏覆盖 | Llama Guard 4 通过率 + 越狱套件结果 |
| 20 | 成本/延迟工程 | Prompt-cache 命中率、p95 延迟、$/query |
| 15 | 漂移监控仪表板 | Phoenix 实时仪表板带每周检索质量趋势 |
| **100** | | |

## 练习

1. 在不同管辖区下构建第二个语料切片（例如 HIPAA 与 GDPR 并存）。在 20 问跨管辖区探测上演示角色+管辖区过滤防止交叉泄露。

2. 衡量一周生产流量中的 prompt-cache 命中率。识别哪些查询破坏了缓存前缀。重构。

3. 添加带 10k-token 摘要缓冲区的多轮记忆。衡量对话增长时 faithfulness 是否下降。

4. 将 Claude Sonnet 4.7 换成自托管的 Llama 3.3 70B。衡量 $/query 和 faithfulness 差异。

5. 添加"不确定"模式：如果 top 重排序分数低于阈值，智能体说"我没有可靠的引用"而非回答。衡量误置信度降低。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Prompt caching | "缓存 system + context" | Claude/OpenAI 特性：缓存前缀 token 命中时打 60-90% 折扣 |
| RAGAS | "RAG 评估器" | 自动评分 faithfulness、answer relevance、context precision |
| 黄金集 | "标注评估" | 200+ 专家标注的带引用 Q/A；真值 |
| 管辖区标签 | "合规标签" | 附加到分块的 GDPR/HIPAA/SOC2 范围；由检索过滤器强制执行 |
| 引用忠实度 | "有据回答率" | 有可检索源片段支撑的声明比例 |
| 漂移 | "检索质量衰减" | nDCG 或引用评分的每周变化；告警阈值 5% |
| 红队 | "对抗性评估" | 发布前的越狱、PII 提取、领域外探测 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — 参考法律生产栈
- [Glean enterprise search](https://www.glean.com) — 参考企业级 RAG
- [Mendable documentation](https://mendable.ai) — 开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管摄入
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆参考
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — 标准 RAG 评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 参考漂移可观测性
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略 rail 框架
