# 毕业项目 12 — 视频理解管道（场景、问答、搜索）

> Twelve Labs 将 Marengo + Pegasus 产品化。VideoDB 发布了 CRUD-for-video API。AI2 的 Molmo 2 发布了开源 VLM checkpoint。Gemini 长上下文原生处理数小时视频。TimeLens-100K 定义了大规模时间定位。2026 年的管道已经定型：场景分割、每场景字幕 + 嵌入、转录对齐、多向量索引，以及用 (start, end) 时间戳加帧预览回答的查询。这个毕业项目要求摄入 100 小时，达到公开基准，并衡量计数和动作问题上的幻觉。

**类型：** 毕业项目
**语言：** Python（管道），TypeScript（UI）
**前置要求：** Phase 4（CV）、Phase 6（语音）、Phase 7（Transformer）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 17（基础设施）
**涉及阶段：** P4 · P6 · P7 · P11 · P12 · P17
**时间：** 30 小时

## 问题

长视频问答是 2026 年规模下带宽需求最大的多模态问题。Gemini 2.5 Pro 可以原生读取 2 小时视频，但将 100 小时视频摄入可查询语料仍需要场景级索引。生产形态结合了场景分割（TransNetV2 或 PySceneDetect）、用 VLM 做每场景字幕（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、转录对齐（Whisper-v3-turbo 带词级时间戳），以及并排存储字幕、帧嵌入和转录的多向量索引。查询管道用 (start, end) 时间戳加帧预览回答。

基准是公开的（ActivityNet-QA、NeXT-GQA）加你自己的 100 问自定义集。计数和动作类型问题上的幻觉是已知的困难失败类别；毕业项目明确衡量它。

## 概念

摄入时三个管道并行运行。**场景分割**将视频切成场景。**VLM 字幕**为每个场景生成字幕和从关键帧提取的帧嵌入。**ASR 对齐**产出词级时间戳。三个流通过 (scene_id, time range) 连接。每个场景在多向量索引（Qdrant）中获得三种向量类型：字幕嵌入、关键帧嵌入、转录嵌入。

查询时，自然语言问题对所有三种向量触发；结果用 RRF 合并；时间定位适配器（TimeLens 风格）在 top 场景内细化 (start, end) 窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）接收查询 + top 场景 + 裁剪帧并用引用时间戳和帧预览回答。

幻觉衡量很重要。计数（"有多少人进入房间？"）和动作类型（"厨师是先倒还是先搅？"）问题出了名地不可靠。将准确率与描述性问题分开报告。

## 架构

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- 场景分割：TransNetV2（2024-26 最先进）或 PySceneDetect
- ASR：Whisper-v3-turbo via faster-whisper 带词级时间戳
- VLM 字幕器 + 回答器：Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2
- 时间定位：TimeLens-100K 训练的适配器或 VideoITG
- 索引：Qdrant 多向量支持（caption / frame / transcript）
- UI：Next.js 15 带 HTML5 视频播放器和场景缩略图
- 评估：ActivityNet-QA、NeXT-GQA、自定义 100 问手工标注集
- 幻觉基准：计数和动作类型子集带手工标签

## 构建步骤

1. **摄入遍历器。** 接受 YouTube URL 或本地 MP4。如需降到 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect 产出 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6k-8k 场景。

3. **ASR 阶段。** 对音频运行 Whisper-v3-turbo；导出词级时间戳；按场景切分转录片段。

4. **VLM 字幕。** 每场景调用 Gemini 2.5 Pro（或 Qwen3-VL-Max），传入关键帧和简短字幕模板。产出字幕 + 帧嵌入。

5. **多向量索引。** Qdrant collection 带三个命名向量。Payload：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题触发三个稠密查询；用 reciprocal rank fusion 合并；top-k=5 场景。

7. **时间定位。** 对 top 场景运行 TimeLens 风格适配器，在场景内细化 (start, end) 窗口。

8. **VLM 合成。** 用查询 + top-3 场景片段（作为图像或短片段）+ 转录调用 Gemini 2.5 Pro。要求 `(video_id, start_ms, end_ms)` 引用。

9. **评估。** 运行 ActivityNet-QA 和 NeXT-GQA。构建 100 问自定义集。报告总体准确率 + 按类别分解（计数、动作、描述性）。

## 使用示例

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付标准

`outputs/skill-video-qa.md` 是交付物。给定一个 YouTube URL 或上传视频，管道索引场景并用带时间戳引用的方式回答问题。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 时间定位 IoU | 保留定位集上的 Intersection-over-union |
| 20 | QA 准确率 | NeXT-GQA 和自定义 100 问 |
| 20 | 摄入吞吐量 | 每美元处理的视频小时数 |
| 20 | UI 和引用体验 | 时间戳链接、缩略图条、跳转到帧 |
| 15 | 幻觉率 | 计数和动作类型准确率单独报告 |
| **100** | | |

## 练习

1. 在字幕阶段将 Gemini 2.5 Pro 换成 Qwen3-VL-Max。在人工评分的 50 场景样本上报告字幕质量差异。

2. 将每场景帧嵌入减少为一个池化向量而非多向量。衡量检索退化。

3. 构建"严格计数"模式：合成器提取每个被计数实例及其时间戳，用户点击验证。衡量用户验证是否减少幻觉。

4. 基准测试摄入成本：三种 VLM 选择下的每美元视频小时数。选出最佳平衡点。

5. 添加说话人分离转录：对音频运行 pyannote 说话人分离并嵌入每说话人转录。演示"Alice 关于 X 说了什么？"查询。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 场景分割 | "镜头检测" | 在镜头边界处将视频切成场景 |
| 多向量索引 | "Caption + frame + transcript" | Qdrant collection 带每种表示的命名向量 |
| 时间定位 | "具体什么时候发生的" | 为查询答案细化 (start, end) 窗口 |
| 帧嵌入 | "视觉表示" | 关键帧的向量嵌入；用于场景视觉相似度 |
| RRF 融合 | "Reciprocal rank fusion" | 跨多个排序列表的合并策略；经典混合检索技巧 |
| 计数幻觉 | "数错" | VLM 在"有多少 X"问题上的已知失败模式 |
| ActivityNet-QA | "视频问答基准" | 长视频问答准确率基准 |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开源 VLM checkpoint
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 大规模时间定位
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — 托管参考
- [VideoDB](https://videodb.io) — CRUD-for-video API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商业参考
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开源替代
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评估基准
