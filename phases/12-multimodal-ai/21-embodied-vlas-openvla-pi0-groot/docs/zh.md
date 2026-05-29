# 具身智能 VLA：RT-2、OpenVLA、π0、GR00T

> 第一次有模型从网站上读取食谱并在厨房机器人中执行是 RT-2（Google DeepMind，2023 年 7 月）。RT-2 将动作离散化为文本 token，在网络数据加机器人动作数据上联合微调 VLM，证明了网络规模的视觉-语言知识可以迁移到机器人控制。OpenVLA（2024 年 6 月）发布了开源 7B 参考实现。Physical Intelligence 的 π0 系列（2024-2025）添加了 flow-matching 动作专家。NVIDIA 的 GR00T N1（2025 年 3 月）为人形机器人大规模交付了双系统（System 1 / System 2）控制。VLA 原语——vision-language-action，一个看、读、行动的单一模型——是本阶段理解模型与 Phase 15 自主系统之间的桥梁。

**Type:** Learn
**Languages:** Python (stdlib, action tokenizer + VLA inference skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 15 (Autonomous Systems, referenced)
**Time:** ~180 minutes

## 学习目标

- 描述动作 tokenization：离散 bin 编码（RT-2）、FAST 高效动作 token、连续 flow-matching 动作（π0）。
- 解释为什么在网络 + 机器人数据上联合微调能保留通用知识迁移到新任务。
- 对比 OpenVLA（开源 7B Llama+VLM）、π0（flow-matching）和 GR00T N1（双系统）在同一机器人任务上的表现。
- 说出 Open X-Embodiment 数据集及其作为 RT-X 训练语料的角色。

## 问题

一个能从自然语言指令做家务的机器人自 1970 年代以来就是研究目标。2020 年代的答案：vision-language-action (VLA) 模型。与用于 VQA 的 VLM 架构相同，但输出是动作（关节力矩、末端执行器位姿、离散命令）而非文本。

VLA 特有的挑战：

1. 动作空间是连续的（关节角度、力）且高维的（7-DOF 手臂 + 3-DOF 夹爪 = 10 维，30 Hz）。
2. 机器人专属训练数据稀缺。Open X-Embodiment 有约 1M 轨迹；网络文本-图像有 5B+。
3. 控制频率很重要。30 Hz 控制循环意味着每个动作 33ms 预算。
4. 安全。错误动作损坏硬件、人员或财产。

## 核心概念

### 动作 tokenization（RT-2）

RT-2 的技巧：将每个关节目标表示为量化文本 token。将归一化的 [-1, 1] 范围离散化为 256 个 bin，将每个 bin 映射到词表 ID。一个 10-DOF 动作在每个控制步变成 10 个 token。

在混合数据上联合微调 PaLM-X VLM：

- 网络图文对（captioning、VQA）。
- 机器人演示，动作作为 token。

模型看到"拿起红色方块"（语言）→ 图像（视觉）→ 10-token 动作序列（离散化关节目标）。网络预训练保留了通用知识迁移：RT-2 可以遵循"移向快速移动的物体"即使"快速移动"不在训练数据中。

RT-2 论文中推理频率 3-5 Hz，受限于 VLM 自回归解码。

### OpenVLA——开源 7B 参考

OpenVLA（Kim et al.，2024 年 6 月）是开源权重的 RT-2 等价物。7B Llama backbone，DINOv2 + SigLIP 双视觉编码器，256 bin 动作 tokenization。

在 Open X-Embodiment（970k 轨迹，跨 22 个机器人）上训练。附带 LoRA 微调支持以适配新机器人。

推理：量化后在 A100 上 4-5 Hz。对慢速操作足够快，对高频控制不够。

### FAST tokenizer——更快的动作解码

Pertsch et al. (2024) 表明离散 bin tokenization 效率低——大多数动作聚集在 bin 空间的小区域。FAST（Frequency-domain Action Sequence Tokenizer）通过 DCT 压缩动作序列并量化系数。

30 步动作轨迹变成约 10 个 FAST token 而非 300 个离散 bin token。推理加速 3-5 倍且无质量损失。

### π0 和 flow-matching 动作

Physical Intelligence 的 π0（Black et al.，2024 年 10 月）用 flow-matching 动作专家替换了离散动作 token：

- 一个小型动作 transformer 读取 VLM 的 hidden states 并通过 rectified flow 输出连续 50 步动作序列。
- 动作 head 用 flow-matching loss 训练；VLM 预训练保持不变。
- 推理：完整动作序列在约 5 个去噪步中输出，有效实现 50 Hz 控制。

π0 的声明：在广泛的操作任务套件上超过 OpenVLA 和 Octo。连续动作公式保留了离散化破坏的平滑性。

π0.5 和 π0-FAST 是增量升级。π0-FAST 结合了 FAST tokenization 和 flow matching。

### GR00T N1——人形机器人的双系统

NVIDIA 的 GR00T N1（2025 年 3 月）为人形机器人（>30 DOF，全身）构建：

- System 2：大型 VLM 读取场景 + 指令，以约 1 Hz 产出高层子目标。
- System 1：小型动作 head transformer 以子目标为条件产出 50-100 Hz 低层关节命令。

这种拆分映射到 Kahneman 的快慢思考：System 2 规划，System 1 行动。好处：慢速 VLM 级规划不阻塞快速控制；System 1 保持小以降低延迟。

GR00T N1.7（2025 年底）改进了数据扩展。GR00T 用 Omniverse 的 sim-to-real 数据微调。

### Open X-Embodiment

训练数据。RT-X（2023 年 10 月）汇集了 22 个数据集，覆盖 22 个机器人的 1M 轨迹。Open X-Embodiment 是所有人使用的语料：

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table。
- 每个样本：（机器人状态、摄像头视角、指令、动作序列）。
- 训练卫生：统一动作空间、归一化关节范围、调整摄像头大小。

OpenVLA 和 π0 在 Open X-Embodiment 上训练。到任何特定机器人的域差距通过 100-1000 个任务专属演示的 LoRA 微调来弥合。

### 联合微调 vs 仅机器人

联合微调混合网络 VQA 数据和机器人轨迹。比例很重要：VQA 太多模型忘记动作；机器人数据太多模型丢失通用知识。

RT-2 的比例：约 1:1。OpenVLA：约 0.5:1 网络对机器人。π0：类似。精确比例是按数据集大小调优的超参数。

仅机器人训练产出任务专属模型，在分布外指令上失败。联合微调是"拿起红色方块（在演示中）"和"从左边拿起第三大的物体（新表述）"之间的区别。

### 安全和动作限制

每个生产级 VLA 都附带：

- 硬关节限制（不能超过规格力矩）。
- 速度限制（软裁剪）。
- 工作空间边界（末端执行器不能离开桌面）。
- 新任务的人在环审批。

这些作为控制层检查位于 VLA 之外。VLA 的输出是建议，不是命令。

## Use It

`code/main.py`：

- 实现 256-bin 动作 tokenization 和反 tokenization。
- 基于 DCT + 量化勾画 FAST tokenizer。
- 对比（离散 bin、FAST、连续 flow）每个动作步的 token 数。
- 打印 RT-2 → OpenVLA → π0 → GR00T 的谱系摘要。

## Ship It

本课产出 `outputs/skill-vla-action-format-picker.md`。给定一个机器人任务（操作、导航、人形全身），在离散 bin + RT-2、FAST + OpenVLA、flow-matching + π0 或双系统 + GR00T 之间选择。

## 练习

1. 10-DOF 手臂，30 Hz 控制率。256 bin 离散 tokenization 每秒输出多少 token？7B VLM 能跟上吗？

2. FAST tokenization 将 30 步轨迹压缩到约 10 个 token。如果轨迹有高频运动（如打鼓），用户会丢失什么？

3. π0 的 flow-matching head 在约 5 步去噪。与 OpenVLA 4-5 Hz 的自回归解码对比吞吐量。

4. GR00T 的 System 1 / System 2 拆分映射到 Kahneman。提出一个可能帮助双足行走的不同拆分（System 3？）。

5. 阅读 Open X-Embodiment Section 4 关于数据集整理。说出防止域泄漏的三条整理规则。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| VLA | "Vision-language-action" | 接受图像 + 指令并输出动作命令的模型 |
| Action tokenization | "离散 bin" | 将连续关节目标量化为每维 256 个 bin，每个是词表 ID |
| FAST tokenizer | "频域动作 token" | DCT + 量化将 30 步轨迹压缩到约 10 个 token |
| Co-fine-tune | "混合网络 + 机器人" | 在网络 VQA 数据和机器人演示上同时训练以保留通用知识 |
| Flow-matching action head | "π0 连续输出" | 通过 rectified flow 输出 50 步动作序列的小型 transformer |
| System 1 / System 2 | "双系统控制" | 大 VLM 慢速规划，小动作 head 快速行动；GR00T 模式 |
| Open X-Embodiment | "RT-X 数据集" | 1M 轨迹跨机器人数据集；训练语料 |

## 延伸阅读

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
