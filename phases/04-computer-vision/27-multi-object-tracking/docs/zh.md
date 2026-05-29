# 多目标跟踪与视频记忆

> 跟踪就是检测加关联。每帧检测。将当前帧的检测与上一帧的轨迹按 ID 匹配。

**类型：** 动手构建
**语言：** Python
**前置课程：** Phase 4 Lesson 06（YOLO 检测）、Phase 4 Lesson 08（Mask R-CNN）、Phase 4 Lesson 24（SAM 3）
**时长：** 约 60 分钟

## 学习目标

- 区分 tracking-by-detection 和基于查询的跟踪，列出算法家族（SORT、DeepSORT、ByteTrack、BoT-SORT、SAM 2 memory tracker、SAM 3.1 Object Multiplex）
- 从零实现 IoU + 匈牙利算法分配，用于经典的 tracking-by-detection
- 解释 SAM 2 的 memory bank 以及为什么它比基于 IoU 的关联更好地处理遮挡
- 读懂三个跟踪指标（MOTA、IDF1、HOTA），并为给定用例选择合适的指标

## 问题背景

检测器告诉你单帧中物体在哪里。跟踪器告诉你帧 `t` 中的哪个检测与帧 `t-1` 中的哪个检测是同一个物体。没有这个，你无法计数穿过线的物体、跟踪穿过遮挡的球，或知道"4 号车已经在车道上 8 秒了"。

跟踪对每个面向视频的产品都至关重要：体育分析、监控、自动驾驶、医学视频分析、野生动物监测、标记计数。核心构建块是共享的：逐帧检测器、运动模型（卡尔曼滤波器或更丰富的模型）、关联步骤（在 IoU / 余弦 / 学到的特征上的匈牙利算法）、以及轨迹生命周期（诞生、更新、死亡）。

2026 年带来了两种新模式：**SAM 2 基于记忆的跟踪**（特征记忆代替运动模型关联）和 **SAM 3.1 Object Multiplex**（同一概念多实例的共享记忆）。本课先走经典栈，再讲基于记忆的方法。

## 核心概念

### Tracking-by-detection

```mermaid
flowchart LR
    F1["Frame t"] --> DET["Detector"] --> D1["Detections at t"]
    PREV["Tracks up to t-1"] --> PREDICT["Motion predict<br/>(Kalman)"]
    PREDICT --> PRED["Predicted tracks at t"]
    D1 --> ASSOC["Hungarian assignment<br/>(IoU / cosine / motion)"]
    PRED --> ASSOC
    ASSOC --> UPDATE["Update matched tracks"]
    ASSOC --> NEW["Birth new tracks"]
    ASSOC --> DEAD["Age unmatched tracks; delete after N"]
    UPDATE --> NEXT["Tracks at t"]
    NEW --> NEXT
    DEAD --> NEXT

    style DET fill:#dbeafe,stroke:#2563eb
    style ASSOC fill:#fef3c7,stroke:#d97706
    style NEXT fill:#dcfce7,stroke:#16a34a
```

2026 年你会遇到的每个跟踪器都是这个循环的变体。差异在于：

- **SORT**（2016）：卡尔曼滤波器 + IoU 匈牙利。简单、快速、无外观模型。
- **DeepSORT**（2017）：SORT + 每轨迹的 CNN 外观特征（ReID 嵌入）。更好地处理交叉。
- **ByteTrack**（2021）：将低置信度检测作为第二阶段关联；不需要外观特征但在 MOT17 上表现顶尖。
- **BoT-SORT**（2022）：Byte + 相机运动补偿 + ReID。
- **StrongSORT / OC-SORT** —— ByteTrack 后代，更好的运动和外观。

### 一段话讲清卡尔曼滤波器

卡尔曼滤波器为每条轨迹维护状态 `(x, y, w, h, dx, dy, dw, dh)` 及协方差。每帧先用匀速模型**预测**状态，再用匹配的检测**更新**。当预测不确定性高时，更新更信任检测。这给出平滑轨迹和在短遮挡（1-5 帧）中延续轨迹的能力。

每个经典跟踪器在运动预测步骤中都使用卡尔曼滤波器。

### 匈牙利算法

给定一个 `M x N` 代价矩阵（轨迹 x 检测），找到最小化总代价的一对一分配。代价通常是 `1 - IoU(track_bbox, detection_bbox)` 或外观特征的负余弦相似度。运行时间 O((M+N)^3)；对于 M, N 到约 1000，通过 `scipy.optimize.linear_sum_assignment` 在 Python 中足够快。

### ByteTrack 的核心思想

标准跟踪器丢弃低置信度检测（< 0.5）。ByteTrack 将它们保留为**第二阶段候选**：在将轨迹与高置信度检测匹配后，未匹配的轨迹尝试用稍宽松的 IoU 阈值匹配低置信度检测。恢复短遮挡、人群附近的 ID 切换。

### SAM 2 基于记忆的跟踪

SAM 2 通过保持逐实例时空特征的 **memory bank** 来处理视频。给定一帧上的 prompt（点击、框、文本），它将实例编码到记忆中。在后续帧上，记忆与新帧特征做交叉注意力，解码器为同一实例在新帧中产出 mask。

无卡尔曼滤波器，无匈牙利算法。关联隐含在记忆-注意力操作中。

优点：
- 对大遮挡鲁棒（记忆跨多帧携带实例身份）。
- 结合 SAM 3 的文本 prompt 时支持开放词汇。
- 无需单独的运动模型。

缺点：
- 多目标跟踪时比 ByteTrack 慢。
- Memory bank 增长；限制上下文窗口。

### SAM 3.1 Object Multiplex

之前的 SAM 2 / SAM 3 跟踪为每个实例保持独立的 memory bank。50 个对象就是 50 个 memory bank。Object Multiplex（2026 年 3 月）将它们折叠为一个共享记忆加**逐实例查询 token**。成本随实例数亚线性增长。

Multiplex 是 2026 年人群跟踪的新默认：演唱会人群、仓库工人、交通路口。

### 三个必知指标

- **MOTA（Multi-Object Tracking Accuracy）** —— 1 - (FN + FP + ID switches) / GT。按错误类型加权；单一指标混合了检测和关联失败。
- **IDF1（ID F1）** —— ID 精确率和召回率的调和平均。专注于每条真值轨迹在时间上保持其 ID 的程度。对 ID 切换敏感的任务优于 MOTA。
- **HOTA（Higher Order Tracking Accuracy）** —— 分解为检测精度（DetA）和关联精度（AssA）。2020 年以来的社区标准；最全面。

监控（谁是谁）：报告 IDF1。体育分析（计数传球）：HOTA。通用学术对比：HOTA。

## 动手构建

### Step 1：基于 IoU 的代价矩阵

```python
import numpy as np


def bbox_iou(a, b):
    """
    a, b: (N, 4) arrays of [x1, y1, x2, y2].
    Returns (N_a, N_b) IoU matrix.
    """
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

### Step 2：最小 SORT 风格跟踪器

为简洁省略了固定匀速卡尔曼——这里我们使用简单的 IoU 关联；生产中卡尔曼预测是必需的。`sort` Python 包提供完整版本。

```python
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, tid, bbox, frame):
        self.id = tid
        self.bbox = bbox
        self.last_frame = frame
        self.hits = 1

    def update(self, bbox, frame):
        self.bbox = bbox
        self.last_frame = frame
        self.hits += 1


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.tracks = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def step(self, detections, frame):
        if not self.tracks:
            for d in detections:
                self.tracks.append(Track(self.next_id, d, frame))
                self.next_id += 1
            return [(t.id, t.bbox) for t in self.tracks]

        track_boxes = np.array([t.bbox for t in self.tracks])
        det_boxes = np.array(detections) if len(detections) else np.empty((0, 4))

        iou = bbox_iou(track_boxes, det_boxes) if len(det_boxes) else np.zeros((len(track_boxes), 0))
        cost = 1 - iou
        cost[iou < self.iou_threshold] = 1e6

        matched_track = set()
        matched_det = set()
        if cost.size > 0:
            row, col = linear_sum_assignment(cost)
            for r, c in zip(row, col):
                if cost[r, c] < 1.0:
                    self.tracks[r].update(det_boxes[c], frame)
                    matched_track.add(r); matched_det.add(c)

        for i, d in enumerate(det_boxes):
            if i not in matched_det:
                self.tracks.append(Track(self.next_id, d, frame))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if frame - t.last_frame <= self.max_age]
        return [(t.id, t.bbox) for t in self.tracks]
```

60 行。接收逐帧检测，返回逐帧轨迹 ID。真实系统会加上卡尔曼预测、ByteTrack 的第二阶段重匹配和外观特征。

### Step 3：合成轨迹测试

```python
def synthetic_frames(num_frames=20, num_objects=3, H=240, W=320, seed=0):
    rng = np.random.default_rng(seed)
    starts = rng.uniform(20, 200, size=(num_objects, 2))
    velocities = rng.uniform(-5, 5, size=(num_objects, 2))
    frames = []
    for f in range(num_frames):
        dets = []
        for i in range(num_objects):
            cx, cy = starts[i] + f * velocities[i]
            dets.append([cx - 10, cy - 10, cx + 10, cy + 10])
        frames.append(dets)
    return frames


tracker = SimpleTracker()
for f, dets in enumerate(synthetic_frames()):
    tracks = tracker.step(dets, f)
```

三个沿直线运动的物体应该在所有 20 帧中保持其 ID。

### Step 4：ID 切换指标

```python
def count_id_switches(tracks_per_frame, gt_per_frame):
    """
    tracks_per_frame:  list of list of (track_id, bbox)
    gt_per_frame:      list of list of (gt_id, bbox)
    Returns number of ID switches.
    """
    prev_assignment = {}
    switches = 0
    for tracks, gts in zip(tracks_per_frame, gt_per_frame):
        if not tracks or not gts:
            continue
        t_boxes = np.array([b for _, b in tracks])
        g_boxes = np.array([b for _, b in gts])
        iou = bbox_iou(g_boxes, t_boxes)
        for g_idx, (gt_id, _) in enumerate(gts):
            j = iou[g_idx].argmax()
            if iou[g_idx, j] > 0.5:
                t_id = tracks[j][0]
                if gt_id in prev_assignment and prev_assignment[gt_id] != t_id:
                    switches += 1
                prev_assignment[gt_id] = t_id
    return switches
```

这是一个简化的类 IDF1 指标：计算真值对象更换其分配的预测轨迹 ID 的次数。真正的 MOTA / IDF1 / HOTA 工具在 `py-motmetrics` 和 `TrackEval` 中。

## 实际使用

2026 年的生产跟踪器：

- `ultralytics` —— YOLOv8 + ByteTrack / BoT-SORT 内置。`results = model.track(source, tracker="bytetrack.yaml")`。默认选择。
- `supervision`（Roboflow）—— ByteTrack 封装加标注工具。
- SAM 2 / SAM 3.1 —— 通过 `processor.track()` 的基于记忆的跟踪。
- 自定义栈：检测器（YOLOv8 / RT-DETR）+ `sort-tracker` / `OC-SORT` / `StrongSORT`。

选择：

- 行人/汽车/箱子 30+ fps：**ByteTrack with ultralytics**。
- 人群中同一类的多实例：**SAM 3.1 Object Multiplex**。
- 有可辨识外观的重度遮挡：**DeepSORT / StrongSORT**（ReID 特征）。
- 体育/复杂交互：**BoT-SORT** 或学习型跟踪器（MOTRv3）。

## 交付产出

本课产出：

- `outputs/prompt-tracker-picker.md` —— 根据场景类型、遮挡模式和延迟预算，在 SORT / ByteTrack / BoT-SORT / SAM 2 / SAM 3.1 之间选择。
- `outputs/skill-mot-evaluator.md` —— 编写一个完整的评估工具，对真值轨迹计算 MOTA / IDF1 / HOTA。

## 练习

1. **（简单）** 用 3、10 和 30 个物体运行上面的合成跟踪器。报告每种情况的 ID 切换次数。找出简单 IoU-only 关联开始失败的位置。
2. **（中等）** 在关联前添加匀速卡尔曼预测步骤。展示短（2-3 帧）遮挡不再导致 ID 切换。
3. **（困难）** 集成 SAM 2 的基于记忆的跟踪器（通过 `transformers`）作为替代跟踪后端。在 30 秒人群视频上同时运行 SimpleTracker 和 SAM 2，手动标注 5 个显著人物的真值 ID，比较 ID 切换次数。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Tracking-by-detection | "检测后关联" | 逐帧检测器 + 在 IoU / 外观上的匈牙利分配 |
| Kalman filter | "运动预测" | 线性动力学 + 协方差，用于平滑轨迹预测和遮挡处理 |
| Hungarian algorithm | "最优分配" | 求解最小代价二部图匹配问题；`scipy.optimize.linear_sum_assignment` |
| ByteTrack | "低置信度第二轮" | 将未匹配轨迹与低置信度检测重新匹配以恢复短遮挡 |
| DeepSORT | "SORT + 外观" | 添加 ReID 特征用于跨帧匹配；更好地保持 ID |
| Memory bank | "SAM 2 技巧" | 跨帧存储的逐实例时空特征；交叉注意力替代显式关联 |
| Object Multiplex | "SAM 3.1 共享记忆" | 单一共享记忆加逐实例查询，用于快速多目标跟踪 |
| HOTA | "现代跟踪指标" | 分解为检测精度和关联精度；社区标准 |

## 延伸阅读

- [SORT (Bewley et al., 2016)](https://arxiv.org/abs/1602.00763) — 最小 tracking-by-detection 论文
- [DeepSORT (Wojke et al., 2017)](https://arxiv.org/abs/1703.07402) — 添加外观特征
- [ByteTrack (Zhang et al., 2022)](https://arxiv.org/abs/2110.06864) — 低置信度第二轮
- [BoT-SORT (Aharon et al., 2022)](https://arxiv.org/abs/2206.14651) — 相机运动补偿
- [HOTA (Luiten et al., 2020)](https://arxiv.org/abs/2009.07736) — 分解式跟踪指标
- [SAM 2 video segmentation (Meta, 2024)](https://ai.meta.com/sam2/) — 基于记忆的跟踪器
- [SAM 3.1 Object Multiplex (Meta, March 2026)](https://ai.meta.com/blog/segment-anything-model-3/)
