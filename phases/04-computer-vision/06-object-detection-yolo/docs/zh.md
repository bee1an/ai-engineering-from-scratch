# 目标检测 — 从零实现 YOLO

> 检测就是分类加回归，在 feature map 的每个位置运行，然后用非极大值抑制清理。

**类型：** Build
**语言：** Python
**前置课程：** Phase 4 Lesson 03（CNN）、Phase 4 Lesson 04（图像分类）、Phase 4 Lesson 05（迁移学习）
**时长：** 约 75 分钟

## 学习目标

- 解释将检测转化为密集预测问题的 grid-and-anchor 设计，并说明输出张量中每个数字的含义
- 计算框之间的 Intersection-over-Union 并从零实现非极大值抑制
- 在预训练 backbone 之上构建一个最小的 YOLO 风格 head，包括分类、objectness 和框回归 loss
- 阅读检测指标行（precision@0.5、recall、mAP@0.5、mAP@0.5:0.95）并判断下一步该调哪个旋钮

## 问题

分类说"这张图是一只狗。"检测说"在像素 (112, 40, 280, 210) 有一只狗，在 (400, 180, 560, 310) 有一只猫，画面中没有其他东西。"这一个结构性变化——预测可变数量的带标签框而不是每张图一个标签——是每个自动驾驶系统、每个监控产品、每个文档版面解析器和每条工厂视觉线所依赖的。

检测也是视觉中每个工程权衡同时出现的地方。你想要准确的框（回归头），你想要每个框的正确类别（分类头），你想要模型知道什么时候没有东西可检测（objectness 分数），你想要每个真实物体恰好一个预测（非极大值抑制）。漏掉其中任何一个，pipeline 要么漏检物体，要么报告幻觉框，要么对同一物体在略微不同的位置预测十五次。

YOLO（You Only Look Once，Redmon et al. 2016）是让所有这些通过卷积网络的单次前向传播实时运行的设计，同样的结构决策仍然是现代检测器（YOLOv8、YOLOv9、YOLO-NAS、RT-DETR）的骨干。学会核心，每个变体都变成同样部件的重新排列。

## 概念

### 检测作为密集预测

分类器每张图输出 C 个数。YOLO 风格检测器每张图输出 `(S x S x (5 + C))` 个数，其中 S 是空间网格大小。

```mermaid
flowchart LR
    IMG["Input 416x416 RGB"] --> BB["Backbone<br/>(ResNet, DarkNet, ...)"]
    BB --> FM["Feature map<br/>(C_feat, 13, 13)"]
    FM --> HEAD["Detection head<br/>(1x1 convs)"]
    HEAD --> OUT["Output tensor<br/>(13, 13, B * (5 + C))"]
    OUT --> DEC["Decode<br/>(grid + sigmoid + exp)"]
    DEC --> NMS["Non-max suppression"]
    NMS --> RESULT["Final boxes"]

    style IMG fill:#dbeafe,stroke:#2563eb
    style HEAD fill:#fef3c7,stroke:#d97706
    style NMS fill:#fecaca,stroke:#dc2626
    style RESULT fill:#dcfce7,stroke:#16a34a
```

`S * S` 个网格单元中的每一个预测 `B` 个框。对于每个框：

- 4 个数描述几何：`tx, ty, tw, th`。
- 1 个数是 objectness 分数："这个单元中心有物体吗？"
- C 个数是类别概率。

每个单元总计：`B * (5 + C)`。对于 VOC，`S=13, B=2, C=20`，就是每个单元 50 个数。

### 为什么要 grid 和 anchor

纯回归会为每个物体预测 `(x, y, w, h)` 作为绝对坐标。这对卷积网络来说很难，因为平移图像不应该把所有预测平移相同的量——每个物体是空间锚定的。Grid 通过将每个 ground-truth 框分配给其中心落入的网格单元来解决这个问题；只有那个单元负责那个物体。

Anchor 解决第二个问题。一个 3x3 conv 不容易从 16 像素感受野的特征单元回归出一个 500 像素宽的框。相反，我们为每个单元预定义 `B` 个先验框形状（anchor），并预测相对于每个 anchor 的小偏移量。模型学习选择正确的 anchor 并微调它，而不是从零回归。

```
Anchor box priors (example for 416x416 input):

  small:   (30,  60)
  medium:  (75,  170)
  large:   (200, 380)

At each grid cell, every anchor emits (tx, ty, tw, th, obj, c_1, ..., c_C).
```

现代检测器通常使用 FPN，每个分辨率有不同的 anchor 集——浅层高分辨率图上用小 anchor，深层低分辨率图上用大 anchor。同样的想法，更多尺度。

### 解码预测

原始的 `tx, ty, tw, th` 不是框坐标；它们是需要变换后才能绘制的回归目标：

```
centre x  = (sigmoid(tx) + cell_x) * stride
centre y  = (sigmoid(ty) + cell_y) * stride
width     = anchor_w * exp(tw)
height    = anchor_h * exp(th)
```

`sigmoid` 保持中心偏移在单元内。`exp` 让宽度从 anchor 自由缩放而不会翻转符号。`stride` 将网格坐标缩放回像素。这个解码步骤从 v2 以来在每个 YOLO 版本中都相同。

### IoU

检测中两个框之间的通用相似度度量：

```
IoU(A, B) = area(A intersect B) / area(A union B)
```

IoU = 1 表示完全相同；IoU = 0 表示无重叠。预测和 ground-truth 框之间的 IoU 决定一个预测是否算作 true positive（通常 IoU >= 0.5）。两个预测之间的 IoU 是 NMS 用来去重的。

### 非极大值抑制

在相邻 anchor 上训练的卷积网络经常会对同一物体预测重叠的框。NMS 保留最高置信度的预测，删除与之 IoU 超过阈值的任何其他预测。

```
NMS(boxes, scores, iou_threshold):
    sort boxes by score descending
    keep = []
    while boxes not empty:
        pick the top-scoring box, add to keep
        remove every box with IoU > iou_threshold to the picked box
    return keep
```

典型阈值：目标检测用 0.45。最近的检测器用 `soft-NMS`、`DIoU-NMS` 替代标准 NMS，或直接学习抑制（RT-DETR），但结构目的相同。

### Loss

YOLO loss 是三个加权 loss 的和：

```
L = lambda_coord * L_box(pred, target, where obj=1)
  + lambda_obj   * L_obj(pred, 1,     where obj=1)
  + lambda_noobj * L_obj(pred, 0,     where obj=0)
  + lambda_cls   * L_cls(pred, target, where obj=1)
```

只有包含物体的单元贡献框回归和分类 loss。没有物体的单元只贡献 objectness loss（教模型保持沉默）。`lambda_noobj` 通常很小（约 0.5），因为绝大多数单元是空的，否则会主导总 loss。

现代变体用 CIoU / DIoU（直接优化 IoU）替换 MSE 框 loss，用 focal loss 处理类别不平衡，用 quality focal loss 平衡 objectness。三组件结构不变。

### 检测指标

准确率不适用于检测。四个适用的数字：

- **Precision@IoU=0.5** — 被算作正例的预测中，有多少实际正确。
- **Recall@IoU=0.5** — 真实物体中，我们找到了多少。
- **AP@0.5** — IoU 阈值 0.5 时的 precision-recall 曲线面积；每个类一个数。
- **mAP@0.5:0.95** — IoU 阈值 0.5、0.55、...、0.95 上 AP 的平均。COCO 指标；最严格也最有信息量。

四个都报告。一个在 mAP@0.5 上强但在 mAP@0.5:0.95 上弱的检测器是定位粗糙但不精确；用更好的框回归 loss 修复。一个高 precision 低 recall 的检测器太保守；降低置信度阈值或增加 objectness 权重。

## 动手构建

### 第 1 步：IoU

整节课的主力。作用在两个 `(x1, y1, x2, y2)` 格式的框数组上。

```python
import numpy as np

def box_iou(boxes_a, boxes_b):
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

返回一个 `(N_a, N_b)` 的成对 IoU 矩阵。对单个 ground-truth 框使用时，让其中一个数组形状为 `(1, 4)`。

### 第 2 步：非极大值抑制

```python
def nms(boxes, scores, iou_threshold=0.45):
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)
```

确定性的，排序带来 `O(N log N)`，在相同输入上与 `torchvision.ops.nms` 行为一致。

### 第 3 步：框编码和解码

在像素坐标和网络实际回归的 `(tx, ty, tw, th)` 目标之间转换。

```python
def encode(box_xyxy, cell_x, cell_y, stride, anchor_wh):
    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    tx = cx / stride - cell_x
    ty = cy / stride - cell_y
    tw = np.log(w / anchor_wh[0] + 1e-8)
    th = np.log(h / anchor_wh[1] + 1e-8)
    return np.array([tx, ty, tw, th])


def decode(tx_ty_tw_th, cell_x, cell_y, stride, anchor_wh):
    tx, ty, tw, th = tx_ty_tw_th
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride
    w = anchor_wh[0] * np.exp(tw)
    h = anchor_wh[1] * np.exp(th)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
```

测试：编码一个框然后解码——你应该得到非常接近原始的结果（sigmoid 逆在 `tx` 不在 post-sigmoid 范围时不完全可逆）。

### 第 4 步：最小 YOLO head

一个 1x1 conv 作用在 feature map 上，reshape 为 `(B, S, S, num_anchors, 5 + C)`。

```python
import torch
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, in_c, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        n, _, h, w = x.shape
        y = self.conv(x)
        y = y.view(n, self.num_anchors, 5 + self.num_classes, h, w)
        y = y.permute(0, 3, 4, 1, 2).contiguous()
        return y
```

输出形状：`(N, H, W, num_anchors, 5 + C)`。最后一维持有 `[tx, ty, tw, th, obj, cls_0, ..., cls_{C-1}]`。

### 第 5 步：Ground-truth 分配

对每个 ground-truth 框，决定哪个 `(cell, anchor)` 负责。

```python
def assign_targets(boxes_xyxy, classes, anchors, stride, grid_size, num_classes):
    num_anchors = len(anchors)
    target = np.zeros((grid_size, grid_size, num_anchors, 5 + num_classes), dtype=np.float32)
    has_obj = np.zeros((grid_size, grid_size, num_anchors), dtype=bool)

    for box, cls in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        gx, gy = int(cx / stride), int(cy / stride)
        bw, bh = x2 - x1, y2 - y1

        ious = np.array([
            (min(bw, aw) * min(bh, ah)) / (bw * bh + aw * ah - min(bw, aw) * min(bh, ah))
            for aw, ah in anchors
        ])
        best = int(np.argmax(ious))
        aw, ah = anchors[best]

        target[gy, gx, best, 0] = cx / stride - gx
        target[gy, gx, best, 1] = cy / stride - gy
        target[gy, gx, best, 2] = np.log(bw / aw + 1e-8)
        target[gy, gx, best, 3] = np.log(bh / ah + 1e-8)
        target[gy, gx, best, 4] = 1.0
        target[gy, gx, best, 5 + cls] = 1.0
        has_obj[gy, gx, best] = True
    return target, has_obj
```

Anchor 选择是"与 ground truth 的最佳形状 IoU"——一个廉价的代理，匹配 YOLOv2/v3 的分配方式。v5 及之后使用更复杂的策略（task-aligned matching、dynamic k）来细化同样的想法。

### 第 6 步：三个 loss

```python
def yolo_loss(pred, target, has_obj, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
    has_obj_t = torch.from_numpy(has_obj).bool()
    target_t = torch.from_numpy(target).float()

    # box-regression loss: only on cells with objects
    box_pred = pred[..., :4][has_obj_t]
    box_true = target_t[..., :4][has_obj_t]
    loss_box = torch.nn.functional.mse_loss(box_pred, box_true, reduction="sum")

    # objectness loss
    obj_pred = pred[..., 4]
    obj_true = target_t[..., 4]
    loss_obj_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[has_obj_t], obj_true[has_obj_t], reduction="sum")
    loss_obj_neg = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[~has_obj_t], obj_true[~has_obj_t], reduction="sum")

    # classification loss on cells with objects
    cls_pred = pred[..., 5:][has_obj_t]
    cls_true = target_t[..., 5:][has_obj_t]
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
        cls_pred, cls_true, reduction="sum")

    total = (lambda_coord * loss_box
             + lambda_obj * loss_obj_pos
             + lambda_noobj * loss_obj_neg
             + lambda_cls * loss_cls)
    return total, {"box": loss_box.item(), "obj_pos": loss_obj_pos.item(),
                   "obj_neg": loss_obj_neg.item(), "cls": loss_cls.item()}
```

五个超参数，每个 YOLO 教程要么硬编码要么搜索。比例很重要：`lambda_coord=5, lambda_noobj=0.5` 镜像了原始 YOLOv1 论文，仍然是合理的默认值。

### 第 7 步：推理 pipeline

解码原始 head 输出，应用 sigmoid/exp，按 objectness 阈值过滤，然后 NMS。

```python
def postprocess(pred_tensor, anchors, stride, img_size, conf_threshold=0.25, iou_threshold=0.45):
    pred = pred_tensor.detach().cpu().numpy()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    num_anchors = len(anchors)

    boxes, scores, classes = [], [], []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                tx, ty, tw, th, obj, *cls = pred[0, gy, gx, a]
                score = sigmoid(obj) * sigmoid(np.array(cls)).max()
                if score < conf_threshold:
                    continue
                cls_idx = int(np.argmax(cls))
                cx = (sigmoid(tx) + gx) * stride
                cy = (sigmoid(ty) + gy) * stride
                w = anchors[a][0] * np.exp(tw)
                h = anchors[a][1] * np.exp(th)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                scores.append(float(score))
                classes.append(cls_idx)

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]
```

这就是完整的评估路径：head -> 解码 -> 阈值 -> NMS。

## 实际使用

`torchvision.models.detection` 提供具有相同概念结构的生产检测器。加载预训练模型只需三行。

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
with torch.no_grad():
    predictions = model([torch.randn(3, 400, 600)])
print(predictions[0].keys())
print(f"boxes:  {predictions[0]['boxes'].shape}")
print(f"scores: {predictions[0]['scores'].shape}")
print(f"labels: {predictions[0]['labels'].shape}")
```

对于实时推理 pipeline，`ultralytics`（YOLOv8/v9）是标准：`from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model(img)`。模型内部处理解码和 NMS，返回你上面构建的同样的 `boxes / scores / labels` 三元组。

## 交付产出

本课产出：

- `outputs/prompt-detection-metric-reader.md` — 一个 prompt，把 `precision, recall, AP, mAP@0.5:0.95` 行转化为一行诊断和最有用的下一个实验。
- `outputs/skill-anchor-designer.md` — 一个 skill，给定 ground-truth 框数据集，对 `(w, h)` 运行 k-means 并返回每个 FPN 层级的 anchor 集加上你需要的覆盖率统计来选择正确的 anchor 数量。

## 练习

1. **（简单）** 实现 `box_iou` 并在 1,000 个随机框对上与 `torchvision.ops.box_iou` 对比。验证最大绝对差异低于 `1e-6`。
2. **（中等）** 将 `yolo_loss` 移植为使用 `CIoU` 框 loss 而非 MSE 的版本。在 100 张图像的合成数据集上展示 CIoU 在相同 epoch 数内收敛到比 MSE 更好的最终 mAP@0.5:0.95。
3. **（困难）** 实现多尺度推理：将同一图像以三个分辨率通过模型，合并框预测，最后运行一次 NMS。在留出集上测量相对于单尺度推理的 mAP 提升。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|----------|----------|
| Anchor | "框先验" | 每个网格单元的预定义框形状，网络从中预测偏移量而非绝对坐标 |
| IoU | "重叠度" | 两个框的 intersection-over-union；检测中的通用相似度度量 |
| NMS | "去重" | 贪心算法，保留最高分预测并移除超过阈值的重叠预测 |
| Objectness | "这里有东西吗" | 每 anchor、每单元的标量，预测该单元中心是否有物体 |
| Grid stride | "下采样因子" | 每个网格单元对应的像素数；416 像素输入配 13 网格 head 的 stride 是 32 |
| mAP | "平均精度均值" | precision-recall 曲线下面积的平均，跨类别和（对 COCO）IoU 阈值取平均 |
| AP@0.5 | "PASCAL VOC AP" | IoU 阈值 0.5 时的平均精度；宽松版本的指标 |
| mAP@0.5:0.95 | "COCO AP" | IoU 阈值 0.5..0.95 步长 0.05 上的平均；严格版本，当前社区标准 |

## 延伸阅读

- [YOLOv1: You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) — 奠基论文；之后的每个 YOLO 都是这个结构的细化
- [YOLOv3 (Redmon & Farhadi, 2018)](https://arxiv.org/abs/1804.02767) — 引入多尺度 FPN 风格 head 的论文；仍然是最清晰的图解
- [Ultralytics YOLOv8 docs](https://docs.ultralytics.com) — 当前生产参考；涵盖数据集格式、增强、训练配方
- [The Illustrated Guide to Object Detection (Jonathan Hui)](https://jonathan-hui.medium.com/object-detection-series-24d03a12f904) — 完整检测器动物园的最佳通俗英文之旅；对理解 DETR、RetinaNet、FCOS 和 YOLO 的关系无价
