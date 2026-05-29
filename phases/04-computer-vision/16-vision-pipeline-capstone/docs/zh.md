# 构建完整视觉流水线 — Capstone

> 生产级视觉系统是一条由模型和规则组成的链路，用数据契约缝合在一起。各个组件在本阶段已经学过了；这个 capstone 把它们端到端地串起来。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 4 Lessons 01-15
**Time:** ~120 minutes

## 学习目标

- 设计一个生产级视觉流水线：检测目标、分类、输出结构化 JSON — 并处理每一条失败路径
- 将检测器（Mask R-CNN 或 YOLO）、分类器（ConvNeXt-Tiny）和数据契约（Pydantic）组装到一个服务中
- 对端到端流水线做基准测试，找出第一个瓶颈（通常是预处理，然后是检测器）
- 交付一个最小化的 FastAPI 服务，接受图片上传、运行流水线、返回带分类的检测结果

## 问题

单个视觉模型有用；视觉产品是它们的链式组合。零售货架审计 = 检测器 + 商品分类器 + 价格 OCR 流水线。自动驾驶 = 2D 检测器 + 3D 检测器 + 分割器 + 跟踪器 + 规划器。医学预筛 = 分割器 + 区域分类器 + 临床 UI。

把这些链路串起来，就是 ML 原型和产品之间的分水岭。模型之间的每个接口都是新的 bug 来源。每次坐标变换、每次归一化、每次 mask resize 都是静默失败的候选。流水线的强度取决于最弱的接口。

这个 capstone 搭建最小可行流水线：检测 + 分类 + 结构化输出 + 服务层。Phase 4 的其他内容都可以插入这个骨架：把 Mask R-CNN 换成 YOLOv8、加 OCR head、加分割分支、加跟踪器。架构是稳定的；组件是可插拔的。

## 概念

### 流水线

```mermaid
flowchart LR
    REQ["HTTP request<br/>+ image bytes"] --> LOAD["Decode<br/>+ preprocess"]
    LOAD --> DET["Detector<br/>(YOLO / Mask R-CNN)"]
    DET --> CROP["Crop + resize<br/>each detection"]
    CROP --> CLS["Classifier<br/>(ConvNeXt-Tiny)"]
    CLS --> AGG["Aggregate<br/>detections + classes"]
    AGG --> SCHEMA["Pydantic<br/>validation"]
    SCHEMA --> RESP["JSON response"]

    REQ -.->|error| RESP

    style DET fill:#fef3c7,stroke:#d97706
    style CLS fill:#dbeafe,stroke:#2563eb
    style SCHEMA fill:#dcfce7,stroke:#16a34a
```

七个阶段。两个模型阶段是昂贵的；另外五个阶段是 bug 藏身之处。

### 用 Pydantic 定义数据契约

每个模型边界变成一个类型化对象。这把静默失败变成了响亮的失败。

```
Detection(
    box: tuple[float, float, float, float],   # (x1, y1, x2, y2), absolute pixels
    score: float,                              # [0, 1]
    class_id: int,                             # from detector's label map
    mask: Optional[list[list[int]]],           # RLE-encoded if present
)

PipelineResult(
    image_id: str,
    detections: list[Detection],
    classifications: list[Classification],
    inference_ms: float,
)
```

当检测器返回 `(cx, cy, w, h)` 格式的框而不是 `(x1, y1, x2, y2)` 时，Pydantic 的验证会在边界处失败，你会立即发现问题，而不是去调试一个静默返回空区域的下游裁剪。

### 延迟分布在哪里

几乎所有视觉流水线都遵循三个事实：

1. **预处理往往是最大的单个块。** 解码 JPEG、转换色彩空间、resize — 这些是 CPU 密集型的，容易被忽略。
2. **检测器占据 GPU 时间的主体。** 70-90% 的 GPU 时间在检测前向传播中。
3. **后处理（NMS、RLE 编解码）在 GPU 上便宜，在 CPU 上昂贵。** 一定要用实际目标来 profile。

知道分布，才能把优化变成一个有优先级的列表。

### 失败模式

- **空检测** — 返回空列表，不要崩溃。记日志。
- **越界框** — 裁剪前 clamp 到图像尺寸。
- **过小裁剪** — 对小于分类器最小输入的框跳过分类。
- **损坏的上传** — 返回 400 和具体错误码，不是 500。
- **模型加载失败** — 在服务启动时失败，不是在第一个请求时。

生产流水线处理每一种情况，不写笼统的 `try/except` 来掩盖失败。每种失败都有命名的错误码和响应。

### 批处理

生产服务服务多个客户端。跨请求批处理检测和分类可以成倍提升吞吐量。代价是：等待批次填满带来的额外延迟。典型设置：收集请求最多 20ms，批量处理，分发响应。`torchserve` 和 `triton` 原生支持；负载可预测的小服务自己实现 micro-batcher。

## 动手构建

### Step 1: 数据契约

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class Detection(BaseModel):
    box: Tuple[float, float, float, float]
    score: float = Field(ge=0, le=1)
    class_id: int = Field(ge=0)
    mask_rle: Optional[str] = None


class Classification(BaseModel):
    detection_index: int
    class_id: int
    class_name: str
    score: float = Field(ge=0, le=1)


class PipelineResult(BaseModel):
    image_id: str
    detections: List[Detection]
    classifications: List[Classification]
    inference_ms: float
```

五秒钟的代码能在任何正经流水线上省一小时的调试。

### Step 2: 最小化 Pipeline 类

```python
import time
import numpy as np
import torch
from PIL import Image

class VisionPipeline:
    def __init__(self, detector, classifier, class_names,
                 device="cpu", min_crop=32):
        self.detector = detector.to(device).eval()
        self.classifier = classifier.to(device).eval()
        self.class_names = class_names
        self.device = device
        self.min_crop = min_crop

    def preprocess(self, image):
        """
        image: PIL.Image or np.ndarray (H, W, 3) uint8
        returns: CHW float tensor on device
        """
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.to(self.device)

    @torch.no_grad()
    def detect(self, image_tensor):
        return self.detector([image_tensor])[0]

    @torch.no_grad()
    def classify(self, crops):
        if len(crops) == 0:
            return []
        batch = torch.stack(crops).to(self.device)
        logits = self.classifier(batch)
        probs = logits.softmax(-1)
        scores, cls = probs.max(-1)
        return list(zip(cls.tolist(), scores.tolist()))

    def run(self, image, image_id="anonymous"):
        t0 = time.perf_counter()
        tensor = self.preprocess(image)
        det = self.detect(tensor)

        crops = []
        detections = []
        valid_indices = []
        for i, (box, score, cls) in enumerate(zip(det["boxes"], det["scores"], det["labels"])):
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            detections.append(Detection(
                box=(x1, y1, x2, y2),
                score=float(score),
                class_id=int(cls),
            ))
            if (x2 - x1) < self.min_crop or (y2 - y1) < self.min_crop:
                continue
            crop = tensor[:, y1:y2, x1:x2]
            crop = torch.nn.functional.interpolate(
                crop.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )[0]
            crops.append(crop)
            valid_indices.append(i)

        class_preds = self.classify(crops)

        classifications = []
        for valid_idx, (cls_id, cls_score) in zip(valid_indices, class_preds):
            classifications.append(Classification(
                detection_index=valid_idx,
                class_id=int(cls_id),
                class_name=self.class_names[cls_id],
                score=float(cls_score),
            ))

        return PipelineResult(
            image_id=image_id,
            detections=detections,
            classifications=classifications,
            inference_ms=(time.perf_counter() - t0) * 1000,
        )
```

每个接口都有类型。每条失败路径都有明确的处理决策。

### Step 3: 接入检测器和分类器

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import convnext_tiny

# Use ImageNet-pretrained weights for a realistic pipeline without training
detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
classifier = convnext_tiny(weights="DEFAULT")
class_names = [f"imagenet_class_{i}" for i in range(1000)]

pipe = VisionPipeline(detector, classifier, class_names)

# Smoke test with a synthetic image
test_image = (np.random.rand(400, 600, 3) * 255).astype(np.uint8)
result = pipe.run(test_image, image_id="demo")
print(result.model_dump_json(indent=2)[:500])
```

### Step 4: FastAPI 服务

```python
from fastapi import FastAPI, UploadFile, HTTPException
from io import BytesIO

app = FastAPI()
pipe = None  # initialised on startup

@app.on_event("startup")
def load():
    global pipe
    detector = maskrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
    classifier = convnext_tiny(weights="DEFAULT").eval()
    pipe = VisionPipeline(detector, classifier, class_names=[f"c{i}" for i in range(1000)])

@app.post("/detect")
async def detect_endpoint(file: UploadFile):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="unsupported image type")
    data = await file.read()
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="cannot decode image")
    result = pipe.run(img, image_id=file.filename or "upload")
    return result.model_dump()
```

用 `uvicorn main:app --host 0.0.0.0 --port 8000` 运行。用 `curl -F 'file=@dog.jpg' http://localhost:8000/detect` 测试。

### Step 5: 流水线基准测试

```python
import time

def benchmark(pipe, num_runs=20, image_size=(400, 600)):
    img = (np.random.rand(*image_size, 3) * 255).astype(np.uint8)
    pipe.run(img)  # warm up

    stages = {"preprocess": [], "detect": [], "classify": [], "total": []}
    for _ in range(num_runs):
        t0 = time.perf_counter()
        tensor = pipe.preprocess(img)
        t1 = time.perf_counter()
        det = pipe.detect(tensor)
        t2 = time.perf_counter()
        crops = []
        for box in det["boxes"]:
            x1, y1, x2, y2 = [max(0, int(b)) for b in box.tolist()]
            x2 = min(x2, tensor.shape[-1])
            y2 = min(y2, tensor.shape[-2])
            if (x2 - x1) >= pipe.min_crop and (y2 - y1) >= pipe.min_crop:
                crop = tensor[:, y1:y2, x1:x2]
                crop = torch.nn.functional.interpolate(
                    crop.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                )[0]
                crops.append(crop)
        pipe.classify(crops)
        t3 = time.perf_counter()
        stages["preprocess"].append((t1 - t0) * 1000)
        stages["detect"].append((t2 - t1) * 1000)
        stages["classify"].append((t3 - t2) * 1000)
        stages["total"].append((t3 - t0) * 1000)

    for stage, times in stages.items():
        times.sort()
        print(f"{stage:12s}  p50={times[len(times)//2]:7.1f} ms  p95={times[int(len(times)*0.95)]:7.1f} ms")
```

CPU 上的典型输出：preprocess ~3 ms，detect 300-500 ms，classify 20-40 ms，total 350-550 ms。在 GPU 上，detect 是 20-40 ms，preprocess + classify 在相对占比上开始变得重要。

## 实际应用

生产模板收敛到相同的结构，再加上：

- **模型版本管理** — 始终在响应中记录模型名称和权重哈希。
- **每请求 trace ID** — 为每个请求的每个阶段记录耗时，以便关联慢响应和具体阶段。
- **降级路径** — 如果分类器超时，返回不带分类的检测结果，而不是整个请求失败。
- **安全过滤** — NSFW / PII 过滤在分类之后、响应离开服务之前运行。
- **批量端点** — 一个 `/detect_batch` 接受图片 URL 列表用于批量处理。

生产级服务方面，`torchserve`、`Triton Inference Server` 和 `BentoML` 开箱即用地处理批处理、版本管理、指标和健康检查。直接用 `FastAPI` 对原型和小规模产品来说没问题。

## 交付产出

本课产出：

- `outputs/prompt-vision-service-shape-reviewer.md` — 一个 prompt，审查视觉服务代码中的契约/响应格式违规，并指出第一个破坏性 bug。
- `outputs/skill-pipeline-budget-planner.md` — 一个 skill，给定目标延迟和吞吐量，为每个流水线阶段分配时间预算，并标记哪个阶段会首先超出预算。

## 练习

1. **（简单）** 在任意开放数据集的 10 张图片上运行流水线。报告每个阶段的平均耗时和每张图片的检测数量分布。
2. **（中等）** 给 `Detection` 添加 mask 输出字段并编码为 RLE。验证即使对 10 个目标的图片，JSON 也保持在 1MB 以下。
3. **（困难）** 在分类器前添加 micro-batcher：收集裁剪最多 10 ms，一次 GPU 调用全部分类，按请求返回结果。在每秒 5 个并发请求下测量吞吐量增益和增加的延迟。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| Pipeline | "系统" | 预处理、推理和后处理步骤的有序链，每对之间有类型化接口 |
| Data contract | "Schema" | Pydantic / dataclass 定义，每个阶段的输入输出都必须符合；在边界处捕获集成 bug |
| Preprocessing | "模型之前" | 解码、色彩转换、resize、归一化；通常是最大的 CPU 时间消耗 |
| Postprocessing | "模型之后" | NMS、mask resize、阈值、RLE 编码；GPU 上便宜，CPU 上昂贵 |
| Microbatcher | "收集再前向" | 等待固定窗口收集多个请求，运行单次批量前向传播的聚合器 |
| Trace ID | "请求 ID" | 在每个阶段记录的每请求标识符，用于端到端追踪慢请求 |
| Failure code | "命名错误" | 每种失败类别的具体错误码，而不是通用 500；支持客户端重试逻辑 |
| Health check | "就绪探针" | 报告服务是否能响应的廉价端点；负载均衡器依赖它 |

## 延伸阅读

- [Full Stack Deep Learning — Deploying Models](https://fullstackdeeplearning.com/course/2022/lecture-5-deployment/) — 生产级 ML 部署的经典概述
- [BentoML docs](https://docs.bentoml.com) — 带批处理、版本管理和指标的服务框架
- [torchserve docs](https://pytorch.org/serve/) — PyTorch 官方服务库
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) — 带批处理和多模型支持的高吞吐量服务
