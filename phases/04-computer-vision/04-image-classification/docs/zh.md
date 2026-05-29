# 图像分类

> 分类器是一个从像素到类别概率分布的函数。其他一切都是管道工程。

**类型：** Build
**语言：** Python
**前置课程：** Phase 2 Lesson 09（模型评估）、Phase 3 Lesson 10（Mini Framework）、Phase 4 Lesson 03（CNN）
**时长：** 约 75 分钟

## 学习目标

- 在 CIFAR-10 上构建端到端图像分类 pipeline：数据集、数据增强、模型、训练循环、评估
- 解释每个组件（dataloader、loss、optimizer、scheduler、augmentation）的作用，并预测破坏其中任何一个会如何体现在 loss 曲线上
- 从零实现 mixup、cutout 和 label smoothing，并说明何时值得添加每一个
- 阅读混淆矩阵和逐类 precision/recall 表，诊断超越总体准确率的数据集和模型故障

## 问题

每个交付的视觉任务在某个层面都归结为图像分类。检测对区域做分类。分割对像素做分类。检索按与类中心的相似度排序。把分类做对——数据集循环、增强策略、loss、评估——是迁移到本阶段其他所有任务的技能。

大多数分类 bug 不在模型里。它们在 pipeline 中：一个坏掉的归一化、一个未 shuffle 的训练集、扭曲标签的增强、被训练数据污染的验证集、在第 30 个 epoch 后静默发散的学习率。一个在正确设置下能达到 93% CIFAR-10 准确率的 CNN，在坏掉的设置下通常只有 70-75%，而 loss 曲线全程看起来都很合理。

这节课手动连接整个 pipeline，使每个部分都可检查。你不会使用任何可能隐藏 bug 的 `torchvision.datasets`。

## 概念

### 分类 pipeline

```mermaid
flowchart LR
    A["Dataset<br/>(images + labels)"] --> B["Augment<br/>(random transforms)"]
    B --> C["Normalise<br/>(mean/std)"]
    C --> D["DataLoader<br/>(batch + shuffle)"]
    D --> E["Model<br/>(CNN)"]
    E --> F["Logits<br/>(N, C)"]
    F --> G["Cross-entropy loss"]
    F --> H["Argmax<br/>at eval"]
    G --> I["Backward"]
    I --> J["Optimizer step"]
    J --> K["Scheduler step"]
    K --> E

    style A fill:#dbeafe,stroke:#2563eb
    style E fill:#fef3c7,stroke:#d97706
    style G fill:#fecaca,stroke:#dc2626
    style H fill:#dcfce7,stroke:#16a34a
```

这个循环中的每一条线都是 bug 可能存在的地方。Cross-entropy 接收原始 logits，不是 softmax 输出，所以在 loss 之前做 `model(x).softmax()` 会静默计算错误的梯度。增强只应用于输入，不应用于标签——除了 mixup，它两者都混合。`optimizer.zero_grad()` 必须每步执行一次；跳过它会累积梯度，看起来像一个极不稳定的学习率。这些 bug 中的每一个都会压平学习曲线而不抛出错误。

### Cross-entropy、logits 和 softmax

分类器为每张图像产生 `C` 个数，称为 logits。应用 softmax 将它们转换为概率分布：

```
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

Cross-entropy 衡量正确类别的负对数概率：

```
CE(z, y) = -log( softmax(z)_y )
        = -z_y + log( sum_j exp(z_j) )
```

右边的形式是数值稳定的（log-sum-exp）。PyTorch 的 `nn.CrossEntropyLoss` 将 softmax + NLL 融合在一个 op 中，直接接收原始 logits。自己先应用 softmax 几乎总是一个 bug——你计算的是 log(softmax(softmax(z)))，一个无意义的量。

### 为什么数据增强有效

CNN 对平移有归纳偏置（来自权重共享），但对裁剪、翻转、颜色抖动或遮挡没有内建的不变性。教会它这些不变性的唯一方法是给它看行使这些变换的像素。训练时的每个随机变换都是在说："这两张图像有相同的标签；学习忽略差异的特征。"

```
Original crop:  "dog facing left"
Flip:           "dog facing right"       <- same label, different pixels
Rotate(+15):    "dog, slight tilt"
Colour jitter:  "dog in warmer light"
RandomErasing:  "dog with patch missing"
```

规则：增强必须保持标签不变。对数字做 Cutout 和旋转可能把"6"翻成"9"；对那个数据集你要用更小的旋转范围，选择尊重数字特定不变性的增强。

### Mixup 和 cutmix

普通增强变换像素但保持标签为 one-hot。**Mixup** 和 **cutmix** 打破了这一点，对两者都做插值。

```
Mixup:
  lambda ~ Beta(a, a)
  x = lambda * x_i + (1 - lambda) * x_j
  y = lambda * y_i + (1 - lambda) * y_j

Cutmix:
  paste a random rectangle of x_j into x_i
  y = area-weighted mix of y_i and y_j
```

为什么有帮助：模型不再记忆尖锐的 one-hot 目标，而是学习在类之间插值。训练 loss 上升，测试准确率上升。这是任何分类器最便宜的鲁棒性升级。

### Label smoothing

Mixup 的近亲。不是对 `[0, 0, 1, 0, 0]` 训练，而是对 `[eps/C, eps/C, 1-eps, eps/C, eps/C]` 训练，其中 `eps` 是一个小值如 0.1。阻止模型产生任意尖锐的 logits，几乎零成本地改善校准。从 PyTorch 1.10 起内建于 `nn.CrossEntropyLoss(label_smoothing=0.1)`。

### 超越准确率的评估

总体准确率隐藏了不平衡。一个 90-10 的二分类器总是预测多数类就能得 90%。真正告诉你发生了什么的工具：

- **逐类准确率** — 每个类一个数字；立即暴露表现不佳的类别。
- **混淆矩阵** — C x C 网格，第 i 行第 j 列 = 真实类 i 被预测为类 j 的计数；对角线是正确的，非对角线是你的模型所在之处。
- **Top-1 / Top-5** — 正确类别是否在前 1 或前 5 个预测中；Top-5 对 ImageNet 很重要，因为像"Norwich terrier"vs"Norfolk terrier"这样的类确实有歧义。
- **校准（ECE）** — 0.8 置信度的预测是否 80% 的时间正确？现代网络系统性地过度自信；用 temperature scaling 或 label smoothing 修复。

## 动手构建

### 第 1 步：确定性合成数据集

CIFAR-10 在磁盘上。为了使本课可复现且快速，我们构建一个看起来像 CIFAR 的合成数据集——32x32 RGB 图像，带有模型必须学习的类特定结构。完全相同的 pipeline 在真实 CIFAR-10 上无需修改即可工作。

```python
import numpy as np
import torch
from torch.utils.data import Dataset


def synthetic_cifar(num_per_class=1000, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    Y = []
    for c in range(num_classes):
        centre = rng.uniform(0, 1, (3,))
        freq = 2 + c
        for _ in range(num_per_class):
            yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
            r = np.sin(xx * freq) * 0.5 + centre[0]
            g = np.cos(yy * freq) * 0.5 + centre[1]
            b = (xx + yy) * 0.5 * centre[2]
            img = np.stack([r, g, b], axis=-1)
            img += rng.normal(0, 0.08, img.shape)
            img = np.clip(img, 0, 1)
            X.append(img.astype(np.float32))
            Y.append(c)
    X = np.stack(X)
    Y = np.array(Y)
    idx = rng.permutation(len(X))
    return X[idx], Y[idx]


class ArrayDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, int(self.Y[i])
```

每个类有自己的颜色调色板和频率模式，加上高斯噪声迫使模型学习信号而非记忆像素。十个类，每类一千张图像，打乱顺序。

### 第 2 步：归一化和数据增强

每个视觉 pipeline 都有的两个变换。

```python
def standardize(mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    def _fn(img):
        return (img - mean) / std
    return _fn


def random_hflip(p=0.5):
    def _fn(img):
        if np.random.random() < p:
            return img[:, ::-1, :].copy()
        return img
    return _fn


def random_crop(pad=4):
    def _fn(img):
        h, w = img.shape[:2]
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        y = np.random.randint(0, 2 * pad)
        x = np.random.randint(0, 2 * pad)
        return padded[y:y + h, x:x + w, :]
    return _fn


def compose(*fns):
    def _fn(img):
        for fn in fns:
            img = fn(img)
        return img
    return _fn
```

裁剪前用 reflect-pad 而非 zero-pad，因为黑色边框是模型会以无用方式学习忽略的信号。

### 第 3 步：Mixup

在训练步骤内混合两张图像和两个标签。实现为 batch 变换，所以它在 forward pass 旁边而不是在 dataset 内部。

```python
def mixup_batch(x, y, num_classes, alpha=0.2):
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_onehot = torch.nn.functional.one_hot(y, num_classes).float()
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return x_mixed, y_mixed


def soft_cross_entropy(logits, soft_targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

`soft_cross_entropy` 是对软标签分布的 cross-entropy。当目标恰好是 one-hot 时，它退化为通常的情况。

### 第 4 步：训练循环

完整配方：数据过一遍，每 batch 一次梯度，scheduler 每 epoch 步进一次。

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def train_one_epoch(model, loader, optimizer, device, num_classes, use_mixup=True):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if use_mixup:
            x_m, y_soft = mixup_batch(x, y, num_classes)
            logits = model(x_m)
            loss = soft_cross_entropy(logits, y_soft)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        # Training accuracy vs the un-mixed labels `y` is only an approximation
        # when mixup is on (the model saw soft targets, not y). Treat it as a
        # rough progress signal; rely on val accuracy for real performance.
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        for t, p in zip(y.cpu(), pred.cpu()):
            cm[t, p] += 1
        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / total, correct / total, cm
```

每次写训练循环时检查的五个不变量：

1. 训练前 `model.train()`，评估前 `model.eval()` — 切换 dropout 和 batchnorm 行为。
2. `.backward()` 之前 `.zero_grad()`。
3. 累积指标时用 `.item()`，这样不会保持计算图存活。
4. 评估时 `@torch.no_grad()` — 节省内存和时间，防止微妙的意外。
5. 对原始 logits 做 argmax，不是 softmax — 结果相同，少一个 op。

### 第 5 步：组装起来

使用上一课的 `TinyResNet`，训练几个 epoch，评估。

```python
from main import synthetic_cifar, ArrayDataset
from main import standardize, random_hflip, random_crop, compose
from main import mixup_batch, soft_cross_entropy
from main import train_one_epoch, evaluate
# TinyResNet comes from the previous lesson (03-cnns-lenet-to-resnet).
# Adjust the import path to wherever you stored the previous lesson's code.
from cnns_lenet_to_resnet import TinyResNet  # example placeholder

X, Y = synthetic_cifar(num_per_class=500)
split = int(0.9 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

mean = [0.5, 0.5, 0.5]
std = [0.25, 0.25, 0.25]
train_tf = compose(random_hflip(), random_crop(pad=4), standardize(mean, std))
eval_tf = standardize(mean, std)

train_ds = ArrayDataset(X_train, Y_train, transform=train_tf)
val_ds = ArrayDataset(X_val, Y_val, transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyResNet(num_classes=10).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, 10, use_mixup=True)
    va_loss, va_acc, _ = evaluate(model, val_loader, device, 10)
    scheduler.step()
    print(f"epoch {epoch:2d}  lr {scheduler.get_last_lr()[0]:.4f}  "
          f"train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
```

在合成数据集上，这在五个 epoch 内就能达到接近完美的验证准确率，这正是重点：pipeline 是正确的，模型能学到可学的东西。把数据集换成真实 CIFAR-10，同样的循环无需修改就能训练到约 90%。

### 第 6 步：阅读混淆矩阵

仅靠准确率永远不会告诉你模型在哪里失败。混淆矩阵会。

```python
def print_confusion(cm, labels=None):
    c = cm.shape[0]
    labels = labels or [str(i) for i in range(c)]
    print(f"{'':>6}" + "".join(f"{l:>5}" for l in labels))
    for i in range(c):
        row = cm[i].tolist()
        print(f"{labels[i]:>6}" + "".join(f"{v:>5}" for v in row))
    print()
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp_min(1)
    rec = tp / (tp + fn).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    for i in range(c):
        print(f"{labels[i]:>6}  prec {prec[i]:.3f}  rec {rec[i]:.3f}  f1 {f1[i]:.3f}")

_, _, cm = evaluate(model, val_loader, device, 10)
print_confusion(cm)
```

行是真实类，列是预测。类 3 和类 5 之间的非对角线计数集中意味着模型混淆了这两个类，给你一个针对性数据收集或类特定增强的起点。

## 实际使用

`torchvision` 把上面所有内容包装成惯用组件。对于真实 CIFAR-10，完整 pipeline 是四行加一个训练循环。

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
train_tf = Compose([
    RandomCrop(32, padding=4, padding_mode="reflect"),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean, std),
])
eval_tf = Compose([ToTensor(), Normalize(mean, std)])

train_ds = CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
val_ds   = CIFAR10(root="./data", train=False, download=True, transform=eval_tf)
```

注意两点：mean/std 是**数据集特定的**——在 CIFAR-10 训练集上计算的，不是 ImageNet——reflect pad 是社区默认的裁剪策略。在这里复制粘贴 ImageNet 统计量是约 1% 的准确率泄漏，没人发现直到有人 profile 模型。

## 交付产出

本课产出：

- `outputs/prompt-classifier-pipeline-auditor.md` — 一个 prompt，审计训练脚本中上述五个不变量并暴露第一个违规。
- `outputs/skill-classification-diagnostics.md` — 一个 skill，给定混淆矩阵和类名列表，总结逐类失败并提出最有影响力的单一修复。

## 练习

1. **（简单）** 在合成数据集上分别用和不用 mixup 训练同一模型五个 epoch。画出两者的训练和验证 loss。解释为什么 mixup 的训练 loss 更高但验证准确率相似或更好。
2. **（中等）** 实现 Cutout——在每张训练图像中随机置零一个 8x8 方块——并运行消融实验：无增强 vs hflip+crop vs hflip+crop+cutout vs hflip+crop+mixup。报告每种的验证准确率。
3. **（困难）** 构建一个 CIFAR-100 pipeline（100 个类，相同输入尺寸）并复现 ResNet-34 训练到与发表准确率相差 1% 以内。额外：扫描三个学习率和两个 weight decay，记录到本地 CSV，产出最终的混淆矩阵-最易混淆类表。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|----------|----------|
| Logits | "原始输出" | 每张图像的 C 个 pre-softmax 数字；cross-entropy 期望这些，不是 softmax 后的值 |
| Cross-entropy | "loss" | 正确类别的负对数概率；将 log-softmax 和 NLL 合并在一个稳定 op 中 |
| DataLoader | "batcher" | 用 shuffling、batching 和（可选的）多 worker 加载包装数据集；一半训练 bug 都怪它 |
| 数据增强 | "随机变换" | 训练时任何保持标签不变的像素级变换；教会 CNN 原生没有的不变性 |
| Mixup / Cutmix | "混合两张图" | 混合输入和标签，使分类器学习平滑插值而非硬边界 |
| Label smoothing | "更软的目标" | 用 (1-eps, eps/(C-1), ...) 替换 one-hot；改善校准并略微提升准确率 |
| Top-k accuracy | "Top-5" | 正确类别在前 k 个最高概率预测中；用于有真正歧义类别的数据集 |
| 混淆矩阵 | "错误在哪里" | C x C 表格，条目 (i, j) 计数真实类 i 被预测为 j 的图像；对角线是对的，非对角线告诉你该修什么 |

## 延伸阅读

- [CS231n: Training Neural Networks](https://cs231n.github.io/neural-networks-3/) — 仍然是单页内最清晰的训练 pipeline 之旅
- [Bag of Tricks for Image Classification (He et al., 2019)](https://arxiv.org/abs/1812.01187) — 每个小技巧合在一起给 ResNet 在 ImageNet 上加 3-4%
- [mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)](https://arxiv.org/abs/1710.09412) — 原始 mixup 论文；三页理论加令人信服的实验
- [Why temperature scaling matters (Guo et al., 2017)](https://arxiv.org/abs/1706.04599) — 证明现代网络校准不良并用一个标量参数修复的论文
