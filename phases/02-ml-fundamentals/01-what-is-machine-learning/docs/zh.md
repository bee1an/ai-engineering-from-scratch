# 什么是机器学习

> 机器学习就是让计算机从数据中发现规律，而不是手动编写规则。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 1（数学基础）
**时间：** 约 45 分钟

## 学习目标

- 解释监督学习、无监督学习和强化学习的区别，并判断给定问题属于哪种类型
- 从零实现一个最近质心分类器，并与随机基线进行对比评估
- 区分分类任务和回归任务，并为每种任务选择合适的 loss function
- 评估一个业务问题是否适合用 ML 解决，还是用确定性规则更好

## 问题

你想做一个垃圾邮件过滤器。传统做法：坐下来写几百条规则。"如果邮件包含 'FREE MONEY'，标记为垃圾。如果感叹号超过 3 个，标记为垃圾。" 你花了几周写规则。然后垃圾邮件发送者换了措辞。你的规则失效了。你再写更多规则。这个循环永远不会结束。

机器学习反转了这个过程。你不写规则，而是给计算机成千上万封标注好的邮件（"垃圾" 或 "非垃圾"），让它自己找出规则。计算机能发现你从未想到的模式。当垃圾邮件发送者改变策略时，你只需用新数据重新训练，而不是重写代码。

从"编写规则"到"从数据中学习"的转变，就是机器学习的核心。每一个推荐引擎、语音助手、自动驾驶汽车和语言模型都是这样工作的。

## 概念

### 从数据中学习，而非从规则

传统编程和机器学习以相反的方向解决问题。

```mermaid
flowchart LR
    subgraph Traditional["Traditional Programming"]
        direction LR
        R[Rules] --> P1[Program]
        D1[Data] --> P1
        P1 --> O1[Output]
    end

    subgraph ML["Machine Learning"]
        direction LR
        D2[Data] --> P2[Learning Algorithm]
        O2[Expected Output] --> P2
        P2 --> M[Model / Rules]
    end
```

传统编程：你编写规则。程序将规则应用于数据来产生输出。

机器学习：你提供数据和期望输出。算法自己发现规则。

训练出来的 "model" 就是规则本身，以数字（weights、parameters）的形式编码。它从见过的样本中泛化，对从未见过的数据做出预测。

### 机器学习的三大类型

```mermaid
flowchart TD
    ML[Machine Learning] --> SL[Supervised Learning]
    ML --> UL[Unsupervised Learning]
    ML --> RL[Reinforcement Learning]

    SL --> C[Classification]
    SL --> R[Regression]

    UL --> CL[Clustering]
    UL --> DR[Dimensionality Reduction]

    RL --> PO[Policy Optimization]
    RL --> VL[Value Learning]
```

**监督学习（Supervised Learning）**：你有输入-输出对。model 学习将输入映射到输出。
- "这里有 10,000 张标注了猫或狗的照片。学会区分它们。"
- "这里有房屋特征和价格。学会预测价格。"

**无监督学习（Unsupervised Learning）**：你只有输入，没有 label。model 自己发现数据中的结构。
- "这里有 10,000 条客户购买记录。找出自然分组。"
- "这里有 1,000 维的数据点。在保持结构的同时降到 2 维。"

**强化学习（Reinforcement Learning）**：一个 agent 在环境中采取行动并获得奖励或惩罚。它学习一种策略（policy）来最大化总奖励。
- "玩这个游戏。赢了 +1，输了 -1。找出策略。"
- "控制这个机械臂。拿起物体 +1，每浪费一秒 -0.01。"

实际工作中你构建的大部分东西都用监督学习。无监督学习常用于预处理和探索。强化学习驱动游戏 AI、机器人和语言模型的 RLHF。

### 超越三大类

上面三个类别很清晰，但现实中的 ML 经常模糊界限。

**半监督学习（Semi-supervised learning）** 使用少量标注数据和大量未标注数据。你可能有 100 张标注的医学图像和 100,000 张未标注的。技术包括：

- **标签传播（Label propagation）：** 构建一个连接相似数据点的图。标签从已标注节点通过图传播到未标注的邻居。
- **伪标签（Pseudo-labeling）：** 在标注数据上训练 model，用它预测未标注数据的标签，然后在所有数据上重新训练。model 自举自己的训练集。
- **一致性正则化（Consistency regularization）：** model 对一个输入和它的轻微扰动版本应该给出相同的预测。即使没有标签也能工作。

**自监督学习（Self-supervised learning）** 从数据本身创造监督信号。完全不需要人工标注。model 从数据的结构中创建自己的预测任务。

- **掩码语言建模（Masked language modeling，BERT）：** 隐藏句子中 15% 的词，训练 model 预测缺失的词。"标签"来自原始文本。
- **对比学习（Contrastive learning，SimCLR）：** 取一张图片，创建两个增强版本。训练 model 识别它们来自同一张图片，同时将它们与其他图片的增强版本区分开。
- **下一个 token 预测（Next-token prediction，GPT）：** 给定所有前面的词，预测下一个词。每个文本文档都变成一个训练样本。

这些不是与三大类别分开的类别。它们是结合了监督和无监督思想的策略。自监督学习技术上是监督的（model 在预测某些东西），但标签是自动生成的，不是人工标注的。

### 分类 vs 回归

这是监督学习的两大主要任务。

| 方面 | Classification | Regression |
|--------|---------------|------------|
| 输出 | 离散类别 | 连续数值 |
| 示例 | "这封邮件是垃圾邮件吗？" | "房价会是多少？" |
| 输出空间 | {猫, 狗, 鸟} | 任意实数 |
| Loss function | Cross-entropy, accuracy | Mean squared error, MAE |
| 决策 | 类别之间的边界 | 拟合数据的曲线 |

分类回答"哪个类别？" 回归回答"多少？"

有些问题可以用两种方式来定义。预测股票涨还是跌是分类。预测确切价格是回归。

### ML 工作流

每个机器学习项目都遵循相同的流程，无论使用什么算法。

```mermaid
flowchart LR
    A[Collect Data] --> B[Clean & Explore]
    B --> C[Feature Engineering]
    C --> D[Split Data]
    D --> E[Train Model]
    E --> F[Evaluate]
    F -->|Not good enough| C
    F -->|Good enough| G[Deploy]
    G --> H[Monitor]
    H -->|Performance drops| A
```

**收集数据**：收集原始数据。更多数据几乎总是更好，但质量比数量更重要。

**清洗与探索**：处理缺失值，去除重复，可视化分布，发现异常。这一步通常占项目总时间的 60-80%。

**特征工程**：将原始数据转换为 model 可以使用的 feature。把日期转成星期几。归一化数值列。编码分类变量。好的 feature 比花哨的算法更重要。

**划分数据**：分为 training set、validation set 和 test set。model 在训练数据上训练，你在验证数据上调超参数，在测试数据上报告最终性能。

**训练模型**：将训练数据输入算法。算法调整内部参数以最小化 loss function。

**评估**：在验证/测试数据上衡量性能。如果性能不可接受，回去尝试不同的 feature、算法或超参数。

**部署**：将 model 放入生产环境，对新数据做预测。

**监控**：跟踪性能随时间的变化。数据分布会变化（data drift），model 会退化。当性能下降时，重新训练。

### Training、Validation 和 Test 划分

这是初学者最容易搞错的概念。你必须在 model 训练时从未见过的数据上评估它。否则你衡量的是记忆能力，而不是学习能力。

```mermaid
flowchart LR
    subgraph Dataset["Full Dataset (100%)"]
        direction LR
        TR["Training Set (70%)"]
        VA["Validation Set (15%)"]
        TE["Test Set (15%)"]
    end

    TR -->|Train model| M[Model]
    M -->|Tune hyperparameters| VA
    VA -->|Final evaluation| TE
```

| 划分 | 目的 | 使用时机 | 典型大小 |
|-------|---------|-----------|-------------|
| Training | model 从这些数据中学习 | 训练期间 | 60-80% |
| Validation | 调超参数，比较 model | 每次训练后 | 10-20% |
| Test | 最终无偏性能估计 | 只用一次，在最后 | 10-20% |

Test set 是神圣的。你只看它一次。如果你不断根据测试性能调整 model，你实际上是在 test set 上训练，你报告的数字毫无意义。

对于小数据集，使用 k-fold cross-validation：将数据分成 k 份，在 k-1 份上训练，在剩余的一份上验证，轮转，取平均结果。

### Overfitting vs Underfitting

```mermaid
flowchart LR
    subgraph UF["Underfitting"]
        U1["Model too simple"]
        U2["High bias"]
        U3["Misses patterns"]
    end

    subgraph GF["Good Fit"]
        G1["Right complexity"]
        G2["Balanced"]
        G3["Generalizes well"]
    end

    subgraph OF["Overfitting"]
        O1["Model too complex"]
        O2["High variance"]
        O3["Memorizes noise"]
    end

    UF -->|Increase complexity| GF
    GF -->|Too much complexity| OF
```

**Underfitting**：model 太简单，无法捕捉数据中的模式。一条直线试图拟合曲线关系。训练误差高。测试误差高。

**Overfitting**：model 太复杂，记住了训练数据，包括其中的噪声。一条弯弯曲曲的曲线穿过每个训练点，但在新数据上表现很差。训练误差低。测试误差高。

**良好拟合**：model 捕捉了真实模式而没有记住噪声。训练误差和测试误差都合理地低。

Overfitting 的迹象：
- 训练准确率远高于验证准确率
- model 在训练数据上表现好但在新数据上表现差
- 增加更多训练数据能提升性能（说明 model 之前在记忆，而不是学习）

Overfitting 的解决方法：
- 获取更多训练数据
- 降低 model 复杂度（更少参数，更简单的架构）
- Regularization（对大权重添加惩罚）
- Dropout（训练时随机将神经元置零）
- Early stopping（当验证误差开始上升时停止训练）

Underfitting 的解决方法：
- 使用更复杂的 model
- 添加更多 feature
- 减少 regularization
- 训练更长时间

### Bias-Variance 权衡

这是 overfitting 和 underfitting 背后的数学框架。

**Bias**：model 中错误假设带来的误差。当真实关系是非线性时，线性 model 有高 bias。高 bias 导致 underfitting。

**Variance**：对训练数据中小波动的敏感性带来的误差。高 variance 的 model 在不同数据子集上训练时会给出非常不同的预测。高 variance 导致 overfitting。

| Model 复杂度 | Bias | Variance | 结果 |
|-----------------|------|----------|--------|
| 太低（对曲线数据用线性 model） | 高 | 低 | Underfitting |
| 刚好 | 中 | 中 | 良好泛化 |
| 太高（对 10 个点用 20 次多项式） | 低 | 高 | Overfitting |

总误差 = Bias^2 + Variance + 不可约噪声

你无法减少不可约噪声（它是数据本身的随机性）。你要找到 bias^2 + variance 最小化的甜蜜点。

### No Free Lunch 定理

没有一种算法对所有问题都是最好的。一种算法在某类问题上表现好，在另一类上就会表现差。这就是为什么数据科学家会尝试多种算法并比较结果。

实际中，选择取决于：
- 你有多少数据
- 有多少 feature
- 关系是线性还是非线性
- 是否需要可解释性
- 能承受多少计算开销

### 什么时候不该用机器学习

ML 很强大，但并不总是正确的工具。在使用 model 之前，先问问你是否真的需要一个。

**不要在以下情况使用 ML：**

- **规则简单且定义明确。** 税务计算、排序算法、单位转换。如果你能用几个 if 语句写出逻辑，model 只会增加复杂性而没有收益。
- **你没有数据或数据很少。** ML 需要样本来学习。只有 10 个数据点，你训练不出任何有意义的东西。先收集数据。
- **犯错的代价是灾难性的，你需要保证正确性。** 药物剂量计算、核反应堆控制、密码学验证。ML model 是概率性的。它们有时会出错。如果"有时出错"不可接受，使用确定性方法。
- **查找表或启发式规则就能解决问题。** 如果一个简单的阈值或表格覆盖了 99% 的情况，添加 ML 只会增加维护成本而没有实质改进。
- **你无法解释决策，而可解释性是必需的。** 受监管行业（贷款、保险、刑事司法）有时要求每个决策都完全可解释。一些 ML model 是可解释的（线性回归、小决策树）。大多数不是。
- **问题变化比你重新训练的速度更快。** 如果规则每天变化而重新训练需要一周，model 永远是过时的。

使用这个决策流程图：

```mermaid
flowchart TD
    A["Do you have data?"] -->|No| B["Collect data first or use rules"]
    A -->|Yes| C["Can you write the rules explicitly?"]
    C -->|"Yes, and they are simple"| D["Use rules. Skip ML."]
    C -->|"No, or they are too complex"| E["Is the cost of errors acceptable?"]
    E -->|"No, need guaranteed correctness"| F["Use deterministic methods"]
    E -->|Yes| G["Do you need explainability?"]
    G -->|"Yes, strictly"| H["Use interpretable models only"]
    G -->|"No, or partially"| I["Use ML"]
    I --> J["Do you have enough labeled data?"]
    J -->|Yes| K["Supervised learning"]
    J -->|"Some labels"| L["Semi-supervised learning"]
    J -->|"No labels"| M["Unsupervised or self-supervised"]
```

## 动手构建

`code/ml_intro.py` 中的代码从零实现了一个最近质心分类器，这是最简单的 ML 算法。它演示了核心思想：从数据中学习，然后对新数据做预测。

### 第 1 步：从零实现最近质心分类器

最近质心分类器计算训练数据中每个类别的中心（均值）。预测时，它将每个新点分配给中心最近的类别。

```python
class NearestCentroid:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.centroids = np.array([
            X[y == c].mean(axis=0) for c in self.classes
        ])

    def predict(self, X):
        distances = np.array([
            np.sqrt(((X - c) ** 2).sum(axis=1))
            for c in self.centroids
        ])
        return self.classes[distances.argmin(axis=0)]
```

这就是整个算法。Fit 计算两个均值。Predict 计算距离。没有 gradient descent，没有迭代，没有超参数。

### 第 2 步：在合成数据上训练

我们生成一个二维分类数据集，包含两个略有重叠的类别。质心分类器在类别中心之间画一条线性决策边界。

```python
rng = np.random.RandomState(42)
X_class0 = rng.randn(100, 2) + np.array([1.0, 1.0])
X_class1 = rng.randn(100, 2) + np.array([-1.0, -1.0])
X = np.vstack([X_class0, X_class1])
y = np.array([0] * 100 + [1] * 100)
```

### 第 3 步：与基线对比

每个 ML model 都应该与一个简单基线进行比较。这里，基线随机预测类别。如果你的 ML model 连随机猜测都打不过，那就有问题了。

```python
baseline_preds = rng.choice([0, 1], size=len(y_test))
baseline_acc = np.mean(baseline_preds == y_test)
```

在这个干净的数据集上，质心分类器应该能达到 90%+ 的准确率。随机基线大约 50%。

### 为什么这很重要

最近质心分类器极其简单。它没有超参数，没有迭代，没有 gradient descent。但它捕捉了 ML 的基本模式：

1. **学习**：从训练数据中学习一种表示（质心）
2. **预测**：使用该表示对新数据做预测（最近距离）
3. **评估**：与基线对比（随机猜测）

从逻辑回归到 transformer，每个 ML 算法都遵循这同样的三步模式。表示变得更复杂，但工作流保持不变。

### 第 4 步：质心分类器做不到什么

最近质心分类器假设每个类别形成一个单一的团。它画线性决策边界。它在以下情况会失败：

- 类别有多个簇（例如，数字 "1" 可以有几种不同的写法）
- 决策边界是非线性的（例如，一个类别包围另一个）
- Feature 的尺度差异很大（距离被最大尺度的 feature 主导）

这些局限性激发了你将学习的每一个其他算法。K-nearest neighbors 处理多个簇。决策树处理非线性边界。Feature scaling 解决尺度问题。每节课都建立在前一节课的局限性之上。

## 实际使用

sklearn 提供了 `NearestCentroid` 和合成数据生成器：

```python
from sklearn.neighbors import NearestCentroid
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=500, n_features=2, n_redundant=0,
    n_clusters_per_class=1, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = NearestCentroid()
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
```

## 交付物

本课产出 `outputs/prompt-ml-problem-framer.md` -- 一个将模糊业务问题转化为具体 ML 任务的 prompt。给它一个问题描述（"我们想减少流失" 或 "预测下季度需求"），它会识别学习类型、定义预测目标、列出候选 feature、选择成功指标、建立基线，并标记数据泄漏或类别不平衡等陷阱。在任何 ML 项目开始时使用它，避免构建错误的东西。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| Model | "那个 AI" | 一个带有可学习参数的数学函数，将输入映射到输出 |
| Training | "教 AI" | 运行优化算法来调整 model 参数，使预测匹配已知输出 |
| Feature | "一个输入列" | 数据的一个可测量属性，model 用它来做预测 |
| Label | "答案" | 训练样本的已知输出，用于计算误差信号 |
| Hyperparameter | "你调的设置" | 训练前设定的参数，控制学习过程（learning rate、层数） |
| Loss function | "model 错了多少" | 衡量预测输出和实际输出之间差距的函数，训练试图最小化它 |
| Overfitting | "它把测试背下来了" | model 学到了训练数据特有的噪声而非通用模式，所以在新数据上失败 |
| Underfitting | "它什么都没学到" | model 太简单，无法捕捉数据中的真实模式 |
| Generalization | "在新数据上有效" | model 对未训练过的数据做出准确预测的能力 |
| Cross-validation | "在不同块上测试" | 反复将数据分成训练/测试折叠并取平均结果，给出更稳健的性能估计 |
| Regularization | "让权重保持小" | 在 loss function 中添加惩罚项，阻止过于复杂的 model |
| Data drift | "世界变了" | 输入数据的统计分布随时间变化，导致 model 性能下降 |

## 练习

1. 取任意数据集（如 Iris、Titanic）。按 70/15/15 划分为 train/validation/test。解释为什么不应该在 test set 上调超参数。
2. 列出三个现实世界的问题。对每个问题，判断它是分类、回归还是聚类，以及是监督还是无监督。
3. 一个 model 在训练数据上获得 99% 准确率，但在测试数据上只有 60%。诊断问题并列出三件你会尝试的事情。

## 延伸阅读

- [An Introduction to Statistical Learning](https://www.statlearning.com/) - 免费教材，涵盖所有经典 ML 方法并附有实际示例
- [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) - 简洁的 ML 概念可视化入门
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) - 用 Python 实现 ML 的实用参考
