# 范数与距离

> 你的距离函数定义了"相似"的含义。选错了，下游所有东西都会崩。

**类型：** Build
**语言：** Python
**前置知识：** Phase 1，第 01 课（线性代数直觉）、第 02 课（向量、矩阵与运算）
**时长：** 约 90 分钟

## 学习目标

- 从零实现 L1、L2、cosine、Mahalanobis、Jaccard 和 edit distance 函数
- 为给定的 ML 任务选择合适的距离度量，并解释为什么其他选项会失败
- 把 L1 和 L2 norm 与 LASSO、Ridge 正则化以及它们的几何约束区域联系起来
- 演示同一份数据在不同度量下会产生不同的最近邻

## The Problem

你有两个向量。它们可能是 word embeddings，可能是用户画像，也可能是像素数组。你需要知道：它们有多接近？

答案完全取决于你选哪个距离函数。两个数据点在某个度量下可能是最近邻，在另一个度量下却相隔甚远。你的 KNN 分类器、推荐引擎、向量数据库、聚类算法、损失函数——它们都依赖于这个选择。一旦选错，你的模型就会朝错误的目标优化。

不存在通用的最优距离。L2 适合空间数据。Cosine similarity 在 NLP 里占主导。Jaccard 处理集合。Edit distance 处理字符串。Mahalanobis 考虑特征相关性。Wasserstein 移动概率质量。每一个都编码了对"相似"含义的不同假设。

本课从零构建每一种主流距离函数，告诉你什么场景下用哪个最合适，并展示同一份数据在不同度量下会产生完全不同的最近邻。

## The Concept

### Norms：度量向量大小

norm 衡量向量的"大小"。任何两个向量之间的距离函数都可以写成它们差值的 norm：d(a, b) = ||a - b||。所以理解 norm 就是理解距离。

### L1 Norm（Manhattan distance）

L1 norm 是所有分量绝对值的和。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它被称为 Manhattan distance，因为它衡量的是在城市方格街道上行走的距离——你只能沿坐标轴方向移动，不能走对角线。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

在网格上，你向东走 3 个街区，向北走 4 个街区。
```

什么时候用 L1：
- 高维稀疏数据（文本特征、one-hot 编码）
- 需要对离群点鲁棒时（单个巨大差异不会主导结果）
- 特征选择问题（L1 正则化促进稀疏性）

与 L1 正则化（Lasso）的关系：把 ||w||_1 加到损失函数里，会惩罚权重绝对值之和。这会把小权重精确推到零，实现自动特征选择。L1 惩罚在权重空间里产生菱形约束区域，菱形的顶点位于坐标轴上——某些权重就是零的位置。

与损失函数的关系：Mean Absolute Error (MAE) 就是预测和目标之间的平均 L1 距离。它对所有误差都线性惩罚，相比 MSE 对离群点更鲁棒。

### L2 Norm（Euclidean distance）

L2 norm 是直线距离，即各分量平方和的平方根。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这就是你在几何课上学过的距离。n 维下的勾股定理。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

直线，斜着穿过网格。
```

什么时候用 L2：
- 低维到中维的连续数据
- 特征尺度可比时
- 物理距离（空间数据、传感器读数）
- 像素级图像相似度

与 L2 正则化（Ridge）的关系：把 ||w||_2^2 加到损失函数里，会惩罚大权重。和 L1 不同，它不会把权重推到零，而是按比例把所有权重缩向零。L2 惩罚产生圆形约束区域，因此坐标轴上没有顶点。权重会变小，但很少精确为零。

与损失函数的关系：Mean Squared Error (MSE) 是 L2 距离平方的均值。平方让大误差被惩罚得比小误差重得多。

```
MAE (L1 loss):  |y - y_hat|         线性惩罚。对离群点鲁棒。
MSE (L2 loss):  (y - y_hat)^2       二次惩罚。对离群点敏感。
```

### Lp Norms：通用家族

L1 和 L2 都是 Lp norm 的特例：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同的 p 值会产生不同形状的"单位球"（到原点距离为 1 的所有点的集合）：

```
p=1:    菱形              （顶点在坐标轴上）
p=2:    圆/球             （常规的圆球）
p=3:    超椭圆            （圆角方形）
p=inf:  方形/超立方体     （边沿坐标轴方向平直）
```

### L-infinity Norm（Chebyshev distance）

当 p 趋于无穷时，Lp norm 收敛到最大绝对分量。

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两点之间的距离由它们差异最大的那一个维度决定。其他维度都被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

什么时候用 L-infinity：
- 任何单一维度的最坏情况偏差很重要时
- 棋盘游戏（国际象棋里的国王走的就是 L-infinity：朝任何方向走一步代价都是 1）
- 制造公差（每个维度都必须在规格内）

### Cosine Similarity 与 Cosine Distance

cosine similarity 衡量两个向量之间的夹角，忽略它们的大小。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

取值范围从 -1（方向相反）到 +1（方向相同）。垂直向量的 cosine similarity 为 0。

cosine distance 把它转成距离：cosine_distance = 1 - cosine_similarity。取值范围从 0（方向相同）到 2（方向相反）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么 cosine 在 NLP 和 embeddings 里占主导：在文本中，文档长度不应该影响相似度。一篇关于猫的文档，长度是另一篇关于猫的文档的两倍，它们仍应"相似"。Cosine similarity 忽略大小（长度），只关心方向。两篇词分布相同但长度不同的文档指向同一方向，它们的 cosine similarity 是 1.0。

什么时候用 cosine similarity：
- 文本相似度（TF-IDF 向量、word embeddings、sentence embeddings）
- 任何"大小是噪声、方向是信号"的领域
- 推荐系统（用户偏好向量）
- Embedding 搜索（向量数据库几乎总是用 cosine 或 dot product）

### Dot Product Similarity 对比 Cosine Similarity

两个向量的 dot product 是：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

cosine similarity 就是 dot product 用两个 magnitudes 归一化后的结果。当两个向量都已经是单位归一化（magnitude = 1）时，dot product 和 cosine similarity 完全相同。

```
若 ||a|| = 1 且 ||b|| = 1：
    a . b = cos(a 和 b 之间的夹角)
```

它们什么时候不同：dot product 包含 magnitude 信息。magnitude 更大的向量得到更高的 dot product 分数。在某些检索系统里这很重要——你希望"热门"项目排名更靠前。Magnitude 充当了一种隐式的质量或重要性信号。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

二者在方向上一致，但 dot product 还反映了 magnitude。
```

实践中：
- 当你想要纯粹的方向相似性时，用 cosine similarity
- 当 magnitudes 携带有意义的信息时，用 dot product
- 许多向量数据库（Pinecone、Weaviate、Qdrant）让你二选一
- 如果你的 embeddings 已经做了 L2 归一化，选哪个都一样

### Mahalanobis Distance

Euclidean distance 平等对待所有维度。但如果你的特征相关或尺度不同，L2 会给出误导性的结果。

Mahalanobis distance 考虑了数据的 covariance 结构。

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据的 covariance matrix。

直观上：Mahalanobis distance 先把数据去相关并归一化（whitening），再在变换后的空间里计算 L2 距离。如果 S 是单位矩阵（特征不相关、单位方差），Mahalanobis distance 退化为 Euclidean distance。

```
例子：身高和体重是相关的。
6'2"、180 磅的人不奇怪。
5'0"、180 磅的人就奇怪了。

Euclidean distance 可能会说他们离均值一样远。
Mahalanobis distance 能正确识别第二个人是离群点，
因为它考虑了身高-体重的相关性。
```

什么时候用 Mahalanobis distance：
- 离群点检测（与均值的 Mahalanobis distance 较大的点是离群点）
- 特征尺度和相关性不同时的分类任务
- 数据足够多到能估计可靠的 covariance matrix
- 制造业质量控制（多元过程监控）

### Jaccard Similarity（用于集合）

Jaccard similarity 衡量两个集合的重叠度。

```
J(A, B) = |A intersect B| / |A union B|
```

取值范围从 0（无重叠）到 1（完全相同）。Jaccard distance = 1 - Jaccard similarity。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

交集 = {cat, fish}                  大小 = 2
并集 = {cat, dog, fish, bird, snake}  大小 = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

什么时候用 Jaccard：
- 比较标签、类别或特征集合
- 基于词出现（而非词频）的文档相似度
- 近重复检测（用 MinHash 近似 Jaccard）
- 比较二值特征向量（存在/不存在数据）
- 评估分割模型（Intersection over Union = Jaccard）

### Edit Distance（Levenshtein Distance）

edit distance 计算把一个字符串变换成另一个字符串所需的最小单字符操作数。操作有：插入、删除、替换。

```
"kitten" -> "sitting"

kitten -> sitten  （把 k 替换为 s）
sitten -> sittin  （把 e 替换为 i）
sittin -> sitting （插入 g）

Edit distance = 3
```

用动态规划计算。填一个矩阵，其中 (i, j) 项是字符串 A 前 i 个字符和字符串 B 前 j 个字符之间的 edit distance。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

什么时候用 edit distance：
- 拼写检查与纠错
- DNA 序列对齐（带权重的操作）
- 模糊字符串匹配
- 杂乱文本数据的去重

### KL Divergence（不是距离，但常被当成距离用）

KL divergence 衡量一个概率分布相对另一个的差异。Lesson 09 涵盖过它，但它属于本话题，因为人们尽管它不是距离，仍把它当"距离"用。

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键性质：KL divergence 不是对称的。

```
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它不满足距离度量的基本要求。它也不满足三角不等式。它是一个 divergence，不是 distance。

Forward KL（D_KL(P || Q)）是"mean-seeking"的：Q 试图覆盖 P 的所有 modes。
Reverse KL（D_KL(Q || P)）是"mode-seeking"的：Q 聚焦于 P 的某一个 mode。

你会在哪些地方看到 KL divergence：
- VAE（ELBO 中的 KL 项把隐变量分布推向 prior）
- 知识蒸馏（学生试图匹配教师的分布）
- RLHF（KL 惩罚让微调模型保持接近 base model）
- 策略梯度方法（约束策略更新）

### Wasserstein Distance（Earth Mover's Distance）

Wasserstein distance 衡量把一个概率分布变换成另一个所需的最小"功"。可以这样想：如果一个分布是一堆土，另一个分布是一个坑，你要搬多少土、搬多远？

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于一维分布，它简化为累积分布函数差的绝对值的积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

为什么 Wasserstein 很重要：
- 它是真正的 metric（对称、满足三角不等式）
- 即使分布没有重叠，它也能提供梯度（KL divergence 此时趋于无穷）
- 这一性质使它成为 Wasserstein GANs (WGANs) 的核心，从而解决了原版 GAN 的训练不稳定问题

```
没有重叠的分布：

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: 无穷（log(0)）
Wasserstein: 4（把所有质量挪 4 个 bin）

Wasserstein 给出有意义的梯度，KL 没有。
```

什么时候用 Wasserstein：
- GAN 训练（WGAN, WGAN-GP）
- 比较可能不重叠的分布
- 最优运输问题
- 图像检索（比较颜色直方图）

### 为什么不同任务需要不同距离

| 任务 | 最佳距离 | 原因 |
|------|--------------|-----|
| 文本相似度 | Cosine | magnitude 是噪声，direction 才是含义 |
| 图像像素比较 | L2 | 空间关系重要，特征尺度可比 |
| 高维稀疏特征 | L1 | 鲁棒，不会放大稀有的大差异 |
| 集合重叠（标签、类别） | Jaccard | 数据天然是集合形式，而非向量 |
| 字符串匹配 | Edit distance | 操作映射到人类编辑直觉 |
| 离群点检测 | Mahalanobis | 考虑特征相关性和尺度 |
| 比较分布 | KL divergence | 衡量用 Q 代替 P 损失的信息 |
| GAN 训练 | Wasserstein | 即便分布不重叠也能提供梯度 |
| Embeddings（向量数据库） | Cosine 或 dot product | Embeddings 被训练成在方向上编码含义 |
| 推荐 | Dot product | magnitude 可编码热度或置信度 |
| DNA 序列 | Weighted edit distance | 替换代价随核苷酸对而变 |
| 制造业 QC | L-infinity | 任一维度的最坏偏差才是关键 |

### 与损失函数的联系

损失函数是应用于"预测 vs 目标"的距离函数。

```
Loss function       使用的距离              行为
MSE                 L2 squared             重罚大误差
MAE                 L1                     所有误差等权惩罚
Huber loss          大误差用 L1，          兼具优点：对离群点鲁棒、
                    小误差用 L2            零附近梯度平滑
Cross-entropy       KL divergence          衡量分布不匹配
Hinge loss          max(0, margin - d)     仅对低于 margin 的部分惩罚
Triplet loss        L2（通常）             把正样本拉近、把负样本推远
Contrastive loss    L2                     相似对靠近、不相似对超过 margin
```

### 与正则化的联系

正则化在损失函数上加一个权重的 norm 惩罚。

```
L1 正则化 (Lasso):   loss + lambda * ||w||_1
  -> 稀疏权重。某些权重精确为零。
  -> 自动特征选择。
  -> 解有顶点（在零处不可导）。

L2 正则化 (Ridge):   loss + lambda * ||w||_2^2
  -> 权重变小。所有权重朝零收缩。
  -> 没有特征选择（不会精确为零）。
  -> 解处处平滑。

Elastic Net:         loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> 兼具 L1 的稀疏性和 L2 的稳定性。
  -> 相关特征组会被一起保留或一起丢弃。
```

为什么 L1 产生稀疏性而 L2 不会：想象 2D 权重空间里的约束区域。L1 是菱形，L2 是圆。损失函数的等高线（椭圆）最有可能在菱形的顶点处接触它，那里某个权重为零。它们和圆相切于一个光滑点，那里两个权重都非零。

### 最近邻搜索

每个距离函数都隐含一个最近邻搜索问题：给一个查询点，找到数据集中最接近的点。

精确最近邻搜索在 n 个 d 维点的数据集上每次查询是 O(n * d)。对于大数据集，这太慢了。

近似最近邻 (ANN) 算法用极小的精度损失换取大幅的速度提升：

```
算法              方法                          被谁使用
KD-trees          坐标轴对齐的空间划分         scikit-learn（低维）
Ball trees        嵌套超球体                   scikit-learn（中维）
LSH               随机哈希投影                 近重复检测
HNSW              分层可导航小世界图           FAISS, Qdrant, Weaviate
IVF               基于聚类搜索的倒排文件索引   FAISS（十亿级）
Product quant.    压缩向量后在压缩空间中搜索  FAISS（内存受限）
```

HNSW (Hierarchical Navigable Small World) 是现代向量数据库里的主流算法。它构建一个多层图，每个节点连向它的近似最近邻。搜索从顶层（稀疏、长跳跃）开始，往下降到底层（密集、短跳跃）。

## Build It

### Step 1: 全部 norm 与 distance 函数

完整实现见 `code/distances.py`。每个函数都只用基本的 Python 数学从零构建。

### Step 2: 同样的数据，不同的距离，不同的近邻

`distances.py` 里的 demo 创建一个数据集，挑选一个查询点，并展示最近邻如何随距离度量变化。在 L1 下"最近"的点，在 L2 或 cosine 下未必最近。

### Step 3: Embedding 相似度搜索

代码包含一个模拟 embedding 相似度搜索：用 cosine similarity 和 L2 distance 找出与查询最相似的"文档"，展示两种排名可能不同。

## Use It

最常见的实际用途：在向量数据库里找相似项。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 然后查询向量数据库时，引擎盖下发生的就是这件事。embedding 模型把文本映射成向量。向量数据库计算你查询向量和每个存储向量之间的 cosine similarity（或 dot product），并用 ANN 算法避免逐一检查所有向量。

## Exercises

1. 计算 (1, 2, 3) 和 (4, 0, 6) 之间的 L1、L2 和 L-infinity distance。验证对任意一对点都有 L-inf <= L2 <= L1。证明这一顺序为什么必然成立。

2. 构造两个向量，使得 cosine similarity 很高（> 0.9）但 L2 distance 很大（> 10）。从几何上解释发生了什么。然后构造两个向量，使得 cosine similarity 很低（< 0.3）但 L2 distance 很小（< 0.5）。

3. 实现一个函数：接受一个数据集和一个查询点，返回 L1、L2、cosine、Mahalanobis distance 下各自的最近邻。找出一个数据集，让这四种度量对哪个点最近的判断各不相同。

4. 用 CDF 方法手算 [0.5, 0.5, 0, 0] 和 [0, 0, 0.5, 0.5] 之间的 Wasserstein distance。然后算 [0.25, 0.25, 0.25, 0.25] 和 [0, 0, 0.5, 0.5] 之间的。哪个更大？为什么？

5. 实现 MinHash 用于近似 Jaccard similarity。生成 100 个随机集合，对所有对计算精确 Jaccard，并用 50、100 和 200 个哈希函数的 MinHash 近似进行比较。画出近似误差。

## Key Terms

| 术语 | 大家怎么说 | 实际含义 |
|------|----------------|----------------------|
| Norm | "向量的大小" | 把向量映射到非负标量的函数，满足三角不等式、绝对齐次性，且只对零向量取零 |
| L1 norm | "Manhattan distance" | 各分量绝对值之和。在优化中产生稀疏性。对离群点鲁棒 |
| L2 norm | "Euclidean distance" | 各分量平方和的平方根。Euclidean 空间里的直线距离 |
| Lp norm | "广义 norm" | 各分量绝对值的 p 次方之和的 p 次方根。L1 和 L2 是特例 |
| L-infinity norm | "Max norm" 或 "Chebyshev distance" | 最大绝对分量值。Lp 在 p 趋于无穷时的极限 |
| Cosine similarity | "向量间夹角" | dot product 用两个 magnitudes 归一化。范围 -1 到 +1。忽略向量长度 |
| Cosine distance | "1 减 cosine similarity" | 把 cosine similarity 转成距离。范围 0 到 2 |
| Dot product | "未归一化的 cosine" | 分量乘积之和。等于 cosine similarity 乘两个 magnitudes |
| Mahalanobis distance | "考虑相关性的距离" | 用数据 covariance matrix 做白化（去相关并归一化）后空间里的 L2 距离 |
| Jaccard similarity | "集合重叠" | 交集大小除以并集大小。用于集合，不是向量 |
| Edit distance | "Levenshtein distance" | 把一个字符串变成另一个所需的最小插入、删除、替换数 |
| KL divergence | "分布间距离" | 不是真正的距离（不对称）。衡量用 Q 编码 P 时的额外比特数 |
| Wasserstein distance | "Earth mover's distance" | 把质量从一个分布运到另一个分布的最小功。是真正的 metric |
| Approximate nearest neighbor | "ANN search" | 算法（HNSW, LSH, IVF）用以远快于精确搜索的速度找近似最近邻 |
| HNSW | "向量数据库的算法" | 分层可导航小世界图。多层图用于快速近似最近邻搜索 |
| L1 regularization | "Lasso" | 把权重的 L1 norm 加到损失里。把权重推向零（稀疏） |
| L2 regularization | "Ridge" 或 "weight decay" | 把权重的 L2 norm 平方加到损失里。把权重朝零收缩但不产生稀疏 |
| Elastic Net | "L1 + L2" | 结合 L1 和 L2 正则化。比单独使用更好地处理相关特征组 |

## Further Reading

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta 的十亿级 ANN 搜索库
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - 把 Earth Mover's distance 引入 GAN 的论文
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 奠基性的 ANN 算法
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec，让 cosine similarity 成为 embeddings 默认选择
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn 中距离度量与近邻算法的实用指南
