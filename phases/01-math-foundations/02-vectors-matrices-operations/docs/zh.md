# Vector、Matrix 与运算

> 每个神经网络本质上就是矩阵乘法加点额外步骤。

**类型：** 构建
**语言：** Python, Julia
**前置条件：** Phase 1, Lesson 01（线性代数直觉）
**时间：** 约 60 分钟

## 学习目标

- 构建一个 Matrix 类，包含逐元素运算、矩阵乘法、transpose、determinant 和 inverse
- 区分逐元素乘法和矩阵乘法，并解释各自的适用场景
- 仅使用从零构建的 Matrix 类实现一个全连接神经网络层（`relu(W @ x + b)`）
- 解释 broadcasting 规则以及 bias 加法在神经网络框架中如何工作

## 问题

你想构建一个神经网络。你读代码时看到这个：

```
output = activation(weights @ input + bias)
```

那个 `@` 是矩阵乘法。`weights` 是一个 matrix。`input` 是一个 vector。如果你不知道这些运算做什么，这行代码就是魔法。如果你知道，它就是一个层的整个前向传播——三个操作。

你的模型处理的每张图片都是像素值的 matrix。每个词 embedding 都是一个 vector。每个神经网络的每一层都是一个矩阵变换。不精通矩阵运算就无法构建 AI 系统，就像不理解变量就无法写代码一样。

本课从零构建这种流利度。

## 概念

### Vector：有序数字列表

Vector 是一组带方向和大小的数字。在 AI 中，vector 表示数据点、特征或参数。

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

2D vector `[3, 4]` 指向平面上的坐标 (3, 4)。它的长度（模长）是 5（3-4-5 三角形）。

### Matrix：数字网格

Matrix 是一个 2D 网格。行和列。一个 m x n 的 matrix 有 m 行 n 列。

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

在神经网络中，权重矩阵将输入 vector 变换为输出 vector。一个有 784 个输入和 128 个输出的层使用一个 128x784 的权重矩阵。

### 为什么形状很重要

矩阵乘法有严格规则：`(m x n) @ (n x p) = (m x p)`。内部维度必须匹配。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

如果你在 PyTorch 中遇到形状不匹配错误，原因就在这里。

### 运算一览

| 运算 | 做什么 | 神经网络用途 |
|-----------|-------------|-------------------|
| 加法 | 逐元素组合 | 给输出加 bias |
| 标量乘法 | 缩放每个元素 | 学习率 * gradient |
| 矩阵乘法 | 变换 vector | 层的前向传播 |
| Transpose | 翻转行列 | 反向传播 |
| Determinant | 单个数字摘要 | 检查可逆性 |
| Inverse | 撤销变换 | 求解线性系统 |
| 单位矩阵 | 什么都不做的 matrix | 初始化，残差连接 |

### 逐元素乘法 vs 矩阵乘法

这个区别经常绊倒初学者。

逐元素：对应位置相乘。两个 matrix 必须形状相同。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵乘法：行和列的 dot product。内部维度必须匹配。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

不同的运算，不同的结果，不同的规则。

### Broadcasting

当你把一个 bias vector 加到输出 matrix 上时，形状不匹配。Broadcasting 会拉伸较小的数组来适配。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

每个现代框架都自动做这件事。理解它能防止形状看起来不对但代码能跑时的困惑。

## 动手构建

### Step 1：Vector 类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Step 2：带核心运算的 Matrix 类

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Step 3：看它工作

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Step 4：连接到神经网络

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是一个全连接层：`output = relu(W @ x + b)`。每个神经网络中的每个全连接层都做的是这件事。

## 实际使用

NumPy 用更少的代码和快几个数量级的速度做上面所有的事。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 中的 `@` 运算符调用 `__matmul__`。NumPy 用 C 和 Fortran 编写的优化 BLAS 例程实现它。同样的数学，快 100 倍。

NumPy 中的 Broadcasting：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 自动将 1D bias 广播到两行。这就是每个神经网络框架中 bias 加法的工作方式。

## 产出

本课产出一个通过几何直觉教矩阵运算的 prompt。见 `outputs/prompt-matrix-operations.md`。

这里构建的 Matrix 类是我们在 Phase 3, Lesson 10 中构建的迷你神经网络框架的基础。

## 练习

1. **验证 inverse。** 将 `A @ A.inverse_2x2()` 相乘，确认得到单位矩阵。用三个不同的 2x2 矩阵试试。当 determinant 为零时会发生什么？

2. **实现 3x3 inverse。** 扩展 Matrix 类，使用伴随矩阵方法计算 3x3 矩阵的 inverse。用 NumPy 的 `np.linalg.inv` 测试。

3. **构建两层网络。** 仅使用你的 Matrix 类（不用 NumPy），创建一个两层神经网络：输入 (3) -> 隐藏层 (4) -> 输出 (2)。随机初始化权重，运行前向传播，验证所有形状正确。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------------|----------------------|
| Vector | "一个箭头" | 有序数字列表。在 AI 中：高维空间中的一个点。 |
| Matrix | "一张数字表" | 线性变换。将 vector 从一个空间映射到另一个空间。 |
| 矩阵乘法 | "把数字乘起来就行" | 第一个 matrix 的每行与第二个 matrix 的每列做 dot product。顺序有关系。 |
| Transpose | "翻转一下" | 交换行和列。将 m x n 矩阵变成 n x m。在反向传播中至关重要。 |
| Determinant | "矩阵的某个数" | 衡量 matrix 缩放面积（2D）或体积（3D）的程度。为零意味着变换压缩了一个维度。 |
| Inverse | "撤销矩阵" | 逆转变换的 matrix。只有当 determinant 不为零时才存在。 |
| 单位矩阵 | "无聊的矩阵" | 等价于乘以 1 的 matrix。用于残差连接（ResNets）。 |
| Broadcasting | "魔法形状修复" | 通过沿缺失维度重复来拉伸较小数组以匹配较大数组。 |
| 逐元素运算 | "普通乘法" | 对应位置相乘。两个数组必须形状相同（或可广播）。 |

## 延伸阅读

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - 本课涵盖的每个运算的视觉直觉
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 遵循的精确规则
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - ML 专用线性代数的简明参考
