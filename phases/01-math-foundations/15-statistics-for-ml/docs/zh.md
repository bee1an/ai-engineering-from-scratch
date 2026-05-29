# Statistics for Machine Learning

> 统计学是你判断模型究竟是真的有效，还是只是运气好的方式。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**Time:** ~120 minutes

## Learning Objectives

- 从零实现描述性统计量、Pearson/Spearman correlation 以及 covariance matrix
- 进行 hypothesis test（t-test、chi-squared）并正确解读 p-value 与 confidence interval
- 用 bootstrap resampling 为任意指标构造无需分布假设的 confidence interval
- 使用 effect size 度量来区分 statistical significance 与 practical significance

## The Problem

你训练了两个模型。Model A 在测试集上得 0.87，Model B 得 0.89。你部署了 Model B。三周后，线上指标比之前还差。发生了什么？

Model B 其实并没有真的优于 Model A。那 0.02 的差距只是噪声。你的测试集太小，或者方差太高，或者两者都有。你把随机性包装成了"改进"上线了。

这种事每天都在发生。Kaggle 排行榜大洗牌、复现失败的论文、只看几百个样本就宣告胜者的 A/B 测试。根因永远是同一个：有人跳过了统计学。

统计学给你工具去区分信号与噪声。它告诉你差异何时是真实的、你应该多大程度地相信它、以及在敢相信结果之前需要多少数据。每条 ML pipeline、每次模型对比、每个实验都需要统计学。没有它，你只是在猜。

## The Concept

### Descriptive Statistics: Summarizing Your Data

在建模之前，你得先知道数据长什么样。Descriptive statistics 把一份数据集压缩成几个能反映其形状的数字。

**Measures of central tendency** 回答的是"中间在哪里？"

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

Mean 是平衡点，median 是中点。当二者偏离时，分布就发生了 skew。收入分布的 mean 远大于 median（亿万富翁拉出的右偏）。训练时的 loss 分布往往 mean 远小于 median（容易样本造成的左偏）。

**Measures of spread** 回答的是"数据有多分散？"

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Percentiles** 把排序后的数据切成 100 等份。25th percentile（Q1）表示有 25% 的值落在该点之下。50th percentile 就是 median，75th percentile 就是 Q3。

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

在 ML 中，你会关心 percentile 用于推理延迟、预测置信度分布以及理解误差分布。一个平均误差很低但 P99 误差极差的模型，对于安全攸关的应用可能完全没法用。

**Sample vs population statistics.** 用样本计算 variance 时，要除以 (n-1) 而不是 n，这就是 Bessel's correction。它弥补了样本均值不是真实总体均值这一事实。如果分母是 n，你会系统性低估真实方差；用 (n-1)，估计就是无偏的。

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

实际中：n 很大时（数千样本）差异可以忽略；n 很小时（几十个样本）则不可忽略。

### Correlation: How Variables Move Together

Correlation 衡量两个变量之间线性关系的强度与方向。

**Pearson correlation coefficient** 衡量线性关联：

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson 假设关系是线性的，且两个变量大致服从正态分布。它对 outlier 很敏感，单个极端点就能把 r 从 0.1 拉到 0.9。

**Spearman rank correlation** 衡量单调关联：

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**何时使用哪一个：**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**黄金法则：** correlation 不意味着 causation。冰淇淋销量与溺亡人数相关，因为两者夏天都增加。你的模型准确率与参数量相关，但加参数并不一定能提高准确率（参见：overfitting）。

### Covariance Matrix

两个变量之间的 covariance 衡量它们如何共同变化：

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

对于 d 个特征，covariance matrix C 是一个 d x d 矩阵，C[i][j] = Cov(feature_i, feature_j)。对角元素 C[i][i] 是各特征的 variance。

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**与 PCA 的关系。** PCA 对 covariance matrix 做特征分解。eigenvector 就是 principal component（最大方差方向），eigenvalue 告诉你每个分量捕获了多少方差。这正是 Lesson 10 涉及的内容，但现在你能看出为什么要分解的对象是 covariance matrix：它编码了数据中所有的成对线性关系。

**与 correlation 的关系。** correlation matrix 就是对标准化后的变量（每个除以自身的 standard deviation）求得的 covariance matrix。Correlation 把 covariance 归一化，使所有取值落在 [-1, 1] 之间。

### Hypothesis Testing

Hypothesis testing 是在不确定性下做决策的一套框架。你先提出一个主张，然后收集数据，判断数据是否与该主张相符。

**基本设定：**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p-value** 是在 H0 为真的前提下，观测到当前这样极端（或更极端）数据的概率。它**不是** H0 为真的概率。这是统计学中最常见的误解。

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Confidence interval** 给出参数的一个合理取值区间：

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

Confidence interval 的宽度反映精度。区间宽意味着不确定性高；区间窄意味着估计精确（但若数据有偏，未必准确）。

### The t-test

t-test 用来比较均值，有几种变体。

**One-sample t-test：** 总体均值是否不同于某个假设值？

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Two-sample t-test (independent)：** 两组的均值是否不同？

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Paired t-test：** 当观测以配对形式出现（同一模型在相同 data split 上评测）：

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

ML 中 paired t-test 很常见：你在同样的 10 个 cross-validation fold 上跑两个模型，再两两配对地比较得分。

### Chi-squared Test

chi-squared test 检验观察频数是否与期望频数相符，适用于 categorical data。

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### A/B Testing for ML Models

ML 中的 A/B test 与网页 A/B test 不一样。模型对比有它独特的难点：

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**流程：**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### Statistical Significance vs Practical Significance

一个结果可以 statistically significant，却在实际中毫无意义。只要数据足够多，再微不足道的差异也会变得 statistically significant。

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Effect size** 量化差异的大小，与样本量无关：

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

p-value 与 effect size 都要报告。p-value 告诉你差异是否真实，effect size 告诉你这差异是否值得在意。

### Multiple Comparison Problem

当你检验很多假设时，总会有一些"显著"是出于偶然。在 alpha = 0.05 下做 20 次检验，即便实际上什么都没发生，你也预期会出现 1 次 false positive。

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni correction：** 将 alpha 除以检验次数。

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

在 ML 中，当你跨多种指标比较模型、尝试众多超参配置或在多个数据集上评估时，这一点尤其重要。

### Bootstrap Methods

Bootstrapping 通过对数据有放回地重采样来估计某个统计量的抽样分布，对原始分布无任何假设。

**算法：**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap confidence interval（百分位法）：**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**为什么 bootstrap 对 ML 很重要：**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**用 bootstrap 比较模型：**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

这比 paired t-test 更稳健，因为它对分布无任何假设。

### Parametric vs Non-parametric Tests

**Parametric tests** 假设数据服从某种特定分布（通常是正态）：

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Non-parametric tests** 不做分布假设：

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**何时使用 non-parametric：**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**何时使用 parametric：**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

ML 实验中，n 通常很小（5 或 10 折 cross-validation），所以 Wilcoxon signed-rank 这类 non-parametric test 往往比 t-test 更合适。

### Central Limit Theorem: Practical Implications

CLT 指出：无论原始总体分布如何，样本均值的分布在 n 增大时会趋于正态分布。

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**它对 ML 为什么重要：**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**CLT 不能做什么：**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### Common Statistical Mistakes in ML Papers

1. **在训练集上做评估。** 一定 overfit。永远要留出模型在训练时见不到的数据。

2. **没有 confidence interval。** 只报告一个准确率数字、不给不确定性，结果就既不可复现也不可验证。

3. **忽视 multiple comparisons。** 试了 50 个配置只汇报最好的那一个，不做修正，会大幅抬高 false positive 率。

4. **混淆 statistical significance 与 practical significance。** 0.01% 的准确率提升即便 p-value 是 0.001 也没意义。

5. **在不平衡数据上用 accuracy。** 99% 负类比例下 99% 的 accuracy 表示模型什么都没学到。要用 precision、recall、F1 或 AUC。

6. **挑指标。** 只报告自己模型胜出的那个指标。诚实的评估应汇报所有相关指标。

7. **train/test split 之间信息泄漏。** 在切分前做归一化，或者用未来数据预测过去。

8. **小测试集且没有方差估计。** 用 100 个样本评估然后宣称 2% 的提升，那是噪声不是信号。

9. **数据不独立时还假设独立。** 同一病人的多张医学影像、同一文档的多句话。组内观测是相关的。

10. **P-hacking。** 不停尝试不同的检验、子集或排除条件，直到 p < 0.05。这种结果只是搜索的产物。

## Building It

You will implement:

1. **Descriptive statistics from scratch** (mean, median, mode, standard deviation, percentiles, IQR)
2. **Correlation functions** (Pearson and Spearman, with the covariance matrix)
3. **Hypothesis tests** (one-sample t-test, two-sample t-test, chi-squared test)
4. **Bootstrap confidence intervals** (for any statistic, no assumptions needed)
5. **A/B test simulator** (generate data, test, check for Type I and Type II errors)
6. **Statistical vs practical significance demo** (showing that large n makes everything "significant")

全部从零实现，仅使用 `math` 和 `random`。不使用 numpy、不使用 scipy。

## Key Terms

| Term | Definition |
|---|---|
| Mean | 所有值之和除以个数。对 outlier 敏感。 |
| Median | 排序后处于中间位置的值。对 outlier 稳健。 |
| Standard deviation | variance 的平方根。以原始单位衡量离散度。 |
| Percentile | 数据中给定比例落在其下的取值。 |
| IQR | Interquartile range。Q3 减 Q1，即中间 50% 数据的跨度。 |
| Pearson correlation | 衡量两个变量间的线性关联。取值范围 [-1, 1]。 |
| Spearman correlation | 基于 rank 衡量单调关联。 |
| Covariance matrix | 所有特征两两 covariance 构成的矩阵。 |
| Null hypothesis | 默认的"无效应或无差异"假设。 |
| p-value | 在 null hypothesis 为真时观测到当前这般极端数据的概率。 |
| Confidence interval | 在给定置信水平下，参数的合理取值区间。 |
| t-test | 检验均值是否存在显著差异，使用 t-distribution。 |
| Chi-squared test | 检验观察频数与期望频数是否存在差异。 |
| Effect size | 差异的量级，与样本量无关。Cohen's d 是常见度量。 |
| Bonferroni correction | 将显著性阈值除以检验次数，以控制 false positive。 |
| Bootstrap | 有放回地重采样来估计抽样分布。 |
| Type I error | False positive。H0 为真却被 reject。 |
| Type II error | False negative。H0 为假却 fail to reject。 |
| Statistical power | 正确 reject 一个错误 H0 的概率。Power = 1 减去 Type II error rate。 |
| Central limit theorem | 样本量增大时，样本均值收敛到正态分布。 |
| Parametric test | 假设数据服从某特定分布（通常是正态）。 |
| Non-parametric test | 不做分布假设，基于 rank 或符号工作。 |
