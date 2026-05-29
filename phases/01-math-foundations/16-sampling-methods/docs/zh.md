# Sampling Methods

> Sampling 是 AI 探索可能性空间的方式。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**Time:** ~120 minutes

## Learning Objectives

- 仅使用均匀随机数，从零实现 inverse CDF、rejection sampling 和 importance sampling
- 为语言模型的 token 生成构建 temperature、top-k 和 top-p (nucleus) sampling
- 解释 reparameterization trick，以及它为何能让梯度在 VAE 的采样过程中反向传播
- 运行 Metropolis-Hastings MCMC，从一个未归一化的目标分布中抽样

## The Problem

一个语言模型刚处理完你的 prompt，输出了一个长度为 50,000 的 logits 向量。词表里每个 token 一个值。现在它得挑一个出来。怎么挑？

如果总是选概率最高的 token，每次回答都一模一样。确定性强，但乏味。如果完全随机均匀地选，输出就是乱码。答案介于这两个极端之间，而那个"中间地带"由 sampling 来掌控。

Sampling 不只是文本生成的事。强化学习通过采样轨迹来估计 policy gradient。VAE 通过从已学到的分布中采样、再让梯度穿过随机性来学习潜在表示。Diffusion 模型通过对噪声进行采样、迭代去噪来生成图像。Monte Carlo 方法用来估计没有解析解的积分。MCMC 算法在无法枚举的高维 posterior 分布中进行探索。

每一个生成式 AI 系统都是一个 sampling 系统。Sampling 策略直接决定了输出的质量、多样性和可控性。本课从均匀随机数开始，从零构建所有主流的 sampling 方法，最终覆盖支撑现代 LLM 和生成模型的核心技术。

## The Concept

### Why Sampling Matters

Sampling 在 AI 和机器学习中扮演四种基础角色：

**Generation。** 语言模型、diffusion 模型和 GAN 都是通过采样来产生输出的。采样算法直接决定了创造性、连贯性和多样性。Temperature、top-k 和 nucleus sampling 是工程师每天都在调的旋钮。

**Training。** Stochastic gradient descent 采样 mini-batch。Dropout 采样要失活的神经元。数据增强采样随机变换。Importance sampling 在强化学习（PPO、TRPO）中通过重新加权样本来降低梯度方差。

**Estimation。** 机器学习中很多量没有解析解。比如数据分布上的期望损失、能量模型的配分函数、贝叶斯推断中的 evidence。Monte Carlo estimation 通过对样本求平均来近似所有这些量。

**Exploration。** MCMC 算法在贝叶斯推断中探索 posterior 分布。Evolutionary strategies 采样参数扰动。Thompson sampling 在 bandit 问题中权衡 exploration 和 exploitation。

核心挑战在于：你只能直接从简单分布（uniform、normal）中采样。其他所有分布，都需要某种方法把简单样本转换成目标分布的样本。

### Uniform Random Sampling

每种 sampling 方法的起点都在这里。一个均匀随机数生成器产生 [0, 1) 之间的值，每一个等长的子区间出现的概率相等。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

要从含 n 个元素的离散集合中均匀采样，生成 U 然后返回 floor(n * U)。要从连续区间 [a, b] 中采样，计算 a + (b - a) * U。

关键洞察：单个均匀随机数所包含的随机性，正好够生成任意分布的一个样本。技巧在于找到正确的变换。

### Inverse CDF Method (Inverse Transform Sampling)

Cumulative distribution function (CDF) 把数值映射到概率：

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

Inverse CDF 把概率映射回数值。如果 U ~ Uniform(0, 1)，那么 X = F_inverse(U) 服从目标分布。

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Exponential distribution 的例子：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

只要 F_inverse 能写出闭式表达式，这种方法就完美适用。Normal distribution 的 inverse CDF 没有闭式形式，所以我们用别的方法（Box-Muller 或数值近似）。

**Discrete version:** 对离散分布，把 CDF 构建成累积和，生成 U，找到第一个累积和超过 U 的索引。这就是 Lesson 06 里 `sample_categorical` 的工作方式。

### Rejection Sampling

当你无法对 CDF 求逆，但能在差一个常数的意义下计算目标 PDF 时，rejection sampling 就派上用场。

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

界 M 越紧，接受率越高。低维（1-3 维）下，rejection sampling 表现很好。高维下，接受率指数级下降，因为大多数 proposal 体积都被拒掉了。这就是 rejection sampling 面临的维度灾难。

**例子：从截断正态分布采样。** 在截断范围上用 uniform proposal。包络 M 取该范围内 normal PDF 的最大值。

**例子：从半圆采样。** 在外接矩形里 uniform 提议。点落在半圆里则接受。这正是 Monte Carlo 计算 pi 的方式：接受率等于面积比 pi/4。

### Importance Sampling

有时你并不需要目标分布 p(x) 的样本。你需要的是 p(x) 下的某个期望，而你手上的样本来自另一个分布 q(x)。

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

这在强化学习里至关重要。在 PPO (Proximal Policy Optimization) 中，你用旧策略 pi_old 收集轨迹，但要优化的是新策略 pi_new。Importance weight 是 pi_new(a|s) / pi_old(a|s)。PPO 会对这些权重做 clip，防止新策略偏离旧策略太远。

Importance sampling 估计量的方差取决于 q 与 p 的相似程度。如果 q 和 p 差异很大，少数样本会拿到极大的权重，主导整个估计。Self-normalized importance sampling 用权重之和做归一化来缓解这个问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte Carlo Estimation

Monte Carlo estimation 通过对随机样本求平均来近似积分。大数定律保证收敛性。

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

误差率与维度无关。这就是为什么 Monte Carlo 方法在网格积分行不通的高维场景中占主导。

**估计 pi：**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**估计期望：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC): Metropolis-Hastings

MCMC 构造一条 Markov chain，使其平稳分布等于目标分布 p(x)。走过足够多的步之后，链上的样本（近似）就是来自 p(x) 的样本。

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

对于对称 proposal（q(x'|x) = q(x|x')），比值简化为 p(x')/p(x)。这就是最初的 Metropolis 算法。

**为什么有效。** 接受规则保证了 detailed balance：处于 x 并跳到 x' 的概率，等于处于 x' 并跳到 x 的概率。Detailed balance 意味着 p(x) 是这条链的平稳分布。

**实践中的注意事项：**
- Burn-in：在链未到达平衡前，丢弃早期样本
- Thinning：每隔 k 步留一个样本，降低自相关
- Proposal scale：太小则链移动缓慢（接受率高，探索慢）；太大则大多数 proposal 被拒（接受率低，原地不动）
- 高维下高斯 proposal 的最优接受率约为 0.234

### Gibbs Sampling

Gibbs sampling 是 MCMC 的一个特例，专门处理多元分布。它不是一次性在所有维度上提议移动，而是按维度逐个变量从条件分布中更新。

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs sampling 要求你能从每个条件分布 p(x_i | x_{-i}) 中采样。对许多模型来说这都很直接：
- Bayesian network：条件分布由图结构决定
- Gaussian mixture：条件分布是 Gaussian
- Ising model：每个 spin 的条件分布只依赖于它的邻居

接受率永远是 1（每个 proposal 都接受），因为从精确条件分布中采样自动满足 detailed balance。

**局限。** 当变量之间高度相关时，Gibbs sampling 混合得很慢，因为一次只更新一个变量，无法在分布上做大幅度的对角线移动。

### Temperature Sampling (Used in LLMs)

语言模型为词表中每个 token 输出 logits z_1, ..., z_V。Softmax 把它们变成概率。Temperature 在 softmax 之前对 logits 做重新缩放：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**为什么有效。** 把 logits 除以 T < 1 会放大它们之间的差距。如果 z_1 = 2、z_2 = 1，除以 T = 0.5 得到 z_1/T = 4 和 z_2/T = 2，差距变大了。经过 softmax 之后，logit 最高的 token 拿到大得多的份额。

**实践中：**
- T = 0.0：greedy decoding，最适合事实型 Q&A
- T = 0.3-0.7：略带创造性，适合代码生成
- T = 0.7-1.0：均衡型，适合一般对话
- T = 1.0-1.5：创意写作、头脑风暴
- T > 1.5：越来越随机，几乎用不上

Temperature 不会改变哪些 token 是可能的，它改变的是分配给每个 token 的概率质量。

### Top-k Sampling

Top-k sampling 把候选集合限制在概率最高的 k 个 token 上，然后重新归一化并从这个受限集合中采样。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k 防止模型选到词表长尾里那些概率极低的 token（错别字、胡话）。问题在于：k 是固定的，不随上下文变化。当模型很有信心时（某个 token 占了 95% 的概率），k = 40 仍然允许 39 个备选。当模型不确定时（概率分散在 1000 个 token 上），k = 40 又把合理选项切掉了。

### Top-p (Nucleus) Sampling

Top-p sampling 动态调整候选集的大小。它不再保留固定数量的 token，而是保留累积概率超过 p 的最小集合。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

模型有信心时，nucleus sampling 只留少数几个 token（也许 2-3 个）。模型不确定时，它会留很多（也许 200 个）。这种自适应行为，正是 nucleus sampling 通常比 top-k 产出更好文本的原因。

**常见组合：**
- Temperature 0.7 + top-p 0.9：通用场景的好选择
- Temperature 0.0（greedy）：确定性任务最优
- Temperature 1.0 + top-k 50：Fan et al. (2018) 原论文的设置

Top-k 和 top-p 可以叠加。先做 top-k，再在剩下的集合上做 top-p。

### Reparameterization Trick (Used in VAEs)

Variational autoencoder (VAE) 的学习方式是：把输入编码成潜在空间中的一个分布，从该分布采样，再把样本解码回去。问题是：你没法在采样操作上做反向传播。

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Reparameterization trick 把随机性从参数中分离出来：

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

它之所以成立，是因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 同分布。关键洞察：把随机性挪到一个不带参数的来源（epsilon），然后把样本写成参数的可微变换。

**在 VAE 训练循环中：**
1. Encoder 为每个输入输出 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 将 z 解码以重建输入
5. 反向传播经过步骤 4、3、2、1（步骤 3 可微，所以能传过去）

没有 reparameterization trick，VAE 没办法用标准反向传播来训练。这一个洞察让 VAE 真正具备了实用性。

### Gumbel-Softmax (Differentiable Categorical Sampling)

Reparameterization trick 适用于连续分布（Gaussian）。对于离散的 categorical 分布，需要另一种方法。Gumbel-Softmax 提供了 categorical sampling 的可微近似。

**Gumbel-Max trick（不可微）：**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax（可微近似）：**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax 给出离散样本的连续松弛版本。输出是一个概率向量（soft one-hot），而不是 hard one-hot。梯度可以穿过 softmax。训练前向传播时可以使用 "straight-through" estimator：前向用 hard argmax，反向用 soft Gumbel-Softmax 的梯度。

**Applications：**
- VAE 中的离散潜变量
- Neural architecture search（在离散操作之间选择）
- Hard attention 机制
- 含离散动作的强化学习

### Stratified Sampling

标准 Monte Carlo 采样可能会因为运气不好在样本空间留下空缺。Stratified sampling 通过把空间划分为若干 stratum 并在每一个里采样，强制实现均匀覆盖。

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Stratified sampling 的方差总是不大于标准 Monte Carlo：

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**Applications：**
- 数值积分（quasi-Monte Carlo）
- 训练集划分（在每一折中确保类别平衡）
- 带分层的 importance sampling（两种技术结合）
- NeRF (Neural Radiance Fields) 沿相机射线使用 stratified sampling

### Connection to Diffusion Models

Diffusion 模型通过一个采样过程生成图像。Forward process 在 T 步内不断给图像加 Gaussian 噪声，直到变成纯噪声。Reverse process 则学习去噪，一步步恢复原始图像。

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

它和本课方法的关联：
- 每一步去噪都用了 reparameterization trick（采噪声、做确定性变换）
- 噪声 schedule {alpha_t} 控制着一种 temperature annealing
- 训练时用 Monte Carlo estimation 来近似 ELBO (evidence lower bound)
- Diffusion 模型中的 ancestral sampling 是一条 Markov chain（每步只依赖当前状态）

整个图像生成过程就是一连串迭代采样：从噪声出发，每一步基于学到的去噪模型条件采样出一个噪声更小的版本。

## Build It

### Step 1: Uniform and inverse CDF sampling

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个 exponential 样本，验证均值是否为 1/lambda。

### Step 2: Rejection sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

用 rejection sampling 从截断正态分布中抽样。通过样本直方图来验证形状。

### Step 3: Importance sampling

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

用 uniform proposal 估计 normal 分布下的 E[X^2]。和已知答案（mu^2 + sigma^2）比对。

### Step 4: Monte Carlo estimation of pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Step 5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从一个双峰分布（两个 Gaussian 的混合）中采样。可视化链的轨迹。

### Step 6: Gibbs sampling

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Step 7: Temperature sampling

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示在一组 token logits 上 temperature 如何改变输出分布。

### Step 8: Top-k and top-p sampling

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Step 9: Reparameterization trick

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示梯度可以穿过重参数化的样本，但无法穿过直接采样。

### Step 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示 temperature 减小时输出如何趋近于 one-hot 向量。

完整实现和所有可视化在 `code/sampling.py` 中。

## Use It

用 NumPy 和 SciPy 的生产级写法：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

大规模 MCMC 用专门的库：
- PyMC：完整的贝叶斯建模，使用 NUTS（adaptive HMC）
- emcee：ensemble MCMC sampler
- NumPyro/JAX：GPU 加速的 MCMC

你已经从零搭好了这些。现在你知道库函数背后到底在做什么。

## Exercises

1. 为 Cauchy 分布实现 inverse CDF sampling。CDF 是 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个样本，把直方图与真实 PDF 一起作图。注意它的重尾（远离中心的极端值）。

2. 用 Uniform(0, 1) proposal 通过 rejection sampling 生成 Beta(2, 5) 分布的样本。把接受到的样本与真实 Beta PDF 一起作图。理论接受率是多少？

3. 用 Monte Carlo 分别以 1,000、10,000 和 100,000 个样本估计 sin(x) 在 0 到 pi 上的积分。比较各级别的误差。验证误差按 O(1/sqrt(N)) 缩放。

4. 实现 Metropolis-Hastings 从二维分布 p(x, y) 正比于 exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) 中采样。把样本和链的轨迹作图。试验不同的 proposal 标准差。

5. 构建一个完整的文本生成 demo：给定 10 个词的词表及其 logits，分别用 (a) greedy、(b) temperature=0.7、(c) top-k=3、(d) top-p=0.9 生成 20 个 token 的序列。比较 5 次运行中输出的多样性。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sampling | "Drawing random values" | 按概率分布生成数值。所有生成式 AI 背后的机制 |
| Uniform distribution | "All equally likely" | [a, b] 中每个值有相同的概率密度 1/(b-a)。所有 sampling 方法的起点 |
| Inverse CDF | "Probability transform" | F_inverse(U) 把均匀样本转成任何已知 CDF 的分布的样本。精确且高效 |
| Rejection sampling | "Propose and accept/reject" | 从一个简单 proposal 生成，按 target/proposal 比值的概率接受。精确但浪费样本 |
| Importance sampling | "Reweight samples" | 用 q(x) 的样本估计 p(x) 下的期望，每个样本按 p(x)/q(x) 加权。强化学习 PPO 的核心 |
| Monte Carlo | "Average random samples" | 用样本均值近似积分。误差为 O(1/sqrt(N))，与维度无关 |
| MCMC | "Random walk that converges" | 构造一条平稳分布等于目标的 Markov chain。Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | "Accept uphill, sometimes downhill" | 提议移动，按密度比接受。Detailed balance 保证收敛到目标分布 |
| Gibbs sampling | "One variable at a time" | 在固定其他变量的条件下，从条件分布更新每个变量。接受率 100% |
| Temperature | "Confidence knob" | 在 softmax 之前把 logits 除以 T。T<1 锐化（更自信），T>1 扁平化（更多样） |
| Top-k sampling | "Keep the k best" | 把除前 k 个最高概率 token 之外的全部置零，重新归一化、采样。候选集大小固定 |
| Nucleus sampling (top-p) | "Keep the probable ones" | 保留累积概率超过 p 的最小 token 集合。候选集大小自适应 |
| Reparameterization trick | "Move randomness outside" | 写成 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)。让采样可微。VAE 训练的关键 |
| Gumbel-Softmax | "Soft categorical sampling" | 用 Gumbel 噪声 + 带 temperature 的 softmax 实现 categorical sampling 的可微近似 |
| Stratified sampling | "Forced coverage" | 把样本空间划分为 stratum，每个采一个。方差总是不大于朴素 Monte Carlo |
| Burn-in | "Warm-up period" | 在链未到达平稳分布之前丢弃的初始 MCMC 样本 |
| Detailed balance | "Reversibility condition" | p(x) * T(x->y) = p(y) * T(y->x)。p 是 Markov chain 平稳分布的充分条件 |
| Diffusion sampling | "Iterative denoising" | 从噪声出发，应用学到的去噪步骤生成数据。每一步都是一次条件采样操作 |

## Further Reading

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 原论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - nucleus (top-p) sampling 论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 引入 reparameterization trick 的 VAE 论文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM 把采样与图像生成连接起来
