# FlowShrink-PDHG

通过缩域加速 PDHG 在网络流或 MCNF 问题上的求解

---

## MCNF 问题描述

这里仅考虑原始的 MCNF 模型，不存在“异构流量”的存在，同时，为了避免问题退化为整数规划，这里仅考虑流量可分割为小数的情况。

### 数学模型

#### 1. 集合与参数 (Givens)

*   $G = (V, E)$：一个有向图，$V$ 是节点集合，$E$ 是有向边（弧）的集合。
*   $K$：商品（commodities）的集合。
*   对于每种商品 $k ∈ K$:
    *   $s_k ∈ V$：商品 $k$ 的源点。
    *   $t_k ∈ V$：商品 $k$ 的汇点。
    *   $d_k > 0$：商品 $k$ 的需求量。
*   对于每条边 $(i, j) ∈ E$:
    *   $u_{ij} ≥ 0$：边 $(i, j)$ 的总容量（所有商品共享）。
    *   $c_{ij}^k ≥ 0$：在边 $(i, j)$ 上运输一单位商品 $k$ 的成本。

#### 2. 决策变量 (Variables)

*   $x_{ij}^k$：表示在边 $(i, j)$ 上流动的商品 $k$ 的流量。

#### 3. 线性规划模型

**目标函数 (Objective Function):**

最小化总运输成本

$$
\text{Minimize} \quad Z = \sum_{k \in K} \sum_{(i, j) \in E} c_{ij}^k x_{ij}^k
$$

**约束条件 (Subject to):**

1.  **容量约束 (Capacity Constraints):**
    对于任意一条边，流经该边的所有商品的总流量不能超过该边的容量。
    $$
    \sum_{k \in K} x_{ij}^k \le u_{ij} \quad \forall (i, j) \in E
    $$

2.  **流量守恒约束 (Flow Conservation Constraints):**
    对于每一种商品和每一个节点，净流量必须等于该节点的供给/需求。
    $$
    \sum_{i: (i, j) \in E} x_{ij}^k - \sum_{l: (j, l) \in E} x_{jl}^k =
    \begin{cases}
    -d_k & \text{if } j = s_k \\
    d_k & \text{if } j = t_k \\
    0 & \text{otherwise}
    \end{cases}
    \quad \forall j \in V, \forall k \in K
    $$
    *   该公式表示：对于商品 $k$ 和节点 $j$，总流入量减去总流出量，必须等于节点 $j$ 对商品 $k$ 的“产出量”。源点 $s_k$ 产出 $d_k$（净流出为 $d_k$），汇点 $t_k$ 消耗 $d_k$（净流入为 $d_k$），中间节点不产出也不消耗（流入等于流出）。

3.  **非负约束 (Non-negativity Constraints):**
    流量不能为负。
    $$
    x_{ij}^k \ge 0 \quad \forall (i, j) \in E, \forall k \in K
    $$


---

**注:**

*   如果这是一个 **可行性问题**，则没有目标函数，只需找到满足所有约束条件的 $x_{ij}^k$ 即可。
*   如果要求流量为整数（例如，运输的货物不能分割），则 $x_{ij}^k$ 必须为整数变量。这将问题从一个（可在多项式时间内求解的）线性规划问题转变为一个（NP-hard的）**整数规划 (Integer Programming, IP)** 问题。


## 实现思路

主要分为两部分：

1. 通过最短路打分+筛边来进行原始问题可行域的缩域
2. 应用 PDHG 来求最优解

### coding env

包管理器使用 uv 或者 miniconda

- Python 3.11 ，用与之兼容的 PyTorch（最近稳定版 2.x）；
- CUDA 选 PyTorch 官方支持的 12.6 （`nvcc --version` 命令可以查看版本）；
- 其他科学库用与之兼容的最新稳定版本（NumPy、SciPy 等）。

### FlowShrink
> 在这里完善内容

### PDHG
> 在这里完善内容


## 性能测试

性能测试主要包括：
1. 缩域后 PDHG 迭代轮次
2. 整体求解时间/加速比

> 后续进行内容补充