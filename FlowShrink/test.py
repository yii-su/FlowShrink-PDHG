import numpy as np

def spmm_min_plus(W, D):
    """
    执行一次基于 (min, +) 半环的稀疏矩阵-矩阵乘法。
    在 NumPy 中，这可以通过广播和维度扩展来实现。
    
    W: 权重矩阵 (N x N)
    D: 当前距离矩阵 (N x k)
    
    返回一个新的距离矩阵 (N x k)。
    """
    # W(i, k) + D(k, j)
    # W          (N, N, 1) -> 扩展一个维度用于和D广播
    # D.T        (k, N)
    # W + D.T    (N, N, k) -> 广播后相加
    #
    # 结果的 D_new(i, j) 在 W[i, :, j] + D[:, j] 这个视图上
    # 为了直观，我们还是用更清晰的循环，因为这里的性能不是首要考量
    # 对于大规模问题，下面这种for循环效率极低，但对于理解算法逻辑非常清晰
    
    N, k = D.shape
    D_new = np.full((N, k), np.inf)

    for i in range(N):
        for j in range(k):
            # 计算 D_new(i, j) = min_m { W(i, m) + D(m, j) }
            # 这等价于 W 的第 i 行和 D 的第 j 列的 (min, +) "点积"
            min_val = np.min(W[i, :] + D[:, j])
            D_new[i, j] = min_val
            
    return D_new

def print_matrix(name, matrix):
    """一个格式化打印矩阵的辅助函数"""
    print(f"--- {name} ---")
    with np.printoptions(linewidth=200, formatter={'float': '{:6.0f}'.format}):
        print(matrix)
    print("\n")

# --- 1. 定义图和初始化 ---

# 节点数量
N = 4

# 初始化权重矩阵 W
# W(i, j) 代表从 j到 i 的权重
W = np.full((N, N), np.inf)

# 填充边权重
W[1, 0] = 10  # 0 -> 1
W[2, 0] = 3   # 0 -> 2
W[1, 1] = 0   # self-loop
W[2, 1] = 1   # 1 -> 2
W[3, 1] = 2   # 1 -> 3
W[2, 2] = 0   # self-loop
W[3, 2] = 4   # 2 -> 3
W[0, 0] = 0   # self-loop
W[3, 3] = 0   # self-loop

print_matrix("权重矩阵 W (W_ij = j->i 的权重)", W)

# 初始化距离矩阵 D (APSP, 所有节点都是源点)
D = np.full((N, N), np.inf)
np.fill_diagonal(D, 0)

print_matrix("初始距离矩阵 D_0", D)


# --- 2. 算法迭代 ---

# 算法最多在 N-1 轮后收敛 (如果没有负权环)
max_rounds = N - 1

D_current = D.copy()

for i in range(max_rounds):
    print(f"==================== Round {i + 1} ====================\n")
    
    # 记录上一轮的结果以便比较
    D_previous = D_current.copy()
    
    # 执行一次 (min, +) SpMM 操作
    D_candidates = spmm_min_plus(W, D_current)
    print_matrix(f"候选距离矩阵 D_{i+1}_candidates", D_candidates)
    
    # 按元素取最小值，合并新发现的更短路径
    D_current = np.minimum(D_previous, D_candidates)
    
    print_matrix(f"更新后的距离矩阵 D_{i+1}", D_current)
    
    # --- 3. 检查终止条件 ---
    if np.array_equal(D_current, D_previous):
        print("矩阵不再变化，算法收敛，提前终止。")
        break

print("==================== 最终结果 ====================")
print_matrix("最终最短路径矩阵 D_final", D_current)
