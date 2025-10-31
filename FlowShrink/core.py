# import torch

from decorators import gpu_timer

# # ============================================================
# # 2. GPU 版本的 min-plus 矩阵乘法
# # ============================================================

# def min_plus_matmul_gpu(W, D, P_old):
#     """
#     GPU 加速的 (min, +) 半环矩阵乘法
#     计算: D_new[i, j] = min_k { W[i, k] + D[k, j] }
#     同时更新前驱矩阵 P
    
#     参数:
#         W: torch.Tensor, shape (N, N), 权重矩阵
#         D: torch.Tensor, shape (N, N), 当前距离矩阵
#         P_old: torch.Tensor, shape (N, N), 当前前驱矩阵
    
#     返回:
#         D_new: torch.Tensor, shape (N, N), 新的距离矩阵
#         P_new: torch.Tensor, shape (N, N), 新的前驱矩阵
#     """
#     N = W.shape[0]
    
#     # 广播计算: W[i, k] + D[k, j]
#     # W.unsqueeze(1): (N, 1, N) - 对应 W[i, :, k]
#     # D.unsqueeze(0): (1, N, N) - 对应 D[:, k, j]
#     # 相加结果: (N, N, N) - [i, j, k] = W[i, k] + D[k, j]
#     path_costs = W.unsqueeze(1) + D.unsqueeze(0)  # (N, N, N)
    
#     # 沿着 k 维度(dim=2)取最小值
#     # D_new[i, j] = min_k { path_costs[i, j, k] }
#     # best_k[i, j] = argmin_k { path_costs[i, j, k] }
#     D_new, best_k = torch.min(path_costs, dim=2)
    
#     # best_k 就是前驱节点矩阵
#     P_new = best_k
    
#     return D_new, P_new


# # ============================================================
# # 3. GPU 版本的 APSP 核心算法
# # ============================================================

# @gpu_timer
# def apsp_gpu(W, device='cuda:0', max_iterations=None, convergence_check=True):
#     """
#     GPU 加速的全源最短路径算法 (APSP)
    
#     参数:
#         W: torch.Tensor, shape (N, N)
#            邻接权重矩阵，0 表示无边
#         device: str, GPU 设备
#         max_iterations: int or None, 最大迭代次数，None 时为 N-1
#         convergence_check: bool, 是否检查收敛（可提前终止）
    
#     返回:
#         D: torch.Tensor, shape (N, N), 最短距离矩阵
#         P: torch.Tensor, shape (N, N), 前驱节点矩阵
#         iterations: int, 实际迭代次数
#     """
#     # 确保输入在正确的设备上，使用 float16
#     W = W.to(device).to(torch.float16)
#     N = W.shape[0]
    
#     # 设置最大迭代次数
#     if max_iterations is None:
#         max_iterations = N - 1
    
#     # 1. 准备权重矩阵 W_prime
#     # 将 0 替换为 inf（表示无边），对角线设为 0
#     W_prime = torch.where(W == 0, torch.tensor(float('inf'), dtype=torch.float16, device=device), W)
#     W_prime.fill_diagonal_(0)
    
#     # 2. 初始化距离矩阵 D
#     D = W_prime.clone()
    
#     # 3. 初始化前驱矩阵 P（使用 long 类型存储节点索引）
#     P = torch.full((N, N), -1, dtype=torch.long, device=device)
    
#     # 初始化前驱矩阵：有直接边的位置，前驱为源节点
#     mask = W_prime < float('inf')
#     P[mask] = torch.arange(N, device=device).unsqueeze(0).expand(N, N)[mask]
    
#     # 4. 迭代执行 SpMM
#     actual_iterations = 0
    
#     for iteration in range(max_iterations):
#         D_old = D.clone()
        
#         # 核心操作: (D_candidates, P_candidates) = W' ⊕.⊗ (D_old, P_old)
#         D_candidates, P_candidates = min_plus_matmul_gpu(W_prime, D_old, P)
        
#         # 更新：只有当新路径更短时才更新
#         update_mask = D_candidates < D
#         D = torch.where(update_mask, D_candidates, D)
#         P = torch.where(update_mask, P_candidates, P)
        
#         actual_iterations += 1
        
#         # 5. 收敛检查
#         if convergence_check and torch.equal(D, D_old):
#             break
    
#     return D, P, actual_iterations


# # ============================================================
# # 4. CPU 上的路径重建函数
# # ============================================================

# def reconstruct_path(P, src, dst):
#     """
#     根据前驱矩阵重建从 src 到 dst 的最短路径
#     在 CPU 上执行
    
#     参数:
#         P: torch.Tensor or numpy.ndarray, 前驱节点矩阵
#         src: int, 源节点
#         dst: int, 目标节点
    
#     返回:
#         list: 路径节点列表 [src, ..., dst]，不可达时返回 None
#     """
#     # 转换到 CPU 和 NumPy（如果需要）
#     if torch.is_tensor(P):
#         P = P.cpu().numpy()
    
#     # 检查是否可达
#     if P[dst, src] == -1:
#         return None
    
#     # 反向追溯路径
#     path = []
#     current = dst
#     visited = set()
#     max_steps = P.shape[0] + 1
    
#     while current != src and len(visited) < max_steps:
#         if current in visited:
#             print(f"警告：检测到路径循环")
#             return None
        
#         visited.add(current)
#         path.append(int(current))
#         current = int(P[current, src])
        
#         if current == -1:
#             return None
    
#     path.append(int(src))
#     path.reverse()
    
#     return path


# def reconstruct_all_paths(P, src):
#     """
#     重建从源点 src 到所有其他节点的路径
    
#     参数:
#         P: torch.Tensor or numpy.ndarray, 前驱矩阵
#         src: int, 源节点
    
#     返回:
#         dict: {dst: path_list} 映射
#     """
#     if torch.is_tensor(P):
#         N = P.shape[0]
#     else:
#         N = P.shape[0]
    
#     paths = {}
#     for dst in range(N):
#         if dst != src:
#             path = reconstruct_path(P, src, dst)
#             if path is not None:
#                 paths[dst] = path
    
#     return paths


# # ============================================================
# # 5. 辅助函数：打印路径信息
# # ============================================================

# def print_path_info(W_original, D, P, src, dst):
#     """
#     打印从 src 到 dst 的路径详细信息
    
#     参数:
#         W_original: 原始权重矩阵（用于显示边权重）
#         D: 距离矩阵
#         P: 前驱矩阵
#         src: 源节点
#         dst: 目标节点
#     """
#     # 转到 CPU
#     if torch.is_tensor(W_original):
#         W_original = W_original.cpu()
#     if torch.is_tensor(D):
#         D = D.cpu()
#     if torch.is_tensor(P):
#         P = P.cpu()
    
#     path = reconstruct_path(P, src, dst)
#     distance = D[dst, src].item()
    
#     print(f"\n从节点 {src} 到节点 {dst}:")
    
#     if path is not None:
#         path_str = " → ".join(map(str, path))
#         print(f"  路径: {path_str}")
#         print(f"  距离: {distance:.6f}")
        
#         # 验证路径
#         if len(path) > 1:
#             total = 0
#             edges = []
#             for i in range(len(path) - 1):
#                 weight = W_original[path[i+1], path[i]].item()
#                 total += weight
#                 edges.append(f"{path[i]}→{path[i+1]}({weight:.3f})")
#             print(f"  详细: {' + '.join(edges)} = {total:.6f}")
#     else:
#         print(f"  状态: 不可达")




import torch
import time
from functools import wraps

# ============================================================
# 计算图的稀疏度
# ============================================================

def compute_sparsity(W):
    """
    计算图的稀疏度（非零边的比例）
    
    参数:
        W: torch.Tensor, 邻接矩阵
    
    返回:
        float: 稀疏度 (0-1 之间)
    """
    N = W.shape[0]
    num_edges = torch.count_nonzero(W).item()
    total_possible = N * N
    return num_edges / total_possible


# ============================================================
# 方法1：稠密矩阵方法（原实现，适用于稠密图）
# ============================================================

def min_plus_matmul_dense(W, D, P_old):
    """
    稠密矩阵版本的 min-plus 矩阵乘法
    适用于稠密图
    """
    path_costs = W.unsqueeze(1) + D.unsqueeze(0)
    D_new, best_k = torch.min(path_costs, dim=2)
    P_new = best_k
    return D_new, P_new


# ============================================================
# 方法2：稀疏边列表方法（新增，适用于稀疏图）
# ============================================================

def min_plus_matmul_sparse(W, D, P_old, edge_list):
    """
    基于边列表的 min-plus 矩阵乘法
    只处理实际存在的边，避免无效计算
    
    参数:
        W: torch.Tensor, 权重矩阵
        D: torch.Tensor, 当前距离矩阵
        P_old: torch.Tensor, 当前前驱矩阵
        edge_list: torch.Tensor, shape (num_edges, 2), 边列表 [from, to]
    
    返回:
        D_new, P_new
    """
    N = W.shape[0]
    device = W.device
    
    # 初始化为当前值
    D_new = D.clone()
    P_new = P_old.clone()
    
    # 提取边的起点和终点
    from_nodes = edge_list[:, 0]  # (num_edges,)
    to_nodes = edge_list[:, 1]    # (num_edges,)
    
    # 提取边的权重
    edge_weights = W[to_nodes, from_nodes]  # (num_edges,)
    
    # 对每个源节点 j 进行处理
    for j in range(N):
        # 当前从源点 j 到所有节点的距离
        D_j = D[:, j]  # (N,)
        
        # 对于每条边 (from_node -> to_node)
        # 计算新路径: D[from_node, j] + W[to_node, from_node]
        new_distances = D_j[from_nodes] + edge_weights  # (num_edges,)
        
        # 更新: 对每个 to_node，检查是否有更短路径
        for idx, (to_node, from_node, new_dist) in enumerate(zip(to_nodes, from_nodes, new_distances)):
            if new_dist < D_new[to_node, j]:
                D_new[to_node, j] = new_dist
                P_new[to_node, j] = from_node
    
    return D_new, P_new


# ============================================================
# 方法3：批量稀疏更新（优化版）
# ============================================================

def min_plus_matmul_sparse_batched(W, D, P_old, edge_list, edge_weights):
    """
    批量处理的稀疏边列表方法
    通过向量化操作加速
    
    参数:
        W: torch.Tensor, 权重矩阵
        D: torch.Tensor, 当前距离矩阵 (N, N)
        P_old: torch.Tensor, 当前前驱矩阵
        edge_list: torch.Tensor, shape (num_edges, 2)
        edge_weights: torch.Tensor, shape (num_edges,)
    
    返回:
        D_new, P_new
    """
    N = W.shape[0]
    num_edges = edge_list.shape[0]
    device = W.device
    
    D_new = D.clone()
    P_new = P_old.clone()
    
    from_nodes = edge_list[:, 0]
    to_nodes = edge_list[:, 1]
    
    # 对每个源节点 j 进行向量化处理
    for j in range(N):
        # 新路径长度: D[from_nodes, j] + edge_weights
        new_distances = D[from_nodes, j] + edge_weights  # (num_edges,)
        
        # 当前到 to_nodes 的距离
        current_distances = D_new[to_nodes, j]  # (num_edges,)
        
        # 找出需要更新的边
        update_mask = new_distances < current_distances
        
        if update_mask.any():
            # 批量更新
            update_to_nodes = to_nodes[update_mask]
            update_from_nodes = from_nodes[update_mask]
            update_distances = new_distances[update_mask]
            
            D_new[update_to_nodes, j] = update_distances
            P_new[update_to_nodes, j] = update_from_nodes
    
    return D_new, P_new


# ============================================================
# 核心：自适应 APSP 算法
# ============================================================

@gpu_timer
def apsp_gpu_adaptive(W, device='cuda:0', max_iterations=None, 
                      convergence_check=True, sparsity_threshold=0.3):
    """
    自适应的 GPU APSP 算法
    根据图的稀疏度自动选择最优算法
    
    参数:
        W: torch.Tensor, 邻接矩阵
        device: str, GPU 设备
        max_iterations: int or None
        convergence_check: bool
        sparsity_threshold: float, 稀疏度阈值（超过则用稠密方法）
    
    返回:
        D, P, iterations
    """
    W = W.to(device).to(torch.float16)
    N = W.shape[0]
    
    if max_iterations is None:
        max_iterations = N - 1
    
    # 计算稀疏度
    sparsity = compute_sparsity(W)
    print(f"图的稀疏度: {sparsity:.4f} ({torch.count_nonzero(W).item()} 条边 / {N*N} 可能边)")
    
    # 选择算法
    use_sparse = sparsity < sparsity_threshold
    
    if use_sparse:
        print(f"✓ 使用稀疏矩阵方法（稀疏度 {sparsity:.4f} < {sparsity_threshold}）")
        
        # 提取边列表（只在初始化时执行一次）
        edge_indices = torch.nonzero(W, as_tuple=False)  # (num_edges, 2)
        edge_weights = W[edge_indices[:, 0], edge_indices[:, 1]]
        
        print(f"  边数: {edge_indices.shape[0]}")
    else:
        print(f"✓ 使用稠密矩阵方法（稀疏度 {sparsity:.4f} >= {sparsity_threshold}）")
    
    # 初始化
    W_prime = torch.where(W == 0, torch.tensor(float('inf'), dtype=torch.float16, device=device), W)
    W_prime.fill_diagonal_(0)
    
    D = W_prime.clone()
    P = torch.full((N, N), -1, dtype=torch.long, device=device)
    
    mask = W_prime < float('inf')
    P[mask] = torch.arange(N, device=device).unsqueeze(0).expand(N, N)[mask]
    
    # 迭代
    actual_iterations = 0
    
    for iteration in range(max_iterations):
        D_old = D.clone()
        
        # 根据选择的方法执行
        if use_sparse:
            D_candidates, P_candidates = min_plus_matmul_sparse_batched(
                W_prime, D_old, P, edge_indices, edge_weights
            )
        else:
            D_candidates, P_candidates = min_plus_matmul_dense(
                W_prime, D_old, P
            )
        
        # 更新
        update_mask = D_candidates < D
        D = torch.where(update_mask, D_candidates, D)
        P = torch.where(update_mask, P_candidates, P)
        
        actual_iterations += 1
        
        if convergence_check and torch.equal(D, D_old):
            break
    
    return D, P, actual_iterations


# ============================================================
# 路径重建（保持不变）
# ============================================================

def reconstruct_path(P, src, dst):
    """路径重建函数"""
    if torch.is_tensor(P):
        P = P.cpu().numpy()
    
    if P[dst, src] == -1:
        return None
    
    path = []
    current = dst
    visited = set()
    max_steps = P.shape[0] + 1
    
    while current != src and len(visited) < max_steps:
        if current in visited:
            return None
        visited.add(current)
        path.append(int(current))
        current = int(P[current, src])
        if current == -1:
            return None
    
    path.append(int(src))
    path.reverse()
    return path
