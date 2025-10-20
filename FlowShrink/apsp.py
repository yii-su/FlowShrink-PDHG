import numpy as np


def apsp_floyd_warshall_matrix(W):
    """
    使用矩阵化的 Floyd-Warshall 算法计算所有节点对的最短路径 (APSP) 及路径。
    
    参数:
    W (np.array): 输入的邻接权重矩阵。0 或 inf 表示边不存在。
    
    返回:
    (np.array, np.array): 最终的距离矩阵 D 和前驱矩阵 P。
    """
    N = W.shape[0]
    
    # 1. 初始化距离矩阵 D
    D = np.where(W == 0, np.inf, W)
    np.fill_diagonal(D, 0)
    
    # 2. 初始化前驱矩阵 P
    # P[i, j] = j 表示从 j 到 i 的直接前驱是 j (如果 D[i,j] 不是 inf)
    initial_predecessors = np.arange(N)
    P = np.tile(initial_predecessors, (N, 1))
    # 对于自身到自身的情况，前驱可以设为自身
    np.fill_diagonal(P, np.arange(N))
    # 没有直接路径的地方，设置前驱为无效值
    P[np.isinf(D)] = -1
    
    print("--- 初始化的距离矩阵 D ---")
    print(np.round(D, 2))
    print("\n--- 初始化的前驱矩阵 P (-1 表示无路径) ---")
    print(P)
    
    # 3. 迭代 (Floyd-Warshall 逻辑)
    for k in range(N):
        print(f"\n--- 迭代轮次 k={k} (允许通过节点 {k}) ---")
        
        # 计算通过中间节点 k 的新路径距离
        D_candidates = D[:, k, np.newaxis] + D[k, :]
        # 找出比当前路径更短的新路径
        update_mask = D_candidates < D
        
        # 更新距离矩阵
        D[update_mask] = D_candidates[update_mask]
        
        # ----------- 修正部分 -----------
        # 如果从 j 到 i 的路径现在通过了 k，那么 i 的新前驱就是 k 的前驱
        # P_new(i, j) = P_old(k, j)
        if np.any(update_mask):
            # 创建一个 N x N 的矩阵，其中每一行都是 P[k, :]
            broadcasted_predecessors = np.tile(P[k, :], (N, 1))
            # 使用 update_mask 来选择正确的更新值，并赋给 P
            P[update_mask] = broadcasted_predecessors[update_mask]
        # --------------------------------
            
        if np.any(update_mask):
            print("更新后的距离矩阵 D:")
            print(np.round(D, 2))
            print("更新后的前驱矩阵 P:")
            print(P)
        else:
            print("本轮无更新。")
            
    return D, P
def reconstruct_path(P, source, dest):
    """根据前驱矩阵回溯最短路径"""
    # 检查输入是否有效
    if P[dest, source] == -1:
        return f"No path found from {source} to {dest}"
    
    path = [dest]
    curr = dest
    # 从目标节点向前回溯，直到回到源节点
    while curr != source:
        # P[curr, source] 给出的是在 "从source到curr" 的路径上，curr的前一个节点
        prev = int(P[curr, source])
        if prev == -1 or prev in path: # 添加了循环检测
            return f"Path reconstruction error or cycle detected near node {curr}"
        path.insert(0, prev)
        curr = prev
    return path
