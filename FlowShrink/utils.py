from collections import deque

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

def create_base_network(N, k, seed=None):
    """
    创建基于k-近邻的有向图初始网络。
    
    参数:
        N (int): 节点数量
        k (int): 每个节点的出度（向外连接的邻居数量）
        seed (int, optional): 随机数种子
    
    返回:
        adj_matrix (np.ndarray): N×N 有向带权邻接矩阵
        node_list (np.ndarray): N×2 节点坐标
    """
    rng = np.random.default_rng(seed)
    
    # 生成节点坐标
    node_list = rng.random((N, 2))
    
    # 初始化邻接矩阵
    # adj_matrix = np.full((N, N), np.inf)
    adj_matrix = np.full((N, N), 0.)
    np.fill_diagonal(adj_matrix, 0)
    
    # 为每个节点添加k条出边
    for i in range(N):
        # 计算到所有其他节点的距离
        distance = np.linalg.norm(node_list[i] - node_list, axis=1)
        
        # 找到k个最近邻居（排除自己）
        neighbors = np.argsort(distance)[1:(k+1)]
        
        # 为每条边生成对数均匀分布的权重 [0.5, 5]
        costs = np.exp(rng.random(k) * (np.log(5) - np.log(0.5)) + np.log(0.5))
        
        # 添加有向边：从i到neighbors
        adj_matrix[i, neighbors] = costs
    
    return adj_matrix


def ensure_weak_connectivity(adj_matrix, seed=None):
    """
    检查有向图的弱连通性，如果不满足则增边使其弱连通。
    
    参数:
        adj_matrix (np.ndarray): N×N 有向带权邻接矩阵
        node_list (np.ndarray): N×2 节点坐标
        seed (int, optional): 随机数种子，仅在需要增边时使用
    
    返回:
        connected_adj_matrix (np.ndarray): 弱连通的 N×N 有向带权邻接矩阵
    """
    N = adj_matrix.shape[0]
    
    # 创建无向版本检查弱连通性
    # 如果 adj[i,j] 或 adj[j,i] 有边，则无向图中 i-j 有边
    undirected = np.isfinite(adj_matrix) | np.isfinite(adj_matrix.T)
    undirected = undirected & (undirected != np.eye(N, dtype=bool))
    
    # 检查连通分量
    n_components, labels = connected_components(
        csgraph=csr_matrix(undirected), 
        directed=False, 
        return_labels=True
    )
    
    # 如果已经弱连通，直接返回
    if n_components == 1:
        return adj_matrix.copy()
    
    # 需要增边，创建随机数生成器
    rng = np.random.default_rng(seed)
    
    # 复制邻接矩阵以避免修改原始数据
    result_adj = adj_matrix.copy()
    
    # 连接相邻编号的连通分量
    for comp_id in range(n_components - 1):
        # 找到属于当前分量和下一个分量的节点
        nodes_comp_i = np.where(labels == comp_id)[0]
        nodes_comp_next = np.where(labels == comp_id + 1)[0]
        
        # 随机选择两个节点
        u = rng.choice(nodes_comp_i)
        v = rng.choice(nodes_comp_next)
        
        # 生成随机权重（对数均匀分布 [0.5, 5]）
        cost = np.exp(rng.random() * (np.log(5) - np.log(0.5)) + np.log(0.5))
        
        # 添加双向边
        result_adj[u, v] = cost
        result_adj[v, u] = cost
    
    return result_adj


def adjacency_to_incidence(adj_matrix):
    """
    将有向邻接矩阵转换为关联矩阵和成本向量。
    
    参数:
        adj_matrix (np.ndarray): N×N 有向带权邻接矩阵（已确保弱连通）
    
    返回:
        A (np.ndarray): 关联矩阵 (N, M)，M为边数
        c (np.ndarray): 边的成本向量，长度M
    """
    N = adj_matrix.shape[0]
    
    # 找到所有有向边
    sources, destinations = np.where((adj_matrix < np.inf) & (adj_matrix > 0))
    
    # 边的数量
    M = len(sources)
    
    # 初始化关联矩阵和成本向量
    A = np.zeros((N, M))
    c = np.zeros(M)
    
    # 构建关联矩阵
    for idx, (u, v) in enumerate(zip(sources, destinations)):
        A[u, idx] = -1  # 起点
        A[v, idx] = 1   # 终点
        c[idx] = adj_matrix[u, v]
    
    return A, c


def create_commodities(adj_matrix, K_commodities, max_demand=10.0, seed=None):
    """
    生成随机的商品数据，确保每个(s,t)对之间存在有向路径。

    参数:
        adj_matrix (np.ndarray): N×N 有向带权邻接矩阵
        K_commodities (int): 要生成的商品数量
        max_demand (float): 单个商品的最大需求量
        seed (int, optional): 随机数种子

    返回:
        list: 商品列表，每个元素是元组 (s_k, t_k, d_k)
    """
    rng = np.random.default_rng(seed)
    N = adj_matrix.shape[0]
    
    if K_commodities > N:
        raise ValueError(f"商品数量 {K_commodities} 不能超过节点数量 {N}")
    
    # 1. 随机选择K个不重复的源节点
    sources = rng.choice(N, K_commodities, replace=False)
    
    commodities = []
    
    for s in sources:
        # 2. 使用BFS找到从s可达的所有节点
        reachable = bfs_reachable(adj_matrix, s)
        
        # 排除源节点本身
        reachable = [node for node in reachable if node != s]
        
        if len(reachable) == 0:
            # 如果没有可达节点，跳过此商品
            print(f"警告: 节点 {s} 没有可达的其他节点，跳过此商品")
            continue
        
        # 3. 从可达节点中随机选择一个作为汇节点
        t = rng.choice(reachable)
        
        # 生成随机需求量
        d = rng.uniform(1.0, max_demand)
        
        commodities.append((int(s), int(t), d))
    
    return commodities


def bfs_reachable(adj_matrix, source):
    """
    使用BFS找到从源节点可达的所有节点。
    
    参数:
        adj_matrix (np.ndarray): N×N 有向带权邻接矩阵
        source (int): 源节点索引
    
    返回:
        list: 从源节点可达的所有节点列表（包括源节点自己）
    """
    N = adj_matrix.shape[0]
    visited = np.zeros(N, dtype=bool)
    queue = deque([source])
    visited[source] = True
    reachable = [source]
    
    while queue:
        current = queue.popleft()
        
        # 找到当前节点的所有邻居（出边）
        # 条件：边权重有限且大于0
        neighbors = np.where((adj_matrix[current] < np.inf) & (adj_matrix[current] > 0))[0]
        
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                reachable.append(int(neighbor))
    
    return reachable

def generate_capacity_constraints(A, commodities, 
                                  capacity_factor_min=1.0, 
                                  capacity_factor_max=3.0,
                                  strategy='uniform'):
    """
    为MCNF问题生成边容量约束
    
    参数:
        A (np.ndarray): 关联矩阵 (N, M)
        commodities (list): 商品列表 [(s_k, t_k, d_k), ...]
        capacity_factor_min (float): 容量下界系数
        capacity_factor_max (float): 容量上界系数
        strategy (str): 容量生成策略
            - 'uniform': 基于平均需求的均匀分布
            - 'scaled_uniform': 基于最大需求的均匀分布
            - 'demand_proportional': 与总需求成比例
    
    返回:
        np.ndarray: 边容量向量，长度M
    """
    N, M = A.shape
    K = len(commodities)
    
    # 提取所有商品的需求量
    demands = np.array([d for _, _, d in commodities])
    
    # 计算需求的统计量
    total_demand = demands.sum()
    avg_demand = demands.mean()
    max_demand = demands.max()
    
    if strategy == 'uniform':
        # 策略1: 每条边的容量在 [avg_demand * factor_min, avg_demand * factor_max] 范围内均匀分布
        # 理念：每条边应能容纳"若干个"平均商品需求
        capacity_min = capacity_factor_min * avg_demand
        capacity_max = capacity_factor_max * avg_demand
        capacity = np.random.uniform(capacity_min, capacity_max, size=M)
        
    elif strategy == 'scaled_uniform':
        # 策略2: 基于最大需求的均匀分布
        # 理念：确保即使最大的单个商品也能通过（但不是所有边都能轻松通过）
        capacity_min = capacity_factor_min * max_demand
        capacity_max = capacity_factor_max * max_demand
        capacity = np.random.uniform(capacity_min, capacity_max, size=M)
        
    elif strategy == 'demand_proportional':
        # 策略3: 与总需求成比例，但在一定范围内波动
        # 理念：总容量应该足够但不过分充裕
        base_capacity = total_demand / M  # 平均每条边分配的容量
        capacity = np.random.uniform(
            capacity_factor_min * base_capacity,
            capacity_factor_max * base_capacity,
            size=M
        )
    
    else:
        raise ValueError(f"未知的容量生成策略: {strategy}")
    
    return capacity
