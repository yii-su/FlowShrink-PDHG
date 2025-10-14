import numpy as np 
# import cvxpy as cp 
# import torch
# import time
# from torch_scatter import scatter, scatter_add
# import argparse


def create_data(N, k):
    """
    创建网络数据，生成节点和连接关系
    
    参数:
        N: 节点数量
        k: 每个节点的邻居数量
    
    返回:
        A: 关联矩阵 (N×edges)，表示节点与边的连接关系
        c: 边的成本向量
    """
    
    # 生成N个随机节点的2D坐标
    node_list = np.array([np.random.rand(N),
                          np.random.rand(N)]).reshape((N,2))
    
    link_list = []
    
    # 为每个节点找到k个最近邻居并创建连接
    for i in range(N):
        # 计算节点i到所有其他节点的欧几里得距离
        distance = np.array([np.linalg.norm(node_list[i]-
                                            node_list[j]) for j in range(N)])
        
        # 找到距离最近的k个邻居节点（排除自己）
        neighbors = np.argsort(distance)[1:(k+1)]
        
        # 创建从节点i出发的边表示：i为起点(-1)，邻居为终点(+1)
        link = np.zeros((N,k))
        link[i,:] = -1  # 起点标记为-1
        link[neighbors,np.array(range(k))] = 1  # 终点标记为+1
        
        # 创建到节点i的边表示：i为终点(+1)，邻居为起点(-1)  
        link2 = np.zeros((N,k))
        link2[i,:] = 1  # 终点标记为+1
        link2[neighbors,np.array(range(k))] = -1  # 起点标记为-1
        
        link_list.append(link)
        link_list.append(link2)
    
    # 合并所有边并去除重复的边
    A = np.hstack(link_list)
    A = np.unique(A, axis=1)  # 去重，避免重复边
    
    # 随机打乱边的顺序
    p = np.random.permutation(A.shape[1])
    
    # 生成随机边成本，服从对数均匀分布 [0.5, 5]
    c = np.exp(np.random.rand(A.shape[1])*(np.log(5)-np.log(0.5))+np.log(0.5))
    
    return A[:,p], c


def incidence_to_adjacency(A):
    """
    高效地将图的关联矩阵转换为邻接矩阵。
    此函数利用NumPy的向量化操作，避免了显式循环，因此性能很高。
    
    参数:
        A (np.ndarray): 关联矩阵，形状为 (N, M)，其中 N 是节点数，M 是边数。
                        每一列代表一条边，-1 表示起点，+1 表示终点。
    
    返回:
        np.ndarray: 二元邻接矩阵，形状为 (N, N)。
                    如果存在从节点 i 到节点 j 的边，则 adj[i, j] = 1，否则为 0。
    """
    # 节点数量
    N = A.shape[0]
    
    # 对于A的每一列（每一条边），找到-1所在的行索引，这就是边的起点
    # np.argmin(A, axis=0) 会返回一个长度为 M 的数组，包含每条边的起点节点索引
    source_nodes = np.argmin(A, axis=0)
    
    # 同理，找到+1所在的行索引，这就是边的终点
    # np.argmax(A, axis=0) 会返回一个长度为 M 的数组，包含每条边的终点节点索引
    dest_nodes = np.argmax(A, axis=0)
    
    # 初始化一个 N x N 的零矩阵
    adj_matrix = np.zeros((N, N), dtype=np.int8)
    
    # 使用NumPy的高级索引，一次性将所有边的连接关系设置为1
    # source_nodes 作为行索引，dest_nodes 作为列索引
    adj_matrix[source_nodes, dest_nodes] = 1
    
    return adj_matrix


def incidence_to_weighted_adjacency(A, costs):
    """
    高效地将关联矩阵和成本向量转换为带权重的邻接矩阵。
    
    这对于运行最短路算法（如Dijkstra或Floyd-Warshall）是必需的。

    参数:
        A (np.ndarray): 关联矩阵，形状为 (N, M)。
        costs (np.ndarray): 边的成本（权重）向量，长度为 M。
    
    返回:
        np.ndarray: 带权重的邻接矩阵，形状为 (N, N)。
                    adj[i, j] 的值是从节点 i 到 j 的边的成本。
                    如果不存在直接的边，则值为无穷大 (inf)。
                    对角线 (i, i) 的值为 0。
    """
    N = A.shape[0]
    
    # 输入检查
    if A.shape[1] != len(costs):
        raise ValueError("关联矩阵的列数必须与成本向量的长度相等。")

    # 找到所有边的起点和终点
    source_nodes = np.argmin(A, axis=0)
    dest_nodes = np.argmax(A, axis=0)
    
    # 初始化一个 N x N 的矩阵，所有值都设为无穷大
    adj_matrix = np.full((N, N), np.inf)
    
    # 使用高级索引，将每条边的成本填充到邻接矩阵的对应位置
    adj_matrix[source_nodes, dest_nodes] = costs
    
    # 在最短路算法中，从一个节点到其自身的成本通常为0
    np.fill_diagonal(adj_matrix, 0)
    
    return adj_matrix


def create_commodities(N, K_commodities, max_demand=10.0):
    """
    生成随机的商品数据。

    参数:
        N (int): 网络中的节点总数。
        K_commodities (int): 要生成的商品数量。
        max_demand (float): 单个商品的最大需求量。

    返回:
        list: 一个商品列表，每个元素是一个元组 (s_k, t_k, d_k)，
              分别代表源节点、汇节点和需求量。
    """
    commodities = []
    for _ in range(K_commodities):
        # 随机选择两个不相同的节点作为源和汇
        s_k, t_k = np.random.choice(N, 2, replace=False)
        # 随机生成需求量
        d_k = np.random.uniform(1.0, max_demand)
        commodities.append((s_k, t_k, d_k))
    return commodities
