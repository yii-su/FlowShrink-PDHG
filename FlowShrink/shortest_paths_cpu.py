from scipy.sparse.csgraph import dijkstra

def calculate_shortest_paths(adj_weighted, commodities):
    """
    为每个商品计算从源到汇的最短路径。

    参数:
        adj_weighted (np.ndarray): 带权重的邻接矩阵 (N x N)。
        commodities (list): 商品列表 [(s_k, t_k, d_k), ...]。

    返回:
        np.ndarray: 前驱节点矩阵 (predecessors matrix)，形状为 (K, N)。
                    predecessors[k, j] 表示在从 s_k 出发的最短路径树中，
                    节点 j 的前一个节点是什么。
    """
    # 提取所有源节点
    source_nodes = [k[0] for k in commodities]
    
    # 一次性对所有需要的源节点运行Dijkstra算法
    # indices=source_nodes 指定只计算这些源点的最短路径树
    # unweighted=False 表示使用我们提供的权重
    # A -9999 in the predecessor matrix indicates that a node is unreachable.
    _dist_matrix, predecessors = dijkstra(
        csgraph=adj_weighted, 
        directed=True, 
        indices=source_nodes, 
        return_predecessors=True
    )
    
    return predecessors
