import numpy as np

from utils import create_data, incidence_to_adjacency, incidence_to_weighted_adjacency

if __name__ == "__main__":
    # 假设 create_data 函数已经定义

    # 1. 生成网络数据
    N_nodes = 5
    k_neighbors = 2
    A_matrix, c_costs = create_data(N_nodes, k_neighbors)
    print(f"生成的关联矩阵 A 的形状: {A_matrix.shape}")
    print("关联矩阵 A (部分):")
    print(A_matrix[:, :10])  # 只打印前10条边
    print("对应的边成本 c (部分):", c_costs[:10])
    print("-" * 30)

    # 2. (可选) 生成二元邻接矩阵
    adj_binary = incidence_to_adjacency(A_matrix)
    print("生成的二元邻接矩阵 (部分):")
    print(adj_binary[:5, :5])
    print("-" * 30)

    # 3. (推荐) 生成带权重的邻接矩阵，为最短路计算做准备
    # 这个 adj_weighted 就是你进行最短路计算所需要的输入
    adj_weighted = incidence_to_weighted_adjacency(A_matrix, c_costs)
    print(f"生成的带权重邻接矩阵，形状: {adj_weighted.shape}")
    print("矩阵中非无穷大的元素数量:", np.sum(adj_weighted != np.inf) - N_nodes)
    print("矩阵中无穷大的元素数量:", np.sum(adj_weighted == np.inf))
    print("带权重邻接矩阵 (部分，inf表示无穷大):")
    print(np.round(adj_weighted[:5, :5], 2))

    # 你现在可以直接将 adj_weighted 传递给 scipy.sparse.csgraph.dijkstra 或 floyd_warshall
