# import numpy as np
import torch

from utils import create_base_network, \
    ensure_weak_connectivity, \
    adjacency_to_incidence, \
    create_commodities
# from core import apsp_gpu
from core import apsp_gpu_adaptive

N = 10000
k = 10
seed = 1
W = create_base_network(N, k, seed)
W = ensure_weak_connectivity(W, seed)

# A, c = adjacency_to_incidence(W)

# K = 3
# commodities = create_commodities(W, K, 10.0, seed)

# 转换为 PyTorch Tensor
W_tensor = torch.from_numpy(W.T).float()

print("="*60)
print("开始 GPU APSP 计算")
print("="*60)

# # 执行 GPU APSP（装饰器会自动打印性能信息）
# D, P, iterations = apsp_gpu(
#     W_tensor, 
#     device='cuda:0',
#     convergence_check=True
# )

# # 打印最终结果
# print("\n--- 最终最短距离矩阵 D ---")
# print(D.cpu().numpy())

# print("\n--- 最终前驱节点矩阵 P ---")
# print(P.cpu().numpy())

D_sparse, P_sparse, iter_sparse = apsp_gpu_adaptive(
    W_tensor, 
    device='cuda:0',
    sparsity_threshold=0.3,
    convergence_check=True
)


# 打印最终结果
print("\n--- 最终最短距离矩阵 D ---")
print(D_sparse.cpu().numpy())

print("\n--- 最终前驱节点矩阵 P ---")
print(P_sparse.cpu().numpy())
