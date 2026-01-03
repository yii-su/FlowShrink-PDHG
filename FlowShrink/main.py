# import numpy as np
import torch

from utils import create_base_network, \
    ensure_weak_connectivity, \
    adjacency_to_incidence, \
    create_commodities, \
    generate_capacity_constraints
from solver import solve_mcnf_cvxpy
# from core import apsp_gpu
from core import apsp_gpu_adaptive

N = 5000
k = 10
seed = 1
W = create_base_network(N, k, seed)
W = ensure_weak_connectivity(W, seed)

A, c = adjacency_to_incidence(W)
# print("关联矩阵 A ")
# print(A)
# print("成本向量 c ")
# print(c)

K = 50
K = N // 5
commodities = create_commodities(W, K, 10.0, seed)

# capacity1 = generate_capacity_constraints(
#     A, commodities,
#     capacity_factor_min=1.0,
#     capacity_factor_max=3.0,
#     strategy='uniform'
# )

# result = solve_mcnf_cvxpy(A, c, commodities, capacity1)

# print(f"求解状态: {result['status']}")
# print(f"最优目标值: {result['objective']:.4f}")
# print(f"流量矩阵维度: {result['flow'].shape}")
# print(f"容量利用率范围: [{(result['total_flow_per_edge']/capacity1).min():.2%}, "
#         f"{(result['total_flow_per_edge']/capacity1).max():.2%}]")

# capacity2 = None  # 测试无容量约束的情况

# result = solve_mcnf_cvxpy(A, c, commodities, capacity2)
# print("松弛容量约束下的结果:")
# print(f"求解状态: {result['status']}")
# print(f"最优目标值: {result['objective']:.4f}")
# print(f"流量矩阵维度: {result['flow'].shape}")


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

print("cpu")
D_sparse, P_sparse, iter_sparse = apsp_gpu_adaptive(
    W_tensor, 
    device='cpu',
    sparsity_threshold=0.3,
    convergence_check=True
)

print("cuda")
D_sparse, P_sparse, iter_sparse = apsp_gpu_adaptive(
    W_tensor, 
    device='cuda:0',
    sparsity_threshold=0.3,
    convergence_check=True
)

# # 打印最终结果
# print("\n--- 最终最短距离矩阵 D ---")
# print(D_sparse.cpu().numpy())

# print("\n--- 最终前驱节点矩阵 P ---")
# print(P_sparse.cpu().numpy())
