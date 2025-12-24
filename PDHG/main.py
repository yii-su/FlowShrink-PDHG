import torch
from pdhg_one_dual import MCNFPDHG
import warnings
# Suppress only DeprecationWarning
warnings.filterwarnings("ignore", category=Warning)
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import FlowShrink.solver as solver
import FlowShrink.utils as utils
import FlowShrink.decorators as decorators
import numpy as np
from torch.profiler import profile,ProfilerActivity,record_function

# 原作者固定商品数量K==N，无法灵活设定商品数,决策变量数量（维度）为NM，且认为dest相同的流量都是同一种商品的，这可以argue：相同dest可以承接不同商品的流量
# 现实问题：不同的商品可以运到相同的dest，也可以从相同的src出发运输
# 这就像一家工厂生产不同产品，客户从这个src订购多种产品，发货到相同dest

def calculate_objective(x,X,w,d,p):
    obj=w@torch.square(X-d)+p@torch.sum(x.reshape(M,K),axis=1)
    return obj

N = 200
k = 10
K = 200
seed = 1

A_adj=utils.create_base_network(N,k,seed)
A_adj=utils.ensure_weak_connectivity(A_adj,seed)
A,c=utils.adjacency_to_incidence(A_adj)
commodities=utils.create_commodities(A_adj,K,seed=seed)
capacity=utils.generate_capacity_constraints(A, commodities, 1.0, 5.0,seed=seed)
W=utils.generate_weight(K,'vector',seed)
# print(f'incidence matrix:\n{A}')
# print(f'costs:\n{c}')
# print(f'commodities:\n{commodities}')
# print(f'capacity:\n{capacity}')
# print(f'weight:\n{W}')

model = MCNFPDHG(torch.float64)
_,M=model.create_data(N,k,K,device='cuda:0',seed=seed)
print(f'vertices:{N}, neighbors:{k}, arcs(edges):{M}, commodities:{K}, seed:{seed}')
#print(f'src:{model.edges_src} | dst:{model.edges_dst}')
x0,X0 = model.make_initials()
# x0 = torch.from_numpy(result['x'].astype(np.float32)).reshape(K*M)
# activities = [ProfilerActivity.CUDA]
# sort_by_keyword = 'cuda' + "_time_total"
# with profile(activities=activities, record_shapes=True) as prof:
#     with record_function("pdhg_solve"):
#x,X,Y=model.pdhg_solve(x0,X0,tol=1e-2)
#print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=50))
#x,X,Y=model.pdhg_solve(x0,X0)
W_scale=torch.tensor(300.0,device='cuda:0')
#result=solver.solve_mcnf_cvxpy_cost(A,c,commodities,capacity,W=W,solver='scs',w_scale=300.0)# numpy format result
#result_gurobi=solver.solve_mcnf_gurobi_cost(A,c,commodities,capacity,W=W,w_scale=300.0)
# print('求解器CVXPY结果')
# print(f"求解状态: {result['status']}")
# print(f"使用求解器: {result['solver']}")
# x_cvxpy=torch.tensor(result['x'],device='cuda:0')
# X_cvxpy=torch.tensor(result['X'],device='cuda:0')
# print(f"PDHG objective: {calculate_objective(x,X,model.W,model.d,model.p)}")
# print(f"CVXPY objective: {calculate_objective(x_cvxpy,X_cvxpy,model.W,model.d,model.p)}")
# print('-----')
# print(f'PDHG x:\n{x.reshape(M,K)}')
# print(f"CVXPY x:\n {x_cvxpy}")
# print('-----')
# print(f'PDHG X:\n{X}')
# print(f"CVXPY X:\n {X_cvxpy}")
# print(f'demand:\n{model.d}')

result_gurobi_gpu=solver.solve_mcnf_gurobi_cost_gpu(A,c,commodities,capacity,W=W,w_scale=300.0)
print('求解器GUROBI_GPU结果')
print(f"求解状态: {result_gurobi_gpu['status']}")
print(f"使用求解器: {result_gurobi_gpu['solver']}")
print(f"求解器内置求解时间: {result_gurobi_gpu['solve_time']}")
x_gurobi=torch.tensor(result_gurobi_gpu['x'],device='cuda:0')
X_gurobi=torch.tensor(result_gurobi_gpu['X'],device='cuda:0')
#print(f"PDHG objective: {calculate_objective(x,X,model.W,model.d,model.p)}")
print(f"GUROBI objective: {calculate_objective(x_gurobi,X_gurobi,model.W,model.d,model.p)}")
print('-----')
#print(f'PDHG x:\n{x.reshape(M,K)}')
print(f"GUROBI x:\n {x_gurobi}")
print('-----')
#print(f'PDHG X:\n{X}')
print(f"GUROBI X:\n {X_gurobi}")
print(f'demand:\n{model.d}')
    
    
#不考虑流量约束的松弛解，一种商品一定走单路径，不会有分叉
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
# W_tensor = torch.from_numpy(W.T).float()

# print("="*60)
# print("开始 GPU APSP 计算")
# print("="*60)

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

# print("cpu")
# D_sparse, P_sparse, iter_sparse = apsp_gpu_adaptive(
#     W_tensor, 
#     device='cpu',
#     sparsity_threshold=0.3,
#     convergence_check=True
# )

# print("cuda")
# D_sparse, P_sparse, iter_sparse = apsp_gpu_adaptive(
#     W_tensor, 
#     device='cuda:0',
#     sparsity_threshold=0.3,
#     convergence_check=True
# )

# # 打印最终结果
# print("\n--- 最终最短距离矩阵 D ---")
# print(D_sparse.cpu().numpy())

# print("\n--- 最终前驱节点矩阵 P ---")
# print(P_sparse.cpu().numpy())
