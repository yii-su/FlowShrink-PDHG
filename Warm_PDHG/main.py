import os
import sys
import warnings
import torch
import numpy as np
from torch.profiler import profile, ProfilerActivity, record_function
import argparse

# --- Local Modules Setup ---
# 确保能够引用上级目录模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pdhg_one_dual_warm_start import MCNFPDHGWARMSTART
import solver
import utils
import decorators as decorators

# --- Global Settings ---
warnings.filterwarnings("ignore", category=Warning)


def calculate_objective(x, X, w, d, p, num_edges, num_commodities):
    """
    计算目标函数值
    """
    # 确保 reshape 维度正确，避免依赖全局变量 M
    term1 = w @ torch.square(X - d)
    # x 通常是打平的，需要 reshape 为 (Edges, Commodities)
    term2 = p @ torch.sum(x.reshape(num_edges, num_commodities), dim=1)
    obj = term1 + term2
    return obj


def generate_graph_data(N, k, K, seed):
    """
    生成图结构和基础数据 (主要用于 Gurobi Solver 和数据校验)
    """
    # 创建基础网络拓扑
    A_adj = utils.create_base_network(N, k, seed)
    A_adj = utils.ensure_weak_connectivity(A_adj, seed)
    
    # 转换为关联矩阵
    A, c = utils.adjacency_to_incidence(A_adj)
    
    # 生成商品和容量约束
    commodities = utils.create_commodities(A_adj, K, seed=seed)
    capacity = utils.generate_capacity_constraints(A, commodities, 1.0, 5.0, seed=seed)
    
    # 生成权重
    W_vec = utils.generate_weight(K, "vector", seed)
    
    return A, c, commodities, capacity, W_vec


def run_pdhg_routine(N, k, K, seed, warm_start, device="cuda:0", pdhg=True):
    """
    执行 PDHG 算法求解流程
    """
    print("-" * 30)
    print(f"开始 PDHG 求解 (Warm Start: {warm_start})")
    
    # 初始化模型
    model = MCNFPDHGWARMSTART(torch.float32)
    
    # 模型内部生成数据 (注意：这里假设 model.create_data 使用相同的 seed 生成相同的数据)
    _, M = model.create_data(N, k, K, seed=seed)
    
    print(f"问题规模 -> 节点(N): {N}, 邻居(k): {k}, 边(M): {M}, 商品(K): {K}")
    
    # 生成初始点
    x0, X0, Y0 = model.make_initials(warm_start=warm_start)

    # # 检查初始目标值
    init_obj = calculate_objective(x0, X0, model.W, model.d, model.p, M, K)
    print(f"初始目标值 (Initial Objective): {init_obj:.4f}")
    
    if not pdhg:
        print("PDHG 求解被跳过，仅返回初始目标值。")
        return init_obj

    # 求解 (此处保留了后续可开启编译优化的位置)
    # model.pdhg_step_fn = torch.compile(model.pdhg_step_fn, mode="max-autotune")
    x_pdhg, X_pdhg, Y = model.pdhg_solve(x0, X0, Y0)
    # x_pdhg, X_pdhg, Y = model.pdhg_solve_cuda_graph(x0, X0, Y0)

    # 计算最终目标值
    final_obj = calculate_objective(x_pdhg, X_pdhg, model.W, model.d, model.p, M, K)
    print(f"PDHG 最终目标值: {final_obj:.4f}")
    
    return final_obj


def run_gurobi_routine(A, c, commodities, capacity, W, scale, device="cuda:0"):
    """
    执行 Gurobi 求解器流程
    """
    print()
    print("-" * 30)
    print("开始 Gurobi 求解")
    
    # 调用求解器
    # W_scale = torch.tensor(scale, device=device) # 如果solver内部需要tensor可取消注释
    result = solver.solve_mcnf_gurobi_cost(
        A, c, commodities, capacity, W=W, w_scale=scale
    )
    
    '''
        return dict(
        mode=mode,
        status=status_str,
        objective=obj_val,
        x=x_val,
        X=X_val,
        solver='GUROBI',
        solver_stats={
            'IterCount': model.IterCount,
            'Runtime': model.Runtime,
            'ObjVal': obj_val
        },
        solve_time=model.Runtime
    )
    '''
    
    print(f"求解状态: {result['status']}")
    print(f"最优值: {result['objective']:.4f}")
    print(f"求解耗时(ms): {result['solve_time']*1000:.0f}") # type: ignore
    
    return result

if __name__ == "__main__":
    # ------------------------------------------
    # 1. 参数配置 (通过命令行接收)
    # ------------------------------------------
    parser = argparse.ArgumentParser()
    
    # 定义命令行参数
    parser.add_argument('--N', type=int, default=100, help='节点数')
    parser.add_argument('--k', type=int, default=10, help='邻居数')
    parser.add_argument('--K', type=int, default=100, help='商品数')
    parser.add_argument('--warm_start', type=int, default=1, help='1开启, 0关闭')
    parser.add_argument('--pdhg', type=int, default=1, help='1开启, 0关闭')
    
    # 这里的 seed, scale, device 如果不常改，可以保留默认值，也可以继续加参数
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--scale', type=float, default=300.0)
    parser.add_argument('--device', type=str, default="cuda:0")

    args = parser.parse_args()

    # 将参数赋值给变量
    N = args.N
    k = args.k
    K = args.K
    seed = args.seed
    scale = args.scale
    device = args.device
    # 将 int (0/1) 转换为 bool
    warm_start = bool(args.warm_start)
    pdhg = bool(args.pdhg)

    print(f"配置确认: N={N}, k={k}, K={K}, warm_start={warm_start}, pdhg={pdhg}")

    # ------------------------------------------
    # 2. PDHG 求解
    # ------------------------------------------
    pdhg_obj = run_pdhg_routine(N, k, K, seed, warm_start, device, pdhg)

    # # ------------------------------------------
    # # 3. Gurobi 求解
    # # ------------------------------------------
    # A, c, commodities, capacity, W = generate_graph_data(N, k, K, seed)
    # gurobi_result = run_gurobi_routine(A, c, commodities, capacity, W, scale, device)
    
    print("-" * 30)
    print("所有求解完成。")