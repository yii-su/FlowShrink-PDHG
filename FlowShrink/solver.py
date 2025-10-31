import cvxpy as cp
import numpy as np
from decorators import cpu_timer

@cpu_timer
def solve_mcnf_cvxpy(A, c, commodities, capacity=None, solver='CLARABEL'):
    """
    使用CVXPY求解多商品网络流问题（支持开源求解器）
    
    参数:
        A (np.ndarray): 关联矩阵 (N, M)
        c (np.ndarray): 边的成本向量，长度M
        commodities (list): 商品列表 [(s_k, t_k, d_k), ...]
        capacity (np.ndarray, optional): 边的容量上限
        solver (str): 求解器选择 'CLARABEL', 'SCS', 'ECOS', 'OSQP'
    
    返回:
        dict: 包含目标值、流量等求解结果
    """
    N, M = A.shape
    K = len(commodities)
    
    # 决策变量: f[k, e] 表示商品k在边e上的流量
    f = cp.Variable((K, M), nonneg=True)
    
    # 目标函数: minimize sum_{k,e} c[e] * f[k, e]
    objective = cp.Minimize(cp.sum(cp.multiply(c, cp.sum(f, axis=0))))
    
    constraints = []
    
    # 约束1: 流量守恒
    for k, (s, t, d) in enumerate(commodities):
        b_k = np.zeros(N)
        b_k[s] = -d
        b_k[t] = d
        
        # A @ f[k, :] == b_k
        constraints.append(A @ f[k, :] == b_k)
    
    # 约束2: 容量约束
    if capacity is not None:
        constraints.append(cp.sum(f, axis=0) <= capacity)
    
    # 构建并求解问题
    problem = cp.Problem(objective, constraints)
    
    # 求解（尝试多个求解器）
    solvers_to_try = [solver, 'CLARABEL', 'SCS', 'ECOS', 'OSQP']
    
    for slv in solvers_to_try:
        try:
            problem.solve(solver=slv, verbose=False)
            if problem.status in ['optimal', 'optimal_inaccurate']:
                break
        except:
            continue
    
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        raise RuntimeError(f"求解失败，状态: {problem.status}")
    
    # 提取结果
    flow_solution = f.value  # shape: (K, M)
    total_flow_per_edge = flow_solution.sum(axis=0)  # type: ignore
    
    return {
        'status': problem.status,
        'objective': problem.value,
        'flow': flow_solution,
        'total_flow_per_edge': total_flow_per_edge,
        'capacity': capacity,
        'capacity_utilization': total_flow_per_edge / capacity if capacity is not None else None,
        'solver_used': problem.solver_stats.solver_name
    }
