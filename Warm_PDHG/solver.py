import numpy as np
from scipy import sparse
import gurobipy as gp
from gurobipy import GRB
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

options = {
    "WLSACCESSID": "f5d6aaec-8085-4838-95ae-66aefadce145",
    "WLSSECRET": "43b2395a-2b49-461b-887d-7810349fe6fe",
    "LICENSEID": 2757273,
}

# @cpu_timer 
def solve_mcnf_gurobi_cost(A, p, commodities, capacity=None, solver='GUROBI', W=None, verbose=True, mode='penalty', w_scale=1.0):
    """
    使用 Gurobi (gurobipy) 求解多商品网络流问题 (New Baseline)
    
    参数与原 CVXPY 版本保持一致。
    """
    # ---- 1. 数据预处理 ----
    # 转换为 CSR 稀疏矩阵，Gurobi 矩阵 API 对此支持最好
    A = sparse.csr_matrix(A)
    N, M = A.shape  # type: ignore

    # 确保成本向量格式正确
    p = np.asarray(p, float)

    # 解析商品数据
    K = len(commodities)
    s = np.zeros(K, int)
    t = np.zeros(K, int)
    d = np.zeros(K, float)

    for i, (si, ti, di) in enumerate(commodities):
        s[i] = si
        t[i] = ti
        d[i] = di

    # 处理权重 W
    if W is None:
        w = np.ones(K) / K
    else:
        if hasattr(W, "ndim") and W.ndim == 2:  # 对角矩阵
            w = np.diag(W)
        else:
            w = np.array(W, float)
    w = w * float(w_scale)

    # 处理容量 (Gurobi 中如果不设上限，默认即为无穷大，但为了统一处理显式设置)
    if capacity is None:
        capacity = np.full(M, GRB.INFINITY)
    else:
        # 将 None 或 inf 替换为 Gurobi 的 INFINITY
        capacity = np.array(capacity, float)
        capacity[np.isinf(capacity)] = GRB.INFINITY

    # ---- 2. 初始化 Gurobi 模型 ----
    env = gp.Env(params=options)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.setParam("Threads", 0)  # 0 = 自动使用所有核心
    # env.start()
    
    model = gp.Model("MCNF_Gurobi", env=env)

    # ---- 3. 定义变量 (使用 Matrix API) ----
    # x: 边流量, shape (M, K)
    x = model.addMVar((M, K), lb=0.0, name="x")
    
    # X: 实际传输量, shape (K,) (仅在非 supply_demand 模式下需要)
    if mode != 'supply_demand':
        X = model.addMVar(K, lb=0.0, name="X")
    else:
        X = None

    # ---- 4. 添加约束 ----

    # 4.1 容量约束: sum(x, axis=1) <= capacity
    # Gurobi Matrix API 支持对 axis 求和
    model.addConstr(x.sum(axis=1) <= capacity, name="capacity")

    # 4.2 流量守恒约束
    for k in range(K):
        if mode == 'supply_demand':
            # A @ x_k == b_k (固定需求)
            bk = np.zeros(N)
            bk[t[k]] = d[k]
            bk[s[k]] = -d[k]
            model.addConstr(A @ x[:, k] == bk, name=f"flow_bal_{k}")  # type: ignore
        else:
            # A @ x_k == fk * X_k  =>  A @ x_k - fk * X_k == 0
            # 构造 fk 向量: t=+1, s=-1
            fk = np.zeros(N)
            fk[t[k]] = 1.0
            fk[s[k]] = -1.0
            
            # 注意: X[k] 是变量，fk 是常数向量。
            # Matrix API 中，常数向量 * 标量变量会自动广播
            model.addConstr(A @ x[:, k] - fk * X[k] == 0, name=f"flow_bal_{k}") # type: ignore

    # 4.3 额外约束 (Hard X)
    if mode == 'hard_X':
        # 强制 X == d
        model.addConstr(X == d, name="hard_demand")

    # ---- 5. 设置目标函数 ----
    
    # 线性运输成本: p^T * sum(x, axis=1)
    # 也可以写成 sum(p @ x[:, k] for k in range(K))
    linear_cost = p @ x.sum(axis=1)

    if mode == 'penalty':
        # 目标: sum(w_k * (X_k - d_k)^2) + linear_cost
        # Gurobi 处理二次项非常高效
        quad_obj = 0
        for k in range(K):
            # 注意：构建二次表达式
            diff = X[k] - d[k] # type: ignore
            quad_obj += w[k] * (diff * diff)
        
        model.setObjective(quad_obj + linear_cost, GRB.MINIMIZE)
    
    elif mode == 'hard_X' or mode == 'supply_demand':
        model.setObjective(linear_cost, GRB.MINIMIZE)
    
    else:
        raise ValueError("mode must be 'penalty'|'hard_X'|'supply_demand'")

    # ---- 6. 求解 ----
    # 对于 QP (penalty 模式)，Gurobi 默认使用 Barrier 算法，非常适合此类问题
    model.optimize()

    # ---- 7. 结果提取与格式化 ----
    
    # 映射 Gurobi 状态码到类似 CVXPY 的字符串
    status_map = {
        GRB.OPTIMAL: 'optimal',
        GRB.INFEASIBLE: 'infeasible',
        GRB.INF_OR_UNBD: 'infeasible_or_unbounded',
        GRB.UNBOUNDED: 'unbounded',
        GRB.ITERATION_LIMIT: 'optimal_inaccurate', # 或其他
        GRB.TIME_LIMIT: 'optimal_inaccurate'
    }
    status_str = status_map.get(model.Status, 'unknown')

    if model.Status == GRB.OPTIMAL or (model.SolCount > 0):
        # 提取变量值 (.X 属性)
        x_val = x.X
        obj_val = model.ObjVal
        if X is not None:
            X_val = X.X
        else:
            X_val = None
    else:
        x_val = None
        X_val = None
        obj_val = float('inf')

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
