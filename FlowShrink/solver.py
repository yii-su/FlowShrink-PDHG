import cvxpy as cp
import numpy as np
from scipy import sparse
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from FlowShrink.decorators import cpu_timer

@cpu_timer
def solve_mcnf_cvxpy(A, c, commodities, capacity=None, solver='SCS',W=None):
    """
    使用CVXPY求解多商品网络流问题（支持开源求解器）
    
    参数:
        A (np.ndarray): 关联矩阵 (N, M)
        c (np.ndarray): 边的成本向量，长度M，暂定所有商品的成本向量相同
        commodities (list): 商品列表 [(s_k, t_k, d_k), ...]
        capacity (np.ndarray, optional): 边的容量上限
        solver (str): 求解器选择 'CLARABEL', 'SCS', 'ECOS', 'OSQP'
        W: 权重对角矩阵,对角线元素均为正数，且和为1
    返回:
        dict: 包含目标值、流量等求解结果
    """
    N, M = A.shape
    K = len(commodities)
    weight=[W[k][k] for k in range(K)]
    
    # 决策变量: f[k, e] 表示商品k在边e上的流量
    f = cp.Variable((K, M), nonneg=True)
    
    # 目标函数: minimize sum_{k,e} c[e] * f[k, e]+sum_{k}^{K} w[k][X_k-d_k]
    d=[c[2] for c in commodities]
    #单商品的运输量cp.sum(f,axis=1)
    #cp.sum(cp.sum(f,axis=1),-d)
    #cp.power(cp.sum(cp.sum(f,axis=1),-d),2)
    objective=cp.Minimize(cp.multiply(cp.power(cp.sum(cp.sum(f,axis=1),-d),2),weight),cp.sum(cp.multiply(c, cp.sum(f, axis=0))))
    #objective = cp.Minimize(cp.sum(cp.multiply(c, cp.sum(f, axis=0))))
    
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
    
@cpu_timer
def solve_mcnf_cvxpy_cost(A, p, commodities, capacity=None, solver='SCS', W=None, verbose=False, mode='penalty', w_scale=1.0):
    """
    使用 CVXPY 求解多商品网络流问题（baseline）
    
    参数:
        A (np.ndarray): 关联矩阵，shape (N, M)。假定为 node-arc incidence:
                        A[v,e]=+1 if edge e enters v, -1 if edge e leaves v.
        p (np.ndarray): 每条边的线性运输成本向量，长度 M (p_e).
        commodities (list): 商品列表 [(s_k, t_k, d_k), ...]，节点索引使用 0-based。
        capacity (np.ndarray, optional): 边的容量上限，长度 M。如果为 None，视为 +inf。
        solver (str): 求解器名称字符串（建议 'SCS','ECOS','OSQP' 等）。默认 'CLARABEL'（若不可用会回退）。
        W (array-like / torch tensor, optional): 权重向量或对角矩阵（或 torch tensor）。
                                                若为对角矩阵，函数会取其对角线为权重。
                                                若 None，则使用均等权重并归一化（和为1）。
        verbose (bool): 是否在求解时打印 CVXPY 日志。
        mode: 'penalty' | 'hard_X' | 'supply_demand'
        'penalty'      : objective has sum_k w_k (X_k - d_k)^2 + linear cost
                       (here W is NOT normalized; you can scale via w_scale)
        'hard_X'       : add constraint X_k == d_k  (forces full delivery if feasible)
        'supply_demand': eliminate X; set A @ x_k == b_k where b_k has -d at s and +d at t
                       (classic flow with fixed demand)
        w_scale: multiply weights w by this scalar (only used in 'penalty' mode)
        
    返回:
        dict: {
            'status': problem.status,
            'objective': objective_value (float),
            'x': x_value (K, M) ndarray,
            'X': X_value (K,) ndarray,
            'used_solver': solver_used (str),
            'solve_time': solve_time (if provided by solver, else None)
        }
    """
    # ---- basic checks & shapes ----
    # convert A to scipy sparse
    A = sparse.csr_matrix(A)
    N, M = A.shape

    # vector cost
    p = np.asarray(p, float)

    K = len(commodities)
    s = np.zeros(K, int)
    t = np.zeros(K, int)
    d = np.zeros(K, float)

    for i, (si, ti, di) in enumerate(commodities):
        s[i] = si
        t[i] = ti
        d[i] = di

    # weights
    if W is None:
        w = np.ones(K) / K
    else:
        if W.ndim == 2:   # diagonal matrix
            w = np.diag(W)
        else:
            w = np.array(W, float)
    w = w * float(w_scale)

    if capacity is None:
        capacity = np.full(M, np.inf)
    else:
        capacity = np.asarray(capacity, float)

    # variables
    x = cp.Variable((K, M), nonneg=True)
    if mode != 'supply_demand':
        X = cp.Variable(K, nonneg=True)
    else:
        X = None

    constraints = []

    # ------------- flow conservation (DPP-safe version) ---------------
    # A * x_k = fk * X_k
    for k in range(K):
        fk = np.zeros(N)
        fk[t[k]] = 1
        fk[s[k]] = -1
        constraints.append(A @ x[k, :] == fk * X[k])

    # ---------------- capacity ----------------------------------------
    constraints.append(cp.sum(x, axis=0) <= capacity)

    if mode == 'penalty' or mode == 'hard_X':
        # DCP-safe flow conservation: A @ x_k == fk * X_k
        for k in range(K):
            fk = np.zeros(N, dtype=float)
            fk[t[k]] = 1.0
            fk[s[k]] = -1.0
            constraints.append(A @ x[k, :] == fk * X[k])
        # capacity
        constraints.append(cp.sum(x, axis=0) <= capacity)
        # objective
        if mode == 'penalty':
            # w is numpy constant; ensure DCP-safety
            quad = cp.sum(cp.multiply(w, cp.square(X - d)))
            linear = p @ cp.sum(x, axis=0)
            objective = cp.Minimize(quad + linear)
        else:  # 'hard_X': enforce X==d
            constraints += [X == d]
            linear = p @ cp.sum(x, axis=0)
            objective = cp.Minimize(linear)  # only minimize transport cost subject to meeting demand
    elif mode == 'supply_demand':
        # enforce exact demand using supply/demand vector -> A @ x_k == b_k
        for k in range(K):
            bk = np.zeros(N, dtype=float)
            bk[t[k]] = d[k]
            bk[s[k]] = -d[k]
            constraints.append(A @ x[k, :] == bk)
        constraints.append(cp.sum(x, axis=0) <= capacity)
        objective = cp.Minimize(p @ cp.sum(x, axis=0))  # minimize transport cost subject to fixed demands
    else:
        raise ValueError("mode must be 'penalty'|'hard_X'|'supply_demand'")

    prob = cp.Problem(objective, constraints)

    # ------------- safe solver config (prevent solver hang) ---------------
    solve_opts = dict()
    if solver.upper() == 'SCS':
        solve_opts = dict(max_iters=5000, eps=1e-4)
        solver_used = cp.SCS
    elif solver.upper() == 'ECOS':
        solver_used = cp.ECOS
    elif solver.upper() == 'OSQP':
        solver_used = cp.OSQP
        solve_opts = dict(max_iter=5000, eps_abs=1e-4, eps_rel=1e-4)
    elif solver.upper() == 'CLARABEL':
        solver_used = cp.CLARABEL
        solve_opts = dict(max_iter=5000, tol_gap_abs=1e-4)
    else:
        solver_used = cp.SCS
        solve_opts = dict(max_iters=5000, eps=1e-4)

    prob.solve(solver=solver_used, verbose=verbose, **solve_opts)

    return dict(
        mode=mode,
        status=prob.status,
        objective=prob.value,
        x=x.value,
        X=X.value,
        solver=str(solver_used),
        solver_stats=prob.solver_stats
    )
