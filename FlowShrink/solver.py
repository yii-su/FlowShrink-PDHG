import cvxpy as cp
import numpy as np
from scipy import sparse
import gurobipy as gp
from gurobipy import GRB
import os
import sys
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
def solve_mcnf_cvxpy_cost(A, p, commodities, capacity=None, solver='SCS', W=None, verbose=True, mode='penalty', w_scale=1.0):
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
    x = cp.Variable((M, K), nonneg=True)
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
        constraints.append(A @ x[:, k] == fk * X[k])

    # ---------------- capacity ----------------------------------------
    constraints.append(cp.sum(x, axis=1) <= capacity)

    if mode == 'penalty' or mode == 'hard_X':
        # DCP-safe flow conservation: A @ x_k == fk * X_k
        for k in range(K):
            fk = np.zeros(N, dtype=float)
            fk[t[k]] = 1.0
            fk[s[k]] = -1.0
            constraints.append(A @ x[:, k] == fk * X[k])
        # capacity
        constraints.append(cp.sum(x, axis=1) <= capacity)
        # objective
        if mode == 'penalty':
            # w is numpy constant; ensure DCP-safety
            quad = cp.sum(cp.multiply(w, cp.square(X - d)))
            linear = p @ cp.sum(x, axis=1)
            objective = cp.Minimize(quad + linear)
        else:  # 'hard_X': enforce X==d
            constraints += [X == d]
            linear = p @ cp.sum(x, axis=1)
            objective = cp.Minimize(linear)  # only minimize transport cost subject to meeting demand
    elif mode == 'supply_demand':
        # enforce exact demand using supply/demand vector -> A @ x_k == b_k
        for k in range(K):
            bk = np.zeros(N, dtype=float)
            bk[t[k]] = d[k]
            bk[s[k]] = -d[k]
            constraints.append(A @ x[:, k] == bk)
        constraints.append(cp.sum(x, axis=1) <= capacity)
        objective = cp.Minimize(p @ cp.sum(x, axis=1))  # minimize transport cost subject to fixed demands
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
   
@cpu_timer 
def solve_mcnf_gurobi_cost(A, p, commodities, capacity=None, solver='GUROBI', W=None, verbose=True, mode='penalty', w_scale=1.0):
    """
    使用 Gurobi (gurobipy) 求解多商品网络流问题 (New Baseline)
    
    参数与原 CVXPY 版本保持一致。
    """
    # ---- 1. 数据预处理 ----
    # 转换为 CSR 稀疏矩阵，Gurobi 矩阵 API 对此支持最好
    A = sparse.csr_matrix(A)
    N, M = A.shape

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
    # 建立环境 (通常学术 license 会自动加载，若报错需检查 gurobi.lic 路径)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.setParam("Threads", 0)  # 0 = 自动使用所有核心
    env.start()
    
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
            model.addConstr(A @ x[:, k] == bk, name=f"flow_bal_{k}")
        else:
            # A @ x_k == fk * X_k  =>  A @ x_k - fk * X_k == 0
            # 构造 fk 向量: t=+1, s=-1
            fk = np.zeros(N)
            fk[t[k]] = 1.0
            fk[s[k]] = -1.0
            
            # 注意: X[k] 是变量，fk 是常数向量。
            # Matrix API 中，常数向量 * 标量变量会自动广播
            model.addConstr(A @ x[:, k] - fk * X[k] == 0, name=f"flow_bal_{k}")

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
            diff = X[k] - d[k]
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
    
def solve_mcnf_gurobi_cost_gpu(A, p, commodities, capacity=None, solver='GUROBI_GPU', W=None, verbose=True, mode='penalty', w_scale=1.0):
    """
    使用 Gurobi 13.0 GPU 加速 (PDHG) 求解多商品网络流问题
    
    环境要求:
    - Gurobi 13.0+
    - Nvidia GPU (Pascal架构或更新)
    - 环境变量配置正确 (LD_LIBRARY_PATH 指向 linux64gpu 等)
    
    参数:
    (与 solve_mcnf_cvxpy_cost / solve_mcnf_gurobi_cost 保持一致)
    """
    # ---- 1. 数据预处理 (CPU端) ----
    A = sparse.csr_matrix(A)
    N, M = A.shape

    p = np.asarray(p, float)

    K = len(commodities)
    s = np.zeros(K, int)
    t = np.zeros(K, int)
    d = np.zeros(K, float)

    for i, (si, ti, di) in enumerate(commodities):
        s[i] = si
        t[i] = ti
        d[i] = di

    if W is None:
        w = np.ones(K) / K
    else:
        if hasattr(W, "ndim") and W.ndim == 2:
            w = np.diag(W)
        else:
            w = np.array(W, float)
    w = w * float(w_scale)

    if capacity is None:
        capacity = np.full(M, GRB.INFINITY)
    else:
        capacity = np.array(capacity, float)
        capacity[np.isinf(capacity)] = GRB.INFINITY

    # ---- 2. 初始化 Gurobi 环境与模型 ----
    # 创建空环境以便设置参数
    env = gp.Env(empty=True)
    
    # --- GPU / PDHG 核心参数设置 ---
    env.setParam("OutputFlag", 1 if verbose else 0)
    
    # Method=6 启用 PDHG (First-Order algorithm)
    # 注意：这是 Gurobi 13.0 针对 LP/QP 引入的一阶方法入口
    env.setParam("Method", 6)
    
    # PDHGGPU=1 启用 GPU 加速
    # 0 = CPU (多线程 AVX2/AVX512), 1 = GPU
    env.setParam("PDHGGPU", 1)
    
    # Crossover=0 关闭交叉操作
    # PDHG 产生的是非基解 (non-vertex solution)。
    # 如果开启 Crossover (默认可能是自动)，求解器会在 PDHG 结束后尝试在 CPU 上用单纯形法推到基解，
    # 这在大规模问题上极其耗时，通常作为 Baseline 对比我们不需要基解，只需数值解。
    env.setParam("Crossover", 0)
    
    # 可选：设置 PDHG 收敛容差 (默认 1e-4 或 1e-6，视版本而定)
    # 对于大规模机器学习/网络流 Baseline，1e-4 通常足够
    env.setParam("PDHGAbsTol", 1e-2) 
    #不使用relative tolerance
    env.setParam("PDHGRelTol", 0)
    env.setParam("PDHGConvTol", 1e-2)

    env.start()
    
    model = gp.Model("MCNF_Gurobi_GPU", env=env)

    # ---- 3. 定义变量 (Matrix API) ----
    # Gurobipy 会自动处理 Numpy 到 C++ 内部结构的传输
    # Gurobi 13.0 会在求解开始时将模型矩阵传输到 GPU 显存
    x = model.addMVar((M, K), lb=0.0, name="x")
    
    if mode != 'supply_demand':
        X = model.addMVar(K, lb=0.0, name="X")
    else:
        X = None

    # ---- 4. 添加约束 ----
    # 4.1 容量约束
    model.addConstr(x.sum(axis=1) <= capacity, name="capacity")

    # 4.2 流量守恒
    for k in range(K):
        if mode == 'supply_demand':
            bk = np.zeros(N)
            bk[t[k]] = d[k]
            bk[s[k]] = -d[k]
            model.addConstr(A @ x[:, k] == bk, name=f"flow_bal_{k}")
        else:
            fk = np.zeros(N)
            fk[t[k]] = 1.0
            fk[s[k]] = -1.0
            model.addConstr(A @ x[:, k] - fk * X[k] == 0, name=f"flow_bal_{k}")

    # 4.3 Hard X 约束
    if mode == 'hard_X':
        model.addConstr(X == d, name="hard_demand")

    # ---- 5. 设置目标函数 ----
    linear_cost = p @ x.sum(axis=1)

    if mode == 'penalty':
        quad_obj = 0
        for k in range(K):
            diff = X[k] - d[k]
            quad_obj += w[k] * (diff * diff)
        # PDHG 原生支持 QP，且在 GPU 上处理二次项效率极高
        model.setObjective(quad_obj + linear_cost, GRB.MINIMIZE)
        
    elif mode in ['hard_X', 'supply_demand']:
        model.setObjective(linear_cost, GRB.MINIMIZE)
    else:
        raise ValueError("mode error")

    # ---- 6. 求解 ----
    # 这一步会触发数据向 GPU 的拷贝以及 CUDA Kernel 的执行
    try:
        model.optimize()
    except gp.GurobiError as e:
        if "GPU" in str(e):
            print(f"Gurobi GPU Error: {e}. Check CUDA drivers or License.")
            return {'status': 'gpu_error', 'objective': float('inf')}
        raise e

    # ---- 7. 结果提取 ----
    status_map = {
        GRB.OPTIMAL: 'optimal',
        GRB.INFEASIBLE: 'infeasible',
        GRB.INF_OR_UNBD: 'infeasible_or_unbounded',
        GRB.UNBOUNDED: 'unbounded',
        GRB.ITERATION_LIMIT: 'optimal_inaccurate',
        GRB.TIME_LIMIT: 'optimal_inaccurate'
    }
    status_str = status_map.get(model.Status, 'unknown')

    # 注意：PDHG 求解器返回的状态通常是 OPTIMAL，但本质上是满足 Tol 范围内的解
    if model.Status == GRB.OPTIMAL or (model.SolCount > 0):
        # 将结果从 Gurobi 对象取回 CPU Numpy 数组
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
        solver='GUROBI_GPU_PDHG',
        solver_stats={
            'IterCount': model.IterCount,
            'Runtime': model.Runtime,
            'ObjVal': obj_val,
            # PDHG 特有的统计信息（如果存在）
            'BarIterCount': model.BarIterCount 
        },
        solve_time=model.Runtime
    )
