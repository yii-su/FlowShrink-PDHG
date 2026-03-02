import os
import sys

import torch
import torch._dynamo
import numpy as np
from scipy.sparse import coo_matrix
import time

import utils
from decorators import timed_ns
import shortest_paths_gpu as ws

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
torch._dynamo.config.suppress_errors = True  # type: ignore # 避免一些无关警告


class MCNFPDHGWARMSTART:
    def __init__(self, dtype=torch.float64):
        self.device = torch.device("cuda:0")
        self.dtype = dtype
        self.weight_update_scaling=torch.tensor(0.5,device=self.device)
        self.best_rp=torch.tensor(torch.inf,device=self.device)
        self.best_rd=torch.tensor(torch.inf,device=self.device)
        

    def create_data(self, num_nodes, k, num_commodities, seed=1):
        self.N = num_nodes
        self.K = num_commodities
        device = self.device
        dtype = self.dtype
        if dtype == torch.float64:
            npdtype = np.float64
        else:
            npdtype = np.float32

        # adjacency and incidence
        W_adj = utils.create_base_network(self.N, k, seed)
        W_adj = utils.ensure_weak_connectivity(W_adj, seed)
        A_inc_np, p_np = utils.adjacency_to_incidence(W_adj)  # (N, M)
        commodities = utils.create_commodities(W_adj, self.K, 10.0, seed)
        self.W_adj = torch.tensor(W_adj, dtype=dtype, device=device)
        del W_adj

        # capacities
        c_np = utils.generate_capacity_constraints(
            A_inc_np, commodities, 1.0, 5.0, seed=seed
        )
        self.c = torch.from_numpy(c_np.astype(npdtype)).to(self.device)
        A_inc = torch.from_numpy(A_inc_np)
        del A_inc_np
        """
        torch.where(condition) 在处理二维张量时，返回的索引是按照 Row-Major（行优先） 顺序排列的，即先扫描第0行，再扫描第1行，以此类推。
        然而，你的 c（容量）、p（费用）以及变量 x 都是按照 Edge Index（列索引 0 到 M-1） 排列的。
        你的假设：edges_src[j] 对应第 j 条边（即 A_inc 的第 j 列）的源节点。
        实际情况：edges_src 只是包含了所有源节点的列表，但按照节点ID排序（受行扫描顺序影响），完全打乱了与边索引 0...M-1 的对应关系。 
        实际变成了沿着dim=1寻找 
        后果：
        PDHG 求解器实际上是在一个 乱连线的图 上进行优化。边的起点和终点被重新洗牌了，但边的容量和费用却保持原序。     
        """
        # self.edges_src=torch.where(A_inc==-1)[0].to(device)# M
        # self.edges_dst=torch.where(A_inc==1)[0].to(device)# M

        # argmin 找到每列最小值的索引（即 -1 所在的行索引）
        self.edges_src = torch.argmin(A_inc, dim=0).to(device)
        # argmax 找到每列最大值的索引（即 1 所在的行索引）
        self.edges_dst = torch.argmax(A_inc, dim=0).to(device)
        self.M = A_inc.shape[1]
        del A_inc

        # edge cost
        p = torch.from_numpy(p_np.astype(npdtype)).to(self.device)
        del p_np

        # W, d
        W_scale = 300.0
        # self.W已经乘了W_scale
        self.W = (
            torch.from_numpy(
                utils.generate_weight(self.K, dimtype="vector", seed=seed) * W_scale
            )
            .to(self.device)
            .to(dtype)
        )
        commodity_src = [c[0] for c in commodities]
        commodity_dst = [c[1] for c in commodities]
        demands = [c[2] for c in commodities]
        self.d = torch.tensor(demands, dtype=dtype, device=self.device)
        # tensors used as indices must be long, int, byte or bool tensors
        self.k_src = torch.tensor(commodity_src, dtype=torch.long, device=self.device)
        self.k_dst = torch.tensor(commodity_dst, dtype=torch.long, device=self.device)
        del commodity_src, commodity_dst, demands, W_scale

        # keep p (M) on device
        self.p = p

        # f_mat (N,K) small-ish dense (-1,0,1)
        f_list = []
        for kk in range(self.K):
            f_np = np.zeros(self.N, dtype=npdtype)
            s_idx, t_idx = commodities[kk][0], commodities[kk][1]
            f_np[s_idx] = -1.0
            f_np[t_idx] = 1.0
            f_list.append(torch.from_numpy(f_np))
        self.f_mat = torch.stack(f_list, dim=1).to(self.device)

        return self.N, self.M

    def pdhg_step_fn(
        self, x_prev, X_prev, Y, x_bar, X_bar, sigma, tau, K, M, overrelax_rho
    ):
        # dual update,explicit,prox is here
        Y_new = Y + sigma * (self.A_matvec(x_bar) - self.S_matvec(X_bar))

        # primal update
        v = (x_prev - tau * self.AT_matvec(Y_new)).reshape(
            M, K
        ) - tau * self.p.unsqueeze(1)
        # x update as projection
        x_new = self.proj(v, self.c)
        # X update as proximal operator
        X_new = self.f1_prox(X_prev + tau * self.ST_matvec(Y_new), tau)

        # overrelaxation
        x_bar = (1 + overrelax_rho) * x_new - overrelax_rho * x_prev
        X_bar = (1 + overrelax_rho) * X_new - overrelax_rho * X_prev

        return x_new, X_new, Y_new, x_bar, X_bar

    def A_matvec(self, x):
        flow = x.view(self.M, self.K)
        # 初始化结果 (N, K)
        div = torch.zeros((self.N, self.K), device=x.device, dtype=x.dtype)
        div.index_add_(0, self.edges_dst, flow)
        div.index_add_(0, self.edges_src, -flow)

        return div.reshape(self.N * self.K)

    def AT_matvec(self, y):
        potentials = y.view(self.N, self.K)
        # tension: (M, K)
        tension = potentials[self.edges_dst] - potentials[self.edges_src]

        return tension.reshape(self.M * self.K)

    def S_matvec(self, X):
        K, N = self.K, self.N

        blocks = self.f_mat * X  # N x K
        return blocks.reshape(N * K)

    def ST_matvec(self, Y):
        K, N = self.K, self.N
        Y_mat = Y.view(N, K)
        return torch.sum(self.f_mat * Y_mat, dim=0)

    def power_iteration_K_norm(self, iters=50):
        """
        Compute ||K||_2 where K = [A  -S].
        A here is mathcal(A), the linear operator in PDHG, not the adjacent or incidence matrix of the graph
        """
        dtype = self.dtype
        device = self.device
        u = torch.randn(self.K * self.M, device=device, dtype=dtype)  # MK
        v = torch.randn(self.K, device=device, dtype=dtype)  # K
        uv = torch.cat([u, v], dim=0)
        uv = uv / uv.norm()  # MK+K

        for _ in range(iters):
            u, v = torch.split(uv, self.K * self.M)
            # K [u; v] = 𝓐(u) - S(v),KN
            Kuv = self.A_matvec(u) - self.S_matvec(v)

            # Kᵀ(K[u;v])
            # Kᵀ y = [𝓐ᵀ y ; -Sᵀ y]
            KT_K_u = self.AT_matvec(Kuv)  # KM=KM*KN * KN
            KT_K_v = -self.ST_matvec(Kuv)  # K=K*KN * KN

            uv_next = torch.cat([KT_K_u, KT_K_v], dim=0)
            norm_next = uv_next.norm()
            uv = uv_next / norm_next  # (M+1)*K

        # sqrt of eigenvalue of KᵀK
        K_norm = norm_next.sqrt()
        return K_norm


    @timed_ns()
    def _pdhg_core(self, x, X, Y, tau, sigma, max_iter, check_interval, tol, overrelax_rho, eta, verbose):
        """
        内部核心迭代函数，专注于计算，便于装饰器统计 GPU 时间。
        """
        x_prev = x.clone()
        X_prev = X.clone()
        x_bar = x.clone()
        X_bar = X.clone()
        
        pweight = torch.tensor(1.0)
        
        # 记录实际迭代次数，用于装饰器统计
        final_iter = 0 

        for it in range(max_iter):
            final_iter = it + 1
            
            x_new, X_new, Y_new, x_bar, X_bar = self.pdhg_step_fn(
                x_prev, X_prev, Y, x_bar, X_bar,
                sigma, tau, self.K, self.M, overrelax_rho,
            )

            # 收敛性检查
            if it % check_interval == 0:
                with torch.no_grad():
                    r_primal = torch.norm(self.A_matvec(x_bar) - self.S_matvec(X_bar))
                    r_dual = (
                        torch.norm(x_new - x_prev) / tau
                        + torch.norm(X_new - X_prev) / tau
                    )
                    
                    # 动态调整步长权重
                    tau, sigma, pweight = self.weight_update(
                        r_primal, r_dual, pweight, eta, tau
                    )
                    
                    rp_val = r_primal.item()
                    rd_val = r_dual.item()

                if verbose:
                    print(f"{it:<8} | {rp_val:.2e}   | {rd_val:.2e} | {tau:.1e}  | {sigma:.1e}")

                if (rp_val < tol) and (rd_val < tol):
                    if verbose:
                        print(f"Converged at iter {it}, r_p={rp_val:.2e}, r_d={rd_val:.2e}")
                    break

            # 更新变量
            x_prev, X_prev = x_new, X_new
            Y = Y_new
            
        return (x_new, X_new, Y_new), None, final_iter

    # -------------------------
    # 对外接口
    # -------------------------
    def pdhg_solve(
        self,
        x0,
        X0,
        Y0,
        tau=None,
        sigma=None,
        max_iter=200000,
        tol=1e-2,
        verbose=True,
        overrelax_rho=1.0,
        check_interval=500,
    ):
        dev = self.device
        
        # 1. 数据准备 (Data Preparation)
        #    这部分通常是 IO 密集或内存操作，不应计入算法的核心 FPS/TPS 评估
        x = x0.to(dev)
        X = X0.to(dev)
        Y = Y0.to(dev).clone() # Y 需要 clone，因为在循环中是迭代更新的
        
        self.p = self.p.to(dev)
        self.c = self.c.to(dev)
        self.d = self.d.to(dev)
        self.W = self.W.to(dev)
        self.f_mat = self.f_mat.to(dev)

        # 2. 参数初始化
        K_norm = self.power_iteration_K_norm()
        eta = 0.9 / K_norm 
        pweight = torch.tensor(1.0)
        
        # 如果未指定，使用 heuristic 计算初始步长
        if tau is None or sigma is None:
            tau = eta / pweight
            sigma = eta * pweight

        if verbose:    
            print(f"{'Iter':<8} | {'P_Res':<10} | {'D_Res':<10} | {'Tao':<8} | {'Sigma':<8}")
            print("-" * 75)

        # 3. 调用核心计算 (Decorated Call)
        #    根据设备选择装饰器逻辑（这里假设主要是 GPU 场景）
        if dev.type == 'cuda':
            # 返回结构为 (Result, Placeholder, Iterations)
            result_tuple, _, _ = self._pdhg_core(
                x, X, Y, tau, sigma, max_iter, check_interval, tol, overrelax_rho, eta, verbose
            )
        else: # 几乎不太可能在 CPU 上运行这个函数，但为了完整性，我们提供一个 fallback 逻辑
            # 如果是在 CPU 上运行，建议另写一个 @cpu_timer 的 core 方法
            # 或者直接调用不带装饰器的逻辑。此处为了演示复用逻辑：
            # 注意：在 CPU 上调用 @gpu_timer 可能会报错（因为 torch.cuda.Event），
            # 实际工程中这里应该根据 device 动态选择方法，或者 _pdhg_core 不加装饰器，
            # 而是在这里手动 wrap。但为了符合你的要求，我们假设在 GPU 运行。
            result_tuple, _, _ = self._pdhg_core(
                x, X, Y, tau, sigma, max_iter, check_interval, tol, overrelax_rho, eta, verbose
            )
        
        # 解包结果
        x_new, X_new, Y_new = result_tuple
        
        return x_new, X_new, Y_new

    def weight_update(self, r_primal, r_dual, pweight, eta, tau):

        scaling = self.weight_update_scaling  # theta
        ratio = r_primal / (r_dual + 1e-12)
        log_p = torch.log(pweight)
        cond1 = ratio > 2.0
        cond2 = ratio < 0.5
        change = torch.where(
            cond1,
            scaling,
            torch.where(cond2, -scaling, torch.tensor(0.0, device=self.device)),
        )
        pweight_new = torch.exp(log_p + change)
        pweight_new = torch.clamp(pweight_new, 1e-5, 1e5)
        tau = eta / pweight_new
        sigma = eta * pweight_new
        return tau, sigma, pweight_new#,eta_new

    # -------------------------
    # prox functions
    # -------------------------
    def f1_prox(self, X_tilde, tau):
        return (X_tilde + 2.0 * tau * self.W * self.d) / (1.0 + 2.0 * tau * self.W)

    def proj(self, U, c):
        """
        Active Set 策略：只对违反容量约束的行进行 Sort 和精确投影。
        保持了数学的精确性（收敛快），同时避免了大量的无效计算（速度快）。
        """
        # 1. 基础处理：任何流量不能为负
        # 这里的 U 还是 (M, K)
        U_clamped = torch.clamp(U, min=0.0)

        # 2. 计算每条边的总流量
        row_sum = U_clamped.sum(dim=1)

        # 3. 找出“坏边”（Active Constraints）：总流量超过容量 c 的边
        # c 是 (M,)
        mask = row_sum > c

        # --- 快速路径 ---
        # 如果没有任何边拥堵，直接返回（这在迭代初期非常常见）
        if not mask.any():
            return U_clamped.reshape(self.M * self.K)

        # --- 慢速路径（只针对坏边） ---
        # 提取坏边的容量和流量
        # c_active: (N_active, )
        # U_active: (N_active, K)
        c_active = c[mask]
        U_active = U_clamped[mask]

        # *** 核心优化：只对 mask 选出来的这部分数据进行 Sort ***
        # 这里的计算量从 M * K log K 降到了 N_active * K log K
        U_sorted, _ = torch.sort(U_active, dim=1, descending=True)

        # --- 下面是标准的单纯形投影逻辑 (Simplex Projection) ---

        # 计算前缀和 cumsum
        cssv = torch.cumsum(U_sorted, dim=1) - c_active.unsqueeze(1)

        # 生成对应的除数 range: [1, 2, 3, ..., K]
        # 注意：为了防止 shape 广播错误，一定要注意维度
        arange_k = torch.arange(1, self.K + 1, device=U.device, dtype=U.dtype)

        # 计算 rho 的候选值
        cond = U_sorted - cssv / arange_k > 0.0

        # 找到满足条件的最大索引 rho
        # cond 是 bool，转 float 算 sum 得到满足条件的个数，减 1 得到下标
        rho = cond.sum(dim=1, keepdim=True).long() - 1

        # 获取对应的 theta (阈值)
        # gather 用于从 active 的 cssv 中取对应 rho 位置的值
        theta = torch.gather(cssv, 1, rho) / (rho.float() + 1)

        # 执行投影：w = max(u - theta, 0)
        w_active = torch.clamp(U_active - theta, min=0.0)

        # --- 结果写回 ---
        # 将计算好的拥堵边投影结果写回大矩阵
        # Clone 是为了不破坏原计算图的 Leaf（视具体 Autograd 需求，但在 PDHG 这种 volatile 上下文中通常是安全的）
        U_out = U_clamped.clone()
        U_out[mask] = w_active

        return U_out.reshape(self.M * self.K)

    def make_initials(self, warm_start=True):
        # 0. 基础全零初始化 (以防 W_adj 未准备好)
        if self.W_adj is None and warm_start:
            print("Warning: W_adj is None, falling back to cold start.")
            return self._generate_cold_start()

        # 1. 根据模式分发
        if warm_start:
            # return self._generate_warm_start()
            return self._generate_mwu_warm_start(batches=20)
        else:
            return self._generate_cold_start()

    @timed_ns()
    def _generate_warm_start(self):
        """
        Warm Start 策略1：
        x, X: Greedy Capacity Filling (考虑容量限制)
        Y: Cost-to-Go Potential (最短路距离)
        """
        device = self.device
        dtype = self.dtype
        K, M, N = self.K, self.M, self.N
        k_src, k_dst = self.k_src, self.k_dst
        demands, capacities = self.d, self.c
        weights = self.W

        # --- APSP ---
        # D, P, _ = ws.apsp_gpu(self.W_adj, dtype=dtype)
        D, P, _ = ws.parallel_bellman_ford_gpu(self.W_adj, dtype=dtype) # 优化过的 apsp_gpu，之后都采用这个
        del self.W_adj
        self.W_adj = None

        # --- 路径追踪 ---
        edge_lookup = torch.full((N, N), -1, dtype=torch.long, device=device)
        edge_lookup[self.edges_src, self.edges_dst] = torch.arange(M, device=device)

        x_flow = torch.zeros((M, K), dtype=dtype, device=device)
        curr_nodes = k_src.clone()
        active_mask = torch.ones(K, dtype=torch.bool, device=device)

        for _ in range(N):
            arrived = curr_nodes == k_dst
            active_mask = active_mask & (~arrived)
            if not active_mask.any():
                break

            next_nodes = P[curr_nodes, k_dst]
            edge_ids = edge_lookup[curr_nodes, next_nodes]
            valid_step = active_mask & (edge_ids != -1)
            
            if not valid_step.any():
                break

            valid_k = torch.nonzero(valid_step, as_tuple=True)[0]
            valid_e = edge_ids[valid_step]
            x_flow[valid_e, valid_k] = demands[valid_step]
            curr_nodes[valid_step] = next_nodes[valid_step]

        # --- Greedy Capacity Filling ---
        sorted_indices = torch.argsort(weights, descending=True)
        inv_sorted_indices = torch.argsort(sorted_indices)

        x_sorted = x_flow[:, sorted_indices]
        d_sorted = demands[sorted_indices]

        cum_flow = torch.cumsum(x_sorted, dim=1)
        prev_flow = cum_flow - x_sorted
        
        caps_expanded = capacities.view(-1, 1)
        residual_caps = (caps_expanded - prev_flow).clamp(min=0)
        allowed_flow_edge = torch.min(x_sorted, residual_caps)

        epsilon = 1e-6
        ratios_edge = allowed_flow_edge / (x_sorted + epsilon)
        path_mask = x_sorted > epsilon
        ratios_edge[~path_mask] = 1.0

        path_bottleneck_ratios, _ = torch.min(ratios_edge, dim=0)

        final_ratios = path_bottleneck_ratios[inv_sorted_indices]
        x_final = x_flow * final_ratios.view(1, -1)
        X_final = demands * final_ratios

        # --- Dual Y (Distance) ---
        Y_matrix = D[:, k_dst]
        finite_mask = torch.isfinite(Y_matrix)
        if finite_mask.any():
            max_val = Y_matrix[finite_mask].max()
            Y_matrix = torch.where(finite_mask, Y_matrix, max_val * 2.0)
        else:
            Y_matrix = torch.zeros_like(Y_matrix)

        return x_final.reshape(-1)*0.8, X_final*0.8, Y_matrix.reshape(-1)

    def _generate_cold_start(self):
        """
        旧的 Cold Start 策略 (复刻)：
        对应原代码中 x0=None, X0=None 的情况。
        不运行 APSP，不进行路径追踪。
        
        逻辑：
        x0 = 0
        X0 = 0
        Y0 = -kappa * (A(x0) - S(X0)) => 0
        """
        device = self.device
        dtype = self.dtype
        
        # 1. 初始化 x0 (全0)
        x0 = torch.zeros(self.M * self.K, device=device, dtype=dtype)
        
        # 2. 初始化 X0 (全0)
        X0 = torch.zeros(self.K, device=device, dtype=dtype)
        
        # 3. 初始化 Y0
        # 对应旧代码 pdhg_solve 中的: rY = self.A_matvec(x) - self.S_matvec(X)
        # 因为 x=0, X=0，所以 rY=0，进而 Y=0。
        # Y 的维度是节点数 N * 商品数 K
        Y0 = torch.zeros(self.N * self.K, device=device, dtype=dtype)

        return x0, X0, Y0


    @timed_ns()
    def _generate_mwu_warm_start(self, batches=10):
        device = self.device
        dtype = self.dtype
        K, M, N = self.K, self.M, self.N
        
        k_src, k_dst = self.k_src, self.k_dst
        demands, capacities = self.d, self.c
        base_cost = self.p 
        
        # [优化1] 预分配结果容器
        x_total = torch.zeros((M, K), dtype=dtype, device=device)
        
        # [优化2] 缓存边的负载 (M,)，避免每轮对 (M, K) 做 sum(dim=1)
        # 用 float32 累加防止溢出精度问题，最后再转回 dtype
        current_edge_loads = torch.zeros(M, dtype=torch.float32, device=device) 
        current_edge_weights = base_cost.clone()
        
        # 预计算每轮流量
        chunk_demands = demands / batches

        # Edge lookup: (N, N) -> edge_index
        edge_lookup = torch.full((N, N), -1, dtype=torch.long, device=device)
        edge_lookup[self.edges_src, self.edges_dst] = torch.arange(M, device=device)

        # 预分配 Buffer，避免循环内重复 malloc
        adj_matrix = torch.full((N, N), float('inf'), dtype=dtype, device=device)
        # adj_matrix.fill_diagonal_(0.0) # 放到循环里一起做

        last_D = None
        if self.W_adj is not None:
            del self.W_adj
            self.W_adj = None

        # 循环不变量：源节点列表不需要在循环里 clone
        # 但我们需要追踪的 curr_nodes 会变，所以只在循环开始初始化
        
        # 预先生成 mask 容器
        active_mask_buffer = torch.ones(K, dtype=torch.bool, device=device)

        for b in range(batches):
            # --- A. 重建邻接矩阵 ---
            # 直接赋值，比 fill + fill_diagonal 略快，利用广播
            adj_matrix.fill_(float('inf'))
            adj_matrix.diagonal().fill_(0.0) 
            adj_matrix[self.edges_src, self.edges_dst] = current_edge_weights

            # --- B. 运行 APSP ---
            # 假设返回 (Distance, Predecessors, _)
            # P[src, dst] = predecessor of dst on path from src
            D, P, _ = ws.parallel_bellman_ford_gpu(adj_matrix, dtype=dtype)
            last_D = D

            # --- C. 路径追踪 (反向回溯: DST -> SRC) ---
            # 重置追踪状态
            curr_nodes = k_dst.clone() 
            active_mask_buffer.fill_(True)
            active_mask = active_mask_buffer # 引用

            # 提取当前批次的需求量 (避免循环内索引)
            # chunks = chunk_demands # 也是 Tensor (K,)

            for _ in range(N):
                # 1. 检查是否抵达源点 (原地操作减少内存开销)
                # arrived = (curr_nodes == k_src)
                # active_mask &= (~arrived)
                # 优化写法：仅对 active 的进行检查
                
                # 这里的逻辑是：如果是 active 且 node == src，则设为 inactive
                # 使用 where 或者 直接索引更新
                reached_src = (curr_nodes == k_src)
                active_mask.masked_fill_(reached_src, False)

                if not active_mask.any():
                    break
                
                # 2. 查找前驱节点 P[k_src, curr_nodes]
                # 只需计算 active 部分，减少索引开销
                # 但为了利用 P 的连续内存，全量索引可能比 mask 后索引更快（取决于 K 大小）
                # 这里保持全量索引，但在 edge_ids 处过滤
                prev_nodes = P[k_src, curr_nodes]

                # 3. 查找边 ID
                edge_ids = edge_lookup[prev_nodes, curr_nodes]
                
                # 4. 确定有效步
                # valid_step = active_mask & (edge_ids != -1) & (prev_nodes != -1)
                # 这里 P 为 -1 意味着不可达 (假设ws库行为)
                # 组合判断逻辑，减少中间变量
                valid_step = active_mask & (edge_ids >= 0) & (prev_nodes >= 0)

                if not valid_step.any():
                    break
                
                # 5. [核心优化] 直接累加到 x_total 和 current_edge_loads
                # 避免创建 temp_x_batch
                
                # 获取 indices
                valid_k_indices = torch.nonzero(valid_step, as_tuple=True)[0]
                valid_e_indices = edge_ids[valid_step]
                flow_values = chunk_demands[valid_step]

                # 更新总流矩阵 (使用 index_put_ accumulate=True)
                # x_total[valid_e_indices, valid_k_indices] += flow_values
                x_total.index_put_((valid_e_indices, valid_k_indices), flow_values, accumulate=True)

                # [优化点] 同步更新边负载缓存，避免 step F 的 sum(dim=1)
                # current_edge_loads[valid_e_indices] += flow_values
                current_edge_loads.index_put_((valid_e_indices,), flow_values.float(), accumulate=True)

                # 6. 移动节点
                curr_nodes[valid_step] = prev_nodes[valid_step]

            # --- D. 拥塞感知更新 (MWU) ---
            if b < batches - 1:
                # [优化] 直接使用缓存的 edge_loads，复杂度 O(M) 而非 O(M*K)
                # congestion = current_edge_loads / (capacities + 1e-6)
                
                # 合并计算步骤减少 kernel launches
                # penalty = 1.0 + 2.0 * relu(load/cap - 1.0)
                # weight = base * penalty
                
                # 利用 JIT friendly 的写法 (虽然这里没开 JIT)
                congestion_ratio = current_edge_loads / (capacities + 1e-6)
                overload = torch.relu_(congestion_ratio - 1.0) # In-place relu
                
                # 向量化计算新权重
                penalty_factor = overload # 复用内存
                penalty_factor.mul_(2.0).add_(1.0) # 1 + 2 * overload
                
                # update weights
                current_edge_weights = base_cost * penalty_factor
                # 确保数值稳定性 (max 是必要的，虽然理论上 penalty >= 1)
                current_edge_weights.max(base_cost) 

        # --- E. 结果组装 ---
        x_final = x_total.mul_(0.9) # In-place mul
        X_final = demands * 0.9

        # --- F. Dual 初始化 ---
        # 确保 last_D 存在
        if last_D is None: 
            # Fallback for batches=0 case
             D, _, _ = ws.parallel_bellman_ford_gpu(adj_matrix, dtype=dtype)
             last_D = D
             
        # Y[u, k] ~ dist(u, k_dst)
        # last_D: (N, N), k_dst: (K,)
        # result: (N, K)
        Y_matrix = last_D[:, k_dst] 
        
        # 处理 Inf
        finite_mask = torch.isfinite(Y_matrix)
        if finite_mask.any():
            max_val = Y_matrix[finite_mask].max()
            # 使用 where 填充
            Y_matrix = torch.where(finite_mask, Y_matrix, max_val * 2.0)
        else:
            Y_matrix.zero_()
        
        return x_final.reshape(-1), X_final, Y_matrix.reshape(-1)
