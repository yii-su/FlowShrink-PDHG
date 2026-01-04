import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import FlowShrink.utils as utils
import FlowShrink.shortest_paths_gpu as ws

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True # 避免一些无关警告
import numpy as np
from scipy.sparse import coo_matrix
import time

class MCNFPDHG:
    def __init__(self,dtype=torch.float64):
        self.device = torch.device('cuda:0')
        self.dtype = dtype
    
    def create_data(self, num_nodes, k, num_commodities, seed=1, warm_start=False):
        self.N = num_nodes
        self.K = num_commodities
        device = self.device
        dtype=self.dtype
        if dtype==torch.float64:
            npdtype=np.float64
        else:
            npdtype=np.float32

        # adjacency and incidence
        W_adj = utils.create_base_network(self.N, k, seed)
        W_adj = utils.ensure_weak_connectivity(W_adj, seed)
        A_inc_np, p_np = utils.adjacency_to_incidence(W_adj)# (N, M)
        commodities = utils.create_commodities(W_adj, self.K, 10.0, seed)
        if warm_start:
            self.W_adj=torch.tensor(W_adj,dtype=dtype,device=device)
        else:
            self.W_adj=None
        del W_adj
        
        # capacities
        c_np = utils.generate_capacity_constraints(A_inc_np, commodities, 1.0, 5.0, seed=seed)
        self.c = torch.from_numpy(c_np.astype(npdtype)).to(self.device)
        A_inc=torch.from_numpy(A_inc_np)
        del A_inc_np
        '''
        torch.where(condition) 在处理二维张量时，返回的索引是按照 Row-Major（行优先） 顺序排列的，即先扫描第0行，再扫描第1行，以此类推。
        然而，你的 c（容量）、p（费用）以及变量 x 都是按照 Edge Index（列索引 0 到 M-1） 排列的。
        你的假设：edges_src[j] 对应第 j 条边（即 A_inc 的第 j 列）的源节点。
        实际情况：edges_src 只是包含了所有源节点的列表，但按照节点ID排序（受行扫描顺序影响），完全打乱了与边索引 0...M-1 的对应关系。 
        实际变成了沿着dim=1寻找 
        后果：
        PDHG 求解器实际上是在一个 乱连线的图 上进行优化。边的起点和终点被重新洗牌了，但边的容量和费用却保持原序。     
        '''
        # self.edges_src=torch.where(A_inc==-1)[0].to(device)# M
        # self.edges_dst=torch.where(A_inc==1)[0].to(device)# M
        
        # argmin 找到每列最小值的索引（即 -1 所在的行索引）
        self.edges_src = torch.argmin(A_inc, dim=0).to(device)
        # argmax 找到每列最大值的索引（即 1 所在的行索引）
        self.edges_dst = torch.argmax(A_inc, dim=0).to(device)
        self.M=A_inc.shape[1]
        del A_inc

        # edge cost
        p = torch.from_numpy(p_np.astype(npdtype)).to(self.device)
        del p_np

        # W, d
        W_scale=300.0
        #self.W已经乘了W_scale
        self.W = torch.from_numpy(utils.generate_weight(self.K,dimtype='vector', seed=seed)*W_scale).to(self.device).to(dtype)
        commodity_src = [c[0] for c in commodities]
        commodity_dst = [c[1] for c in commodities]
        demands = [c[2] for c in commodities]
        self.d = torch.tensor(demands, dtype=dtype, device=self.device)
        # tensors used as indices must be long, int, byte or bool tensors
        self.k_src=torch.tensor(commodity_src, dtype=torch.long, device=self.device)
        self.k_dst=torch.tensor(commodity_dst, dtype=torch.long, device=self.device)
        del commodity_src,commodity_dst,demands,W_scale

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
    
    def pdhg_step_fn(self,x_prev, X_prev, Y, x_bar,X_bar, sigma, tau,  
                 K, M, overrelax_rho):
        # dual update,explicit,prox is here
        Y_new = Y + sigma * (self.A_matvec(x_bar) - self.S_matvec(X_bar))

        #primal update
        v=(x_prev-tau*self.AT_matvec(Y_new)).reshape(M,K)-tau*self.p.unsqueeze(1)
        #x update as projection
        x_new= self.proj(v,self.c)
        #X update as proximal operator
        X_new = self.f1_prox(X_prev + tau * self.ST_matvec(Y_new), tau)
        
        #overrelaxation
        x_bar=(1+overrelax_rho)*x_new-overrelax_rho*x_prev
        X_bar=(1+overrelax_rho)*X_new-overrelax_rho*X_prev
    
        return x_new, X_new, Y_new, x_bar, X_bar


    # -------------------------
    # 矩阵-向量接口（稀疏化）
    # -------------------------
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
        #pytorch的广播机制用于标量*向量时：
        #f_mat为N×K矩阵，此处操作应为对f_mat的每一列，用标量X_k去乘
        #正确方法应为将X广播为1行K列，对应f_mat的K列，每一列一个标量与该列做乘法（列线性变换），即X_col = X.unsqueeze(0)
        #或直接省略unsqueeze，运算符*会触发pytorch的自动标量乘广播
        #此处X_col = X.unsqueeze(1)没有报错，是因为测试数据中N==K，掩盖了维度的不匹配
        blocks = self.f_mat * X   # N x K
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
        dtype=self.dtype
        device = self.device
        u = torch.randn(self.K*self.M, device=device, dtype=dtype)#MK
        v=torch.randn(self.K, device=device, dtype=dtype)#K
        uv=torch.cat([u,v],dim=0)
        uv=uv/uv.norm()#MK+K
        
        for _ in range(iters):
            u,v=torch.split(uv,self.K*self.M)
            # K [u; v] = 𝓐(u) - S(v),KN
            Kuv = self.A_matvec(u) - self.S_matvec(v)

            # Kᵀ(K[u;v])
            # Kᵀ y = [𝓐ᵀ y ; -Sᵀ y]
            KT_K_u = self.AT_matvec(Kuv)#KM=KM*KN * KN
            KT_K_v = -self.ST_matvec(Kuv)#K=K*KN * KN

            uv_next = torch.cat([KT_K_u, KT_K_v], dim=0)
            norm_next = uv_next.norm()
            uv = uv_next / norm_next#(M+1)*K

        # sqrt of eigenvalue of KᵀK
        K_norm = norm_next.sqrt()
        return K_norm


    # -------------------------
    # PDHG solver with automated tau/sigma tuning and relaxation theta
    # -------------------------
    def pdhg_solve(self,
                x0=None, X0=None,
                tau=None, sigma=None,
                kappa_Y=1.0,
                max_iter=100000, tol=1e-2,
                verbose=True, overrelax_rho=1.0, check_interval=500):
        dev = self.device
        K, M = self.K, self.M
        dtype = self.dtype

        if x0 is None:
            x = torch.zeros(M*K, device=dev, dtype=dtype)
        else:
            x = x0.clone().to(dev)
        if X0 is None:
            X = torch.zeros(K, device=dev, dtype=dtype)
        else:
            X = X0.clone().to(dev)

        # ensure data on device
        self.p = self.p.to(dev)
        self.c = self.c.to(dev)
        self.d = self.d.to(dev)
        self.W = self.W.to(dev)
        self.f_mat = self.f_mat.to(dev)
           
        # calculate the 2-norm of linear operator in our problem formulation to ensure convergence
        K_norm = self.power_iteration_K_norm()
        eta = 0.9 / K_norm # safer estimation, K_norm is derived by iteration
        pweight = torch.tensor(1.0)
        tau = eta/pweight
        sigma = eta*pweight

        # residual-based dual init
        rY = self.A_matvec(x) - self.S_matvec(X)
        Y = -kappa_Y * rY
        x_prev = x.clone()
        X_prev = X.clone()
        x_bar = x.clone()
        X_bar = X.clone()

        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        for it in range(max_iter):
            x_new, X_new, Y_new, x_bar, X_bar = self.pdhg_step_fn(x_prev,X_prev,Y,x_bar,X_bar,sigma,tau,self.K,self.M,overrelax_rho)
            
            if (it == max_iter - 1):
                print(f'Max iterations reached, r_p={r_primal:.2e}, r_d={r_dual:.2e}')

            if it % check_interval == 0:
                # residuals
                with torch.no_grad():
                    r_primal = torch.norm(self.A_matvec(x_bar) - self.S_matvec(X_bar))
                    r_dual = torch.norm(x_new - x_prev)/tau + torch.norm(X_new - X_prev)/tau
                    tau, sigma, pweight = self.weight_update(r_primal,r_dual,pweight,eta,tau)
                    rp_val=r_primal.item()
                    rd_val=r_dual.item()

                
                if verbose:
                    print(f'Iter {it:6d} | r_p={r_primal:.2e} | r_d={r_dual:.2e} | pweight={pweight}')
                    
                if (rp_val < tol) and (rd_val < tol):
                    print(f'Converged at iter {it}, r_p={r_primal:.2e}, r_d={r_dual:.2e}')
                    break
            
            #shift iteration
            x_prev,X_prev=x_new,X_new
            Y=Y_new

        if dev.type == 'cuda':
            torch.cuda.synchronize()
        if verbose:
            print('PDHG total time:', time.time()-t0)
        return x_new, X_new, Y_new

    def weight_update(self, r_primal,r_dual,pweight, eta, tau):
        scaling = torch.tensor(0.5, device=self.device) # theta
        ratio = r_primal / (r_dual + 1e-12)
        log_p=torch.log(pweight)
        cond1=ratio>10.0
        cond2=ratio<0.1
        change = torch.where(cond1, scaling, torch.where(cond2, -scaling, torch.tensor(0.0, device=self.device)))
        pweight_new=torch.exp(log_p + change)
        pweight_new = torch.clamp(pweight_new, 1e-5, 1e5)
        tau = eta / pweight_new
        sigma = eta * pweight_new
        return tau, sigma, pweight_new
    
    # -------------------------
    # prox functions
    # -------------------------
    def f1_prox(self, X_tilde, tau):
        return (X_tilde+2.0*tau*self.W*self.d) / (1.0+2.0*tau*self.W)
    
    
    def proj(self, U, c):
        """
        U: (M, K) unconstrained flow M ROW K COL
        c: (M,) flow capacity per edge
        """
        c_expanded = c.unsqueeze(1) # (M, 1)
        
        # 1. Clip negative values
        U_clipped = torch.clamp(U, min=0)
        
        # 2. Check sum constraint along dim=1 (Commodities)
        row_sum = U_clipped.sum(dim=1, keepdim=True) # (M, 1)
        
        # 3. Sort along dim=1
        U_sorted, _ = torch.sort(U_clipped, dim=1, descending=True)
        
        # 4. Cumsum along dim=1
        S_cum = U_sorted.cumsum(dim=1)
        
        # 5. Calculate Tau Candidates
        # (M, K) - (M, 1) / (1, K) -> (M, K)
        tau_candidates = (S_cum - c_expanded) / torch.arange(1,self.K+1,device= U.device,dtype=U.dtype).view(1,self.K)
        
        # 6. Find rho (active set size)
        cond = U_sorted > tau_candidates
        # Count true values along dim=1
        rho = cond.type(torch.int8).sum(dim=1) - 1
        rho = torch.clamp(rho, min=0) # (M,)
        
        # 7. Gather Tau
        # rho shape (M,), need (M, 1) to gather from (M, K)
        tau_selected = torch.gather(tau_candidates, 1, rho.unsqueeze(1)) # (M, 1)
        
        # 8. Projection
        x_proj = torch.clamp(U_clipped - tau_selected, min=0)
        
        # 9. Final Select
        # If sum <= c, keep original, else project
        need_proj = row_sum > c_expanded
        x_out = torch.where(need_proj, x_proj, U_clipped)
        
        return x_out.reshape(self.M*self.K)

    def make_initials(self):
        dtype=self.dtype
        dev = self.device
        if self.W_adj is None:
            x0 = torch.zeros(self.M * self.K, device=dev, dtype=dtype)
            X0 = torch.zeros(self.K, device=dev, dtype=dtype)
        else:
            x0,X0=self.generate_initial_flow_gpu()
        return x0, X0
    
    def generate_initial_flow_gpu(self):
        """
        在 GPU 上根据前驱（Next-Hop）矩阵 P 重建路径并生成初始流 x0。
            
        返回:
            x0: torch.Tensor, (M * K), 展平的初始流量向量
            X0: torch.Tensor, (K,), 初始送达量向量, 与需求相同
        """
        device=self.device
        dtype=self.dtype
        # 1. 数据准备
        K = self.K
        # 各个commodities的原点和汇点
        k_src = self.k_src
        k_dst = self.k_dst
        demands = self.d # (K,)
        edges_src=self.edges_src
        edges_dst=self.edges_dst
        M=self.M
        N=self.N
        _,P,_=ws.apsp_gpu(self.W_adj,dtype=dtype)
        del self.W_adj
        self.W_adj=None

        # 2. 构建 (u, v) -> edge_index 的快速查找表
        # 这一步只需要做一次。如果是类成员函数，可以在 __init__ 或 create_data 中缓存 edge_lookup
        edge_lookup = torch.full((N, N), -1, dtype=torch.long, device=device)
        edge_lookup[edges_src, edges_dst] = torch.arange(M, device=device)

        # 3. 初始化流量矩阵 (M, K)
        x_flow = torch.zeros((M, K), dtype=dtype, device=device)
        
        # 4. 并行路径追踪 (Pointer Chasing)
        # 所有商品并行从 s 出发走向 t
        curr_nodes = k_src.clone()
        
        # 记录哪些商品已经到达终点，避免多余计算
        active_mask = torch.ones(K, dtype=torch.bool, device=device)
        
        # 循环次数上限设为 N (最坏情况路径长度)
        # 实际上对于稀疏图和小直径网络，这个循环会非常快
        for _ in range(N):
            # 如果所有商品都到达终点，提前退出
            # 注意：这里需要排除掉那些 s==t 的琐碎情况（如果有的话）
            arrived = (curr_nodes == k_dst)
            active_mask = active_mask & (~arrived)
            
            if not active_mask.any():
                break
                
            # --- 核心逻辑 ---
            
            # 1. 查找下一跳节点
            # P shape [N, N]. gather indices: curr_nodes [K], t_indices [K]
            # P[u, v] 代表从 u 去 v 的下一个节点
            # 利用 Advanced Indexing: P[row_indices, col_indices]
            next_nodes = P[curr_nodes, k_dst] # shape (K,)
            
            # 2. 查找对应的边索引
            edge_ids = edge_lookup[curr_nodes, next_nodes] # shape (K,)
            
            # 3. 只有 active 且 边存在的商品才更新流量
            # edge_ids == -1 说明图不连通或 P 矩阵指引了不存在的边
            valid_step = active_mask & (edge_ids != -1)
            
            if not valid_step.any():
                # 所有活跃的商品都找不到路了（图不连通），直接退出防止死循环
                break
                
            # 4. 填入流量
            # 选取有效的 k 索引
            valid_k = torch.nonzero(valid_step, as_tuple=True)[0]
            valid_e = edge_ids[valid_step]
            
            x_flow[valid_e, valid_k] = demands[valid_step]
            
            # 5. 更新位置
            curr_nodes[valid_step] = next_nodes[valid_step]

            # 5. 展平并返回 (M*K)和demand（松弛容量约束，则一定全部送达，X==d）
        return x_flow.reshape(-1),self.d
