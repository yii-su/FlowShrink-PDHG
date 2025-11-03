import torch
import numpy as np
from scipy.sparse import coo_matrix
import time
import FlowShrink.utils as utils

class MultiCommodityNetworkFlowPDHG:
    def __init__(self, num_nodes, num_edges, num_commodities, device='cuda:0'):
        self.N = num_nodes  # 节点数
        self.M = num_edges  # 边数
        self.K = num_commodities  # 商品数
        self.device = device
        # k是每个节点的出度
        k = 10
        seed = 1
        # 生成网络邻接矩阵W_adj（N*N)
        W_adj = utils.create_base_network(self.N, k, seed)
        W_adj = utils.ensure_weak_connectivity(W_adj, seed)

        # 生成网络关联矩阵A_inc（N*M)和成本向量p(M)
        self.A_inc, p = utils.adjacency_to_incidence(W_adj)

        # 商品列表，每个元素是元组 (s_k, t_k, d_k)
        commodities = utils.create_commodities(W_adj, self.K, 10.0, seed)
        
        # 流量守恒聚合矩阵A(KN x KM)
        self.A = torch.kron(torch.eye(self.K), self.A_inc).to(device)
        
        # 容量聚合矩阵C (M x KM)
        self.C = torch.kron(torch.ones(1, self.K), torch.eye(self.M)).to(device)  
        
        # 权重矩阵W (K x K)
        self.W = utils.generate_weight(self.K).to(self.device) 
        
        # 运输成本向量p_tilde (KM),暂定为所有商品共有相同的边成本
        self.p_tilde = torch.kron(torch.ones(self.K,device=self.device),p) 
        
        # 源汇矩阵S (KN x K)
        f_list=[]
        for k in range (self.K):
            f=torch.zeros[self.N]
            f[commodities[k][0]]=-1.0
            f[commodities[k][1]]=1.0
            f_list.append(f)
        f_mat=torch.stack(f_list)
        self.S=torch.zeros(self.K*self.N,self.K)
        for k in range(self.K):
            self.S[k*self.N:(k+1)*self.N, k] = f_mat[k]
        self.S.to(self.device)
        
        # 容量向量c (M)
        self.c = utils.generate_capacity_constraints(self.A_inc,commodities,1.0,5.0).to(device)
        
        # 需求向量d (K)
        self.d = torch.tensor(commodities[:][2],device=self.device)  
    
    def initialize_variables(self):
        """初始化PDHG变量"""
        # 原始变量
        self.x = torch.zeros(self.K * self.M, device=self.device)  # 流量变量
        self.X = torch.zeros(self.K, device=self.device)  # 净流量变量
        
        # 对偶变量
        self.y = torch.zeros(self.K * self.N, device=self.device)  # 流量守恒乘子
        self.z = torch.zeros(self.M, device=self.device)  # 容量约束乘子
        
        # 扩展运输成本向量
        self.expandp = torch.cat([self.p] * self.K).to(self.device)

    def pdhg_solve(self,A, S, C, c,
                x0, X0, tau=1e-2, sigma=1e-2,
                kappa_Y=1.0, kappa_Z=1.0,
                max_iter=10000, tol=1e-5, device='cuda:0'):
        """
        PDHG solver with residual-based dual initialization (GPU accelerated).
        
        Parameters
        ----------
        A, S, C : torch.Tensor
            Linear operators (matrices) for constraints.
        c : torch.Tensor
            Capacity vector.
        x0, X0 : torch.Tensor
            Initial primal variables.
        tau, sigma : float
            Step sizes for primal and dual updates.
        kappa_Y, kappa_Z : float
            Scaling factors for residual initialization.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance.
        device : str
            GPU device (e.g. 'cuda:0').
        """

        # === move to device ===
        A = A.to(device)
        S = S.to(device)
        C = C.to(device)
        c = c.to(device)
        x = x0.clone().to(device)
        X = X0.clone().to(device)

        torch.cuda.synchronize()
        start_time = time.time() # start timing
        # === residual-based initialization (方案2) ===
        rY = A @ x - S @ X
        rZ = C @ x - c
        Y = -kappa_Y * rY
        Z = torch.clamp(kappa_Z * rZ, min=0.0)

        # === preallocate for extrapolation ===
        x_bar = x.clone()
        X_bar = X.clone()

        # === iteration loop ===
        for it in range(max_iter):
            # Dual updates
            Y = Y + sigma * (A @ x_bar - S @ X_bar)
            Z = torch.clamp(Z + sigma * (C @ x_bar - c), min=0.0)

            # Primal updates via proximal operators
            x_prev = x.clone()
            X_prev = X.clone()

            # gradient step
            grad_x = A.T @ Y + C.T @ Z
            grad_X = -S.T @ Y

            x = self.f2_prox(x - tau * grad_x, tau)
            X = self.f1_prox(X - tau * grad_X, tau)

            # Extrapolation
            x_bar = 2 * x - x_prev
            X_bar = 2 * X - X_prev

            # Compute residuals
            r_primal = torch.norm(A @ x - S @ X)
            r_dual = torch.norm(x - x_prev) + torch.norm(X - X_prev)

            if (r_primal < tol) and (r_dual < tol):
                print(f'Converged at iter {it}, r_p={r_primal:.2e}, r_d={r_dual:.2e}')
                break

            if it % 1000 == 0:
                print(f'Iter {it:5d} | r_p={r_primal:.2e} | r_d={r_dual:.2e}')
                
        torch.cuda.synchronize()
        print('pdhg to optimal time:', time.time()-start_time)
        return x, X, Y, Z

    def f1_prox(self, v, tau):
        """
        prox for f1(X) = (X - d)^T W (X - d)
        => prox_{tau f1}(v) = (I + 2*tau*W)^{-1} (v + 2*tau*W*d)
        """
        I = torch.eye(len(self.d), device=self.device)
        M = I + 2 * tau * self.W
        rhs = v + 2 * tau * (self.W @ self.d)
        return torch.linalg.solve(M, rhs)

    def f2_prox(self, v, tau):
        """
        prox for f2(x) = p_tilde^T x + I_{x>=0}
        => prox_{tau f2}(v) = max(0, v - tau * p_tilde)
        """
        return torch.clamp(v - tau * self.p_tilde, min=0.0)
    
    
    def solve(self, ε_rel=1e-6, ε_abs=1e-8, max_iter=10000, check_interval=100):
        """PDHG主求解函数"""
        print("开始PDHG求解...")
        start_time = time.time()
        
        # 初始化变量
        self.initialize_variables()
        
        # 计算算子范数并设置步长
        L = self.compute_operator_norm()
        print(f"估计的算子范数: {L:.6f}")
        
        # 设置步长 (满足 tausigma‖A‖² < 1)
        tau = 0.9 / L
        sigma = 0.9 / L
        
        tau_x = tau  # x的步长
        tau_X = tau  # X的步长  
        sigma_y = sigma  # y的步长
        sigma_z = sigma  # z的步长
        
        print(f"步长设置: tau_x={tau_x:.6f}, tau_X={tau_X:.6f}, sigma_y={sigma_y:.6f}, sigma_z={sigma_z:.6f}")
        
        # 计算初始残差
        R_0, R_pri_0, R_dual_0, R_comp_0 = self.compute_optimality_residual(tau_x, tau_X)
        print(f"初始最优残差: {R_0:.6e}")
        
        # PDHG迭代
        for k in range(max_iter):
            # 保存前一次迭代值
            x_old = self.x.clone()
            X_old = self.X.clone()
            
            # ========== 原始变量更新 ==========
            
            # 1. 更新x (流量变量)
            grad_x = torch.matmul(self.expandA.t(), self.y) + torch.matmul(self.C.t(), self.z) + self.expandp
            x_temp = self.x - tau_x * grad_x
            self.x = torch.relu(x_temp)  # 投影到非负象限
            
            # 2. 更新X (净流量变量)
            # 使用二次项+非负约束的精确近端算子
            numerator = self.X + tau_X * (torch.matmul(self.S.t(), self.y) + 2 * torch.matmul(self.W, self.d))
            denominator = 1 + 2 * tau_X * torch.diag(self.W)
            X_temp = numerator / denominator
            self.X = torch.relu(X_temp)  # 投影到非负象限
            
            # ========== 对偶变量更新 (使用外推) ==========
            
            # 计算外推点
            x_bar = 2 * self.x - x_old
            X_bar = 2 * self.X - X_old
            
            # 1. 更新y (流量守恒乘子)
            dual_grad_y = torch.matmul(self.expandA, x_bar) - torch.matmul(self.S, X_bar)
            self.y = self.y + sigma_y * dual_grad_y
            
            # 2. 更新z (容量约束乘子)
            dual_grad_z = torch.matmul(self.C, x_bar) - self.c
            z_temp = self.z + sigma_z * dual_grad_z
            self.z = torch.relu(z_temp)  # 投影到非负象限
            
            # ========== 收敛检查 ==========
            if k % check_interval == 0 or k == max_iter - 1:
                R_current, R_pri, R_dual, R_comp = self.compute_optimality_residual(tau_x, tau_X)
                
                # 计算目标函数值
                objective = torch.dot((self.X - self.d), torch.matmul(self.W, (self.X - self.d))) + torch.dot(self.expandp, self.x)
                
                print(f"迭代 {k:4d}: 目标值={objective.item():.6e}, "
                      f"残差={R_current:.2e}, R_pri={R_pri:.2e}, R_dual={R_dual:.2e}, R_comp={R_comp:.2e}")
                
                # 收敛检查
                if R_current < ε_rel * R_0:
                    print(f"相对收敛于迭代 {k}, 残差: {R_current:.2e}")
                    break
                if R_current < ε_abs:
                    print(f"绝对收敛于迭代 {k}, 残差: {R_current:.2e}")
                    break
                    
        end_time = time.time()
        print(f"求解完成! 总耗时: {end_time - start_time:.2f}秒")
        print(f"总迭代次数: {k+1}")
        
        return self.get_solution()