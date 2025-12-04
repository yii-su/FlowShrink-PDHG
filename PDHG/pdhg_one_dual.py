import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import FlowShrink.utils as utils

import torch
import numpy as np
from scipy.sparse import coo_matrix
import time

class MCNFPDHG:
    def __init__(self):
        self.device = torch.device('cuda:0')

    def create_data(self, num_nodes, k, num_commodities, device='cuda:0', seed=1):
        self.N = num_nodes
        self.K = num_commodities
        self.device = torch.device(device)
        dtype = torch.float32

        # adjacency and incidence
        W_adj = utils.create_base_network(self.N, k, seed)
        W_adj = utils.ensure_weak_connectivity(W_adj, seed)
        A_inc_np, p_np = utils.adjacency_to_incidence(W_adj)   # (N, M)
        commodities = utils.create_commodities(W_adj, self.K, 10.0, seed)
        del W_adj
        
        # capacities
        c_np = utils.generate_capacity_constraints(A_inc_np, commodities, 1.0, 5.0, seed=seed)
        self.c = torch.from_numpy(c_np.astype(np.float32)).to(self.device)

        # to sparse coo
        A_inc_coo = coo_matrix(A_inc_np)
        del A_inc_np

        rows = torch.from_numpy(A_inc_coo.row.astype(np.int64))
        cols = torch.from_numpy(A_inc_coo.col.astype(np.int64))
        vals = torch.from_numpy(A_inc_coo.data.astype(np.float32))
        indices = torch.stack([rows, cols], dim=0)
        A_inc_sparse = torch.sparse_coo_tensor(indices, vals, (self.N, A_inc_coo.shape[1]), dtype=dtype)
        self.A_inc = A_inc_sparse.coalesce().to(self.device)
        del rows, cols, vals, indices, A_inc_coo

        # edge cost
        p = torch.from_numpy(p_np.astype(np.float32)).to(self.device)
        del p_np

        self.M = self.A_inc.shape[1]

        # W, d
        w_scale=300.0
        self.W = torch.from_numpy(utils.generate_weight(self.K,dimtype='vector', seed=seed)*w_scale).to(self.device).to(dtype)
        demands = [c[2] for c in commodities]
        self.d = torch.tensor(demands, dtype=dtype, device=self.device)

        # keep p (M) on device
        self.p = p

        # f_mat (K,N) small-ish dense (-1,0,1)
        f_list = []
        for kk in range(self.K):
            f_np = np.zeros(self.N, dtype=np.float32)
            s_idx, t_idx = commodities[kk][0], commodities[kk][1]
            f_np[s_idx] = -1.0
            f_np[t_idx] = 1.0
            f_list.append(torch.from_numpy(f_np))
        self.f_mat = torch.stack(f_list, dim=0).to(self.device)

        return self.N, self.M

    # -------------------------
    # 矩阵-向量接口（稀疏化）
    # -------------------------
    def A_matvec(self, x):
        K, N, M = self.K, self.N, self.M
        X = x.view(K, M)
        X_T = X.t().contiguous()           # M x K
        Y = torch.sparse.mm(self.A_inc, X_T)  # N x K
        return Y.t().reshape(K * N)

    def AT_matvec(self, y):
        K, N, M = self.K, self.N, self.M
        Y = y.view(K, N)
        Y_T = Y.t().contiguous()           # N x K
        A_t = self.A_inc.transpose(0,1).coalesce()  # M x N
        X_T = torch.sparse.mm(A_t, Y_T)    # M x K
        return X_T.t().reshape(K * M)

    def S_matvec(self, X):
        K, N = self.K, self.N
        X_col = X.view(K, 1)
        blocks = self.f_mat * X_col   # K x N
        return blocks.reshape(K * N)

    def ST_matvec(self, Y):
        K, N = self.K, self.N
        Y_mat = Y.view(K, N)
        return torch.sum(self.f_mat * Y_mat, dim=1)
    
    def power_iteration_K_norm(self,A, f_stack, K, M, device='cuda', iters=20):
        """
        Compute ||K||_2 where K = [A  -S].
        A: (N, M) sparse or dense tensor
        f_stack: (K, N) tensor, each row = f_k^T
        K: number of commodities
        M: number of edges
        """
        A_t = A.t()

        # random initial vector v = (x, X)
        x = torch.randn(K, M, device=device)
        X = torch.randn(K, device=device)

        # normalize
        norm = torch.sqrt((x**2).sum() + (X**2).sum())
        x = x / norm
        X = X / norm

        for _ in range(iters):

            # --- Compute u = K v = A x - S X ---
            # A x_k for all k
            Ax = A @ x.transpose(0,1)          # (N, K)
            Ax = Ax.transpose(0,1).contiguous()# (K, N)

            # S X = X[k] * f_k
            SX = X.unsqueeze(1) * f_stack      # (K, N)

            u = Ax - SX                        # (K, N)

            # --- Compute v_new = K^T u ---
            # x-part: A^T u_k
            Atu = A_t @ u.transpose(0,1)       # (M, K)
            Atu = Atu.transpose(0,1).contiguous()

            # X-part:
            # -(f_k^T u_k)
            SXu = -(u * f_stack).sum(dim=1)    # (K,)

            x_new = Atu
            X_new = SXu

            # normalize
            norm = torch.sqrt((x_new**2).sum() + (X_new**2).sum())
            x = x_new / norm
            X = X_new / norm

        return norm  # this is ||K||_2


    # -------------------------
    # PDHG solver with automated tau/sigma tuning and relaxation theta
    # -------------------------
    def pdhg_solve(self,
                x0=None, X0=None,
                tau=None, sigma=None,
                kappa_Y=1.0,
                max_iter=100000, tol=1e-2, device=None,
                verbose=True, overrelax_rho=1.9):
        dev = self.device if device is None else torch.device(device)
        K, M = self.K, self.M
        dtype = torch.float32

        if x0 is None:
            x = torch.zeros(K*M, device=dev, dtype=dtype)
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
        self.A_inc = self.A_inc.to(dev)
           
        # calculate the 2-norm of linear operator in our problem formulation to ensure convergence
        K_norm = self.power_iteration_K_norm(self.A_inc, self.f_mat, K, M)
        eta = 1.0 / K_norm
        pweight = torch.tensor(1.0)
        tau = eta/pweight
        sigma = eta*pweight
        wu_it = 100

        # residual-based dual init
        rY = self.A_matvec(x) - self.S_matvec(X)
        Y = -kappa_Y * rY
        x_prev = x.clone()
        X_prev = X.clone()
        x_bar = x.clone()
        X_bar = X.clone()
        vars = [x,X,Y]

        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        for it in range(max_iter):
            # dual update,explicit,prox is here
            Y_new = Y + sigma * (self.A_matvec(x_bar) - self.S_matvec(X_bar))

            #primal update
            v=x_prev-tau*self.AT_matvec(Y_new)
            #x update as projection
            x_new = self.proj((v.reshape(K,M)-tau*self.p),self.c).squeeze()
            #X update as proximal operator
            X_new = self.f1_prox(X_prev + tau * self.ST_matvec(Y_new), tau)
            
            #overrelaxation
            x_bar=(1+overrelax_rho)*x_new-overrelax_rho*x_prev
            X_bar=(1+overrelax_rho)*X_new-overrelax_rho*X_prev

            # residuals
            r_primal = torch.norm(self.A_matvec(x_bar) - self.S_matvec(X_bar))
            r_dual = torch.norm(x_new - x_prev)/tau + torch.norm(X_new - X_prev)/tau

            if (r_primal.item() < tol) and (r_dual.item() < tol):
                if verbose:
                    print(f'Converged at iter {it}, r_p={r_primal:.2e}, r_d={r_dual:.2e}')
                break

            if (it == max_iter - 1):
                print(f'Max iterations reached, r_p={r_primal:.2e}, r_d={r_dual:.2e}')

            if verbose and (it % 500 == 0):
                print(f'Iter {it:6d} | r_p={r_primal:.2e} | r_d={r_dual:.2e}')
                #print(f'x:\n{x}')
                #print(f'X:\n{X}')
            
            if it%wu_it == 0:
                tau, sigma, pweight = self.weight_update(x_new,X_new,vars[0],vars[1],Y_new,vars[2],pweight,eta)
            
            vars = [x_new,X_new,Y_new]
            
            #shift iteration
            x_prev,X_prev=x_new,X_new
            Y=Y_new

        if dev.type == 'cuda':
            torch.cuda.synchronize()
        if verbose:
            print('pdhg total time:', time.time()-t0)
        return x_new, X_new, Y_new

    def weight_update(self, x, X, x_prev, X_prev, Y, Y_prev,pweight, eta, eps_zero=1e-5):
        # use 2 norms of primal/dual changes to adapt pweight
        del_x = torch.norm(x - x_prev)
        del_X = torch.norm(X - X_prev)
        del_Y = torch.norm(Y - Y_prev)
        # primal/dual change magnitude (combine x and X)
        del_primal = torch.pow((del_x + del_X) / 2.0,0.5)
        del_dual = torch.pow(del_Y,0.5)
        # smooth parameter theta
        theta=0.5
        if (del_primal > eps_zero) and (del_dual > eps_zero):
            pweight = torch.exp(theta * torch.log(del_dual / del_primal) + theta * torch.log(pweight))
        tau = eta / pweight
        sigma = eta * pweight
        return tau, sigma, pweight
    
    # -------------------------
    # prox functions
    # -------------------------
    def f1_prox(self, X_tilde, tau):
        return (X_tilde+2.0*tau*self.W*self.d) / (1.0+2.0*tau*self.W)
    
    def proj(self,U, c):
        """
        U: (K,M) tensor, unconstrained flows
        c: (M,) tensor, per-edge capacities
        Return: X: (KM) projected tensor
        """
        device = U.device
        K, M = self.K,self.M

        # Step 0: initial clipping
        U_clipped = torch.clamp(U, min=0)

        # quick exit: columns with sum <= c do not need projection
        col_sum = U_clipped.sum(dim=0)  # (M,)
        no_proj_mask = col_sum <= c  # (M,)

        # initialize result
        X = U_clipped.clone()

        # need projection only where sum > c
        if no_proj_mask.all():
            return X.reshape(K*M).squeeze()  # nothing to do

        # extract columns needing projection
        mask = ~no_proj_mask  # (M,)
        U_proj = U[:, mask]   # (K, M_active)
        c_proj = c[mask]      # (M_active,)

        # K-by-M_active
        # Step 1: sort each column descending
        U_sorted, _ = torch.sort(U_proj, dim=0, descending=True)

        # Step 2: cumulative sums
        S = U_sorted.cumsum(dim=0)  # (K, M_active)

        # Step 3: all possible tau candidates
        j = torch.arange(1, K+1, device=device, dtype=U.dtype).view(K, 1)  # (K,1)
        tau_candidates = (S - c_proj.unsqueeze(0)) / j  # (K, M_active)

        # Step 4: find j* = largest j with U_sorted[j] > tau_candidates[j]
        cond = U_sorted > tau_candidates  # (K, M_active)
        # find last True along dim=0
        j_star = cond.float().argmax(dim=0)  # WRONG if argmax returns 0 for all-False
        # we want last True: so reverse and compute index
        cond_rev = cond.flip(0)
        idx_rev = cond_rev.float().argmax(dim=0)
        j_star = K - 1 - idx_rev  # (M_active,)

        # Step 5: gather tau for each column
        tau = tau_candidates[j_star, torch.arange(tau_candidates.shape[1], device=device)]

        # Step 6: final projection
        X_proj = torch.clamp(U_proj - tau.unsqueeze(0), min=0)

        # put back into result
        X[:, mask] = X_proj
        return X.reshape(K*M).squeeze()

    def make_initials(self, device=None):
        dev = torch.device(device if device is not None else self.device)
        x0 = torch.zeros(self.K * self.M, device=dev, dtype=torch.float32)
        X0 = torch.zeros(self.K, device=dev, dtype=torch.float32)
        return x0, X0
