import torch
from functools import wraps

def gpu_timer(func):
    """
    GPU 性能测试装饰器
    测量：GPU执行时间、迭代次数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 确保之前的 GPU 操作完成
        torch.cuda.synchronize()
        
        # 创建 CUDA 事件用于精确计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # 开始计时
        start_event.record(torch.cuda.current_stream())
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 结束计时
        end_event.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        
        # 计算执行时间（毫秒）
        gpu_time = start_event.elapsed_time(end_event)
        
        # 打印性能报告
        print("\n" + "="*60)
        print(f"函数: {func.__name__}")
        print("="*60)
        print(f"GPU 执行时间: {gpu_time:.2f} ms ({gpu_time/1000:.4f} s)")
        
        # 如果返回值包含迭代次数信息
        if isinstance(result, tuple) and len(result) == 3:
            D, P, iterations = result
            print(f"迭代次数: {iterations}")
            print(f"平均每轮时间: {gpu_time/iterations:.2f} ms")
            print("="*60 + "\n")
            return D, P, iterations
        else:
            print("="*60 + "\n")
            return result
    
    return wrapper

# GPU-based min-plus matrix multiplication
def min_plus_matmul_gpu(W, D):
    """
    GPU 加速的 (min, +) 半环矩阵乘法
    计算: D_new[i, j] = min_k { W[i, k] + D[k, j] }
    同时更新前驱矩阵 P
    
    参数:
        W: torch.Tensor, shape (N, N), 权重矩阵
        D: torch.Tensor, shape (N, N), 当前距离矩阵
        P_old: torch.Tensor, shape (N, N), 当前前驱矩阵
    
    返回:
        D_new: torch.Tensor, shape (N, N), 新的距离矩阵
        P_new: torch.Tensor, shape (N, N), 新的前驱矩阵
    """
    
    # 广播计算: W[i, k] + D[k, j]
    # W.unsqueeze(1): (N, 1, N) - 对应 W[i, :, k]
    # D.unsqueeze(0): (1, N, N) - 对应 D[:, k, j]
    # 相加结果: (N, N, N) - [i, j, k] = W[i, k] + D[k, j]
    path_costs = W.unsqueeze(1) + D.unsqueeze(0)  # (N, N, N)
    
    # 沿着 k 维度(dim=2)取最小值
    # D_new[i, j] = min_k { path_costs[i, j, k] }
    # best_k[i, j] = argmin_k { path_costs[i, j, k] }
    D_new, best_k = torch.min(path_costs, dim=2)
    
    # best_k 就是前驱节点矩阵
    P_new = best_k
    
    return D_new, P_new


# GPU 版本的 APSP 核心算法
@gpu_timer
def apsp_gpu(W, device='cuda:0', max_iterations=None, convergence_check=True,dtype=torch.float64):
    """
    GPU 加速的全源最短路径算法 (APSP)
    
    参数:
        W: torch.Tensor, shape (N, N)
           邻接权重矩阵，0 表示无边
        device: str, GPU 设备
        max_iterations: int or None, 最大迭代次数，None 时为 N-1
        convergence_check: bool, 是否检查收敛（可提前终止）
    
    返回:
        D: torch.Tensor, shape (N, N), 最短距离矩阵
        P: torch.Tensor, shape (N, N), 前驱节点矩阵
        iterations: int, 实际迭代次数
    """
    # 确保输入在正确的设备上
    W = W.to(device).to(dtype)
    N = W.shape[0]
    
    # 设置最大迭代次数
    if max_iterations is None:
        max_iterations = N - 1
    
    # 1. 准备权重矩阵 W_prime
    # 将 0 替换为 inf（表示无边），对角线设为 0
    W_prime = torch.where(W == 0, torch.tensor(float('inf'), dtype=dtype, device=device), W)
    W_prime.fill_diagonal_(0)
    
    # 2. 初始化距离矩阵 D
    D = W_prime.clone()
    
    # 3. 初始化前驱矩阵 P（使用 long 类型存储节点索引）
    P = torch.full((N, N), -1, dtype=torch.long, device=device)
    
    # 初始化前驱矩阵：有直接边的位置，前驱为源节点
    mask = W_prime < float('inf')
    P[mask] = torch.arange(N, device=device).unsqueeze(0).expand(N, N)[mask]
    
    # 4. 迭代执行 SpMM
    actual_iterations = 0
    
    for _ in range(max_iterations):
        D_old = D.clone()
        
        D_candidates, P_candidates = min_plus_matmul_gpu(W_prime, D_old)
        
        # 更新：只有当新路径更短时才更新
        update_mask = D_candidates < D
        D = torch.where(update_mask, D_candidates, D)
        P = torch.where(update_mask, P_candidates, P)
        
        actual_iterations += 1
        
        # 5. 收敛检查
        if convergence_check and torch.equal(D, D_old):
            break
    
    return D, P, actual_iterations

@gpu_timer
def parallel_bellman_ford_gpu(W, device='cuda:0', max_iterations=None, convergence_check=True, dtype=torch.float64, block_size=128):
    """
    基于分块矩阵运算的并行 Bellman-Ford 算法 (GPU)
    
    优势：
    1. 解决了 N=1000 时原始 APSP 的显存爆炸问题 (OOM)。
    2. 使用分块计算 (Blocking) 平衡计算速度与显存占用。
    
    参数:
        W: torch.Tensor, shape (N, N), 邻接权重矩阵 (0 表示无边)
        device: str, GPU 设备
        max_iterations: int, 最大迭代次数
        convergence_check: bool, 是否提前终止
        dtype: 数据类型
        block_size: int, 分块大小，建议 128-512，越小显存占用越低
    
    返回:
        D: (N, N) 最短距离矩阵
        P: (N, N) 前驱节点矩阵
        iterations: 实际迭代次数
    """
    W = W.to(device).to(dtype)
    N = W.shape[0]
    
    if max_iterations is None:
        max_iterations = N - 1
    # 1. 预处理权重矩阵 W_prime
    # 将 0 替换为 inf，对角线保持为 0 (或 inf，取决于定义，这里保持标准 APSP 定义)
    # 注意：为了矩阵加法逻辑，我们希望 W[i, j] 代表 i->j 的边权
    inf_val = float('inf')
    W_prime = torch.where(W == 0, torch.tensor(inf_val, dtype=dtype, device=device), W)
    W_prime.fill_diagonal_(0)
    # 2. 初始化距离矩阵 D
    # 初始状态：D[i, j] = W_prime[i, j]
    D = W_prime.clone()
    
    # 3. 初始化前驱矩阵 P
    # P[i, j] 表示从 i 到 j 的路径中，j 的前驱
    # 初始：如果有边 i->j，则前驱为 i，否则为 -1
    P = torch.full((N, N), -1, dtype=torch.long, device=device)
    mask_init = (W_prime != inf_val)
    # src_indices: 行号 i
    src_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, N)
    P[mask_init] = src_indices[mask_init]
    # 对角线前驱设为自己 (可选)
    P.fill_diagonal_(-1) 
    actual_iterations = 0
    
    # 4. 迭代松弛
    # D_new[i, j] = min_k ( D[i, k] + W[k, j] )
    for _ in range(max_iterations):
        D_old = D.clone()
        has_update = False
        
        # --- 分块核心优化 ---
        # 为了避免生成 (N, N, N) 的巨大张量，我们按列(dst)分块处理
        for col_start in range(0, N, block_size):
            col_end = min(col_start + block_size, N)
            
            # 取出当前块对应的权重列 W[:, col_start:col_end] -> Shape (N, Block)
            # W_chunk[k, j] 表示 k -> j 的边权
            W_chunk = W_prime[:, col_start:col_end] 
            
            # 计算路径开销： D[i, k] + W[k, j]
            # 我们需要广播加法得到 (N, N, Block)
            # D.unsqueeze(2): (N, N, 1) -> 对应 D[i, k]
            # W_chunk.unsqueeze(0): (1, N, Block) -> 对应 W[k, j]
            
            # 注意显存： (1000, 1000, 128) * 8字节 ≈ 1GB (安全)
            path_costs = D.unsqueeze(2) + W_chunk.unsqueeze(0)
            
            # 在 k 维度 (dim=1) 找最小值
            # min_vals: (N, Block), min_indices: (N, Block)
            # min_indices 正好就是中间节点 k，也就是新的前驱
            chunk_min_dist, chunk_predecessors = torch.min(path_costs, dim=1)
            
            # 更新 D 和 P 的对应块
            # 只有当新路径更短时才更新
            current_dist_chunk = D[:, col_start:col_end]
            update_mask = chunk_min_dist < current_dist_chunk
            
            if convergence_check:
                if update_mask.any():
                    has_update = True
            
            # 原地更新 D 的块
            D[:, col_start:col_end] = torch.where(update_mask, chunk_min_dist, current_dist_chunk)
            
            # 更新 P 的块
            # 注意：chunk_predecessors 就是中间点 k。
            # 这里的逻辑是：如果 D[i, k] + W[k, j] 更短，那么 j 的新前驱就是 k 到 j 这条边的起点，即 k。
            # 所以直接用 chunk_predecessors 是对的。
            current_P_chunk = P[:, col_start:col_end]
            P[:, col_start:col_end] = torch.where(update_mask, chunk_predecessors, current_P_chunk)
        actual_iterations += 1
        
        # 5. 收敛检查
        if convergence_check and not has_update:
            break
            
    return D, P, actual_iterations
