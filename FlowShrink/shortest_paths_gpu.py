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
        start_event.record()
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 结束计时
        end_event.record()
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