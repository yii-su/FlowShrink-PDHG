from functools import wraps
import time

import torch

# ============================================================
# 1. 性能测试装饰器
# ============================================================

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
        start_event.record() # pyright: ignore[reportCallIssue]
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 结束计时
        end_event.record() # pyright: ignore[reportCallIssue]
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


def cpu_timer(func):
    """
    CPU 性能测试装饰器
    测量：CPU执行时间、迭代次数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 开始计时
        start_time = time.perf_counter()
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 结束计时
        end_time = time.perf_counter()
        
        # 计算执行时间（秒转毫秒）
        cpu_time = (end_time - start_time) * 1000
        
        # 打印性能报告
        print("\n" + "="*60)
        print(f"函数: {func.__name__}")
        print("="*60)
        print(f"CPU 执行时间: {cpu_time:.2f} ms ({cpu_time/1000:.4f} s)")
        
        # 如果返回值包含迭代次数信息
        if isinstance(result, tuple) and len(result) == 3:
            D, P, iterations = result
            print(f"迭代次数: {iterations}")
            print(f"平均每轮时间: {cpu_time/iterations:.2f} ms")
            print("="*60 + "\n")
            return D, P, iterations
        else:
            print("="*60 + "\n")
            return result
    
    return wrapper
