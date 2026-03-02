# perf_tools.py
import functools
import time
# import cupy as cp
import torch
from typing import Callable, Any


def timed_ns(repeat: int = 1, warmup: int = 0) -> Callable:
    """纳秒级GPU性能测试装饰器（支持异步操作）"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            device = torch.cuda.current_device() if torch.cuda.is_available() else None
            

            # 预热阶段（含同步）
            for _ in range(warmup):
                func(*args, **kwargs)
                if device is not None:
                    torch.cuda.synchronize()

            # 正式测试
            durations_ns = []
            result = None
            for _ in range(repeat):
                if device is not None:
                    torch.cuda.synchronize()  # 确保之前的 GPU 操作完成
                start = time.perf_counter_ns()
                result = func(*args, **kwargs)
                if device is not None:
                    torch.cuda.synchronize()  # 确保GPU操作完成
                end = time.perf_counter_ns()
                durations_ns.append(end - start)

            # # 打印结果
            # _print_perf_result(func.__name__, durations_ns, repeat, warmup)
            # 简化打印
            _print_perf_result_simple(func.__name__, durations_ns, repeat, warmup)
            return result
        return wrapper
    return decorator


def _print_perf_result(name: str, durations: list[int], repeat: int, warmup: int) -> None:
    """格式化输出性能测试结果"""
    avg_ns = sum(durations) / repeat
    print(f"\n[Perf] {name} (ns)")
    print(f"  Runs: {repeat} (warmup {warmup})")
    # print(f"  Min: {min(durations):,}ns | Avg: {avg_ns:,.0f}ns | Max: {max(durations):,}ns")
    # print(f"  Min: {min(durations) / 1_000_000:.0f} ms | Avg: {avg_ns / 1_000_000:.0f} ms | Max: {max(durations) / 1_000_000:.0f} ms")
    # 使用微妙输出，四舍五入到整数
    print(f"  Min: {min(durations) / 1_000:.0f} µs | Avg: {avg_ns / 1_000:.0f} µs | Max: {max(durations) / 1_000:.0f} µs")
    print(f"  Total: {sum(durations):,}ns")
    
# 简化打印    
def _print_perf_result_simple(name: str, durations: list[int], repeat: int, warmup: int) -> None:
    """格式化输出性能测试结果"""
    print(f"\n[Perf] {name}")
    # 使用毫秒输出，四舍五入到整数
    print(f"  Total: {sum(durations) / 1_000_000:.0f} ms")


def benchmark(func: Callable, *args, repeat: int = 5, **kwargs) -> dict:
    """独立基准测试函数

    Args:
        func: 待测试函数
        *args: 位置参数
        repeat: 测试次数 (default: 5)
       **​kwargs: 关键字参数

    Returns:
        包含测试结果的字典
    """
    durations = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        func(*args, **kwargs)
        durations.append(time.perf_counter_ns() - start)

    return {
        "name": func.__name__,
        "min": min(durations),
        "avg": sum(durations) / repeat,
        "max": max(durations),
        "unit": "ns"
    }
