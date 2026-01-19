#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performance Monitor - 性能监控模块

提供性能监控、内存管理和计算优化功能
"""

import time
import psutil
import threading
import os
from collections import deque
from typing import Dict, Any, List, Callable, Optional
from contextlib import contextmanager
import logging


class PerformanceMonitor:
    """
    性能监控器

    监控计算性能、内存使用和系统资源
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.execution_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.cpu_usage = deque(maxlen=window_size)

        self.logger = logging.getLogger("PerformanceMonitor")

    @contextmanager
    def monitor_execution(self, operation_name: str = "operation"):
        """
        执行监控上下文管理器

        用法:
        with monitor.monitor_execution("matrix_multiplication"):
            result = np.dot(A, B)
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory

            self.execution_times.append(execution_time)
            self.memory_usage.append(memory_delta)

            self.logger.debug(".4f"
                            f"内存变化: {memory_delta:+.2f}MB")

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.execution_times:
            return {
                'avg_execution_time': 0,
                'max_execution_time': 0,
                'min_execution_time': 0,
                'avg_memory_delta': 0,
                'total_operations': 0
            }

        return {
            'avg_execution_time': sum(self.execution_times) / len(self.execution_times),
            'max_execution_time': max(self.execution_times),
            'min_execution_time': min(self.execution_times),
            'avg_memory_delta': sum(self.memory_usage) / len(self.memory_usage),
            'total_operations': len(self.execution_times),
            'current_memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'cpu_percent': psutil.cpu_percent(interval=0.1)
        }

    def reset(self):
        """重置监控数据"""
        self.execution_times.clear()
        self.memory_usage.clear()
        self.cpu_usage.clear()


class ComputationOptimizer:
    """
    计算优化器

    提供向量化计算、内存管理和并行处理优化
    """

    def __init__(self):
        self.vectorization_enabled = True
        self.memory_pool = {}
        self.cache = {}

    def vectorized_operation(
        self,
        operation: Callable,
        data: List,
        use_cache: bool = True
    ) -> Any:
        """
        向量化操作包装器

        自动选择最优的计算方式
        """
        if not self.vectorization_enabled:
            return [operation(item) for item in data]

        # 检查缓存
        cache_key = hash(str(data) + str(operation))
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]

        try:
            import numpy as np
            # 尝试向量化操作
            vectorized_data = np.array(data)
            result = operation(vectorized_data)

            # 缓存结果
            if use_cache:
                self.cache[cache_key] = result

            return result

        except Exception:
            # 回退到循环方式
            result = [operation(item) for item in data]

            if use_cache:
                self.cache[cache_key] = result

            return result

    def memory_efficient_batch(
        self,
        data: List,
        batch_size: int = 1000,
        operation: Callable = None
    ) -> List:
        """
        内存高效批处理

        将大数据集分批处理，避免内存溢出
        """
        if not data:
            return []

        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            if operation:
                batch_result = self.vectorized_operation(operation, batch, use_cache=False)
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    results.append(batch_result)
            else:
                results.extend(batch)

        return results

    def adaptive_batch_size(
        self,
        data_size: int,
        memory_limit_mb: float = 500
    ) -> int:
        """
        自适应批大小计算

        根据可用内存自动调整批处理大小
        """
        available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
        safe_memory = min(available_memory * 0.8, memory_limit_mb)

        # 估算每个数据项的内存使用（保守估计）
        estimated_item_size = 100  # bytes
        max_batch_size = int(safe_memory * 1024 * 1024 / estimated_item_size)

        # 确保批大小合理
        batch_size = min(max_batch_size, data_size, 10000)
        batch_size = max(batch_size, 1)

        return batch_size


class ResourceManager:
    """
    资源管理器

    管理计算资源、线程池和内存池
    """

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(4, psutil.cpu_count())
        self.active_threads = 0
        self.thread_lock = threading.Lock()

    def parallel_execute(
        self,
        tasks: List[Callable],
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """
        并行执行任务

        控制并发数量，避免资源耗尽
        """
        if max_concurrent is None:
            max_concurrent = self.max_workers

        results = []
        threads = []

        def worker(task, index):
            try:
                result = task()
                results.append((index, result))
            except Exception as e:
                results.append((index, e))
            finally:
                with self.thread_lock:
                    self.active_threads -= 1

        # 启动线程
        for i, task in enumerate(tasks):
            # 等待可用线程
            while True:
                with self.thread_lock:
                    if self.active_threads < max_concurrent:
                        self.active_threads += 1
                        break
                time.sleep(0.01)  # 短暂等待

            thread = threading.Thread(target=worker, args=(task, i))
            thread.daemon = True
            thread.start()
            threads.append(thread)

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 排序结果
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_total': psutil.virtual_memory().total / 1024 / 1024,  # MB
            'memory_available': psutil.virtual_memory().available / 1024 / 1024,  # MB
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': self._get_disk_usage()
        }

    def _get_disk_usage(self) -> Optional[float]:
        """跨平台获取磁盘使用率"""
        try:
            # Windows使用C:驱动器，Linux/macOS使用根目录
            if os.name == 'nt':  # Windows
                disk_usage = psutil.disk_usage('C:')
            else:  # Linux/Unix/macOS
                disk_usage = psutil.disk_usage(os.path.sep)
            return disk_usage.percent
        except Exception:
            # 如果无法获取磁盘信息，返回None
            return None


# 全局实例
performance_monitor = PerformanceMonitor()
computation_optimizer = ComputationOptimizer()
resource_manager = ResourceManager()


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    return performance_monitor


def get_computation_optimizer() -> ComputationOptimizer:
    """获取全局计算优化器"""
    return computation_optimizer


def get_resource_manager() -> ResourceManager:
    """获取全局资源管理器"""
    return resource_manager


