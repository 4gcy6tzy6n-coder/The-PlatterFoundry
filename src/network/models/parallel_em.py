#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""并行化EM算法实现

提供EM算法的并行化版本，支持多进程并行计算，提升大规模数据处理性能。
"""

import numpy as np
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from multiprocessing import Pool, cpu_count
import warnings

# 忽略多进程相关警告
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")


@dataclass
class ParallelEMConfig:
    """并行EM配置"""
    n_jobs: int = None  # 并行工作进程数，None表示使用CPU核心数
    chunk_size: int = 1000  # 每个进程处理的数据块大小
    max_iterations: int = 100  # 最大迭代次数
    convergence_threshold: float = 1e-6  # 收敛阈值
    verbose: bool = False  # 是否输出详细信息


class ParallelEMEstimator:
    """
    并行化EM算法估计器
    
    特性：
    1. 多进程并行计算
    2. 自动数据分块
    3. 进度监控
    4. 内存优化
    """
    
    def __init__(self, config: Optional[ParallelEMConfig] = None):
        self.config = config or ParallelEMConfig()
        
        # 自动设置并行度
        if self.config.n_jobs is None:
            self.config.n_jobs = max(1, cpu_count() - 1)  # 保留一个核心
        
        self.logger = logging.getLogger("ParallelEMEstimator")
        
        # 统计信息
        self.iterations = 0
        self.convergence_time = 0.0
        self.total_time = 0.0
    
    def fit(self, 
            loss_sequence: List[int], 
            n_states: int = 3,
            initial_params: Optional[dict] = None) -> Tuple[dict, dict]:
        """
        并行EM算法拟合
        
        Args:
            loss_sequence: 丢包序列 (0/1)
            n_states: 状态数 (默认3: GOOD/BAD/DEAD)
            initial_params: 初始参数
            
        Returns:
            Tuple[dict, dict]: (估计的参数, 诊断信息)
        """
        import time
        start_time = time.time()
        
        n = len(loss_sequence)
        if n < 100:
            self.logger.warning("数据量过小，建议使用串行EM算法")
            return self._fit_serial(loss_sequence, n_states, initial_params)
        
        # 数据分块
        chunks = self._split_data(loss_sequence, n_states)
        
        if self.config.verbose:
            self.logger.info(f"开始并行EM算法，数据量={n}，分块数={len(chunks)}，"
                           f"并行度={self.config.n_jobs}")
        
        # 并行E步
        e_results = self._parallel_e_step(chunks, n_states, initial_params)
        
        # M步（串行，因为需要全局统计）
        m_result = self._m_step(e_results, n_states)
        
        # 迭代优化
        params = m_result
        log_likelihoods = [m_result['log_likelihood']]
        
        for iteration in range(1, self.config.max_iterations):
            if self.config.verbose and iteration % 10 == 0:
                self.logger.info(f"迭代 {iteration}/{self.config.max_iterations}")
            
            # 并行E步
            e_results = self._parallel_e_step(chunks, n_states, params)
            
            # M步
            new_params = self._m_step(e_results, n_states)
            
            # 检查收敛
            log_likelihoods.append(new_params['log_likelihood'])
            if self._check_convergence(log_likelihoods, iteration):
                self.logger.info(f"EM算法在 {iteration} 次迭代后收敛")
                self.iterations = iteration
                break
            
            params = new_params
        
        self.total_time = time.time() - start_time
        
        # 准备诊断信息
        diagnostics = {
            'iterations': self.iterations,
            'converged': self.iterations < self.config.max_iterations,
            'log_likelihoods': log_likelihoods,
            'total_time': self.total_time,
            'n_jobs': self.config.n_jobs,
            'data_size': n
        }
        
        return params, diagnostics
    
    def _split_data(self, loss_sequence: List[int], n_states: int) -> List[List[int]]:
        """将数据分割成多个块"""
        n = len(loss_sequence)
        n_chunks = max(1, n // self.config.chunk_size)
        
        if n_chunks <= 1:
            return [loss_sequence]
        
        chunk_size = n // n_chunks
        chunks = []
        
        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < n_chunks - 1 else n
            chunks.append(loss_sequence[start:end])
        
        return chunks
    
    def _parallel_e_step(self, 
                        chunks: List[List[int]], 
                        n_states: int,
                        params: Optional[dict] = None) -> List[dict]:
        """并行E步计算"""
        if len(chunks) == 1:
            # 单块数据，直接计算
            return [self._e_step_single(chunks[0], n_states, params)]
        
        # 多进程并行
        with Pool(processes=self.config.n_jobs) as pool:
            # 准备参数
            chunk_params = [(chunk, n_states, params) for chunk in chunks]
            
            # 并行执行E步
            results = pool.starmap(self._e_step_single, chunk_params)
        
        return results
    
    def _e_step_single(self, 
                      chunk: List[int], 
                      n_states: int,
                      params: Optional[dict] = None) -> dict:
        """单个数据块的E步计算"""
        if params is None:
            # 初始化参数
            params = self._initialize_params(n_states)
        
        n = len(chunk)
        if n == 0:
            return {
                'gamma': np.zeros((0, n_states)),
                'xi': np.zeros((0, n_states, n_states)),
                'log_likelihood': 0.0,
                'n_samples': 0
            }
        
        # 前向后向算法
        alpha, c = self._forward(chunk, params)
        beta = self._backward(chunk, params, c)
        
        # 计算gamma和xi
        gamma = alpha * beta
        gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-10)
        
        xi = np.zeros((n - 1, n_states, n_states))
        for t in range(n - 1):
            for i in range(n_states):
                for j in range(n_states):
                    xi[t, i, j] = (
                        alpha[t, i] *
                        params['transition'][i, j] *
                        self._emission_prob(chunk[t + 1], params['emission'][j]) *
                        beta[t + 1, j] /
                        (c[t + 1] + 1e-10)
                    )
        
        # 对数似然
        log_likelihood = np.sum(np.log(c + 1e-10))
        
        return {
            'gamma': gamma,
            'xi': xi,
            'log_likelihood': log_likelihood,
            'n_samples': n
        }
    
    def _forward(self, 
                chunk: List[int], 
                params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """前向算法"""
        n = len(chunk)
        n_states = len(params['initial'])
        
        alpha = np.zeros((n, n_states))
        c = np.zeros(n)
        
        # 初始化
        for j in range(n_states):
            alpha[0, j] = params['initial'][j] * self._emission_prob(chunk[0], params['emission'][j])
        
        c[0] = np.sum(alpha[0])
        if c[0] > 0:
            alpha[0] /= c[0]
        
        # 迭代
        for t in range(1, n):
            for j in range(n_states):
                alpha[t, j] = np.sum(alpha[t - 1] * params['transition'][:, j]) * \
                             self._emission_prob(chunk[t], params['emission'][j])
            
            c[t] = np.sum(alpha[t])
            if c[t] > 0:
                alpha[t] /= c[t]
        
        return alpha, c
    
    def _backward(self, 
                 chunk: List[int], 
                 params: dict, 
                 c: np.ndarray) -> np.ndarray:
        """后向算法"""
        n = len(chunk)
        n_states = len(params['initial'])
        
        beta = np.zeros((n, n_states))
        beta[n - 1] = 1.0
        
        for t in range(n - 2, -1, -1):
            for i in range(n_states):
                beta[t, i] = np.sum(
                    params['transition'][i, :] *
                    self._emission_prob(chunk[t + 1], params['emission'][:]) *
                    beta[t + 1, :]
                )
            beta[t] /= (c[t + 1] + 1e-10)
        
        return beta
    
    def _m_step(self, e_results: List[dict], n_states: int) -> dict:
        """M步计算"""
        # 聚合统计量
        total_gamma = None
        total_xi = None
        total_log_likelihood = 0.0
        total_samples = 0
        
        for result in e_results:
            if result['n_samples'] == 0:
                continue
            
            if total_gamma is None:
                total_gamma = result['gamma'].copy()
                total_xi = result['xi'].copy()
            else:
                total_gamma = np.vstack([total_gamma, result['gamma']])
                total_xi = np.vstack([total_xi, result['xi']])
            
            total_log_likelihood += result['log_likelihood']
            total_samples += result['n_samples']
        
        if total_samples == 0:
            return self._initialize_params(n_states)
        
        # 更新初始状态分布
        initial = total_gamma[0] / np.sum(total_gamma[0])
        
        # 更新转移概率
        transition = np.zeros((n_states, n_states))
        for i in range(n_states):
            for j in range(n_states):
                if total_xi.shape[0] > 0:
                    transition[i, j] = np.sum(total_xi[:, i, j]) / np.sum(total_gamma[:-1, i])
                else:
                    transition[i, j] = 1.0 / n_states
        
        # 归一化转移矩阵
        transition = transition / transition.sum(axis=1, keepdims=True)
        
        # 更新发射概率
        emission = np.zeros(n_states)
        for j in range(n_states):
            if np.sum(total_gamma[:, j]) > 0:
                emission[j] = np.sum(total_gamma[:, j] * np.array([1 if x == 1 else 0 for x in chunk])) / np.sum(total_gamma[:, j])
            else:
                emission[j] = 0.5
        
        return {
            'initial': initial,
            'transition': transition,
            'emission': emission,
            'log_likelihood': total_log_likelihood,
            'n_samples': total_samples
        }
    
    def _initialize_params(self, n_states: int) -> dict:
        """初始化参数"""
        if n_states == 3:
            # GOOD/BAD/DEAD
            initial = np.array([0.8, 0.15, 0.05])
            transition = np.array([
                [0.9, 0.08, 0.02],
                [0.2, 0.7, 0.1],
                [0.1, 0.3, 0.6]
            ])
            emission = np.array([0.01, 0.4, 0.95])
        else:
            # 随机初始化
            initial = np.random.dirichlet(np.ones(n_states))
            transition = np.random.dirichlet(np.ones(n_states), size=n_states)
            emission = np.random.uniform(0, 1, n_states)
        
        return {
            'initial': initial,
            'transition': transition,
            'emission': emission,
            'log_likelihood': 0.0,
            'n_samples': 0
        }
    
    def _emission_prob(self, y: int, p: float) -> float:
        """发射概率"""
        return p if y == 1 else (1 - p)
    
    def _check_convergence(self, log_likelihoods: List[float], iteration: int) -> bool:
        """检查收敛"""
        if len(log_likelihoods) < 2:
            return False
        
        current = log_likelihoods[-1]
        previous = log_likelihoods[-2]
        
        if abs(current - previous) < self.config.convergence_threshold:
            return True
        
        return False
    
    def _fit_serial(self, 
                   loss_sequence: List[int], 
                   n_states: int,
                   initial_params: Optional[dict]) -> Tuple[dict, dict]:
        """串行EM算法（用于小数据）"""
        from .improved_loss_model import ImprovedLossModel
        from .improved_loss_model import LossModelConfig
        
        # 使用现有的ImprovedLossModel
        model = ImprovedLossModel()
        config, metrics = model.em_parameter_estimation(
            loss_sequence,
            max_iterations=self.config.max_iterations,
            convergence_threshold=self.config.convergence_threshold
        )
        
        # 转换为标准格式
        params = {
            'initial': np.array([0.8, 0.15, 0.05]),  # 简化
            'transition': np.array([
                [0.9, 0.08, 0.02],
                [0.2, 0.7, 0.1],
                [0.1, 0.3, 0.6]
            ]),
            'emission': np.array([config.loss_rate_good, config.loss_rate_bad, config.loss_rate_dead]),
            'log_likelihood': metrics.log_likelihood,
            'n_samples': len(loss_sequence)
        }
        
        diagnostics = {
            'iterations': 1,
            'converged': metrics.convergence,
            'log_likelihoods': [metrics.log_likelihood],
            'total_time': 0.0,
            'n_jobs': 1,
            'data_size': len(loss_sequence)
        }
        
        return params, diagnostics


def parallel_em_fit(loss_sequence: List[int], 
                   n_states: int = 3,
                   config: Optional[ParallelEMConfig] = None) -> Tuple[dict, dict]:
    """
    并行EM算法快速接口
    
    Args:
        loss_sequence: 丢包序列 (0/1)
        n_states: 状态数
        config: 并行配置
        
    Returns:
        Tuple[dict, dict]: (参数, 诊断信息)
    """
    estimator = ParallelEMEstimator(config)
    return estimator.fit(loss_sequence, n_states)


# 使用示例
if __name__ == "__main__":
    import time
    
    # 生成测试数据
    np.random.seed(42)
    n = 50000
    loss_sequence = np.random.binomial(1, 0.15, n).tolist()
    
    print(f"测试数据: {n} 个样本")
    
    # 并行EM算法
    config = ParallelEMConfig(
        n_jobs=4,
        chunk_size=5000,
        max_iterations=50,
        verbose=True
    )
    
    start_time = time.time()
    params, diagnostics = parallel_em_fit(loss_sequence, n_states=3, config=config)
    elapsed = time.time() - start_time
    
    print(f"\n并行EM算法结果:")
    print(f"  运行时间: {elapsed:.4f}秒")
    print(f"  迭代次数: {diagnostics['iterations']}")
    print(f"  收敛: {diagnostics['converged']}")
    print(f"  对数似然: {params['log_likelihood']:.2f}")
    print(f"  初始状态: {params['initial']}")
    print(f"  发射概率: {params['emission']}")
    print(f"  转移矩阵:\n{params['transition']}")