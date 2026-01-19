#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Improved Loss Model - 改进的丢包模型

创新点：
1. 三状态马尔可夫模型：GOOD, BAD, DEAD 三个状态
2. 自适应参数学习：从历史数据学习转移概率
3. EM算法参数估计：从丢包序列自动学习最优模型参数
4. 多时间尺度分析：毫秒级到秒级的丢包模式
5. 相关性建模：考虑丢包的时间相关性
6. 预测性丢包概率：基于模型的预测能力分析
7. 状态驻留时间分布：量化状态持续时间特性

这解决了传统Gilbert-Elliott模型只有两个状态、参数固定的问题。
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from scipy import stats


class LossState:
    """丢包状态枚举"""
    GOOD = 0
    BAD = 1
    DEAD = 2


@dataclass
class TransitionMatrix:
    """转移概率矩阵"""
    p_good_to_good: float = 0.95
    p_good_to_bad: float = 0.04
    p_good_to_dead: float = 0.01
    p_bad_to_good: float = 0.30
    p_bad_to_bad: float = 0.60
    p_bad_to_dead: float = 0.10
    p_dead_to_good: float = 0.20
    p_dead_to_bad: float = 0.50
    p_dead_to_dead: float = 0.30
    
    def to_matrix(self) -> np.ndarray:
        """转换为numpy矩阵"""
        return np.array([
            [self.p_good_to_good, self.p_good_to_bad, self.p_good_to_dead],
            [self.p_bad_to_good, self.p_bad_to_bad, self.p_bad_to_dead],
            [self.p_dead_to_good, self.p_dead_to_bad, self.p_dead_to_dead]
        ])
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'TransitionMatrix':
        """从矩阵创建"""
        return cls(
            p_good_to_good=matrix[0, 0],
            p_good_to_bad=matrix[0, 1],
            p_good_to_dead=matrix[0, 2],
            p_bad_to_good=matrix[1, 0],
            p_bad_to_bad=matrix[1, 1],
            p_bad_to_dead=matrix[1, 2],
            p_dead_to_good=matrix[2, 0],
            p_dead_to_bad=matrix[2, 1],
            p_dead_to_dead=matrix[2, 2]
        )
    
    def validate(self) -> bool:
        """验证转移概率矩阵的有效性"""
        matrix = self.to_matrix()
        row_sums = np.sum(matrix, axis=1)
        return all(abs(rs - 1.0) < 1e-6 for rs in row_sums)


@dataclass
class LossModelConfig:
    """丢包模型配置"""
    loss_rate_good: float = 0.01
    loss_rate_bad: float = 0.40
    loss_rate_dead: float = 0.95
    transition_matrix: TransitionMatrix = field(default_factory=TransitionMatrix)
    initial_state_distribution: Tuple[float, float, float] = (0.85, 0.14, 0.01)
    min_burst_length: int = 1
    max_burst_length: int = 10


@dataclass
class BurstMetrics:
    """突发性指标 - 增强版"""
    burst_index: float = 0.0
    run_test_statistic: float = 0.0
    entropy: float = 0.0
    run_count: int = 0
    avg_run_length: float = 0.0
    loss_probability: float = 0.0
    burst_length_distribution: Dict[str, float] = field(default_factory=dict)

    # 增强指标
    burst_intensity: float = 0.0  # 突发强度
    periodicity_index: float = 0.0  # 周期性指标
    autocorrelation_coefficient: float = 0.0  # 自相关系数
    fractal_dimension: float = 0.0  # 分形维度
    burst_pattern_complexity: float = 0.0  # 突发模式复杂度


@dataclass
class ModelFittingMetrics:
    """模型拟合指标"""
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    num_params: int = 0
    convergence: bool = False


class ImprovedLossModel:
    """
    改进的三状态丢包模型
    
    核心创新：
    1. 三状态设计：更精确地模拟网络丢包行为
    2. 自适应学习：从观测数据学习模型参数
    3. EM参数估计：自动学习最优转移概率
    4. 突发检测：自动识别丢包突发
    5. 状态持久化：记录状态历史用于分析
    6. 预测性分析：基于模型的预测能力评估
    """
    
    def __init__(self, config: Optional[LossModelConfig] = None):
        self.config = config or LossModelConfig()
        self.transition_matrix = self.config.transition_matrix.to_matrix()
        self.initial_state = np.array(self.config.initial_state_distribution)
        
        self.state_history = deque(maxlen=10000)
        self.loss_history = deque(maxlen=10000)
        
        self.current_state = self._sample_initial_state()
        self.state_duration = 0
        self.current_burst_id = 0
        
        self.total_packets = 0
        self.total_losses = 0
        self.burst_count = 0
        self.burst_lengths: List[int] = []
        
        self.state_durations: List[int] = []
        
        self.logger = logging.getLogger("ImprovedLossModel")
    
    def _sample_initial_state(self) -> int:
        """采样初始状态"""
        rng = np.random.random()
        cumsum = np.cumsum(self.initial_state)
        for i, threshold in enumerate(cumsum):
            if rng < threshold:
                return i
        return LossState.GOOD
    
    def _update_state(self):
        """更新状态（马尔可夫转移）"""
        if np.random.random() < self.transition_matrix[self.current_state, self.current_state]:
            self.state_duration += 1
        else:
            probs = self.transition_matrix[self.current_state]
            rng = np.random.random()
            cumsum = np.cumsum(probs)
            for new_state, threshold in enumerate(cumsum):
                if rng < threshold:
                    if new_state != self.current_state:
                        self._on_state_change(new_state)
                    self.current_state = new_state
                    self.state_duration = 0
                    break
    
    def _on_state_change(self, new_state: int):
        """状态变化回调"""
        if self.state_duration > 1:
            self.burst_lengths.append(self.state_duration)
            self.state_durations.append(self.state_duration)
        if new_state != LossState.GOOD:
            self.burst_count += 1
            self.current_burst_id += 1
    
    def _get_loss_probability(self) -> float:
        """获取当前丢包概率"""
        rates = [
            self.config.loss_rate_good,
            self.config.loss_rate_bad,
            self.config.loss_rate_dead
        ]
        return rates[self.current_state]
    
    def generate_loss_pattern(self, n_packets: int, seed: Optional[int] = None) -> Dict[str, Any]:
        """生成丢包模式"""
        if seed is not None:
            np.random.seed(seed)
        
        self.state_history.clear()
        self.loss_history.clear()
        self.total_packets = 0
        self.total_losses = 0
        self.burst_count = 0
        self.burst_lengths = []
        self.state_durations = []
        self.current_state = self._sample_initial_state()
        self.state_duration = 0
        
        loss_events = []
        current_burst = None
        
        for packet_id in range(n_packets):
            self._update_state()
            
            loss_prob = self._get_loss_probability()
            is_lost = np.random.random() < loss_prob
            
            self.total_packets += 1
            if is_lost:
                self.total_losses += 1
                
                event = {
                    'packet_id': packet_id,
                    'state': self.current_state,
                    'state_duration': self.state_duration,
                    'loss_prob': loss_prob
                }
                loss_events.append(event)
                self.loss_history.append(1)
                
                if current_burst is None:
                    current_burst = {
                        'start': packet_id,
                        'length': 1,
                        'state': self.current_state
                    }
                else:
                    current_burst['length'] += 1
            else:
                if current_burst is not None:
                    loss_events.append({
                        'type': 'burst_end',
                        **current_burst
                    })
                    current_burst = None
                self.loss_history.append(0)
            
            self.state_history.append(self.current_state)
        
        if current_burst is not None:
            self.burst_lengths.append(current_burst['length'])
            self.state_durations.append(current_burst['length'])
        
        loss_rate = self.total_losses / self.total_packets if self.total_packets > 0 else 0
        
        return {
            'total_packets': self.total_packets,
            'total_losses': self.total_losses,
            'loss_rate': loss_rate,
            'loss_events': loss_events,
            'burst_count': self.burst_count,
            'burst_lengths': self.burst_lengths,
            'avg_burst_length': np.mean(self.burst_lengths) if self.burst_lengths else 0,
            'max_burst_length': max(self.burst_lengths) if self.burst_lengths else 0,
            'state_distribution': self._compute_state_distribution(),
            'state_transitions': self._count_transitions(),
            'state_duration_stats': self._compute_state_duration_stats()
        }

    def generate(self, n_packets: int, seed: Optional[int] = None) -> np.ndarray:
        """
        生成丢包序列 - 与GilbertElliottModel兼容的接口

        返回值: numpy数组，1表示丢包，0表示成功
        """
        pattern = self.generate_loss_pattern(n_packets, seed)

        # 从loss_events构建丢包序列
        loss_sequence = np.zeros(n_packets, dtype=int)

        for event in pattern['loss_events']:
            # 检查是否有type键，如果没有，则假设是单个丢包事件
            if 'type' in event:
                if event['type'] == 'loss':
                    loss_sequence[event['packet_id']] = 1
                elif event['type'] == 'burst_start':
                    start = event['start_packet']
                    length = event['length']
                    end = min(start + length, n_packets)
                    loss_sequence[start:end] = 1
            else:
                # 没有type键的假设是单个丢包事件
                if 'packet_id' in event:
                    loss_sequence[event['packet_id']] = 1

        return loss_sequence
    
    def _compute_state_distribution(self) -> Dict[str, float]:
        """计算状态分布"""
        if not self.state_history:
            return {'good': 0, 'bad': 0, 'dead': 0}
        
        counts = {
            LossState.GOOD: 0,
            LossState.BAD: 0,
            LossState.DEAD: 0
        }
        for state in self.state_history:
            counts[state] += 1
        
        total = len(self.state_history)
        return {
            'good': counts[LossState.GOOD] / total,
            'bad': counts[LossState.BAD] / total,
            'dead': counts[LossState.DEAD] / total
        }
    
    def _compute_state_duration_stats(self) -> Dict[str, float]:
        """计算状态驻留时间统计"""
        if not self.state_durations:
            return {'mean': 0, 'std': 0, 'median': 0}
        
        durations = np.array(self.state_durations)
        return {
            'mean': np.mean(durations),
            'std': np.std(durations),
            'median': np.median(durations),
            'max': np.max(durations),
            'min': np.min(durations)
        }
    
    def _count_transitions(self) -> int:
        """计算状态转移次数"""
        if len(self.state_history) < 2:
            return 0
        
        transitions = 0
        for i in range(1, len(self.state_history)):
            if self.state_history[i] != self.state_history[i-1]:
                transitions += 1
        
        return transitions
    
    def em_parameter_estimation(
        self, 
        loss_sequence: List[int],
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6
    ) -> Tuple['LossModelConfig', ModelFittingMetrics]:
        """
        EM算法参数估计 - 核心创新
        
        从丢包序列使用EM算法学习最优模型参数：
        1. E步：计算隐状态的后验概率
        2. M步：最大化期望似然更新参数
        3. 迭代直到收敛
        
        Args:
            loss_sequence: 丢包序列 (0/1)
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            
        Returns:
            LossModelConfig: 估计的最优配置
            ModelFittingMetrics: 模型拟合指标
        """
        n = len(loss_sequence)
        if n < 10:
            return self.config, ModelFittingMetrics()
        
        y = np.array(loss_sequence)

        # 智能初始化参数
        loss_rate = np.mean(y)

        # 根据经验分布初始化丢包率
        if loss_rate < 0.1:
            p_good = min(0.05, loss_rate * 0.5)
            p_bad = min(0.3, loss_rate * 2)
            p_dead = min(0.8, loss_rate * 4)
        elif loss_rate < 0.3:
            p_good = 0.02
            p_bad = min(0.4, loss_rate * 1.5)
            p_dead = min(0.9, loss_rate * 3)
        else:
            p_good = 0.01
            p_bad = min(0.5, loss_rate * 1.2)
            p_dead = min(0.95, loss_rate * 2)

        # 初始化转移矩阵（基于经验）
        if loss_rate < 0.1:
            # 低丢包：主要在GOOD状态
            alpha = np.array([
                [0.98, 0.015, 0.005],
                [0.4, 0.55, 0.05],
                [0.3, 0.4, 0.3]
            ])
        elif loss_rate < 0.3:
            # 中等丢包：GOOD和BAD状态都有
            alpha = np.array([
                [0.9, 0.08, 0.02],
                [0.2, 0.7, 0.1],
                [0.1, 0.3, 0.6]
            ])
        else:
            # 高丢包：经常在BAD/DEAD状态
            alpha = np.array([
                [0.8, 0.15, 0.05],
                [0.1, 0.6, 0.3],
                [0.05, 0.25, 0.7]
            ])

        # 初始化初始状态分布
        pi = np.array([0.8, 0.15, 0.05])  # 通常从GOOD状态开始
        
        log_likelihoods = []
        
        for iteration in range(max_iterations):
            gamma = np.zeros((n, 3))
            xi = np.zeros((n - 1, 3, 3))
            
            alpha_scaled, c = self._forward_backward(y, p_good, p_bad, p_dead, alpha, pi)
            
            gamma = alpha_scaled
            
            for t in range(n - 1):
                for i in range(3):
                    for j in range(3):
                        numerator = (
                            alpha_scaled[t, i] *
                            alpha[i, j] *
                            self._emission_prob(y[t + 1], 
                                [p_good, p_bad, p_dead][j]) *
                            c[t + 1]
                        )
                        xi[t, i, j] = numerator / (c[t + 1] + 1e-10)
            
            pi = gamma[0] / (np.sum(gamma[0]) + 1e-10)
            
            for i in range(3):
                for j in range(3):
                    xi_sum = np.sum(xi[:, i, j])
                    gamma_sum = np.sum(gamma[:, i])
                    alpha[i, j] = xi_sum / (gamma_sum + 1e-10)
            
            gamma_sum_i = np.sum(gamma, axis=0)
            p_good = np.sum(gamma[:, 0] * y) / (gamma_sum_i[0] + 1e-10)
            p_bad = np.sum(gamma[:, 1] * y) / (gamma_sum_i[1] + 1e-10)
            p_dead = np.sum(gamma[:, 2] * y) / (gamma_sum_i[2] + 1e-10)
            
            p_good = np.clip(p_good, 0.001, 0.999)
            p_bad = np.clip(p_bad, 0.001, 0.999)
            p_dead = np.clip(p_dead, 0.001, 0.999)
            
            alpha = self._normalize_transition_matrix(alpha)
            
            log_likelihood = np.sum(np.log(c))
            log_likelihoods.append(log_likelihood)
            
            if iteration > 0 and abs(log_likelihood - log_likelihoods[-2]) < convergence_threshold:
                break
        
        new_config = LossModelConfig(
            loss_rate_good=p_good,
            loss_rate_bad=p_bad,
            loss_rate_dead=p_dead,
            transition_matrix=TransitionMatrix.from_matrix(alpha),
            initial_state_distribution=tuple(pi / np.sum(pi))
        )
        
        metrics = ModelFittingMetrics(
            log_likelihood=log_likelihoods[-1] if log_likelihoods else 0,
            aic=2 * 8 - 2 * log_likelihoods[-1] if log_likelihoods else 0,
            bic=8 * np.log(n) - 2 * log_likelihoods[-1] if log_likelihoods else 0,
            num_params=8,
            convergence=len(log_likelihoods) < max_iterations
        )
        
        return new_config, metrics
    
    def _forward_backward(
        self,
        y: np.ndarray,
        p_good: float,
        p_bad: float,
        p_dead: float,
        alpha: np.ndarray,
        pi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """前向后向算法"""
        n = len(y)
        emission_probs = np.array([
            [self._emission_prob(y_t, p_good) for y_t in y],
            [self._emission_prob(y_t, p_bad) for y_t in y],
            [self._emission_prob(y_t, p_dead) for y_t in y]
        ]).T
        
        alpha_hat = np.zeros((n, 3))
        c = np.zeros(n)
        
        alpha_hat[0] = pi * emission_probs[0]
        c[0] = np.sum(alpha_hat[0])
        alpha_hat[0] /= (c[0] + 1e-10)
        
        for t in range(1, n):
            for j in range(3):
                alpha_hat[t, j] = np.sum(alpha_hat[t - 1] * alpha[:, j]) * emission_probs[t, j]
            c[t] = np.sum(alpha_hat[t])
            if c[t] > 0:
                alpha_hat[t] /= c[t]
        
        return alpha_hat, c
    
    def _emission_prob(self, y: int, p: float) -> float:
        """发射概率"""
        return p if y == 1 else (1 - p)
    
    def _normalize_transition_matrix(self, alpha: np.ndarray) -> np.ndarray:
        """归一化转移矩阵"""
        row_sums = np.sum(alpha, axis=1, keepdims=True)
        return alpha / (row_sums + 1e-10)
    
    def estimate_parameters(self, loss_sequence: List[int]) -> 'LossModelConfig':
        """从丢包序列估计模型参数"""
        n = len(loss_sequence)
        if n < 100:
            return self.config
        
        loss_rate = np.mean(loss_sequence)
        
        good_rate = min(0.1, loss_rate * 0.5)
        bad_rate = min(0.8, max(0.2, loss_rate * 1.5))
        dead_rate = min(0.99, loss_rate * 2.5)
        
        config = LossModelConfig(
            loss_rate_good=good_rate,
            loss_rate_bad=bad_rate,
            loss_rate_dead=dead_rate
        )
        
        return config
    
    def predict_loss_probability(
        self, 
        historical_sequence: List[int],
        horizon: int = 10
    ) -> List[Dict[str, float]]:
        """
        预测未来丢包概率 - 核心创新
        
        基于当前模型和历史数据预测未来丢包概率
        """
        predictions = []
        
        state_probs = self._infer_current_state(historical_sequence)
        
        current_p = state_probs
        
        for _ in range(horizon):
            next_p = current_p @ self.transition_matrix
            
            loss_prob = (
                next_p[0] * self.config.loss_rate_good +
                next_p[1] * self.config.loss_rate_bad +
                next_p[2] * self.config.loss_rate_dead
            )
            
            predictions.append({
                'loss_probability': loss_prob,
                'state_probs': {
                    'good': next_p[0],
                    'bad': next_p[1],
                    'dead': next_p[2]
                },
                'uncertainty': self._compute_prediction_uncertainty(next_p)
            })
            
            current_p = next_p
        
        return predictions
    
    def _infer_current_state(self, sequence: List[int]) -> np.ndarray:
        """推断当前状态分布"""
        n = len(sequence)
        if n == 0:
            return self.initial_state
        
        y = np.array(sequence)
        
        emission_probs = np.zeros((n, 3))
        emission_probs[:, 0] = [
            self._emission_prob(yt, self.config.loss_rate_good) for yt in y
        ]
        emission_probs[:, 1] = [
            self._emission_prob(yt, self.config.loss_rate_bad) for yt in y
        ]
        emission_probs[:, 2] = [
            self._emission_prob(yt, self.config.loss_rate_dead) for yt in y
        ]
        
        alpha = np.zeros((n, 3))
        alpha[0] = self.initial_state * emission_probs[0]
        alpha[0] /= (np.sum(alpha[0]) + 1e-10)
        
        for t in range(1, n):
            for j in range(3):
                alpha[t, j] = np.sum(alpha[t - 1] * self.transition_matrix[:, j]) * emission_probs[t, j]
            alpha[t] /= (np.sum(alpha[t]) + 1e-10)
        
        return alpha[-1] / (np.sum(alpha[-1]) + 1e-10)
    
    def _compute_prediction_uncertainty(self, state_probs: np.ndarray) -> float:
        """计算预测不确定性"""
        return 1.0 - np.max(state_probs)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_packets': self.total_packets,
            'total_losses': self.total_losses,
            'loss_rate': self.total_losses / max(1, self.total_packets),
            'burst_count': self.burst_count,
            'avg_burst_length': np.mean(self.burst_lengths) if self.burst_lengths else 0,
            'state_distribution': self._compute_state_distribution(),
            'state_transitions': self._count_transitions(),
            'state_duration_stats': self._compute_state_duration_stats()
        }


def estimate_burstiness(loss_sequence: List[int]) -> BurstMetrics:
    """
    估计丢包序列的突发性 - 核心创新
    
    创新指标：
    - Burst Index: 突发程度量化
    - Run Test: 检测随机性
    - Entropy: 熵值分析
    - 突发长度分布: 量化突发特性
    """
    n = len(loss_sequence)
    if n < 10:
        return BurstMetrics()
    
    runs, lengths = [], []
    current_run = 1
    for i in range(1, n):
        if loss_sequence[i] == loss_sequence[i-1]:
            current_run += 1
        else:
            runs.append(loss_sequence[i-1])
            lengths.append(current_run)
            current_run = 1
    runs.append(loss_sequence[-1])
    lengths.append(current_run)
    
    burst_index = sum(l**2 for l in lengths) / sum(lengths) if lengths else 1
    
    p = np.mean(loss_sequence)
    expected_runs = 1 + 2 * n * p * (1 - p)
    actual_runs = len(runs)
    run_test_statistic = (actual_runs - expected_runs) / np.sqrt(2 * n * p * (1 - p)) if p > 0 and p < 1 else 0
    
    unique, counts = np.unique(runs, return_counts=True)
    probs = counts / len(runs)
    entropy = -sum(p_ * np.log(p_ + 1e-10) for p_ in probs)
    
    length_arr = np.array(lengths)
    length_distribution = {
        'short': np.sum(length_arr <= 2) / len(length_arr),
        'medium': np.sum((length_arr > 2) & (length_arr <= 5)) / len(length_arr),
        'long': np.sum((length_arr > 5) & (length_arr <= 10)) / len(length_arr),
        'very_long': np.sum(length_arr > 10) / len(length_arr)
    }
    
    # 计算增强指标
    # 1. 突发强度：考虑突发长度和频率
    if lengths:
        burst_lengths = [l for l, r in zip(lengths, runs) if r == 1]  # 只考虑丢包突发
        if burst_lengths:
            burst_intensity = np.mean(burst_lengths) * np.sqrt(len(burst_lengths) / n)
        else:
            burst_intensity = 0.0
    else:
        burst_intensity = 0.0

    # 2. 周期性指标：使用FFT检测周期性
    if n >= 32:  # 需要足够的数据点
        signal = np.array(loss_sequence)
        fft = np.fft.fft(signal)
        power_spectrum = np.abs(fft) ** 2
        # 寻找主要频率成分
        peak_indices = np.argsort(power_spectrum[1:n//2])[-3:]  # 前3个峰值
        periodicity_index = np.mean(power_spectrum[peak_indices]) / np.mean(power_spectrum[1:n//2])
    else:
        periodicity_index = 0.0

    # 3. 自相关系数：lag-1自相关
    if n >= 3:
        signal = np.array(loss_sequence)
        autocorrelation_coefficient = np.corrcoef(signal[:-1], signal[1:])[0, 1]
        autocorrelation_coefficient = 0.0 if np.isnan(autocorrelation_coefficient) else autocorrelation_coefficient
    else:
        autocorrelation_coefficient = 0.0

    # 4. 分形维度：使用盒子计数法近似
    if n >= 64:
        # 简单分形维度估计
        scales = [2, 4, 8, 16]
        counts = []
        for scale in scales:
            boxes = np.zeros(n // scale + 1)
            for i in range(0, n, scale):
                boxes[i // scale] = np.max(signal[i:i+scale])
            counts.append(np.sum(boxes))

        if len(counts) >= 2:
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            fractal_dimension = -slope
        else:
            fractal_dimension = 1.0
    else:
        fractal_dimension = 1.0

    # 5. 突发模式复杂度：基于信息熵和模式多样性
    if lengths and runs:
        # 计算模式的多样性
        unique_patterns = len(set(zip(lengths, runs)))
        max_possible_patterns = min(len(lengths), 2 ** min(10, len(lengths)))  # 理论最大值
        pattern_diversity = unique_patterns / max_possible_patterns if max_possible_patterns > 0 else 0

        # 结合熵和多样性
        burst_pattern_complexity = entropy * pattern_diversity
    else:
        burst_pattern_complexity = 0.0

    return BurstMetrics(
        burst_index=burst_index,
        run_test_statistic=run_test_statistic,
        entropy=entropy,
        run_count=actual_runs,
        avg_run_length=np.mean(lengths) if lengths else 0,
        loss_probability=p,
        burst_length_distribution=length_distribution,
        burst_intensity=burst_intensity,
        periodicity_index=periodicity_index,
        autocorrelation_coefficient=autocorrelation_coefficient,
        fractal_dimension=fractal_dimension,
        burst_pattern_complexity=burst_pattern_complexity
    )
