#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standard Kalman Filter - 标准卡尔曼滤波器基准实现

这是一个经典的卡尔曼滤波器实现，作为自适应估计算法的基准对比。
使用固定参数设置，提供稳定的状态估计性能。

核心特性：
1. 经典卡尔曼滤波算法实现
2. 固定过程噪声和测量噪声参数
3. 状态预测和测量更新
4. 适用于线性高斯系统
5. 提供置信区间估计

作为传统方法的代表，用于对比自适应算法的优势。
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """卡尔曼滤波器状态"""

    x: float  # 状态估计值
    P: float  # 状态协方差
    K: float = 0.0  # 卡尔曼增益


@dataclass
class StandardKalmanResult:
    """标准卡尔曼滤波器结果"""

    estimate: float
    confidence_lower: float
    confidence_upper: float
    confidence_std: float
    kalman_gain: float
    innovation: float
    residual: float


class StandardKalmanFilter:
    """
    标准卡尔曼滤波器实现

    使用经典的卡尔曼滤波算法进行状态估计，
    参数固定不变，作为传统方法的基准。
    """

    def __init__(self,
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1,
                 initial_state: float = 0.0,
                 initial_covariance: float = 1.0):
        """
        初始化标准卡尔曼滤波器

        Args:
            process_noise: 过程噪声方差 (固定参数)
            measurement_noise: 测量噪声方差 (固定参数)
            initial_state: 初始状态估计
            initial_covariance: 初始状态协方差
        """
        self.Q = process_noise  # 过程噪声方差
        self.R = measurement_noise  # 测量噪声方差

        # 初始化状态
        self.state = KalmanState(
            x=initial_state,
            P=initial_covariance
        )

        # 状态转移矩阵 (一维情况)
        self.F = 1.0  # 状态转移矩阵
        self.H = 1.0  # 观测矩阵

        # 控制输入 (无控制输入)
        self.B = 0.0
        self.u = 0.0

        logger.info(f"StandardKalmanFilter initialized with Q={self.Q}, R={self.R}")

    def predict(self) -> KalmanState:
        """
        预测步骤

        使用状态转移矩阵预测下一时刻的状态

        Returns:
            预测后的状态
        """
        # 状态预测
        x_pred = self.F * self.state.x + self.B * self.u

        # 协方差预测
        P_pred = self.F * self.state.P * self.F + self.Q

        predicted_state = KalmanState(x=x_pred, P=P_pred)
        return predicted_state

    def update(self, measurement: float) -> StandardKalmanResult:
        """
        更新步骤

        使用测量值更新状态估计

        Args:
            measurement: 测量值

        Returns:
            更新后的估计结果
        """
        # 计算卡尔曼增益
        K = (self.state.P * self.H) / (self.H * self.state.P * self.H + self.R)

        # 状态更新
        innovation = measurement - self.H * self.state.x
        x_updated = self.state.x + K * innovation

        # 协方差更新
        P_updated = (1 - K * self.H) * self.state.P

        # 更新内部状态
        self.state.x = x_updated
        self.state.P = P_updated
        self.state.K = K

        # 计算置信区间 (假设3σ)
        confidence_std = np.sqrt(P_updated)
        confidence_lower = x_updated - 3 * confidence_std
        confidence_upper = x_updated + 3 * confidence_std

        # 计算残差
        residual = measurement - x_updated

        result = StandardKalmanResult(
            estimate=x_updated,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            confidence_std=confidence_std,
            kalman_gain=K,
            innovation=innovation,
            residual=residual
        )

        return result

    def estimate(self, measurement: float) -> StandardKalmanResult:
        """
        执行完整的估计步骤：预测 + 更新

        Args:
            measurement: 当前测量值

        Returns:
            估计结果
        """
        # 预测步骤
        self.predict()

        # 更新步骤
        result = self.update(measurement)

        return result

    def reset(self, initial_state: float = 0.0, initial_covariance: float = 1.0):
        """
        重置滤波器状态

        Args:
            initial_state: 新的初始状态
            initial_covariance: 新的初始协方差
        """
        self.state = KalmanState(x=initial_state, P=initial_covariance)
        logger.info(f"Kalman filter reset to x={initial_state}, P={initial_covariance}")

    def get_state_info(self) -> Dict[str, Any]:
        """
        获取当前状态信息

        Returns:
            状态信息字典
        """
        return {
            'current_estimate': self.state.x,
            'current_covariance': self.state.P,
            'process_noise': self.Q,
            'measurement_noise': self.R,
            'last_kalman_gain': self.state.K
        }


class StandardKalmanFilterExperiment:
    """
    标准卡尔曼滤波器实验包装器

    提供实验友好的接口，与其他估计算法保持一致
    """

    def __init__(self,
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1):
        """
        初始化实验包装器

        Args:
            process_noise: 过程噪声参数
            measurement_noise: 测量噪声参数
        """
        self.filter = StandardKalmanFilter(
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )

        self.results = []
        self.measurements = []
        self.estimates = []

    def run_experiment(self, measurements: list, reset_between_runs: bool = True) -> Dict[str, Any]:
        """
        运行完整的实验

        Args:
            measurements: 测量值序列
            reset_between_runs: 是否在每次运行之间重置滤波器

        Returns:
            实验结果字典
        """
        if reset_between_runs:
            self.filter.reset()

        self.results = []
        self.measurements = measurements
        self.estimates = []

        for measurement in measurements:
            result = self.filter.estimate(measurement)

            self.results.append(result)
            self.estimates.append(result.estimate)

        # 计算性能指标
        estimates = np.array(self.estimates)
        measurements = np.array(self.measurements)

        rmse = np.sqrt(np.mean((estimates - measurements) ** 2))
        mae = np.mean(np.abs(estimates - measurements))

        # 计算收敛时间 (误差首次小于某个阈值)
        errors = np.abs(estimates - measurements)
        convergence_threshold = 0.1  # 10%的收敛阈值
        convergence_time = None

        for i, error in enumerate(errors):
            if error < convergence_threshold:
                convergence_time = i
                break

        if convergence_time is None:
            convergence_time = len(measurements)  # 如果未收敛，使用总长度

        return {
            'rmse': rmse,
            'mae': mae,
            'convergence_time': convergence_time,
            'estimates': self.estimates,
            'results': self.results,
            'measurements': self.measurements,
            'filter_params': self.filter.get_state_info()
        }

    def get_performance_summary(self) -> Dict[str, float]:
        """
        获取性能汇总

        Returns:
            性能指标字典
        """
        if not self.results:
            return {}

        estimates = np.array(self.estimates)
        measurements = np.array(self.measurements)

        return {
            'rmse': np.sqrt(np.mean((estimates - measurements) ** 2)),
            'mae': np.mean(np.abs(estimates - measurements)),
            'mean_estimate': np.mean(estimates),
            'std_estimate': np.std(estimates),
            'final_estimate': estimates[-1] if len(estimates) > 0 else 0,
            'total_measurements': len(measurements)
        }


# 便捷函数
def create_standard_kalman_filter(process_noise: float = 0.01,
                                 measurement_noise: float = 0.1) -> StandardKalmanFilter:
    """
    创建标准卡尔曼滤波器实例

    Args:
        process_noise: 过程噪声
        measurement_noise: 测量噪声

    Returns:
        配置好的卡尔曼滤波器实例
    """
    return StandardKalmanFilter(
        process_noise=process_noise,
        measurement_noise=measurement_noise
    )


def run_kalman_comparison(measurements: list,
                         process_noises: list = [0.001, 0.01, 0.1],
                         measurement_noises: list = [0.01, 0.1, 1.0]) -> Dict[str, Any]:
    """
    运行卡尔曼滤波器参数对比实验

    Args:
        measurements: 测量值序列
        process_noises: 过程噪声参数列表
        measurement_noises: 测量噪声参数列表

    Returns:
        参数对比结果
    """
    results = {}

    for q in process_noises:
        for r in measurement_noises:
            key = f"Q_{q}_R_{r}"
            filter_exp = StandardKalmanFilterExperiment(
                process_noise=q,
                measurement_noise=r
            )

            result = filter_exp.run_experiment(measurements)
            results[key] = result

    return results


if __name__ == "__main__":
    # 示例用法
    print("Standard Kalman Filter - 标准卡尔曼滤波器基准实现")
    print("=" * 60)

    # 创建测试数据
    np.random.seed(42)
    true_values = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.5
    measurements = true_values + np.random.normal(0, 0.1, len(true_values))

    # 创建卡尔曼滤波器
    kf = StandardKalmanFilter(process_noise=0.01, measurement_noise=0.1)

    print("运行标准卡尔曼滤波器...")
    estimates = []

    for measurement in measurements[:20]:  # 只显示前20个
        result = kf.estimate(measurement)
        estimates.append(result.estimate)
        print(".4f")

    print("\n性能统计:")
    print(".4f")
    print(".4f")
    # 参数对比
    print("\n参数对比实验...")
    comparison_results = run_kalman_comparison(measurements[:50])
    for key, result in comparison_results.items():
        print(".4f")