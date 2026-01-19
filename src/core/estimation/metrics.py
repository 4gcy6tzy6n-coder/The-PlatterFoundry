#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimation Metrics - 估计精度指标模块

创新贡献：
1. MASE (Mean Absolute Scaled Error): 无尺度误差度量
2. MAPE/SMAPE: 百分比误差度量
3. RMSE/RMSSE: 均方根误差及其缩放版本
4. 置信区间覆盖率: 评估估计不确定性质量

这解决了传统误差指标无法跨场景比较的问题。
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class EstimationMetrics:
    """综合估计指标"""
    mae: float = 0.0
    mape: float = 0.0
    smape: float = 0.0
    rmse: float = 0.0
    mase: float = 0.0
    coverage: float = 0.0
    mean_interval_width: float = 0.0


@dataclass
class EstimationAccuracyMetrics:
    """估计精度指标（用于创新点量化）"""
    point_accuracy: float = 0.0
    uncertainty_quality: float = 0.0
    trend_detection_accuracy: float = 0.0
    burst_detection_delay: float = 0.0
    convergence_speed: float = 0.0


class MetricsCalculator:
    """
    估计精度指标计算器
    
    提供多种创新性评估指标：
    1. MASE: 相对于朴素预测器的误差缩放
    2. 置信区间覆盖率: 评估不确定性估计质量
    3. 趋势检测准确率: 评估趋势估计能力
    """
    
    def __init__(self, seasonality_period: int = 1):
        self.seasonality_period = seasonality_period
    
    def calculate_mase(
        self, 
        actual: List[float], 
        predicted: List[float],
        naive_forecast: List[float] = None
    ) -> float:
        """
        计算MASE (Mean Absolute Scaled Error)
        
        MASE = mean(|e_t|) / mean(|e_t^naive|)
        
        创新点：使用朴素预测器作为基准，实现跨场景可比较的误差度量
        """
        n = len(actual)
        if n < 2:
            return 0.0
        
        errors = np.abs(np.array(actual) - np.array(predicted))
        
        if naive_forecast is None:
            naive_forecast = actual[:-1] + [actual[-1]]
        
        naive_errors = np.abs(np.array(actual[1:]) - np.array(actual[:-1]))
        
        mean_error = np.mean(errors)
        mean_naive_error = np.mean(naive_errors) + 1e-10
        
        return mean_error / mean_naive_error
    
    def calculate_mape(self, actual: List[float], predicted: List[float]) -> float:
        """
        计算MAPE (Mean Absolute Percentage Error)
        
        MAPE = (1/n) * Σ|e_t / y_t|
        """
        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)
        
        mask = actual_arr != 0
        if not np.any(mask):
            return 0.0
        
        mape = np.mean(np.abs((actual_arr[mask] - predicted_arr[mask]) / actual_arr[mask]))
        return mape
    
    def calculate_smape(self, actual: List[float], predicted: List[float]) -> float:
        """
        计算SMAPE (Symmetric MAPE)
        
        SMAPE = (1/n) * Σ|e_t| / (|y_t| + |ŷ_t|)
        
        创新点：对称MAPE避免了大值的主导效应
        """
        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)
        
        denominator = np.abs(actual_arr) + np.abs(predicted_arr)
        mask = denominator != 0
        
        if not np.any(mask):
            return 0.0
        
        smape = np.mean(np.abs(actual_arr[mask] - predicted_arr[mask]) / denominator[mask])
        return smape
    
    def calculate_rmse(self, actual: List[float], predicted: List[float]) -> float:
        """计算RMSE (Root Mean Square Error)"""
        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)
        
        mse = np.mean((actual_arr - predicted_arr) ** 2)
        return np.sqrt(mse)
    
    def calculate_coverage(
        self, 
        actual: List[float], 
        lower_bounds: List[float], 
        upper_bounds: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        计算置信区间覆盖率
        
        创新点：评估估计不确定性的可靠性
        
        Returns:
            coverage: 实际值在置信区间内的比例
            avg_width: 平均置信区间宽度
        """
        actual_arr = np.array(actual)
        lower_arr = np.array(lower_bounds)
        upper_arr = np.array(upper_bounds)
        
        within_interval = (actual_arr >= lower_arr) & (actual_arr <= upper_arr)
        coverage = np.mean(within_interval)
        
        avg_width = np.mean(upper_arr - lower_arr)
        
        return coverage, avg_width
    
    def calculate_uncertainty_quality(
        self,
        predictions: List[float],
        actuals: List[float],
        lower_bounds: List[float],
        upper_bounds: List[float]
    ) -> Dict[str, float]:
        """
        计算不确定性质量指标
        
        创新点：多维度评估不确定性估计质量
        """
        coverage, width = self.calculate_coverage(actuals, lower_bounds, upper_bounds)
        
        prediction_arr = np.array(predictions)
        actual_arr = np.array(actuals)
        
        within_10pct = 0
        within_20pct = 0
        for i, (pred, actual_val, lower, upper) in enumerate(zip(predictions, actuals, lower_bounds, upper_bounds)):
            if actual_val > 0:
                relative_error = abs(pred - actual_val) / actual_val
                if relative_error <= 0.10:
                    within_10pct += 1
                if relative_error <= 0.20:
                    within_20pct += 1
        
        n = len(actuals)
        
        return {
            'coverage': coverage,
            'avg_interval_width': width,
            'within_10_percent': within_10pct / n if n > 0 else 0,
            'within_20_percent': within_20pct / n if n > 0 else 0,
            'sharpness': width / (np.mean(actual_arr) + 1e-10)
        }
    
    def calculate_trend_detection_accuracy(
        self,
        actual: List[float],
        detected_trends: List[str],
        tolerance: int = 2
    ) -> float:
        """
        计算趋势检测准确率
        
        创新点：评估估计器识别趋势变化的能力
        """
        if len(actual) < 3 or len(detected_trends) != len(actual) - 1:
            return 0.0
        
        actual_trends = []
        for i in range(1, len(actual)):
            diff = actual[i] - actual[i-1]
            if abs(diff) < 1e-6:
                actual_trends.append("stable")
            elif diff > 0:
                actual_trends.append("rising")
            else:
                actual_trends.append("falling")
        
        correct = 0
        for i, (detected, actual_trend) in enumerate(zip(detected_trends, actual_trends)):
            if i >= len(detected_trends):
                break
            if detected == actual_trend:
                correct += 1
            elif detected == "stable" and actual_trend == "stable":
                correct += 1
        
        return correct / len(actual_trends) if actual_trends else 0.0
    
    def calculate_convergence_speed(
        self,
        errors: List[float],
        threshold: float = 0.1
    ) -> int:
        """
        计算收敛速度
        
        创新点：量化估计器达到稳定状态所需的迭代次数
        """
        if not errors:
            return 0
        
        initial_error = errors[0]
        if initial_error == 0:
            return 0
        
        for i, error in enumerate(errors):
            relative_error = error / initial_error
            if relative_error <= threshold:
                return i
        
        return len(errors)
    
    def calculate_all_metrics(
        self,
        actual: List[float],
        predicted: List[float],
        lower_bounds: List[float] = None,
        upper_bounds: List[float] = None
    ) -> EstimationMetrics:
        """
        计算所有估计指标
        
        Returns:
            EstimationMetrics: 包含所有指标的容器
        """
        metrics = EstimationMetrics()
        
        metrics.mae = np.mean(np.abs(np.array(actual) - np.array(predicted)))
        metrics.mape = self.calculate_mape(actual, predicted)
        metrics.smape = self.calculate_smape(actual, predicted)
        metrics.rmse = self.calculate_rmse(actual, predicted)
        metrics.mase = self.calculate_mase(actual, predicted)
        
        if lower_bounds is not None and upper_bounds is not None:
            coverage, width = self.calculate_coverage(actual, lower_bounds, upper_bounds)
            metrics.coverage = coverage
            metrics.mean_interval_width = width
        
        return metrics


def compute_composite_accuracy_score(
    accuracy_metrics: Dict[str, float],
    weights: Dict[str, float] = None
) -> float:
    """
    计算综合精度分数
    
    创新点：多维度指标加权融合
    """
    if weights is None:
        weights = {
            'point_accuracy': 0.4,
            'uncertainty_quality': 0.3,
            'trend_detection': 0.2,
            'convergence_speed': 0.1
        }
    
    score = 0.0
    for metric_name, weight in weights.items():
        if metric_name in accuracy_metrics:
            score += weight * accuracy_metrics[metric_name]
    
    return score
