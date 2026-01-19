#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adaptive Network Estimator - 自适应网络状态估计器

创新点：
1. 双时间尺度估计：短期响应(10样本) + 长期趋势(100样本)，解决动态网络适应问题
2. 变结构卡尔曼滤波：自动调整过程噪声参数，适应不同网络环境
3. 突发检测算法：基于Z-score的早期预警，识别网络状态突变
4. 置信区间量化：提供估计可靠性指标，支持决策边界分析
5. 自适应噪声调整：根据估计误差动态调整噪声参数
6. 多尺度融合：融合短期和长期估计结果

这解决了传统固定参数估计器无法适应网络动态变化的问题。
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class EstimateResult:
    """估计结果，包含置信区间"""

    value: float
    confidence_lower: float
    confidence_upper: float
    confidence: float
    is_stable: bool
    trend: str
    uncertainty_measure: float = 0.0


@dataclass
class AdaptiveNoiseConfig:
    """自适应噪声配置"""

    base_process_noise: float = 0.001
    base_measurement_noise: float = 0.01
    noise_increase_rate: float = 1.1
    noise_decrease_rate: float = 0.95
    max_process_noise: float = 0.1
    min_process_noise: float = 0.0001
    adaptation_speed: float = 0.1


@dataclass
class AblationConfig:
    """消融实验配置 - 允许关闭特定组件"""

    enable_adaptive_noise: bool = True  # 自适应噪声调整
    enable_burst_detection: bool = True  # 突发检测
    enable_multi_scale_fusion: bool = True  # 多尺度融合
    enable_trend_estimation: bool = True  # 趋势估计


class AdaptiveEstimator:
    """
    自适应网络状态估计器 - 真正的卡尔曼滤波实现

    核心创新：
    - 变结构卡尔曼滤波：完整的状态空间模型
    - 突发检测：早期识别网络状态变化
    - 置信量化：提供估计可靠性指标
    - 自适应噪声：动态调整噪声参数

    卡尔曼滤波状态空间模型：
    x[k+1] = F * x[k] + w[k]    # 状态转移
    z[k] = H * x[k] + v[k]      # 观测方程

    其中：
    - x: 状态向量 [value, trend]
    - F: 状态转移矩阵
    - H: 观测矩阵
    - Q: 过程噪声协方差
    - R: 观测噪声协方差
    """

    def __init__(
        self,
        name: str,
        short_window: int = 10,
        long_window: int = 100,
        burst_threshold: float = 2.0,
        stability_threshold: float = 0.1,
        initial_value: float = 0.1,
        noise_config: AdaptiveNoiseConfig | None = None,
        ablation_config: AblationConfig | None = None,
    ):
        self.name: str = name
        self.short_window: int = short_window
        self.long_window: int = long_window
        self.burst_threshold: float = burst_threshold
        self.stability_threshold: float = stability_threshold

        self.noise_config: AdaptiveNoiseConfig = noise_config or AdaptiveNoiseConfig()
        self.ablation_config: AblationConfig = ablation_config or AblationConfig()
        self.logger: logging.Logger = logging.getLogger(f"Estimator_{name}")

        self.short_history: deque[float] = deque(maxlen=short_window)
        self.long_history: deque[float] = deque(maxlen=long_window)

        # 卡尔曼滤波状态向量 [value, trend]
        self.x: np.ndarray = np.array([initial_value, 0.0], dtype=float).reshape(2, 1)
        self.state_cov: np.ndarray = np.eye(2, dtype=float) * 0.1
        self.F: np.ndarray = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
        self.H: np.ndarray = np.array([[1.0, 0.0]], dtype=float)

        # 过程噪声协方差 Q (使用保护成员以避免常量重赋值)
        self.process_noise_cov: np.ndarray = (
            np.eye(2, dtype=float) * self.noise_config.base_process_noise
        )

        # 观测噪声协方差 R
        self.measurement_noise_cov: np.ndarray = np.array(
            [[self.noise_config.base_measurement_noise]], dtype=float
        )

        self.kalman_gain: np.ndarray = np.zeros((2, 1), dtype=float)
        self.last_update_time: float = 0.0

        self.burst_count: int = 0
        self.last_burst_time: float = 0.0

        self.prediction_error: deque[float] = deque(maxlen=20)

        self.current_estimate: float = initial_value
        self.trend: float = 0.0
        self.variance: float = 0.0
        self.process_noise: float = self.noise_config.base_process_noise
        self.measurement_noise: float = self.noise_config.base_measurement_noise
        self._single_history: deque[float] = deque(
            maxlen=max(short_window, long_window)
        )

    def update(self, measurement: float, timestamp: float) -> EstimateResult:
        """更新估计值 - 真正的卡尔曼滤波实现"""
        # 计算预测误差用于自适应调整（在卡尔曼滤波之前）
        if self.short_history:
            prediction = self._predict()
            error = abs(measurement - prediction)
            self.prediction_error.append(error)

        # 异常值检测和处理
        if self._is_outlier(measurement):
            self.logger.debug(f"异常值检测: {measurement:.4f}")
            measurement = self._smooth_outlier(measurement)

        # 添加到历史记录 - 多尺度融合组件
        if self.ablation_config.enable_multi_scale_fusion:
            self.short_history.append(measurement)
            self.long_history.append(measurement)
        else:
            # 如果关闭多尺度融合，只使用短期历史
            if not hasattr(self, "_single_history"):
                self._single_history = deque(
                    maxlen=max(self.short_window, self.long_window)
                )
            self._single_history.append(measurement)

        # 自适应调整噪声参数
        if self.ablation_config.enable_adaptive_noise:
            self._adaptive_noise_adjustment(measurement)

        # 执行真正的卡尔曼滤波
        self._adaptive_kalman_update(measurement)

        # 趋势估计
        if self.ablation_config.enable_trend_estimation:
            self._estimate_trend()

        # 检测突发和稳定性
        if self.ablation_config.enable_burst_detection:
            _is_burst = self._detect_burst(measurement)
            is_stable = self._check_stability()
        else:
            is_stable = True

        # 计算置信区间（基于卡尔曼滤波协方差）
        confidence_lower, confidence_upper = self._compute_confidence_interval()
        confidence = self._compute_confidence()

        # 计算不确定性度量
        uncertainty = self._compute_uncertainty_measure()

        # 更新时间戳
        self.last_update_time = timestamp

        return EstimateResult(
            value=float(self.x[0, 0]),
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            confidence=confidence,
            is_stable=is_stable,
            trend=self._get_trend_direction(),
            uncertainty_measure=uncertainty,
        )

    def _predict(self) -> float:
        """一步预测 - 基于卡尔曼滤波状态"""
        x_pred = self.F @ self.x
        return float(x_pred[0, 0])

    def _adaptive_noise_adjustment(self, measurement: float) -> None:
        """自适应调整噪声参数"""
        if len(self.prediction_error) < 3:
            return

        recent_errors = list(self.prediction_error)[-5:]
        mean_error = float(np.mean(recent_errors))
        error_trend = (
            float(recent_errors[-1] - recent_errors[0])
            if len(recent_errors) >= 2
            else 0.0
        )

        # 动态调整过程噪声协方差 Q
        current_q_scale = (
            self.process_noise_cov[0, 0] / self.noise_config.base_process_noise
        )
        if mean_error > 0.01 and error_trend > 0:
            new_q_scale = min(
                10.0, current_q_scale * self.noise_config.noise_increase_rate
            )
        elif mean_error > 0.01 and error_trend < 0:
            new_q_scale = max(
                0.1, current_q_scale * self.noise_config.noise_decrease_rate
            )
        else:
            new_q_scale = current_q_scale

        # Update process noise covariance diagonal elements
        self.process_noise_cov[0, 0] = (
            self.noise_config.base_process_noise * new_q_scale
        )
        self.process_noise_cov[1, 1] = (
            self.noise_config.base_process_noise * new_q_scale
        )

        # 动态调整观测噪声协方差 R
        new_r_scale = float(max(0.1, min(5.0, 1 + mean_error * 10)))
        self.measurement_noise_cov[0, 0] = (
            self.noise_config.base_measurement_noise * new_r_scale
        )

        # 为了向后兼容，更新传统变量
        self.process_noise = float(self.process_noise_cov[0, 0])
        self.measurement_noise = float(self.measurement_noise_cov[0, 0])

    def _is_outlier(self, measurement: float) -> bool:
        """检测异常值"""
        history = (
            self.short_history
            if self.ablation_config.enable_multi_scale_fusion
            else self._single_history
        )

        if len(history) < 3:
            return False
        mean = float(np.mean(history))
        std = float(np.std(history))
        if std < 1e-6:
            return False
        return bool(abs(measurement - mean) > self.burst_threshold * std)

    def _smooth_outlier(self, measurement: float) -> float:
        """平滑异常值"""
        history = (
            self.short_history
            if self.ablation_config.enable_multi_scale_fusion
            else self._single_history
        )

        if len(history) < 2:
            return measurement
        return float(np.median(history))

    def _adaptive_kalman_update(self, measurement: float) -> None:
        """真正的卡尔曼滤波更新 - 核心创新"""
        # 预测步骤 (Prediction)
        x_pred = self.F @ self.x  # 状态预测
        P_pred = (
            self.F @ self.state_cov @ self.F.T + self.process_noise_cov
        )  # 协方差预测

        # 计算测量残差 (innovation)
        z = np.array([[measurement]], dtype=float)  # 观测向量

        y = z - self.H @ x_pred  # 观测残差 (innovation)

        S = self.H @ P_pred @ self.H.T + self.measurement_noise_cov  # 残差协方差
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # 卡尔曼增益

        # 更新状态和协方差
        self.x = x_pred + K @ y  # 保持2x1形状
        self.state_cov = (np.eye(2) - K @ self.H) @ P_pred
        self.state_cov = np.asarray(self.state_cov, dtype=float)

        # 存储卡尔曼增益用于调试
        self.kalman_gain = K

        # 更新传统变量以保持兼容性
        self.current_estimate = float(self.x[0, 0])  # 提取估计值
        self.trend = float(self.x[1, 0])  # 提取趋势
        self.variance = float(self.state_cov[0, 0])  # 提取方差

    def _estimate_trend(self) -> None:
        """估计趋势 - 增强版"""
        history = (
            self.long_history
            if self.ablation_config.enable_multi_scale_fusion
            else self._single_history
        )

        if len(history) < 10:
            self.trend = 0.0
            return

        recent = np.array(list(history)[-20:], dtype=float)

        # 方法1: 线性回归趋势
        if len(recent) >= 10:
            x = np.arange(len(recent))
            slope, _ = np.polyfit(x, recent, 1)
            linear_trend = float(slope)
        else:
            linear_trend = 0.0

        # 方法2: 加权移动平均趋势
        if len(recent) >= 5:
            weights = np.exp(np.linspace(-1, 0, len(recent)), dtype=float)
            weights /= np.sum(weights)
            weighted_avg = float(np.average(recent, weights=weights))
            prev_avg = (
                float(np.average(recent[:-5], weights=weights[: len(recent) - 5]))
                if len(recent) > 5
                else weighted_avg
            )
            weighted_trend = (weighted_avg - prev_avg) / 5
        else:
            weighted_trend = 0.0

        # 方法3: 指数平滑趋势
        if len(recent) >= 3:
            alpha = 0.3  # 平滑系数
            smoothed = np.zeros(len(recent), dtype=float)
            smoothed[0] = float(recent[0])
            for i in range(1, len(recent)):
                smoothed[i] = alpha * float(recent[i]) + (1 - alpha) * smoothed[i - 1]

            # 计算平滑序列的趋势
            if len(smoothed) >= 5:
                smooth_trend = float((smoothed[-1] - smoothed[-5]) / 4)
            else:
                smooth_trend = float(
                    (smoothed[-1] - smoothed[0]) / max(1, len(smoothed) - 1)
                )
        else:
            smooth_trend = 0.0

        # 融合三种趋势估计（加权平均）
        weights = np.array([0.5, 0.3, 0.2], dtype=float)
        trends = np.array([linear_trend, weighted_trend, smooth_trend], dtype=float)

        # 处理NaN值
        valid_mask = ~np.isnan(trends)
        if np.any(valid_mask):
            self.trend = float(
                np.average(trends[valid_mask], weights=weights[valid_mask])
            )
        else:
            self.trend = 0.0

    def _detect_burst(self, measurement: float) -> bool:
        """检测突发变化 - 核心创新"""
        short_hist = (
            self.short_history
            if self.ablation_config.enable_multi_scale_fusion
            else self._single_history
        )
        long_hist = (
            self.long_history
            if self.ablation_config.enable_multi_scale_fusion
            else self._single_history
        )

        if len(short_hist) < 5:
            return False

        # 策略1: Z-score检测
        mean = float(np.mean(short_hist))
        std = float(np.std(short_hist))

        if std < 1e-6:
            return False

        z_score = abs(measurement - mean) / std
        z_score_burst = z_score > self.burst_threshold

        # 策略2: 移动平均偏差检测
        if len(long_hist) >= 10:
            recent_avg = float(np.mean(list(long_hist)[-10:]))
            long_term_avg = (
                float(np.mean(list(long_hist)[:-10]))
                if len(long_hist) > 10
                else recent_avg
            )
            deviation_ratio = abs(recent_avg - long_term_avg) / (
                abs(long_term_avg) + 1e-6
            )
            deviation_burst = deviation_ratio > 0.5
        else:
            deviation_burst = False

        # 策略3: 趋势突变检测
        if len(short_hist) >= 3:
            recent_trend = np.polyfit(np.arange(3), list(short_hist)[-3:], 1)[0]
            if len(short_hist) >= 6:
                prev_trend = float(
                    np.polyfit(np.arange(3), list(short_hist)[-6:-3], 1)[0]
                )
                trend_change = abs(recent_trend - prev_trend)
                trend_burst = trend_change > 0.1
            else:
                trend_burst = False
        else:
            trend_burst = False

        # 多重确认：至少两种策略检测到突发
        burst_detected = (
            (z_score_burst and deviation_burst)
            or (z_score_burst and trend_burst)
            or (deviation_burst and trend_burst)
        )

        if burst_detected:
            self.burst_count += 1
            self.last_burst_time = self.last_update_time
            self.logger.debug(
                f"突发检测: z_score={z_score:.2f}, deviation_burst={deviation_burst}, trend_burst={trend_burst}"
            )
            return True

        return False

    def _check_stability(self) -> bool:
        """检查稳定性"""
        history = (
            self.short_history
            if self.ablation_config.enable_multi_scale_fusion
            else self._single_history
        )

        if len(history) < (
            self.short_window if self.ablation_config.enable_multi_scale_fusion else 10
        ):
            return False

        recent = list(history)[-10:]
        if len(recent) < 2:
            return False

        cv = float(np.std(recent)) / (float(np.mean(recent)) + 1e-10)
        return bool(cv < self.stability_threshold)

    def _compute_confidence_interval(self) -> Tuple[float, float]:
        """计算置信区间 - 基于卡尔曼滤波协方差"""
        state_variance = self.H @ self.state_cov @ self.H.T
        std_dev = float(np.sqrt(max(0, state_variance[0, 0])))

        margin = 1.96 * std_dev

        lower = float(max(0.0, self.x[0, 0] - margin))
        upper = float(min(1.0, self.x[0, 0] + margin))

        return (lower, upper)

    def _compute_confidence(self) -> float:
        """计算置信度"""
        if len(self.short_history) < 5:
            return 0.5

        history_factor = min(1.0, len(self.short_history) / self.short_window)
        stability_factor = 1.0 if self._check_stability() else 0.7

        variance_factor = max(0.5, 1.0 - min(1.0, self.variance * 10))

        confidence = (
            history_factor * 0.3 + stability_factor * 0.4 + variance_factor * 0.3
        )
        return float(min(1.0, confidence))

    def _compute_uncertainty_measure(self) -> float:
        """计算不确定性度量 - 核心创新"""
        if len(self.short_history) < 2:
            return 1.0

        variance_component = min(1.0, self.variance * 10)

        spread = float(np.std(list(self.short_history))) / (
            float(np.mean(list(self.short_history))) + 1e-10
        )
        spread_component = min(1.0, spread)

        instability_penalty = 0.0 if self._check_stability() else 0.3

        uncertainty = (variance_component + spread_component) / 2 + instability_penalty
        return float(min(1.0, uncertainty))

    def _get_trend_direction(self) -> str:
        """获取趋势方向"""
        if abs(self.trend) < 1e-6:
            return "stable"
        elif self.trend > 0:
            return "rising"
        else:
            return "falling"

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        result: Dict[str, Any] = {
            "name": self.name,
            "current_estimate": self.current_estimate,
            "trend": self.trend,
            "variance": self.variance,
            "kalman_gain": self.kalman_gain.tolist()
            if isinstance(self.kalman_gain, np.ndarray)
            else self.kalman_gain,
            "process_noise": self.process_noise,
            "measurement_noise": self.measurement_noise,
            "burst_count": self.burst_count,
            "is_stable": self._check_stability(),
            "sample_count": len(self.short_history),
            "confidence": self._compute_confidence(),
            "uncertainty": self._compute_uncertainty_measure(),
            "trend_direction": self._get_trend_direction(),
        }
        return result

    def reset(self) -> None:
        """重置状态"""
        self.short_history.clear()
        self.long_history.clear()
        self._single_history.clear()
        self.current_estimate = 0.1
        self.trend = 0.0
        self.variance = 0.0
        self.burst_count = 0
        self.prediction_error.clear()

        # Reset Kalman filter state
        self.x = np.array([0.1, 0.0], dtype=float).reshape(2, 1)
        self.state_cov = np.eye(2, dtype=float) * 0.1
        self.process_noise_cov = (
            np.eye(2, dtype=float) * self.noise_config.base_process_noise
        )
        self.measurement_noise_cov = np.array(
            [[self.noise_config.base_measurement_noise]], dtype=float
        )


class MultiScaleEstimator:
    """
    多尺度融合估计器 - 核心创新

    融合短期和长期估计结果：
    - 短期：高响应速度，适合快速变化的网络
    - 长期：高稳定性，适合缓慢变化的网络
    - 自适应融合权重：根据网络状态动态调整
    """

    def __init__(
        self, short_window: int = 10, long_window: int = 100, initial_value: float = 0.1
    ):
        self.short_estimator: AdaptiveEstimator = AdaptiveEstimator(
            name="short_term",
            short_window=short_window,
            long_window=long_window,
            initial_value=initial_value,
        )

        self.long_estimator: AdaptiveEstimator = AdaptiveEstimator(
            name="long_term",
            short_window=short_window * 2,
            long_window=long_window,
            initial_value=initial_value,
        )

        self.fusion_weights: Dict[str, float] = {"short": 0.5, "long": 0.5}
        self.logger: logging.Logger = logging.getLogger("MultiScaleEstimator")

    def update(self, measurement: float, timestamp: float) -> EstimateResult:
        """更新并融合估计"""
        short_result = self.short_estimator.update(measurement, timestamp)
        long_result = self.long_estimator.update(measurement, timestamp)

        self._update_fusion_weights(short_result, long_result)

        fused_value = self.fusion_weights["short"] * float(
            short_result.value
        ) + self.fusion_weights["long"] * float(long_result.value)

        confidence = (
            self.fusion_weights["short"] * short_result.confidence
            + self.fusion_weights["long"] * long_result.confidence
        )

        uncertainty = (
            self.fusion_weights["short"] * short_result.uncertainty_measure
            + self.fusion_weights["long"] * long_result.uncertainty_measure
        )

        trend = short_result.trend

        lower = min(short_result.confidence_lower, long_result.confidence_lower)
        upper = max(short_result.confidence_upper, long_result.confidence_upper)

        return EstimateResult(
            value=fused_value,
            confidence_lower=lower,
            confidence_upper=upper,
            confidence=confidence,
            is_stable=short_result.is_stable and long_result.is_stable,
            trend=trend,
            uncertainty_measure=uncertainty,
        )

    def _update_fusion_weights(
        self, short_result: EstimateResult, long_result: EstimateResult
    ) -> None:
        """更新融合权重 - 核心创新"""
        short_weight = short_result.confidence
        long_weight = long_result.confidence

        total = short_weight + long_weight
        if total > 0:
            self.fusion_weights["short"] = short_weight / total
            self.fusion_weights["long"] = long_weight / total

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        result: Dict[str, Any] = {
            "fusion_weights": self.fusion_weights,
            "short_estimator": self.short_estimator.get_statistics(),
            "long_estimator": self.long_estimator.get_statistics(),
        }
        return result


class NetworkStateEstimator:
    """
    网络状态综合估计器

    整合多个估计器，提供完整的网络状态估计
    """

    def __init__(self, ablation_config: AblationConfig | None = None):
        self.ablation_config: AblationConfig = ablation_config or AblationConfig()
        self.noise_config: AdaptiveNoiseConfig = AdaptiveNoiseConfig()

        self.loss_estimator: AdaptiveEstimator = AdaptiveEstimator(
            name="loss_rate",
            short_window=10,
            long_window=100,
            initial_value=0.1,
            noise_config=self.noise_config,
            ablation_config=self.ablation_config,
        )
        self.latency_estimator: AdaptiveEstimator = AdaptiveEstimator(
            name="latency",
            short_window=5,
            long_window=50,
            initial_value=50.0,
            noise_config=self.noise_config,
            ablation_config=self.ablation_config,
        )
        self.jitter_estimator: AdaptiveEstimator = AdaptiveEstimator(
            name="jitter",
            short_window=5,
            long_window=30,
            initial_value=5.0,
            noise_config=self.noise_config,
            ablation_config=self.ablation_config,
        )

        self.logger: logging.Logger = logging.getLogger("NetworkStateEstimator")

    def update(self, loss_rate: float, latency: float, jitter: float, timestamp: float):
        """更新所有估计器"""
        loss_result = self.loss_estimator.update(loss_rate, timestamp)
        latency_result = self.latency_estimator.update(latency, timestamp)
        jitter_result = self.jitter_estimator.update(jitter, timestamp)

        return {
            "loss_rate": loss_result,
            "latency": latency_result,
            "jitter": jitter_result,
        }

    def get_network_state(self) -> Dict[str, Any]:
        """获取网络状态摘要"""
        result: Dict[str, Any] = {
            "loss_rate": self.loss_estimator.get_statistics(),
            "latency": self.latency_estimator.get_statistics(),
            "jitter": self.jitter_estimator.get_statistics(),
            "overall_health": self._compute_health_score(),
            "multi_scale_insights": self._generate_insights(),
        }
        return result

    def _compute_health_score(self) -> float:
        """计算网络健康评分"""
        loss_stat = self.loss_estimator.get_statistics()
        latency_stat = self.latency_estimator.get_statistics()

        loss_penalty = min(1.0, loss_stat["current_estimate"] * 5)
        latency_penalty = min(1.0, latency_stat["current_estimate"] / 200)

        health = 1.0 - 0.6 * loss_penalty - 0.4 * latency_penalty
        return float(max(0.0, health))

    def _generate_insights(self) -> Dict[str, Any]:
        """生成网络洞察"""
        loss_stat = self.loss_estimator.get_statistics()
        latency_stat = self.latency_estimator.get_statistics()

        insights: Dict[str, Any] = {
            "loss_trend": loss_stat["trend_direction"],
            "latency_trend": latency_stat["trend_direction"],
            "estimated_reliability": loss_stat["confidence"],
            "estimated_uncertainty": float(loss_stat.get("uncertainty", 0.5)),
            "bursts_detected": bool(loss_stat["burst_count"] > 0),
            "recommendations": [],
        }

        recommendations: list[str] = []

        if loss_stat["trend_direction"] == "rising":
            recommendations.append("丢包率上升，建议监控网络状态")

        if latency_stat["trend_direction"] == "rising":
            recommendations.append("延迟上升，可能存在网络拥塞")

        if loss_stat["burst_count"] > 5:
            recommendations.append("检测到频繁突发，建议检查物理连接")

        insights["recommendations"] = recommendations

        return insights
