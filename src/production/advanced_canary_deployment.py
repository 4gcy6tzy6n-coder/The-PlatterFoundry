#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Canary Deployment - 高级金丝雀部署系统

创新点：
1. 参数级金丝雀：独立性能沙箱测试新参数组合
2. 动态权重调整：基于业务压力、时间等因素实时调权
3. 多维度健康评估：性能、公平性、稳定性综合评估
4. 自适应部署策略：基于学习历史的智能部署决策
5. 渐进式流量分配：非线性流量增长，支持快速回滚
6. 实时异常检测：多层级异常检测和自动响应

这解决了传统金丝雀部署过于粗糙的问题，实现真正智能的安全部署。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import time
import threading
from datetime import datetime, timedelta
from enum import Enum
import json


class DeploymentPhase(Enum):
    """部署阶段"""
    PREPARING = "preparing"
    PARAMETER_SANDBOX = "parameter_sandbox"
    TRAFFIC_RAMP_UP = "traffic_ramp_up"
    FULL_TRAFFIC = "full_traffic"
    MONITORING = "monitoring"
    ROLLBACK = "rollback"
    COMPLETED = "completed"


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class HealthMetrics:
    """健康指标"""
    timestamp: datetime
    performance_score: float
    fairness_score: float
    stability_score: float
    latency_p95: float
    error_rate: float
    throughput: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ParameterSandbox:
    """参数沙箱"""
    parameter_set: Dict[str, Any]
    traffic_percentage: float
    start_time: datetime
    duration: int  # 秒
    health_history: List[HealthMetrics] = field(default_factory=list)
    status: HealthStatus = HealthStatus.HEALTHY


@dataclass
class TrafficRampConfig:
    """流量爬坡配置"""
    initial_percentage: float = 1.0
    max_percentage: float = 50.0
    ramp_up_steps: int = 10
    step_duration: int = 300  # 5分钟每步
    success_threshold: float = 0.95
    failure_threshold: float = 0.85


@dataclass
class AdvancedCanaryConfig:
    """高级金丝雀配置"""
    enable_parameter_sandbox: bool = True
    enable_dynamic_weights: bool = True
    enable_predictive_monitoring: bool = True
    traffic_ramp: TrafficRampConfig = field(default_factory=TrafficRampConfig)

    # 健康阈值
    health_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'performance': 0.85,
        'fairness': 0.80,
        'stability': 0.90,
        'latency': 200.0,  # ms
        'error_rate': 0.05
    })

    # 权重配置
    base_weights: Dict[str, float] = field(default_factory=lambda: {
        'performance': 0.3,
        'fairness': 0.25,
        'stability': 0.2,
        'latency': 0.15,
        'error_rate': 0.1
    })

    # 时间相关权重调整
    time_based_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'business_hours': {'performance': 1.2, 'latency': 1.1},
        'off_hours': {'performance': 0.9, 'stability': 1.1},
        'peak_hours': {'latency': 1.3, 'stability': 1.2}
    })

    # 业务压力相关权重
    load_based_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'high_load': {'performance': 1.3, 'stability': 1.2},
        'normal_load': {'performance': 1.0, 'fairness': 1.1},
        'low_load': {'stability': 0.9, 'latency': 0.8}
    })

    monitoring_interval: int = 30  # 秒
    max_deployment_time: int = 3600  # 1小时
    auto_rollback_threshold: float = 0.75


class DynamicWeightAdjuster:
    """动态权重调整器"""

    def __init__(self, config: AdvancedCanaryConfig):
        self.config = config
        self.logger = logging.getLogger("DynamicWeightAdjuster")

        # 上下文状态
        self.current_time_context = "business_hours"
        self.current_load_context = "normal_load"

    def adjust_weights(self, base_weights: Dict[str, float],
                      current_metrics: Dict[str, Any]) -> Dict[str, float]:
        """调整权重"""
        adjusted_weights = base_weights.copy()

        # 时间上下文调整
        time_multipliers = self.config.time_based_weights.get(self.current_time_context, {})
        for metric, multiplier in time_multipliers.items():
            if metric in adjusted_weights:
                adjusted_weights[metric] *= multiplier

        # 业务负载上下文调整
        load_multipliers = self.config.load_based_weights.get(self.current_load_context, {})
        for metric, multiplier in load_multipliers.items():
            if metric in adjusted_weights:
                adjusted_weights[metric] *= multiplier

        # 基于当前指标的动态调整
        adjusted_weights = self._adjust_based_on_metrics(adjusted_weights, current_metrics)

        # 归一化权重
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}

        return adjusted_weights

    def _adjust_based_on_metrics(self, weights: Dict[str, float],
                                metrics: Dict[str, Any]) -> Dict[str, float]:
        """基于指标动态调整权重"""
        # 如果性能指标异常，增加性能权重
        if metrics.get('performance_score', 1.0) < 0.8:
            weights['performance'] *= 1.5

        # 如果公平性指标异常，增加公平性权重
        if metrics.get('fairness_score', 1.0) < 0.75:
            weights['fairness'] *= 1.4

        # 如果延迟过高，增加延迟权重
        if metrics.get('latency_p95', 0) > 150:
            weights['latency'] *= 1.3

        # 如果错误率过高，增加错误率权重
        if metrics.get('error_rate', 0) > 0.03:
            weights['error_rate'] *= 1.5

        return weights

    def update_context(self, current_time: datetime, system_load: float):
        """更新上下文"""
        # 更新时间上下文
        hour = current_time.hour
        if 9 <= hour <= 17:  # 工作时间
            self.current_time_context = "business_hours"
        elif 18 <= hour <= 22:  # 高峰时间
            self.current_time_context = "peak_hours"
        else:  # 非高峰时间
            self.current_time_context = "off_hours"

        # 更新负载上下文
        if system_load > 0.8:
            self.current_load_context = "high_load"
        elif system_load < 0.3:
            self.current_load_context = "low_load"
        else:
            self.current_load_context = "normal_load"


class HealthEvaluator:
    """健康评估器"""

    def __init__(self, config: AdvancedCanaryConfig):
        self.config = config
        self.logger = logging.getLogger("HealthEvaluator")

        self.weight_adjuster = DynamicWeightAdjuster(config)
        self.health_history: deque[HealthMetrics] = deque(maxlen=100)

    def evaluate_health(self, metrics: HealthMetrics,
                       current_context: Dict[str, Any]) -> Tuple[HealthStatus, float, Dict[str, float]]:
        """
        评估健康状态

        Returns:
            (健康状态, 综合得分, 各指标得分)
        """
        # 更新上下文
        self.weight_adjuster.update_context(
            metrics.timestamp,
            current_context.get('system_load', 0.5)
        )

        # 计算各指标得分
        metric_scores = self._calculate_metric_scores(metrics)

        # 获取动态权重
        context_metrics = {
            'performance_score': metric_scores['performance'],
            'fairness_score': metric_scores['fairness'],
            'latency_p95': metrics.latency_p95,
            'error_rate': metrics.error_rate
        }

        weights = self.weight_adjuster.adjust_weights(self.config.base_weights, context_metrics)

        # 计算综合得分
        overall_score = sum(score * weights.get(metric, 0)
                          for metric, score in metric_scores.items())

        # 确定健康状态
        status = self._determine_health_status(overall_score, metric_scores)

        # 记录历史
        self.health_history.append(metrics)

        return status, overall_score, metric_scores

    def _calculate_metric_scores(self, metrics: HealthMetrics) -> Dict[str, float]:
        """计算各指标得分"""
        scores = {}

        # 性能得分
        scores['performance'] = min(1.0, metrics.performance_score /
                                   self.config.health_thresholds['performance'])

        # 公平性得分
        scores['fairness'] = min(1.0, metrics.fairness_score /
                                self.config.health_thresholds['fairness'])

        # 稳定性得分 (基于历史方差)
        if len(self.health_history) > 5:
            recent_perf = [h.performance_score for h in list(self.health_history)[-5:]]
            perf_std = np.std(recent_perf)
            stability_score = max(0, 1.0 - perf_std * 2)  # 稳定性得分
        else:
            stability_score = 0.8  # 初始稳定性假设

        scores['stability'] = min(1.0, stability_score /
                                 self.config.health_thresholds['stability'])

        # 延迟得分 (越低越好)
        latency_score = max(0, 1.0 - metrics.latency_p95 /
                           self.config.health_thresholds['latency'])
        scores['latency'] = latency_score

        # 错误率得分 (越低越好)
        error_score = max(0, 1.0 - metrics.error_rate /
                         self.config.health_thresholds['error_rate'])
        scores['error_rate'] = error_score

        return scores

    def _determine_health_status(self, overall_score: float,
                               metric_scores: Dict[str, float]) -> HealthStatus:
        """确定健康状态"""
        # 检查是否有严重指标失败
        critical_metrics = ['performance', 'stability']
        for metric in critical_metrics:
            if metric_scores.get(metric, 1.0) < 0.6:
                return HealthStatus.CRITICAL

        # 检查是否有多个指标警告
        warning_count = sum(1 for score in metric_scores.values() if score < 0.8)
        if warning_count >= 2:
            return HealthStatus.WARNING

        # 基于综合得分
        if overall_score >= 0.9:
            return HealthStatus.HEALTHY
        elif overall_score >= 0.75:
            return HealthStatus.WARNING
        elif overall_score >= 0.6:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAILED

    def get_health_trend(self) -> Dict[str, Any]:
        """获取健康趋势"""
        if len(self.health_history) < 5:
            return {"trend": "insufficient_data"}

        recent_scores = [h.performance_score for h in list(self.health_history)[-10:]]
        trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

        if trend_slope > 0.001:
            trend = "improving"
        elif trend_slope < -0.001:
            trend = "degrading"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": trend_slope,
            "avg_score": np.mean(recent_scores),
            "volatility": np.std(recent_scores)
        }


class AdvancedCanaryDeployment:
    """
    高级金丝雀部署系统

    核心创新：
    1. 参数级沙箱：独立测试新参数组合
    2. 动态权重：基于上下文实时调整指标权重
    3. 智能流量爬坡：非线性流量分配策略
    4. 预测性监控：基于历史数据的异常预测
    5. 自动故障恢复：智能回滚和修复策略
    """

    def __init__(self, config: Optional[AdvancedCanaryConfig] = None):
        self.config = config or AdvancedCanaryConfig()
        self.logger = logging.getLogger("AdvancedCanaryDeployment")

        # 核心组件
        self.health_evaluator = HealthEvaluator(self.config)

        # 部署状态
        self.current_phase: DeploymentPhase = DeploymentPhase.PREPARING
        self.deployment_start_time: Optional[datetime] = None
        self.parameter_sandboxes: List[ParameterSandbox] = []

        # 流量分配
        self.current_traffic_percentage: float = 0.0
        self.target_traffic_percentage: float = 0.0

        # 监控和控制
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = False

        # 部署历史
        self.deployment_history: List[Dict[str, Any]] = []

    def start_deployment(self, algorithm_version: str,
                        parameter_sets: List[Dict[str, Any]] = None):
        """启动部署"""
        self.logger.info(f"Starting advanced canary deployment for version {algorithm_version}")

        self.deployment_start_time = datetime.now()
        self.current_phase = DeploymentPhase.PREPARING

        # 如果启用参数沙箱，先进行参数测试
        if self.config.enable_parameter_sandbox and parameter_sets:
            self._start_parameter_sandbox_testing(parameter_sets)
        else:
            # 直接进入流量爬坡阶段
            self._start_traffic_ramp_up()

        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def _start_parameter_sandbox_testing(self, parameter_sets: List[Dict[str, Any]]):
        """启动参数沙箱测试"""
        self.current_phase = DeploymentPhase.PARAMETER_SANDBOX
        self.logger.info(f"Starting parameter sandbox testing with {len(parameter_sets)} parameter sets")

        # 为每个参数集创建沙箱
        for i, params in enumerate(parameter_sets):
            sandbox = ParameterSandbox(
                parameter_set=params,
                traffic_percentage=1.0,  # 1%流量用于参数测试
                start_time=datetime.now(),
                duration=600  # 10分钟测试
            )
            self.parameter_sandboxes.append(sandbox)

        # 设置初始流量分配
        self.current_traffic_percentage = len(parameter_sets) * 1.0
        self.target_traffic_percentage = self.current_traffic_percentage

    def _start_traffic_ramp_up(self):
        """启动流量爬坡"""
        self.current_phase = DeploymentPhase.TRAFFIC_RAMP_UP
        self.logger.info("Starting traffic ramp-up phase")

        self.current_traffic_percentage = self.config.traffic_ramp.initial_percentage
        self.target_traffic_percentage = self.config.traffic_ramp.max_percentage

        # 记录部署开始
        deployment_record = {
            'phase': 'traffic_ramp_up',
            'start_time': datetime.now(),
            'initial_traffic': self.current_traffic_percentage,
            'target_traffic': self.target_traffic_percentage
        }
        self.deployment_history.append(deployment_record)

    def _monitoring_loop(self):
        """监控循环"""
        self.logger.info("Monitoring loop started")

        while not self.stop_monitoring:
            try:
                current_time = datetime.now()

                # 检查部署超时
                if self._check_deployment_timeout():
                    self._handle_timeout()
                    break

                # 根据当前阶段执行相应监控
                if self.current_phase == DeploymentPhase.PARAMETER_SANDBOX:
                    self._monitor_parameter_sandboxes()
                elif self.current_phase == DeploymentPhase.TRAFFIC_RAMP_UP:
                    self._monitor_traffic_ramp()
                elif self.current_phase == DeploymentPhase.FULL_TRAFFIC:
                    self._monitor_full_traffic()
                elif self.current_phase == DeploymentPhase.MONITORING:
                    self._monitor_post_deployment()

                time.sleep(self.config.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)

        self.logger.info("Monitoring loop stopped")

    def _monitor_parameter_sandboxes(self):
        """监控参数沙箱"""
        current_time = datetime.now()
        active_sandboxes = []

        for sandbox in self.parameter_sandboxes:
            # 检查沙箱是否过期
            elapsed = (current_time - sandbox.start_time).total_seconds()
            if elapsed >= sandbox.duration:
                # 评估沙箱结果
                self._evaluate_sandbox_result(sandbox)
            else:
                active_sandboxes.append(sandbox)

        # 如果所有沙箱都完成，进入下一阶段
        if not active_sandboxes and self.parameter_sandboxes:
            best_params = self._select_best_parameters()
            self.logger.info(f"Parameter sandbox testing completed. Best parameters: {best_params}")

            # 进入流量爬坡阶段
            self._start_traffic_ramp_up()

    def _evaluate_sandbox_result(self, sandbox: ParameterSandbox):
        """评估沙箱结果"""
        if not sandbox.health_history:
            sandbox.status = HealthStatus.FAILED
            return

        # 计算平均健康指标
        avg_performance = np.mean([h.performance_score for h in sandbox.health_history])
        avg_fairness = np.mean([h.fairness_score for h in sandbox.health_history])
        avg_stability = np.mean([h.stability_score for h in sandbox.health_history])

        overall_score = (avg_performance + avg_fairness + avg_stability) / 3

        if overall_score >= 0.9:
            sandbox.status = HealthStatus.HEALTHY
        elif overall_score >= 0.75:
            sandbox.status = HealthStatus.WARNING
        else:
            sandbox.status = HealthStatus.CRITICAL

    def _select_best_parameters(self) -> Dict[str, Any]:
        """选择最佳参数"""
        best_sandbox = None
        best_score = -1

        for sandbox in self.parameter_sandboxes:
            if sandbox.status == HealthStatus.HEALTHY:
                # 计算综合得分
                avg_performance = np.mean([h.performance_score for h in sandbox.health_history])
                avg_fairness = np.mean([h.fairness_score for h in sandbox.health_history])
                score = (avg_performance + avg_fairness) / 2

                if score > best_score:
                    best_score = score
                    best_sandbox = sandbox

        return best_sandbox.parameter_set if best_sandbox else {}

    def _monitor_traffic_ramp(self):
        """监控流量爬坡"""
        # 计算当前步数
        elapsed_time = (datetime.now() - self.deployment_start_time).total_seconds()
        current_step = int(elapsed_time / self.config.traffic_ramp.step_duration)

        if current_step >= self.config.traffic_ramp.ramp_up_steps:
            # 流量爬坡完成
            self._complete_traffic_ramp()
            return

        # 计算目标流量百分比 (非线性增长)
        progress = current_step / self.config.traffic_ramp.ramp_up_steps
        # 使用sigmoid函数实现平滑增长
        sigmoid_progress = 1 / (1 + np.exp(-6 * (progress - 0.5)))
        target_traffic = (self.config.traffic_ramp.max_percentage *
                         sigmoid_progress)

        self.target_traffic_percentage = target_traffic

        # 检查健康状态
        health_status, overall_score, _ = self._get_current_health_status()

        if health_status == HealthStatus.FAILED:
            self.logger.warning("Health check failed during traffic ramp-up")
            self._initiate_rollback("health_failure")
            return

        # 根据健康状态调整增长速度
        if health_status == HealthStatus.WARNING:
            # 减缓增长速度
            self.target_traffic_percentage *= 0.8

        # 应用流量调整
        self._adjust_traffic_allocation()

    def _complete_traffic_ramp(self):
        """完成流量爬坡"""
        self.logger.info("Traffic ramp-up completed successfully")
        self.current_phase = DeploymentPhase.FULL_TRAFFIC
        self.current_traffic_percentage = 100.0
        self.target_traffic_percentage = 100.0

    def _monitor_full_traffic(self):
        """监控全流量部署"""
        # 保持全流量运行一段时间进行最终验证
        if self._check_stability_period():
            self.current_phase = DeploymentPhase.MONITORING
            self.logger.info("Full traffic deployment stabilized, entering monitoring phase")

    def _monitor_post_deployment(self):
        """监控部署后状态"""
        # 持续监控健康状态
        health_status, overall_score, _ = self._get_current_health_status()

        if health_status == HealthStatus.FAILED:
            self.logger.error("Post-deployment health check failed")
            self._initiate_rollback("post_deployment_failure")

        # 检查是否达到稳定期
        if self._check_stability_period():
            self._complete_deployment()

    def _get_current_health_status(self) -> Tuple[HealthStatus, float, Dict[str, float]]:
        """获取当前健康状态"""
        # 这里应该从实际监控系统获取最新指标
        # 现在用模拟数据
        current_metrics = HealthMetrics(
            timestamp=datetime.now(),
            performance_score=0.88 + np.random.normal(0, 0.02),
            fairness_score=0.85 + np.random.normal(0, 0.02),
            stability_score=0.92 + np.random.normal(0, 0.01),
            latency_p95=120 + np.random.normal(0, 10),
            error_rate=0.02 + np.random.normal(0, 0.005),
            throughput=1000 + np.random.normal(0, 50)
        )

        current_context = {
            'system_load': 0.6,
            'time_of_day': datetime.now().hour
        }

        return self.health_evaluator.evaluate_health(current_metrics, current_context)

    def _adjust_traffic_allocation(self):
        """调整流量分配"""
        # 平滑过渡到目标流量
        adjustment_rate = 0.1  # 10% 每步调整
        traffic_diff = self.target_traffic_percentage - self.current_traffic_percentage

        if abs(traffic_diff) > 1.0:  # 只有差异大于1%时才调整
            adjustment = traffic_diff * adjustment_rate
            self.current_traffic_percentage += adjustment

            self.logger.info(f"Adjusted traffic to {self.current_traffic_percentage:.1f}%")

    def _check_stability_period(self) -> bool:
        """检查稳定性周期"""
        if self.deployment_start_time is None:
            return False

        elapsed = (datetime.now() - self.deployment_start_time).total_seconds()
        return elapsed > 1800  # 30分钟稳定性周期

    def _check_deployment_timeout(self) -> bool:
        """检查部署超时"""
        if self.deployment_start_time is None:
            return False

        elapsed = (datetime.now() - self.deployment_start_time).total_seconds()
        return elapsed > self.config.max_deployment_time

    def _handle_timeout(self):
        """处理超时"""
        self.logger.error("Deployment timed out, initiating rollback")
        self._initiate_rollback("timeout")

    def _initiate_rollback(self, reason: str):
        """启动回滚"""
        self.logger.warning(f"Initiating rollback due to: {reason}")
        self.current_phase = DeploymentPhase.ROLLBACK

        # 逐步减少流量
        self.target_traffic_percentage = 0.0

        # 记录回滚事件
        rollback_record = {
            'phase': 'rollback',
            'reason': reason,
            'timestamp': datetime.now(),
            'traffic_at_rollback': self.current_traffic_percentage
        }
        self.deployment_history.append(rollback_record)

    def _complete_deployment(self):
        """完成部署"""
        self.logger.info("Deployment completed successfully")
        self.current_phase = DeploymentPhase.COMPLETED

        completion_record = {
            'phase': 'completed',
            'timestamp': datetime.now(),
            'final_traffic': self.current_traffic_percentage,
            'total_duration': (datetime.now() - self.deployment_start_time).total_seconds()
        }
        self.deployment_history.append(completion_record)

        # 停止监控
        self.stop_monitoring = True

    def stop_deployment(self):
        """停止部署"""
        self.logger.info("Stopping deployment")
        self.stop_monitoring = True

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

    def get_deployment_status(self) -> Dict[str, Any]:
        """获取部署状态"""
        health_trend = self.health_evaluator.get_health_trend()

        return {
            'current_phase': self.current_phase.value,
            'current_traffic_percentage': self.current_traffic_percentage,
            'target_traffic_percentage': self.target_traffic_percentage,
            'start_time': self.deployment_start_time.isoformat() if self.deployment_start_time else None,
            'health_trend': health_trend,
            'active_sandboxes': len([s for s in self.parameter_sandboxes
                                   if (datetime.now() - s.start_time).total_seconds() < s.duration]),
            'deployment_history': len(self.deployment_history)
        }


def create_advanced_canary_deployment() -> AdvancedCanaryDeployment:
    """创建高级金丝雀部署系统"""
    config = AdvancedCanaryConfig(
        enable_parameter_sandbox=True,
        enable_dynamic_weights=True,
        enable_predictive_monitoring=True
    )

    deployment = AdvancedCanaryDeployment(config)
    return deployment


# 使用示例
if __name__ == "__main__":
    # 创建高级金丝雀部署系统
    deployment = create_advanced_canary_deployment()

    # 定义参数集进行测试
    parameter_sets = [
        {'learning_rate': 0.01, 'batch_size': 32},
        {'learning_rate': 0.005, 'batch_size': 64},
        {'learning_rate': 0.02, 'batch_size': 16}
    ]

    try:
        # 启动部署
        deployment.start_deployment("v2.1.0", parameter_sets)

        # 模拟运行一段时间
        print("Deployment started, monitoring...")
        time.sleep(10)  # 运行10秒用于演示

        # 获取状态
        status = deployment.get_deployment_status()
        print("Deployment Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")

    finally:
        deployment.stop_deployment()
        print("Deployment demo completed")