#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Continuous Validation - 连续验证系统

创新点：
1. 实时反馈环：生产环境数据自动回传到测试环境
2. 自进化测试集：基于生产观测动态生成新测试用例
3. 异常事件驱动：性能退化和公平性异常触发测试更新
4. 学习增强：利用生产数据改进测试场景生成
5. 渐进式验证：从小规模测试到大规模验证的连续过程
6. 风险感知部署：基于验证结果的智能部署决策

这解决了传统"测试-部署"割裂的问题，实现真正的连续验证。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import time
import threading
from datetime import datetime, timedelta
import json


@dataclass
class ProductionEvent:
    """生产环境事件"""
    timestamp: datetime
    event_type: str  # 'performance_degradation', 'fairness_violation', 'anomaly'
    severity: float  # 0-1, 严重程度
    metrics: Dict[str, Any]  # 相关指标数据
    context: Dict[str, Any]  # 上下文信息
    algorithm_version: str   # 算法版本


@dataclass
class ValidationResult:
    """验证结果"""
    scenario_id: str
    timestamp: datetime
    performance_score: float
    fairness_score: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    recommendations: List[str]
    confidence: float


@dataclass
class ContinuousValidationConfig:
    """连续验证配置"""
    feedback_interval: int = 300  # 反馈间隔(秒)
    learning_rate: float = 0.1    # 学习率
    adaptation_threshold: float = 0.7  # 适应阈值
    risk_tolerance: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.8, 'medium': 0.6, 'high': 0.4, 'critical': 0.2
    })
    max_test_scenarios: int = 1000
    event_retention_days: int = 30
    auto_adaptation: bool = True


class ProductionMonitor:
    """生产环境监控器"""

    def __init__(self, config: ContinuousValidationConfig):
        self.config = config
        self.logger = logging.getLogger("ProductionMonitor")

        # 事件存储
        self.events: deque[ProductionEvent] = deque(maxlen=10000)

        # 监控指标
        self.current_metrics = {}
        self.baseline_metrics = {}

        # 异常检测
        self.anomaly_thresholds = {
            'performance_drop': 0.1,    # 10%性能下降
            'fairness_drop': 0.15,     # 15%公平性下降
            'latency_spike': 100,      # 100ms延迟尖峰
            'packet_loss_spike': 0.05  # 5%丢包率尖峰
        }

    def record_event(self, event: ProductionEvent):
        """记录生产事件"""
        self.events.append(event)
        self.logger.info(f"Recorded production event: {event.event_type} "
                        f"(severity: {event.severity:.3f})")

        # 更新当前指标
        self._update_current_metrics(event)

        # 检查是否需要触发验证
        if self._should_trigger_validation(event):
            self._trigger_validation_cycle(event)

    def _update_current_metrics(self, event: ProductionEvent):
        """更新当前指标"""
        for metric_name, value in event.metrics.items():
            self.current_metrics[metric_name] = value

    def _should_trigger_validation(self, event: ProductionEvent) -> bool:
        """判断是否应该触发验证"""
        # 高严重性事件直接触发
        if event.severity > 0.8:
            return True

        # 性能退化事件
        if event.event_type == 'performance_degradation':
            return event.metrics.get('performance_drop', 0) > self.anomaly_thresholds['performance_drop']

        # 公平性异常事件
        if event.event_type == 'fairness_violation':
            return event.metrics.get('fairness_drop', 0) > self.anomaly_thresholds['fairness_drop']

        # 延迟尖峰
        if event.event_type == 'latency_spike':
            return event.metrics.get('latency_increase', 0) > self.anomaly_thresholds['latency_spike']

        # 丢包尖峰
        if event.event_type == 'packet_loss_spike':
            return event.metrics.get('loss_rate_increase', 0) > self.anomaly_thresholds['packet_loss_spike']

        return False

    def _trigger_validation_cycle(self, triggering_event: ProductionEvent):
        """触发验证周期"""
        self.logger.info(f"Triggering validation cycle due to: {triggering_event.event_type}")

        # 这里会通知ContinuousValidator启动新的验证周期
        # 实际实现中会通过事件系统或回调函数

    def get_recent_events(self, hours: int = 24) -> List[ProductionEvent]:
        """获取最近的事件"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [event for event in self.events if event.timestamp > cutoff_time]

    def analyze_event_patterns(self) -> Dict[str, Any]:
        """分析事件模式"""
        recent_events = self.get_recent_events(24)

        if not recent_events:
            return {}

        # 事件类型统计
        event_types = defaultdict(int)
        severities = []
        timestamps = []

        for event in recent_events:
            event_types[event.event_type] += 1
            severities.append(event.severity)
            timestamps.append(event.timestamp)

        # 时间序列分析
        if len(timestamps) > 1:
            time_diffs = np.diff([t.timestamp() for t in sorted(timestamps)])
            avg_interval = np.mean(time_diffs) if len(time_diffs) > 0 else 0
        else:
            avg_interval = 0

        return {
            'total_events': len(recent_events),
            'event_distribution': dict(event_types),
            'avg_severity': np.mean(severities),
            'max_severity': np.max(severities),
            'avg_event_interval': avg_interval,
            'trend_analysis': self._analyze_trend(severities)
        }

    def _analyze_trend(self, severities: List[float]) -> str:
        """分析趋势"""
        if len(severities) < 5:
            return "insufficient_data"

        # 简单线性趋势分析
        x = np.arange(len(severities))
        slope = np.polyfit(x, severities, 1)[0]

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"


class TestScenarioLearner:
    """测试场景学习器"""

    def __init__(self, config: ContinuousValidationConfig):
        self.config = config
        self.logger = logging.getLogger("TestScenarioLearner")

        # 学习数据
        self.production_events = deque(maxlen=5000)
        self.generated_scenarios = []

        # 学习模型参数
        self.scenario_templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, Dict]:
        """初始化场景模板"""
        return {
            'performance_degradation': {
                'network_conditions': {
                    'loss_rate': (0.05, 0.25),
                    'delay': (50, 150),
                    'jitter': (10, 40)
                },
                'traffic_pattern': 'bursty',
                'trigger_conditions': ['high_load', 'congestion']
            },
            'fairness_violation': {
                'network_conditions': {
                    'loss_rate': (0.02, 0.15),
                    'delay': (20, 80),
                    'jitter': (5, 25)
                },
                'traffic_pattern': 'fluctuating',
                'trigger_conditions': ['resource_contention', 'unequal_distribution']
            },
            'latency_spike': {
                'network_conditions': {
                    'loss_rate': (0.01, 0.10),
                    'delay': (100, 300),
                    'jitter': (20, 60)
                },
                'traffic_pattern': 'periodic',
                'trigger_conditions': ['queue_overflow', 'bgp_flap']
            }
        }

    def learn_from_production_event(self, event: ProductionEvent):
        """从生产事件学习"""
        self.production_events.append(event)

        # 更新场景模板
        self._update_scenario_templates(event)

        # 生成新的测试场景
        new_scenarios = self._generate_scenarios_from_event(event)
        self.generated_scenarios.extend(new_scenarios)

        self.logger.info(f"Learned from {event.event_type} event, generated {len(new_scenarios)} new scenarios")

    def _update_scenario_templates(self, event: ProductionEvent):
        """更新场景模板"""
        event_type = event.event_type

        if event_type in self.scenario_templates:
            template = self.scenario_templates[event_type]

            # 使用指数移动平均更新参数范围
            for metric_name, value in event.metrics.items():
                if metric_name in template['network_conditions']:
                    min_val, max_val = template['network_conditions'][metric_name]
                    # 向观测值靠拢
                    center = (min_val + max_val) / 2
                    new_center = center * (1 - self.config.learning_rate) + value * self.config.learning_rate
                    spread = (max_val - min_val) / 2
                    template['network_conditions'][metric_name] = (
                        new_center - spread,
                        new_center + spread
                    )

    def _generate_scenarios_from_event(self, event: ProductionEvent) -> List[Dict]:
        """从事件生成测试场景"""
        scenarios = []
        event_type = event.event_type

        if event_type not in self.scenario_templates:
            return scenarios

        template = self.scenario_templates[event_type]

        # 生成多个变体场景
        n_variants = min(5, max(1, int(event.severity * 10)))  # 严重程度决定变体数量

        for i in range(n_variants):
            scenario = {
                'id': f"{event_type}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}_{i}",
                'event_type': event_type,
                'severity': event.severity,
                'network_conditions': {},
                'traffic_pattern': template['traffic_pattern'],
                'trigger_conditions': template['trigger_conditions'].copy(),
                'confidence': 0.8  # 基于事件的置信度
            }

            # 从模板生成具体参数
            for param_name, (min_val, max_val) in template['network_conditions'].items():
                # 添加一些随机性
                variation = np.random.normal(0, 0.1)  # 10%变异
                value = np.random.uniform(min_val, max_val) * (1 + variation)
                scenario['network_conditions'][param_name] = max(0, value)

            scenarios.append(scenario)

        return scenarios

    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        if not self.production_events:
            return {}

        event_types = defaultdict(int)
        severities = []

        for event in self.production_events:
            event_types[event.event_type] += 1
            severities.append(event.severity)

        return {
            'total_learned_events': len(self.production_events),
            'event_type_distribution': dict(event_types),
            'avg_severity': np.mean(severities),
            'generated_scenarios': len(self.generated_scenarios),
            'template_updates': len([t for t in self.scenario_templates.values() if t.get('updated', False)])
        }


class ContinuousValidator:
    """
    连续验证器

    核心创新：
    1. 实时反馈环：生产→测试的双向数据流
    2. 自适应测试集：基于生产观测动态演化
    3. 事件驱动验证：异常事件触发验证周期
    4. 风险感知部署：基于验证历史的智能决策
    5. 渐进式验证：从小规模到大规模的连续过程
    """

    def __init__(self, config: Optional[ContinuousValidationConfig] = None):
        self.config = config or ContinuousValidationConfig()
        self.logger = logging.getLogger("ContinuousValidator")

        # 核心组件
        self.production_monitor = ProductionMonitor(self.config)
        self.scenario_learner = TestScenarioLearner(self.config)

        # 验证历史
        self.validation_history: deque[ValidationResult] = deque(maxlen=1000)

        # 验证状态
        self.is_validating = False
        self.last_validation_time = None

        # 线程控制
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = False

    def start_continuous_validation(self):
        """启动连续验证"""
        self.logger.info("Starting continuous validation system")

        # 启动监控线程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        self.logger.info("Continuous validation system started")

    def stop_continuous_validation(self):
        """停止连续验证"""
        self.logger.info("Stopping continuous validation system")
        self.stop_monitoring = True

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

        self.logger.info("Continuous validation system stopped")

    def record_production_event(self, event: ProductionEvent):
        """记录生产事件"""
        self.production_monitor.record_event(event)

        # 同时让学习器学习
        self.scenario_learner.learn_from_production_event(event)

    def trigger_validation_cycle(self, priority: str = "normal"):
        """触发验证周期"""
        if self.is_validating:
            self.logger.warning("Validation already in progress, skipping")
            return

        self.is_validating = True
        self.last_validation_time = datetime.now()

        try:
            self.logger.info(f"Starting validation cycle with priority: {priority}")

            # 获取新生成的测试场景
            new_scenarios = self.scenario_learner.generated_scenarios[-10:]  # 最近10个

            # 执行验证
            results = self._execute_validation_cycle(new_scenarios, priority)

            # 存储结果
            self.validation_history.extend(results)

            # 生成部署建议
            deployment_recommendation = self._generate_deployment_recommendation(results)

            self.logger.info(f"Validation cycle completed. Recommendation: {deployment_recommendation}")

        finally:
            self.is_validating = False

    def _monitoring_loop(self):
        """监控循环"""
        self.logger.info("Monitoring loop started")

        while not self.stop_monitoring:
            try:
                # 检查是否需要触发定期验证
                if self._should_trigger_periodic_validation():
                    self.trigger_validation_cycle("periodic")

                # 检查学习统计
                if len(self.scenario_learner.production_events) % 50 == 0:  # 每50个事件
                    stats = self.scenario_learner.get_learning_statistics()
                    self.logger.info(f"Learning stats: {stats}")

                time.sleep(self.config.feedback_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # 错误时等待更长时间

        self.logger.info("Monitoring loop stopped")

    def _should_trigger_periodic_validation(self) -> bool:
        """判断是否应该触发定期验证"""
        if self.last_validation_time is None:
            return True

        time_since_last = datetime.now() - self.last_validation_time
        return time_since_last.total_seconds() > self.config.feedback_interval

    def _execute_validation_cycle(self, scenarios: List[Dict], priority: str) -> List[ValidationResult]:
        """执行验证周期"""
        results = []

        for scenario in scenarios:
            # 这里应该是实际的验证执行
            # 现在用模拟结果
            result = self._simulate_validation(scenario, priority)
            results.append(result)

        return results

    def _simulate_validation(self, scenario: Dict, priority: str) -> ValidationResult:
        """模拟验证执行"""
        # 基于场景特征生成验证结果
        base_performance = 0.75
        base_fairness = 0.80

        # 根据场景严重性调整
        severity_penalty = scenario.get('severity', 0.5) * 0.2
        performance_score = max(0.1, base_performance - severity_penalty + np.random.normal(0, 0.05))
        fairness_score = max(0.1, base_fairness - severity_penalty + np.random.normal(0, 0.05))

        # 确定风险级别
        avg_score = (performance_score + fairness_score) / 2
        if avg_score > self.config.risk_tolerance['low']:
            risk_level = 'low'
        elif avg_score > self.config.risk_tolerance['medium']:
            risk_level = 'medium'
        elif avg_score > self.config.risk_tolerance['high']:
            risk_level = 'high'
        else:
            risk_level = 'critical'

        # 生成建议
        recommendations = self._generate_recommendations(risk_level, scenario)

        return ValidationResult(
            scenario_id=scenario['id'],
            timestamp=datetime.now(),
            performance_score=performance_score,
            fairness_score=fairness_score,
            risk_level=risk_level,
            recommendations=recommendations,
            confidence=0.85
        )

    def _generate_recommendations(self, risk_level: str, scenario: Dict) -> List[str]:
        """生成建议"""
        recommendations = []

        if risk_level == 'critical':
            recommendations.extend([
                "强烈建议暂停部署",
                "需要进行全面的回归测试",
                f"重点关注{scenario.get('event_type', 'unknown')}场景"
            ])
        elif risk_level == 'high':
            recommendations.extend([
                "建议进行金丝雀部署",
                "增加监控指标",
                "准备回滚计划"
            ])
        elif risk_level == 'medium':
            recommendations.extend([
                "可以进行受控部署",
                "增加监控频率",
                "准备降级策略"
            ])
        else:  # low
            recommendations.extend([
                "可以正常部署",
                "保持标准监控",
                "收集反馈数据"
            ])

        return recommendations

    def _generate_deployment_recommendation(self, results: List[ValidationResult]) -> str:
        """生成部署建议"""
        if not results:
            return "no_data"

        # 统计风险级别
        risk_counts = defaultdict(int)
        for result in results:
            risk_counts[result.risk_level] += 1

        # 如果有critical风险，强烈建议不部署
        if risk_counts.get('critical', 0) > 0:
            return "block_deployment"

        # 如果high风险超过30%，建议谨慎部署
        high_ratio = risk_counts.get('high', 0) / len(results)
        if high_ratio > 0.3:
            return "canary_deployment"

        # 如果medium风险超过50%，建议分阶段部署
        medium_ratio = risk_counts.get('medium', 0) / len(results)
        if medium_ratio > 0.5:
            return "staged_deployment"

        # 其他情况可以正常部署
        return "normal_deployment"

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'is_validating': self.is_validating,
            'last_validation_time': self.last_validation_time.isoformat() if self.last_validation_time else None,
            'total_validations': len(self.validation_history),
            'production_events': len(self.production_monitor.events),
            'generated_scenarios': len(self.scenario_learner.generated_scenarios),
            'learning_stats': self.scenario_learner.get_learning_statistics(),
            'event_analysis': self.production_monitor.analyze_event_patterns()
        }


def create_continuous_validator() -> ContinuousValidator:
    """创建连续验证器"""
    config = ContinuousValidationConfig(
        feedback_interval=300,  # 5分钟
        learning_rate=0.1,
        adaptation_threshold=0.7,
        auto_adaptation=True
    )

    validator = ContinuousValidator(config)
    return validator


# 使用示例
if __name__ == "__main__":
    # 创建连续验证器
    validator = create_continuous_validator()
    validator.start_continuous_validation()

    try:
        # 模拟生产事件
        print("Simulating production events...")

        # 模拟一个性能退化事件
        event1 = ProductionEvent(
            timestamp=datetime.now(),
            event_type='performance_degradation',
            severity=0.7,
            metrics={
                'performance_drop': 0.12,
                'latency_increase': 45,
                'throughput_drop': 0.08
            },
            context={
                'network_load': 'high',
                'algorithm_version': 'v2.1.3'
            },
            algorithm_version='v2.1.3'
        )

        validator.record_production_event(event1)

        # 等待一段时间
        time.sleep(2)

        # 手动触发验证周期
        validator.trigger_validation_cycle("manual")

        # 获取系统状态
        status = validator.get_system_status()
        print("System Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")

    finally:
        validator.stop_continuous_validation()
        print("Continuous validation demo completed")