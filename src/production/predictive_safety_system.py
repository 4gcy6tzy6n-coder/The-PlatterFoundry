#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predictive Safety System - 预测性安全系统

创新点：
1. 风险预测模型：基于历史数据预测回滚概率
2. 多维度风险评估：性能、公平性、稳定性综合风险评分
3. 主动风险识别：识别潜在风险点并提供缓解建议
4. 历史学习机制：从部署历史中持续学习和改进
5. 实时风险监控：部署过程中的动态风险评估
6. 决策支持系统：为部署决策提供量化依据

这解决了传统被动监控的问题，实现真正预测性的安全保障。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json


@dataclass
class DeploymentRecord:
    """部署记录"""
    algorithm_version: str
    deployment_time: datetime
    rollback_occurred: bool
    rollback_time: Optional[datetime]
    rollback_reason: Optional[str]
    canary_duration: int  # 金丝雀持续时间(秒)
    max_traffic_percentage: float
    final_status: str  # 'success', 'rolled_back', 'partial_success'

    # 性能指标
    avg_performance_score: float
    avg_fairness_score: float
    avg_stability_score: float
    peak_latency: float
    avg_error_rate: float

    # 上下文信息
    network_conditions: Dict[str, Any]
    system_load: float
    deployment_hour: int
    is_business_hours: bool

    # 风险指标
    risk_score: float = 0.0
    risk_factors: List[str] = field(default_factory=list)


@dataclass
class RiskPrediction:
    """风险预测结果"""
    rollback_probability: float
    risk_score: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    primary_risk_factors: List[str]
    mitigation_suggestions: List[str]
    confidence_interval: Tuple[float, float]
    recommended_actions: List[str]


@dataclass
class PredictiveSafetyConfig:
    """预测性安全配置"""
    model_update_interval: int = 3600  # 模型更新间隔(秒)
    min_training_samples: int = 50     # 最少训练样本数
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.2, 'medium': 0.4, 'high': 0.6, 'critical': 0.8
    })
    feature_importance_threshold: float = 0.1
    enable_online_learning: bool = True
    model_save_path: str = "models/risk_prediction_model.pkl"


class FeatureEngineer:
    """特征工程器"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.logger = logging.getLogger("FeatureEngineer")

    def extract_features(self, deployment_record: DeploymentRecord) -> Dict[str, float]:
        """从部署记录中提取特征"""
        features = {}

        # 基础性能特征
        features['avg_performance_score'] = deployment_record.avg_performance_score
        features['avg_fairness_score'] = deployment_record.avg_fairness_score
        features['avg_stability_score'] = deployment_record.avg_stability_score
        features['peak_latency'] = deployment_record.peak_latency
        features['avg_error_rate'] = deployment_record.avg_error_rate

        # 时间相关特征
        features['canary_duration'] = deployment_record.canary_duration
        features['max_traffic_percentage'] = deployment_record.max_traffic_percentage
        features['deployment_hour'] = deployment_record.deployment_hour
        features['is_business_hours'] = 1.0 if deployment_record.is_business_hours else 0.0

        # 系统负载特征
        features['system_load'] = deployment_record.system_load

        # 网络条件特征
        network = deployment_record.network_conditions
        features['network_loss_rate'] = network.get('loss_rate', 0.01)
        features['network_delay'] = network.get('delay', 20.0)
        features['network_jitter'] = network.get('jitter', 2.0)
        features['network_load'] = network.get('load', 0.5)

        # 派生特征
        features['performance_fairness_ratio'] = (
            features['avg_performance_score'] / (features['avg_fairness_score'] + 1e-6)
        )
        features['stability_variability'] = (
            1.0 - features['avg_stability_score']
        ) * features['peak_latency']

        features['load_pressure_index'] = (
            features['system_load'] * features['network_load']
        )

        features['time_risk_factor'] = (
            1.0 if features['is_business_hours'] else 0.5
        ) * (1.0 + features['deployment_hour'] / 24.0)

        return features

    def prepare_training_data(self, deployment_records: List[DeploymentRecord]) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        if len(deployment_records) < 10:
            raise ValueError("Insufficient training data")

        features_list = []
        labels = []

        for record in deployment_records:
            features = self.extract_features(record)
            features_list.append(list(features.values()))
            labels.append(1.0 if record.rollback_occurred else 0.0)

        X = np.array(features_list)
        y = np.array(labels)

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y

    def transform_features(self, deployment_record: DeploymentRecord) -> np.ndarray:
        """转换特征为模型输入格式"""
        features = self.extract_features(deployment_record)
        feature_values = np.array(list(features.values())).reshape(1, -1)
        return self.scaler.transform(feature_values)


class RiskPredictionModel:
    """风险预测模型"""

    def __init__(self, config: PredictiveSafetyConfig):
        self.config = config
        self.logger = logging.getLogger("RiskPredictionModel")

        # 模型组件
        self.rollback_classifier: Optional[RandomForestClassifier] = None
        self.risk_regressor: Optional[GradientBoostingRegressor] = None
        self.feature_engineer = FeatureEngineer()

        # 训练数据
        self.training_records: deque[DeploymentRecord] = deque(maxlen=1000)

        # 模型状态
        self.last_update_time: Optional[datetime] = None
        self.is_trained = False

    def add_training_record(self, record: DeploymentRecord):
        """添加训练记录"""
        self.training_records.append(record)

        # 检查是否需要更新模型
        if self._should_update_model():
            self.update_model()

    def update_model(self):
        """更新模型"""
        if len(self.training_records) < self.config.min_training_samples:
            self.logger.warning(f"Insufficient training samples: {len(self.training_records)}")
            return

        try:
            # 准备训练数据
            X, y_rollback = self.feature_engineer.prepare_training_data(list(self.training_records))

            # 训练回滚分类器
            self.rollback_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.rollback_classifier.fit(X, y_rollback)

            # 训练风险回归器 (使用性能指标作为风险代理)
            risk_scores = np.array([
                1.0 - (r.avg_performance_score + r.avg_fairness_score + r.avg_stability_score) / 3.0
                for r in self.training_records
            ])

            self.risk_regressor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            self.risk_regressor.fit(X, risk_scores)

            self.is_trained = True
            self.last_update_time = datetime.now()

            self.logger.info("Risk prediction model updated successfully")

        except Exception as e:
            self.logger.error(f"Failed to update model: {e}")

    def predict_risk(self, deployment_record: DeploymentRecord) -> RiskPrediction:
        """预测风险"""
        if not self.is_trained:
            # 如果模型未训练，使用基于规则的预测
            return self._rule_based_prediction(deployment_record)

        try:
            # 提取特征
            X = self.feature_engineer.transform_features(deployment_record)

            # 预测回滚概率
            rollback_prob = self.rollback_classifier.predict_proba(X)[0][1]

            # 预测风险评分
            risk_score = self.risk_regressor.predict(X)[0]
            risk_score = np.clip(risk_score, 0, 1)

            # 确定风险等级
            risk_level = self._determine_risk_level(risk_score)

            # 识别主要风险因素
            risk_factors = self._identify_risk_factors(X[0])

            # 生成缓解建议
            mitigation_suggestions = self._generate_mitigation_suggestions(risk_factors, deployment_record)

            # 计算置信区间
            confidence_interval = self._calculate_confidence_interval(rollback_prob, risk_score)

            # 生成推荐行动
            recommended_actions = self._generate_recommended_actions(risk_level, risk_factors)

            return RiskPrediction(
                rollback_probability=rollback_prob,
                risk_score=risk_score,
                risk_level=risk_level,
                primary_risk_factors=risk_factors,
                mitigation_suggestions=mitigation_suggestions,
                confidence_interval=confidence_interval,
                recommended_actions=recommended_actions
            )

        except Exception as e:
            self.logger.error(f"Risk prediction failed: {e}")
            return self._rule_based_prediction(deployment_record)

    def _rule_based_prediction(self, record: DeploymentRecord) -> RiskPrediction:
        """基于规则的风险预测 (当模型未训练时使用)"""
        risk_score = 0.0

        # 基于性能指标计算基础风险
        if record.avg_performance_score < 0.8:
            risk_score += 0.3
        if record.avg_fairness_score < 0.75:
            risk_score += 0.25
        if record.avg_stability_score < 0.85:
            risk_score += 0.2
        if record.peak_latency > 150:
            risk_score += 0.15
        if record.avg_error_rate > 0.03:
            risk_score += 0.1

        # 时间因素
        if record.is_business_hours:
            risk_score += 0.1

        # 系统负载因素
        risk_score += record.system_load * 0.1

        risk_score = min(risk_score, 1.0)
        risk_level = self._determine_risk_level(risk_score)

        return RiskPrediction(
            rollback_probability=risk_score * 0.7,  # 粗略估计
            risk_score=risk_score,
            risk_level=risk_level,
            primary_risk_factors=["rule_based_assessment"],
            mitigation_suggestions=["Increase monitoring", "Prepare rollback plan"],
            confidence_interval=(risk_score * 0.8, risk_score * 1.2),
            recommended_actions=["Enhanced monitoring", "Gradual rollout"]
        )

    def _determine_risk_level(self, risk_score: float) -> str:
        """确定风险等级"""
        if risk_score <= self.config.risk_thresholds['low']:
            return 'low'
        elif risk_score <= self.config.risk_thresholds['medium']:
            return 'medium'
        elif risk_score <= self.config.risk_thresholds['high']:
            return 'high'
        else:
            return 'critical'

    def _identify_risk_factors(self, feature_values: np.ndarray) -> List[str]:
        """识别主要风险因素"""
        if self.rollback_classifier is None:
            return ["unknown"]

        # 获取特征重要性
        feature_importance = self.rollback_classifier.feature_importances_

        # 找到最重要的特征
        important_indices = np.where(feature_importance > self.config.feature_importance_threshold)[0]

        risk_factors = []
        feature_names = [
            'avg_performance_score', 'avg_fairness_score', 'avg_stability_score',
            'peak_latency', 'avg_error_rate', 'canary_duration', 'max_traffic_percentage',
            'deployment_hour', 'is_business_hours', 'system_load',
            'network_loss_rate', 'network_delay', 'network_jitter', 'network_load',
            'performance_fairness_ratio', 'stability_variability',
            'load_pressure_index', 'time_risk_factor'
        ]

        for idx in important_indices:
            if idx < len(feature_names):
                risk_factors.append(feature_names[idx])

        return risk_factors[:5]  # 返回前5个最重要的因素

    def _generate_mitigation_suggestions(self, risk_factors: List[str],
                                       record: DeploymentRecord) -> List[str]:
        """生成缓解建议"""
        suggestions = []

        if 'avg_performance_score' in risk_factors:
            suggestions.append("Optimize algorithm performance before deployment")

        if 'avg_fairness_score' in risk_factors:
            suggestions.append("Review and improve fairness constraints")

        if 'peak_latency' in risk_factors:
            suggestions.append("Implement latency optimization measures")

        if 'system_load' in risk_factors:
            suggestions.append("Schedule deployment during low load periods")

        if 'is_business_hours' in risk_factors:
            suggestions.append("Consider off-hours deployment")

        if 'network_loss_rate' in risk_factors:
            suggestions.append("Test under various network loss conditions")

        if not suggestions:
            suggestions.append("Increase monitoring and prepare rollback plan")

        return suggestions

    def _calculate_confidence_interval(self, rollback_prob: float,
                                    risk_score: float) -> Tuple[float, float]:
        """计算置信区间"""
        # 基于模型的不确定性估算置信区间
        uncertainty = 0.1  # 简化假设

        lower = max(0, risk_score - uncertainty)
        upper = min(1, risk_score + uncertainty)

        return (lower, upper)

    def _generate_recommended_actions(self, risk_level: str,
                                    risk_factors: List[str]) -> List[str]:
        """生成推荐行动"""
        actions = []

        if risk_level == 'critical':
            actions.extend([
                "Block deployment",
                "Require additional testing",
                "Senior review required"
            ])
        elif risk_level == 'high':
            actions.extend([
                "Canary deployment only",
                "Extended monitoring period",
                "Prepare immediate rollback"
            ])
        elif risk_level == 'medium':
            actions.extend([
                "Gradual rollout",
                "Enhanced monitoring",
                "Stakeholder notification"
            ])
        else:  # low
            actions.extend([
                "Normal deployment",
                "Standard monitoring",
                "Post-deployment review"
            ])

        return actions

    def _should_update_model(self) -> bool:
        """判断是否应该更新模型"""
        if self.last_update_time is None:
            return True

        elapsed = (datetime.now() - self.last_update_time).total_seconds()
        return elapsed >= self.config.model_update_interval

    def get_model_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        if not self.is_trained:
            return {"status": "not_trained"}

        stats = {
            "status": "trained",
            "training_samples": len(self.training_records),
            "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
            "feature_importance": {}
        }

        if self.rollback_classifier:
            feature_names = [
                'avg_performance_score', 'avg_fairness_score', 'avg_stability_score',
                'peak_latency', 'avg_error_rate', 'canary_duration', 'max_traffic_percentage',
                'deployment_hour', 'is_business_hours', 'system_load',
                'network_loss_rate', 'network_delay', 'network_jitter', 'network_load',
                'performance_fairness_ratio', 'stability_variability',
                'load_pressure_index', 'time_risk_factor'
            ]

            for name, importance in zip(feature_names, self.rollback_classifier.feature_importances_):
                if importance > self.config.feature_importance_threshold:
                    stats["feature_importance"][name] = float(importance)

        return stats

    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_trained:
            return

        model_data = {
            'rollback_classifier': self.rollback_classifier,
            'risk_regressor': self.risk_regressor,
            'scaler': self.feature_engineer.scaler,
            'last_update_time': self.last_update_time,
            'training_records_count': len(self.training_records)
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> bool:
        """加载模型"""
        try:
            model_data = joblib.load(filepath)

            self.rollback_classifier = model_data['rollback_classifier']
            self.risk_regressor = model_data['risk_regressor']
            self.feature_engineer.scaler = model_data['scaler']
            self.last_update_time = model_data['last_update_time']

            self.is_trained = True
            self.logger.info(f"Model loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False


class PredictiveSafetySystem:
    """
    预测性安全系统

    核心创新：
    1. 主动风险预测：在部署前量化风险
    2. 历史学习：从部署历史中持续改进
    3. 多维度评估：性能、公平性、稳定性综合考虑
    4. 决策支持：为部署决策提供量化依据
    5. 实时监控：部署过程中的动态风险评估
    """

    def __init__(self, config: Optional[PredictiveSafetyConfig] = None):
        self.config = config or PredictiveSafetyConfig()
        self.logger = logging.getLogger("PredictiveSafetySystem")

        # 核心组件
        self.risk_model = RiskPredictionModel(self.config)

        # 部署历史
        self.deployment_history: deque[DeploymentRecord] = deque(maxlen=2000)

        # 风险监控
        self.active_predictions: Dict[str, RiskPrediction] = {}

    def assess_deployment_risk(self, algorithm_version: str,
                              performance_metrics: Dict[str, float],
                              network_conditions: Dict[str, Any],
                              system_context: Dict[str, Any]) -> RiskPrediction:
        """
        评估部署风险

        Args:
            algorithm_version: 算法版本
            performance_metrics: 性能指标
            network_conditions: 网络条件
            system_context: 系统上下文

        Returns:
            风险预测结果
        """
        # 创建部署记录用于预测
        deployment_record = DeploymentRecord(
            algorithm_version=algorithm_version,
            deployment_time=datetime.now(),
            rollback_occurred=False,  # 预测用，不需要实际值
            rollback_time=None,
            rollback_reason=None,
            canary_duration=300,  # 假设5分钟金丝雀
            max_traffic_percentage=10.0,  # 假设10%流量
            final_status="predicting",

            # 性能指标
            avg_performance_score=performance_metrics.get('performance', 0.85),
            avg_fairness_score=performance_metrics.get('fairness', 0.80),
            avg_stability_score=performance_metrics.get('stability', 0.90),
            peak_latency=performance_metrics.get('latency', 100.0),
            avg_error_rate=performance_metrics.get('error_rate', 0.02),

            # 上下文信息
            network_conditions=network_conditions,
            system_load=system_context.get('load', 0.5),
            deployment_hour=datetime.now().hour,
            is_business_hours=9 <= datetime.now().hour <= 17
        )

        # 进行风险预测
        prediction = self.risk_model.predict_risk(deployment_record)

        # 存储预测结果
        self.active_predictions[algorithm_version] = prediction

        self.logger.info(f"Risk assessment completed for {algorithm_version}: "
                        f"{prediction.risk_level} risk ({prediction.risk_score:.3f})")

        return prediction

    def record_deployment_outcome(self, algorithm_version: str,
                                 success: bool, rollback_reason: Optional[str] = None):
        """
        记录部署结果

        Args:
            algorithm_version: 算法版本
            success: 是否成功
            rollback_reason: 回滚原因
        """
        if algorithm_version not in self.active_predictions:
            self.logger.warning(f"No prediction found for {algorithm_version}")
            return

        prediction = self.active_predictions[algorithm_version]

        # 创建完整的部署记录
        deployment_record = DeploymentRecord(
            algorithm_version=algorithm_version,
            deployment_time=datetime.now() - timedelta(minutes=30),  # 假设30分钟前开始
            rollback_occurred=not success,
            rollback_time=datetime.now() if not success else None,
            rollback_reason=rollback_reason,
            canary_duration=600,  # 10分钟
            max_traffic_percentage=25.0,  # 25%流量
            final_status="success" if success else "rolled_back",

            # 使用预测时的指标作为近似值
            avg_performance_score=0.85,  # 应该从实际监控获取
            avg_fairness_score=0.80,
            avg_stability_score=0.90,
            peak_latency=120.0,
            avg_error_rate=0.025,

            # 上下文信息 (应该从部署时保存)
            network_conditions={'loss_rate': 0.02, 'delay': 25.0},
            system_load=0.6,
            deployment_hour=14,
            is_business_hours=True,

            # 风险信息
            risk_score=prediction.risk_score,
            risk_factors=prediction.primary_risk_factors
        )

        # 添加到历史记录
        self.deployment_history.append(deployment_record)

        # 添加到模型训练数据
        self.risk_model.add_training_record(deployment_record)

        # 清理活跃预测
        del self.active_predictions[algorithm_version]

        self.logger.info(f"Deployment outcome recorded for {algorithm_version}: {'success' if success else 'rollback'}")

    def get_risk_trends(self, days: int = 30) -> Dict[str, Any]:
        """获取风险趋势"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_records = [r for r in self.deployment_history if r.deployment_time > cutoff_date]

        if not recent_records:
            return {"status": "no_data"}

        # 计算趋势
        rollback_rate = sum(1 for r in recent_records if r.rollback_occurred) / len(recent_records)
        avg_risk_score = np.mean([r.risk_score for r in recent_records])

        # 按时间排序
        sorted_records = sorted(recent_records, key=lambda r: r.deployment_time)
        risk_scores_over_time = [r.risk_score for r in sorted_records]

        # 计算趋势斜率
        if len(risk_scores_over_time) > 5:
            x = np.arange(len(risk_scores_over_time))
            trend_slope = np.polyfit(x, risk_scores_over_time, 1)[0]
        else:
            trend_slope = 0

        return {
            "period_days": days,
            "total_deployments": len(recent_records),
            "rollback_rate": rollback_rate,
            "avg_risk_score": avg_risk_score,
            "risk_trend": "increasing" if trend_slope > 0.001 else "decreasing" if trend_slope < -0.001 else "stable",
            "trend_slope": trend_slope
        }

    def get_risk_insights(self) -> Dict[str, Any]:
        """获取风险洞察"""
        if len(self.deployment_history) < 10:
            return {"status": "insufficient_data"}

        # 分析常见回滚原因
        rollback_reasons = defaultdict(int)
        for record in self.deployment_history:
            if record.rollback_occurred and record.rollback_reason:
                rollback_reasons[record.rollback_reason] += 1

        # 分析高风险时间段
        hourly_risks = defaultdict(list)
        for record in self.deployment_history:
            hourly_risks[record.deployment_hour].append(record.risk_score)

        high_risk_hours = [
            hour for hour, scores in hourly_risks.items()
            if np.mean(scores) > 0.5
        ]

        # 分析网络条件对风险的影响
        network_risk_correlation = self._analyze_network_risk_correlation()

        return {
            "common_rollback_reasons": dict(rollback_reasons),
            "high_risk_hours": high_risk_hours,
            "network_risk_correlation": network_risk_correlation,
            "model_statistics": self.risk_model.get_model_statistics()
        }

    def _analyze_network_risk_correlation(self) -> Dict[str, float]:
        """分析网络条件与风险的相关性"""
        correlations = {}

        if len(self.deployment_history) < 20:
            return correlations

        risk_scores = [r.risk_score for r in self.deployment_history]

        # 分析不同网络条件的相关性
        network_features = ['loss_rate', 'delay', 'jitter', 'load']
        for feature in network_features:
            feature_values = []
            for record in self.deployment_history:
                value = record.network_conditions.get(feature, 0)
                feature_values.append(value)

            if len(feature_values) == len(risk_scores):
                correlation = np.corrcoef(feature_values, risk_scores)[0, 1]
                correlations[feature] = correlation

        return correlations

    def save_system_state(self):
        """保存系统状态"""
        try:
            self.risk_model.save_model(self.config.model_save_path)
            self.logger.info("System state saved")
        except Exception as e:
            self.logger.error(f"Failed to save system state: {e}")

    def load_system_state(self) -> bool:
        """加载系统状态"""
        try:
            success = self.risk_model.load_model(self.config.model_save_path)
            if success:
                self.logger.info("System state loaded")
            return success
        except Exception as e:
            self.logger.error(f"Failed to load system state: {e}")
            return False


def create_predictive_safety_system() -> PredictiveSafetySystem:
    """创建预测性安全系统"""
    config = PredictiveSafetyConfig(
        enable_online_learning=True,
        model_save_path="models/risk_prediction_model.pkl"
    )

    system = PredictiveSafetySystem(config)
    return system


# 使用示例
if __name__ == "__main__":
    # 创建预测性安全系统
    safety_system = create_predictive_safety_system()

    # 模拟风险评估
    performance_metrics = {
        'performance': 0.82,
        'fairness': 0.78,
        'stability': 0.85,
        'latency': 140.0,
        'error_rate': 0.025
    }

    network_conditions = {
        'loss_rate': 0.03,
        'delay': 30.0,
        'jitter': 8.0,
        'load': 0.7
    }

    system_context = {
        'load': 0.6
    }

    # 评估部署风险
    prediction = safety_system.assess_deployment_risk(
        "v2.1.5",
        performance_metrics,
        network_conditions,
        system_context
    )

    print("Risk Assessment Results:")
    print(f"  Rollback Probability: {prediction.rollback_probability:.3f}")
    print(f"  Risk Score: {prediction.risk_score:.3f}")
    print(f"  Risk Level: {prediction.risk_level}")
    print(f"  Primary Risk Factors: {prediction.primary_risk_factors}")
    print(f"  Mitigation Suggestions: {prediction.mitigation_suggestions}")
    print(f"  Recommended Actions: {prediction.recommended_actions}")

    # 模拟部署结果
    safety_system.record_deployment_outcome("v2.1.5", success=True)

    # 获取系统洞察
    insights = safety_system.get_risk_insights()
    print("\\nRisk Insights:")
    for key, value in insights.items():
        print(f"  {key}: {value}")

    print("\\nPredictive safety system demo completed")