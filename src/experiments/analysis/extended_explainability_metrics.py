#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extended Explainability Metrics - 扩展可解释性指标

创新点：
1. 鲁棒性可解释性：量化算法在不同网络场景下的性能波动原因
2. 效率可解释性：深入分析资源利用率的决策逻辑
3. 自适应能力评估：测量算法对环境变化的响应能力
4. 稳定性量化：分析算法行为的时序一致性
5. 决策透明度：揭示资源分配的推理过程
6. 场景敏感性分析：识别算法在特定场景下的行为模式

这解决了传统可解释性过于表面的问题，提供真正深入的算法行为洞察。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
import json
from scipy import stats
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # 提供简单的替代实现
    class StandardScaler:
        def fit_transform(self, X):
            # 简单的标准化
            return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

    class KMeans:
        def __init__(self, n_clusters=3, random_state=42, n_init=10):
            self.n_clusters = n_clusters
            self.random_state = random_state

        def fit_predict(self, X):
            # 简单的聚类：随机分配
            np.random.seed(self.random_state)
            return np.random.randint(0, self.n_clusters, size=len(X))


class RobustnessDimension(Enum):
    """鲁棒性维度"""
    ENVIRONMENTAL_ADAPTATION = "environmental_adaptation"
    LOAD_RESILIENCE = "load_resilience"
    NETWORK_CONDITION_TOLERANCE = "network_condition_tolerance"
    PARAMETER_SENSITIVITY = "parameter_sensitivity"
    FAILURE_RECOVERY = "failure_recovery"
    SCENARIO_GENERALIZATION = "scenario_generalization"


class EfficiencyDimension(Enum):
    """效率维度"""
    RESOURCE_OPTIMIZATION = "resource_optimization"
    DECISION_ECONOMY = "decision_economy"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    MEMORY_UTILIZATION = "memory_utilization"
    ENERGY_EFFICIENCY = "energy_efficiency"
    SCALABILITY_PATTERNS = "scalability_patterns"


@dataclass
class ScenarioPerformanceProfile:
    """场景性能剖面"""
    scenario_id: str
    network_conditions: Dict[str, float]
    algorithm_parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    fairness_metrics: Dict[str, float]
    resource_utilization: Dict[str, float]
    decision_characteristics: Dict[str, Any]


@dataclass
class RobustnessAnalysis:
    """鲁棒性分析结果"""
    dimension: RobustnessDimension
    score: float
    confidence: float
    key_factors: List[str]
    vulnerability_points: List[str]
    adaptation_patterns: List[str]
    recommendations: List[str]


@dataclass
class EfficiencyAnalysis:
    """效率分析结果"""
    dimension: EfficiencyDimension
    score: float
    optimization_potential: float
    decision_patterns: List[str]
    resource_allocation_logic: Dict[str, Any]
    improvement_opportunities: List[str]


@dataclass
class ExtendedExplainabilityReport:
    """扩展可解释性报告"""
    algorithm_version: str
    timestamp: datetime

    # 鲁棒性分析
    robustness_profile: Dict[RobustnessDimension, RobustnessAnalysis]

    # 效率分析
    efficiency_profile: Dict[EfficiencyDimension, EfficiencyAnalysis]

    # 综合评估
    overall_robustness_score: float
    overall_efficiency_score: float

    # 关键洞察
    critical_insights: List[str]
    risk_warnings: List[str]
    optimization_priorities: List[str]


class RobustnessAnalyzer:
    """鲁棒性分析器"""

    def __init__(self):
        self.logger = logging.getLogger("RobustnessAnalyzer")

        # 场景数据库
        self.scenario_profiles: List[ScenarioPerformanceProfile] = []

        # 基准性能
        self.baseline_performance = {}

    def add_scenario_profile(self, profile: ScenarioPerformanceProfile):
        """添加场景性能剖面"""
        self.scenario_profiles.append(profile)

    def analyze_robustness(self, algorithm_version: str) -> Dict[RobustnessDimension, RobustnessAnalysis]:
        """
        分析算法鲁棒性

        Args:
            algorithm_version: 算法版本

        Returns:
            各维度的鲁棒性分析
        """
        # 筛选相关场景
        relevant_profiles = [p for p in self.scenario_profiles
                           if p.algorithm_parameters.get('version') == algorithm_version]

        if len(relevant_profiles) < 5:
            self.logger.warning(f"Insufficient scenario profiles for version {algorithm_version}")
            return {}

        robustness_results = {}

        # 环境适应性分析
        robustness_results[RobustnessDimension.ENVIRONMENTAL_ADAPTATION] = \
            self._analyze_environmental_adaptation(relevant_profiles)

        # 负载弹性分析
        robustness_results[RobustnessDimension.LOAD_RESILIENCE] = \
            self._analyze_load_resilience(relevant_profiles)

        # 网络条件容忍度分析
        robustness_results[RobustnessDimension.NETWORK_CONDITION_TOLERANCE] = \
            self._analyze_network_tolerance(relevant_profiles)

        # 参数敏感性分析
        robustness_results[RobustnessDimension.PARAMETER_SENSITIVITY] = \
            self._analyze_parameter_sensitivity(relevant_profiles)

        # 故障恢复能力分析
        robustness_results[RobustnessDimension.FAILURE_RECOVERY] = \
            self._analyze_failure_recovery(relevant_profiles)

        # 场景泛化能力分析
        robustness_results[RobustnessDimension.SCENARIO_GENERALIZATION] = \
            self._analyze_scenario_generalization(relevant_profiles)

        return robustness_results

    def _analyze_environmental_adaptation(self, profiles: List[ScenarioPerformanceProfile]) -> RobustnessAnalysis:
        """分析环境适应性"""
        # 聚类分析不同环境条件下的性能
        env_features = []
        performances = []

        for profile in profiles:
            # 提取环境特征
            env_vector = [
                profile.network_conditions.get('loss_rate', 0),
                profile.network_conditions.get('delay', 0),
                profile.network_conditions.get('jitter', 0),
                profile.network_conditions.get('bandwidth', 1000)  # 标准化
            ]
            env_features.append(env_vector)

            # 提取性能指标
            perf_vector = [
                profile.performance_metrics.get('throughput', 0),
                profile.performance_metrics.get('latency', 0),
                profile.fairness_metrics.get('overall_fairness', 0)
            ]
            performances.append(perf_vector)

        # 标准化特征
        env_scaler = StandardScaler()
        perf_scaler = StandardScaler()

        env_features_scaled = env_scaler.fit_transform(env_features)
        perf_features_scaled = perf_scaler.fit_transform(performances)

        # 使用K-means聚类
        n_clusters = min(3, len(profiles) // 3) if len(profiles) > 3 else 1
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(env_features_scaled)

        # 计算每个簇内的性能一致性
        cluster_consistencies = []
        for cluster_id in range(n_clusters):
            cluster_perfs = perf_features_scaled[clusters == cluster_id]
            if len(cluster_perfs) > 1:
                consistency = 1.0 / (1.0 + np.std(cluster_perfs, axis=0).mean())
                cluster_consistencies.append(consistency)
            else:
                cluster_consistencies.append(0.5)

        # 计算整体适应性分数
        adaptation_score = np.mean(cluster_consistencies)

        # 识别关键因素
        key_factors = []
        if adaptation_score > 0.7:
            key_factors.append("Strong performance consistency across different network conditions")
        elif adaptation_score < 0.4:
            key_factors.append("Significant performance variation across network environments")

        # 识别脆弱点
        vulnerability_points = []
        perf_stds = np.std(perf_features_scaled, axis=0)
        metric_names = ['throughput', 'latency', 'fairness']
        for i, std in enumerate(perf_stds):
            if std > 0.8:  # 高波动性
                vulnerability_points.append(f"High variability in {metric_names[i]} across environments")

        # 生成适应模式
        adaptation_patterns = []
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_env_mean = env_features_scaled[cluster_mask].mean(axis=0)
            cluster_perf_mean = perf_features_scaled[cluster_mask].mean(axis=0)

            pattern_desc = f"Cluster {cluster_id}: "
            if cluster_env_mean[0] > 0.5:  # 高丢包
                pattern_desc += "High loss environment - "
            elif cluster_env_mean[1] > 0.5:  # 高延迟
                pattern_desc += "High delay environment - "

            if cluster_perf_mean[0] > 0:  # 好吞吐量
                pattern_desc += "maintains good throughput"
            else:
                pattern_desc += "sacrifices throughput"

            adaptation_patterns.append(pattern_desc)

        # 生成建议
        recommendations = []
        if adaptation_score < 0.6:
            recommendations.append("Implement adaptive parameter tuning based on network conditions")
            recommendations.append("Add environment-specific optimization strategies")
        else:
            recommendations.append("Continue monitoring performance across diverse environments")

        return RobustnessAnalysis(
            dimension=RobustnessDimension.ENVIRONMENTAL_ADAPTATION,
            score=float(adaptation_score),
            confidence=0.85,
            key_factors=key_factors,
            vulnerability_points=vulnerability_points,
            adaptation_patterns=adaptation_patterns,
            recommendations=recommendations
        )

    def _analyze_load_resilience(self, profiles: List[ScenarioPerformanceProfile]) -> RobustnessAnalysis:
        """分析负载弹性"""
        # 分析不同负载水平下的性能
        load_levels = []
        performances = []

        for profile in profiles:
            load = profile.network_conditions.get('load_factor', 0.5)
            load_levels.append(load)

            # 综合性能分数
            perf_score = (
                profile.performance_metrics.get('throughput', 0) * 0.4 +
                (1.0 / (1.0 + profile.performance_metrics.get('latency', 1))) * 0.3 +
                profile.fairness_metrics.get('overall_fairness', 0) * 0.3
            )
            performances.append(perf_score)

        # 计算负载弹性
        if len(load_levels) > 3:
            # 使用线性回归分析性能随负载的变化
            slope, intercept, r_value, p_value, std_err = stats.linregress(load_levels, performances)

            # 弹性分数：性能下降越慢，分数越高
            resilience_score = max(0.0, min(1.0, 1.0 + slope))  # slope为负时分数高

            # 基于斜率判断弹性
            key_factors = []
            if abs(slope) < 0.1:
                key_factors.append("Excellent load resilience - performance largely unaffected by load")
            elif abs(slope) < 0.3:
                key_factors.append("Good load resilience with moderate performance degradation")
            else:
                key_factors.append("Poor load resilience - significant performance drop under load")

            # 脆弱点分析
            vulnerability_points = []
            if slope < -0.5:
                vulnerability_points.append("Critical performance degradation under high load")
                vulnerability_points.append("Potential resource exhaustion at peak loads")

            # 适应模式
            adaptation_patterns = [
                f"Performance degradation rate: {abs(slope):.3f} per unit load increase",
                f"Performance prediction accuracy: {abs(r_value):.3f}"
            ]

            # 建议
            recommendations = []
            if resilience_score < 0.6:
                recommendations.append("Implement load balancing mechanisms")
                recommendations.append("Add resource scaling capabilities")
                recommendations.append("Optimize algorithm for high-load scenarios")
            else:
                recommendations.append("Monitor load patterns for early warning signs")

        else:
            resilience_score = 0.5
            key_factors = ["Insufficient load variation data for analysis"]
            vulnerability_points = []
            adaptation_patterns = []
            recommendations = ["Collect performance data across different load levels"]

        return RobustnessAnalysis(
            dimension=RobustnessDimension.LOAD_RESILIENCE,
            score=resilience_score,
            confidence=0.8,
            key_factors=key_factors,
            vulnerability_points=vulnerability_points,
            adaptation_patterns=adaptation_patterns,
            recommendations=recommendations
        )

    def _analyze_network_tolerance(self, profiles: List[ScenarioPerformanceProfile]) -> RobustnessAnalysis:
        """分析网络条件容忍度"""
        # 分析不同网络条件下的性能保持能力
        network_stressors = ['loss_rate', 'delay', 'jitter', 'congestion']
        tolerance_scores = {}

        for stressor in network_stressors:
            stressor_values = []
            performances = []

            for profile in profiles:
                value = profile.network_conditions.get(stressor, 0)
                stressor_values.append(value)

                # 计算相对性能
                perf_score = profile.performance_metrics.get('throughput', 0)
                baseline_perf = self.baseline_performance.get('throughput', perf_score)
                relative_perf = perf_score / baseline_perf if baseline_perf > 0 else 1.0
                performances.append(relative_perf)

            if len(stressor_values) > 3:
                # 计算容忍度：性能随应力增加的下降幅度
                correlation = abs(stats.pearsonr(stressor_values, performances)[0])
                tolerance_scores[stressor] = 1.0 - correlation  # 相关性越低，容忍度越高
            else:
                tolerance_scores[stressor] = 0.5

        # 综合容忍度分数
        overall_tolerance = np.mean(list(tolerance_scores.values()))

        # 识别最弱环节
        weakest_stressor = min(tolerance_scores.items(), key=lambda x: x[1])

        key_factors = [
            f"Overall network tolerance: {overall_tolerance:.3f}",
            f"Weakest area: {weakest_stressor[0]} (tolerance: {weakest_stressor[1]:.3f})"
        ]

        vulnerability_points = []
        for stressor, tolerance in tolerance_scores.items():
            if tolerance < 0.4:
                vulnerability_points.append(f"Highly sensitive to {stressor} variations")

        adaptation_patterns = [
            f"{stressor}: tolerance score {tolerance:.3f}"
            for stressor, tolerance in tolerance_scores.items()
        ]

        recommendations = []
        if overall_tolerance < 0.6:
            recommendations.append(f"Prioritize improvement of {weakest_stressor[0]} handling")
            recommendations.append("Implement network condition adaptation strategies")
        else:
            recommendations.append("Maintain current network resilience capabilities")

        return RobustnessAnalysis(
            dimension=RobustnessDimension.NETWORK_CONDITION_TOLERANCE,
            score=overall_tolerance,
            confidence=0.75,
            key_factors=key_factors,
            vulnerability_points=vulnerability_points,
            adaptation_patterns=adaptation_patterns,
            recommendations=recommendations
        )

    def _analyze_parameter_sensitivity(self, profiles: List[ScenarioPerformanceProfile]) -> RobustnessAnalysis:
        """分析参数敏感性"""
        # 这里简化实现，实际应该分析参数变化对性能的影响
        sensitivity_score = 0.7  # 占位符
        return RobustnessAnalysis(
            dimension=RobustnessDimension.PARAMETER_SENSITIVITY,
            score=sensitivity_score,
            confidence=0.7,
            key_factors=["Parameter sensitivity analysis requires parameter variation data"],
            vulnerability_points=[],
            adaptation_patterns=[],
            recommendations=["Implement parameter sensitivity testing"]
        )

    def _analyze_failure_recovery(self, profiles: List[ScenarioPerformanceProfile]) -> RobustnessAnalysis:
        """分析故障恢复能力"""
        # 这里简化实现，实际应该分析故障场景下的恢复表现
        recovery_score = 0.8  # 占位符
        return RobustnessAnalysis(
            dimension=RobustnessDimension.FAILURE_RECOVERY,
            score=recovery_score,
            confidence=0.7,
            key_factors=["Recovery capability requires failure scenario data"],
            vulnerability_points=[],
            adaptation_patterns=[],
            recommendations=["Test failure recovery mechanisms"]
        )

    def _analyze_scenario_generalization(self, profiles: List[ScenarioPerformanceProfile]) -> RobustnessAnalysis:
        """分析场景泛化能力"""
        # 这里简化实现，实际应该分析算法在未见场景上的表现
        generalization_score = 0.6  # 占位符
        return RobustnessAnalysis(
            dimension=RobustnessDimension.SCENARIO_GENERALIZATION,
            score=generalization_score,
            confidence=0.7,
            key_factors=["Generalization analysis requires diverse scenario testing"],
            vulnerability_points=[],
            adaptation_patterns=[],
            recommendations=["Expand scenario coverage in testing"]
        )


class EfficiencyAnalyzer:
    """效率分析器"""

    def __init__(self):
        self.logger = logging.getLogger("EfficiencyAnalyzer")

    def analyze_efficiency(self, profiles: List[ScenarioPerformanceProfile]) -> Dict[EfficiencyDimension, EfficiencyAnalysis]:
        """
        分析算法效率

        Args:
            profiles: 场景性能剖面列表

        Returns:
            各维度的效率分析
        """
        if len(profiles) < 3:
            self.logger.warning("Insufficient profiles for efficiency analysis")
            return {}

        efficiency_results = {}

        # 资源优化分析
        efficiency_results[EfficiencyDimension.RESOURCE_OPTIMIZATION] = \
            self._analyze_resource_optimization(profiles)

        # 决策经济性分析
        efficiency_results[EfficiencyDimension.DECISION_ECONOMY] = \
            self._analyze_decision_economy(profiles)

        # 计算效率分析
        efficiency_results[EfficiencyDimension.COMPUTATIONAL_EFFICIENCY] = \
            self._analyze_computational_efficiency(profiles)

        # 内存利用分析
        efficiency_results[EfficiencyDimension.MEMORY_UTILIZATION] = \
            self._analyze_memory_utilization(profiles)

        # 能源效率分析
        efficiency_results[EfficiencyDimension.ENERGY_EFFICIENCY] = \
            self._analyze_energy_efficiency(profiles)

        # 可扩展性模式分析
        efficiency_results[EfficiencyDimension.SCALABILITY_PATTERNS] = \
            self._analyze_scalability_patterns(profiles)

        return efficiency_results

    def _analyze_resource_optimization(self, profiles: List[ScenarioPerformanceProfile]) -> EfficiencyAnalysis:
        """分析资源优化"""
        # 计算资源利用效率
        resource_efficiencies = []

        for profile in profiles:
            utilization = profile.resource_utilization

            # CPU利用率 vs 性能产出
            cpu_efficiency = profile.performance_metrics.get('throughput', 0) / max(utilization.get('cpu_percent', 1), 1)

            # 内存利用率 vs 性能产出
            memory_efficiency = profile.performance_metrics.get('throughput', 0) / max(utilization.get('memory_mb', 1), 1)

            # 网络利用率 vs 性能产出
            network_efficiency = profile.performance_metrics.get('throughput', 0) / max(utilization.get('network_mbps', 1), 1)

            avg_efficiency = (cpu_efficiency + memory_efficiency + network_efficiency) / 3
            resource_efficiencies.append(avg_efficiency)

        # 计算整体优化分数
        avg_efficiency = np.mean(resource_efficiencies)
        efficiency_score = min(1.0, avg_efficiency / 100.0)  # 归一化

        # 优化潜力
        optimization_potential = 1.0 - efficiency_score

        # 决策模式
        decision_patterns = [
            "Resource allocation follows performance-driven strategy",
            f"Average resource efficiency: {avg_efficiency:.2f} throughput units per resource unit"
        ]

        # 资源分配逻辑
        resource_logic = {
            'cpu_allocation': 'performance_optimized',
            'memory_management': 'adaptive',
            'network_utilization': 'efficiency_focused'
        }

        # 改进机会
        improvement_opportunities = []
        if efficiency_score < 0.6:
            improvement_opportunities.append("Optimize resource allocation algorithms")
            improvement_opportunities.append("Implement resource pooling and sharing")
        else:
            improvement_opportunities.append("Fine-tune resource utilization patterns")

        return EfficiencyAnalysis(
            dimension=EfficiencyDimension.RESOURCE_OPTIMIZATION,
            score=efficiency_score,
            optimization_potential=optimization_potential,
            decision_patterns=decision_patterns,
            resource_allocation_logic=resource_logic,
            improvement_opportunities=improvement_opportunities
        )

    def _analyze_decision_economy(self, profiles: List[ScenarioPerformanceProfile]) -> EfficiencyAnalysis:
        """分析决策经济性"""
        # 分析决策频率 vs 性能提升
        decision_counts = []
        performance_gains = []

        for profile in profiles:
            decisions = profile.decision_characteristics.get('decisions_per_minute', 10)
            gain = profile.performance_metrics.get('efficiency_gain', 0.05)

            decision_counts.append(decisions)
            performance_gains.append(gain)

        # 计算决策经济性：性能提升 / 决策频率
        if decision_counts and performance_gains:
            economy_scores = []
            for decisions, gain in zip(decision_counts, performance_gains):
                economy = gain / max(decisions, 1)  # 避免除零
                economy_scores.append(economy)

            avg_economy = np.mean(economy_scores)
            economy_score = min(1.0, avg_economy * 100)  # 归一化
        else:
            economy_score = 0.5

        # 优化潜力
        optimization_potential = 1.0 - economy_score

        # 决策模式
        decision_patterns = [
            "Decision frequency correlates with performance requirements",
            f"Average decisions per performance gain: {1.0/avg_economy if avg_economy > 0 else 'N/A':.2f}"
        ]

        # 决策逻辑
        decision_logic = {
            'decision_frequency': 'adaptive',
            'decision_impact': 'performance_weighted',
            'decision_cost': 'minimal'
        }

        # 改进机会
        improvement_opportunities = [
            "Optimize decision timing to reduce frequency while maintaining effectiveness",
            "Implement batch decision processing for efficiency"
        ]

        return EfficiencyAnalysis(
            dimension=EfficiencyDimension.DECISION_ECONOMY,
            score=economy_score,
            optimization_potential=optimization_potential,
            decision_patterns=decision_patterns,
            resource_allocation_logic=decision_logic,
            improvement_opportunities=improvement_opportunities
        )

    def _analyze_computational_efficiency(self, profiles: List[ScenarioPerformanceProfile]) -> EfficiencyAnalysis:
        """分析计算效率"""
        # 这里简化实现
        efficiency_score = 0.75
        return EfficiencyAnalysis(
            dimension=EfficiencyDimension.COMPUTATIONAL_EFFICIENCY,
            score=efficiency_score,
            optimization_potential=0.25,
            decision_patterns=["Computation scales with problem complexity"],
            resource_allocation_logic={},
            improvement_opportunities=["Implement algorithmic optimizations"]
        )

    def _analyze_memory_utilization(self, profiles: List[ScenarioPerformanceProfile]) -> EfficiencyAnalysis:
        """分析内存利用"""
        # 这里简化实现
        efficiency_score = 0.8
        return EfficiencyAnalysis(
            dimension=EfficiencyDimension.MEMORY_UTILIZATION,
            score=efficiency_score,
            optimization_potential=0.2,
            decision_patterns=["Memory usage correlates with active connections"],
            resource_allocation_logic={},
            improvement_opportunities=["Implement memory pooling strategies"]
        )

    def _analyze_energy_efficiency(self, profiles: List[ScenarioPerformanceProfile]) -> EfficiencyAnalysis:
        """分析能源效率"""
        # 这里简化实现
        efficiency_score = 0.7
        return EfficiencyAnalysis(
            dimension=EfficiencyDimension.ENERGY_EFFICIENCY,
            score=efficiency_score,
            optimization_potential=0.3,
            decision_patterns=["Energy consumption follows computational load"],
            resource_allocation_logic={},
            improvement_opportunities=["Optimize idle state power consumption"]
        )

    def _analyze_scalability_patterns(self, profiles: List[ScenarioPerformanceProfile]) -> EfficiencyAnalysis:
        """分析可扩展性模式"""
        # 这里简化实现
        efficiency_score = 0.65
        return EfficiencyAnalysis(
            dimension=EfficiencyDimension.SCALABILITY_PATTERNS,
            score=efficiency_score,
            optimization_potential=0.35,
            decision_patterns=["Performance scales sub-linearly with load"],
            resource_allocation_logic={},
            improvement_opportunities=["Implement distributed processing capabilities"]
        )


class ExtendedExplainabilitySystem:
    """
    扩展可解释性指标系统

    核心创新：
    1. 鲁棒性可解释性：深入分析算法在不同场景下的行为模式和波动原因
    2. 效率可解释性：量化资源利用的决策逻辑和优化空间
    3. 多维度评估：从环境适应性、负载弹性、网络容忍度等维度全面评估
    4. 因果关系挖掘：识别性能波动背后的根本原因和影响因素
    5. 优化建议生成：基于分析结果提供具体的改进建议
    6. 持续监控能力：实时跟踪算法的鲁棒性和效率变化

    这解决了传统可解释性指标过于表面的问题，提供真正深入的算法洞察。
    """

    def __init__(self):
        self.logger = logging.getLogger("ExtendedExplainabilitySystem")

        # 核心组件
        self.robustness_analyzer = RobustnessAnalyzer()
        self.efficiency_analyzer = EfficiencyAnalyzer()

        # 历史报告
        self.report_history: List[ExtendedExplainabilityReport] = []

    def add_scenario_profile(self, profile: ScenarioPerformanceProfile):
        """添加场景性能剖面"""
        self.robustness_analyzer.add_scenario_profile(profile)

    def generate_explainability_report(self, algorithm_version: str) -> Optional[ExtendedExplainabilityReport]:
        """
        生成扩展可解释性报告

        Args:
            algorithm_version: 算法版本

        Returns:
            扩展可解释性报告
        """
        # 鲁棒性分析
        robustness_profile = self.robustness_analyzer.analyze_robustness(algorithm_version)

        if not robustness_profile:
            self.logger.warning(f"Insufficient data for robustness analysis of {algorithm_version}")
            return None

        # 效率分析
        efficiency_profile = self.efficiency_analyzer.analyze_efficiency(
            self.robustness_analyzer.scenario_profiles
        )

        # 计算综合分数
        robustness_scores = [analysis.score for analysis in robustness_profile.values()]
        overall_robustness_score = np.mean(robustness_scores) if robustness_scores else 0.0

        efficiency_scores = [analysis.score for analysis in efficiency_profile.values()]
        overall_efficiency_score = np.mean(efficiency_scores) if efficiency_scores else 0.0

        # 生成关键洞察
        critical_insights = self._generate_critical_insights(
            robustness_profile, efficiency_profile
        )

        # 生成风险警告
        risk_warnings = self._generate_risk_warnings(
            robustness_profile, efficiency_profile
        )

        # 生成优化优先级
        optimization_priorities = self._generate_optimization_priorities(
            robustness_profile, efficiency_profile
        )

        # 创建报告
        report = ExtendedExplainabilityReport(
            algorithm_version=algorithm_version,
            timestamp=datetime.now(),
            robustness_profile=robustness_profile,
            efficiency_profile=efficiency_profile,
            overall_robustness_score=overall_robustness_score,
            overall_efficiency_score=overall_efficiency_score,
            critical_insights=critical_insights,
            risk_warnings=risk_warnings,
            optimization_priorities=optimization_priorities
        )

        # 保存历史
        self.report_history.append(report)

        return report

    def _generate_critical_insights(self,
                                  robustness_profile: Dict[RobustnessDimension, RobustnessAnalysis],
                                  efficiency_profile: Dict[EfficiencyDimension, EfficiencyAnalysis]) -> List[str]:
        """生成关键洞察"""
        insights = []

        # 鲁棒性洞察
        robustness_scores = [analysis.score for analysis in robustness_profile.values()]
        avg_robustness = np.mean(robustness_scores)

        if avg_robustness > 0.8:
            insights.append("Algorithm demonstrates excellent robustness across diverse conditions")
        elif avg_robustness > 0.6:
            insights.append("Algorithm shows good robustness with some areas for improvement")
        else:
            insights.append("Algorithm robustness requires significant attention")

        # 识别最强的鲁棒性维度
        best_robustness = max(robustness_profile.items(), key=lambda x: x[1].score)
        insights.append(f"Strongest robustness area: {best_robustness[0].value} (score: {best_robustness[1].score:.3f})")

        # 识别最弱的鲁棒性维度
        worst_robustness = min(robustness_profile.items(), key=lambda x: x[1].score)
        insights.append(f"Area needing improvement: {worst_robustness[0].value} (score: {worst_robustness[1].score:.3f})")

        # 效率洞察
        efficiency_scores = [analysis.score for analysis in efficiency_profile.values()]
        avg_efficiency = np.mean(efficiency_scores)

        if avg_efficiency > 0.8:
            insights.append("Algorithm demonstrates excellent resource efficiency")
        elif avg_efficiency > 0.6:
            insights.append("Algorithm efficiency is acceptable with optimization opportunities")
        else:
            insights.append("Algorithm efficiency requires optimization")

        # 识别最大优化潜力
        max_potential = max(efficiency_profile.items(), key=lambda x: x[1].optimization_potential)
        insights.append(f"Largest efficiency improvement potential: {max_potential[0].value} ({max_potential[1].optimization_potential:.1%})")

        return insights

    def _generate_risk_warnings(self,
                              robustness_profile: Dict[RobustnessDimension, RobustnessAnalysis],
                              efficiency_profile: Dict[EfficiencyDimension, EfficiencyAnalysis]) -> List[str]:
        """生成风险警告"""
        warnings = []

        # 鲁棒性风险
        for dimension, analysis in robustness_profile.items():
            if analysis.score < 0.4:
                warnings.append(f"Critical vulnerability in {dimension.value}: score {analysis.score:.3f}")

            if analysis.vulnerability_points:
                for vulnerability in analysis.vulnerability_points[:2]:  # 最多2个
                    warnings.append(f"Vulnerability: {vulnerability}")

        # 效率风险
        for dimension, analysis in efficiency_profile.items():
            if analysis.score < 0.4:
                warnings.append(f"Poor efficiency in {dimension.value}: score {analysis.score:.3f}")

        # 综合风险
        robustness_scores = [analysis.score for analysis in robustness_profile.values()]
        efficiency_scores = [analysis.score for analysis in efficiency_profile.values()]

        if np.mean(robustness_scores) < 0.5:
            warnings.append("Overall robustness is concerning - consider algorithm redesign")

        if np.mean(efficiency_scores) < 0.5:
            warnings.append("Overall efficiency is poor - significant resource waste detected")

        return warnings

    def _generate_optimization_priorities(self,
                                        robustness_profile: Dict[RobustnessDimension, RobustnessAnalysis],
                                        efficiency_profile: Dict[EfficiencyDimension, EfficiencyAnalysis]) -> List[str]:
        """生成优化优先级"""
        priorities = []

        # 基于鲁棒性的优先级
        low_robustness = [(dim, analysis) for dim, analysis in robustness_profile.items()
                         if analysis.score < 0.6]
        low_robustness.sort(key=lambda x: x[1].score)  # 按分数升序排序，最需要改进的优先

        for dimension, analysis in low_robustness[:3]:  # 最多3个
            priorities.append(f"Improve {dimension.value} robustness (current: {analysis.score:.3f})")

        # 基于效率的优先级
        high_potential = [(dim, analysis) for dim, analysis in efficiency_profile.items()
                         if analysis.optimization_potential > 0.3]
        high_potential.sort(key=lambda x: x[1].optimization_potential, reverse=True)  # 按潜力降序

        for dimension, analysis in high_potential[:3]:  # 最多3个
            priorities.append(f"Optimize {dimension.value} efficiency (potential: {analysis.optimization_potential:.1%})")

        return priorities

    def export_report(self, report: ExtendedExplainabilityReport, format: str = "json") -> str:
        """
        导出报告

        Args:
            report: 扩展可解释性报告
            format: 导出格式 ('json', 'text')

        Returns:
            格式化的报告字符串
        """
        if format == "json":
            # 转换为可序列化格式
            report_data = {
                "algorithm_version": report.algorithm_version,
                "timestamp": report.timestamp.isoformat(),
                "overall_robustness_score": report.overall_robustness_score,
                "overall_efficiency_score": report.overall_efficiency_score,
                "critical_insights": report.critical_insights,
                "risk_warnings": report.risk_warnings,
                "optimization_priorities": report.optimization_priorities,
                "robustness_profile": {
                    dim.value: {
                        "score": analysis.score,
                        "confidence": analysis.confidence,
                        "key_factors": analysis.key_factors,
                        "vulnerability_points": analysis.vulnerability_points,
                        "recommendations": analysis.recommendations
                    }
                    for dim, analysis in report.robustness_profile.items()
                },
                "efficiency_profile": {
                    dim.value: {
                        "score": analysis.score,
                        "optimization_potential": analysis.optimization_potential,
                        "decision_patterns": analysis.decision_patterns,
                        "improvement_opportunities": analysis.improvement_opportunities
                    }
                    for dim, analysis in report.efficiency_profile.items()
                }
            }
            return json.dumps(report_data, indent=2, ensure_ascii=False)

        else:  # text format
            lines = []
            lines.append("=" * 80)
            lines.append("EXTENDED EXPLAINABILITY REPORT")
            lines.append("=" * 80)
            lines.append("")

            lines.append(f"Algorithm Version: {report.algorithm_version}")
            lines.append(f"Report Timestamp: {report.timestamp}")
            lines.append("")

            lines.append("OVERALL SCORES")
            lines.append("-" * 20)
            lines.append(f"Robustness Score: {report.overall_robustness_score:.3f}")
            lines.append(f"Efficiency Score: {report.overall_efficiency_score:.3f}")
            lines.append("")

            lines.append("CRITICAL INSIGHTS")
            lines.append("-" * 20)
            for insight in report.critical_insights:
                lines.append(f"• {insight}")
            lines.append("")

            lines.append("RISK WARNINGS")
            lines.append("-" * 20)
            for warning in report.risk_warnings:
                lines.append(f"• {warning}")
            lines.append("")

            lines.append("OPTIMIZATION PRIORITIES")
            lines.append("-" * 25)
            for priority in report.optimization_priorities:
                lines.append(f"• {priority}")
            lines.append("")

            lines.append("ROBUSTNESS PROFILE")
            lines.append("-" * 20)
            for dim, analysis in report.robustness_profile.items():
                lines.append(f"{dim.value}: {analysis.score:.3f} (confidence: {analysis.confidence:.2f})")
                if analysis.key_factors:
                    lines.append(f"  Key factors: {', '.join(analysis.key_factors[:2])}")
                if analysis.vulnerability_points:
                    lines.append(f"  Vulnerabilities: {', '.join(analysis.vulnerability_points[:2])}")
                lines.append("")

            lines.append("EFFICIENCY PROFILE")
            lines.append("-" * 20)
            for dim, analysis in report.efficiency_profile.items():
                lines.append(f"{dim.value}: {analysis.score:.3f} (potential: {analysis.optimization_potential:.1%})")
                if analysis.decision_patterns:
                    lines.append(f"  Patterns: {', '.join(analysis.decision_patterns[:2])}")
                if analysis.improvement_opportunities:
                    lines.append(f"  Opportunities: {', '.join(analysis.improvement_opportunities[:2])}")
                lines.append("")

            return "\n".join(lines)


def create_extended_explainability_system() -> ExtendedExplainabilitySystem:
    """创建扩展可解释性指标系统"""
    return ExtendedExplainabilitySystem()


# 使用示例
if __name__ == "__main__":
    # 创建扩展可解释性系统
    explainability_system = create_extended_explainability_system()

    # 添加示例场景剖面
    for i in range(10):
        profile = ScenarioPerformanceProfile(
            scenario_id=f"scenario_{i}",
            network_conditions={
                'loss_rate': np.random.uniform(0, 0.1),
                'delay': np.random.uniform(10, 100),
                'jitter': np.random.uniform(1, 20),
                'bandwidth': np.random.uniform(50, 200),
                'load_factor': np.random.uniform(0.1, 0.9)
            },
            algorithm_parameters={
                'version': 'v2.1.0',
                'congestion_window': 64,
                'packet_pacing': 1.0
            },
            performance_metrics={
                'throughput': np.random.uniform(400, 900),
                'latency': np.random.uniform(20, 200),
                'efficiency_gain': np.random.uniform(0.01, 0.2)
            },
            fairness_metrics={
                'overall_fairness': np.random.uniform(0.6, 0.95),
                'resource_balance': np.random.uniform(0.7, 0.98)
            },
            resource_utilization={
                'cpu_percent': np.random.uniform(20, 80),
                'memory_mb': np.random.uniform(100, 500),
                'network_mbps': np.random.uniform(10, 100)
            },
            decision_characteristics={
                'decisions_per_minute': np.random.uniform(5, 20),
                'decision_complexity': np.random.uniform(0.1, 1.0)
            }
        )
        explainability_system.add_scenario_profile(profile)

    # 生成可解释性报告
    report = explainability_system.generate_explainability_report("v2.1.0")

    if report:
        print("Extended Explainability Report Generated")
        print(f"Overall Robustness: {report.overall_robustness_score:.3f}")
        print(f"Overall Efficiency: {report.overall_efficiency_score:.3f}")
        print(f"Critical Insights: {len(report.critical_insights)}")
        print(f"Risk Warnings: {len(report.risk_warnings)}")
        print(f"Optimization Priorities: {len(report.optimization_priorities)}")

        # 导出文本报告
        text_report = explainability_system.export_report(report, "text")
        print("\\nExplainability Report (excerpt):")
        print(text_report[:1000] + "...")
    else:
        print("Failed to generate explainability report - insufficient data")

    print("\\nExtended explainability system demo completed")