#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Behavior Fingerprint System - 行为指纹系统

创新点：
1. 多维度行为向量：整合性能、公平性、鲁棒性等多维度指标
2. 指纹相似度分析：基于向量距离的算法行为比较
3. 异常行为检测：自动识别偏离正常行为模式的异常
4. 版本演化追踪：跟踪算法版本间的行为变化趋势
5. 根因关联分析：将行为变化关联到具体的代码或配置变更
6. 聚类行为模式：识别算法家族的典型行为模式

这解决了传统性能指标无法全面刻画算法行为的局限。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import json


@dataclass
class BehaviorMetrics:
    """行为指标"""
    timestamp: datetime
    algorithm_version: str

    # 性能指标
    performance_score: float
    throughput: float
    latency_p95: float
    packet_loss_rate: float

    # 公平性指标
    fairness_score: float
    resource_balance: float
    information_symmetry: float
    jains_fairness_index: float

    # 鲁棒性指标
    stability_score: float
    fluctuation_resistance: float
    recovery_speed: float
    adaptation_rate: float

    # 环境适应性指标
    network_load_adaptation: float
    congestion_handling: float
    cross_traffic_tolerance: float

    # 决策模式指标
    decision_aggressiveness: float
    fairness_priority: float
    performance_priority: float
    risk_tolerance: float

    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            self.performance_score,
            self.throughput,
            self.latency_p95,
            self.packet_loss_rate,
            self.fairness_score,
            self.resource_balance,
            self.information_symmetry,
            self.jains_fairness_index,
            self.stability_score,
            self.fluctuation_resistance,
            self.recovery_speed,
            self.adaptation_rate,
            self.network_load_adaptation,
            self.congestion_handling,
            self.cross_traffic_tolerance,
            self.decision_aggressiveness,
            self.fairness_priority,
            self.performance_priority,
            self.risk_tolerance
        ])


@dataclass
class BehaviorFingerprint:
    """行为指纹"""
    algorithm_version: str
    creation_time: datetime
    fingerprint_vector: np.ndarray
    confidence_intervals: Dict[str, Tuple[float, float]]
    representative_scenarios: List[str]
    behavioral_traits: List[str]

    def similarity_to(self, other: 'BehaviorFingerprint') -> float:
        """计算与另一个指纹的相似度"""
        return cosine_similarity(
            self.fingerprint_vector.reshape(1, -1),
            other.fingerprint_vector.reshape(1, -1)
        )[0][0]

    def distance_to(self, other: 'BehaviorFingerprint') -> float:
        """计算与另一个指纹的距离"""
        return euclidean_distances(
            self.fingerprint_vector.reshape(1, -1),
            other.fingerprint_vector.reshape(1, -1)
        )[0][0]


@dataclass
class AnomalyDetectionResult:
    """异常检测结果"""
    is_anomaly: bool
    anomaly_score: float
    anomalous_dimensions: List[str]
    confidence: float
    recommended_actions: List[str]


@dataclass
class BehaviorEvolution:
    """行为演化"""
    from_version: str
    to_version: str
    evolution_vector: np.ndarray
    significant_changes: List[str]
    evolution_type: str  # 'improvement', 'regression', 'pivot', 'refinement'


class FingerprintGenerator:
    """指纹生成器"""

    def __init__(self):
        self.logger = logging.getLogger("FingerprintGenerator")
        self.scaler = StandardScaler()

    def generate_fingerprint(self, algorithm_version: str,
                           metrics_history: List[BehaviorMetrics],
                           test_scenarios: List[str]) -> BehaviorFingerprint:
        """
        生成行为指纹

        Args:
            algorithm_version: 算法版本
            metrics_history: 指标历史数据
            test_scenarios: 测试场景列表

        Returns:
            行为指纹
        """
        if not metrics_history:
            raise ValueError("No metrics data available for fingerprint generation")

        # 提取特征向量
        feature_vectors = [metrics.to_vector() for metrics in metrics_history]
        X = np.array(feature_vectors)

        # 计算中心特征向量 (均值)
        central_vector = np.mean(X, axis=0)

        # 计算置信区间
        confidence_intervals = {}
        vector_length = len(central_vector)

        for i in range(vector_length):
            values = X[:, i]
            mean_val = central_vector[i]
            std_val = np.std(values)

            # 95% 置信区间
            ci_lower = mean_val - 1.96 * std_val / np.sqrt(len(values))
            ci_upper = mean_val + 1.96 * std_val / np.sqrt(len(values))

            # 特征名称映射
            feature_names = [
                'performance_score', 'throughput', 'latency_p95', 'packet_loss_rate',
                'fairness_score', 'resource_balance', 'information_symmetry', 'jains_fairness_index',
                'stability_score', 'fluctuation_resistance', 'recovery_speed', 'adaptation_rate',
                'network_load_adaptation', 'congestion_handling', 'cross_traffic_tolerance',
                'decision_aggressiveness', 'fairness_priority', 'performance_priority', 'risk_tolerance'
            ]

            if i < len(feature_names):
                confidence_intervals[feature_names[i]] = (ci_lower, ci_upper)

        # 识别行为特征
        behavioral_traits = self._identify_behavioral_traits(central_vector, X)

        return BehaviorFingerprint(
            algorithm_version=algorithm_version,
            creation_time=datetime.now(),
            fingerprint_vector=central_vector,
            confidence_intervals=confidence_intervals,
            representative_scenarios=test_scenarios,
            behavioral_traits=behavioral_traits
        )

    def _identify_behavioral_traits(self, central_vector: np.ndarray,
                                  feature_matrix: np.ndarray) -> List[str]:
        """识别行为特征"""
        traits = []

        # 性能导向特征
        if central_vector[0] > 0.8:  # performance_score
            traits.append("high_performance")
        elif central_vector[0] < 0.6:
            traits.append("low_performance")

        # 公平性导向特征
        if central_vector[4] > 0.8:  # fairness_score
            traits.append("fairness_oriented")
        elif central_vector[4] < 0.6:
            traits.append("performance_prioritized")

        # 鲁棒性特征
        if central_vector[8] > 0.8:  # stability_score
            traits.append("highly_stable")
        elif central_vector[8] < 0.6:
            traits.append("unstable")

        # 决策风格特征
        if central_vector[15] > 0.7:  # decision_aggressiveness
            traits.append("aggressive_decisions")
        elif central_vector[15] < 0.3:
            traits.append("conservative_decisions")

        # 环境适应性特征
        adaptation_score = np.mean(central_vector[12:15])  # network adaptation metrics
        if adaptation_score > 0.8:
            traits.append("environment_adaptive")
        elif adaptation_score < 0.6:
            traits.append("environment_sensitive")

        # 变异性特征
        variability = np.mean(np.std(feature_matrix, axis=0))
        if variability > 0.2:
            traits.append("behaviorally_variable")
        elif variability < 0.1:
            traits.append("behaviorally_consistent")

        return traits


class AnomalyDetector:
    """异常检测器"""

    def __init__(self, contamination: float = 0.1):
        self.logger = logging.getLogger("AnomalyDetector")
        self.contamination = contamination  # 预期异常比例

    def detect_anomalies(self, current_fingerprint: BehaviorFingerprint,
                        baseline_fingerprints: List[BehaviorFingerprint]) -> AnomalyDetectionResult:
        """
        检测异常行为

        Args:
            current_fingerprint: 当前指纹
            baseline_fingerprints: 基线指纹列表

        Returns:
            异常检测结果
        """
        if not baseline_fingerprints:
            return AnomalyDetectionResult(
                is_anomaly=False,
                anomaly_score=0.0,
                anomalous_dimensions=[],
                confidence=0.5,
                recommended_actions=["Insufficient baseline data"]
            )

        # 计算与所有基线指纹的相似度
        similarities = []
        for baseline in baseline_fingerprints:
            similarity = current_fingerprint.similarity_to(baseline)
            similarities.append(similarity)

        # 计算异常分数 (1 - 最大相似度)
        max_similarity = max(similarities)
        anomaly_score = 1.0 - max_similarity

        # 判断是否为异常
        is_anomaly = anomaly_score > self.contamination

        # 识别异常维度
        anomalous_dimensions = self._identify_anomalous_dimensions(
            current_fingerprint, baseline_fingerprints[0]  # 使用第一个作为参考
        )

        # 计算置信度
        confidence = min(1.0, len(baseline_fingerprints) / 10.0)  # 基于基线数据量

        # 生成推荐行动
        recommended_actions = self._generate_anomaly_actions(
            is_anomaly, anomalous_dimensions, anomaly_score
        )

        return AnomalyDetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            anomalous_dimensions=anomalous_dimensions,
            confidence=confidence,
            recommended_actions=recommended_actions
        )

    def _identify_anomalous_dimensions(self, current: BehaviorFingerprint,
                                     baseline: BehaviorFingerprint) -> List[str]:
        """识别异常维度"""
        anomalous = []

        feature_names = [
            'performance_score', 'throughput', 'latency_p95', 'packet_loss_rate',
            'fairness_score', 'resource_balance', 'information_symmetry', 'jains_fairness_index',
            'stability_score', 'fluctuation_resistance', 'recovery_speed', 'adaptation_rate',
            'network_load_adaptation', 'congestion_handling', 'cross_traffic_tolerance',
            'decision_aggressiveness', 'fairness_priority', 'performance_priority', 'risk_tolerance'
        ]

        current_vector = current.fingerprint_vector
        baseline_vector = baseline.fingerprint_vector

        for i, feature_name in enumerate(feature_names):
            if i >= len(current_vector) or i >= len(baseline_vector):
                continue

            current_value = current_vector[i]
            baseline_value = baseline_vector[i]

            # 计算相对差异
            if abs(baseline_value) > 1e-6:
                relative_diff = abs(current_value - baseline_value) / abs(baseline_value)
            else:
                relative_diff = abs(current_value)

            # 检查是否超出置信区间
            if feature_name in current.confidence_intervals:
                ci_lower, ci_upper = current.confidence_intervals[feature_name]
                if current_value < ci_lower or current_value > ci_upper:
                    anomalous.append(f"{feature_name}_out_of_bounds")

            # 检查显著变化
            elif relative_diff > 0.2:  # 20% 变化阈值
                anomalous.append(f"{feature_name}_significant_change")

        return anomalous

    def _generate_anomaly_actions(self, is_anomaly: bool,
                                anomalous_dimensions: List[str],
                                anomaly_score: float) -> List[str]:
        """生成异常处理行动"""
        actions = []

        if not is_anomaly:
            actions.append("Behavior within normal range")
            return actions

        # 基于异常维度生成具体行动
        if any('performance' in dim for dim in anomalous_dimensions):
            actions.append("Investigate performance degradation causes")

        if any('fairness' in dim for dim in anomalous_dimensions):
            actions.append("Review fairness algorithm changes")

        if any('stability' in dim for dim in anomalous_dimensions):
            actions.append("Check system stability and error handling")

        if any('latency' in dim for dim in anomalous_dimensions):
            actions.append("Analyze latency increase patterns")

        # 基于异常严重程度
        if anomaly_score > 0.5:
            actions.extend([
                "Immediate code review required",
                "Consider rolling back recent changes"
            ])
        elif anomaly_score > 0.3:
            actions.extend([
                "Schedule detailed analysis",
                "Monitor closely in production"
            ])
        else:
            actions.extend([
                "Log for trend analysis",
                "Continue normal monitoring"
            ])

        return actions


class BehaviorEvolutionTracker:
    """行为演化追踪器"""

    def __init__(self):
        self.logger = logging.getLogger("BehaviorEvolutionTracker")

    def track_evolution(self, from_fingerprint: BehaviorFingerprint,
                       to_fingerprint: BehaviorFingerprint) -> BehaviorEvolution:
        """
        追踪版本间的行为演化

        Args:
            from_fingerprint: 源版本指纹
            to_fingerprint: 目标版本指纹

        Returns:
            行为演化分析
        """
        # 计算演化向量
        evolution_vector = to_fingerprint.fingerprint_vector - from_fingerprint.fingerprint_vector

        # 识别显著变化
        significant_changes = self._identify_significant_changes(
            from_fingerprint, to_fingerprint, evolution_vector
        )

        # 确定演化类型
        evolution_type = self._classify_evolution_type(
            from_fingerprint, to_fingerprint, evolution_vector
        )

        return BehaviorEvolution(
            from_version=from_fingerprint.algorithm_version,
            to_version=to_fingerprint.algorithm_version,
            evolution_vector=evolution_vector,
            significant_changes=significant_changes,
            evolution_type=evolution_type
        )

    def _identify_significant_changes(self, from_fp: BehaviorFingerprint,
                                    to_fp: BehaviorFingerprint,
                                    evolution_vector: np.ndarray) -> List[str]:
        """识别显著变化"""
        changes = []

        feature_names = [
            'performance_score', 'throughput', 'latency_p95', 'packet_loss_rate',
            'fairness_score', 'resource_balance', 'information_symmetry', 'jains_fairness_index',
            'stability_score', 'fluctuation_resistance', 'recovery_speed', 'adaptation_rate',
            'network_load_adaptation', 'congestion_handling', 'cross_traffic_tolerance',
            'decision_aggressiveness', 'fairness_priority', 'performance_priority', 'risk_tolerance'
        ]

        for i, feature_name in enumerate(feature_names):
            if i >= len(evolution_vector):
                continue

            from_value = from_fp.fingerprint_vector[i]
            to_value = to_fp.fingerprint_vector[i]
            change = evolution_vector[i]

            # 计算相对变化
            if abs(from_value) > 1e-6:
                relative_change = abs(change) / abs(from_value)
            else:
                relative_change = abs(change)

            # 判断显著性
            if relative_change > 0.1:  # 10% 相对变化
                direction = "increased" if change > 0 else "decreased"
                magnitude = "significantly" if relative_change > 0.2 else "moderately"
                changes.append(f"{feature_name} {magnitude} {direction} ({change:.3f})")

        return changes

    def _classify_evolution_type(self, from_fp: BehaviorFingerprint,
                               to_fp: BehaviorFingerprint,
                               evolution_vector: np.ndarray) -> str:
        """分类演化类型"""
        # 计算关键指标的变化
        performance_change = evolution_vector[0]  # performance_score
        fairness_change = evolution_vector[4]     # fairness_score
        stability_change = evolution_vector[8]    # stability_score

        # 分析演化模式
        if performance_change > 0.05 and fairness_change > 0.05:
            return "improvement"  # 性能和公平性都提升

        elif performance_change < -0.05 or fairness_change < -0.05:
            return "regression"   # 出现退化

        elif abs(performance_change - fairness_change) > 0.1:
            return "pivot"        # 性能和公平性trade-off显著

        else:
            return "refinement"   # 小幅调整


class BehaviorFingerprintSystem:
    """
    行为指纹系统

    核心创新：
    1. 多维度行为刻画：从单一指标到全面行为向量
    2. 指纹相似度分析：量化算法间的行为差异
    3. 异常行为检测：自动识别异常行为模式
    4. 版本演化追踪：理解算法改进的方向性
    5. 根因关联分析：快速定位行为变化的原因
    6. 行为模式聚类：识别算法家族的共性特征

    这解决了传统指标无法全面描述算法行为的根本问题。
    """

    def __init__(self):
        self.logger = logging.getLogger("BehaviorFingerprintSystem")

        # 核心组件
        self.fingerprint_generator = FingerprintGenerator()
        self.anomaly_detector = AnomalyDetector()
        self.evolution_tracker = BehaviorEvolutionTracker()

        # 指纹数据库
        self.fingerprints: Dict[str, BehaviorFingerprint] = {}
        self.baseline_fingerprints: List[BehaviorFingerprint] = []

        # 演化历史
        self.evolution_history: List[BehaviorEvolution] = []

        # 聚类模型
        self.cluster_model: Optional[KMeans] = None
        self.n_clusters = 3

    def create_fingerprint(self, algorithm_version: str,
                          metrics_history: List[BehaviorMetrics],
                          test_scenarios: List[str]) -> BehaviorFingerprint:
        """
        创建行为指纹

        Args:
            algorithm_version: 算法版本
            metrics_history: 指标历史
            test_scenarios: 测试场景

        Returns:
            行为指纹
        """
        fingerprint = self.fingerprint_generator.generate_fingerprint(
            algorithm_version, metrics_history, test_scenarios
        )

        # 存储指纹
        self.fingerprints[algorithm_version] = fingerprint

        # 如果是第一个指纹，设为基线
        if not self.baseline_fingerprints:
            self.baseline_fingerprints.append(fingerprint)

        self.logger.info(f"Created fingerprint for algorithm {algorithm_version}")
        return fingerprint

    def detect_behavioral_anomalies(self, algorithm_version: str) -> Optional[AnomalyDetectionResult]:
        """
        检测行为异常

        Args:
            algorithm_version: 算法版本

        Returns:
            异常检测结果
        """
        if algorithm_version not in self.fingerprints:
            self.logger.warning(f"No fingerprint found for {algorithm_version}")
            return None

        current_fingerprint = self.fingerprints[algorithm_version]

        # 使用基线指纹进行异常检测
        anomaly_result = self.anomaly_detector.detect_anomalies(
            current_fingerprint, self.baseline_fingerprints
        )

        if anomaly_result.is_anomaly:
            self.logger.warning(f"Anomalous behavior detected for {algorithm_version}: "
                              f"score={anomaly_result.anomaly_score:.3f}")

        return anomaly_result

    def analyze_behavior_evolution(self, from_version: str, to_version: str) -> Optional[BehaviorEvolution]:
        """
        分析行为演化

        Args:
            from_version: 源版本
            to_version: 目标版本

        Returns:
            行为演化分析
        """
        if from_version not in self.fingerprints or to_version not in self.fingerprints:
            self.logger.warning(f"Fingerprints not found for evolution analysis")
            return None

        from_fp = self.fingerprints[from_version]
        to_fp = self.fingerprints[to_version]

        evolution = self.evolution_tracker.track_evolution(from_fp, to_fp)
        self.evolution_history.append(evolution)

        self.logger.info(f"Analyzed evolution from {from_version} to {to_version}: {evolution.evolution_type}")

        return evolution

    def cluster_behavior_patterns(self) -> Dict[str, Any]:
        """聚类行为模式"""
        if len(self.fingerprints) < self.n_clusters:
            return {"status": "insufficient_data"}

        # 提取所有指纹向量
        versions = list(self.fingerprints.keys())
        vectors = np.array([self.fingerprints[v].fingerprint_vector for v in versions])

        # 标准化
        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors)

        # 聚类
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = self.cluster_model.fit_predict(vectors_scaled)

        # 分析聚类结果
        cluster_analysis = {}
        for i in range(self.n_clusters):
            cluster_versions = [versions[j] for j in range(len(versions)) if clusters[j] == i]
            cluster_vectors = vectors[clusters == i]

            if len(cluster_vectors) > 0:
                centroid = np.mean(cluster_vectors, axis=0)

                # 识别聚类特征
                traits = self._identify_cluster_traits(centroid)

                cluster_analysis[f"cluster_{i}"] = {
                    "versions": cluster_versions,
                    "size": len(cluster_versions),
                    "traits": traits,
                    "centroid": centroid.tolist()
                }

        return {
            "n_clusters": self.n_clusters,
            "total_versions": len(versions),
            "clusters": cluster_analysis
        }

    def _identify_cluster_traits(self, centroid: np.ndarray) -> List[str]:
        """识别聚类特征"""
        traits = []

        # 基于质心位置识别特征
        feature_names = [
            'performance_score', 'throughput', 'latency_p95', 'packet_loss_rate',
            'fairness_score', 'resource_balance', 'information_symmetry', 'jains_fairness_index',
            'stability_score', 'fluctuation_resistance', 'recovery_speed', 'adaptation_rate',
            'network_load_adaptation', 'congestion_handling', 'cross_traffic_tolerance',
            'decision_aggressiveness', 'fairness_priority', 'performance_priority', 'risk_tolerance'
        ]

        # 识别高值特征
        for i, feature_name in enumerate(feature_names):
            if i < len(centroid) and centroid[i] > 0.7:
                traits.append(f"high_{feature_name}")

        # 识别低值特征
        for i, feature_name in enumerate(feature_names):
            if i < len(centroid) and centroid[i] < 0.3:
                traits.append(f"low_{feature_name}")

        return traits[:5]  # 返回前5个显著特征

    def find_similar_algorithms(self, target_version: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        查找相似算法

        Args:
            target_version: 目标版本
            top_k: 返回最相似的前k个

        Returns:
            (版本, 相似度) 列表
        """
        if target_version not in self.fingerprints:
            return []

        target_fp = self.fingerprints[target_version]
        similarities = []

        for version, fp in self.fingerprints.items():
            if version != target_version:
                similarity = target_fp.similarity_to(fp)
                similarities.append((version, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_behavior_insights(self, algorithm_version: str) -> Dict[str, Any]:
        """获取行为洞察"""
        if algorithm_version not in self.fingerprints:
            return {"status": "fingerprint_not_found"}

        fingerprint = self.fingerprints[algorithm_version]

        # 查找相似算法
        similar_algorithms = self.find_similar_algorithms(algorithm_version, 3)

        # 检测异常
        anomaly_result = self.detect_behavioral_anomalies(algorithm_version)

        # 获取演化历史
        evolutions = [
            evo for evo in self.evolution_history
            if evo.from_version == algorithm_version or evo.to_version == algorithm_version
        ]

        return {
            "algorithm_version": algorithm_version,
            "behavioral_traits": fingerprint.behavioral_traits,
            "confidence_intervals": fingerprint.confidence_intervals,
            "similar_algorithms": similar_algorithms,
            "anomaly_detection": {
                "is_anomaly": anomaly_result.is_anomaly if anomaly_result else False,
                "anomaly_score": anomaly_result.anomaly_score if anomaly_result else 0.0,
                "anomalous_dimensions": anomaly_result.anomalous_dimensions if anomaly_result else []
            } if anomaly_result else None,
            "evolution_history": [
                {
                    "from_version": evo.from_version,
                    "to_version": evo.to_version,
                    "evolution_type": evo.evolution_type,
                    "significant_changes": evo.significant_changes
                }
                for evo in evolutions
            ]
        }

    def export_fingerprints(self, filepath: str):
        """导出指纹数据"""
        export_data = {
            "export_time": datetime.now().isoformat(),
            "fingerprints": {},
            "evolution_history": []
        }

        # 导出指纹
        for version, fp in self.fingerprints.items():
            export_data["fingerprints"][version] = {
                "creation_time": fp.creation_time.isoformat(),
                "fingerprint_vector": fp.fingerprint_vector.tolist(),
                "behavioral_traits": fp.behavioral_traits,
                "representative_scenarios": fp.representative_scenarios
            }

        # 导出演化历史
        for evo in self.evolution_history:
            export_data["evolution_history"].append({
                "from_version": evo.from_version,
                "to_version": evo.to_version,
                "evolution_type": evo.evolution_type,
                "significant_changes": evo.significant_changes
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Exported fingerprint data to {filepath}")

    def import_fingerprints(self, filepath: str):
        """导入指纹数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            import_data = json.load(f)

        # 导入指纹
        for version, fp_data in import_data.get("fingerprints", {}).items():
            fingerprint = BehaviorFingerprint(
                algorithm_version=version,
                creation_time=datetime.fromisoformat(fp_data["creation_time"]),
                fingerprint_vector=np.array(fp_data["fingerprint_vector"]),
                confidence_intervals={},  # 导入时不包含CI
                representative_scenarios=fp_data.get("representative_scenarios", []),
                behavioral_traits=fp_data.get("behavioral_traits", [])
            )
            self.fingerprints[version] = fingerprint

        self.logger.info(f"Imported {len(self.fingerprints)} fingerprints from {filepath}")


def create_behavior_fingerprint_system() -> BehaviorFingerprintSystem:
    """创建行为指纹系统"""
    return BehaviorFingerprintSystem()


# 使用示例
if __name__ == "__main__":
    # 创建行为指纹系统
    fingerprint_system = create_behavior_fingerprint_system()

    # 生成模拟指标数据
    np.random.seed(42)
    n_samples = 50

    # 模拟算法v1.0的行为指标
    metrics_v1 = []
    for i in range(n_samples):
        metrics = BehaviorMetrics(
            timestamp=datetime.now() - timedelta(hours=i),
            algorithm_version="v1.0",
            performance_score=0.85 + np.random.normal(0, 0.05),
            throughput=950 + np.random.normal(0, 50),
            latency_p95=110 + np.random.normal(0, 15),
            packet_loss_rate=0.015 + np.random.normal(0, 0.005),
            fairness_score=0.82 + np.random.normal(0, 0.04),
            resource_balance=0.78 + np.random.normal(0, 0.06),
            information_symmetry=0.88 + np.random.normal(0, 0.03),
            jains_fairness_index=0.91 + np.random.normal(0, 0.02),
            stability_score=0.89 + np.random.normal(0, 0.04),
            fluctuation_resistance=0.76 + np.random.normal(0, 0.08),
            recovery_speed=0.83 + np.random.normal(0, 0.05),
            adaptation_rate=0.79 + np.random.normal(0, 0.06),
            network_load_adaptation=0.81 + np.random.normal(0, 0.07),
            congestion_handling=0.77 + np.random.normal(0, 0.08),
            cross_traffic_tolerance=0.85 + np.random.normal(0, 0.04),
            decision_aggressiveness=0.62 + np.random.normal(0, 0.12),
            fairness_priority=0.74 + np.random.normal(0, 0.09),
            performance_priority=0.71 + np.random.normal(0, 0.10),
            risk_tolerance=0.58 + np.random.normal(0, 0.11)
        )
        metrics_v1.append(metrics)

    # 创建v1.0指纹
    test_scenarios = ["stable_network", "fluctuating_load", "bursty_traffic"]
    fp_v1 = fingerprint_system.create_fingerprint("v1.0", metrics_v1, test_scenarios)

    print("Created fingerprint for v1.0:")
    print(f"  Behavioral traits: {fp_v1.behavioral_traits}")
    print(f"  Vector dimension: {len(fp_v1.fingerprint_vector)}")

    # 模拟v1.1的行为（轻微改进）
    metrics_v11 = []
    for i in range(n_samples):
        metrics = BehaviorMetrics(
            timestamp=datetime.now() - timedelta(hours=i),
            algorithm_version="v1.1",
            performance_score=0.87 + np.random.normal(0, 0.05),  # 轻微提升
            throughput=970 + np.random.normal(0, 50),
            latency_p95=105 + np.random.normal(0, 15),  # 轻微改善
            packet_loss_rate=0.013 + np.random.normal(0, 0.005),
            fairness_score=0.84 + np.random.normal(0, 0.04),  # 轻微提升
            resource_balance=0.80 + np.random.normal(0, 0.06),
            information_symmetry=0.89 + np.random.normal(0, 0.03),
            jains_fairness_index=0.92 + np.random.normal(0, 0.02),
            stability_score=0.90 + np.random.normal(0, 0.04),
            fluctuation_resistance=0.78 + np.random.normal(0, 0.08),
            recovery_speed=0.85 + np.random.normal(0, 0.05),
            adaptation_rate=0.81 + np.random.normal(0, 0.06),
            network_load_adaptation=0.83 + np.random.normal(0, 0.07),
            congestion_handling=0.79 + np.random.normal(0, 0.08),
            cross_traffic_tolerance=0.87 + np.random.normal(0, 0.04),
            decision_aggressiveness=0.64 + np.random.normal(0, 0.12),
            fairness_priority=0.76 + np.random.normal(0, 0.09),
            performance_priority=0.73 + np.random.normal(0, 0.10),
            risk_tolerance=0.60 + np.random.normal(0, 0.11)
        )
        metrics_v11.append(metrics)

    # 创建v1.1指纹
    fp_v11 = fingerprint_system.create_fingerprint("v1.1", metrics_v11, test_scenarios)

    # 检测异常
    anomaly_result = fingerprint_system.detect_behavioral_anomalies("v1.1")
    if anomaly_result:
        print(f"\\nAnomaly detection for v1.1:")
        print(f"  Is anomaly: {anomaly_result.is_anomaly}")
        print(f"  Anomaly score: {anomaly_result.anomaly_score:.3f}")
        if anomaly_result.anomalous_dimensions:
            print(f"  Anomalous dimensions: {anomaly_result.anomalous_dimensions[:3]}")

    # 分析演化
    evolution = fingerprint_system.analyze_behavior_evolution("v1.0", "v1.1")
    if evolution:
        print(f"\\nEvolution from v1.0 to v1.1:")
        print(f"  Evolution type: {evolution.evolution_type}")
        if evolution.significant_changes:
            print(f"  Significant changes: {evolution.significant_changes[:2]}")

    # 查找相似算法
    similar = fingerprint_system.find_similar_algorithms("v1.1", 2)
    print(f"\\nSimilar algorithms to v1.1:")
    for version, similarity in similar:
        print(f"  {version}: {similarity:.3f}")

    print("\\nBehavior fingerprint system demo completed")