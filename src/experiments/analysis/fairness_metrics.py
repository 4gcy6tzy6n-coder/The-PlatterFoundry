#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Fairness Metrics - 综合公平性指标

创新贡献：
1. Jain's Fairness Index: 资源分配公平性度量
2. 熵基公平性指标: 基于信息论的公平性评估
3. 多维度公平性综合评估
4. 统计显著性边界分析

这解决了传统公平性指标单一、无法全面评估的问题。
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class FairnessDimension(Enum):
    """公平性维度枚举"""
    RESOURCE_ALLOCATION = "resource_allocation"
    INFORMATION_SYMMETRY = "information_symmetry"
    STATISTICAL_SIMILARITY = "statistical_similarity"
    PERFORMANCE_CONSISTENCY = "performance_consistency"
    OPPORTUNITY_EQUALITY = "opportunity_equality"


@dataclass
class ComprehensiveFairnessMetrics:
    """综合公平性指标"""
    jains_index: float = 1.0
    entropy_based_index: float = 1.0
    resource_balance_score: float = 1.0
    information_symmetry_score: float = 1.0
    statistical_similarity_score: float = 1.0
    performance_consistency_score: float = 1.0
    overall_fairness_score: float = 1.0
    confidence_lower: float = 0.0
    confidence_upper: float = 1.0


class ComprehensiveFairnessAnalyzer:
    """
    综合公平性分析器
    
    提供多维度公平性评估：
    1. Jain's Fairness Index
    2. 熵基公平性指标
    3. 统计相似性分析
    4. 性能一致性分析
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def jains_fairness_index(self, values: List[float]) -> float:
        """
        计算Jain's Fairness Index
        
        JFI = (Σx_i)² / (n * Σx_i²)
        
        范围：[0, 1]，1表示完全公平
        """
        n = len(values)
        if n < 2:
            return 1.0
        
        values_arr = np.array(values, dtype=float)
        
        sum_values = np.sum(values_arr)
        if sum_values == 0:
            return 1.0
        
        sum_squares = np.sum(values_arr ** 2)
        
        jfi = (sum_values ** 2) / (n * sum_squares)
        
        return min(1.0, max(0.0, jfi))
    
    def entropy_based_fairness(self, values: List[float]) -> float:
        """
        计算熵基公平性指数
        
        H = -Σ(p_i * log(p_i)) / log(n)
        
        创新点：使用信息熵度量分配均匀性
        """
        n = len(values)
        if n < 2:
            return 1.0
        
        total = sum(values)
        if total == 0:
            return 1.0
        
        proportions = [v / total for v in values]
        
        entropy = 0.0
        for p in proportions:
            if p > 0:
                entropy -= p * np.log(p + 1e-10)
        
        max_entropy = np.log(n)
        
        return entropy / max_entropy if max_entropy > 0 else 1.0
    
    def coefficient_of_variation_fairness(self, values: List[float]) -> float:
        """
        基于变异系数的公平性
        
        CV = σ / μ
        
        公平性 = 1 / (1 + CV)
        """
        if len(values) < 2:
            return 1.0
        
        mean_val = np.mean(values)
        if mean_val == 0:
            return 1.0
        
        cv = np.std(values) / mean_val
        
        return 1.0 / (1.0 + cv)
    
    def analyze_resource_allocation_fairness(
        self,
        resource_allocations: Dict[str, float]
    ) -> Dict[str, float]:
        """
        分析资源分配公平性
        
        创新点：多角度评估资源分配公平性
        """
        values = list(resource_allocations.values())
        
        if len(values) < 2:
            return {
                'jains_index': 1.0,
                'entropy_fairness': 1.0,
                'cv_fairness': 1.0,
                'overall': 1.0
            }
        
        jains = self.jains_fairness_index(values)
        entropy_fairness = self.entropy_based_fairness(values)
        cv_fairness = self.coefficient_of_variation_fairness(values)
        
        overall = (jains + entropy_fairness + cv_fairness) / 3
        
        return {
            'jains_index': jains,
            'entropy_fairness': entropy_fairness,
            'cv_fairness': cv_fairness,
            'overall': overall
        }
    
    def analyze_statistical_similarity(
        self,
        algorithm_results: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        分析统计相似性
        
        创新点：综合多种统计检验评估分布相似性
        """
        if len(algorithm_results) < 2:
            return {'similarity': 1.0, 'significant_difference': False}
        
        algo_names = list(algorithm_results.keys())
        results_list = [algorithm_results[name] for name in algo_names]
        
        ks_similarities = []
        for i in range(len(results_list)):
            for j in range(i + 1, len(results_list)):
                data1, data2 = results_list[i], results_list[j]
                if len(data1) >= 2 and len(data2) >= 2:
                    stat, _ = self._ks_statistic(data1, data2)
                    similarity = 1 - stat
                    ks_similarities.append(similarity)
        
        avg_similarity = np.mean(ks_similarities) if ks_similarities else 1.0
        
        all_data = [x for results in results_list for x in results]
        mean_perf = np.mean(all_data)
        std_perf = np.std(all_data)
        
        cv = std_perf / (mean_perf + 1e-10) if mean_perf > 0 else 0
        consistency = 1 / (1 + cv)
        
        return {
            'avg_ks_similarity': avg_similarity,
            'performance_consistency': consistency,
            'significant_difference': avg_similarity < (1 - self.significance_level),
            'effect_size': cv
        }
    
    def _ks_statistic(self, data1: List[float], data2: List[float]) -> Tuple[float, float]:
        """计算KS统计量"""
        from scipy import stats
        
        if len(data1) < 2 or len(data2) < 2:
            return 0.0, 1.0
        
        return stats.ks_2samp(data1, data2)
    
    def calculate_comprehensive_fairness(
        self,
        resource_allocations: Dict[str, float],
        information_access: Dict[str, bool],
        performance_results: Dict[str, List[float]]
    ) -> ComprehensiveFairnessMetrics:
        """
        计算综合公平性指标
        
        创新点：多维度公平性综合评估
        """
        metrics = ComprehensiveFairnessMetrics()
        
        resource_analysis = self.analyze_resource_allocation_fairness(resource_allocations)
        metrics.resource_balance_score = resource_analysis['overall']
        
        metrics.information_symmetry_score = self._calculate_information_symmetry(information_access)
        
        stat_analysis = self.analyze_statistical_similarity(performance_results)
        metrics.statistical_similarity_score = stat_analysis['avg_ks_similarity']
        metrics.performance_consistency_score = stat_analysis['performance_consistency']
        
        values = list(resource_allocations.values())
        metrics.jains_index = self.jains_fairness_index(values)
        metrics.entropy_based_index = self.entropy_based_fairness(values)
        
        weights = {
            'resource': 0.30,
            'information': 0.25,
            'statistical': 0.25,
            'performance': 0.20
        }
        
        metrics.overall_fairness_score = (
            weights['resource'] * metrics.resource_balance_score +
            weights['information'] * metrics.information_symmetry_score +
            weights['statistical'] * metrics.statistical_similarity_score +
            weights['performance'] * metrics.performance_consistency_score
        )
        
        metrics.confidence_lower = max(0, metrics.overall_fairness_score - 0.1)
        metrics.confidence_upper = min(1, metrics.overall_fairness_score + 0.1)
        
        return metrics

    def analyze_comprehensive_fairness(self, algorithm_results: Dict[str, List[Dict[str, Any]]]) -> ComprehensiveFairnessMetrics:
        """
        分析算法结果的综合公平性

        适用于实验三的数据格式
        """
        metrics = ComprehensiveFairnessMetrics()

        if not algorithm_results:
            return metrics

        # 提取资源分配（基于成功率）
        resource_allocations = {}
        information_access = {}
        performance_results = {}

        for algo_name, results in algorithm_results.items():
            if not results:
                continue

            # 计算平均成功率作为资源分配指标
            success_rates = [r.get('success_rate', 0) for r in results]
            resource_allocations[algo_name] = np.mean(success_rates)

            # 检查是否有隐藏状态访问
            has_hidden_access = any(r.get('had_hidden_state_access', False) for r in results)
            information_access[algo_name] = has_hidden_access

            # 性能结果
            performance_results[algo_name] = success_rates

        # 如果没有数据，返回默认值
        if not resource_allocations:
            return metrics

        # 使用现有的综合公平性计算
        return self.calculate_comprehensive_fairness(
            resource_allocations,
            information_access,
            performance_results
        )

    def _calculate_information_symmetry(self, information_access: Dict[str, bool]) -> float:
        """计算信息对称性分数"""
        if not information_access:
            return 1.0
        
        has_hidden_access = sum(1 for has_access in information_access.values() if has_access)
        symmetry = 1 - has_hidden_access / len(information_access)
        
        return symmetry
    
    def generate_fairness_report(
        self,
        fairness_metrics: ComprehensiveFairnessMetrics
    ) -> Dict[str, Any]:
        """生成公平性报告"""
        return {
            'summary': {
                'overall_score': fairness_metrics.overall_fairness_score,
                'grade': self._get_fairness_grade(fairness_metrics.overall_fairness_score),
                'passed': fairness_metrics.overall_fairness_score > 0.8
            },
            'dimensions': {
                'resource_allocation': {
                    'score': fairness_metrics.resource_balance_score,
                    'jains_index': fairness_metrics.jains_index,
                    'interpretation': self._interpret_score(fairness_metrics.resource_balance_score)
                },
                'information_symmetry': {
                    'score': fairness_metrics.information_symmetry_score,
                    'interpretation': self._interpret_score(fairness_metrics.information_symmetry_score)
                },
                'statistical_similarity': {
                    'score': fairness_metrics.statistical_similarity_score,
                    'interpretation': self._interpret_score(fairness_metrics.statistical_similarity_score)
                },
                'performance_consistency': {
                    'score': fairness_metrics.performance_consistency_score,
                    'interpretation': self._interpret_score(fairness_metrics.performance_consistency_score)
                }
            },
            'confidence_interval': [
                fairness_metrics.confidence_lower,
                fairness_metrics.confidence_upper
            ]
        }
    
    def _get_fairness_grade(self, score: float) -> str:
        """获取公平性等级"""
        if score >= 0.9:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Good)"
        elif score >= 0.7:
            return "B (Fair)"
        elif score >= 0.6:
            return "C (Poor)"
        else:
            return "F (Fail)"
    
    def _interpret_score(self, score: float) -> str:
        """解释分数"""
        if score >= 0.9:
            return "Excellent - No significant fairness issues"
        elif score >= 0.8:
            return "Good - Minor fairness concerns"
        elif score >= 0.7:
            return "Fair - Some fairness issues detected"
        elif score >= 0.6:
            return "Poor - Significant fairness issues"
        else:
            return "Fail - Major fairness problems"
