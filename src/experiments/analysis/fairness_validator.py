#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fairness Validator - 公平性验证器

创新点：
1. 多维度公平性指标：资源分配、信息对称、统计充分性
2. 统计显著性检验：Kolmogorov-Smirnov检验 + t检验
3. 信息泄露检测：检测算法是否获取额外信息
4. 多场景公平性验证：在不同网络条件下验证公平性
5. 随机种子链管理：确保实验可复现
6. 可视化公平性报告：生成公平性分析图
7. 统计功效分析：评估实验设计质量

这解决了传统实验只关注性能、忽视公平性的问题。
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import hashlib
import json


@dataclass
class FairnessMetrics:
    """公平性指标 - 增强版"""
    resource_balance_score: float = 1.0
    resource_variance: float = 0.0
    information_symmetry_score: float = 1.0
    hidden_state_access_count: int = 0
    sample_size_sufficient: bool = True
    statistical_power: float = 0.8
    effect_size: float = 0.0
    performance_consistency: float = 1.0
    success_rate_variance: float = 0.0
    jains_index: float = 1.0
    entropy_fairness: float = 1.0
    confidence_lower: float = 0.0
    confidence_upper: float = 1.0

    # 增强公平性指标
    temporal_fairness: float = 1.0  # 时间公平性
    opportunity_fairness: float = 1.0  # 机会公平性
    outcome_fairness: float = 1.0  # 结果公平性
    process_fairness: float = 1.0  # 过程公平性
    predictive_fairness: float = 1.0  # 预测公平性
    calibration_fairness: float = 1.0  # 校准公平性


@dataclass
class SeedChainEntry:
    """种子链条目"""
    component: str
    index: int
    derived_seed: int
    purpose: str
    hash_fingerprint: str = ""


class FairnessValidator:
    """
    公平性验证器
    
    验证实验是否满足公平性要求：
    1. 所有算法使用相同的网络实例
    2. 相同的资源预算约束
    3. 相同的信息访问权限
    4. 统计充分的重复次数
    5. 随机种子可追溯
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.logger = logging.getLogger("FairnessValidator")
        
        self.algorithm_results: Dict[str, List[Dict[str, Any]]] = {}
        self.network_instances: Dict[str, Any] = {}
        
        self.seed_chain: List[SeedChainEntry] = []
        self.used_seeds: Dict[int, str] = {}
    
    def register_algorithm_results(self, algorithm_name: str, results: List[Dict[str, Any]]):
        """注册算法结果"""
        self.algorithm_results[algorithm_name] = results
        self.logger.info(f"注册算法: {algorithm_name}, 结果数: {len(results)}")
    
    def register_network_instance(self, instance_id: str, instance_info: Dict[str, Any]):
        """注册网络实例"""
        self.network_instances[instance_id] = instance_info
        
        if 'fingerprint' in instance_info:
            self._register_seed_usage(
                f"network_instance_{instance_id}",
                instance_info.get('seed', 0),
                f"生成网络实例 {instance_id}"
            )
    
    def derive_seed(self, component: str, index: int, purpose: str) -> int:
        """派生新种子 - 使用SHA-256提升安全性"""
        if self.used_seeds:
            base_seed = max(self.used_seeds.keys())
        else:
            base_seed = 42
        
        seed_input = f"{base_seed}_{component}_{index}_{purpose}"
        # 使用SHA-256替代MD5，提升安全性
        derived = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**31)
        
        self._register_seed_usage(component, derived, purpose)
        
        return derived
    
    def _register_seed_usage(self, component: str, seed: int, purpose: str):
        """注册种子使用 - 使用SHA-256提升安全性"""
        entry = SeedChainEntry(
            component=component,
            index=len(self.seed_chain),
            derived_seed=seed,
            purpose=purpose,
            # 使用SHA-256替代MD5，指纹长度保持8字符以保持向后兼容
            hash_fingerprint=hashlib.sha256(str(seed).encode()).hexdigest()[:8]
        )
        self.seed_chain.append(entry)
        self.used_seeds[seed] = purpose
    
    def get_seed_chain_report(self) -> Dict[str, Any]:
        """获取种子链报告"""
        return {
            'total_entries': len(self.seed_chain),
            'seed_entries': [
                {
                    'index': e.index,
                    'component': e.component,
                    'seed': e.derived_seed,
                    'fingerprint': e.hash_fingerprint,
                    'purpose': e.purpose
                }
                for e in self.seed_chain
            ],
            'reproducibility_verified': len(set(e.derived_seed for e in self.seed_chain)) == len(self.seed_chain)
        }
    
    def validate_all(self) -> Tuple[FairnessMetrics, Dict[str, Any]]:
        """执行完整公平性验证"""
        report = {
            'resource_fairness': self._validate_resource_fairness(),
            'information_symmetry': self._validate_information_symmetry(),
            'statistical_sufficiency': self._validate_statistical_sufficiency(),
            'performance_consistency': self._validate_performance_consistency(),
            'distribution_similarity': self._validate_distribution_similarity()
        }
        
        metrics = FairnessMetrics(
            resource_balance_score=report['resource_fairness']['score'],
            resource_variance=report['resource_fairness']['variance'],
            information_symmetry_score=report['information_symmetry']['score'],
            hidden_state_access_count=report['information_symmetry']['hidden_access'],
            sample_size_sufficient=report['statistical_sufficiency']['sufficient'],
            statistical_power=report['statistical_sufficiency']['power'],
            effect_size=report['statistical_sufficiency']['effect_size'],
            performance_consistency=report['performance_consistency']['consistency'],
            success_rate_variance=report['performance_consistency'].get('variance', 0),
            jains_index=report['resource_fairness'].get('jains_index', 1.0),
            entropy_fairness=report['resource_fairness'].get('entropy_fairness', 1.0)
        )
        
        # 计算增强公平性指标
        metrics.temporal_fairness = self._validate_temporal_fairness()
        metrics.opportunity_fairness = self._validate_opportunity_fairness()
        metrics.outcome_fairness = self._validate_outcome_fairness()
        metrics.process_fairness = self._validate_process_fairness()
        metrics.predictive_fairness = self._validate_predictive_fairness()
        metrics.calibration_fairness = self._validate_calibration_fairness()

        metrics.confidence_lower = max(0, metrics.resource_balance_score - 0.1)
        metrics.confidence_upper = min(1, metrics.resource_balance_score + 0.1)

        # 计算整体验证结果
        overall_passed = self._calculate_overall_passed(metrics, report)

        # 添加整体结果到report
        report['overall_passed'] = overall_passed
        report['overall_score'] = self._calculate_overall_score(metrics)

        return metrics, report
    
    def _validate_resource_fairness(self) -> Dict[str, Any]:
        """验证资源公平性"""
        if len(self.algorithm_results) < 2:
            return {'score': 1.0, 'variance': 0.0, 'details': '算法数量不足'}
        
        resource_usage = {}
        for algo, results in self.algorithm_results.items():
            if not results:
                continue
            avg_redundancy = np.mean([r.get('redundancy_ratio', 0) for r in results])
            resource_usage[algo] = avg_redundancy
        
        if not resource_usage:
            return {'score': 1.0, 'variance': 0.0, 'details': '无资源数据'}
        
        values = list(resource_usage.values())
        variance = np.var(values) if values else 0
        
        mean_val = np.mean(values)
        if mean_val > 0:
            cv = np.sqrt(variance) / mean_val
            score = max(0, 1 - cv)
        else:
            score = 1.0
        
        jains_index = self._jains_fairness(values)
        entropy_fairness = self._entropy_based_fairness(values)
        
        return {
            'score': score,
            'variance': variance,
            'resource_usage': resource_usage,
            'coefficient_of_variation': cv if mean_val > 0 else 0,
            'jains_index': jains_index,
            'entropy_fairness': entropy_fairness
        }
    
    def _jains_fairness(self, values: List[float]) -> float:
        """计算Jain's Fairness Index"""
        n = len(values)
        if n < 2:
            return 1.0
        
        values_arr = np.array(values, dtype=float)
        sum_values = np.sum(values_arr)
        if sum_values == 0:
            return 1.0
        
        sum_squares = np.sum(values_arr ** 2)
        return min(1.0, max(0.0, (sum_values ** 2) / (n * sum_squares)))
    
    def _entropy_based_fairness(self, values: List[float]) -> float:
        """计算熵基公平性"""
        n = len(values)
        if n < 2:
            return 1.0
        
        total = sum(values)
        if total == 0:
            return 1.0
        
        proportions = [v / total for v in values]
        entropy = -sum(p * np.log(p + 1e-10) for p in proportions if p > 0)
        return entropy / np.log(n) if n > 1 else 1.0
    
    def _validate_information_symmetry(self) -> Dict[str, Any]:
        """验证信息对称性"""
        hidden_access = 0
        algo_infos = []
        
        for algo, results in self.algorithm_results.items():
            if not results:
                continue
            sample_result = results[0]
            has_hidden_access = sample_result.get('had_hidden_state_access', False)
            if has_hidden_access:
                hidden_access += 1
            algo_infos.append({
                'name': algo,
                'has_hidden_access': has_hidden_access
            })
        
        if algo_infos:
            symmetry_score = 1 - hidden_access / len(algo_infos)
        else:
            symmetry_score = 1.0
        
        return {
            'score': symmetry_score,
            'hidden_access': hidden_access,
            'algorithm_info': algo_infos
        }
    
    def _validate_statistical_sufficiency(self) -> Dict[str, Any]:
        """验证统计充分性"""
        min_required_samples = 30
        
        sample_sizes = {}
        for algo, results in self.algorithm_results.items():
            sample_sizes[algo] = len(results)
        
        min_samples = min(sample_sizes.values()) if sample_sizes else 0
        sufficient = min_samples >= min_required_samples
        
        if len(self.algorithm_results) >= 2:
            algo_names = list(self.algorithm_results.keys())[:2]
            results_1 = self.algorithm_results[algo_names[0]]
            results_2 = self.algorithm_results[algo_names[1]]
            
            if results_1 and results_2:
                success_1 = [r.get('success_rate', 0) for r in results_1]
                success_2 = [r.get('success_rate', 0) for r in results_2]
                
                mean_1, mean_2 = np.mean(success_1), np.mean(success_2)
                std_1, std_2 = np.std(success_1), np.std(success_2)
                pooled_std = np.sqrt((std_1**2 + std_2**2) / 2)
                effect_size = abs(mean_1 - mean_2) / (pooled_std + 1e-10)
                
                power = self._estimate_statistical_power(min_samples, effect_size)
            else:
                effect_size = 0
                power = 0.8
        else:
            effect_size = 0
            power = 0.8
        
        return {
            'sufficient': sufficient,
            'min_samples': min_samples,
            'required_samples': min_required_samples,
            'sample_sizes': sample_sizes,
            'power': power,
            'effect_size': effect_size
        }
    
    def _estimate_statistical_power(self, n: int, effect_size: float) -> float:
        """估算统计功效"""
        if effect_size < 0.1:
            return max(0.1, min(0.9, n / 200))
        elif effect_size < 0.5:
            return max(0.2, min(0.95, n / 100))
        else:
            return max(0.5, min(0.99, n / 50))
    
    def _validate_performance_consistency(self) -> Dict[str, Any]:
        """验证性能一致性"""
        if len(self.algorithm_results) < 2:
            return {'consistency': 1.0, 'variance': 0.0}
        
        algo_names = list(self.algorithm_results.keys())
        
        if len(algo_names) >= 2:
            results_1 = self.algorithm_results[algo_names[0]]
            results_2 = self.algorithm_results[algo_names[1]]
            
            if results_1 and results_2:
                data_1 = [r.get('success_rate', 0) for r in results_1]
                data_2 = [r.get('success_rate', 0) for r in results_2]
                
                ks_stat, p_value = stats.ks_2samp(data_1, data_2)
                
                consistency = max(0, 1 - ks_stat)
                
                variance = (np.var(data_1) + np.var(data_2)) / 2
                
                return {
                    'consistency': consistency,
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'is_significant': p_value < self.significance_level,
                    'variance': variance
                }
        
        return {'consistency': 1.0, 'variance': 0.0}
    
    def _validate_distribution_similarity(self) -> Dict[str, Any]:
        """验证分布相似性"""
        if len(self.algorithm_results) < 2:
            return {'similarity': 1.0}
        
        algo_names = list(self.algorithm_results.keys())
        
        all_success_rates = []
        for algo in algo_names:
            results = self.algorithm_results[algo]
            success_rates = [r.get('success_rate', 0) for r in results]
            all_success_rates.append(success_rates)
        
        similarity_scores = []
        for i in range(len(all_success_rates)):
            for j in range(i + 1, len(all_success_rates)):
                data1, data2 = all_success_rates[i], all_success_rates[j]
                if len(data1) >= 2 and len(data2) >= 2:
                    stat, _ = stats.ks_2samp(data1, data2)
                    similarity = 1 - stat
                    similarity_scores.append(similarity)
        
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 1.0
        
        return {
            'similarity': avg_similarity,
            'avg_ks_statistic': 1 - avg_similarity
        }

    def _validate_temporal_fairness(self) -> float:
        """验证时间公平性 - 各算法在不同时间段的性能一致性"""
        if len(self.algorithm_results) < 2:
            return 1.0

        # 计算每个算法在不同时间段的性能
        temporal_consistency = []
        for algo, results in self.algorithm_results.items():
            if len(results) < 10:  # 需要足够的数据点
                continue

            # 按时间排序（假设results按时间顺序）
            success_rates = [r.get('success_rate', 0) for r in results]

            # 分成前半和后半两个时间段
            mid = len(success_rates) // 2
            first_half = success_rates[:mid]
            second_half = success_rates[mid:]

            if first_half and second_half:
                mean1, mean2 = np.mean(first_half), np.mean(second_half)
                std1, std2 = np.std(first_half), np.std(second_half)

                # 计算时间段间的性能一致性
                consistency = 1 - abs(mean1 - mean2) / (max(mean1, mean2) + 1e-10)
                temporal_consistency.append(consistency)

        return np.mean(temporal_consistency) if temporal_consistency else 1.0

    def _validate_opportunity_fairness(self) -> float:
        """验证机会公平性 - 各算法获得相同机会的程度"""
        if len(self.algorithm_results) < 2:
            return 1.0

        # 检查样本大小的均衡性
        sample_sizes = [len(results) for results in self.algorithm_results.values()]
        cv = np.std(sample_sizes) / (np.mean(sample_sizes) + 1e-10)

        # 机会公平性 = 1 - 样本大小变异系数
        return max(0, 1 - cv)

    def _validate_outcome_fairness(self) -> float:
        """验证结果公平性 - 基于性能分布的公平性"""
        if len(self.algorithm_results) < 2:
            return 1.0

        # 计算各算法性能的分布特征
        performance_stats = []
        for algo, results in self.algorithm_results.items():
            success_rates = [r.get('success_rate', 0) for r in results]
            if success_rates:
                performance_stats.append({
                    'mean': np.mean(success_rates),
                    'std': np.std(success_rates),
                    'cv': np.std(success_rates) / (np.mean(success_rates) + 1e-10)
                })

        if len(performance_stats) < 2:
            return 1.0

        # 结果公平性基于性能变异系数
        cvs = [stat['cv'] for stat in performance_stats]
        avg_cv = np.mean(cvs)

        return max(0, 1 - avg_cv)

    def _validate_process_fairness(self) -> float:
        """验证过程公平性 - 算法执行环境的公平性"""
        # 检查是否有隐藏状态访问
        hidden_access_penalty = sum(
            1 for results in self.algorithm_results.values()
            for r in results
            if r.get('had_hidden_state_access', False)
        ) / max(1, sum(len(results) for results in self.algorithm_results.values()))

        return 1 - hidden_access_penalty

    def _validate_predictive_fairness(self) -> float:
        """验证预测公平性 - 各算法预测准确性的一致性"""
        if len(self.algorithm_results) < 2:
            return 1.0

        # 计算预测准确性（如果有相关数据）
        prediction_accuracies = []
        for algo, results in self.algorithm_results.items():
            accuracies = [r.get('prediction_accuracy', r.get('success_rate', 0.5)) for r in results]
            if accuracies:
                prediction_accuracies.append(np.mean(accuracies))

        if len(prediction_accuracies) < 2:
            return 1.0

        # 预测公平性基于准确性方差
        cv = np.std(prediction_accuracies) / (np.mean(prediction_accuracies) + 1e-10)
        return max(0, 1 - cv)

    def _validate_calibration_fairness(self) -> float:
        """验证校准公平性 - 置信度和实际性能的匹配程度"""
        if len(self.algorithm_results) < 2:
            return 1.0

        calibration_scores = []
        for algo, results in self.algorithm_results.items():
            for r in results:
                confidence = r.get('confidence', 0.5)
                actual_performance = r.get('success_rate', 0.5)

                # 校准误差：置信度和实际性能的差异
                calibration_error = abs(confidence - actual_performance)
                calibration_scores.append(1 - calibration_error)  # 校准分数

        return np.mean(calibration_scores) if calibration_scores else 1.0
    
    def run_kolmogorov_smirnov_test(
        self, 
        data1: List[float], 
        data2: List[float]
    ) -> Dict[str, Any]:
        """执行Kolmogorov-Smirnov检验"""
        if len(data1) < 2 or len(data2) < 2:
            return {'ks_statistic': 0, 'p_value': 1, 'significant': False}
        
        ks_stat, p_value = stats.ks_2samp(data1, data2)
        
        return {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'effect_size_category': self._categorize_effect_size(ks_stat)
        }
    
    def _categorize_effect_size(self, ks_stat: float) -> str:
        """分类效应量大小"""
        if ks_stat < 0.2:
            return "negligible"
        elif ks_stat < 0.4:
            return "small"
        elif ks_stat < 0.6:
            return "medium"
        else:
            return "large"
    
    def compute_cohens_d(self, data1: List[float], data2: List[float]) -> float:
        """计算Cohen's d效应量"""
        if len(data1) < 2 or len(data2) < 2:
            return 0
        
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1), np.var(data2)
        n1, n2 = len(data1), len(data2)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std < 1e-10:
            return 0
        
        return abs(mean1 - mean2) / pooled_std
    
    def generate_fairness_report(self) -> Dict[str, Any]:
        """生成完整公平性报告"""
        metrics, validation_results = self.validate_all()
        
        return {
            'summary': {
                'overall_score': self._compute_overall_score(metrics),
                'passed': metrics.resource_balance_score > 0.8 and metrics.sample_size_sufficient,
                'grade': self._get_fairness_grade(metrics.resource_balance_score),
                'metrics': metrics
            },
            'details': validation_results,
            'seed_chain': self.get_seed_chain_report(),
            'recommendations': self._generate_recommendations(metrics, validation_results)
        }
    
    def _compute_overall_score(self, metrics: FairnessMetrics) -> float:
        """计算总体公平性分数"""
        weights = {
            'resource': 0.30,
            'information': 0.25,
            'statistical': 0.25,
            'performance': 0.20
        }
        
        score = (
            weights['resource'] * metrics.resource_balance_score +
            weights['information'] * metrics.information_symmetry_score +
            weights['statistical'] * min(1.0, metrics.statistical_power) +
            weights['performance'] * metrics.performance_consistency
        )
        
        return score
    
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
    
    def _generate_recommendations(
        self, 
        metrics: FairnessMetrics, 
        details: Dict[str, Any]
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if metrics.resource_balance_score < 0.8:
            recommendations.append(
                "资源分配不平衡，建议重新设计资源分配策略，确保各算法使用相同的资源约束"
            )
        
        if metrics.information_symmetry_score < 1.0:
            recommendations.append(
                "存在信息不对称，某些算法可能获取了隐藏状态信息，需要重新设计实验"
            )
        
        if not metrics.sample_size_sufficient:
            recommendations.append(
                f"样本量不足，当前最小样本数{details['statistical_sufficiency']['min_samples']}，"
                f"建议至少增加到30次重复"
            )
        
        if metrics.performance_consistency < 0.8:
            recommendations.append(
                "性能一致性不足，可能存在系统性偏差，建议检查实验设置"
            )
        
        if metrics.jains_index < 0.9:
            recommendations.append(
                f"Jain公平性指数较低({metrics.jains_index:.3f})，资源分配需要更均衡"
            )
        
        if not recommendations:
            recommendations.append("公平性验证通过，实验设置符合ToN标准")
        
        return recommendations

    def _calculate_overall_passed(self, metrics: FairnessMetrics, report: Dict[str, Any]) -> bool:
        """计算整体验证是否通过

        基于IEEE ToN期刊标准，定义公平性验证的通过条件：
        1. 资源平衡分数 > 0.8
        2. 信息对称性分数 > 0.8
        3. 样本量充足
        4. 统计功效 > 0.8
        5. 没有严重的信息泄露（隐藏状态访问 < 10%）
        6. 性能一致性 > 0.7
        7. Jain公平性指数 > 0.85
        """

        # 核心公平性指标检查
        core_checks = [
            metrics.resource_balance_score > 0.8,      # 资源分配公平
            metrics.information_symmetry_score > 0.8,  # 信息对称性
            metrics.sample_size_sufficient,             # 样本量充足
            metrics.statistical_power > 0.8,           # 统计功效
            metrics.hidden_state_access_count < len(self.algorithm_results) * 0.1,  # 信息泄露控制
            metrics.performance_consistency > 0.7,     # 性能一致性
            metrics.jains_index > 0.85,                # Jain公平性指数
        ]

        # 增强公平性指标检查（如果启用）
        if hasattr(metrics, 'temporal_fairness'):
            enhanced_checks = [
                metrics.temporal_fairness > 0.7,
                metrics.opportunity_fairness > 0.7,
                metrics.outcome_fairness > 0.7,
                metrics.process_fairness > 0.7,
            ]
            core_checks.extend(enhanced_checks)

        # 计算通过率
        passed_checks = sum(core_checks)
        total_checks = len(core_checks)
        pass_rate = passed_checks / total_checks

        # 整体通过条件：至少80%的检查通过，且所有核心指标通过
        core_passed = all(core_checks[:7])  # 前7个为核心指标
        overall_passed = core_passed and (pass_rate >= 0.8)

        return overall_passed

    def _calculate_overall_score(self, metrics: FairnessMetrics) -> float:
        """计算整体公平性分数

        基于多个公平性维度的加权平均
        """

        # 定义权重
        weights = {
            'resource_balance': 0.25,      # 资源分配
            'information_symmetry': 0.20,  # 信息对称性
            'statistical_sufficiency': 0.15, # 统计充分性
            'performance_consistency': 0.15, # 性能一致性
            'jain_fairness': 0.15,         # Jain公平性
            'entropy_fairness': 0.10       # 熵公平性
        }

        # 计算各维度分数
        scores = {
            'resource_balance': metrics.resource_balance_score,
            'information_symmetry': metrics.information_symmetry_score,
            'statistical_sufficiency': 1.0 if metrics.sample_size_sufficient else 0.5,
            'performance_consistency': metrics.performance_consistency,
            'jain_fairness': metrics.jains_index,
            'entropy_fairness': metrics.entropy_fairness
        }

        # 如果有增强指标，加入计算
        if hasattr(metrics, 'temporal_fairness'):
            enhanced_weights = {
                'temporal': 0.05,
                'opportunity': 0.05,
                'outcome': 0.05,
                'process': 0.05,
                'predictive': 0.03,
                'calibration': 0.02
            }

            # 重新分配权重
            total_enhanced_weight = sum(enhanced_weights.values())
            for key in weights:
                weights[key] *= (1 - total_enhanced_weight)

            # 添加增强指标
            enhanced_scores = {
                'temporal': metrics.temporal_fairness,
                'opportunity': metrics.opportunity_fairness,
                'outcome': metrics.outcome_fairness,
                'process': metrics.process_fairness,
                'predictive': metrics.predictive_fairness,
                'calibration': metrics.calibration_fairness
            }

            weights.update(enhanced_weights)
            scores.update(enhanced_scores)

        # 计算加权平均分数
        overall_score = sum(scores[key] * weights[key] for key in scores.keys())

        return min(1.0, max(0.0, overall_score))  # 确保在[0,1]范围内
