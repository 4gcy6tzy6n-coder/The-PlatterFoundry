#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistical Analyzer - 统计分析器

创新点：
1. 完整的统计检验套件：t检验、KS检验、ANOVA
2. 置信区间计算：Bootstrap和理论方法
3. 多重比较校正：Bonferroni、FDR
4. 效应量和统计功效分析
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
import logging


@dataclass
class StatisticalTestResult:
    """统计检验结果"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_size_label: str
    confidence_interval: Tuple[float, float]
    interpretation: str


class StatisticalAnalyzer:
    """
    统计分析器
    
    提供完整的统计分析功能：
    1. 描述性统计
    2. 假设检验
    3. 效应量分析
    4. 多重比较校正
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.logger = logging.getLogger("StatisticalAnalyzer")
    
    def descriptive_statistics(self, data) -> Dict[str, float]:
        """计算描述性统计"""
        # 处理numpy数组
        if hasattr(data, '__len__') and hasattr(data, '__iter__'):
            # 转换为numpy数组
            arr = np.asarray(data)
            if arr.size == 0:
                return {}
        elif hasattr(data, '__len__'):
            if len(data) == 0:
                return {}
            arr = np.array(data)
        else:
            return {}
        
        arr = np.array(data)
        n = len(arr)
        
        # 基本统计
        mean = np.mean(arr)
        median = np.median(arr)
        std = np.std(arr, ddof=1)  # 样本标准差
        variance = np.var(arr, ddof=1)
        min_val = np.min(arr)
        max_val = np.max(arr)
        
        # 分位数
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        
        # 偏度和峰度
        skewness = stats.skew(arr) if n > 2 else 0
        kurtosis = stats.kurtosis(arr) if n > 3 else 0
        
        return {
            'n': n,
            'mean': mean,
            'median': median,
            'std': std,
            'variance': variance,
            'min': min_val,
            'max': max_val,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'range': max_val - min_val,
            'cv': std / (mean + 1e-10)  # 变异系数
        }
    
    def independent_samples_t_test(
        self, 
        group1: List[float], 
        group2: List[float]
    ) -> StatisticalTestResult:
        """
        独立样本t检验
        
        检验两组均值是否显著不同
        """
        if len(group1) < 2 or len(group2) < 2:
            return self._create_empty_result("Independent t-test")
        
        # 执行检验
        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        
        # 计算效应量 (Cohen's d)
        effect_size = self._cohens_d(group1, group2)
        effect_label = self._interpret_effect_size(effect_size)
        
        # 置信区间
        ci = self._compute_mean_difference_ci(group1, group2)
        
        # 解释
        interpretation = self._interpret_t_test(statistic, p_value, effect_size, len(group1), len(group2))
        
        return StatisticalTestResult(
            test_name="Independent t-test",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.significance_level,
            effect_size=effect_size,
            effect_size_label=effect_label,
            confidence_interval=ci,
            interpretation=interpretation
        )
    
    def paired_samples_t_test(
        self, 
        before: List[float], 
        after: List[float]
    ) -> StatisticalTestResult:
        """配对样本t检验"""
        if len(before) < 2 or len(before) != len(after):
            return self._create_empty_result("Paired t-test")
        
        statistic, p_value = stats.ttest_rel(before, after)
        effect_size = self._cohens_d(before, after)
        effect_label = self._interpret_effect_size(effect_size)
        
        diff = [a - b for a, b in zip(after, before)]
        ci = self._compute_ci_from_data(diff)
        
        interpretation = f"配对样本t检验: t={statistic:.3f}, p={p_value:.4f}"
        
        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.significance_level,
            effect_size=effect_size,
            effect_size_label=effect_label,
            confidence_interval=ci,
            interpretation=interpretation
        )
    
    def kolmogorov_smirnov_test(
        self, 
        data1: List[float], 
        data2: List[float]
    ) -> StatisticalTestResult:
        """Kolmogorov-Smirnov检验"""
        if len(data1) < 3 or len(data2) < 3:
            return self._create_empty_result("Kolmogorov-Smirnov test")
        
        statistic, p_value = stats.ks_2samp(data1, data2)
        
        # KS统计量本身就是一种效应量
        effect_size = statistic
        effect_label = self._interpret_ks_effect(statistic)
        
        interpretation = f"KS检验: D={statistic:.3f}, p={p_value:.4f}"
        
        return StatisticalTestResult(
            test_name="Kolmogorov-Smirnov test",
            statistic=statistic,
            p_value=p_value,
            significant=p_value < self.significance_level,
            effect_size=effect_size,
            effect_size_label=effect_label,
            confidence_interval=(0, 0),
            interpretation=interpretation
        )

    def kruskal_wallis_test(self, groups: List[List[float]]) -> Dict[str, Any]:
        """
        Kruskal-Wallis H检验 - 非参数单因素方差分析

        用于检验多个独立样本是否来自同一总体
        当数据不满足正态性假设时使用
        """
        if len(groups) < 2:
            return {'passed': False, 'error': '至少需要2组数据'}

        try:
            h_statistic, p_value = stats.kruskal(*groups)

            # 计算效应量 (eta-squared)
            n_total = sum(len(g) for g in groups)
            k = len(groups)

            # Kruskal-Wallis效应量
            eta_squared = (h_statistic - k + 1) / (n_total - k) if n_total > k else 0
            eta_squared = min(1.0, max(0.0, eta_squared))

            return {
                'test_name': 'Kruskal-Wallis H-test',
                'h_statistic': h_statistic,
                'p_value': p_value,
                'significant': p_value < self.significance_level,
                'eta_squared': eta_squared,
                'effect_size_label': self._interpret_eta_squared(eta_squared),
                'df': k - 1,
                'n_groups': k,
                'total_n': n_total
            }
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def friedman_test(self, groups: List[List[float]]) -> Dict[str, Any]:
        """
        Friedman检验 - 非参数重复测量方差分析

        用于检验相关样本的多个处理条件间是否存在差异
        """
        if len(groups) < 2:
            return {'passed': False, 'error': '至少需要2组数据'}

        # 检查每组数据长度是否相同
        group_lengths = [len(g) for g in groups]
        if len(set(group_lengths)) != 1:
            return {'passed': False, 'error': 'Friedman检验要求所有组有相同数量的观测值'}

        try:
            statistic, p_value = stats.friedmanchisquare(*groups)

            # Kendall's W 效应量
            k = len(groups)
            n = group_lengths[0]

            w = statistic / (k * (k - 1) * (n * k - k + 1) / 12) if k > 1 else 0
            w = min(1.0, max(0.0, w))

            return {
                'test_name': 'Friedman test',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < self.significance_level,
                'kendall_w': w,
                'effect_size_label': self._interpret_kendall_w(w),
                'df': k - 1,
                'n_groups': k,
                'n_subjects': n
            }
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _interpret_kendall_w(self, w: float) -> str:
        """解释Kendall's W效应量"""
        if w < 0.1:
            return "negligible"
        elif w < 0.3:
            return "small"
        elif w < 0.5:
            return "medium"
        else:
            return "large"
    
    def one_way_anova(
        self, 
        groups: List[List[float]]
    ) -> Dict[str, Any]:
        """
        单因素方差分析
        
        检验多个组的均值是否都相等
        """
        if len(groups) < 2:
            return {}
        
        # 执行ANOVA
        f_statistic, p_value = stats.f_oneway(*groups)
        
        # 计算效应量 (Eta-squared)
        all_data = [x for g in groups for x in g]
        grand_mean = np.mean(all_data)
        
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum((x - grand_mean)**2 for x in all_data)
        
        eta_squared = ss_between / (ss_total + 1e-10)
        
        # 计算组间自由度
        k = len(groups)
        df_between = k - 1
        df_within = len(all_data) - k
        
        return {
            'test_name': 'One-way ANOVA',
            'f_statistic': f_statistic,
            'p_value': p_value,
            'significant': p_value < self.significance_level,
            'eta_squared': eta_squared,
            'effect_size_label': self._interpret_eta_squared(eta_squared),
            'df_between': df_between,
            'df_within': df_within,
            'num_groups': k
        }
    
    def bootstrap_confidence_interval(
        self, 
        data: List[float], 
        statistic: str = 'mean',
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Bootstrap置信区间
        
        创新点：使用Bootstrap方法计算置信区间，不依赖分布假设
        """
        if len(data) < 2:
            return (data[0] if data else 0, data[0] if data else 0)
        
        np.random.seed(42)  # 可复现性
        
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            if statistic == 'mean':
                bootstrap_stats.append(np.mean(sample))
            elif statistic == 'median':
                bootstrap_stats.append(np.median(sample))
            elif statistic == 'std':
                bootstrap_stats.append(np.std(sample))
        
        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_stats, alpha * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)
        
        return (lower, upper)
    
    def bonferroni_correction(
        self, 
        p_values: List[float], 
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Bonferroni多重比较校正
        
        控制FWER (Family-Wise Error Rate)
        """
        n = len(p_values)
        corrected_alpha = alpha / n
        
        results = []
        for i, p in enumerate(p_values):
            results.append({
                'index': i,
                'original_p_value': p,
                'corrected_p_value': min(1.0, p * n),
                'significant': p < corrected_alpha
            })
        
        return {
            'correction_method': 'Bonferroni',
            'original_alpha': alpha,
            'corrected_alpha': corrected_alpha,
            'n_comparisons': n,
            'n_significant': sum(1 for r in results if r['significant']),
            'results': results
        }
    
    def false_discovery_rate_correction(
        self, 
        p_values: List[float], 
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        FDR (False Discovery Rate) 校正
        
        Benjamini-Hochberg程序
        控制期望FDR
        """
        n = len(p_values)
        if n == 0:
            return {'significant_count': 0}
        
        # 排序p值
        indexed_pvals = list(enumerate(p_values))
        sorted_pvals = sorted(indexed_pvals, key=lambda x: x[1])
        
        # BH校正
        bh_significant = []
        for i, (idx, p) in enumerate(sorted_pvals):
            adjusted_p = p * n / (i + 1)
            adjusted_p = min(1.0, adjusted_p)
            if adjusted_p < alpha:
                bh_significant.append({
                    'index': idx,
                    'original_p_value': p,
                    'adjusted_p_value': adjusted_p,
                    'significant': True
                })
        
        return {
            'correction_method': 'Benjamini-Hochberg FDR',
            'alpha': alpha,
            'n_comparisons': n,
            'n_significant': len(bh_significant),
            'significant_indices': [r['index'] for r in bh_significant]
        }
    
    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """计算Cohen's d效应量"""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std < 1e-10:
            return 0
        
        return abs(mean1 - mean2) / pooled_std
    
    def _interpret_effect_size(self, d: float) -> str:
        """解释Cohen's d效应量"""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_ks_effect(self, d: float) -> str:
        """解释KS统计量效应"""
        if d < 0.2:
            return "negligible"
        elif d < 0.4:
            return "small"
        elif d < 0.6:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta2: float) -> str:
        """解释Eta-squared效应量"""
        if eta2 < 0.01:
            return "negligible"
        elif eta2 < 0.06:
            return "small"
        elif eta2 < 0.14:
            return "medium"
        else:
            return "large"
    
    def _compute_mean_difference_ci(
        self, 
        group1: List[float], 
        group2: List[float]
    ) -> Tuple[float, float]:
        """计算均值差置信区间"""
        mean_diff = np.mean(group1) - np.mean(group2)
        se = np.sqrt(np.var(group1, ddof=1)/len(group1) + np.var(group2, ddof=1)/len(group2))
        
        df = len(group1) + len(group2) - 2
        t_critical = stats.t.ppf(0.975, df)
        
        margin = t_critical * se
        return (mean_diff - margin, mean_diff + margin)
    
    def _compute_ci_from_data(self, data: List[float]) -> Tuple[float, float]:
        """从数据计算置信区间"""
        mean = np.mean(data)
        se = np.std(data, ddof=1) / np.sqrt(len(data))
        
        df = len(data) - 1
        t_critical = stats.t.ppf(0.975, df)
        
        margin = t_critical * se
        return (mean - margin, mean + margin)
    
    def _interpret_t_test(
        self, 
        t_stat: float, 
        p_val: float, 
        effect: float, 
        n1: int, 
        n2: int
    ) -> str:
        """解释t检验结果"""
        significance = "显著" if p_val < self.significance_level else "不显著"
        effect_desc = self._interpret_effect_size(effect)
        
        return f"t检验结果: {significance} (p={p_val:.4f}), 效应量: {effect_desc} (d={effect:.3f})"
    
    def _create_empty_result(self, test_name: str) -> StatisticalTestResult:
        """创建空结果"""
        return StatisticalTestResult(
            test_name=test_name,
            statistic=0,
            p_value=1.0,
            significant=False,
            effect_size=0,
            effect_size_label="N/A",
            confidence_interval=(0, 0),
            interpretation="数据不足，无法执行检验"
        )

    def robust_descriptive_statistics(self, data: List[float]) -> Dict[str, float]:
        """
        稳健描述性统计 - 对异常值不敏感

        使用trimmed mean和MAD等稳健统计量
        """
        if not data:
            return {}

        arr = np.array(data)
        n = len(arr)

        # 传统统计量
        mean = np.mean(arr)
        median = np.median(arr)
        std = np.std(arr, ddof=1)

        # 稳健统计量
        # Trimmed mean (去除极端值)
        trim_percent = 0.1  # 去除10%的极端值
        trim_size = int(n * trim_percent)
        if trim_size > 0:
            sorted_data = np.sort(arr)
            trimmed_data = sorted_data[trim_size:-trim_size] if trim_size * 2 < n else sorted_data
            trimmed_mean = np.mean(trimmed_data)
        else:
            trimmed_mean = mean

        # MAD (Median Absolute Deviation)
        mad = np.median(np.abs(arr - median))

        # 标准化MAD (类似标准差)
        mad_std = mad * 1.4826  # 对于正态分布的标准化常数

        # 胜点估计 (Winsorized statistics)
        winsor_percent = 0.1
        winsor_size = int(n * winsor_percent)
        if winsor_size > 0:
            sorted_data = np.sort(arr)
            lower_bound = sorted_data[winsor_size]
            upper_bound = sorted_data[-(winsor_size + 1)]
            winsorized_data = np.clip(arr, lower_bound, upper_bound)
            winsorized_mean = np.mean(winsorized_data)
        else:
            winsorized_mean = mean

        # 异常值比例
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        outlier_bounds = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
        outlier_count = np.sum((arr < outlier_bounds[0]) | (arr > outlier_bounds[1]))
        outlier_ratio = outlier_count / n

        return {
            'n': n,
            'mean': mean,
            'median': median,
            'std': std,
            'trimmed_mean': trimmed_mean,
            'winsorized_mean': winsorized_mean,
            'mad': mad,
            'mad_std': mad_std,
            'outlier_ratio': outlier_ratio,
            'robust_cv': mad_std / (median + 1e-10),  # 稳健变异系数
            'skewness_robust': (trimmed_mean - median) / (mad_std + 1e-10)  # 稳健偏度
        }
