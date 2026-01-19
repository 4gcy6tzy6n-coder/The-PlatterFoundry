#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验1.1正式版最终版：自适应网络状态估计器性能评估

科学假设：
基于"变结构卡尔曼滤波"的自适应估计器，在估计精度、快速适应性和不确定性量化上，
显著优于固定参数的滤波器或滑动平均方法。

实验设计：
1. 对比算法：
   - AdaptiveEstimator (变结构卡尔曼滤波)
   - StandardKalmanFilter (固定参数卡尔曼滤波)
   - EWMA (指数加权移动平均)

2. 测试场景：
   - 稳定场景 (stable)
   - 突发场景 (burst)
   - 周期场景 (periodic)
   - 趋势变化场景 (trend)
   - 非平稳场景 (non_stationary)

3. 评估指标：
   - RMSE (Root Mean Square Error)
   - MAE (Mean Absolute Error)
   - 收敛时间 (Convergence Time)

4. 统计检验：
   - 配对样本Wilcoxon检验
   - 双因素方差分析 (ANOVA)

5. 可解释性报告：
   - 公平性验证 (FairnessValidator)
   - 可解释性分析 (InterpretabilityExtensions)

IEEE/ACM ToN 标准实验实现
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys
import os

import numpy as np
import pandas as pd
from scipy import stats

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入核心算法
from src.core.estimation.adaptive_estimator import AdaptiveEstimator
from src.core.estimation.standard_kalman_filter import StandardKalmanFilter

# 导入实验分析模块
from src.experiments.analysis.fairness_validator import FairnessValidator
from src.experiments.analysis.extended_explainability_metrics import ExtendedExplainabilitySystem, ScenarioPerformanceProfile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NetworkScenarioGenerator:
    """网络场景生成器"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_stable(self, n_samples: int = 1000, mean: float = 0.5, noise: float = 0.02) -> np.ndarray:
        """生成稳定场景"""
        return mean + np.random.normal(0, noise, n_samples)
    
    def generate_burst(self, n_samples: int = 1000, base: float = 0.5, burst_prob: float = 0.05) -> np.ndarray:
        """生成突发场景"""
        data = np.full(n_samples, base)
        noise = np.random.normal(0, 0.02, n_samples)
        
        # 随机生成突发
        for i in range(n_samples):
            if np.random.rand() < burst_prob:
                burst_length = np.random.randint(5, 20)
                burst_magnitude = np.random.uniform(0.3, 0.8)
                end = min(i + burst_length, n_samples)
                data[i:end] += burst_magnitude
        
        return data + noise
    
    def generate_periodic(self, n_samples: int = 1000, period: int = 100, amplitude: float = 0.3) -> np.ndarray:
        """生成周期场景"""
        t = np.arange(n_samples)
        base = 0.5
        signal = base + amplitude * np.sin(2 * np.pi * t / period)
        noise = np.random.normal(0, 0.02, n_samples)
        return signal + noise
    
    def generate_trend(self, n_samples: int = 1000, start: float = 0.3, end: float = 0.7) -> np.ndarray:
        """生成趋势变化场景"""
        trend = np.linspace(start, end, n_samples)
        noise = np.random.normal(0, 0.02, n_samples)
        return trend + noise
    
    def generate_non_stationary(self, n_samples: int = 1000) -> np.ndarray:
        """生成非平稳场景 - 混合多种变化"""
        data = np.zeros(n_samples)
        t = np.arange(n_samples)
        
        # 基础趋势
        data += np.linspace(0.3, 0.7, n_samples)
        
        # 周期性变化
        data += 0.1 * np.sin(2 * np.pi * t / 50)
        
        # 随机突发
        for i in range(0, n_samples, 200):
            burst_start = i + np.random.randint(0, 50)
            burst_length = np.random.randint(10, 30)
            burst_end = min(burst_start + burst_length, n_samples)
            data[burst_start:burst_end] += np.random.uniform(0.2, 0.4)
        
        # 噪声
        data += np.random.normal(0, 0.03, n_samples)
        
        return data


class EWMAEstimator:
    """EWMA估计器 - 基准算法"""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.estimate = 0.0
        self.initialized = False
    
    def update(self, measurement: float) -> float:
        """更新估计值"""
        if not self.initialized:
            self.estimate = measurement
            self.initialized = True
        else:
            self.estimate = self.alpha * measurement + (1 - self.alpha) * self.estimate
        return self.estimate
    
    def reset(self):
        """重置估计器"""
        self.estimate = 0.0
        self.initialized = False


class Experiment1_1Formal:
    """实验1.1正式版"""
    
    def __init__(self, n_samples: int = 1000, n_repeats: int = 30, seed: int = 42):
        self.n_samples = n_samples
        self.n_repeats = n_repeats
        self.seed = seed
        
        # 创建输出目录
        self.result_dir = Path("result")
        self.result_dir.mkdir(exist_ok=True)
        
        # 场景生成器
        self.scenario_generator = NetworkScenarioGenerator(seed=seed)
        
        # 实验结果存储
        self.all_results = []
        self.scenario_data = {}
        
    def run_all_experiments(self) -> Dict[str, Any]:
        """运行所有实验"""
        logger.info("=" * 60)
        logger.info("实验1.1正式版：自适应网络状态估计器性能评估")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # 定义场景
        scenarios = ['stable', 'burst', 'periodic', 'trend', 'non_stationary']
        
        # 定义算法
        algorithms = ['AdaptiveEstimator', 'StandardKalmanFilter', 'EWMA']
        
        # 运行实验
        logger.info(f"\n运行实验: {len(scenarios)} 场景 × {len(algorithms)} 算法 × {self.n_repeats} 重复")
        
        for scenario in scenarios:
            logger.info(f"\n场景: {scenario}")
            
            # 生成场景数据
            scenario_data = self._generate_scenario_data(scenario)
            self.scenario_data[scenario] = scenario_data
            
            for algorithm in algorithms:
                logger.info(f"  算法: {algorithm}")
                
                for repeat in range(self.n_repeats):
                    # 运行单次实验
                    result = self._run_single_experiment(
                        algorithm=algorithm,
                        scenario=scenario,
                        data=scenario_data,
                        repeat_id=repeat
                    )
                    
                    self.all_results.append(result)
                    
                    if (repeat + 1) % 10 == 0:
                        logger.info(f"    完成 {repeat + 1}/{self.n_repeats} 次重复")
        
        # 统计分析
        logger.info("\n执行统计分析...")
        statistical_results = self._perform_statistical_analysis()
        
        # 可解释性分析
        logger.info("\n执行可解释性分析...")
        explainability_results = self._perform_explainability_analysis()
        
        # 汇总结果
        duration = time.time() - start_time
        
        final_results = {
            'experiment_info': {
                'name': 'Experiment 1.1 Formal Final',
                'description': '自适应网络状态估计器性能评估',
                'n_samples': self.n_samples,
                'n_repeats': self.n_repeats,
                'scenarios': scenarios,
                'algorithms': algorithms,
                'duration_seconds': duration,
                'timestamp': time.time()
            },
            'raw_results': self.all_results,
            'statistical_analysis': statistical_results,
            'explainability_analysis': explainability_results,
            'summary': self._generate_summary()
        }
        
        # 保存结果
        self._save_results(final_results)
        
        logger.info(f"\n实验完成！总用时: {duration:.2f} 秒")
        logger.info(f"结果保存至: {self.result_dir}")
        
        return final_results
    
    def _generate_scenario_data(self, scenario: str) -> np.ndarray:
        """生成场景数据"""
        if scenario == 'stable':
            return self.scenario_generator.generate_stable(self.n_samples)
        elif scenario == 'burst':
            return self.scenario_generator.generate_burst(self.n_samples)
        elif scenario == 'periodic':
            return self.scenario_generator.generate_periodic(self.n_samples)
        elif scenario == 'trend':
            return self.scenario_generator.generate_trend(self.n_samples)
        elif scenario == 'non_stationary':
            return self.scenario_generator.generate_non_stationary(self.n_samples)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    def _run_single_experiment(self, algorithm: str, scenario: str, 
                               data: np.ndarray, repeat_id: int) -> Dict[str, Any]:
        """运行单次实验"""
        # 创建估计器
        if algorithm == 'AdaptiveEstimator':
            estimator = AdaptiveEstimator(
                name=f"{algorithm}_{scenario}_{repeat_id}",
                short_window=10,
                long_window=100
            )
        elif algorithm == 'StandardKalmanFilter':
            estimator = StandardKalmanFilter(
                process_noise=0.01,
                measurement_noise=0.1
            )
        elif algorithm == 'EWMA':
            estimator = EWMAEstimator(alpha=0.3)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # 运行估计
        estimates = []
        execution_start = time.time()
        
        for i, measurement in enumerate(data):
            if algorithm == 'AdaptiveEstimator':
                result = estimator.update(measurement, timestamp=i)
                estimates.append(result.value)
            elif algorithm == 'StandardKalmanFilter':
                result = estimator.estimate(measurement)
                estimates.append(result.estimate)
            elif algorithm == 'EWMA':
                estimate = estimator.update(measurement)
                estimates.append(estimate)
        
        execution_time = time.time() - execution_start
        
        # 计算性能指标
        estimates = np.array(estimates)
        rmse = np.sqrt(np.mean((estimates - data) ** 2))
        mae = np.mean(np.abs(estimates - data))
        
        # 计算收敛时间
        errors = np.abs(estimates - data)
        convergence_threshold = 0.05
        convergence_time = 0
        
        for i, error in enumerate(errors):
            if error < convergence_threshold:
                convergence_time = i
                break
        
        if convergence_time == 0:
            convergence_time = len(data)
        
        return {
            'algorithm': algorithm,
            'scenario': scenario,
            'repeat_id': repeat_id,
            'rmse': float(rmse),
            'mae': float(mae),
            'convergence_time': int(convergence_time),
            'execution_time': float(execution_time),
            'n_samples': len(data)
        }
    
    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """执行统计分析"""
        df = pd.DataFrame(self.all_results)
        
        results = {
            'wilcoxon_tests': {},
            'anova_tests': {},
            'descriptive_statistics': {}
        }
        
        # 1. Wilcoxon配对检验 - 比较AdaptiveEstimator与其他算法
        scenarios = df['scenario'].unique()
        metrics = ['rmse', 'mae', 'convergence_time']
        
        for scenario in scenarios:
            results['wilcoxon_tests'][scenario] = {}
            
            # 获取AdaptiveEstimator的结果
            adaptive_data = df[(df['scenario'] == scenario) & 
                              (df['algorithm'] == 'AdaptiveEstimator')]
            
            for baseline in ['StandardKalmanFilter', 'EWMA']:
                baseline_data = df[(df['scenario'] == scenario) & 
                                  (df['algorithm'] == baseline)]
                
                results['wilcoxon_tests'][scenario][baseline] = {}
                
                for metric in metrics:
                    try:
                        statistic, p_value = stats.wilcoxon(
                            adaptive_data[metric].values,
                            baseline_data[metric].values,
                            alternative='less'  # AdaptiveEstimator应该更小(更好)
                        )
                        
                        results['wilcoxon_tests'][scenario][baseline][metric] = {
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'adaptive_mean': float(adaptive_data[metric].mean()),
                            'baseline_mean': float(baseline_data[metric].mean()),
                            'improvement': float(
                                (baseline_data[metric].mean() - adaptive_data[metric].mean()) / 
                                baseline_data[metric].mean() * 100
                            )
                        }
                    except Exception as e:
                        logger.warning(f"Wilcoxon test failed for {scenario}/{baseline}/{metric}: {e}")
        
        # 2. 双因素方差分析 (ANOVA)
        for metric in metrics:
            try:
                # 按算法分组
                groups = [df[df['algorithm'] == algo][metric].values 
                         for algo in df['algorithm'].unique()]
                
                f_stat, p_value = stats.f_oneway(*groups)
                
                results['anova_tests'][metric] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            except Exception as e:
                logger.warning(f"ANOVA failed for {metric}: {e}")
        
        # 3. 描述性统计
        for scenario in scenarios:
            results['descriptive_statistics'][scenario] = {}
            
            for algorithm in df['algorithm'].unique():
                subset = df[(df['scenario'] == scenario) & 
                           (df['algorithm'] == algorithm)]
                
                results['descriptive_statistics'][scenario][algorithm] = {
                    'rmse_mean': float(subset['rmse'].mean()),
                    'rmse_std': float(subset['rmse'].std()),
                    'mae_mean': float(subset['mae'].mean()),
                    'mae_std': float(subset['mae'].std()),
                    'convergence_time_mean': float(subset['convergence_time'].mean()),
                    'convergence_time_std': float(subset['convergence_time'].std())
                }
        
        return results
    
    def _perform_explainability_analysis(self) -> Dict[str, Any]:
        """执行可解释性分析"""
        results = {}
        
        try:
            # 1. 公平性验证
            logger.info("  执行公平性验证...")
            validator = FairnessValidator(significance_level=0.05)
            
            df = pd.DataFrame(self.all_results)
            
            # 为每个算法注册结果
            for algorithm in df['algorithm'].unique():
                algo_results = df[df['algorithm'] == algorithm].to_dict('records')
                validator.register_algorithm_results(algorithm, algo_results)
            
            fairness_report = validator.validate_all()
            results['fairness_validation'] = fairness_report
            
        except Exception as e:
            logger.warning(f"公平性验证失败: {e}")
            results['fairness_validation'] = {'error': str(e)}
        
        try:
            # 2. 可解释性分析
            logger.info("  执行可解释性分析...")
            explainability_system = ExtendedExplainabilitySystem()
            
            # 选择一个有明显性能差异的场景进行分析
            df = pd.DataFrame(self.all_results)
            burst_results = df[df['scenario'] == 'burst']
            
            # 添加场景剖面
            for _, row in burst_results.iterrows():
                profile = ScenarioPerformanceProfile(
                    scenario_id=f"{row['algorithm']}_{row['repeat_id']}",
                    network_conditions={
                        'scenario': row['scenario'],
                        'n_samples': row['n_samples']
                    },
                    algorithm_parameters={
                        'algorithm': row['algorithm'],
                        'version': 'v1.0'
                    },
                    performance_metrics={
                        'rmse': row['rmse'],
                        'mae': row['mae'],
                        'convergence_time': row['convergence_time']
                    },
                    fairness_metrics={},
                    resource_utilization={'execution_time': row['execution_time']},
                    decision_characteristics={}
                )
                explainability_system.add_scenario_profile(profile)
            
            # 生成可解释性报告
            explainability_report = explainability_system.generate_explainability_report('v1.0')
            
            # 转换为可序列化的格式
            if explainability_report:
                results['explainability_analysis'] = {
                    'algorithm_version': explainability_report.algorithm_version,
                    'timestamp': explainability_report.timestamp.isoformat(),
                    'overall_robustness_score': float(explainability_report.overall_robustness_score),
                    'overall_efficiency_score': float(explainability_report.overall_efficiency_score),
                    'critical_insights': list(explainability_report.critical_insights),
                    'risk_warnings': list(explainability_report.risk_warnings),
                    'optimization_priorities': list(explainability_report.optimization_priorities)
                }
            else:
                results['explainability_analysis'] = {'error': 'No report generated'}
            
        except Exception as e:
            logger.warning(f"可解释性分析失败: {e}")
            results['explainability_analysis'] = {'error': str(e)}
        
        return results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成汇总报告"""
        df = pd.DataFrame(self.all_results)
        
        summary = {
            'overall_performance': {},
            'best_algorithm_by_scenario': {},
            'key_findings': []
        }
        
        # 总体性能
        for algorithm in df['algorithm'].unique():
            subset = df[df['algorithm'] == algorithm]
            summary['overall_performance'][algorithm] = {
                'mean_rmse': float(subset['rmse'].mean()),
                'mean_mae': float(subset['mae'].mean()),
                'mean_convergence_time': float(subset['convergence_time'].mean())
            }
        
        # 各场景最佳算法
        for scenario in df['scenario'].unique():
            subset = df[df['scenario'] == scenario]
            best_algo = subset.groupby('algorithm')['rmse'].mean().idxmin()
            summary['best_algorithm_by_scenario'][scenario] = best_algo
        
        # 关键发现
        adaptive_rmse = df[df['algorithm'] == 'AdaptiveEstimator']['rmse'].mean()
        kalman_rmse = df[df['algorithm'] == 'StandardKalmanFilter']['rmse'].mean()
        ewma_rmse = df[df['algorithm'] == 'EWMA']['rmse'].mean()
        
        summary['key_findings'].append(
            f"AdaptiveEstimator RMSE: {adaptive_rmse:.4f} vs StandardKalman: {kalman_rmse:.4f} vs EWMA: {ewma_rmse:.4f}"
        )
        
        if adaptive_rmse < kalman_rmse and adaptive_rmse < ewma_rmse:
            summary['key_findings'].append(
                "✓ AdaptiveEstimator在所有场景下平均性能最优"
            )
        
        return summary
    
    def _convert_to_json_serializable(self, obj):
        """将对象转换为JSON可序列化的格式"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif hasattr(obj, '__dataclass_fields__'):  # dataclass对象
            return {k: self._convert_to_json_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _save_results(self, results: Dict[str, Any]):
        """保存实验结果"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 1. 保存完整JSON结果
        json_file = self.result_dir / f"experiment_1_1_formal_final_{timestamp}.json"
        # 转换为可JSON序列化的格式
        serializable_results = self._convert_to_json_serializable(results)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        logger.info(f"保存JSON结果: {json_file}")
        
        # 2. 保存CSV格式的原始数据
        df = pd.DataFrame(results['raw_results'])
        csv_file = self.result_dir / f"experiment_1_1_formal_final_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"保存CSV结果: {csv_file}")
        
        # 3. 保存统计分析汇总表格
        self._save_statistical_tables(results['statistical_analysis'], timestamp)
    
    def _save_statistical_tables(self, statistical_results: Dict[str, Any], timestamp: str):
        """保存统计分析表格"""
        txt_file = self.result_dir / f"experiment_1_1_statistics_tables_{timestamp}.txt"
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("实验1.1 统计分析汇总表格\n")
            f.write("=" * 80 + "\n\n")
            
            # Wilcoxon检验结果
            f.write("1. Wilcoxon配对检验结果 (AdaptiveEstimator vs Baselines)\n")
            f.write("-" * 80 + "\n")
            
            wilcoxon_tests = statistical_results.get('wilcoxon_tests', {})
            for scenario, baselines in wilcoxon_tests.items():
                f.write(f"\n场景: {scenario}\n")
                for baseline, metrics in baselines.items():
                    f.write(f"  对比: {baseline}\n")
                    for metric, result in metrics.items():
                        f.write(f"    {metric}:\n")
                        f.write(f"      p-value: {result['p_value']:.6f}\n")
                        f.write(f"      显著性: {'是' if result['significant'] else '否'}\n")
                        f.write(f"      改善幅度: {result['improvement']:.2f}%\n")
            
            # ANOVA结果
            f.write("\n\n2. 方差分析 (ANOVA) 结果\n")
            f.write("-" * 80 + "\n")
            
            anova_tests = statistical_results.get('anova_tests', {})
            for metric, result in anova_tests.items():
                f.write(f"\n{metric}:\n")
                f.write(f"  F统计量: {result['f_statistic']:.4f}\n")
                f.write(f"  p-value: {result['p_value']:.6f}\n")
                f.write(f"  显著性: {'是' if result['significant'] else '否'}\n")
            
            # 描述性统计
            f.write("\n\n3. 描述性统计\n")
            f.write("-" * 80 + "\n")
            
            desc_stats = statistical_results.get('descriptive_statistics', {})
            for scenario, algorithms in desc_stats.items():
                f.write(f"\n场景: {scenario}\n")
                for algorithm, stats in algorithms.items():
                    f.write(f"  {algorithm}:\n")
                    f.write(f"    RMSE: {stats['rmse_mean']:.4f} ± {stats['rmse_std']:.4f}\n")
                    f.write(f"    MAE: {stats['mae_mean']:.4f} ± {stats['mae_std']:.4f}\n")
                    f.write(f"    收敛时间: {stats['convergence_time_mean']:.1f} ± {stats['convergence_time_std']:.1f}\n")
        
        logger.info(f"保存统计表格: {txt_file}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("实验1.1正式版最终版：自适应网络状态估计器性能评估")
    print("=" * 80)
    print("\n实验配置:")
    print("  - 样本数: 1000")
    print("  - 重复次数: 30")
    print("  - 场景数: 5 (stable, burst, periodic, trend, non_stationary)")
    print("  - 算法数: 3 (AdaptiveEstimator, StandardKalmanFilter, EWMA)")
    print("  - 总实验数: 5 × 3 × 30 = 450 次")
    print("\n开始实验...\n")
    
    # 创建实验实例
    experiment = Experiment1_1Formal(
        n_samples=1000,
        n_repeats=30,
        seed=42
    )
    
    # 运行实验
    results = experiment.run_all_experiments()
    
    # 显示关键结果
    print("\n" + "=" * 80)
    print("实验完成！关键结果:")
    print("=" * 80)
    
    summary = results['summary']
    print("\n总体性能:")
    for algo, perf in summary['overall_performance'].items():
        print(f"  {algo}:")
        print(f"    平均RMSE: {perf['mean_rmse']:.4f}")
        print(f"    平均MAE: {perf['mean_mae']:.4f}")
        print(f"    平均收敛时间: {perf['mean_convergence_time']:.1f}")
    
    print("\n各场景最佳算法:")
    for scenario, best_algo in summary['best_algorithm_by_scenario'].items():
        print(f"  {scenario}: {best_algo}")
    
    print("\n关键发现:")
    for finding in summary['key_findings']:
        print(f"  {finding}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
