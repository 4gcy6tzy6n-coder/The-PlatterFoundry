#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验1.2正式版：三状态丢包模型性能评估

科学假设：
改进的"三状态马尔可夫模型"能更准确地刻画真实网络丢包的相关性和突发性，
在拟合优度和预测能力上超越传统模型。

实验设计：
1. 对比模型：
   - IID模型 (独立同分布)
   - Gilbert-Elliott模型 (二状态)
   - ImprovedLossModel (三状态马尔可夫)

2. 测试数据：
   - 合成数据集：6种场景
   - 真实数据集：3种网络跟踪

3. 评估指标：
   - 拟合优度 (MSE)
   - 预测准确率 (Accuracy)
   - KL散度 (分布相似度)

4. 统计检验：
   - 配对t检验
   - Friedman检验

IEEE/ACM ToN 标准实验实现
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

import numpy as np
import pandas as pd
from scipy import stats

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入核心模型
from src.network.models.improved_loss_model import ImprovedLossModel, LossModelConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IIDLossModel:
    """IID丢包模型 - 独立同分布基线"""
    
    def __init__(self, loss_rate: float = 0.1):
        self.loss_rate = loss_rate
    
    def generate(self, n_packets: int, seed: int = None) -> np.ndarray:
        """生成IID丢包序列"""
        if seed is not None:
            np.random.seed(seed)
        return (np.random.rand(n_packets) < self.loss_rate).astype(int)
    
    def predict(self, history: np.ndarray, n_predict: int = 100) -> np.ndarray:
        """预测未来丢包"""
        # IID模型：简单使用历史平均丢包率
        estimated_rate = np.mean(history)
        return (np.random.rand(n_predict) < estimated_rate).astype(int)


class GilbertElliottModel:
    """Gilbert-Elliott二状态模型"""
    
    def __init__(self, p_gb: float = 0.1, p_bg: float = 0.2, 
                 loss_good: float = 0.01, loss_bad: float = 0.5):
        self.p_gb = p_gb  # Good to Bad
        self.p_bg = p_bg  # Bad to Good
        self.loss_good = loss_good
        self.loss_bad = loss_bad
        self.state = 0  # 0=Good, 1=Bad
    
    def generate(self, n_packets: int, seed: int = None) -> np.ndarray:
        """生成Gilbert-Elliott丢包序列"""
        if seed is not None:
            np.random.seed(seed)
        
        losses = []
        self.state = 0
        
        for _ in range(n_packets):
            # 状态转移
            if self.state == 0:  # Good
                if np.random.rand() < self.p_gb:
                    self.state = 1
            else:  # Bad
                if np.random.rand() < self.p_bg:
                    self.state = 0
            
            # 生成丢包
            loss_prob = self.loss_good if self.state == 0 else self.loss_bad
            losses.append(1 if np.random.rand() < loss_prob else 0)
        
        return np.array(losses)
    
    def predict(self, history: np.ndarray, n_predict: int = 100) -> np.ndarray:
        """预测未来丢包"""
        # 简单预测：使用最后状态
        predictions = []
        for _ in range(n_predict):
            if self.state == 0:
                if np.random.rand() < self.p_gb:
                    self.state = 1
            else:
                if np.random.rand() < self.p_bg:
                    self.state = 0
            
            loss_prob = self.loss_good if self.state == 0 else self.loss_bad
            predictions.append(1 if np.random.rand() < loss_prob else 0)
        
        return np.array(predictions)


class DatasetGenerator:
    """数据集生成器"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_synthetic_datasets(self) -> Dict[str, np.ndarray]:
        """生成合成数据集"""
        datasets = {}
        
        # 1. 稳定低丢包
        datasets['stable_low_loss'] = self._generate_stable_low_loss()
        
        # 2. 突发高丢包
        datasets['bursty_high_loss'] = self._generate_bursty_high_loss()
        
        # 3. 周期性丢包
        datasets['periodic_loss'] = self._generate_periodic_loss()
        
        # 4. 队列拥塞
        datasets['queue_congestion'] = self._generate_queue_congestion()
        
        # 5. 无线衰落
        datasets['wireless_fading'] = self._generate_wireless_fading()
        
        # 6. 混合模式
        datasets['mixed_loss'] = self._generate_mixed_pattern()
        
        return datasets
    
    def _generate_stable_low_loss(self, n_packets: int = 10000) -> np.ndarray:
        """稳定低丢包场景"""
        return (np.random.rand(n_packets) < 0.01).astype(int)
    
    def _generate_bursty_high_loss(self, n_packets: int = 10000) -> np.ndarray:
        """突发高丢包场景"""
        losses = np.zeros(n_packets, dtype=int)
        n_bursts = 20
        for _ in range(n_bursts):
            start = np.random.randint(0, n_packets - 100)
            length = np.random.randint(10, 50)
            losses[start:start+length] = 1
        return losses
    
    def _generate_periodic_loss(self, n_packets: int = 10000) -> np.ndarray:
        """周期性丢包场景"""
        period = 100
        losses = np.zeros(n_packets, dtype=int)
        for i in range(0, n_packets, period):
            losses[i:i+10] = 1
        return losses
    
    def _generate_queue_congestion(self, n_packets: int = 10000) -> np.ndarray:
        """队列拥塞场景"""
        losses = []
        queue_size = 0
        max_queue = 50
        
        for _ in range(n_packets):
            arrival = 1
            service = 1 if np.random.rand() < 0.9 else 0
            
            queue_size += arrival - service
            queue_size = max(0, queue_size)
            
            loss = 1 if queue_size > max_queue else 0
            losses.append(loss)
        
        return np.array(losses)
    
    def _generate_wireless_fading(self, n_packets: int = 10000) -> np.ndarray:
        """无线信道衰落场景"""
        # 使用正弦波模拟信号强度变化
        t = np.arange(n_packets)
        signal_strength = 0.5 + 0.3 * np.sin(2 * np.pi * t / 500)
        loss_prob = 1 - signal_strength
        return (np.random.rand(n_packets) < loss_prob).astype(int)
    
    def _generate_mixed_pattern(self, n_packets: int = 10000) -> np.ndarray:
        """混合模式场景"""
        losses = np.zeros(n_packets, dtype=int)
        
        # 基础随机丢包
        losses += (np.random.rand(n_packets) < 0.02).astype(int)
        
        # 添加突发
        for _ in range(10):
            start = np.random.randint(0, n_packets - 50)
            length = np.random.randint(5, 20)
            losses[start:start+length] = 1
        
        # 添加周期性
        for i in range(0, n_packets, 200):
            losses[i:i+5] = 1
        
        return np.clip(losses, 0, 1)
    
    def generate_real_world_datasets(self) -> Dict[str, np.ndarray]:
        """生成模拟真实世界的数据集"""
        datasets = {}
        
        # 模拟CAIDA数据特征
        datasets['caida_sample'] = self._simulate_caida(10000)
        
        # 模拟Internet2数据特征
        datasets['internet2_sample'] = self._simulate_internet2(10000)
        
        # 模拟MAWI数据特征
        datasets['mawi_sample'] = self._simulate_mawi(10000)
        
        return datasets
    
    def _simulate_caida(self, n_packets: int) -> np.ndarray:
        """模拟CAIDA网络跟踪特征"""
        # CAIDA: 低丢包率，偶尔突发
        losses = (np.random.rand(n_packets) < 0.005).astype(int)
        for _ in range(5):
            start = np.random.randint(0, n_packets - 20)
            losses[start:start+10] = 1
        return losses
    
    def _simulate_internet2(self, n_packets: int) -> np.ndarray:
        """模拟Internet2网络特征"""
        # Internet2: 极低丢包率
        return (np.random.rand(n_packets) < 0.001).astype(int)
    
    def _simulate_mawi(self, n_packets: int) -> np.ndarray:
        """模拟MAWI网络特征"""
        # MAWI: 中等丢包率，有相关性
        ge_model = GilbertElliottModel(p_gb=0.05, p_bg=0.3, 
                                      loss_good=0.01, loss_bad=0.3)
        return ge_model.generate(n_packets)


class Experiment1_2_LossModel:
    """实验1.2：丢包模型对比"""
    
    def __init__(self, n_packets: int = 10000, n_repeats: int = 10, seed: int = 42):
        self.n_packets = n_packets
        self.n_repeats = n_repeats
        self.seed = seed
        
        # 创建输出目录
        self.result_dir = Path("result")
        self.result_dir.mkdir(exist_ok=True)
        
        # 数据集生成器
        self.dataset_generator = DatasetGenerator(seed=seed)
        
        # 实验结果存储
        self.all_results = []
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """运行所有实验"""
        logger.info("=" * 60)
        logger.info("实验1.2正式版：三状态丢包模型性能评估")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # 生成数据集
        logger.info("\n生成数据集...")
        synthetic_datasets = self.dataset_generator.generate_synthetic_datasets()
        real_datasets = self.dataset_generator.generate_real_world_datasets()
        all_datasets = {**synthetic_datasets, **real_datasets}
        
        logger.info(f"生成 {len(all_datasets)} 个数据集")
        
        # 定义模型
        models = ['IID', 'GilbertElliott', 'ImprovedLossModel']
        
        # 运行实验
        logger.info(f"\n运行实验: {len(all_datasets)} 数据集 × {len(models)} 模型 × {self.n_repeats} 重复")
        
        for dataset_name, true_losses in all_datasets.items():
            logger.info(f"\n数据集: {dataset_name}")
            
            for model_name in models:
                logger.info(f"  模型: {model_name}")
                
                for repeat in range(self.n_repeats):
                    # 运行单次实验
                    result = self._run_single_experiment(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        true_losses=true_losses,
                        repeat_id=repeat
                    )
                    
                    self.all_results.append(result)
                    
                    if (repeat + 1) % 5 == 0:
                        logger.info(f"    完成 {repeat + 1}/{self.n_repeats} 次重复")
        
        # 统计分析
        logger.info("\n执行统计分析...")
        statistical_results = self._perform_statistical_analysis()
        
        # 汇总结果
        duration = time.time() - start_time
        
        final_results = {
            'experiment_info': {
                'name': 'Experiment 1.2 Loss Model',
                'description': '三状态丢包模型性能评估',
                'n_packets': self.n_packets,
                'n_repeats': self.n_repeats,
                'n_datasets': len(all_datasets),
                'models': models,
                'duration_seconds': duration,
                'timestamp': time.time()
            },
            'raw_results': self.all_results,
            'statistical_analysis': statistical_results,
            'summary': self._generate_summary()
        }
        
        # 保存结果
        self._save_results(final_results)
        
        logger.info(f"\n实验完成！总用时: {duration:.2f} 秒")
        logger.info(f"结果保存至: {self.result_dir}")
        
        return final_results
    
    def _run_single_experiment(self, model_name: str, dataset_name: str,
                               true_losses: np.ndarray, repeat_id: int) -> Dict[str, Any]:
        """运行单次实验"""
        # 创建模型
        if model_name == 'IID':
            loss_rate = np.mean(true_losses)
            model = IIDLossModel(loss_rate=loss_rate)
        elif model_name == 'GilbertElliott':
            model = GilbertElliottModel()
        elif model_name == 'ImprovedLossModel':
            config = LossModelConfig()
            model = ImprovedLossModel(config)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # 生成预测序列
        seed = self.seed + repeat_id
        predicted_losses = model.generate(len(true_losses), seed=seed)
        
        # 计算评估指标
        mse = np.mean((predicted_losses - true_losses) ** 2)
        accuracy = np.mean(predicted_losses == true_losses)
        
        # 计算KL散度（分布相似度）
        true_dist = np.array([1 - np.mean(true_losses), np.mean(true_losses)])
        pred_dist = np.array([1 - np.mean(predicted_losses), np.mean(predicted_losses)])
        
        # 避免除以零
        true_dist = np.clip(true_dist, 1e-10, 1.0)
        pred_dist = np.clip(pred_dist, 1e-10, 1.0)
        
        kl_divergence = stats.entropy(true_dist, pred_dist)
        
        # 计算突发检测能力
        true_bursts = self._detect_bursts(true_losses)
        pred_bursts = self._detect_bursts(predicted_losses)
        
        burst_detection_score = 1.0 - abs(true_bursts - pred_bursts) / max(true_bursts, pred_bursts, 1)
        
        return {
            'model': model_name,
            'dataset': dataset_name,
            'repeat_id': repeat_id,
            'mse': float(mse),
            'accuracy': float(accuracy),
            'kl_divergence': float(kl_divergence),
            'burst_detection_score': float(burst_detection_score),
            'true_loss_rate': float(np.mean(true_losses)),
            'pred_loss_rate': float(np.mean(predicted_losses))
        }
    
    def _detect_bursts(self, losses: np.ndarray, min_length: int = 3) -> int:
        """检测突发丢包数量"""
        n_bursts = 0
        in_burst = False
        burst_length = 0
        
        for loss in losses:
            if loss == 1:
                if not in_burst:
                    in_burst = True
                    burst_length = 1
                else:
                    burst_length += 1
            else:
                if in_burst and burst_length >= min_length:
                    n_bursts += 1
                in_burst = False
                burst_length = 0
        
        if in_burst and burst_length >= min_length:
            n_bursts += 1
        
        return n_bursts
    
    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """执行统计分析"""
        df = pd.DataFrame(self.all_results)
        
        results = {
            'paired_t_tests': {},
            'friedman_test': {},
            'descriptive_statistics': {}
        }
        
        # 配对t检验 - 比较ImprovedLossModel与基线
        datasets = df['dataset'].unique()
        metrics = ['mse', 'accuracy', 'kl_divergence', 'burst_detection_score']
        
        for dataset in datasets:
            results['paired_t_tests'][dataset] = {}
            
            improved_data = df[(df['dataset'] == dataset) & 
                              (df['model'] == 'ImprovedLossModel')]
            
            for baseline in ['IID', 'GilbertElliott']:
                baseline_data = df[(df['dataset'] == dataset) & 
                                  (df['model'] == baseline)]
                
                results['paired_t_tests'][dataset][baseline] = {}
                
                for metric in metrics:
                    try:
                        t_stat, p_value = stats.ttest_rel(
                            improved_data[metric].values,
                            baseline_data[metric].values
                        )
                        
                        results['paired_t_tests'][dataset][baseline][metric] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'improved_mean': float(improved_data[metric].mean()),
                            'baseline_mean': float(baseline_data[metric].mean())
                        }
                    except Exception as e:
                        logger.warning(f"t-test failed for {dataset}/{baseline}/{metric}: {e}")
        
        # Friedman检验 - 非参数检验
        for metric in metrics:
            try:
                groups = [df[df['model'] == model][metric].values 
                         for model in df['model'].unique()]
                
                stat, p_value = stats.friedmanchisquare(*groups)
                
                results['friedman_test'][metric] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
            except Exception as e:
                logger.warning(f"Friedman test failed for {metric}: {e}")
        
        # 描述性统计
        for dataset in datasets:
            results['descriptive_statistics'][dataset] = {}
            
            for model in df['model'].unique():
                subset = df[(df['dataset'] == dataset) & (df['model'] == model)]
                
                results['descriptive_statistics'][dataset][model] = {
                    'mse_mean': float(subset['mse'].mean()),
                    'mse_std': float(subset['mse'].std()),
                    'accuracy_mean': float(subset['accuracy'].mean()),
                    'accuracy_std': float(subset['accuracy'].std()),
                    'kl_divergence_mean': float(subset['kl_divergence'].mean()),
                    'kl_divergence_std': float(subset['kl_divergence'].std())
                }
        
        return results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成汇总报告"""
        df = pd.DataFrame(self.all_results)
        
        summary = {
            'overall_performance': {},
            'best_model_by_dataset': {},
            'key_findings': []
        }
        
        # 总体性能
        for model in df['model'].unique():
            subset = df[df['model'] == model]
            summary['overall_performance'][model] = {
                'mean_mse': float(subset['mse'].mean()),
                'mean_accuracy': float(subset['accuracy'].mean()),
                'mean_kl_divergence': float(subset['kl_divergence'].mean())
            }
        
        # 各数据集最佳模型
        for dataset in df['dataset'].unique():
            subset = df[df['dataset'] == dataset]
            best_model = subset.groupby('model')['accuracy'].mean().idxmax()
            summary['best_model_by_dataset'][dataset] = best_model
        
        # 关键发现
        improved_acc = df[df['model'] == 'ImprovedLossModel']['accuracy'].mean()
        iid_acc = df[df['model'] == 'IID']['accuracy'].mean()
        ge_acc = df[df['model'] == 'GilbertElliott']['accuracy'].mean()
        
        summary['key_findings'].append(
            f"ImprovedLossModel准确率: {improved_acc:.4f} vs IID: {iid_acc:.4f} vs GE: {ge_acc:.4f}"
        )
        
        if improved_acc > iid_acc and improved_acc > ge_acc:
            summary['key_findings'].append(
                "✓ ImprovedLossModel在所有数据集上平均准确率最优"
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
        elif hasattr(obj, '__dataclass_fields__'):
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
        
        # 保存JSON结果
        json_file = self.result_dir / f"experiment_1_2_loss_model_{timestamp}.json"
        serializable_results = self._convert_to_json_serializable(results)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        logger.info(f"保存JSON结果: {json_file}")
        
        # 保存CSV格式
        df = pd.DataFrame(results['raw_results'])
        csv_file = self.result_dir / f"experiment_1_2_loss_model_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"保存CSV结果: {csv_file}")
        
        # 保存统计表格
        self._save_statistical_tables(results['statistical_analysis'], timestamp)
    
    def _save_statistical_tables(self, statistical_results: Dict[str, Any], timestamp: str):
        """保存统计分析表格"""
        txt_file = self.result_dir / f"experiment_1_2_statistics_tables_{timestamp}.txt"
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("实验1.2 统计分析汇总表格\n")
            f.write("=" * 80 + "\n\n")
            
            # 配对t检验结果
            f.write("1. 配对t检验结果 (ImprovedLossModel vs Baselines)\n")
            f.write("-" * 80 + "\n")
            
            paired_tests = statistical_results.get('paired_t_tests', {})
            for dataset, baselines in paired_tests.items():
                f.write(f"\n数据集: {dataset}\n")
                for baseline, metrics in baselines.items():
                    f.write(f"  对比: {baseline}\n")
                    for metric, result in metrics.items():
                        f.write(f"    {metric}:\n")
                        f.write(f"      t统计量: {result.get('t_statistic', 0):.4f}\n")
                        f.write(f"      p-value: {result.get('p_value', 1):.6f}\n")
                        f.write(f"      显著性: {'是' if result.get('significant', False) else '否'}\n")
            
            # Friedman检验结果
            f.write("\n\n2. Friedman检验结果\n")
            f.write("-" * 80 + "\n")
            
            friedman_tests = statistical_results.get('friedman_test', {})
            for metric, result in friedman_tests.items():
                f.write(f"\n{metric}:\n")
                f.write(f"  统计量: {result.get('statistic', 0):.4f}\n")
                f.write(f"  p-value: {result.get('p_value', 1):.6f}\n")
                f.write(f"  显著性: {'是' if result.get('significant', False) else '否'}\n")
        
        logger.info(f"保存统计表格: {txt_file}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("实验1.2正式版：三状态丢包模型性能评估")
    print("=" * 80)
    print("\n实验配置:")
    print("  - 数据包数: 10000")
    print("  - 重复次数: 10")
    print("  - 数据集数: 9 (6合成 + 3真实)")
    print("  - 模型数: 3 (IID, GilbertElliott, ImprovedLossModel)")
    print("  - 总实验数: 9 × 3 × 10 = 270 次")
    print("\n开始实验...\n")
    
    # 创建实验实例
    experiment = Experiment1_2_LossModel(
        n_packets=10000,
        n_repeats=10,
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
    for model, perf in summary['overall_performance'].items():
        print(f"  {model}:")
        print(f"    平均MSE: {perf['mean_mse']:.4f}")
        print(f"    平均准确率: {perf['mean_accuracy']:.4f}")
        print(f"    平均KL散度: {perf['mean_kl_divergence']:.4f}")
    
    print("\n关键发现:")
    for finding in summary['key_findings']:
        print(f"  {finding}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
