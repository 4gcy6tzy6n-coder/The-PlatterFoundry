#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验2.1正式版：MCTS智能缺陷发现

科学假设：
集成"MCTS智能测试生成"的平台，能在相同实验预算内，发现比传统场景枚举方法
更多、更严重的算法缺陷。

实验设计：
1. 目标算法：
   - TCP CUBIC (传统拥塞控制)
   - BBR (基于带宽延迟积)

2. 对照组（传统评估）：
   - 25个预定义场景（固定RTT、丢包率、带宽组合）

3. 实验组（MCTS探索）：
   - MCTS智能测试生成
   - 相同的计算预算

4. 评估指标：
   - 缺陷发现数量
   - 缺陷严重度加权得分
   - 场景多样性（参数空间覆盖度）

IEEE/ACM ToN 标准实验实现
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys
import os
import random

import numpy as np
import pandas as pd
from scipy import stats

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NetworkScenario:
    """网络场景定义"""
    
    def __init__(self, rtt_ms: float, loss_rate: float, bandwidth_mbps: float, 
                 jitter_ms: float = 0, scenario_id: str = None):
        self.rtt_ms = rtt_ms
        self.loss_rate = loss_rate
        self.bandwidth_mbps = bandwidth_mbps
        self.jitter_ms = jitter_ms
        self.scenario_id = scenario_id or f"rtt{rtt_ms}_loss{loss_rate}_bw{bandwidth_mbps}"
    
    def to_dict(self) -> Dict:
        return {
            'rtt_ms': self.rtt_ms,
            'loss_rate': self.loss_rate,
            'bandwidth_mbps': self.bandwidth_mbps,
            'jitter_ms': self.jitter_ms,
            'scenario_id': self.scenario_id
        }


class AlgorithmSimulator:
    """算法模拟器 - 简化的TCP/BBR行为"""
    
    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.cwnd = 10  # 初始拥塞窗口
        self.ssthresh = 64
    
    def run(self, scenario: NetworkScenario, duration_sec: int = 10) -> Dict[str, Any]:
        """运行算法并返回性能指标"""
        # 简化模拟：基于场景参数计算性能
        
        if self.algorithm_name == "CUBIC":
            # CUBIC对丢包敏感
            throughput = scenario.bandwidth_mbps * (1 - scenario.loss_rate * 10)
            throughput = max(0.1, throughput)  # 最小吞吐量
            
            # CUBIC在高RTT下性能下降
            rtt_penalty = 1 / (1 + scenario.rtt_ms / 100)
            throughput *= rtt_penalty
            
            # 计算丢包导致的吞吐量损失
            fairness = 0.8 if scenario.loss_rate < 0.05 else 0.5
            latency = scenario.rtt_ms * (1 + scenario.loss_rate * 5)
            
        elif self.algorithm_name == "BBR":
            # BBR对带宽延迟积敏感
            bdp = scenario.bandwidth_mbps * scenario.rtt_ms / 8000  # 简化计算
            throughput = min(scenario.bandwidth_mbps * 0.95, bdp * 10)
            
            # BBR在低丢包时表现更好
            if scenario.loss_rate < 0.01:
                throughput *= 1.1
            elif scenario.loss_rate > 0.1:
                throughput *= 0.7
            
            fairness = 0.9
            latency = scenario.rtt_ms * (1 + scenario.loss_rate * 2)
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")
        
        # 添加抖动影响
        throughput *= (1 - scenario.jitter_ms / 100)
        latency += scenario.jitter_ms
        
        return {
            'throughput_mbps': max(0.1, throughput),
            'latency_ms': latency,
            'fairness': fairness,
            'packet_loss': scenario.loss_rate
        }


class DefectDetector:
    """缺陷检测器"""
    
    def __init__(self):
        self.defect_thresholds = {
            'low_throughput': 1.0,  # Mbps
            'high_latency': 500,  # ms
            'poor_fairness': 0.7,
            'high_packet_loss': 0.15
        }
    
    def detect_defects(self, result: Dict[str, Any], scenario: NetworkScenario) -> List[Dict[str, Any]]:
        """检测性能缺陷"""
        defects = []
        
        # 检测吞吐量过低
        if result['throughput_mbps'] < self.defect_thresholds['low_throughput']:
            severity = 'critical' if result['throughput_mbps'] < 0.5 else 'high'
            defects.append({
                'type': 'low_throughput',
                'severity': severity,
                'value': result['throughput_mbps'],
                'expected': self.defect_thresholds['low_throughput'],
                'scenario': scenario.scenario_id
            })
        
        # 检测延迟过高
        if result['latency_ms'] > self.defect_thresholds['high_latency']:
            severity = 'high' if result['latency_ms'] < 1000 else 'critical'
            defects.append({
                'type': 'high_latency',
                'severity': severity,
                'value': result['latency_ms'],
                'expected': self.defect_thresholds['high_latency'],
                'scenario': scenario.scenario_id
            })
        
        # 检测公平性差
        if result['fairness'] < self.defect_thresholds['poor_fairness']:
            severity = 'medium' if result['fairness'] > 0.5 else 'high'
            defects.append({
                'type': 'poor_fairness',
                'severity': severity,
                'value': result['fairness'],
                'expected': self.defect_thresholds['poor_fairness'],
                'scenario': scenario.scenario_id
            })
        
        # 检测丢包率异常
        if result['packet_loss'] > self.defect_thresholds['high_packet_loss']:
            defects.append({
                'type': 'high_packet_loss',
                'severity': 'medium',
                'value': result['packet_loss'],
                'expected': self.defect_thresholds['high_packet_loss'],
                'scenario': scenario.scenario_id
            })
        
        return defects


class TraditionalTester:
    """传统测试方法 - 预定义场景枚举"""
    
    def __init__(self, n_scenarios: int = 100):
        self.n_scenarios = n_scenarios
        self.scenarios = self._generate_predefined_scenarios()
    
    def _generate_predefined_scenarios(self) -> List[NetworkScenario]:
        """生成预定义的测试场景（扩展到100个）"""
        scenarios = []
        
        # RTT: 5ms到1000ms，10个档位
        rtts = [5, 10, 20, 50, 100, 150, 200, 300, 500, 1000]
        
        # 丢包率: 0%到25%，10个档位
        loss_rates = [0.0, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25]
        
        # 带宽: 1Mbps到10000Mbps
        bandwidths = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
        
        # 抖动: 0ms到50ms
        jitters = [0, 5, 10, 20, 30, 50]
        
        # 组合生成（确保100个场景）
        count = 0
        for rtt in rtts:
            for loss in loss_rates:
                if count >= self.n_scenarios:
                    break
                bw = bandwidths[count % len(bandwidths)]
                jitter = jitters[count % len(jitters)]
                scenarios.append(NetworkScenario(
                    rtt_ms=rtt,
                    loss_rate=loss,
                    bandwidth_mbps=bw,
                    jitter_ms=jitter,
                    scenario_id=f"traditional_{count}"
                ))
                count += 1
            if count >= self.n_scenarios:
                break
        
        return scenarios
    
    def run_tests(self, algorithm_name: str) -> Tuple[List[Dict], List[Dict]]:
        """运行传统测试"""
        simulator = AlgorithmSimulator(algorithm_name)
        detector = DefectDetector()
        
        all_results = []
        all_defects = []
        
        for scenario in self.scenarios:
            result = simulator.run(scenario)
            result['scenario'] = scenario.to_dict()
            result['algorithm'] = algorithm_name
            all_results.append(result)
            
            # 检测缺陷
            defects = detector.detect_defects(result, scenario)
            all_defects.extend(defects)
        
        return all_results, all_defects


class MCTSNode:
    """MCTS节点"""
    
    def __init__(self, scenario: NetworkScenario, parent=None):
        self.scenario = scenario
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.defect_score = 0.0
    
    def is_fully_expanded(self) -> bool:
        """检查是否完全展开"""
        return len(self.children) >= 5  # 每个节点最多5个子节点
    
    def best_child(self, exploration_weight: float = 1.414) -> 'MCTSNode':
        """选择最佳子节点（UCB1）"""
        if not self.children:
            return None
        
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                return child
            
            exploitation = child.reward / child.visits
            exploration = exploration_weight * np.sqrt(np.log(self.visits) / child.visits)
            choices_weights.append(exploitation + exploration)
        
        return self.children[np.argmax(choices_weights)]


class MCTSTester:
    """MCTS智能测试生成（优化版）"""
    
    def __init__(self, n_simulations: int = 100, exploration_weight: float = 2.5):
        self.n_simulations = n_simulations
        self.exploration_weight = exploration_weight
        self.detector = DefectDetector()
        self.defect_focused = True  # 启用缺陷导向探索
    
    def run_tests(self, algorithm_name: str) -> Tuple[List[Dict], List[Dict]]:
        """运行MCTS测试"""
        simulator = AlgorithmSimulator(algorithm_name)
        
        # 初始根节点 - 标准场景
        root_scenario = NetworkScenario(rtt_ms=100, loss_rate=0.05, bandwidth_mbps=100)
        root = MCTSNode(root_scenario)
        
        all_results = []
        all_defects = []
        explored_scenarios = set()
        
        for sim in range(self.n_simulations):
            # MCTS四个步骤
            node = self._select(root)
            
            if node.visits > 0 and not node.is_fully_expanded():
                node = self._expand(node)
            
            reward, result, defects = self._simulate(node, simulator, algorithm_name)
            
            # 记录结果
            if node.scenario.scenario_id not in explored_scenarios:
                all_results.append(result)
                all_defects.extend(defects)
                explored_scenarios.add(node.scenario.scenario_id)
            
            self._backpropagate(node, reward)
        
        return all_results, all_defects
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """选择阶段"""
        while node.children and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """扩展阶段 - 生成新的网络场景"""
        # 基于当前场景生成变体
        base = node.scenario
        
        # 随机扰动参数
        perturbations = [
            ('rtt', np.random.uniform(0.5, 2.0)),
            ('loss', np.random.uniform(0.5, 2.0)),
            ('bandwidth', np.random.uniform(0.5, 2.0)),
            ('jitter', np.random.uniform(0, 50))
        ]
        
        param, factor = random.choice(perturbations)
        
        if param == 'rtt':
            new_rtt = np.clip(base.rtt_ms * factor, 1, 1000)
            new_scenario = NetworkScenario(new_rtt, base.loss_rate, base.bandwidth_mbps, base.jitter_ms)
        elif param == 'loss':
            new_loss = np.clip(base.loss_rate * factor, 0, 0.5)
            new_scenario = NetworkScenario(base.rtt_ms, new_loss, base.bandwidth_mbps, base.jitter_ms)
        elif param == 'bandwidth':
            new_bw = np.clip(base.bandwidth_mbps * factor, 1, 10000)
            new_scenario = NetworkScenario(base.rtt_ms, base.loss_rate, new_bw, base.jitter_ms)
        else:  # jitter
            new_jitter = factor
            new_scenario = NetworkScenario(base.rtt_ms, base.loss_rate, base.bandwidth_mbps, new_jitter)
        
        new_scenario.scenario_id = f"mcts_{len(node.children)}_{random.randint(1000,9999)}"
        child = MCTSNode(new_scenario, parent=node)
        node.children.append(child)
        
        return child
    
    def _simulate(self, node: MCTSNode, simulator: AlgorithmSimulator, 
                  algorithm_name: str) -> Tuple[float, Dict, List[Dict]]:
        """模拟阶段（优化版 - 增强缺陷发现能力）"""
        result = simulator.run(node.scenario)
        result['scenario'] = node.scenario.to_dict()
        result['algorithm'] = algorithm_name
        
        # 检测缺陷
        defects = self.detector.detect_defects(result, node.scenario)
        
        # 优化的奖励函数：大幅提升缺陷奖励
        severity_weights = {'critical': 10.0, 'high': 5.0, 'medium': 2.0, 'low': 1.0}
        defect_score = sum(severity_weights.get(d['severity'], 0) for d in defects)
        
        # 额外奖励：发现多种类型的缺陷
        unique_types = len(set(d['type'] for d in defects))
        diversity_bonus = unique_types * 2.0
        
        # 额外奖励：极端场景探索
        extremity_bonus = 0
        if node.scenario.rtt_ms > 300 or node.scenario.loss_rate > 0.15:
            extremity_bonus = 3.0
        
        # 总奖励
        reward = defect_score + diversity_bonus + extremity_bonus
        
        return reward, result, defects
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """反向传播阶段"""
        while node is not None:
            node.visits += 1
            node.reward += reward
            node = node.parent


class Experiment2_1_MCTSDefectDiscovery:
    """实验2.1：MCTS缺陷发现对比（优化版）"""
    
    def __init__(self, n_simulations: int = 100, seed: int = 42):
        self.n_simulations = n_simulations
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # 创建输出目录
        self.result_dir = Path("result")
        self.result_dir.mkdir(exist_ok=True)
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """运行所有实验"""
        logger.info("=" * 60)
        logger.info("实验2.1正式版：MCTS智能缺陷发现")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        algorithms = ['CUBIC', 'BBR']
        
        results = {
            'traditional': {},
            'mcts': {}
        }
        
        for algorithm in algorithms:
            logger.info(f"\n测试算法: {algorithm}")
            
            # 传统方法
            logger.info("  运行传统枚举测试...")
            traditional_tester = TraditionalTester(n_scenarios=self.n_simulations)
            trad_results, trad_defects = traditional_tester.run_tests(algorithm)
            
            results['traditional'][algorithm] = {
                'results': trad_results,
                'defects': trad_defects,
                'n_defects': len(trad_defects),
                'unique_defect_types': len(set(d['type'] for d in trad_defects))
            }
            
            logger.info(f"    发现缺陷: {len(trad_defects)}")
            
            # MCTS方法
            logger.info("  运行MCTS智能测试...")
            mcts_tester = MCTSTester(n_simulations=self.n_simulations)
            mcts_results, mcts_defects = mcts_tester.run_tests(algorithm)
            
            results['mcts'][algorithm] = {
                'results': mcts_results,
                'defects': mcts_defects,
                'n_defects': len(mcts_defects),
                'unique_defect_types': len(set(d['type'] for d in mcts_defects))
            }
            
            logger.info(f"    发现缺陷: {len(mcts_defects)}")
        
        # 统计分析
        logger.info("\n执行统计分析...")
        statistical_results = self._perform_statistical_analysis(results)
        
        duration = time.time() - start_time
        
        final_results = {
            'experiment_info': {
                'name': 'Experiment 2.1 MCTS Defect Discovery',
                'description': 'MCTS智能缺陷发现 vs 传统枚举测试',
                'n_simulations': self.n_simulations,
                'algorithms': algorithms,
                'duration_seconds': duration,
                'timestamp': time.time()
            },
            'results': results,
            'statistical_analysis': statistical_results,
            'summary': self._generate_summary(results)
        }
        
        # 保存结果
        self._save_results(final_results)
        
        logger.info(f"\n实验完成！总用时: {duration:.2f} 秒")
        logger.info(f"结果保存至: {self.result_dir}")
        
        return final_results
    
    def _perform_statistical_analysis(self, results: Dict) -> Dict[str, Any]:
        """执行统计分析"""
        analysis = {
            'defect_comparison': {},
            'severity_analysis': {},
            'diversity_analysis': {}
        }
        
        for algorithm in ['CUBIC', 'BBR']:
            trad = results['traditional'][algorithm]
            mcts = results['mcts'][algorithm]
            
            # 缺陷数量对比
            analysis['defect_comparison'][algorithm] = {
                'traditional_count': trad['n_defects'],
                'mcts_count': mcts['n_defects'],
                'improvement': ((mcts['n_defects'] - trad['n_defects']) / 
                               max(trad['n_defects'], 1) * 100)
            }
            
            # 严重度分析
            trad_severity = self._calculate_severity_score(trad['defects'])
            mcts_severity = self._calculate_severity_score(mcts['defects'])
            
            analysis['severity_analysis'][algorithm] = {
                'traditional_score': trad_severity,
                'mcts_score': mcts_severity,
                'improvement': ((mcts_severity - trad_severity) / 
                               max(trad_severity, 1) * 100)
            }
            
            # 多样性分析
            trad_scenarios = {r['scenario']['scenario_id'] for r in trad['results']}
            mcts_scenarios = {r['scenario']['scenario_id'] for r in mcts['results']}
            
            analysis['diversity_analysis'][algorithm] = {
                'traditional_scenarios': len(trad_scenarios),
                'mcts_scenarios': len(mcts_scenarios),
                'unique_to_mcts': len(mcts_scenarios - trad_scenarios)
            }
        
        return analysis
    
    def _calculate_severity_score(self, defects: List[Dict]) -> float:
        """计算缺陷严重度加权得分"""
        severity_weights = {'critical': 3.0, 'high': 2.0, 'medium': 1.0, 'low': 0.5}
        return sum(severity_weights.get(d['severity'], 0) for d in defects)
    
    def _generate_summary(self, results: Dict) -> Dict[str, Any]:
        """生成汇总报告"""
        summary = {
            'total_defects': {},
            'best_method': {},
            'key_findings': []
        }
        
        for algorithm in ['CUBIC', 'BBR']:
            trad_count = results['traditional'][algorithm]['n_defects']
            mcts_count = results['mcts'][algorithm]['n_defects']
            
            summary['total_defects'][algorithm] = {
                'traditional': trad_count,
                'mcts': mcts_count,
                'improvement_pct': ((mcts_count - trad_count) / max(trad_count, 1) * 100)
            }
            
            if mcts_count > trad_count:
                summary['best_method'][algorithm] = 'MCTS'
                summary['key_findings'].append(
                    f"MCTS在{algorithm}上发现了{mcts_count - trad_count}个额外缺陷"
                )
            else:
                summary['best_method'][algorithm] = 'Traditional'
        
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
        json_file = self.result_dir / f"experiment_2_1_mcts_defect_{timestamp}.json"
        serializable_results = self._convert_to_json_serializable(results)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        logger.info(f"保存JSON结果: {json_file}")
        
        # 保存汇总报告
        self._save_summary_report(results, timestamp)
    
    def _save_summary_report(self, results: Dict[str, Any], timestamp: str):
        """保存汇总报告"""
        txt_file = self.result_dir / f"experiment_2_1_summary_{timestamp}.txt"
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("实验2.1 MCTS智能缺陷发现汇总报告\n")
            f.write("=" * 80 + "\n\n")
            
            summary = results['summary']
            
            f.write("1. 缺陷发现对比\n")
            f.write("-" * 80 + "\n")
            for algo, counts in summary['total_defects'].items():
                f.write(f"\n{algo}:\n")
                f.write(f"  传统方法: {counts['traditional']} 个缺陷\n")
                f.write(f"  MCTS方法: {counts['mcts']} 个缺陷\n")
                f.write(f"  改进幅度: {counts['improvement_pct']:.1f}%\n")
            
            f.write("\n\n2. 关键发现\n")
            f.write("-" * 80 + "\n")
            for finding in summary['key_findings']:
                f.write(f"  • {finding}\n")
        
        logger.info(f"保存汇总报告: {txt_file}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("实验2.1优化版：MCTS智能缺陷发现（参数调优）")
    print("=" * 80)
    print("\n实验配置:")
    print("  - 模拟次数: 100 (优化后)")
    print("  - 探索权重: 2.5 (提升探索能力)")
    print("  - 奖励函数: 优化版（缺陷导向+多样性+极端场景）")
    print("  - 测试算法: CUBIC, BBR")
    print("  - 对照组: 传统枚举测试 (100个预定义场景)")
    print("  - 实验组: MCTS智能探索 (100次模拟)")
    print("\n开始实验...\n")
    
    # 创建实验实例
    experiment = Experiment2_1_MCTSDefectDiscovery(
        n_simulations=100,
        seed=42
    )
    
    # 运行实验
    results = experiment.run_all_experiments()
    
    # 显示关键结果
    print("\n" + "=" * 80)
    print("实验完成！关键结果:")
    print("=" * 80)
    
    summary = results['summary']
    print("\n缺陷发现对比:")
    for algo, counts in summary['total_defects'].items():
        print(f"  {algo}:")
        print(f"    传统方法: {counts['traditional']} 个")
        print(f"    MCTS方法: {counts['mcts']} 个")
        print(f"    改进幅度: {counts['improvement_pct']:.1f}%")
    
    print("\n关键发现:")
    for finding in summary['key_findings']:
        print(f"  • {finding}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
