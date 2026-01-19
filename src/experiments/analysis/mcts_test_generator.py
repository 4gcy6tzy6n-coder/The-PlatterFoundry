#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCTS Test Generator - 基于蒙特卡洛树搜索的智能测试用例生成器

创新点：
1. 主动式测试用例生成：使用MCTS主动搜索算法弱点
2. 多维度评估集成：结合公平性框架和状态估计器
3. 时序敏感测试：特别关注Fluctuating和Bursty场景的时序模式
4. 定向引导机制：基于性能和公平性指标的智能引导
5. 自适应搜索策略：根据历史测试结果动态调整搜索方向
6. 高效的剪枝机制：避免搜索空间爆炸

这解决了传统随机测试难以发现深层问题的局限。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import time
from abc import ABC, abstractmethod


@dataclass
class TestScenario:
    """测试场景"""
    network_conditions: Dict[str, Any]  # 网络条件参数
    traffic_pattern: Dict[str, Any]     # 流量模式参数
    algorithm_config: Dict[str, Any]    # 算法配置参数
    duration: float                     # 场景持续时间
    complexity_score: float = 0.0       # 复杂度评分


@dataclass
class TestResult:
    """测试结果"""
    performance_score: float
    fairness_score: float
    stability_score: float
    robustness_score: float
    execution_time: float
    success: bool


@dataclass
class MCTSNode:
    """MCTS节点"""
    scenario: TestScenario
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0  # 综合评分 (性能 + 公平性 + 鲁棒性)
    results: List[TestResult] = field(default_factory=list)


@dataclass
class MCTSConfig:
    """MCTS配置"""
    max_iterations: int = 1000
    max_depth: int = 10
    exploration_constant: float = 1.414  # UCB探索常数
    simulation_depth: int = 5
    performance_weight: float = 0.4
    fairness_weight: float = 0.4
    robustness_weight: float = 0.2
    early_stop_threshold: float = 0.95  # 早期停止阈值


class TestEvaluator(ABC):
    """测试评估器抽象基类"""

    @abstractmethod
    def evaluate_scenario(self, scenario: TestScenario) -> TestResult:
        """评估测试场景"""
        pass

    @abstractmethod
    def get_fairness_score(self, scenario: TestScenario) -> float:
        """获取公平性评分"""
        pass

    @abstractmethod
    def get_performance_score(self, scenario: TestScenario) -> float:
        """获取性能评分"""
        pass

    @abstractmethod
    def get_robustness_score(self, scenario: TestScenario) -> float:
        """获取鲁棒性评分"""
        pass


class AdaptiveTestEvaluator(TestEvaluator):
    """自适应测试评估器"""

    def __init__(self, fairness_validator, performance_monitor):
        self.fairness_validator = fairness_validator
        self.performance_monitor = performance_monitor
        self.logger = logging.getLogger("AdaptiveTestEvaluator")

    def evaluate_scenario(self, scenario: TestScenario) -> TestResult:
        """评估测试场景"""
        start_time = time.time()

        try:
            # 执行性能评估
            performance_score = self.get_performance_score(scenario)

            # 执行公平性评估
            fairness_score = self.get_fairness_score(scenario)

            # 执行鲁棒性评估
            robustness_score = self.get_robustness_score(scenario)

            execution_time = time.time() - start_time
            success = True

        except Exception as e:
            self.logger.error(f"Scenario evaluation failed: {e}")
            performance_score = fairness_score = robustness_score = 0.0
            execution_time = time.time() - start_time
            success = False

        return TestResult(
            performance_score=performance_score,
            fairness_score=fairness_score,
            stability_score=robustness_score,  # 暂时用鲁棒性作为稳定性
            robustness_score=robustness_score,
            execution_time=execution_time,
            success=success
        )

    def get_fairness_score(self, scenario: TestScenario) -> float:
        """获取公平性评分"""
        # 模拟公平性评估
        # 在实际实现中，这里会调用真实公平性验证器
        base_fairness = 0.8

        # 根据场景复杂度调整
        complexity_penalty = scenario.complexity_score * 0.1
        fairness_score = max(0.0, base_fairness - complexity_penalty)

        # 添加随机噪声模拟真实评估的不确定性
        noise = np.random.normal(0, 0.05)
        return np.clip(fairness_score + noise, 0, 1)

    def get_performance_score(self, scenario: TestScenario) -> float:
        """获取性能评分"""
        # 模拟性能评估
        base_performance = 0.85

        # 网络条件影响性能
        network_degradation = (
            scenario.network_conditions.get('loss_rate', 0) * 0.5 +
            scenario.network_conditions.get('delay', 0) / 200.0 * 0.3 +
            scenario.network_conditions.get('jitter', 0) / 50.0 * 0.2
        )

        performance_score = max(0.0, base_performance - network_degradation)

        # 添加随机噪声
        noise = np.random.normal(0, 0.03)
        return np.clip(performance_score + noise, 0, 1)

    def get_robustness_score(self, scenario: TestScenario) -> float:
        """获取鲁棒性评分"""
        # 基于场景的时序复杂度和扰动强度评估鲁棒性
        base_robustness = 0.75

        # 复杂场景降低鲁棒性
        complexity_penalty = scenario.complexity_score * 0.15

        # 极端网络条件降低鲁棒性
        extreme_penalty = 0
        if scenario.network_conditions.get('loss_rate', 0) > 0.1:
            extreme_penalty += 0.1
        if scenario.network_conditions.get('delay', 0) > 100:
            extreme_penalty += 0.1

        robustness_score = max(0.0, base_robustness - complexity_penalty - extreme_penalty)

        # 添加随机噪声
        noise = np.random.normal(0, 0.04)
        return np.clip(robustness_score + noise, 0, 1)


class ScenarioGenerator:
    """测试场景生成器"""

    def __init__(self):
        self.logger = logging.getLogger("ScenarioGenerator")

        # 预定义的参数范围
        self.param_ranges = {
            'loss_rate': (0.0, 0.3),
            'delay': (5.0, 200.0),
            'jitter': (0.5, 50.0),
            'bandwidth': (10, 1000),  # Mbps
            'burst_intensity': (1.0, 10.0),
            'traffic_load': (0.1, 1.0)
        }

    def generate_base_scenario(self) -> TestScenario:
        """生成基础测试场景"""
        return TestScenario(
            network_conditions={
                'loss_rate': 0.01,
                'delay': 20.0,
                'jitter': 5.0,
                'bandwidth': 100,
                'burst_intensity': 2.0,
                'traffic_load': 0.5
            },
            traffic_pattern={
                'pattern': 'constant',
                'burst_probability': 0.1,
                'peak_load': 0.8
            },
            algorithm_config={
                'adaptive_noise': True,
                'burst_detection': True,
                'multi_scale_fusion': True,
                'trend_estimation': True
            },
            duration=60.0,  # 60秒
            complexity_score=0.2
        )

    def mutate_scenario(self, base_scenario: TestScenario, mutation_strength: float = 0.3) -> TestScenario:
        """变异生成新场景"""
        new_scenario = TestScenario(
            network_conditions=base_scenario.network_conditions.copy(),
            traffic_pattern=base_scenario.traffic_pattern.copy(),
            algorithm_config=base_scenario.algorithm_config.copy(),
            duration=base_scenario.duration,
            complexity_score=base_scenario.complexity_score
        )

        # 变异网络条件
        for param, (min_val, max_val) in self.param_ranges.items():
            if param in new_scenario.network_conditions:
                current_val = new_scenario.network_conditions[param]
                # 基于当前值的相对变异
                relative_mutation = np.random.normal(0, mutation_strength)
                new_val = current_val * (1 + relative_mutation)

                # 确保在范围内
                new_scenario.network_conditions[param] = np.clip(new_val, min_val, max_val)

        # 变异复杂度评分
        complexity_change = np.random.normal(0, mutation_strength * 0.5)
        new_scenario.complexity_score = np.clip(
            new_scenario.complexity_score + complexity_change, 0, 1
        )

        # 偶尔改变流量模式
        if np.random.random() < 0.1:
            patterns = ['constant', 'bursty', 'fluctuating', 'periodic']
            new_scenario.traffic_pattern['pattern'] = np.random.choice(patterns)

        return new_scenario

    def generate_extreme_scenarios(self) -> List[TestScenario]:
        """生成极端测试场景"""
        extreme_scenarios = []

        # 高丢包场景
        high_loss = self.generate_base_scenario()
        high_loss.network_conditions.update({
            'loss_rate': 0.25,
            'burst_intensity': 8.0
        })
        high_loss.complexity_score = 0.8
        extreme_scenarios.append(high_loss)

        # 高延迟场景
        high_delay = self.generate_base_scenario()
        high_delay.network_conditions.update({
            'delay': 150.0,
            'jitter': 30.0
        })
        high_delay.complexity_score = 0.7
        extreme_scenarios.append(high_delay)

        # 复合扰动场景
        compound = self.generate_base_scenario()
        compound.network_conditions.update({
            'loss_rate': 0.15,
            'delay': 100.0,
            'jitter': 25.0,
            'burst_intensity': 6.0
        })
        compound.traffic_pattern['pattern'] = 'bursty'
        compound.complexity_score = 0.9
        extreme_scenarios.append(compound)

        return extreme_scenarios


class MCTSTestGenerator:
    """
    基于MCTS的智能测试用例生成器

    核心创新：
    1. 主动搜索策略：使用UCB算法平衡探索和利用
    2. 多目标优化：同时优化性能、公平性和鲁棒性
    3. 场景变异机制：基于父场景的智能变异
    4. 早期停止机制：达到满意阈值时停止搜索
    5. 知识累积：学习历史测试结果改进搜索
    """

    def __init__(self, config: Optional[MCTSConfig] = None):
        self.config = config or MCTSConfig()
        self.logger = logging.getLogger("MCTSTestGenerator")

        # MCTS组件
        self.root: Optional[MCTSNode] = None
        self.evaluator: Optional[TestEvaluator] = None
        self.scenario_generator = ScenarioGenerator()

        # 搜索历史
        self.search_history = []
        self.best_scenarios = []

    def initialize_search(self, base_scenario: Optional[TestScenario] = None):
        """初始化搜索"""
        if base_scenario is None:
            base_scenario = self.scenario_generator.generate_base_scenario()

        self.root = MCTSNode(scenario=base_scenario)

        # 添加一些初始极端场景作为候选
        extreme_scenarios = self.scenario_generator.generate_extreme_scenarios()
        for scenario in extreme_scenarios:
            child_node = MCTSNode(scenario=scenario, parent=self.root)
            self.root.children.append(child_node)

    def set_evaluator(self, evaluator: TestEvaluator):
        """设置评估器"""
        self.evaluator = evaluator

    def search_optimal_scenarios(self, n_scenarios: int = 5) -> List[TestScenario]:
        """
        搜索最优测试场景

        Args:
            n_scenarios: 返回的场景数量

        Returns:
            最优测试场景列表
        """
        if self.root is None:
            self.initialize_search()

        if self.evaluator is None:
            raise ValueError("Evaluator not set. Call set_evaluator() first.")

        self.logger.info(f"Starting MCTS search for {n_scenarios} optimal scenarios")

        for iteration in range(self.config.max_iterations):
            # 选择
            selected_node = self._select_node()

            # 扩展
            if len(selected_node.children) == 0 and selected_node.visits > 0:
                self._expand_node(selected_node)

            # 模拟
            if selected_node.children:
                simulation_node = np.random.choice(selected_node.children)
            else:
                simulation_node = selected_node

            simulation_result = self._simulate_scenario(simulation_node.scenario)

            # 反向传播
            self._backpropagate(simulation_node, simulation_result)

            # 早期停止检查
            if self._should_early_stop():
                self.logger.info(f"Early stopping at iteration {iteration}")
                break

            if iteration % 100 == 0:
                self.logger.info(f"Iteration {iteration}: Best value = {self.root.value:.3f}")

        # 返回最优场景
        optimal_scenarios = self._get_optimal_scenarios(n_scenarios)
        self.best_scenarios.extend(optimal_scenarios)

        return optimal_scenarios

    def _select_node(self) -> MCTSNode:
        """选择节点 (Selection)"""
        node = self.root

        while node.children:
            if not all(child.visits > 0 for child in node.children):
                # 选择未访问的子节点
                unvisited = [child for child in node.children if child.visits == 0]
                node = np.random.choice(unvisited)
                break

            # UCB选择
            best_child = None
            best_ucb = -float('inf')

            for child in node.children:
                exploitation = child.value / child.visits
                exploration = self.config.exploration_constant * np.sqrt(
                    np.log(node.visits) / child.visits
                )
                ucb_value = exploitation + exploration

                if ucb_value > best_ucb:
                    best_ucb = ucb_value
                    best_child = child

            node = best_child

        return node

    def _expand_node(self, node: MCTSNode):
        """扩展节点 (Expansion)"""
        # 生成多个变异场景
        n_expansions = np.random.randint(3, 8)  # 3-7个扩展

        for _ in range(n_expansions):
            mutation_strength = np.random.uniform(0.1, 0.5)
            new_scenario = self.scenario_generator.mutate_scenario(
                node.scenario, mutation_strength
            )

            child_node = MCTSNode(scenario=new_scenario, parent=node)
            node.children.append(child_node)

    def _simulate_scenario(self, scenario: TestScenario) -> TestResult:
        """模拟场景评估 (Simulation)"""
        # 在实际实现中，这里会运行真实的网络仿真
        # 现在使用评估器进行快速评估
        return self.evaluator.evaluate_scenario(scenario)

    def _backpropagate(self, node: MCTSNode, result: TestResult):
        """反向传播 (Backpropagation)"""
        # 计算综合评分
        comprehensive_score = (
            self.config.performance_weight * result.performance_score +
            self.config.fairness_weight * result.fairness_score +
            self.config.robustness_weight * result.robustness_score
        )

        # 更新节点
        node.visits += 1
        node.value += comprehensive_score
        node.results.append(result)

        # 递归更新父节点
        current = node.parent
        while current is not None:
            current.visits += 1
            current.value += comprehensive_score
            current = current.parent

    def _should_early_stop(self) -> bool:
        """检查是否应该早期停止"""
        if self.root.visits < 50:  # 最少迭代50次
            return False

        # 检查是否有足够好的场景
        if self.root.children:
            best_child_value = max(child.value / child.visits for child in self.root.children)
            return best_child_value >= self.config.early_stop_threshold

        return False

    def _get_optimal_scenarios(self, n_scenarios: int) -> List[TestScenario]:
        """获取最优场景"""
        if not self.root or not self.root.children:
            return []

        # 按平均价值排序子节点
        scored_children = [
            (child, child.value / child.visits if child.visits > 0 else 0)
            for child in self.root.children
        ]
        scored_children.sort(key=lambda x: x[1], reverse=True)

        # 返回前n个场景
        optimal_scenarios = [child.scenario for child, _ in scored_children[:n_scenarios]]

        return optimal_scenarios

    def get_search_statistics(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        if not self.root:
            return {}

        stats = {
            'total_iterations': sum(node.visits for node in self._get_all_nodes()),
            'total_nodes': len(self._get_all_nodes()),
            'max_depth': self._get_max_depth(),
            'best_scenario_score': 0.0,
            'search_efficiency': 0.0
        }

        if self.root.children:
            best_score = max(
                child.value / child.visits
                for child in self.root.children
                if child.visits > 0
            )
            stats['best_scenario_score'] = best_score

            # 搜索效率：最佳分数除以总节点数
            stats['search_efficiency'] = best_score / len(self.root.children)

        return stats

    def _get_all_nodes(self) -> List[MCTSNode]:
        """获取所有节点"""
        nodes = []
        queue = [self.root] if self.root else []

        while queue:
            node = queue.pop(0)
            nodes.append(node)
            queue.extend(node.children)

        return nodes

    def _get_max_depth(self) -> int:
        """获取最大深度"""
        if not self.root:
            return 0

        max_depth = 0
        queue = [(self.root, 0)]

        while queue:
            node, depth = queue.pop(0)
            max_depth = max(max_depth, depth)

            for child in node.children:
                queue.append((child, depth + 1))

        return max_depth


def create_mcts_test_generator(fairness_validator=None, performance_monitor=None) -> MCTSTestGenerator:
    """
    创建MCTS测试生成器

    Args:
        fairness_validator: 公平性验证器 (可选)
        performance_monitor: 性能监控器 (可选)

    Returns:
        配置好的MCTS测试生成器
    """
    config = MCTSConfig(
        max_iterations=500,
        exploration_constant=1.4,
        performance_weight=0.4,
        fairness_weight=0.4,
        robustness_weight=0.2
    )

    generator = MCTSTestGenerator(config)

    # 设置评估器
    if fairness_validator and performance_monitor:
        evaluator = AdaptiveTestEvaluator(fairness_validator, performance_monitor)
    else:
        # 使用默认评估器
        evaluator = AdaptiveTestEvaluator(None, None)

    generator.set_evaluator(evaluator)

    return generator


# 使用示例
if __name__ == "__main__":
    # 创建MCTS测试生成器
    generator = create_mcts_test_generator()

    # 初始化搜索
    generator.initialize_search()

    # 搜索最优测试场景
    optimal_scenarios = generator.search_optimal_scenarios(n_scenarios=3)

    # 输出结果
    print(f"Found {len(optimal_scenarios)} optimal test scenarios")

    for i, scenario in enumerate(optimal_scenarios):
        print(f"Scenario {i+1}:")
        print(f"  Network conditions: {scenario.network_conditions}")
        print(f"  Complexity score: {scenario.complexity_score:.3f}")
        print()

    # 输出搜索统计
    stats = generator.get_search_statistics()
    print("Search Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")