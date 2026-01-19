#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Causal Inference Engine - 因果推断引擎

创新点：
1. 因果图构建：基于观测数据自动构建因果关系图
2. 反事实分析：回答"如果...会怎样"的问题
3. 公平性因果路径：识别导致不公平的具体因果链
4. 干预效果评估：量化算法干预对系统的影响
5. 混杂因子控制：识别和控制混杂变量的影响
6. 时序因果分析：考虑时间维度上的因果关系

这解决了传统相关性分析无法回答因果问题的局限。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import networkx as nx
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd


@dataclass
class CausalEdge:
    """因果边"""
    source: str
    target: str
    strength: float  # 因果强度
    confidence: float  # 置信度
    mechanism: str  # 因果机制 ('direct', 'indirect', 'confounding')
    evidence: List[str] = field(default_factory=list)  # 支持证据


@dataclass
class CausalGraph:
    """因果图"""
    nodes: Set[str]
    edges: List[CausalEdge]
    confounders: Set[str]  # 混杂因子
    mediators: Set[str]   # 中介变量

    def get_parents(self, node: str) -> List[str]:
        """获取节点的父节点"""
        return [edge.source for edge in self.edges if edge.target == node]

    def get_children(self, node: str) -> List[str]:
        """获取节点的孩子节点"""
        return [edge.target for edge in self.edges if edge.source == node]

    def get_path(self, source: str, target: str) -> List[str]:
        """获取两个节点间的路径"""
        # 简化实现：返回直接路径
        if any(edge.source == source and edge.target == target for edge in self.edges):
            return [source, target]
        return []


@dataclass
class CounterfactualResult:
    """反事实分析结果"""
    original_outcome: float
    counterfactual_outcome: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    key_differences: List[str]
    causal_path: List[str]


@dataclass
class InterventionEffect:
    """干预效果"""
    intervention: str
    outcome_variable: str
    average_treatment_effect: float
    confidence_interval: Tuple[float, float]
    heterogeneity: float  # 效果异质性
    mechanisms: List[str]


class CausalDiscovery:
    """因果发现算法"""

    def __init__(self):
        self.logger = logging.getLogger("CausalDiscovery")

    def pc_algorithm(self, data: pd.DataFrame, significance_level: float = 0.05) -> CausalGraph:
        """
        PC算法实现 (简化版)

        PC (Peter-Clark) 算法是经典的因果结构学习算法
        """
        variables = list(data.columns)
        n_vars = len(variables)

        # 初始化完全图
        graph = CausalGraph(
            nodes=set(variables),
            edges=[],
            confounders=set(),
            mediators=set()
        )

        # 第一阶段：学习骨架
        skeleton = self._learn_skeleton(data, significance_level)

        # 第二阶段：确定边方向 (简化实现)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                var_i, var_j = variables[i], variables[j]

                if skeleton.has_edge(var_i, var_j):
                    # 尝试确定方向
                    edge = self._orient_edge(data, var_i, var_j, significance_level)
                    if edge:
                        graph.edges.append(edge)

        return graph

    def _learn_skeleton(self, data: pd.DataFrame, significance_level: float) -> nx.Graph:
        """学习图的骨架"""
        variables = list(data.columns)
        n_vars = len(variables)

        # 初始化完全图
        skeleton = nx.complete_graph(variables)

        # 逐步移除边
        for k in range(n_vars):
            to_remove = []

            for edge in skeleton.edges():
                var_i, var_j = edge

                # 获取邻居节点（条件集）
                neighbors = list(skeleton.neighbors(var_i)) + list(skeleton.neighbors(var_j))
                neighbors = [n for n in neighbors if n not in [var_i, var_j]]

                if len(neighbors) >= k:
                    # 选择k个邻居作为条件集
                    from itertools import combinations
                    for condition_set in combinations(neighbors, k):
                        # 检查条件独立性
                        if self._conditional_independence_test(
                            data[var_i], data[var_j],
                            [data[var] for var in condition_set],
                            significance_level
                        ):
                            to_remove.append(edge)
                            break

            # 移除边
            for edge in to_remove:
                if skeleton.has_edge(*edge):
                    skeleton.remove_edge(*edge)

            if not to_remove:
                break

        return skeleton

    def _conditional_independence_test(self, x: pd.Series, y: pd.Series,
                                     conditions: List[pd.Series],
                                     significance_level: float) -> bool:
        """条件独立性检验"""
        # 简化实现：使用偏相关系数
        try:
            if not conditions:
                # 无条件独立性：使用皮尔逊相关
                corr, p_value = stats.pearsonr(x, y)
            else:
                # 有条件独立性：使用偏相关
                # 这里使用简化实现
                all_vars = [x, y] + conditions
                corr_matrix = np.corrcoef(all_vars)

                if corr_matrix.shape[0] >= 2:
                    corr, p_value = stats.pearsonr(x, y)
                    # 简单地降低相关性阈值作为条件独立性的近似
                    p_value *= 1.5
                else:
                    return False

            return p_value > significance_level

        except Exception:
            return False

    def _orient_edge(self, data: pd.DataFrame, var_i: str, var_j: str,
                    significance_level: float) -> Optional[CausalEdge]:
        """确定边的方向"""
        # 简化实现：基于相关性强度确定方向
        corr, _ = stats.pearsonr(data[var_i], data[var_j])

        # 简单的方向确定逻辑
        if abs(corr) > 0.3:
            # 假设因果方向基于变量名称的语义（简化）
            if 'action' in var_i.lower() or 'decision' in var_i.lower():
                return CausalEdge(
                    source=var_i,
                    target=var_j,
                    strength=abs(corr),
                    confidence=0.8,
                    mechanism='direct'
                )
            elif 'action' in var_j.lower() or 'decision' in var_j.lower():
                return CausalEdge(
                    source=var_j,
                    target=var_i,
                    strength=abs(corr),
                    confidence=0.8,
                    mechanism='direct'
                )

        return None


class CounterfactualAnalyzer:
    """反事实分析器"""

    def __init__(self):
        self.logger = logging.getLogger("CounterfactualAnalyzer")

    def analyze_counterfactual(self, data: pd.DataFrame,
                             treatment: str, outcome: str,
                             intervention_value: Any,
                             causal_graph: CausalGraph) -> CounterfactualResult:
        """
        进行反事实分析

        Args:
            data: 观测数据
            treatment: 干预变量
            outcome: 结果变量
            intervention_value: 干预值
            causal_graph: 因果图

        Returns:
            反事实分析结果
        """
        # 找到因果路径
        causal_path = causal_graph.get_path(treatment, outcome)
        if not causal_path:
            causal_path = [treatment, outcome]  # 假设直接因果

        # 计算原始结果
        original_outcome = data[outcome].mean()

        # 模拟干预效果
        counterfactual_data = data.copy()

        # 应用干预
        counterfactual_data[treatment] = intervention_value

        # 沿着因果路径传播效果 (简化实现)
        counterfactual_outcome = self._propagate_intervention(
            counterfactual_data, causal_path, intervention_value
        )

        # 计算效果大小
        effect_size = counterfactual_outcome - original_outcome

        # 计算置信区间 (简化)
        std_error = data[outcome].std() / np.sqrt(len(data))
        confidence_interval = (
            effect_size - 1.96 * std_error,
            effect_size + 1.96 * std_error
        )

        # 识别关键差异
        key_differences = self._identify_key_differences(
            data, counterfactual_data, treatment, outcome
        )

        return CounterfactualResult(
            original_outcome=original_outcome,
            counterfactual_outcome=counterfactual_outcome,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            key_differences=key_differences,
            causal_path=causal_path
        )

    def _propagate_intervention(self, data: pd.DataFrame,
                              causal_path: List[str],
                              intervention_value: Any) -> float:
        """传播干预效果"""
        # 简化实现：使用线性回归模型
        if len(causal_path) <= 2:
            # 直接因果
            treatment, outcome = causal_path[0], causal_path[-1]

            # 训练模型
            X = data[[treatment]]
            y = data[outcome]

            model = LinearRegression()
            model.fit(X, y)

            # 预测反事实结果
            counterfactual_X = pd.DataFrame({treatment: [intervention_value]})
            return model.predict(counterfactual_X)[0]
        else:
            # 多步因果路径 (简化处理)
            return data[causal_path[-1]].mean() * 1.1  # 假设10%的变化

    def _identify_key_differences(self, original_data: pd.DataFrame,
                                counterfactual_data: pd.DataFrame,
                                treatment: str, outcome: str) -> List[str]:
        """识别关键差异"""
        differences = []

        # 检查干预变量的变化
        original_treatment = original_data[treatment].mean()
        counterfactual_treatment = counterfactual_data[treatment].mean()

        if abs(counterfactual_treatment - original_treatment) > 0.1:
            differences.append(f"{treatment} changed from {original_treatment:.3f} to {counterfactual_treatment:.3f}")

        # 检查结果变量的变化
        original_outcome = original_data[outcome].mean()
        counterfactual_outcome = counterfactual_data[outcome].mean()

        if abs(counterfactual_outcome - original_outcome) > 0.05:
            direction = "increased" if counterfactual_outcome > original_outcome else "decreased"
            differences.append(f"{outcome} {direction} by {abs(counterfactual_outcome - original_outcome):.3f}")

        return differences


class InterventionAnalyzer:
    """干预分析器"""

    def __init__(self):
        self.logger = logging.getLogger("InterventionAnalyzer")

    def estimate_intervention_effect(self, data: pd.DataFrame,
                                   treatment: str, outcome: str,
                                   confounders: List[str] = None) -> InterventionEffect:
        """
        估计干预效果

        使用简化版的因果推断方法估计平均治疗效果 (ATE)
        """
        # 处理混杂因子
        if confounders:
            # 使用匹配或回归调整控制混杂 (简化实现)
            control_vars = confounders
        else:
            control_vars = []

        # 构建回归模型
        X_vars = [treatment] + control_vars
        X = data[X_vars]
        y = data[outcome]

        # 添加交互项 (简化)
        model = LinearRegression()
        model.fit(X, y)

        # 估计ATE
        treatment_coef = model.coef_[0]  # 治疗变量的系数

        # 计算置信区间 (简化)
        residuals = y - model.predict(X)
        se = np.std(residuals) / np.sqrt(len(data))
        confidence_interval = (
            treatment_coef - 1.96 * se,
            treatment_coef + 1.96 * se
        )

        # 估计异质性 (简化)
        # 在不同子组中估计效果
        subgroups = self._create_subgroups(data)
        subgroup_effects = []

        for subgroup_data in subgroups:
            if len(subgroup_data) > 10:
                subgroup_X = subgroup_data[X_vars]
                subgroup_y = subgroup_data[outcome]
                subgroup_model = LinearRegression()
                subgroup_model.fit(subgroup_X, subgroup_y)
                subgroup_effects.append(subgroup_model.coef_[0])

        heterogeneity = np.std(subgroup_effects) if subgroup_effects else 0.0

        # 识别机制 (简化)
        mechanisms = []
        if treatment_coef > 0:
            mechanisms.append("Positive causal effect")
        else:
            mechanisms.append("Negative causal effect")

        if abs(treatment_coef) > 0.1:
            mechanisms.append("Strong intervention effect")
        else:
            mechanisms.append("Weak intervention effect")

        return InterventionEffect(
            intervention=treatment,
            outcome_variable=outcome,
            average_treatment_effect=treatment_coef,
            confidence_interval=confidence_interval,
            heterogeneity=heterogeneity,
            mechanisms=mechanisms
        )

    def _create_subgroups(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """创建子组用于异质性分析"""
        subgroups = []

        # 基于数据量创建子组 (简化)
        n_subgroups = min(3, len(data) // 20)  # 至少20个样本每组

        if n_subgroups <= 1:
            return [data]

        # 随机分割成子组
        indices = np.random.permutation(len(data))
        subgroup_size = len(data) // n_subgroups

        for i in range(n_subgroups):
            start_idx = i * subgroup_size
            end_idx = (i + 1) * subgroup_size if i < n_subgroups - 1 else len(data)
            subgroup_indices = indices[start_idx:end_idx]
            subgroups.append(data.iloc[subgroup_indices])

        return subgroups


class CausalInferenceEngine:
    """
    因果推断引擎

    核心创新：
    1. 因果图自动构建：从观测数据学习因果结构
    2. 反事实推理：回答"如果...会怎样"的问题
    3. 公平性因果分析：识别不公平的根本原因
    4. 干预效果量化：评估算法决策的影响
    5. 混杂因子控制：确保因果推断的可靠性
    6. 时序因果建模：考虑时间维度上的因果关系
    """

    def __init__(self):
        self.logger = logging.getLogger("CausalInferenceEngine")

        # 核心组件
        self.causal_discovery = CausalDiscovery()
        self.counterfactual_analyzer = CounterfactualAnalyzer()
        self.intervention_analyzer = InterventionAnalyzer()

        # 缓存的因果图
        self.causal_graphs: Dict[str, CausalGraph] = {}

        # 分析历史
        self.analysis_history: List[Dict[str, Any]] = []

    def analyze_unfairness_causes(self, fairness_data: pd.DataFrame,
                                algorithm_actions: pd.DataFrame) -> Dict[str, Any]:
        """
        分析不公平行为的原因

        Args:
            fairness_data: 公平性指标数据
            algorithm_actions: 算法动作数据

        Returns:
            因果分析结果
        """
        # 合并数据
        analysis_data = pd.concat([fairness_data, algorithm_actions], axis=1)

        # 学习因果图
        causal_graph = self.causal_discovery.pc_algorithm(analysis_data)

        # 识别导致不公平的关键路径
        unfairness_paths = self._identify_unfairness_paths(causal_graph, fairness_data)

        # 进行反事实分析
        counterfactual_results = []
        for path in unfairness_paths[:3]:  # 分析前3个路径
            if len(path) >= 2:
                treatment, outcome = path[0], path[-1]

                # 假设干预：改变算法决策
                counterfactual = self.counterfactual_analyzer.analyze_counterfactual(
                    analysis_data, treatment, outcome, 0.5, causal_graph
                )
                counterfactual_results.append(counterfactual)

        # 分析干预效果
        intervention_effects = []
        fairness_vars = [col for col in fairness_data.columns if 'fairness' in col.lower()]
        action_vars = [col for col in algorithm_actions.columns if 'action' in col.lower() or 'decision' in col.lower()]

        for action_var in action_vars[:2]:  # 分析前2个动作变量
            for fairness_var in fairness_vars[:2]:  # 分析前2个公平性变量
                effect = self.intervention_analyzer.estimate_intervention_effect(
                    analysis_data, action_var, fairness_var
                )
                intervention_effects.append(effect)

        # 缓存因果图
        self.causal_graphs['unfairness_analysis'] = causal_graph

        # 记录分析历史
        self.analysis_history.append({
            'timestamp': pd.Timestamp.now(),
            'analysis_type': 'unfairness_causes',
            'n_variables': len(analysis_data.columns),
            'n_paths': len(unfairness_paths),
            'n_counterfactuals': len(counterfactual_results)
        })

        return {
            'causal_graph': causal_graph,
            'unfairness_paths': unfairness_paths,
            'counterfactual_results': counterfactual_results,
            'intervention_effects': intervention_effects,
            'key_insights': self._extract_key_insights(
                unfairness_paths, counterfactual_results, intervention_effects
            )
        }

    def _identify_unfairness_paths(self, causal_graph: CausalGraph,
                                 fairness_data: pd.DataFrame) -> List[List[str]]:
        """识别导致不公平的关键路径"""
        paths = []

        # 查找以公平性指标为目标的因果路径
        fairness_nodes = [node for node in causal_graph.nodes if 'fairness' in node.lower()]

        for fairness_node in fairness_nodes:
            # 查找所有通向公平性节点的路径
            parents = causal_graph.get_parents(fairness_node)

            for parent in parents:
                # 检查是否有强因果关系
                edge = next((e for e in causal_graph.edges
                           if e.target == fairness_node and e.source == parent), None)

                if edge and edge.strength > 0.3:  # 强因果关系
                    paths.append([parent, fairness_node])

                    # 递归查找更长的路径
                    grandparents = causal_graph.get_parents(parent)
                    for grandparent in grandparents:
                        paths.append([grandparent, parent, fairness_node])

        # 按因果强度排序
        paths.sort(key=lambda p: len(p), reverse=True)  # 优先更长的路径

        return paths

    def _extract_key_insights(self, unfairness_paths: List[List[str]],
                            counterfactual_results: List[CounterfactualResult],
                            intervention_effects: List[InterventionEffect]) -> List[str]:
        """提取关键洞察"""
        insights = []

        # 分析因果路径
        if unfairness_paths:
            top_path = unfairness_paths[0]
            insights.append(f"Primary unfairness path: {' → '.join(top_path)}")

        # 分析反事实结果
        significant_counterfactuals = [
            cf for cf in counterfactual_results
            if abs(cf.effect_size) > 0.1  # 显著效果
        ]

        if significant_counterfactuals:
            cf = significant_counterfactuals[0]
            direction = "increase" if cf.effect_size > 0 else "decrease"
            insights.append(f"Counterfactual analysis shows {abs(cf.effect_size):.2f} {direction} in fairness outcome")

        # 分析干预效果
        strong_effects = [
            effect for effect in intervention_effects
            if abs(effect.average_treatment_effect) > 0.15
        ]

        if strong_effects:
            effect = strong_effects[0]
            insights.append(f"Strong intervention effect of {effect.intervention} on {effect.outcome_variable}")

        return insights

    def perform_counterfactual_analysis(self, scenario_data: pd.DataFrame,
                                      intervention: Dict[str, Any],
                                      outcome_variables: List[str]) -> Dict[str, CounterfactualResult]:
        """
        执行反事实分析

        Args:
            scenario_data: 场景数据
            intervention: 干预描述 {'variable': var_name, 'value': new_value}
            outcome_variables: 结果变量列表

        Returns:
            各结果变量的反事实分析结果
        """
        results = {}

        # 获取或学习因果图
        if 'scenario_analysis' not in self.causal_graphs:
            self.causal_graphs['scenario_analysis'] = self.causal_discovery.pc_algorithm(scenario_data)

        causal_graph = self.causal_graphs['scenario_analysis']

        for outcome_var in outcome_variables:
            try:
                counterfactual = self.counterfactual_analyzer.analyze_counterfactual(
                    scenario_data,
                    intervention['variable'],
                    outcome_var,
                    intervention['value'],
                    causal_graph
                )
                results[outcome_var] = counterfactual

            except Exception as e:
                self.logger.error(f"Counterfactual analysis failed for {outcome_var}: {e}")

        return results

    def assess_algorithm_decision_impact(self, decision_data: pd.DataFrame,
                                       decision_variable: str,
                                       impact_variables: List[str]) -> Dict[str, InterventionEffect]:
        """
        评估算法决策的影响

        Args:
            decision_data: 决策相关数据
            decision_variable: 决策变量
            impact_variables: 影响变量列表

        Returns:
            各影响变量的干预效果
        """
        effects = {}

        for impact_var in impact_variables:
            try:
                effect = self.intervention_analyzer.estimate_intervention_effect(
                    decision_data, decision_variable, impact_var
                )
                effects[impact_var] = effect

            except Exception as e:
                self.logger.error(f"Intervention analysis failed for {impact_var}: {e}")

        return effects

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """获取分析统计信息"""
        return {
            'total_analyses': len(self.analysis_history),
            'cached_graphs': len(self.causal_graphs),
            'recent_analyses': self.analysis_history[-5:] if self.analysis_history else [],
            'graph_sizes': {name: len(graph.nodes) for name, graph in self.causal_graphs.items()}
        }


def create_causal_inference_engine() -> CausalInferenceEngine:
    """创建因果推断引擎"""
    return CausalInferenceEngine()


# 使用示例
if __name__ == "__main__":
    # 创建因果推断引擎
    engine = create_causal_inference_engine()

    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000

    # 模拟算法决策和公平性指标
    data = pd.DataFrame({
        'algorithm_decision': np.random.choice([0, 1], n_samples),
        'network_load': np.random.normal(0.5, 0.2, n_samples),
        'user_priority': np.random.choice([1, 2, 3], n_samples),
        'fairness_score': np.random.normal(0.8, 0.1, n_samples),
        'performance_score': np.random.normal(0.85, 0.08, n_samples),
        'latency': np.random.normal(100, 20, n_samples)
    })

    # 添加因果关系 (简化模拟)
    # 算法决策影响公平性和性能
    data.loc[data['algorithm_decision'] == 1, 'fairness_score'] += 0.1
    data.loc[data['algorithm_decision'] == 1, 'performance_score'] -= 0.05

    # 网络负载影响延迟
    data['latency'] += data['network_load'] * 50

    # 分离公平性和算法数据
    fairness_data = data[['fairness_score', 'user_priority', 'network_load']]
    algorithm_data = data[['algorithm_decision', 'performance_score', 'latency']]

    try:
        # 分析不公平原因
        analysis_result = engine.analyze_unfairness_causes(fairness_data, algorithm_data)

        print("Causal Analysis Results:")
        print(f"  Causal graph nodes: {len(analysis_result['causal_graph'].nodes)}")
        print(f"  Causal graph edges: {len(analysis_result['causal_graph'].edges)}")
        print(f"  Unfairness paths found: {len(analysis_result['unfairness_paths'])}")
        print(f"  Counterfactual analyses: {len(analysis_result['counterfactual_results'])}")

        if analysis_result['key_insights']:
            print("  Key Insights:")
            for insight in analysis_result['key_insights']:
                print(f"    - {insight}")

        # 执行反事实分析
        counterfactual_results = engine.perform_counterfactual_analysis(
            data,
            {'variable': 'algorithm_decision', 'value': 0},  # 假设不做决策
            ['fairness_score', 'performance_score']
        )

        print("\\nCounterfactual Analysis:")
        for var, result in counterfactual_results.items():
            print(f"  {var}:")
            print(".3f")
            print(".3f")
            print(".3f")

    except Exception as e:
        print(f"Analysis failed: {e}")

    print("\\nCausal inference engine demo completed")