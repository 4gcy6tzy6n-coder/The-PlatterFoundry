#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
决策复盘系统 - 用于功平性-因果推断集成实验

核心功能：
1. 捕获算法决策的完整上下文
2. 重建决策推理链
3. 识别不公平行为模式
4. 生成复盘报告
5. 提供改进建议
"""

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class DecisionImpact:
    """决策影响分析"""

    immediate_effects: Dict[str, float]
    short_term_effects: Dict[str, float]
    medium_term_effects: Dict[str, float]
    long_term_effects: Dict[str, float]
    side_effects: List[str]
    risk_assessment: str
    user_impact: str

    def to_dict(self) -> Dict[str, Any]:
        """转为字典格式"""
        return dataclasses.asdict(self)


class DecisionType(Enum):
    """决策类型枚举"""

    ROUTING = "routing"  # 路由决策
    CONGESTION = "congestion"  # 拥塞控制
    RATE_CONTROL = "rate_control"  # 速率控制
    FAIRNESS = "fairness"  # 公平性决策
    FAIRNESS_ALLOCATION = "fairness_allocation"  # 公平性分配
    ERROR_RECOVERY = "error_recovery"  # 错误 recovery
    SCHEDULING = "scheduling"  # 调度决策
    RESOURCE_ALLOCATION = "resource_allocation"  # 资源分配
    LOAD_BALANCING = "load_balancing"  # 负载均衡


@dataclass
class DecisionContext:
    """决策上下文 - 完整记录决策时的环境状态"""

    timestamp: datetime
    algorithm_version: str
    decision_type: DecisionType
    network_conditions: Dict[str, Any]
    system_load: float
    queue_lengths: Dict[str, int]
    current_throughput: float
    current_latency: float
    current_fairness: float
    recent_decisions: List[Dict[str, Any]]
    performance_trend: List[float]
    fairness_trend: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """转为字典格式"""
        return {
            "timestamp": self.timestamp.isoformat()
            if isinstance(self.timestamp, datetime)
            else str(self.timestamp),
            "algorithm_version": self.algorithm_version,
            "decision_type": self.decision_type.value,
            "network_conditions": self.network_conditions,
            "system_load": self.system_load,
            "queue_lengths": self.queue_lengths,
            "current_throughput": self.current_throughput,
            "current_latency": self.current_latency,
            "current_fairness": self.current_fairness,
            "recent_decisions": self.recent_decisions,
            "performance_trend": self.performance_trend,
            "fairness_trend": self.fairness_trend,
        }


@dataclass
class DecisionFactors:
    """决策影响因素 - 量化各因素对决策的影响程度"""

    primary_trigger: str
    contributing_factors: List[str]
    weights: Dict[str, float]
    thresholds: Dict[str, float]
    alternatives_considered: List[str]

    # 公平性相关因素 (kept for compatibility)
    fairness_weight: float = 0.0
    priority_factor: float = 0.0

    # 效率相关因素
    throughput_urgency: float = 0.0
    latency_sensitivity: float = 0.0

    # 网络条件相关
    congestion_level: float = 0.0
    loss_rate: float = 0.0

    # 历史因素
    historical_performance: float = 0.0
    stability_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转为字典格式"""
        result = dataclasses.asdict(self)

        # Ensure all required fields exist
        result["primary_trigger"] = result.get("primary_trigger", "")
        result["contributing_factors"] = result.get("contributing_factors", [])
        result["weights"] = result.get("weights", {})
        result["thresholds"] = result.get("thresholds", {})
        result["alternatives_considered"] = result.get("alternatives_considered", [])

        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DecisionFactors":
        """从字典创建"""
        # Ensure required fields exist
        data["primary_trigger"] = data.get("primary_trigger", "")
        data["contributing_factors"] = data.get("contributing_factors", [])
        data["weights"] = data.get("weights", {})
        data["thresholds"] = data.get("thresholds", {})
        data["alternatives_considered"] = data.get("alternatives_considered", [])
        return DecisionFactors(**data)


@dataclass
class DecisionRecord:
    """决策记录 - 完整的决策元数据"""

    context: DecisionContext
    factors: DecisionFactors
    chosen_action: str
    alternative_actions: List[str]
    confidence: float
    expected_outcome: Dict[str, float]
    decision_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def id(self) -> str:
        return self.decision_id

    def to_dict(self) -> Dict[str, Any]:
        """转为字典格式"""
        return {
            "context": self.context.to_dict(),
            "factors": self.factors.to_dict(),
            "chosen_action": self.chosen_action,
            "alternative_actions": self.alternative_actions,
            "confidence": self.confidence,
            "expected_outcome": self.expected_outcome,
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat()
            if isinstance(self.timestamp, datetime)
            else str(self.timestamp),
        }


@dataclass
class InequalityAnalysis:
    """不公平性分析结果"""

    affected_users: List[str]
    inequality_type: str
    severity_score: float
    root_cause: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class DecisionReplaySystem:
    """
    决策复盘系统

    功能：
    1. 记录和存储决策历史
    2. 重建决策推理链
    3. 分析决策公平性
    4. 识别改进机会
    """

    def __init__(self):
        self.decision_history: List[DecisionRecord] = []
        self.inequality_analysis_results: List[InequalityAnalysis] = []
        self.improvement_suggestions: List[Dict[str, Any]] = []

        # 性能统计
        self.total_decisions: int = 0
        self.unequal_decisions: int = 0

    def record_decision(
        self,
        context: DecisionContext,
        factors: DecisionFactors,
        chosen_action: str,
        alternative_actions: List[str],
        confidence: float,
        expected_outcome: Dict[str, float],
        decision_id: str = "",
        timestamp: datetime = None,
    ) -> str:
        """记录单个决策"""

        if timestamp is None:
            timestamp = datetime.now()

        if not decision_id:
            decision_id = f"decision_{self.total_decisions}"

        decision_record = DecisionRecord(
            context=context,
            factors=factors,
            chosen_action=chosen_action,
            alternative_actions=alternative_actions,
            confidence=confidence,
            expected_outcome=expected_outcome,
            decision_id=decision_id,
            timestamp=timestamp,
        )

        self.decision_history.append(decision_record)
        self.total_decisions += 1

        return decision_id

    def start_decision_recording(
        self, recording_name: str, context: DecisionContext, factors: DecisionFactors
    ) -> str:
        """启动决策记录流程（兼容API）"""
        if context.recent_decisions:
            primary_action = context.recent_decisions[0]["action"]
        else:
            primary_action = "unknown"

        decision_id = f"rec_{recording_name}_{self.total_decisions}"

        self.record_decision(
            context=context,
            factors=factors,
            chosen_action=primary_action,
            alternative_actions=factors.alternatives_considered,
            confidence=0.8,
            expected_outcome={
                "fairness": context.current_fairness,
                "throughput": context.current_throughput,
                "latency": context.current_latency,
            },
            decision_id=decision_id,
            timestamp=context.timestamp,
        )

        return decision_id

    def complete_decision_recording(
        self,
        recording_id: str,
        chosen_action: str,
        reasoning_chain: List[Dict[str, Any]],
        expected_outcomes: Dict[str, float],
        actual_outcomes: DecisionImpact,
    ) -> Dict[str, Any]:
        """完成决策记录并生成回放数据

        Args:
            recording_id: 录制ID
            chosen_action: 选择的行动
            reasoning_chain: 推理链
            expected_outcomes: 预期结果
            actual_outcomes: 实际结果

        Returns:
            回放数据字典
        """
        # Find the recording
        recording = self.get_full_recording(recording_id)
        if not recording:
            return {"error": "Recording not found"}

        # Generate replay data
        replay_data = {
            "recording_id": recording_id,
            "context": recording.get("context", {}),
            "factors": recording.get("factors", {}),
            "chosen_action": chosen_action,
            "reasoning_chain": reasoning_chain,
            "expected_outcomes": expected_outcomes,
            "actual_outcomes": actual_outcomes.to_dict()
            if hasattr(actual_outcomes, "to_dict")
            else actual_outcomes,
            "replay_timestamp": datetime.now().isoformat(),
        }

        return replay_data

    def generate_replay_report(self, recording_id: str, format: str = "text") -> str:
        """生成复盘报告

        Args:
            recording_id: 录制ID
            format: 报告格式 ("text" 或 "json")

        Returns:
            复盘报告内容
        """
        recording = self.get_full_recording(recording_id)
        if not recording:
            return f"Recording {recording_id} not found"

        if format == "json":
            return json.dumps(recording, indent=2, ensure_ascii=False, default=str)
        elif format == "text":
            # 生成文本格式报告
            context = recording.get("context", {})
            factors = recording.get("factors", {})

            lines = []
            lines.append("=" * 60)
            lines.append("决策复盘报告")
            lines.append("=" * 60)
            lines.append(f"录制ID: {recording_id}")
            lines.append(f"算法版本: {context.get('algorithm_version', 'unknown')}")
            lines.append(f"决策类型: {context.get('decision_type', 'unknown')}")
            lines.append(f"时间戳: {context.get('timestamp', 'unknown')}")
            lines.append("")
            lines.append("网络条件:")
            for key, value in context.get("network_conditions", {}).items():
                lines.append(f"  {key}: {value}")
            lines.append("")
            lines.append("系统负载:")
            lines.append(f"  当前吞吐量: {context.get('current_throughput', 0)}")
            lines.append(f"  当前延迟: {context.get('current_latency', 0)}")
            lines.append(f"  当前公平性: {context.get('current_fairness', 0)}")
            lines.append(f"  系统负载: {context.get('system_load', 0)}")
            lines.append("")
            lines.append("决策因素:")
            if factors.get("primary_trigger"):
                lines.append(f"  主要触发: {factors['primary_trigger']}")
            lines.append("  关键指标权重:")
            weights = factors.get("weights", {})
            for key, value in weights.items():
                lines.append(f"    {key}: {value}")
            lines.append("")
            lines.append("选择的行动:")
            lines.append(f"  {recording.get('chosen_action', 'unknown')}")
            lines.append("")

            if recording.get("reasoning_chain"):
                lines.append("推理链:")
                for i, step in enumerate(recording.get("reasoning_chain", []), 1):
                    lines.append(f"  步骤{i}: {step.get('reason', 'unknown')}")
                    lines.append(f"    依据: {step.get('evidence', 'N/A')}")
            lines.append("")
            lines.append("=" * 60)

            return "\n".join(lines)
        else:
            return f"Unsupported format: {format}"

    def get_full_recording(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """获取完整录制的决策数据"""
        for decision in self.decision_history:
            if decision.decision_id == recording_id:
                return decision.to_dict()
        return None

    def record_decisions_batch(self, decisions: List[Dict[str, Any]]) -> int:
        """批量记录决策"""
        recorded_count = 0

        for decision_data in decisions:
            try:
                # 解析决策数据
                context = DecisionContext(
                    timestamp=decision_data.get("timestamp", datetime.now()),
                    algorithm_version=decision_data.get("algorithm_version", "unknown"),
                    decision_type=DecisionType(decision_data["decision_type"]),
                    network_conditions=decision_data.get("network_conditions", {}),
                    system_load=decision_data.get("system_load", 0.0),
                    queue_lengths=decision_data.get("queue_lengths", {}),
                    current_throughput=decision_data.get("current_throughput", 0.0),
                    current_latency=decision_data.get("current_latency", 0.0),
                    current_fairness=decision_data.get("current_fairness", 0.0),
                    recent_decisions=decision_data.get("recent_decisions", []),
                    performance_trend=decision_data.get("performance_trend", []),
                    fairness_trend=decision_data.get("fairness_trend", []),
                )

                factors = DecisionFactors.from_dict(decision_data.get("factors", {}))

                # 记录决策
                self.record_decision(
                    context=context,
                    factors=factors,
                    chosen_action=decision_data.get("chosen_action", "unknown"),
                    alternative_actions=decision_data.get("alternative_actions", []),
                    confidence=decision_data.get("confidence", 0.0),
                    expected_outcome=decision_data.get("expected_outcome", {}),
                    decision_id=decision_data.get("decision_id", ""),
                    timestamp=decision_data.get("timestamp", datetime.now()),
                )

                recorded_count += 1

            except Exception as e:
                print(f"记录决策失败: {type(e).__name__}: {e}")
                continue

        return recorded_count

    def analyze_fairness_violations(self) -> List[InequalityAnalysis]:
        """分析公平性违反情况"""

        violation_examples = []

        # 1. 基于优先级因子的不公平性分析
        # 检测是否有用户长期处于低优先级
        priority_violations = self._analyze_priority_inequality()
        violation_examples.extend(priority_violations)

        # 2. 基于网络状态的不公平性分析
        # 检测算法是否对特定网络条件的用户不公平
        condition_violations = self._analyze_condition_inequality()
        violation_examples.extend(condition_violations)

        # 3. 基于历史性能的不公平性分析
        # 检测是否有"富者愈富"的不公平模式
        history_violations = self._analyze_history_inequality()
        violation_examples.extend(history_violations)

        self.inequality_analysis_results = violation_examples
        self.unequal_decisions = len(violation_examples)

        return violation_examples

    def _analyze_priority_inequality(self) -> List[InequalityAnalysis]:
        """分析优先级导致的不公平性"""

        results = []

        # 提取优先级相关决策
        priority_decisions = [
            d
            for d in self.decision_history
            if d.factors.priority_factor > 0.8  # 高优先级决策
        ]

        if len(priority_decisions) < 5:
            return results

        # 计算平均优先级
        priority_values = [d.factors.priority_factor for d in priority_decisions]
        avg_priority = np.mean(priority_values)
        std_priority = np.std(priority_values)

        # 检测异常高优先级（可能意味着不公平）
        high_priority_threshold = avg_priority + 2 * std_priority

        for decision in priority_decisions:
            if decision.factors.priority_factor > high_priority_threshold:
                analysis = InequalityAnalysis(
                    affected_users=[
                        decision.context.context_data.get("stream_id", "unknown")
                    ],
                    inequality_type="excessive_priority",
                    severity_score=min(1.0, decision.factors.priority_factor),
                    root_cause=[
                        "Priority factor heavily biased toward specific flow",
                        "High priority without corresponding need verification",
                        "Potential resource monopolization risk",
                    ],
                )
                results.append(analysis)

        return results

    def _analyze_condition_inequality(self) -> List[InequalityAnalysis]:
        """分析网络条件导致的不公平性"""

        results = []

        # 按网络质量分组决策
        good_network = []
        poor_network = []

        for decision in self.decision_history:
            loss = decision.context.network_state.get("loss_rate", 0.0)
            latency = decision.context.network_state.get("latency_ms", 0.0)

            # 简单的网络质量评估
            if loss < 0.05 and latency < 100:
                good_network.append(decision)
            else:
                poor_network.append(decision)

        # 分析决策差异
        if good_network and poor_network:
            good_fairness = np.mean([d.factors.fairness_weight for d in good_network])
            poor_fairness = np.mean([d.factors.fairness_weight for d in poor_network])

            fairness_gap = abs(good_fairness - poor_fairness)

            if fairness_gap > 0.3:  # 显著差异
                # 找出最劣势的决策
                worst_poor = min(poor_network, key=lambda d: d.factors.fairness_weight)

                analysis = InequalityAnalysis(
                    affected_users=[
                        worst_poor.context.context_data.get("stream_id", "unknown")
                    ],
                    inequality_type="network_condition_bias",
                    severity_score=fairness_gap,
                    root_cause=[
                        f"Fairness gap: {good_fairness:.3f} vs {poor_fairness:.3f}",
                        "Algorithm may be biased against poor network conditions",
                        "Insufficient fairness compensation for disadvantaged users",
                    ],
                )
                results.append(analysis)

        return results

    def _analyze_history_inequality(self) -> List[InequalityAnalysis]:
        """分析历史性能导致的不公平性"""

        results = []

        # 检测历史依赖模式
        historical_correlation = []

        for decision in self.decision_history:
            historical_perf = decision.factors.historical_performance
            fairness_weight = decision.factors.fairness_weight

            # 检测"富者愈富"模式：历史表现好的获得更高公平权重
            historical_correlation.append((historical_perf, fairness_weight))

        if len(historical_correlation) >= 10:
            hists, fair = zip(*historical_correlation)

            # 计算相关性
            correlation = np.corrcoef(hists, fair)[0, 1]

            # 如果高度正相关，可能存在"富者愈富"问题
            if correlation > 0.6:
                # 找出相关系数最大的决策（最典型的不公平案例）
                worst_case = max(
                    self.decision_history,
                    key=lambda d: d.factors.historical_performance,
                )

                analysis = InequalityAnalysis(
                    affected_users=[
                        worst_case.context.context_data.get("stream_id", "unknown")
                    ],
                    inequality_type="historical_performane_bias",
                    severity_score=min(1.0, float(correlation)),
                    root_cause=[
                        f"Historical bias correlation: {correlation:.3f}",
                        "Algorithm overly favors historically good performers",
                        "New or struggling streams denied opportunities",
                    ],
                )
                results.append(analysis)

        return results

    def replay_decision_sequence(self, start_index: int, count: int) -> Dict[str, Any]:
        """重放决策序列，重建推理链"""

        if start_index < 0 or start_index >= len(self.decision_history):
            return {"error": "Invalid start index"}

        end_index = min(start_index + count, len(self.decision_history))

        replay_data = {
            "sequence_info": {
                "start_index": start_index,
                "end_index": end_index - 1,
                "num_decisions": end_index - start_index,
                "time_range": {
                    "start": self.decision_history[start_index].context.timestamp,
                    "end": self.decision_history[end_index - 1].context.timestamp,
                },
            },
            "decisions": [],
        }

        for i in range(start_index, end_index):
            decision = self.decision_history[i]

            # 分析这个决策的关键影响因素
            decision_analysis = {
                "timestamp": decision.context.timestamp,
                "decision_type": decision.context.decision_type.value,
                "decision_id": decision.context.decision_id,
                "chosen_action": decision.chosen_action,
                "confidence": decision.confidence,
                "key_factors": self._extract_key_factors(decision),
                "context_summary": self._summarize_context(decision.context),
                "alternative_options": decision.alternative_actions,
            }

            replay_data["decisions"].append(decision_analysis)

        return replay_data

    def _extract_key_factors(self, decision: DecisionRecord) -> Dict[str, float]:
        """提取决策的关键影响因素"""
        factors = decision.factors

        # Try new fields first
        if factors.weights:
            return {k: float(v) for k, v in factors.weights.items()}

        # Fall back to old fields
        factor_weights = {
            "fairness_weight": factors.fairness_weight * 1.2,
            "priority_factor": factors.priority_factor * 1.1,
            "congestion_level": factors.congestion_level * 1.0,
            "throughput_urgency": factors.throughput_urgency * 0.9,
            "latency_sensitivity": factors.latency_sensitivity * 0.8,
        }

        # 按权重排序
        sorted_factors = sorted(
            factor_weights.items(), key=lambda x: abs(x[1]), reverse=True
        )

        return {name: float(weight) for name, weight in sorted_factors[:5]}

    def _summarize_context(self, context: DecisionContext) -> Dict[str, Any]:
        """总结决策上下文"""
        return {
            "network_conditions": {
                "loss_rate": context.network_state.get("loss_rate", 0.0),
                "latency": context.network_state.get("latency_ms", 0.0),
                "bandwidth": context.network_state.get("bandwidth_mbps", 0.0),
            },
            "decision_reason": context.context_data.get("reason", "unknown"),
        }

    def identify_improvement_areas(self) -> List[Dict[str, Any]]:
        """识别改进领域"""

        suggestions = []

        # 分析公平性违反
        inequality_issues = self.analyze_fairness_violations()

        # 生成改进策略
        for issue in inequality_issues:
            suggestion = {
                "issue_type": issue.inequality_type,
                "severity": issue.severity_score,
                "specific_instances": len(issue.affected_users),
                "root_causes": issue.root_cause,
                "suggested_actions": [],
            }

            # 根据问题类型生成改进建议
            if issue.inequality_type == "excessive_priority":
                suggestion["suggested_actions"].extend(
                    [
                        "Implement priority capping mechanism",
                        "Add priority verification checks",
                        "Use adaptive priority normalization",
                    ]
                )

            elif issue.inequality_type == "network_condition_bias":
                suggestion["suggested_actions"].extend(
                    [
                        "Add fairness compensation for poor conditions",
                        "Implement condition-aware fairness metrics",
                        "Use robust statistical methods for fairness",
                    ]
                )

            elif issue.inequality_type == "historical_performane_bias":
                suggestion["suggested_actions"].extend(
                    [
                        "Implement equal opportunity incentive",
                        "Use decayed historical performance",
                        "Add historical bias correction factor",
                    ]
                )

            suggestions.append(suggestion)

        self.improvement_suggestions = suggestions
        return suggestions

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合复盘报告"""

        # 进行各项分析
        inequality_analysis = self.analyze_fairness_violations()
        improvement_areas = self.identify_improvement_areas()

        # 计算统计数据
        stats = {
            "total_decisions": self.total_decisions,
            "unequal_decisions": self.unequal_decisions,
            "inequality_rate": self.unequal_decisions / max(1, self.total_decisions),
            "avg_priority_factor": np.mean(
                [d.factors.priority_factor for d in self.decision_history]
            )
            if self.decision_history
            else 0.0,
            "avg_fairness_weight": np.mean(
                [d.factors.fairness_weight for d in self.decision_history]
            )
            if self.decision_history
            else 0.0,
        }

        # 生成洞察
        insights = []

        if stats["inequality_rate"] > 0.3:
            insights.append("High level of unfairness detected in decision patterns")

        if stats["avg_fairness_weight"] < 0.5:
            insights.append("Fairness weighting may be insufficient")

        if improvement_areas:
            insights.append(f"Found {len(improvement_areas)} key areas for improvement")

        # 构建报告
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_decisions_analyzed": self.total_decisions,
                "analysis_depth": "comprehensive",
            },
            "statistics": stats,
            "inequality_analysis": [ia.to_dict() for ia in inequality_analysis],
            "improvement_suggestions": improvement_areas,
            "replay_data_available": len(self.decision_history) > 0,
            "insights": insights,
            "overall_assessment": self._get_overall_assessment(
                stats, inequality_analysis
            ),
        }

        return report

    def _get_overall_assessment(
        self, stats: Dict[str, Any], inequalities: List[InequalityAnalysis]
    ) -> Dict[str, Any]:
        """获取整体评估"""

        # 计算公平性评分
        inequality_score = stats["inequality_rate"] * 100
        bias_severity = (
            np.mean([i.severity_score for i in inequalities]) if inequalities else 0.0
        )

        overall_score = max(0, 100 - inequality_score * 100 - bias_severity * 20)

        if overall_score >= 80:
            assessment = "Excellent"
            color = "green"
        elif overall_score >= 60:
            assessment = "Good"
            color = "yellow"
        elif overall_score >= 40:
            assessment = "Fair"
            color = "orange"
        else:
            assessment = "Poor"
            color = "red"

        return {
            "score": float(overall_score),
            "assessment": assessment,
            "color": color,
            "recommendation": "Focus on reducing bias in fairness weighting"
            if overall_score < 70
            else "Continue monitoring and periodic review",
        }

    def save_results(self, output_path: str) -> bool:
        """保存复盘结果到文件"""

        try:
            # 生成报告
            report = self.generate_comprehensive_report()

            # 添加详细决策数据
            report["decision_samples"] = [
                d.to_dict()
                for d in self.decision_history[:100]  # Limit to first 100
            ]

            # 保存到文件
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

            return True

        except Exception as e:
            print(f"保存结果失败: {type(e).__name__}: {e}")
            return False

    def reset(self) -> None:
        """重置系统状态"""
        self.decision_history.clear()
        self.inequality_analysis_results.clear()
        self.improvement_suggestions.clear()
        self.total_decisions = 0
        self.unequal_decisions = 0


class MockDecisionReplaySystem(DecisionReplaySystem):
    """
    模拟决策复盘系统 - 用于测试和演示
    """

    def __init__(self):
        super().__init__()

    def generate_mock_decisions(self, num_decisions: int = 50) -> None:
        """生成模拟决策数据"""

        import random

        for i in range(num_decisions):
            timestamp = float(i * 10)  # 每10秒一个决策

            # 随机决定决策类型
            decision_type = random.choice(list(DecisionType))

            # 生成网络状态
            network_state = {
                "loss_rate": random.uniform(0.0, 0.3),
                "latency_ms": random.uniform(10, 200),
                "bandwidth_mbps": random.uniform(10, 100),
                "jitter_ms": random.uniform(0, 20),
            }

            # 生成系统状态
            system_state = {
                "queue_length": random.uniform(0, 10),
                "cpu_usage": random.uniform(0.1, 0.9),
                "memory_usage": random.uniform(0.2, 0.8),
            }

            # 生成输入特征
            input_features = {
                "flow_priority": random.uniform(0, 1),
                "historical_throughput": random.uniform(5, 50),
                "packet_size": random.uniform(500, 1500),
            }

            # 生成上下文数据
            context_data = {
                "stream_id": f"stream_{random.randint(1, 5)}",
                "source_ip": "192.168.1.1",
                "dest_ip": "10.0.0.1",
                "reason": "congestion_avoidance"
                if random.random() > 0.5
                else "fairness_optimization",
            }

            # 创建决策上下文
            context = DecisionContext(
                timestamp=timestamp,
                decision_type=decision_type,
                decision_id=f"decision_{i}",
                network_state=network_state,
                system_state=system_state,
                input_features=input_features,
                context_data=context_data,
            )

            # 创建决策因素（引入一些不公平性用于测试）
            factors = DecisionFactors(
                fairness_weight=0.3 + random.random() * 0.7,  # 推成不均
                priority_factor=0.2 + random.random() * 0.8,
                congestion_level=network_state["loss_rate"],
                throughput_urgency=random.uniform(0, 1),
                latency_sensitivity=random.uniform(0, 1),
                loss_rate=network_state["loss_rate"],
                historical_performance=random.uniform(0, 1),
                stability_score=random.uniform(0, 1),
            )

            # 选择行动
            action_options = [
                "rate_reduce",
                "rate_increase",
                "fairness_adjust",
                "queue_prioritize",
            ]
            chosen_action = random.choice(action_options)
            alternative_actions = [a for a in action_options if a != chosen_action][:2]

            # 生成预期结果
            expected_outcome = {
                "fairness_change": random.uniform(-0.2, 0.2),
                "throughput_change": random.uniform(-0.1, 0.1),
                "latency_change": random.uniform(-10, 10),
            }

            # 记录决策
            self.record_decision(
                context=context,
                factors=factors,
                chosen_action=chosen_action,
                alternative_actions=alternative_actions,
                confidence=0.8 + random.random() * 0.2,
                expected_outcome=expected_outcome,
            )


__all__ = [
    "DecisionReplaySystem",
    "MockDecisionReplaySystem",
    "DecisionContext",
    "DecisionFactors",
    "DecisionRecord",
    "InequalityAnalysis",
    "DecisionType",
    "DecisionImpact",
]
