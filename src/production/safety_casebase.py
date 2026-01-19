#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Safety Casebase - 安全案例库

创新点：
1. 智能案例归档：自动分类和标注异常事件
2. 模式挖掘：识别异常发生的规律和趋势
3. 反例教学：为算法开发者提供具体案例学习
4. 风险预测：基于历史案例预测未来风险
5. 持续学习：从案例中提取改进建议
6. 知识图谱：构建安全知识的关联网络

这解决了传统日志记录的被动性，实现真正智能的安全知识管理。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
from pathlib import Path


class CaseSeverity(Enum):
    """案例严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CaseCategory(Enum):
    """案例分类"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    FAIRNESS_VIOLATION = "fairness_violation"
    STABILITY_ISSUE = "stability_issue"
    SECURITY_VULNERABILITY = "security_vulnerability"
    RESOURCE_LEAK = "resource_leak"
    CONFIGURATION_ERROR = "configuration_error"
    EXTERNAL_ATTACK = "external_attack"
    UNKNOWN = "unknown"


@dataclass
class SafetyCase:
    """安全案例"""
    case_id: str
    timestamp: datetime
    algorithm_version: str
    category: CaseCategory
    severity: CaseSeverity

    # 事件描述
    description: str
    trigger_conditions: Dict[str, Any]
    symptoms: List[str]

    # 根本原因分析
    root_cause: str
    contributing_factors: List[str]

    # 影响评估
    impact_assessment: Dict[str, Any]
    affected_users: int
    downtime_duration: float  # 分钟

    # 恢复措施
    recovery_actions: List[str]
    recovery_success: bool
    lessons_learned: List[str]

    # 元数据
    detection_method: str
    resolution_time: float  # 分钟
    prevention_measures: List[str]

    # 关联案例
    related_cases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class PatternInsight:
    """模式洞察"""
    pattern_id: str
    pattern_type: str  # 'temporal', 'algorithmic', 'environmental'
    description: str
    affected_cases: List[str]
    frequency: int
    trend: str  # 'increasing', 'decreasing', 'stable'
    risk_score: float
    recommendations: List[str]


@dataclass
class RiskPrediction:
    """风险预测"""
    prediction_id: str
    algorithm_version: str
    predicted_risks: List[Dict[str, Any]]
    confidence_level: float
    time_horizon: str  # 'short_term', 'medium_term', 'long_term'
    mitigation_strategies: List[str]


class CaseArchiver:
    """案例归档器"""

    def __init__(self, db_path: str = "data/safety_casebase.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.logger = logging.getLogger("CaseArchiver")

        # 初始化数据库
        self._initialize_database()

    def _initialize_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 创建案例表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS safety_cases (
                    case_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    algorithm_version TEXT,
                    category TEXT,
                    severity TEXT,
                    description TEXT,
                    trigger_conditions TEXT,
                    symptoms TEXT,
                    root_cause TEXT,
                    contributing_factors TEXT,
                    impact_assessment TEXT,
                    affected_users INTEGER,
                    downtime_duration REAL,
                    recovery_actions TEXT,
                    recovery_success BOOLEAN,
                    lessons_learned TEXT,
                    detection_method TEXT,
                    resolution_time REAL,
                    prevention_measures TEXT,
                    related_cases TEXT,
                    tags TEXT
                )
            ''')

            # 创建模式洞察表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_insights (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    description TEXT,
                    affected_cases TEXT,
                    frequency INTEGER,
                    trend TEXT,
                    risk_score REAL,
                    recommendations TEXT,
                    last_updated TEXT
                )
            ''')

            # 创建风险预测表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_predictions (
                    prediction_id TEXT PRIMARY KEY,
                    algorithm_version TEXT,
                    predicted_risks TEXT,
                    confidence_level REAL,
                    time_horizon TEXT,
                    mitigation_strategies TEXT,
                    created_at TEXT
                )
            ''')

            conn.commit()

    def archive_case(self, safety_case: SafetyCase):
        """归档安全案例"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO safety_cases
                (case_id, timestamp, algorithm_version, category, severity, description,
                 trigger_conditions, symptoms, root_cause, contributing_factors,
                 impact_assessment, affected_users, downtime_duration, recovery_actions,
                 recovery_success, lessons_learned, detection_method, resolution_time,
                 prevention_measures, related_cases, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                safety_case.case_id,
                safety_case.timestamp.isoformat(),
                safety_case.algorithm_version,
                safety_case.category.value,
                safety_case.severity.value,
                safety_case.description,
                json.dumps(safety_case.trigger_conditions),
                json.dumps(safety_case.symptoms),
                safety_case.root_cause,
                json.dumps(safety_case.contributing_factors),
                json.dumps(safety_case.impact_assessment),
                safety_case.affected_users,
                safety_case.downtime_duration,
                json.dumps(safety_case.recovery_actions),
                safety_case.recovery_success,
                json.dumps(safety_case.lessons_learned),
                safety_case.detection_method,
                safety_case.resolution_time,
                json.dumps(safety_case.prevention_measures),
                json.dumps(safety_case.related_cases),
                json.dumps(safety_case.tags)
            ))

            conn.commit()

        self.logger.info(f"Safety case archived: {safety_case.case_id}")

    def retrieve_case(self, case_id: str) -> Optional[SafetyCase]:
        """检索案例"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM safety_cases WHERE case_id = ?', (case_id,))
            row = cursor.fetchone()

            if row:
                return self._row_to_safety_case(row)

        return None

    def search_cases(self, filters: Dict[str, Any]) -> List[SafetyCase]:
        """搜索案例"""
        query = "SELECT * FROM safety_cases WHERE 1=1"
        params = []

        if 'category' in filters:
            query += " AND category = ?"
            params.append(filters['category'].value)

        if 'severity' in filters:
            query += " AND severity = ?"
            params.append(filters['severity'].value)

        if 'algorithm_version' in filters:
            query += " AND algorithm_version = ?"
            params.append(filters['algorithm_version'])

        if 'date_from' in filters:
            query += " AND timestamp >= ?"
            params.append(filters['date_from'].isoformat())

        if 'date_to' in filters:
            query += " AND timestamp <= ?"
            params.append(filters['date_to'].isoformat())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            cases = []
            for row in cursor.fetchall():
                cases.append(self._row_to_safety_case(row))

        return cases

    def _row_to_safety_case(self, row) -> SafetyCase:
        """将数据库行转换为SafetyCase对象"""
        return SafetyCase(
            case_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            algorithm_version=row[2],
            category=CaseCategory(row[3]),
            severity=CaseSeverity(row[4]),
            description=row[5],
            trigger_conditions=json.loads(row[6]),
            symptoms=json.loads(row[7]),
            root_cause=row[8],
            contributing_factors=json.loads(row[9]),
            impact_assessment=json.loads(row[10]),
            affected_users=row[11],
            downtime_duration=row[12],
            recovery_actions=json.loads(row[13]),
            recovery_success=row[14],
            lessons_learned=json.loads(row[15]),
            detection_method=row[16],
            resolution_time=row[17],
            prevention_measures=json.loads(row[18]),
            related_cases=json.loads(row[19]),
            tags=json.loads(row[20])
        )


class PatternMiner:
    """模式挖掘器"""

    def __init__(self, archiver: CaseArchiver):
        self.archiver = archiver
        self.logger = logging.getLogger("PatternMiner")

    def mine_patterns(self) -> List[PatternInsight]:
        """挖掘模式"""
        # 获取所有案例
        all_cases = self.archiver.search_cases({})

        if len(all_cases) < 5:
            self.logger.warning("Insufficient cases for pattern mining")
            return []

        patterns = []

        # 时序模式
        temporal_patterns = self._mine_temporal_patterns(all_cases)
        patterns.extend(temporal_patterns)

        # 算法模式
        algorithmic_patterns = self._mine_algorithmic_patterns(all_cases)
        patterns.extend(algorithmic_patterns)

        # 环境模式
        environmental_patterns = self._mine_environmental_patterns(all_cases)
        patterns.extend(environmental_patterns)

        # 保存模式洞察
        self._save_pattern_insights(patterns)

        return patterns

    def _mine_temporal_patterns(self, cases: List[SafetyCase]) -> List[PatternInsight]:
        """挖掘时序模式"""
        patterns = []

        # 按时间排序
        sorted_cases = sorted(cases, key=lambda c: c.timestamp)

        # 计算时间窗口内的案例频率
        time_windows = [
            (datetime.now() - timedelta(days=7), "last_week"),
            (datetime.now() - timedelta(days=30), "last_month"),
            (datetime.now() - timedelta(days=90), "last_quarter")
        ]

        for cutoff_time, window_name in time_windows:
            recent_cases = [c for c in sorted_cases if c.timestamp >= cutoff_time]

            if len(recent_cases) > 2:
                # 计算严重程度分布
                severity_counts = defaultdict(int)
                for case in recent_cases:
                    severity_counts[case.severity.value] += 1

                # 检查是否有严重案例增加的趋势
                critical_cases = [c for c in recent_cases if c.severity == CaseSeverity.CRITICAL]

                if len(critical_cases) >= 2:
                    pattern = PatternInsight(
                        pattern_id=f"temporal_critical_{window_name}",
                        pattern_type="temporal",
                        description=f"Increased frequency of critical safety incidents in {window_name.replace('_', ' ')}",
                        affected_cases=[c.case_id for c in critical_cases],
                        frequency=len(critical_cases),
                        trend="increasing",
                        risk_score=min(1.0, len(critical_cases) / 5.0),
                        recommendations=[
                            "Increase monitoring frequency",
                            "Review recent algorithm changes",
                            "Consider additional safety measures"
                        ]
                    )
                    patterns.append(pattern)

        return patterns

    def _mine_algorithmic_patterns(self, cases: List[SafetyCase]) -> List[PatternInsight]:
        """挖掘算法模式"""
        patterns = []

        # 按算法版本分组
        version_groups = defaultdict(list)
        for case in cases:
            version_groups[case.algorithm_version].append(case)

        # 分析每个版本的问题模式
        for version, version_cases in version_groups.items():
            if len(version_cases) < 3:
                continue

            # 计算最常见的问题类型
            category_counts = defaultdict(int)
            for case in version_cases:
                category_counts[case.category.value] += 1

            most_common_category = max(category_counts.items(), key=lambda x: x[1])

            if most_common_category[1] >= 2:
                pattern = PatternInsight(
                    pattern_id=f"algorithmic_{version}_{most_common_category[0]}",
                    pattern_type="algorithmic",
                    description=f"Version {version} frequently experiences {most_common_category[0].replace('_', ' ')} issues",
                    affected_cases=[c.case_id for c in version_cases if c.category.value == most_common_category[0]],
                    frequency=most_common_category[1],
                    trend="stable",
                    risk_score=min(1.0, most_common_category[1] / len(version_cases)),
                    recommendations=[
                        f"Review {most_common_category[0].replace('_', ' ')} handling in version {version}",
                        "Consider algorithm redesign for this issue type",
                        f"Add specific tests for {most_common_category[0].replace('_', ' ')} scenarios"
                    ]
                )
                patterns.append(pattern)

        return patterns

    def _mine_environmental_patterns(self, cases: List[SafetyCase]) -> List[PatternInsight]:
        """挖掘环境模式"""
        patterns = []

        # 分析触发条件中的环境因素
        environmental_factors = defaultdict(int)

        for case in cases:
            trigger_conditions = case.trigger_conditions

            # 检查常见环境因素
            if 'network_load' in trigger_conditions and trigger_conditions['network_load'] > 0.8:
                environmental_factors['high_network_load'] += 1

            if 'system_load' in trigger_conditions and trigger_conditions['system_load'] > 0.8:
                environmental_factors['high_system_load'] += 1

            if 'time_of_day' in trigger_conditions:
                hour = trigger_conditions['time_of_day']
                if 9 <= hour <= 17:
                    environmental_factors['business_hours'] += 1

        # 生成环境模式洞察
        for factor, count in environmental_factors.items():
            if count >= 3:  # 至少3个案例
                pattern = PatternInsight(
                    pattern_id=f"environmental_{factor}",
                    pattern_type="environmental",
                    description=f"Safety incidents frequently occur under {factor.replace('_', ' ')} conditions",
                    affected_cases=[],  # 这里可以关联具体案例
                    frequency=count,
                    trend="stable",
                    risk_score=min(1.0, count / len(cases)),
                    recommendations=[
                        f"Strengthen monitoring during {factor.replace('_', ' ')} periods",
                        f"Implement additional safeguards for {factor.replace('_', ' ')} scenarios",
                        f"Consider load balancing to avoid {factor.replace('_', ' ')} situations"
                    ]
                )
                patterns.append(pattern)

        return patterns

    def _save_pattern_insights(self, patterns: List[PatternInsight]):
        """保存模式洞察"""
        with sqlite3.connect(self.archiver.db_path) as conn:
            cursor = conn.cursor()

            for pattern in patterns:
                cursor.execute('''
                    INSERT OR REPLACE INTO pattern_insights
                    (pattern_id, pattern_type, description, affected_cases, frequency,
                     trend, risk_score, recommendations, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.pattern_id,
                    pattern.pattern_type,
                    pattern.description,
                    json.dumps(pattern.affected_cases),
                    pattern.frequency,
                    pattern.trend,
                    pattern.risk_score,
                    json.dumps(pattern.recommendations),
                    datetime.now().isoformat()
                ))

            conn.commit()


class RiskPredictor:
    """风险预测器"""

    def __init__(self, archiver: CaseArchiver, pattern_miner: PatternMiner):
        self.archiver = archiver
        self.pattern_miner = pattern_miner
        self.logger = logging.getLogger("RiskPredictor")

    def predict_risks(self, algorithm_version: str, time_horizon: str = "medium_term") -> RiskPrediction:
        """
        预测风险

        Args:
            algorithm_version: 算法版本
            time_horizon: 时间跨度 ('short_term', 'medium_term', 'long_term')

        Returns:
            风险预测结果
        """
        # 获取历史案例
        historical_cases = self.archiver.search_cases({
            'algorithm_version': algorithm_version,
            'date_from': datetime.now() - timedelta(days=90)  # 最近90天
        })

        # 获取模式洞察
        patterns = self.pattern_miner.mine_patterns()

        # 基于历史和模式预测风险
        predicted_risks = self._analyze_risk_factors(historical_cases, patterns, time_horizon)

        # 计算置信度
        confidence_level = self._calculate_prediction_confidence(historical_cases, patterns)

        # 生成缓解策略
        mitigation_strategies = self._generate_mitigation_strategies(predicted_risks)

        prediction = RiskPrediction(
            prediction_id=f"pred_{algorithm_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            algorithm_version=algorithm_version,
            predicted_risks=predicted_risks,
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            mitigation_strategies=mitigation_strategies
        )

        # 保存预测结果
        self._save_prediction(prediction)

        return prediction

    def _analyze_risk_factors(self, cases: List[SafetyCase],
                            patterns: List[PatternInsight],
                            time_horizon: str) -> List[Dict[str, Any]]:
        """分析风险因素"""
        risk_factors = []

        # 基于历史案例的频率分析
        if cases:
            category_counts = defaultdict(int)
            severity_counts = defaultdict(int)

            for case in cases:
                category_counts[case.category.value] += 1
                severity_counts[case.severity.value] += 1

            # 识别高频风险
            for category, count in category_counts.items():
                frequency = count / len(cases)
                if frequency > 0.2:  # 20%以上
                    risk_factors.append({
                        'risk_type': 'frequent_incidents',
                        'category': category,
                        'frequency': frequency,
                        'severity': 'medium',
                        'description': f"High frequency of {category.replace('_', ' ')} incidents"
                    })

        # 基于模式洞察的风险
        for pattern in patterns:
            if pattern.risk_score > 0.3:  # 中等及以上风险
                risk_factors.append({
                    'risk_type': 'pattern_based',
                    'pattern_id': pattern.pattern_id,
                    'pattern_type': pattern.pattern_type,
                    'risk_score': pattern.risk_score,
                    'severity': 'high' if pattern.risk_score > 0.7 else 'medium',
                    'description': pattern.description
                })

        # 时间跨度调整
        time_multiplier = {
            'short_term': 1.2,   # 短期风险更高
            'medium_term': 1.0,  # 中期风险正常
            'long_term': 0.8     # 长期风险较低
        }.get(time_horizon, 1.0)

        for risk in risk_factors:
            risk['risk_score'] = min(1.0, risk.get('risk_score', 0.5) * time_multiplier)

        return risk_factors

    def _calculate_prediction_confidence(self, cases: List[SafetyCase],
                                       patterns: List[PatternInsight]) -> float:
        """计算预测置信度"""
        base_confidence = 0.5

        # 基于案例数量调整
        case_confidence = min(1.0, len(cases) / 20.0)  # 20个案例达到满置信度
        base_confidence += case_confidence * 0.3

        # 基于模式数量调整
        pattern_confidence = min(1.0, len(patterns) / 10.0)  # 10个模式达到满置信度
        base_confidence += pattern_confidence * 0.2

        # 时间跨度调整 (最近案例权重更高)
        if cases:
            recent_cases = [c for c in cases if c.timestamp > datetime.now() - timedelta(days=7)]
            recency_confidence = len(recent_cases) / len(cases)
            base_confidence += recency_confidence * 0.1

        return min(1.0, base_confidence)

    def _generate_mitigation_strategies(self, risks: List[Dict[str, Any]]) -> List[str]:
        """生成缓解策略"""
        strategies = []

        # 基于风险类型生成策略
        for risk in risks:
            risk_type = risk.get('risk_type')
            category = risk.get('category', '')

            if risk_type == 'frequent_incidents':
                if 'performance' in category:
                    strategies.append("Implement additional performance monitoring and alerting")
                elif 'fairness' in category:
                    strategies.append("Add fairness validation checks in deployment pipeline")
                elif 'stability' in category:
                    strategies.append("Enhance stability testing and chaos engineering")

            elif risk_type == 'pattern_based':
                pattern_type = risk.get('pattern_type')
                if pattern_type == 'temporal':
                    strategies.append("Strengthen monitoring during high-risk time periods")
                elif pattern_type == 'environmental':
                    strategies.append("Add environmental stress testing")

        # 去重并添加通用策略
        strategies = list(set(strategies))
        strategies.extend([
            "Increase test coverage for identified risk areas",
            "Implement gradual rollout with automated rollback triggers",
            "Enhance logging and observability for faster root cause analysis"
        ])

        return strategies[:10]  # 最多10个策略

    def _save_prediction(self, prediction: RiskPrediction):
        """保存预测结果"""
        with sqlite3.connect(self.archiver.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO risk_predictions
                (prediction_id, algorithm_version, predicted_risks, confidence_level,
                 time_horizon, mitigation_strategies, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.prediction_id,
                prediction.algorithm_version,
                json.dumps(prediction.predicted_risks),
                prediction.confidence_level,
                prediction.time_horizon,
                json.dumps(prediction.mitigation_strategies),
                datetime.now().isoformat()
            ))

            conn.commit()


class SafetyKnowledgeBase:
    """
    安全知识库

    核心创新：
    1. 案例归档：系统化存储和检索安全事件
    2. 模式挖掘：自动发现安全事件的规律和趋势
    3. 风险预测：基于历史数据预测未来风险
    4. 知识关联：构建安全事件的关联网络
    5. 持续学习：从案例中提取改进建议
    6. 开发者支持：提供反例学习和最佳实践

    这解决了传统安全日志的被动性和不可学习性。
    """

    def __init__(self, db_path: str = "data/safety_casebase.db"):
        self.archiver = CaseArchiver(db_path)
        self.pattern_miner = PatternMiner(self.archiver)
        self.risk_predictor = RiskPredictor(self.archiver, self.pattern_miner)
        self.logger = logging.getLogger("SafetyKnowledgeBase")

    def add_case(self, safety_case: SafetyCase):
        """添加安全案例"""
        self.archiver.archive_case(safety_case)
        self.logger.info(f"Safety case added: {safety_case.case_id}")

    def get_similar_cases(self, query_case: SafetyCase, top_k: int = 5) -> List[SafetyCase]:
        """查找相似案例"""
        # 简单的相似性搜索 (可以扩展为更复杂的算法)
        all_cases = self.archiver.search_cases({
            'category': query_case.category,
            'algorithm_version': query_case.algorithm_version
        })

        # 计算相似度分数
        case_scores = []
        for case in all_cases:
            if case.case_id == query_case.case_id:
                continue

            score = self._calculate_case_similarity(query_case, case)
            case_scores.append((case, score))

        # 按相似度排序
        case_scores.sort(key=lambda x: x[1], reverse=True)

        return [case for case, score in case_scores[:top_k]]

    def _calculate_case_similarity(self, case1: SafetyCase, case2: SafetyCase) -> float:
        """计算案例相似度"""
        similarity = 0.0

        # 类别匹配
        if case1.category == case2.category:
            similarity += 0.3

        # 严重程度接近
        severity_similarity = 1.0 - abs(case1.severity.value == case2.severity.value)
        similarity += severity_similarity * 0.2

        # 根本原因相似
        if case1.root_cause.lower() in case2.root_cause.lower() or \
           case2.root_cause.lower() in case1.root_cause.lower():
            similarity += 0.3

        # 影响相似
        impact_sim = 1.0 - abs(case1.affected_users - case2.affected_users) / max(case1.affected_users, case2.affected_users, 1)
        similarity += impact_sim * 0.2

        return similarity

    def generate_learning_materials(self, algorithm_version: str = None) -> Dict[str, Any]:
        """生成学习材料"""
        materials = {
            'common_failure_patterns': [],
            'best_practices': [],
            'risk_trends': [],
            'case_studies': []
        }

        # 获取案例
        filters = {'algorithm_version': algorithm_version} if algorithm_version else {}
        cases = self.archiver.search_cases(filters)

        if not cases:
            return materials

        # 分析常见失败模式
        category_counts = defaultdict(int)
        root_causes = defaultdict(int)

        for case in cases:
            category_counts[case.category.value] += 1
            root_causes[case.root_cause] += 1

        # 生成常见失败模式
        for category, count in category_counts.items():
            if count >= 3:  # 至少3次
                materials['common_failure_patterns'].append({
                    'pattern': category.replace('_', ' '),
                    'frequency': count,
                    'description': f"Common {category.replace('_', ' ')} issues observed {count} times"
                })

        # 生成最佳实践
        materials['best_practices'] = [
            "Implement comprehensive monitoring and alerting",
            "Use gradual rollout strategies for new deployments",
            "Maintain detailed incident response procedures",
            "Regularly review and update safety testing procedures",
            "Foster culture of learning from failures"
        ]

        # 分析风险趋势
        recent_cases = [c for c in cases if c.timestamp > datetime.now() - timedelta(days=30)]
        if recent_cases:
            trend = "increasing" if len(recent_cases) > len(cases) * 0.3 else "stable"
            materials['risk_trends'].append({
                'period': 'last_30_days',
                'incident_count': len(recent_cases),
                'trend': trend
            })

        # 生成案例研究
        severe_cases = [c for c in cases if c.severity in [CaseSeverity.HIGH, CaseSeverity.CRITICAL]]
        for case in severe_cases[:3]:  # 最多3个案例研究
            materials['case_studies'].append({
                'title': f"{case.category.value.replace('_', ' ').title()} Incident",
                'description': case.description,
                'lessons': case.lessons_learned,
                'prevention': case.prevention_measures
            })

        return materials

    def get_knowledge_graph(self) -> Dict[str, Any]:
        """获取知识图谱"""
        # 这里可以构建案例之间的关联图谱
        cases = self.archiver.search_cases({})

        graph = {
            'nodes': [],
            'edges': []
        }

        # 添加节点
        for case in cases[:50]:  # 限制节点数量
            graph['nodes'].append({
                'id': case.case_id,
                'type': case.category.value,
                'severity': case.severity.value,
                'timestamp': case.timestamp.isoformat()
            })

        # 添加边 (基于相似性或因果关系)
        for i, case1 in enumerate(cases[:20]):
            for j, case2 in enumerate(cases[:20]):
                if i != j and case1.category == case2.category:
                    similarity = self._calculate_case_similarity(case1, case2)
                    if similarity > 0.5:
                        graph['edges'].append({
                            'source': case1.case_id,
                            'target': case2.case_id,
                            'weight': similarity,
                            'type': 'similarity'
                        })

        return graph

    def export_knowledge_base(self, filepath: str):
        """导出知识库"""
        knowledge = {
            'export_timestamp': datetime.now().isoformat(),
            'case_count': len(self.archiver.search_cases({})),
            'patterns': [insight.__dict__ for insight in self.pattern_miner.mine_patterns()],
            'learning_materials': self.generate_learning_materials(),
            'knowledge_graph': self.get_knowledge_graph()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Knowledge base exported to {filepath}")

    def import_knowledge_base(self, filepath: str):
        """导入知识库"""
        with open(filepath, 'r', encoding='utf-8') as f:
            knowledge = json.load(f)

        # 这里可以实现知识库的导入逻辑
        self.logger.info(f"Knowledge base import from {filepath} completed")


def create_safety_casebase(db_path: str = "data/safety_casebase.db") -> SafetyKnowledgeBase:
    """创建安全案例库"""
    return SafetyKnowledgeBase(db_path)


# 使用示例
if __name__ == "__main__":
    # 创建安全案例库
    casebase = create_safety_casebase()

    # 创建示例安全案例
    sample_case = SafetyCase(
        case_id="case_20240101_001",
        timestamp=datetime.now(),
        algorithm_version="v2.1.0",
        category=CaseCategory.PERFORMANCE_DEGRADATION,
        severity=CaseSeverity.HIGH,
        description="Sudden performance drop during peak hours",
        trigger_conditions={
            'network_load': 0.85,
            'time_of_day': 14,
            'system_load': 0.78
        },
        symptoms=[
            "Throughput decreased by 25%",
            "Latency increased by 150ms",
            "Error rate spiked to 5%"
        ],
        root_cause="Congestion control algorithm oscillation under high load",
        contributing_factors=[
            "Insufficient damping in control loop",
            "No load-adaptive parameter tuning"
        ],
        impact_assessment={
            'user_impact': 'degraded_experience',
            'business_impact': 'revenue_loss',
            'technical_impact': 'system_instability'
        },
        affected_users=50000,
        downtime_duration=45.0,
        recovery_actions=[
            "Reduced traffic by 30%",
            "Applied emergency parameter adjustments",
            "Rolled back to previous version"
        ],
        recovery_success=True,
        lessons_learned=[
            "Need better load testing before deployment",
            "Implement adaptive parameter tuning",
            "Add oscillation detection mechanisms"
        ],
        detection_method="automated_monitoring",
        resolution_time=120.0,
        prevention_measures=[
            "Implement chaos engineering testing",
            "Add load-adaptive algorithms",
            "Enhance monitoring for oscillation patterns"
        ],
        tags=["performance", "congestion_control", "high_load"]
    )

    # 添加案例
    casebase.add_case(sample_case)

    # 查找相似案例
    similar_cases = casebase.get_similar_cases(sample_case, 3)
    print(f"Found {len(similar_cases)} similar cases")

    # 生成学习材料
    materials = casebase.generate_learning_materials("v2.1.0")
    print(f"Generated learning materials with {len(materials['common_failure_patterns'])} patterns")

    # 预测风险
    risk_prediction = casebase.risk_predictor.predict_risks("v2.1.0", "medium_term")
    print(f"Risk prediction confidence: {risk_prediction.confidence_level:.3f}")
    print(f"Predicted risks: {len(risk_prediction.predicted_risks)}")

    print("\\nSafety casebase demo completed")