#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Network Perturbation Model - 高级网络扰动模型

创新点：
1. 多维度网络扰动建模：超越简单丢包，包含延迟、抖动、路由变化
2. 队列延迟突变模型：基于M/M/1队列理论的动态延迟建模
3. BGP路由抖动模拟：基于AS路径变化的路由不稳定性
4. 跨运营商交互模型：考虑不同ISP之间的协同效应
5. 时空相关性建模：网络扰动的时空分布特征
6. 自适应扰动强度：根据观测数据动态调整扰动参数

这解决了传统丢包模型过于简单的问题。
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from scipy import stats
from enum import Enum


class PerturbationType(Enum):
    """扰动类型枚举"""
    PACKET_LOSS = "packet_loss"
    QUEUE_DELAY = "queue_delay"
    BGP_FLAP = "bgp_flap"
    CROSS_ISP = "cross_isp"
    CONGESTION_SPIKE = "congestion_spike"


@dataclass
class QueueDelayConfig:
    """队列延迟配置"""
    service_rate: float = 1000.0  # 服务率 (packets/sec)
    arrival_rate: float = 800.0   # 到达率 (packets/sec)
    buffer_size: int = 100        # 缓冲区大小
    spike_probability: float = 0.1  # 突发概率
    spike_intensity: float = 5.0   # 突发强度倍数
    recovery_rate: float = 0.8     # 恢复率


@dataclass
class BGPConfig:
    """BGP配置"""
    flap_probability: float = 0.05      # 路由抖动概率
    flap_duration_mean: float = 30.0    # 抖动持续时间均值 (seconds)
    flap_duration_std: float = 10.0     # 抖动持续时间标准差
    path_change_probability: float = 0.2  # 路径变化概率
    convergence_delay: float = 5.0      # 收敛延迟 (seconds)


@dataclass
class CrossISPConfig:
    """跨运营商配置"""
    isp_count: int = 3                    # ISP数量
    interaction_strength: float = 0.3     # 交互强度
    correlation_matrix: Optional[np.ndarray] = None  # ISP间相关性矩阵
    peering_delay_matrix: Optional[np.ndarray] = None  # 对等延迟矩阵


@dataclass
class AdvancedPerturbationConfig:
    """高级扰动配置"""
    queue_delay: QueueDelayConfig = field(default_factory=QueueDelayConfig)
    bgp_config: BGPConfig = field(default_factory=BGPConfig)
    cross_isp: CrossISPConfig = field(default_factory=CrossISPConfig)
    temporal_correlation: float = 0.7     # 时空相关性
    spatial_correlation: float = 0.5      # 空间相关性
    enable_adaptive_intensity: bool = True  # 自适应强度


@dataclass
class PerturbationEvent:
    """扰动事件"""
    timestamp: float
    perturbation_type: PerturbationType
    intensity: float
    duration: float
    affected_parameters: Dict[str, Any]
    spatial_scope: str  # "local", "regional", "global"


@dataclass
class NetworkState:
    """网络状态"""
    timestamp: float
    packet_loss_rate: float
    queue_delay: float
    jitter: float
    bgp_stable: bool
    cross_isp_latency: float
    congestion_level: float


class AdvancedNetworkPerturbationModel:
    """
    高级网络扰动模型

    核心创新：
    1. 多维度扰动建模：丢包 + 延迟 + 路由 + 跨ISP交互
    2. 队列理论基础：基于M/M/1的科学延迟建模
    3. 时空相关性：考虑扰动的时空分布特征
    4. 自适应强度：根据观测数据动态调整
    5. 事件驱动：基于事件的扰动生成机制
    """

    def __init__(self, config: Optional[AdvancedPerturbationConfig] = None):
        self.config = config or AdvancedPerturbationConfig()
        self.logger = logging.getLogger("AdvancedPerturbationModel")

        # 状态历史
        self.state_history = deque(maxlen=10000)
        self.event_history = deque(maxlen=5000)

        # 当前状态
        self.current_state = NetworkState(
            timestamp=0.0,
            packet_loss_rate=0.01,
            queue_delay=10.0,
            jitter=2.0,
            bgp_stable=True,
            cross_isp_latency=50.0,
            congestion_level=0.1
        )

        # 初始化ISP相关性矩阵
        self._initialize_isp_matrices()

        # 自适应参数
        self.adaptation_history = deque(maxlen=1000)

    def _initialize_isp_matrices(self):
        """初始化ISP相关性和延迟矩阵"""
        n_isp = self.config.cross_isp.isp_count

        # 相关性矩阵 (随机生成但保证正定)
        if self.config.cross_isp.correlation_matrix is None:
            A = np.random.randn(n_isp, n_isp)
            self.config.cross_isp.correlation_matrix = np.dot(A, A.T)
            # 归一化到[0,1]
            self.config.cross_isp.correlation_matrix /= np.max(np.abs(self.config.cross_isp.correlation_matrix))

        # 对等延迟矩阵
        if self.config.cross_isp.peering_delay_matrix is None:
            self.config.cross_isp.peering_delay_matrix = np.random.exponential(20, (n_isp, n_isp))
            np.fill_diagonal(self.config.cross_isp.peering_delay_matrix, 0)  # 对角线为0

    def generate_perturbation_sequence(self, duration: float, time_step: float = 0.1) -> List[NetworkState]:
        """
        生成扰动序列

        Args:
            duration: 持续时间 (seconds)
            time_step: 时间步长 (seconds)

        Returns:
            网络状态序列
        """
        states = []
        current_time = 0.0

        while current_time < duration:
            # 更新网络状态
            self._update_network_state(current_time)

            # 记录状态
            state_copy = NetworkState(**self.current_state.__dict__)
            states.append(state_copy)

            current_time += time_step
            self.current_state.timestamp = current_time

        self.state_history.extend(states)
        return states

    def _update_network_state(self, timestamp: float):
        """更新网络状态"""
        # 1. 生成基础扰动事件
        events = self._generate_perturbation_events(timestamp)

        # 2. 计算各维度扰动强度
        queue_delay_perturbation = self._compute_queue_delay_perturbation(events)
        bgp_perturbation = self._compute_bgp_perturbation(events)
        cross_isp_perturbation = self._compute_cross_isp_perturbation(events)

        # 3. 应用时空相关性
        correlated_perturbations = self._apply_spatial_temporal_correlation(
            queue_delay_perturbation, bgp_perturbation, cross_isp_perturbation
        )

        # 4. 更新状态
        self._apply_perturbations_to_state(correlated_perturbations)

        # 5. 记录事件
        self.event_history.extend(events)

    def _generate_perturbation_events(self, timestamp: float) -> List[PerturbationEvent]:
        """生成扰动事件"""
        events = []

        # 队列延迟突变
        if np.random.random() < self.config.queue_delay.spike_probability:
            event = PerturbationEvent(
                timestamp=timestamp,
                perturbation_type=PerturbationType.QUEUE_DELAY,
                intensity=np.random.exponential(self.config.queue_delay.spike_intensity),
                duration=np.random.exponential(10.0),  # 持续时间
                affected_parameters={"queue_length": "spike"},
                spatial_scope="local"
            )
            events.append(event)

        # BGP路由抖动
        if np.random.random() < self.config.bgp_config.flap_probability:
            duration = np.random.normal(
                self.config.bgp_config.flap_duration_mean,
                self.config.bgp_config.flap_duration_std
            )
            event = PerturbationEvent(
                timestamp=timestamp,
                perturbation_type=PerturbationType.BGP_FLAP,
                intensity=1.0,
                duration=max(1.0, duration),
                affected_parameters={"routing_table": "flapping"},
                spatial_scope="regional"
            )
            events.append(event)

        # 跨ISP交互
        if np.random.random() < 0.05:  # 5%概率
            event = PerturbationEvent(
                timestamp=timestamp,
                perturbation_type=PerturbationType.CROSS_ISP,
                intensity=np.random.beta(2, 5),  # 偏向小值
                duration=np.random.exponential(30.0),
                affected_parameters={"isp_interaction": "correlated_delay"},
                spatial_scope="global"
            )
            events.append(event)

        return events

    def _compute_queue_delay_perturbation(self, events: List[PerturbationEvent]) -> float:
        """计算队列延迟扰动"""
        base_delay = self._compute_base_queue_delay()

        # 查找活跃的队列延迟事件
        active_spikes = [
            event for event in events
            if event.perturbation_type == PerturbationType.QUEUE_DELAY
        ]

        if active_spikes:
            # 叠加所有活跃事件的强度
            total_intensity = sum(event.intensity for event in active_spikes)
            perturbation = base_delay * (1 + total_intensity)
        else:
            perturbation = base_delay

        return perturbation

    def _compute_base_queue_delay(self) -> float:
        """计算基础队列延迟 (基于M/M/1理论)"""
        ρ = self.config.queue_delay.arrival_rate / self.config.queue_delay.service_rate

        if ρ >= 1:
            # 系统不稳定，返回极大延迟
            return 1000.0

        # M/M/1平均队列延迟
        avg_delay = 1 / (self.config.queue_delay.service_rate * (1 - ρ))
        return avg_delay

    def _compute_bgp_perturbation(self, events: List[PerturbationEvent]) -> float:
        """计算BGP扰动"""
        active_flaps = [
            event for event in events
            if event.perturbation_type == PerturbationType.BGP_FLAP
        ]

        if active_flaps:
            # BGP抖动导致的额外延迟
            convergence_delay = self.config.bgp_config.convergence_delay
            perturbation = convergence_delay * len(active_flaps)
        else:
            perturbation = 0.0

        return perturbation

    def _compute_cross_isp_perturbation(self, events: List[PerturbationEvent]) -> float:
        """计算跨ISP扰动"""
        active_interactions = [
            event for event in events
            if event.perturbation_type == PerturbationType.CROSS_ISP
        ]

        if active_interactions:
            # 基于ISP相关性矩阵计算交互效应
            correlation_effect = np.random.choice(
                self.config.cross_isp.correlation_matrix.flatten(),
                size=len(active_interactions)
            ).mean()

            base_delay = self.config.cross_isp.peering_delay_matrix.mean()
            perturbation = base_delay * correlation_effect
        else:
            perturbation = 0.0

        return perturbation

    def _apply_spatial_temporal_correlation(self, queue_delay: float, bgp_delay: float,
                                          cross_isp_delay: float) -> Dict[str, float]:
        """应用时空相关性"""
        # 时空相关性矩阵
        correlation_matrix = np.array([
            [1.0, self.config.temporal_correlation, self.config.spatial_correlation],
            [self.config.temporal_correlation, 1.0, self.config.spatial_correlation],
            [self.config.spatial_correlation, self.config.spatial_correlation, 1.0]
        ])

        perturbations = np.array([queue_delay, bgp_delay, cross_isp_delay])

        # 应用相关性
        correlated = np.dot(correlation_matrix, perturbations)

        return {
            'queue_delay': max(0, correlated[0]),
            'bgp_delay': max(0, correlated[1]),
            'cross_isp_delay': max(0, correlated[2])
        }

    def _apply_perturbations_to_state(self, perturbations: Dict[str, float]):
        """将扰动应用到网络状态"""
        # 更新队列延迟
        self.current_state.queue_delay = perturbations['queue_delay']

        # 更新抖动 (基于延迟变化)
        delay_std = np.std([s.queue_delay for s in list(self.state_history)[-10:]]) if self.state_history else 1.0
        self.current_state.jitter = delay_std

        # 更新BGP稳定性
        self.current_state.bgp_stable = perturbations['bgp_delay'] < 10.0

        # 更新跨ISP延迟
        self.current_state.cross_isp_latency = 50.0 + perturbations['cross_isp_delay']

        # 更新拥塞水平 (基于队列延迟)
        congestion_factor = min(1.0, self.current_state.queue_delay / 100.0)
        self.current_state.congestion_level = congestion_factor

        # 轻微影响丢包率
        base_loss = 0.01
        congestion_loss = congestion_factor * 0.1
        self.current_state.packet_loss_rate = base_loss + congestion_loss

    def adapt_from_observations(self, observed_states: List[NetworkState]):
        """从观测数据自适应调整参数"""
        if not self.config.enable_adaptive_intensity:
            return

        # 分析观测数据的统计特性
        queue_delays = [s.queue_delay for s in observed_states]
        packet_losses = [s.packet_loss_rate for s in observed_states]

        # 更新队列延迟配置
        if len(queue_delays) > 10:
            observed_mean_delay = np.mean(queue_delays)
            observed_std_delay = np.std(queue_delays)

            # 自适应调整服务率以匹配观测延迟
            target_service_rate = 1.0 / (observed_mean_delay * 0.001)  # 粗略估计
            self.config.queue_delay.service_rate = np.clip(target_service_rate, 100, 10000)

        # 记录适应历史
        adaptation_record = {
            'timestamp': observed_states[-1].timestamp if observed_states else 0,
            'queue_delays_mean': np.mean(queue_delays) if queue_delays else 0,
            'packet_losses_mean': np.mean(packet_losses) if packet_losses else 0,
            'adapted_service_rate': self.config.queue_delay.service_rate
        }
        self.adaptation_history.append(adaptation_record)

    def get_perturbation_statistics(self) -> Dict[str, Any]:
        """获取扰动统计信息"""
        if not self.event_history:
            return {}

        events_by_type = {}
        for event in self.event_history:
            event_type = event.perturbation_type.value
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(event)

        stats = {}
        for event_type, events in events_by_type.items():
            intensities = [e.intensity for e in events]
            durations = [e.duration for e in events]

            stats[event_type] = {
                'count': len(events),
                'avg_intensity': np.mean(intensities),
                'avg_duration': np.mean(durations),
                'max_intensity': np.max(intensities),
                'total_duration': np.sum(durations)
            }

        # 状态统计
        if self.state_history:
            queue_delays = [s.queue_delay for s in self.state_history]
            packet_losses = [s.packet_loss_rate for s in self.state_history]

            stats['network_state'] = {
                'avg_queue_delay': np.mean(queue_delays),
                'max_queue_delay': np.max(queue_delays),
                'avg_packet_loss': np.mean(packet_losses),
                'bgp_stability_ratio': np.mean([s.bgp_stable for s in self.state_history])
            }

        return stats

    def reset(self):
        """重置模型状态"""
        self.state_history.clear()
        self.event_history.clear()
        self.adaptation_history.clear()

        # 重置为初始状态
        self.current_state = NetworkState(
            timestamp=0.0,
            packet_loss_rate=0.01,
            queue_delay=10.0,
            jitter=2.0,
            bgp_stable=True,
            cross_isp_latency=50.0,
            congestion_level=0.1
        )