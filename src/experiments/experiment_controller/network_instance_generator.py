"""Network Instance Generator - 网络条件生成器"""

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class LossModel:
    INDEPENDENT = "independent"
    GILBERT_ELLIOTT = "gilbert_elliott"


@dataclass
class LossEvent:
    packet_id: str
    timestamp: int
    loss_type: str
    in_burst: bool
    burst_id: Optional[int] = None
    burst_position: Optional[int] = None


@dataclass
class BurstInfo:
    burst_id: int
    start_time: int
    duration: int
    total_packets_lost: int
    loss_events: List[LossEvent]


@dataclass
class NetworkInstance:
    instance_id: str
    seed: int
    loss_model: str
    total_packets: int
    total_duration_ms: int
    loss_events: List[LossEvent] = None
    bursts: List[BurstInfo] = None
    base_loss_rate: float = 0.0
    base_latency_ms: float = 0.0
    base_jitter_ms: float = 0.0
    fingerprint: str = ""

    def __post_init__(self):
        if self.loss_events is None:
            self.loss_events = []
        if self.bursts is None:
            self.bursts = []

    def compute_fingerprint(self) -> str:
        """计算网络实例指纹 - 使用SHA-256提升安全性"""
        data = {
            "seed": self.seed,
            "loss_model": self.loss_model,
            "total_packets": self.total_packets,
            "loss_events": [
                {"packet_id": e.packet_id, "timestamp": e.timestamp, "in_burst": e.in_burst}
                for e in self.loss_events
            ],
        }
        json_str = json.dumps(data, sort_keys=True)
        # 使用SHA-256替代MD5，提升安全性
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        total_loss = len(self.loss_events)
        burst_loss = sum(1 for e in self.loss_events if e.in_burst)
        return {
            "total_packets": self.total_packets,
            "total_loss": total_loss,
            "loss_rate": total_loss / self.total_packets if self.total_packets > 0 else 0.0,
            "burst_count": len(self.bursts),
            "burst_loss": burst_loss,
            "random_loss": total_loss - burst_loss
        }


class NetworkInstanceGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_gilbert_elliott_loss(
        self,
        total_packets: int,
        good_to_bad_prob: float = 0.05,
        bad_to_good_prob: float = 0.2,
        loss_in_good: float = 0.01,
        loss_in_bad: float = 0.5
    ) -> NetworkInstance:
        loss_events = []
        bursts = []
        current_state = "GOOD"
        packet_id = 0
        burst_id = 0
        in_burst = False
        current_burst_events = []

        for packet_id in range(total_packets):
            if current_state == "GOOD":
                if self.rng.random() < good_to_bad_prob:
                    current_state = "BAD"
                    if not in_burst:
                        in_burst = True
                        current_burst_events = []
            else:
                if self.rng.random() < bad_to_good_prob:
                    current_state = "GOOD"
                    if in_burst and current_burst_events:
                        bursts.append(BurstInfo(
                            burst_id=burst_id,
                            start_time=current_burst_events[0].timestamp,
                            duration=len(current_burst_events),
                            total_packets_lost=len(current_burst_events),
                            loss_events=current_burst_events
                        ))
                        burst_id += 1
                        current_burst_events = []
                        in_burst = False

            loss_prob = loss_in_good if current_state == "GOOD" else loss_in_bad
            if self.rng.random() < loss_prob:
                event = LossEvent(
                    packet_id=f"p_{packet_id}",
                    timestamp=packet_id,
                    loss_type="burst" if in_burst else "random",
                    in_burst=in_burst,
                    burst_id=burst_id if in_burst else None,
                    burst_position=len(current_burst_events) if in_burst else None
                )
                loss_events.append(event)
                if in_burst:
                    current_burst_events.append(event)

        instance = NetworkInstance(
            instance_id=f"inst_{self.seed}_{total_packets}",
            seed=self.seed,
            loss_model=LossModel.GILBERT_ELLIOTT,
            total_packets=total_packets,
            total_duration_ms=total_packets,
            loss_events=loss_events,
            bursts=bursts,
            base_loss_rate=len(loss_events) / total_packets if total_packets > 0 else 0.0
        )
        instance.fingerprint = instance.compute_fingerprint()
        return instance

    def generate_independent_loss(
        self,
        total_packets: int,
        loss_rate: float = 0.1
    ) -> NetworkInstance:
        loss_events = []
        for packet_id in range(total_packets):
            if self.rng.random() < loss_rate:
                event = LossEvent(
                    packet_id=f"p_{packet_id}",
                    timestamp=packet_id,
                    loss_type="random",
                    in_burst=False
                )
                loss_events.append(event)

        instance = NetworkInstance(
            instance_id=f"inst_{self.seed}_{total_packets}",
            seed=self.seed,
            loss_model=LossModel.INDEPENDENT,
            total_packets=total_packets,
            total_duration_ms=total_packets,
            loss_events=loss_events,
            base_loss_rate=loss_rate
        )
        instance.fingerprint = instance.compute_fingerprint()
        return instance
