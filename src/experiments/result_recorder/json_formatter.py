"""JSON Formatter - 实验结果格式化"""

import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any


@dataclass
class ExperimentMetadata:
    experiment_id: str
    experiment_name: str
    timestamp: str
    random_seed: int
    platform_version: str = "1.0.0"
    network_instance_seed: int = 0
    network_instance_fingerprint: str = ""
    loss_model: str = "gilbert_elliott"
    total_packets: int = 0
    algorithms: List[str] = field(default_factory=list)
    scenarios: List[str] = field(default_factory=list)
    repeat_count: int = 30

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class AlgorithmRunData:
    algorithm_name: str
    run_index: int
    random_seed: int
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    redundancy_ratio: float = 0.0
    total_packets_sent: int = 0
    start_time: str = ""
    end_time: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    metadata: ExperimentMetadata
    algorithm_results: Dict[str, List[AlgorithmRunData]]
    summary_statistics: Dict = None
    fairness_report: Dict = None

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata.to_dict(),
            "algorithm_results": {
                name: [run.to_dict() for run in runs]
                for name, runs in self.algorithm_results.items()
            },
            "summary_statistics": self.summary_statistics or {},
            "fairness_report": self.fairness_report or {}
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class ExperimentDataFormatter:
    def create_readme(self, result: ExperimentResult) -> str:
        lines = [
            f"# {result.metadata.experiment_name}",
            "",
            "## Metadata",
            f"- ID: {result.metadata.experiment_id}",
            f"- Seed: {result.metadata.random_seed}",
            f"- Algorithms: {', '.join(result.metadata.algorithms)}",
            f"- Scenarios: {', '.join(result.metadata.scenarios)}",
            f"- Repeat Count: {result.metadata.repeat_count}",
            "",
            "## Results Summary"
        ]
        for algo_name, runs in result.algorithm_results.items():
            if runs:
                avg_success = sum(r.success_rate for r in runs) / len(runs)
                avg_latency = sum(r.avg_latency_ms for r in runs) / len(runs)
                lines.append(f"\n### {algo_name}")
                lines.append(f"- Runs: {len(runs)}")
                lines.append(f"- Avg Success Rate: {avg_success:.2%}")
                lines.append(f"- Avg Latency: {avg_latency:.1f}ms")
        return "\n".join(lines)

    def format_statistics(self, data: List[float]) -> Dict[str, float]:
        import statistics
        if not data:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": statistics.mean(data),
            "std": statistics.stdev(data) if len(data) > 1 else 0,
            "min": min(data),
            "max": max(data)
        }
