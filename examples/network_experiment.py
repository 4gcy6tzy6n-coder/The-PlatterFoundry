#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Network Experiment Demo - 网络实验示例
"""

import time
from src.experiments import (
    NetworkInstanceGenerator,
    RandomExtractor
)


class ScenarioConfig:
    def __init__(self, name, loss_rate, good_to_bad, bad_to_good, loss_good, loss_bad):
        self.name = name
        self.loss_rate = loss_rate
        self.good_to_bad_prob = good_to_bad
        self.bad_to_good_prob = bad_to_good
        self.loss_in_good = loss_good
        self.loss_in_bad = loss_bad


def run_network_experiment(master_seed=42, total_packets=1000, repeat_count=10):
    print(f"=== Network Platform 实验演示 ===")
    print(f"Seed: {master_seed}, Packets: {total_packets}, Repeats: {repeat_count}\n")

    scenarios = [
        ScenarioConfig("low_loss", 0.05, 0.02, 0.3, 0.01, 0.4),
        ScenarioConfig("medium_loss", 0.15, 0.05, 0.2, 0.02, 0.5),
        ScenarioConfig("high_loss", 0.30, 0.10, 0.15, 0.05, 0.7)
    ]

    algorithms = ["Algorithm_A", "Algorithm_B", "Algorithm_C"]

    extractor = RandomExtractor(master_seed)
    results = {}

    for scenario in scenarios:
        print(f"场景: {scenario.name}")
        generator = NetworkInstanceGenerator(extractor.extract_for_scenario(scenario.name))
        instance = generator.generate_gilbert_elliott_loss(
            total_packets=total_packets,
            good_to_bad_prob=scenario.good_to_bad_prob,
            bad_to_good_prob=scenario.bad_to_good_prob,
            loss_in_good=scenario.loss_in_good,
            loss_in_bad=scenario.loss_in_bad
        )

        stats = instance.get_statistics()
        print(f"  网络实例: 总包={stats['total_packets']}, 丢包率={stats['loss_rate']:.2%}, 突发数={stats['burst_count']}")

        results[scenario.name] = {}

        for algo in algorithms:
            print(f"  算法: {algo}")
            run_results = []

            for repeat in range(repeat_count):
                run_seed = extractor.extract_for_run(scenario.name, algo, repeat)
                performance = simulate_algorithm(algo, instance, run_seed)
                run_results.append(performance)
                print(f"    Run {repeat+1}: 成功率={performance['success_rate']:.2%}")

            avg_success = sum(r['success_rate'] for r in run_results) / len(run_results)
            avg_latency = sum(r['avg_latency_ms'] for r in run_results) / len(run_results)
            print(f"    平均: 成功率={avg_success:.2%}, 延迟={avg_latency:.1f}ms")
            results[scenario.name][algo] = {
                'runs': run_results,
                'avg_success': avg_success,
                'avg_latency': avg_latency
            }
        print()

    print("=== 实验完成 ===\n")
    print("汇总:")
    for scenario_name, algo_results in results.items():
        print(f"\n{scenario_name}:")
        for algo, data in algo_results.items():
            print(f"  {algo}: 成功率={data['avg_success']:.2%}, 延迟={data['avg_latency']:.1f}ms")

    return results


def simulate_algorithm(algorithm_type, network_instance, seed):
    import random
    rng = random.Random(seed)
    loss_rate = len(network_instance.loss_events) / network_instance.total_packets

    if algorithm_type == "Algorithm_A":
        base_success = 0.95 - loss_rate * 0.3
        base_latency = 50 + loss_rate * 100
    elif algorithm_type == "Algorithm_B":
        base_success = 0.90 - loss_rate * 0.5
        base_latency = 100 + loss_rate * 200
    else:
        base_success = 0.85 - loss_rate * 0.4
        base_latency = 80 + loss_rate * 150

    success_rate = max(0.0, min(1.0, base_success + rng.gauss(0, 0.02)))
    avg_latency = max(0.0, base_latency + rng.gauss(0, 10))

    return {
        'success_rate': success_rate,
        'avg_latency_ms': avg_latency,
        'run_seed': seed
    }


if __name__ == "__main__":
    run_network_experiment()
