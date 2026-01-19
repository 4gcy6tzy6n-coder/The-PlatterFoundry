#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Production Demo - 生产部署示例
"""

import time
from src.production import ProductionController, ProductionConfig


def run_production_demo():
    print("=== NetworkPlatform 生产部署演示 ===\n")

    config = ProductionConfig(
        target_ip="127.0.0.1",
        target_port=9999,
        protocol="UDP",
        test_duration=10,
        packet_interval=0.1,
        enable_redundancy=True,
        auto_tuning=True
    )

    controller = ProductionController(config)
    print(f"配置: {config}\n")

    if controller.start():
        print("生产控制器已启动\n")

        for i in range(5):
            time.sleep(2)
            status = controller.get_status()
            print(f"状态更新 #{i+1}:")
            print(f"  运行: {status['is_running']}")
            print(f"  数据包: {status['packet_count']}")
            stats = status.get('adapter_stats', {})
            if stats:
                print(f"  成功率: {stats.get('success_rate', 0):.1f}%")
            print()

        controller.stop()
        print("生产控制器已停止")
    else:
        print("启动失败（这是正常的，因为没有真实服务器）")


if __name__ == "__main__":
    run_production_demo()
