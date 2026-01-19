#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Production Controller - 生产部署控制器
"""

import time
import threading
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..network.real_interface import RealNetworkInterface, SocketProtocol


@dataclass
class ProductionConfig:
    target_ip: str
    target_port: int
    protocol: str = "UDP"
    local_port: Optional[int] = None
    test_duration: int = 300
    packet_interval: float = 0.05
    packet_size: int = 1400
    enable_redundancy: bool = True
    auto_tuning: bool = True


class ProductionController:
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.adapter = None
        self.is_running = False
        self.packet_counter = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger("ProductionController")

    def initialize(self) -> bool:
        try:
            protocol_map = {"UDP": SocketProtocol.UDP, "TCP": SocketProtocol.TCP}
            self.adapter = RealNetworkInterface(
                target_ip=self.config.target_ip,
                target_port=self.config.target_port,
                protocol=protocol_map.get(self.config.protocol, SocketProtocol.UDP),
                local_port=self.config.local_port
            )
            if not self.adapter.initialize():
                return False
            return True
        except Exception as e:
            print(f"初始化失败: {e}")
            return False

    def start(self) -> bool:
        if self.is_running:
            return False
        if not self.initialize():
            return False
        self.is_running = True
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()
        return True

    def _main_loop(self):
        start_time = time.time()
        next_packet_time = start_time
        while self.is_running:
            current_time = time.time()
            if current_time - start_time > self.config.test_duration:
                break
            if current_time >= next_packet_time:
                self._send_packet()
                next_packet_time = current_time + self.config.packet_interval
            time.sleep(0.001)

    def _send_packet(self):
        with self.lock:
            self.packet_counter += 1

        try:
            data = f"packet_{self.packet_counter}_{time.time()}".encode()
            redundancy = self.config.enable_redundancy if self.config.enable_redundancy else 1
            self.adapter.send_packet(data, redundancy=redundancy)
        except Exception:
            pass

    def stop(self):
        self.is_running = False
        if self.adapter:
            self.adapter.close()
        if hasattr(self, 'main_thread'):
            self.main_thread.join(timeout=2.0)

    def get_status(self) -> Dict[str, Any]:
        return {
            'is_running': self.is_running,
            'packet_count': self.packet_counter,
            'adapter_stats': self.adapter.get_stats() if self.adapter else {},
            'config': {
                'target_ip': self.config.target_ip,
                'target_port': self.config.target_port,
                'protocol': self.config.protocol,
                'test_duration': self.config.test_duration,
                'packet_interval': self.config.packet_interval,
                'enable_redundancy': self.config.enable_redundancy
            }
        }

    def run_test(self, duration: int = 60) -> Dict[str, Any]:
        original_duration = self.config.test_duration
        self.config.test_duration = duration
        self.start()
        time.sleep(duration)
        self.stop()
        self.config.test_duration = original_duration
        return self.get_status()
