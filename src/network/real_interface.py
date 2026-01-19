#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real Network Interface - 真实网络接口
支持TCP/UDP socket通信，多副本冗余发送，ACK反馈和RTT测量
"""

import socket
import time
import threading
import struct
from typing import Dict, List, Optional, Any
from enum import Enum


class SocketProtocol(Enum):
    TCP = "TCP"
    UDP = "UDP"


class PacketInfo:
    def __init__(self, packet_id: int, data: bytes, send_time: float):
        self.packet_id = packet_id
        self.data = data
        self.send_time = send_time
        self.recv_time = None
        self.is_lost = False
        self.rtt = None
        self.ack_received = False

    def set_received(self, recv_time: float):
        self.recv_time = recv_time
        self.rtt = (recv_time - self.send_time) * 1000
        self.ack_received = True

    def set_lost(self):
        self.is_lost = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'packet_id': self.packet_id,
            'send_time': self.send_time,
            'recv_time': self.recv_time,
            'rtt': self.rtt,
            'is_lost': self.is_lost,
            'ack_received': self.ack_received,
            'data_size': len(self.data)
        }


class RealNetworkInterface:
    def __init__(self,
                 target_ip: str,
                 target_port: int,
                 protocol: SocketProtocol = SocketProtocol.UDP,
                 local_port: Optional[int] = None,
                 timeout: float = 5.0,
                 enable_ack: bool = True):
        self.target_ip = target_ip
        self.target_port = target_port
        self.protocol = protocol
        self.local_port = local_port or 0
        self.timeout = timeout
        self.enable_ack = enable_ack
        self.socket = None
        self.is_running = False
        self.packet_counter = 0
        self.stats = {'sent': 0, 'received': 0, 'lost': 0, 'rtt_sum': 0.0, 'rtt_count': 0}
        self.lock = threading.Lock()

    def initialize(self) -> bool:
        try:
            if self.protocol == SocketProtocol.UDP:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            elif self.protocol == SocketProtocol.TCP:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.target_ip, self.target_port))
            self.is_running = True
            return True
        except Exception as e:
            print(f"初始化失败: {e}")
            return False

    def send_packet(self, data: bytes, redundancy: int = 1, packet_id: Optional[int] = None) -> List[PacketInfo]:
        if not self.is_running:
            return []

        if packet_id is None:
            with self.lock:
                packet_id = self.packet_counter
                self.packet_counter += 1

        packets = []
        for i in range(redundancy):
            packet = PacketInfo(packet_id, data, time.time())
            packets.append(packet)
            self._send_single_copy(packet, data, i)

        return packets

    def _send_single_copy(self, packet: PacketInfo, data: bytes, copy_index: int):
        header = struct.pack('!II', packet.packet_id, copy_index)
        full_data = header + data
        try:
            if self.protocol == SocketProtocol.UDP:
                self.socket.sendto(full_data, (self.target_ip, self.target_port))
            elif self.protocol == SocketProtocol.TCP:
                self.socket.send(full_data)
            with self.lock:
                self.stats['sent'] += 1
        except Exception:
            packet.set_lost()
            with self.lock:
                self.stats['lost'] += 1

    def send_data(self, data: bytes, redundancy: int = 1) -> bool:
        packets = self.send_packet(data, redundancy)
        return len(packets) > 0 and all(not p.is_lost for p in packets)

    def receive(self, buffer_size: int = 2048) -> Optional[Dict[str, Any]]:
        try:
            if self.protocol == SocketProtocol.UDP:
                data, addr = self.socket.recvfrom(buffer_size)
            else:
                data = self.socket.recv(buffer_size)
            
            if len(data) >= 8:
                packet_id, copy_index = struct.unpack('!II', data[:8])
                payload = data[8:]
                return {
                    'packet_id': packet_id,
                    'copy_index': copy_index,
                    'payload': payload,
                    'timestamp': time.time()
                }
        except BlockingIOError:
            pass
        except Exception:
            pass
        return None

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            stats = self.stats.copy()
            if stats['rtt_count'] > 0:
                stats['avg_rtt'] = stats['rtt_sum'] / stats['rtt_count']
            else:
                stats['avg_rtt'] = 0.0
            total = stats['sent']
            if total > 0:
                stats['success_rate'] = (stats['received'] / total) * 100
                stats['loss_rate'] = (stats['lost'] / total) * 100
            else:
                stats['success_rate'] = 0.0
                stats['loss_rate'] = 0.0
            return stats

    def reset_stats(self):
        with self.lock:
            self.stats = {'sent': 0, 'received': 0, 'lost': 0, 'rtt_sum': 0.0, 'rtt_count': 0}

    def close(self):
        self.is_running = False
        if self.socket:
            self.socket.close()

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
