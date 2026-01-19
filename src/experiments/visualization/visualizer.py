#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization Module - 可视化模块

创新点：
1. 公平性雷达图：多维度公平性可视化
2. 性能对比热力图：直观展示各算法性能
3. 置信区间可视化：显示估计不确定性
4. 统计显著性标记：显著差异一目了然
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import io
import base64


@dataclass
class PlotConfig:
    """绘图配置"""
    figure_size: Tuple[int, int] = (10, 6)
    dpi: int = 100
    color_scheme: str = 'viridis'
    font_size: int = 12
    show_significance: bool = True
    save_path: Optional[str] = None


class NetworkVisualizer:
    """
    网络实验可视化器
    
    提供多种可视化图表：
    1. 公平性雷达图
    2. 性能对比热力图
    3. 置信区间图
    4. 统计分布箱线图
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        plt.style.use('default')
    
    def plot_fairness_radar(
        self, 
        fairness_metrics: Dict[str, float],
        title: str = "Fairness Metrics"
    ) -> str:
        """
        绘制公平性雷达图
        
        展示多个公平性维度的得分
        """
        categories = list(fairness_metrics.keys())
        values = list(fairness_metrics.values())
        
        # 确保闭环
        categories.append(categories[0])
        values.append(values[0])
        
        # 角度
        angles = np.linspace(0, 2 * np.pi, len(categories) - 1, endpoint=False)
        angles = np.concatenate([angles, [angles[0] + 2 * np.pi]])
        
        fig, ax = plt.subplots(figsize=self.config.figure_size, subplot_kw=dict(polar=True))
        
        ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1], size=self.config.font_size)
        ax.set_ylim(0, 1)
        
        ax.set_title(title, size=self.config.font_size + 2, pad=20)
        
        plt.tight_layout()
        
        if self.config.save_path:
            plt.savefig(self.config.save_path, dpi=self.config.dpi)
        
        return self._fig_to_base64(fig)
    
    def plot_performance_heatmap(
        self,
        data: Dict[str, Dict[str, float]],
        metric: str = "success_rate",
        title: str = "Performance Comparison"
    ) -> str:
        """
        绘制性能对比热力图
        
        展示不同场景下各算法的性能
        """
        scenarios = list(data.keys())
        algorithms = list(data[scenarios[0]].keys()) if scenarios else []
        
        matrix = np.zeros((len(scenarios), len(algorithms)))
        for i, scenario in enumerate(scenarios):
            for j, algo in enumerate(algorithms):
                matrix[i, j] = data[scenario].get(metric, 0) * 100  # 转换为百分比
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # 设置标签
        ax.set_xticks(np.arange(len(algorithms)))
        ax.set_yticks(np.arange(len(scenarios)))
        ax.set_xticklabels(algorithms, size=self.config.font_size)
        ax.set_yticklabels(scenarios, size=self.config.font_size)
        
        # 添加数值标签
        for i in range(len(scenarios)):
            for j in range(len(algorithms)):
                text = ax.text(j, i, f'{matrix[i, j]:.1f}%',
                              ha="center", va="center", color="black", size=10)
        
        ax.set_title(title, size=self.config.font_size + 2, pad=10)
        ax.set_xlabel("Algorithm", size=self.config.font_size)
        ax.set_ylabel("Scenario", size=self.config.font_size)
        
        plt.colorbar(im, ax=ax, label=f'{metric} (%)')
        plt.tight_layout()
        
        if self.config.save_path:
            plt.savefig(self.config.save_path, dpi=self.config.dpi)
        
        return self._fig_to_base64(fig)
    
    def plot_confidence_intervals(
        self,
        data: Dict[str, Dict[str, Tuple[float, float, float]]],
        title: str = "Performance with 95% Confidence Intervals"
    ) -> str:
        """
        绘制置信区间图
        
        展示各算法性能的置信区间
        """
        algorithms = list(data.keys())
        means = [data[algo].get('mean', 0) for algo in algorithms]
        lowers = [data[algo].get('ci_lower', 0) for algo in algorithms]
        uppers = [data[algo].get('ci_upper', 0) for algo in algorithms]
        
        errors = [
            [means[i] - lowers[i] for i in range(len(algorithms))],
            [uppers[i] - means[i] for i in range(len(algorithms))]
        ]
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        x = np.arange(len(algorithms))
        bars = ax.bar(x, means, yerr=errors, capsize=5, 
                     color='steelblue', alpha=0.7, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, size=self.config.font_size)
        ax.set_ylabel('Success Rate', size=self.config.font_size)
        ax.set_title(title, size=self.config.font_size + 2, pad=10)
        ax.set_ylim(0, 1.1)
        
        # 添加数值标签
        for i, (mean, lower, upper) in enumerate(zip(means, lowers, uppers)):
            ax.text(i, upper + 0.02, f'{mean:.2%}', ha='center', va='bottom', size=9)
        
        plt.tight_layout()
        
        if self.config.save_path:
            plt.savefig(self.config.save_path, dpi=self.config.dpi)
        
        return self._fig_to_base64(fig)
    
    def plot_boxplot_comparison(
        self,
        data: Dict[str, List[float]],
        title: str = "Distribution Comparison"
    ) -> str:
        """
        绘制箱线图对比
        
        展示各算法结果的分布
        """
        algorithms = list(data.keys())
        values = [data[algo] for algo in algorithms]
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        bp = ax.boxplot(values, labels=algorithms, patch_artist=True)
        
        # 设置颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Success Rate', size=self.config.font_size)
        ax.set_title(title, size=self.config.font_size + 2, pad=10)
        
        plt.tight_layout()
        
        if self.config.save_path:
            plt.savefig(self.config.save_path, dpi=self.config.dpi)
        
        return self._fig_to_base64(fig)
    
    def plot_loss_pattern(
        self,
        loss_sequence: List[int],
        title: str = "Loss Pattern Over Time"
    ) -> str:
        """
        绘制丢包模式图
        
        展示丢包事件随时间的变化
        """
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        x = np.arange(len(loss_sequence))
        ax.scatter(x, loss_sequence, c=loss_sequence, cmap='RdYlGn_r', 
                  s=10, alpha=0.6)
        
        ax.set_xlabel('Packet Index', size=self.config.font_size)
        ax.set_ylabel('Loss (1=lost, 0=received)', size=self.config.font_size)
        ax.set_title(title, size=self.config.font_size + 2, pad=10)
        
        # 添加丢包率
        loss_rate = np.mean(loss_sequence)
        ax.axhline(y=loss_rate, color='red', linestyle='--', 
                  label=f'Overall Loss Rate: {loss_rate:.2%}')
        ax.legend()
        
        plt.tight_layout()
        
        if self.config.save_path:
            plt.savefig(self.config.save_path, dpi=self.config.dpi)
        
        return self._fig_to_base64(fig)
    
    def plot_state_distribution(
        self,
        state_dist: Dict[str, float],
        title: str = "Network State Distribution"
    ) -> str:
        """
        绘制状态分布饼图
        
        展示网络各状态的分布
        """
        labels = list(state_dist.keys())
        sizes = list(state_dist.values())
        
        # 过滤掉0值
        non_zero = [(l, s) for l, s in zip(labels, sizes) if s > 0]
        labels, sizes = zip(*non_zero) if non_zero else ([], [])
        
        colors = ['#2ecc71', '#e74c3c', '#f39c12'][:len(labels)]
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90
        )
        
        ax.set_title(title, size=self.config.font_size + 2, pad=10)
        
        plt.tight_layout()
        
        if self.config.save_path:
            plt.savefig(self.config.save_path, dpi=self.config.dpi)
        
        return self._fig_to_base64(fig)
    
    def create_summary_dashboard(
        self,
        performance_data: Dict[str, Dict[str, float]],
        fairness_data: Dict[str, float],
        statistical_results: Dict[str, Any]
    ) -> str:
        """
        创建综合仪表板
        
        将多个图表整合到一个页面
        """
        fig = plt.figure(figsize=(14, 10))
        
        # 子图1: 性能热力图
        ax1 = fig.add_subplot(2, 2, 1)
        scenarios = list(performance_data.keys())
        algorithms = list(performance_data[scenarios[0]].keys()) if scenarios else []
        matrix = np.array([[performance_data[s].get('mean', 0) for s in scenarios] for _ in algorithms])
        
        im = ax1.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_yticks(range(len(algorithms)))
        ax1.set_xticklabels(scenarios, rotation=45, ha='right')
        ax1.set_yticklabels(algorithms)
        ax1.set_title('Performance Heatmap')
        
        # 子图2: 公平性雷达图
        ax2 = fig.add_subplot(2, 2, 2, polar=True)
        categories = list(fairness_data.keys())
        values = list(fairness_data.values())
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        ax2.plot(angles, values, 'o-', linewidth=2)
        ax2.fill(angles, values, alpha=0.25)
        ax2.set_xticks(angles)
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 1)
        ax2.set_title('Fairness Metrics')
        
        # 子图3: 统计显著性
        ax3 = fig.add_subplot(2, 2, 3)
        metrics = list(statistical_results.keys())[:5]
        p_values = [statistical_results[m].get('p_value', 1) for m in metrics]
        colors = ['green' if p < 0.05 else 'red' for p in p_values]
        ax3.barh(metrics, p_values, color=colors)
        ax3.axvline(x=0.05, color='black', linestyle='--', label='α=0.05')
        ax3.set_xlabel('p-value')
        ax3.set_title('Statistical Significance')
        ax3.legend()
        
        # 子图4: 关键指标
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        summary_text = f"""
        Key Findings:
        
        - Overall Fairness Score: {fairness_data.get('overall', 0):.2f}
        
        - Best Performing Algorithm: {max(algorithms, key=lambda a: np.mean([performance_data[s].get('mean', 0) for s in scenarios])) if algorithms else 'N/A'}
        
        - Significant Differences: {sum(1 for p in p_values if p < 0.05)}
        
        - Sample Size: {statistical_results.get('sample_size', 'N/A')}
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=12, 
                verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        
        if self.config.save_path:
            plt.savefig(self.config.save_path, dpi=self.config.dpi)
        
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """将matplotlib图转换为base64编码"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.config.dpi, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_str}"
    
    def plot_network_state_timeseries(
        self,
        loss_rates: List[float],
        latencies: List[float],
        timestamps: List[float] = None,
        burst_events: List[Dict[str, Any]] = None
    ) -> str:
        """
        绘制网络状态时间序列图

        显示丢包率、延迟等网络指标的时间演变
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_size)

        # 时间轴
        if timestamps is None:
            timestamps = list(range(len(loss_rates)))

        # 子图1: 丢包率时间序列
        ax1.plot(timestamps, loss_rates, 'b-', linewidth=1, alpha=0.7, label='Loss Rate')
        ax1.set_ylabel('Loss Rate', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title('Network Loss Rate Over Time')
        ax1.grid(True, alpha=0.3)

        # 标记突发事件
        if burst_events:
            for event in burst_events:
                if 'timestamp' in event:
                    ax1.axvline(x=event['timestamp'], color='r', linestyle='--', alpha=0.7, linewidth=1)

        # 子图2: 延迟时间序列
        ax2.plot(timestamps, latencies, 'r-', linewidth=1, alpha=0.7, label='Latency')
        ax2.set_ylabel('Latency (ms)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_title('Network Latency Over Time')
        ax2.grid(True, alpha=0.3)

        # 子图3: 丢包率分布
        ax3.hist(loss_rates, bins=30, alpha=0.7, color='blue', density=True)
        ax3.set_xlabel('Loss Rate')
        ax3.set_ylabel('Density')
        ax3.set_title('Loss Rate Distribution')
        ax3.grid(True, alpha=0.3)

        # 子图4: 延迟分布
        ax4.hist(latencies, bins=30, alpha=0.7, color='red', density=True)
        ax4.set_xlabel('Latency (ms)')
        ax4.set_ylabel('Density')
        ax4.set_title('Latency Distribution')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def plot_burst_analysis(
        self,
        loss_sequence: List[int],
        burst_events: List[Dict[str, Any]],
        burst_metrics: Dict[str, Any]
    ) -> str:
        """
        绘制突发分析图

        可视化丢包突发模式和统计特性
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_size)

        n = len(loss_sequence)
        timestamps = list(range(n))

        # 子图1: 丢包序列
        ax1.plot(timestamps, loss_sequence, 'k-', linewidth=0.5, alpha=0.7)
        ax1.fill_between(timestamps, 0, loss_sequence, color='red', alpha=0.3)
        ax1.set_ylabel('Loss Event (0/1)')
        ax1.set_title('Loss Sequence with Bursts')
        ax1.grid(True, alpha=0.3)

        # 标记突发事件
        for event in burst_events:
            if 'start' in event and 'length' in event:
                start = event['start']
                end = start + event['length']
                ax1.axvspan(start, end, color='red', alpha=0.2)

        # 子图2: 突发长度分布
        if burst_events:
            burst_lengths = [event.get('length', 0) for event in burst_events if 'length' in event]
            if burst_lengths:
                ax2.hist(burst_lengths, bins=min(20, len(burst_lengths)), alpha=0.7, color='orange')
                ax2.set_xlabel('Burst Length')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Burst Length Distribution')
                ax2.grid(True, alpha=0.3)

        # 子图3: 突发统计指标雷达图
        if burst_metrics:
            categories = ['Burst Index', 'Entropy', 'Run Test', 'Avg Length']
            values = [
                burst_metrics.get('burst_index', 0),
                burst_metrics.get('entropy', 0),
                abs(burst_metrics.get('run_test_statistic', 0)),  # 取绝对值
                burst_metrics.get('avg_run_length', 0) / 10  # 归一化
            ]

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            ax3 = fig.add_subplot(2, 2, 3, polar=True)
            ax3.plot(angles, values, 'o-', linewidth=2, color='green')
            ax3.fill(angles, values, alpha=0.25, color='green')
            ax3.set_xticks(angles)
            ax3.set_xticklabels(categories, fontsize=8)
            ax3.set_title('Burst Statistics', fontsize=10)
            ax3.grid(True, alpha=0.3)

        # 子图4: 自相关分析
        if n > 20:
            # 计算自相关
            autocorr = []
            for lag in range(1, min(20, n//2)):
                corr = np.corrcoef(loss_sequence[:-lag], loss_sequence[lag:])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0)

            ax4.bar(range(1, len(autocorr) + 1), autocorr, alpha=0.7, color='purple')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_xlabel('Lag')
            ax4.set_ylabel('Autocorrelation')
            ax4.set_title('Loss Sequence Autocorrelation')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def save_figure(self, fig, filename: str):
        """保存图表到文件"""
        fig.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
        plt.close(fig)
