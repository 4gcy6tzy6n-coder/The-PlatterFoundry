#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式可视化器 - 使用Plotly创建交互式图表

特性：
1. 交互式仪表板
2. 动态图表
3. 悬停信息
4. 图表导出
5. 响应式设计
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging


@dataclass
class InteractivePlotConfig:
    """交互式绘图配置"""
    width: int = 1000
    height: int = 600
    theme: str = "plotly_white"  # 主题: plotly_white, plotly_dark, presentation
    show_legend: bool = True
    hover_mode: str = "closest"  # 悬停模式: closest, x, y, x unified, y unified
    export_format: str = "html"  # 导出格式: html, png, svg, pdf


class InteractiveVisualizer:
    """
    交互式可视化器
    
    使用Plotly创建交互式图表，支持：
    - 公平性雷达图
    - 性能热力图
    - 置信区间图
    - 时间序列图
    - 综合仪表板
    """
    
    def __init__(self, config: Optional[InteractivePlotConfig] = None):
        self.config = config or InteractivePlotConfig()
        self.logger = logging.getLogger("InteractiveVisualizer")
        
        # 检查Plotly是否可用
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            self.plotly_available = True
        except ImportError:
            self.plotly_available = False
            self.logger.warning("Plotly未安装，交互式可视化将不可用")
    
    def is_available(self) -> bool:
        """检查Plotly是否可用"""
        return self.plotly_available
    
    def plot_fairness_radar(self, 
                           fairness_metrics: Dict[str, float],
                           title: str = "公平性雷达图") -> Any:
        """
        创建公平性雷达图
        
        Args:
            fairness_metrics: 公平性指标字典
            title: 图表标题
            
        Returns:
            Plotly图表对象
        """
        if not self.plotly_available:
            raise ImportError("Plotly未安装，请运行: pip install plotly")
        
        import plotly.graph_objects as go
        
        # 准备数据
        dimensions = list(fairness_metrics.keys())
        values = list(fairness_metrics.values())
        
        # 闭合雷达图
        dimensions_closed = dimensions + [dimensions[0]]
        values_closed = values + [values[0]]
        
        # 创建雷达图
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=dimensions_closed,
            fill='toself',
            name='公平性指标',
            line=dict(color='rgb(31, 119, 180)', width=2),
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            showlegend=self.config.show_legend,
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme,
            hovermode=self.config.hover_mode
        )
        
        return fig
    
    def plot_performance_heatmap(self,
                                performance_data: Dict[str, Dict[str, float]],
                                title: str = "性能热力图") -> Any:
        """
        创建性能热力图
        
        Args:
            performance_data: 性能数据，格式: {场景: {算法: 性能}}
            title: 图表标题
            
        Returns:
            Plotly图表对象
        """
        if not self.plotly_available:
            raise ImportError("Plotly未安装，请运行: pip install plotly")
        
        import plotly.graph_objects as go
        
        # 准备数据
        scenarios = list(performance_data.keys())
        algorithms = list(performance_data[scenarios[0]].keys())
        
        # 创建矩阵
        matrix = []
        for scenario in scenarios:
            row = [performance_data[scenario].get(algo, 0) for algo in algorithms]
            matrix.append(row)
        
        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=algorithms,
            y=scenarios,
            colorscale='Viridis',
            colorbar=dict(title="性能得分"),
            hoverongaps=False,
            hovertemplate="场景: %{y}<br>算法: %{x}<br>性能: %{z:.4f}<extra></extra>"
        ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            xaxis=dict(
                title="算法",
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title="场景",
                tickfont=dict(size=12)
            ),
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme,
            hovermode=self.config.hover_mode
        )
        
        return fig
    
    def plot_confidence_interval(self,
                                x_values: List[float],
                                mean_values: List[float],
                                ci_lower: List[float],
                                ci_upper: List[float],
                                title: str = "置信区间图",
                                x_label: str = "X轴",
                                y_label: str = "Y轴") -> Any:
        """
        创建置信区间图
        
        Args:
            x_values: X轴值
            mean_values: 平均值
            ci_lower: 置信区间下限
            ci_upper: 置信区间上限
            title: 图表标题
            x_label: X轴标签
            y_label: Y轴标签
            
        Returns:
            Plotly图表对象
        """
        if not self.plotly_available:
            raise ImportError("Plotly未安装，请运行: pip install plotly")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # 添加置信区间
        fig.add_trace(go.Scatter(
            x=x_values + x_values[::-1],
            y=ci_upper + ci_lower[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name='95%置信区间'
        ))
        
        # 添加均值线
        fig.add_trace(go.Scatter(
            x=x_values,
            y=mean_values,
            mode='lines+markers',
            line=dict(color='rgb(31, 119, 180)', width=2),
            marker=dict(size=6),
            name='均值',
            hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y:.4f}}<extra></extra>"
        ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            xaxis=dict(
                title=x_label,
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title=y_label,
                tickfont=dict(size=12)
            ),
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme,
            showlegend=True,
            hovermode=self.config.hover_mode
        )
        
        return fig
    
    def plot_time_series(self,
                        time_values: List[float],
                        values: List[float],
                        title: str = "时间序列图",
                        x_label: str = "时间",
                        y_label: str = "值",
                        series_name: str = "序列") -> Any:
        """
        创建时间序列图
        
        Args:
            time_values: 时间值
            values: 序列值
            title: 图表标题
            x_label: X轴标签
            y_label: Y轴标签
            series_name: 序列名称
            
        Returns:
            Plotly图表对象
        """
        if not self.plotly_available:
            raise ImportError("Plotly未安装，请运行: pip install plotly")
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_values,
            y=values,
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=4),
            name=series_name,
            hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y:.4f}}<extra></extra>"
        ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            xaxis=dict(
                title=x_label,
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title=y_label,
                tickfont=dict(size=12)
            ),
            width=self.config.width,
            height=self.config.height,
            template=self.config.theme,
            hovermode=self.config.hover_mode
        )
        
        return fig
    
    def create_summary_dashboard(self,
                                performance_data: Dict[str, Dict[str, float]],
                                fairness_metrics: Dict[str, float],
                                statistical_results: Dict[str, Any],
                                title: str = "NetworkPlatform 综合仪表板") -> Any:
        """
        创建综合仪表板
        
        Args:
            performance_data: 性能数据
            fairness_metrics: 公平性指标
            statistical_results: 统计结果
            title: 仪表板标题
            
        Returns:
            Plotly图表对象
        """
        if not self.plotly_available:
            raise ImportError("Plotly未安装，请运行: pip install plotly")
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # 创建子图布局 (2x2)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('性能热力图', '公平性雷达图', 
                          '置信区间', '统计摘要'),
            specs=[[{'type': 'heatmap'}, {'type': 'polar'}],
                   [{'type': 'scatter'}, {'type': 'table'}]]
        )
        
        # 1. 性能热力图
        scenarios = list(performance_data.keys())
        algorithms = list(performance_data[scenarios[0]].keys())
        matrix = []
        for scenario in scenarios:
            row = [performance_data[scenario].get(algo, 0) for algo in algorithms]
            matrix.append(row)
        
        fig.add_trace(
            go.Heatmap(z=matrix, x=algorithms, y=scenarios, colorscale='Viridis'),
            row=1, col=1
        )
        
        # 2. 公平性雷达图
        dimensions = list(fairness_metrics.keys())
        values = list(fairness_metrics.values())
        dimensions_closed = dimensions + [dimensions[0]]
        values_closed = values + [values[0]]
        
        fig.add_trace(
            go.Scatterpolar(r=values_closed, theta=dimensions_closed, fill='toself'),
            row=1, col=2
        )
        
        # 3. 置信区间（如果有）
        if 'confidence_intervals' in statistical_results:
            ci_data = statistical_results['confidence_intervals']
            fig.add_trace(
                go.Scatter(x=ci_data['x'], y=ci_data['mean'], 
                          error_y=dict(type='data', array=ci_data['upper'], 
                                     arrayminus=ci_data['lower'])),
                row=2, col=1
            )
        
        # 4. 统计摘要表格
        if 'summary_stats' in statistical_results:
            stats = statistical_results['summary_stats']
            table_data = []
            for key, value in stats.items():
                table_data.append([key, f"{value:.4f}"])
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['指标', '值']),
                    cells=dict(values=[[row[0] for row in table_data], 
                                      [row[1] for row in table_data]])
                ),
                row=2, col=2
            )
        
        # 更新整体布局
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=24)
            ),
            width=self.config.width * 1.5,
            height=self.config.height * 1.5,
            template=self.config.theme,
            showlegend=False
        )
        
        return fig
    
    def export_plot(self, fig: Any, filename: str, format: Optional[str] = None) -> str:
        """
        导出图表
        
        Args:
            fig: Plotly图表对象
            filename: 文件名（不含扩展名）
            format: 导出格式，如果为None则使用配置中的格式
            
        Returns:
            导出的文件路径
        """
        if not self.plotly_available:
            raise ImportError("Plotly未安装")
        
        if format is None:
            format = self.config.export_format
        
        if format == "html":
            filepath = f"{filename}.html"
            fig.write_html(filepath)
        elif format == "png":
            filepath = f"{filename}.png"
            fig.write_image(filepath, width=self.config.width, height=self.config.height)
        elif format == "svg":
            filepath = f"{filename}.svg"
            fig.write_image(filepath, width=self.config.width, height=self.config.height)
        elif format == "pdf":
            filepath = f"{filename}.pdf"
            fig.write_image(filepath, width=self.config.width, height=self.config.height)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
        
        self.logger.info(f"图表已导出到: {filepath}")
        return filepath


# 使用示例
if __name__ == "__main__":
    import sys
    
    # 检查Plotly是否安装
    try:
        import plotly
        print("[完成] Plotly已安装")
    except ImportError:
        print("[错误] Plotly未安装，请运行: pip install plotly")
        sys.exit(1)
    
    # 创建可视化器
    visualizer = InteractiveVisualizer()
    
    # 示例1: 公平性雷达图
    print("\n[数据] 示例1: 公平性雷达图")
    fairness_metrics = {
        "资源公平性": 0.92,
        "信息对称性": 1.0,
        "统计充分性": 0.85,
        "性能一致性": 0.88,
        "Jain指数": 0.95
    }
    
    fig1 = visualizer.plot_fairness_radar(fairness_metrics, "公平性评估")
    visualizer.export_plot(fig1, "fairness_radar", "html")
    print("  [完成] 公平性雷达图已保存")
    
    # 示例2: 性能热力图
    print("\n[数据] 示例2: 性能热力图")
    performance_data = {
        "低丢包": {"Baseline": 0.85, "Proposed": 0.90, "Advanced": 0.92},
        "中丢包": {"Baseline": 0.70, "Proposed": 0.80, "Advanced": 0.85},
        "高丢包": {"Baseline": 0.50, "Proposed": 0.65, "Advanced": 0.72}
    }
    
    fig2 = visualizer.plot_performance_heatmap(performance_data, "算法性能对比")
    visualizer.export_plot(fig2, "performance_heatmap", "html")
    print("  [完成] 性能热力图已保存")
    
    # 示例3: 置信区间图
    print("\n[数据] 示例3: 置信区间图")
    x = [1, 2, 3, 4, 5]
    mean = [0.85, 0.87, 0.86, 0.88, 0.87]
    lower = [0.82, 0.84, 0.83, 0.85, 0.84]
    upper = [0.88, 0.90, 0.89, 0.91, 0.90]
    
    fig3 = visualizer.plot_confidence_interval(
        x, mean, lower, upper, 
        "置信区间分析", "实验次数", "成功率"
    )
    visualizer.export_plot(fig3, "confidence_interval", "html")
    print("  [完成] 置信区间图已保存")
    
    # 示例4: 时间序列图
    print("\n[数据] 示例4: 时间序列图")
    time_vals = list(range(100))
    values = [0.05 + 0.01 * np.sin(t/10) + np.random.randn() * 0.005 for t in time_vals]
    
    fig4 = visualizer.plot_time_series(
        time_vals, values,
        "丢包率时间序列", "时间", "丢包率", "网络丢包"
    )
    visualizer.export_plot(fig4, "time_series", "html")
    print("  [完成] 时间序列图已保存")
    
    # 示例5: 综合仪表板
    print("\n[数据] 示例5: 综合仪表板")
    statistical_results = {
        "confidence_intervals": {
            "x": [1, 2, 3, 4, 5],
            "mean": [0.85, 0.87, 0.86, 0.88, 0.87],
            "lower": [0.82, 0.84, 0.83, 0.85, 0.84],
            "upper": [0.88, 0.90, 0.89, 0.91, 0.90]
        },
        "summary_stats": {
            "平均性能": 0.865,
            "标准差": 0.012,
            "最小值": 0.85,
            "最大值": 0.88,
            "样本量": 150
        }
    }
    
    fig5 = visualizer.create_summary_dashboard(
        performance_data, fairness_metrics, statistical_results,
        "NetworkPlatform 综合分析仪表板"
    )
    visualizer.export_plot(fig5, "summary_dashboard", "html")
    print("  [完成] 综合仪表板已保存")
    
    print("\n" + "=" * 70)
    print("所有交互式图表已生成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  - fairness_radar.html")
    print("  - performance_heatmap.html")
    print("  - confidence_interval.html")
    print("  - time_series.html")
    print("  - summary_dashboard.html")
    print("\n提示: 在浏览器中打开HTML文件查看交互式图表")