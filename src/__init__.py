# NetworkPlatform - 科研级网络传输实验平台

from .network.real_interface import RealNetworkInterface, SocketProtocol, PacketInfo
from .production.production_controller import ProductionController, ProductionConfig
from .experiments.experiment_controller.network_instance_generator import NetworkInstanceGenerator, NetworkInstance
from .experiments.experiment_scheduler.random_extractor import RandomExtractor, RandomSeed
from .experiments.result_recorder.json_formatter import ExperimentDataFormatter
from .core.estimation.adaptive_estimator import AdaptiveEstimator, NetworkStateEstimator, EstimateResult
from .core.estimation.metrics import MetricsCalculator
from .experiments.analysis.fairness_validator import FairnessValidator, FairnessMetrics
from .experiments.analysis.statistical_analyzer import StatisticalAnalyzer, StatisticalTestResult
from .experiments.visualization.visualizer import NetworkVisualizer
from .exceptions import (
    NetworkPlatformError, EstimationError, ModelError, AnalysisError,
    VisualizationError, ValidationError, ConfigurationError, DataError,
    ConvergenceError, ErrorHandler, get_global_error_handler
)
from .utils.performance_monitor import (
    PerformanceMonitor, ComputationOptimizer, ResourceManager,
    get_performance_monitor, get_computation_optimizer, get_resource_manager
)

__version__ = "1.0.0"
__all__ = [
    "RealNetworkInterface",
    "SocketProtocol",
    "PacketInfo",
    "ProductionController",
    "ProductionConfig",
    "NetworkInstanceGenerator",
    "NetworkInstance",
    "RandomExtractor",
    "RandomSeed",
    "ExperimentDataFormatter",
    "AdaptiveEstimator",
    "NetworkStateEstimator",
    "EstimateResult",
    "MetricsCalculator",
    "FairnessValidator",
    "FairnessMetrics",
    "StatisticalAnalyzer",
    "StatisticalTestResult",
    "NetworkVisualizer",
    "NetworkPlatformError",
    "EstimationError",
    "ModelError",
    "AnalysisError",
    "VisualizationError",
    "ValidationError",
    "ConfigurationError",
    "DataError",
    "ConvergenceError",
    "ErrorHandler",
    "get_global_error_handler",
    "PerformanceMonitor",
    "ComputationOptimizer",
    "ResourceManager",
    "get_performance_monitor",
    "get_computation_optimizer",
    "get_resource_manager"
]
