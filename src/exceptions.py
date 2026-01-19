#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NetworkPlatform Exceptions - 异常处理模块

定义平台的所有异常类，提供统一的错误处理机制
"""

from typing import Dict, Any, Optional


class NetworkPlatformError(Exception):
    """
    网络平台基础异常类

    所有平台异常的基类，提供统一的错误信息格式
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}

        # 构建完整错误信息
        full_message = f"[{self.error_code}] {message}"
        if self.details:
            full_message += f" | Details: {self.details}"

        super().__init__(full_message)


class EstimationError(NetworkPlatformError):
    """
    估计相关异常

    用于自适应估计器的错误
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "ESTIMATION_ERROR", details)


class ModelError(NetworkPlatformError):
    """
    模型相关异常

    用于丢包模型的错误
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_ERROR", details)


class AnalysisError(NetworkPlatformError):
    """
    分析相关异常

    用于统计分析和公平性验证的错误
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "ANALYSIS_ERROR", details)


class VisualizationError(NetworkPlatformError):
    """
    可视化相关异常

    用于绘图和可视化组件的错误
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VISUALIZATION_ERROR", details)


class ValidationError(NetworkPlatformError):
    """
    验证相关异常

    用于平台验证和数据验证的错误
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "VALIDATION_ERROR", details)


class ConfigurationError(NetworkPlatformError):
    """
    配置相关异常

    用于配置文件的错误
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)


class DataError(NetworkPlatformError):
    """
    数据相关异常

    用于数据处理和输入验证的错误
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATA_ERROR", details)


class ConvergenceError(NetworkPlatformError):
    """
    收敛相关异常

    用于EM算法等迭代方法的收敛失败
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONVERGENCE_ERROR", details)


def handle_exceptions(func):
    """
    异常处理装饰器

    自动捕获和转换异常，提供统一的错误处理
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NetworkPlatformError:
            # 重新抛出平台异常
            raise
        except ValueError as e:
            # 转换数值错误
            raise DataError(f"Invalid data: {str(e)}")
        except RuntimeError as e:
            # 转换运行时错误
            raise ValidationError(f"Runtime error: {str(e)}")
        except Exception as e:
            # 捕获其他未知异常
            raise NetworkPlatformError(f"Unexpected error: {str(e)}", details={
                'original_exception': type(e).__name__,
                'function': func.__name__
            })

    return wrapper


class ErrorHandler:
    """
    错误处理器

    提供错误记录、报告和恢复功能
    """

    def __init__(self):
        self.error_log = []

    def log_error(self, error: NetworkPlatformError):
        """记录错误"""
        error_entry = {
            'timestamp': None,  # 可以添加时间戳
            'error_code': error.error_code,
            'message': error.message,
            'details': error.details
        }
        self.error_log.append(error_entry)

    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        if not self.error_log:
            return {'total_errors': 0, 'error_types': {}}

        error_types = {}
        for entry in self.error_log:
            error_type = entry['error_code']
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'most_common_error': max(error_types.items(), key=lambda x: x[1]) if error_types else None
        }

    def clear_log(self):
        """清除错误日志"""
        self.error_log.clear()


# 全局错误处理器实例
global_error_handler = ErrorHandler()


def get_global_error_handler() -> ErrorHandler:
    """获取全局错误处理器"""
    return global_error_handler


