"""
SBP Analyzer Package

用於機器學習模型分析與監控的工具包，專注於吞嚥障礙評估聲音數據。
支持與MicDysphagiaFramework整合，提供豐富的模型分析與監控功能。
"""

# 版本信息
__version__ = '0.1.0'

# 從各個模塊導出主要的類和函數

# 從回調模塊導出回調
from .callbacks import (
    BaseCallback,
    DataMonitorCallback,
    TensorBoardCallback,
    DataDriftCallback,
    FeatureMonitorCallback,
    OutlierDetectionCallback,
    ModelAnalyticsCallback
)

# 從模型鉤子模塊導出鉤子工具
from .model_hooks import (
    ActivationHook,
    GradientHook,
    ModelHookManager
)

# 導出公開的API
__all__ = [
    # 回調
    'BaseCallback',
    'DataMonitorCallback',
    'TensorBoardCallback',
    'DataDriftCallback',
    'FeatureMonitorCallback',
    'OutlierDetectionCallback',
    'ModelAnalyticsCallback',
    
    # 模型鉤子
    'ActivationHook',
    'GradientHook',
    'ModelHookManager',
]
