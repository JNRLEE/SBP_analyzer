"""
SBP Analyzer Package

用於機器學習模型分析與監控的工具包，專注於吞嚥障礙評估聲音數據。
支持與MicDysphagiaFramework整合，提供豐富的模型分析與監控功能。
"""

# 版本信息
__version__ = '0.1.0'

# 從各個模塊導出主要的類和函數
from data_loader import ExperimentLoader, HookDataLoader
from metrics import distribution_metrics
from visualization import distribution_plots, performance_plots
from utils import file_utils, tensor_utils
from interfaces import analyzer_interface

# 導出公開的API
__all__ = [
    # 數據載入器
    'ExperimentLoader',
    'HookDataLoader',
    
    # 指標計算
    'distribution_metrics',
    
    # 視覺化工具
    'distribution_plots',
    'performance_plots',
    
    # 工具函數
    'file_utils',
    'tensor_utils',
    
    # 接口
    'analyzer_interface',
]
