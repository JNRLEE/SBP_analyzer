"""
SBP Analyzer Package

用於機器學習模型分析與監控的工具包，專注於吞嚥障礙評估聲音數據。
支持與MicDysphagiaFramework整合，提供豐富的模型分析與監控功能。
"""

# 版本信息
__version__ = '0.1.0'

# 設置包導入路徑，確保測試環境也能正確導入
import os
import sys

# 從analyzer模塊導出分析器
from .analyzer.base_analyzer import BaseAnalyzer
from .analyzer.model_structure_analyzer import ModelStructureAnalyzer
from .analyzer.training_dynamics_analyzer import TrainingDynamicsAnalyzer
from .analyzer.intermediate_data_analyzer import IntermediateDataAnalyzer

# 從visualization模塊導出繪圖工具
from .visualization.plotter import BasePlotter
from .visualization.model_structure_plots import ModelStructurePlotter
from .visualization.distribution_plots import DistributionPlotter
from .visualization.performance_plots import PerformancePlotter

# 從data_loader模塊導出數據加載工具
from .data_loader.hook_data_loader import HookDataLoader
from .data_loader.experiment_loader import ExperimentLoader

# 從reporter模塊導出報告生成工具
from .reporter.report_generator import ReportGenerator

# 從interfaces模塊導出用戶介面
from .interfaces.analyzer_interface import SBPAnalyzer, AnalysisResults

# 從metrics模塊導出主要的度量標準
from .metrics.layer_activity_metrics import (
    calculate_activation_statistics,
    calculate_activation_sparsity,
    detect_dead_neurons,
    calculate_layer_similarity
)

# 導出公開的API
__all__ = [
    # 分析器
    'BaseAnalyzer',
    'ModelStructureAnalyzer',
    'TrainingDynamicsAnalyzer',
    'IntermediateDataAnalyzer',
    
    # 繪圖工具
    'BasePlotter',
    'ModelStructurePlotter',
    'DistributionPlotter',
    'PerformancePlotter',
    
    # 數據加載工具
    'HookDataLoader',
    'ExperimentLoader',
    
    # 報告生成工具
    'ReportGenerator',
    
    # 用戶介面
    'SBPAnalyzer',
    'AnalysisResults',
    
    # 主要度量標準
    'calculate_activation_statistics',
    'calculate_activation_sparsity',
    'detect_dead_neurons',
    'calculate_layer_similarity'
]
