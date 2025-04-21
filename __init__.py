"""
SBP Analyzer Package

用於機器學習模型分析的工具包，專注於吞嚥障礙評估聲音數據的離線分析。
支持基於MicDysphagiaFramework實驗結果的深入分析。
"""

# 版本信息
__version__ = '0.1.0'

# 從各個模塊導出主要的類和函數

# 從分析器模塊導出分析器
from analyzer.base_analyzer import BaseAnalyzer
from analyzer.model_structure_analyzer import ModelStructureAnalyzer
from analyzer.training_dynamics_analyzer import TrainingDynamicsAnalyzer
from analyzer.intermediate_data_analyzer import IntermediateDataAnalyzer

# 從數據加載器模塊導出加載工具
from data_loader.base_loader import BaseLoader
from data_loader.experiment_loader import ExperimentLoader
from data_loader.hook_data_loader import HookDataLoader

# 從報告生成器模塊導出報告工具
from reporter.report_generator import ReportGenerator

# 導出公開的API
__all__ = [
    # 分析器
    'BaseAnalyzer',
    'ModelStructureAnalyzer',
    'TrainingDynamicsAnalyzer',
    'IntermediateDataAnalyzer',
    
    # 數據加載器
    'BaseLoader',
    'ExperimentLoader',
    'HookDataLoader',
    
    # 報告生成器
    'ReportGenerator',
]
