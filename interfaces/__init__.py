"""
接口模組包。

此模組包含用於整合各種分析器並提供統一接口的功能。

Classes:
    SBPAnalyzer: SBP分析器的主接口類別。
    AnalysisResults: 封裝分析結果的類別。
"""

from .analyzer_interface import SBPAnalyzer, AnalysisResults

__all__ = [
    'SBPAnalyzer',
    'AnalysisResults'
]
