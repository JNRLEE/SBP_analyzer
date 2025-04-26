"""
interfaces 模組

此模組包含以下主要組件：

Classes:
    AnalysisResults: 封裝分析結果的類別。

此類別用於整合和存儲來自不同分析器的分析結果，並提供方便的訪問方法。

Attributes:
    model_structure_results (Dict): 模型結構分析結果。
    training_dynamics_results (Dict): 訓練動態分析結果。
    intermediate_data_results (Dict): 中間層數據分析結果。
    inference_results (Dict): 推理性能分析結果。
    plots (Dict[str, Any]): 生成的圖表。 (analyzer_interface.py)
    SBPAnalyzer: SBP分析器的主接口類。

此類整合了各種不同類型的分析器，並提供了統一的接口。

Attributes:
    experiment_dir (str): 實驗結果的目錄路徑。
    experiment_loader (ExperimentLoader): 實驗數據加載器。
    model_structure_analyzer (ModelStructureAnalyzer): 模型結構分析器。
    training_dynamics_analyzer (TrainingDynamicsAnalyzer): 訓練動態分析器。
    intermediate_data_analyzer (IntermediateDataAnalyzer): 中間層數據分析器。
    inference_analyzer (InferenceAnalyzer): 推理性能分析器。
    results (AnalysisResults): 分析結果。 (analyzer_interface.py)


完整的模組索引請參見：FUNCTION_CLASS_INDEX.md
"""

from .analyzer_interface import SBPAnalyzer, AnalysisResults

__all__ = [
    'SBPAnalyzer',
    'AnalysisResults'
]