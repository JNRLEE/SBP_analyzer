"""
analyzer 模組

此模組包含以下主要組件：

Classes:
    AdaptiveThresholdAnalyzer: 自適應閾值分析器類

此分析器擴展了中間數據分析器的功能，通過分析每一層激活的分佈特性，
自適應地調整用於檢測死亡神經元和飽和神經元的閾值。支持多種自適應
方法，包括基於百分位數、分佈特性和混合方法。

Args:
    experiment_name: 實驗名稱
    output_dir: 輸出目錄路徑
    adaptive_method: 自適應方法，可選 'percentile'、'distribution' 或 'hybrid'
    adaptation_strength: 自適應強度，範圍 [0, 1]，越高表示越強的自適應調整
    base_dead_threshold: 基礎死亡神經元閾值，預設為 0 (adaptive_threshold_analyzer.py)
    BaseAnalyzer: 分析器的抽象基礎類別，提供通用接口和方法。

這個類別定義了所有分析器共有的屬性和方法。子類別應重寫抽象方法來實現特定的分析功能。

Attributes:
    experiment_dir (str): 實驗結果的目錄路徑。
    config (Dict): 實驗配置。
    logger (logging (base_analyzer.py)
    InferenceAnalyzer: 推理分析器，用於分析模型推理性能和行為。

此分析器主要關注模型推理的時間效率、資源使用、輸出特性和準確度等方面。

Attributes:
    experiment_dir (str): 實驗結果的目錄路徑。
    inference_logs (Dict): 儲存推理日誌資料。
    model_structure (Dict): 模型結構資訊。
    results (Dict): 分析結果。
    device (str): 執行推理的設備。 (inference_analyzer.py)
    IntermediateDataAnalyzer: 中間層數據分析器，用於分析模型中間層輸出的特性。

此分析器關注模型訓練過程中通過hooks獲取的中間層數據，分析其分布、統計特性和變化趨勢。

Attributes:
    experiment_dir (str): 實驗結果的目錄路徑。
    hooks_dir (str): 存放hooks數據的目錄。
    available_epochs (List[int]): 可用的輪次列表。
    available_layers (Dict[int, List[str]]): 每個輪次可用的層列表。
    experiment_name (str): 實驗名稱。
    output_dir (str): 輸出目錄路徑。
    dead_neuron_threshold (float): 死亡神經元的閾值。
    saturated_neuron_threshold (float): 飽和神經元的閾值。
    layer_results (Dict): 依照層名稱組織的分析結果。 (intermediate_data_analyzer.py)
    LayerCorrelationAnalysis: 神經網絡層內神經元相關性分析類

此類提供方法分析神經網絡層內神經元之間的相關性模式，包括相關性矩陣計算、
神經元群聚識別、主成分分析等。 (layer_correlation_analysis.py)
    LayerRelationshipAnalysis: 神經網絡層間關係分析類

此類提供方法分析神經網絡中不同層之間的關係，包括相關性、
信息流動、表示相似度等。 (layer_relationship_analysis.py)
    ModelStructureAnalyzer: 模型結構分析器，用於分析模型結構和參數分布。

此分析器主要關注模型架構、層級結構、參數分布和其他結構特性。

Attributes:
    experiment_dir (str): 實驗結果的目錄路徑。
    model_structure (Dict): 儲存解析後的模型結構。
    model_stats (Dict): 儲存模型統計信息。 (model_structure_analyzer.py)
    NumpyEncoder: 專門處理NumPy類型的JSON編碼器 (base_analyzer.py)
    TrainingDynamicsAnalyzer: 訓練動態分析器，用於分析模型訓練過程中的性能指標變化。

此分析器關注訓練過程中損失函數、評估指標等數據的變化趨勢，可用於評估模型收斂性、穩定性等。

Attributes:
    experiment_dir (str): 實驗結果的目錄路徑。
    training_history (Dict): 訓練歷史記錄。
    metrics_data (pd (training_dynamics_analyzer.py)


完整的模組索引請參見：FUNCTION_CLASS_INDEX.md
"""

from .base_analyzer import BaseAnalyzer
from .model_structure_analyzer import ModelStructureAnalyzer
from .training_dynamics_analyzer import TrainingDynamicsAnalyzer
from .intermediate_data_analyzer import IntermediateDataAnalyzer
from .inference_analyzer import InferenceAnalyzer

__all__ = [
    'BaseAnalyzer',
    'ModelStructureAnalyzer',
    'TrainingDynamicsAnalyzer',
    'IntermediateDataAnalyzer',
    'InferenceAnalyzer'
]