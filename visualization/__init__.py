"""
visualization 模組

此模組包含以下主要組件：

Classes:
    BasePlotter: 繪圖器基類，提供所有繪圖器共用的基本功能。

Attributes:
    output_dir (str): 圖表保存的輸出目錄。
    style (str): Matplotlib 繪圖風格。
    figsize (Tuple[int, int]): 圖表大小。
    dpi (int): 圖表 DPI。 (plotter.py)
    DistributionPlotter: 分佈視覺化器，用於繪製各種分佈相關的圖表。

此類提供多種方法來視覺化權重、激活值和其他張量數據的分佈情況。

Attributes:
    output_dir (str): 圖表保存的輸出目錄。
    style (str): Matplotlib 繪圖風格。
    figsize (Tuple[int, int]): 圖表大小。
    dpi (int): 圖表 DPI。 (distribution_plots.py)
    ModelStructurePlotter: 模型結構視覺化工具，用於繪製模型架構圖、參數分布等圖表。

Attributes:
    default_figsize (Tuple[int, int]): 默認圖表大小。
    default_dpi (int): 默認解析度。
    layer_colors (Dict[str, str]): 不同層類型的顏色映射。 (model_structure_plots.py)
    PerformancePlotter: 性能視覺化器，用於繪製訓練過程中的性能指標和動態。

此類提供多種方法來視覺化訓練過程中的指標變化，如損失曲線、學習率變化等。

Attributes:
    output_dir (str): 圖表保存的輸出目錄。
    style (str): Matplotlib 繪圖風格。
    figsize (Tuple[int, int]): 圖表大小。
    dpi (int): 圖表 DPI。 (performance_plots.py)

Functions:
    visualize_activation_across_samples: 視覺化不同樣本間的激活值變化。

Args:
    activations: 層級激活值張量
    layer_name: 層名稱，用於圖表標題
    max_samples: 最多顯示的樣本數量
    save_path: 保存圖表的路徑，若為None則不保存
    
Returns:
    matplotlib圖表物件 (layer_visualizer.py)
    visualize_activation_distribution: 視覺化層級激活值分佈。

Args:
    activations: 層級激活值張量
    layer_name: 層名稱，用於圖表標題
    save_path: 保存圖表的路徑，若為None則不保存
    
Returns:
    matplotlib圖表物件 (layer_visualizer.py)
    visualize_dead_and_saturated_neurons: 視覺化死亡和飽和神經元。

Args:
    activations: 層級激活值張量
    layer_name: 層名稱，用於圖表標題
    dead_threshold: 判定神經元死亡的最大激活閾值
    saturation_threshold: 判定神經元飽和的閾值
    save_path: 保存圖表的路徑，若為None則不保存
    
Returns:
    matplotlib圖表物件 (layer_visualizer.py)
    visualize_feature_maps: 視覺化卷積層的特徵圖。

Args:
    activations: 卷積層級激活值張量 [batch, channels, height, width]
    layer_name: 層名稱，用於圖表標題
    sample_idx: 要顯示的樣本索引
    max_features: 最多顯示的特徵圖數量
    save_path: 保存圖表的路徑，若為None則不保存
    
Returns:
    matplotlib圖表物件，若輸入不是卷積層則返回None (layer_visualizer.py)
    visualize_layer_metrics: 為層級激活值生成一整套可視化圖表。

Args:
    activations: 層級激活值張量
    layer_name: 層名稱，用於圖表標題和文件名
    output_dir: 保存圖表的目錄
    include_feature_maps: 是否包括特徵圖可視化（僅適用於卷積層）
    
Returns:
    包含所有生成圖表路徑的字典 (layer_visualizer.py)
    visualize_neuron_activity: 視覺化神經元活躍度，顯示最活躍和最不活躍的神經元。

Args:
    activations: 層級激活值張量
    layer_name: 層名稱，用於圖表標題
    top_k: 顯示的頂部和底部神經元數量
    threshold: 神經元被視為活躍的激活閾值
    save_path: 保存圖表的路徑，若為None則不保存
    
Returns:
    matplotlib圖表物件 (layer_visualizer.py)


完整的模組索引請參見：FUNCTION_CLASS_INDEX.md
"""

from .plotter import BasePlotter
from .model_structure_plots import ModelStructurePlotter
from .distribution_plots import DistributionPlotter
from .performance_plots import PerformancePlotter

__all__ = [
    'BasePlotter',
    'ModelStructurePlotter',
    'DistributionPlotter',
    'PerformancePlotter'
]