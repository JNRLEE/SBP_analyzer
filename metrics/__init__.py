"""
metrics 模組

此模組包含以下主要組件：

Functions:
    _calculate_feature_vectors_coherence: 計算特徵向量之間的一致性

Args:
    flattened: 已展平的張量，形狀為 [batch, channels/features, flattened_dim]
    
Returns:
    一致性指標 (layer_activity_metrics.py)
    _calculate_improvement_rate: 計算指標的改進速率。

Args:
    values (List[float]): 值序列。
    is_higher_better (bool, optional): 是否指標值越高越好。默認為True。
    
Returns:
    float: 改進速率。 (performance_metrics.py)
    _calculate_moving_average: 計算移動平均。

Args:
    values (List[float]): 值序列。
    window_size (int, optional): 窗口大小。默認為5。
    
Returns:
    List[float]: 移動平均序列。 (performance_metrics.py)
    _detect_spikes: 檢測數據中的異常尖峰。

Args:
    values (List[float]): 值序列。
    window_size (int, optional): 檢測窗口大小。默認為3。
    threshold (float, optional): 尖峰判定Z分數閾值。默認為2 (performance_metrics.py)
    _find_convergence_point: 查找收斂點。

Args:
    values (List[float]): 值序列。
    window_size (int, optional): 窗口大小。默認為5。
    threshold (float, optional): 收斂閾值。默認為0 (performance_metrics.py)
    analyze_layer_relationships: # 分析不同層之間的相互關係

此函數分析神經網絡中不同層之間的關係，包括相關性、互信息和結構相似性等指標。

Args:
    activations_dict: 層名稱到激活值的字典
    reference_layer: 參考層名稱，如果提供，則分析所有層與此層的關係
    metrics: 要計算的關係指標列表
    
Returns:
    包含層間關係分析結果的字典，包括:
        - layer_pairs: 分析的層對列表
        - metrics: 每種度量的值列表
        - layer_graph: 層之間關係的圖表示 (layer_activity_metrics.py)
    analyze_learning_efficiency: 分析模型的學習效率。

Args:
    loss_values (List[float]): 損失值序列。
    epochs (List[int], optional): 對應的輪次。默認為None，使用從0開始的序列。
    iterations (List[int], optional): 對應的迭代次數。默認為None。
    
Returns:
    Dict[str, Any]: 包含學習效率分析結果的字典。 (performance_metrics.py)
    analyze_loss_curve: 分析損失曲線的收斂性、穩定性等特性。

Args:
    loss_values (List[float]): 訓練損失值序列。
    val_loss_values (Optional[List[float]], optional): 驗證損失值序列。默認為None。
    window_size (int, optional): 移動平均窗口大小。默認為5。
    convergence_threshold (float, optional): 收斂閾值。默認為0 (performance_metrics.py)
    analyze_lr_schedule_impact: 分析學習率調度對訓練的影響。

Args:
    loss_values (List[float]): 損失值序列。
    learning_rates (List[float]): 對應的學習率序列。
    
Returns:
    Dict[str, Any]: 包含學習率影響分析結果的字典。 (performance_metrics.py)
    analyze_metric_curve: 分析評估指標曲線的特性。

Args:
    metric_values (List[float]): 訓練指標值序列。
    val_metric_values (Optional[List[float]], optional): 驗證指標值序列。默認為None。
    is_higher_better (bool, optional): 是否指標值越高越好。默認為True。
    window_size (int, optional): 移動平均窗口大小。默認為5。
    convergence_threshold (float, optional): 收斂閾值。默認為0 (performance_metrics.py)
    calculate_activation_dynamics: 計算不同輪次間的激活值變化。

Args:
    epoch_activations: 輪次到激活值的字典
    
Returns:
    包含動態變化指標的字典 (layer_activity_metrics.py)
    calculate_activation_saturation: 計算激活值的飽和度，即接近最大值的比例。

Args:
    activations: 層級激活值張量
    saturation_threshold: 飽和閾值
    
Returns:
    飽和度 (0-1之間的值) (layer_activity_metrics.py)
    calculate_activation_sparsity: 計算激活值的稀疏度，即接近零的值的比例。

Args:
    activations: 層級激活值張量
    threshold: 判定為零的閾值
    
Returns:
    稀疏度 (0-1之間的值，1表示全為零) (layer_activity_metrics.py)
    calculate_activation_statistics: 計算激活值的統計指標。這個函數是 compute_activation_distribution 的擴展版本，
增加了更多的統計指標。

Args:
    activations: 層級激活值張量
    
Returns:
    包含統計指標的字典 (layer_activity_metrics.py)
    calculate_distribution_entropy: 計算分布的香農熵。

Args:
    dist: 輸入分布數據
    bins: 用於構建直方圖的bins數量
    
Returns:
    分布的香農熵值 (以位元為單位) (distribution_metrics.py)
    calculate_distribution_similarity: 使用指定的方法計算兩個分布之間的相似度或距離。

Args:
    dist1: 第一個分布的數據
    dist2: 第二個分布的數據
    method: 相似度計算方法，選項包括：
        - 'kl_divergence': KL散度 (值越小越相似)
        - 'js_divergence': JS散度 (值越小越相似)
        - 'wasserstein': Wasserstein距離 (值越小越相似)
        - 'correlation': Pearson相關係數 (值越接近1越相似)
    
Returns:
    兩個分布之間的相似度或距離度量 (distribution_metrics.py)
    calculate_distribution_stats: 計算分布的各種統計量。

Args:
    dist: 輸入分布數據
    
Returns:
    包含統計量的字典:
        - mean: 平均值
        - std: 標準差
        - min: 最小值
        - max: 最大值
        - median: 中位數
        - skewness: 偏度
        - kurtosis: 峰度 (distribution_metrics.py)
    calculate_effective_rank: 計算激活值矩陣的有效秩，用於衡量特徵的多樣性。

Args:
    activations: 層級激活值張量
    
Returns:
    有效秩 (layer_activity_metrics.py)
    calculate_feature_coherence: 計算特徵間的一致性，衡量不同通道/特徵之間的相關程度。

Args:
    activations: 層級激活值張量，形狀為 [batch, channels, height, width] 或 [batch, seq_len, features]
    
Returns:
    一致性指標 (layer_activity_metrics.py)
    calculate_histogram_intersection: 計算兩個分布之間的直方圖交叉值，用來衡量分布的相似度。

Args:
    dist1: 第一個分布的數據
    dist2: 第二個分布的數據
    bins: 用於構建直方圖的bins數量
    
Returns:
    兩個直方圖的交叉值 (0到1之間，1表示完全相同) (distribution_metrics.py)
    calculate_layer_similarity: 計算兩個層的激活值之間的相似度。

Args:
    activations1: 第一個層的激活值
    activations2: 第二個層的激活值
    
Returns:
    包含不同相似度指標的字典 (layer_activity_metrics.py)
    compare_tensor_distributions: 比較兩個張量的分布，可以沿著指定維度進行比較。

Args:
    tensor1: 第一個張量
    tensor2: 第二個張量
    dim: 要比較的維度，若為None則比較整個張量
    methods: 要使用的比較方法列表
    
Returns:
    包含各種相似度度量的字典 (distribution_metrics.py)
    compute_activation_distribution: 計算激活值分佈的基本屬性，包括零值、正值和負值的比例。

Args:
    activations: 層級激活值張量
    
Returns:
    包含分佈特性的字典 (layer_activity_metrics.py)
    compute_layer_similarity: 計算兩組層級激活值之間的相似度。

Args:
    activations1: 第一組層級激活值
    activations2: 第二組層級激活值
    method: 相似度計算方法，可選'cosine'或'correlation'
    
Returns:
    相似度值 (layer_activity_metrics.py)
    compute_neuron_activity: 計算神經元活躍度指標，包括活躍神經元的數量和平均活躍度。

Args:
    activations: 層級激活值張量
    threshold: 神經元被視為活躍的激活閾值
    
Returns:
    包含神經元活躍度指標的字典 (layer_activity_metrics.py)
    compute_saturation_metrics: 計算神經元飽和度指標，識別飽和神經元(輸出始終接近最大值)。

Args:
    activations: 層級激活值張量
    saturation_threshold: 判定神經元飽和的閾值
    
Returns:
    包含飽和度指標的字典 (layer_activity_metrics.py)
    compute_stability_over_batches: 計算多個批次激活值的穩定性指標。

Args:
    batch_activations: 多個批次的激活值列表，每個元素是一個批次的激活張量
    
Returns:
    包含穩定性指標的字典，如方差、變異係數、批次間相似度等 (layer_activity_metrics.py)
    detect_activation_anomalies: 檢測神經網路層的異常激活模式。

此函數分析激活值張量，檢測多種潛在異常：
1 (layer_activity_metrics.py)
    detect_dead_neurons: 檢測層中的死亡神經元（始終輸出接近零的神經元）。

Args:
    activations: 層級激活值張量
    threshold: 判定神經元死亡的最大激活閾值
    return_indices: 是否返回死亡神經元的索引
    
Returns:
    包含死亡神經元信息的字典 (layer_activity_metrics.py)
    detect_oscillations: 檢測訓練曲線中的震盪現象。

Args:
    values (List[float]): 值序列。
    window_size (int, optional): 檢測窗口大小。默認為3。
    threshold (float, optional): 震盪判定閾值。默認為0 (performance_metrics.py)
    detect_overfitting: 檢測過擬合現象。

Args:
    train_values (List[float]): 訓練值序列。
    val_values (List[float]): 驗證值序列。
    window_size (int, optional): 移動平均窗口大小。默認為5。
    threshold (float, optional): 過擬合判定閾值。默認為0 (performance_metrics.py)
    detect_plateaus: 檢測訓練曲線中的平台期。

Args:
    values (List[float]): 值序列。
    window_size (int, optional): 檢測窗口大小。默認為5。
    threshold (float, optional): 平台期判定閾值。默認為0 (performance_metrics.py)
    detect_training_anomalies: 檢測訓練過程中的異常。

Args:
    loss_values (List[float]): 訓練損失值序列。
    val_loss_values (Optional[List[float]], optional): 驗證損失值序列。默認為None。
    metrics (Optional[Dict[str, List[float]]], optional): 其他指標序列的字典。默認為None。
    
Returns:
    Dict[str, Any]: 包含異常檢測結果的字典。 (performance_metrics.py)


完整的模組索引請參見：FUNCTION_CLASS_INDEX.md
"""

from .performance_metrics import (
    analyze_loss_curve,
    analyze_metric_curve,
    analyze_learning_efficiency,
    analyze_lr_schedule_impact,
    detect_training_anomalies,
    detect_overfitting,
    detect_oscillations,
    detect_plateaus
)
from .distribution_metrics import (
    calculate_histogram_intersection,
    calculate_distribution_similarity,
    compare_tensor_distributions,
    calculate_distribution_stats,
    calculate_distribution_entropy
)

__all__ = [
    # 性能指標
    'analyze_loss_curve',
    'analyze_metric_curve',
    'analyze_learning_efficiency',
    'analyze_lr_schedule_impact',
    'detect_training_anomalies',
    'detect_overfitting',
    'detect_oscillations',
    'detect_plateaus',
    
    # 分布指標
    'calculate_histogram_intersection',
    'calculate_distribution_similarity',
    'compare_tensor_distributions',
    'calculate_distribution_stats',
    'calculate_distribution_entropy'
]