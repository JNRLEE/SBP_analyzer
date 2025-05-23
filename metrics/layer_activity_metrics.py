"""
# 神經網絡層級活動度指標計算模組
# 
# 此模組負責計算神經網絡各層激活值的統計指標，包括：
# - 激活分佈分析（正/負/零值比例）
# - 神經元活躍度指標
# - 死亡神經元檢測
# - 飽和度指標計算
# - 層級響應穩定性分析
# - 異常激活模式檢測
# - 層間關係分析
#
# 使用說明:
# ```python
# import torch
# from metrics.layer_activity_metrics import detect_activation_anomalies, analyze_layer_relationships
#
# # 檢測異常激活模式
# activations = torch.randn(32, 64, 16, 16)
# anomalies = detect_activation_anomalies(activations)
# if anomalies['has_anomaly']:
#     print(f"檢測到異常，分數: {anomalies['anomaly_score']}")
#
# # 分析層間關係
# layer_activations = {
#     'layer1': torch.randn(16, 100),
#     'layer2': torch.randn(16, 100),
#     'layer3': torch.randn(16, 100)
# }
# relationships = analyze_layer_relationships(layer_activations)
# print(f"高相關層對: {relationships['high_correlation_pairs']}")
# ```
"""

import torch
import numpy as np
import scipy.stats
from typing import Dict, Any, Tuple, List, Union, Optional


def compute_activation_distribution(activations: torch.Tensor) -> Dict[str, float]:
    """
    計算激活值分佈的基本屬性，包括零值、正值和負值的比例。
    
    Args:
        activations: 層級激活值張量
        
    Returns:
        包含分佈特性的字典
    """
    # 確保輸入是張量
    if not isinstance(activations, torch.Tensor):
        activations = torch.tensor(activations)
    
    # 將張量轉換為一維向量
    flat_activations = activations.reshape(-1)
    total_elements = flat_activations.numel()
    
    # 計算零值、正值和負值的數量
    zero_count = torch.sum(flat_activations == 0.0).item()
    positive_count = torch.sum(flat_activations > 0.0).item()
    negative_count = torch.sum(flat_activations < 0.0).item()
    
    # 計算比例
    zero_fraction = zero_count / total_elements
    positive_fraction = positive_count / total_elements
    negative_fraction = negative_count / total_elements
    
    return {
        'zero_fraction': zero_fraction,
        'positive_fraction': positive_fraction,
        'negative_fraction': negative_fraction
    }


def compute_neuron_activity(activations: torch.Tensor, threshold: float = 0.0) -> Dict[str, Union[int, float]]:
    """
    計算神經元活躍度指標，包括活躍神經元的數量和平均活躍度。
    
    Args:
        activations: 層級激活值張量
        threshold: 神經元被視為活躍的激活閾值
        
    Returns:
        包含神經元活躍度指標的字典
    """
    # 確定張量的形狀和維度
    tensor_shape = activations.shape
    
    # 計算神經元維度上的平均激活值
    if len(tensor_shape) == 4:  # 卷積層 [batch, channels, height, width]
        # 對每個通道(卷積核)計算平均激活值
        neuron_means = activations.mean(dim=[0, 2, 3])  # 結果形狀: [channels]
    elif len(tensor_shape) == 2:  # 全連接層 [batch, neurons]
        # 對每個神經元計算平均激活值
        neuron_means = activations.mean(dim=0)  # 結果形狀: [neurons]
    else:
        raise ValueError(f"不支援的激活值張量形狀: {tensor_shape}")
    
    # 計算活躍神經元的數量和比例
    active_neurons = torch.sum(neuron_means > threshold).item()
    total_neurons = neuron_means.numel()
    active_ratio = active_neurons / total_neurons if total_neurons > 0 else 0.0
    
    # 計算平均活躍度
    mean_activity = neuron_means.mean().item()
    
    return {
        'active_neurons': active_neurons,
        'total_neurons': total_neurons,
        'active_ratio': active_ratio,
        'mean_activity': mean_activity
    }


def detect_dead_neurons(activations: torch.Tensor, 
                         threshold: float = 1e-6,
                         return_indices: bool = False) -> Dict[str, Any]:
    """
    檢測層中的死亡神經元（始終輸出接近零的神經元）。
    
    Args:
        activations: 層級激活值張量
        threshold: 判定神經元死亡的最大激活閾值
        return_indices: 是否返回死亡神經元的索引
        
    Returns:
        包含死亡神經元信息的字典
    """
    # 確定張量的形狀和維度
    tensor_shape = activations.shape
    
    # 計算神經元維度上的最大激活值
    if len(tensor_shape) == 4:  # 卷積層 [batch, channels, height, width]
        # 對每個通道(卷積核)計算最大激活值
        neuron_max = activations.abs().max(dim=0)[0].max(dim=1)[0].max(dim=1)[0]  # 結果形狀: [channels]
    elif len(tensor_shape) == 2:  # 全連接層 [batch, neurons]
        # 對每個神經元計算最大激活值
        neuron_max = activations.abs().max(dim=0)[0]  # 結果形狀: [neurons]
    elif len(tensor_shape) == 3:  # 序列/Transformer層 [batch, seq_len, features]
        # 對每個特徵計算最大激活值
        neuron_max = activations.abs().max(dim=0)[0].max(dim=0)[0]  # 結果形狀: [features]
    else:
        # 嘗試平坦化處理，保留最後一個維度作為神經元維度
        try:
            batch_size = tensor_shape[0]
            flattened = activations.reshape(batch_size, -1)
            neuron_max = flattened.abs().max(dim=0)[0]
        except Exception as e:
            raise ValueError(f"不支援的激活值張量形狀: {tensor_shape}，錯誤：{str(e)}")
    
    # 檢測死亡神經元
    dead_mask = neuron_max <= threshold
    dead_count = torch.sum(dead_mask).item()
    total_neurons = neuron_max.numel()
    dead_ratio = dead_count / total_neurons if total_neurons > 0 else 0.0
    
    result = {
        'dead_count': dead_count,
        'total_neurons': total_neurons,
        'dead_ratio': dead_ratio
    }
    
    # 如果需要，返回死亡神經元的索引
    if return_indices:
        result['dead_indices'] = torch.nonzero(dead_mask).squeeze().tolist()
        if isinstance(result['dead_indices'], int) and dead_count == 1:
            result['dead_indices'] = [result['dead_indices']]
        elif dead_count == 0:
            result['dead_indices'] = []
    
    return result


def compute_saturation_metrics(activations: torch.Tensor, 
                               saturation_threshold: float = 0.95) -> Dict[str, Any]:
    """
    計算神經元飽和度指標，識別飽和神經元(輸出始終接近最大值)。
    
    Args:
        activations: 層級激活值張量
        saturation_threshold: 判定神經元飽和的閾值
        
    Returns:
        包含飽和度指標的字典
    """
    # 確定張量的形狀和維度
    tensor_shape = activations.shape
    
    # 標準化激活值到[0,1]區間，便於計算飽和度
    if activations.min() < 0 or activations.max() > 1:
        # 如果激活值不在[0,1]範圍內，進行標準化
        normalized = (activations - activations.min()) / (activations.max() - activations.min() + 1e-8)
    else:
        normalized = activations
    
    # 計算神經元維度上的平均激活值
    if len(tensor_shape) == 4:  # 卷積層 [batch, channels, height, width]
        # 對每個通道(卷積核)計算平均激活值
        neuron_means = normalized.mean(dim=[0, 2, 3])  # 結果形狀: [channels]
    elif len(tensor_shape) == 2:  # 全連接層 [batch, neurons]
        # 對每個神經元計算平均激活值
        neuron_means = normalized.mean(dim=0)  # 結果形狀: [neurons]
    elif len(tensor_shape) == 3:  # 序列/Transformer層 [batch, seq_len, features]
        # 對每個特徵計算平均激活值
        neuron_means = normalized.mean(dim=[0, 1])  # 結果形狀: [features]
    else:
        # 嘗試平坦化處理
        try:
            batch_size = tensor_shape[0]
            flattened = normalized.reshape(batch_size, -1)
            neuron_means = flattened.mean(dim=0)
        except Exception as e:
            raise ValueError(f"不支援的激活值張量形狀: {tensor_shape}，錯誤：{str(e)}")
    
    # 檢測飽和神經元
    saturated_mask = neuron_means >= saturation_threshold
    saturated_count = torch.sum(saturated_mask).item()
    total_neurons = neuron_means.numel()
    saturation_ratio = saturated_count / total_neurons if total_neurons > 0 else 0.0
    
    result = {
        'saturated_count': saturated_count,
        'total_neurons': total_neurons,
        'saturation_ratio': saturation_ratio,
        'mean_saturation': neuron_means.mean().item()
    }
    
    return result


def compute_layer_similarity(activations1: torch.Tensor, 
                             activations2: torch.Tensor, 
                             method: str = 'cosine') -> float:
    """
    計算兩組層級激活值之間的相似度。
    
    Args:
        activations1: 第一組層級激活值
        activations2: 第二組層級激活值
        method: 相似度計算方法，可選'cosine'或'correlation'
        
    Returns:
        相似度值
    """
    # 將張量展平為一維向量
    flat1 = activations1.reshape(-1)
    flat2 = activations2.reshape(-1)
    
    # 如果形狀不同，截斷到相同長度
    if flat1.shape[0] != flat2.shape[0]:
        min_length = min(flat1.shape[0], flat2.shape[0])
        flat1 = flat1[:min_length]
        flat2 = flat2[:min_length]
    
    if method == 'cosine':
        # 計算餘弦相似度
        sim = torch.nn.functional.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0))
        return sim.item()
    elif method == 'correlation':
        # 計算皮爾遜相關係數
        centered1 = flat1 - flat1.mean()
        centered2 = flat2 - flat2.mean()
        
        norm1 = torch.norm(centered1)
        norm2 = torch.norm(centered2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        correlation = torch.dot(centered1, centered2) / (norm1 * norm2)
        return correlation.item()
    else:
        raise ValueError(f"不支援的相似度計算方法: {method}")


def compute_stability_over_batches(batch_activations: List[torch.Tensor]) -> Dict[str, float]:
    """
    計算多個批次激活值的穩定性指標。
    
    Args:
        batch_activations: 多個批次的激活值列表，每個元素是一個批次的激活張量
        
    Returns:
        包含穩定性指標的字典，如方差、變異係數、批次間相似度等
    """
    if len(batch_activations) < 2:
        raise ValueError("需要至少兩個批次才能計算穩定性")
    
    # 轉換為numpy數組以便計算
    batches = [t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t for t in batch_activations]
    
    # 計算每個批次的平均激活值
    batch_means = [np.mean(batch) for batch in batches]
    
    # 計算批次平均值的方差
    variance = np.var(batch_means)
    
    # 計算變異係數 (CV = 標準差 / 平均值)
    mean_of_means = np.mean(batch_means)
    cv = np.std(batch_means) / (mean_of_means + 1e-10)
    
    # 計算相鄰批次間的相似度
    similarities = []
    for i in range(len(batches) - 1):
        flat1 = batches[i].reshape(-1)
        flat2 = batches[i + 1].reshape(-1)
        min_len = min(len(flat1), len(flat2))
        similarity = np.corrcoef(flat1[:min_len], flat2[:min_len])[0, 1]
        similarities.append(similarity)
    
    avg_similarity = np.mean(similarities)
    
    return {
        'batch_variance': float(variance),
        'coefficient_of_variation': float(cv),
        'avg_batch_similarity': float(avg_similarity),
        'batch_similarities': [float(s) for s in similarities]
    }


def detect_activation_anomalies(activations: Union[torch.Tensor, np.ndarray], 
                              threshold_z: float = 3.0,
                              skew_threshold: float = 1.0,
                              kurt_threshold: float = 3.0,
                              gradient_threshold: float = 2.0) -> Dict[str, Any]:
    """
    檢測神經網路層的異常激活模式。

    此函數分析激活值張量，檢測多種潛在異常：
    1. 極端值（異常高或低的值）
    2. 分布異常（高偏斜度或峰度）
    3. 梯度異常（鄰近神經元間的激活值變化過大）
    4. 激活值過度集中（不均勻分布）
    
    Args:
        activations: 要分析的激活值張量
        threshold_z: Z分數閾值，用於判定極端值 (默認: 3.0)
        skew_threshold: 偏斜度異常判定閾值 (默認: 1.0)
        kurt_threshold: 峰度異常判定閾值 (默認: 3.0)
        gradient_threshold: 梯度異常判定閾值 (默認: 2.0)
        
    Returns:
        Dict[str, Any]: 包含異常檢測結果的字典，包括：
            - has_anomaly: 是否檢測到異常
            - anomaly_score: 綜合異常分數 (0-1)
            - extreme_value_ratio: 極端值占比
            - skewness: 分布偏斜度
            - kurtosis: 分布峰度
            - skewness_abnormal: 偏斜度是否異常
            - kurtosis_abnormal: 峰度是否異常
            - gradient_abnormal_ratio: 梯度異常的比例
            - concentration_abnormal: 激活值是否過度集中
            - extreme_value_examples: 極端值示例列表
    """
    # 確保輸入為numpy數組
    if isinstance(activations, torch.Tensor):
        act_np = activations.detach().cpu().numpy()
    else:
        act_np = activations.copy()
    
    # 将张量展平成1D数组用于统计分析
    flat_act = act_np.reshape(-1)
    
    # 1. 基本統計量計算
    mean = np.mean(flat_act)
    std = np.std(flat_act)
    min_val = np.min(flat_act)
    max_val = np.max(flat_act)
    
    # 2. 極端值分析
    z_scores = np.abs((flat_act - mean) / (std + 1e-10))
    extreme_mask = z_scores > threshold_z
    extreme_value_ratio = np.sum(extreme_mask) / flat_act.size
    extreme_values = flat_act[extreme_mask]
    
    # 取前10個極端值作為示例 (如果有的話)
    extreme_indices = np.argsort(z_scores)[-10:][::-1]
    extreme_examples = flat_act[extreme_indices].tolist()
    
    # 3. 分布形狀分析
    # 計算偏斜度 (skewness)
    skewness = float(scipy.stats.skew(flat_act))
    
    # 計算峰度 (kurtosis)
    # 注意：scipy.stats.kurtosis 返回的是超出正態分布峰度(3)的部分
    kurtosis = float(scipy.stats.kurtosis(flat_act) + 3)  # 轉換為原始峰度定義
    
    # 判斷偏斜度和峰度是否異常
    skewness_abnormal = abs(skewness) > skew_threshold
    kurtosis_abnormal = kurtosis > kurt_threshold + 3 or kurtosis < 3 - kurt_threshold
    
    # 4. 梯度分析（檢測相鄰神經元間的激活值變化）
    gradient_abnormal_count = 0
    total_gradients = 0
    
    # 對每個維度進行梯度計算
    for axis in range(act_np.ndim):
        # 計算沿著該軸的梯度
        grad_slices = [slice(None)] * act_np.ndim
        grad_slices[axis] = slice(1, None)
        prev_slices = [slice(None)] * act_np.ndim
        prev_slices[axis] = slice(None, -1)
        
        gradients = act_np[tuple(grad_slices)] - act_np[tuple(prev_slices)]
        
        # 標準化梯度
        grad_mean = np.mean(gradients)
        grad_std = np.std(gradients)
        norm_gradients = np.abs((gradients - grad_mean) / (grad_std + 1e-10))
        
        # 計算異常梯度的數量
        gradient_abnormal_count += np.sum(norm_gradients > gradient_threshold)
        total_gradients += gradients.size
    
    gradient_abnormal_ratio = gradient_abnormal_count / (total_gradients + 1e-10)
    
    # 5. 激活值分布集中度分析
    # 使用基尼係數衡量激活值的集中程度
    sorted_act = np.sort(np.abs(flat_act))
    n = len(sorted_act)
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * sorted_act)) / (n * np.sum(sorted_act))
    
    # 判斷激活值是否過度集中 (基尼係數過高)
    concentration_abnormal = gini > 0.8  # 閾值可根據經驗調整
    
    # 6. 計算綜合異常分數 (0-1)
    anomaly_factors = [
        extreme_value_ratio * 2.0,  # 極端值加權
        skewness_abnormal * 0.5,    # 偏斜度異常
        kurtosis_abnormal * 0.5,    # 峰度異常
        gradient_abnormal_ratio,    # 梯度異常
        concentration_abnormal * 0.5 # 集中度異常
    ]
    anomaly_score = min(1.0, sum(anomaly_factors))
    
    # 7. 判斷是否存在異常
    has_anomaly = anomaly_score > 0.3  # 異常閾值可調整
    
    return {
        'has_anomaly': has_anomaly,
        'anomaly_score': float(anomaly_score),
        'extreme_value_ratio': float(extreme_value_ratio),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'skewness_abnormal': bool(skewness_abnormal),
        'kurtosis_abnormal': bool(kurtosis_abnormal),
        'gradient_abnormal_ratio': float(gradient_abnormal_ratio),
        'concentration_abnormal': bool(concentration_abnormal),
        'extreme_value_examples': extreme_examples,
        'gini_coefficient': float(gini),
        'min_value': float(min_val),
        'max_value': float(max_val)
    }


def analyze_layer_relationships(activations_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
                               reference_layer: Optional[str] = None,
                               metrics: List[str] = ['correlation', 'mutual_information', 'structural_similarity']) -> Dict[str, Any]:
    """
    # 分析不同層之間的相互關係
    
    此函數分析神經網絡中不同層之間的關係，包括相關性、互信息和結構相似性等指標。
    
    Args:
        activations_dict: 層名稱到激活值的字典
        reference_layer: 參考層名稱，如果提供，則分析所有層與此層的關係
        metrics: 要計算的關係指標列表
        
    Returns:
        包含層間關係分析結果的字典，包括:
            - layer_pairs: 分析的層對列表
            - metrics: 每種度量的值列表
            - layer_graph: 層之間關係的圖表示
    """
    import skimage.metrics
    from scipy.stats import entropy
    
    if len(activations_dict) < 2:
        return {'error': '需要至少兩個層的激活值進行關係分析'}
    
    # 預處理：將所有層的激活值調整為可比較的格式
    processed_activations = {}
    for layer_name, activation in activations_dict.items():
        # 轉換為numpy陣列
        if isinstance(activation, torch.Tensor):
            activation_np = activation.detach().cpu().numpy()
        else:
            activation_np = np.asarray(activation)
        
        # 對於每個批次樣本，將激活值展平為特徵向量
        if len(activation_np.shape) > 2:
            # 保留批次維度，合併其他維度
            activation_np = activation_np.reshape(activation_np.shape[0], -1)
        
        processed_activations[layer_name] = activation_np
    
    # 確定要比較的層
    layer_names = list(processed_activations.keys())
    
    if reference_layer is not None and reference_layer in layer_names:
        # 如果指定了參考層，所有其他層都與參考層比較
        comparison_pairs = [(reference_layer, layer) for layer in layer_names if layer != reference_layer]
    else:
        # 否則，計算所有可能的層對
        comparison_pairs = [(layer1, layer2) 
                            for i, layer1 in enumerate(layer_names) 
                            for layer2 in layer_names[i+1:]]
    
    # 初始化結果字典
    results = {
        'layer_pairs': [],
        'metrics': {},
        'layer_graph': {}  # 初始化層級關係圖
    }
    
    for metric in metrics:
        results['metrics'][metric] = []
    
    # 初始化層級關係圖
    for layer_name in layer_names:
        results['layer_graph'][layer_name] = {}
    
    # 對每對層計算關係指標
    for layer1, layer2 in comparison_pairs:
        results['layer_pairs'].append((layer1, layer2))
        
        act1 = processed_activations[layer1]
        act2 = processed_activations[layer2]
        
        # 確保批次大小一致
        min_batch = min(act1.shape[0], act2.shape[0])
        act1 = act1[:min_batch]
        act2 = act2[:min_batch]
        
        # 用於存儲各指標的分數
        metric_scores = {}
        
        # 對於每個度量指標計算相應的關係值
        if 'correlation' in metrics:
            # 計算Pearson相關係數
            corr_matrix = np.zeros((min_batch,))
            for i in range(min_batch):
                # 向量可能有不同長度，需要進行處理
                # 方法1：使用較小的維度
                feature_len1 = act1.shape[1]
                feature_len2 = act2.shape[1]
                
                # 使用批次內的向量相似度
                if feature_len1 != feature_len2:
                    # 不同維度的向量需要特殊處理
                    # 我們使用PCA的思想，將較大維度的向量投影到較小維度
                    if feature_len1 > feature_len2:
                        # 計算第一主成分
                        vec1_centered = act1[i] - np.mean(act1[i])
                        # 使用簡化版本，取前feature_len2個元素
                        vec1 = vec1_centered[:feature_len2]
                        vec2 = act2[i] - np.mean(act2[i])
                    else:
                        vec1 = act1[i] - np.mean(act1[i])
                        vec2_centered = act2[i] - np.mean(act2[i])
                        # 使用簡化版本，取前feature_len1個元素
                        vec2 = vec2_centered[:feature_len1]
                else:
                    # 標準化向量
                    vec1 = act1[i] - np.mean(act1[i])
                    vec2 = act2[i] - np.mean(act2[i])
                
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                # 避免除零
                if norm1 > 0 and norm2 > 0:
                    corr = np.dot(vec1, vec2) / (norm1 * norm2)
                    corr_matrix[i] = corr
                else:
                    corr_matrix[i] = 0
            
            avg_corr = np.mean(corr_matrix)
            results['metrics']['correlation'].append(float(avg_corr))
            metric_scores['correlation'] = float(avg_corr)
        
        if 'mutual_information' in metrics:
            # 使用直方圖估算互信息
            mi_values = []
            for i in range(min_batch):
                # 由於向量可能有不同長度，我們使用直方圖方法計算互信息，不需要相同大小
                try:
                    # 將連續值離散化為直方圖
                    hist_bins = 20
                    
                    # 選取數據進行處理
                    x_sample = act1[i]
                    y_sample = act2[i]
                    
                    # 如果向量太大，隨機抽樣以加速計算
                    max_samples = 10000
                    if len(x_sample) > max_samples:
                        indices = np.random.choice(len(x_sample), max_samples, replace=False)
                        x_sample = x_sample[indices]
                    
                    if len(y_sample) > max_samples:
                        indices = np.random.choice(len(y_sample), max_samples, replace=False)
                        y_sample = y_sample[indices]
                    
                    # 創建聯合直方圖
                    hist_2d, _, _ = np.histogram2d(x_sample, y_sample, bins=hist_bins)
                    
                    # 確保和為1以形成聯合概率質量函數
                    hist_sum = np.sum(hist_2d)
                    if hist_sum > 0:
                        hist_2d /= hist_sum
                    
                    # 計算邊緣分布
                    p_x = np.sum(hist_2d, axis=1)
                    p_y = np.sum(hist_2d, axis=0)
                    
                    # 避免log(0)
                    hist_2d = np.where(hist_2d > 0, hist_2d, 1e-10)
                    p_x = np.where(p_x > 0, p_x, 1e-10)
                    p_y = np.where(p_y > 0, p_y, 1e-10)
                    
                    # 計算互信息
                    # I(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
                    outer_product = np.outer(p_x, p_y)
                    mi = np.sum(hist_2d * np.log(hist_2d / outer_product))
                    mi_values.append(mi)
                except:
                    # 出錯時跳過
                    continue
            
            avg_mi = np.mean(mi_values) if mi_values else 0.0
            results['metrics']['mutual_information'].append(float(avg_mi))
            metric_scores['mutual_information'] = float(avg_mi)
        
        if 'structural_similarity' in metrics:
            # 如果原始激活值是卷積特徵圖，使用結構相似性指數(SSIM)
            # 檢查原始維度
            orig_shapes = {
                layer1: activations_dict[layer1].shape,
                layer2: activations_dict[layer2].shape
            }
            
            # 需要4D張量用於SSIM [batch, channels, height, width]
            if len(orig_shapes[layer1]) == 4 and len(orig_shapes[layer2]) == 4:
                # 獲取原始卷積特徵圖
                feat1 = activations_dict[layer1]
                feat2 = activations_dict[layer2]
                
                if isinstance(feat1, torch.Tensor):
                    feat1 = feat1.detach().cpu().numpy()
                if isinstance(feat2, torch.Tensor):
                    feat2 = feat2.detach().cpu().numpy()
                
                # 計算每個批次的結構相似性
                ssim_values = []
                for i in range(min(feat1.shape[0], feat2.shape[0])):
                    # 需要調整兩個特徵圖到相同大小
                    # 選擇較小的尺寸
                    f1 = feat1[i]
                    f2 = feat2[i]
                    
                    c1 = min(f1.shape[0], f2.shape[0])
                    h1, h2 = f1.shape[1], f2.shape[1]
                    w1, w2 = f1.shape[2], f2.shape[2]
                    
                    # 對於每個通道計算SSIM，然後取平均
                    channel_ssim = []
                    
                    for c in range(c1):
                        # 獲取通道
                        ch1 = f1[c]
                        ch2 = f2[c]
                        
                        # 調整大小以匹配
                        if h1 != h2 or w1 != w2:
                            # 使用最簡單的縮放策略：截取較小尺寸
                            min_h = min(h1, h2)
                            min_w = min(w1, w2)
                            ch1_resized = ch1[:min_h, :min_w]
                            ch2_resized = ch2[:min_h, :min_w]
                        else:
                            ch1_resized = ch1
                            ch2_resized = ch2
                        
                        # 確保有足夠的像素進行SSIM計算
                        if ch1_resized.shape[0] > 7 and ch1_resized.shape[1] > 7:
                            try:
                                # 計算相似性，濾波器大小改為7（默認為11，可能對小圖像不合適）
                                ch1_normalized = (ch1_resized - ch1_resized.min()) / (ch1_resized.max() - ch1_resized.min() + 1e-10)
                                ch2_normalized = (ch2_resized - ch2_resized.min()) / (ch2_resized.max() - ch2_resized.min() + 1e-10)
                                
                                ssim = skimage.metrics.structural_similarity(
                                    ch1_normalized, ch2_normalized, 
                                    data_range=1.0, win_size=7
                                )
                                channel_ssim.append(ssim)
                            except Exception as e:
                                # 忽略錯誤
                                continue
                
                    # 計算平均SSIM
                    if channel_ssim:
                        avg_channel_ssim = np.mean(channel_ssim)
                        ssim_values.append(avg_channel_ssim)
                
                # 返回所有批次和通道的平均SSIM
                avg_ssim = np.mean(ssim_values) if ssim_values else 0.0
                results['metrics']['structural_similarity'].append(float(avg_ssim))
                metric_scores['structural_similarity'] = float(avg_ssim)
            else:
                # 對於非卷積特徵，標記為不適用
                results['metrics']['structural_similarity'].append(0.0)
                metric_scores['structural_similarity'] = 0.0
        
        # 更新層級關係圖
        # 使用相關性度量（如果可用）或第一個可用的度量
        relation_score = metric_scores.get('correlation', 
                                          metric_scores.get(next(iter(metric_scores), 0.0)))
        
        # 將關係添加到圖中（雙向）
        results['layer_graph'][layer1][layer2] = relation_score
        results['layer_graph'][layer2][layer1] = relation_score
    
    return results

# 添加適配函數，以便測試文件能夠使用 calculate_ 前綴
def calculate_activation_statistics(activations: torch.Tensor) -> Dict[str, float]:
    """
    計算激活值的統計指標。這個函數是 compute_activation_distribution 的擴展版本，
    增加了更多的統計指標。
    
    Args:
        activations: 層級激活值張量
        
    Returns:
        包含統計指標的字典
    """
    # 確保輸入是張量
    if not isinstance(activations, torch.Tensor):
        activations = torch.tensor(activations, dtype=torch.float32)
    
    # 將張量轉換為一維向量進行統計分析
    flat_act = activations.reshape(-1)
    
    # 基本統計量
    mean = torch.mean(flat_act).item()
    std = torch.std(flat_act).item()
    variance = torch.var(flat_act).item()
    min_val = torch.min(flat_act).item()
    max_val = torch.max(flat_act).item()
    
    # 計算分位數
    sorted_act, _ = torch.sort(flat_act)
    n = len(sorted_act)
    median = sorted_act[n // 2].item() if n % 2 == 1 else (sorted_act[n // 2 - 1] + sorted_act[n // 2]).item() / 2
    q1 = sorted_act[n // 4].item()
    q3 = sorted_act[3 * n // 4].item()
    iqr = q3 - q1
    
    # 計算偏度和峰度
    # 將張量轉換為numpy數組以使用scipy統計函數
    np_act = flat_act.detach().cpu().numpy()
    skewness = float(scipy.stats.skew(np_act))
    kurtosis = float(scipy.stats.kurtosis(np_act))
    
    # 重用已有函數中的計算結果
    dist_stats = compute_activation_distribution(activations)
    sparsity = calculate_activation_sparsity(activations)
    
    # 計算熵 (分布的不確定性)
    # 首先創建直方圖並正規化為概率分布
    hist, _ = np.histogram(np_act, bins=100, density=True)
    hist = hist[hist > 0]  # 移除零概率
    entropy_val = -np.sum(hist * np.log(hist))
    
    return {
        'mean': mean,
        'std': std,
        'variance': variance,
        'min': min_val,
        'max': max_val,
        'median': median,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'range': max_val - min_val,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'sparsity': sparsity,
        'positive_fraction': dist_stats['positive_fraction'],
        'entropy': float(entropy_val)
    }

def calculate_activation_sparsity(activations: torch.Tensor, threshold: float = 1e-6) -> float:
    """
    計算激活值的稀疏度，即接近零的值的比例。
    
    Args:
        activations: 層級激活值張量
        threshold: 判定為零的閾值
        
    Returns:
        稀疏度 (0-1之間的值，1表示全為零)
    """
    if not isinstance(activations, torch.Tensor):
        activations = torch.tensor(activations)
    
    flat_act = activations.reshape(-1)
    zero_ratio = torch.sum(torch.abs(flat_act) <= threshold).float() / flat_act.numel()
    
    return zero_ratio.item()

def calculate_activation_saturation(activations: torch.Tensor, saturation_threshold: float = 0.95) -> float:
    """
    計算激活值的飽和度，即接近最大值的比例。
    
    Args:
        activations: 層級激活值張量
        saturation_threshold: 飽和閾值
        
    Returns:
        飽和度 (0-1之間的值)
    """
    result = compute_saturation_metrics(activations, saturation_threshold)
    return result['saturation_ratio']

def calculate_effective_rank(activations: torch.Tensor) -> float:
    """
    計算激活值矩陣的有效秩，用於衡量特徵的多樣性。
    
    Args:
        activations: 層級激活值張量
        
    Returns:
        有效秩
    """
    # 特殊情況：形狀為 (1,1,1) 或接近單元素的張量
    if activations.numel() <= 1 or (len(activations.shape) == 3 and activations.shape[0] == 1 and activations.shape[1] == 1 and activations.shape[2] == 1):
        return 1.0

    # 處理張量維度
    if len(activations.shape) == 4:  # 卷積層 [batch, channels, height, width]
        # 將每個通道視為一個特徵，將空間維度展平
        batch_size, channels, height, width = activations.shape
        reshaped = activations.permute(0, 2, 3, 1).reshape(-1, channels)
    elif len(activations.shape) == 2:  # 全連接層 [batch, neurons]
        reshaped = activations
    elif len(activations.shape) == 3:  # 序列數據 [batch, seq_len, features]
        batch_size, seq_len, features = activations.shape
        reshaped = activations.reshape(-1, features)
    else:
        # 嘗試將張量展平為2維
        try:
            reshaped = activations.reshape(activations.shape[0], -1)
        except:
            return 1.0  # 如果無法重塑，直接返回 1.0
    
    # 確保矩陣有足夠的元素用於SVD
    if reshaped.shape[0] <= 1 or reshaped.shape[1] <= 1:
        return 1.0
    
    # 使用奇異值分解計算有效秩
    try:
        # 中心化數據
        centered = reshaped - torch.mean(reshaped, dim=0)
        # 奇異值分解
        _, s, _ = torch.svd(centered)
        
        # 確保奇異值非零
        if torch.sum(s) < 1e-10:
            return 1.0
            
        # 計算有效秩
        normalized_s = s / torch.sum(s)
        entropy = -torch.sum(normalized_s * torch.log(normalized_s + 1e-10))
        effective_rank = torch.exp(entropy).item()
        
        # 檢查結果是否為 NaN
        if torch.isnan(torch.tensor(effective_rank)):
            return 1.0
            
        return effective_rank
    except Exception as e:
        # 失敗時，返回最小維度作為替代
        return min(reshaped.shape)

def calculate_feature_coherence(activations: torch.Tensor) -> float:
    """
    計算特徵間的一致性，衡量不同通道/特徵之間的相關程度。
    
    Args:
        activations: 層級激活值張量，形狀為 [batch, channels, height, width] 或 [batch, seq_len, features]
        
    Returns:
        一致性指標
    """
    tensor_shape = activations.shape
    
    # 判斷張量維度
    if len(tensor_shape) == 4:  # 卷積層 [batch, channels, height, width]
        batch_size, channels, height, width = tensor_shape
        
        if channels <= 1:
            return 1.0  # 只有一個通道時，一致性為1
        
        # 將每個通道展平為向量
        flattened = activations.reshape(batch_size, channels, -1)
        
        # 計算通道之間的相關性
        return _calculate_feature_vectors_coherence(flattened)
        
    elif len(tensor_shape) == 3:  # 序列/Transformer層 [batch, seq_len, features]
        batch_size, seq_len, features = tensor_shape
        
        if features <= 1:
            return 1.0  # 只有一個特徵時，一致性為1
        
        # 將seq_len視為空間維度，features視為通道
        flattened = activations.permute(0, 2, 1)  # 變為 [batch, features, seq_len]
        flattened = flattened.reshape(batch_size, features, -1)
        
        # 計算特徵之間的相關性
        return _calculate_feature_vectors_coherence(flattened)
    
    else:
        raise ValueError(f"需要3D或4D張量，當前形狀: {tensor_shape}")

def _calculate_feature_vectors_coherence(flattened: torch.Tensor) -> float:
    """
    計算特徵向量之間的一致性
    
    Args:
        flattened: 已展平的張量，形狀為 [batch, channels/features, flattened_dim]
        
    Returns:
        一致性指標
    """
    batch_size, channels, _ = flattened.shape
    
    # 計算每對通道/特徵之間的相關係數
    correlations = []
    
    # 對於每個批次樣本
    for b in range(batch_size):
        # 計算當前批次的通道間相關係數
        batch_correlations = []
        for i in range(channels):
            for j in range(i+1, channels):
                # 提取兩個通道的特徵向量
                vec1 = flattened[b, i]
                vec2 = flattened[b, j]
                
                # 標準化
                vec1 = (vec1 - torch.mean(vec1)) / (torch.std(vec1) + 1e-8)
                vec2 = (vec2 - torch.mean(vec2)) / (torch.std(vec2) + 1e-8)
                
                # 計算相關係數
                corr = torch.mean(vec1 * vec2).item()
                batch_correlations.append(abs(corr))  # 使用絕對值，因為我們關心的是關係強度而非方向
        
        # 計算當前批次的平均相關
        if batch_correlations:
            correlations.append(np.mean(batch_correlations))
    
    # 計算所有批次的平均相關
    if correlations:
        return np.mean(correlations)
    else:
        return 0.0  # 如果無法計算，返回零

def calculate_activation_dynamics(epoch_activations: Dict[int, torch.Tensor]) -> Dict[str, Any]:
    """
    計算不同輪次間的激活值變化。
    
    Args:
        epoch_activations: 輪次到激活值的字典
        
    Returns:
        包含動態變化指標的字典
    """
    if len(epoch_activations) < 2:
        return {'error': '至少需要兩個輪次的數據'}
    
    epochs = sorted(epoch_activations.keys())
    
    # 計算每個輪次的基本統計量
    stats = {}
    for epoch in epochs:
        activations = epoch_activations[epoch]
        stats[epoch] = {
            'mean': torch.mean(activations).item(),
            'std': torch.std(activations).item(),
            'min': torch.min(activations).item(),
            'max': torch.max(activations).item()
        }
    
    # 計算輪次間的變化
    mean_changes = []
    std_changes = []
    
    for i in range(1, len(epochs)):
        prev_epoch = epochs[i-1]
        curr_epoch = epochs[i]
        
        mean_change = stats[curr_epoch]['mean'] - stats[prev_epoch]['mean']
        mean_changes.append(mean_change)
        
        std_change = stats[curr_epoch]['std'] - stats[prev_epoch]['std']
        std_changes.append(std_change)
    
    # 計算平均變化率
    mean_change_rate = sum(abs(change) for change in mean_changes) / len(mean_changes) if mean_changes else 0
    
    return {
        'epochs': epochs,
        'epoch_stats': stats,
        'mean_changes': mean_changes,
        'std_changes': std_changes,
        'mean_change_rate': mean_change_rate
    }

def calculate_layer_similarity(activations1: torch.Tensor, activations2: torch.Tensor) -> Dict[str, float]:
    """
    計算兩個層的激活值之間的相似度。
    
    Args:
        activations1: 第一個層的激活值
        activations2: 第二個層的激活值
        
    Returns:
        包含不同相似度指標的字典
    """
    # 計算餘弦相似度，compute_layer_similarity函數已被修改為處理不同形狀
    cosine_sim = compute_layer_similarity(activations1, activations2, method='cosine')
    
    # 計算皮爾遜相關係數
    correlation = compute_layer_similarity(activations1, activations2, method='correlation')
    
    # 如果是批次數據，計算每個樣本的相關
    if activations1.dim() > 1 and activations1.shape[0] > 1 and activations2.dim() > 1 and activations2.shape[0] > 1:
        # 確保批次維度相同，否則取最小值
        common_batch_size = min(activations1.shape[0], activations2.shape[0])
        
        batch_correlations = []
        for i in range(common_batch_size):
            sample1 = activations1[i].reshape(-1)
            sample2 = activations2[i].reshape(-1)
            
            # 如果形狀不同，截斷到相同長度
            if sample1.shape[0] != sample2.shape[0]:
                min_length = min(sample1.shape[0], sample2.shape[0])
                sample1 = sample1[:min_length]
                sample2 = sample2[:min_length]
            
            # 標準化
            sample1 = (sample1 - torch.mean(sample1)) / (torch.std(sample1) + 1e-8)
            sample2 = (sample2 - torch.mean(sample2)) / (torch.std(sample2) + 1e-8)
            
            # 計算相關
            batch_corr = torch.mean(sample1 * sample2).item()
            batch_correlations.append(batch_corr)
        
        mean_correlation = np.mean(batch_correlations) if batch_correlations else 0.0
    else:
        mean_correlation = correlation
    
    return {
        'cosine_similarity': cosine_sim,
        'correlation': correlation,
        'mean_correlation': mean_correlation
    } 