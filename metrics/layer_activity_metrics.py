"""
# 神經網絡層級活動度指標計算模組
# 
# 此模組負責計算神經網絡各層激活值的統計指標，包括：
# - 激活分佈分析（正/負/零值比例）
# - 神經元活躍度指標
# - 死亡神經元檢測
# - 飽和度指標計算
# - 層級響應穩定性分析
"""

import torch
import numpy as np
import scipy.stats
from typing import Dict, Any, Tuple, List, Union


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
    else:
        raise ValueError(f"不支援的激活值張量形狀: {tensor_shape}")
    
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
    else:
        raise ValueError(f"不支援的激活值張量形狀: {tensor_shape}")
    
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
    # 確保張量維度匹配
    if activations1.shape != activations2.shape:
        raise ValueError(f"激活值形狀不匹配: {activations1.shape} vs {activations2.shape}")
    
    # 將張量展平為一維向量
    flat1 = activations1.reshape(-1)
    flat2 = activations2.reshape(-1)
    
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
    計算層級激活值在不同批次之間的穩定性。
    
    Args:
        batch_activations: 包含多個批次激活值的列表
        
    Returns:
        包含穩定性指標的字典
    """
    if len(batch_activations) < 2:
        raise ValueError("需要至少兩個批次的激活值來計算穩定性")
    
    # 計算每個批次的平均激活值
    batch_means = [tensor.mean().item() for tensor in batch_activations]
    
    # 計算批次間的方差和離散度
    batch_variance = np.var(batch_means)
    batch_cv = np.std(batch_means) / np.mean(batch_means) if np.mean(batch_means) != 0 else 0.0
    
    # 計算相鄰批次之間的相似度
    similarities = []
    for i in range(len(batch_activations) - 1):
        similarity = compute_layer_similarity(batch_activations[i], batch_activations[i+1])
        similarities.append(similarity)
    
    avg_similarity = np.mean(similarities) if similarities else 0.0
    
    return {
        'batch_variance': batch_variance,
        'batch_coefficient_variation': batch_cv,
        'avg_batch_similarity': avg_similarity
    }


def analyze_layer_gradient_flow(gradients: torch.Tensor) -> Dict[str, float]:
    """
    分析層級梯度流動情況，檢測梯度消失或爆炸。
    
    Args:
        gradients: 層參數梯度張量
        
    Returns:
        包含梯度分析指標的字典
    """
    if gradients is None:
        return {
            'has_gradient': False,
            'mean_gradient': 0.0,
            'gradient_norm': 0.0,
            'zero_gradients_ratio': 1.0,
            'gradient_signal_to_noise': 0.0
        }
    
    # 計算梯度範數
    gradient_norm = torch.norm(gradients).item()
    
    # 計算梯度的平均值和標準差
    mean_gradient = gradients.mean().item()
    std_gradient = gradients.std().item()
    
    # 計算零梯度的比例
    flat_gradients = gradients.reshape(-1)
    zero_count = torch.sum(flat_gradients == 0.0).item()
    zero_ratio = zero_count / flat_gradients.numel()
    
    # 計算梯度信噪比 (信號強度/噪聲強度)
    signal_to_noise = abs(mean_gradient) / (std_gradient + 1e-8) if std_gradient > 0 else 0.0
    
    # 檢測梯度是否消失或爆炸
    vanishing = gradient_norm < 1e-4
    exploding = gradient_norm > 1e3
    
    return {
        'has_gradient': True,
        'mean_gradient': mean_gradient,
        'gradient_norm': gradient_norm,
        'zero_gradients_ratio': zero_ratio,
        'gradient_signal_to_noise': signal_to_noise,
        'gradient_vanishing': vanishing,
        'gradient_exploding': exploding
    }


def calculate_activation_statistics(activations: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    # 計算激活值的統計指標
    
    Args:
        activations: 層級激活值張量或numpy陣列
        
    Returns:
        包含統計指標的字典
    """
    # 將輸入轉換為numpy陣列以便計算統計量
    if isinstance(activations, torch.Tensor):
        activations_np = activations.detach().cpu().numpy()
    else:
        activations_np = np.asarray(activations)
    
    # 將激活值展平
    flat_activations = activations_np.reshape(-1)
    
    # 計算基本統計量
    mean = np.mean(flat_activations)
    std = np.std(flat_activations)
    minimum = np.min(flat_activations)
    maximum = np.max(flat_activations)
    median = np.median(flat_activations)
    
    # 計算分位數
    q1 = np.percentile(flat_activations, 25)
    q3 = np.percentile(flat_activations, 75)
    iqr = q3 - q1
    
    # 計算高階統計量
    skewness = scipy.stats.skew(flat_activations)
    kurtosis = scipy.stats.kurtosis(flat_activations)
    
    # 計算範圍和方差
    value_range = maximum - minimum
    variance = np.var(flat_activations)
    
    # 計算稀疏度 (零值的比例)
    sparsity = np.sum(np.abs(flat_activations) < 1e-6) / len(flat_activations)
    
    # 計算正值比例
    positive_fraction = np.sum(flat_activations > 0) / len(flat_activations)
    
    # 計算熵 (使用直方圖估算)
    hist, bin_edges = np.histogram(flat_activations, bins=50, density=True)
    hist = hist[hist > 0]  # 排除零概率區域
    entropy = -np.sum(hist * np.log2(hist)) * (bin_edges[1] - bin_edges[0])
    
    return {
        'mean': float(mean),
        'std': float(std),
        'min': float(minimum),
        'max': float(maximum),
        'median': float(median),
        'q1': float(q1),
        'q3': float(q3),
        'iqr': float(iqr),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'range': float(value_range),
        'variance': float(variance),
        'sparsity': float(sparsity),
        'positive_fraction': float(positive_fraction),
        'entropy': float(entropy)
    }


def calculate_activation_sparsity(activations: Union[torch.Tensor, np.ndarray], 
                                 threshold: float = 1e-6) -> float:
    """
    # 計算激活值的稀疏度 (零值或接近零值的比例)
    
    Args:
        activations: 層級激活值張量或numpy陣列
        threshold: 視為零的閾值
        
    Returns:
        稀疏度 (0-1之間的值，1表示完全稀疏，全是零)
    """
    if isinstance(activations, torch.Tensor):
        # PyTorch實現
        return (torch.abs(activations) < threshold).float().mean().item()
    else:
        # NumPy實現
        activations_np = np.asarray(activations)
        return np.sum(np.abs(activations_np) < threshold) / activations_np.size


def calculate_activation_saturation(activations: Union[torch.Tensor, np.ndarray],
                                   threshold: float = 0.9) -> float:
    """
    # 計算激活值的飽和度 (接近最大值的比例)
    
    Args:
        activations: 層級激活值張量或numpy陣列
        threshold: 視為飽和的閾值 (相對於最大可能值)
        
    Returns:
        飽和度 (0-1之間的值，1表示完全飽和)
    """
    if isinstance(activations, torch.Tensor):
        # 標準化激活值
        if activations.min() < 0 or activations.max() > 1:
            max_val = activations.max()
            normalized = activations / (max_val + 1e-8) if max_val > 0 else activations
        else:
            normalized = activations
        
        # 計算飽和比例
        return (normalized > threshold).float().mean().item()
    else:
        # NumPy實現
        activations_np = np.asarray(activations)
        max_val = np.max(activations_np)
        normalized = activations_np / (max_val + 1e-8) if max_val > 0 else activations_np
        return np.sum(normalized > threshold) / activations_np.size


def calculate_effective_rank(activations: Union[torch.Tensor, np.ndarray]) -> float:
    """
    # 計算激活值矩陣的有效秩 (使用奇異值)
    
    Args:
        activations: 層級激活值張量或numpy陣列
        
    Returns:
        有效秩 (浮點數，反映數據內在維度)
    """
    # 轉換為numpy陣列進行SVD分解
    if isinstance(activations, torch.Tensor):
        activations_np = activations.detach().cpu().numpy()
    else:
        activations_np = np.asarray(activations)
    
    # 如果是高維張量，將其重塑為矩陣
    if len(activations_np.shape) > 2:
        # 對於卷積層張量 [batch, channels, height, width]，轉為 [batch, channels*height*width]
        activations_np = activations_np.reshape(activations_np.shape[0], -1)
    
    # 對於一維張量，轉換為列向量
    if len(activations_np.shape) == 1:
        activations_np = activations_np.reshape(-1, 1)
    
    # 處理極小的矩陣
    if min(activations_np.shape) <= 1:
        return 1.0
    
    # 計算SVD
    try:
        u, s, vh = np.linalg.svd(activations_np, full_matrices=False)
        
        # 標準化奇異值
        normalized_s = s / np.sum(s)
        
        # 計算有效秩 (基於奇異值的熵)
        entropy = -np.sum(normalized_s * np.log(normalized_s + 1e-10))
        effective_rank = np.exp(entropy)
        
        return float(effective_rank)
    except:
        # 如果SVD計算失敗，返回最小維度作為有效秩的估計
        return float(min(activations_np.shape))


def calculate_feature_coherence(activations: Union[torch.Tensor, np.ndarray]) -> float:
    """
    # 計算特徵圖間的一致性 (對卷積層特別有用)
    
    Args:
        activations: 層級激活值張量或numpy陣列，形狀為 [batch, channels, height, width]
        
    Returns:
        特徵一致性值 (-1到1之間，1表示完全一致)
    """
    # 檢查輸入維度
    if isinstance(activations, torch.Tensor):
        if len(activations.shape) < 3:
            raise ValueError("特徵一致性計算需要至少3維張量 [batch, channels, ...]")
        
        # 處理卷積層特徵圖
        if len(activations.shape) == 4:  # [batch, channels, height, width]
            batch_size, channels = activations.shape[0], activations.shape[1]
            
            # 將特徵圖展平
            reshaped = activations.reshape(batch_size, channels, -1)
            
            # 計算每一批次中通道間的相關性
            corr_sum = 0.0
            count = 0
            
            for b in range(batch_size):
                for i in range(channels):
                    for j in range(i+1, channels):
                        # 計算通道i和j之間的相關係數
                        x = reshaped[b, i]
                        y = reshaped[b, j]
                        
                        x_centered = x - torch.mean(x)
                        y_centered = y - torch.mean(y)
                        
                        x_norm = torch.norm(x_centered)
                        y_norm = torch.norm(y_centered)
                        
                        if x_norm > 0 and y_norm > 0:
                            corr = torch.dot(x_centered, y_centered) / (x_norm * y_norm)
                            corr_sum += corr.item()
                            count += 1
            
            # 計算平均相關係數
            mean_corr = corr_sum / count if count > 0 else 0.0
            return mean_corr
        else:
            # 非標準卷積層特徵，簡化計算
            activations = activations.reshape(activations.shape[0], activations.shape[1], -1)
            return calculate_feature_coherence(activations)
    else:
        # 轉換為PyTorch張量並重新調用函數
        return calculate_feature_coherence(torch.tensor(activations))


def calculate_layer_similarity(layer1: Union[torch.Tensor, np.ndarray], 
                              layer2: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    # 計算兩個層之間的相似度
    
    Args:
        layer1: 第一個層的激活值
        layer2: 第二個層的激活值
        
    Returns:
        包含不同相似度指標的字典
    """
    # 轉換為PyTorch張量
    if not isinstance(layer1, torch.Tensor):
        layer1 = torch.tensor(layer1)
    if not isinstance(layer2, torch.Tensor):
        layer2 = torch.tensor(layer2)
    
    # 將張量展平為矩陣 [batch, features]
    if len(layer1.shape) > 2:
        layer1 = layer1.reshape(layer1.shape[0], -1)
    if len(layer2.shape) > 2:
        layer2 = layer2.reshape(layer2.shape[0], -1)
    
    # 確保批次大小一致
    if layer1.shape[0] != layer2.shape[0]:
        raise ValueError(f"層的批次大小不匹配: {layer1.shape[0]} vs {layer2.shape[0]}")
    
    # 計算相關係數
    correlations = []
    for i in range(layer1.shape[0]):
        vec1 = layer1[i].float()
        vec2 = layer2[i].float()
        
        # 中心化
        vec1 = vec1 - vec1.mean()
        vec2 = vec2 - vec2.mean()
        
        # 標準化
        norm1 = torch.norm(vec1)
        norm2 = torch.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            corr = torch.dot(vec1, vec2) / (norm1 * norm2)
            correlations.append(corr.item())
    
    # 計算餘弦相似度
    layer1_flat = layer1.reshape(-1).float()
    layer2_flat = layer2.reshape(-1).float()
    cosine_sim = torch.nn.functional.cosine_similarity(layer1_flat.unsqueeze(0), 
                                                      layer2_flat.unsqueeze(0)).item()
    
    return {
        'mean_correlation': np.mean(correlations) if correlations else 0.0,
        'cosine_similarity': cosine_sim
    }


def calculate_activation_dynamics(epochs_activations: Dict[int, Union[torch.Tensor, np.ndarray]]) -> Dict[str, Any]:
    """
    # 計算激活值隨訓練進行的動態變化
    
    Args:
        epochs_activations: 字典，鍵為輪次，值為該輪次的激活值
        
    Returns:
        包含動態變化指標的字典
    """
    # 排序輪次
    epochs = sorted(epochs_activations.keys())
    
    if len(epochs) < 2:
        return {
            'epochs': epochs,
            'mean_changes': [],
            'std_changes': [],
            'mean_change_rate': 0.0
        }
    
    mean_values = []
    std_values = []
    
    # 計算每個輪次的統計量
    for epoch in epochs:
        activations = epochs_activations[epoch]
        
        # 轉換為numpy陣列
        if isinstance(activations, torch.Tensor):
            act_np = activations.detach().cpu().numpy()
        else:
            act_np = np.asarray(activations)
        
        mean_values.append(np.mean(act_np))
        std_values.append(np.std(act_np))
    
    # 計算相鄰輪次之間的變化
    mean_changes = [abs(mean_values[i+1] - mean_values[i]) for i in range(len(mean_values)-1)]
    std_changes = [abs(std_values[i+1] - std_values[i]) for i in range(len(std_values)-1)]
    
    # 計算變化率
    mean_change_rate = np.mean(mean_changes) / (np.mean(mean_values) + 1e-8)
    
    return {
        'epochs': epochs,
        'mean_changes': mean_changes,
        'std_changes': std_changes,
        'mean_change_rate': float(mean_change_rate)
    } 