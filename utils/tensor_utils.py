"""
張量處理工具模組。

此模組提供用於處理PyTorch張量的實用工具。

Functions:
    flatten_tensor: 將張量平坦化為一維向量
    tensor_to_numpy: 將PyTorch張量轉換為NumPy數組
    get_tensor_statistics: 計算張量的基本統計量
    normalize_tensor: 規範化張量
    sample_tensor: 從張量中抽樣
"""

import logging
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

def flatten_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    將張量平坦化為一維向量。
    
    Args:
        tensor (torch.Tensor): 輸入張量
    
    Returns:
        torch.Tensor: 平坦化後的一維張量
    """
    return tensor.flatten()

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    將PyTorch張量轉換為NumPy數組。
    
    處理了張量在GPU上的情況及梯度追蹤。
    
    Args:
        tensor (torch.Tensor): 輸入張量
    
    Returns:
        np.ndarray: 轉換後的NumPy數組
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    if tensor.is_cuda:
        tensor = tensor.cpu()
        
    return tensor.numpy()

def get_tensor_statistics(tensor: torch.Tensor, flatten: bool = True) -> Dict[str, float]:
    """
    計算張量的基本統計量。
    
    計算包括均值、標準差、最小值、最大值、中位數等。
    
    Args:
        tensor (torch.Tensor): 輸入張量
        flatten (bool, optional): 是否將張量平坦化。默認為True
    
    Returns:
        Dict[str, float]: 包含統計量的字典
    """
    if flatten:
        tensor = flatten_tensor(tensor)
    
    # 轉換為numpy以便計算更多統計量
    np_tensor = tensor_to_numpy(tensor)
    
    try:
        stats = {
            'mean': float(np.mean(np_tensor)),
            'std': float(np.std(np_tensor)),
            'min': float(np.min(np_tensor)),
            'max': float(np.max(np_tensor)),
            'median': float(np.median(np_tensor)),
            'q1': float(np.percentile(np_tensor, 25)),
            'q3': float(np.percentile(np_tensor, 75)),
            'iqr': float(np.percentile(np_tensor, 75) - np.percentile(np_tensor, 25)),
            'skewness': float(get_skewness(np_tensor)),
            'kurtosis': float(get_kurtosis(np_tensor)),
            'sparsity': float(get_sparsity(np_tensor)),
            'shape': tensor.shape
        }
        return stats
    except Exception as e:
        logger.error(f"計算張量統計量時出錯: {e}")
        return {
            'mean': float(tensor.mean().item()),
            'std': float(tensor.std().item()),
            'min': float(tensor.min().item()),
            'max': float(tensor.max().item()),
            'shape': tensor.shape
        }

def normalize_tensor(tensor: torch.Tensor, method: str = 'min_max') -> torch.Tensor:
    """
    規範化張量。
    
    支持多種規範化方法，包括min-max和z-score。
    
    Args:
        tensor (torch.Tensor): 輸入張量
        method (str, optional): 規範化方法，'min_max'或'z_score'。默認為'min_max'
    
    Returns:
        torch.Tensor: 規範化後的張量
    """
    if method == 'min_max':
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 0:
            return (tensor - min_val) / (max_val - min_val)
        else:
            logger.warning("張量最大值等於最小值，返回零張量")
            return torch.zeros_like(tensor)
    
    elif method == 'z_score':
        mean = tensor.mean()
        std = tensor.std()
        if std > 0:
            return (tensor - mean) / std
        else:
            logger.warning("張量標準差為零，返回零張量")
            return torch.zeros_like(tensor)
    
    else:
        logger.warning(f"不支持的規範化方法: {method}，使用原始張量")
        return tensor

def sample_tensor(tensor: torch.Tensor, max_samples: int = 10000) -> torch.Tensor:
    """
    從張量中抽樣。
    
    用於處理很大的張量，通過抽樣減少計算量。
    
    Args:
        tensor (torch.Tensor): 輸入張量
        max_samples (int, optional): 最大抽樣數量。默認為10000
    
    Returns:
        torch.Tensor: 抽樣後的張量
    """
    flat_tensor = flatten_tensor(tensor)
    n = flat_tensor.numel()
    
    if n <= max_samples:
        return flat_tensor
    
    # 隨機抽樣
    indices = torch.randperm(n)[:max_samples]
    return flat_tensor[indices]

def get_sparsity(arr: np.ndarray, threshold: float = 1e-6) -> float:
    """
    計算數組的稀疏度。
    
    稀疏度定義為接近零的元素所佔比例。
    
    Args:
        arr (np.ndarray): 輸入數組
        threshold (float, optional): 判定為零的閾值。默認為1e-6
    
    Returns:
        float: 稀疏度 (0-1之間)
    """
    zero_count = np.sum(np.abs(arr) < threshold)
    return float(zero_count) / float(arr.size)

def get_skewness(arr: np.ndarray) -> float:
    """
    計算數組的偏度。
    
    Args:
        arr (np.ndarray): 輸入數組
    
    Returns:
        float: 偏度值
    """
    from scipy import stats
    return float(stats.skew(arr.flatten()))

def get_kurtosis(arr: np.ndarray) -> float:
    """
    計算數組的峰度。
    
    Args:
        arr (np.ndarray): 輸入數組
    
    Returns:
        float: 峰度值
    """
    from scipy import stats
    return float(stats.kurtosis(arr.flatten()))

def reshape_activations(activations: torch.Tensor) -> torch.Tensor:
    """
    重塑激活值張量以便於處理。
    
    根據激活值的形狀進行適當的重整，使其更便於分析。
    
    Args:
        activations (torch.Tensor): 激活值張量
    
    Returns:
        torch.Tensor: 重塑後的張量
    """
    # 對於4D張量 (通常是卷積層的輸出)
    if len(activations.shape) == 4:
        # [batch, channels, height, width] -> [batch, channels, height*width]
        return activations.reshape(activations.shape[0], activations.shape[1], -1)
    
    # 對於3D張量 (通常是注意力層的輸出)
    elif len(activations.shape) == 3:
        # 保持原樣
        return activations
    
    # 對於2D張量 (通常是全連接層的輸出)
    elif len(activations.shape) == 2:
        # 保持原樣
        return activations
    
    # 其他情況
    else:
        return activations 