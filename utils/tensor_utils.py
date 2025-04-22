"""
# 實現張量數據的預處理和正規化功能
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any


def to_numpy(tensor: Union[torch.Tensor, np.ndarray, List]) -> np.ndarray:
    """
    將張量轉換為NumPy陣列
    
    Args:
        tensor: 輸入張量，可以是PyTorch張量、NumPy陣列或列表
        
    Returns:
        np.ndarray: 轉換後的NumPy陣列
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        return np.array(tensor)
    else:
        raise TypeError(f"無法轉換類型 {type(tensor)} 為 numpy 陣列")


def normalize_tensor(tensor: Union[torch.Tensor, np.ndarray], 
                     method: str = 'minmax', 
                     dim: Optional[int] = None) -> Union[torch.Tensor, np.ndarray]:
    """
    正規化張量數據
    
    Args:
        tensor: 輸入張量
        method: 正規化方法，可選 'minmax', 'zscore', 'l1', 'l2'
        dim: 應用正規化的維度，None表示整個張量
        
    Returns:
        與輸入相同類型的正規化後張量
    """
    is_torch = isinstance(tensor, torch.Tensor)
    
    if not is_torch:
        tensor_np = to_numpy(tensor)
        if method == 'minmax':
            min_val = tensor_np.min(axis=dim, keepdims=True)
            max_val = tensor_np.max(axis=dim, keepdims=True)
            # 防止除零
            denominator = (max_val - min_val)
            denominator = np.where(denominator != 0, denominator, 1)
            return (tensor_np - min_val) / denominator
        elif method == 'zscore':
            mean_val = tensor_np.mean(axis=dim, keepdims=True)
            std_val = tensor_np.std(axis=dim, keepdims=True)
            # 防止除零
            std_val = np.where(std_val != 0, std_val, 1)
            return (tensor_np - mean_val) / std_val
        elif method == 'l1':
            norm = np.sum(np.abs(tensor_np), axis=dim, keepdims=True)
            # 防止除零
            norm = np.where(norm != 0, norm, 1)
            return tensor_np / norm
        elif method == 'l2':
            norm = np.sqrt(np.sum(tensor_np**2, axis=dim, keepdims=True))
            # 防止除零
            norm = np.where(norm != 0, norm, 1)
            return tensor_np / norm
        else:
            raise ValueError(f"未知的正規化方法: {method}")
    else:
        if method == 'minmax':
            min_val = tensor.min(dim=dim, keepdim=True)[0] if dim is not None else tensor.min()
            max_val = tensor.max(dim=dim, keepdim=True)[0] if dim is not None else tensor.max()
            # 防止除零
            denominator = (max_val - min_val)
            denominator = torch.where(denominator != 0, denominator, torch.tensor(1.0, device=tensor.device))
            return (tensor - min_val) / denominator
        elif method == 'zscore':
            mean_val = tensor.mean(dim=dim, keepdim=True) if dim is not None else tensor.mean()
            std_val = tensor.std(dim=dim, keepdim=True) if dim is not None else tensor.std()
            # 防止除零
            std_val = torch.where(std_val != 0, std_val, torch.tensor(1.0, device=tensor.device))
            return (tensor - mean_val) / std_val
        elif method == 'l1':
            norm = torch.sum(torch.abs(tensor), dim=dim, keepdim=True) if dim is not None else torch.sum(torch.abs(tensor))
            # 防止除零
            norm = torch.where(norm != 0, norm, torch.tensor(1.0, device=tensor.device))
            return tensor / norm
        elif method == 'l2':
            norm = torch.sqrt(torch.sum(tensor**2, dim=dim, keepdim=True)) if dim is not None else torch.sqrt(torch.sum(tensor**2))
            # 防止除零
            norm = torch.where(norm != 0, norm, torch.tensor(1.0, device=tensor.device))
            return tensor / norm
        else:
            raise ValueError(f"未知的正規化方法: {method}")


def preprocess_activations(activations: torch.Tensor, 
                          flatten: bool = False, 
                          normalize: Optional[str] = None,
                          remove_outliers: bool = False,
                          outlier_threshold: float = 3.0) -> torch.Tensor:
    """
    預處理激活值數據
    
    Args:
        activations: 輸入激活值張量
        flatten: 是否將激活值展平
        normalize: 正規化方法，None表示不進行正規化
        remove_outliers: 是否移除離群值
        outlier_threshold: 離群值閾值（標準差的倍數）
        
    Returns:
        torch.Tensor: 預處理後的激活值
    """
    processed = activations.clone()
    
    # 移除離群值（將其替換為平均值）
    if remove_outliers and processed.numel() > 1:
        mean = processed.mean()
        std = processed.std()
        mask = torch.abs(processed - mean) > (outlier_threshold * std)
        processed[mask] = mean
    
    # 正規化
    if normalize is not None:
        processed = normalize_tensor(processed, method=normalize)
    
    # 展平
    if flatten and processed.dim() > 1:
        processed = processed.view(processed.size(0), -1)
    
    return processed


def batch_to_features(batch_data: Dict[str, torch.Tensor], 
                     feature_key: str = 'features') -> np.ndarray:
    """
    從批次數據中提取特徵
    
    Args:
        batch_data: 批次數據字典
        feature_key: 特徵的鍵名
        
    Returns:
        np.ndarray: 特徵陣列
    """
    if feature_key in batch_data:
        return to_numpy(batch_data[feature_key])
    else:
        raise KeyError(f"批次數據中不包含 '{feature_key}' 鍵")
        

def compute_batch_statistics(tensor: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
    """
    計算張量的基礎統計指標
    
    Args:
        tensor: 輸入張量
        
    Returns:
        Dict[str, float]: 包含平均值、標準差、最小值、最大值、中位數等統計信息的字典
    """
    if isinstance(tensor, torch.Tensor):
        tensor = to_numpy(tensor)
    
    flat_tensor = tensor.reshape(-1)
    
    return {
        'mean': float(np.mean(flat_tensor)),
        'std': float(np.std(flat_tensor)),
        'min': float(np.min(flat_tensor)),
        'max': float(np.max(flat_tensor)),
        'median': float(np.median(flat_tensor)),
        'q1': float(np.percentile(flat_tensor, 25)),
        'q3': float(np.percentile(flat_tensor, 75)),
        'sparsity': float((flat_tensor == 0).sum() / flat_tensor.size),
        'num_elements': int(flat_tensor.size)
    }


def tensor_correlation(tensor1: Union[torch.Tensor, np.ndarray], 
                      tensor2: Union[torch.Tensor, np.ndarray]) -> float:
    """
    計算兩個張量之間的皮爾森相關係數
    
    Args:
        tensor1: 第一個張量
        tensor2: 第二個張量
        
    Returns:
        float: 相關係數
    """
    tensor1_np = to_numpy(tensor1).flatten()
    tensor2_np = to_numpy(tensor2).flatten()
    
    # 確保兩個張量具有相同的形狀
    if tensor1_np.shape != tensor2_np.shape:
        min_size = min(tensor1_np.size, tensor2_np.size)
        tensor1_np = tensor1_np[:min_size]
        tensor2_np = tensor2_np[:min_size]
    
    return float(np.corrcoef(tensor1_np, tensor2_np)[0, 1])


def average_activations(activations_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    計算多個激活值的平均值
    
    Args:
        activations_dict: 層名稱到激活值張量的映射字典
        
    Returns:
        Dict[str, torch.Tensor]: 層名稱到平均激活值的映射字典
    """
    result = {}
    for layer_name, activation in activations_dict.items():
        # 計算每個特徵圖的平均值
        if activation.dim() > 2:  # 對於卷積層的輸出
            # 通道維度通常為1，保留它
            result[layer_name] = activation.mean(dim=(2, 3)) if activation.dim() == 4 else activation.mean(dim=2)
        else:
            # 對於全連接層，直接平均
            result[layer_name] = activation.mean(dim=0)
    
    return result


def extract_feature_maps(activation: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, List[int]]:
    """
    從激活值中提取最活躍的特徵圖
    
    Args:
        activation: 激活值張量
        top_k: 需要提取的特徵圖數量
        
    Returns:
        Tuple[torch.Tensor, List[int]]: 提取的特徵圖和對應的索引
    """
    if activation.dim() < 4:  # 需要是卷積層的輸出
        raise ValueError("激活值的維度應為4 (batch, channels, height, width)")
    
    # 計算每個特徵圖的平均激活強度
    mean_activation = activation.mean(dim=(0, 2, 3))
    
    # 找出最活躍的特徵圖索引
    if top_k >= mean_activation.size(0):
        top_indices = list(range(mean_activation.size(0)))
    else:
        _, top_indices = torch.topk(mean_activation, top_k)
        top_indices = top_indices.tolist()
    
    # 提取這些特徵圖
    top_feature_maps = activation[:, top_indices, :, :]
    
    return top_feature_maps, top_indices 