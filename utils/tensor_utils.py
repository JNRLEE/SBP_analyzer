"""
# 張量工具
# 提供處理 PyTorch 張量的輔助函數
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def to_numpy(tensor: Any) -> np.ndarray:
    """
    將 PyTorch 張量轉換為 NumPy 數組。
    
    Args:
        tensor: 要轉換的數據，可以是 PyTorch 張量、列表或 NumPy 數組
        
    Returns:
        np.ndarray: 轉換後的 NumPy 數組
        
    Description:
        安全地將 PyTorch 張量轉換為 NumPy 數組，處理不同的設備類型和數據類型。
        如果輸入已經是 NumPy 數組或標量，則直接返回。
        
    References:
        - PyTorch to NumPy conversion: https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html
    """
    if not HAS_TORCH:
        if isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.array(tensor)
    
    if isinstance(tensor, torch.Tensor):
        # 確保張量在 CPU 上並且是連續的
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        return np.array(tensor)
    elif isinstance(tensor, (int, float, bool)):
        return np.array(tensor)
    else:
        raise TypeError(f"無法轉換類型 {type(tensor)} 為 NumPy 數組")


def to_tensor(
    array: Any,
    device: Optional[str] = None,
    dtype: Optional[Any] = None
) -> "torch.Tensor":
    """
    將 NumPy 數組或其他數據轉換為 PyTorch 張量。
    
    Args:
        array: 要轉換的數據，可以是 NumPy 數組、列表或 PyTorch 張量
        device: 張量的目標設備，例如 'cpu' 或 'cuda'
        dtype: 張量的目標數據類型
        
    Returns:
        torch.Tensor: 轉換後的 PyTorch 張量
        
    Description:
        將 NumPy 數組或其他數據安全地轉換為 PyTorch 張量，並可選擇指定設備和數據類型。
        如果輸入已經是 PyTorch 張量，則將其移至指定的設備並轉換為指定的數據類型。
        
    References:
        - NumPy to PyTorch conversion: https://pytorch.org/docs/stable/torch.html#torch.tensor
    """
    if not HAS_TORCH:
        raise ImportError("未安裝 PyTorch，無法轉換為張量")
    
    if isinstance(array, torch.Tensor):
        # 如果已經是張量，則移動到指定設備和數據類型
        tensor = array
        if device is not None:
            tensor = tensor.to(device)
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor
    
    # 從 NumPy 數組或其他數據創建張量
    kwargs = {}
    if device is not None:
        kwargs['device'] = device
    if dtype is not None:
        kwargs['dtype'] = dtype
    
    return torch.tensor(array, **kwargs)


def flatten_nested_tensors(
    nested_tensors: Any,
    prefix: str = ''
) -> Dict[str, np.ndarray]:
    """
    將嵌套的張量字典攤平為單層字典。
    
    Args:
        nested_tensors: 包含嵌套張量的數據結構
        prefix: 用於生成鍵名的前綴
        
    Returns:
        Dict[str, np.ndarray]: 扁平化的字典，鍵為路徑，值為 NumPy 數組
        
    Description:
        將嵌套的數據結構（字典、列表、元組）中的所有張量攤平為單層字典，
        鍵名使用點號分隔的路徑。
        
    References:
        None
    """
    flattened = {}
    
    if isinstance(nested_tensors, dict):
        for key, value in nested_tensors.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list, tuple)):
                flattened.update(flatten_nested_tensors(value, new_prefix))
            else:
                try:
                    # 嘗試轉換為 NumPy 數組
                    flattened[new_prefix] = to_numpy(value)
                except:
                    # 如果不能轉換，則跳過
                    pass
    
    elif isinstance(nested_tensors, (list, tuple)):
        for i, value in enumerate(nested_tensors):
            new_prefix = f"{prefix}[{i}]"
            if isinstance(value, (dict, list, tuple)):
                flattened.update(flatten_nested_tensors(value, new_prefix))
            else:
                try:
                    flattened[new_prefix] = to_numpy(value)
                except:
                    pass
    
    else:
        try:
            if prefix:
                flattened[prefix] = to_numpy(nested_tensors)
        except:
            pass
    
    return flattened


def get_tensor_statistics(
    tensor: Any,
    axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """
    計算張量的統計量。
    
    Args:
        tensor: 要分析的張量或數組
        axis: 計算統計量的軸
        
    Returns:
        Dict[str, Union[float, np.ndarray]]: 包含統計量的字典
        
    Description:
        計算張量的基本統計量，如均值、中位數、最小值、最大值、標準差等。
        可以沿著指定的軸計算，或者計算整個張量的統計量。
        
    References:
        None
    """
    # 轉換為 NumPy 數組
    array = to_numpy(tensor)
    
    # 計算基本統計量
    stats = {
        'mean': np.mean(array, axis=axis),
        'median': np.median(array, axis=axis),
        'min': np.min(array, axis=axis),
        'max': np.max(array, axis=axis),
        'std': np.std(array, axis=axis),
        'var': np.var(array, axis=axis),
        'shape': array.shape,
        'size': array.size,
        'ndim': array.ndim
    }
    
    # 計算非零元素的比例
    non_zeros = np.count_nonzero(array, axis=axis)
    if axis is None:
        stats['non_zero_ratio'] = non_zeros / array.size
    else:
        # 計算每個軸的非零比例
        axis_size = np.prod([array.shape[i] for i in range(array.ndim) if i != axis])
        stats['non_zero_ratio'] = non_zeros / axis_size
    
    return stats 