"""
分布度量模組。

此模組提供用於計算兩個分布之間相似度或差異性的指標。

Functions:
    calculate_kl_divergence: 計算KL散度
    calculate_js_divergence: 計算JS散度
    calculate_wasserstein_distance: 計算Wasserstein距離
    calculate_histogram_intersection: 計算直方圖交集
"""

import logging
import numpy as np
import torch
from scipy import stats
from typing import List, Tuple, Optional, Union, Dict

# 更改為使用絕對導入路徑
from utils.tensor_utils import tensor_to_numpy
from utils.stat_utils import detect_outliers

logger = logging.getLogger(__name__)

def calculate_kl_divergence(
    p_dist: Union[np.ndarray, torch.Tensor],
    q_dist: Union[np.ndarray, torch.Tensor],
    min_value: float = 1e-10
) -> float:
    """
    計算兩個分布間的Kullback-Leibler散度。
    
    KL散度測量從一個分布到另一個分布的資訊損失。
    
    Args:
        p_dist (Union[np.ndarray, torch.Tensor]): 參考分布
        q_dist (Union[np.ndarray, torch.Tensor]): 與參考分布比較的分布
        min_value (float, optional): 添加到零值以避免數值問題的小常數。默認為1e-10
    
    Returns:
        float: KL散度值
    
    Raises:
        ValueError: 分布的維度不一致時
    """
    # 轉換為numpy數組
    if isinstance(p_dist, torch.Tensor):
        p_dist = tensor_to_numpy(p_dist)
    if isinstance(q_dist, torch.Tensor):
        q_dist = tensor_to_numpy(q_dist)
    
    # 確保形狀一致
    if p_dist.shape != q_dist.shape:
        raise ValueError(f"分布形狀不一致: p_dist {p_dist.shape}, q_dist {q_dist.shape}")
    
    # 扁平化
    p = p_dist.flatten()
    q = q_dist.flatten()
    
    # 正規化
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # 避免零值
    p = np.maximum(p, min_value)
    q = np.maximum(q, min_value)
    
    # 重新正規化
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # 計算KL散度
    return float(np.sum(p * np.log(p / q)))

def calculate_js_divergence(
    p_dist: Union[np.ndarray, torch.Tensor],
    q_dist: Union[np.ndarray, torch.Tensor],
    min_value: float = 1e-10
) -> float:
    """
    計算兩個分布間的Jensen-Shannon散度。
    
    JS散度是KL散度的對稱版本，其值介於0和1之間。
    
    Args:
        p_dist (Union[np.ndarray, torch.Tensor]): 第一個分布
        q_dist (Union[np.ndarray, torch.Tensor]): 第二個分布
        min_value (float, optional): 避免數值問題的小常數。默認為1e-10
    
    Returns:
        float: JS散度值
    """
    # 轉換為numpy數組
    if isinstance(p_dist, torch.Tensor):
        p_dist = tensor_to_numpy(p_dist)
    if isinstance(q_dist, torch.Tensor):
        q_dist = tensor_to_numpy(q_dist)
    
    # 確保形狀一致
    if p_dist.shape != q_dist.shape:
        # 嘗試廣播以使兩個數組兼容
        try:
            np.broadcast(p_dist, q_dist)
        except ValueError:
            raise ValueError(f"無法廣播分布形狀: p_dist {p_dist.shape}, q_dist {q_dist.shape}")
    
    # 扁平化
    p = p_dist.flatten()
    q = q_dist.flatten()
    
    # 正規化
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # 避免零值
    p = np.maximum(p, min_value)
    q = np.maximum(q, min_value)
    
    # 重新正規化
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # 計算M分布
    m = 0.5 * (p + q)
    
    # 計算JS散度
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

def calculate_wasserstein_distance(
    p_dist: Union[torch.Tensor, np.ndarray],
    q_dist: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    計算兩個分布間的Wasserstein距離（地球移動者距離）。
    
    該距離測量將一個分布轉化為另一個所需的最小"工作量"。
    
    Args:
        p_dist (Union[torch.Tensor, np.ndarray]): 第一個分布
        q_dist (Union[torch.Tensor, np.ndarray]): 第二個分布
    
    Returns:
        float: Wasserstein距離值
    """
    # 轉換為numpy數組
    if isinstance(p_dist, torch.Tensor):
        p_dist = tensor_to_numpy(p_dist)
    if isinstance(q_dist, torch.Tensor):
        q_dist = tensor_to_numpy(q_dist)
    
    # 扁平化並排序
    p_values = np.sort(p_dist.flatten())
    q_values = np.sort(q_dist.flatten())
    
    # 確保長度相同
    if len(p_values) != len(q_values):
        min_length = min(len(p_values), len(q_values))
        p_values = np.linspace(
            p_values.min(), p_values.max(), min_length
        )
        q_values = np.linspace(
            q_values.min(), q_values.max(), min_length
        )
    
    # 計算Wasserstein距離
    # 使用正確的 scipy.stats 引用
    return float(stats.wasserstein_distance(p_values, q_values))

def calculate_histogram_intersection(
    p_dist: Union[torch.Tensor, np.ndarray],
    q_dist: Union[torch.Tensor, np.ndarray],
    bins: int = 100,
    density: bool = True
) -> float:
    """
    計算兩個分布的直方圖交集。
    
    直方圖交集測量兩個分布之間的相似性。
    
    Args:
        p_dist (Union[torch.Tensor, np.ndarray]): 第一個分布
        q_dist (Union[torch.Tensor, np.ndarray]): 第二個分布
        bins (int, optional): 直方圖的柱數。默認為100
        density (bool, optional): 是否將直方圖正規化為密度函數。默認為True
    
    Returns:
        float: 直方圖交集值，範圍從0（無交集）到1（完全重疊）
    """
    # 轉換為numpy數組
    if isinstance(p_dist, torch.Tensor):
        p_dist = tensor_to_numpy(p_dist)
    if isinstance(q_dist, torch.Tensor):
        q_dist = tensor_to_numpy(q_dist)
    
    # 扁平化
    p = p_dist.flatten()
    q = q_dist.flatten()
    
    # 求出共同的範圍
    min_val = min(p.min(), q.min())
    max_val = max(p.max(), q.max())
    
    # 如果範圍太小，添加一點偏移以避免錯誤
    if max_val - min_val < 1e-10:
        max_val = min_val + 1.0
    
    # 計算直方圖
    p_hist, bin_edges = np.histogram(
        p, bins=bins, range=(min_val, max_val), density=density
    )
    q_hist, _ = np.histogram(
        q, bins=bins, range=(min_val, max_val), density=density
    )
    
    # 正規化直方圖，確保其和為1
    p_hist = p_hist / np.sum(p_hist) if np.sum(p_hist) > 0 else p_hist
    q_hist = q_hist / np.sum(q_hist) if np.sum(q_hist) > 0 else q_hist
    
    # 計算交集
    intersection = np.sum(np.minimum(p_hist, q_hist))
    
    # 確保返回值在0到1之間
    # 由於直方圖已正規化，交集的最大值為1
    return float(min(max(intersection, 0.0), 1.0))

def compare_distributions(dist1: np.ndarray, dist2: np.ndarray) -> Dict[str, float]:
    """
    全面比較兩個分布，使用多種度量方法。
    
    Args:
        dist1 (np.ndarray): 第一個分布。
        dist2 (np.ndarray): 第二個分布。
    
    Returns:
        Dict[str, float]: 包含各種分布度量的字典。
    """
    # 標準化並平滑分布
    dist1 = np.asarray(dist1, dtype=np.float64).flatten()
    dist2 = np.asarray(dist2, dtype=np.float64).flatten()
    
    # 如果分布長度不同，則調整為相同長度
    if len(dist1) < len(dist2):
        dist1 = np.pad(dist1, (0, len(dist2) - len(dist1)), 'constant')
    elif len(dist1) > len(dist2):
        dist2 = np.pad(dist2, (0, len(dist1) - len(dist2)), 'constant')
    
    # 歸一化分布
    dist1 = dist1 / np.sum(dist1) if np.sum(dist1) > 0 else dist1
    dist2 = dist2 / np.sum(dist2) if np.sum(dist2) > 0 else dist2
    
    # 計算各種度量
    metrics = {
        'kl_divergence': calculate_kl_divergence(dist1, dist2),
        'js_divergence': calculate_js_divergence(dist1, dist2),
        'wasserstein_distance': calculate_wasserstein_distance(dist1, dist2),
        'histogram_intersection': calculate_histogram_intersection(dist1, dist2)
    }
    
    return metrics 

def calculate_distribution_overlap(
    p_dist: Union[np.ndarray, torch.Tensor],
    q_dist: Union[np.ndarray, torch.Tensor],
    method: str = 'js'
) -> float:
    """
    計算兩個分布之間的重疊或相似度。
    
    根據指定的方法，選擇適當的度量。
    
    Args:
        p_dist (Union[np.ndarray, torch.Tensor]): 第一個分布
        q_dist (Union[np.ndarray, torch.Tensor]): 第二個分布
        method (str, optional): 使用的方法，可以是'js'（Jensen-Shannon散度），
            'kl'（Kullback-Leibler散度），'wasserstein'（Wasserstein距離）
            或'hist'（直方圖交集）。默認為'js'
    
    Returns:
        float: 所選方法下的分布重疊或相似度
    
    Raises:
        ValueError: 如果指定了未知的方法
    """
    if method == 'js':
        # JS散度，轉換為相似度 [0-1]
        js_div = calculate_js_divergence(p_dist, q_dist)
        return float(1.0 - js_div)
    
    elif method == 'kl':
        # KL散度，轉換為有界的相似度 [0-1]
        kl_div = calculate_kl_divergence(p_dist, q_dist)
        return float(np.exp(-kl_div))
    
    elif method == 'wasserstein':
        # Wasserstein距離，轉換為相似度 [0-1]
        w_dist = calculate_wasserstein_distance(p_dist, q_dist)
        return float(1.0 / (1.0 + w_dist))
    
    elif method == 'hist':
        # 直方圖交集，本身就是相似度 [0-1]
        return float(calculate_histogram_intersection(p_dist, q_dist))
    
    else:
        raise ValueError(f"未知的分布對比方法: {method}")

def calculate_distribution_statistics(
    dist: Union[np.ndarray, torch.Tensor]
) -> dict:
    """
    計算分布的統計特性。
    
    包括中心趨勢、變異性和形狀特徵等多種統計量。
    
    Args:
        dist (Union[np.ndarray, torch.Tensor]): 輸入分布
    
    Returns:
        dict: 包含各種統計量的字典
    """
    # 轉換為numpy數組
    if isinstance(dist, torch.Tensor):
        dist = tensor_to_numpy(dist)
    
    # 扁平化
    flat_dist = dist.flatten()
    
    # 基本統計量
    result_stats = {
        'mean': float(np.mean(flat_dist)),
        'median': float(np.median(flat_dist)),
        'std': float(np.std(flat_dist)),
        'var': float(np.var(flat_dist)),
        'min': float(np.min(flat_dist)),
        'max': float(np.max(flat_dist)),
        'range': float(np.max(flat_dist) - np.min(flat_dist)),
    }
    
    # 百分位數
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    for p in percentiles:
        result_stats[f'p{p}'] = float(np.percentile(flat_dist, p))
    
    # 形狀特徵
    try:
        result_stats['skewness'] = float(stats.skew(flat_dist))
        result_stats['kurtosis'] = float(stats.kurtosis(flat_dist))
    except:
        # 如果scipy不可用，使用基本計算
        result_stats['skewness'] = float(0.0)
        result_stats['kurtosis'] = float(0.0)
    
    # 檢測異常值
    outliers = detect_outliers(flat_dist, method='iqr')
    result_stats['outlier_count'] = len(outliers)
    result_stats['outlier_ratio'] = float(len(outliers) / len(flat_dist))
    
    return result_stats

def calculate_distribution_similarity(
    dist1: Union[np.ndarray, torch.Tensor],
    dist2: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """
    計算兩個分布之間的相似度。
    
    此函數是 compare_distributions 的替代名稱，提供相同的功能。
    
    Args:
        dist1 (Union[np.ndarray, torch.Tensor]): 第一個分布
        dist2 (Union[np.ndarray, torch.Tensor]): 第二個分布
    
    Returns:
        Dict[str, float]: 包含各種分布相似度度量的字典
    """
    # 轉換為numpy數組
    if isinstance(dist1, torch.Tensor):
        dist1 = tensor_to_numpy(dist1)
    if isinstance(dist2, torch.Tensor):
        dist2 = tensor_to_numpy(dist2)
    
    return compare_distributions(dist1, dist2)

def compare_tensor_distributions(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    flatten: bool = True
) -> Dict[str, float]:
    """
    比較兩個PyTorch張量的分布。
    
    Args:
        tensor1 (torch.Tensor): 第一個張量
        tensor2 (torch.Tensor): 第二個張量
        flatten (bool, optional): 是否在比較前將張量扁平化，默認為True
    
    Returns:
        Dict[str, float]: 包含各種分布相似度度量的字典
    """
    # 轉換為numpy數組
    if flatten:
        dist1 = tensor_to_numpy(tensor1.flatten())
        dist2 = tensor_to_numpy(tensor2.flatten())
    else:
        dist1 = tensor_to_numpy(tensor1)
        dist2 = tensor_to_numpy(tensor2)
    
    return compare_distributions(dist1, dist2) 