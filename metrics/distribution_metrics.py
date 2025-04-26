"""
這個模組提供了計算和比較數據分布之間差異的函數。
可用於分析神經網絡中不同層的激活值分布、權重分布等。
"""

import numpy as np
import torch
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, Union, List, Tuple, Optional, Any


def calculate_histogram_intersection(
    dist1: Union[np.ndarray, torch.Tensor], 
    dist2: Union[np.ndarray, torch.Tensor], 
    bins: int = 100
) -> float:
    """
    計算兩個分布之間的直方圖交叉值，用來衡量分布的相似度。
    
    Args:
        dist1: 第一個分布的數據
        dist2: 第二個分布的數據
        bins: 用於構建直方圖的bins數量
        
    Returns:
        兩個直方圖的交叉值 (0到1之間，1表示完全相同)
    """
    # 將張量轉換為numpy數組
    if isinstance(dist1, torch.Tensor):
        dist1 = dist1.detach().cpu().numpy().flatten()
    else:
        dist1 = np.asarray(dist1).flatten()
        
    if isinstance(dist2, torch.Tensor):
        dist2 = dist2.detach().cpu().numpy().flatten()
    else:
        dist2 = np.asarray(dist2).flatten()
    
    # 計算兩個分布的範圍
    min_val = min(np.min(dist1), np.min(dist2))
    max_val = max(np.max(dist1), np.max(dist2))
    
    # 創建共同的bin範圍
    bin_range = (min_val, max_val)
    
    # 計算直方圖
    hist1, bin_edges = np.histogram(dist1, bins=bins, range=bin_range, density=True)
    hist2, _ = np.histogram(dist2, bins=bins, range=bin_range, density=True)
    
    # 正規化直方圖
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # 計算交叉值
    intersection = np.sum(np.minimum(hist1, hist2))
    
    return float(intersection)


def calculate_distribution_similarity(
    dist1: Union[np.ndarray, torch.Tensor], 
    dist2: Union[np.ndarray, torch.Tensor], 
    method: str = 'kl_divergence'
) -> float:
    """
    使用指定的方法計算兩個分布之間的相似度或距離。
    
    Args:
        dist1: 第一個分布的數據
        dist2: 第二個分布的數據
        method: 相似度計算方法，選項包括：
            - 'kl_divergence': KL散度 (值越小越相似)
            - 'js_divergence': JS散度 (值越小越相似)
            - 'wasserstein': Wasserstein距離 (值越小越相似)
            - 'correlation': Pearson相關係數 (值越接近1越相似)
        
    Returns:
        兩個分布之間的相似度或距離度量
    """
    # 將張量轉換為numpy數組
    if isinstance(dist1, torch.Tensor):
        dist1 = dist1.detach().cpu().numpy().flatten()
    else:
        dist1 = np.asarray(dist1).flatten()
        
    if isinstance(dist2, torch.Tensor):
        dist2 = dist2.detach().cpu().numpy().flatten()
    else:
        dist2 = np.asarray(dist2).flatten()
    
    if method == 'kl_divergence':
        # 使用epsilon避免log(0)的情況
        epsilon = 1e-10
        
        # 計算直方圖
        bins = min(100, len(dist1) // 10)
        hist1, bin_edges = np.histogram(dist1, bins=bins, density=True)
        hist2, _ = np.histogram(dist2, bins=bin_edges, density=True)
        
        # 添加平滑以避免零概率
        hist1 = hist1 + epsilon
        hist2 = hist2 + epsilon
        
        # 正規化
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        # 計算KL散度
        kl_div = np.sum(hist1 * np.log(hist1 / hist2))
        return float(kl_div)
    
    elif method == 'js_divergence':
        # 計算直方圖
        bins = min(100, len(dist1) // 10)
        hist1, bin_edges = np.histogram(dist1, bins=bins, density=True)
        hist2, _ = np.histogram(dist2, bins=bin_edges, density=True)
        
        # 添加平滑以避免零概率
        epsilon = 1e-10
        hist1 = hist1 + epsilon
        hist2 = hist2 + epsilon
        
        # 正規化
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        # 計算JS散度
        js_div = jensenshannon(hist1, hist2)
        return float(js_div)
    
    elif method == 'wasserstein':
        # 計算Wasserstein距離 (Earth Mover's Distance)
        return float(stats.wasserstein_distance(dist1, dist2))
    
    elif method == 'correlation':
        # 計算Pearson相關係數
        return float(np.corrcoef(dist1, dist2)[0, 1])
    
    else:
        raise ValueError(f"未知的相似度計算方法: {method}")


def compare_tensor_distributions(
    tensor1: torch.Tensor, 
    tensor2: torch.Tensor,
    dim: Optional[int] = None,
    methods: List[str] = ['histogram_intersection', 'kl_divergence', 'wasserstein']
) -> Dict[str, Union[float, List[float]]]:
    """
    比較兩個張量的分布，可以沿著指定維度進行比較。
    
    Args:
        tensor1: 第一個張量
        tensor2: 第二個張量
        dim: 要比較的維度，若為None則比較整個張量
        methods: 要使用的比較方法列表
        
    Returns:
        包含各種相似度度量的字典
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"張量形狀不匹配: {tensor1.shape} vs {tensor2.shape}")
    
    results = {}
    
    if dim is None:
        # 比較整個張量
        flat_tensor1 = tensor1.reshape(-1)
        flat_tensor2 = tensor2.reshape(-1)
        
        for method in methods:
            if method == 'histogram_intersection':
                results[method] = calculate_histogram_intersection(flat_tensor1, flat_tensor2)
            else:
                results[method] = calculate_distribution_similarity(flat_tensor1, flat_tensor2, method=method)
    else:
        # 沿著指定維度比較
        slices = []
        for i in range(tensor1.size(dim)):
            if dim == 0:
                slice1 = tensor1[i].reshape(-1)
                slice2 = tensor2[i].reshape(-1)
            elif dim == 1:
                slice1 = tensor1[:, i].reshape(-1)
                slice2 = tensor2[:, i].reshape(-1)
            elif dim == 2:
                slice1 = tensor1[:, :, i].reshape(-1)
                slice2 = tensor2[:, :, i].reshape(-1)
            else:
                raise ValueError(f"目前僅支持0, 1, 2維度的比較, 收到: {dim}")
            
            slices.append((slice1, slice2))
        
        # 初始化結果字典
        for method in methods:
            results[method] = []
        
        # 計算每個切片的相似度
        for slice1, slice2 in slices:
            for method in methods:
                if method == 'histogram_intersection':
                    results[method].append(calculate_histogram_intersection(slice1, slice2))
                else:
                    results[method].append(calculate_distribution_similarity(slice1, slice2, method=method))
    
    return results


def calculate_distribution_stats(
    dist: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """
    計算分布的各種統計量。
    
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
            - kurtosis: 峰度
    """
    # 將張量轉換為numpy數組
    if isinstance(dist, torch.Tensor):
        dist_np = dist.detach().cpu().numpy().flatten()
    else:
        dist_np = np.asarray(dist).flatten()
    
    # 計算基本統計量
    stats_dict = {
        'mean': float(np.mean(dist_np)),
        'std': float(np.std(dist_np)),
        'min': float(np.min(dist_np)),
        'max': float(np.max(dist_np)),
        'median': float(np.median(dist_np)),
        'skewness': float(stats.skew(dist_np)),
        'kurtosis': float(stats.kurtosis(dist_np))
    }
    
    return stats_dict


def calculate_distribution_entropy(
    dist: Union[np.ndarray, torch.Tensor], 
    bins: int = 100
) -> float:
    """
    計算分布的香農熵。
    
    Args:
        dist: 輸入分布數據
        bins: 用於構建直方圖的bins數量
        
    Returns:
        分布的香農熵值 (以位元為單位)
    """
    # 將張量轉換為numpy數組
    if isinstance(dist, torch.Tensor):
        dist_np = dist.detach().cpu().numpy().flatten()
    else:
        dist_np = np.asarray(dist).flatten()
    
    # 創建直方圖
    counts, _ = np.histogram(dist_np, bins=bins)
    
    # 計算概率
    probabilities = counts / np.sum(counts)
    
    # 過濾掉零概率 (log(0)會導致問題)
    probabilities = probabilities[probabilities > 0]
    
    # 計算熵
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return float(entropy) 