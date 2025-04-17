"""
# 分佈指標計算
# 提供計算數據分佈相關統計量的函數
"""

import numpy as np
from typing import Dict, Union, Optional, List, Tuple


def calculate_basic_statistics(data: Union[np.ndarray, List]) -> Dict[str, float]:
    """
    計算數據的基本統計量。
    
    Args:
        data: 要分析的數據，可以是 numpy 數組或列表
        
    Returns:
        Dict[str, float]: 包含基本統計指標的字典
        
    Description:
        計算數據的均值、中位數、最小值、最大值、標準差、方差、偏度和峰度等基本統計量。
        
    References:
        - NumPy Statistics: https://numpy.org/doc/stable/reference/routines.statistics.html
    """
    # 確保數據是 numpy 數組
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # 處理空數據
    if data.size == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
            'std': np.nan,
            'var': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'q1': np.nan,
            'q3': np.nan,
            'iqr': np.nan,
            'count': 0,
            'non_zero_count': 0,
            'zero_ratio': np.nan
        }
    
    # 處理包含 NaN 或 Inf 的數據
    data_clean = data[np.isfinite(data)]
    
    # 計算基本統計量
    count = data.size
    non_zero_count = np.count_nonzero(data_clean)
    
    stats = {
        'mean': np.mean(data_clean),
        'median': np.median(data_clean),
        'min': np.min(data_clean),
        'max': np.max(data_clean),
        'std': np.std(data_clean),
        'var': np.var(data_clean),
        'count': count,
        'non_zero_count': non_zero_count,
        'zero_ratio': 1.0 - (non_zero_count / count) if count > 0 else np.nan
    }
    
    # 計算分位數
    q1, q3 = np.percentile(data_clean, [25, 75])
    stats['q1'] = q1
    stats['q3'] = q3
    stats['iqr'] = q3 - q1
    
    # 計算偏度和峰度
    # 偏度：數據分佈的不對稱程度
    # 峰度：數據分佈的尖峭程度
    try:
        from scipy import stats as sp_stats
        stats['skewness'] = sp_stats.skew(data_clean)
        stats['kurtosis'] = sp_stats.kurtosis(data_clean)
    except ImportError:
        # 如果沒有 scipy，使用簡化版本的計算
        mean = stats['mean']
        std = stats['std']
        if std > 0:
            # 計算偏度的近似值
            skewness = np.mean(((data_clean - mean) / std) ** 3)
            # 計算峰度的近似值
            kurtosis = np.mean(((data_clean - mean) / std) ** 4) - 3
            stats['skewness'] = skewness
            stats['kurtosis'] = kurtosis
        else:
            stats['skewness'] = np.nan
            stats['kurtosis'] = np.nan
    
    return stats


def calculate_distribution_comparison(
    data1: Union[np.ndarray, List],
    data2: Union[np.ndarray, List],
    bins: Optional[int] = None,
    range: Optional[Tuple[float, float]] = None
) -> Dict[str, float]:
    """
    比較兩個數據分佈的差異。
    
    Args:
        data1: 第一個數據集
        data2: 第二個數據集
        bins: 用於計算直方圖的區間數量
        range: 用於計算直方圖的數據範圍
        
    Returns:
        Dict[str, float]: 包含比較指標的字典
        
    Description:
        計算兩個數據分佈之間的基本差異指標，包括均值差、中位數差、分佈重疊度等。
        
    References:
        None
    """
    # 確保數據是 numpy 數組
    if not isinstance(data1, np.ndarray):
        data1 = np.array(data1)
    if not isinstance(data2, np.ndarray):
        data2 = np.array(data2)
    
    # 計算各自的基本統計量
    stats1 = calculate_basic_statistics(data1)
    stats2 = calculate_basic_statistics(data2)
    
    # 計算差異
    comparison = {
        'mean_diff': stats2['mean'] - stats1['mean'],
        'median_diff': stats2['median'] - stats1['median'],
        'std_ratio': stats2['std'] / stats1['std'] if stats1['std'] > 0 else np.nan,
        'range_ratio': (stats2['max'] - stats2['min']) / (stats1['max'] - stats1['min']) if (stats1['max'] - stats1['min']) > 0 else np.nan
    }
    
    # 計算分佈重疊度 (通過直方圖)
    if bins is None:
        # 估計適當的 bins 數量
        bins = int(np.sqrt(min(data1.size, data2.size)))
        bins = max(10, min(bins, 100))  # 至少 10 個，最多 100 個
    
    if range is None:
        # 估計適當的範圍
        min_val = min(np.min(data1), np.min(data2))
        max_val = max(np.max(data1), np.max(data2))
        range = (min_val, max_val)
    
    # 計算直方圖
    hist1, bin_edges = np.histogram(data1, bins=bins, range=range, density=True)
    hist2, _ = np.histogram(data2, bins=bins, range=range, density=True)
    
    # 計算直方圖重疊度 (Histogram Intersection)
    # 重疊度 = 兩個直方圖共同部分的面積 / 兩個直方圖最大可能共同部分的面積
    min_overlap = np.sum(np.minimum(hist1, hist2))
    max_overlap = np.sum(np.maximum(hist1, hist2))
    comparison['histogram_overlap'] = min_overlap / max_overlap if max_overlap > 0 else 0.0
    
    return comparison 