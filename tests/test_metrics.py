"""
# 度量計算模組測試
# 測試分佈指標計算功能

測試日期: 2024-05-28
測試目的: 驗證分佈指標計算模組的功能
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from metrics import distribution_metrics


@pytest.fixture
def mock_distribution_data():
    """
    創建用於測試分佈指標的模擬數據。
    
    Returns:
        dict: 包含不同分佈數據的字典
    """
    np.random.seed(42)  # 確保測試結果可重複
    torch.manual_seed(42)
    
    # 創建測試數據
    data = {
        # 正態分佈數據
        'normal': torch.tensor(np.random.normal(0, 1, 1000)),
        
        # 均勻分佈數據
        'uniform': torch.tensor(np.random.uniform(-1, 1, 1000)),
        
        # 偏態分佈數據（對數正態）
        'skewed': torch.tensor(np.random.lognormal(0, 0.5, 1000)),
        
        # 多峰分佈數據（混合高斯）
        'multimodal': torch.tensor(np.concatenate([
            np.random.normal(-2, 0.5, 500),
            np.random.normal(2, 0.5, 500)
        ])),
        
        # 非常小的分佈（近乎常數）
        'near_constant': torch.tensor(np.random.normal(0, 0.01, 1000)),
        
        # 2D 數據用於測試激活指標
        'activations': torch.tensor(np.random.normal(0, 1, (100, 10)))
    }
    return data


def test_calculate_mean(mock_distribution_data):
    """
    測試計算平均值功能。
    """
    # 測試正態分佈的平均值
    normal_data = mock_distribution_data['normal']
    mean = distribution_metrics.calculate_mean(normal_data)
    
    # 檢查返回值是否為浮點數或張量
    assert isinstance(mean, (float, torch.Tensor))
    # 檢查計算結果是否接近於0（正態分佈的理論均值）
    assert abs(float(mean)) < 0.1
    
    # 測試計算結果是否與 numpy 計算結果一致
    numpy_mean = np.mean(normal_data.numpy())
    assert abs(float(mean) - numpy_mean) < 1e-6


def test_calculate_variance(mock_distribution_data):
    """
    測試計算方差功能。
    """
    # 測試正態分佈的方差
    normal_data = mock_distribution_data['normal']
    variance = distribution_metrics.calculate_variance(normal_data)
    
    # 檢查返回值是否為浮點數或張量
    assert isinstance(variance, (float, torch.Tensor))
    # 檢查計算結果是否接近於1（正態分佈的理論方差）
    assert abs(float(variance) - 1.0) < 0.1
    
    # 測試計算結果是否與 numpy 計算結果一致
    numpy_var = np.var(normal_data.numpy())
    assert abs(float(variance) - numpy_var) < 1e-6


def test_calculate_skewness(mock_distribution_data):
    """
    測試計算偏態功能。
    """
    # 測試偏態分佈的偏度
    skewed_data = mock_distribution_data['skewed']
    skewness = distribution_metrics.calculate_skewness(skewed_data)
    
    # 檢查返回值是否為浮點數或張量
    assert isinstance(skewness, (float, torch.Tensor))
    # 檢查對數正態分佈的偏度是否為正值（右偏）
    assert float(skewness) > 0
    
    # 測試對稱分佈（正態）的偏度是否接近於0
    normal_data = mock_distribution_data['normal']
    normal_skewness = distribution_metrics.calculate_skewness(normal_data)
    assert abs(float(normal_skewness)) < 0.2


def test_calculate_kurtosis(mock_distribution_data):
    """
    測試計算峰度功能。
    """
    # 測試正態分佈的峰度
    normal_data = mock_distribution_data['normal']
    kurtosis = distribution_metrics.calculate_kurtosis(normal_data)
    
    # 檢查返回值是否為浮點數或張量
    assert isinstance(kurtosis, (float, torch.Tensor))
    # 檢查正態分佈的超峰度是否接近於0
    assert abs(float(kurtosis)) < 0.5
    
    # 測試多峰分佈的峰度
    multimodal_data = mock_distribution_data['multimodal']
    multimodal_kurtosis = distribution_metrics.calculate_kurtosis(multimodal_data)
    # 雙峰分佈通常具有負的超峰度
    assert float(multimodal_kurtosis) < 0


def test_calculate_percentile(mock_distribution_data):
    """
    測試計算百分位數功能。
    """
    # 測試均勻分佈的百分位數
    uniform_data = mock_distribution_data['uniform']
    
    # 測試中位數（50%分位數）
    median = distribution_metrics.calculate_percentile(uniform_data, 50)
    # 檢查返回值是否為浮點數或張量
    assert isinstance(median, (float, torch.Tensor))
    # 檢查均勻分佈[-1,1]的中位數是否接近於0
    assert abs(float(median)) < 0.1
    
    # 測試第25百分位數
    p25 = distribution_metrics.calculate_percentile(uniform_data, 25)
    # 檢查均勻分佈[-1,1]的25%分位數是否接近於-0.5
    assert float(p25) < 0
    
    # 測試第75百分位數
    p75 = distribution_metrics.calculate_percentile(uniform_data, 75)
    # 檢查均勻分佈[-1,1]的75%分位數是否接近於0.5
    assert float(p75) > 0


def test_calculate_sparsity(mock_distribution_data):
    """
    測試計算稀疏度功能。
    """
    # 測試正態分佈的稀疏度
    normal_data = mock_distribution_data['normal']
    sparsity = distribution_metrics.calculate_sparsity(normal_data)
    
    # 檢查返回值是否為浮點數或張量
    assert isinstance(sparsity, (float, torch.Tensor))
    # 稀疏度應該在0到1之間
    assert 0 <= float(sparsity) <= 1
    
    # 測試非常接近常數的數據的稀疏度
    near_constant_data = mock_distribution_data['near_constant']
    constant_sparsity = distribution_metrics.calculate_sparsity(near_constant_data)
    # 接近常數的數據應該具有較高的稀疏度
    assert float(constant_sparsity) > float(sparsity)


def test_calculate_activation_pattern(mock_distribution_data):
    """
    測試計算激活模式功能。
    """
    # 測試2D激活數據
    activations = mock_distribution_data['activations']
    pattern = distribution_metrics.calculate_activation_pattern(activations)
    
    # 檢查返回值是否為張量
    assert isinstance(pattern, (np.ndarray, torch.Tensor))
    # 檢查形狀是否正確
    assert pattern.shape[0] == activations.shape[1]  # 神經元數量
    
    # 激活模式的值應該在0到1之間
    assert (pattern >= 0).all()
    assert (pattern <= 1).all()


def test_calculate_entropy(mock_distribution_data):
    """
    測試計算熵功能。
    """
    # 測試正態分佈的熵
    normal_data = mock_distribution_data['normal']
    entropy = distribution_metrics.calculate_entropy(normal_data)
    
    # 檢查返回值是否為浮點數或張量
    assert isinstance(entropy, (float, torch.Tensor))
    # 熵應該為正值
    assert float(entropy) > 0
    
    # 比較不同分佈的熵
    near_constant_data = mock_distribution_data['near_constant']
    constant_entropy = distribution_metrics.calculate_entropy(near_constant_data)
    # 接近常數的分佈應該具有較低的熵
    assert float(constant_entropy) < float(entropy) 