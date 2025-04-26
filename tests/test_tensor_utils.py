"""
# 測試張量工具函數

測試日期: 2025-04-30
測試目的: 驗證張量工具模組中函數的功能正確性
"""
import pytest
import torch
import numpy as np
import sys
import os

# 添加專案根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.tensor_utils import (
    to_numpy,
    normalize_tensor,
    preprocess_activations,
    get_tensor_statistics,
    tensor_correlation,
    average_activations,
    extract_feature_maps,
    handle_nan_values,
    reduce_dimensions,
    handle_sparse_tensor,
    batch_process_activations
)


def test_to_numpy():
    """測試張量轉換為NumPy陣列"""
    # 測試PyTorch張量
    torch_tensor = torch.randn(10, 10)
    np_from_torch = to_numpy(torch_tensor)
    assert isinstance(np_from_torch, np.ndarray)
    assert np_from_torch.shape == (10, 10)
    
    # 測試NumPy陣列直接返回
    np_array = np.random.randn(10, 10)
    np_from_np = to_numpy(np_array)
    assert isinstance(np_from_np, np.ndarray)
    assert np_from_np is np_array  # 應該是相同的物件
    
    # 測試Python列表
    py_list = [[1, 2], [3, 4]]
    np_from_list = to_numpy(py_list)
    assert isinstance(np_from_list, np.ndarray)
    assert np_from_list.shape == (2, 2)
    
    # 測試錯誤輸入
    with pytest.raises(TypeError):
        to_numpy("這不是一個有效的輸入")


def test_normalize_tensor_minmax():
    """測試最小最大值正規化"""
    # 測試PyTorch張量
    torch_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = normalize_tensor(torch_tensor, method='minmax')
    assert isinstance(normalized, torch.Tensor)
    assert torch.allclose(normalized, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    
    # 測試NumPy陣列
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = normalize_tensor(np_array, method='minmax')
    assert isinstance(normalized, np.ndarray)
    assert np.allclose(normalized, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    
    # 測試維度參數
    tensor_2d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    normalized = normalize_tensor(tensor_2d, method='minmax', dim=1)
    assert normalized.shape == (2, 3)
    assert torch.allclose(normalized[0], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(normalized[1], torch.tensor([0.0, 0.5, 1.0]))


def test_normalize_tensor_zscore():
    """測試Z分數正規化"""
    # 創建均值為0、標準差為1的數據
    np_array = np.random.normal(0, 1, size=1000)
    normalized = normalize_tensor(np_array, method='zscore')
    
    # Z分數正規化後，均值應接近0，標準差應接近1
    assert -0.1 < np.mean(normalized) < 0.1
    assert 0.9 < np.std(normalized) < 1.1


def test_normalize_tensor_l1_l2():
    """測試L1和L2正規化"""
    # 測試L1正規化
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    l1_norm = normalize_tensor(tensor, method='l1')
    assert torch.isclose(torch.sum(torch.abs(l1_norm)), torch.tensor(1.0))
    
    # 測試L2正規化
    l2_norm = normalize_tensor(tensor, method='l2')
    assert torch.isclose(torch.sqrt(torch.sum(l2_norm**2)), torch.tensor(1.0))


def test_normalize_tensor_error_handling():
    """測試正規化錯誤處理"""
    tensor = torch.randn(10)
    
    # 測試未知方法
    with pytest.raises(ValueError):
        normalize_tensor(tensor, method='unknown_method')
    
    # 測試零張量
    zero_tensor = torch.zeros(10)
    normalized = normalize_tensor(zero_tensor, method='minmax')
    assert torch.all(normalized == 0)  # 應為全零
    
    normalized = normalize_tensor(zero_tensor, method='l2')
    assert torch.all(torch.isfinite(normalized))  # 應是有限值


def test_preprocess_activations():
    """測試激活值預處理"""
    # 創建激活值
    activation = torch.randn(32, 64, 16, 16)
    
    # 測試展平
    processed = preprocess_activations(activation, flatten=True)
    assert processed.shape == (32, 64*16*16)
    
    # 測試正規化
    processed = preprocess_activations(activation, normalize='minmax')
    assert torch.min(processed) >= 0
    assert torch.max(processed) <= 1
    
    # 測試離群值移除
    # 創建包含離群值的激活值
    activation_with_outliers = torch.randn(100)
    activation_with_outliers[0] = 100.0  # 添加離群值
    
    processed = preprocess_activations(activation_with_outliers, remove_outliers=True)
    assert torch.max(processed) < 10  # 離群值應被替換
    
    # 測試組合選項
    processed = preprocess_activations(
        activation, 
        flatten=True, 
        normalize='zscore', 
        remove_outliers=True
    )
    assert processed.shape == (32, 64*16*16)
    assert -5 < processed.mean() < 5
    assert 0 < processed.std() < 5


def test_get_tensor_statistics():
    """測試計算張量統計信息"""
    # 創建張量
    tensor = torch.randn(32, 10)
    
    # 計算統計信息
    stats = get_tensor_statistics(tensor)
    
    # 檢查返回的統計信息
    expected_keys = ['mean', 'std', 'min', 'max']
    for key in expected_keys:
        assert key in stats
    
    # 檢查數值合理性
    assert -2 < stats['mean'] < 2
    assert 0 < stats['std'] < 2
    assert stats['min'] <= stats['max']


def test_tensor_correlation():
    """測試張量相關性計算"""
    # 創建相關的張量
    base = torch.randn(100)
    correlated = base + torch.randn(100) * 0.1  # 高度相關
    anti_correlated = -base + torch.randn(100) * 0.1  # 負相關
    uncorrelated = torch.randn(100)  # 不相關
    
    # 計算相關係數
    corr = tensor_correlation(base, correlated)
    anti_corr = tensor_correlation(base, anti_correlated)
    no_corr = tensor_correlation(base, uncorrelated)
    
    # 檢查結果
    assert 0.8 < corr <= 1.0  # 高度正相關
    assert -1.0 <= anti_corr < -0.8  # 高度負相關
    assert -0.5 < no_corr < 0.5  # 低相關
    
    # 測試不同形狀的張量
    tensor1 = torch.randn(100)
    tensor2 = torch.randn(150)
    corr = tensor_correlation(tensor1, tensor2)
    assert -1.0 <= corr <= 1.0


def test_average_activations():
    """測試計算平均激活值"""
    # 創建激活值字典
    activations = {
        'conv1': torch.randn(32, 64, 16, 16),
        'conv2': torch.randn(32, 128, 8, 8),
        'fc1': torch.randn(32, 256)
    }
    
    # 計算平均激活值
    avg_activations = average_activations(activations)
    
    # 檢查結果
    assert 'conv1' in avg_activations
    assert 'conv2' in avg_activations
    assert 'fc1' in avg_activations
    
    assert avg_activations['conv1'].shape == (32, 64)  # 應該保留批次和通道維度
    assert avg_activations['conv2'].shape == (32, 128)
    assert avg_activations['fc1'].shape == (32,)  # 對於全連接層，保留批次維度


def test_extract_feature_maps():
    """測試提取特徵圖"""
    # 創建激活值
    activation = torch.randn(32, 64, 16, 16)
    
    # 將某些通道設為更活躍
    activation[:, 1, :, :] *= 5
    activation[:, 5, :, :] *= 4
    activation[:, 10, :, :] *= 3
    
    # 提取最活躍的特徵圖
    feature_maps, indices = extract_feature_maps(activation, top_k=3)
    
    # 檢查結果
    assert feature_maps.shape == (32, 3, 16, 16)  # 應提取3個特徵圖
    assert len(indices) == 3
    assert 1 in indices
    assert 5 in indices
    assert 10 in indices
    
    # 測試top_k大於通道數的情況
    feature_maps, indices = extract_feature_maps(activation, top_k=100)
    assert feature_maps.shape == (32, 64, 16, 16)  # 應返回所有特徵圖
    assert len(indices) == 64
    
    # 測試錯誤處理
    with pytest.raises(ValueError):
        extract_feature_maps(torch.randn(32, 64))  # 維度不夠


def test_handle_nan_values():
    """測試處理NaN值的功能"""
    # 創建包含NaN的張量
    tensor = torch.tensor([1.0, float('nan'), 3.0, float('nan'), 5.0])
    
    # 測試zero策略
    processed = handle_nan_values(tensor, strategy='zero')
    assert torch.allclose(processed, torch.tensor([1.0, 0.0, 3.0, 0.0, 5.0]), equal_nan=False)
    
    # 測試fill策略
    processed = handle_nan_values(tensor, strategy='fill', fill_value=999.0)
    assert torch.allclose(processed, torch.tensor([1.0, 999.0, 3.0, 999.0, 5.0]), equal_nan=False)
    
    # 測試mean策略
    processed = handle_nan_values(tensor, strategy='mean')
    assert torch.allclose(processed, torch.tensor([1.0, 3.0, 3.0, 3.0, 5.0]), equal_nan=False)
    
    # 測試median策略
    processed = handle_nan_values(tensor, strategy='median')
    assert torch.allclose(processed, torch.tensor([1.0, 3.0, 3.0, 3.0, 5.0]), equal_nan=False)
    
    # 測試2D張量的插值策略
    tensor_2d = torch.tensor([[1.0, float('nan'), 3.0], 
                             [float('nan'), 5.0, 6.0]])
    processed = handle_nan_values(tensor_2d, strategy='interpolate')
    assert torch.isfinite(processed).all()
    
    # 測試NumPy數組
    np_array = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    processed = handle_nan_values(np_array, strategy='mean')
    assert np.allclose(processed, np.array([1.0, 3.0, 3.0, 3.0, 5.0]), equal_nan=False)
    
    # 測試錯誤處理
    with pytest.raises(ValueError):
        handle_nan_values(tensor, strategy='unknown_strategy')
    
    # 測試無NaN的情況
    no_nan_tensor = torch.tensor([1.0, 2.0, 3.0])
    processed = handle_nan_values(no_nan_tensor)
    assert torch.equal(processed, no_nan_tensor)


def test_reduce_dimensions():
    """測試降維功能"""
    # 創建4D張量
    tensor = torch.randn(32, 64, 16, 16)
    
    # 測試mean降維
    reduced = reduce_dimensions(tensor, method='mean')
    assert reduced.shape == (32, 1, 1, 1)
    
    # 測試max降維
    reduced = reduce_dimensions(tensor, method='max')
    assert reduced.shape == (32, 1, 1, 1)
    
    # 測試sum降維
    reduced = reduce_dimensions(tensor, method='sum')
    assert reduced.shape == (32, 1, 1, 1)
    
    # 測試flatten降維
    reduced = reduce_dimensions(tensor, method='flatten')
    assert reduced.shape == (32*64*16*16,)
    
    # 測試指定維度降維
    reduced = reduce_dimensions(tensor, method='mean', dims=[2, 3])
    assert reduced.shape == (32, 64, 1, 1)
    
    # 測試NumPy數組
    np_array = np.random.randn(32, 64, 16, 16)
    reduced = reduce_dimensions(np_array, method='mean')
    assert reduced.shape == (32, 1, 1, 1)
    
    # 測試錯誤處理
    with pytest.raises(ValueError):
        reduce_dimensions(tensor, method='unknown_method')


def test_handle_sparse_tensor():
    """測試處理稀疏張量功能"""
    # 創建稀疏張量
    dense = torch.zeros(10, 10)
    dense[0, 0] = 1.0
    dense[5, 5] = 2.0
    dense[9, 9] = 3.0
    
    # 測試keep策略
    processed = handle_sparse_tensor(dense, strategy='keep')
    assert torch.equal(processed, dense)
    
    # 測試remove_zeros策略
    processed = handle_sparse_tensor(dense, strategy='remove_zeros')
    assert processed.shape == (3,)
    assert torch.allclose(processed, torch.tensor([1.0, 2.0, 3.0]))
    
    # 測試compress策略
    processed = handle_sparse_tensor(dense, strategy='compress')
    assert processed.is_sparse
    assert torch.equal(processed.to_dense(), dense)
    
    # 測試NumPy數組
    np_array = np.zeros((10, 10))
    np_array[0, 0] = 1.0
    np_array[5, 5] = 2.0
    np_array[9, 9] = 3.0
    
    processed = handle_sparse_tensor(np_array, strategy='remove_zeros')
    assert processed.shape == (3,)
    assert np.allclose(processed, np.array([1.0, 2.0, 3.0]))
    
    # 測試錯誤處理
    with pytest.raises(ValueError):
        handle_sparse_tensor(dense, strategy='unknown_strategy')


def test_batch_process_activations():
    """測試批處理激活值功能"""
    # 創建激活值字典
    activations = {
        'conv1': torch.randn(32, 64, 16, 16),
        'conv2': torch.zeros(32, 128, 8, 8),  # 稀疏層
        'fc1': torch.randn(32, 256)
    }
    
    # 創建操作列表
    operations = [
        {'type': 'preprocess', 'flatten': True},
        {'type': 'normalize', 'method': 'minmax'}
    ]
    
    # 測試批處理
    processed = batch_process_activations(activations, operations)
    
    assert 'conv1' in processed
    assert 'conv2' in processed
    assert 'fc1' in processed
    
    assert processed['conv1'].shape == (32, 64*16*16)
    assert processed['conv2'].shape == (32, 128*8*8)
    assert processed['fc1'].shape == (32, 256)
    
    assert 0 <= processed['conv1'].min() <= processed['conv1'].max() <= 1
    assert 0 <= processed['conv2'].min() <= processed['conv2'].max() <= 1
    assert 0 <= processed['fc1'].min() <= processed['fc1'].max() <= 1
    
    # 測試複雜的操作組合
    complex_operations = [
        {'type': 'handle_nan', 'strategy': 'zero'},
        {'type': 'normalize', 'method': 'zscore'},
        {'type': 'reduce_dims', 'method': 'mean'},
        {'type': 'preprocess', 'flatten': True}
    ]
    
    processed = batch_process_activations(activations, complex_operations)
    for key, value in processed.items():
        assert value.dim() == 2
        assert value.shape[0] == 32
        assert value.shape[1] == 1 