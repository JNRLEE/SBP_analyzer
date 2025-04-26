"""
測試工具模組的單元測試。
"""

import os
# import unittest # 移除 unittest
import numpy as np
import torch
from pathlib import Path
import pytest
import pandas as pd

from utils.file_utils import get_files_by_extension, ensure_dir_exists
from utils.tensor_utils import (
    flatten_tensor, tensor_to_numpy, get_tensor_statistics,
    normalize_tensor, sample_tensor
)
from utils.stat_utils import (
    calculate_moving_average, find_convergence_point,
    calculate_stability_metrics, detect_outliers,
    detect_overfitting_simple,
    calculate_distribution_stats, # Now imported from stat_utils
    detect_distribution_type    # Now imported from stat_utils
)

# --- Fixtures ---
@pytest.fixture
def sample_tensor_normal():
    return torch.randn(16, 32, 8, 8) * 0.1 # Lower std dev

@pytest.fixture
def sample_tensor_sparse():
    t = torch.randn(16, 100)
    mask = torch.rand(16, 100) > 0.9 # 90% sparse
    return t * mask.float()

@pytest.fixture
def sample_tensor_saturated():
    return torch.ones(16, 50) * 0.99 # Close to 1 (assuming range [0,1])

@pytest.fixture
def sample_tensor_dead():
    return torch.zeros(16, 20)

@pytest.fixture
def sample_data_for_outliers():
    # Data with clear outliers
    return np.array([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, -50])

@pytest.fixture
def sample_data_converging():
    # Data that clearly converges
    return np.array([10, 8, 6, 4, 2, 1, 0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.29])

@pytest.fixture
def sample_data_not_converging():
    return np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

@pytest.fixture
def sample_data_stable():
    return np.array([1.0, 1.1, 1.0, 0.9, 1.0, 1.1, 1.0, 0.9, 1.0, 1.1])

@pytest.fixture
def sample_data_unstable():
    return np.array([1, 5, 2, 8, 3, 10, 4, 9, 2, 7])

# --- Test Functions for File Utils (改為函數式) ---

def test_get_files_by_extension(tmp_path):
    """
    測試獲取指定擴展名的文件列表。
    """
    test_dir = tmp_path / "file_utils_test"
    test_dir.mkdir()
    test_files = ['test1.txt', 'test2.txt', 'test.json']
    for file in test_files:
        (test_dir / file).write_text('test')

    # 測試獲取.txt文件
    txt_files = get_files_by_extension(str(test_dir), '.txt')
    assert len(txt_files) == 2

    # 測試獲取.json文件
    json_files = get_files_by_extension(str(test_dir), '.json')
    assert len(json_files) == 1

def test_ensure_dir_exists(tmp_path):
    """
    測試確保目錄存在。
    """
    test_dir = tmp_path / "ensure_dir_test"
    test_dir.mkdir()

    # 測試確保已存在的目錄
    result = ensure_dir_exists(str(test_dir))
    assert result == str(test_dir)
    assert os.path.exists(result)

    # 測試創建新目錄
    new_dir = tmp_path / "ensure_dir_test" / "new_dir"
    if os.path.exists(new_dir):
        os.rmdir(new_dir)

    result = ensure_dir_exists(str(new_dir))
    assert result == str(new_dir)
    assert os.path.exists(result)

# --- Test Functions for Tensor Utils (改為函數式) ---

def test_flatten_tensor():
    """
    測試平坦化張量。
    """
    tensor = torch.randn(2, 3, 4, 5)
    flat_tensor = flatten_tensor(tensor)
    assert flat_tensor.dim() == 1
    assert flat_tensor.numel() == tensor.numel()

def test_tensor_to_numpy():
    """
    測試張量轉換為NumPy數組。
    """
    tensor = torch.randn(2, 3)
    np_array = tensor_to_numpy(tensor)
    assert isinstance(np_array, np.ndarray)
    assert np_array.shape == tensor.shape

    tensor.requires_grad = True
    np_array = tensor_to_numpy(tensor)
    assert isinstance(np_array, np.ndarray)

def test_get_tensor_statistics():
    """
    測試計算張量統計量。
    """
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = get_tensor_statistics(tensor)
    assert isinstance(stats, dict)
    assert 'mean' in stats
    assert stats['mean'] == 3.0
    assert stats['min'] == 1.0
    assert stats['max'] == 5.0

def test_normalize_tensor(sample_tensor_normal):
    """
    測試規範化張量。
    """
    normalized = normalize_tensor(sample_tensor_normal, method='min_max')
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0

    normalized_z = normalize_tensor(sample_tensor_normal, method='z_score')
    assert np.isclose(normalized_z.mean(), 0.0, atol=1e-6)
    assert np.isclose(normalized_z.std(), 1.0, atol=1e-6)

    with pytest.raises(ValueError):
        normalize_tensor(sample_tensor_normal, method='invalid_method')

def test_sample_tensor():
    """
    測試從張量中抽樣。
    """
    tensor = torch.randn(1000)
    max_samples = 100
    sampled = sample_tensor(tensor, max_samples=max_samples)
    assert len(sampled) == max_samples

    small_tensor = torch.randn(50)
    sampled = sample_tensor(small_tensor, max_samples=max_samples)
    assert len(sampled) == 50

# --- Test Functions for Stat Utils (改為函數式) ---

def test_calculate_moving_average():
    """
    測試計算移動平均。
    """
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    window_size = 3
    ma = calculate_moving_average(values, window_size=window_size)
    assert len(ma) == len(values) - window_size + 1
    assert ma[0] == 2.0
    assert ma[-1] == 9.0

def test_find_convergence_point(sample_data_converging, sample_data_not_converging):
    """
    測試查找收斂點。
    """
    convergence_point = find_convergence_point(sample_data_converging, window_size=3, threshold=0.05)
    assert convergence_point is not None
    assert convergence_point >= len(sample_data_converging) - 5

    no_convergence_point = find_convergence_point(sample_data_not_converging, window_size=3, threshold=0.01)
    assert no_convergence_point is None

def test_calculate_stability_metrics(sample_data_stable, sample_data_unstable):
    """
    測試計算穩定性指標。
    """
    stable_metrics = calculate_stability_metrics(sample_data_stable, window_size=5)
    unstable_metrics = calculate_stability_metrics(sample_data_unstable, window_size=5)

    assert 'overall_std' in stable_metrics
    assert stable_metrics['overall_std'] < unstable_metrics['overall_std']
    assert stable_metrics['last_window_std'] < unstable_metrics['last_window_std']

def test_detect_outliers(sample_data_for_outliers):
    """
    測試檢測異常值。
    """
    outlier_info = detect_outliers(sample_data_for_outliers, method='iqr', factor=1.5)
    assert isinstance(outlier_info, dict)
    assert outlier_info['outliers_count'] == 2
    assert set(outlier_info['outliers_indices']) == {5, 10}
    assert set(outlier_info['outliers_values']) == {100, -50}

# Test the functions moved to stat_utils
def test_moved_calculate_distribution_stats(sample_tensor_normal, sample_tensor_dead):
    stats_normal = calculate_distribution_stats(sample_tensor_normal)
    assert 'mean' in stats_normal
    assert stats_normal['sparsity'] < 0.01

    stats_dead = calculate_distribution_stats(sample_tensor_dead)
    assert stats_dead['mean'] == 0.0
    assert stats_dead['sparsity'] == 1.0
    assert np.isnan(stats_dead['entropy']) or np.isneginf(stats_dead['entropy'])

def test_moved_detect_distribution_type(sample_tensor_normal, sample_tensor_sparse, sample_tensor_saturated, sample_tensor_dead):
    # 檢查稀疏性
    sparse_data = sample_tensor_sparse.detach().cpu().numpy().flatten()
    zero_count = np.sum(np.abs(sparse_data) < 1e-6)
    total_count = sparse_data.size
    manual_sparsity = zero_count / total_count
    print(f"\n調試信息 - 稀疏張量：零值比例 = {manual_sparsity:.4f}, 零值數量 = {zero_count}, 總元素數 = {total_count}")
    
    # 執行測試
    assert detect_distribution_type(sample_tensor_normal) == 'normal'
    assert detect_distribution_type(sample_tensor_sparse) == 'sparse'
    assert detect_distribution_type(sample_tensor_saturated, saturation_threshold=0.95, saturation_range=(0, 1)) == 'saturated'
    assert detect_distribution_type(sample_tensor_dead, sparse_threshold=0.99) == 'sparse'

def test_detect_overfitting_simple():
    """
    測試簡單的過擬合檢測功能 (現在從 utils.stat_utils 導入)。
    """
    train_loss_good = np.array([1.0, 0.8, 0.6, 0.4, 0.3])
    val_loss_good = np.array([1.1, 0.9, 0.7, 0.5, 0.4])
    train_loss_overfit = np.array([1.0, 0.8, 0.6, 0.4, 0.3])
    val_loss_overfit = np.array([1.1, 0.9, 0.7, 0.8, 0.9])

    # 測試正常情況（無過擬合）
    status_good = detect_overfitting_simple(train_loss_good, val_loss_good)
    assert status_good == 'No sign of overfitting'

    # 測試過擬合情況
    status_overfit = detect_overfitting_simple(train_loss_overfit, val_loss_overfit)
    assert status_overfit == 'Overfitting'

# Removed: if __name__ == '__main__': unittest.main() 