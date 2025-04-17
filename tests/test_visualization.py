"""
# 視覺化模組測試
# 測試分佈和性能繪圖功能

測試日期: 2024-05-28
測試目的: 驗證視覺化模組能夠正確生成圖表並保存
"""

import os
import pytest
import tempfile
import torch
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pathlib import Path

from visualization import distribution_plots, performance_plots


@pytest.fixture
def temp_output_dir():
    """
    創建臨時輸出目錄用於保存生成的圖表。
    
    Returns:
        str: 臨時輸出目錄的路徑
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_data():
    """
    創建用於視覺化測試的模擬數據。
    
    Returns:
        dict: 包含各種類型數據的字典
    """
    # 創建測試數據
    data = {
        'layer_values': torch.randn(100),
        'layer_activations': torch.randn(100, 10),
        'layer_weights': torch.randn(10, 20),
        'training_history': {
            'loss': [0.9, 0.8, 0.7, 0.6, 0.5],
            'val_loss': [0.95, 0.85, 0.75, 0.65, 0.55],
            'accuracy': [0.5, 0.6, 0.7, 0.8, 0.9],
            'val_accuracy': [0.45, 0.55, 0.65, 0.75, 0.85]
        },
        'model_performance': {
            'precision': 0.8,
            'recall': 0.7,
            'f1_score': 0.75,
            'accuracy': 0.85
        },
        'experiment_id': 'test_exp_001'
    }
    return data


def test_plot_distribution(mock_data, temp_output_dir):
    """
    測試繪製分佈圖的功能。
    """
    layer_values = mock_data['layer_values']
    
    # 測試正態分佈圖
    fig = distribution_plots.plot_distribution(
        layer_values, 
        title="測試分佈圖",
        bins=20
    )
    
    assert isinstance(fig, Figure)
    
    # 測試保存圖表
    output_path = os.path.join(temp_output_dir, f"{mock_data['experiment_id']}_test_plot_distribution.png")
    plt.savefig(output_path)
    plt.close(fig)
    
    assert os.path.exists(output_path)


def test_plot_activation_heatmap(mock_data, temp_output_dir):
    """
    測試繪製激活熱圖的功能。
    """
    activations = mock_data['layer_activations']
    
    fig = distribution_plots.plot_activation_heatmap(
        activations,
        title="激活熱圖測試",
        xlabel="神經元",
        ylabel="樣本"
    )
    
    assert isinstance(fig, Figure)
    
    # 測試保存圖表
    output_path = os.path.join(temp_output_dir, f"{mock_data['experiment_id']}_test_plot_activation_heatmap.png")
    plt.savefig(output_path)
    plt.close(fig)
    
    assert os.path.exists(output_path)


def test_plot_weight_distribution(mock_data, temp_output_dir):
    """
    測試繪製權重分佈圖的功能。
    """
    weights = mock_data['layer_weights']
    
    fig = distribution_plots.plot_weight_distribution(
        weights,
        layer_name="測試層",
        title="權重分佈測試"
    )
    
    assert isinstance(fig, Figure)
    
    # 測試保存圖表
    output_path = os.path.join(temp_output_dir, f"{mock_data['experiment_id']}_test_plot_weight_distribution.png")
    plt.savefig(output_path)
    plt.close(fig)
    
    assert os.path.exists(output_path)


def test_plot_training_curves(mock_data, temp_output_dir):
    """
    測試繪製訓練曲線的功能。
    """
    history = mock_data['training_history']
    
    fig = performance_plots.plot_training_curves(
        history['loss'],
        history['val_loss'],
        title="損失曲線測試",
        xlabel="輪次",
        ylabel="損失值"
    )
    
    assert isinstance(fig, Figure)
    
    # 測試保存圖表
    output_path = os.path.join(temp_output_dir, f"{mock_data['experiment_id']}_test_plot_training_curves.png")
    plt.savefig(output_path)
    plt.close(fig)
    
    assert os.path.exists(output_path)


def test_plot_metrics_comparison(mock_data, temp_output_dir):
    """
    測試繪製指標比較圖的功能。
    """
    metrics = mock_data['model_performance']
    metrics_list = list(metrics.values())
    labels = list(metrics.keys())
    
    fig = performance_plots.plot_metrics_comparison(
        metrics_list,
        labels=labels,
        title="模型性能指標比較"
    )
    
    assert isinstance(fig, Figure)
    
    # 測試保存圖表
    output_path = os.path.join(temp_output_dir, f"{mock_data['experiment_id']}_test_plot_metrics_comparison.png")
    plt.savefig(output_path)
    plt.close(fig)
    
    assert os.path.exists(output_path)


def test_plot_confusion_matrix(mock_data, temp_output_dir):
    """
    測試繪製混淆矩陣的功能。
    """
    # 創建模擬混淆矩陣
    confusion_matrix = np.array([[80, 20], [10, 90]])
    class_names = ['類別 0', '類別 1']
    
    fig = performance_plots.plot_confusion_matrix(
        confusion_matrix,
        class_names=class_names,
        title="混淆矩陣測試"
    )
    
    assert isinstance(fig, Figure)
    
    # 測試保存圖表
    output_path = os.path.join(temp_output_dir, f"{mock_data['experiment_id']}_test_plot_confusion_matrix.png")
    plt.savefig(output_path)
    plt.close(fig)
    
    assert os.path.exists(output_path) 