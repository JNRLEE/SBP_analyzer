"""
# 接口模組測試
# 測試分析器接口功能

測試日期: 2024-05-28
測試目的: 驗證分析器接口模組的功能和用戶交互
"""

import os
import pytest
import tempfile
import torch
import numpy as np
import json
from pathlib import Path

from interfaces.analyzer_interface import AnalyzerInterface
from visualization import distribution_plots
from metrics import distribution_metrics


@pytest.fixture
def mock_experiment():
    """
    創建一個模擬的實驗環境，包括實驗目錄、配置和各種測試文件。
    
    Returns:
        dict: 包含實驗路徑和配置的字典
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # 創建實驗子目錄
        models_dir = os.path.join(temp_dir, "models")
        hooks_dir = os.path.join(temp_dir, "hooks")
        results_dir = os.path.join(temp_dir, "results")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(hooks_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # 創建實驗配置文件
        config = {
            "experiment_id": "test_exp_001",
            "experiment_name": "測試實驗",
            "model_name": "test_model",
            "batch_size": 32,
            "learning_rate": 0.001,
            "dataset": "test_dataset",
            "epochs": 10
        }
        
        with open(os.path.join(temp_dir, "config.json"), "w") as f:
            json.dump(config, f)
        
        # 創建訓練歷史文件
        history = {
            "loss": [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25],
            "val_loss": [0.95, 0.85, 0.75, 0.65, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3],
            "accuracy": [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87, 0.9, 0.92],
            "val_accuracy": [0.45, 0.55, 0.65, 0.7, 0.75, 0.77, 0.8, 0.82, 0.85, 0.87]
        }
        
        with open(os.path.join(temp_dir, "training_history.json"), "w") as f:
            json.dump(history, f)
        
        # 創建模擬的模型層激活數據
        epoch_dir = os.path.join(hooks_dir, "epoch_9")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 創建層激活數據
        for layer_idx in range(3):
            layer_activation = torch.randn(100, 20)  # 模擬激活數據
            torch.save(layer_activation, os.path.join(epoch_dir, f"layer_{layer_idx}_activation.pt"))
        
        # 創建層權重數據
        for layer_idx in range(3):
            layer_weights = torch.randn(20, 20)  # 模擬權重數據
            torch.save(layer_weights, os.path.join(epoch_dir, f"layer_{layer_idx}_weights.pt"))
        
        # 創建實驗日誌文件
        with open(os.path.join(temp_dir, "experiments.log"), "w") as f:
            f.write(f"Experiment ID: {config['experiment_id']}\n")
            f.write(f"Date: 2024-05-28\n")
            f.write(f"Model: {config['model_name']}\n")
            f.write(f"Dataset: {config['dataset']}\n")
            f.write(f"Final Loss: {history['loss'][-1]}\n")
            f.write(f"Final Accuracy: {history['accuracy'][-1]}\n")
        
        yield {
            "experiment_dir": temp_dir,
            "config": config,
            "history": history
        }


@pytest.fixture
def analyzer_interface(mock_experiment):
    """
    創建分析器接口實例用於測試。
    
    Args:
        mock_experiment: 模擬實驗環境
        
    Returns:
        AnalyzerInterface: 分析器接口實例
    """
    # 創建分析器接口
    interface = AnalyzerInterface(mock_experiment["experiment_dir"])
    return interface


def test_analyzer_interface_initialization(analyzer_interface, mock_experiment):
    """
    測試分析器接口初始化。
    """
    # 檢查接口是否正確初始化
    assert analyzer_interface.experiment_dir == mock_experiment["experiment_dir"]
    assert hasattr(analyzer_interface, "experiment_loader")
    assert hasattr(analyzer_interface, "hook_data_loader")


def test_analyzer_interface_load_experiment_config(analyzer_interface, mock_experiment):
    """
    測試分析器接口載入實驗配置。
    """
    # 載入配置
    config = analyzer_interface.load_experiment_config()
    
    # 驗證配置內容
    assert config is not None
    assert config["experiment_id"] == mock_experiment["config"]["experiment_id"]
    assert config["experiment_name"] == mock_experiment["config"]["experiment_name"]
    assert config["model_name"] == mock_experiment["config"]["model_name"]
    assert config["batch_size"] == mock_experiment["config"]["batch_size"]


def test_analyzer_interface_load_training_history(analyzer_interface, mock_experiment):
    """
    測試分析器接口載入訓練歷史。
    """
    # 載入訓練歷史
    history = analyzer_interface.load_training_history()
    
    # 驗證歷史數據
    assert history is not None
    assert "loss" in history
    assert "val_loss" in history
    assert "accuracy" in history
    assert "val_accuracy" in history
    assert len(history["loss"]) == mock_experiment["config"]["epochs"]
    assert history["loss"][-1] == mock_experiment["history"]["loss"][-1]


def test_analyzer_interface_analyze_training_dynamics(analyzer_interface, tmp_path):
    """
    測試分析器接口分析訓練動態。
    """
    # 設置輸出路徑
    output_dir = tmp_path
    
    # 分析訓練動態
    metrics = analyzer_interface.analyze_training_dynamics(output_dir=output_dir)
    
    # 驗證返回的指標
    assert metrics is not None
    assert "final_loss" in metrics
    assert "final_val_loss" in metrics
    assert "final_accuracy" in metrics
    assert "final_val_accuracy" in metrics
    assert "convergence_epoch" in metrics
    
    # 檢查生成的圖表文件
    expected_files = [
        "test_exp_001_training_loss.png",
        "test_exp_001_training_accuracy.png"
    ]
    
    for file_name in expected_files:
        assert os.path.exists(os.path.join(output_dir, file_name))


def test_analyzer_interface_analyze_layer_activations(analyzer_interface, tmp_path):
    """
    測試分析器接口分析層激活。
    """
    # 設置輸出路徑
    output_dir = tmp_path
    
    # 分析層激活
    metrics = analyzer_interface.analyze_layer_activations(
        epoch=9,
        layer_indices=[0, 1, 2],
        output_dir=output_dir
    )
    
    # 驗證返回的指標列表
    assert metrics is not None
    assert isinstance(metrics, list)
    assert len(metrics) == 3  # 三個層
    
    # 檢查每個層的指標
    for layer_metric in metrics:
        assert "layer_idx" in layer_metric
        assert "mean" in layer_metric
        assert "variance" in layer_metric
        assert "skewness" in layer_metric
        assert "kurtosis" in layer_metric
        assert "sparsity" in layer_metric
        assert "entropy" in layer_metric
    
    # 檢查生成的圖表文件
    for layer_idx in range(3):
        expected_files = [
            f"test_exp_001_layer_{layer_idx}_activation_distribution.png",
            f"test_exp_001_layer_{layer_idx}_activation_heatmap.png"
        ]
        
        for file_name in expected_files:
            assert os.path.exists(os.path.join(output_dir, file_name))


def test_analyzer_interface_analyze_weight_distributions(analyzer_interface, tmp_path):
    """
    測試分析器接口分析權重分佈。
    """
    # 設置輸出路徑
    output_dir = tmp_path
    
    # 分析權重分佈
    metrics = analyzer_interface.analyze_weight_distributions(
        epoch=9,
        layer_indices=[0, 1, 2],
        output_dir=output_dir
    )
    
    # 驗證返回的指標列表
    assert metrics is not None
    assert isinstance(metrics, list)
    assert len(metrics) == 3  # 三個層
    
    # 檢查每個層的指標
    for layer_metric in metrics:
        assert "layer_idx" in layer_metric
        assert "mean" in layer_metric
        assert "variance" in layer_metric
        assert "l1_norm" in layer_metric
        assert "l2_norm" in layer_metric
    
    # 檢查生成的圖表文件
    for layer_idx in range(3):
        file_name = f"test_exp_001_layer_{layer_idx}_weight_distribution.png"
        assert os.path.exists(os.path.join(output_dir, file_name))


def test_analyzer_interface_generate_experiment_report(analyzer_interface, tmp_path):
    """
    測試分析器接口生成實驗報告。
    """
    # 設置輸出路徑
    output_dir = tmp_path
    
    # 生成實驗報告
    report_path = analyzer_interface.generate_experiment_report(output_dir=output_dir)
    
    # 驗證報告文件
    assert report_path is not None
    assert os.path.exists(report_path)
    
    # 檢查報告內容
    with open(report_path, 'r') as f:
        report_content = f.read()
    
    # 報告應包含實驗ID、名稱等信息
    assert "test_exp_001" in report_content
    assert "測試實驗" in report_content 