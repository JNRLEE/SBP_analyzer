"""
# 數據載入器測試
# 測試數據載入器的基本功能

測試日期: 2024-05-28
測試目的: 驗證數據載入器能夠正確地載入和解析實驗數據
"""

import os
import pytest
import json
import tempfile
import torch
import numpy as np
from pathlib import Path

from data_loader import ExperimentLoader, HookDataLoader


@pytest.fixture
def mock_experiment_dir():
    """
    創建模擬的實驗目錄結構和文件用於測試。
    
    Returns:
        str: 臨時實驗目錄的路徑
    """
    # 創建臨時目錄
    with tempfile.TemporaryDirectory() as temp_dir:
        # 創建必要的子目錄
        models_dir = os.path.join(temp_dir, "models")
        hooks_dir = os.path.join(temp_dir, "hooks")
        results_dir = os.path.join(temp_dir, "results")
        epoch_0_dir = os.path.join(hooks_dir, "epoch_0")
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(epoch_0_dir, exist_ok=True)
        
        # 創建模擬的配置文件
        config = {
            "experiment_name": "test_experiment",
            "model": "test_model",
            "batch_size": 32,
            "learning_rate": 0.001
        }
        
        with open(os.path.join(temp_dir, "config.json"), "w") as f:
            json.dump(config, f)
        
        # 創建模擬的模型結構文件
        model_structure = {
            "layers": [
                {"name": "layer1", "type": "Linear", "params": 1024},
                {"name": "layer2", "type": "ReLU", "params": 0}
            ],
            "total_params": 1024
        }
        
        with open(os.path.join(temp_dir, "model_structure.json"), "w") as f:
            json.dump(model_structure, f)
        
        # 創建模擬的訓練歷史文件
        training_history = {
            "loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "accuracy": [0.7, 0.8, 0.9],
            "val_accuracy": [0.6, 0.7, 0.8]
        }
        
        with open(os.path.join(temp_dir, "training_history.json"), "w") as f:
            json.dump(training_history, f)
        
        # 創建模擬的結果文件
        results = {
            "final_loss": 0.3,
            "final_accuracy": 0.9
        }
        
        with open(os.path.join(results_dir, "results.json"), "w") as f:
            json.dump(results, f)
        
        # 創建模擬的模型檢查點文件
        with open(os.path.join(models_dir, "checkpoint_epoch_0.pth"), "w") as f:
            f.write("dummy model checkpoint")
        
        # 創建模擬的鉤子數據
        # 輪次摘要
        epoch_summary = {
            "epoch": 0,
            "loss": 0.5,
            "accuracy": 0.7
        }
        torch.save(epoch_summary, os.path.join(epoch_0_dir, "epoch_summary.pt"))
        
        # 層激活值
        layer_activation = torch.randn(10, 5)  # 模擬張量
        torch.save(layer_activation, os.path.join(epoch_0_dir, "layer1_activation_batch_0.pt"))
        
        yield temp_dir


def test_experiment_loader_init(mock_experiment_dir):
    """
    測試 ExperimentLoader 的初始化。
    """
    loader = ExperimentLoader(mock_experiment_dir)
    assert loader.experiment_dir == mock_experiment_dir


def test_experiment_loader_load_config(mock_experiment_dir):
    """
    測試 ExperimentLoader 載入配置文件。
    """
    loader = ExperimentLoader(mock_experiment_dir)
    config = loader.load(data_type='config')
    assert config is not None
    assert config['experiment_name'] == "test_experiment"
    assert config['model'] == "test_model"
    assert config['batch_size'] == 32
    assert config['learning_rate'] == 0.001


def test_experiment_loader_load_model_structure(mock_experiment_dir):
    """
    測試 ExperimentLoader 載入模型結構文件。
    """
    loader = ExperimentLoader(mock_experiment_dir)
    structure = loader.load(data_type='model_structure')
    assert structure is not None
    assert 'layers' in structure
    assert structure['total_params'] == 1024


def test_experiment_loader_load_training_history(mock_experiment_dir):
    """
    測試 ExperimentLoader 載入訓練歷史文件。
    """
    loader = ExperimentLoader(mock_experiment_dir)
    history = loader.load(data_type='training_history')
    assert history is not None
    assert 'loss' in history
    assert len(history['loss']) == 3
    assert 'accuracy' in history
    assert history['loss'][0] == 0.5


def test_experiment_loader_load_results(mock_experiment_dir):
    """
    測試 ExperimentLoader 載入結果文件。
    """
    loader = ExperimentLoader(mock_experiment_dir)
    results = loader.load(data_type='results')
    assert results is not None
    assert results['final_loss'] == 0.3
    assert results['final_accuracy'] == 0.9


def test_experiment_loader_list_available_models(mock_experiment_dir):
    """
    測試 ExperimentLoader 列出可用的模型檢查點。
    """
    loader = ExperimentLoader(mock_experiment_dir)
    models = loader.list_available_models()
    assert models is not None
    assert len(models) == 1
    assert "checkpoint_epoch_0.pth" in models


def test_hook_data_loader_init(mock_experiment_dir):
    """
    測試 HookDataLoader 的初始化。
    """
    loader = HookDataLoader(mock_experiment_dir)
    assert loader.experiment_dir == mock_experiment_dir
    assert loader.hooks_dir == os.path.join(mock_experiment_dir, 'hooks')


def test_hook_data_loader_list_available_epochs(mock_experiment_dir):
    """
    測試 HookDataLoader 列出可用的輪次。
    """
    loader = HookDataLoader(mock_experiment_dir)
    epochs = loader.list_available_epochs()
    assert epochs is not None
    assert len(epochs) == 1
    assert 0 in epochs


def test_hook_data_loader_load_epoch_summary(mock_experiment_dir):
    """
    測試 HookDataLoader 載入輪次摘要。
    """
    loader = HookDataLoader(mock_experiment_dir)
    summary = loader.load(data_type='epoch_summary', epoch=0)
    assert summary is not None
    assert summary['epoch'] == 0
    assert summary['loss'] == 0.5
    assert summary['accuracy'] == 0.7


def test_hook_data_loader_load_layer_activation(mock_experiment_dir):
    """
    測試 HookDataLoader 載入層激活值。
    """
    loader = HookDataLoader(mock_experiment_dir)
    activation = loader.load(data_type='layer_activation', layer_name='layer1', epoch=0, batch=0)
    assert activation is not None
    assert isinstance(activation, torch.Tensor)
    assert activation.shape == (10, 5)


def test_hook_data_loader_list_available_layer_activations(mock_experiment_dir):
    """
    測試 HookDataLoader 列出可用的層激活值。
    """
    loader = HookDataLoader(mock_experiment_dir)
    layers = loader.list_available_layer_activations(0)
    assert layers is not None
    assert len(layers) == 1
    assert 'layer1' in layers 