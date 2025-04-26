"""
此模組包含SBP_analyzer的端到端測試案例，
測試整個分析流程，從數據載入到報告生成。
"""

import os
import pytest
import torch
import numpy as np
import json
import shutil
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime

from data_loader.hook_data_loader import HookDataLoader
from data_loader.experiment_loader import ExperimentLoader
from metrics.layer_activity_metrics import (
    calculate_activation_statistics,
    calculate_activation_sparsity,
    detect_dead_neurons
)
from reporter.report_generator import ReportGenerator
from utils.tensor_utils import normalize_tensor_zscore

# 測試目錄結構常量
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'test_output')

@pytest.fixture(scope="module")
def setup_test_data():
    """
    設置測試數據環境
    
    該fixture會創建一個模擬的實驗結果目錄結構，包含：
    - config.json: 實驗配置
    - model_structure.json: 模型結構信息
    - training_history.json: 訓練歷史
    - hooks/: 激活值數據
      - training_summary.pt: 訓練摘要
      - epoch_0/: 第0輪數據
        - layer1_activation_batch_0.pt: 層1的激活值
        - layer2_activation_batch_0.pt: 層2的激活值
    """
    # 創建測試目錄
    experiment_dir = os.path.join(TEST_DATA_DIR, 'mock_experiment')
    hooks_dir = os.path.join(experiment_dir, 'hooks')
    epoch0_dir = os.path.join(hooks_dir, 'epoch_0')
    epoch1_dir = os.path.join(hooks_dir, 'epoch_1')
    
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(hooks_dir, exist_ok=True)
    os.makedirs(epoch0_dir, exist_ok=True)
    os.makedirs(epoch1_dir, exist_ok=True)
    
    # 1. 創建 config.json
    config = {
        "experiment_name": "mock_test",
        "model": {
            "type": "test_model",
            "params": {
                "num_layers": 2,
                "hidden_dim": 64
            }
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
    }
    
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 2. 創建 model_structure.json
    model_structure = {
        "model_name": "TestModel",
        "model_type": "classification",
        "total_parameters": 10000,
        "total_layers": 5,
        "layer_types": {
            "Linear": 2,
            "Conv2d": 2,
            "BatchNorm2d": 1
        },
        "layers": [
            {
                "name": "layer1",
                "type": "Conv2d",
                "parameters": 1000,
                "shape": [64, 3, 3, 3]
            },
            {
                "name": "layer2",
                "type": "Linear",
                "parameters": 9000,
                "shape": [10, 64]
            }
        ]
    }
    
    with open(os.path.join(experiment_dir, 'model_structure.json'), 'w') as f:
        json.dump(model_structure, f, indent=2)
    
    # 3. 創建 training_history.json
    training_history = {
        "loss": [2.0, 1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.55, 0.5, 0.48],
        "val_loss": [2.1, 1.6, 1.3, 1.1, 0.9, 0.85, 0.8, 0.75, 0.7, 0.68],
        "accuracy": [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87, 0.89, 0.9],
        "val_accuracy": [0.48, 0.58, 0.68, 0.72, 0.76, 0.78, 0.8, 0.82, 0.84, 0.85],
        "lr": [0.001, 0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0005, 0.0001, 0.0001, 0.0001]
    }
    
    with open(os.path.join(experiment_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # 4. 創建模擬的激活值數據
    # 層1的激活值：正常分布
    layer1_activation = torch.randn(32, 64, 7, 7)
    torch.save(layer1_activation, os.path.join(epoch0_dir, 'layer1_activation_batch_0.pt'))
    
    # 層2的激活值：包含一些死亡神經元
    layer2_activation = torch.zeros(32, 10)
    # 50%的神經元是活躍的
    active_neurons = torch.randperm(10)[:5]
    for i in range(32):
        layer2_activation[i, active_neurons] = torch.randn(5)
    torch.save(layer2_activation, os.path.join(epoch0_dir, 'layer2_activation_batch_0.pt'))
    
    # 另一個輪次的層1激活值
    layer1_activation_epoch1 = torch.randn(32, 64, 7, 7) * 0.8  # 更低的方差
    torch.save(layer1_activation_epoch1, os.path.join(epoch1_dir, 'layer1_activation_batch_0.pt'))
    
    # 另一個輪次的層2激活值：更少的死亡神經元
    layer2_activation_epoch1 = torch.zeros(32, 10)
    # 70%的神經元是活躍的
    active_neurons_epoch1 = torch.randperm(10)[:7]
    for i in range(32):
        layer2_activation_epoch1[i, active_neurons_epoch1] = torch.randn(7)
    torch.save(layer2_activation_epoch1, os.path.join(epoch1_dir, 'layer2_activation_batch_0.pt'))
    
    # 5. 創建訓練摘要
    training_summary = {
        "total_training_time": 3600,  # 1小時
        "epochs_completed": 10,
        "best_epoch": 8,
        "best_metric": 0.89,
        "convergence_epoch": 7,
        "early_stopping": False
    }
    torch.save(training_summary, os.path.join(hooks_dir, 'training_summary.pt'))
    
    # 6. 創建輪次摘要
    epoch0_summary = {
        "epoch": 0,
        "loss": 2.0,
        "accuracy": 0.5,
        "learning_rate": 0.001,
        "time_taken": 360  # 6分鐘
    }
    torch.save(epoch0_summary, os.path.join(epoch0_dir, 'epoch_summary.pt'))
    
    epoch1_summary = {
        "epoch": 1,
        "loss": 1.5,
        "accuracy": 0.6,
        "learning_rate": 0.001,
        "time_taken": 358  # 接近6分鐘
    }
    torch.save(epoch1_summary, os.path.join(epoch1_dir, 'epoch_summary.pt'))
    
    # 清理之前的測試輸出目錄
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    
    yield experiment_dir
    
    # 測試後不清理，保留測試數據以供檢查
    # shutil.rmtree(TEST_DATA_DIR)

def test_hook_data_loader(setup_test_data):
    """
    測試HookDataLoader能否正確載入並處理hook數據
    """
    experiment_dir = setup_test_data
    loader = HookDataLoader(experiment_dir)
    
    # 測試可用輪次列表
    epochs = loader.list_available_epochs()
    assert len(epochs) == 2
    assert 0 in epochs
    assert 1 in epochs
    
    # 測試可用層列表
    layers = loader.list_available_layers(epoch=0)
    assert len(layers) == 2
    assert "layer1" in layers
    assert "layer2" in layers
    
    # 測試載入激活值
    layer1_activation = loader.load_layer_activation("layer1", epoch=0, batch=0)
    assert layer1_activation is not None
    assert layer1_activation.shape == (32, 64, 7, 7)
    
    layer2_activation = loader.load_layer_activation("layer2", epoch=0, batch=0)
    assert layer2_activation is not None
    assert layer2_activation.shape == (32, 10)
    
    # 測試載入不同輪次的激活值
    layer1_activation_epoch1 = loader.load_layer_activation("layer1", epoch=1, batch=0)
    assert layer1_activation_epoch1 is not None
    assert layer1_activation_epoch1.shape == (32, 64, 7, 7)
    
    # 測試載入訓練摘要
    training_summary = loader.load_training_summary()
    assert training_summary is not None
    assert training_summary["total_training_time"] == 3600
    assert training_summary["epochs_completed"] == 10
    
    # 測試載入輪次摘要
    epoch0_summary = loader.load_epoch_summary(0)
    assert epoch0_summary is not None
    assert epoch0_summary["epoch"] == 0
    assert epoch0_summary["loss"] == 2.0
    
    # 測試獲取實驗摘要
    experiment_summary = loader.get_experiment_summary()
    assert experiment_summary["status"] == "hooks可用"
    assert len(experiment_summary["available_epochs"]) == 2
    assert len(experiment_summary["available_layers"]) == 2
    assert experiment_summary["training_summary_available"] == True

def test_experiment_loader(setup_test_data):
    """
    測試ExperimentLoader能否正確載入實驗配置和訓練歷史
    """
    experiment_dir = setup_test_data
    loader = ExperimentLoader(experiment_dir)
    
    # 測試載入配置
    config = loader.load_config()
    assert config is not None
    assert config["experiment_name"] == "mock_test"
    assert config["model"]["type"] == "test_model"
    
    # 測試載入模型結構
    model_structure = loader.load_model_structure()
    assert model_structure is not None
    assert model_structure["model_name"] == "TestModel"
    assert model_structure["total_parameters"] == 10000
    
    # 測試載入訓練歷史
    training_history = loader.load_training_history()
    assert training_history is not None
    assert len(training_history["loss"]) == 10
    assert training_history["accuracy"][-1] == 0.9

def test_layer_activity_analysis(setup_test_data):
    """
    測試層級活動分析功能
    """
    experiment_dir = setup_test_data
    loader = HookDataLoader(experiment_dir)
    
    # 載入層激活值
    layer1_activation = loader.load_layer_activation("layer1", epoch=0, batch=0)
    layer2_activation = loader.load_layer_activation("layer2", epoch=0, batch=0)
    
    # 分析層1的激活值統計
    layer1_stats = calculate_activation_statistics(layer1_activation)
    assert "mean" in layer1_stats
    assert "std" in layer1_stats
    assert "min" in layer1_stats
    assert "max" in layer1_stats
    assert "sparsity" in layer1_stats
    
    # 驗證層1的激活值分布（應該接近正態分布）
    assert abs(layer1_stats["mean"]) < 0.1  # 均值應接近0
    assert abs(layer1_stats["std"] - 1.0) < 0.1  # 標準差應接近1
    
    # 分析層2的激活值稀疏度
    layer2_sparsity = calculate_activation_sparsity(layer2_activation)
    # 我們創建的數據中，有50%的神經元是死亡的
    assert layer2_sparsity > 0.45  # 應該接近0.5
    
    # 檢測層2的死亡神經元
    dead_info = detect_dead_neurons(layer2_activation)
    assert dead_info["dead_count"] == 5  # 我們設置了5個死亡神經元
    assert abs(dead_info["dead_ratio"] - 0.5) < 0.01  # 死亡比例應接近0.5

def test_report_generator(setup_test_data):
    """
    測試報告生成器功能
    """
    experiment_dir = setup_test_data
    hook_loader = HookDataLoader(experiment_dir)
    exp_loader = ExperimentLoader(experiment_dir)
    
    # 載入必要的數據
    model_structure = exp_loader.load_model_structure()
    training_history = exp_loader.load_training_history()
    
    # 創建一些分析圖表
    # 損失曲線圖
    plt.figure(figsize=(10, 6))
    plt.plot(training_history["loss"], label="Train Loss")
    plt.plot(training_history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    loss_plot_path = os.path.join(TEST_OUTPUT_DIR, "loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()
    
    # 準確率曲線圖
    plt.figure(figsize=(10, 6))
    plt.plot(training_history["accuracy"], label="Train Accuracy")
    plt.plot(training_history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    acc_plot_path = os.path.join(TEST_OUTPUT_DIR, "accuracy_curve.png")
    plt.savefig(acc_plot_path)
    plt.close()
    
    # 創建層激活值分布圖
    layer1_activation = hook_loader.load_layer_activation("layer1", epoch=0, batch=0)
    plt.figure(figsize=(8, 5))
    plt.hist(layer1_activation.flatten().numpy(), bins=50, alpha=0.7)
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.title("Layer 1 Activation Distribution")
    layer1_plot_path = os.path.join(TEST_OUTPUT_DIR, "layer1_distribution.png")
    plt.savefig(layer1_plot_path)
    plt.close()
    
    # 準備報告數據
    # 將分析結果封裝成 MockAnalysisResults 類以適配 ReportGenerator
    class MockAnalysisResults:
        def __init__(self, data):
            self.data = data
            
        def to_dict(self):
            return {
                'model_structure': self.data['model_structure'],
                'training_dynamics': {
                    'convergence_analysis': {
                        'loss': {
                            'convergence_epoch': 7,
                            'convergence_rate': 0.1,
                            'final_value': 0.48
                        },
                        'accuracy': {
                            'convergence_epoch': 6,
                            'convergence_rate': 0.05,
                            'final_value': 0.9
                        }
                    },
                    'stability_analysis': {
                        'loss': {
                            'variance': 0.25,
                            'max_fluctuation': 0.2
                        },
                        'accuracy': {
                            'variance': 0.01,
                            'max_fluctuation': 0.05
                        }
                    }
                },
                'intermediate_data': {
                    'layer_results': {
                        'layer1': {
                            'epochs_analyzed': [0, 1],
                            'statistics': {
                                0: self.data['layer_activations']['layer1']['statistics'],
                                1: {'mean': 0.1, 'std': 0.8, 'min': -2.1, 'max': 2.3, 'median': 0.05}
                            },
                            'sparsity': {
                                0: {'l1_sparsity': 0.2, 'zero_ratio': 0.1},
                                1: {'l1_sparsity': 0.15, 'zero_ratio': 0.08}
                            }
                        },
                        'layer2': {
                            'epochs_analyzed': [0],
                            'statistics': {
                                0: self.data['layer_activations']['layer2']['statistics']
                            },
                            'sparsity': {
                                0: {'l1_sparsity': 0.5, 'zero_ratio': 0.5}
                            }
                        }
                    }
                }
            }
    
    analysis_results = MockAnalysisResults({
        "model_structure": model_structure,
        "training_history": training_history,
        "layer_activations": {
            "layer1": {
                "statistics": calculate_activation_statistics(layer1_activation),
                "distribution_plot": layer1_plot_path
            },
            "layer2": {
                "statistics": calculate_activation_statistics(
                    hook_loader.load_layer_activation("layer2", epoch=0, batch=0)
                ),
                "dead_neurons": detect_dead_neurons(
                    hook_loader.load_layer_activation("layer2", epoch=0, batch=0)
                )
            }
        },
        "plots": [
            {
                "title": "訓練和驗證損失",
                "path": loss_plot_path,
                "caption": "訓練過程中的損失函數變化趨勢"
            },
            {
                "title": "訓練和驗證準確率",
                "path": acc_plot_path,
                "caption": "訓練過程中的準確率變化趨勢"
            }
        ]
    })
    
    # 創建報告生成器
    report_generator = ReportGenerator(
        experiment_name="mock_test",
        output_dir=TEST_OUTPUT_DIR
    )
    
    # 生成Markdown報告
    markdown_path = report_generator.generate(
        analysis_results,
        format='markdown'
    )
    
    # 檢查報告文件是否存在
    assert os.path.exists(markdown_path)
    
    # 簡單檢查報告內容
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
        assert "mock_test" in content
        assert "模型結構分析" in content
        assert "訓練動態分析" in content
        assert "中間層數據分析" in content
    
    # 生成HTML報告
    html_path = report_generator.generate(
        analysis_results,
        format='html'
    )
    
    # 檢查HTML報告文件是否存在
    assert os.path.exists(html_path)
    
    print(f"\n測試報告已生成在: {TEST_OUTPUT_DIR}")

def test_end_to_end_workflow(setup_test_data):
    """
    測試完整的端到端分析流程
    """
    experiment_dir = setup_test_data
    
    # 1. 載入實驗數據
    hook_loader = HookDataLoader(experiment_dir)
    exp_loader = ExperimentLoader(experiment_dir)
    
    config = exp_loader.load_config()
    model_structure = exp_loader.load_model_structure()
    training_history = exp_loader.load_training_history()
    
    # 確認數據載入正確
    assert config is not None
    assert model_structure is not None
    assert training_history is not None
    
    # 2. 分析層激活值
    layer_activations = {}
    
    # 獲取所有可用的輪次和層
    epochs = hook_loader.list_available_epochs()
    first_epoch = epochs[0]
    layers = hook_loader.list_available_layers(first_epoch)
    
    for layer_name in layers:
        layer_activation = hook_loader.load_layer_activation(layer_name, first_epoch, batch=0)
        if layer_activation is not None:
            # 計算層統計信息
            stats = calculate_activation_statistics(layer_activation)
            
            # 檢測死亡神經元
            dead_info = detect_dead_neurons(layer_activation)
            
            # 創建激活值分布圖
            plt.figure(figsize=(8, 5))
            plt.hist(layer_activation.flatten().numpy(), bins=50, alpha=0.7)
            plt.xlabel("Activation Value")
            plt.ylabel("Frequency")
            plt.title(f"Layer {layer_name} Activation Distribution")
            plot_path = os.path.join(TEST_OUTPUT_DIR, f"{layer_name}_distribution.png")
            plt.savefig(plot_path)
            plt.close()
            
            # 保存分析結果
            layer_activations[layer_name] = {
                "statistics": stats,
                "dead_neurons": dead_info,
                "distribution_plot": plot_path
            }
    
    # 3. 創建訓練曲線圖表
    # 損失曲線
    plt.figure(figsize=(10, 6))
    plt.plot(training_history["loss"], label="Train Loss")
    plt.plot(training_history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    loss_plot_path = os.path.join(TEST_OUTPUT_DIR, "e2e_loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()
    
    # 準確率曲線
    plt.figure(figsize=(10, 6))
    plt.plot(training_history["accuracy"], label="Train Accuracy")
    plt.plot(training_history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    acc_plot_path = os.path.join(TEST_OUTPUT_DIR, "e2e_accuracy_curve.png")
    plt.savefig(acc_plot_path)
    plt.close()
    
    # 4. 整合分析結果
    # 將分析結果封裝成 MockAnalysisResults 類以適配 ReportGenerator
    class MockAnalysisResults:
        def __init__(self, data):
            self.data = data
            
        def to_dict(self):
            return {
                'model_structure': self.data['model_structure'],
                'training_dynamics': {
                    'convergence_analysis': {
                        'loss': {
                            'convergence_epoch': 7,
                            'convergence_rate': 0.1,
                            'final_value': 0.48
                        },
                        'accuracy': {
                            'convergence_epoch': 6,
                            'convergence_rate': 0.05,
                            'final_value': 0.9
                        }
                    },
                    'stability_analysis': {
                        'loss': {
                            'variance': 0.25,
                            'max_fluctuation': 0.2
                        },
                        'accuracy': {
                            'variance': 0.01,
                            'max_fluctuation': 0.05
                        }
                    }
                },
                'intermediate_data': {
                    'layer_results': {
                        'layer1': {
                            'epochs_analyzed': [0, 1],
                            'statistics': {
                                0: self.data['layer_activations']['layer1']['statistics'],
                                1: {'mean': 0.1, 'std': 0.8, 'min': -2.1, 'max': 2.3, 'median': 0.05}
                            }
                        },
                        'layer2': {
                            'epochs_analyzed': [0],
                            'statistics': {
                                0: self.data['layer_activations']['layer2']['statistics']
                            }
                        }
                    }
                }
            }
    
    analysis_results = MockAnalysisResults({
        "model_structure": model_structure,
        "training_history": training_history,
        "layer_activations": layer_activations,
        "plots": [
            {
                "title": "訓練和驗證損失",
                "path": loss_plot_path,
                "caption": "訓練過程中的損失函數變化趨勢"
            },
            {
                "title": "訓練和驗證準確率",
                "path": acc_plot_path,
                "caption": "訓練過程中的準確率變化趨勢"
            }
        ]
    })
    
    # 5. 生成分析報告
    report_output_dir = os.path.join(TEST_OUTPUT_DIR, "end_to_end_report")
    os.makedirs(report_output_dir, exist_ok=True)
    
    report_generator = ReportGenerator(
        experiment_name=config["experiment_name"],
        output_dir=report_output_dir
    )
    
    # 生成HTML報告
    html_path = report_generator.generate(
        analysis_results,
        format='html'
    )
    
    # 檢查報告是否成功生成
    assert os.path.exists(html_path)
    
    print(f"\n端到端測試報告已生成在: {report_output_dir}")
    
    # 驗證流程完整性
    assert layer_activations["layer1"]["statistics"]["mean"] is not None
    assert "dead_ratio" in layer_activations["layer2"]["dead_neurons"]

if __name__ == "__main__":
    # 直接運行此文件時，執行完整的端到端測試
    test_data_dir = setup_test_data()
    test_hook_data_loader(test_data_dir)
    test_experiment_loader(test_data_dir)
    test_layer_activity_analysis(test_data_dir)
    test_report_generator(test_data_dir)
    test_end_to_end_workflow(test_data_dir)
    print("所有測試完成！") 