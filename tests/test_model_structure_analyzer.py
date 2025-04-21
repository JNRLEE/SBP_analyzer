"""
測試模型結構分析器模組。

此模組包含用於測試ModelStructureAnalyzer類的單元測試。
"""

import os
import json
import pytest
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock

from analyzer.model_structure_analyzer import ModelStructureAnalyzer

# 建立測試資料目錄和檔案的夾具
@pytest.fixture
def setup_test_dir():
    """建立臨時測試目錄並創建必要的測試檔案"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # 創建analysis子目錄
        analysis_dir = os.path.join(temp_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 創建模型結構檔案
        model_structure = {
            "model_name": "TestCNN",
            "total_parameters": 1000000,
            "layer_info": [
                {
                    "name": "conv1",
                    "type": "Conv2d",
                    "parameters": 100000,
                    "trainable": True,
                    "shape": [64, 3, 3, 3]
                },
                {
                    "name": "norm1",
                    "type": "BatchNorm2d",
                    "parameters": 128,
                    "trainable": True,
                    "shape": [64]
                },
                {
                    "name": "conv2",
                    "type": "Conv2d",
                    "parameters": 200000,
                    "trainable": True,
                    "shape": [128, 64, 3, 3]
                },
                {
                    "name": "norm2",
                    "type": "BatchNorm2d",
                    "parameters": 256,
                    "trainable": True,
                    "shape": [128]
                },
                {
                    "name": "fc1",
                    "type": "Linear",
                    "parameters": 300000,
                    "trainable": True,
                    "shape": [512, 2048]
                },
                {
                    "name": "fc2",
                    "type": "Linear",
                    "parameters": 399616,
                    "trainable": True,
                    "shape": [1000, 512]
                }
            ]
        }
        
        with open(os.path.join(temp_dir, 'model_structure.json'), 'w') as f:
            json.dump(model_structure, f, indent=2)
        
        yield temp_dir

# 測試基本功能
def test_basic_functionality(setup_test_dir):
    """測試ModelStructureAnalyzer的基本功能"""
    analyzer = ModelStructureAnalyzer(setup_test_dir)
    results = analyzer.analyze(save_results=True)
    
    # 驗證基本屬性
    assert results['model_name'] == "TestCNN"
    assert results['total_params'] == 1000000
    assert results['layer_count'] == 6
    assert 'Conv2d' in results['layer_types']
    assert 'Linear' in results['layer_types']
    assert 'BatchNorm2d' in results['layer_types']
    
    # 驗證參數統計
    assert results['trainable_params'] == 1000000
    assert results['non_trainable_params'] == 0

# 測試稀疏度估計功能
def test_sparsity_estimation(setup_test_dir):
    """測試稀疏度估計功能"""
    analyzer = ModelStructureAnalyzer(setup_test_dir)
    analyzer.load_model_structure()
    layer_info = analyzer.model_structure['layer_info']
    
    sparsity = analyzer._estimate_sparsity(layer_info)
    
    # 稀疏度應該在0-1之間
    assert 0 <= sparsity <= 1
    
    # 驗證不同層類型的稀疏度估計是否合理
    mock_layers = [
        {"type": "Conv2d", "parameters": 1000},
        {"type": "BatchNorm2d", "parameters": 1000},
        {"type": "Linear", "parameters": 1000}
    ]
    
    sparsity = analyzer._estimate_sparsity(mock_layers)
    assert sparsity == (1000 * 0.2 + 1000 * 0.05 + 1000 * 0.3) / 3000

# 測試計算複雜度功能
def test_flops_calculation(setup_test_dir):
    """測試FLOPs計算功能"""
    analyzer = ModelStructureAnalyzer(setup_test_dir)
    analyzer.load_model_structure()
    
    flops_data = analyzer.calculate_flops()
    
    assert 'total_flops' in flops_data
    assert 'flops_by_layer' in flops_data
    assert flops_data['total_flops'] > 0
    
    # 檢查每一層是否都有對應的FLOPs值
    layer_names = [layer['name'] for layer in analyzer.model_structure['layer_info']]
    for name in layer_names:
        assert name in flops_data['flops_by_layer']

# 測試記憶體佔用估計功能
def test_memory_footprint(setup_test_dir):
    """測試記憶體佔用估計功能"""
    analyzer = ModelStructureAnalyzer(setup_test_dir)
    analyzer.load_model_structure()
    
    memory_data = analyzer.calculate_memory_footprint()
    
    assert 'total_memory' in memory_data
    assert 'parameters_memory' in memory_data
    assert 'activations_memory' in memory_data
    assert 'gradients_memory' in memory_data
    
    # 參數記憶體應該等於參數數量*4（假設float32）
    assert memory_data['parameters_memory'] == 1000000 * 4
    
    # 總記憶體應該大於參數記憶體
    assert memory_data['total_memory'] > memory_data['parameters_memory']

# 測試感受野計算功能
def test_receptive_field_calculation(setup_test_dir):
    """測試理論感受野計算功能"""
    analyzer = ModelStructureAnalyzer(setup_test_dir)
    analyzer.load_model_structure()
    
    rf_data = analyzer.calculate_receptive_field()
    
    assert 'receptive_fields' in rf_data
    assert 'sorted_by_size' in rf_data
    assert 'max_receptive_field' in rf_data
    assert rf_data['max_receptive_field'] >= 1
    
    # 確認每一層都有感受野信息
    layer_names = [layer['name'] for layer in analyzer.model_structure['layer_info']]
    for name in layer_names:
        assert name in rf_data['receptive_fields']
        assert 'receptive_field_size' in rf_data['receptive_fields'][name]

# 測試視覺化功能
@patch('matplotlib.pyplot.savefig')  # Mock掉savefig以避免實際生成圖像
def test_visualization(mock_savefig, setup_test_dir):
    """測試模型結構視覺化功能"""
    analyzer = ModelStructureAnalyzer(setup_test_dir)
    analyzer.analyze()
    
    # 測試帶有輸出路徑的情況
    output_path = os.path.join(setup_test_dir, 'model_structure.png')
    fig = analyzer.visualize_model_structure(output_path)
    
    # 確認返回了圖形對象
    assert fig is not None
    
    # 確認嘗試保存圖像
    mock_savefig.assert_called()

# 測試分析結果保存和載入
def test_save_and_load_results(setup_test_dir):
    """測試分析結果的保存和載入功能"""
    analyzer = ModelStructureAnalyzer(setup_test_dir)
    results = analyzer.analyze(save_results=True)
    
    # 檢查結果文件是否存在
    result_path = os.path.join(setup_test_dir, 'analysis', 'modelstructureanalyzer_results.json')
    assert os.path.exists(result_path)
    
    # 創建新的分析器並載入結果
    new_analyzer = ModelStructureAnalyzer(setup_test_dir)
    loaded_results = new_analyzer.load_results(result_path)
    
    # 確認載入的結果與原始結果相同
    assert loaded_results['model_name'] == results['model_name']
    assert loaded_results['total_params'] == results['total_params']
    assert loaded_results['layer_count'] == results['layer_count'] 