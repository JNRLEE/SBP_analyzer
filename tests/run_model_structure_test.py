"""
直接運行模型結構分析器測試的腳本，繞過 pytest 框架
"""

import os
import sys
import json
import tempfile
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端

# 將父目錄添加到路徑中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analyzer.model_structure_analyzer import ModelStructureAnalyzer
from visualization.model_structure_plots import ModelStructurePlotter

def setup_test_data():
    """設置測試數據。"""
    # 創建臨時目錄作為實驗目錄
    temp_dir = tempfile.mkdtemp()
    
    # 創建模型結構文件
    model_structure = {
        "model_name": "ResNet18Model",
        "layers": [
            {
                "id": "input_layer",
                "name": "input_layer",
                "type": "input",
                "parameters": 0,
                "trainable": True,
                "input_shape": [224, 224, 3],
                "output_shape": [224, 224, 3],
                "inputs": []
            },
            {
                "id": "conv1",
                "name": "conv1",
                "type": "Conv2D",
                "parameters": 9408,
                "trainable": True,
                "input_shape": [224, 224, 3],
                "output_shape": [112, 112, 64],
                "kernel_size": [7, 7],
                "inputs": ["input_layer"],
                "weight_stats": {
                    "mean": 0.0,
                    "std": 0.02,
                    "min": -0.06,
                    "max": 0.06,
                    "sparsity": 0.05,
                    "histogram": [10, 20, 30, 40, 30, 20, 10]
                }
            },
            {
                "id": "pool1",
                "name": "pool1",
                "type": "MaxPool2D",
                "parameters": 0,
                "trainable": True,
                "input_shape": [112, 112, 64],
                "output_shape": [56, 56, 64],
                "inputs": ["conv1"]
            }
        ]
    }
    
    # 寫入模型結構文件
    with open(os.path.join(temp_dir, 'model_structure.json'), 'w') as f:
        json.dump(model_structure, f)
    
    return temp_dir

def test_model_structure_analyzer():
    """測試模型結構分析器的基本功能。"""
    test_dir = setup_test_data()
    
    try:
        # 初始化分析器
        analyzer = ModelStructureAnalyzer(test_dir)
        print("初始化分析器成功")
        
        # 載入模型結構
        structure = analyzer.load_model_structure()
        print(f"載入模型結構成功: {structure['model_name']}")
        
        # 進行分析
        results = analyzer.analyze(save_results=False)
        print(f"分析完成，模型類型: {results['architecture_type']}")
        print(f"總參數數量: {results['total_params']}")
        
        # 獲取層級結構
        hierarchy = analyzer.get_layer_hierarchy()
        print(f"獲取層級結構成功，共 {len(hierarchy['layers'])} 層")
        
        # 初始化繪圖工具
        plotter = ModelStructurePlotter()
        print("初始化繪圖工具成功")
        
        # 繪製模型結構圖
        fig = plotter.plot_model_structure(hierarchy)
        print("繪製模型結構圖成功")
        
        # 繪製參數分布圖
        param_distribution = analyzer.get_parameter_distribution()
        fig = plotter.plot_parameter_distribution(param_distribution)
        print("繪製參數分布圖成功")
        
        print("所有測試通過!")
        return True
    
    except Exception as e:
        print(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理臨時目錄
        import shutil
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_model_structure_analyzer() 