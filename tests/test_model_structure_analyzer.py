"""
測試模型結構分析模組的單元測試。

此模組測試analyzer/model_structure_analyzer.py中的功能。
"""

import os
import json
import tempfile
import shutil
import pytest
import numpy as np
from collections import defaultdict

from analyzer.model_structure_analyzer import ModelStructureAnalyzer

class TestModelStructureAnalyzer:
    """測試ModelStructureAnalyzer類的各個方法"""
    
    @pytest.fixture
    def temp_experiment_dir(self):
        """創建一個臨時實驗目錄用於測試"""
        temp_dir = tempfile.mkdtemp()
        
        # 創建分析結果目錄
        os.makedirs(os.path.join(temp_dir, "analysis"), exist_ok=True)
        
        # 創建模型結構JSON文件
        model_structure = {
            "model_name": "TestModel",
            "layers": [
                {
                    "id": "input",
                    "name": "input_layer",
                    "type": "input",
                    "parameters": 0,
                    "input_size": [32, 224, 224],
                    "output_size": [32, 224, 224],
                    "trainable": True
                },
                {
                    "id": "conv1",
                    "name": "conv1",
                    "type": "convolution",
                    "parameters": 1728,  # 3x3 kernel, 3 in channels, 64 out channels
                    "input_size": [32, 224, 224],
                    "output_size": [64, 112, 112],
                    "kernel_size": [3, 3],
                    "trainable": True,
                    "inputs": ["input"]
                },
                {
                    "id": "bn1",
                    "name": "batch_norm1",
                    "type": "batch_norm",
                    "parameters": 128,  # 2 params per channel, 64 channels
                    "input_size": [64, 112, 112],
                    "output_size": [64, 112, 112],
                    "trainable": True,
                    "inputs": ["conv1"]
                },
                {
                    "id": "relu1",
                    "name": "relu1",
                    "type": "activation",
                    "parameters": 0,
                    "input_size": [64, 112, 112],
                    "output_size": [64, 112, 112],
                    "trainable": False,
                    "inputs": ["bn1"]
                },
                {
                    "id": "conv2",
                    "name": "conv2",
                    "type": "convolution",
                    "parameters": 36864,  # 3x3 kernel, 64 in channels, 64 out channels
                    "input_size": [64, 112, 112],
                    "output_size": [64, 112, 112],
                    "kernel_size": [3, 3],
                    "trainable": True,
                    "inputs": ["relu1"]
                },
                {
                    "id": "fc1",
                    "name": "fc1",
                    "type": "linear",
                    "parameters": 8192000,  # 64*112*112 in, 100 out
                    "input_size": [64*112*112],
                    "output_size": [100],
                    "trainable": True,
                    "inputs": ["conv2"]
                },
                {
                    "id": "output",
                    "name": "output",
                    "type": "linear",
                    "parameters": 1000,  # 100 in, 10 out
                    "input_size": [100],
                    "output_size": [10],
                    "trainable": True,
                    "inputs": ["fc1"]
                }
            ]
        }
        
        with open(os.path.join(temp_dir, "model_structure.json"), "w") as f:
            json.dump(model_structure, f)
        
        yield temp_dir
        
        # 清理臨時目錄
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, temp_experiment_dir):
        """測試分析器初始化"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        assert analyzer.experiment_dir == temp_experiment_dir
        assert analyzer.model_structure == {}
    
    def test_load_model_structure(self, temp_experiment_dir):
        """測試載入模型結構文件"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        structure = analyzer.load_model_structure()
        
        assert structure["model_name"] == "TestModel"
        assert len(structure["layers"]) == 7
        assert structure["layers"][0]["id"] == "input"
    
    def test_analyze_basic_info(self, temp_experiment_dir):
        """測試模型基本信息分析"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        analyzer.load_model_structure()
        analyzer._analyze_basic_info()
        
        # 檢查層數量
        assert analyzer.results["layer_count"] == 7
        
        # 檢查參數統計
        assert analyzer.results["total_params"] == 8231720
        assert analyzer.results["trainable_params"] == 8231720
        assert analyzer.results["non_trainable_params"] == 0
        
        # 檢查層類型分佈
        layer_types = analyzer.results["layer_types"]
        assert "convolution" in layer_types
        assert layer_types["convolution"] == 2
        assert "linear" in layer_types
        assert layer_types["linear"] == 2
    
    def test_identify_architecture_type(self, temp_experiment_dir):
        """測試架構類型識別"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        analyzer.load_model_structure()
        analyzer._identify_architecture_type()
        
        assert analyzer.results["architecture_type"] == "cnn"
    
    def test_analyze_layer_hierarchy(self, temp_experiment_dir):
        """測試層級結構分析"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        analyzer.load_model_structure()
        analyzer._analyze_layer_hierarchy()
        
        # 檢查模型深度
        assert analyzer.results["model_depth"] == 6  # 從input到output的路徑長度
    
    def test_analyze_parameter_distribution(self, temp_experiment_dir):
        """測試參數分佈分析"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        analyzer.load_model_structure()
        analyzer._analyze_parameter_distribution()
        
        param_dist = analyzer.results["parameter_distribution"]
        
        # 檢查按類型的參數分佈
        assert "by_type" in param_dist
        by_type = param_dist["by_type"]
        assert "convolution" in by_type
        assert "linear" in by_type
        
        # 檢查按類型的百分比
        assert "by_type_pct" in param_dist
        by_type_pct = param_dist["by_type_pct"]
        assert by_type_pct["linear"] > by_type_pct["convolution"]  # 全連接層應該佔大部分參數
        
        # 檢查Gini係數
        assert "concentration" in param_dist
        assert 0 <= param_dist["concentration"]["gini_coefficient"] <= 1
        
        # 檢查百分位數
        assert "percentiles" in param_dist
        percentiles = param_dist["percentiles"]
        assert percentiles["p10"] <= percentiles["p25"] <= percentiles["p50"]
        assert percentiles["p50"] <= percentiles["p75"] <= percentiles["p90"]
    
    def test_analyze_weight_distributions(self, temp_experiment_dir):
        """測試權重分佈分析"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        analyzer.load_model_structure()
        analyzer._analyze_weight_distributions()
        
        weight_dists = analyzer.results["weight_distributions"]
        
        # 檢查是否分析了每一層
        assert len(weight_dists) > 0
        
        # 檢查卷積層的分佈
        assert "conv1" in weight_dists
        conv_dist = weight_dists["conv1"]
        assert "distribution_type" in conv_dist
        assert "estimated_mean" in conv_dist
        assert "estimated_std" in conv_dist
        
        # 檢查全連接層的分佈
        assert "fc1" in weight_dists
        fc_dist = weight_dists["fc1"]
        assert fc_dist["distribution_type"] in ["xavier", "unknown"]
    
    def test_analyze_layer_complexity(self, temp_experiment_dir):
        """測試層級複雜度分析"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        analyzer.load_model_structure()
        analyzer._analyze_layer_complexity()
        
        complexities = analyzer.results["layer_complexity"]
        
        # 檢查是否分析了每一層
        assert len(complexities) > 0
        
        # 檢查卷積層的複雜度
        assert "conv1" in complexities
        conv_complexity = complexities["conv1"]
        assert "parameter_complexity" in conv_complexity
        assert "computational_complexity" in conv_complexity
        assert "io_complexity" in conv_complexity
        
        # 檢查模型整體複雜度
        assert "model_complexity" in analyzer.results
        model_complexity = analyzer.results["model_complexity"]
        assert "total_flops" in model_complexity
        assert "total_memory" in model_complexity
        assert "flops_per_param" in model_complexity
    
    def test_identify_largest_layers(self, temp_experiment_dir):
        """測試識別最大層"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        analyzer.load_model_structure()
        analyzer._identify_largest_layers(top_n=3)
        
        largest_layers = analyzer.results["largest_layers"]
        
        # 檢查是否找到了前3大的層
        assert len(largest_layers) == 3
        
        # 檢查層是否按參數量降序排序
        assert largest_layers[0]["parameters"] >= largest_layers[1]["parameters"]
        assert largest_layers[1]["parameters"] >= largest_layers[2]["parameters"]
        
        # 檢查最大層是否是fc1（最多參數）
        assert largest_layers[0]["layer_id"] == "fc1"
    
    def test_analyze_sequential_paths(self, temp_experiment_dir):
        """測試序列路徑分析"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        analyzer.load_model_structure()
        analyzer._analyze_sequential_paths()
        
        paths = analyzer.results["sequential_paths"]
        
        # 檢查是否找到了路徑
        assert len(paths) > 0
        
        # 檢查最長路徑是否包含所有層
        longest_path = paths[0]
        assert len(longest_path) == 7  # 應該包含所有7個層
        
        # 檢查路徑開始和結束
        assert longest_path[0]["id"] == "input"
        assert longest_path[-1]["id"] == "output"
    
    def test_calculate_parameter_efficiency(self, temp_experiment_dir):
        """測試參數效率計算"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        analyzer.load_model_structure()
        analyzer._analyze_layer_complexity()  # 先分析複雜度
        analyzer._calculate_parameter_efficiency()
        
        efficiency = analyzer.results["parameter_efficiency"]
        
        # 檢查效率指標
        assert "flops_per_param" in efficiency
        assert "params_utilization" in efficiency
        assert "relative_efficiency" in efficiency
        
        # 檢查值的範圍
        assert efficiency["flops_per_param"] >= 0
        assert 0 <= efficiency["params_utilization"] <= 1
    
    def test_analyze_layer_connectivity(self, temp_experiment_dir):
        """測試層間連接性分析"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        analyzer.load_model_structure()
        analyzer._analyze_layer_connectivity()
        
        connectivity = analyzer.results["layer_connectivity"]
        
        # 檢查連接度統計
        assert "avg_indegree" in connectivity
        assert "avg_outdegree" in connectivity
        assert "max_indegree" in connectivity
        assert "max_outdegree" in connectivity
        assert "bottleneck_layers" in connectivity
        assert "hub_layers" in connectivity
        
        # 檢查平均進出度
        assert connectivity["avg_indegree"] > 0
        assert connectivity["avg_outdegree"] > 0
    
    def test_analyze_computational_complexity(self, temp_experiment_dir):
        """測試計算複雜度分析"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        analyzer.load_model_structure()
        analyzer._analyze_layer_complexity()  # 先分析層級複雜度
        analyzer._analyze_computational_complexity()
        
        if "model_complexity" in analyzer.results:
            complexity = analyzer.results["model_complexity"]
            
            # 檢查是否包含GFLOPs
            if "gflops" in complexity:
                assert complexity["gflops"] >= 0
                
                # 檢查硬件估計
                assert "hardware_estimates" in complexity
                hw_est = complexity["hardware_estimates"]
                assert "cpu_inference_ms" in hw_est
                assert "gpu_inference_ms" in hw_est
                
                # 檢查複雜度分類
                assert "complexity_class" in complexity
                assert complexity["complexity_class"] in [
                    "very light", "light", "moderate", "heavy", "very heavy", "unknown"
                ]
    
    def test_full_analyze(self, temp_experiment_dir):
        """測試完整的分析過程"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        results = analyzer.analyze(save_results=True)
        
        # 檢查分析結果是否已保存
        analysis_path = os.path.join(temp_experiment_dir, "analysis", "ModelStructureAnalyzer_results.json")
        assert os.path.exists(analysis_path)
        
        # 檢查分析結果是否包含所有預期的鍵
        assert "model_name" in results
        assert "layer_count" in results
        assert "total_params" in results
        assert "architecture_type" in results
        assert "parameter_distribution" in results
    
    def test_get_model_summary(self, temp_experiment_dir):
        """測試獲取模型摘要"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        summary = analyzer.get_model_summary()
        
        # 檢查摘要內容
        assert "model_name" in summary
        assert "architecture_type" in summary
        assert "total_params" in summary
        assert "trainable_params" in summary
        assert "non_trainable_params" in summary
        assert "layer_count" in summary
        assert "model_depth" in summary
        assert "largest_layers" in summary
        assert len(summary["largest_layers"]) <= 3  # 只應包含前3個最大層
    
    def test_get_layer_hierarchy(self, temp_experiment_dir):
        """測試獲取層級結構"""
        analyzer = ModelStructureAnalyzer(temp_experiment_dir)
        hierarchy = analyzer.get_layer_hierarchy()
        
        # 檢查層級結構
        assert "nodes" in hierarchy
        assert "roots" in hierarchy
        
        # 檢查節點數和根節點
        assert len(hierarchy["nodes"]) == 7  # 總共7個層
        assert len(hierarchy["roots"]) == 1  # 應該只有一個根節點 (input)
        assert "input" in hierarchy["roots"]
        
        # 檢查子節點關係
        nodes = hierarchy["nodes"]
        assert "children" in nodes["input"]
        assert "conv1" in nodes["input"]["children"] 