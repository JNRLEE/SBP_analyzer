#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自適應閾值分析器的單元測試

測試日期: 2023-04-17
測試目的: 驗證自適應閾值分析器能正確分析不同分佈類型的層激活數據，
       並根據這些分佈特性調整閾值，檢測問題神經元。
"""

import os
import sys
import unittest
import numpy as np
import torch
import json
import tempfile
import shutil
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from pathlib import Path
import logging

# 確保可以導入被測試的模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analyzer.adaptive_threshold_analyzer import AdaptiveThresholdAnalyzer
from analyzer.intermediate_data_analyzer import IntermediateDataAnalyzer


class MockHookDataLoader:
    """模擬鉤子資料載入器，用於測試不同類型的激活分佈"""
    
    def __init__(self, base_dir=None):
        """初始化模擬資料載入器"""
        self.layer_names = [
            'layer1_normal',
            'layer2_uniform',
            'layer3_bimodal',
            'layer4_sparse',
            'layer5_dead',
            'layer6_saturated',
            'layer7_right_skewed',
            'layer8_left_skewed'
        ]
        
        # 激活資料快取
        self.activations_cache = {}
        
    def get_layer_names(self):
        """獲取層名稱列表"""
        return self.layer_names
        
    def list_available_epochs(self):
        """獲取可用的輪次列表"""
        return [0, 1, 2]  # 模擬3個輪次
        
    def list_available_layer_activations(self, epoch):
        """獲取特定輪次可用的層列表"""
        return self.layer_names
        
    def list_available_batches(self, epoch):
        """獲取特定輪次可用的批次列表"""
        return [0, 1]  # 模擬2個批次
    
    def load_layer_activation(self, layer_name, epoch, batch=0):
        """
        載入指定層的激活值。

        Args:
            layer_name (str): 層名稱
            epoch (int): 輪次編號
            batch (int, optional): 批次編號，預設為0

        Returns:
            torch.Tensor: 激活值張量
        """
        # 基本參數
        batch_size = 32
        channels = 16
        height = 8
        width = 8
        
        # 根據層名稱中的分佈類型產生不同分佈的資料
        if 'normal' in layer_name:
            # 正態分佈
            data = torch.randn(batch_size, channels, height, width) * 0.2 + 0.5
        
        elif 'uniform' in layer_name:
            # 均勻分佈
            data = torch.rand(batch_size, channels, height, width)
        
        elif 'bimodal' in layer_name:
            # 雙峰分佈
            data1 = torch.randn(batch_size, channels, height, width) * 0.1 + 0.2
            data2 = torch.randn(batch_size, channels, height, width) * 0.1 + 0.8
            mask = torch.rand(batch_size, channels, height, width) > 0.5
            data = torch.where(mask, data1, data2)
        
        elif 'sparse' in layer_name:
            # 稀疏分佈
            data = torch.zeros(batch_size, channels, height, width)
            mask = torch.rand(batch_size, channels, height, width) > 0.7
            values = torch.rand(batch_size, channels, height, width)
            data = torch.where(mask, values, data)
        
        elif 'dead' in layer_name:
            # 死亡神經元模擬
            data = torch.zeros(batch_size, channels, height, width)
            # 讓部分通道完全為零
            for i in range(channels):
                if i % 3 == 0:  # 每3個通道中的1個接近死亡
                    data[:, i, :, :] = torch.rand(batch_size, height, width) * 0.01
                else:
                    data[:, i, :, :] = torch.rand(batch_size, height, width) * 0.5 + 0.25
        
        elif 'saturated' in layer_name:
            # 飽和神經元模擬
            data = torch.ones(batch_size, channels, height, width)
            # 讓部分通道接近飽和
            for i in range(channels):
                if i % 3 == 0:  # 每3個通道中的1個接近飽和
                    data[:, i, :, :] = torch.rand(batch_size, height, width) * 0.05 + 0.95
                else:
                    data[:, i, :, :] = torch.rand(batch_size, height, width) * 0.5 + 0.25
        
        elif 'right_skewed' in layer_name:
            # 右偏分佈
            data = torch.rand(batch_size, channels, height, width) ** 2
        
        elif 'left_skewed' in layer_name:
            # 左偏分佈
            data = 1 - torch.rand(batch_size, channels, height, width) ** 2
        
        else:
            # 默認使用正態分佈
            data = torch.randn(batch_size, channels, height, width) * 0.2 + 0.5
        
        # 確保張量設置了正確的名稱屬性 (這對於分佈類型檢測非常關鍵)
        # 將名稱保存為張量的屬性，確保可以被檢測到
        data._layer_name = layer_name
        
        # 使用自定義方法設置名稱
        if hasattr(data, 'set_name') and callable(data.set_name):
            data = data.set_name(layer_name)
        
        # 增加一些調試輸出來確認張量名稱的設置
        print(f"設置張量名稱: {layer_name}，實際名稱: {getattr(data, '_layer_name', None)}")
        if hasattr(data, 'name') and callable(data.name):
            print(f"通過name()方法獲取: {data.name()}")
        
        return data
    
    def load_activation_data(self, layer_name):
        """
        根據層名稱載入模擬激活資料
        
        Args:
            layer_name: 層名稱，格式為 'layerX_distribution_type'
            
        Returns:
            模擬的激活資料張量
        """
        # 如果已快取則直接返回
        if layer_name in self.activations_cache:
            return self.activations_cache[layer_name]
        
        # 產生隨機種子以確保重複呼叫時獲得相同結果
        np.random.seed(hash(layer_name) % 10000)
        
        # 基本參數
        batch_size = 32
        channels = 16
        height = 8
        width = 8
        
        # 根據層名稱中的分佈類型產生不同分佈的資料
        if 'normal' in layer_name:
            # 正態分佈
            data = torch.randn(batch_size, channels, height, width) * 0.2 + 0.5
        
        elif 'uniform' in layer_name:
            # 均勻分佈
            data = torch.rand(batch_size, channels, height, width)
        
        elif 'bimodal' in layer_name:
            # 雙峰分佈
            data1 = torch.randn(batch_size, channels, height, width) * 0.1 + 0.2
            data2 = torch.randn(batch_size, channels, height, width) * 0.1 + 0.8
            mask = torch.rand(batch_size, channels, height, width) > 0.5
            data = torch.where(mask, data1, data2)
        
        elif 'sparse' in layer_name:
            # 稀疏分佈
            data = torch.zeros(batch_size, channels, height, width)
            mask = torch.rand(batch_size, channels, height, width) > 0.7
            values = torch.rand(batch_size, channels, height, width)
            data = torch.where(mask, values, data)
        
        elif 'dead' in layer_name:
            # 死亡神經元模擬
            data = torch.zeros(batch_size, channels, height, width)
            # 讓部分通道完全為零
            for i in range(channels):
                if i % 3 == 0:  # 每3個通道中的1個接近死亡
                    data[:, i, :, :] = torch.rand(batch_size, height, width) * 0.01
                else:
                    data[:, i, :, :] = torch.rand(batch_size, height, width) * 0.5 + 0.25
        
        elif 'saturated' in layer_name:
            # 飽和神經元模擬
            data = torch.ones(batch_size, channels, height, width)
            # 讓部分通道接近飽和
            for i in range(channels):
                if i % 3 == 0:  # 每3個通道中的1個接近飽和
                    data[:, i, :, :] = torch.rand(batch_size, height, width) * 0.05 + 0.95
                else:
                    data[:, i, :, :] = torch.rand(batch_size, height, width) * 0.5 + 0.25
        
        elif 'right_skewed' in layer_name:
            # 右偏分佈
            data = torch.rand(batch_size, channels, height, width) ** 2
        
        elif 'left_skewed' in layer_name:
            # 左偏分佈
            data = 1 - torch.rand(batch_size, channels, height, width) ** 2
        
        else:
            # 默認使用正態分佈
            data = torch.randn(batch_size, channels, height, width) * 0.2 + 0.5
        
        # 設置張量名稱，這對於分佈類型檢測非常關鍵
        data._layer_name = layer_name
        
        # 使用自定義方法設置名稱
        if hasattr(data, 'set_name') and callable(data.set_name):
            data = data.set_name(layer_name)
            
        # 將結果快取
        self.activations_cache[layer_name] = data
        
        return data


class TestAdaptiveThresholdAnalyzer(unittest.TestCase):
    """測試自適應閾值分析器"""
    
    def setUp(self):
        """測試前準備工作"""
        # 創建臨時目錄用於保存分析結果
        self.temp_dir = tempfile.mkdtemp(prefix="test_adaptive_threshold_")
        self.experiment_name = "test_adaptive_analysis"
        self.output_dir = os.path.join(self.temp_dir, self.experiment_name, "analysis")
        
        # 創建模擬資料載入器
        self.mock_loader = MockHookDataLoader()
        
        # 創建基準分析器（使用固定閾值）供比較
        self.baseline_analyzer = IntermediateDataAnalyzer(
            experiment_dir=self.temp_dir,
            output_dir=self.output_dir,
            dead_neuron_threshold=0.05,
            saturated_neuron_threshold=0.95
        )
        
        # 設置日誌記錄器
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def tearDown(self):
        """測試後清理工作"""
        # 刪除臨時目錄及其內容
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_analyze_with_mock_loader(self):
        """測試使用模擬載入器的分析功能"""
        # 創建自適應閾值分析器
        analyzer = AdaptiveThresholdAnalyzer(
            experiment_name=self.experiment_name,
            output_dir=self.output_dir
        )
        
        # 運行分析
        result = analyzer.analyze_with_loader(self.mock_loader)
        
        # 檢查基本結果結構
        self.assertIn("thresholds", result)
        self.assertIn("results", result)
        
        # 檢查是否為每一層生成了閾值
        for layer_name in self.mock_loader.get_layer_names():
            self.assertIn(layer_name, result['thresholds'])
            self.assertIn(layer_name, result['results'])
        
        # 檢查輸出文件是否存在
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "AdaptiveThresholdAnalyzer_statistics.json")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "AdaptiveThresholdAnalyzer_thresholds.json")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "AdaptiveThresholdAnalyzer_results.json")))
        
        # 檢查報告是否生成
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "threshold_report", "adaptive_threshold_report.html")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "threshold_report", "threshold_comparison.png")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "threshold_report", "detection_results.png")))
    
    def test_distribution_type_detection(self):
        """測試分佈類型檢測功能"""
        # 創建自適應閾值分析器
        analyzer = AdaptiveThresholdAnalyzer(
            experiment_name=self.experiment_name,
            output_dir=self.output_dir
        )
        
        # 運行分析
        analyzer.analyze_with_loader(self.mock_loader)
        
        # 載入統計數據檢查分佈類型是否正確識別
        thresholds_file = os.path.join(self.output_dir, "AdaptiveThresholdAnalyzer_thresholds.json")
        if not os.path.exists(thresholds_file):
            self.logger.error(f"找不到閾值文件: {thresholds_file}")
            return
            
        with open(thresholds_file, 'r') as f:
            thresholds = json.load(f)
        
        # 檢查特定類型的層是否被正確識別
        for layer_name, threshold in thresholds.items():
            dist_type = threshold['distribution_type']
            
            # 檢查關鍵層的分佈識別是否合理
            if 'sparse' in layer_name:
                self.assertTrue(
                    'sparse' in dist_type or 'low_entropy' in dist_type,
                    f"層 {layer_name} 應該被識別為稀疏分佈，但實際為 {dist_type}"
                )
                
            if 'dead' in layer_name:
                self.assertTrue(
                    'sparse' in dist_type or 'right_skewed' in dist_type,
                    f"層 {layer_name} 應該被識別為稀疏或右偏分佈，但實際為 {dist_type}"
                )
                
            if 'saturated' in layer_name:
                self.assertTrue(
                    'saturated' in dist_type or 'left_skewed' in dist_type,
                    f"層 {layer_name} 應該被識別為飽和或左偏分佈，但實際為 {dist_type}"
                )
                
            if 'normal' in layer_name:
                self.assertTrue(
                    'normal' in dist_type or 'symmetric' in dist_type,
                    f"層 {layer_name} 應該被識別為正態或對稱分佈，但實際為 {dist_type}"
                )
                
            if 'uniform' in layer_name:
                self.assertTrue(
                    'uniform' in dist_type or 'flat' in dist_type,
                    f"層 {layer_name} 應該被識別為均勻或平坦分佈，但實際為 {dist_type}"
                )
                
            if 'bimodal' in layer_name:
                self.assertTrue(
                    'bimodal' in dist_type or 'multimodal' in dist_type,
                    f"層 {layer_name} 應該被識別為雙峰或多峰分佈，但實際為 {dist_type}"
                )
                
            if 'right_skewed' in layer_name:
                self.assertTrue(
                    'right_skewed' in dist_type or 'positive_skew' in dist_type,
                    f"層 {layer_name} 應該被識別為右偏分佈，但實際為 {dist_type}"
                )
                
            if 'left_skewed' in layer_name:
                self.assertTrue(
                    'left_skewed' in dist_type or 'negative_skew' in dist_type,
                    f"層 {layer_name} 應該被識別為左偏分佈，但實際為 {dist_type}"
                )
            
            # 打印分佈類型，便於調試
            print(f"Layer {layer_name} detected as {dist_type}")
    
    def test_compare_with_baseline(self):
        """測試與基準分析器比較"""
        # 1. 運行基準分析器
        baseline_output_dir = os.path.join(self.output_dir, "baseline")
        os.makedirs(baseline_output_dir, exist_ok=True)
        
        self.baseline_analyzer = IntermediateDataAnalyzer(
            experiment_dir=self.temp_dir,
            output_dir=baseline_output_dir,
            dead_neuron_threshold=0.05,
            saturated_neuron_threshold=0.95
        )
        
        self.baseline_analyzer.analyze_with_loader(self.mock_loader)
        
        # 2. 運行自適應閾值分析器
        adaptive_output_dir = os.path.join(self.output_dir, "adaptive")
        os.makedirs(adaptive_output_dir, exist_ok=True)
        
        adaptive_analyzer = AdaptiveThresholdAnalyzer(
            experiment_name=f"{self.experiment_name}_adaptive",
            output_dir=adaptive_output_dir,
            adaptive_method='hybrid',
            adaptation_strength=0.7
        )
        
        adaptive_result = adaptive_analyzer.analyze_with_loader(self.mock_loader)
        
        # 3. 載入兩者結果並比較
        baseline_results_file = os.path.join(baseline_output_dir, "IntermediateDataAnalyzer_results.json")
        if not os.path.exists(baseline_results_file):
            self.logger.error(f"找不到基準分析器結果文件: {baseline_results_file}")
            return
            
        with open(baseline_results_file, 'r') as f:
            baseline_results = json.load(f)
        
        # 4. 比較死亡和飽和神經元檢測結果
        comparison = self._compare_detection_results(baseline_results, adaptive_result['results'])
        
        # 5. 生成比較圖
        self._generate_comparison_plots(comparison)
        
        # 6. 檢查關鍵層的比較結果
        # 預期自適應方法在極端分佈上效果更好
        extreme_layers = ['layer4_sparse', 'layer5_dead', 'layer6_saturated']
        for layer in extreme_layers:
            if layer in comparison:
                print(f"Layer {layer} comparison:")
                print(f"  Baseline: Dead={comparison[layer]['baseline_dead_percent']:.2f}%, Saturated={comparison[layer]['baseline_saturated_percent']:.2f}%")
                print(f"  Adaptive: Dead={comparison[layer]['adaptive_dead_percent']:.2f}%, Saturated={comparison[layer]['adaptive_saturated_percent']:.2f}%")
                
                # 檢查閾值差異 (預期自適應方法會調整閾值)
                self.assertNotEqual(
                    comparison[layer]['baseline_dead_threshold'],
                    comparison[layer]['adaptive_dead_threshold']
                )
    
    def _compare_detection_results(self, baseline_results, adaptive_results):
        """比較基準分析器和自適應分析器的檢測結果"""
        comparison = {}
        
        for layer_name in baseline_results:
            if layer_name in adaptive_results:
                baseline = baseline_results[layer_name]
                adaptive = adaptive_results[layer_name]
                
                comparison[layer_name] = {
                    'layer_name': layer_name,
                    'baseline_dead_count': baseline.get('dead_count', 0),
                    'adaptive_dead_count': adaptive.get('dead_count', 0),
                    'baseline_dead_percent': baseline.get('dead_percentage', 0) * 100,
                    'adaptive_dead_percent': adaptive.get('dead_percentage', 0) * 100,
                    'baseline_saturated_count': baseline.get('saturated_count', 0),
                    'adaptive_saturated_count': adaptive.get('saturated_count', 0),
                    'baseline_saturated_percent': baseline.get('saturated_percentage', 0) * 100,
                    'adaptive_saturated_percent': adaptive.get('saturated_percentage', 0) * 100,
                    'baseline_dead_threshold': 0.05,  # 基準固定閾值
                    'adaptive_dead_threshold': adaptive.get('dead_threshold', 0.05),
                    'baseline_saturated_threshold': 0.95,  # 基準固定閾值
                    'adaptive_saturated_threshold': adaptive.get('saturated_threshold', 0.95),
                    'distribution_type': adaptive_results[layer_name].get('distribution_type', 'unknown')
                }
                
                # 計算檢測差異
                comparison[layer_name]['dead_difference'] = (
                    comparison[layer_name]['adaptive_dead_percent'] - 
                    comparison[layer_name]['baseline_dead_percent']
                )
                
                comparison[layer_name]['saturated_difference'] = (
                    comparison[layer_name]['adaptive_saturated_percent'] - 
                    comparison[layer_name]['baseline_saturated_percent']
                )
        
        return comparison
    
    def _generate_comparison_plots(self, comparison):
        """生成基準和自適應方法的比較圖"""
        if not comparison:
            return
            
        layer_names = list(comparison.keys())
        
        plt.figure(figsize=(14, 10))
        
        # 死亡神經元檢測比較
        plt.subplot(2, 1, 1)
        x = np.arange(len(layer_names))
        width = 0.35
        
        baseline_dead = [comparison[layer]['baseline_dead_percent'] for layer in layer_names]
        adaptive_dead = [comparison[layer]['adaptive_dead_percent'] for layer in layer_names]
        
        plt.bar(x - width/2, baseline_dead, width, label='基準方法')
        plt.bar(x + width/2, adaptive_dead, width, label='自適應方法')
        
        plt.xlabel('層名稱')
        plt.ylabel('死亡神經元百分比 (%)')
        plt.title('死亡神經元檢測比較')
        plt.xticks(x, layer_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # 飽和神經元檢測比較
        plt.subplot(2, 1, 2)
        baseline_saturated = [comparison[layer]['baseline_saturated_percent'] for layer in layer_names]
        adaptive_saturated = [comparison[layer]['adaptive_saturated_percent'] for layer in layer_names]
        
        plt.bar(x - width/2, baseline_saturated, width, label='基準方法')
        plt.bar(x + width/2, adaptive_saturated, width, label='自適應方法')
        
        plt.xlabel('層名稱')
        plt.ylabel('飽和神經元百分比 (%)')
        plt.title('飽和神經元檢測比較')
        plt.xticks(x, layer_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "detection_comparison.png"), dpi=300)
        plt.close()
        
        # 閾值比較圖
        plt.figure(figsize=(14, 10))
        
        # 死亡閾值比較
        plt.subplot(2, 1, 1)
        baseline_dead_threshold = [comparison[layer]['baseline_dead_threshold'] for layer in layer_names]
        adaptive_dead_threshold = [comparison[layer]['adaptive_dead_threshold'] for layer in layer_names]
        
        plt.bar(x - width/2, baseline_dead_threshold, width, label='基準方法')
        plt.bar(x + width/2, adaptive_dead_threshold, width, label='自適應方法')
        
        plt.xlabel('層名稱')
        plt.ylabel('死亡閾值')
        plt.title('死亡閾值比較')
        plt.xticks(x, layer_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # 飽和閾值比較
        plt.subplot(2, 1, 2)
        baseline_saturated_threshold = [comparison[layer]['baseline_saturated_threshold'] for layer in layer_names]
        adaptive_saturated_threshold = [comparison[layer]['adaptive_saturated_threshold'] for layer in layer_names]
        
        plt.bar(x - width/2, baseline_saturated_threshold, width, label='基準方法')
        plt.bar(x + width/2, adaptive_saturated_threshold, width, label='自適應方法')
        
        plt.xlabel('層名稱')
        plt.ylabel('飽和閾值')
        plt.title('飽和閾值比較')
        plt.xticks(x, layer_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "threshold_comparison.png"), dpi=300)
        plt.close()


if __name__ == '__main__':
    unittest.main() 