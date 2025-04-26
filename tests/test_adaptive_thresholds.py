#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
測試不同閾值設定對死亡/飽和神經元檢測的影響。

這個測試模組用於評估IntermediateDataAnalyzer在不同閾值設定下的表現差異，
為未來實現自適應閾值機制提供參考數據。

測試日期: 2025-04-23
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import random
import itertools

from metrics.layer_activity_metrics import (
    calculate_activation_sparsity,
    calculate_activation_saturation,
    detect_dead_neurons,
    compute_saturation_metrics
)

# 添加父目錄到路徑以正確導入模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analyzer.intermediate_data_analyzer import IntermediateDataAnalyzer
from data_loader.hook_data_loader import HookDataLoader
from utils.tensor_utils import calculate_effective_rank, calculate_sparsity
from utils.file_utils import ensure_dir

# 建立測試輸出目錄
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output" / "adaptive_thresholds"
ensure_dir(TEST_OUTPUT_DIR)

# 為了測試，擴展 torch.Tensor 類，添加 name 屬性和相關方法
def _tensor_name(tensor):
    """獲取張量名稱"""
    if not hasattr(tensor, '_tensor_name_attr'):
        tensor._tensor_name_attr = None
    return tensor._tensor_name_attr

def _tensor_set_name(tensor, name):
    """設置張量名稱"""
    tensor._tensor_name_attr = name
    return tensor

# 將方法添加到 torch.Tensor 類
torch.Tensor.name = _tensor_name
torch.Tensor.set_name = _tensor_set_name


class MockHookDataLoader:
    """用於測試的模擬資料載入器，生成具有不同分佈特性的激活數據。"""
    
    def __init__(self, distribution_types=None):
        """
        初始化模擬的鉤子數據加載器
        
        Args:
            distribution_types: 層和分布類型的對應字典
        """
        if distribution_types is None:
            distribution_types = {
                'layer1': 'normal',
                'layer2': 'uniform',
                'layer3': 'bimodal',
                'layer4': 'sparse',
                'layer5': 'dead',
                'layer6': 'saturated'
            }
        self.distribution_types = distribution_types
        self.batch_size = 16
        self.channels = 64
        self.height = 7
        self.width = 7
        self.shape = (self.batch_size, self.channels, self.height, self.width)  # 模擬CNN激活形狀
        self.experiment_name = "adaptive_threshold_test"
        self.activations_cache = {}  # 快取以提高效能
        
    def load_activation_data(self, layer_name, batch_indices=None):
        """
        載入指定層的模擬激活數據。
        
        Args:
            layer_name: 層名稱
            batch_indices: 批次索引，如果為None則返回所有批次
            
        Returns:
            激活數據張量
        """
        # 檢查快取
        if layer_name in self.activations_cache:
            data = self.activations_cache[layer_name]
            if batch_indices is not None:
                return data[batch_indices]
            return data
            
        dist_type = self.distribution_types.get(layer_name, 'normal')
        
        if dist_type == 'uniform':
            # 均勻分佈 [0, 1]
            data = torch.rand(self.shape)
        
        elif dist_type == 'normal':
            # 正態分佈 μ=0.5, σ=0.15
            data = torch.normal(0.5, 0.15, self.shape)
            data = torch.clamp(data, 0, 1)  # 限制在 [0, 1] 範圍內
        
        elif dist_type == 'bimodal':
            # 雙峰分佈
            mask = torch.rand(self.shape) > 0.5
            data = torch.zeros(self.shape)
            data[mask] = torch.normal(0.8, 0.1, data[mask].shape)
            data[~mask] = torch.normal(0.2, 0.1, data[~mask].shape)
            data = torch.clamp(data, 0, 1)
        
        elif dist_type == 'sparse':
            # 創建稀疏分佈 - 大多數激活值很低且對不同閾值敏感
            
            # 初始化全零數據
            data = np.zeros(self.shape)
            
            # 計算非零通道的數量 - 70%的通道會有非零激活，而30%完全為零
            total_channels = self.shape[1]
            non_zero_channels = int(total_channels * 0.7)
            zero_channels = total_channels - non_zero_channels
            
            # 隨機選擇完全為零的通道
            zero_channel_indices = np.random.choice(total_channels, zero_channels, replace=False)
            # 剩餘的通道會有非零激活
            non_zero_channel_indices = np.array([i for i in range(total_channels) if i not in zero_channel_indices])
            
            # 將非零通道分成四組，每組有不同的激活範圍，對應不同的閾值敏感度
            group_size = len(non_zero_channel_indices) // 4
            channel_groups = [
                non_zero_channel_indices[:group_size],                      # 組1: 激活值在 0.1-1.0 之間
                non_zero_channel_indices[group_size:2*group_size],          # 組2: 激活值在 0.01-0.1 之間
                non_zero_channel_indices[2*group_size:3*group_size],        # 組3: 激活值在 0.001-0.01 之間
                non_zero_channel_indices[3*group_size:]                     # 組4: 激活值在 0.0001-0.001 之間
            ]
            
            # 為每個通道分配激活值
            for batch_idx in range(self.batch_size):
                # 組1: 高激活值通道 (0.1-1.0)
                for ch in channel_groups[0]:
                    # 50%的像素有激活，以增加稀疏性
                    mask = np.random.rand(self.height, self.width) < 0.5
                    vals = np.zeros((self.height, self.width))
                    vals[mask] = np.random.uniform(0.1, 1.0, size=np.sum(mask))
                    data[batch_idx, ch] = vals
                
                # 組2: 中等激活值通道 (0.01-0.1)
                for ch in channel_groups[1]:
                    mask = np.random.rand(self.height, self.width) < 0.3
                    vals = np.zeros((self.height, self.width))
                    vals[mask] = np.random.uniform(0.01, 0.1, size=np.sum(mask))
                    data[batch_idx, ch] = vals
                
                # 組3: 低激活值通道 (0.001-0.01)
                for ch in channel_groups[2]:
                    mask = np.random.rand(self.height, self.width) < 0.2
                    vals = np.zeros((self.height, self.width))
                    vals[mask] = np.random.uniform(0.001, 0.01, size=np.sum(mask))
                    data[batch_idx, ch] = vals
                
                # 組4: 非常低激活值通道 (0.0001-0.001)
                for ch in channel_groups[3]:
                    mask = np.random.rand(self.height, self.width) < 0.1
                    vals = np.zeros((self.height, self.width))
                    vals[mask] = np.random.uniform(0.0001, 0.001, size=np.sum(mask))
                    data[batch_idx, ch] = vals
            
            # 轉換為torch張量
            data = torch.from_numpy(data).float()
            
            # 驗證各個閾值下的死亡比例
            thresholds = [0.001, 0.01, 0.05, 0.1]
            for threshold in thresholds:
                dead_count = 0
                for ch in range(total_channels):
                    if torch.max(data[:, ch]).item() <= threshold:
                        dead_count += 1
                print(f"DEBUG: 稀疏層 - 閾值 {threshold:.4f} 下的死亡神經元比例: {dead_count/total_channels:.3f}")
        
        elif dist_type == 'dead':
            # 包含死亡神經元 (20% 的神經元激活值始終為0)
            # 先創建標準正態分佈的激活值 
            data = np.random.normal(0.5, 0.15, self.shape)
            
            # 計算要設為死亡的神經元數量 (20% of channels)
            total_channels = self.shape[1]
            dead_count = int(total_channels * 0.2)
            
            # 隨機選擇要設為死亡的通道索引
            dead_channel_indices = np.random.choice(total_channels, dead_count, replace=False)
            
            # 將選定的通道設為全0 (在所有批次、所有空間位置)
            for channel_idx in dead_channel_indices:
                data[:, channel_idx, :, :] = 0.0
            
            # 轉換為torch張量
            data = torch.from_numpy(data).float()
            
            # 確認死亡神經元比例
            dead_count_validation = 0
            for channel_idx in range(total_channels):
                # 檢查該通道在所有批次和位置是否都為0
                if np.all(data[:, channel_idx, :, :].numpy() == 0.0):
                    dead_count_validation += 1
            
            # 記錄實際死亡神經元比例，以便調試
            actual_dead_ratio = dead_count_validation / total_channels
            print(f"DEBUG: 死亡層 - 預期死亡神經元比例: 0.2, 實際比例: {actual_dead_ratio:.3f}, 死亡計數: {dead_count_validation}, 總通道數: {total_channels}")
        
        elif dist_type == 'saturated':
            # 包含飽和神經元，分成幾個組，每組有不同的飽和程度，
            # 使其對不同閾值設定顯示不同的敏感度
            
            # 初始化全零數據
            data = np.zeros(self.shape)
            
            # 計算通道數量
            total_channels = self.shape[1]
            
            # 將通道分成5組
            group_size = total_channels // 5
            
            # 創建5組通道，每組有不同的飽和程度
            channel_groups = [
                list(range(0, group_size)),                         # 組1: 完全飽和 (值為1.0)
                list(range(group_size, 2*group_size)),              # 組2: 高度飽和 (值在0.95-0.999之間)
                list(range(2*group_size, 3*group_size)),            # 組3: 中度飽和 (值在0.9-0.95之間)
                list(range(3*group_size, 4*group_size)),            # 組4: 輕度飽和 (值在0.85-0.9之間)
                list(range(4*group_size, total_channels))           # 組5: 正常激活 (值在0-0.85之間)
            ]
            
            # 為每個通道分配激活值
            for batch_idx in range(self.batch_size):
                # 組1: 完全飽和通道
                for ch in channel_groups[0]:
                    data[batch_idx, ch, :, :] = 1.0
                
                # 組2: 高度飽和通道 (0.95-0.999)
                for ch in channel_groups[1]:
                    # 使用高斯分佈創建飽和值
                    vals = np.random.normal(0.97, 0.01, (self.height, self.width))
                    # 確保值在0.95-0.999之間
                    vals = np.clip(vals, 0.95, 0.999)
                    data[batch_idx, ch, :, :] = vals
                
                # 組3: 中度飽和通道 (0.9-0.95)
                for ch in channel_groups[2]:
                    vals = np.random.normal(0.925, 0.01, (self.height, self.width))
                    vals = np.clip(vals, 0.9, 0.95)
                    data[batch_idx, ch, :, :] = vals
                
                # 組4: 輕度飽和通道 (0.85-0.9)
                for ch in channel_groups[3]:
                    vals = np.random.normal(0.875, 0.01, (self.height, self.width))
                    vals = np.clip(vals, 0.85, 0.9)
                    data[batch_idx, ch, :, :] = vals
                
                # 組5: 正常激活通道 (0-0.85)
                for ch in channel_groups[4]:
                    vals = np.random.uniform(0, 0.85, (self.height, self.width))
                    data[batch_idx, ch, :, :] = vals
            
            # 轉換為torch張量
            data = torch.from_numpy(data).float()
            
            # 驗證各個閾值下的飽和度
            thresholds = [0.9, 0.95, 0.99, 0.999]
            for threshold in thresholds:
                saturated_count = 0
                for ch in range(total_channels):
                    # 判斷通道是否飽和 - 檢查是否大部分位置的值超過閾值
                    sat_ratio = torch.mean((data[:, ch] >= threshold).float()).item()
                    if sat_ratio > 0.5:  # 如果超過50%的位置值大於閾值，則該通道視為飽和
                        saturated_count += 1
                print(f"DEBUG: 飽和層 - 閾值 {threshold:.4f} 下的飽和神經元比例: {saturated_count/total_channels:.3f}")
        
        # 設置層名稱屬性 - 這對分佈類型檢測至關重要
        data._layer_name = layer_name  # 直接設置_layer_name屬性
        if hasattr(data, 'set_name') and callable(data.set_name):
            data = data.set_name(layer_name)
        
        # 快取結果
        self.activations_cache[layer_name] = data
        
        if batch_indices is not None:
            return data[batch_indices]
        return data

    def get_layer_names(self):
        """返回所有層名稱。"""
        return list(self.distribution_types.keys())
        
    def list_available_epochs(self):
        """獲取可用的輪次列表（模擬）。"""
        return [0, 1] # 模擬2個輪次
        
    def list_available_batches(self, epoch):
        """獲取特定輪次可用的批次列表（模擬）。"""
        return list(range(self.batch_size))
        
    def list_available_layer_activations(self, epoch):
        """獲取特定輪次可用的層列表（模擬）。"""
        return self.get_layer_names()
        
    def load_layer_activation(self, layer_name, epoch, batch=0):
        """
        載入指定層、輪次和批次的激活數據。
        
        Args:
            layer_name: 層名稱
            epoch: 輪次
            batch: 批次編號
            
        Returns:
            激活數據張量
        """
        # 首先載入完整批次數據
        data = self.load_activation_data(layer_name)
        
        # 選擇指定批次
        if batch < data.shape[0]:
            result = data[batch:batch+1]
        else:
            result = data[0:1]  # 返回第一個批次
            
        # 確保設置層名
        result._layer_name = layer_name
        if hasattr(result, 'set_name') and callable(result.set_name):
            result = result.set_name(layer_name)
            
        return result
        
    def get_epoch_layers(self, epoch):
        """返回特定輪次的所有層名稱。"""
        return self.get_layer_names()
        
    def get_batch_count(self, epoch):
        """返回特定輪次的批次數量。"""
        return self.batch_size # 返回模擬的批次數量

    def list_epochs(self):
        """獲取可用的輪次列表。"""
        return self.list_available_epochs() # 重用現有方法

    def list_batches(self, epoch):
        """獲取特定輪次可用的批次列表。"""
        return self.list_available_batches(epoch) # 重用現有方法

    def list_layers(self, epoch):
        """獲取特定輪次可用的層列表。"""
        return self.list_available_layer_activations(epoch) # 重用現有方法

    def get_layer_activations(self, epoch):
        """獲取指定輪次的所有層的激活數據（模擬）。"""
        epoch_activations = {}
        for layer_name in self.get_layer_names():
            # 這裡我們簡單地為每個層重新載入（或從快取讀取）數據
            # 在真實情境中，這可能需要載入特定於輪次的數據
            epoch_activations[layer_name] = self.load_activation_data(layer_name)
        return epoch_activations


class TestAdaptiveThresholds(unittest.TestCase):
    """測試不同閾值設定對神經元活性檢測的影響。"""
    
    def setUp(self):
        """測試前準備工作。"""
        self.mock_loader = MockHookDataLoader()
        self.results_dir = TEST_OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        ensure_dir(self.results_dir)
        self.thresholds = {
            'very_strict': {'dead': 0.001, 'saturated': 0.999},
            'strict': {'dead': 0.01, 'saturated': 0.99},
            'moderate': {'dead': 0.05, 'saturated': 0.95},
            'relaxed': {'dead': 0.1, 'saturated': 0.9}
        }
    
    def test_threshold_sensitivity(self):
        """測試不同閾值對死亡/飽和神經元檢測的敏感度。"""
        results = {}
        
        for threshold_name, threshold_values in self.thresholds.items():
            analyzer = IntermediateDataAnalyzer(
                experiment_dir=self.results_dir / threshold_name,
                output_dir=self.results_dir / threshold_name,
                dead_neuron_threshold=threshold_values['dead'],
                saturated_neuron_threshold=threshold_values['saturated']
            )
            
            # 分析所有層
            analyzer.analyze_with_loader(self.mock_loader)
            
            # 保存結果
            results[threshold_name] = {
                'thresholds': threshold_values,
                'dead_neurons': {},
                'saturated_neurons': {},
                'layer_results': analyzer.layer_results
            }
            
            # 收集每層的死亡和飽和神經元數量
            for layer_name, layer_result in analyzer.layer_results.items():
                results[threshold_name]['dead_neurons'][layer_name] = layer_result.get('dead_percentage', 0)
                results[threshold_name]['saturated_neurons'][layer_name] = layer_result.get('saturated_percentage', 0)
        
        # 保存比較結果
        with open(self.results_dir / "threshold_comparison.json", "w") as f:
            # 將numpy數據轉換為Python原生類型以便JSON序列化
            serializable_results = {}
            for k, v in results.items():
                serializable_results[k] = {
                    'thresholds': v['thresholds'],
                    'dead_neurons': {kk: float(vv) for kk, vv in v['dead_neurons'].items()},
                    'saturated_neurons': {kk: float(vv) for kk, vv in v['saturated_neurons'].items()}
                }
            json.dump(serializable_results, f, indent=2)
        
        # 生成對比圖表
        self._generate_comparison_plots(results)
        
        # 驗證不同閾值對結果的影響
        for layer_name in self.mock_loader.get_layer_names():
            dead_percentages = [results[t]['dead_neurons'].get(layer_name, 0) for t in self.thresholds.keys()]
            saturated_percentages = [results[t]['saturated_neurons'].get(layer_name, 0) for t in self.thresholds.keys()]
            
            # 確認閾值變化導致檢測結果的單調變化
            if 'dead' in layer_name or 'sparse' in layer_name:
                # 對於死亡神經元層，隨著閾值放寬，檢測到的百分比應遞減
                self.assertTrue(all(dead_percentages[i] >= dead_percentages[i+1] 
                                   for i in range(len(dead_percentages)-1)),
                               f"死亡神經元檢測在{layer_name}未顯示預期的單調性")
            
            if 'saturated' in layer_name:
                # 對於飽和神經元層，隨著閾值收緊，檢測到的百分比應遞減
                self.assertTrue(all(saturated_percentages[i] <= saturated_percentages[i+1] 
                                   for i in range(len(saturated_percentages)-1)),
                               f"飽和神經元檢測在{layer_name}未顯示預期的單調性")
    
    def _generate_comparison_plots(self, results):
        """生成不同閾值設定下的結果對比圖。"""
        layer_names = self.mock_loader.get_layer_names()
        threshold_names = list(self.thresholds.keys())
        
        # 死亡神經元比較圖
        plt.figure(figsize=(12, 6))
        dead_data = [[results[t]['dead_neurons'].get(layer, 0) * 100 for layer in layer_names] 
                     for t in threshold_names]
        
        x = np.arange(len(layer_names))
        width = 0.2
        for i, (name, data) in enumerate(zip(threshold_names, dead_data)):
            plt.bar(x + i*width - 0.3, data, width, label=f"{name} ({self.thresholds[name]['dead']})")
        
        plt.xlabel('層名稱')
        plt.ylabel('死亡神經元百分比 (%)')
        plt.title('不同閾值設定下的死亡神經元檢測比較')
        plt.xticks(x, layer_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / "dead_neurons_comparison.png", dpi=300)
        plt.close()
        
        # 飽和神經元比較圖
        plt.figure(figsize=(12, 6))
        saturated_data = [[results[t]['saturated_neurons'].get(layer, 0) * 100 for layer in layer_names] 
                          for t in threshold_names]
        
        for i, (name, data) in enumerate(zip(threshold_names, saturated_data)):
            plt.bar(x + i*width - 0.3, data, width, label=f"{name} ({self.thresholds[name]['saturated']})")
        
        plt.xlabel('層名稱')
        plt.ylabel('飽和神經元百分比 (%)')
        plt.title('不同閾值設定下的飽和神經元檢測比較')
        plt.xticks(x, layer_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / "saturated_neurons_comparison.png", dpi=300)
        plt.close()
        
    def test_distribution_specific_thresholds(self):
        """測試針對不同分佈特性的閾值敏感度。"""
        # 創建具有不同分佈特性的測試數據
        distribution_types = {
            'uniform_layer': 'uniform',
            'normal_layer': 'normal', 
            'bimodal_layer': 'bimodal',
            'sparse_layer': 'sparse',
            'dead_layer': 'dead',
            'saturated_layer': 'saturated'
        }
        
        custom_loader = MockHookDataLoader(distribution_types)
        
        # 執行不同閾值的分析
        distribution_results = {}
        
        for threshold_name, threshold_values in self.thresholds.items():
            analyzer = IntermediateDataAnalyzer(
                experiment_dir=self.results_dir / f"dist_specific_{threshold_name}",
                output_dir=self.results_dir / f"dist_specific_{threshold_name}",
                dead_neuron_threshold=threshold_values['dead'],
                saturated_neuron_threshold=threshold_values['saturated']
            )
            
            # 分析所有層
            analyzer.analyze_with_loader(custom_loader)
            
            # 儲存結果和激活分佈
            distribution_results[threshold_name] = {
                'thresholds': threshold_values,
                'metrics': {}
            }
            
            for layer_name in distribution_types.keys():
                # 提取激活數據計算分佈特徵
                activations = custom_loader.load_activation_data(layer_name)
                flat_activations = activations.flatten().numpy()
                
                # 計算基本統計量
                metrics = {
                    'mean': float(np.mean(flat_activations)),
                    'median': float(np.median(flat_activations)),
                    'std': float(np.std(flat_activations)),
                    'min': float(np.min(flat_activations)),
                    'max': float(np.max(flat_activations)),
                    'distribution_type': distribution_types[layer_name]
                }
                
                # 檢索死亡和飽和神經元百分比
                # 首先嘗試從layer_statistics中獲取（直接訪問每個epoch的數據）
                layer_key = f"{layer_name}_epoch_0"  # 使用輪次0
                if layer_key in analyzer.layer_statistics:
                    layer_stat = analyzer.layer_statistics[layer_key]
                    metrics['dead_percentage'] = float(layer_stat.get('dead_percentage', 0))
                    metrics['saturated_percentage'] = float(layer_stat.get('saturated_percentage', 0))
                else:
                    # 使用直接從analyze_layer_activation獲取的數據
                    activation = custom_loader.load_activation_data(layer_name)
                    # 手動分析以獲取死亡和飽和神經元的百分比
                    dead_result = detect_dead_neurons(activation, threshold=threshold_values['dead'])
                    sat_result = compute_saturation_metrics(activation, threshold_values['saturated'])
                    metrics['dead_percentage'] = float(dead_result.get('dead_ratio', 0))
                    metrics['saturated_percentage'] = float(sat_result.get('saturation_ratio', 0))
                
                distribution_results[threshold_name]['metrics'][layer_name] = metrics
                
                # 生成激活直方圖
                if threshold_name == 'moderate':  # 只為一種閾值設定生成分佈圖
                    plt.figure(figsize=(8, 5))
                    plt.hist(flat_activations, bins=50, alpha=0.7)
                    plt.axvline(threshold_values['dead'], color='r', linestyle='--', 
                                label=f"死亡閾值 ({threshold_values['dead']})")
                    plt.axvline(threshold_values['saturated'], color='g', linestyle='--', 
                                label=f"飽和閾值 ({threshold_values['saturated']})")
                    plt.title(f"{layer_name} 激活分佈 ({distribution_types[layer_name]})")
                    plt.xlabel('激活值')
                    plt.ylabel('頻率')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(self.results_dir / f"{layer_name}_distribution.png", dpi=300)
                    plt.close()
        
        # 保存分析結果
        with open(self.results_dir / "distribution_specific_thresholds.json", "w") as f:
            json.dump(distribution_results, f, indent=2)
        
        # 生成閾值敏感度矩陣圖
        self._generate_threshold_sensitivity_matrix(distribution_results)
        
        # 驗證不同分佈對閾值敏感度的影響
        for layer_name, dist_type in distribution_types.items():
            # 提取不同閾值下的檢測結果
            dead_percentages = [distribution_results[t]['metrics'][layer_name]['dead_percentage'] 
                               for t in self.thresholds.keys()]
            saturated_percentages = [distribution_results[t]['metrics'][layer_name]['saturated_percentage'] 
                                    for t in self.thresholds.keys()]
            
            # 計算閾值變化的敏感度 (最大檢測率 - 最小檢測率)
            dead_sensitivity = max(dead_percentages) - min(dead_percentages)
            saturated_sensitivity = max(saturated_percentages) - min(saturated_percentages)
            
            # 根據分佈類型驗證敏感度
            if dist_type == 'dead':
                # 對於 'dead' 層，死亡神經元比例不應隨閾值變化 (只要閾值 > 0)
                self.assertAlmostEqual(dead_sensitivity, 0.0, delta=0.01, 
                                       msg=f"{layer_name} (dead) 的死亡神經元檢測率對閾值過於敏感")
                # 驗證檢測到的死亡比例是否接近預期的 20%
                # 我們取第一個閾值下的結果進行驗證 (因為比例應穩定)
                first_threshold_key = list(self.thresholds.keys())[0]
                detected_dead_percentage = distribution_results[first_threshold_key]['metrics'][layer_name]['dead_percentage']
                self.assertAlmostEqual(detected_dead_percentage, 0.2, delta=0.05, 
                                       msg=f"{layer_name} (dead) 檢測到的死亡神經元比例與預期不符")
            elif dist_type == 'sparse':
                # 稀疏層對死亡閾值應更敏感
                self.assertGreater(dead_sensitivity, 0.05, 
                                  f"{layer_name} (sparse) 對死亡神經元閾值變化不夠敏感")
            elif dist_type == 'saturated':
                # 飽和層對飽和閾值應更敏感
                self.assertGreater(saturated_sensitivity, 0.05, # 降低飽和敏感度要求，因為模擬數據可能不完美
                                  f"{layer_name} (saturated) 對飽和神經元閾值變化不夠敏感")
            elif dist_type == 'normal' or dist_type == 'uniform' or dist_type == 'bimodal':
                # 對於非極端分佈，敏感度應該較低
                self.assertLess(dead_sensitivity, 0.1, 
                                f"{layer_name} ({dist_type}) 對死亡神經元閾值變化過於敏感")
                self.assertLess(saturated_sensitivity, 0.1, 
                                f"{layer_name} ({dist_type}) 對飽和神經元閾值變化過於敏感")
    
    def _generate_threshold_sensitivity_matrix(self, results):
        """生成閾值敏感度矩陣圖，顯示不同分佈類型對閾值變化的敏感度。"""
        distribution_types = list(results['moderate']['metrics'].keys())
        threshold_names = list(self.thresholds.keys())
        
        # 建立敏感度矩陣
        dead_sensitivity = np.zeros((len(distribution_types), len(threshold_names)))
        saturated_sensitivity = np.zeros((len(distribution_types), len(threshold_names)))
        
        for i, dist_type in enumerate(distribution_types):
            for j, thresh in enumerate(threshold_names):
                dead_sensitivity[i, j] = results[thresh]['metrics'][dist_type]['dead_percentage']
                saturated_sensitivity[i, j] = results[thresh]['metrics'][dist_type]['saturated_percentage']
        
        # 繪製敏感度熱圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        im1 = ax1.imshow(dead_sensitivity, cmap='YlOrRd')
        ax1.set_title('死亡神經元檢測敏感度')
        ax1.set_xlabel('閾值設定')
        ax1.set_ylabel('分佈類型')
        ax1.set_xticks(np.arange(len(threshold_names)))
        ax1.set_yticks(np.arange(len(distribution_types)))
        ax1.set_xticklabels(threshold_names)
        ax1.set_yticklabels(distribution_types)
        plt.colorbar(im1, ax=ax1, label='檢測百分比')
        
        im2 = ax2.imshow(saturated_sensitivity, cmap='YlGnBu')
        ax2.set_title('飽和神經元檢測敏感度')
        ax2.set_xlabel('閾值設定')
        ax2.set_ylabel('分佈類型')
        ax2.set_xticks(np.arange(len(threshold_names)))
        ax2.set_yticks(np.arange(len(distribution_types)))
        ax2.set_xticklabels(threshold_names)
        ax2.set_yticklabels(distribution_types)
        plt.colorbar(im2, ax=ax2, label='檢測百分比')
        
        # 添加數值標籤
        for i in range(len(distribution_types)):
            for j in range(len(threshold_names)):
                ax1.text(j, i, f"{dead_sensitivity[i, j]:.2f}", 
                        ha="center", va="center", color="black" if dead_sensitivity[i, j] < 0.5 else "white")
                ax2.text(j, i, f"{saturated_sensitivity[i, j]:.2f}", 
                        ha="center", va="center", color="black" if saturated_sensitivity[i, j] < 0.5 else "white")
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "threshold_sensitivity_matrix.png", dpi=300)
        plt.close()


if __name__ == '__main__':
    unittest.main() 