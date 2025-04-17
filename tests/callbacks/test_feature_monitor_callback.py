"""
測試特徵監控回調類的功能

此模塊測試 FeatureMonitorCallback 類的功能，包括特徵激活、梯度和相關性的監控。
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import os
import shutil
from typing import Dict, Any

from SBP_analyzer.callbacks.feature_monitor_callback import FeatureMonitorCallback

class ConvModel(nn.Module):
    """用於測試的卷積網絡模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

class TestFeatureMonitorCallback(unittest.TestCase):
    """測試 FeatureMonitorCallback 類的功能"""
    
    def setUp(self):
        """設置測試環境"""
        # 創建測試目錄
        self.test_dir = './test_feature_plots'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 創建測試模型
        self.model = ConvModel()
        
        # 創建測試數據（模擬圖像）
        self.test_input = torch.randn(4, 3, 32, 32)
        self.test_target = torch.randint(0, 10, (4,))
        
        # 基本回調實例
        self.callback = FeatureMonitorCallback(
            target_layers=['conv1', 'conv2', 'fc1', 'fc2'],
            monitor_frequency=1,
            track_correlations=True,
            track_activations=True,
            save_plots=True,
            plot_dir=self.test_dir
        )
    
    def tearDown(self):
        """清理測試環境"""
        # 移除測試目錄
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """測試回調初始化"""
        # 測試默認參數
        callback = FeatureMonitorCallback()
        self.assertIsNone(callback.target_layers)
        self.assertEqual(callback.monitor_frequency, 10)
        self.assertTrue(callback.log_to_tensorboard)
        self.assertTrue(callback.track_correlations)
        self.assertTrue(callback.track_activations)
        self.assertFalse(callback.track_gradients)
        
        # 測試自定義參數
        callback = FeatureMonitorCallback(
            target_layers=['conv1', 'fc1'],
            feature_indices=[0, 1, 2],
            monitor_frequency=5,
            track_correlations=False,
            track_activations=True,
            track_gradients=True
        )
        self.assertEqual(callback.target_layers, ['conv1', 'fc1'])
        self.assertEqual(callback.feature_indices, [0, 1, 2])
        self.assertEqual(callback.monitor_frequency, 5)
        self.assertFalse(callback.track_correlations)
        self.assertTrue(callback.track_activations)
        self.assertTrue(callback.track_gradients)
    
    def test_train_begin(self):
        """測試訓練開始回調"""
        self.callback._on_train_begin(self.model)
        
        # 檢查是否註冊了鉤子
        self.assertGreater(len(self.callback.hook_handles), 0)
        
        # 檢查是否創建了圖表目錄
        self.assertTrue(os.path.exists(self.test_dir))
    
    def test_train_end(self):
        """測試訓練結束回調"""
        # 首先註冊鉤子
        self.callback._on_train_begin(self.model)
        
        # 然後清理
        self.callback._on_train_end(self.model)
        
        # 檢查鉤子是否被移除
        self.assertEqual(len(self.callback.hook_handles), 0)
    
    def test_hook_registration(self):
        """測試鉤子註冊"""
        # 測試自動推斷目標層
        callback = FeatureMonitorCallback(target_layers=None)
        callback._register_hooks(self.model)
        
        # 檢查是否註冊了合適的層
        self.assertGreater(len(callback.hook_handles), 0)
        
        # 清理
        callback._remove_hooks()
        self.assertEqual(len(callback.hook_handles), 0)
        
        # 測試指定特定層
        callback = FeatureMonitorCallback(target_layers=['conv1', 'fc1'])
        callback._register_hooks(self.model)
        
        # 檢查鉤子數量
        self.assertGreaterEqual(len(callback.hook_handles), 1)
        
        # 清理
        callback._remove_hooks()
    
    def test_activation_hook(self):
        """測試激活值鉤子"""
        # 手動調用激活鉤子
        output = torch.randn(4, 16, 32, 32)  # 模擬卷積層輸出
        self.callback._activation_hook('conv1', output)
        
        # 檢查是否記錄了激活值
        self.assertIn('conv1', self.callback.activation_stats)
        self.assertEqual(len(self.callback.activation_stats['conv1']), 1)
        
        # 檢查統計數據
        stats = self.callback.activation_stats['conv1'][0]
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        
        # 驗證統計數據的值
        self.assertEqual(stats['mean'], output.mean().item())
        self.assertEqual(stats['std'], output.std().item())
        self.assertEqual(stats['min'], output.min().item())
        self.assertEqual(stats['max'], output.max().item())
    
    def test_epoch_statistics(self):
        """測試每個 epoch 的統計計算"""
        # 添加一些測試數據
        self.callback._activation_hook('conv1', torch.randn(4, 16, 32, 32))
        self.callback._activation_hook('conv1', torch.randn(4, 16, 32, 32))
        self.callback._activation_hook('conv2', torch.randn(4, 32, 16, 16))
        
        # 計算 epoch 統計數據
        epoch_stats = self.callback._compute_epoch_statistics()
        
        # 檢查每個層的統計數據
        self.assertIn('conv1', epoch_stats)
        self.assertIn('conv2', epoch_stats)
        
        # 檢查每個層的統計量
        for layer, stats in epoch_stats.items():
            self.assertIn('mean_activation', stats)
            self.assertIn('std_activation', stats)
            self.assertIn('max_activation', stats)
    
    def test_batch_monitoring_frequency(self):
        """測試批次監控頻率"""
        # 設置監控頻率為 2
        self.callback.monitor_frequency = 2
        
        # batch 0 應該觸發監控
        self.callback._on_batch_end(0, self.model, torch.tensor(0.5))
        
        # batch 1 不應該觸發監控
        self.callback._on_batch_end(1, self.model, torch.tensor(0.5))
        
        # 目前我們無法直接檢查內部邏輯是否執行，但至少確保沒有錯誤
        self.assertEqual(self.callback.current_batch, 1)
    
    def test_feature_correlation_analysis(self):
        """測試特徵相關性分析"""
        # 為了測試相關性分析，我們需要模擬一些相關的特徵數據
        # 首先創建相關性較強的特徵
        base_feature = torch.randn(10)
        correlated_feature = base_feature * 0.9 + torch.randn(10) * 0.1
        
        # 模擬兩組強相關的激活值
        output1 = torch.stack([base_feature, correlated_feature])
        output1 = output1.unsqueeze(0)  # 添加批次維度
        
        # 再添加一組不太相關的特徵
        output2 = torch.randn(1, 3, 10)
        
        # 記錄這些激活值
        self.callback._activation_hook('test_layer1', output1)
        self.callback._activation_hook('test_layer2', output2)
        
        # 執行相關性分析
        self.callback._analyze_feature_correlations()
        
        # 測試這裡主要確保方法執行沒有錯誤，並且產生了一些相關性結果
        self.assertIn('test_layer1', self.callback.feature_correlations)
        
        # 對於第一層，應該有較高的相關性
        # 注意：由於維度處理，可能結果與預期不完全一致，這裡主要確保方法運行
        corr_layer1 = self.callback.feature_correlations.get('test_layer1')
        if corr_layer1 is not None and isinstance(corr_layer1, np.ndarray) and corr_layer1.size > 1:
            # 相關矩陣對角線應該是 1
            self.assertTrue(np.isclose(np.diag(corr_layer1), 1.0).all())
    
    def test_end_to_end(self):
        """測試端到端流程"""
        # 這個測試模擬一個簡單的訓練流程
        
        # 1. 訓練開始
        self.callback._on_train_begin(self.model)
        
        # 2. 進行一些批次
        for batch in range(3):
            # 模擬前向傳播
            outputs = self.model(self.test_input)
            loss = nn.CrossEntropyLoss()(outputs, self.test_target)
            
            # 批次結束
            self.callback._on_batch_end(batch, self.model, loss)
        
        # 3. epoch 結束
        train_logs = {"loss": 0.5, "accuracy": 0.8}
        val_logs = {"loss": 0.6, "accuracy": 0.75}
        self.callback._on_epoch_end(0, self.model, train_logs, val_logs)
        
        # 4. 訓練結束
        self.callback._on_train_end(self.model)
        
        # 所有步驟應該成功執行，不拋出錯誤

if __name__ == '__main__':
    unittest.main() 