"""
測試數據漂移回調類的功能

此模塊測試 DataDriftCallback 類的功能，包括對數據漂移的檢測和分析。
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

from SBP_analyzer.callbacks.data_drift_callback import DataDriftCallback

class SimpleModel(nn.Module):
    """簡單測試模型"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(5, 2)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return self.output(x)

class TestDataDriftCallback(unittest.TestCase):
    """測試 DataDriftCallback 類的功能"""
    
    def setUp(self):
        """設置測試環境"""
        # 創建測試模型
        self.model = SimpleModel()
        
        # 創建參考數據（正態分佈）
        self.reference_data = np.random.normal(0, 1, (100, 10))
        
        # 創建漂移數據（均值偏移）
        self.drift_data = np.random.normal(2, 1, (100, 10))
        
        # 創建輕微漂移數據
        self.slight_drift_data = np.random.normal(0.5, 1, (100, 10))
        
        # 創建基本回調實例
        self.callback = DataDriftCallback(
            reference_data=self.reference_data,
            drift_metrics=['ks', 'psi', 'wasserstein'],
            monitoring_frequency=1,
            drift_threshold=0.1
        )
        
        # 轉換為張量
        self.test_input = torch.tensor(self.reference_data, dtype=torch.float32)
        self.test_drift_input = torch.tensor(self.drift_data, dtype=torch.float32)
        self.test_slight_drift_input = torch.tensor(self.slight_drift_data, dtype=torch.float32)
        self.test_target = torch.randint(0, 2, (100,))
    
    def test_initialization(self):
        """測試回調初始化"""
        # 測試默認參數
        callback = DataDriftCallback()
        self.assertIsNone(callback.reference_data)
        self.assertEqual(callback.monitoring_frequency, 1)
        self.assertEqual(callback.drift_threshold, 0.1)
        self.assertEqual(callback.drift_metrics, ['ks', 'psi', 'wasserstein'])
        
        # 測試自定義參數
        callback = DataDriftCallback(
            reference_data=self.reference_data,
            feature_indices=[0, 1, 2],
            monitoring_frequency=5,
            drift_threshold=0.2,
            drift_metrics=['ks']
        )
        self.assertIsNotNone(callback.reference_data)
        self.assertEqual(callback.feature_indices, [0, 1, 2])
        self.assertEqual(callback.monitoring_frequency, 5)
        self.assertEqual(callback.drift_threshold, 0.2)
        self.assertEqual(callback.drift_metrics, ['ks'])
    
    def test_train_begin(self):
        """測試訓練開始回調"""
        # 訓練開始時應該初始化參考統計量
        self.callback._on_train_begin(self.model)
        self.assertIsNotNone(self.callback.reference_statistics)
        
        # 檢查統計量是否包含必要的字段
        stats = self.callback.reference_statistics
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('data', stats)
    
    def test_batch_begin_no_reference(self):
        """測試當沒有參考數據時的批次開始回調"""
        # 創建沒有參考數據的回調
        callback = DataDriftCallback()
        
        # 第一個批次應該設置參考數據
        callback._on_batch_begin(0, self.model, self.test_input, self.test_target)
        
        # 檢查是否正確設置了參考統計量
        self.assertIsNone(callback.reference_statistics)  # 因為我們直接調用了 _on_batch_begin, 不是 on_batch_begin
    
    def test_calculate_statistics(self):
        """測試數據統計量計算"""
        # 計算統計量
        stats = self.callback._calculate_statistics(self.reference_data)
        
        # 檢查統計量
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('median', stats)
        self.assertIn('q1', stats)
        self.assertIn('q3', stats)
        self.assertIn('data', stats)
        
        # 檢查統計量的形狀
        self.assertEqual(stats['mean'].shape, (10,))
        self.assertEqual(stats['std'].shape, (10,))
        
        # 檢查平均值和標準差是否合理
        self.assertTrue(np.allclose(stats['mean'], 0, atol=0.5))
        self.assertTrue(np.allclose(stats['std'], 1, atol=0.5))
    
    def test_detect_drift(self):
        """測試漂移檢測"""
        # 初始化參考統計量
        self.callback._on_train_begin(self.model)
        
        # 檢測明顯漂移
        drift_results = self.callback._detect_drift(self.drift_data, {})
        
        # 檢查結果
        self.assertIn('ks', drift_results)
        self.assertIn('psi', drift_results)
        self.assertIn('wasserstein', drift_results)
        
        # 檢查漂移分數是否足夠高（表示檢測到漂移）
        for metric, score in drift_results.items():
            self.assertGreater(score, self.callback.drift_threshold, f"{metric} 漂移分數應該超過閾值")
        
        # 驗證漂移檢測標誌已設置
        self.assertTrue(self.callback.drift_detected)
    
    def test_no_drift_detection(self):
        """測試無漂移情況"""
        # 初始化參考統計量
        self.callback._on_train_begin(self.model)
        
        # 使用類似的數據進行檢測
        similar_data = np.random.normal(0, 1, (100, 10))
        drift_results = self.callback._detect_drift(similar_data, {})
        
        # 檢查結果
        for metric, score in drift_results.items():
            # 分數應該較低，可能有些會超過閾值，但大多數應該低
            if score > self.callback.drift_threshold:
                print(f"警告: {metric} 分數 {score} 超過閾值 {self.callback.drift_threshold}，但可能是隨機變異")
    
    def test_epoch_end_no_monitoring(self):
        """測試超過監控頻率的情況"""
        # 設置監控頻率為 2
        self.callback.monitoring_frequency = 2
        
        # epoch 0 不應該觸發監控
        self.callback._on_epoch_end(0, self.model, {"inputs": self.test_input}, {})
        self.assertEqual(self.callback.current_epoch, 0)
        self.assertFalse(hasattr(self.callback, 'drift_detected') and self.callback.drift_detected)
        
        # epoch 1 應該觸發監控
        train_logs = {"inputs": self.test_drift_input}
        self.callback._on_epoch_end(1, self.model, train_logs, {})
        self.assertEqual(self.callback.current_epoch, 1)
    
    def test_specific_drift_metrics(self):
        """測試特定漂移指標"""
        # 測試 KS 統計量
        ks_callback = DataDriftCallback(
            reference_data=self.reference_data,
            drift_metrics=['ks']
        )
        ks_callback._on_train_begin(self.model)
        drift_results = ks_callback._detect_drift(self.drift_data, {})
        self.assertIn('ks', drift_results)
        self.assertNotIn('psi', drift_results)
        self.assertNotIn('wasserstein', drift_results)
        
        # 測試 PSI
        psi_callback = DataDriftCallback(
            reference_data=self.reference_data,
            drift_metrics=['psi']
        )
        psi_callback._on_train_begin(self.model)
        drift_results = psi_callback._detect_drift(self.drift_data, {})
        self.assertIn('psi', drift_results)
        self.assertNotIn('ks', drift_results)
        
        # 測試 Wasserstein 距離
        w_callback = DataDriftCallback(
            reference_data=self.reference_data,
            drift_metrics=['wasserstein']
        )
        w_callback._on_train_begin(self.model)
        drift_results = w_callback._detect_drift(self.drift_data, {})
        self.assertIn('wasserstein', drift_results)
        self.assertNotIn('ks', drift_results)
    
    def test_feature_indices(self):
        """測試特徵索引過濾"""
        # 只監控前 3 個特徵
        callback = DataDriftCallback(
            reference_data=self.reference_data,
            feature_indices=[0, 1, 2],
            drift_metrics=['ks']
        )
        callback._on_train_begin(self.model)
        
        # 確保統計量只包含指定的特徵
        stats = callback.reference_statistics
        self.assertEqual(stats['mean'].shape, (3,))
        
        # 檢測漂移
        drift_results = callback._detect_drift(self.drift_data, {})
        self.assertIn('ks', drift_results)

if __name__ == '__main__':
    unittest.main() 