"""
測試Outlier Detection Callback模組

本模組包含用於測試OutlierDetectionCallback功能的測試，
包括離群值檢測、監控和保存離群樣本等功能。
"""

import os
import pytest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# 確保可以導入要測試的類
from SBP_analyzer.callbacks.outlier_detection_callback import OutlierDetectionCallback


class TestOutlierDetectionCallback:
    
    def setup_method(self):
        """每個測試方法前設置環境"""
        # 創建測試數據
        self.batch_size = 8
        
        # 創建正常數據
        self.normal_data = torch.randn(40, 10)
        
        # 創建離群值數據
        self.outliers = torch.randn(8, 10) * 5 + 10
        
        # 合併數據
        self.all_data = torch.cat([self.normal_data, self.outliers])
        self.targets = torch.randint(0, 2, (48,))
        
        # 創建數據加載器
        self.dataset = TensorDataset(self.all_data, self.targets)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        
        # 創建臨時輸出目錄
        self.output_dir = "./temp_outlier_test"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def teardown_method(self):
        """每個測試方法後清理環境"""
        # 刪除測試過程中創建的文件
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
    
    def test_initialization(self):
        """測試回調函數的初始化"""
        # 使用默認參數
        callback = OutlierDetectionCallback()
        assert callback.detection_method == 'iforest'
        assert callback.monitoring_level == 'batch'
        assert callback.threshold == 0.99
        assert callback.save_outliers is False
        
        # 使用自定義參數
        callback = OutlierDetectionCallback(
            detection_method='lof',
            monitoring_level='all',
            threshold=0.95,
            save_outliers=True,
            output_dir=self.output_dir
        )
        assert callback.detection_method == 'lof'
        assert callback.monitoring_level == 'all'
        assert callback.threshold == 0.95
        assert callback.save_outliers is True
        assert callback.output_dir == self.output_dir
    
    def test_train_begin(self, simple_model):
        """測試訓練開始時的回調處理"""
        callback = OutlierDetectionCallback(monitoring_level='all')
        
        # 模擬訓練開始
        callback.on_train_begin(simple_model)
        
        # 檢查初始化的收集器
        assert hasattr(callback, 'all_data')
        assert len(callback.all_data) == 0
        assert callback.model == simple_model
    
    def test_batch_end(self, simple_model):
        """測試批次結束時的回調處理"""
        callback = OutlierDetectionCallback(monitoring_level='batch')
        callback.on_train_begin(simple_model)
        
        # 獲取一個批次的數據
        for inputs, targets in self.dataloader:
            break
        
        # 模擬批次結束
        callback.on_batch_end(inputs, targets, None)
        
        # 檢查批次監控
        if callback.batch_detector:
            assert callback.batch_detector.n_samples_ == len(inputs)
    
    def test_train_end_monitoring_all(self, simple_model):
        """測試整體訓練數據的監控"""
        callback = OutlierDetectionCallback(
            monitoring_level='all',
            detection_method='iforest',
            threshold=0.95
        )
        callback.on_train_begin(simple_model)
        
        # 模擬收集所有數據
        for inputs, targets in self.dataloader:
            callback.all_data.extend(inputs.detach().numpy())
        
        # 模擬訓練結束
        callback.on_train_end()
        
        # 檢查是否創建了探測器
        assert callback.all_detector is not None
        
        # 檢查數據尺寸
        assert callback.all_detector.n_samples_ == len(self.all_data)
    
    def test_detect_outliers_iforest(self):
        """測試使用IsolationForest檢測離群值"""
        callback = OutlierDetectionCallback(
            detection_method='iforest',
            threshold=0.9
        )
        
        # 使用已知數據檢測離群值
        normal_data_np = self.normal_data.numpy()
        outliers_np = self.outliers.numpy()
        all_data_np = np.vstack([normal_data_np, outliers_np])
        
        # 執行離群值檢測
        outlier_indices = callback._detect_outliers(all_data_np)
        
        # 驗證檢測結果 (由於算法的隨機性，我們只檢查是否檢測到離群值)
        assert len(outlier_indices) > 0
    
    def test_detect_outliers_lof(self):
        """測試使用LocalOutlierFactor檢測離群值"""
        callback = OutlierDetectionCallback(
            detection_method='lof',
            threshold=0.9
        )
        
        # 使用已知數據檢測離群值
        normal_data_np = self.normal_data.numpy()
        outliers_np = self.outliers.numpy()
        all_data_np = np.vstack([normal_data_np, outliers_np])
        
        # 執行離群值檢測
        outlier_indices = callback._detect_outliers(all_data_np)
        
        # 驗證檢測結果
        assert len(outlier_indices) > 0
    
    def test_handle_outliers(self, simple_model):
        """測試處理檢測到的離群值"""
        callback = OutlierDetectionCallback(
            detection_method='iforest',
            threshold=0.9,
            monitoring_level='batch'
        )
        callback.on_train_begin(simple_model)
        
        # 獲取一個批次的數據
        for inputs, targets in self.dataloader:
            break
        
        # 強制設置一些離群值的索引
        outlier_indices = [0, 2, 4]  # 假設這些是離群值
        batch_data = inputs.numpy()
        
        # 手動調用處理離群值的方法
        callback._handle_outliers(outlier_indices, batch_data, targets.numpy())
        
        # 檢查是否記錄了離群值信息
        assert len(callback.outlier_info) == len(outlier_indices)
    
    def test_save_outlier_samples(self):
        """測試保存離群樣本"""
        callback = OutlierDetectionCallback(
            save_outliers=True,
            output_dir=self.output_dir
        )
        
        # 模擬一些離群值
        outlier_data = self.outliers[:3].numpy()
        outlier_targets = np.array([0, 1, 0])
        scores = np.array([0.8, 0.9, 0.85])
        
        # 創建離群值信息
        for i in range(len(outlier_data)):
            callback.outlier_info.append({
                'data': outlier_data[i],
                'target': outlier_targets[i],
                'score': scores[i],
                'batch_idx': 0,
                'sample_idx': i
            })
        
        # 保存離群樣本
        callback._save_outlier_samples()
        
        # 檢查是否創建了CSV文件
        csv_path = os.path.join(self.output_dir, "outliers_info.csv")
        assert os.path.exists(csv_path)
        
        # 驗證CSV內容
        df = pd.read_csv(csv_path)
        assert len(df) == len(callback.outlier_info)
        assert 'score' in df.columns
        assert 'batch_idx' in df.columns
    
    def test_end_to_end_training(self, simple_model):
        """端到端測試整個訓練過程中的離群值檢測"""
        callback = OutlierDetectionCallback(
            detection_method='iforest',
            monitoring_level='all',
            threshold=0.95,
            save_outliers=True,
            output_dir=self.output_dir
        )
        
        # 模擬完整的訓練過程
        callback.on_train_begin(simple_model)
        
        # 模擬多個批次的數據處理
        for i, (inputs, targets) in enumerate(self.dataloader):
            callback.on_batch_end(inputs, targets, epoch=0)
        
        # 模擬訓練結束
        callback.on_train_end()
        
        # 檢查最終結果
        if callback.save_outliers and callback.outlier_info:
            csv_path = os.path.join(self.output_dir, "outliers_info.csv")
            assert os.path.exists(csv_path)
    
    def test_with_different_thresholds(self):
        """測試不同閾值對離群值檢測的影響"""
        # 創建測試數據
        normal_data_np = self.normal_data.numpy()
        outliers_np = self.outliers.numpy()
        all_data_np = np.vstack([normal_data_np, outliers_np])
        
        # 使用不同閾值進行測試
        thresholds = [0.99, 0.95, 0.9, 0.8]
        results = []
        
        for threshold in thresholds:
            callback = OutlierDetectionCallback(
                detection_method='iforest',
                threshold=threshold
            )
            
            # 執行離群值檢測
            outlier_indices = callback._detect_outliers(all_data_np)
            results.append(len(outlier_indices))
        
        # 檢查閾值降低時，檢測到的離群值數量是否增加
        for i in range(1, len(results)):
            assert results[i] >= results[i-1] 