"""
測試性能指標計算模組。

此模組包含對performance_metrics.py中函數的單元測試。
"""

import unittest
import os
import json
import numpy as np
import sys
from pathlib import Path

# 將專案根目錄添加到Python路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics.performance_metrics import (
    analyze_loss_curve,
    analyze_metric_curve,
    analyze_learning_efficiency,
    analyze_lr_schedule_impact,
    detect_training_anomalies,
    detect_overfitting,
    detect_oscillations,
    detect_plateaus,
    _calculate_moving_average,
    _find_convergence_point
)


class TestPerformanceMetrics(unittest.TestCase):
    """
    測試性能指標計算函數。
    """
    
    def setUp(self):
        """
        設置測試環境。
        """
        # 簡單的損失曲線
        self.loss_values = [10.0, 8.0, 6.0, 5.0, 4.0, 3.5, 3.2, 3.0, 2.9, 2.8]
        self.val_loss_values = [11.0, 9.0, 7.0, 6.0, 5.5, 5.0, 4.8, 4.7, 4.6, 4.7]  # 最後有輕微過擬合
        
        # 簡單的指標曲線
        self.metric_values = [0.5, 0.55, 0.6, 0.65, 0.7, 0.72, 0.74, 0.75, 0.76, 0.77]
        self.val_metric_values = [0.48, 0.53, 0.58, 0.62, 0.65, 0.68, 0.69, 0.7, 0.7, 0.69]  # 最後有輕微過擬合
        
        # 學習率序列
        self.learning_rates = [0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001]
        
        # 不收斂的損失曲線
        self.non_converging_loss = [5.0, 4.9, 4.8, 4.9, 4.7, 4.8, 4.7, 4.6, 4.7, 4.5]
        
        # 振盪的損失曲線
        self.oscillating_loss = [5.0, 4.0, 4.5, 3.5, 4.0, 3.0, 3.5, 2.5, 3.0, 2.0]
        
        # 平台期的損失曲線
        self.plateau_loss = [5.0, 4.0, 3.0, 2.9, 2.89, 2.9, 2.88, 2.9, 2.5, 2.0]
    
    def test_analyze_loss_curve(self):
        """
        測試損失曲線分析函數。
        """
        # 測試基本分析功能
        result = analyze_loss_curve(self.loss_values, self.val_loss_values)
        
        # 檢查結果是否包含預期的鍵
        self.assertIn('train_loss', result)
        self.assertIn('val_loss', result)
        self.assertIn('train_val_difference', result)
        
        # 檢查訓練損失分析
        self.assertEqual(result['train_loss']['min_value'], min(self.loss_values))
        self.assertEqual(result['train_loss']['min_epoch'], self.loss_values.index(min(self.loss_values)))
        
        # 檢查驗證損失分析
        self.assertEqual(result['val_loss']['min_value'], min(self.val_loss_values))
        self.assertEqual(result['val_loss']['min_epoch'], self.val_loss_values.index(min(self.val_loss_values)))
        
        # 檢查收斂點分析
        self.assertIn('convergence_epoch', result['train_loss'])
    
    def test_analyze_metric_curve(self):
        """
        測試指標曲線分析函數。
        """
        # 測試基本分析功能
        result = analyze_metric_curve(self.metric_values, self.val_metric_values)
        
        # 檢查結果是否包含預期的鍵
        self.assertIn('train_metric', result)
        self.assertIn('val_metric', result)
        
        # 檢查指標分析
        self.assertIn('improvement_rate', result)
        self.assertTrue(result['improvement_rate'] > 0)
        
        # 使用is_higher_better=False測試
        result_reversed = analyze_metric_curve(self.metric_values, self.val_metric_values, is_higher_better=False)
        self.assertTrue(result_reversed['improvement_rate'] < 0)
    
    def test_analyze_learning_efficiency(self):
        """
        測試學習效率分析函數。
        """
        result = analyze_learning_efficiency(self.loss_values)
        
        # 檢查結果是否包含預期的鍵
        self.assertIn('initial_learning_rate', result)
        self.assertIn('overall_learning_rate', result)
        self.assertIn('learning_rate_curve', result)
        
        # 檢查學習率分析
        self.assertTrue(result['initial_learning_rate'] > 0)
        self.assertTrue(result['overall_learning_rate'] > 0)
    
    def test_analyze_lr_schedule_impact(self):
        """
        測試學習率調度影響分析函數。
        """
        result = analyze_lr_schedule_impact(self.loss_values, self.learning_rates)
        
        # 檢查結果是否包含預期的鍵
        self.assertIn('lr_changes', result)
        self.assertIn('lr_impact', result)
        
        # 檢查學習率變化點
        self.assertEqual(len(result['lr_changes']), 2)  # 應該有2個變化點
        self.assertEqual(result['lr_changes'], [4, 7])  # 變化點應該在索引4和7
    
    def test_detect_training_anomalies(self):
        """
        測試訓練異常檢測函數。
        """
        # 測試震盪損失曲線
        result = detect_training_anomalies(self.oscillating_loss)
        self.assertTrue(result['has_anomalies'])
        self.assertIn('oscillations', result['details'])
        
        # 測試平台期損失曲線
        result = detect_training_anomalies(self.plateau_loss)
        self.assertTrue(result['has_anomalies'])
        self.assertIn('plateaus', result['details'])
        
        # 測試過擬合檢測
        result = detect_training_anomalies(self.loss_values, self.val_loss_values)
        self.assertIn('overfitting', result['details'])
    
    def test_detect_overfitting(self):
        """
        測試過擬合檢測函數。
        """
        # 創建明顯過擬合的序列
        train_loss = [10, 8, 6, 4, 3, 2, 1.5, 1.2, 1.0, 0.8]
        val_loss = [11, 9, 7, 5, 4, 3.5, 3.7, 4.0, 4.5, 5.0]
        
        result = detect_overfitting(train_loss, val_loss)
        self.assertTrue(result['has_overfitting'])
        self.assertGreater(result['severity'], 0)
        
        # 測試不過擬合的情況
        train_loss2 = [10, 8, 6, 4, 3, 2.5, 2.2, 2.0, 1.9, 1.8]
        val_loss2 = [11, 9, 7, 5, 4, 3.5, 3.2, 3.0, 2.9, 2.8]
        
        result2 = detect_overfitting(train_loss2, val_loss2)
        self.assertFalse(result2['has_overfitting'])
    
    def test_detect_oscillations(self):
        """
        測試震盪檢測函數。
        """
        result = detect_oscillations(self.oscillating_loss)
        self.assertTrue(result.get('has_oscillations', False))
        
        # 測試不震盪的情況
        result2 = detect_oscillations(self.loss_values)
        self.assertFalse(result2.get('has_oscillations', False))
    
    def test_detect_plateaus(self):
        """
        測試平台期檢測函數。
        """
        result = detect_plateaus(self.plateau_loss)
        self.assertTrue(result.get('has_plateaus', False))
        
        # 測試沒有平台期的情況
        result2 = detect_plateaus(self.loss_values)
        self.assertFalse(result2.get('has_plateaus', False))
    
    def test_calculate_moving_average(self):
        """
        測試移動平均計算函數。
        """
        values = [1, 2, 3, 4, 5]
        ma = _calculate_moving_average(values, window_size=3)
        self.assertEqual(len(ma), 3)
        self.assertEqual(ma[0], 2.0)  # (1+2+3)/3
        self.assertEqual(ma[1], 3.0)  # (2+3+4)/3
        self.assertEqual(ma[2], 4.0)  # (3+4+5)/3
    
    def test_find_convergence_point(self):
        """
        測試收斂點查找函數。
        """
        # 測試收斂的序列
        converge_point = _find_convergence_point(self.loss_values, window_size=3, threshold=0.1)
        self.assertIsNotNone(converge_point)
        
        # 測試不收斂的序列
        non_converge_point = _find_convergence_point(self.non_converging_loss, window_size=3, threshold=0.01)
        self.assertIsNone(non_converge_point)


if __name__ == '__main__':
    unittest.main() 