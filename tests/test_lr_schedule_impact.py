"""
測試學習率調度影響分析和可視化功能。

此模組測試 metrics/performance_metrics.py 中的 analyze_lr_schedule_impact 函數
以及 visualization/performance_plots.py 中的 plot_lr_schedule_impact 函數。
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from pathlib import Path

# 導入需要測試的功能
from metrics.performance_metrics import analyze_lr_schedule_impact
from visualization.performance_plots import plot_lr_schedule_impact
from utils.stat_utils import find_convergence_point

class TestLRScheduleImpact(unittest.TestCase):
    """測試學習率調度對訓練效果影響的分析和可視化功能。"""
    
    def setUp(self):
        """設置測試環境和模擬數據。"""
        # 設置臨時目錄用於存儲測試生成的圖像
        self.test_dir = tempfile.mkdtemp()
        
        # 創建模擬訓練數據
        # 模擬不同學習率調度類型的訓練過程
        self.runs = {
            # 常數學習率
            'Constant LR': {
                'train_loss': [8.0, 6.0, 4.5, 3.5, 3.0, 2.8, 2.7, 2.6, 2.55, 2.5],
                'val_loss': [8.5, 6.5, 5.0, 4.0, 3.5, 3.3, 3.25, 3.2, 3.15, 3.1],
                'learning_rate': [0.01] * 10
            },
            # 階梯式學習率
            'Step LR': {
                'train_loss': [8.0, 5.0, 3.0, 2.0, 1.8, 1.5, 1.3, 1.25, 1.2, 1.18],
                'val_loss': [8.5, 5.5, 3.5, 2.5, 2.3, 2.0, 1.8, 1.7, 1.65, 1.6],
                'learning_rate': [0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001]
            },
            # 循環式學習率
            'Cyclic LR': {
                'train_loss': [8.0, 5.5, 6.0, 4.0, 4.5, 3.2, 3.5, 2.8, 3.0, 2.5],
                'val_loss': [8.5, 6.0, 6.5, 4.5, 5.0, 3.7, 4.0, 3.3, 3.5, 3.0],
                'learning_rate': [0.001, 0.01, 0.1, 0.01, 0.001, 0.01, 0.1, 0.01, 0.001, 0.0001]
            },
            # 自定義學習率調度
            'Custom LR': {
                'train_loss': [8.0, 6.5, 5.0, 4.0, 3.2, 2.5, 2.0, 1.8, 1.5, 1.4],
                'val_loss': [8.5, 7.0, 5.5, 4.5, 3.7, 3.0, 2.5, 2.3, 2.0, 1.9],
                'learning_rate': [0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.008, 0.005, 0.003]
            }
        }
        
    def tearDown(self):
        """清理測試環境。"""
        # 刪除測試期間創建的圖像
        for file in Path(self.test_dir).glob('*.png'):
            file.unlink()
        os.rmdir(self.test_dir)
    
    def test_analyze_lr_schedule_detection(self):
        """測試學習率調度類型的自動檢測。"""
        # 分析學習率調度
        result = analyze_lr_schedule_impact(self.runs)
        
        # 檢查是否成功檢測各種調度類型
        self.assertEqual(result['lr_schedules']['Constant LR']['type'], 'constant')
        self.assertEqual(result['lr_schedules']['Step LR']['type'], 'step')
        self.assertEqual(result['lr_schedules']['Cyclic LR']['type'], 'cyclic')
        self.assertEqual(result['lr_schedules']['Custom LR']['type'], 'custom')
        
        # 檢查是否包含必要的分析結果
        for run_name in self.runs.keys():
            self.assertIn(run_name, result['lr_schedules'])
            self.assertIn('initial_lr', result['lr_schedules'][run_name])
            self.assertIn('final_lr', result['lr_schedules'][run_name])
            self.assertIn('lr_loss_correlation', result['lr_schedules'][run_name])
    
    def test_analyze_lr_correlation(self):
        """測試學習率與損失相關性分析。"""
        result = analyze_lr_schedule_impact(self.runs)
        
        # 檢查相關性計算
        # 預期常數學習率的相關性為0（因為所有學習率值相同）
        self.assertEqual(result['lr_schedules']['Constant LR']['lr_loss_correlation'], 0)
        
        # 其他調度應有非零相關性
        self.assertNotEqual(result['lr_schedules']['Step LR']['lr_loss_correlation'], 0)
        self.assertNotEqual(result['lr_schedules']['Cyclic LR']['lr_loss_correlation'], 0)
        self.assertNotEqual(result['lr_schedules']['Custom LR']['lr_loss_correlation'], 0)
    
    def test_compare_performance(self):
        """測試不同學習率調度的性能比較。"""
        result = analyze_lr_schedule_impact(self.runs)
        
        # 檢查是否正確識別最佳性能的運行
        self.assertIn('best_performing_run', result['comparative_analysis'])
        
        # 檢查最低損失值是否正確計算
        min_loss_values = result['comparative_analysis']['min_loss_by_run']
        for run_name, run_data in self.runs.items():
            expected_min = min(run_data['train_loss'])
            self.assertEqual(min_loss_values[run_name], expected_min)
        
        # 檢查收斂點檢測
        convergence_epochs = result['comparative_analysis'].get('convergence_epochs', {})
        for run_name, epoch in convergence_epochs.items():
            # 確保收斂點是合理的整數
            self.assertIsInstance(epoch, int)
            self.assertGreaterEqual(epoch, 0)
            self.assertLess(epoch, len(self.runs[run_name]['train_loss']))
    
    def test_plot_lr_schedule_impact(self):
        """測試學習率調度影響可視化功能。"""
        # 分析學習率調度
        result = analyze_lr_schedule_impact(self.runs)
        
        # 生成可視化圖表
        fig = plot_lr_schedule_impact(result)
        
        # 檢查是否返回了有效的Figure對象
        self.assertIsInstance(fig, plt.Figure)
        
        # 保存圖表到臨時目錄
        filepath = os.path.join(self.test_dir, 'lr_schedule_impact.png')
        fig.savefig(filepath)
        
        # 檢查文件是否成功創建
        self.assertTrue(os.path.exists(filepath))
        
        # 關閉圖表，避免顯示
        plt.close(fig)
    
    def test_with_missing_data(self):
        """測試處理缺失數據的情況。"""
        # 創建缺少必要字段的模擬數據
        invalid_runs = {
            'Missing Loss': {
                'learning_rate': [0.01, 0.01, 0.01]
                # 沒有train_loss或loss字段
            },
            'Missing LR': {
                'train_loss': [8.0, 6.0, 4.0]
                # 沒有learning_rate字段
            },
            'Valid Run': {
                'train_loss': [8.0, 6.0, 4.0],
                'learning_rate': [0.01, 0.01, 0.01]
            }
        }
        
        # 分析結果應只包含有效的運行
        result = analyze_lr_schedule_impact(invalid_runs)
        
        # 檢查結果
        self.assertIn('lr_schedules', result)
        self.assertNotIn('Missing Loss', result['lr_schedules'])
        self.assertNotIn('Missing LR', result['lr_schedules'])
        self.assertIn('Valid Run', result['lr_schedules'])
    
    def test_recommendations_generation(self):
        """測試是否生成學習率調度建議。"""
        result = analyze_lr_schedule_impact(self.runs)
        
        # 檢查建議是否存在
        self.assertIn('recommendations', result)
        
        # 修改數據以使一個運行明顯優於其他運行
        modified_runs = self.runs.copy()
        modified_runs['Step LR']['train_loss'] = [8.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        
        # 再次分析
        enhanced_result = analyze_lr_schedule_impact(modified_runs)
        
        # 檢查建議是否反映了Step LR的優越性
        self.assertIn('recommendations', enhanced_result)
        # 注意：這裡不檢查具體建議內容，因為它可能基於多種因素

    def test_plot_with_english_labels(self):
        """測試使用英文標籤模式的學習率調度影響可視化功能。"""
        # 分析學習率調度
        result = analyze_lr_schedule_impact(self.runs)
        
        # 使用英文標籤生成可視化圖表
        fig = plot_lr_schedule_impact(result, use_english=True)
        
        # 檢查是否返回了有效的Figure對象
        self.assertIsInstance(fig, plt.Figure)
        
        # 保存圖表到臨時目錄
        filepath = os.path.join(self.test_dir, 'lr_schedule_impact_english.png')
        fig.savefig(filepath)
        
        # 檢查文件是否成功創建
        self.assertTrue(os.path.exists(filepath))
        
        # 關閉圖表，避免顯示
        plt.close(fig)

if __name__ == '__main__':
    unittest.main() 