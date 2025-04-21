"""
測試訓練動態分析功能。

此模組包含測試TrainingDynamicsAnalyzer類別功能的相關測試案例。
"""

import os
import sys
import unittest
import shutil
import tempfile
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加專案根目錄至路徑，以便引入模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analyzer.training_dynamics_analyzer import TrainingDynamicsAnalyzer

class TestTrainingDynamicsAnalyzer(unittest.TestCase):
    """
    測試TrainingDynamicsAnalyzer類別的功能。
    """
    
    def setUp(self):
        """
        在每個測試案例前設置測試環境。
        """
        # 創建臨時目錄作為測試實驗目錄
        self.test_dir = tempfile.mkdtemp()
        
        # 創建虛擬的訓練歷史數據
        self.train_loss = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.9, 0.8, 0.7, 0.6]
        self.val_loss = [12.0, 9.0, 7.0, 5.0, 3.0, 2.5, 2.2, 2.1, 2.0, 1.9]
        self.accuracy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
        
        # 模擬訓練歷史JSON文件
        self.training_history = {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'accuracy': self.accuracy,
            'training_time': 100.0,
            'best_val_loss': 1.9
        }
        
        # 將訓練歷史保存為JSON文件
        with open(os.path.join(self.test_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f)
        
        # 初始化分析器
        self.analyzer = TrainingDynamicsAnalyzer(self.test_dir)
    
    def tearDown(self):
        """
        在每個測試案例後清理測試環境。
        """
        # 清理臨時目錄
        shutil.rmtree(self.test_dir)
    
    def test_load_training_history(self):
        """
        測試載入訓練歷史功能。
        """
        # 調用載入方法
        loaded_history = self.analyzer.load_training_history()
        
        # 驗證載入的數據
        self.assertEqual(loaded_history['train_loss'], self.train_loss)
        self.assertEqual(loaded_history['val_loss'], self.val_loss)
        self.assertEqual(loaded_history['accuracy'], self.accuracy)
        self.assertEqual(loaded_history['training_time'], 100.0)
        self.assertEqual(loaded_history['best_val_loss'], 1.9)
    
    def test_preprocess_metrics(self):
        """
        測試指標數據預處理功能。
        """
        # 調用預處理方法
        metrics_df = self.analyzer.preprocess_metrics()
        
        # 驗證預處理結果
        self.assertIsInstance(metrics_df, pd.DataFrame)
        self.assertEqual(len(metrics_df), 10)  # 10個輪次
        
        # 檢查列名
        expected_columns = {'train_loss', 'val_loss', 'accuracy'}
        self.assertTrue(expected_columns.issubset(set(metrics_df.columns)))
        
        # 檢查數據內容
        np.testing.assert_array_equal(metrics_df['train_loss'].values, self.train_loss)
        np.testing.assert_array_equal(metrics_df['val_loss'].values, self.val_loss)
        np.testing.assert_array_equal(metrics_df['accuracy'].values, self.accuracy)
    
    def test_analyze_convergence(self):
        """
        測試收斂性分析功能。
        """
        # 先執行預處理
        self.analyzer.preprocess_metrics()
        
        # 調用收斂性分析方法
        convergence_results = self.analyzer.analyze_convergence('train_loss')
        
        # 驗證收斂性分析結果
        self.assertIn('has_converged', convergence_results)
        self.assertIn('convergence_point', convergence_results)
        self.assertIn('min_value', convergence_results)
        self.assertIn('min_value_epoch', convergence_results)
        
        # 驗證最小值
        self.assertEqual(convergence_results['min_value'], 0.6)
        self.assertEqual(convergence_results['min_value_epoch'], 9)
        
        # 驗證收斂點（這裡應該是6，根據find_convergence_point的實現）
        self.assertEqual(convergence_results['convergence_point'], 6)
    
    def test_analyze_stability(self):
        """
        測試穩定性分析功能。
        """
        # 先執行預處理
        self.analyzer.preprocess_metrics()
        
        # 調用穩定性分析方法
        stability_results = self.analyzer.analyze_stability('train_loss')
        
        # 驗證穩定性分析結果
        self.assertIn('variance', stability_results)
        self.assertIn('mean', stability_results)
        self.assertIn('std', stability_results)
        self.assertIn('max_fluctuation', stability_results)
        self.assertIn('moving_average', stability_results)
        
        # 驗證基本統計量
        self.assertAlmostEqual(stability_results['mean'], np.mean(self.train_loss))
        self.assertAlmostEqual(stability_results['std'], np.std(self.train_loss))
        self.assertAlmostEqual(stability_results['variance'], np.var(self.train_loss))
        
        # 驗證最大波動
        expected_max_fluctuation = max([abs(self.train_loss[i] - self.train_loss[i-1]) for i in range(1, len(self.train_loss))])
        self.assertEqual(stability_results['max_fluctuation'], expected_max_fluctuation)
    
    def test_get_metric_correlation(self):
        """
        測試指標相關性分析功能。
        """
        # 先執行預處理
        self.analyzer.preprocess_metrics()
        
        # 調用指標相關性分析方法
        correlation_results = self.analyzer.get_metric_correlation()
        
        # 驗證相關性分析結果
        self.assertIsInstance(correlation_results, pd.DataFrame)
        self.assertGreater(len(correlation_results), 0)
        
        # 驗證列名
        expected_columns = {'metric1', 'metric2', 'correlation', 'strength', 'direction'}
        self.assertTrue(expected_columns.issubset(set(correlation_results.columns)))
        
        # 驗證train_loss和val_loss之間應該有很強的正相關
        train_val_correlation = correlation_results[
            (correlation_results['metric1'] == 'train_loss') & (correlation_results['metric2'] == 'val_loss') |
            (correlation_results['metric1'] == 'val_loss') & (correlation_results['metric2'] == 'train_loss')
        ]
        
        if not train_val_correlation.empty:
            self.assertGreater(train_val_correlation.iloc[0]['correlation'], 0.9)
    
    def test_analyze(self):
        """
        測試完整分析功能。
        """
        # 執行完整分析
        results = self.analyzer.analyze(save_results=False)
        
        # 驗證分析結果的主要部分
        self.assertIn('convergence_analysis', results)
        self.assertIn('stability_analysis', results)
        self.assertIn('trend_analysis', results)
        self.assertIn('metric_correlations', results)
        self.assertIn('training_efficiency', results)
        self.assertIn('anomaly_detection', results)
        self.assertIn('training_summary', results)
        
        # 檢查訓練摘要
        summary = results['training_summary']
        self.assertEqual(summary['total_epochs'], 10)
        self.assertEqual(summary['training_time'], 100.0)
        self.assertEqual(summary['best_val_loss'], 1.9)
        
        # 檢查收斂性分析
        conv_analysis = results['convergence_analysis']
        self.assertIn('train_loss', conv_analysis)
        self.assertIn('val_loss', conv_analysis)
        
        # 檢查趨勢分析
        trend_analysis = results['trend_analysis']
        self.assertIn('train_loss', trend_analysis)
        self.assertIn('val_loss', trend_analysis)
        self.assertIn('accuracy', trend_analysis)
    
    def test_visualize_metrics(self):
        """
        測試指標可視化功能。
        """
        # 先執行預處理
        self.analyzer.preprocess_metrics()
        
        # 調用可視化方法
        figure = self.analyzer.visualize_metrics()
        
        # 驗證是否返回了Matplotlib圖表
        self.assertIsInstance(figure, plt.Figure)
        
        # 檢查figure中是否包含正確數量的子圖
        self.assertGreaterEqual(len(figure.axes), 1)
        
        # 關閉圖表避免顯示
        plt.close(figure)
        
        # 測試僅可視化部分指標
        figure = self.analyzer.visualize_metrics(metrics=['train_loss', 'val_loss'])
        self.assertIsInstance(figure, plt.Figure)
        plt.close(figure)

if __name__ == '__main__':
    unittest.main() 