"""
測試指標計算模組的單元測試。
"""

import unittest
import numpy as np
import torch
from pathlib import Path

from metrics.distribution_metrics import (
    calculate_kl_divergence, calculate_js_divergence, 
    calculate_wasserstein_distance, calculate_histogram_intersection,
    calculate_distribution_similarity, compare_tensor_distributions
)
from metrics.performance_metrics import (
    analyze_loss_curve, analyze_metric_curve,
    calculate_convergence_metrics, analyze_training_efficiency
)

class TestDistributionMetrics(unittest.TestCase):
    """
    測試分布指標模組。
    """

    def test_calculate_kl_divergence(self):
        """
        測試計算KL散度。
        """
        # 創建兩個簡單的分布
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        
        # 測試計算KL散度
        kl = calculate_kl_divergence(p, q)
        self.assertGreaterEqual(kl, 0.0, "KL散度應該大於等於0")
        
        # 測試相同分布的KL散度應接近0
        kl_same = calculate_kl_divergence(p, p)
        self.assertAlmostEqual(kl_same, 0.0, places=5, msg="相同分布的KL散度應接近0")

    def test_calculate_js_divergence(self):
        """
        測試計算JS散度。
        """
        # 創建兩個簡單的分布
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        
        # 測試計算JS散度
        js = calculate_js_divergence(p, q)
        self.assertGreaterEqual(js, 0.0, "JS散度應該大於等於0")
        self.assertLessEqual(js, 1.0, "JS散度應該小於等於1")
        
        # 測試相同分布的JS散度應為0
        js_same = calculate_js_divergence(p, p)
        self.assertAlmostEqual(js_same, 0.0, places=5, msg="相同分布的JS散度應為0")
        
        # 測試JS散度的對稱性
        js_pq = calculate_js_divergence(p, q)
        js_qp = calculate_js_divergence(q, p)
        self.assertAlmostEqual(js_pq, js_qp, places=5, msg="JS散度應該是對稱的")

    def test_calculate_wasserstein_distance(self):
        """
        測試計算Wasserstein距離。
        """
        # 創建兩組樣本
        p_values = np.array([1, 2, 3, 4, 5])
        q_values = np.array([2, 3, 4, 5, 6])
        
        # 測試計算Wasserstein距離
        wd = calculate_wasserstein_distance(p_values, q_values)
        self.assertGreaterEqual(wd, 0.0, "Wasserstein距離應該大於等於0")
        
        # 測試相同分布的Wasserstein距離應為0
        wd_same = calculate_wasserstein_distance(p_values, p_values)
        self.assertAlmostEqual(wd_same, 0.0, places=5, msg="相同分布的Wasserstein距離應為0")

    def test_calculate_histogram_intersection(self):
        """
        測試計算直方圖交集。
        """
        # 創建兩個直方圖
        hist1 = np.array([0.1, 0.2, 0.3, 0.4])
        hist2 = np.array([0.2, 0.3, 0.4, 0.1])
        
        # 測試計算交集
        intersection = calculate_histogram_intersection(hist1, hist2)
        self.assertGreaterEqual(intersection, 0.0, "交集應該大於等於0")
        self.assertLessEqual(intersection, 1.0, "交集應該小於等於1")
        
        # 測試相同直方圖的交集應為1
        intersection_same = calculate_histogram_intersection(hist1, hist1)
        self.assertAlmostEqual(intersection_same, 1.0, places=5, msg="相同直方圖的交集應為1")

    def test_calculate_distribution_similarity(self):
        """
        測試計算分布相似度。
        """
        # 創建兩組樣本
        dist1 = np.array([1, 2, 3, 4, 5])
        dist2 = np.array([2, 3, 4, 5, 6])
        
        # 測試計算相似度
        similarity = calculate_distribution_similarity(dist1, dist2)
        self.assertIsInstance(similarity, dict, "應該返回字典")
        self.assertIn('histogram_intersection', similarity, "應該包含histogram_intersection")
        
        # 檢查直方圖交集值是否在0到1之間
        self.assertGreaterEqual(similarity['histogram_intersection'], 0.0, "交集應該大於等於0")
        self.assertLessEqual(similarity['histogram_intersection'], 1.0, "交集應該小於等於1")

    def test_compare_tensor_distributions(self):
        """
        測試比較張量分布。
        """
        # 創建兩個張量
        tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        tensor2 = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
        
        # 測試比較分布
        results = compare_tensor_distributions(tensor1, tensor2)
        self.assertIsInstance(results, dict, "應該返回字典")
        self.assertIn('histogram_intersection', results, "應該包含histogram_intersection")
        
        # 測試flatten參數
        results_no_flatten = compare_tensor_distributions(tensor1, tensor2, flatten=False)
        self.assertIsInstance(results_no_flatten, dict, "應該返回字典")

class TestPerformanceMetrics(unittest.TestCase):
    """
    測試性能指標模組。
    """

    def test_analyze_loss_curve(self):
        """
        測試分析損失曲線。
        """
        # 創建一個測試損失序列
        loss_values = [10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        
        # 測試分析損失曲線
        analysis = analyze_loss_curve(loss_values)
        self.assertIsInstance(analysis, dict, "應該返回字典")
        self.assertIn('min_loss', analysis, "應該包含min_loss")
        self.assertIn('min_loss_epoch', analysis, "應該包含min_loss_epoch")
        self.assertIn('improvement_ratio', analysis, "應該包含improvement_ratio")
        
        # 檢查最小損失值和對應的輪次
        self.assertEqual(analysis['min_loss'], 1.0, "最小損失應該是1.0")
        self.assertEqual(analysis['min_loss_epoch'], 7, "最小損失所在輪次應該是7")

    def test_analyze_metric_curve(self):
        """
        測試分析評估指標曲線。
        """
        # 創建一個測試指標序列
        metric_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        # 測試分析指標曲線 (假設較高的值更好)
        analysis = analyze_metric_curve(metric_values, higher_is_better=True)
        self.assertIsInstance(analysis, dict, "應該返回字典")
        self.assertIn('best_value', analysis, "應該包含best_value")
        self.assertIn('best_value_epoch', analysis, "應該包含best_value_epoch")
        
        # 檢查最佳值和對應的輪次
        self.assertEqual(analysis['best_value'], 0.8, "最佳值應該是0.8")
        self.assertEqual(analysis['best_value_epoch'], 7, "最佳值所在輪次應該是7")
        
        # 測試較低的值更好的情況
        analysis_lower = analyze_metric_curve(metric_values, higher_is_better=False)
        self.assertEqual(analysis_lower['best_value'], 0.1, "最佳值應該是0.1")
        self.assertEqual(analysis_lower['best_value_epoch'], 0, "最佳值所在輪次應該是0")

    def test_calculate_convergence_metrics(self):
        """
        測試計算收斂指標。
        """
        # 創建訓練損失和驗證損失序列
        train_loss = [10.0, 8.0, 6.0, 4.0, 3.0, 2.5, 2.2, 2.1, 2.0, 1.9]
        val_loss = [11.0, 9.0, 7.0, 5.0, 4.0, 3.8, 3.7, 3.7, 3.8, 4.0]
        
        # 測試計算收斂指標
        metrics = calculate_convergence_metrics(train_loss, val_loss, threshold=0.1)
        self.assertIsInstance(metrics, dict, "應該返回字典")
        self.assertIn('train_convergence_epoch', metrics, "應該包含train_convergence_epoch")
        self.assertIn('val_convergence_epoch', metrics, "應該包含val_convergence_epoch")
        
        # 只提供訓練損失的情況
        metrics_train_only = calculate_convergence_metrics(train_loss)
        self.assertIsInstance(metrics_train_only, dict, "應該返回字典")
        self.assertIn('train_convergence_epoch', metrics_train_only, "應該包含train_convergence_epoch")

    def test_analyze_training_efficiency(self):
        """
        測試分析訓練效率。
        """
        # 創建測試數據
        loss_values = [10.0, 8.0, 6.0, 4.0, 3.0, 2.5, 2.2, 2.1, 2.0, 1.9]
        training_time = 100.0  # 假設總訓練時間為100秒
        epoch_times = [12.0, 11.0, 10.0, 10.0, 10.0, 9.5, 9.5, 9.0, 9.0, 10.0]  # 每輪訓練時間
        
        # 測試分析訓練效率
        efficiency = analyze_training_efficiency(loss_values, training_time, epoch_times)
        self.assertIsInstance(efficiency, dict, "應該返回字典")
        self.assertIn('total_training_time', efficiency, "應該包含total_training_time")
        self.assertIn('time_per_epoch', efficiency, "應該包含time_per_epoch")
        self.assertIn('mean_epoch_time', efficiency, "應該包含mean_epoch_time")
        
        # 測試只提供損失值的情況
        efficiency_loss_only = analyze_training_efficiency(loss_values)
        self.assertIsInstance(efficiency_loss_only, dict, "應該返回字典")

if __name__ == '__main__':
    unittest.main() 