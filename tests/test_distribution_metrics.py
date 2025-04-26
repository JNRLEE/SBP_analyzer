"""
分布度量功能的測試模組。

測試日期: 2023-05-22
測試目的: 驗證分布相關統計函數的正確性與功能
"""

import unittest
import torch
import numpy as np
from metrics.distribution_metrics import (
    calculate_histogram_intersection,
    calculate_distribution_similarity,
    compare_tensor_distributions,
    calculate_distribution_stats,
    calculate_distribution_entropy
)


class TestDistributionMetrics(unittest.TestCase):
    """測試分布度量功能的類。"""
    
    def setUp(self):
        """設置測試環境，創建測試數據。"""
        # 建立兩個相似但不同的正態分布
        np.random.seed(42)
        torch.manual_seed(42)
        
        # NumPy 數組用於測試
        self.dist1_np = np.random.normal(0, 1, 1000)
        self.dist2_np = np.random.normal(0.2, 1.1, 1000)
        
        # PyTorch 張量用於測試
        self.dist1_torch = torch.randn(1000)
        self.dist2_torch = torch.randn(1000) * 1.1 + 0.2
        
        # 多維張量
        self.tensor1 = torch.randn(10, 20, 30)
        self.tensor2 = torch.randn(10, 20, 30) * 1.1 + 0.1

    def test_histogram_intersection(self):
        """測試直方圖交叉計算功能。"""
        intersection = calculate_histogram_intersection(self.dist1_np, self.dist2_np)
        self.assertGreaterEqual(intersection, 0.0)
        self.assertLessEqual(intersection, 1.0)
        
        # 相同分布應有高交叉率
        same_dist_intersection = calculate_histogram_intersection(self.dist1_np, self.dist1_np)
        self.assertAlmostEqual(same_dist_intersection, 1.0, delta=0.01)

    def test_distribution_similarity(self):
        """測試分布相似度計算功能。"""
        # 測試不同相似度度量方法
        for method in ['kl_divergence', 'js_divergence', 'wasserstein', 'correlation']:
            similarity = calculate_distribution_similarity(
                self.dist1_np, self.dist2_np, method=method
            )
            self.assertIsNotNone(similarity)
            
            # 相同分布的相似度應該接近理想值
            same_dist_similarity = calculate_distribution_similarity(
                self.dist1_np, self.dist1_np, method=method
            )
            if method in ['kl_divergence', 'js_divergence', 'wasserstein']:
                self.assertAlmostEqual(same_dist_similarity, 0.0, delta=0.01)
            elif method == 'correlation':
                self.assertAlmostEqual(same_dist_similarity, 1.0, delta=0.01)
    
    def test_compare_tensor_distributions(self):
        """測試張量分布比較功能。"""
        comparison_results = compare_tensor_distributions(self.tensor1, self.tensor2)
        
        # 檢查結果包含預期的度量
        self.assertIn('histogram_intersection', comparison_results)
        self.assertIn('kl_divergence', comparison_results)
        self.assertIn('wasserstein', comparison_results)
        
        # 測試指定維度
        dim = 1
        dim_results = compare_tensor_distributions(self.tensor1, self.tensor2, dim=dim)
        self.assertEqual(len(dim_results['histogram_intersection']), self.tensor1.size(dim))

    def test_distribution_stats(self):
        """測試分布統計計算功能。"""
        stats = calculate_distribution_stats(self.dist1_np)
        
        # 檢查返回的統計量
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('median', stats)
        self.assertIn('skewness', stats)
        self.assertIn('kurtosis', stats)
        
        # 簡單驗證
        self.assertAlmostEqual(stats['mean'], np.mean(self.dist1_np), delta=0.01)
        self.assertAlmostEqual(stats['std'], np.std(self.dist1_np), delta=0.01)

    def test_distribution_entropy(self):
        """測試分布熵計算功能。"""
        entropy = calculate_distribution_entropy(self.dist1_np)
        self.assertGreater(entropy, 0)
        
        # 確定熵在合理範圍內
        max_entropy = np.log2(len(self.dist1_np))
        self.assertLess(entropy, max_entropy)
        
        # 單一值的分布應該有零熵
        single_value_dist = np.ones(100)
        single_value_entropy = calculate_distribution_entropy(single_value_dist)
        self.assertAlmostEqual(single_value_entropy, 0, delta=0.01)


if __name__ == '__main__':
    unittest.main() 