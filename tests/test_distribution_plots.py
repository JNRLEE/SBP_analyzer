"""
測試分佈視覺化模組的單元測試。

此模組測試visualization/distribution_plots.py中的功能。
"""

import os
import numpy as np
import torch
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from visualization.distribution_plots import DistributionPlotter

class TestDistributionPlotter:
    """測試DistributionPlotter類的各個方法"""
    
    @pytest.fixture
    def plotter(self, tmp_path):
        """創建一個DistributionPlotter實例用於測試"""
        output_dir = tmp_path / "plot_output"
        output_dir.mkdir()
        return DistributionPlotter(str(output_dir))
    
    @pytest.fixture
    def sample_data(self):
        """生成測試用的數據樣本"""
        # 使用正態分佈數據
        np.random.seed(42)  # 確保結果可重現
        return np.random.normal(0, 1, 1000)
    
    @pytest.fixture
    def sample_data_dict(self):
        """生成測試用的多個分佈數據"""
        np.random.seed(42)  # 確保結果可重現
        return {
            'normal': np.random.normal(0, 1, 1000),
            'uniform': np.random.uniform(-1, 1, 1000),
            'exponential': np.random.exponential(1, 1000)
        }
    
    @pytest.fixture
    def sample_tensor(self):
        """生成測試用的PyTorch張量"""
        torch.manual_seed(42)  # 確保結果可重現
        return torch.randn(10, 10)
    
    def test_plot_histogram(self, plotter, sample_data):
        """測試plot_histogram方法能否正確繪製直方圖"""
        # 使用numpy數組繪圖
        fig = plotter.plot_histogram(sample_data, title="Test Histogram", show=False)
        assert isinstance(fig, Figure)
        
        # 檢查是否有正確保存
        output_path = os.path.join(plotter.output_dir, "Test Histogram.png")
        assert os.path.exists(output_path)
        
        plt.close(fig)
    
    def test_plot_histogram_with_tensor(self, plotter, sample_tensor):
        """測試plot_histogram方法能否處理PyTorch張量"""
        fig = plotter.plot_histogram(sample_tensor, title="Tensor Histogram", show=False)
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_distribution_comparison(self, plotter, sample_data_dict):
        """測試plot_distribution_comparison方法能否比較多個分佈"""
        fig = plotter.plot_distribution_comparison(
            sample_data_dict,
            title="Distribution Comparison",
            show=False
        )
        assert isinstance(fig, Figure)
        
        # 檢查是否有正確保存
        output_path = os.path.join(plotter.output_dir, "Distribution Comparison.png")
        assert os.path.exists(output_path)
        
        plt.close(fig)
    
    def test_plot_boxplot(self, plotter, sample_data_dict):
        """測試plot_boxplot方法能否繪製箱形圖"""
        fig = plotter.plot_boxplot(
            sample_data_dict,
            title="Boxplot Test",
            show=False
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_heatmap(self, plotter):
        """測試plot_heatmap方法能否繪製熱力圖"""
        # 創建一個2D數組用於熱力圖
        data = np.random.rand(10, 10)
        fig = plotter.plot_heatmap(
            data,
            title="Heatmap Test",
            show=False
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_qq_plot(self, plotter, sample_data):
        """測試plot_qq_plot方法能否繪製Q-Q圖"""
        fig = plotter.plot_qq_plot(
            sample_data,
            title="Q-Q Plot Test",
            show=False
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_with_different_data_types(self, plotter):
        """測試各種方法是否能處理不同類型的數據"""
        # Python列表
        python_list = [1, 2, 3, 4, 5]
        fig1 = plotter.plot_histogram(python_list, title="Python List", show=False)
        assert isinstance(fig1, Figure)
        plt.close(fig1)
        
        # NumPy數組
        numpy_array = np.array([1, 2, 3, 4, 5])
        fig2 = plotter.plot_histogram(numpy_array, title="NumPy Array", show=False)
        assert isinstance(fig2, Figure)
        plt.close(fig2)
        
        # PyTorch張量
        torch_tensor = torch.tensor([1, 2, 3, 4, 5.0])
        fig3 = plotter.plot_histogram(torch_tensor, title="PyTorch Tensor", show=False)
        assert isinstance(fig3, Figure)
        plt.close(fig3) 