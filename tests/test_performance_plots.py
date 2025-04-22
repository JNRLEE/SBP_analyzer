"""
測試性能視覺化模組的單元測試。

此模組測試visualization/performance_plots.py中的功能。
"""

import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from visualization.performance_plots import PerformancePlotter

class TestPerformancePlotter:
    """測試PerformancePlotter類的各個方法"""
    
    @pytest.fixture
    def plotter(self, tmp_path):
        """創建一個PerformancePlotter實例用於測試"""
        output_dir = tmp_path / "plot_output"
        output_dir.mkdir()
        return PerformancePlotter(str(output_dir))
    
    @pytest.fixture
    def sample_loss_history(self):
        """生成測試用的損失歷史數據"""
        np.random.seed(42)  # 確保結果可重現
        epochs = 50
        
        # 模擬訓練和驗證損失
        train_loss = 1.0 - 0.8 * np.exp(-np.arange(epochs) / 20) + 0.1 * np.random.randn(epochs)
        val_loss = 1.2 - 0.7 * np.exp(-np.arange(epochs) / 25) + 0.2 * np.random.randn(epochs)
        
        return {
            'train': train_loss.tolist(),
            'val': val_loss.tolist()
        }
    
    @pytest.fixture
    def sample_metric_history(self):
        """生成測試用的指標歷史數據"""
        np.random.seed(42)  # 確保結果可重現
        epochs = 50
        
        # 模擬訓練和驗證準確率
        train_acc = 0.7 + 0.25 * (1 - np.exp(-np.arange(epochs) / 15)) + 0.05 * np.random.randn(epochs)
        train_acc = np.clip(train_acc, 0, 1).tolist()
        
        val_acc = 0.6 + 0.3 * (1 - np.exp(-np.arange(epochs) / 20)) + 0.1 * np.random.randn(epochs)
        val_acc = np.clip(val_acc, 0, 1).tolist()
        
        # 模擬F1分數
        train_f1 = 0.65 + 0.3 * (1 - np.exp(-np.arange(epochs) / 18)) + 0.05 * np.random.randn(epochs)
        train_f1 = np.clip(train_f1, 0, 1).tolist()
        
        val_f1 = 0.55 + 0.35 * (1 - np.exp(-np.arange(epochs) / 22)) + 0.1 * np.random.randn(epochs)
        val_f1 = np.clip(val_f1, 0, 1).tolist()
        
        return {
            'train': {'accuracy': train_acc, 'f1': train_f1},
            'val': {'accuracy': val_acc, 'f1': val_f1}
        }
    
    @pytest.fixture
    def sample_lr_history(self):
        """生成測試用的學習率歷史數據"""
        epochs = 50
        # 模擬學習率遞減
        return [0.01 * (0.1 ** (e // 20)) for e in range(epochs)]
    
    def test_plot_loss_curve(self, plotter, sample_loss_history):
        """測試plot_loss_curve方法能否正確繪製損失曲線"""
        fig = plotter.plot_loss_curve(
            sample_loss_history,
            title="Test Loss Curve",
            filename="test_loss.png",
            show=False
        )
        assert isinstance(fig, Figure)
        
        # 檢查是否有正確保存
        output_path = os.path.join(plotter.output_dir, "test_loss.png")
        assert os.path.exists(output_path)
        
        plt.close(fig)
    
    def test_plot_metric_curves(self, plotter, sample_metric_history):
        """測試plot_metric_curves方法能否正確繪製多個指標曲線"""
        # 測試子圖模式
        fig1 = plotter.plot_metric_curves(
            sample_metric_history,
            title="Test Metrics (Subplots)",
            filename="test_metrics_subplots.png",
            subplots=True,
            show=False
        )
        assert isinstance(fig1, Figure)
        
        # 測試共享圖模式
        fig2 = plotter.plot_metric_curves(
            sample_metric_history,
            title="Test Metrics (Single Plot)",
            filename="test_metrics_single.png",
            subplots=False,
            show=False
        )
        assert isinstance(fig2, Figure)
        
        # 測試單個指標
        fig3 = plotter.plot_metric_curves(
            sample_metric_history,
            title="Test Accuracy Only",
            filename="test_accuracy.png",
            metric_names=['accuracy'],
            show=False
        )
        assert isinstance(fig3, Figure)
        
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
    
    def test_plot_learning_rate(self, plotter, sample_lr_history):
        """測試plot_learning_rate方法能否正確繪製學習率曲線"""
        fig = plotter.plot_learning_rate(
            sample_lr_history,
            title="Test Learning Rate",
            filename="test_lr.png",
            show=False
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_convergence_analysis(self, plotter, sample_loss_history):
        """測試plot_convergence_analysis方法能否正確分析收斂情況"""
        # 假設第20輪收斂
        convergence_epoch = 20
        fig = plotter.plot_convergence_analysis(
            sample_loss_history['train'],
            convergence_epoch,
            title="Test Convergence Analysis",
            filename="test_convergence.png",
            show=False
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_training_stability(self, plotter, sample_loss_history):
        """測試plot_training_stability方法能否正確分析訓練穩定性"""
        fig = plotter.plot_training_stability(
            sample_loss_history['train'],
            moving_avg_window=5,
            title="Test Training Stability",
            filename="test_stability.png",
            show=False
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_gradient_norm(self, plotter):
        """測試plot_gradient_norm方法能否正確繪製梯度範數曲線"""
        # 模擬梯度範數
        np.random.seed(42)
        epochs = 50
        gradient_norms = 10.0 * np.exp(-np.arange(epochs) / 10) + np.random.rand(epochs)
        
        # 在中間添加一些異常值
        gradient_norms[15] = 15.0  # 異常值
        gradient_norms[30] = 12.0  # 異常值
        
        fig = plotter.plot_gradient_norm(
            gradient_norms.tolist(),
            title="Test Gradient Norm",
            filename="test_gradient.png",
            show=False
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_plot_training_heatmap(self, plotter):
        """測試plot_training_heatmap方法能否正確繪製訓練熱力圖"""
        # 模擬每個批次在每個輪次的損失
        np.random.seed(42)
        epochs = 10
        batches = 20
        
        # 創建一個逐漸減小的損失矩陣，帶有一些隨機性
        epoch_batch_data = np.zeros((epochs, batches))
        for e in range(epochs):
            for b in range(batches):
                base_loss = 1.0 - 0.5 * (e / epochs) - 0.3 * (b / batches)
                noise = 0.1 * np.random.randn()
                epoch_batch_data[e, b] = max(0.1, base_loss + noise)
        
        fig = plotter.plot_training_heatmap(
            epoch_batch_data,
            title="Test Training Heatmap",
            filename="test_heatmap.png",
            show=False
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_error_handling(self, plotter):
        """測試各種錯誤情況的處理"""
        # 測試空數據
        with pytest.raises(Exception):
            plotter.plot_loss_curve({})
        
        # 測試不兼容的數據類型
        with pytest.raises(Exception):
            plotter.plot_training_heatmap("not_an_array")
        
        # 測試損失歷史中的無效值
        invalid_loss = {'train': [1, 2, 'invalid', 4, 5]}
        with pytest.raises(Exception):
            plotter.plot_loss_curve(invalid_loss) 