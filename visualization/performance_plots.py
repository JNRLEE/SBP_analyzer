"""
性能視覺化模組。

此模組提供用於視覺化模型訓練過程中的性能指標和訓練動態的功能。

Classes:
    PerformancePlotter: 用於繪製性能相關圖表的類。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
import torch

from .plotter import BasePlotter


class PerformancePlotter(BasePlotter):
    """
    性能視覺化器，用於繪製訓練過程中的性能指標和動態。
    
    此類提供多種方法來視覺化訓練過程中的指標變化，如損失曲線、學習率變化等。
    
    Attributes:
        output_dir (str): 圖表保存的輸出目錄。
        style (str): Matplotlib 繪圖風格。
        figsize (Tuple[int, int]): 圖表大小。
        dpi (int): 圖表 DPI。
    """
    
    def __init__(self, output_dir: str = None, style: str = 'seaborn-whitegrid', 
                 figsize: Tuple[int, int] = (10, 6), dpi: int = 100):
        """
        初始化性能視覺化器。
        
        Args:
            output_dir (str, optional): 圖表保存的輸出目錄。若為 None，則不保存圖表。
            style (str, optional): Matplotlib 繪圖風格。默認為 'seaborn-whitegrid'。
            figsize (Tuple[int, int], optional): 圖表大小。默認為 (10, 6)。
            dpi (int, optional): 圖表 DPI。默認為 100。
        """
        super().__init__(output_dir, style, figsize, dpi)
    
    def plot_loss_curve(self, loss_history: Dict[str, List[float]], 
                        title: str = 'Training and Validation Loss',
                        xlabel: str = 'Epoch',
                        ylabel: str = 'Loss',
                        log_scale: bool = False,
                        mark_min: bool = True,
                        filename: str = None,
                        show: bool = True) -> plt.Figure:
        """
        繪製訓練和驗證損失曲線。
        
        Args:
            loss_history (Dict[str, List[float]]): 損失歷史記錄，例如 {'train': [...], 'val': [...]}。
            title (str, optional): 圖表標題。默認為 'Training and Validation Loss'。
            xlabel (str, optional): X 軸標籤。默認為 'Epoch'。
            ylabel (str, optional): Y 軸標籤。默認為 'Loss'。
            log_scale (bool, optional): 是否使用對數尺度。默認為 False。
            mark_min (bool, optional): 是否標記最小值。默認為 True。
            filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
            show (bool, optional): 是否顯示圖表。默認為 True。
            
        Returns:
            plt.Figure: Matplotlib 圖表對象。
        """
        # 創建圖表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 設置顏色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # 繪製每個損失曲線
        for i, (loss_type, losses) in enumerate(loss_history.items()):
            epochs = range(1, len(losses) + 1)
            line, = ax.plot(epochs, losses, marker='o', linestyle='-', linewidth=2, 
                           markersize=4, label=loss_type.capitalize(), color=colors[i % len(colors)])
            
            # 標記最小值
            if mark_min and losses:
                min_loss = min(losses)
                min_epoch = losses.index(min_loss) + 1
                ax.plot(min_epoch, min_loss, 'o', markersize=8, markerfacecolor='none', 
                       markeredgewidth=2, markeredgecolor=line.get_color())
                ax.annotate(f'Min: {min_loss:.4f}', 
                           xy=(min_epoch, min_loss),
                           xytext=(min_epoch + 1, min_loss * 1.1),
                           arrowprops=dict(arrowstyle='->'))
        
        # 設置標題和標籤
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # 設置對數尺度
        if log_scale:
            ax.set_yscale('log')
        
        # 添加圖例和網格線
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 美化圖表
        plt.tight_layout()
        
        # 保存圖表
        if self.output_dir and filename:
            self._save_fig(fig, filename)
        
        # 顯示圖表
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_metric_curves(self, metric_history: Dict[str, Dict[str, List[float]]],
                          title: str = 'Training Metrics',
                          xlabel: str = 'Epoch',
                          metric_names: List[str] = None,
                          subplots: bool = True,
                          filename: str = None,
                          show: bool = True) -> plt.Figure:
        """
        繪製多個性能指標的曲線圖。
        
        Args:
            metric_history: 指標歷史記錄，例如 {'train': {'acc': [...], 'f1': [...]}, 'val': {...}}。
            title (str, optional): 圖表標題。默認為 'Training Metrics'。
            xlabel (str, optional): X 軸標籤。默認為 'Epoch'。
            metric_names (List[str], optional): 要繪製的指標名稱。若為 None，則繪製所有指標。
            subplots (bool, optional): 是否為每個指標創建子圖。默認為 True。
            filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
            show (bool, optional): 是否顯示圖表。默認為 True。
            
        Returns:
            plt.Figure: Matplotlib 圖表對象。
        """
        # 獲取所有指標名稱
        if not metric_names:
            # 從所有數據集中獲取唯一的指標名稱
            metric_names = set()
            for dataset_metrics in metric_history.values():
                metric_names.update(dataset_metrics.keys())
            metric_names = sorted(list(metric_names))
        
        # 確定子圖佈局
        n_metrics = len(metric_names)
        if subplots and n_metrics > 1:
            n_rows = int(np.ceil(n_metrics / 2))
            n_cols = min(2, n_metrics)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.figsize[0], self.figsize[1] * n_rows / 2), 
                                     dpi=self.dpi, sharex=True)
            if n_metrics > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
        else:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            axes = [ax] * n_metrics  # 所有指標共用同一個子圖
        
        # 設置顏色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # 繪製每個指標的曲線
        for i, metric_name in enumerate(metric_names):
            ax = axes[i if subplots and n_metrics > 1 else 0]
            
            for j, (dataset, metrics) in enumerate(metric_history.items()):
                if metric_name in metrics:
                    values = metrics[metric_name]
                    epochs = range(1, len(values) + 1)
                    label = f'{dataset} {metric_name}'
                    ax.plot(epochs, values, marker='o', linestyle='-', linewidth=2,
                           markersize=4, label=label, color=colors[j % len(colors)])
            
            # 設置標題和標籤
            metric_title = f'{metric_name.capitalize()}' if subplots else title
            ax.set_title(metric_title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric_name.capitalize())
            
            # 添加圖例和網格線
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # 隱藏多餘的子圖
        if subplots and n_metrics > 1:
            for i in range(n_metrics, len(axes)):
                axes[i].axis('off')
        
        # 設置整體標題
        if subplots and n_metrics > 1:
            plt.suptitle(title, fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # 為 suptitle 留出空間
        else:
            plt.tight_layout()
        
        # 保存圖表
        if self.output_dir and filename:
            self._save_fig(fig, filename)
        
        # 顯示圖表
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_learning_rate(self, lr_history: List[float],
                          title: str = 'Learning Rate Schedule',
                          xlabel: str = 'Epoch',
                          ylabel: str = 'Learning Rate',
                          log_scale: bool = True,
                          filename: str = None,
                          show: bool = True) -> plt.Figure:
        """
        繪製學習率變化曲線。
        
        Args:
            lr_history (List[float]): 學習率歷史記錄。
            title (str, optional): 圖表標題。默認為 'Learning Rate Schedule'。
            xlabel (str, optional): X 軸標籤。默認為 'Epoch'。
            ylabel (str, optional): Y 軸標籤。默認為 'Learning Rate'。
            log_scale (bool, optional): 是否使用對數尺度。默認為 True。
            filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
            show (bool, optional): 是否顯示圖表。默認為 True。
            
        Returns:
            plt.Figure: Matplotlib 圖表對象。
        """
        # 創建圖表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 繪製學習率曲線
        epochs = range(1, len(lr_history) + 1)
        ax.plot(epochs, lr_history, marker='o', linestyle='-', linewidth=2, markersize=4, color='#1f77b4')
        
        # 設置標題和標籤
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # 設置對數尺度
        if log_scale:
            ax.set_yscale('log')
        
        # 添加網格線
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 美化圖表
        plt.tight_layout()
        
        # 保存圖表
        if self.output_dir and filename:
            self._save_fig(fig, filename)
        
        # 顯示圖表
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_convergence_analysis(self, loss_history: List[float], 
                                 convergence_epoch: int,
                                 title: str = 'Convergence Analysis',
                                 xlabel: str = 'Epoch',
                                 ylabel: str = 'Loss',
                                 filename: str = None,
                                 show: bool = True) -> plt.Figure:
        """
        繪製收斂分析圖，包括收斂點標記。
        
        Args:
            loss_history (List[float]): 損失歷史記錄。
            convergence_epoch (int): 收斂點的輪次。
            title (str, optional): 圖表標題。默認為 'Convergence Analysis'。
            xlabel (str, optional): X 軸標籤。默認為 'Epoch'。
            ylabel (str, optional): Y 軸標籤。默認為 'Loss'。
            filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
            show (bool, optional): 是否顯示圖表。默認為 True。
            
        Returns:
            plt.Figure: Matplotlib 圖表對象。
        """
        # 創建圖表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 繪製損失曲線
        epochs = range(1, len(loss_history) + 1)
        ax.plot(epochs, loss_history, marker='o', linestyle='-', linewidth=2, markersize=4, color='#1f77b4', label='Loss')
        
        # 標記收斂點
        if 0 < convergence_epoch <= len(loss_history):
            conv_loss = loss_history[convergence_epoch - 1]
            ax.axvline(x=convergence_epoch, color='red', linestyle='--', label=f'Convergence at epoch {convergence_epoch}')
            ax.plot(convergence_epoch, conv_loss, 'ro', markersize=8)
            
            # 將損失曲線分為收斂前和收斂後兩部分
            ax.fill_between(epochs[:convergence_epoch], loss_history[:convergence_epoch], alpha=0.2, color='orange', label='Pre-convergence')
            if convergence_epoch < len(loss_history):
                ax.fill_between(epochs[convergence_epoch:], loss_history[convergence_epoch:], alpha=0.2, color='green', label='Post-convergence')
        
        # 設置標題和標籤
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # 添加圖例和網格線
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 美化圖表
        plt.tight_layout()
        
        # 保存圖表
        if self.output_dir and filename:
            self._save_fig(fig, filename)
        
        # 顯示圖表
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_training_stability(self, loss_history: List[float],
                               moving_avg_window: int = 5,
                               title: str = 'Training Stability Analysis',
                               xlabel: str = 'Epoch',
                               ylabel: str = 'Loss',
                               filename: str = None,
                               show: bool = True) -> plt.Figure:
        """
        繪製訓練穩定性分析圖，包括移動平均和波動性。
        
        Args:
            loss_history (List[float]): 損失歷史記錄。
            moving_avg_window (int, optional): 移動平均的窗口大小。默認為 5。
            title (str, optional): 圖表標題。默認為 'Training Stability Analysis'。
            xlabel (str, optional): X 軸標籤。默認為 'Epoch'。
            ylabel (str, optional): Y 軸標籤。默認為 'Loss'。
            filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
            show (bool, optional): 是否顯示圖表。默認為 True。
            
        Returns:
            plt.Figure: Matplotlib 圖表對象。
        """
        # 創建圖表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 繪製原始損失曲線
        epochs = range(1, len(loss_history) + 1)
        ax.plot(epochs, loss_history, marker='o', linestyle='-', linewidth=1, markersize=3, 
               alpha=0.6, color='#1f77b4', label='Original Loss')
        
        # 計算並繪製移動平均
        if len(loss_history) >= moving_avg_window:
            moving_avg = []
            for i in range(len(loss_history) - moving_avg_window + 1):
                moving_avg.append(np.mean(loss_history[i:i+moving_avg_window]))
            
            # 移動平均的繪製位置需要偏移
            ma_epochs = range(moving_avg_window, len(loss_history) + 1)
            ax.plot(ma_epochs, moving_avg, marker='', linestyle='-', linewidth=2, 
                   color='red', label=f'{moving_avg_window}-Epoch Moving Average')
        
        # 計算並顯示波動性指標
        if len(loss_history) > 1:
            diffs = np.diff(loss_history)
            volatility = np.std(diffs)
            ax.text(0.02, 0.98, f'Volatility: {volatility:.4f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 設置標題和標籤
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # 添加圖例和網格線
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 美化圖表
        plt.tight_layout()
        
        # 保存圖表
        if self.output_dir and filename:
            self._save_fig(fig, filename)
        
        # 顯示圖表
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_gradient_norm(self, gradient_norms: List[float],
                          title: str = 'Gradient Norm History',
                          xlabel: str = 'Epoch',
                          ylabel: str = 'Gradient Norm',
                          log_scale: bool = False,
                          filename: str = None,
                          show: bool = True) -> plt.Figure:
        """
        繪製梯度範數歷史曲線。
        
        Args:
            gradient_norms (List[float]): 梯度範數歷史記錄。
            title (str, optional): 圖表標題。默認為 'Gradient Norm History'。
            xlabel (str, optional): X 軸標籤。默認為 'Epoch'。
            ylabel (str, optional): Y 軸標籤。默認為 'Gradient Norm'。
            log_scale (bool, optional): 是否使用對數尺度。默認為 False。
            filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
            show (bool, optional): 是否顯示圖表。默認為 True。
            
        Returns:
            plt.Figure: Matplotlib 圖表對象。
        """
        # 創建圖表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 繪製梯度範數曲線
        epochs = range(1, len(gradient_norms) + 1)
        ax.plot(epochs, gradient_norms, marker='o', linestyle='-', linewidth=2, markersize=4, color='#1f77b4')
        
        # 標記異常值
        mean_norm = np.mean(gradient_norms)
        std_norm = np.std(gradient_norms)
        threshold = mean_norm + 2 * std_norm
        
        for i, norm in enumerate(gradient_norms):
            if norm > threshold:
                ax.plot(i + 1, norm, 'ro', markersize=8)
                ax.annotate(f'Anomaly: {norm:.2f}', 
                           xy=(i + 1, norm),
                           xytext=(i + 1 + 0.5, norm * 1.1),
                           arrowprops=dict(arrowstyle='->'))
        
        # 設置標題和標籤
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # 設置對數尺度
        if log_scale:
            ax.set_yscale('log')
        
        # 添加網格線
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 美化圖表
        plt.tight_layout()
        
        # 保存圖表
        if self.output_dir and filename:
            self._save_fig(fig, filename)
        
        # 顯示圖表
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_training_heatmap(self, epoch_batch_data: np.ndarray,
                             title: str = 'Training Progress Heatmap',
                             xlabel: str = 'Batch',
                             ylabel: str = 'Epoch',
                             cmap: str = 'viridis',
                             vmin: float = None,
                             vmax: float = None,
                             filename: str = None,
                             show: bool = True) -> plt.Figure:
        """
        繪製訓練進度熱力圖，每個單元格表示特定輪次和批次的某個指標。
        
        Args:
            epoch_batch_data (np.ndarray): 形狀為 [n_epochs, n_batches] 的數據。
            title (str, optional): 圖表標題。默認為 'Training Progress Heatmap'。
            xlabel (str, optional): X 軸標籤。默認為 'Batch'。
            ylabel (str, optional): Y 軸標籤。默認為 'Epoch'。
            cmap (str, optional): 顏色映射。默認為 'viridis'。
            vmin (float, optional): 顏色映射的最小值。默認為 None。
            vmax (float, optional): 顏色映射的最大值。默認為 None。
            filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
            show (bool, optional): 是否顯示圖表。默認為 True。
            
        Returns:
            plt.Figure: Matplotlib 圖表對象。
        """
        # 創建圖表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 繪製熱力圖
        im = ax.imshow(epoch_batch_data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        
        # 添加顏色條
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Value')
        
        # 設置標題和標籤
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # 設置刻度
        ax.set_xticks(np.arange(0, epoch_batch_data.shape[1], max(1, epoch_batch_data.shape[1] // 10)))
        ax.set_yticks(np.arange(0, epoch_batch_data.shape[0], max(1, epoch_batch_data.shape[0] // 10)))
        
        # 美化圖表
        plt.tight_layout()
        
        # 保存圖表
        if self.output_dir and filename:
            self._save_fig(fig, filename)
        
        # 顯示圖表
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig 