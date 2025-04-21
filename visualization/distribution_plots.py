"""
分布視覺化模組。

此模組提供用於視覺化數據分布的功能。

Functions:
    plot_tensor_histogram: 繪製張量的直方圖
    plot_tensor_kde: 繪製張量的核密度估計
    plot_distribution_comparison: 比較兩個分布
    plot_layer_activations: 繪製神經網絡層的激活值分布
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import torch

from ..utils.tensor_utils import tensor_to_numpy, flatten_tensor, sample_tensor

logger = logging.getLogger(__name__)

def plot_tensor_histogram(tensor: torch.Tensor, title: str = 'Tensor Distribution', 
                        xlabel: str = 'Value', ylabel: str = 'Frequency',
                        bins: int = 50, figsize: Tuple[int, int] = (10, 6), 
                        color: str = 'blue', alpha: float = 0.7,
                        log_scale: bool = False, max_samples: int = 100000) -> plt.Figure:
    """
    繪製張量的直方圖。
    
    Args:
        tensor (torch.Tensor): 輸入張量
        title (str, optional): 圖表標題. 默認為'Tensor Distribution'
        xlabel (str, optional): x軸標籤. 默認為'Value'
        ylabel (str, optional): y軸標籤. 默認為'Frequency'
        bins (int, optional): 直方圖的柱數. 默認為50
        figsize (Tuple[int, int], optional): 圖表尺寸. 默認為(10, 6)
        color (str, optional): 直方圖顏色. 默認為'blue'
        alpha (float, optional): 透明度. 默認為0.7
        log_scale (bool, optional): 是否使用對數尺度. 默認為False
        max_samples (int, optional): 最大樣本數. 默認為100000
    
    Returns:
        plt.Figure: Matplotlib圖表對象
    """
    # 對大型張量進行抽樣
    if tensor.numel() > max_samples:
        tensor = sample_tensor(tensor, max_samples)
    
    # 轉換為numpy數組並平坦化
    values = tensor_to_numpy(flatten_tensor(tensor))
    
    # 計算統計量
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 繪製直方圖
    n, bins, patches = ax.hist(values, bins=bins, color=color, alpha=alpha, density=True)
    
    # 繪製垂直線標記均值和中位數
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.4f}')
    ax.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.4f}')
    
    # 添加統計信息
    stats_text = f'Mean: {mean_val:.4f}\nMedian: {median_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # 設置標題和標籤
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # 設置對數尺度（如果指定）
    if log_scale:
        ax.set_yscale('log')
    
    # 添加圖例
    ax.legend()
    
    # 調整佈局
    plt.tight_layout()
    
    return fig

def plot_tensor_kde(tensor: torch.Tensor, title: str = 'Tensor Density', 
                   xlabel: str = 'Value', ylabel: str = 'Density',
                   figsize: Tuple[int, int] = (10, 6), color: str = 'blue',
                   max_samples: int = 100000) -> plt.Figure:
    """
    繪製張量的核密度估計(KDE)圖。
    
    Args:
        tensor (torch.Tensor): 輸入張量
        title (str, optional): 圖表標題. 默認為'Tensor Density'
        xlabel (str, optional): x軸標籤. 默認為'Value'
        ylabel (str, optional): y軸標籤. 默認為'Density'
        figsize (Tuple[int, int], optional): 圖表尺寸. 默認為(10, 6)
        color (str, optional): 線條顏色. 默認為'blue'
        max_samples (int, optional): 最大樣本數. 默認為100000
    
    Returns:
        plt.Figure: Matplotlib圖表對象
    """
    # 對大型張量進行抽樣
    if tensor.numel() > max_samples:
        tensor = sample_tensor(tensor, max_samples)
    
    # 轉換為numpy數組並平坦化
    values = tensor_to_numpy(flatten_tensor(tensor))
    
    # 計算統計量
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用Seaborn繪製KDE圖
    sns.kdeplot(values, ax=ax, color=color, fill=True, alpha=0.3)
    
    # 標記均值和中位數
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.4f}')
    ax.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.4f}')
    
    # 添加統計信息
    stats_text = f'Mean: {mean_val:.4f}\nMedian: {median_val:.4f}\nStd: {std_val:.4f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # 設置標題和標籤
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # 添加圖例
    ax.legend()
    
    # 調整佈局
    plt.tight_layout()
    
    return fig

def plot_distribution_comparison(tensor1: torch.Tensor, tensor2: torch.Tensor,
                                title: str = 'Distribution Comparison',
                                labels: List[str] = ['Distribution 1', 'Distribution 2'],
                                figsize: Tuple[int, int] = (12, 8),
                                bins: int = 50, max_samples: int = 50000,
                                plot_type: str = 'both') -> plt.Figure:
    """
    比較兩個張量的分布。
    
    Args:
        tensor1 (torch.Tensor): 第一個張量
        tensor2 (torch.Tensor): 第二個張量
        title (str, optional): 圖表標題. 默認為'Distribution Comparison'
        labels (List[str], optional): 兩個分布的標籤. 默認為['Distribution 1', 'Distribution 2']
        figsize (Tuple[int, int], optional): 圖表尺寸. 默認為(12, 8)
        bins (int, optional): 直方圖的柱數. 默認為50
        max_samples (int, optional): 每個張量的最大樣本數. 默認為50000
        plot_type (str, optional): 繪圖類型，'hist', 'kde'或'both'. 默認為'both'
    
    Returns:
        plt.Figure: Matplotlib圖表對象
    """
    # 對大型張量進行抽樣
    if tensor1.numel() > max_samples:
        tensor1 = sample_tensor(tensor1, max_samples)
    
    if tensor2.numel() > max_samples:
        tensor2 = sample_tensor(tensor2, max_samples)
    
    # 轉換為numpy數組並平坦化
    values1 = tensor_to_numpy(flatten_tensor(tensor1))
    values2 = tensor_to_numpy(flatten_tensor(tensor2))
    
    # 創建圖表
    if plot_type == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 直方圖
        ax1.hist(values1, bins=bins, alpha=0.5, label=labels[0], density=True)
        ax1.hist(values2, bins=bins, alpha=0.5, label=labels[1], density=True)
        ax1.set_title(f'{title} - Histogram')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        # KDE圖
        sns.kdeplot(values1, ax=ax2, label=labels[0], fill=True, alpha=0.3)
        sns.kdeplot(values2, ax=ax2, label=labels[1], fill=True, alpha=0.3)
        ax2.set_title(f'{title} - Density')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Density')
        ax2.legend()
    
    elif plot_type == 'hist':
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(values1, bins=bins, alpha=0.5, label=labels[0], density=True)
        ax.hist(values2, bins=bins, alpha=0.5, label=labels[1], density=True)
        ax.set_title(f'{title}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    elif plot_type == 'kde':
        fig, ax = plt.subplots(figsize=figsize)
        sns.kdeplot(values1, ax=ax, label=labels[0], fill=True, alpha=0.3)
        sns.kdeplot(values2, ax=ax, label=labels[1], fill=True, alpha=0.3)
        ax.set_title(f'{title}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
    
    # 調整佈局
    plt.tight_layout()
    
    return fig

def plot_layer_activations(activations: torch.Tensor, layer_name: str = '',
                          title: str = None, figsize: Tuple[int, int] = (12, 8),
                          max_channels: int = 9, flatten: bool = False) -> plt.Figure:
    """
    繪製神經網絡層的激活值分布。
    
    可以顯示多個通道的直方圖或密度圖。
    
    Args:
        activations (torch.Tensor): 激活值張量
        layer_name (str, optional): 層名稱. 默認為''
        title (str, optional): 圖表標題. 默認為None
        figsize (Tuple[int, int], optional): 圖表尺寸. 默認為(12, 8)
        max_channels (int, optional): 要顯示的最大通道數. 默認為9
        flatten (bool, optional): 是否將所有通道合併為一個分布. 默認為False
    
    Returns:
        plt.Figure: Matplotlib圖表對象
    """
    # 確定標題
    if title is None:
        title = f'Layer Activations: {layer_name}' if layer_name else 'Layer Activations'
    
    # 獲取激活值
    act_np = tensor_to_numpy(activations)
    
    # 如果激活值是標量 (沒有通道維度)
    if len(act_np.shape) <= 2:
        return plot_tensor_histogram(torch.tensor(act_np), title=title, figsize=figsize)
    
    # 如果要將所有通道合併
    if flatten:
        return plot_tensor_histogram(torch.tensor(act_np), title=title, figsize=figsize)
    
    # 否則，為多個通道創建子圖
    batch_size, num_channels = act_np.shape[0], act_np.shape[1]
    
    # 如果通道數超過max_channels，則只顯示前max_channels個
    if num_channels > max_channels:
        logger.info(f"層有 {num_channels} 個通道，只顯示前 {max_channels} 個")
        num_channels = max_channels
    
    # 計算子圖的行數和列數
    n_rows = int(np.ceil(np.sqrt(num_channels)))
    n_cols = int(np.ceil(num_channels / n_rows))
    
    # 創建圖表
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # 平坦化axes數組以便迭代
    if num_channels > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # 為每個通道繪製直方圖
    for i in range(num_channels):
        # 獲取該通道的所有批次的值
        channel_values = act_np[:, i].flatten()
        
        # 繪製直方圖
        axes[i].hist(channel_values, bins=30, alpha=0.7)
        axes[i].set_title(f'Channel {i}')
        
        # 添加均值和標準差
        mean_val = np.mean(channel_values)
        std_val = np.std(channel_values)
        axes[i].axvline(mean_val, color='red', linestyle='dashed', linewidth=1)
        axes[i].text(0.95, 0.95, f'μ={mean_val:.4f}\nσ={std_val:.4f}',
                   transform=axes[i].transAxes, fontsize=8,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # 隱藏多餘的子圖
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
    
    # 設置整體標題
    plt.suptitle(title, fontsize=14)
    
    # 調整佈局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 為suptitle留出空間
    
    return fig 