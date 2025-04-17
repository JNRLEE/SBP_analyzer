"""
# 分佈視覺化
# 提供用於繪製數據分佈的函數
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import os


def plot_histogram(
    data: Union[np.ndarray, List],
    title: str = 'Data Distribution',
    xlabel: str = 'Value',
    ylabel: str = 'Frequency',
    bins: Optional[int] = None,
    range: Optional[Tuple[float, float]] = None,
    density: bool = False,
    stats: Optional[Dict[str, float]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    color: str = 'skyblue',
    alpha: float = 0.7,
    grid: bool = True,
    save_path: Optional[str] = None,
    experiment_id: Optional[str] = None,
    dpi: int = 100
) -> Tuple[plt.Figure, plt.Axes]:
    """
    繪製數據的直方圖。
    
    Args:
        data: 要繪製的數據，可以是 numpy 數組或列表
        title: 圖表標題
        xlabel: X 軸標籤
        ylabel: Y 軸標籤
        bins: 直方圖的區間數量
        range: 直方圖的數據範圍
        density: 是否將直方圖標準化以形成密度
        stats: 可選的統計量字典，將顯示在圖表上
        ax: 可選的 Matplotlib Axes 實例
        figsize: 圖表尺寸
        color: 直方圖顏色
        alpha: 直方圖透明度
        grid: 是否顯示網格
        save_path: 保存圖表的路徑
        experiment_id: 實驗 ID，用於生成文件名
        dpi: 圖表分辨率
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: 包含圖表和軸的元組
        
    Description:
        繪製數據的直方圖，可選擇性地顯示統計信息和密度曲線，並提供保存圖表的選項。
        
    References:
        - Matplotlib Histograms: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
    """
    # 確保數據是 numpy 數組
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # 處理空數據
    if data.size == 0:
        raise ValueError("無法繪製空數據的直方圖")
    
    # 處理 nan 和 inf
    data_clean = data[np.isfinite(data)]
    if data_clean.size == 0:
        raise ValueError("所有數據都是 NaN 或 Inf，無法繪製直方圖")
    
    # 自動選擇 bins 數量
    if bins is None:
        bins = int(np.sqrt(data_clean.size))
        bins = max(5, min(bins, 100))  # 至少 5 個，最多 100 個
    
    # 自動選擇合適的範圍
    if range is None:
        range = (np.min(data_clean), np.max(data_clean))
    
    # 創建或使用現有的軸
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # 繪製直方圖
    n, bins, patches = ax.hist(
        data_clean, bins=bins, range=range, density=density,
        color=color, alpha=alpha, edgecolor='black', linewidth=0.8
    )
    
    # 設置標題和標籤
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # 顯示網格
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加統計信息
    if stats:
        info_text = (
            f"Mean: {stats.get('mean', np.mean(data_clean)):.4f}\n"
            f"Median: {stats.get('median', np.median(data_clean)):.4f}\n"
            f"Std: {stats.get('std', np.std(data_clean)):.4f}\n"
            f"Min: {stats.get('min', np.min(data_clean)):.4f}\n"
            f"Max: {stats.get('max', np.max(data_clean)):.4f}\n"
            f"Count: {stats.get('count', data_clean.size)}"
        )
        
        # 如果有偏度和峰度的信息
        if 'skewness' in stats and 'kurtosis' in stats:
            info_text += f"\nSkewness: {stats['skewness']:.4f}\n"
            info_text += f"Kurtosis: {stats['kurtosis']:.4f}"
        
        # 將統計信息放在圖的右上角
        ax.text(
            0.95, 0.95, info_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    # 如果啟用了密度，添加核密度估計
    if density:
        from scipy import stats as sp_stats
        
        # 計算核密度估計
        x = np.linspace(range[0], range[1], 1000)
        kde = sp_stats.gaussian_kde(data_clean)
        ax.plot(x, kde(x), 'r-', linewidth=2, label='KDE')
        ax.legend()
    
    plt.tight_layout()
    
    # 保存圖表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 處理文件名
        if experiment_id:
            # 從 save_path 中提取基本文件名和目錄
            base_dir = os.path.dirname(save_path)
            base_name, ext = os.path.splitext(os.path.basename(save_path))
            if not ext:  # 如果沒有擴展名，使用默認的
                ext = '.png'
            
            # 構建符合標準命名格式的文件名
            filename = f"{experiment_id}_histogram_{base_name}{ext}"
            save_path = os.path.join(base_dir, filename)
        
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    return fig, ax


def plot_box_plot(
    data: Union[np.ndarray, List, Dict[str, np.ndarray]],
    title: str = 'Box Plot',
    xlabel: str = 'Data Groups',
    ylabel: str = 'Values',
    labels: Optional[List[str]] = None,
    vert: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    grid: bool = True,
    save_path: Optional[str] = None,
    experiment_id: Optional[str] = None,
    dpi: int = 100,
    showfliers: bool = True,
    notch: bool = False,
    patch_artist: bool = True,
    colors: Optional[List[str]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    繪製數據的箱型圖。
    
    Args:
        data: 要繪製的數據，可以是單個數組或字典（鍵為標籤，值為數組）
        title: 圖表標題
        xlabel: X 軸標籤
        ylabel: Y 軸標籤
        labels: 數據組的標籤列表
        vert: 是否垂直繪製箱型圖
        ax: 可選的 Matplotlib Axes 實例
        figsize: 圖表尺寸
        grid: 是否顯示網格
        save_path: 保存圖表的路徑
        experiment_id: 實驗 ID，用於生成文件名
        dpi: 圖表分辨率
        showfliers: 是否顯示異常值
        notch: 是否在箱體上繪製凹口
        patch_artist: 是否填充箱體
        colors: 箱體顏色列表
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: 包含圖表和軸的元組
        
    Description:
        繪製數據的箱型圖，可以比較多個數據組的分佈情況，並提供保存圖表的選項。
        
    References:
        - Matplotlib Box Plots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html
    """
    # 處理輸入數據
    if isinstance(data, dict):
        data_list = list(data.values())
        if labels is None:
            labels = list(data.keys())
    else:
        # 如果是單個數組，轉換為列表包含的一個數組
        if isinstance(data, np.ndarray) or isinstance(data, list):
            if data and not isinstance(data[0], (list, np.ndarray)):
                data_list = [data]
            else:
                data_list = data
        else:
            raise ValueError("data 必須是數組、列表或字典")
    
    # 創建或使用現有的軸
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # 設置默認顏色
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(data_list)))
    
    # 繪製箱型圖
    bp = ax.boxplot(
        data_list, 
        labels=labels, 
        vert=vert, 
        notch=notch, 
        patch_artist=patch_artist,
        showfliers=showfliers
    )
    
    # 如果使用 patch_artist，為箱體填充顏色
    if patch_artist:
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    
    # 設置標題和標籤
    ax.set_title(title, fontsize=14)
    if vert:
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
    else:
        ax.set_xlabel(ylabel, fontsize=12)
        ax.set_ylabel(xlabel, fontsize=12)
    
    # 顯示網格
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存圖表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 處理文件名
        if experiment_id:
            # 從 save_path 中提取基本文件名和目錄
            base_dir = os.path.dirname(save_path)
            base_name, ext = os.path.splitext(os.path.basename(save_path))
            if not ext:  # 如果沒有擴展名，使用默認的
                ext = '.png'
            
            # 構建符合標準命名格式的文件名
            filename = f"{experiment_id}_boxplot_{base_name}{ext}"
            save_path = os.path.join(base_dir, filename)
        
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    return fig, ax 