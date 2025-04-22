"""
分佈視覺化模組。

此模組提供用於視覺化各種分佈數據的繪圖功能，包括參數分佈、激活值分佈等。

Classes:
    DistributionPlotter: 用於繪製分佈相關圖表的類。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
import torch

from .plotter import BasePlotter


class DistributionPlotter(BasePlotter):
    """
    分佈視覺化器，用於繪製各種分佈相關的圖表。
    
    此類提供多種方法來視覺化權重、激活值和其他張量數據的分佈情況。
    
    Attributes:
        output_dir (str): 圖表保存的輸出目錄。
        style (str): Matplotlib 繪圖風格。
        figsize (Tuple[int, int]): 圖表大小。
        dpi (int): 圖表 DPI。
    """
    
    def __init__(self, output_dir: str = None, style: str = 'seaborn-whitegrid', 
                 figsize: Tuple[int, int] = (10, 6), dpi: int = 100):
        """
        初始化分佈視覺化器。
        
        Args:
            output_dir (str, optional): 圖表保存的輸出目錄。若為 None，則不保存圖表。
            style (str, optional): Matplotlib 繪圖風格。默認為 'seaborn-whitegrid'。
            figsize (Tuple[int, int], optional): 圖表大小。默認為 (10, 6)。
            dpi (int, optional): 圖表 DPI。默認為 100。
        """
        super().__init__(output_dir, style, figsize, dpi)
    
    def plot_histogram(self, data: Union[np.ndarray, torch.Tensor, List], 
                       title: str = 'Distribution Histogram', 
                       xlabel: str = 'Value', 
                       ylabel: str = 'Frequency',
                       bins: int = 50,
                       color: str = '#2077B4',
                       kde: bool = True,
                       filename: str = None,
                       show: bool = True) -> plt.Figure:
        """
        繪製直方圖來視覺化數據分佈。
        
        Args:
            data: 要繪製的數據，可以是 NumPy 數組、PyTorch 張量或列表。
            title (str, optional): 圖表標題。默認為 'Distribution Histogram'。
            xlabel (str, optional): X 軸標籤。默認為 'Value'。
            ylabel (str, optional): Y 軸標籤。默認為 'Frequency'。
            bins (int, optional): 直方圖的箱數。默認為 50。
            color (str, optional): 直方圖的顏色。默認為 '#2077B4'。
            kde (bool, optional): 是否顯示核密度估計曲線。默認為 True。
            filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
            show (bool, optional): 是否顯示圖表。默認為 True。
            
        Returns:
            plt.Figure: Matplotlib 圖表對象。
        """
        # 轉換數據為 NumPy 數組
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        data = np.array(data).flatten()
        
        # 創建圖表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 繪製直方圖
        sns.histplot(data, bins=bins, kde=kde, color=color, ax=ax)
        
        # 設置標題和標籤
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
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
    
    def plot_distribution_comparison(self, data_dict: Dict[str, Union[np.ndarray, torch.Tensor, List]],
                                    title: str = 'Distribution Comparison',
                                    xlabel: str = 'Value',
                                    ylabel: str = 'Density',
                                    filename: str = None,
                                    show: bool = True) -> plt.Figure:
        """
        繪製多個分佈的比較圖。
        
        Args:
            data_dict: 字典，鍵為分佈名稱，值為數據。
            title (str, optional): 圖表標題。默認為 'Distribution Comparison'。
            xlabel (str, optional): X 軸標籤。默認為 'Value'。
            ylabel (str, optional): Y 軸標籤。默認為 'Density'。
            filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
            show (bool, optional): 是否顯示圖表。默認為 True。
            
        Returns:
            plt.Figure: Matplotlib 圖表對象。
        """
        # 創建圖表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 為每個分佈繪製密度圖
        for name, data in data_dict.items():
            # 轉換數據為 NumPy 數組
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            
            data = np.array(data).flatten()
            
            # 繪製密度圖
            sns.kdeplot(data, ax=ax, label=name)
        
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
    
    def plot_boxplot(self, data: Union[Dict[str, Union[np.ndarray, torch.Tensor, List]], 
                                       pd.DataFrame],
                     title: str = 'Boxplot Comparison',
                     xlabel: str = 'Category',
                     ylabel: str = 'Value',
                     vert: bool = True,
                     filename: str = None,
                     show: bool = True) -> plt.Figure:
        """
        繪製箱形圖比較不同類別的數據分佈。
        
        Args:
            data: 字典（鍵為類別名稱，值為數據）或 DataFrame。
            title (str, optional): 圖表標題。默認為 'Boxplot Comparison'。
            xlabel (str, optional): X 軸標籤。默認為 'Category'。
            ylabel (str, optional): Y 軸標籤。默認為 'Value'。
            vert (bool, optional): 是否垂直繪製箱形圖。默認為 True。
            filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
            show (bool, optional): 是否顯示圖表。默認為 True。
            
        Returns:
            plt.Figure: Matplotlib 圖表對象。
        """
        # 將輸入數據轉換為適合 boxplot 的格式
        if isinstance(data, dict):
            df = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})
        else:
            df = data
        
        # 創建圖表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 繪製箱形圖
        sns.boxplot(data=df, orient='v' if vert else 'h', ax=ax)
        
        # 設置標題和標籤
        ax.set_title(title)
        if vert:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            ax.set_xlabel(ylabel)
            ax.set_ylabel(xlabel)
        
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
    
    def plot_heatmap(self, data: Union[np.ndarray, torch.Tensor, List[List]],
                    title: str = 'Distribution Heatmap',
                    xlabel: str = 'X',
                    ylabel: str = 'Y',
                    cmap: str = 'viridis',
                    annot: bool = False,
                    filename: str = None,
                    show: bool = True) -> plt.Figure:
        """
        繪製熱力圖來視覺化二維數據。
        
        Args:
            data: 二維數據，可以是 NumPy 數組、PyTorch 張量或嵌套列表。
            title (str, optional): 圖表標題。默認為 'Distribution Heatmap'。
            xlabel (str, optional): X 軸標籤。默認為 'X'。
            ylabel (str, optional): Y 軸標籤。默認為 'Y'。
            cmap (str, optional): 顏色映射。默認為 'viridis'。
            annot (bool, optional): 是否在熱力圖中顯示數值。默認為 False。
            filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
            show (bool, optional): 是否顯示圖表。默認為 True。
            
        Returns:
            plt.Figure: Matplotlib 圖表對象。
        """
        # 轉換數據為 NumPy 數組
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        data = np.array(data)
        
        # 確保數據是二維的
        if data.ndim != 2:
            raise ValueError(f"數據必須是二維的，而不是 {data.ndim} 維")
        
        # 創建圖表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 繪製熱力圖
        sns.heatmap(data, cmap=cmap, annot=annot, ax=ax)
        
        # 設置標題和標籤
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
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
    
    def plot_qq_plot(self, data: Union[np.ndarray, torch.Tensor, List],
                    title: str = 'Q-Q Plot',
                    filename: str = None,
                    show: bool = True) -> plt.Figure:
        """
        繪製 Q-Q 圖來檢查數據是否符合正態分佈。
        
        Args:
            data: 一維數據，可以是 NumPy 數組、PyTorch 張量或列表。
            title (str, optional): 圖表標題。默認為 'Q-Q Plot'。
            filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
            show (bool, optional): 是否顯示圖表。默認為 True。
            
        Returns:
            plt.Figure: Matplotlib 圖表對象。
        """
        # 轉換數據為 NumPy 數組
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        data = np.array(data).flatten()
        
        # 創建圖表
        from scipy import stats
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 繪製 Q-Q 圖
        stats.probplot(data, plot=ax)
        
        # 設置標題
        ax.set_title(title)
        
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