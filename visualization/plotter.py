"""
繪圖工具基類模組。

此模組提供用於視覺化的基本繪圖功能，包括繪圖器基類和通用的繪圖功能。

Classes:
    BasePlotter: 所有繪圖器的基類。
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional, Union
import matplotlib
from pathlib import Path

class BasePlotter:
    """
    繪圖器基類，提供所有繪圖器共用的基本功能。
    
    Attributes:
        output_dir (str): 圖表保存的輸出目錄。
        style (str): Matplotlib 繪圖風格。
        figsize (Tuple[int, int]): 圖表大小。
        dpi (int): 圖表 DPI。
    """
    
    def __init__(self, output_dir: Optional[str] = None, style: str = 'seaborn-whitegrid', 
                 figsize: Tuple[int, int] = (10, 6), dpi: int = 100):
        """
        初始化繪圖器基類。
        
        Args:
            output_dir (str, optional): 圖表保存的輸出目錄。若為 None，則不保存圖表。
            style (str, optional): Matplotlib 繪圖風格。默認為 'seaborn-whitegrid'。
            figsize (Tuple[int, int], optional): 圖表大小。默認為 (10, 6)。
            dpi (int, optional): 圖表 DPI。默認為 100。
        """
        self.output_dir = output_dir
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        # 設置 Matplotlib 繪圖風格
        try:
            plt.style.use(style)
        except Exception as e:
            print(f"無法設置繪圖風格 '{style}': {e}")
            print("使用默認風格")
        
        # 如果輸出目錄不存在，則創建
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def _save_fig(self, fig: plt.Figure, filename: str) -> str:
        """
        保存圖表到文件。
        
        Args:
            fig (plt.Figure): 要保存的圖表。
            filename (str): 文件名，不包含擴展名。
            
        Returns:
            str: 保存文件的完整路徑。
        """
        if not self.output_dir:
            return None
        
        # 確保文件名有正確的擴展名
        if not filename.endswith(('.png', '.jpg', '.svg', '.pdf')):
            filename = f"{filename}.png"
        
        # 構建完整路徑
        filepath = os.path.join(self.output_dir, filename)
        
        # 確保目錄存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存圖表
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        
        return filepath
    
    def set_output_dir(self, output_dir: str) -> None:
        """
        設置輸出目錄。
        
        Args:
            output_dir (str): 新的輸出目錄。
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def set_style(self, style: str) -> None:
        """
        設置繪圖風格。
        
        Args:
            style (str): Matplotlib 繪圖風格。
        """
        self.style = style
        try:
            plt.style.use(style)
        except Exception as e:
            print(f"無法設置繪圖風格 '{style}': {e}")
            print("保持當前風格")
    
    def set_figsize(self, figsize: Tuple[int, int]) -> None:
        """
        設置圖表大小。
        
        Args:
            figsize (Tuple[int, int]): 圖表大小，如 (寬度, 高度)。
        """
        self.figsize = figsize
    
    def set_dpi(self, dpi: int) -> None:
        """
        設置圖表 DPI。
        
        Args:
            dpi (int): 圖表 DPI。
        """
        self.dpi = dpi
    
    def plot_histogram(self, data, title="Histogram", bins=50, xlabel="Value", ylabel="Frequency", 
                       filename=None, show=True) -> plt.Figure:
        """
        繪製直方圖。
        
        Args:
            data (array-like): 要繪製的數據。
            title (str, optional): 圖表標題。默認為 'Histogram'。
            bins (int, optional): 直方圖的箱數。默認為 50。
            xlabel (str, optional): X 軸標籤。默認為 'Value'。
            ylabel (str, optional): Y 軸標籤。默認為 'Frequency'。
            filename (str, optional): 保存文件名。若為 None，則不保存。
            show (bool, optional): 是否顯示圖表。默認為 True。
            
        Returns:
            plt.Figure: Matplotlib 圖表對象。
        """
        # 創建圖表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 繪製直方圖
        ax.hist(np.asarray(data).flatten(), bins=bins)
        
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

    def plot_waveform(self, waveform, sample_rate, title="Waveform", xlabel="Time (s)", ylabel="Amplitude",
                      filename=None, show=True) -> plt.Figure:
        """
        繪製波形圖。
        
        Args:
            waveform (array-like): 一維音頻波形數據。
            sample_rate (int): 音頻的採樣率。
            title (str, optional): 圖表標題。默認為 'Waveform'。
            xlabel (str, optional): X 軸標籤。默認為 'Time (s)'。
            ylabel (str, optional): Y 軸標籤。默認為 'Amplitude'。
            filename (str, optional): 保存文件名。若為 None，則不保存。
            show (bool, optional): 是否顯示圖表。默認為 True。
            
        Returns:
            plt.Figure: Matplotlib 圖表對象。
        """
        # 創建圖表
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 轉換數據
        waveform = np.asarray(waveform).flatten()
        time_axis = np.arange(len(waveform)) / sample_rate
        
        # 繪製波形
        ax.plot(time_axis, waveform)
        
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
