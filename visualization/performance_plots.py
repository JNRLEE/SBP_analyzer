"""
性能指標視覺化模組。

此模組提供用於視覺化模型訓練性能指標的功能。

Functions:
    plot_loss_curve: 繪製損失曲線
    plot_metric_curves: 繪製多個指標曲線
    plot_learning_rate: 繪製學習率曲線
    plot_training_comparison: 比較多次訓練運行的性能
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from ..utils.stat_utils import calculate_moving_average, detect_outliers, smooth_curve

logger = logging.getLogger(__name__)

def plot_loss_curve(loss_values: List[float], val_loss_values: Optional[List[float]] = None,
                   title: str = 'Training Loss', figsize: Tuple[int, int] = (10, 6),
                   smooth_factor: Optional[float] = None, 
                   highlight_min: bool = True, log_scale: bool = False,
                   include_statistics: bool = True) -> plt.Figure:
    """
    繪製訓練損失曲線。
    
    Args:
        loss_values (List[float]): 訓練損失值列表
        val_loss_values (Optional[List[float]], optional): 驗證損失值列表。默認為None
        title (str, optional): 圖表標題. 默認為'Training Loss'
        figsize (Tuple[int, int], optional): 圖表尺寸. 默認為(10, 6)
        smooth_factor (Optional[float], optional): 平滑因子 (0-1)。默認為None，表示不平滑
        highlight_min (bool, optional): 是否突出顯示最小值. 默認為True
        log_scale (bool, optional): 是否使用對數尺度. 默認為False
        include_statistics (bool, optional): 是否包含統計信息. 默認為True
    
    Returns:
        plt.Figure: Matplotlib圖表對象
    """
    epochs = range(1, len(loss_values) + 1)
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 如果指定了平滑因子，則應用平滑
    if smooth_factor is not None:
        smoothed_loss = smooth_curve(loss_values, smooth_factor)
        ax.plot(epochs, smoothed_loss, 'b-', linewidth=2, label='Smoothed Train Loss')
        ax.plot(epochs, loss_values, 'b-', alpha=0.3, label='Train Loss')
    else:
        ax.plot(epochs, loss_values, 'b-', label='Train Loss')
    
    # 如果有驗證損失，也繪製它
    if val_loss_values is not None:
        if len(val_loss_values) != len(loss_values):
            logger.warning(f"驗證損失長度 ({len(val_loss_values)}) 與訓練損失長度 ({len(loss_values)}) 不一致")
            val_epochs = range(1, len(val_loss_values) + 1)
        else:
            val_epochs = epochs
        
        if smooth_factor is not None and len(val_loss_values) > 2:
            smoothed_val_loss = smooth_curve(val_loss_values, smooth_factor)
            ax.plot(val_epochs, smoothed_val_loss, 'r-', linewidth=2, label='Smoothed Val Loss')
            ax.plot(val_epochs, val_loss_values, 'r-', alpha=0.3, label='Val Loss')
        else:
            ax.plot(val_epochs, val_loss_values, 'r-', label='Val Loss')
    
    # 找出並標記最小損失
    if highlight_min:
        min_loss_epoch = loss_values.index(min(loss_values)) + 1
        min_loss = min(loss_values)
        ax.plot(min_loss_epoch, min_loss, 'bo', markersize=8)
        ax.annotate(f'Min: {min_loss:.4f}', 
                  xy=(min_loss_epoch, min_loss),
                  xytext=(min_loss_epoch + 1, min_loss * 1.1),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        # 如果有驗證損失，也標記其最小值
        if val_loss_values is not None:
            min_val_loss_epoch = val_loss_values.index(min(val_loss_values)) + 1
            min_val_loss = min(val_loss_values)
            ax.plot(min_val_loss_epoch, min_val_loss, 'ro', markersize=8)
            ax.annotate(f'Min: {min_val_loss:.4f}',
                      xy=(min_val_loss_epoch, min_val_loss),
                      xytext=(min_val_loss_epoch + 1, min_val_loss * 1.1),
                      arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # 添加統計信息
    if include_statistics:
        stats_text = [
            f"Initial Loss: {loss_values[0]:.4f}",
            f"Final Loss: {loss_values[-1]:.4f}",
            f"Min Loss: {min(loss_values):.4f} (Epoch {loss_values.index(min(loss_values))+1})",
            f"Mean Loss: {np.mean(loss_values):.4f}",
            f"Loss Reduction: {(loss_values[0] - loss_values[-1]) / loss_values[0]:.2%}"
        ]
        
        if val_loss_values is not None:
            stats_text.extend([
                f"Initial Val Loss: {val_loss_values[0]:.4f}",
                f"Final Val Loss: {val_loss_values[-1]:.4f}",
                f"Min Val Loss: {min(val_loss_values):.4f} (Epoch {val_loss_values.index(min(val_loss_values))+1})",
                f"Mean Val Loss: {np.mean(val_loss_values):.4f}",
                f"Val Loss Reduction: {(val_loss_values[0] - val_loss_values[-1]) / val_loss_values[0]:.2%}"
            ])
        
        ax.text(0.02, 0.02, '\n'.join(stats_text), transform=ax.transAxes, fontsize=9,
              verticalalignment='bottom', horizontalalignment='left',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 設置對數尺度（如果指定）
    if log_scale:
        ax.set_yscale('log')
    
    # 設置標題和標籤
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加圖例
    ax.legend()
    
    # 調整佈局
    plt.tight_layout()
    
    return fig

def plot_metric_curves(metrics: Dict[str, List[float]], title: str = 'Training Metrics',
                      figsize: Tuple[int, int] = (12, 8), smooth_factor: Optional[float] = None,
                      log_scale: bool = False, highlight_best: bool = True) -> plt.Figure:
    """
    繪製多個指標曲線。
    
    Args:
        metrics (Dict[str, List[float]]): 指標名稱到值列表的字典
        title (str, optional): 圖表標題. 默認為'Training Metrics'
        figsize (Tuple[int, int], optional): 圖表尺寸. 默認為(12, 8)
        smooth_factor (Optional[float], optional): 平滑因子 (0-1)。默認為None，表示不平滑
        log_scale (bool, optional): 是否使用對數尺度. 默認為False
        highlight_best (bool, optional): 是否突出顯示最佳值. 默認為True
    
    Returns:
        plt.Figure: Matplotlib圖表對象
    """
    if not metrics:
        logger.warning("沒有提供指標數據")
        return plt.figure(figsize=figsize)
    
    # 獲取所有指標的最大長度
    max_length = max(len(values) for values in metrics.values())
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 為每個指標繪製曲線
    for i, (metric_name, values) in enumerate(metrics.items()):
        if not values:
            continue
            
        epochs = range(1, len(values) + 1)
        
        # 獲取一個顏色
        color = plt.cm.tab10(i % 10)
        
        # 如果指定了平滑因子，則應用平滑
        if smooth_factor is not None and len(values) > 2:
            smoothed_values = smooth_curve(values, smooth_factor)
            ax.plot(epochs, smoothed_values, color=color, linewidth=2, label=f'{metric_name} (Smoothed)')
            ax.plot(epochs, values, color=color, alpha=0.3, label=f'{metric_name}')
        else:
            ax.plot(epochs, values, color=color, label=metric_name)
        
        # 標記最佳值
        if highlight_best:
            # 假設較高的值更好
            best_value = max(values)
            best_epoch = values.index(best_value) + 1
            
            ax.plot(best_epoch, best_value, 'o', color=color, markersize=8)
            ax.annotate(f'{best_value:.4f}',
                      xy=(best_epoch, best_value),
                      xytext=(best_epoch + 0.1, best_value),
                      color=color,
                      arrowprops=dict(facecolor=color, shrink=0.05, width=1.5))
    
    # 設置對數尺度（如果指定）
    if log_scale:
        ax.set_yscale('log')
    
    # 設置標題和標籤
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric Value')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加圖例
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 調整佈局
    plt.tight_layout()
    
    return fig

def plot_learning_rate(lr_values: List[float], title: str = 'Learning Rate Schedule',
                      figsize: Tuple[int, int] = (10, 6), log_scale: bool = True) -> plt.Figure:
    """
    繪製學習率曲線。
    
    Args:
        lr_values (List[float]): 學習率值列表
        title (str, optional): 圖表標題. 默認為'Learning Rate Schedule'
        figsize (Tuple[int, int], optional): 圖表尺寸. 默認為(10, 6)
        log_scale (bool, optional): 是否使用對數尺度. 默認為True
    
    Returns:
        plt.Figure: Matplotlib圖表對象
    """
    if not lr_values:
        logger.warning("沒有提供學習率數據")
        return plt.figure(figsize=figsize)
    
    epochs = range(1, len(lr_values) + 1)
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 繪製學習率曲線
    ax.plot(epochs, lr_values, 'b-', linewidth=2)
    
    # 標記重要點
    ax.plot(1, lr_values[0], 'ro', markersize=8)
    ax.annotate(f'Initial: {lr_values[0]:.6f}',
              xy=(1, lr_values[0]),
              xytext=(2, lr_values[0] * 1.1),
              arrowprops=dict(facecolor='red', shrink=0.05, width=1.5))
    
    ax.plot(len(lr_values), lr_values[-1], 'go', markersize=8)
    ax.annotate(f'Final: {lr_values[-1]:.6f}',
              xy=(len(lr_values), lr_values[-1]),
              xytext=(len(lr_values) - len(lr_values)//5, lr_values[-1] * 1.1),
              arrowprops=dict(facecolor='green', shrink=0.05, width=1.5))
    
    # 設置對數尺度（如果指定）
    if log_scale:
        ax.set_yscale('log')
    
    # 設置標題和標籤
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 調整佈局
    plt.tight_layout()
    
    return fig

def plot_training_comparison(runs: Dict[str, Dict[str, List[float]]], 
                           metric: str = 'train_loss', 
                           title: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 8),
                           smooth_factor: Optional[float] = 0.7,
                           log_scale: bool = False) -> plt.Figure:
    """
    比較多次訓練運行的性能。
    
    Args:
        runs (Dict[str, Dict[str, List[float]]]): 包含多次訓練運行數據的字典
                                               格式為 {run_name: {'train_loss': [...], 'val_loss': [...], ...}}
        metric (str, optional): 要比較的指標名稱. 默認為'train_loss'
        title (Optional[str], optional): 圖表標題. 默認為None
        figsize (Tuple[int, int], optional): 圖表尺寸. 默認為(12, 8)
        smooth_factor (Optional[float], optional): 平滑因子 (0-1)。默認為0.7
        log_scale (bool, optional): 是否使用對數尺度. 默認為False
    
    Returns:
        plt.Figure: Matplotlib圖表對象
    """
    if not runs:
        logger.warning("沒有提供訓練運行數據")
        return plt.figure(figsize=figsize)
    
    # 確定標題
    if title is None:
        title = f"Comparison of {metric}"
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用各種顏色繪製每次運行的曲線
    colors = plt.cm.tab10(range(len(runs)))
    
    for i, (run_name, run_data) in enumerate(runs.items()):
        if metric not in run_data:
            logger.warning(f"運行 '{run_name}' 不包含指標 '{metric}'")
            continue
        
        values = run_data[metric]
        epochs = range(1, len(values) + 1)
        
        # 如果指定了平滑因子，則應用平滑
        if smooth_factor is not None and len(values) > 2:
            smoothed_values = smooth_curve(values, smooth_factor)
            ax.plot(epochs, smoothed_values, color=colors[i], linewidth=2, label=f'{run_name}')
            ax.plot(epochs, values, color=colors[i], alpha=0.2)
        else:
            ax.plot(epochs, values, color=colors[i], linewidth=2, label=run_name)
        
        # 標記最小值
        min_value = min(values)
        min_epoch = values.index(min_value) + 1
        
        ax.plot(min_epoch, min_value, 'o', color=colors[i], markersize=6)
        ax.annotate(f'{min_value:.4f}',
                  xy=(min_epoch, min_value),
                  xytext=(min_epoch + 0.5, min_value * 1.05),
                  color=colors[i],
                  arrowprops=dict(facecolor=colors[i], shrink=0.05, width=1))
    
    # 設置對數尺度（如果指定）
    if log_scale:
        ax.set_yscale('log')
    
    # 設置標題和標籤
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加圖例
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 調整佈局
    plt.tight_layout()
    
    return fig

def plot_training_radar(runs: Dict[str, Dict[str, float]], 
                       metrics: List[str],
                       title: str = 'Training Performance Comparison',
                       figsize: Tuple[int, int] = (10, 10),
                       higher_is_better: Dict[str, bool] = None) -> plt.Figure:
    """
    使用雷達圖比較多次訓練運行的性能。
    
    Args:
        runs (Dict[str, Dict[str, float]]): 包含多次訓練運行性能指標的字典
                                         格式為 {run_name: {'metric1': value1, 'metric2': value2, ...}}
        metrics (List[str]): 要比較的指標名稱列表
        title (str, optional): 圖表標題. 默認為'Training Performance Comparison'
        figsize (Tuple[int, int], optional): 圖表尺寸. 默認為(10, 10)
        higher_is_better (Dict[str, bool], optional): 指定每個指標較高的值是否更好。
                                                   默認為None，此時所有指標都假設較高的值更好。
    
    Returns:
        plt.Figure: Matplotlib圖表對象
    """
    if not runs or not metrics:
        logger.warning("沒有提供訓練運行數據或指標列表")
        return plt.figure(figsize=figsize)
    
    # 預設高於值是否更好
    if higher_is_better is None:
        higher_is_better = {metric: True for metric in metrics}
    
    # 確保所有要比較的指標都可用
    available_metrics = []
    for metric in metrics:
        if all(metric in run_data for run_data in runs.values()):
            available_metrics.append(metric)
        else:
            logger.warning(f"指標 '{metric}' 在某些運行中不可用")
    
    if not available_metrics:
        logger.warning("沒有所有運行都可用的指標")
        return plt.figure(figsize=figsize)
    
    # 歸一化指標值
    normalized_data = {}
    for metric in available_metrics:
        values = [run_data[metric] for run_data in runs.values()]
        min_val, max_val = min(values), max(values)
        
        # 處理最小值等於最大值的情況
        if min_val == max_val:
            for run_name in runs:
                if run_name not in normalized_data:
                    normalized_data[run_name] = {}
                normalized_data[run_name][metric] = 1.0
        else:
            for run_name, run_data in runs.items():
                if run_name not in normalized_data:
                    normalized_data[run_name] = {}
                
                # 歸一化，並根據higher_is_better反轉，使得更好的值總是更大
                if higher_is_better.get(metric, True):
                    normalized_data[run_name][metric] = (run_data[metric] - min_val) / (max_val - min_val)
                else:
                    normalized_data[run_name][metric] = (max_val - run_data[metric]) / (max_val - min_val)
    
    # 準備雷達圖數據
    angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 閉合圓
    
    # 準備指標標籤
    labels = available_metrics.copy()
    labels = np.concatenate((labels, [labels[0]]))  # 閉合圓
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # 使用各種顏色繪製每次運行的雷達圖
    colors = plt.cm.tab10(range(len(runs)))
    
    for i, (run_name, run_data) in enumerate(normalized_data.items()):
        values = [run_data[metric] for metric in available_metrics]
        values = np.concatenate((values, [values[0]]))  # 閉合圓
        
        ax.plot(angles, values, color=colors[i], linewidth=2, label=run_name)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    # 設置角度刻度位置
    ax.set_xticks(angles[:-1])
    
    # 設置標籤
    ax.set_xticklabels(labels[:-1])
    
    # 設置y軸範圍
    ax.set_ylim(0, 1.1)
    
    # 設置標題
    ax.set_title(title)
    
    # 添加圖例
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    
    # 調整佈局
    plt.tight_layout()
    
    return fig 