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
from utils.stat_utils import calculate_moving_average, detect_outliers, smooth_curve

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

def plot_learning_stages(loss_values: List[float], 
                        learning_stages: List[Dict[str, Any]],
                        title: str = 'Learning Stages Analysis',
                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    視覺化訓練過程中的不同學習階段。
    
    Args:
        loss_values (List[float]): 訓練損失值列表。
        learning_stages (List[Dict[str, Any]]): 學習階段列表，每個階段是包含'type'、'start'、'end'、'avg_slope'等鍵的字典。
        title (str, optional): 圖表標題. 默認為'Learning Stages Analysis'
        figsize (Tuple[int, int], optional): 圖表尺寸. 默認為(12, 8)
        
    Returns:
        plt.Figure: Matplotlib圖表對象
    """
    if not loss_values or not learning_stages:
        logger.warning("沒有提供損失值或學習階段數據")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 繪製損失曲線
    epochs = range(1, len(loss_values) + 1)
    ax.plot(epochs, loss_values, 'b-', label='Training Loss', alpha=0.7)
    
    # 定義不同階段的顏色映射
    stage_colors = {
        'initial': 'green',
        'plateau': 'red',
        'acceleration': 'purple',
        'slow_progress': 'orange',
        'convergence': 'blue'
    }
    
    # 繪製不同學習階段的背景色
    for i, stage in enumerate(learning_stages):
        stage_type = stage.get('type', f'stage_{i}')
        start = max(0, stage.get('start', 0))
        end = min(len(loss_values) - 1, stage.get('end', len(loss_values) - 1))
        
        # 轉換為以1開始的輪次索引
        start_epoch = start + 1
        end_epoch = end + 1
        
        # 獲取階段顏色
        color = stage_colors.get(stage_type, plt.cm.tab10(i % 10))
        
        # 使用矩形填充背景色
        ax.axvspan(start_epoch, end_epoch, alpha=0.2, color=color, label=f'{stage_type.capitalize()} ({start_epoch}-{end_epoch})')
        
        # 添加階段標籤
        mid_point = (start_epoch + end_epoch) / 2
        y_pos = min(loss_values[start:end+1]) * 0.9  # 標籤位置在階段最小值下方
        ax.text(mid_point, y_pos, stage_type.capitalize(), 
              ha='center', va='top', fontsize=10, 
              bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.3))
    
    # 添加收斂線性趨勢
    for i, stage in enumerate(learning_stages):
        start = max(0, stage.get('start', 0))
        end = min(len(loss_values) - 1, stage.get('end', len(loss_values) - 1))
        
        if end > start:
            # 根據該階段的斜率添加線性趨勢線
            x = np.array(range(start + 1, end + 2))  # 轉換為以1開始的輪次索引
            avg_slope = stage.get('avg_slope', 0)
            
            if avg_slope != 0:
                # 計算趨勢線
                y_start = loss_values[start]
                trend_line = [y_start + avg_slope * (i - start) for i in range(start, end + 1)]
                
                # 繪製趨勢線
                ax.plot(x, trend_line, '--', color=stage_colors.get(stage.get('type', f'stage_{i}'), plt.cm.tab10(i % 10)),
                      linewidth=1, alpha=0.8)
    
    # 設置y軸範圍
    min_loss = min(loss_values) * 0.8  # 留出空間顯示標籤
    max_loss = max(loss_values) * 1.1
    ax.set_ylim(min_loss, max_loss)
    
    # 設置標題和標籤
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 調整佈局
    handles, labels = ax.get_legend_handles_labels()
    # 去除重複的圖例項
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels, loc='upper right')
    
    plt.tight_layout()
    
    return fig

def plot_learning_efficiency_comparison(runs: Dict[str, Dict[str, Dict]], 
                                      metrics: List[str] = ['loss_reduction_efficiency', 'convergence_percent'],
                                      title: str = 'Learning Efficiency Comparison',
                                      figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    比較多個訓練運行的學習效率指標。
    
    Args:
        runs (Dict[str, Dict[str, Dict]]): 包含多個訓練運行學習效率分析結果的字典，
                                         格式為 {run_name: {'efficiency_metrics': {...}, 'convergence_metrics': {...}, ...}}
        metrics (List[str], optional): 要比較的指標列表. 默認為['loss_reduction_efficiency', 'convergence_percent']
        title (str, optional): 圖表標題. 默認為'Learning Efficiency Comparison'
        figsize (Tuple[int, int], optional): 圖表尺寸. 默認為(12, 8)
        
    Returns:
        plt.Figure: Matplotlib圖表對象
    """
    if not runs or not metrics:
        logger.warning("沒有提供訓練運行數據或指標列表")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
    
    # 提取要比較的指標數據
    comparison_data = {}
    
    for run_name, run_data in runs.items():
        comparison_data[run_name] = {}
        
        for metric in metrics:
            # 找到指標所在的字典
            for category in ['efficiency_metrics', 'convergence_metrics', 'time_metrics', 'generalization_metrics']:
                if category in run_data and metric in run_data[category]:
                    comparison_data[run_name][metric] = run_data[category][metric]
                    break
    
    # 創建圖表
    n_metrics = len(metrics)
    n_runs = len(runs)
    
    # 使用雷達圖或柱狀圖，取決於指標數量
    if n_metrics >= 3:
        # 雷達圖適用於3個或更多指標
        # 使用已有的plot_training_radar函數
        # 先規範化數據
        normalized_runs = {}
        for run_name, run_metrics in comparison_data.items():
            normalized_runs[run_name] = {metric: run_metrics.get(metric, 0) for metric in metrics}
        
        fig = plot_training_radar(normalized_runs, metrics, title=title, figsize=figsize)
    else:
        # 柱狀圖適用於1-2個指標
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(n_metrics)  # 指標位置
        width = 0.8 / n_runs  # 柱狀寬度
        
        for i, (run_name, run_metrics) in enumerate(comparison_data.items()):
            values = [run_metrics.get(metric, 0) for metric in metrics]
            ax.bar(x - 0.4 + (i + 0.5) * width, values, width, label=run_name, alpha=0.7)
        
        # 設置標題和標籤
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        plt.tight_layout()
    
    return fig

def plot_lr_schedule_impact(lr_impact_analysis: Dict[str, Any], 
                          title: str = 'Learning Rate Schedule Impact Analysis',
                          figsize: Tuple[int, int] = (14, 10),
                          use_english: bool = False) -> plt.Figure:
    """
    可視化學習率調度對學習效率的影響分析結果。
    
    Args:
        lr_impact_analysis (Dict[str, Any]): 由analyze_lr_schedule_impact函數產生的分析結果
        title (str, optional): 圖表標題. 默認為'Learning Rate Schedule Impact Analysis'
        figsize (Tuple[int, int], optional): 圖表尺寸. 默認為(14, 10)
        use_english (bool, optional): 是否使用英文標籤. 默認為False
    
    Returns:
        plt.Figure: Matplotlib圖表對象
    """
    if not lr_impact_analysis or not lr_impact_analysis.get('lr_schedules'):
        logger.warning("沒有提供學習率調度分析數據")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
    
    run_names = list(lr_impact_analysis['lr_schedules'].keys())
    n_runs = len(run_names)
    
    # 創建2x2佈局的圖表
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    
    # 1. 學習率調度類型與性能比較
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 組織數據
    lr_types = {name: lr_impact_analysis['lr_schedules'][name]['type'] for name in run_names}
    min_losses = lr_impact_analysis['comparative_analysis']['min_loss_by_run']
    epochs_to_min = lr_impact_analysis['comparative_analysis']['epochs_to_min_loss']
    
    # 按學習率調度類型分組
    grouped_data = {}
    for name, lr_type in lr_types.items():
        if lr_type not in grouped_data:
            grouped_data[lr_type] = []
        grouped_data[lr_type].append((name, min_losses[name], epochs_to_min[name]))
    
    # 繪製條形圖
    x_pos = np.arange(n_runs)
    bars = ax1.bar(x_pos, [min_losses[name] for name in run_names], width=0.6, alpha=0.7)
    
    # 設置條形顏色按學習率調度類型
    colors = {
        'constant': 'blue', 
        'step': 'green', 
        'cyclic': 'red', 
        'custom': 'orange',
        'warmup': 'purple'
    }
    for i, name in enumerate(run_names):
        bars[i].set_color(colors.get(lr_types[name], 'gray'))
    
    # 添加標籤和圖例
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{name}\n({lr_types[name]})" for name in run_names], rotation=45)
    ax1.set_ylabel('Minimum Loss')
    ax1.set_title('Minimum Loss by Learning Rate Schedule Type')
    
    # 創建顏色標記的圖例
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors.values()]
    labels = list(colors.keys())
    ax1.legend(handles, labels, title="Schedule Type")
    
    # 2. 收斂速度比較
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 繪製收斂輪次條形圖
    convergence_epochs = lr_impact_analysis['comparative_analysis'].get('convergence_epochs', {})
    valid_runs = [name for name in run_names if name in convergence_epochs]
    
    if valid_runs:
        x_pos_conv = np.arange(len(valid_runs))
        bars_conv = ax2.bar(x_pos_conv, [convergence_epochs[name] for name in valid_runs], 
                          width=0.6, alpha=0.7)
        
        # 設置條形顏色
        for i, name in enumerate(valid_runs):
            bars_conv[i].set_color(colors.get(lr_types[name], 'gray'))
        
        ax2.set_xticks(x_pos_conv)
        ax2.set_xticklabels([f"{name}\n({lr_types[name]})" for name in valid_runs], rotation=45)
        ax2.set_ylabel('Convergence Epoch')
        ax2.set_title('Convergence Speed by Learning Rate Schedule Type')
    else:
        ax2.text(0.5, 0.5, "No convergence data available", ha='center', va='center')
    
    # 3. 學習率變化與損失相關性
    ax3 = fig.add_subplot(gs[1, :])
    
    # 為每個運行創建一個子圖
    for i, name in enumerate(run_names):
        schedule_info = lr_impact_analysis['lr_schedules'][name]
        corr = schedule_info['lr_loss_correlation']
        changes = schedule_info['changes_count']
        lr_range = schedule_info['lr_range']
        lr_type = lr_types[name]
        
        # 根據學習率調度類型顯示合適的描述
        type_descriptions = {
            'constant': {'zh': '固定學習率', 'en': 'Constant LR'},
            'step': {'zh': '階梯式下降', 'en': 'Step Decay'},
            'cyclic': {'zh': '循環變化', 'en': 'Cyclic Variation'},
            'custom': {'zh': '自定義變化', 'en': 'Custom Schedule'},
            'warmup': {'zh': '預熱式上升', 'en': 'Warmup Schedule'}
        }
        
        type_description = type_descriptions.get(lr_type, {'zh': lr_type, 'en': lr_type})
        display_type = type_description['en'] if use_english else type_description['zh']
        
        # 在圖表上放置一個註釋方框
        props = dict(boxstyle='round', facecolor=colors.get(lr_type, 'gray'), alpha=0.2)
        
        if use_english:
            text_info = (f"Type: {display_type}\n"
                       f"LR-Loss Correlation: {corr:.3f}\n"
                       f"LR Changes: {changes}\n"
                       f"LR Range: {lr_range:.6f}")
        else:
            text_info = (f"Type/類型: {display_type}\n"
                       f"LR-Loss Correlation/相關性: {corr:.3f}\n"
                       f"LR Changes/變化次數: {changes}\n"
                       f"LR Range/範圍: {lr_range:.6f}")
        
        x_pos = 0.1 + 0.8 * (i / (n_runs or 1))  # 平均分佈在x軸上
        ax3.text(x_pos, 0.5, text_info, transform=ax3.transAxes, fontsize=9,
               bbox=props, ha='center', va='center')
    
    ax3.set_title('Learning Rate Schedule Characteristics')
    ax3.axis('off')  # 不顯示座標軸
    
    # 添加推薦信息
    if 'recommendations' in lr_impact_analysis and lr_impact_analysis['recommendations']:
        recommendations = lr_impact_analysis['recommendations']
        rec_text = "\n".join([f"• {rec}" for rec in recommendations])
        
        props = dict(boxstyle='round', facecolor='lightgray', alpha=0.5)
        rec_title = "Recommendations:" if use_english else "建議/Recommendations:"
        ax3.text(0.5, 0.15, f"{rec_title}\n{rec_text}",
               transform=ax3.transAxes, fontsize=10,
               bbox=props, ha='center', va='center')
    
    # 設置總標題
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 為suptitle留出空間
    
    return fig 