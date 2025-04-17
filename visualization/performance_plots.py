"""
# 性能視覺化
# 提供用於繪製模型訓練性能指標的函數
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import os


def plot_learning_curve(
    training_history: Dict[str, List],
    metrics: Union[str, List[str]],
    title: str = 'Learning Curve',
    xlabel: str = 'Epoch',
    ylabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    grid: bool = True,
    save_path: Optional[str] = None,
    experiment_id: Optional[str] = None,
    dpi: int = 100,
    smooth_factor: Optional[float] = None,
    log_scale: bool = False,
    val_prefix: str = 'val_',
    colors: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    markers: Optional[List[str]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    繪製訓練歷史中的學習曲線。
    
    Args:
        training_history: 訓練歷史字典，通常包含 'loss'、'val_loss' 等鍵
        metrics: 要繪製的指標名稱，可以是單個字符串或多個字符串的列表
        title: 圖表標題
        xlabel: X 軸標籤
        ylabel: Y 軸標籤，如果為 None，將根據指標自動設置
        ax: 可選的 Matplotlib Axes 實例
        figsize: 圖表尺寸
        grid: 是否顯示網格
        save_path: 保存圖表的路徑
        experiment_id: 實驗 ID，用於生成文件名
        dpi: 圖表分辨率
        smooth_factor: 平滑因子，表示每個點的移動平均窗口大小（0-1之間的小數）
        log_scale: 是否使用對數刻度
        val_prefix: 驗證指標的前綴
        colors: 線條顏色列表
        linestyles: 線條樣式列表
        markers: 標記樣式列表
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: 包含圖表和軸的元組
        
    Description:
        繪製模型訓練過程中的學習曲線，可以同時繪製多個指標，並支持平滑和對數刻度等選項。
        
    References:
        - Matplotlib Line Plots: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    """
    # 確保 metrics 是列表
    if isinstance(metrics, str):
        metrics = [metrics]
    
    # 創建或使用現有的軸
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # 設置默認顏色循環
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics) * 2))
    
    # 設置默認線條樣式和標記
    if linestyles is None:
        linestyles = ['-', '--']
    if markers is None:
        markers = ['o', '^', 's', 'd', 'x', '+']
    
    # 繪製每個指標的曲線
    all_lines = []
    for i, metric in enumerate(metrics):
        # 訓練曲線
        if metric in training_history:
            color_idx = i % len(colors)
            marker_idx = i % len(markers)
            
            # 獲取數據
            epochs = range(1, len(training_history[metric]) + 1)
            values = training_history[metric]
            
            # 如果需要平滑
            if smooth_factor is not None and smooth_factor > 0:
                values = smooth_curve(values, smooth_factor)
            
            # 繪製訓練曲線
            line_train, = ax.plot(
                epochs, values,
                linestyle=linestyles[0],
                color=colors[color_idx],
                marker=markers[marker_idx],
                markevery=max(1, len(epochs) // 10),
                markersize=6,
                label=f'{metric.capitalize()}'
            )
            all_lines.append(line_train)
            
            # 繪製驗證曲線
            val_metric = f"{val_prefix}{metric}"
            if val_metric in training_history:
                val_values = training_history[val_metric]
                
                # 如果需要平滑
                if smooth_factor is not None and smooth_factor > 0:
                    val_values = smooth_curve(val_values, smooth_factor)
                
                line_val, = ax.plot(
                    epochs, val_values,
                    linestyle=linestyles[1],
                    color=colors[color_idx],
                    marker=markers[(marker_idx + len(markers) // 2) % len(markers)],
                    markevery=max(1, len(epochs) // 10),
                    markersize=6,
                    label=f'Val {metric.capitalize()}'
                )
                all_lines.append(line_val)
    
    # 設置圖表標題和標籤
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    
    # 如果沒有提供 Y 軸標籤，根據指標自動設置
    if ylabel is None:
        if len(metrics) == 1:
            ylabel = metrics[0].capitalize()
        else:
            ylabel = 'Value'
    ax.set_ylabel(ylabel, fontsize=12)
    
    # 設置對數刻度
    if log_scale:
        ax.set_yscale('log')
    
    # 顯示網格
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加圖例
    if all_lines:
        ax.legend(handles=all_lines, loc='best')
    
    # 調整布局
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
            # 用下划線連接多個指標名稱
            metrics_str = '_'.join(metrics)
            filename = f"{experiment_id}_learning_curve_{metrics_str}{ext}"
            save_path = os.path.join(base_dir, filename)
        
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    return fig, ax


def smooth_curve(points: List[float], factor: float = 0.6) -> np.ndarray:
    """
    使用指數移動平均平滑曲線。
    
    Args:
        points: 原始數據點
        factor: 平滑因子，表示過去值的權重
        
    Returns:
        np.ndarray: 平滑後的數據點
        
    Description:
        使用指數移動平均來平滑數據，factor 值越大，平滑效果越強。
        
    References:
        None
    """
    # 轉換為 numpy 數組
    points = np.array(points)
    
    # 應用指數移動平均
    smoothed_points = np.zeros_like(points, dtype=float)
    for i in range(len(points)):
        if i == 0:
            smoothed_points[i] = points[i]
        else:
            smoothed_points[i] = factor * smoothed_points[i-1] + (1 - factor) * points[i]
    
    return smoothed_points


def plot_metric_comparison(
    metrics_data: Dict[str, Dict[str, Union[float, List[float]]]],
    title: str = 'Metrics Comparison',
    xlabel: str = 'Experiment',
    ylabel: str = 'Value',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (12, 8),
    grid: bool = True,
    save_path: Optional[str] = None,
    experiment_id: Optional[str] = None,
    dpi: int = 100,
    plot_type: str = 'bar',
    sort_by: Optional[str] = None,
    colors: Optional[List[str]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    比較多個實驗的指標值。
    
    Args:
        metrics_data: 實驗指標數據，格式為 {實驗名: {指標名: 值}}
        title: 圖表標題
        xlabel: X 軸標籤
        ylabel: Y 軸標籤
        ax: 可選的 Matplotlib Axes 實例
        figsize: 圖表尺寸
        grid: 是否顯示網格
        save_path: 保存圖表的路徑
        experiment_id: 實驗 ID，用於生成文件名
        dpi: 圖表分辨率
        plot_type: 圖表類型，可選 'bar' 或 'radar'
        sort_by: 可選的排序指標名
        colors: 顏色列表
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: 包含圖表和軸的元組
        
    Description:
        比較多個實驗的指標值，可以使用條形圖或雷達圖進行可視化。
        
    References:
        None
    """
    # 檢查輸入數據格式
    if not metrics_data:
        raise ValueError("metrics_data 不能為空")
    
    # 獲取所有唯一的指標名
    all_metrics = set()
    for exp_metrics in metrics_data.values():
        all_metrics.update(exp_metrics.keys())
    all_metrics = sorted(list(all_metrics))
    
    # 獲取實驗名列表，可能需要排序
    experiment_names = list(metrics_data.keys())
    if sort_by is not None and sort_by in all_metrics:
        # 根據指定的指標排序
        experiment_names.sort(
            key=lambda x: metrics_data[x].get(sort_by, 0) 
            if isinstance(metrics_data[x].get(sort_by, 0), (int, float)) 
            else np.mean(metrics_data[x].get(sort_by, [0]))
        )
    
    # 處理每個指標可能是單個值或列表的情況
    # 對於列表，計算均值
    processed_data = {}
    for exp_name, exp_metrics in metrics_data.items():
        processed_data[exp_name] = {}
        for metric, value in exp_metrics.items():
            if isinstance(value, list):
                processed_data[exp_name][metric] = np.mean(value)
            else:
                processed_data[exp_name][metric] = value
    
    # 選擇繪圖類型
    if plot_type == 'bar':
        return _plot_bar_comparison(
            processed_data, experiment_names, all_metrics,
            title, xlabel, ylabel, ax, figsize, grid,
            save_path, experiment_id, dpi, colors
        )
    elif plot_type == 'radar':
        return _plot_radar_comparison(
            processed_data, experiment_names, all_metrics,
            title, ax, figsize, save_path, experiment_id, dpi, colors
        )
    else:
        raise ValueError(f"不支持的繪圖類型: {plot_type}，僅支持 'bar' 或 'radar'")


def _plot_bar_comparison(
    processed_data: Dict[str, Dict[str, float]],
    experiment_names: List[str],
    all_metrics: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    ax: Optional[plt.Axes],
    figsize: Tuple[int, int],
    grid: bool,
    save_path: Optional[str],
    experiment_id: Optional[str],
    dpi: int,
    colors: Optional[List[str]]
) -> Tuple[plt.Figure, plt.Axes]:
    """繪製條形圖比較"""
    # 創建或使用現有的軸
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # 設置默認顏色
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_metrics)))
    
    # 計算條形圖的位置
    num_metrics = len(all_metrics)
    num_experiments = len(experiment_names)
    bar_width = 0.8 / num_metrics
    indices = np.arange(num_experiments)
    
    # 繪製每個指標的條形
    for i, metric in enumerate(all_metrics):
        # 收集所有實驗的該指標值
        values = [processed_data[exp].get(metric, 0) for exp in experiment_names]
        
        # 繪製條形
        ax.bar(
            indices + i * bar_width - 0.4 + bar_width / 2,
            values,
            bar_width,
            label=metric,
            color=colors[i % len(colors)],
            edgecolor='black',
            linewidth=0.8
        )
    
    # 設置 X 軸標籤和刻度
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(indices)
    ax.set_xticklabels(experiment_names, rotation=45, ha='right')
    
    # 顯示網格
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 添加圖例
    ax.legend()
    
    # 調整布局
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
            filename = f"{experiment_id}_metrics_comparison{ext}"
            save_path = os.path.join(base_dir, filename)
        
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    return fig, ax


def _plot_radar_comparison(
    processed_data: Dict[str, Dict[str, float]],
    experiment_names: List[str],
    all_metrics: List[str],
    title: str,
    ax: Optional[plt.Axes],
    figsize: Tuple[int, int],
    save_path: Optional[str],
    experiment_id: Optional[str],
    dpi: int,
    colors: Optional[List[str]]
) -> Tuple[plt.Figure, plt.Axes]:
    """繪製雷達圖比較"""
    try:
        import matplotlib.pyplot as plt
        
        # 計算角度
        num_metrics = len(all_metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # 閉合雷達圖
        
        # 處理數據：標準化所有指標
        # 獲取每個指標的最大值和最小值
        min_values = {}
        max_values = {}
        for metric in all_metrics:
            values = [exp_metrics.get(metric, 0) for exp_metrics in processed_data.values()]
            min_values[metric] = min(values)
            max_values[metric] = max(values)
        
        # 標準化數據（映射到 0-1 範圍）
        normalized_data = {}
        for exp_name, exp_metrics in processed_data.items():
            normalized_data[exp_name] = {}
            for metric in all_metrics:
                value = exp_metrics.get(metric, 0)
                min_val = min_values[metric]
                max_val = max_values[metric]
                if max_val == min_val:  # 避免除以零
                    normalized_data[exp_name][metric] = 0.5
                else:
                    normalized_data[exp_name][metric] = (value - min_val) / (max_val - min_val)
        
        # 創建或使用現有的軸
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        else:
            fig = ax.figure
        
        # 設置默認顏色
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_names)))
        
        # 繪製每個實驗的雷達圖
        for i, exp_name in enumerate(experiment_names):
            # 準備數據
            values = [normalized_data[exp_name].get(metric, 0) for metric in all_metrics]
            values += values[:1]  # 閉合雷達圖
            
            # 繪製雷達線和填充
            ax.plot(
                angles, values, 'o-',
                linewidth=2, label=exp_name,
                color=colors[i % len(colors)]
            )
            ax.fill(
                angles, values, alpha=0.1,
                color=colors[i % len(colors)]
            )
        
        # 設置刻度和標籤
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_metrics)
        
        # 調整布局
        ax.set_title(title, fontsize=14, pad=20)
        
        # 添加圖例
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 調整布局
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
                filename = f"{experiment_id}_radar_comparison{ext}"
                save_path = os.path.join(base_dir, filename)
            
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        
        return fig, ax
    
    except Exception as e:
        print(f"繪製雷達圖時出錯: {str(e)}")
        # 如果雷達圖繪製失敗，退回到條形圖
        return _plot_bar_comparison(
            processed_data, experiment_names, all_metrics,
            title, "Experiment", "Value", ax, figsize, True,
            save_path, experiment_id, dpi, colors
        ) 