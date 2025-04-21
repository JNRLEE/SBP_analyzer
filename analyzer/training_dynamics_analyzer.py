"""
訓練動態分析模組。

此模組提供用於分析深度學習模型訓練過程動態的功能，包括損失函數變化、評估指標趨勢等。

Classes:
    TrainingDynamicsAnalyzer: 分析訓練動態的類別。
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base_analyzer import BaseAnalyzer
from metrics.performance_metrics import (
    analyze_loss_curve, 
    analyze_metric_curve,
    analyze_training_efficiency,
    evaluate_convergence,
    detect_performance_anomalies,
    analyze_learning_efficiency,
    analyze_lr_schedule_impact
)
from utils.stat_utils import (
    calculate_correlation, 
    calculate_moving_average, 
    find_convergence_point,
    calculate_stability_metrics
)

class TrainingDynamicsAnalyzer(BaseAnalyzer):
    """
    訓練動態分析器，用於分析模型訓練過程中的性能指標變化。
    
    此分析器關注訓練過程中損失函數、評估指標等數據的變化趨勢，可用於評估模型收斂性、穩定性等。
    
    Attributes:
        experiment_dir (str): 實驗結果的目錄路徑。
        training_history (Dict): 訓練歷史記錄。
        metrics_data (pd.DataFrame): 處理後的指標數據。
    """
    
    def __init__(self, experiment_dir: str):
        """
        初始化訓練動態分析器。
        
        Args:
            experiment_dir (str): 實驗結果的目錄路徑。
        """
        super().__init__(experiment_dir)
        self.training_history = {}
        self.metrics_data = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_training_history(self) -> Dict:
        """
        從training_history.json載入訓練歷史。
        
        Returns:
            Dict: 包含訓練歷史的字典。
        """
        history_path = os.path.join(self.experiment_dir, 'training_history.json')
        try:
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
                return self.training_history
        except Exception as e:
            self.logger.error(f"載入訓練歷史文件失敗: {e}")
            raise
    
    def preprocess_metrics(self) -> pd.DataFrame:
        """
        預處理訓練指標數據，轉換為易於分析的格式。
        
        Returns:
            pd.DataFrame: 包含處理後指標數據的DataFrame。
        """
        if not self.training_history:
            self.load_training_history()
        
        # 將訓練歷史轉換為DataFrame格式
        metrics_data = pd.DataFrame()
        
        # 處理常見的指標類型
        epoch_column = []
        metrics_dict = {}
        
        # 確定有多少輪次
        max_epochs = 0
        for key, values in self.training_history.items():
            if isinstance(values, list):
                max_epochs = max(max_epochs, len(values))
        
        # 創建輪次列
        epoch_column = list(range(max_epochs))
        metrics_dict['epoch'] = epoch_column
        
        # 提取訓練歷史中的所有指標
        for key, values in self.training_history.items():
            if isinstance(values, list):
                # 確保長度一致，如果需要，用NaN填充
                if len(values) < max_epochs:
                    values = values + [float('nan')] * (max_epochs - len(values))
                metrics_dict[key] = values[:max_epochs]
        
        # 創建DataFrame
        metrics_data = pd.DataFrame(metrics_dict)
        
        # 設置輪次為索引
        if 'epoch' in metrics_data.columns:
            metrics_data.set_index('epoch', inplace=True)
        
        # 存儲處理後的數據
        self.metrics_data = metrics_data
        return metrics_data
    
    def analyze(self, metrics: Optional[List[str]] = None, save_results: bool = True) -> Dict:
        """
        分析訓練動態並計算相關統計信息。
        
        Args:
            metrics (List[str], optional): 要分析的指標列表。如果為None，則分析所有可用指標。
            save_results (bool, optional): 是否保存分析結果。默認為True。
        
        Returns:
            Dict: 包含分析結果的字典。
        """
        # 載入訓練歷史並預處理
        self.preprocess_metrics()
        
        # 初始化結果字典
        self.results = {
            'convergence_analysis': {},
            'stability_analysis': {},
            'trend_analysis': {},
            'metric_correlations': {},
            'training_efficiency': {},
            'anomaly_detection': {},
            'enhanced_learning_efficiency': {}
        }
        
        try:
            # 確定要分析的指標
            if not metrics:
                # 默認分析所有可用指標
                metrics = self.metrics_data.columns.tolist()
            else:
                # 僅分析指定指標中可用的部分
                metrics = [m for m in metrics if m in self.metrics_data.columns]
            
            # 獲取訓練時間
            training_time = self.training_history.get('training_time')
            
            # 主要分析損失指標
            loss_metrics = [m for m in metrics if 'loss' in m.lower()]
            
            # 如果存在訓練損失，優先分析訓練損失
            primary_loss = None
            for loss_name in ['train_loss', 'loss', 'training_loss']:
                if loss_name in loss_metrics:
                    primary_loss = loss_name
                    break
            
            # 如果沒有找到訓練損失，選擇第一個損失指標
            if not primary_loss and loss_metrics:
                primary_loss = loss_metrics[0]
            
            # 分析主要損失指標
            if primary_loss:
                loss_values = self.metrics_data[primary_loss].dropna().tolist()
                
                # 1. 收斂性分析
                self.results['convergence_analysis'][primary_loss] = self.analyze_convergence(primary_loss)
                
                # 2. 穩定性分析
                self.results['stability_analysis'][primary_loss] = self.analyze_stability(primary_loss)
                
                # 3. 趨勢分析
                loss_trend_analysis = analyze_loss_curve(loss_values)
                self.results['trend_analysis'][primary_loss] = loss_trend_analysis
                
                # 4. 訓練效率分析
                training_efficiency = analyze_training_efficiency(loss_values, training_time)
                self.results['training_efficiency'] = training_efficiency
                
                # 5. 異常檢測
                anomalies = detect_performance_anomalies(loss_values)
                self.results['anomaly_detection'][primary_loss] = anomalies
            
            # 分析驗證損失（如果存在）
            val_loss = None
            for loss_name in ['val_loss', 'validation_loss']:
                if loss_name in loss_metrics:
                    val_loss = loss_name
                    break
            
            if val_loss:
                val_loss_values = self.metrics_data[val_loss].dropna().tolist()
                
                # 收斂性分析
                self.results['convergence_analysis'][val_loss] = self.analyze_convergence(val_loss)
                
                # 穩定性分析
                self.results['stability_analysis'][val_loss] = self.analyze_stability(val_loss)
                
                # 趨勢分析
                val_loss_trend_analysis = analyze_loss_curve(val_loss_values)
                self.results['trend_analysis'][val_loss] = val_loss_trend_analysis
                
                # 異常檢測
                val_anomalies = detect_performance_anomalies(val_loss_values)
                self.results['anomaly_detection'][val_loss] = val_anomalies
                
                # 如果有訓練損失和驗證損失，進行增強的學習效率分析
                if primary_loss:
                    # 提取學習率數據（如果存在）
                    learning_rates = None
                    for lr_name in ['lr', 'learning_rate', 'optimizer_lr']:
                        if lr_name in self.metrics_data.columns:
                            learning_rates = self.metrics_data[lr_name].dropna().tolist()
                            break
                    
                    # 進行增強的學習效率分析
                    learning_efficiency = analyze_learning_efficiency(
                        loss_values=loss_values,
                        val_loss_values=val_loss_values,
                        training_time=training_time,
                        learning_rates=learning_rates
                    )
                    
                    # 添加到結果中
                    self.results['enhanced_learning_efficiency'] = learning_efficiency
            
            # 分析其他非損失指標
            evaluation_metrics = [m for m in metrics if 'loss' not in m.lower()]
            for metric in evaluation_metrics:
                metric_values = self.metrics_data[metric].dropna().tolist()
                if not metric_values:
                    continue
                
                # 判斷指標是否越高越好
                higher_is_better = True  # 默認大多數評估指標是越高越好
                if any(term in metric.lower() for term in ['error', 'distance', 'divergence']):
                    higher_is_better = False
                
                # 趨勢分析
                metric_trend_analysis = analyze_metric_curve(metric_values, higher_is_better)
                self.results['trend_analysis'][metric] = metric_trend_analysis
                
                # 穩定性分析
                self.results['stability_analysis'][metric] = self.analyze_stability(metric)
                
                # 異常檢測
                metric_anomalies = detect_performance_anomalies(metric_values)
                self.results['anomaly_detection'][metric] = metric_anomalies
            
            # 計算指標相關性
            if len(metrics) > 1:
                self.results['metric_correlations'] = self.get_metric_correlation()
            
            # 計算整體訓練摘要
            self.results['training_summary'] = self._generate_training_summary()
            
            if save_results:
                self.save_results(os.path.join(self.experiment_dir, 'analysis'))
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"分析訓練動態時出錯: {e}")
            raise
    
    def analyze_convergence(self, metric: str = 'train_loss') -> Dict:
        """
        分析指定指標的收斂性。
        
        Args:
            metric (str, optional): 要分析的指標名稱。默認為'train_loss'。
        
        Returns:
            Dict: 包含收斂分析結果的字典。
        """
        if metric not in self.metrics_data.columns:
            return {
                'has_converged': False,
                'convergence_point': None,
                'convergence_epoch': None,
                'convergence_value': None,
                'total_epochs': None
            }
        
        # 獲取指標值
        values = self.metrics_data[metric].dropna().tolist()
        
        # 如果沒有數據，返回空結果
        if not values:
            return {
                'has_converged': False,
                'convergence_point': None,
                'convergence_epoch': None,
                'convergence_value': None,
                'total_epochs': 0
            }
        
        # 分析收斂性
        convergence_info = evaluate_convergence(values)
        
        # 獲取收斂點
        convergence_point = find_convergence_point(values)
        
        result = {
            'has_converged': convergence_info['has_converged'],
            'convergence_point': convergence_point,
            'convergence_epoch': convergence_point if convergence_point is not None else None,
            'convergence_value': values[convergence_point] if convergence_point is not None else None,
            'total_epochs': len(values),
            'min_value': min(values),
            'min_value_epoch': values.index(min(values)),
            'final_value': values[-1],
            'converged_percentage': (convergence_point / len(values) * 100) if convergence_point is not None else None
        }
        
        return result
    
    def analyze_stability(self, metric: str = 'train_loss', window_size: int = 5) -> Dict:
        """
        分析指定指標的穩定性。
        
        Args:
            metric (str, optional): 要分析的指標名稱。默認為'train_loss'。
            window_size (int, optional): 用於計算移動平均的窗口大小。默認為5。
        
        Returns:
            Dict: 包含穩定性分析結果的字典。
        """
        if metric not in self.metrics_data.columns:
            return {
                'variance': None,
                'moving_average': None,
                'mean': None,
                'std': None,
                'max_fluctuation': None
            }
        
        # 獲取指標值
        values = self.metrics_data[metric].dropna().tolist()
        
        # 如果沒有數據，返回空結果
        if not values:
            return {
                'variance': None,
                'moving_average': None,
                'mean': None,
                'std': None,
                'max_fluctuation': None
            }
        
        # 計算穩定性指標
        stability_metrics = calculate_stability_metrics(values, window_size)
        
        # 計算移動平均
        moving_avg = calculate_moving_average(values, window_size)
        
        # 計算最大波動
        if len(values) > 1:
            diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
            max_fluctuation = max(diffs)
            max_fluctuation_epoch = diffs.index(max_fluctuation) + 1  # +1 因為是與前一個值的差異
        else:
            max_fluctuation = 0
            max_fluctuation_epoch = None
        
        # 計算基本統計量
        mean_value = np.mean(values)
        std_value = np.std(values)
        variance = std_value ** 2
        
        result = {
            'variance': variance,
            'moving_average': moving_avg,
            'mean': mean_value,
            'std': std_value,
            'max_fluctuation': max_fluctuation,
            'max_fluctuation_epoch': max_fluctuation_epoch,
            'coefficient_of_variation': stability_metrics['coefficient_of_variation'],
            'overall_trend': stability_metrics['overall_trend']
        }
        
        return result
    
    def get_metric_correlation(self) -> pd.DataFrame:
        """
        計算不同指標之間的相關性。
        
        Returns:
            pd.DataFrame: 包含指標相關性的DataFrame。
        """
        if self.metrics_data is None or self.metrics_data.empty:
            return pd.DataFrame()
        
        # 填充缺失值，以便計算相關性
        filled_data = self.metrics_data.ffill().bfill()
        
        # 計算相關性矩陣
        correlation_matrix = filled_data.corr()
        
        # 提取每對指標的相關係數
        metric_pairs = []
        
        for i, metric1 in enumerate(correlation_matrix.columns):
            for j, metric2 in enumerate(correlation_matrix.columns):
                if i < j:  # 只保留上三角矩陣，避免重複
                    correlation = correlation_matrix.loc[metric1, metric2]
                    metric_pairs.append({
                        'metric1': metric1,
                        'metric2': metric2,
                        'correlation': correlation,
                        'strength': abs(correlation),
                        'direction': 'positive' if correlation > 0 else 'negative'
                    })
        
        # 按相關性強度排序
        metric_pairs.sort(key=lambda x: x['strength'], reverse=True)
        
        # 轉換為DataFrame
        correlation_df = pd.DataFrame(metric_pairs)
        
        return correlation_df

    def _generate_training_summary(self) -> Dict:
        """
        生成訓練過程的整體摘要。
        
        Returns:
            Dict: 包含訓練摘要的字典。
        """
        summary = {}
        
        # 獲取基本訓練信息
        total_epochs = 0
        for key, values in self.training_history.items():
            if isinstance(values, list):
                total_epochs = max(total_epochs, len(values))
        
        summary['total_epochs'] = total_epochs
        summary['training_time'] = self.training_history.get('training_time')
        
        # 如果有best_val_loss，添加到摘要
        if 'best_val_loss' in self.training_history:
            summary['best_val_loss'] = self.training_history['best_val_loss']
        
        # 收集最終指標值
        final_metrics = {}
        for key, values in self.training_history.items():
            if isinstance(values, list) and values:
                final_metrics[f'final_{key}'] = values[-1]
        
        summary['final_metrics'] = final_metrics
        
        # 計算平均每輪時間
        if summary['training_time'] and total_epochs > 0:
            summary['avg_time_per_epoch'] = summary['training_time'] / total_epochs
        
        return summary
    
    def visualize_metrics(self, metrics: Optional[List[str]] = None, 
                         output_path: Optional[str] = None) -> plt.Figure:
        """
        將訓練指標可視化為曲線圖。
        
        Args:
            metrics (List[str], optional): 要可視化的指標列表。如果為None，則使用所有可用指標。
            output_path (str, optional): 輸出圖像的路徑。如果為None，則僅返回圖像對象。
            
        Returns:
            plt.Figure: Matplotlib圖像對象。
        """
        # 確保已預處理指標數據
        if self.metrics_data is None:
            self.preprocess_metrics()
        
        # 確定要可視化的指標
        if not metrics:
            # 默認可視化主要指標
            metrics = []
            # 首先添加損失指標
            for col in self.metrics_data.columns:
                if 'loss' in col.lower():
                    metrics.append(col)
            # 然後添加其他指標
            for col in self.metrics_data.columns:
                if 'loss' not in col.lower():
                    metrics.append(col)
        else:
            # 僅可視化指定指標中可用的部分
            metrics = [m for m in metrics if m in self.metrics_data.columns]
        
        # 創建圖形
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            # 如果沒有可用指標，創建空圖
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No metrics available", ha='center', va='center')
            plt.close()
            return fig
        
        # 將損失指標和其他指標分開可視化
        loss_metrics = [m for m in metrics if 'loss' in m.lower()]
        other_metrics = [m for m in metrics if 'loss' not in m.lower()]
        
        # 確定子圖數量和布局
        n_subplots = 1 + (len(other_metrics) > 0)
        
        # 創建圖形和子圖
        fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 5 * n_subplots), sharex=True)
        
        # 確保axes始終是列表
        if n_subplots == 1:
            axes = [axes]
        
        # 繪製損失曲線
        ax = axes[0]
        for metric in loss_metrics:
            values = self.metrics_data[metric].dropna()
            ax.plot(values.index, values, label=metric)
        
        ax.set_title("Loss Metrics")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Value")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 繪製其他指標曲線
        if other_metrics:
            ax = axes[1]
            for metric in other_metrics:
                values = self.metrics_data[metric].dropna()
                ax.plot(values.index, values, label=metric)
            
            ax.set_title("Other Metrics")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Metric Value")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # 進行分析以標記重要點
        if self.results and loss_metrics:
            primary_loss = loss_metrics[0]
            
            # 標記收斂點
            if 'convergence_analysis' in self.results and primary_loss in self.results['convergence_analysis']:
                convergence_info = self.results['convergence_analysis'][primary_loss]
                convergence_point = convergence_info.get('convergence_point')
                
                if convergence_point is not None:
                    values = self.metrics_data[primary_loss].dropna()
                    if convergence_point < len(values):
                        axes[0].scatter([convergence_point], [values.iloc[convergence_point]], 
                                      color='red', s=100, zorder=5, label='Convergence Point')
                        axes[0].axvline(x=convergence_point, color='r', linestyle='--', alpha=0.5)
            
            # 標記最佳點
            if 'trend_analysis' in self.results and primary_loss in self.results['trend_analysis']:
                trend_info = self.results['trend_analysis'][primary_loss]
                min_loss_epoch = trend_info.get('min_loss_epoch')
                
                if min_loss_epoch is not None:
                    values = self.metrics_data[primary_loss].dropna()
                    if min_loss_epoch < len(values):
                        axes[0].scatter([min_loss_epoch], [values.iloc[min_loss_epoch]], 
                                      color='green', s=100, zorder=5, label='Best Model')
        
        # 調整布局
        plt.tight_layout()
        
        # 保存圖像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_learning_stages(self, output_path: Optional[str] = None) -> plt.Figure:
        """
        可視化學習階段分析結果。
        
        Args:
            output_path (str, optional): 輸出圖像的路徑。如果為None，則僅返回圖像對象。
            
        Returns:
            plt.Figure: Matplotlib圖像對象。
        """
        from visualization.performance_plots import plot_learning_stages
        
        # 檢查是否已進行學習效率分析
        if 'enhanced_learning_efficiency' not in self.results or not self.results['enhanced_learning_efficiency']:
            self.logger.warning("未找到學習效率分析結果，無法繪製學習階段")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No learning efficiency analysis results available", ha='center', va='center')
            return fig
        
        # 獲取主要損失指標
        primary_loss = None
        for loss_name in ['train_loss', 'loss', 'training_loss']:
            if loss_name in self.metrics_data.columns:
                primary_loss = loss_name
                break
        
        if not primary_loss:
            self.logger.warning("未找到主要損失指標，無法繪製學習階段")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No primary loss metric available", ha='center', va='center')
            return fig
        
        # 獲取損失值和學習階段
        loss_values = self.metrics_data[primary_loss].dropna().tolist()
        
        learning_efficiency_results = self.results['enhanced_learning_efficiency']
        
        if 'learning_dynamics' not in learning_efficiency_results or 'identified_stages' not in learning_efficiency_results['learning_dynamics']:
            self.logger.warning("未找到學習階段識別結果，無法繪製學習階段")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No learning stages identified", ha='center', va='center')
            return fig
        
        learning_stages = learning_efficiency_results['learning_dynamics']['identified_stages']
        
        # 繪製學習階段
        fig = plot_learning_stages(loss_values, learning_stages, 
                               title=f'Learning Stages Analysis - {self.experiment_dir}')
        
        # 保存圖像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_efficiency_metrics(self, metrics: Optional[List[str]] = None, 
                                   output_path: Optional[str] = None) -> plt.Figure:
        """
        可視化學習效率指標。
        
        Args:
            metrics (List[str], optional): 要可視化的指標列表。如果為None，則使用默認指標。
            output_path (str, optional): 輸出圖像的路徑。如果為None，則僅返回圖像對象。
            
        Returns:
            plt.Figure: Matplotlib圖像對象。
        """
        from visualization.performance_plots import plot_learning_efficiency_comparison
        
        # 檢查是否已進行學習效率分析
        if 'enhanced_learning_efficiency' not in self.results or not self.results['enhanced_learning_efficiency']:
            self.logger.warning("未找到學習效率分析結果，無法繪製效率指標")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No learning efficiency analysis results available", ha='center', va='center')
            return fig
        
        # 獲取相關類別的指標
        efficiency_metrics = self.results['enhanced_learning_efficiency'].get('efficiency_metrics', {})
        convergence_metrics = self.results['enhanced_learning_efficiency'].get('convergence_metrics', {})
        time_metrics = self.results['enhanced_learning_efficiency'].get('time_metrics', {})
        generalization_metrics = self.results['enhanced_learning_efficiency'].get('generalization_metrics', {})
        
        # 收集所有可用指標
        all_metrics = {}
        for category, metrics_dict in [
            ('efficiency', efficiency_metrics),
            ('convergence', convergence_metrics),
            ('time', time_metrics),
            ('generalization', generalization_metrics)
        ]:
            for metric, value in metrics_dict.items():
                all_metrics[f"{category}_{metric}"] = value
        
        # 確定要可視化的指標
        if not metrics:
            # 默認使用一些重要的效率指標
            metrics = [
                'efficiency_loss_reduction_efficiency',
                'efficiency_percent_loss_reduction',
                'convergence_convergence_percent'
            ]
            # 添加時間指標（如果有）
            if time_metrics:
                metrics.append('time_loss_reduction_per_second')
            # 添加泛化指標（如果有）
            if generalization_metrics:
                metrics.append('generalization_generalization_gap')
        
        # 過濾不可用的指標
        available_metrics = [m for m in metrics if m in all_metrics]
        
        if not available_metrics:
            self.logger.warning("沒有可用的效率指標，無法繪製")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No available efficiency metrics", ha='center', va='center')
            return fig
        
        # 準備數據
        runs_data = {'This Run': {}}
        for metric in available_metrics:
            category, metric_name = metric.split('_', 1)
            if category not in runs_data['This Run']:
                runs_data['This Run'][category] = {}
            runs_data['This Run'][category][metric_name] = all_metrics[metric]
        
        # 繪製效率指標
        fig = plot_learning_efficiency_comparison(
            runs=runs_data,
            metrics=available_metrics,
            title=f'Learning Efficiency Metrics - {self.experiment_dir}'
        )
        
        # 保存圖像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_lr_schedule_impact(self, other_runs: Optional[Dict[str, Dict[str, List[float]]]] = None) -> Dict[str, Any]:
        """
        分析學習率調度對訓練效果的影響。
        
        Args:
            other_runs (Dict[str, Dict[str, List[float]]], optional): 其他訓練運行的數據，格式為：
                {
                    'run_name': {
                        'train_loss': [float, ...],
                        'val_loss': [float, ...],
                        'learning_rate': [float, ...]
                    }
                }
                如果為None，則僅分析當前實驗。
                
        Returns:
            Dict[str, Any]: 學習率調度影響分析結果。
        """
        from metrics.performance_metrics import analyze_lr_schedule_impact
        
        # 檢查是否已載入訓練歷史
        if not hasattr(self, 'training_history') or not self.training_history:
            self.logger.warning("未載入訓練歷史，無法分析學習率調度")
            return {}
        
        # 檢查當前實驗是否有學習率數據
        lr_metric = None
        for metric_name in ['learning_rate', 'lr', 'optimizer_lr']:
            if metric_name in self.metrics_data.columns:
                lr_metric = metric_name
                break
        
        if not lr_metric:
            self.logger.warning("當前實驗中未找到學習率數據")
            return {}
        
        # 準備當前實驗的數據
        current_run_data = {
            'Current Experiment': {
                'train_loss': self.metrics_data['train_loss'].tolist() if 'train_loss' in self.metrics_data.columns else [],
                'val_loss': self.metrics_data['val_loss'].tolist() if 'val_loss' in self.metrics_data.columns else [],
                'learning_rate': self.metrics_data[lr_metric].tolist()
            }
        }
        
        # 合併其他運行的數據
        all_runs_data = {}
        if other_runs:
            all_runs_data.update(other_runs)
        all_runs_data.update(current_run_data)
        
        # 執行分析
        analysis_results = analyze_lr_schedule_impact(all_runs_data)
        
        # 保存分析結果
        if 'lr_schedule_impact_analysis' not in self.results:
            self.results['lr_schedule_impact_analysis'] = {}
        self.results['lr_schedule_impact_analysis'] = analysis_results
        
        return analysis_results
    
    def visualize_lr_schedule_impact(self, other_runs: Optional[Dict[str, Dict[str, List[float]]]] = None,
                                   output_path: Optional[str] = None) -> plt.Figure:
        """
        可視化學習率調度對訓練效果的影響。
        
        Args:
            other_runs (Dict[str, Dict[str, List[float]]], optional): 其他訓練運行的數據，格式同analyze_lr_schedule_impact
            output_path (str, optional): 輸出圖像的路徑。如果為None，則僅返回圖像對象。
            
        Returns:
            plt.Figure: Matplotlib圖像對象。
        """
        from visualization.performance_plots import plot_lr_schedule_impact
        
        # 先執行分析（如果需要）
        if 'lr_schedule_impact_analysis' not in self.results or not self.results['lr_schedule_impact_analysis']:
            self.analyze_lr_schedule_impact(other_runs)
        
        analysis_results = self.results.get('lr_schedule_impact_analysis', {})
        
        if not analysis_results:
            self.logger.warning("沒有學習率調度影響分析結果，無法生成可視化")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No learning rate schedule impact analysis results available", ha='center', va='center')
            return fig
        
        # 生成可視化
        fig = plot_lr_schedule_impact(
            analysis_results,
            title=f'Learning Rate Schedule Impact Analysis - {self.experiment_dir}'
        )
        
        # 保存圖像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig 