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
    analyze_learning_efficiency,
    analyze_lr_schedule_impact,
    detect_training_anomalies,
    detect_overfitting,
    detect_oscillations,
    detect_plateaus
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
        
        # 遍歷所有指標
        for metric, values in self.training_history.items():
            metrics_data[metric] = values
        
        # 添加輪次索引
        metrics_data['epoch'] = range(1, len(metrics_data) + 1)
        
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
        if not self.metrics_data:
            self.preprocess_metrics()
        
        # 初始化結果字典
        self.results = {
            'convergence_analysis': {},
            'stability_analysis': {},
            'anomaly_detection': {},
            'metric_correlations': {},
            'learning_efficiency': {}
        }
        
        try:
            # 確定要分析的指標
            if metrics is None:
                # 尋找訓練和驗證損失
                loss_metrics = []
                val_loss_metrics = []
                accuracy_metrics = []
                val_accuracy_metrics = []
                other_metrics = []
                
                for metric in self.training_history.keys():
                    if metric == 'loss':
                        loss_metrics.append(metric)
                    elif metric == 'val_loss':
                        val_loss_metrics.append(metric)
                    elif 'accuracy' in metric and 'val' not in metric:
                        accuracy_metrics.append(metric)
                    elif 'val_accuracy' in metric or 'val_acc' in metric:
                        val_accuracy_metrics.append(metric)
                    elif metric not in ['lr', 'learning_rate']:
                        other_metrics.append(metric)
                
                # 分析損失曲線
                if loss_metrics and val_loss_metrics:
                    self._analyze_loss_curves(loss_metrics[0], val_loss_metrics[0])
                elif loss_metrics:
                    self._analyze_loss_curves(loss_metrics[0])
                
                # 分析準確度曲線
                if accuracy_metrics and val_accuracy_metrics:
                    self._analyze_accuracy_curves(accuracy_metrics[0], val_accuracy_metrics[0])
                elif accuracy_metrics:
                    self._analyze_accuracy_curves(accuracy_metrics[0])
                
                # 分析其他指標
                for metric in other_metrics:
                    val_metric = f"val_{metric}" if f"val_{metric}" in self.training_history else None
                    self._analyze_other_metric(metric, val_metric)
                
                # 分析學習效率
                if 'lr' in self.training_history or 'learning_rate' in self.training_history:
                    lr_key = 'lr' if 'lr' in self.training_history else 'learning_rate'
                    if loss_metrics:
                        self._analyze_learning_efficiency(loss_metrics[0], lr_key)
                
                # 分析異常
                if loss_metrics and val_loss_metrics:
                    self._detect_anomalies(loss_metrics[0], val_loss_metrics[0])
                elif loss_metrics:
                    self._detect_anomalies(loss_metrics[0])
            else:
                # 使用指定的指標進行分析
                for metric in metrics:
                    if metric in self.training_history:
                        val_metric = f"val_{metric}" if f"val_{metric}" in self.training_history else None
                        if metric == 'loss':
                            self._analyze_loss_curves(metric, val_metric)
                        elif 'accuracy' in metric:
                            self._analyze_accuracy_curves(metric, val_metric)
                        else:
                            self._analyze_other_metric(metric, val_metric)
            
            # 計算指標相關性
            self._calculate_metric_correlations()
            
            if save_results:
                self.save_results(os.path.join(self.experiment_dir, 'analysis'))
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"分析訓練動態時出錯: {e}")
            raise
    
    def _analyze_loss_curves(self, loss_metric: str, val_loss_metric: Optional[str] = None) -> None:
        """
        分析訓練和驗證損失曲線。
        
        Args:
            loss_metric (str): 訓練損失指標名稱。
            val_loss_metric (Optional[str], optional): 驗證損失指標名稱。默認為None。
        """
        # 獲取訓練損失值
        loss_values = self.training_history[loss_metric]
        val_loss_values = self.training_history[val_loss_metric] if val_loss_metric else None
        
        # 使用性能指標模組分析損失曲線
        loss_analysis = analyze_loss_curve(loss_values, val_loss_values)
        
        # 存儲分析結果
        self.results['convergence_analysis'][loss_metric] = {
            'convergence_epoch': loss_analysis.get('train_loss', {}).get('convergence_epoch'),
            'convergence_value': loss_analysis.get('train_loss', {}).get('convergence_value'),
            'convergence_rate': loss_analysis.get('train_loss', {}).get('convergence_rate'),
            'final_value': loss_analysis.get('train_loss', {}).get('final_value'),
            'min_value': loss_analysis.get('train_loss', {}).get('min_value'),
            'min_epoch': loss_analysis.get('train_loss', {}).get('min_epoch')
        }
        
        if 'stability' in loss_analysis.get('train_loss', {}):
            self.results['stability_analysis'][loss_metric] = loss_analysis['train_loss']['stability']
        
        # 如果有驗證損失，也存儲對應結果
        if val_loss_metric:
            self.results['convergence_analysis'][val_loss_metric] = {
                'convergence_epoch': loss_analysis.get('val_loss', {}).get('convergence_epoch'),
                'convergence_value': loss_analysis.get('val_loss', {}).get('convergence_value'),
                'convergence_rate': loss_analysis.get('val_loss', {}).get('convergence_rate'),
                'final_value': loss_analysis.get('val_loss', {}).get('final_value'),
                'min_value': loss_analysis.get('val_loss', {}).get('min_value'),
                'min_epoch': loss_analysis.get('val_loss', {}).get('min_epoch')
            }
            
            if 'stability' in loss_analysis.get('val_loss', {}):
                self.results['stability_analysis'][val_loss_metric] = loss_analysis['val_loss']['stability']
            
            # 訓練/驗證差異分析
            if 'train_val_difference' in loss_analysis:
                self.results['train_val_difference'] = loss_analysis['train_val_difference']
            
            # 過擬合分析
            if 'overfitting_detection' in loss_analysis:
                self.results['anomaly_detection']['overfitting'] = loss_analysis['overfitting_detection']
    
    def _analyze_accuracy_curves(self, acc_metric: str, val_acc_metric: Optional[str] = None) -> None:
        """
        分析訓練和驗證準確度曲線。
        
        Args:
            acc_metric (str): 訓練準確度指標名稱。
            val_acc_metric (Optional[str], optional): 驗證準確度指標名稱。默認為None。
        """
        # 獲取準確度值
        acc_values = self.training_history[acc_metric]
        val_acc_values = self.training_history[val_acc_metric] if val_acc_metric else None
        
        # 使用性能指標模組分析準確度曲線
        acc_analysis = analyze_metric_curve(acc_values, val_acc_values, is_higher_better=True)
        
        # 存儲分析結果
        self.results['convergence_analysis'][acc_metric] = {
            'convergence_epoch': acc_analysis.get('train_metric', {}).get('convergence_epoch'),
            'convergence_value': acc_analysis.get('train_metric', {}).get('convergence_value'),
            'convergence_rate': acc_analysis.get('train_metric', {}).get('convergence_rate'),
            'final_value': acc_analysis.get('train_metric', {}).get('final_value'),
            'max_value': acc_analysis.get('train_metric', {}).get('max_value'),
            'max_epoch': acc_analysis.get('train_metric', {}).get('max_epoch')
        }
        
        # 如果有驗證準確度，也存儲對應結果
        if val_acc_metric:
            self.results['convergence_analysis'][val_acc_metric] = {
                'convergence_epoch': acc_analysis.get('val_metric', {}).get('convergence_epoch'),
                'convergence_value': acc_analysis.get('val_metric', {}).get('convergence_value'),
                'convergence_rate': acc_analysis.get('val_metric', {}).get('convergence_rate'),
                'final_value': acc_analysis.get('val_metric', {}).get('final_value'),
                'max_value': acc_analysis.get('val_metric', {}).get('max_value'),
                'max_epoch': acc_analysis.get('val_metric', {}).get('max_epoch')
            }
        
        # 存儲改進率分析
        self.results['improvement_rate'] = {
            'accuracy': acc_analysis.get('improvement_rate')
        }
    
    def _analyze_other_metric(self, metric: str, val_metric: Optional[str] = None) -> None:
        """
        分析其他指標曲線。
        
        Args:
            metric (str): 指標名稱。
            val_metric (Optional[str], optional): 對應的驗證指標名稱。默認為None。
        """
        # 獲取指標值
        metric_values = self.training_history[metric]
        val_metric_values = self.training_history[val_metric] if val_metric else None
        
        # 嘗試判斷指標是否越高越好
        is_higher_better = 'accuracy' in metric or 'precision' in metric or 'recall' in metric or 'f1' in metric
        
        # 使用性能指標模組分析指標曲線
        metric_analysis = analyze_metric_curve(metric_values, val_metric_values, is_higher_better=is_higher_better)
        
        # 存儲分析結果
        self.results['convergence_analysis'][metric] = {
            'convergence_epoch': metric_analysis.get('train_metric', {}).get('convergence_epoch'),
            'convergence_value': metric_analysis.get('train_metric', {}).get('convergence_value'),
            'final_value': metric_analysis.get('train_metric', {}).get('final_value')
        }
        
        # 添加最大值或最小值
        if is_higher_better:
            self.results['convergence_analysis'][metric]['max_value'] = metric_analysis.get('train_metric', {}).get('max_value')
            self.results['convergence_analysis'][metric]['max_epoch'] = metric_analysis.get('train_metric', {}).get('max_epoch')
        else:
            self.results['convergence_analysis'][metric]['min_value'] = metric_analysis.get('train_metric', {}).get('min_value')
            self.results['convergence_analysis'][metric]['min_epoch'] = metric_analysis.get('train_metric', {}).get('min_epoch')
        
        # 如果有驗證指標，也存儲對應結果
        if val_metric:
            self.results['convergence_analysis'][val_metric] = {
                'convergence_epoch': metric_analysis.get('val_metric', {}).get('convergence_epoch'),
                'convergence_value': metric_analysis.get('val_metric', {}).get('convergence_value'),
                'final_value': metric_analysis.get('val_metric', {}).get('final_value')
            }
            
            # 添加最大值或最小值
            if is_higher_better:
                self.results['convergence_analysis'][val_metric]['max_value'] = metric_analysis.get('val_metric', {}).get('max_value')
                self.results['convergence_analysis'][val_metric]['max_epoch'] = metric_analysis.get('val_metric', {}).get('max_epoch')
            else:
                self.results['convergence_analysis'][val_metric]['min_value'] = metric_analysis.get('val_metric', {}).get('min_value')
                self.results['convergence_analysis'][val_metric]['min_epoch'] = metric_analysis.get('val_metric', {}).get('min_epoch')
        
        # 存儲改進率分析
        self.results['improvement_rate'][metric] = metric_analysis.get('improvement_rate')
    
    def _analyze_learning_efficiency(self, loss_metric: str, lr_metric: str) -> None:
        """
        分析學習效率和學習率調度的影響。
        
        Args:
            loss_metric (str): 損失指標名稱。
            lr_metric (str): 學習率指標名稱。
        """
        # 獲取損失值和學習率
        loss_values = self.training_history[loss_metric]
        learning_rates = self.training_history[lr_metric]
        
        # 分析學習效率
        efficiency_analysis = analyze_learning_efficiency(loss_values)
        self.results['learning_efficiency'] = efficiency_analysis
        
        # 如果學習率序列存在，分析學習率調度的影響
        if len(learning_rates) == len(loss_values):
            lr_impact_analysis = analyze_lr_schedule_impact(loss_values, learning_rates)
            self.results['learning_efficiency']['lr_schedule_impact'] = lr_impact_analysis
    
    def _detect_anomalies(self, loss_metric: str, val_loss_metric: Optional[str] = None) -> None:
        """
        檢測訓練過程中的異常。
        
        Args:
            loss_metric (str): 損失指標名稱。
            val_loss_metric (Optional[str], optional): 驗證損失指標名稱。默認為None。
        """
        # 獲取損失值
        loss_values = self.training_history[loss_metric]
        val_loss_values = self.training_history[val_loss_metric] if val_loss_metric else None
        
        # 構建其他指標字典
        other_metrics = {}
        for metric, values in self.training_history.items():
            if metric not in [loss_metric, val_loss_metric, 'lr', 'learning_rate'] and len(values) == len(loss_values):
                other_metrics[metric] = values
        
        # 檢測訓練異常
        anomalies = detect_training_anomalies(loss_values, val_loss_values, other_metrics)
        
        # 存儲異常檢測結果
        if anomalies['has_anomalies']:
            self.results['anomaly_detection'].update(anomalies['details'])
    
    def _calculate_metric_correlations(self) -> None:
        """
        計算不同指標之間的相關性。
        """
        if self.metrics_data is None or len(self.metrics_data) <= 1:
            return
        
        # 計算相關性矩陣，排除epoch列
        metrics_only = self.metrics_data.drop(columns=['epoch'], errors='ignore')
        correlation_matrix = metrics_only.corr()
        
        # 轉換為字典格式存儲
        self.results['metric_correlations'] = correlation_matrix.to_dict()
    
    def analyze_convergence(self, metric: str = 'loss') -> Dict:
        """
        分析指定指標的收斂性。
        
        Args:
            metric (str, optional): 要分析的指標名稱。默認為'loss'。
        
        Returns:
            Dict: 包含收斂分析結果的字典。
        """
        if not self.training_history:
            self.load_training_history()
        
        if metric not in self.training_history:
            self.logger.warning(f"指標 {metric} 不存在於訓練歷史中")
            return {}
        
        # 獲取指標值
        values = self.training_history[metric]
        val_metric = f"val_{metric}" if f"val_{metric}" in self.training_history else None
        val_values = self.training_history[val_metric] if val_metric else None
        
        # 使用性能指標模組分析
        if metric == 'loss':
            analysis = analyze_loss_curve(values, val_values)
            result = {
                'train': analysis.get('train_loss', {}),
                'val': analysis.get('val_loss', {}) if val_values else {}
            }
        else:
            # 嘗試判斷指標是否越高越好
            is_higher_better = 'accuracy' in metric or 'precision' in metric or 'recall' in metric or 'f1' in metric
            analysis = analyze_metric_curve(values, val_values, is_higher_better=is_higher_better)
            result = {
                'train': analysis.get('train_metric', {}),
                'val': analysis.get('val_metric', {}) if val_values else {}
            }
        
        return result
    
    def analyze_stability(self, metric: str = 'loss', window_size: int = 5) -> Dict:
        """
        分析指定指標的穩定性。
        
        Args:
            metric (str, optional): 要分析的指標名稱。默認為'loss'。
            window_size (int, optional): 用於計算移動平均的窗口大小。默認為5。
        
        Returns:
            Dict: 包含穩定性分析結果的字典。
        """
        if not self.training_history:
            self.load_training_history()
        
        if metric not in self.training_history:
            self.logger.warning(f"指標 {metric} 不存在於訓練歷史中")
            return {}
        
        # 獲取指標值
        values = self.training_history[metric]
        
        # 檢測震盪
        oscillations = detect_oscillations(values, window_size=window_size)
        
        # 檢測平台期
        plateaus = detect_plateaus(values, window_size=window_size)
        
        # 計算穩定性指標
        if len(values) > window_size * 2:
            second_half = values[len(values)//2:]
            stability = {
                'variance': float(np.var(second_half)),
                'std': float(np.std(second_half)),
                'max_fluctuation': float(max(second_half) - min(second_half))
            }
        else:
            stability = {}
        
        # 整合結果
        result = {
            'stability_metrics': stability,
            'oscillations': oscillations,
            'plateaus': plateaus
        }
        
        return result
    
    def get_metric_correlation(self) -> pd.DataFrame:
        """
        計算不同指標之間的相關性。
        
        Returns:
            pd.DataFrame: 包含指標相關性的DataFrame。
        """
        if not self.metrics_data:
            self.preprocess_metrics()
        
        # 計算相關性矩陣，排除epoch列
        metrics_only = self.metrics_data.drop(columns=['epoch'], errors='ignore')
        return metrics_only.corr() 