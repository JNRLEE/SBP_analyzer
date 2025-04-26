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
from utils.stat_utils import (
    find_convergence_point,
    calculate_stability_metrics,
    detect_overfitting_simple
)
# from reporter.plot_utils import PlotConfig # Commented out as PlotConfig is not found
# from utils.log_utils import setup_logger # Commented out as setup_logger is not found
from utils.file_utils import load_json_file, ensure_dir

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
        預處理訓練指標數據，轉換為易於分析的格式，並處理常見問題。

        處理邏輯:
        - 如果歷史記錄為空，返回空 DataFrame 並記錄。
        - 檢查指標列表長度是否一致，如果不一致，截斷至最短長度並記錄警告。
        - 檢查是否存在 'epoch' 鍵，如果不存在，自動生成並記錄警告。
        - 檢查是否存在 NaN 值，並記錄警告。

        Returns:
            pd.DataFrame: 包含處理後指標數據的DataFrame。
        """
        if not self.training_history:
            try:
                self.load_training_history()
            except FileNotFoundError:
                self.logger.warning("訓練歷史文件不存在，無法預處理。")
                self.metrics_data = pd.DataFrame()
                return self.metrics_data
            except json.JSONDecodeError:
                self.logger.error("訓練歷史文件 JSON 格式錯誤，無法預處理。")
                self.metrics_data = pd.DataFrame()
                return self.metrics_data
        
        if not self.training_history:
            self.logger.info("訓練歷史為空，返回空的 DataFrame。")
            self.metrics_data = pd.DataFrame()
            return self.metrics_data

        # 檢查數值列表長度是否一致
        lengths = {metric: len(values) for metric, values in self.training_history.items() if isinstance(values, list)}
        if not lengths:
             self.logger.warning("訓練歷史中沒有找到列表格式的指標數據，返回空 DataFrame。")
             self.metrics_data = pd.DataFrame()
             return self.metrics_data
             
        min_len = min(lengths.values()) if lengths else 0
        max_len = max(lengths.values()) if lengths else 0

        processed_history = {}
        has_inconsistent_lengths = False
        if min_len != max_len:
            has_inconsistent_lengths = True
            self.logger.warning(f"訓練歷史指標列表長度不一致 (最短: {min_len}, 最長: {max_len})。將截斷至最短長度 {min_len}。")
            for metric, values in self.training_history.items():
                if isinstance(values, list):
                    processed_history[metric] = values[:min_len]
                else: # 保留非列表值
                     processed_history[metric] = values
        else:
             processed_history = self.training_history.copy()

        # 檢查並生成 'epoch' 鍵
        if 'epoch' not in processed_history and min_len > 0:
            self.logger.warning("訓練歷史中缺少 'epoch' 鍵，將自動生成從 0 開始的輪次。")
            processed_history['epoch'] = list(range(min_len))
        elif 'epoch' in processed_history and len(processed_history['epoch']) != min_len:
             self.logger.warning(f"'epoch' 鍵長度 ({len(processed_history['epoch'])}) 與其他指標長度 ({min_len}) 不匹配，將重新生成。")
             processed_history['epoch'] = list(range(min_len))

        # 轉換為 DataFrame
        try:
            # 創建 DataFrame 前確保所有 list 長度一致 (已經截斷)
            self.metrics_data = pd.DataFrame(processed_history)
        except ValueError as e:
            # 捕獲潛在的 Pandas 錯誤 (理論上不應發生，因為已截斷)
            self.logger.error(f"創建 DataFrame 失敗: {e}。 歷史數據: {processed_history}")
            self.metrics_data = pd.DataFrame() # 返回空 DataFrame
            return self.metrics_data
            
        # 檢查 NaN 值
        if self.metrics_data.isnull().values.any():
            nan_cols = self.metrics_data.columns[self.metrics_data.isnull().any()].tolist()
            self.logger.warning(f"指標數據包含 NaN 值。涉及的列: {nan_cols}")

        return self.metrics_data
    
    def analyze(self, save_results: bool = True, 
                convergence_params: Dict[str, Any] = None,
                stability_params: Dict[str, Any] = None,
                overfitting_params: Dict[str, Any] = None) -> Dict:
        """
        執行所有訓練動態分析步驟。

        分析內容:
        - 收斂性 (Convergence)
        - 穩定性 (Stability)
        - 過擬合 (Overfitting)
        - 指標相關性 (Metric Correlations)
        - 學習效率 (Learning Efficiency)
        - 異常檢測 (Anomaly Detection)

        Args:
            save_results (bool): 是否將結果保存到文件。
            convergence_params (Dict[str, Any], optional): 傳遞給 analyze_convergence 的參數。
            stability_params (Dict[str, Any], optional): 傳遞給 analyze_stability 的參數。
            overfitting_params (Dict[str, Any], optional): 傳遞給 analyze_overfitting 的參數。

        Returns:
            Dict: 包含所有分析結果的字典。
        """
        # 載入和預處理數據
        if self.metrics_data is None:
            try:
                self.load_training_history()
                self.preprocess_metrics()
            except Exception as e:
                self.logger.error(f"加載/預處理數據失敗: {e}")
        
        if self.metrics_data is None or self.metrics_data.empty:
            self.logger.warning("沒有可用的指標數據進行分析。")
            return { # Return structure consistent with successful run but indicating error
                'error': 'No metrics data available for analysis.',
                'convergence_analysis': {},
                'stability_analysis': {},
                'overfitting_analysis': {'status': 'N/A', 'reason': 'No data available'},
                'metric_correlations': {},
                'learning_efficiency': {},
                'anomaly_detection': {},
            }
        
        # 設置默認參數 (如果未提供)
        convergence_params = convergence_params or {'window': 5, 'threshold': 0.01, 'patience': 5}
        stability_params = stability_params or {'window_size': 5}
        overfitting_params = overfitting_params or {'patience': 5}

        # 初始化結果字典
        self.results = {
            'convergence_analysis': {},
            'stability_analysis': {},
            'overfitting_analysis': {'status': 'Unknown'},  # 提供默認值
            'anomaly_detection': {},
            'metric_correlations': {},
            'learning_efficiency': {}
        }

        # --- 執行核心分析 --- 
        metrics_to_analyze = []
        loss_metric = None
        val_loss_metric = None
        lr_metric = None

        for col in self.metrics_data.columns:
            if col == 'epoch': continue # Skip epoch column
            if col == 'loss' or col == 'train_loss': loss_metric = col
            if col == 'val_loss': val_loss_metric = col
            if col in ['lr', 'learning_rate']: lr_metric = col
            # Analyze convergence and stability for all numeric metrics
            if pd.api.types.is_numeric_dtype(self.metrics_data[col]):
                metrics_to_analyze.append(col)
        
        # 1. 分析收斂性和穩定性
        for metric in metrics_to_analyze:
            self.logger.debug(f"Analyzing convergence for {metric}...")
            self.results['convergence_analysis'][metric] = self.analyze_convergence(metric=metric, **convergence_params)
            self.logger.debug(f"Analyzing stability for {metric}...")
            self.results['stability_analysis'][metric] = self.analyze_stability(metric=metric, **stability_params)
        
        # 2. 分析過擬合 (如果存在 loss 和 val_loss)
        if loss_metric and val_loss_metric:
            self.logger.debug("Analyzing overfitting...")
            overfitting_result = self.analyze_overfitting(
                loss_metric=loss_metric, 
                val_loss_metric=val_loss_metric, 
                **overfitting_params
            )
            # 確保返回的結果是字典而非 None 或元組
            if isinstance(overfitting_result, dict):
                self.results['overfitting_analysis'] = overfitting_result
            else:
                # 處理意外的返回類型
                self.logger.warning(f"Unexpected return type from analyze_overfitting: {type(overfitting_result)}")
                self.results['overfitting_analysis'] = {
                    'status': 'Error',
                    'reason': f'Unexpected return type: {type(overfitting_result)}'
                }
        else:
            self.logger.warning("無法分析過擬合，缺少 'loss' 或 'val_loss' 指標。")
            self.results['overfitting_analysis'] = {'status': 'N/A', 'reason': 'Missing loss or val_loss metrics'}

        # 3. 分析學習效率 (如果存在 loss 和 lr)
        if loss_metric and lr_metric:
            self.logger.debug("Analyzing learning efficiency...")
            self._analyze_learning_efficiency(loss_metric, lr_metric)
        else:
            self.logger.warning("無法分析學習效率，缺少 'loss' 或學習率指標 ('lr'/'learning_rate')。")
            self.results['learning_efficiency'] = {'error': 'Missing required metrics'}

        # 4. 檢測異常 (如果存在 loss)
        if loss_metric:
            self.logger.debug("Detecting anomalies...")
            self._detect_anomalies(loss_metric, val_loss_metric)
        else:
            self.logger.warning("無法檢測異常，缺少 'loss' 指標。")
            self.results['anomaly_detection'] = {'error': 'Missing loss metric'}

        # 5. 計算指標相關性
        self._calculate_metric_correlations()

        # 6. 保存結果 (如果需要)
        if save_results:
            self.save_results(os.path.join(self.experiment_dir, 'analysis'))

        return self.results
    
    def _analyze_loss_curves(self, analysis_params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyzes training and validation loss curves."""
        if self.metrics_data is None or 'loss' not in self.metrics_data.columns:
            logging.warning("Loss data not found or not preprocessed. Cannot analyze loss curves.")
            return {'error': 'Loss data not available'}
        
        loss_values = self.metrics_data['loss'].tolist()
        val_loss_values = self.metrics_data['val_loss'].tolist() if 'val_loss' in self.metrics_data.columns else None
        
        # Extract relevant params for analyze_loss_curve
        window_size = analysis_params.get('convergence_window', 5)
        convergence_threshold = analysis_params.get('convergence_threshold', 0.01)
        patience = analysis_params.get('overfitting_patience', 5) # Use overfitting patience for loss curve analysis patience

        return analyze_loss_curve(
            loss_values,
            val_loss_values,
            window_size=window_size,
            convergence_threshold=convergence_threshold,
            patience=patience
        )

    def _analyze_metric_curves(self, metric_name: str, analysis_params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyzes training and validation metric curves."""
        train_metric_col = metric_name
        val_metric_col = f"val_{metric_name}"

        if self.metrics_data is None or train_metric_col not in self.metrics_data.columns:
            logging.warning(f"Metric '{metric_name}' not found or data not preprocessed.")
            return {'error': f"Metric '{metric_name}' data not available"}

        metric_values = self.metrics_data[train_metric_col].tolist()
        val_metric_values = self.metrics_data[val_metric_col].tolist() if val_metric_col in self.metrics_data.columns else None
        is_higher_better = analysis_params.get('is_higher_better', False) # Default assuming lower is better

        # Extract relevant params for analyze_metric_curve
        window_size = analysis_params.get('convergence_window', 5)
        convergence_threshold = analysis_params.get('convergence_threshold', 0.01)
        patience = analysis_params.get('overfitting_patience', 5) # Use overfitting patience here too

        return analyze_metric_curve(
            metric_values,
            val_metric_values,
            is_higher_better=is_higher_better,
            window_size=window_size,
            convergence_threshold=convergence_threshold,
            patience=patience
        )
    
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
        if 'improvement_rate' in metric_analysis:
            self.results['improvement_rate'][metric] = metric_analysis.get('improvement_rate')
        else:
            # 如果沒有改進率，則使用默認值
            self.results['improvement_rate'][metric] = {'rate': 'N/A', 'slope': 0.0}
    
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
        """計算指標之間的相關性矩陣。"""
        if self.metrics_data is None or self.metrics_data.empty:
            self.logger.warning("無法計算指標相關性：沒有可用的指標數據。")
            self.results['metric_correlations'] = {}
            return

        # 過濾出數值型列
        numeric_metrics = self.metrics_data.select_dtypes(include=['number'])
        if 'epoch' in numeric_metrics.columns:
            numeric_metrics = numeric_metrics.drop(columns=['epoch'])  # 排除輪次列

        if numeric_metrics.empty or len(numeric_metrics.columns) < 2:
            self.logger.warning("無法計算相關性：數值型指標數量不足。")
            self.results['metric_correlations'] = {}
            return

        # 計算相關性矩陣 - 處理數據點不足的情況
        if len(numeric_metrics) <= 1:  # 只有一個數據點時，返回空DataFrame
            self.logger.warning("數據點數量不足 (≤1)，無法計算相關性。")
            self.results['metric_correlations'] = {}
            return
        
        # 計算相關性矩陣
        correlation_matrix = numeric_metrics.corr()
        
        # 將相關性矩陣存儲到結果中
        self.results['metric_correlations'] = correlation_matrix.to_dict()

    def get_summary(self) -> Dict[str, Any]:
        """
        生成分析結果的摘要字典。
        
        Returns:
            Dict[str, Any]: 包含分析摘要的字典。
        """
        if not hasattr(self, 'results') or not self.results:
            self.logger.warning("分析尚未執行或無結果可摘要")
            return {}
        
        summary = {
            'metrics_analyzed': [],
            'analysis_keys': list(self.results.keys()),
            'convergence_summary': {},
            'stability_summary': {},
            'overfitting_status': 'Unknown'
        }
        
        # 提取已分析的指標
        if 'convergence_analysis' in self.results:
            metrics = [m for m in self.results['convergence_analysis'].keys() 
                      if not isinstance(self.results['convergence_analysis'][m], str)]
            summary['metrics_analyzed'] = metrics
            
            # 收集收斂性摘要
            for metric in metrics:
                metric_result = self.results['convergence_analysis'].get(metric, {})
                if isinstance(metric_result, dict) and 'converged' in metric_result:
                    converged = metric_result.get('converged', False)
                    point = metric_result.get('convergence_point', None)
                    summary['convergence_summary'][metric] = {
                        'converged': converged,
                        'at_epoch': point
                    }
        
        # 提取穩定性摘要
        if 'stability_analysis' in self.results:
            for metric in summary['metrics_analyzed']:
                metric_result = self.results['stability_analysis'].get(metric, {})
                if isinstance(metric_result, dict):
                    overall_std = metric_result.get('overall_std', float('nan'))
                    last_std = metric_result.get('last_window_std', float('nan'))
                    summary['stability_summary'][metric] = {
                        'overall_std': overall_std,
                        'recent_std': last_std,
                        'is_stable': last_std < overall_std
                    }
        
        # 提取過擬合狀態
        if 'overfitting_analysis' in self.results:
            overfitting_result = self.results['overfitting_analysis']
            if isinstance(overfitting_result, dict):
                summary['overfitting_status'] = overfitting_result.get('status', 'Unknown')
                if 'divergence_point' in overfitting_result:
                    summary['overfitting_epoch'] = overfitting_result['divergence_point']
        
        return summary

    def get_metric_correlation(self) -> pd.DataFrame:
        """
        獲取指標間的相關性矩陣。
        
        Returns:
            pd.DataFrame: 包含指標間相關性的DataFrame，如果數據不足則返回空DataFrame。
        """
        if self.metrics_data is None or self.metrics_data.empty:
            self.logger.warning("無法計算指標相關性：沒有可用的指標數據。")
            return pd.DataFrame()

        # 過濾出數值型列
        numeric_metrics = self.metrics_data.select_dtypes(include=['number'])
        if 'epoch' in numeric_metrics.columns:
            numeric_metrics = numeric_metrics.drop(columns=['epoch'])  # 排除輪次列

        if numeric_metrics.empty or len(numeric_metrics.columns) < 2:
            self.logger.warning("無法計算相關性：數值型指標數量不足。")
            return pd.DataFrame()

        # 計算相關性矩陣 - 處理數據點不足的情況
        if len(numeric_metrics) <= 1:  # 只有一個數據點時，返回空DataFrame
            self.logger.warning("數據點數量不足 (≤1)，無法計算相關性。")
            return pd.DataFrame()
        
        # 計算相關性矩陣
        return numeric_metrics.corr()

    def analyze_convergence(self, metric: str = 'loss', **kwargs) -> Dict[str, Any]:
        """Analyzes the convergence of a specific metric."""
        # 首先檢查 metrics_data 是否已加載
        if self.metrics_data is None or self.metrics_data.empty:
            self.logger.warning(f"Cannot analyze convergence for '{metric}': No metrics data available.")
            return {'error': 'No metrics data available'}
         
        # 檢查指定的指標是否存在
        # 1. 修改: 現在先檢查精確匹配
        if metric in self.metrics_data.columns:
            metric_name = metric
        # 2. 然後嘗試在 train_loss/val_loss 種類中搜索
        elif f'train_{metric}' in self.metrics_data.columns:
            metric_name = f'train_{metric}'
            self.logger.info(f"使用 'train_{metric}' 替代 '{metric}'")
        elif f'val_{metric}' in self.metrics_data.columns:
            metric_name = f'val_{metric}'
            self.logger.info(f"使用 'val_{metric}' 替代 '{metric}'")
        else:
            # 3. 最後嘗試模糊匹配
            possible_matches = [col for col in self.metrics_data.columns if metric in col]
            if possible_matches:
                metric_name = possible_matches[0]
                self.logger.warning(f"Metric '{metric}' not found exactly, using '{metric_name}' instead.")
            else:
                self.logger.error(f"Metric '{metric}' not found in history columns: {list(self.metrics_data.columns)}.")
                return {'error': f"Metric '{metric}' not found in history data"}
            
        # 使用找到的指標名來提取值
        values = self.metrics_data[metric_name].tolist()
        
        # 使用 kwargs 配置參數
        window = kwargs.get('window', 5)
        threshold = kwargs.get('threshold', 0.01)
        
        try:
            # 處理單一數據點的情況
            if len(values) <= 1:
                return {
                    'converged': False,
                    'convergence_point': None,
                    'convergence_value': values[0] if values else None,
                    'convergence_rate': 0.0,
                    'note': 'Insufficient data points for convergence analysis'
                }
            
            # 正確的參數名稱是 window_size
            convergence_info = find_convergence_point(values, window_size=window, threshold=threshold) 
            # 檢查返回值 - 確保是字典，以防止 None 返回
            if convergence_info is None:
                return {
                    'converged': False,
                    'convergence_point': None,
                    'note': 'Convergence analysis failed or returned None'
                }
            return convergence_info
        except Exception as e:
            self.logger.error(f"Error analyzing convergence for '{metric}': {e}", exc_info=True)
            return {'error': str(e)}

    def analyze_stability(self, metric: str = 'loss', **kwargs) -> Dict[str, Any]:
        """Analyzes the stability of a specific metric."""
        # 首先檢查 metrics_data 是否已加載
        if self.metrics_data is None or self.metrics_data.empty:
            self.logger.warning(f"Cannot analyze stability for '{metric}': No metrics data available.")
            return {'error': 'No metrics data available'}

        # 檢查指定的指標是否存在 - 使用與 analyze_convergence 相同的改進邏輯
        if metric in self.metrics_data.columns:
            metric_name = metric
        elif f'train_{metric}' in self.metrics_data.columns:
            metric_name = f'train_{metric}'
            self.logger.info(f"使用 'train_{metric}' 替代 '{metric}'")
        elif f'val_{metric}' in self.metrics_data.columns:
            metric_name = f'val_{metric}'
            self.logger.info(f"使用 'val_{metric}' 替代 '{metric}'")
        else:
            possible_matches = [col for col in self.metrics_data.columns if metric in col]
            if possible_matches:
                metric_name = possible_matches[0]
                self.logger.warning(f"Metric '{metric}' not found exactly, using '{metric_name}' instead.")
            else:
                self.logger.error(f"Metric '{metric}' not found in history columns: {list(self.metrics_data.columns)}.")
                return {'error': f"Metric '{metric}' not found in history data"}
        
        # 使用找到的指標名來提取值
        values = self.metrics_data[metric_name].tolist()
        window_size = kwargs.get('window_size', 5)

        try:
            # 處理單一數據點的情況
            if len(values) <= 1:
                return {
                    'overall_std': 0.0,
                    'last_window_std': 0.0,
                    'stability_index': 1.0,
                    'note': 'Insufficient data points for stability analysis'
                }
            
            # 直接調用底層函數
            stability_info = calculate_stability_metrics(values, window_size=window_size)
            return stability_info
        except Exception as e:
            self.logger.error(f"Error analyzing stability for '{metric}': {e}", exc_info=True)
            return {'error': str(e)}

    def analyze_overfitting(self, loss_metric: str = 'loss', val_loss_metric: str = 'val_loss', **kwargs) -> Dict[str, Any]:
        """Analyzes overfitting based on training and validation loss metrics."""
        # 首先檢查 metrics_data 是否已加載
        if self.metrics_data is None or self.metrics_data.empty:
            self.logger.warning(f"Cannot analyze overfitting: No metrics data available.")
            return {'status': 'Error', 'reason': 'No metrics data available'}
         
        # 檢查訓練損失指標是否存在 - 使用更靈活的指標匹配
        train_metric = None
        val_metric = None
        
        # 嘗試匹配訓練損失
        if loss_metric in self.metrics_data.columns:
            train_metric = loss_metric
        elif f'train_{loss_metric}' in self.metrics_data.columns:
            train_metric = f'train_{loss_metric}'
        else:
            # 模糊匹配訓練損失
            train_matches = [col for col in self.metrics_data.columns 
                            if (col.startswith('train_') and loss_metric in col) or 
                               (not col.startswith('val_') and loss_metric in col)]
            if train_matches:
                train_metric = train_matches[0]
                self.logger.warning(f"Loss metric '{loss_metric}' not found exactly, using '{train_metric}' instead.")
            
        # 嘗試匹配驗證損失
        if val_loss_metric in self.metrics_data.columns:
            val_metric = val_loss_metric
        elif f'val_{loss_metric}' in self.metrics_data.columns:
            val_metric = f'val_{loss_metric}'
        else:
            # 模糊匹配驗證損失
            val_matches = [col for col in self.metrics_data.columns 
                          if col.startswith('val_') and loss_metric in col]
            if val_matches:
                val_metric = val_matches[0]
                self.logger.warning(f"Validation metric '{val_loss_metric}' not found exactly, using '{val_metric}' instead.")
        
        # 檢查是否找到了指標
        if not train_metric or not val_metric:
            self.logger.error(f"Required metrics not found in columns: {list(self.metrics_data.columns)}.")
            return {'status': 'Error', 'reason': f"Metrics '{loss_metric}' or '{val_loss_metric}' not found"}

        # 提取指標值
        loss_values = self.metrics_data[train_metric].tolist()
        val_loss_values = self.metrics_data[val_metric].tolist()
        patience = kwargs.get('patience', 5)
        
        try:
            # 處理單一數據點的情況
            if len(loss_values) <= 1 or len(val_loss_values) <= 1:
                return {
                    'status': 'None',
                    'divergence_point': None,
                    'train_val_gap': abs(loss_values[0] - val_loss_values[0]) if loss_values and val_loss_values else 0,
                    'note': 'Insufficient data points for overfitting analysis'
                }
            
            # 調用底層函數
            overfitting_result = detect_overfitting_simple(loss_values, val_loss_values, patience=patience)
            
            # 處理新的返回格式：現在函數返回字符串而非字典或元組
            if isinstance(overfitting_result, str):
                # 將字符串結果映射到預期的狀態格式
                if overfitting_result == 'Overfitting':
                    # 找到過擬合的開始點 - 驗證損失最低點
                    min_val_loss_idx = np.argmin(val_loss_values)
                    return {
                        'status': 'Overfitting',
                        'divergence_point': min_val_loss_idx,
                        'train_val_gap': val_loss_values[-1] - loss_values[-1]
                    }
                elif overfitting_result == 'No sign of overfitting':
                    return {
                        'status': 'None',
                        'divergence_point': None,
                        'train_val_gap': val_loss_values[-1] - loss_values[-1]
                    }
                else:
                    # 其他未知情況
                    return {
                        'status': overfitting_result,
                        'divergence_point': None
                    }
            
            # 向後兼容 - 處理舊的返回格式
            elif isinstance(overfitting_result, dict):
                return overfitting_result
            elif isinstance(overfitting_result, tuple) and len(overfitting_result) >= 2:
                # 兼容舊格式元組返回值 (status, details)
                return {
                    'status': overfitting_result[0],
                    'divergence_point': overfitting_result[1].get('divergence_point') if len(overfitting_result) > 1 and isinstance(overfitting_result[1], dict) else None
                }
            else:
                # 如果是其他類型，返回錯誤
                return {
                    'status': 'Error',
                    'reason': f"Unexpected return type from detect_overfitting_simple: {type(overfitting_result)}"
                }
        except Exception as e:
            self.logger.error(f"Error analyzing overfitting: {e}", exc_info=True)
            return {'status': 'Error', 'reason': str(e)}

    def plot_loss_curves(self, show: bool = True, save_path: Optional[str] = None):
        """繪製訓練和驗證損失曲線。"""
        logging.info("Plotting loss curves...")
        if self.metrics_data is None or 'loss' not in self.metrics_data.columns:
            self.logger.warning("Cannot plot loss curves: No data available or 'loss' metric missing.")
            fig, ax = plt.subplots()
            ax.set_title("No Loss Data Available")
            # Optionally save/show empty plot based on args
            if save_path:
                ensure_dir(os.path.dirname(save_path))
                fig.savefig(save_path)
                self.logger.info(f"Empty loss plot saved to {save_path}")
            if show:
                plt.show()
            plt.close(fig) # Close the figure to prevent display in non-interactive envs
            return fig, ax
        
        fig, ax = plt.subplots()
        ax.plot(self.metrics_data.index, self.metrics_data['loss'], label='Training Loss')
        if 'val_loss' in self.metrics_data.columns:
            ax.plot(self.metrics_data.index, self.metrics_data['val_loss'], label='Validation Loss')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            fig.savefig(save_path)
            self.logger.info(f"Loss plot saved to {save_path}")
        if show:
            plt.show()
        plt.close(fig) # Close the figure
        return fig, ax

    def plot_metric_curves(self, metric_name: str, show: bool = True, save_path: Optional[str] = None):
        """繪製指定指標的訓練和驗證曲線。"""
        logging.info(f"Plotting metric curves for '{metric_name}'...")
        train_metric_col = metric_name
        val_metric_col = f"val_{metric_name}"
        
        if self.metrics_data is None or train_metric_col not in self.metrics_data.columns:
             self.logger.warning(f"Cannot plot metric curves for '{metric_name}': No data available or '{train_metric_col}' metric missing.")
             fig, ax = plt.subplots()
             ax.set_title(f"No {metric_name} Data Available")
             if save_path:
                 ensure_dir(os.path.dirname(save_path))
                 fig.savefig(save_path)
                 self.logger.info(f"Empty {metric_name} plot saved to {save_path}")
             if show:
                 plt.show()
             plt.close(fig) # Close the figure
             return fig, ax
             
        fig, ax = plt.subplots()
        ax.plot(self.metrics_data.index, self.metrics_data[train_metric_col], label=f"Training {metric_name}")
        if val_metric_col in self.metrics_data.columns:
            ax.plot(self.metrics_data.index, self.metrics_data[val_metric_col], label=f"Validation {metric_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Training and Validation {metric_name}")
        ax.legend()

        if save_path:
            ensure_dir(os.path.dirname(save_path))
            fig.savefig(save_path)
            self.logger.info(f"{metric_name} plot saved to {save_path}")
        if show:
            plt.show()
        plt.close(fig) # Close the figure
        return fig, ax
        
    # Add find_best_epoch method if it's not already present or imported correctly
    # Assuming find_best_epoch is available (e.g., from utils.data_utils)
    def find_best_epoch(self, metric: str = 'val_loss', criteria: str = 'min') -> Optional[int]:
        """Finds the best epoch based on a given metric and criteria."""
        if self.metrics_data is None:
            logging.warning("Cannot find best epoch: history_df is not loaded.")
            return None
        if metric not in self.metrics_data.columns:
            logging.warning(f"Metric '{metric}' not found in history_df. Cannot find best epoch.")
            return None
        
        try:
            best_epoch, _ = find_best_epoch(self.metrics_data, metric, criteria)
            return best_epoch
        except Exception as e:
            logging.error(f"Error finding best epoch for metric '{metric}': {e}", exc_info=True)
            return None