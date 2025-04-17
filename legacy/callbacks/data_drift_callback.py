"""
數據漂移監控回調模塊

該模塊實現了數據漂移監控的回調功能，用於檢測訓練過程中數據分佈的變化。
"""

from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
from .base_callback import BaseCallback

class DataDriftCallback(BaseCallback):
    """
    數據漂移監控回調
    
    監控訓練過程中數據分佈的變化，檢測是否存在數據漂移問題。
    可以通過多種統計方法（如KS測試、PSI、Wasserstein距離等）來衡量分佈差異。
    
    屬性:
        reference_data: 參考數據，用作基準分佈
        feature_indices: 要監控的特徵索引列表
        monitoring_frequency: 監控頻率（每隔多少個epoch檢測一次）
        drift_threshold: 漂移閾值，超過此值時觸發警告
        drift_metrics: 用於檢測漂移的統計指標列表
    """
    
    def __init__(
        self,
        reference_data: Optional[Union[torch.Tensor, np.ndarray]] = None,
        feature_indices: Optional[List[int]] = None,
        monitoring_frequency: int = 1,
        drift_threshold: float = 0.1,
        drift_metrics: List[str] = ['ks', 'psi', 'wasserstein'],
        trigger_condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        log_to_tensorboard: bool = False
    ):
        """初始化數據漂移監控回調
        
        Args:
            reference_data: 參考數據，用作基準分佈。如果為None，將使用第一個batch作為參考
            feature_indices: 要監控的特徵索引列表。如果為None，監控所有特徵
            monitoring_frequency: 監控頻率（每隔多少個epoch檢測一次）
            drift_threshold: 漂移閾值，超過此值時觸發警告
            drift_metrics: 用於檢測漂移的統計指標列表，可選值有['ks', 'psi', 'wasserstein']
            trigger_condition: 觸發條件，決定何時執行回調邏輯
            log_to_tensorboard: 是否記錄到TensorBoard
        """
        super().__init__(trigger_condition)
        self.reference_data = reference_data
        self.feature_indices = feature_indices
        self.monitoring_frequency = monitoring_frequency
        self.drift_threshold = drift_threshold
        self.drift_metrics = drift_metrics
        self.log_to_tensorboard = log_to_tensorboard
        
        # 內部狀態
        self.reference_statistics = None
        self.current_epoch = 0
        self.tensorboard_writer = None
        self.drift_detected = False
        self.drift_scores = {}
    
    def _on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """訓練開始時初始化參考數據統計量
        
        如果提供了參考數據，則計算其統計特性
        
        Args:
            model: 正在訓練的模型
            logs: 訓練相關信息
        """
        if logs is None:
            logs = {}
        
        # 獲取tensorboard寫入器（如果有）
        self.tensorboard_writer = logs.get('tensorboard_writer', None)
        
        if self.reference_data is not None:
            if isinstance(self.reference_data, torch.Tensor):
                self.reference_data = self.reference_data.cpu().numpy()
            self.reference_statistics = self._calculate_statistics(self.reference_data)
    
    def _on_batch_begin(self, batch: int, model: nn.Module, inputs: torch.Tensor, 
                       targets: torch.Tensor, logs: Dict[str, Any] = None) -> None:
        """批次開始時，如果尚未設置參考數據，則使用第一個batch作為參考
        
        Args:
            batch: 當前批次索引
            model: 正在訓練的模型
            inputs: 輸入數據批次
            targets: 目標數據批次
            logs: 批次相關信息
        """
        if batch == 0 and self.reference_statistics is None:
            # 使用第一個batch作為參考數據
            if isinstance(inputs, torch.Tensor):
                reference_data = inputs.detach().cpu().numpy()
            else:
                reference_data = inputs
            self.reference_statistics = self._calculate_statistics(reference_data)
    
    def _on_epoch_end(self, epoch: int, model: nn.Module, train_logs: Dict[str, Any], 
                     val_logs: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
        """每個epoch結束時檢測數據漂移
        
        Args:
            epoch: 當前epoch索引
            model: 正在訓練的模型
            train_logs: 訓練集上的指標和結果
            val_logs: 驗證集上的指標和結果
            logs: 其他相關信息
        """
        self.current_epoch = epoch
        
        # 根據監控頻率決定是否檢測漂移
        if epoch % self.monitoring_frequency != 0:
            return
        
        # 獲取當前批次數據
        current_data = None
        if 'inputs' in train_logs:
            current_data = train_logs['inputs']
        elif 'last_batch' in train_logs:
            current_data = train_logs.get('last_batch', {}).get('inputs', None)
        
        if current_data is not None:
            if isinstance(current_data, torch.Tensor):
                current_data = current_data.detach().cpu().numpy()
            self._detect_drift(current_data, logs or {})
    
    def _calculate_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """計算數據統計量
        
        Args:
            data: 輸入數據
            
        Returns:
            包含各種統計量的字典
        """
        # 選擇要監控的特徵
        if self.feature_indices is not None:
            if len(data.shape) > 1:
                data = data[:, self.feature_indices]
        
        # 計算基本統計量
        statistics = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'median': np.median(data, axis=0),
            'q1': np.percentile(data, 25, axis=0),
            'q3': np.percentile(data, 75, axis=0),
            'data': data  # 保存原始數據以便進行更複雜的統計分析
        }
        
        return statistics
    
    def _detect_drift(self, current_data: np.ndarray, logs: Dict[str, Any]) -> Dict[str, Any]:
        """檢測數據漂移
        
        Args:
            current_data: 當前數據
            logs: 其他相關信息
            
        Returns:
            包含漂移檢測結果的字典
        """
        if self.reference_statistics is None:
            return {}
        
        current_statistics = self._calculate_statistics(current_data)
        drift_results = {}
        
        # 計算各種漂移指標
        for metric in self.drift_metrics:
            if metric == 'ks':
                drift_results['ks'] = self._compute_ks_statistic(
                    self.reference_statistics['data'], 
                    current_statistics['data']
                )
            elif metric == 'psi':
                drift_results['psi'] = self._compute_psi(
                    self.reference_statistics['data'], 
                    current_statistics['data']
                )
            elif metric == 'wasserstein':
                drift_results['wasserstein'] = self._compute_wasserstein_distance(
                    self.reference_statistics['data'], 
                    current_statistics['data']
                )
        
        # 檢查是否超過漂移閾值
        self.drift_detected = any(score > self.drift_threshold for score in drift_results.values())
        self.drift_scores = drift_results
        
        # 記錄漂移分數
        if self.log_to_tensorboard and self.tensorboard_writer is not None:
            for metric_name, value in drift_results.items():
                self.tensorboard_writer.add_scalar(
                    f'data_drift/{metric_name}', 
                    value, 
                    self.current_epoch
                )
            
            # 記錄分佈變化的統計量
            for stat_name in ['mean', 'std', 'min', 'max']:
                ref_val = self.reference_statistics[stat_name]
                curr_val = current_statistics[stat_name]
                
                # 計算相對變化
                if np.all(ref_val != 0):
                    rel_change = np.mean(np.abs((curr_val - ref_val) / ref_val))
                    self.tensorboard_writer.add_scalar(
                        f'data_drift/rel_change_{stat_name}',
                        rel_change,
                        self.current_epoch
                    )
        
        return drift_results
    
    def _compute_ks_statistic(self, ref_data: np.ndarray, curr_data: np.ndarray) -> float:
        """計算Kolmogorov-Smirnov統計量
        
        Args:
            ref_data: 參考數據
            curr_data: 當前數據
            
        Returns:
            KS統計量
        """
        # 簡化實現：計算每個特徵的均值的KS統計量
        ref_means = np.mean(ref_data, axis=1)
        curr_means = np.mean(curr_data, axis=1)
        
        # 計算經驗累積分佈函數(ECDF)
        def ecdf(x):
            x = np.sort(x)
            y = np.arange(1, len(x) + 1) / len(x)
            return x, y
        
        x1, y1 = ecdf(ref_means)
        x2, y2 = ecdf(curr_means)
        
        # 將兩個ECDF插值到相同的x值上
        x = np.unique(np.concatenate([x1, x2]))
        y1_interp = np.interp(x, x1, y1)
        y2_interp = np.interp(x, x2, y2)
        
        # 計算KS統計量
        ks_stat = np.max(np.abs(y1_interp - y2_interp))
        return float(ks_stat)
    
    def _compute_psi(self, ref_data: np.ndarray, curr_data: np.ndarray, bins: int = 10) -> float:
        """計算人口穩定性指數 (Population Stability Index)
        
        Args:
            ref_data: 參考數據
            curr_data: 當前數據
            bins: 分箱數量
            
        Returns:
            PSI值
        """
        # 簡化實現：計算每個特徵的均值的PSI
        ref_means = np.mean(ref_data, axis=1)
        curr_means = np.mean(curr_data, axis=1)
        
        # 計算分箱邊界
        min_val = min(np.min(ref_means), np.min(curr_means))
        max_val = max(np.max(ref_means), np.max(curr_means))
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # 計算各個分箱中的數量
        ref_counts, _ = np.histogram(ref_means, bins=bin_edges)
        curr_counts, _ = np.histogram(curr_means, bins=bin_edges)
        
        # 計算比例
        ref_pct = ref_counts / np.sum(ref_counts)
        curr_pct = curr_counts / np.sum(curr_counts)
        
        # 避免除以零和log(0)
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)
        
        # 計算PSI
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(psi)
    
    def _compute_wasserstein_distance(self, ref_data: np.ndarray, curr_data: np.ndarray) -> float:
        """計算Wasserstein距離
        
        Args:
            ref_data: 參考數據
            curr_data: 當前數據
            
        Returns:
            Wasserstein距離
        """
        # 簡化實現：計算每個特徵的均值的Wasserstein距離
        ref_means = np.mean(ref_data, axis=1)
        curr_means = np.mean(curr_data, axis=1)
        
        # 排序數據
        ref_means = np.sort(ref_means)
        curr_means = np.sort(curr_means)
        
        # 計算Wasserstein距離（實際上是1D情況下的EMD）
        n = len(ref_means)
        m = len(curr_means)
        
        # 如果長度不同，則通過線性插值使其長度相同
        if n != m:
            indices = np.linspace(0, n - 1, m)
            ref_means = np.interp(indices, np.arange(n), ref_means)
            n = m
        
        # 計算1D Wasserstein距離（等價於排序後對應點之間差的絕對值的平均）
        distance = np.mean(np.abs(ref_means - curr_means))
        return float(distance) 