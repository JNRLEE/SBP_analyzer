"""
離群值檢測回調模塊

該模塊實現了用於訓練過程中的離群值檢測功能，可以監控輸入數據、中間層特徵和預測結果中的異常模式。
"""

from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from .base_callback import BaseCallback

class OutlierDetectionCallback(BaseCallback):
    """
    離群值檢測回調
    
    監控並檢測訓練過程中的異常數據樣本，可用於識別潛在的問題數據、數據漂移，
    以及模型異常行為。
    
    屬性:
        detection_method: 使用的離群值檢測方法（'iforest'、'lof'、'mahalanobis'）
        monitoring_level: 監控的模型層次（'input'、'feature'、'output'或'all'）
        threshold: 離群值判定閾值
        target_layers: 要監控的特定層列表
        batch_size: 離群值檢測的批次大小
        window_size: 用於計算統計信息的樣本窗口大小
        outlier_handling: 檢測到離群值時的處理策略
    """
    
    def __init__(
        self,
        detection_method: str = 'iforest',
        monitoring_level: str = 'all',
        threshold: float = 0.95,
        target_layers: Optional[List[str]] = None,
        batch_size: int = 32,
        window_size: int = 500,
        check_frequency: int = 10,
        outlier_handling: str = 'log',
        notify_threshold: int = 10,
        adaptive_threshold: bool = True,
        save_outliers: bool = False,
        outlier_save_dir: str = './outliers',
        trigger_condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        """初始化離群值檢測回調
        
        Args:
            detection_method: 使用的離群值檢測方法：
                - 'iforest': 孤立森林算法
                - 'lof': 局部離群因子
                - 'mahalanobis': 馬哈拉諾比斯距離
            monitoring_level: 檢測的層級：
                - 'input': 僅檢測輸入數據
                - 'feature': 檢測中間層特徵
                - 'output': 檢測模型輸出
                - 'all': 檢測所有層級
            threshold: 判定為離群值的閾值
            target_layers: 監控的特定層名稱列表
            batch_size: 離群值檢測的批次大小
            window_size: 用於計算統計信息的樣本窗口大小
            check_frequency: 每隔多少個batch檢測一次
            outlier_handling: 檢測到離群值的處理方式：
                - 'log': 僅記錄
                - 'notify': 發出通知
                - 'save': 保存離群樣本
                - 'remove': 從訓練中暫時移除（僅影響當前batch）
            notify_threshold: 達到多少個離群值時發出通知
            adaptive_threshold: 是否自適應調整離群閾值
            save_outliers: 是否保存離群樣本
            outlier_save_dir: 離群樣本保存目錄
            trigger_condition: 觸發條件，決定何時執行回調邏輯
        """
        super().__init__(trigger_condition)
        self.detection_method = detection_method
        self.monitoring_level = monitoring_level
        self.threshold = threshold
        self.target_layers = target_layers
        self.batch_size = batch_size
        self.window_size = window_size
        self.check_frequency = check_frequency
        self.outlier_handling = outlier_handling
        self.notify_threshold = notify_threshold
        self.adaptive_threshold = adaptive_threshold
        self.save_outliers = save_outliers
        self.outlier_save_dir = outlier_save_dir
        
        # 內部狀態
        self.hook_handles = []
        self.input_samples = []
        self.feature_samples = defaultdict(list)
        self.output_samples = []
        self.outlier_indices = []
        self.outlier_scores = defaultdict(list)
        self.current_batch = 0
        self.current_epoch = 0
        self.models = {}
        self.statistics = {}
        self.total_outliers = 0
        self.batch_outlier_counts = []
        
        # 初始化日誌記錄器
        self.logger = logging.getLogger("OutlierDetection")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # 創建保存目錄（如果需要）
        if self.save_outliers:
            import os
            os.makedirs(self.outlier_save_dir, exist_ok=True)
    
    def _on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """訓練開始時設置模型鉤子和初始化檢測器
        
        Args:
            model: 正在訓練的模型
            logs: 訓練相關信息
        """
        if logs is None:
            logs = {}
        
        # 如果監控特徵層，註冊層鉤子
        if self.monitoring_level in ['feature', 'all']:
            self._register_hooks(model)
        
        # 初始化檢測器字典
        self.models = {}
    
    def _on_train_end(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """訓練結束時清理鉤子並生成最終報告
        
        Args:
            model: 訓練完成的模型
            logs: 訓練相關信息
        """
        # 移除所有鉤子
        self._remove_hooks()
        
        # 生成最終離群值分析報告
        self._generate_final_report()
        
        # 記錄總體離群值統計信息
        self.logger.info(f"訓練結束: 共檢測到 {self.total_outliers} 個離群值樣本")
    
    def _on_batch_begin(self, batch: int, logs: Dict[str, Any] = None) -> None:
        """每個batch開始時記錄輸入數據
        
        Args:
            batch: 當前batch索引
            logs: batch相關信息
        """
        self.current_batch = batch
        
        # 檢查是否有輸入數據，並且是否需要監控輸入層
        if logs and 'inputs' in logs and self.monitoring_level in ['input', 'all']:
            inputs = logs['inputs']
            if isinstance(inputs, torch.Tensor):
                # 將輸入轉換為numpy數組並保存
                inputs_np = inputs.detach().cpu().numpy()
                
                # 如果是圖像或序列等高維數據，需要平坦化
                if inputs_np.ndim > 2:
                    # 保持第一維（批次維），平坦化其餘維度
                    batch_size = inputs_np.shape[0]
                    inputs_np = inputs_np.reshape(batch_size, -1)
                
                # 將樣本添加到窗口中
                self.input_samples.extend(inputs_np)
                # 保持窗口大小
                if len(self.input_samples) > self.window_size:
                    self.input_samples = self.input_samples[-self.window_size:]
    
    def _on_batch_end(self, batch: int, model: nn.Module, loss: torch.Tensor, 
                     logs: Dict[str, Any] = None) -> None:
        """每個batch結束時檢測離群值
        
        Args:
            batch: 當前batch索引
            model: 正在訓練的模型
            loss: 當前batch的損失值
            logs: batch相關信息
        """
        # 每隔指定批次進行檢測
        if batch % self.check_frequency != 0:
            return
        
        # 檢查是否有足夠的樣本進行檢測
        if ((self.monitoring_level in ['input', 'all'] and len(self.input_samples) < 50) or 
            (self.monitoring_level in ['feature', 'all'] and all(len(samples) < 50 for samples in self.feature_samples.values()))):
            return
        
        # 檢測輸入數據中的離群值
        if self.monitoring_level in ['input', 'all'] and self.input_samples:
            input_outliers, input_scores = self._detect_outliers('input', np.array(self.input_samples[-self.batch_size:]))
            if input_outliers:
                self._handle_outliers('input', input_outliers, input_scores, logs)
        
        # 檢測特徵層的離群值
        if self.monitoring_level in ['feature', 'all']:
            for layer_name, samples in self.feature_samples.items():
                if len(samples) < self.batch_size:
                    continue
                
                feature_outliers, feature_scores = self._detect_outliers(
                    f'feature_{layer_name}', 
                    np.array(samples[-self.batch_size:])
                )
                if feature_outliers:
                    self._handle_outliers(f'feature_{layer_name}', feature_outliers, feature_scores, logs)
        
        # 檢測輸出數據中的離群值
        if self.monitoring_level in ['output', 'all'] and logs and 'outputs' in logs:
            outputs = logs['outputs']
            if isinstance(outputs, torch.Tensor):
                outputs_np = outputs.detach().cpu().numpy()
                
                # 儲存輸出樣本
                self.output_samples.extend(outputs_np)
                if len(self.output_samples) > self.window_size:
                    self.output_samples = self.output_samples[-self.window_size:]
                
                output_outliers, output_scores = self._detect_outliers('output', outputs_np)
                if output_outliers:
                    self._handle_outliers('output', output_outliers, output_scores, logs)
    
    def _on_epoch_end(self, epoch: int, model: nn.Module, train_logs: Dict[str, Any], 
                     val_logs: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
        """每個epoch結束時更新檢測器並分析離群趨勢
        
        Args:
            epoch: 當前epoch索引
            model: 正在訓練的模型
            train_logs: 訓練集上的指標和結果
            val_logs: 驗證集上的指標和結果
            logs: 其他相關信息
        """
        self.current_epoch = epoch
        
        # 計算本epoch的離群值率
        if self.batch_outlier_counts:
            epoch_outlier_rate = sum(self.batch_outlier_counts) / (len(self.batch_outlier_counts) * self.batch_size)
            self.logger.info(f"Epoch {epoch}: 離群值率 = {epoch_outlier_rate:.4f}")
            
            # 重置batch離群計數
            self.batch_outlier_counts = []
        
        # 如果使用自適應閾值，根據離群值率調整閾值
        if self.adaptive_threshold and hasattr(self, 'epoch_outlier_rates'):
            self._adjust_threshold()
        
        # 如果有足夠的數據，重新訓練檢測器
        if epoch % 5 == 0:  # 每5個epoch重新訓練
            self._retrain_detectors()
    
    def _register_hooks(self, model: nn.Module) -> None:
        """註冊層鉤子以收集中間層特徵
        
        Args:
            model: 要監控的模型
        """
        if self.target_layers is None:
            # 如果未指定目標層，自動選擇可能的目標層
            layers_dict = {}
            for name, module in model.named_modules():
                # 監控卷積層、線性層和批歸一化層
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
                    layers_dict[name] = module
            
            self.target_layers = list(layers_dict.keys())
        
        # 為每個目標層註冊前向鉤子
        for name, module in model.named_modules():
            if name in self.target_layers:
                handle = module.register_forward_hook(
                    lambda m, inp, outp, name=name: self._feature_hook(name, outp)
                )
                self.hook_handles.append(handle)
    
    def _remove_hooks(self) -> None:
        """移除所有註冊的鉤子"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    def _feature_hook(self, layer_name: str, output: torch.Tensor) -> None:
        """收集層特徵的鉤子函數
        
        Args:
            layer_name: 層名稱
            output: 層輸出張量
        """
        if isinstance(output, tuple):
            output = output[0]
        
        # 將輸出轉換為numpy數組
        output_np = output.detach().cpu().numpy()
        
        # 如果是高維特徵，平坦化處理
        if output_np.ndim > 2:
            # 保持第一維（批次維），平坦化其餘維度
            batch_size = output_np.shape[0]
            output_np = output_np.reshape(batch_size, -1)
        
        # 存儲特徵
        self.feature_samples[layer_name].extend(output_np)
        # 保持窗口大小
        if len(self.feature_samples[layer_name]) > self.window_size:
            self.feature_samples[layer_name] = self.feature_samples[layer_name][-self.window_size:]
    
    def _detect_outliers(self, data_source: str, samples: np.ndarray) -> Tuple[List[int], List[float]]:
        """使用選定的方法檢測離群值
        
        Args:
            data_source: 數據源標識（'input', 'feature_layerX', 'output'）
            samples: 要檢測的樣本數組 (batch_size x features)
            
        Returns:
            outlier_indices: 離群樣本的索引列表
            outlier_scores: 離群樣本的異常分數列表
        """
        # 檢查樣本是否為空或者維度不正確
        if samples.size == 0 or samples.ndim != 2:
            return [], []
        
        # 確保模型存在，如果不存在則初始化
        if data_source not in self.models:
            self._initialize_detector(data_source)
        
        # 檢測離群值
        detector = self.models[data_source]
        
        if self.detection_method == 'iforest':
            # 使用孤立森林算法
            scores = detector.score_samples(samples)
            # 孤立森林返回的是負異常分數，分數越低越異常
            # 將其轉換為正常分數，分數越高越異常
            scores = -scores
            
        elif self.detection_method == 'lof':
            # 使用局部離群因子
            scores = -detector.score_samples(samples)
            
        elif self.detection_method == 'mahalanobis':
            # 使用馬哈拉諾比斯距離
            scores = np.zeros(samples.shape[0])
            for i, sample in enumerate(samples):
                # 計算樣本與均值之間的馬哈拉諾比斯距離
                diff = sample - self.statistics[data_source]['mean']
                scores[i] = np.sqrt(diff.dot(self.statistics[data_source]['inv_cov']).dot(diff))
        
        # 根據閾值確定離群值
        if self.detection_method in ['iforest', 'lof']:
            # 對於這些算法，分數越高越異常
            threshold_value = np.percentile(scores, self.threshold * 100)
            outlier_indices = np.where(scores > threshold_value)[0].tolist()
            outlier_scores = scores[outlier_indices].tolist()
        else:
            # 對於馬哈拉諾比斯距離，距離越大越異常
            threshold_value = np.percentile(scores, self.threshold * 100)
            outlier_indices = np.where(scores > threshold_value)[0].tolist()
            outlier_scores = scores[outlier_indices].tolist()
        
        return outlier_indices, outlier_scores
    
    def _initialize_detector(self, data_source: str) -> None:
        """初始化指定數據源的離群值檢測器
        
        Args:
            data_source: 數據源標識（'input', 'feature_layerX', 'output'）
        """
        # 獲取訓練樣本
        if data_source == 'input':
            if len(self.input_samples) < 50:  # 需要足夠的樣本
                return
            samples = np.array(self.input_samples)
        elif data_source.startswith('feature_'):
            layer_name = data_source[len('feature_'):]
            if layer_name not in self.feature_samples or len(self.feature_samples[layer_name]) < 50:
                return
            samples = np.array(self.feature_samples[layer_name])
        elif data_source == 'output':
            if len(self.output_samples) < 50:
                return
            samples = np.array(self.output_samples)
        else:
            self.logger.warning(f"未知的數據源: {data_source}")
            return
        
        # 確保樣本是二維的
        if samples.ndim != 2:
            self.logger.warning(f"樣本維度不正確，期望為2，實際為{samples.ndim}")
            return
        
        # 初始化檢測器
        if self.detection_method == 'iforest':
            # 孤立森林算法
            detector = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=0.1,  # 預期的離群比例
                random_state=42
            )
            detector.fit(samples)
            self.models[data_source] = detector
            
        elif self.detection_method == 'lof':
            # 局部離群因子
            detector = LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.1,
                novelty=True
            )
            detector.fit(samples)
            self.models[data_source] = detector
            
        elif self.detection_method == 'mahalanobis':
            # 馬哈拉諾比斯距離
            # 計算均值和協方差矩陣
            mean = np.mean(samples, axis=0)
            cov = np.cov(samples, rowvar=False)
            
            # 使用偽逆來處理可能的奇異協方差矩陣
            inv_cov = np.linalg.pinv(cov)
            
            self.statistics[data_source] = {
                'mean': mean,
                'cov': cov,
                'inv_cov': inv_cov
            }
            
            # 馬哈拉諾比斯方法不需要顯式模型
            self.models[data_source] = None
        
        else:
            self.logger.warning(f"未支持的檢測方法: {self.detection_method}")
    
    def _handle_outliers(self, data_source: str, outlier_indices: List[int], 
                        outlier_scores: List[float], logs: Dict[str, Any] = None) -> None:
        """處理檢測到的離群值
        
        Args:
            data_source: 數據源標識
            outlier_indices: 離群樣本的索引
            outlier_scores: 離群樣本的異常分數
            logs: batch相關信息
        """
        # 更新總體離群值計數
        num_outliers = len(outlier_indices)
        self.total_outliers += num_outliers
        self.batch_outlier_counts.append(num_outliers)
        
        # 儲存離群值分數
        self.outlier_scores[data_source].extend(
            [(self.current_epoch, self.current_batch, idx, score) 
             for idx, score in zip(outlier_indices, outlier_scores)]
        )
        
        # 根據處理策略採取行動
        if self.outlier_handling == 'log' or self.outlier_handling == 'notify':
            self.logger.info(
                f"Epoch {self.current_epoch}, Batch {self.current_batch}: "
                f"檢測到 {num_outliers} 個離群值，來源: {data_source}"
            )
            
            # 如果離群值數量超過通知閾值，發出警告
            if self.outlier_handling == 'notify' and num_outliers >= self.notify_threshold:
                self.logger.warning(
                    f"警告: 離群值數量 ({num_outliers}) 超過通知閾值 ({self.notify_threshold})，"
                    f"數據源: {data_source}"
                )
        
        # 如果需要保存離群樣本
        if self.save_outliers:
            self._save_outlier_samples(data_source, outlier_indices, outlier_scores, logs)
    
    def _save_outlier_samples(self, data_source: str, outlier_indices: List[int], 
                             outlier_scores: List[float], logs: Dict[str, Any] = None) -> None:
        """保存離群樣本供後續分析
        
        Args:
            data_source: 數據源標識
            outlier_indices: 離群樣本的索引
            outlier_scores: 離群樣本的異常分數
            logs: batch相關信息
        """
        import os
        import pickle
        
        # 創建保存目錄
        save_dir = os.path.join(self.outlier_save_dir, f"epoch_{self.current_epoch}", data_source)
        os.makedirs(save_dir, exist_ok=True)
        
        # 獲取樣本數據
        if data_source == 'input' and logs and 'inputs' in logs:
            inputs = logs['inputs']
            if isinstance(inputs, torch.Tensor):
                samples = inputs.detach().cpu().numpy()
                for i, idx in enumerate(outlier_indices):
                    if idx < len(samples):
                        # 保存樣本和分數
                        outlier_data = {
                            'sample': samples[idx],
                            'score': outlier_scores[i],
                            'epoch': self.current_epoch,
                            'batch': self.current_batch,
                            'source': data_source
                        }
                        
                        # 如果有標籤，也保存標籤
                        if logs and 'targets' in logs:
                            targets = logs['targets']
                            if isinstance(targets, torch.Tensor) and idx < len(targets):
                                outlier_data['target'] = targets[idx].detach().cpu().numpy()
                        
                        # 保存為pickle文件
                        file_path = os.path.join(
                            save_dir, 
                            f"outlier_batch{self.current_batch}_idx{idx}_score{outlier_scores[i]:.4f}.pkl"
                        )
                        with open(file_path, 'wb') as f:
                            pickle.dump(outlier_data, f)
        
        elif data_source.startswith('feature_'):
            # 特徵離群值當前無法直接保存，因為鉤子收集的是批次級別的
            # 可以實現更複雜的追踪機制來匹配特定樣本
            pass
        
        elif data_source == 'output' and logs and 'outputs' in logs:
            outputs = logs['outputs']
            if isinstance(outputs, torch.Tensor):
                samples = outputs.detach().cpu().numpy()
                for i, idx in enumerate(outlier_indices):
                    if idx < len(samples):
                        # 保存樣本和分數
                        outlier_data = {
                            'sample': samples[idx],
                            'score': outlier_scores[i],
                            'epoch': self.current_epoch,
                            'batch': self.current_batch,
                            'source': data_source
                        }
                        
                        # 如果有標籤，也保存標籤
                        if logs and 'targets' in logs:
                            targets = logs['targets']
                            if isinstance(targets, torch.Tensor) and idx < len(targets):
                                outlier_data['target'] = targets[idx].detach().cpu().numpy()
                        
                        # 保存為pickle文件
                        file_path = os.path.join(
                            save_dir, 
                            f"outlier_batch{self.current_batch}_idx{idx}_score{outlier_scores[i]:.4f}.pkl"
                        )
                        with open(file_path, 'wb') as f:
                            pickle.dump(outlier_data, f)
    
    def _retrain_detectors(self) -> None:
        """重新訓練所有檢測器以適應數據分佈的變化"""
        # 重新訓練輸入層檢測器
        if 'input' in self.models and len(self.input_samples) >= 50:
            self._initialize_detector('input')
        
        # 重新訓練特徵層檢測器
        for layer_name in self.feature_samples.keys():
            data_source = f'feature_{layer_name}'
            if data_source in self.models and len(self.feature_samples[layer_name]) >= 50:
                self._initialize_detector(data_source)
        
        # 重新訓練輸出層檢測器
        if 'output' in self.models and len(self.output_samples) >= 50:
            self._initialize_detector('output')
    
    def _adjust_threshold(self) -> None:
        """根據離群值率自適應調整閾值"""
        # 計算最近幾個epoch的平均離群值率
        recent_rates = self.epoch_outlier_rates[-3:] if len(self.epoch_outlier_rates) >= 3 else self.epoch_outlier_rates
        avg_rate = sum(recent_rates) / len(recent_rates)
        
        # 目標離群值率，通常在1-5%範圍內
        target_rate = 0.03  # 3%
        
        # 如果當前離群值率過高，提高閾值（更嚴格）
        if avg_rate > target_rate * 1.5:
            self.threshold = min(0.99, self.threshold + 0.01)
            self.logger.info(f"離群值率過高 ({avg_rate:.4f})，提高閾值至 {self.threshold:.4f}")
        
        # 如果當前離群值率過低，降低閾值（更寬鬆）
        elif avg_rate < target_rate * 0.5:
            self.threshold = max(0.8, self.threshold - 0.01)
            self.logger.info(f"離群值率過低 ({avg_rate:.4f})，降低閾值至 {self.threshold:.4f}")
    
    def _generate_final_report(self) -> None:
        """生成最終的離群值分析報告"""
        # 記錄總體統計信息
        self.logger.info("=" * 50)
        self.logger.info("離群值檢測最終報告")
        self.logger.info("=" * 50)
        self.logger.info(f"總訓練樣本數: {self.current_epoch * self.batch_size * self.check_frequency}")
        self.logger.info(f"檢測到的離群值總數: {self.total_outliers}")
        self.logger.info(f"總體離群值率: {self.total_outliers / (self.current_epoch * self.batch_size * self.check_frequency + 1e-10):.4f}")
        
        # 記錄每個數據源的離群值統計
        for data_source, scores in self.outlier_scores.items():
            if scores:
                self.logger.info(f"\n數據源 {data_source}:")
                self.logger.info(f"  離群值數量: {len(scores)}")
                
                # 計算平均異常分數
                avg_score = sum(score for _, _, _, score in scores) / len(scores)
                self.logger.info(f"  平均異常分數: {avg_score:.4f}")
                
                # 按epoch分組統計
                epoch_counts = defaultdict(int)
                for epoch, _, _, _ in scores:
                    epoch_counts[epoch] += 1
                
                # 找出離群值最多的epoch
                if epoch_counts:
                    max_epoch = max(epoch_counts.items(), key=lambda x: x[1])
                    self.logger.info(f"  離群值最多的Epoch: {max_epoch[0]}，數量: {max_epoch[1]}")
        
        # 如果保存了圖表，記錄路徑
        if self.save_outliers:
            self.logger.info(f"\n離群樣本保存在: {os.path.abspath(self.outlier_save_dir)}")
        
        self.logger.info("=" * 50) 