"""
中間層數據分析模組。

此模組提供用於分析深度學習模型中間層輸出（激活值）的功能，包括值分布、統計特性等。

Classes:
    IntermediateDataAnalyzer: 分析中間層數據的類別。
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import torch
import numpy as np
import pandas as pd

from .base_analyzer import BaseAnalyzer

class IntermediateDataAnalyzer(BaseAnalyzer):
    """
    中間層數據分析器，用於分析模型中間層輸出的特性。
    
    此分析器關注模型訓練過程中通過hooks獲取的中間層數據，分析其分布、統計特性和變化趨勢。
    
    Attributes:
        experiment_dir (str): 實驗結果的目錄路徑。
        hooks_dir (str): 存放hooks數據的目錄。
        available_epochs (List[int]): 可用的輪次列表。
        available_layers (Dict[int, List[str]]): 每個輪次可用的層列表。
    """
    
    def __init__(self, experiment_dir: str):
        """
        初始化中間層數據分析器。
        
        Args:
            experiment_dir (str): 實驗結果的目錄路徑。
        """
        super().__init__(experiment_dir)
        self.hooks_dir = os.path.join(experiment_dir, 'hooks')
        self.available_epochs = []
        self.available_layers = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 確認hooks目錄是否存在
        if not os.path.exists(self.hooks_dir):
            self.logger.warning(f"Hooks目錄不存在: {self.hooks_dir}")
        else:
            self._scan_available_data()
    
    def _scan_available_data(self) -> None:
        """
        掃描並記錄可用的輪次和層。
        """
        try:
            # 掃描輪次目錄
            epoch_dirs = [d for d in os.listdir(self.hooks_dir) 
                          if d.startswith('epoch_') and os.path.isdir(os.path.join(self.hooks_dir, d))]
            
            self.available_epochs = sorted([int(d.split('_')[1]) for d in epoch_dirs])
            
            # 對每個輪次掃描可用的層
            for epoch in self.available_epochs:
                epoch_dir = os.path.join(self.hooks_dir, f'epoch_{epoch}')
                
                # 查找形如 *_activation_batch_*.pt 的文件
                activation_files = [f for f in os.listdir(epoch_dir) if f.endswith('.pt') and 'activation_batch' in f]
                
                # 提取層名
                layer_names = []
                for file in activation_files:
                    parts = file.split('_activation_batch_')
                    if len(parts) == 2:
                        layer_names.append(parts[0])
                
                self.available_layers[epoch] = sorted(set(layer_names))
            
            self.logger.info(f"找到 {len(self.available_epochs)} 個輪次的數據")
            for epoch, layers in self.available_layers.items():
                self.logger.info(f"輪次 {epoch}: 找到 {len(layers)} 個層的數據")
        
        except Exception as e:
            self.logger.error(f"掃描可用數據時出錯: {e}")
    
    def load_activation(self, layer_name: str, epoch: int, batch: int = 0) -> Optional[torch.Tensor]:
        """
        載入特定層、特定輪次、特定批次的激活值。
        
        Args:
            layer_name (str): 層的名稱。
            epoch (int): 輪次。
            batch (int, optional): 批次索引。默認為0。
        
        Returns:
            Optional[torch.Tensor]: 激活值張量，如果數據不存在則返回None。
        """
        if epoch not in self.available_epochs:
            self.logger.warning(f"輪次 {epoch} 的數據不可用")
            return None
        
        if layer_name not in self.available_layers.get(epoch, []):
            self.logger.warning(f"輪次 {epoch} 中不存在層 {layer_name} 的數據")
            return None
        
        file_path = os.path.join(self.hooks_dir, f'epoch_{epoch}', f'{layer_name}_activation_batch_{batch}.pt')
        
        try:
            if os.path.exists(file_path):
                return torch.load(file_path)
            else:
                self.logger.warning(f"文件不存在: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"載入激活值數據時出錯: {e}")
            return None
    
    def analyze_layer(self, layer_name: str, epochs: Optional[List[int]] = None, batch: int = 0) -> Dict:
        """
        分析特定層在一個或多個輪次的激活值特性。
        
        Args:
            layer_name (str): 層的名稱。
            epochs (List[int], optional): 要分析的輪次列表。如果為None，則分析所有可用輪次。
            batch (int, optional): 要分析的批次索引。默認為0。
        
        Returns:
            Dict: 包含分析結果的字典。
        """
        if epochs is None:
            epochs = self.available_epochs
        
        # 篩選可用的輪次
        valid_epochs = [e for e in epochs if e in self.available_epochs and layer_name in self.available_layers.get(e, [])]
        
        if not valid_epochs:
            self.logger.warning(f"沒有找到層 {layer_name} 在指定輪次的數據")
            return {}
        
        layer_results = {
            'layer_name': layer_name,
            'epochs_analyzed': valid_epochs,
            'batch_analyzed': batch,
            'statistics': {},
            'distribution': {},
            'sparsity': {},
            'changes_over_epochs': {}
        }
        
        try:
            # 載入每個輪次的數據並分析
            epoch_data = {}
            for epoch in valid_epochs:
                activation = self.load_activation(layer_name, epoch, batch)
                if activation is not None:
                    epoch_data[epoch] = activation
                    
                    # 計算基本統計量
                    stats = self._compute_statistics(activation)
                    layer_results['statistics'][epoch] = stats
                    
                    # 計算分布特性
                    dist = self._compute_distribution(activation)
                    layer_results['distribution'][epoch] = dist
                    
                    # 計算稀疏度
                    sparsity = self._compute_sparsity(activation)
                    layer_results['sparsity'][epoch] = sparsity
            
            # 分析跨輪次的變化
            if len(epoch_data) > 1:
                layer_results['changes_over_epochs'] = self._analyze_epoch_changes(epoch_data)
            
            return layer_results
            
        except Exception as e:
            self.logger.error(f"分析層 {layer_name} 時出錯: {e}")
            return {}
    
    def analyze(self, layers: Optional[List[str]] = None, epochs: Optional[List[int]] = None, 
                batch: int = 0, save_results: bool = True) -> Dict:
        """
        分析一個或多個層在一個或多個輪次的激活值特性。
        
        Args:
            layers (List[str], optional): 要分析的層列表。如果為None，則分析所有可用層。
            epochs (List[int], optional): 要分析的輪次列表。如果為None，則分析所有可用輪次。
            batch (int, optional): 要分析的批次索引。默認為0。
            save_results (bool, optional): 是否保存分析結果。默認為True。
        
        Returns:
            Dict: 包含分析結果的字典。
        """
        # 如果未指定輪次，使用所有可用輪次
        if epochs is None:
            epochs = self.available_epochs
        
        # 如果未指定層，使用所有可用層
        if layers is None:
            # 使用所有輪次中出現的層的并集
            all_layers = set()
            for epoch in epochs:
                if epoch in self.available_layers:
                    all_layers.update(self.available_layers[epoch])
            layers = sorted(all_layers)
        
        # 初始化結果字典
        self.results = {
            'layers_analyzed': layers,
            'epochs_analyzed': epochs,
            'batch_analyzed': batch,
            'layer_results': {}
        }
        
        try:
            # 對每一層進行分析
            for layer_name in layers:
                layer_result = self.analyze_layer(layer_name, epochs, batch)
                if layer_result:
                    self.results['layer_results'][layer_name] = layer_result
            
            if save_results:
                self.save_results(os.path.join(self.experiment_dir, 'analysis'))
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"分析中間層數據時出錯: {e}")
            raise
    
    def _compute_statistics(self, activation: torch.Tensor) -> Dict:
        """
        計算激活值的基本統計量。
        
        Args:
            activation (torch.Tensor): 激活值張量。
        
        Returns:
            Dict: 包含統計量的字典。
        """
        # 轉換為NumPy數組以便計算
        if activation.is_cuda:
            activation = activation.cpu()
        act_np = activation.detach().numpy()
        
        # 計算基本統計量
        return {
            'mean': float(np.mean(act_np)),
            'std': float(np.std(act_np)),
            'min': float(np.min(act_np)),
            'max': float(np.max(act_np)),
            'median': float(np.median(act_np)),
            '25th_percentile': float(np.percentile(act_np, 25)),
            '75th_percentile': float(np.percentile(act_np, 75))
        }
    
    def _compute_distribution(self, activation: torch.Tensor) -> Dict:
        """
        分析激活值的分布特性。
        
        Args:
            activation (torch.Tensor): 激活值張量。
        
        Returns:
            Dict: 包含分布特性的字典。
        """
        # 轉換為NumPy數組以便計算
        if activation.is_cuda:
            activation = activation.cpu()
        act_np = activation.detach().numpy().flatten()
        
        # 計算分布特性
        return {
            'skewness': float(self._compute_skewness(act_np)),
            'kurtosis': float(self._compute_kurtosis(act_np)),
            'histogram_data': self._compute_histogram(act_np)
        }
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """
        計算數據的偏度。
        
        Args:
            data (np.ndarray): 數據數組。
        
        Returns:
            float: 偏度值。
        """
        # 具體實現...
        return 0.0
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """
        計算數據的峰度。
        
        Args:
            data (np.ndarray): 數據數組。
        
        Returns:
            float: 峰度值。
        """
        # 具體實現...
        return 0.0
    
    def _compute_histogram(self, data: np.ndarray) -> Dict:
        """
        計算數據的直方圖。
        
        Args:
            data (np.ndarray): 數據數組。
        
        Returns:
            Dict: 包含直方圖數據的字典。
        """
        # 具體實現...
        hist, bin_edges = np.histogram(data, bins=50)
        return {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
    
    def _compute_sparsity(self, activation: torch.Tensor) -> Dict:
        """
        計算激活值的稀疏度。
        
        Args:
            activation (torch.Tensor): 激活值張量。
        
        Returns:
            Dict: 包含稀疏度指標的字典。
        """
        # 轉換為NumPy數組以便計算
        if activation.is_cuda:
            activation = activation.cpu()
        act_np = activation.detach().numpy()
        
        # 計算L1稀疏度 (越接近0越稀疏)
        l1_sparsity = np.mean(np.abs(act_np))
        
        # 計算零元素比例
        zero_threshold = 1e-6  # 接近零的閾值
        zero_ratio = np.mean(np.abs(act_np) < zero_threshold)
        
        return {
            'l1_sparsity': float(l1_sparsity),
            'zero_ratio': float(zero_ratio)
        }
    
    def _analyze_epoch_changes(self, epoch_data: Dict[int, torch.Tensor]) -> Dict:
        """
        分析層在不同輪次之間的變化。
        
        Args:
            epoch_data (Dict[int, torch.Tensor]): 不同輪次的激活值數據。
        
        Returns:
            Dict: 包含變化分析結果的字典。
        """
        # 具體實現...
        return {} 