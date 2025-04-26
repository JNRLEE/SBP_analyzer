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
from data_loader.hook_data_loader import HookDataLoader
from metrics.layer_activity_metrics import *
from utils.tensor_utils import preprocess_activations, to_numpy, get_tensor_statistics

class IntermediateDataAnalyzer(BaseAnalyzer):
    """
    中間層數據分析器，用於分析模型中間層輸出的特性。
    
    此分析器關注模型訓練過程中通過hooks獲取的中間層數據，分析其分布、統計特性和變化趨勢。
    
    Attributes:
        experiment_dir (str): 實驗結果的目錄路徑。
        hooks_dir (str): 存放hooks數據的目錄。
        available_epochs (List[int]): 可用的輪次列表。
        available_layers (Dict[int, List[str]]): 每個輪次可用的層列表。
        experiment_name (str): 實驗名稱。
        output_dir (str): 輸出目錄路徑。
        dead_neuron_threshold (float): 死亡神經元的閾值。
        saturated_neuron_threshold (float): 飽和神經元的閾值。
        layer_results (Dict): 依照層名稱組織的分析結果。
    """
    
    def __init__(self, experiment_dir: str, hook_data_loader: Optional[HookDataLoader] = None,
                 experiment_name: Optional[str] = None, output_dir: Optional[str] = None,
                 dead_neuron_threshold: float = 0.01, saturated_neuron_threshold: float = 0.99):
        """
        初始化中間層數據分析器。
        
        Args:
            experiment_dir (str): 實驗結果的目錄路徑。
            hook_data_loader: 可選的HookDataLoader實例，如果不提供則自動創建。
            experiment_name: 可選的實驗名稱，用於生成報告和保存結果。
            output_dir: 可選的輸出目錄路徑，用於保存分析結果。
            dead_neuron_threshold: 死亡神經元的閾值，默認為0.01。
            saturated_neuron_threshold: 飽和神經元的閾值，默認為0.99。
        """
        super().__init__(experiment_dir)
        self.hooks_dir = os.path.join(experiment_dir, 'hooks')
        self.available_epochs = []
        self.available_layers = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.experiment_name = experiment_name or os.path.basename(experiment_dir)
        self.output_dir = output_dir or os.path.join(experiment_dir, 'analysis')
        self.dead_neuron_threshold = dead_neuron_threshold
        self.saturated_neuron_threshold = saturated_neuron_threshold
        
        # 確認hooks目錄是否存在
        if not os.path.exists(self.hooks_dir):
            self.logger.warning(f"Hooks目錄不存在: {self.hooks_dir}")
        else:
            self._scan_available_data()
        
        # 初始化數據載入器
        if hook_data_loader is None:
            self.hook_loader = HookDataLoader(experiment_dir)
        else:
            self.hook_loader = hook_data_loader
            
        # 儲存分析結果
        self.layer_statistics: Dict[str, Dict] = {}
        self.activation_dynamics: Dict[str, Dict] = {}
        self.layer_relationships: Dict[str, Dict[str, float]] = {}
        self.dead_neurons_analysis: Dict[str, Dict] = {}
        self.layer_results: Dict[str, Dict] = {}  # 新增: 存儲以層名稱組織的分析結果
    
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
    
    def analyze_with_loader(self, 
                          hook_loader: HookDataLoader,
                          epochs: Optional[List[int]] = None, 
                          layers: Optional[List[str]] = None,
                          batch_idx: int = 0,
                          preprocess_args: Dict = None) -> Dict:
        """
        使用提供的 HookDataLoader 分析中間層數據
        
        此方法允許使用外部提供的數據加載器進行分析，便於測試和模擬數據分析。
        
        Args:
            hook_loader: 用於加載數據的HookDataLoader實例
            epochs: 要分析的輪次列表，如果為None則分析所有可用輪次
            layers: 要分析的層名稱列表，如果為None則分析所有可用層
            batch_idx: 要分析的批次索引，預設為0
            preprocess_args: 預處理參數，可包含normalize, flatten, remove_outliers等選項
            
        Returns:
            Dict: 分析結果摘要
        """
        self.logger.info("使用提供的數據加載器開始分析中間層數據...")
        
        # 臨時保存原始的hook_loader
        original_hook_loader = self.hook_loader
        
        try:
            # 設置為提供的hook_loader
            self.hook_loader = hook_loader
            
            # 使用提供的hook_loader進行分析
            results = self.analyze(epochs, layers, batch_idx, preprocess_args)
            
            return results
            
        finally:
            # 恢復原始的hook_loader
            self.hook_loader = original_hook_loader
    
    def analyze(self, 
               epochs: Optional[List[int]] = None, 
               layers: Optional[List[str]] = None,
               batch_idx: int = 0,
               preprocess_args: Dict = None) -> Dict:
        """
        分析中間層數據
        
        Args:
            epochs: 要分析的輪次列表，如果為None則分析所有可用輪次
            layers: 要分析的層名稱列表，如果為None則分析所有可用層
            batch_idx: 要分析的批次索引，預設為0
            preprocess_args: 預處理參數，可包含normalize, flatten, remove_outliers等選項
            
        Returns:
            Dict: 分析結果摘要
        """
        self.logger.info("開始分析中間層數據...")
        
        # --- Clear previous results to prevent state leakage --- 
        self.layer_statistics.clear()
        self.activation_dynamics.clear()
        self.layer_relationships.clear()
        self.dead_neurons_analysis.clear()
        # --- End clearing --- 
        
        # 確定要分析的輪次
        available_epochs = self.hook_loader.list_available_epochs()
        if not available_epochs:
            self.logger.warning("找不到任何可用的輪次數據")
            return {"error": "No epoch data available"}
            
        if epochs is None:
            epochs_to_analyze = available_epochs
        else:
            epochs_to_analyze = [e for e in epochs if e in available_epochs]
            if not epochs_to_analyze:
                self.logger.warning(f"指定的輪次 {epochs} 不可用，可用輪次為 {available_epochs}")
                return {"error": f"Specified epochs not available. Available epochs: {available_epochs}"}
        
        # 分析每個輪次的層激活值
        all_layer_metrics = {}
        
        # 設置預處理參數
        if preprocess_args is None:
            preprocess_args = {
                'normalize': None,  # 可選: 'minmax', 'zscore', 'l1', 'l2'
                'flatten': False,
                'remove_outliers': False
            }
        
        # 逐輪次分析
        for epoch in epochs_to_analyze:
            # 獲取該輪次可用的層
            available_layers = self.hook_loader.list_available_layer_activations(epoch)
            if not available_layers:
                self.logger.warning(f"輪次 {epoch} 沒有可用的層激活數據")
                continue
                
            # 確定要分析的層
            if layers is None:
                layers_to_analyze = available_layers
            else:
                layers_to_analyze = [l for l in layers if l in available_layers]
                if not layers_to_analyze:
                    self.logger.warning(f"輪次 {epoch} 中沒有指定的層 {layers}，可用層為 {available_layers}")
                    continue
            
            # 載入並分析每個層
            for layer_name in layers_to_analyze:
                try:
                    # 載入激活值
                    activation = self.hook_loader.load_layer_activation(layer_name, epoch, batch_idx)
                    
                    # 預處理激活值
                    processed_activation = preprocess_activations(
                        activation,
                        flatten=preprocess_args.get('flatten', False),
                        normalize=preprocess_args.get('normalize'),
                        remove_outliers=preprocess_args.get('remove_outliers', False)
                    )
                    
                    # 分析該層的激活值
                    layer_metrics = self._analyze_layer_activation(processed_activation, layer_name, epoch)
                    
                    # 儲存結果
                    layer_key = f"{layer_name}_epoch_{epoch}"
                    all_layer_metrics[layer_key] = layer_metrics
                    
                    # 更新類屬性
                    self.layer_statistics[layer_key] = layer_metrics
                    
                except Exception as e:
                    self.logger.error(f"分析層 {layer_name} 在輪次 {epoch} 時出錯: {e}")
        
        # 分析層之間的關係
        self._analyze_layer_relationships(epochs_to_analyze, layers_to_analyze, batch_idx)
        
        # 分析激活值的時間變化（跨輪次）
        self._analyze_activation_dynamics(epochs_to_analyze, layers_to_analyze, batch_idx)
        
        # --- DEBUG: Print keys before reorganization ---
        # self.logger.debug(f"DEBUG: self.layer_statistics keys before reorganization: {list(self.layer_statistics.keys())}")
        # --- END DEBUG --- 
        
        # 重新組織層統計結果，按層名稱分組
        layer_results = {}
        for layer_key, metrics in self.layer_statistics.items():
            parts = layer_key.split('_epoch_')
            if len(parts) == 2:
                layer_name, epoch_str = parts
                epoch = int(epoch_str)
                
                # 如果層名稱不在結果字典中，則創建
                if layer_name not in layer_results:
                    layer_results[layer_name] = {}
                
                # 添加輪次數據
                layer_results[layer_name][str(epoch)] = metrics
        
        # 更新實例屬性
        self.layer_results = layer_results
        
        return {
            "layer_results": layer_results,  # 使用重組後的結構
            "activation_dynamics": self.activation_dynamics,
            "layer_relationships": self.layer_relationships,
            "dead_neurons_analysis": self.dead_neurons_analysis
        }
    
    def _analyze_layer_activation(self, 
                                activation: torch.Tensor, 
                                layer_name: str, 
                                epoch: int) -> Dict:
        """
        分析單個層的激活值
        
        Args:
            activation: 層的激活值張量
            layer_name: 層名稱
            epoch: 輪次編號
            
        Returns:
            Dict: 層激活值分析結果
        """
        # 計算基本統計量
        basic_stats = calculate_activation_statistics(activation)
        
        # 計算稀疏度
        sparsity = calculate_activation_sparsity(activation)
        
        # 計算飽和度 - 使用實例中設置的飽和閾值
        saturation = calculate_activation_saturation(activation, saturation_threshold=self.saturated_neuron_threshold)
        
        # 計算有效秩
        try:
            effective_rank = calculate_effective_rank(activation)
        except Exception as e:
            self.logger.warning(f"計算有效秩時出錯: {e}")
            effective_rank = None
        
        # 計算特徵一致性
        try:
            if activation.dim() >= 3:
                coherence = calculate_feature_coherence(activation)
            else:
                coherence = None
        except Exception as e:
            self.logger.warning(f"計算特徵一致性時出錯: {e}")
            coherence = None
        
        # 檢測失活神經元 - 使用實例中設置的死亡閾值
        try:
            dead_neurons = detect_dead_neurons(activation, threshold=self.dead_neuron_threshold)
            self.dead_neurons_analysis[f"{layer_name}_epoch_{epoch}"] = dead_neurons
        except Exception as e:
            self.logger.warning(f"檢測失活神經元時出錯: {e}")
            dead_neurons = {"error": str(e)}
        
        # 整合結果
        results = {
            "basic_statistics": basic_stats,
            "sparsity": sparsity,
            "saturation": saturation,
            "effective_rank": effective_rank,
            "feature_coherence": coherence,
            "dead_neurons": dead_neurons,
            # 添加百分比數據以便輕鬆訪問
            "dead_percentage": dead_neurons.get("dead_ratio", 0),
            "saturated_percentage": saturation  # 直接使用浮點數值，不再嘗試使用 .get()
        }
        
        return results
    
    def _analyze_layer_relationships(self, 
                                   epochs: List[int], 
                                   layers: List[str],
                                   batch_idx: int) -> None:
        """
        分析不同層之間的關係
        
        Args:
            epochs: 要分析的輪次列表
            layers: 要分析的層列表
            batch_idx: 批次索引
        """
        if not epochs or not layers or len(layers) < 2:
            return
        
        # 只分析第一個輪次的層間關係，避免計算太多
        epoch = epochs[0]
        
        # 載入所有層的激活值
        layer_activations = {}
        for layer_name in layers:
            try:
                activation = self.hook_loader.load_layer_activation(layer_name, epoch, batch_idx)
                layer_activations[layer_name] = activation
            except:
                continue
        
        # 如果少於2個層，無法計算關係
        if len(layer_activations) < 2:
            return
        
        # 計算每對層之間的相似度
        layer_names = list(layer_activations.keys())
        for i in range(len(layer_names)):
            for j in range(i+1, len(layer_names)):
                layer1 = layer_names[i]
                layer2 = layer_names[j]
                
                try:
                    similarity = calculate_layer_similarity(
                        layer_activations[layer1], 
                        layer_activations[layer2]
                    )
                    
                    # 儲存結果
                    key = f"{layer1}_vs_{layer2}_epoch_{epoch}"
                    self.layer_relationships[key] = similarity
                except Exception as e:
                    self.logger.warning(f"計算層 {layer1} 和 {layer2} 的相似度時出錯: {e}")
    
    def _analyze_activation_dynamics(self, 
                                   epochs: List[int], 
                                   layers: List[str],
                                   batch_idx: int) -> None:
        """
        分析層激活值隨時間變化的動態特性
        
        Args:
            epochs: 要分析的輪次列表
            layers: 要分析的層列表
            batch_idx: 批次索引
        """
        if len(epochs) < 2:
            return  # 至少需要兩個輪次來計算變化
            
        # 按層組織激活值
        for layer_name in layers:
            activations_by_epoch = {}
            
            # 載入每個輪次的激活值
            for epoch in epochs:
                try:
                    activation = self.hook_loader.load_layer_activation(layer_name, epoch, batch_idx)
                    activations_by_epoch[epoch] = activation
                except:
                    continue
            
            # 如果至少有兩個輪次的數據，才計算動態特性
            if len(activations_by_epoch) >= 2:
                try:
                    dynamics = calculate_activation_dynamics(activations_by_epoch)
                    self.activation_dynamics[layer_name] = dynamics
                except Exception as e:
                    self.logger.warning(f"計算層 {layer_name} 的激活動態時出錯: {e}")
    
    def get_layer_statistics(self, layer_name: str, epoch: int) -> Optional[Dict]:
        """
        獲取指定層和輪次的統計分析結果
        
        Args:
            layer_name: 層名稱
            epoch: 輪次編號
            
        Returns:
            Optional[Dict]: 層統計信息，如果不存在則返回None
        """
        key = f"{layer_name}_epoch_{epoch}"
        return self.layer_statistics.get(key)
    
    def get_layer_dynamics(self, layer_name: str) -> Optional[Dict]:
        """
        獲取指定層的激活動態分析結果
        
        Args:
            layer_name: 層名稱
            
        Returns:
            Optional[Dict]: 層動態信息，如果不存在則返回None
        """
        return self.activation_dynamics.get(layer_name)
    
    def get_layer_relationship(self, layer1: str, layer2: str, epoch: int) -> Optional[Dict]:
        """
        獲取兩個層之間的關係分析結果
        
        Args:
            layer1: 第一個層的名稱
            layer2: 第二個層的名稱
            epoch: 輪次編號
            
        Returns:
            Optional[Dict]: 層關係信息，如果不存在則返回None
        """
        # 嘗試兩種順序的鍵
        key1 = f"{layer1}_vs_{layer2}_epoch_{epoch}"
        key2 = f"{layer2}_vs_{layer1}_epoch_{epoch}"
        
        return self.layer_relationships.get(key1) or self.layer_relationships.get(key2)
    
    def get_problematic_layers(self, criteria: Dict[str, float] = None) -> List[Dict]:
        """
        檢測模型中存在問題的層。
        
        分析所有層的統計數據，找出符合問題標準的層，如死亡神經元比例高、
        飽和神經元比例高、過大或過小的激活值等。
        
        Args:
            criteria (Dict[str, float], optional): 問題判斷標準，包含閾值。
                可包含 'dead_neuron_threshold', 'saturated_neuron_threshold',
                'low_variance_threshold' 等。如果為None，則使用預設標準。
            
        Returns:
            List[Dict]: 每個問題層的信息，包含層名稱、問題類型、問題指標值等。
        """
        self.logger.info("檢測問題層...")
        
        # 預設標準
        default_criteria = {
            'dead_neuron_threshold': self.dead_neuron_threshold,       # 死亡神經元比例閾值
            'saturated_neuron_threshold': self.saturated_neuron_threshold,  # 飽和神經元比例閾值
            'low_variance_threshold': 0.01,     # 低方差閾值
            'high_mean_threshold': 5.0,         # 高均值閾值
            'weight_update_threshold': 0.001    # 權重更新閾值
        }
        
        # 使用提供的標準或預設標準
        if criteria is None:
            criteria = default_criteria
        else:
            # 合併自訂和預設標準
            for k, v in default_criteria.items():
                if k not in criteria:
                    criteria[k] = v
        
        problematic_layers = []
        
        # 遍歷所有層統計數據
        for layer_key, stats in self.layer_statistics.items():
            layer_name, epoch_str = layer_key.split('_epoch_')
            epoch = int(epoch_str)
            
            problems = []
            problem_metrics = {}
            
            # 檢查死亡神經元
            if 'dead_neurons' in stats and stats['dead_neurons'] is not None:
                dead_ratio = stats['dead_neurons'].get('ratio', 0)
                if dead_ratio >= criteria['dead_neuron_threshold']:
                    problems.append('high_dead_neurons')
                    problem_metrics['dead_neuron_ratio'] = dead_ratio
            
            # 檢查飽和神經元（如果可用）
            if 'saturated_neurons' in stats and stats['saturated_neurons'] is not None:
                sat_ratio = stats['saturated_neurons'].get('ratio', 0)
                if sat_ratio >= criteria['saturated_neuron_threshold']:
                    problems.append('high_saturated_neurons')
                    problem_metrics['saturated_neuron_ratio'] = sat_ratio
            
            # 檢查低方差
            if 'basic_statistics' in stats and 'std' in stats['basic_statistics']:
                std = stats['basic_statistics']['std']
                if std < criteria['low_variance_threshold']:
                    problems.append('low_variance')
                    problem_metrics['std'] = std
            
            # 檢查高均值（可能表示梯度爆炸）
            if 'basic_statistics' in stats and 'mean' in stats['basic_statistics']:
                mean = abs(stats['basic_statistics']['mean'])
                if mean > criteria['high_mean_threshold']:
                    problems.append('high_mean')
                    problem_metrics['mean'] = mean
            
            # 如果檢測到問題，添加到結果列表
            if problems:
                problematic_layers.append({
                    'layer_name': layer_name,
                    'epoch': epoch,
                    'problems': problems,
                    'metrics': problem_metrics
                })
        
        # 根據問題嚴重程度排序（問題數量越多越嚴重）
        problematic_layers.sort(key=lambda x: len(x['problems']), reverse=True)
        
        self.logger.info(f"檢測到 {len(problematic_layers)} 個問題層")
        return problematic_layers
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        生成包含所有層統計信息的摘要數據框。
        
        將所有層的統計數據整合到一個pandas DataFrame中，方便查詢和可視化。
        數據框包含層名稱、輪次、基本統計量（平均值、標準差等）、稀疏度、有效秩等信息。
        
        Returns:
            pd.DataFrame: 包含所有層統計信息的數據框。
        """
        self.logger.info("生成層統計摘要數據框...")
        
        # 初始化數據列表
        data = []
        
        # 遍歷所有層統計數據
        for layer_key, stats in self.layer_statistics.items():
            # 解析層名稱和輪次
            try:
                layer_name, epoch_str = layer_key.split('_epoch_')
                epoch = int(epoch_str)
            except ValueError:
                self.logger.warning(f"無法解析層鍵 {layer_key}，跳過")
                continue
            
            # 基本數據結構
            row_data = {
                'layer_name': layer_name,
                'epoch': epoch
            }
            
            # 提取基本統計數據
            if 'basic_statistics' in stats:
                for stat_name, stat_value in stats['basic_statistics'].items():
                    row_data[f'stat_{stat_name}'] = stat_value
            
            # 提取稀疏度
            if 'sparsity' in stats:
                row_data['sparsity'] = stats['sparsity']
            
            # 提取有效秩
            if 'effective_rank' in stats:
                row_data['effective_rank'] = stats['effective_rank']
            
            # 提取死亡神經元信息
            if 'dead_neurons' in stats and stats['dead_neurons'] is not None:
                row_data['dead_neuron_count'] = stats['dead_neurons'].get('count', 0)
                row_data['dead_neuron_ratio'] = stats['dead_neurons'].get('ratio', 0.0)
            
            # 提取飽和神經元信息
            if 'saturated_neurons' in stats and stats['saturated_neurons'] is not None:
                row_data['saturated_neuron_count'] = stats['saturated_neurons'].get('count', 0)
                row_data['saturated_neuron_ratio'] = stats['saturated_neurons'].get('ratio', 0.0)
            
            # 提取其他可能存在的指標
            for key, value in stats.items():
                if key not in ['basic_statistics', 'sparsity', 'effective_rank', 'dead_neurons', 'saturated_neurons']:
                    if isinstance(value, (int, float, str, bool)):
                        row_data[key] = value
            
            # 添加到數據列表
            data.append(row_data)
        
        # 創建數據框
        if data:
            df = pd.DataFrame(data)
            
            # 設置索引
            df = df.set_index(['layer_name', 'epoch'])
            
            # 按層名稱和輪次排序
            df = df.sort_index()
            
            # 重置索引，便於後續處理
            df = df.reset_index()
            
            self.logger.info(f"生成了包含 {len(df)} 行數據的摘要數據框")
            return df
        else:
            self.logger.warning("無法生成摘要數據框，沒有有效的層統計數據")
            return pd.DataFrame()  # 返回空數據框
    
    def generate_report(self) -> Dict:
        """
        生成分析報告
        
        Returns:
            Dict: 包含所有分析結果的報告字典
        """
        report = {
            'summary': {
                'total_layers_analyzed': len(self.layer_statistics),
                'layers_with_dynamics': len(self.activation_dynamics),
                'layer_relationships_analyzed': len(self.layer_relationships)
            },
            'problematic_layers': self.get_problematic_layers(),
            'layer_statistics': self.layer_statistics,
            'activation_dynamics': self.activation_dynamics,
            'layer_relationships': self.layer_relationships
        }
        
        return report 