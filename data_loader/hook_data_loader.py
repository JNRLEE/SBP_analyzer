"""
Hook數據載入器模組。

此模組提供用於載入模型鉤子(hooks)數據的功能，包括中間層激活值等。

Classes:
    HookDataLoader: 載入模型鉤子數據的類別。
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import json

import torch
import numpy as np
import glob
from pathlib import Path

from .base_loader import BaseLoader
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.tensor_utils import (
    preprocess_activations, normalize_tensor, handle_nan_values, 
    reduce_dimensions, handle_sparse_tensor, batch_process_activations
)

logger = logging.getLogger(__name__)

class HookDataLoader(BaseLoader):
    """
    Hook數據載入器，用於載入模型中間層的激活值等數據。
    
    此載入器可以載入hooks/目錄下的.pt文件，包括各輪次各層的激活值。
    
    Attributes:
        experiment_dir (str): 實驗結果的目錄路徑。
        hooks_dir (str): 存放hooks數據的目錄。
        available_epochs (List[int]): 可用的輪次列表。
        available_layers (Dict[Tuple[int, int], List[str]]): 每個輪次、每個批次可用的層列表 (鍵為 (epoch, batch))。
        available_batches (Dict[int, List[int]]): 每個輪次可用的批次列表。
    """
    
    def __init__(self, experiment_dir: str):
        """
        初始化Hook數據載入器。
        
        Args:
            experiment_dir (str): 實驗結果的目錄路徑。
        """
        super().__init__(experiment_dir)
        self.hooks_dir = os.path.join(experiment_dir, 'hooks')
        self.available_epochs = []
        self.available_layers: Dict[Tuple[int, int], List[str]] = {}
        self.available_batches: Dict[int, List[int]] = {}
        
        if not os.path.exists(self.hooks_dir):
            logger.warning(f"Hooks目錄不存在: {self.hooks_dir}")
            self.hooks_available = False
        else:
            self.hooks_available = True
            logger.info(f"已找到Hooks目錄: {self.hooks_dir}")
            self._scan_available_data()
    
    def _scan_available_data(self) -> None:
        """
        掃描並記錄可用的輪次、批次和層。
        """
        try:
            epoch_dirs = [d for d in Path(self.hooks_dir).iterdir()
                          if d.is_dir() and d.name.startswith('epoch_')]
            
            self.available_epochs = sorted([int(d.name.split('_')[1]) for d in epoch_dirs])
            self.available_layers = {}
            self.available_batches = {}
            
            for epoch_dir_path in epoch_dirs:
                epoch = int(epoch_dir_path.name.split('_')[1])
                batches_in_epoch = set()
                layers_in_epoch_batch: Dict[Tuple[int, int], Set[str]] = {}
                
                # 掃描 epoch 目錄下的所有 .pt 文件
                for pt_file in epoch_dir_path.glob('*.pt'):
                    filename = pt_file.name
                    # 檢查激活文件格式: {layer_name}_activation_batch_{batch_idx}.pt
                    if '_activation_batch_' in filename and filename.endswith('.pt'):
                        parts = filename.rsplit('_activation_batch_', 1)
                        layer_name = parts[0]
                        try:
                            batch_idx_str = parts[1].split('.')[0]
                            batch_idx = int(batch_idx_str)
                            batches_in_epoch.add(batch_idx)
                            epoch_batch_key = (epoch, batch_idx)
                            if epoch_batch_key not in layers_in_epoch_batch:
                                layers_in_epoch_batch[epoch_batch_key] = set()
                            layers_in_epoch_batch[epoch_batch_key].add(layer_name)
                        except (IndexError, ValueError):
                            logger.warning(f"無法解析文件名格式: {filename} in {epoch_dir_path}")
                            continue
                    # 可以添加對其他文件格式的掃描邏輯，例如 batch_{batch_idx}_data.pt
                    elif filename.startswith('batch_') and filename.endswith('_data.pt'):
                         try:
                            batch_idx = int(filename.split('_')[1])
                            batches_in_epoch.add(batch_idx)
                         except (IndexError, ValueError):
                             logger.warning(f"無法解析批次數據文件名格式: {filename} in {epoch_dir_path}")
                             continue
                
                if batches_in_epoch:
                    self.available_batches[epoch] = sorted(list(batches_in_epoch))
                    for key, layers_set in layers_in_epoch_batch.items():
                        self.available_layers[key] = sorted(list(layers_set))
                else:
                     self.available_batches[epoch] = []
            
            logger.info(f"找到 {len(self.available_epochs)} 個輪次的數據")
            total_layers = sum(len(v) for v in self.available_layers.values())
            logger.info(f"共找到 {total_layers} 個層激活記錄分佈在各輪次和批次中")
            # for epoch in self.available_epochs:
            #     logger.info(f"輪次 {epoch}: 可用批次 {self.available_batches.get(epoch, [])}")
            #     for batch in self.available_batches.get(epoch, []):
            #         logger.info(f"  批次 {batch}: 可用層 {self.available_layers.get((epoch, batch), [])}")
        
        except Exception as e:
            logger.exception(f"掃描可用數據時出錯: {e}", exc_info=True)
            self.available_epochs = []
            self.available_layers = {}
            self.available_batches = {}
    
    def load(self, layer_name: Optional[str] = None, epoch: Optional[int] = None, 
             batch: int = 0) -> Dict[str, Any]:
        """
        載入hook數據。
        
        Args:
            layer_name (str, optional): 要載入的層名稱。如果為None，則返回所有可用層的信息。
            epoch (int, optional): 要載入的輪次。如果為None，則返回所有可用輪次的信息。
            batch (int, optional): 要載入的批次索引。默認為0。
        
        Returns:
            Dict[str, Any]: 包含載入數據的字典。
        """
        if layer_name is None and epoch is None:
            # 返回可用資源概覽
            return {
                'available_epochs': self.available_epochs,
                'available_layers': self.available_layers,
                'hooks_dir': self.hooks_dir
            }
        
        if epoch is not None and layer_name is not None:
            # 載入特定層的激活值
            activation = self.load_activation(layer_name, epoch, batch)
            if activation is not None:
                return {
                    'layer_name': layer_name,
                    'epoch': epoch,
                    'batch': batch,
                    'activation': activation
                }
            else:
                return {}
        
        if epoch is not None:
            # 載入特定輪次的摘要
            summary = self.load_epoch_summary(epoch)
            return {
                'epoch': epoch,
                'summary': summary,
                'available_layers': self.available_layers.get((epoch, batch), [])
            }
        
        if layer_name is not None:
            # 載入特定層在所有可用輪次的數據
            layer_data = {}
            for e in self.available_epochs:
                if layer_name in self.available_layers.get((e, batch), []):
                    activation = self.load_activation(layer_name, e, batch)
                    if activation is not None:
                        layer_data[e] = activation
            
            return {
                'layer_name': layer_name,
                'batch': batch,
                'available_epochs': list(layer_data.keys()),
                'activations': layer_data
            }
        
        return {}
    
    def load_training_summary(self) -> Optional[Dict]:
        """
        載入整體訓練摘要
        
        Returns:
            包含訓練摘要資訊的字典，若檔案不存在則返回None
        """
        if not self.hooks_available:
            logger.warning("Hooks不可用，無法載入訓練摘要")
            return None
            
        summary_path = os.path.join(self.hooks_dir, 'training_summary.pt')
        if os.path.exists(summary_path):
            try:
                return torch.load(summary_path)
            except Exception as e:
                logger.error(f"載入訓練摘要時發生錯誤: {str(e)}")
                return None
        else:
            logger.warning(f"訓練摘要檔案不存在: {summary_path}")
            return None
    
    def load_evaluation_results(self, dataset: str = 'test') -> Optional[Dict]:
        """
        載入特定資料集的評估結果
        
        Args:
            dataset: 資料集名稱，預設為'test'
            
        Returns:
            包含評估結果的字典，若檔案不存在則返回None
        """
        if not self.hooks_available:
            return None
            
        eval_path = os.path.join(self.hooks_dir, f'evaluation_results_{dataset}.pt')
        if os.path.exists(eval_path):
            try:
                return torch.load(eval_path)
            except Exception as e:
                logger.error(f"載入評估結果時發生錯誤: {str(e)}")
                return None
        else:
            logger.warning(f"評估結果檔案不存在: {eval_path}")
            return None
    
    def load_epoch_summary(self, epoch: int) -> Optional[Dict]:
        """
        載入特定輪次的摘要
        
        Args:
            epoch: 輪次編號
            
        Returns:
            包含輪次摘要的字典，若檔案不存在則返回None
            
        Raises:
            FileNotFoundError: 當指定的輪次目錄不存在時。
        """
        if not self.hooks_available:
            raise FileNotFoundError("Hooks數據不可用")
            
        epoch_dir = os.path.join(self.hooks_dir, f'epoch_{epoch}')
        if not os.path.exists(epoch_dir):
            raise FileNotFoundError(f"輪次目錄不存在: {epoch_dir}")
            
        summary_path = os.path.join(epoch_dir, 'epoch_summary.pt')
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"輪次摘要檔案不存在: {summary_path}")
            
        try:
            return torch.load(summary_path)
        except Exception as e:
            logger.error(f"載入輪次摘要時發生錯誤: {str(e)}")
            raise FileNotFoundError(f"無法載入輪次摘要: {summary_path}") from e
    
    def load_batch_data(self, epoch: int, batch: int) -> Optional[Dict]:
        """
        載入特定批次的資料
        
        Args:
            epoch: 輪次編號
            batch: 批次編號
            
        Returns:
            包含批次資料的字典，若檔案不存在則返回None
        """
        if not self.hooks_available:
            return None
            
        batch_path = os.path.join(self.hooks_dir, f'epoch_{epoch}', f'batch_{batch}_data.pt')
        if os.path.exists(batch_path):
            try:
                return torch.load(batch_path)
            except Exception as e:
                logger.error(f"載入批次資料時發生錯誤: {str(e)}")
                return None
        else:
            logger.warning(f"批次資料檔案不存在: {batch_path}")
            return None
    
    def load_layer_activation(self, layer_name: str, epoch: int, batch: int = 0) -> Optional[torch.Tensor]:
        """
        載入指定層在特定輪次和批次的激活值。

        Args:
            layer_name (str): 層名稱。
            epoch (int): 輪次編號。
            batch (int): 批次索引，默認為0。

        Returns:
            Optional[torch.Tensor]: 激活值張量，如果文件不存在或無法載入則返回None。

        Raises:
            FileNotFoundError: 如果輪次目錄不存在。
        """
        if not self.hooks_available:
            logger.warning("Hooks不可用，無法載入激活值。")
            return None

        epoch_dir = Path(self.hooks_dir) / f'epoch_{epoch}'
        if not epoch_dir.is_dir():
            raise FileNotFoundError(f"輪次目錄不存在: {epoch_dir}")

        # 使用新的文件名格式
        activation_filename = f"{layer_name}_activation_batch_{batch}.pt"
        activation_path = epoch_dir / activation_filename

        if activation_path.exists():
            try:
                activation_data = torch.load(activation_path, map_location=torch.device('cpu'))
                logger.debug(f"成功載入激活值: {activation_path}")
                return activation_data
            except FileNotFoundError:
                # 這個內部異常應該由 activation_path.exists() 捕獲，但為了健壯性保留
                logger.warning(f"激活值文件不存在（內部錯誤）：{activation_path}")
                raise FileNotFoundError(f"找不到層 {layer_name} 在輪次 {epoch} 批次 {batch} 的激活值檔案")
            except Exception as e:
                logger.error(f"無法載入激活值文件 {activation_path}: {str(e)}", exc_info=True)
                return None
        else:
            logger.warning(f"激活值文件不存在: {activation_path}")
            # 為了與之前的測試兼容，這裡也引發 FileNotFoundError
            raise FileNotFoundError(f"找不到層 {layer_name} 在輪次 {epoch} 批次 {batch} 的激活值檔案")
    
    def list_available_epochs(self) -> List[int]:
        """
        返回可用的輪次列表。
        """
        return self.available_epochs
    
    def list_available_batches(self, epoch: int) -> List[int]:
        """
        返回指定輪次可用的批次索引列表。

        Args:
            epoch (int): 輪次編號。

        Returns:
            List[int]: 可用的批次索引列表。
        """
        return self.available_batches.get(epoch, [])
    
    def list_available_layers(self, epoch: int, batch: int = 0) -> List[str]:
        """
        返回指定輪次和批次可用的層名稱列表。

        Args:
            epoch (int): 輪次編號。
            batch (int): 批次索引。

        Returns:
            List[str]: 可用的層名稱列表。
        """
        return self.available_layers.get((epoch, batch), [])
    
    def list_available_layer_activations(self, epoch: int) -> List[str]:
        """[兼容性] 返回指定輪次第一個可用批次的層列表。建議使用 list_available_layers。"""
        batches = self.list_available_batches(epoch)
        if batches:
            logger.warning("調用了舊的 list_available_layer_activations 方法，僅返回第一個批次的層。推薦使用 list_available_layers(epoch, batch)。")
            return self.list_available_layers(epoch, batches[0])
        return []
    
    def get_activation_shape(self, layer_name: str, epoch: int, batch: int = 0) -> Optional[Tuple]:
        """
        獲取特定層激活值的形狀
        
        Args:
            layer_name: 層的名稱
            epoch: 輪次編號
            batch: 批次編號，預設為0
            
        Returns:
            激活值張量的形狀，若無法載入則返回None
        """
        activation = self.load_layer_activation(layer_name, epoch, batch)
        if activation is not None:
            return activation.shape
        return None
    
    def load_all_epoch_activations(self, layer_name: str, batch: int = 0) -> Dict[int, torch.Tensor]:
        """
        載入特定層在所有可用輪次中的激活值
        
        Args:
            layer_name: 層的名稱
            batch: 批次編號，預設為0
            
        Returns:
            以輪次為鍵，激活值張量為值的字典
        """
        results = {}
        epochs = self.list_available_epochs()
        
        for epoch in epochs:
            activation = self.load_layer_activation(layer_name, epoch, batch)
            if activation is not None:
                results[epoch] = activation
                
        return results
    
    def load_all_layer_activations_for_epoch(self, epoch: int, batch: int = 0) -> Dict[str, torch.Tensor]:
        """
        載入特定輪次中所有層的激活值
        
        Args:
            epoch: 輪次編號
            batch: 批次編號，預設為0
            
        Returns:
            以層名稱為鍵，激活值張量為值的字典
        """
        results = {}
        layers = self.list_available_layers(epoch, batch)
        
        for layer in layers:
            activation = self.load_layer_activation(layer, epoch, batch)
            if activation is not None:
                results[layer] = activation
                
        return results
    
    def get_experiment_summary(self) -> Dict:
        """
        獲取實驗摘要，包含hooks資料的概覽
        
        Returns:
            包含實驗摘要的字典
        """
        if not self.hooks_available:
            return {"status": "hooks不可用"}
            
        epochs = self.list_available_epochs()
        if not epochs:
            return {"status": "沒有找到epoch資料"}
            
        # 選擇第一個epoch和batch來獲取可用層
        first_epoch = epochs[0]
        layers = self.list_available_layers(first_epoch)
        
        # 獲取訓練摘要
        training_summary = self.load_training_summary()
        
        return {
            "status": "hooks可用",
            "available_epochs": epochs,
            "available_layers": layers,
            "training_summary_available": training_summary is not None,
            "evaluation_results_available": self.load_evaluation_results() is not None
        }

    def load_activation(self, layer_name: str, epoch: int, batch: int = 0) -> Optional[torch.Tensor]:
        """
        載入特定層在特定輪次和批次的激活值。
        
        Args:
            layer_name (str): 層的名稱。
            epoch (int): 輪次。
            batch (int, optional): 批次索引。默認為0。
        
        Returns:
            Optional[torch.Tensor]: 激活值張量。
            
        Raises:
            FileNotFoundError: 當指定的層或輪次不存在時。
        """
        if not self.hooks_available:
            raise FileNotFoundError("Hooks數據不可用")
            
        epoch_dir = os.path.join(self.hooks_dir, f'epoch_{epoch}')
        if not os.path.exists(epoch_dir):
            raise FileNotFoundError(f"輪次目錄不存在: {epoch_dir}")
            
        file_path = os.path.join(epoch_dir, f'{layer_name}_activation_batch_{batch}.pt')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到層 {layer_name} 在輪次 {epoch} 批次 {batch} 的激活值檔案")
            
        try:
            return torch.load(file_path)
        except Exception as e:
            logger.error(f"載入激活值數據時出錯: {e}")
            raise FileNotFoundError(f"無法載入文件: {file_path}") from e

    def load_activation_with_preprocessing(self, layer_name: str, epoch: int, batch: int = 0,
                                        preprocess_config: Optional[Dict[str, Any]] = None) -> Optional[torch.Tensor]:
        """
        載入並預處理激活值數據。
        
        Args:
            layer_name (str): 層的名稱。
            epoch (int): 輪次。
            batch (int, optional): 批次索引。默認為0。
            preprocess_config (Optional[Dict[str, Any]], optional): 預處理配置。
                可能的選項包括：
                - normalize: 正規化方法 ('minmax', 'zscore', 'robust')
                - handle_outliers: 是否處理離群值
                - reduce_dims: 降維方法
                - flatten: 是否將多維張量展平為2D
                - remove_outliers: 是否移除離群值
                - outlier_threshold: 離群值閾值
                
        Returns:
            Optional[torch.Tensor]: 預處理後的激活值張量。
        """
        # 檢查文件是否存在
        file_path = os.path.join(self.hooks_dir, f'epoch_{epoch}', f'{layer_name}_activation_batch_{batch}.pt')
        if os.path.exists(file_path):
            try:
                activation = torch.load(file_path)
            except Exception as e:
                logger.error(f"載入激活值數據時出錯: {e}")
                return None
        else:
            return None
            
        if activation is None:
            return None
            
        if preprocess_config is None:
            return activation
            
        # 預處理步驟
        if preprocess_config.get('handle_outliers'):
            activation = handle_nan_values(activation)
            
        if preprocess_config.get('remove_outliers'):
            threshold = preprocess_config.get('outlier_threshold', 3.0)
            mean = activation.mean()
            std = activation.std()
            activation = torch.where(
                torch.abs(activation - mean) > threshold * std,
                mean,
                activation
            )
            
        if 'normalize' in preprocess_config:
            method = preprocess_config['normalize']
            activation = normalize_tensor(activation, method=method)
            
        if preprocess_config.get('reduce_dims'):
            method = preprocess_config['reduce_dims']
            activation = reduce_dimensions(activation, method=method)
            
        if preprocess_config.get('handle_sparse'):
            activation = handle_sparse_tensor(activation)
            
        # 展平多維張量
        if preprocess_config.get('flatten'):
            batch_size = activation.shape[0]
            activation = activation.reshape(batch_size, -1)
            
        return activation

    def load_activations_batch(self, layer_names: List[str], epochs: List[int], 
                             batch: int = 0, preprocess_config: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        批量載入多個層和多個輪次的激活值。
        
        Args:
            layer_names (List[str]): 層名稱列表。
            epochs (List[int]): 輪次列表。
            batch (int, optional): 批次索引。默認為0。
            preprocess_config (Optional[Dict[str, Any]], optional): 預處理配置。
                
        Returns:
            Dict[str, Dict[int, torch.Tensor]]: 層名稱到輪次激活值映射的映射。
        """
        results = {}
        for layer_name in layer_names:
            layer_results = {}
            for epoch in epochs:
                try:
                    if preprocess_config:
                        activation = self.load_activation_with_preprocessing(
                            layer_name, epoch, batch, preprocess_config
                        )
                    else:
                        activation = self.load_activation(layer_name, epoch, batch)
                        
                    if activation is not None:
                        layer_results[epoch] = activation
                except FileNotFoundError:
                    continue
            if layer_results:
                results[layer_name] = layer_results
        return results

    def load_layer_statistics(self, layer_name: str, epoch: int, batch: int = 0) -> Optional[Dict[str, Any]]:
        """
        載入層統計數據，如果不存在則計算基本統計信息。
        
        Args:
            layer_name (str): 層的名稱。
            epoch (int): 輪次。
            batch (int, optional): 批次索引。默認為0。
            
        Returns:
            Optional[Dict[str, Any]]: 層統計數據字典。
            
        Raises:
            FileNotFoundError: 當指定的層或輪次不存在時。
        """
        # 先尋找是否有預先計算的統計數據
        stats_path = os.path.join(self.hooks_dir, f'epoch_{epoch}', f'{layer_name}_stats.json')
        
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"載入層統計數據時出錯: {e}")
            
        # 未找到預計算的統計數據，嘗試讀取激活值並計算
        try:
            activation = self.load_activation(layer_name, epoch, batch)
        except FileNotFoundError:
            return None
            
        if activation is None:
            return None
        
        # 計算基本統計信息
        result = {
            'layer_name': layer_name,
            'epoch': epoch,
            'batch': batch,
            'shape': list(activation.shape),
            'dtype': str(activation.dtype),
            'min': float(activation.min().item()),
            'max': float(activation.max().item()),
            'mean': float(activation.mean().item()),
            'std': float(activation.std().item()),
            'sparsity': float((activation == 0).sum().item() / activation.numel()),
            'has_nan': bool(torch.isnan(activation).any().item()),
            'has_inf': bool(torch.isinf(activation).any().item())
        }
        
        return result

    def get_activation_metadata(self) -> Dict[str, Any]:
        """
        獲取所有可用激活值數據的元數據。
        
        Returns:
            Dict[str, Any]: 包含元數據的字典。
        """
        metadata = {
            'available_epochs': self.available_epochs,
            'available_layers': self.available_layers,
            'layer_details': {}
        }
        
        # 收集每個層的基本信息
        for epoch in self.available_epochs:
            for layer_name in self.available_layers.get((epoch, 0), []):
                if layer_name not in metadata['layer_details']:
                    metadata['layer_details'][layer_name] = {'epochs': []}
                
                # 添加這個輪次的層信息
                activation = self.load_activation(layer_name, epoch, 0)
                if activation is not None:
                    epoch_info = {
                        'epoch': epoch,
                        'shape': list(activation.shape),
                        'dtype': str(activation.dtype)
                    }
                    metadata['layer_details'][layer_name]['epochs'].append(epoch_info)
        
        return metadata

    def save_processed_activation(self, activation: torch.Tensor, 
                                  layer_name: str, epoch: int, batch: int = 0,
                                  suffix: str = 'processed') -> str:
        """
        保存處理後的激活值數據。
        
        Args:
            activation (torch.Tensor): 處理後的激活值張量。
            layer_name (str): 層的名稱。
            epoch (int): 輪次。
            batch (int, optional): 批次索引。默認為0。
            suffix (str, optional): 文件名後綴。默認為'processed'。
            
        Returns:
            str: 保存的文件路徑。
        """
        epoch_dir = os.path.join(self.hooks_dir, f'epoch_{epoch}')
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir, exist_ok=True)
        
        file_path = os.path.join(epoch_dir, f'{layer_name}_{suffix}_batch_{batch}.pt')
        torch.save(activation, file_path)
        
        return file_path

    def load_hook_data(self, epochs: Optional[List[int]] = None, 
                    layers: Optional[List[str]] = None, 
                    batch_idx: int = 0) -> Dict[str, Any]:
        """
        載入指定層和輪次的所有hook數據。
        
        這個方法會載入多個層和輪次的激活值數據，作為analyzer_interface中analyze方法的輸入。
        
        Args:
            epochs (Optional[List[int]], optional): 要載入的輪次列表。如果為None，載入所有可用輪次。
            layers (Optional[List[str]], optional): 要載入的層列表。如果為None，載入所有可用層。
            batch_idx (int, optional): 要載入的批次索引。默認為0。
            
        Returns:
            Dict[str, Any]: 包含所有載入的數據的字典，結構為:
                {
                    'metadata': {
                        'epochs': [載入的輪次列表],
                        'layers': [載入的層列表],
                        'batch_idx': 批次索引
                    },
                    'activations': {
                        'layer_name1': {
                            epoch1: tensor1,
                            epoch2: tensor2,
                            ...
                        },
                        'layer_name2': {
                            ...
                        }
                    },
                    'summaries': {
                        epoch1: 輪次1摘要,
                        epoch2: 輪次2摘要,
                        ...
                    },
                    'training_summary': 訓練摘要(如果可用)
                }
        """
        if not self.hooks_available:
            logger.warning("Hooks數據不可用")
            return {}
            
        # 確定要載入的輪次
        available_epochs = self.list_available_epochs()
        if not epochs:
            epochs_to_load = available_epochs
        else:
            epochs_to_load = [e for e in epochs if e in available_epochs]
            
        if not epochs_to_load:
            logger.warning("沒有找到可用的輪次數據")
            return {}
            
        # 對於第一個輪次，獲取可用的層
        first_epoch = epochs_to_load[0]
        available_layers = self.list_available_layers(first_epoch, batch_idx)
        
        # 確定要載入的層
        if not layers:
            layers_to_load = available_layers
        else:
            layers_to_load = [l for l in layers if l in available_layers]
            
        if not layers_to_load:
            logger.warning(f"在輪次 {first_epoch} 中沒有找到可用的層數據")
            return {}
            
        # 準備返回結構
        result = {
            'metadata': {
                'epochs': epochs_to_load,
                'layers': layers_to_load,
                'batch_idx': batch_idx
            },
            'activations': {},
            'summaries': {},
            'training_summary': self.load_training_summary()
        }
        
        # 載入每個層的數據
        for layer_name in layers_to_load:
            result['activations'][layer_name] = {}
            for epoch in epochs_to_load:
                # 檢查這個輪次是否有這個層的數據
                if epoch in self.available_epochs and layer_name in self.available_layers.get((epoch, batch_idx), []):
                    activation = self.load_activation(layer_name, epoch, batch_idx)
                    if activation is not None:
                        result['activations'][layer_name][epoch] = activation
        
        # 載入每個輪次的摘要
        for epoch in epochs_to_load:
            summary = self.load_epoch_summary(epoch)
            if summary is not None:
                result['summaries'][epoch] = summary
                
        logger.info(f"已載入 {len(layers_to_load)} 個層在 {len(epochs_to_load)} 個輪次的數據")
        return result 

    def load_all_layer_activations(self, epoch: int, batch: int = 0) -> Dict[str, torch.Tensor]:
        """
        載入特定輪次和批次的所有層激活值。
        
        Args:
            epoch (int): 輪次編號。
            batch (int, optional): 批次索引。默認為0。
            
        Returns:
            Dict[str, torch.Tensor]: 層名稱到激活值的映射。
        """
        if not self.hooks_available:
            return {}
            
        epoch_dir = os.path.join(self.hooks_dir, f'epoch_{epoch}')
        if not os.path.exists(epoch_dir):
            logger.warning(f"輪次目錄不存在: {epoch_dir}")
            return {}
            
        # 獲取所有可用的層
        layers = self.list_available_layers(epoch, batch)
        if not layers:
            return {}
            
        # 載入每個層的激活值
        activations = {}
        for layer_name in layers:
            activation = self.load_layer_activation(layer_name, epoch, batch)
            if activation is not None:
                activations[layer_name] = activation
                
        return activations 

    def load_layer_activations_across_epochs(self, layer_name: str, epochs: List[int], batch: int = 0) -> Dict[int, torch.Tensor]:
        """
        載入特定層在多個輪次的激活值。
        
        Args:
            layer_name (str): 層名稱。
            epochs (List[int]): 輪次列表。
            batch (int, optional): 批次索引。默認為0。
            
        Returns:
            Dict[int, torch.Tensor]: 輪次到激活值的映射。
        """
        if not self.hooks_available:
            return {}
            
        activations = {}
        for epoch in epochs:
            activation = self.load_layer_activation(layer_name, epoch, batch)
            if activation is not None:
                activations[epoch] = activation
                
        return activations 

    def load_all_activations_across_epochs(self, epochs: List[int], batch: int = 0) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        載入所有層在多個輪次的激活值。
        
        Args:
            epochs (List[int]): 輪次列表。
            batch (int, optional): 批次索引。默認為0。
            
        Returns:
            Dict[str, Dict[int, torch.Tensor]]: 層名稱到輪次激活值映射的映射。
        """
        if not self.hooks_available:
            return {}
            
        # 獲取所有可用的層
        layers = self.list_available_layers(epochs[0], batch)
        if not layers:
            return {}
            
        # 載入每個層在所有輪次的激活值
        all_activations = {}
        for layer_name in layers:
            layer_activations = self.load_layer_activations_across_epochs(layer_name, epochs, batch)
            if layer_activations:
                all_activations[layer_name] = layer_activations
                
        return all_activations 