"""
Hook數據載入器模組。

此模組提供用於載入模型鉤子(hooks)數據的功能，包括中間層激活值等。

Classes:
    HookDataLoader: 載入模型鉤子數據的類別。
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Set, Tuple

import torch

from .base_loader import BaseLoader

class HookDataLoader(BaseLoader):
    """
    Hook數據載入器，用於載入模型中間層的激活值等數據。
    
    此載入器可以載入hooks/目錄下的.pt文件，包括各輪次各層的激活值。
    
    Attributes:
        experiment_dir (str): 實驗結果的目錄路徑。
        hooks_dir (str): 存放hooks數據的目錄。
        available_epochs (List[int]): 可用的輪次列表。
        available_layers (Dict[int, List[str]]): 每個輪次可用的層列表。
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
        self.available_layers = {}
        
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
                'available_layers': self.available_layers.get(epoch, [])
            }
        
        if layer_name is not None:
            # 載入特定層在所有可用輪次的數據
            layer_data = {}
            for e in self.available_epochs:
                if layer_name in self.available_layers.get(e, []):
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
        載入整體訓練摘要。
        
        Returns:
            Optional[Dict]: 訓練摘要字典，如果文件不存在則返回None。
        """
        summary_path = os.path.join(self.hooks_dir, 'training_summary.pt')
        try:
            if os.path.exists(summary_path):
                return torch.load(summary_path)
            else:
                self.logger.warning(f"訓練摘要文件不存在: {summary_path}")
                return None
        except Exception as e:
            self.logger.error(f"載入訓練摘要失敗: {e}")
            return None
    
    def load_evaluation_results(self, dataset: str = 'test') -> Optional[Dict]:
        """
        載入評估結果。
        
        Args:
            dataset (str, optional): 數據集名稱。默認為'test'。
        
        Returns:
            Optional[Dict]: 評估結果字典，如果文件不存在則返回None。
        """
        eval_path = os.path.join(self.hooks_dir, f'evaluation_results_{dataset}.pt')
        try:
            if os.path.exists(eval_path):
                return torch.load(eval_path)
            else:
                self.logger.warning(f"評估結果文件不存在: {eval_path}")
                return None
        except Exception as e:
            self.logger.error(f"載入評估結果失敗: {e}")
            return None
    
    def load_epoch_summary(self, epoch: int) -> Optional[Dict]:
        """
        載入特定輪次的摘要。
        
        Args:
            epoch (int): 輪次。
        
        Returns:
            Optional[Dict]: 輪次摘要字典，如果文件不存在則返回None。
        """
        if epoch not in self.available_epochs:
            self.logger.warning(f"輪次 {epoch} 的數據不可用")
            return None
        
        summary_path = os.path.join(self.hooks_dir, f'epoch_{epoch}', 'epoch_summary.pt')
        try:
            if os.path.exists(summary_path):
                return torch.load(summary_path)
            else:
                self.logger.warning(f"輪次摘要文件不存在: {summary_path}")
                return None
        except Exception as e:
            self.logger.error(f"載入輪次摘要失敗: {e}")
            return None
    
    def load_batch_data(self, epoch: int, batch: int) -> Optional[Dict]:
        """
        載入特定批次的數據。
        
        Args:
            epoch (int): 輪次。
            batch (int): 批次索引。
        
        Returns:
            Optional[Dict]: 批次數據字典，如果文件不存在則返回None。
        """
        if epoch not in self.available_epochs:
            self.logger.warning(f"輪次 {epoch} 的數據不可用")
            return None
        
        batch_path = os.path.join(self.hooks_dir, f'epoch_{epoch}', f'batch_{batch}_data.pt')
        try:
            if os.path.exists(batch_path):
                return torch.load(batch_path)
            else:
                self.logger.warning(f"批次數據文件不存在: {batch_path}")
                return None
        except Exception as e:
            self.logger.error(f"載入批次數據失敗: {e}")
            return None
    
    def load_activation(self, layer_name: str, epoch: int, batch: int = 0) -> Optional[torch.Tensor]:
        """
        載入特定層在特定輪次和批次的激活值。
        
        Args:
            layer_name (str): 層的名稱。
            epoch (int): 輪次。
            batch (int, optional): 批次索引。默認為0。
        
        Returns:
            Optional[torch.Tensor]: 激活值張量，如果文件不存在則返回None。
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
    
    def list_available_epochs(self) -> List[int]:
        """
        列出可用的輪次。
        
        Returns:
            List[int]: 可用輪次的列表。
        """
        return self.available_epochs
    
    def list_available_layers(self, epoch: Optional[int] = None) -> Union[Dict[int, List[str]], List[str]]:
        """
        列出可用的層。
        
        Args:
            epoch (int, optional): 特定的輪次。如果為None，則返回所有輪次的層。
        
        Returns:
            Union[Dict[int, List[str]], List[str]]: 若指定輪次，則返回該輪次可用層的列表；否則返回所有輪次的層字典。
        """
        if epoch is not None:
            return self.available_layers.get(epoch, [])
        else:
            return self.available_layers 