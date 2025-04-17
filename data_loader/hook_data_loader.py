"""
# 鉤子數據載入器
# 用於載入由模型鉤子儲存的數據
"""

import logging
import os
from typing import Dict, Any, List, Optional

import torch

from .base_loader import BaseLoader


class HookDataLoader(BaseLoader):
    """
    模型鉤子數據載入器，用於載入和解析由模型鉤子儲存的數據。
    
    Args:
        experiment_dir: 實驗結果目錄的路徑
        log_level: 日誌級別，預設為 INFO
        
    Returns:
        None
        
    Description:
        此類別負責載入模型鉤子產生的數據，包括訓練摘要、輪次摘要和層激活值等。
        鉤子數據通常以 .pt 格式儲存在 hooks/ 子目錄中。
        
    References:
        None
    """
    
    def __init__(self, experiment_dir: str, log_level: int = logging.INFO):
        super().__init__(experiment_dir, log_level)
        self.hooks_dir = os.path.join(experiment_dir, 'hooks')
        if not os.path.exists(self.hooks_dir):
            self.logger.warning(f"模型鉤子目錄 {self.hooks_dir} 不存在")
    
    def load(self, data_type: str, **kwargs) -> Any:
        """
        根據指定的數據類型載入相應的鉤子數據。
        
        Args:
            data_type: 要載入的數據類型，可選值為 'training_summary', 'evaluation_results', 'epoch_summary', 'batch_data', 'layer_activation'
            **kwargs: 數據載入需要的額外參數，如 epoch, batch, layer_name 等
            
        Returns:
            Any: 載入的數據
            
        Description:
            根據指定的數據類型調用相應的載入方法，並返回載入的數據。
            
        References:
            None
        """
        if data_type == 'training_summary':
            return self.load_training_summary()
        elif data_type == 'evaluation_results':
            dataset = kwargs.get('dataset', 'test')
            return self.load_evaluation_results(dataset)
        elif data_type == 'epoch_summary':
            epoch = kwargs.get('epoch')
            if epoch is None:
                self.logger.error("載入輪次摘要時必須指定 'epoch'")
                return None
            return self.load_epoch_summary(epoch)
        elif data_type == 'batch_data':
            epoch = kwargs.get('epoch')
            batch = kwargs.get('batch')
            if epoch is None or batch is None:
                self.logger.error("載入批次數據時必須指定 'epoch' 和 'batch'")
                return None
            return self.load_batch_data(epoch, batch)
        elif data_type == 'layer_activation':
            layer_name = kwargs.get('layer_name')
            epoch = kwargs.get('epoch')
            batch = kwargs.get('batch')
            if layer_name is None or epoch is None or batch is None:
                self.logger.error("載入層激活值時必須指定 'layer_name', 'epoch' 和 'batch'")
                return None
            return self.load_layer_activation(layer_name, epoch, batch)
        else:
            self.logger.error(f"未知的數據類型：{data_type}")
            return None
    
    def load_training_summary(self) -> Optional[Dict[str, Any]]:
        """
        載入整體訓練摘要。
        
        Args:
            None
            
        Returns:
            Optional[Dict[str, Any]]: 訓練摘要數據，如果文件不存在則返回 None
            
        Description:
            從 hooks/training_summary.pt 文件載入整體訓練摘要數據。
            
        References:
            None
        """
        summary_path = os.path.join(self.hooks_dir, 'training_summary.pt')
        if self.check_file_exists(summary_path):
            try:
                summary = torch.load(summary_path)
                self.logger.info(f"成功載入訓練摘要 {summary_path}")
                return summary
            except Exception as e:
                self.logger.error(f"載入訓練摘要時出錯：{str(e)}")
        return None
    
    def load_evaluation_results(self, dataset: str = 'test') -> Optional[Dict[str, Any]]:
        """
        載入評估結果。
        
        Args:
            dataset: 數據集名稱，預設為 'test'
            
        Returns:
            Optional[Dict[str, Any]]: 評估結果數據，如果文件不存在則返回 None
            
        Description:
            從 hooks/evaluation_results_{dataset}.pt 文件載入評估結果數據。
            
        References:
            None
        """
        eval_path = os.path.join(self.hooks_dir, f'evaluation_results_{dataset}.pt')
        if self.check_file_exists(eval_path):
            try:
                results = torch.load(eval_path)
                self.logger.info(f"成功載入評估結果 {eval_path}")
                return results
            except Exception as e:
                self.logger.error(f"載入評估結果時出錯：{str(e)}")
        return None
    
    def load_epoch_summary(self, epoch: int) -> Optional[Dict[str, Any]]:
        """
        載入特定輪次的摘要。
        
        Args:
            epoch: 輪次編號
            
        Returns:
            Optional[Dict[str, Any]]: 輪次摘要數據，如果文件不存在則返回 None
            
        Description:
            從 hooks/epoch_{epoch}/epoch_summary.pt 文件載入特定輪次的摘要數據。
            
        References:
            None
        """
        epoch_dir = os.path.join(self.hooks_dir, f'epoch_{epoch}')
        if not os.path.exists(epoch_dir):
            self.logger.warning(f"輪次目錄 {epoch_dir} 不存在")
            return None
        
        summary_path = os.path.join(epoch_dir, 'epoch_summary.pt')
        if self.check_file_exists(summary_path):
            try:
                summary = torch.load(summary_path)
                self.logger.info(f"成功載入輪次摘要 {summary_path}")
                return summary
            except Exception as e:
                self.logger.error(f"載入輪次摘要時出錯：{str(e)}")
        return None
    
    def load_batch_data(self, epoch: int, batch: int) -> Optional[Dict[str, Any]]:
        """
        載入特定批次的數據。
        
        Args:
            epoch: 輪次編號
            batch: 批次編號
            
        Returns:
            Optional[Dict[str, Any]]: 批次數據，如果文件不存在則返回 None
            
        Description:
            從 hooks/epoch_{epoch}/batch_{batch}_data.pt 文件載入特定批次的數據。
            
        References:
            None
        """
        batch_path = os.path.join(self.hooks_dir, f'epoch_{epoch}', f'batch_{batch}_data.pt')
        if self.check_file_exists(batch_path):
            try:
                batch_data = torch.load(batch_path)
                self.logger.info(f"成功載入批次數據 {batch_path}")
                return batch_data
            except Exception as e:
                self.logger.error(f"載入批次數據時出錯：{str(e)}")
        return None
    
    def load_layer_activation(self, layer_name: str, epoch: int, batch: int) -> Optional[torch.Tensor]:
        """
        載入特定層在特定批次的激活值。
        
        Args:
            layer_name: 層名稱
            epoch: 輪次編號
            batch: 批次編號
            
        Returns:
            Optional[torch.Tensor]: 層激活值數據，如果文件不存在則返回 None
            
        Description:
            從 hooks/epoch_{epoch}/{layer_name}_activation_batch_{batch}.pt 文件載入特定層的激活值數據。
            
        References:
            None
        """
        pattern = f"{layer_name}_activation_batch_{batch}.pt"
        path = os.path.join(self.hooks_dir, f'epoch_{epoch}', pattern)
        if self.check_file_exists(path):
            try:
                activation = torch.load(path)
                self.logger.info(f"成功載入層激活值 {path}")
                return activation
            except Exception as e:
                self.logger.error(f"載入層激活值時出錯：{str(e)}")
        return None
    
    def list_available_epochs(self) -> List[int]:
        """
        列出可用的輪次。
        
        Args:
            None
            
        Returns:
            List[int]: 可用的輪次編號列表
            
        Description:
            掃描 hooks/ 目錄，並返回所有以 'epoch_' 開頭的子目錄對應的輪次編號。
            
        References:
            None
        """
        if not os.path.exists(self.hooks_dir):
            return []
        
        epochs = []
        for item in os.listdir(self.hooks_dir):
            if item.startswith('epoch_') and os.path.isdir(os.path.join(self.hooks_dir, item)):
                try:
                    epoch_num = int(item.split('_')[1])
                    epochs.append(epoch_num)
                except ValueError:
                    pass
        
        epochs.sort()
        return epochs
    
    def list_available_layer_activations(self, epoch: int) -> List[str]:
        """
        列出特定輪次中可用的層激活值文件。
        
        Args:
            epoch: 輪次編號
            
        Returns:
            List[str]: 可用的層名稱列表
            
        Description:
            掃描 hooks/epoch_{epoch}/ 目錄，並返回所有以 '_activation_batch_' 結尾的文件對應的層名稱。
            
        References:
            None
        """
        epoch_dir = os.path.join(self.hooks_dir, f'epoch_{epoch}')
        if not os.path.exists(epoch_dir):
            self.logger.warning(f"輪次目錄 {epoch_dir} 不存在")
            return []
        
        layer_names = []
        for item in os.listdir(epoch_dir):
            if '_activation_batch_' in item and item.endswith('.pt'):
                try:
                    layer_name = item.split('_activation_batch_')[0]
                    layer_names.append(layer_name)
                except Exception:
                    pass
        
        return list(set(layer_names))  # 去重 