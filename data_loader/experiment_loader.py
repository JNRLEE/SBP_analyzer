"""
# 實驗載入器
# 用於載入實驗的基本信息和訓練結果
"""

import json
import logging
import os
from typing import Dict, Any, Optional, List

from .base_loader import BaseLoader


class ExperimentLoader(BaseLoader):
    """
    實驗數據載入器，用於載入和解析實驗的基本信息和訓練結果。
    
    Args:
        experiment_dir: 實驗結果目錄的路徑
        log_level: 日誌級別，預設為 INFO
        
    Returns:
        None
        
    Description:
        此類別負責載入實驗配置、模型結構、訓練歷史和最終結果等數據。
        
    References:
        None
    """
    
    def __init__(self, experiment_dir: str, log_level: int = logging.INFO):
        super().__init__(experiment_dir, log_level)
    
    def load(self, data_type: str, **kwargs) -> Any:
        """
        根據指定的數據類型載入相應的數據。
        
        Args:
            data_type: 要載入的數據類型，可選值為 'config', 'model_structure', 'training_history', 'results'
            **kwargs: 額外的參數
            
        Returns:
            Any: 載入的數據
            
        Description:
            根據指定的數據類型調用相應的載入方法，並返回載入的數據。
            
        References:
            None
        """
        if data_type == 'config':
            return self.load_config()
        elif data_type == 'model_structure':
            return self.load_model_structure()
        elif data_type == 'training_history':
            return self.load_training_history()
        elif data_type == 'results':
            return self.load_results()
        else:
            self.logger.error(f"未知的數據類型：{data_type}")
            return None
    
    def load_config(self) -> Optional[Dict[str, Any]]:
        """
        載入 config.json 中的實驗配置。
        
        Args:
            None
            
        Returns:
            Optional[Dict[str, Any]]: 實驗配置數據，如果文件不存在則返回 None
            
        Description:
            從實驗目錄中的 config.json 文件載入實驗配置數據。
            
        References:
            None
        """
        config_path = os.path.join(self.experiment_dir, 'config.json')
        if self.check_file_exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.logger.info(f"成功載入配置文件 {config_path}")
                return config
            except Exception as e:
                self.logger.error(f"載入配置文件時出錯：{str(e)}")
        return None
    
    def load_model_structure(self) -> Optional[Dict[str, Any]]:
        """
        載入 model_structure.json 中的模型結構信息。
        
        Args:
            None
            
        Returns:
            Optional[Dict[str, Any]]: 模型結構數據，如果文件不存在則返回 None
            
        Description:
            從實驗目錄中的 model_structure.json 文件載入模型結構信息。
            
        References:
            None
        """
        structure_path = os.path.join(self.experiment_dir, 'model_structure.json')
        if self.check_file_exists(structure_path):
            try:
                with open(structure_path, 'r', encoding='utf-8') as f:
                    structure = json.load(f)
                self.logger.info(f"成功載入模型結構文件 {structure_path}")
                return structure
            except Exception as e:
                self.logger.error(f"載入模型結構文件時出錯：{str(e)}")
        return None
    
    def load_training_history(self) -> Optional[Dict[str, Any]]:
        """
        載入 training_history.json 中的訓練歷史記錄。
        
        Args:
            None
            
        Returns:
            Optional[Dict[str, Any]]: 訓練歷史數據，如果文件不存在則返回 None
            
        Description:
            從實驗目錄中的 training_history.json 文件載入訓練歷史記錄。
            
        References:
            None
        """
        history_path = os.path.join(self.experiment_dir, 'training_history.json')
        if self.check_file_exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                self.logger.info(f"成功載入訓練歷史文件 {history_path}")
                return history
            except Exception as e:
                self.logger.error(f"載入訓練歷史文件時出錯：{str(e)}")
        return None
    
    def load_results(self) -> Optional[Dict[str, Any]]:
        """
        載入 results/results.json 中的最終結果摘要。
        
        Args:
            None
            
        Returns:
            Optional[Dict[str, Any]]: 最終結果數據，如果文件不存在則返回 None
            
        Description:
            從實驗目錄中的 results/results.json 文件載入最終結果摘要。
            
        References:
            None
        """
        results_path = os.path.join(self.experiment_dir, 'results', 'results.json')
        if self.check_file_exists(results_path):
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                self.logger.info(f"成功載入結果文件 {results_path}")
                return results
            except Exception as e:
                self.logger.error(f"載入結果文件時出錯：{str(e)}")
        return None
    
    def list_available_models(self) -> List[str]:
        """
        列出 models/ 目錄中可用的模型檢查點文件。
        
        Args:
            None
            
        Returns:
            List[str]: 可用的模型檢查點文件列表
            
        Description:
            掃描實驗目錄中的 models/ 子目錄，並返回所有 .pth 文件的列表。
            
        References:
            None
        """
        models_dir = os.path.join(self.experiment_dir, 'models')
        if not os.path.exists(models_dir):
            self.logger.warning(f"模型目錄 {models_dir} 不存在")
            return []
        
        return [f for f in os.listdir(models_dir) if f.endswith('.pth')] 