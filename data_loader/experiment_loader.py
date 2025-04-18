"""
實驗數據載入器模組。

此模組提供用於載入實驗基本數據的功能，包括配置、模型結構和訓練歷史記錄。

Classes:
    ExperimentLoader: 載入實驗基本數據的類別。
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

from .base_loader import BaseLoader

class ExperimentLoader(BaseLoader):
    """
    實驗數據載入器，用於載入實驗的基本信息和訓練結果。
    
    此載入器可以載入實驗目錄中的config.json、model_structure.json和training_history.json等文件。
    
    Attributes:
        experiment_dir (str): 實驗結果的目錄路徑。
        config (Dict): 實驗配置。
        model_structure (Dict): 模型結構信息。
        training_history (Dict): 訓練歷史記錄。
        results (Dict): 結果摘要。
    """
    
    def __init__(self, experiment_dir: str):
        """
        初始化實驗數據載入器。
        
        Args:
            experiment_dir (str): 實驗結果的目錄路徑。
        """
        super().__init__(experiment_dir)
        self.config = None
        self.model_structure = None
        self.training_history = None
        self.results = None
    
    def load(self, load_config: bool = True, load_model_structure: bool = True, 
             load_training_history: bool = True, load_results: bool = True) -> Dict:
        """
        載入實驗數據。
        
        Args:
            load_config (bool, optional): 是否載入配置。默認為True。
            load_model_structure (bool, optional): 是否載入模型結構。默認為True。
            load_training_history (bool, optional): 是否載入訓練歷史。默認為True。
            load_results (bool, optional): 是否載入結果摘要。默認為True。
        
        Returns:
            Dict: 包含載入數據的字典。
        """
        result = {}
        
        if load_config:
            self.config = self.load_config()
            result['config'] = self.config
        
        if load_model_structure:
            self.model_structure = self.load_model_structure()
            result['model_structure'] = self.model_structure
        
        if load_training_history:
            self.training_history = self.load_training_history()
            result['training_history'] = self.training_history
        
        if load_results:
            self.results = self.load_results()
            result['results'] = self.results
        
        return result
    
    def load_config(self) -> Optional[Dict]:
        """
        載入config.json中的實驗配置。
        
        Returns:
            Optional[Dict]: 包含配置的字典，如果文件不存在則返回None。
        """
        config_path = os.path.join(self.experiment_dir, 'config.json')
        return self._load_json(config_path)
    
    def load_model_structure(self) -> Optional[Dict]:
        """
        載入model_structure.json中的模型結構信息。
        
        Returns:
            Optional[Dict]: 包含模型結構的字典，如果文件不存在則返回None。
        """
        structure_path = os.path.join(self.experiment_dir, 'model_structure.json')
        return self._load_json(structure_path)
    
    def load_training_history(self) -> Optional[Dict]:
        """
        載入training_history.json中的訓練歷史記錄。
        
        Returns:
            Optional[Dict]: 包含訓練歷史的字典，如果文件不存在則返回None。
        """
        history_path = os.path.join(self.experiment_dir, 'training_history.json')
        return self._load_json(history_path)
    
    def load_results(self) -> Optional[Dict]:
        """
        載入results/results.json中的最終結果摘要。
        
        Returns:
            Optional[Dict]: 包含結果摘要的字典，如果文件不存在則返回None。
        """
        results_path = os.path.join(self.experiment_dir, 'results', 'results.json')
        return self._load_json(results_path)
    
    def get_experiment_info(self) -> Dict:
        """
        獲取實驗的基本信息。
        
        Returns:
            Dict: 包含實驗信息的字典。
        """
        if not self.config:
            self.config = self.load_config()
        
        info = {
            'experiment_name': os.path.basename(self.experiment_dir),
            'experiment_dir': self.experiment_dir
        }
        
        if self.config:
            info.update({
                'model_name': self.config.get('model', {}).get('name', 'Unknown'),
                'dataset': self.config.get('dataset', {}).get('name', 'Unknown'),
                'task_type': self.config.get('task', {}).get('type', 'Unknown'),
                'created_at': self.config.get('created_at', 'Unknown')
            })
        
        return info 