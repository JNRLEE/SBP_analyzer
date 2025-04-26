"""
基礎數據載入器模組。

此模組提供用於載入實驗數據的基礎類別。

Classes:
    BaseLoader: 所有數據載入器的基礎類別。
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

class BaseLoader(ABC):
    """
    數據載入器的抽象基礎類別，提供通用接口和方法。
    
    這個類別定義了所有數據載入器共有的屬性和方法。子類別應重寫抽象方法來實現特定的數據載入功能。
    
    Attributes:
        experiment_dir (str): 實驗結果的目錄路徑。
        logger (logging.Logger): 日誌記錄器。
    """
    
    def __init__(self, experiment_dir: str):
        """
        初始化基礎載入器。
        
        Args:
            experiment_dir (str): 實驗結果的目錄路徑。
        """
        self.experiment_dir = experiment_dir
        
        # 設置日誌
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 確認實驗目錄是否存在
        if not os.path.exists(experiment_dir):
            self.logger.warning(f"實驗目錄不存在: {experiment_dir}")
    
    @abstractmethod
    def load(self, *args, **kwargs) -> Any:
        """
        載入數據並返回結果。
        
        這是一個抽象方法，必須由子類實現。
        
        Returns:
            Any: 載入的數據。
        """
        pass
    
    def _check_file_exists(self, file_path: str) -> bool:
        """
        檢查文件是否存在。
        
        Args:
            file_path (str): 文件路徑。
        
        Returns:
            bool: 文件是否存在。
        """
        exists = os.path.exists(file_path)
        if not exists:
            self.logger.warning(f"文件不存在: {file_path}")
        return exists
    
    def _load_json(self, file_path: str) -> Optional[Dict]:
        """
        載入JSON文件。
        
        Args:
            file_path (str): JSON文件路徑。
        
        Returns:
            Optional[Dict]: 載入的JSON數據，如果文件不存在或無法解析則返回None。
        """
        if not self._check_file_exists(file_path):
            return None
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"載入JSON文件失敗: {e}")
            return None 