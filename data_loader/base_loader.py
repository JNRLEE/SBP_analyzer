"""
# 基礎載入器類別
# 定義載入器的基本接口和共用功能
"""

from abc import ABC, abstractmethod
import logging
import os
from typing import Any, Dict, Optional


class BaseLoader(ABC):
    """
    數據載入器的基礎抽象類別，定義所有載入器的通用接口。
    
    Args:
        experiment_dir: 實驗結果目錄的路徑
        log_level: 日誌級別，預設為 INFO
        
    Returns:
        None
        
    Description:
        此類別提供載入器的基本結構和通用功能，包括設置日誌和檢查目錄等。
        子類應實現具體的數據載入邏輯。
        
    References:
        None
    """
    
    def __init__(self, experiment_dir: str, log_level: int = logging.INFO):
        self.experiment_dir = experiment_dir
        self.logger = self._setup_logger(log_level)
        
        if not os.path.exists(experiment_dir):
            self.logger.warning(f"實驗目錄 {experiment_dir} 不存在")
        
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """
        設置載入器的日誌記錄器。
        
        Args:
            log_level: 日誌級別
            
        Returns:
            logging.Logger: 設置好的日誌記錄器
            
        Description:
            創建並配置一個日誌記錄器實例，用於記錄數據載入過程中的信息。
            
        References:
            None
        """
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(log_level)
        
        # 避免重複處理器
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    @abstractmethod
    def load(self, *args, **kwargs) -> Any:
        """
        載入數據，這個方法需要被子類實現。
        
        Args:
            *args: 位置參數
            **kwargs: 關鍵字參數
            
        Returns:
            Any: 載入的數據
            
        Description:
            此抽象方法定義了數據載入的基本接口，子類應實現具體的載入邏輯。
            
        References:
            None
        """
        pass
    
    def check_file_exists(self, file_path: str) -> bool:
        """
        檢查文件是否存在，並記錄日誌。
        
        Args:
            file_path: 要檢查的文件路徑
            
        Returns:
            bool: 文件是否存在
            
        Description:
            檢查指定的文件是否存在，並適當地記錄日誌。
            
        References:
            None
        """
        exists = os.path.exists(file_path)
        if not exists:
            self.logger.warning(f"文件 {file_path} 不存在")
        return exists 