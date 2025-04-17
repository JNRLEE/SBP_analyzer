"""
# 基礎分析器類別
# 定義分析器的基本接口和共用功能
"""

from abc import ABC, abstractmethod
import logging
import os
from typing import Dict, Any, List, Optional, Union


class BaseAnalyzer(ABC):
    """
    分析器的基礎抽象類別，定義所有分析器的通用接口。
    
    Args:
        experiment_dir: 實驗結果目錄的路徑
        log_level: 日誌級別，預設為 INFO
        
    Returns:
        None
        
    Description:
        此類別提供分析器的基本結構和通用功能，包括設置日誌、加載數據和保存結果等。
        子類應實現具體的分析邏輯。
        
    References:
        None
    """
    
    def __init__(self, experiment_dir: str, log_level: int = logging.INFO):
        self.experiment_dir = experiment_dir
        self.logger = self._setup_logger(log_level)
        self.results = {}
        
        if not os.path.exists(experiment_dir):
            self.logger.warning(f"實驗目錄 {experiment_dir} 不存在")
        
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """
        設置分析器的日誌記錄器。
        
        Args:
            log_level: 日誌級別
            
        Returns:
            logging.Logger: 設置好的日誌記錄器
            
        Description:
            創建並配置一個日誌記錄器實例，用於記錄分析過程中的信息。
            
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
    def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        執行分析，這個方法需要被子類實現。
        
        Args:
            **kwargs: 分析所需的額外參數
            
        Returns:
            Dict[str, Any]: 分析結果字典
            
        Description:
            此抽象方法定義了分析的基本接口，子類應實現具體的分析邏輯。
            
        References:
            None
        """
        pass
    
    def save_results(self, output_dir: str, filename: str = "analysis_results.json") -> str:
        """
        將分析結果保存到文件。
        
        Args:
            output_dir: 輸出目錄
            filename: 輸出文件名，預設為 "analysis_results.json"
            
        Returns:
            str: 保存的文件路徑
            
        Description:
            將分析結果序列化並保存到指定的文件中。
            
        References:
            None
        """
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # 將 numpy 數組和 PyTorch 張量轉換為列表
        serializable_results = self._make_serializable(self.results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"分析結果已保存至 {output_path}")
        return output_path
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        將對象轉換為可序列化的形式。
        
        Args:
            obj: 要轉換的對象
            
        Returns:
            Any: 轉換後的可序列化對象
            
        Description:
            處理特殊類型（如 numpy 數組、PyTorch 張量）使其可以被 JSON 序列化。
            
        References:
            None
        """
        import numpy as np
        
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj 