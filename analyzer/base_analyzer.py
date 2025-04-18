"""
基礎分析器模組，提供分析器的基本結構和方法。

這個模組定義了SBP_analyzer中所有分析器的父類別，提供通用功能如數據載入和結果輸出。
所有特定類型的分析器都應繼承自BaseAnalyzer類別並實現其抽象方法。

Classes:
    BaseAnalyzer: 所有分析器的基礎類別，定義通用接口和實用方法。
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

class BaseAnalyzer(ABC):
    """
    分析器的抽象基礎類別，提供通用接口和方法。
    
    這個類別定義了所有分析器共有的屬性和方法。子類別應重寫抽象方法來實現特定的分析功能。
    
    Attributes:
        experiment_dir (str): 實驗結果的目錄路徑。
        config (Dict): 實驗配置。
        logger (logging.Logger): 日誌記錄器。
        results (Dict): 分析結果存儲字典。
    """
    
    def __init__(self, experiment_dir: str):
        """
        初始化基礎分析器。
        
        Args:
            experiment_dir (str): 實驗結果的目錄路徑。
        """
        self.experiment_dir = experiment_dir
        self.config = {}
        self.results = {}
        
        # 設置日誌
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 確認實驗目錄是否存在
        if not os.path.exists(experiment_dir):
            self.logger.error(f"實驗目錄不存在: {experiment_dir}")
            raise FileNotFoundError(f"實驗目錄不存在: {experiment_dir}")
    
    def load_config(self) -> Dict:
        """
        載入實驗配置。
        
        Returns:
            Dict: 實驗配置字典。
        """
        config_path = os.path.join(self.experiment_dir, 'config.json')
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                return self.config
        except Exception as e:
            self.logger.error(f"載入配置文件失敗: {e}")
            raise
    
    @abstractmethod
    def analyze(self, *args, **kwargs) -> Dict:
        """
        執行分析並返回結果。
        
        這是一個抽象方法，必須由子類實現。
        
        Returns:
            Dict: 分析結果字典。
        """
        pass
    
    def save_results(self, output_dir: str) -> None:
        """
        將分析結果保存到輸出目錄。
        
        Args:
            output_dir (str): 保存結果的目錄路徑。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_path = os.path.join(output_dir, f"{self.__class__.__name__}_results.json")
        try:
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=4)
            self.logger.info(f"分析結果已保存到: {output_path}")
        except Exception as e:
            self.logger.error(f"保存結果失敗: {e}")
            raise 