"""
SBP Analyzer 本地回調接口副本

這個模塊定義了標準的回調接口，複製自 MicDysphagiaFramework，
以確保 SBP_analyzer 的回調能夠與 MicDysphagiaFramework 的訓練器兼容，
同時保持 SBP_analyzer 包的獨立性。
"""

from typing import Dict, Any, List, Callable, Optional, Union
import torch
import torch.nn as nn

class CallbackInterface:
    """定義回調接口，用於集成 SBP_analyzer 的回調功能"""
    
    def on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """訓練開始時調用
        
        Args:
            model: 正在訓練的模型
            logs: 包含各種訓練相關資訊的字典
        """
        pass
        
    def on_epoch_begin(self, epoch: int, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """每個 epoch 開始時調用
        
        Args:
            epoch: 當前epoch索引
            model: 正在訓練的模型
            logs: 包含各種訓練相關資訊的字典
        """
        pass
    
    def on_batch_begin(self, batch: int, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, logs: Dict[str, Any] = None) -> None:
        """每個批次開始時調用
        
        Args:
            batch: 當前批次索引
            model: 正在訓練的模型
            inputs: 輸入數據批次
            targets: 目標數據批次
            logs: 包含各種訓練相關資訊的字典
        """
        pass
    
    def on_batch_end(self, batch: int, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, 
                    outputs: torch.Tensor, loss: torch.Tensor, logs: Dict[str, Any] = None) -> None:
        """每個批次結束時調用，可獲取損失和梯度等信息
        
        Args:
            batch: 當前批次索引
            model: 正在訓練的模型
            inputs: 輸入數據批次
            targets: 目標數據批次
            outputs: 模型輸出結果
            loss: 計算的損失值
            logs: 包含各種訓練相關資訊的字典
        """
        pass
    
    def on_epoch_end(self, epoch: int, model: nn.Module, train_logs: Dict[str, Any], 
                   val_logs: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
        """每個 epoch 結束時調用，可獲取評估指標等信息
        
        Args:
            epoch: 當前epoch索引
            model: 正在訓練的模型
            train_logs: 訓練集上的指標和結果
            val_logs: 驗證集上的指標和結果
            logs: 包含各種訓練相關資訊的字典
        """
        pass
    
    def on_train_end(self, model: nn.Module, history: Dict[str, List], logs: Dict[str, Any] = None) -> None:
        """訓練結束時調用
        
        Args:
            model: 訓練完成的模型
            history: 訓練歷史記錄，包含損失和指標
            logs: 包含各種訓練相關資訊的字典
        """
        pass

    def on_evaluation_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """評估開始時調用
        
        Args:
            model: 要評估的模型
            logs: 包含各種評估相關資訊的字典
        """
        pass

    def on_evaluation_end(self, model: nn.Module, results: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
        """評估結束時調用
        
        Args:
            model: 評估的模型
            results: 評估結果
            logs: 包含各種評估相關資訊的字典
        """
        pass 