"""
基礎回調類，定義了所有回調類的共同行為和接口。
"""

from typing import Dict, Any, List, Optional, Callable
import torch
import torch.nn as nn

# 導入本地的回調接口副本
from ..interfaces.callback_interface import CallbackInterface

class BaseCallback(CallbackInterface):
    """
    基礎回調類，所有 SBP_analyzer 中的回調都應繼承此類。
    實現了基本的接口，並可以添加額外的通用功能，如條件觸發。
    """
    def __init__(self, trigger_condition: Optional[Callable[[Dict[str, Any]], bool]] = None):
        """初始化基礎回調
        
        Args:
            trigger_condition: 一個可選的函數，用於決定是否執行回調邏輯。
                               該函數接受 logs 字典作為輸入，返回 True 或 False。
        """
        self.trigger_condition = trigger_condition
        # 注意：不再需要 self.trainer，因為 trainer 實例和相關信息
        # 會通過回調方法的參數 (如 model, logs) 傳遞進來。

    def should_trigger(self, logs: Optional[Dict[str, Any]] = None) -> bool:
        """檢查是否應該觸發回調的業務邏輯
        
        Args:
            logs: 當前回調點的日誌信息字典。
            
        Returns:
            bool: 如果回調應被觸發，則返回 True。
        """
        if self.trigger_condition is None:
            return True
        # 確保 logs 不是 None
        logs = logs or {}
        try:
            return self.trigger_condition(logs)
        except Exception as e:
            print(f"Error evaluating trigger condition: {e}") # 使用 print 或 logger
            return False # 默認不觸發以避免錯誤

    # --- 重寫 CallbackInterface 中的方法 --- 
    # 每個方法首先檢查是否應觸發，然後調用一個內部方法來執行實際邏輯
    # 子類可以選擇重寫內部方法 (_on_...) 或外部方法 (on_...)。
    # 重寫外部方法可以完全自定義觸發邏輯。

    def on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        if self.should_trigger(logs):
            self._on_train_begin(model, logs)

    def _on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """子類應重寫此方法以實現訓練開始時的邏輯"""
        pass

    def on_epoch_begin(self, epoch: int, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        if self.should_trigger(logs):
            self._on_epoch_begin(epoch, model, logs)

    def _on_epoch_begin(self, epoch: int, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """子類應重寫此方法以實現 epoch 開始時的邏輯"""
        pass

    def on_batch_begin(self, batch: int, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, logs: Dict[str, Any] = None) -> None:
        if self.should_trigger(logs):
            self._on_batch_begin(batch, model, inputs, targets, logs)
            
    def _on_batch_begin(self, batch: int, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, logs: Dict[str, Any] = None) -> None:
        """子類應重寫此方法以實現批次開始時的邏輯"""
        pass

    def on_batch_end(self, batch: int, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, 
                    outputs: torch.Tensor, loss: torch.Tensor, logs: Dict[str, Any] = None) -> None:
        if self.should_trigger(logs):
            self._on_batch_end(batch, model, inputs, targets, outputs, loss, logs)
            
    def _on_batch_end(self, batch: int, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, 
                    outputs: torch.Tensor, loss: torch.Tensor, logs: Dict[str, Any] = None) -> None:
        """子類應重寫此方法以實現批次結束時的邏輯"""
        pass

    def on_epoch_end(self, epoch: int, model: nn.Module, train_logs: Dict[str, Any], 
                   val_logs: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
        if self.should_trigger(logs):
            self._on_epoch_end(epoch, model, train_logs, val_logs, logs)
            
    def _on_epoch_end(self, epoch: int, model: nn.Module, train_logs: Dict[str, Any], 
                   val_logs: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
        """子類應重寫此方法以實現 epoch 結束時的邏輯"""
        pass

    def on_train_end(self, model: nn.Module, history: Dict[str, List], logs: Dict[str, Any] = None) -> None:
        if self.should_trigger(logs):
            self._on_train_end(model, history, logs)
            
    def _on_train_end(self, model: nn.Module, history: Dict[str, List], logs: Dict[str, Any] = None) -> None:
        """子類應重寫此方法以實現訓練結束時的邏輯"""
        pass

    def on_evaluation_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        if self.should_trigger(logs):
            self._on_evaluation_begin(model, logs)
            
    def _on_evaluation_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """子類應重寫此方法以實現評估開始時的邏輯"""
        pass

    def on_evaluation_end(self, model: nn.Module, results: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
        if self.should_trigger(logs):
            self._on_evaluation_end(model, results, logs)
            
    def _on_evaluation_end(self, model: nn.Module, results: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
        """子類應重寫此方法以實現評估結束時的邏輯"""
        pass
