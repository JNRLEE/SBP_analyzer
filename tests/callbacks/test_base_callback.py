"""
測試基礎回調類的功能

此模塊測試 BaseCallback 類的基本功能，包括條件觸發機制和回調方法的調用。
"""

import unittest
import torch
import torch.nn as nn
from typing import Dict, Any

from SBP_analyzer.callbacks.base_callback import BaseCallback

class SimpleModel(nn.Module):
    """簡單測試模型"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(5, 2)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return self.output(x)

class TestBaseCallback(unittest.TestCase):
    """測試 BaseCallback 類的基本功能"""
    
    def setUp(self):
        """設置測試環境"""
        # 創建測試模型
        self.model = SimpleModel()
        
        # 創建一個子類以追蹤方法調用
        class TrackingCallback(BaseCallback):
            def __init__(self, trigger_condition=None):
                super().__init__(trigger_condition)
                self.call_history = {
                    'train_begin': 0,
                    'epoch_begin': 0,
                    'batch_begin': 0,
                    'batch_end': 0,
                    'epoch_end': 0,
                    'train_end': 0,
                    'evaluation_begin': 0,
                    'evaluation_end': 0
                }
            
            def _on_train_begin(self, model, logs=None):
                self.call_history['train_begin'] += 1
            
            def _on_epoch_begin(self, epoch, model, logs=None):
                self.call_history['epoch_begin'] += 1
            
            def _on_batch_begin(self, batch, model, inputs, targets, logs=None):
                self.call_history['batch_begin'] += 1
            
            def _on_batch_end(self, batch, model, inputs, targets, outputs, loss, logs=None):
                self.call_history['batch_end'] += 1
            
            def _on_epoch_end(self, epoch, model, train_logs, val_logs, logs=None):
                self.call_history['epoch_end'] += 1
            
            def _on_train_end(self, model, history, logs=None):
                self.call_history['train_end'] += 1
            
            def _on_evaluation_begin(self, model, logs=None):
                self.call_history['evaluation_begin'] += 1
            
            def _on_evaluation_end(self, model, results, logs=None):
                self.call_history['evaluation_end'] += 1
        
        self.callback = TrackingCallback()
        
        # 創建測試數據
        self.test_input = torch.randn(4, 10)
        self.test_target = torch.randint(0, 2, (4,))
        self.test_output = self.model(self.test_input)
        self.test_loss = torch.tensor(0.5)
    
    def test_initialization(self):
        """測試回調初始化"""
        self.assertIsNone(self.callback.trigger_condition)
        
        # 測試帶有觸發條件的初始化
        def always_trigger(logs):
            return True
        
        callback_with_trigger = BaseCallback(trigger_condition=always_trigger)
        self.assertEqual(callback_with_trigger.trigger_condition, always_trigger)
    
    def test_should_trigger(self):
        """測試觸發條件機制"""
        # 無條件的回調應該始終觸發
        self.assertTrue(self.callback.should_trigger())
        self.assertTrue(self.callback.should_trigger({"test": 1}))
        
        # 測試始終觸發的條件
        def always_trigger(logs):
            return True
        
        callback = BaseCallback(trigger_condition=always_trigger)
        self.assertTrue(callback.should_trigger())
        self.assertTrue(callback.should_trigger({"test": 1}))
        
        # 測試從不觸發的條件
        def never_trigger(logs):
            return False
        
        callback = BaseCallback(trigger_condition=never_trigger)
        self.assertFalse(callback.should_trigger())
        self.assertFalse(callback.should_trigger({"test": 1}))
        
        # 測試基於日誌內容的條件
        def trigger_on_loss(logs):
            return logs.get("loss", 0) > 0.5
        
        callback = BaseCallback(trigger_condition=trigger_on_loss)
        self.assertFalse(callback.should_trigger({"loss": 0.3}))
        self.assertTrue(callback.should_trigger({"loss": 0.7}))
    
    def test_on_train_begin(self):
        """測試訓練開始回調"""
        self.callback.on_train_begin(self.model)
        self.assertEqual(self.callback.call_history['train_begin'], 1)
        
        # 測試帶有觸發條件的回調
        def never_trigger(logs):
            return False
        
        callback = self.callback.__class__(trigger_condition=never_trigger)
        callback.on_train_begin(self.model)
        self.assertEqual(callback.call_history['train_begin'], 0)
    
    def test_on_epoch_begin(self):
        """測試 epoch 開始回調"""
        self.callback.on_epoch_begin(1, self.model)
        self.assertEqual(self.callback.call_history['epoch_begin'], 1)
    
    def test_on_batch_callbacks(self):
        """測試批次相關回調"""
        # 測試批次開始回調
        self.callback.on_batch_begin(0, self.model, self.test_input, self.test_target)
        self.assertEqual(self.callback.call_history['batch_begin'], 1)
        
        # 測試批次結束回調
        self.callback.on_batch_end(0, self.model, self.test_input, self.test_target, 
                                  self.test_output, self.test_loss)
        self.assertEqual(self.callback.call_history['batch_end'], 1)
    
    def test_on_epoch_end(self):
        """測試 epoch 結束回調"""
        train_logs = {"loss": 0.5, "accuracy": 0.8}
        val_logs = {"loss": 0.6, "accuracy": 0.75}
        
        self.callback.on_epoch_end(1, self.model, train_logs, val_logs)
        self.assertEqual(self.callback.call_history['epoch_end'], 1)
    
    def test_on_train_end(self):
        """測試訓練結束回調"""
        history = {
            "loss": [0.5, 0.4, 0.3],
            "accuracy": [0.7, 0.8, 0.85]
        }
        
        self.callback.on_train_end(self.model, history)
        self.assertEqual(self.callback.call_history['train_end'], 1)
    
    def test_on_evaluation_callbacks(self):
        """測試評估相關回調"""
        # 測試評估開始回調
        self.callback.on_evaluation_begin(self.model)
        self.assertEqual(self.callback.call_history['evaluation_begin'], 1)
        
        # 測試評估結束回調
        results = {"loss": 0.4, "accuracy": 0.85}
        self.callback.on_evaluation_end(self.model, results)
        self.assertEqual(self.callback.call_history['evaluation_end'], 1)
    
    def test_error_handling_in_trigger(self):
        """測試觸發條件中的錯誤處理"""
        def faulty_trigger(logs):
            # 故意引發錯誤
            return logs["non_existent_key"] > 0
        
        callback = BaseCallback(trigger_condition=faulty_trigger)
        # 應該不觸發，而不是拋出錯誤
        self.assertFalse(callback.should_trigger({}))

if __name__ == '__main__':
    unittest.main() 