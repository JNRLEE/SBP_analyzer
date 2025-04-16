"""
模型分析回調：使用模型鉤子監控和分析模型內部狀態

此回調使用 ModelHookManager 監控模型訓練過程中的內部狀態，
包括層激活值、參數梯度和權重分布等，並提供豐富的分析功能。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
import logging
import os
import json
import numpy as np
from datetime import datetime

from ..interfaces.callback_interface import CallbackInterface
from ..model_hooks import ModelHookManager

logger = logging.getLogger(__name__)

class ModelAnalyticsCallback(CallbackInterface):
    """模型分析回調，用於監控和分析模型內部狀態
    
    此回調利用 ModelHookManager 在訓練過程中收集模型內部狀態，
    包括層激活值、參數梯度和權重分布等，並進行深度分析。
    """
    
    def __init__(self, 
                 output_dir: str = 'analysis_results',
                 monitored_layers: Optional[List[str]] = None,
                 monitored_params: Optional[List[str]] = None,
                 save_frequency: int = 1,
                 save_format: List[str] = ['json', 'tensorboard']):
        """初始化模型分析回調
        
        Args:
            output_dir: 分析結果保存目錄
            monitored_layers: 要監控的層名稱列表，如果為 None 則監控所有層
            monitored_params: 要監控的參數名稱列表，如果為 None 則監控所有參數
            save_frequency: 保存頻率(每多少個epoch保存一次)
            save_format: 保存格式，可選 'json', 'tensorboard', 'csv'
        """
        self.output_dir = output_dir
        self.monitored_layers = monitored_layers
        self.monitored_params = monitored_params
        self.save_frequency = save_frequency
        self.save_format = save_format
        
        self.hook_manager = None
        self.experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = None
        
        # 創建目錄
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        """確保輸出目錄存在"""
        self.results_dir = os.path.join(self.output_dir, f'model_analytics_{self.experiment_id}')
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"模型分析結果將保存到: {self.results_dir}")
    
    def on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """訓練開始時初始化模型鉤子管理器
        
        Args:
            model: 正在訓練的模型
            logs: 包含訓練相關信息的字典
        """
        # 初始化模型鉤子管理器
        self.hook_manager = ModelHookManager(
            model,
            monitored_layers=self.monitored_layers,
            monitored_params=self.monitored_params
        )
        
        # 保存模型結構
        self._save_model_structure(model)
        
        logger.info("模型分析回調已初始化，開始監控模型內部狀態")
    
    def _save_model_structure(self, model: nn.Module):
        """保存模型結構信息
        
        Args:
            model: 要分析的模型
        """
        # 獲取模型摘要
        model_summary = self.hook_manager.get_model_summary()
        
        # 構建層結構信息
        layers_info = []
        for name, module in model.named_modules():
            if name:  # 排除根模型自身
                params = sum(p.numel() for p in module.parameters())
                trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                layers_info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': params,
                    'trainable_parameters': trainable_params
                })
        
        # 組合完整的模型結構信息
        model_structure = {
            'summary': model_summary,
            'layers': layers_info
        }
        
        # 保存為 JSON 文件
        with open(os.path.join(self.results_dir, 'model_structure.json'), 'w') as f:
            json.dump(model_structure, f, indent=2)
    
    def on_batch_end(self, batch: int, model: nn.Module, inputs: torch.Tensor, 
                    targets: torch.Tensor, outputs: torch.Tensor, 
                    loss: torch.Tensor, logs: Dict[str, Any] = None) -> None:
        """每個批次結束時分析模型狀態
        
        Args:
            batch: 當前批次索引
            model: 正在訓練的模型
            inputs: 輸入張量
            targets: 目標張量
            outputs: 模型輸出
            loss: 損失值
            logs: 包含訓練相關信息的字典
        """
        # 更新批次數據
        self.hook_manager.update_batch_data(
            inputs=inputs,
            outputs=outputs,
            targets=targets,
            loss=loss
        )
        
        # 僅記錄或處理需要的數據，避免每個批次都保存大量數據
        
    def on_epoch_end(self, epoch: int, model: nn.Module, train_logs: Dict[str, Any], 
                   val_logs: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
        """每個 epoch 結束時分析和保存模型狀態
        
        Args:
            epoch: 當前 epoch 索引
            model: 正在訓練的模型
            train_logs: 訓練日誌
            val_logs: 驗證日誌
            logs: 附加日誌
        """
        # 檢查是否需要保存
        if epoch % self.save_frequency != 0:
            return
        
        # 收集模型分析數據
        analysis_data = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'gradient_statistics': self.hook_manager.get_gradient_statistics(),
            'weight_statistics': self.hook_manager.get_weight_statistics(),
            'layer_output_statistics': self.hook_manager.get_layer_output_statistics()
        }
        
        # 保存分析數據
        self._save_analysis_data(analysis_data, epoch)
        
        # 如果使用 TensorBoard，記錄到 TensorBoard
        if 'json' in self.save_format and logs and 'tensorboard_writer' in logs:
            writer = logs['tensorboard_writer']
            self._write_to_tensorboard(writer, analysis_data, epoch)
    
    def _save_analysis_data(self, analysis_data: Dict[str, Any], epoch: int):
        """保存分析數據
        
        Args:
            analysis_data: 分析數據字典
            epoch: 當前 epoch
        """
        # 保存為 JSON
        if 'json' in self.save_format:
            epoch_dir = os.path.join(self.results_dir, f'epoch_{epoch}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            with open(os.path.join(epoch_dir, 'analysis.json'), 'w') as f:
                json.dump(analysis_data, f, indent=2)
        
        # 保存為 CSV (只保存摘要)
        if 'csv' in self.save_format:
            # 這裡可以實現 CSV 保存邏輯
            pass
    
    def _write_to_tensorboard(self, writer, analysis_data: Dict[str, Any], epoch: int):
        """將分析數據寫入 TensorBoard
        
        Args:
            writer: TensorBoard SummaryWriter
            analysis_data: 分析數據字典
            epoch: 當前 epoch
        """
        # 寫入梯度統計
        for param_name, stats in analysis_data['gradient_statistics'].items():
            for stat_name, value in stats.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f'gradient/{param_name}/{stat_name}', value, epoch)
        
        # 寫入權重統計
        for param_name, stats in analysis_data['weight_statistics'].items():
            for stat_name, value in stats.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f'weights/{param_name}/{stat_name}', value, epoch)
        
        # 寫入層輸出統計
        for layer_name, stats in analysis_data['layer_output_statistics'].items():
            for stat_name, value in stats.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f'activations/{layer_name}/{stat_name}', value, epoch)
    
    def on_train_end(self, model: nn.Module, history: Dict[str, List], logs: Dict[str, Any] = None) -> None:
        """訓練結束時執行最終分析
        
        Args:
            model: 訓練完成的模型
            history: 訓練歷史
            logs: 附加日誌
        """
        # 保存最終訓練結果摘要
        final_summary = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_epochs': len(history.get('train_loss', [])),
            'final_train_loss': history.get('train_loss', [])[-1] if history.get('train_loss', []) else None,
            'final_val_loss': history.get('val_loss', [])[-1] if history.get('val_loss', []) else None,
            'model_summary': self.hook_manager.get_model_summary() if self.hook_manager else None
        }
        
        # 保存最終摘要
        with open(os.path.join(self.results_dir, 'training_summary.json'), 'w') as f:
            json.dump(final_summary, f, indent=2)
        
        # 清除鉤子
        if self.hook_manager:
            self.hook_manager.remove_hooks()
            self.hook_manager = None
        
        logger.info(f"模型分析完成，最終結果已保存到 {self.results_dir}") 