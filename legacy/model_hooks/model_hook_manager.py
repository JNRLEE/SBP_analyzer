"""
模型鉤子管理器：綜合管理模型內部狀態的鉤子系統

這個模組提供了 ModelHookManager 類，用於統一管理模型中的各種鉤子，
包括激活值鉤子、梯度鉤子等，並提供全面的模型內部狀態分析功能。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union
import logging
import weakref

from .activation_hook import ActivationHook
from .gradient_hook import GradientHook

logger = logging.getLogger(__name__)

class ModelHookManager:
    """模型鉤子管理器，用於綜合管理和獲取模型內部狀態
    
    使用方法:
        model = YourModel()
        hook_manager = ModelHookManager(model)
        
        # 訓練
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # 獲取內部狀態
        activations = hook_manager.get_activations()
        gradients = hook_manager.get_gradients()
        weights = hook_manager.get_weights()
    """
    
    def __init__(self, model: nn.Module,
                 monitored_layers: Optional[List[str]] = None,
                 monitored_params: Optional[List[str]] = None):
        """初始化模型鉤子管理器
        
        Args:
            model: 要監控的 PyTorch 模型
            monitored_layers: 要監控激活值的層名稱列表
            monitored_params: 要監控梯度的參數名稱列表
        """
        self.model = model
        self.monitored_layers = monitored_layers
        self.monitored_params = monitored_params
        
        # 初始化鉤子
        self.activation_hook = ActivationHook(model, monitored_layers)
        self.gradient_hook = GradientHook(model, monitored_params)
        
        # 監控的批次數據
        self.current_inputs = None
        self.current_outputs = None
        self.current_targets = None
        self.current_loss = None
    
    def update_batch_data(self, 
                          inputs: Optional[torch.Tensor] = None,
                          outputs: Optional[torch.Tensor] = None,
                          targets: Optional[torch.Tensor] = None,
                          loss: Optional[torch.Tensor] = None):
        """更新當前批次數據
        
        Args:
            inputs: 輸入張量
            outputs: 模型輸出張量
            targets: 目標張量
            loss: 損失值
        """
        if inputs is not None:
            self.current_inputs = inputs.detach() if isinstance(inputs, torch.Tensor) else inputs
        if outputs is not None:
            self.current_outputs = outputs.detach() if isinstance(outputs, torch.Tensor) else outputs
        if targets is not None:
            self.current_targets = targets.detach() if isinstance(targets, torch.Tensor) else targets
        if loss is not None:
            self.current_loss = loss.detach() if isinstance(loss, torch.Tensor) else loss
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """獲取層激活值
        
        Returns:
            Dict[str, torch.Tensor]: 層名到激活值的映射
        """
        return self.activation_hook.get_activations()
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """獲取參數梯度
        
        Returns:
            Dict[str, torch.Tensor]: 參數名到梯度的映射
        """
        return self.gradient_hook.get_gradients()
    
    def get_gradient_statistics(self) -> Dict[str, Dict[str, float]]:
        """獲取梯度統計數據
        
        Returns:
            Dict[str, Dict[str, float]]: 參數名到統計數據的映射
        """
        return self.gradient_hook.get_gradient_statistics()
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """獲取模型權重
        
        Returns:
            Dict[str, torch.Tensor]: 參數名到權重的映射
        """
        weights = {}
        for name, param in self.model.named_parameters():
            weights[name] = param.data.detach().clone()
        return weights
    
    def get_weight_statistics(self) -> Dict[str, Dict[str, float]]:
        """獲取權重統計數據 (均值、標準差、最大值、最小值等)
        
        Returns:
            Dict[str, Dict[str, float]]: 參數名到統計數據的映射
        """
        weights = self.get_weights()
        stats = {}
        for name, weight in weights.items():
            try:
                flat_weight = weight.flatten()
                stats[name] = {
                    'mean': float(flat_weight.mean().item()),
                    'std': float(flat_weight.std().item()),
                    'min': float(flat_weight.min().item()),
                    'max': float(flat_weight.max().item()),
                    'norm': float(torch.norm(flat_weight).item()),
                    'sparsity': float(torch.sum(flat_weight == 0).item() / flat_weight.numel())
                }
            except Exception as e:
                logger.warning(f"計算權重統計量時出錯 ({name}): {str(e)}")
                stats[name] = {'error': str(e)}
        return stats
    
    def get_layer_output_statistics(self) -> Dict[str, Dict[str, float]]:
        """獲取層輸出的統計數據 (均值、標準差、激活比例等)
        
        Returns:
            Dict[str, Dict[str, float]]: 層名到統計數據的映射
        """
        activations = self.get_activations()
        stats = {}
        for name, activation in activations.items():
            try:
                # 某些激活可能是列表或元組，處理這種情況
                if isinstance(activation, (list, tuple)):
                    activation = activation[0]  # 取第一個元素作為代表
                
                # 展平以獲取整體統計量
                flat_act = activation.flatten()
                
                # 計算統計量
                stats[name] = {
                    'mean': float(flat_act.mean().item()),
                    'std': float(flat_act.std().item()),
                    'min': float(flat_act.min().item()),
                    'max': float(flat_act.max().item()),
                    'norm': float(torch.norm(flat_act).item()),
                    'activation_ratio': float(torch.sum(flat_act > 0).item() / flat_act.numel()),
                    'saturation_ratio_high': float(torch.sum(flat_act > 0.9 * flat_act.max()).item() / flat_act.numel()),
                    'saturation_ratio_low': float(torch.sum(flat_act < 0.1 * flat_act.max()).item() / flat_act.numel())
                }
            except Exception as e:
                logger.warning(f"計算層輸出統計量時出錯 ({name}): {str(e)}")
                stats[name] = {'error': str(e)}
        return stats
    
    def get_model_summary(self) -> Dict[str, Any]:
        """獲取模型概要信息
        
        Returns:
            Dict[str, Any]: 模型概要信息
        """
        summary = {
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'layer_types': {}
        }
        
        # 統計不同類型層的數量
        for module in self.model.modules():
            module_type = type(module).__name__
            if module_type not in summary['layer_types']:
                summary['layer_types'][module_type] = 0
            summary['layer_types'][module_type] += 1
        
        return summary
    
    def clear(self):
        """清除所有保存的數據"""
        self.activation_hook.clear_activations()
        self.gradient_hook.clear_gradients()
        self.current_inputs = None
        self.current_outputs = None
        self.current_targets = None
        self.current_loss = None
    
    def remove_hooks(self):
        """移除所有已註冊的鉤子"""
        self.activation_hook.remove_hooks()
        self.gradient_hook.remove_hooks()
    
    def __del__(self):
        """析構時移除鉤子"""
        self.remove_hooks() 