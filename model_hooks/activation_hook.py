"""
激活值鉤子模組：用於獲取模型中間層激活值

這個模組提供了 ActivationHook 類，用於在不修改原始模型架構的情況下，
獲取模型中間層的激活值，用於分析和監控模型內部狀態。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging
import weakref  # 使用弱引用避免內存洩漏

logger = logging.getLogger(__name__)

class ActivationHook:
    """獲取模型中間層激活值的鉤子
    
    使用方法:
        model = YourModel()
        activation_hook = ActivationHook(model)
        
        # 訓練或推理
        outputs = model(inputs)
        
        # 獲取激活值
        activations = activation_hook.get_activations()
    """
    
    def __init__(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        """初始化激活值鉤子
        
        Args:
            model: 要監控的 PyTorch 模型
            layer_names: 要監控的層名稱列表，如果為 None 則監控所有層
        """
        self.model = model
        self.layer_names = layer_names
        self.hooks = []
        self.activations = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """註冊前向傳播鉤子到模型的各層"""
        for name, module in self.model.named_modules():
            if self.layer_names is None or name in self.layer_names:
                # 為了避免在 lambda 中捕獲循環變量，需要創建函數工廠
                def get_hook(name):
                    def hook(module, input, output):
                        self.activations[name] = output.detach()
                    return hook
                
                # 註冊鉤子
                hook = module.register_forward_hook(get_hook(name))
                self.hooks.append(hook)
                logger.info(f"註冊激活值鉤子: {name}")
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """獲取當前保存的激活值
        
        Returns:
            Dict[str, torch.Tensor]: 層名到激活值的映射
        """
        return self.activations
    
    def clear_activations(self):
        """清除保存的激活值"""
        self.activations.clear()
    
    def remove_hooks(self):
        """移除所有已註冊的鉤子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("移除所有激活值鉤子")
    
    def __del__(self):
        """析構時移除鉤子"""
        self.remove_hooks() 