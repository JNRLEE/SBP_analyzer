"""
梯度鉤子模組：用於獲取模型參數梯度

這個模組提供了 GradientHook 類，用於在不修改原始模型架構的情況下，
獲取模型參數的梯度，用於分析和監控模型訓練過程中的梯度流動。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class GradientHook:
    """獲取模型參數梯度的鉤子
    
    使用方法:
        model = YourModel()
        gradient_hook = GradientHook(model)
        
        # 訓練
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # 獲取梯度
        gradients = gradient_hook.get_gradients()
    """
    
    def __init__(self, model: nn.Module, param_names: Optional[List[str]] = None):
        """初始化梯度鉤子
        
        Args:
            model: 要監控的 PyTorch 模型
            param_names: 要監控的參數名稱列表，如果為 None 則監控所有參數
        """
        self.model = model
        self.param_names = param_names
        self.hooks = []
        self.gradients = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """註冊參數的梯度鉤子"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and (self.param_names is None or name in self.param_names):
                # 為了避免在 lambda 中捕獲循環變量，需要創建函數工廠
                def get_hook(name):
                    def hook(grad):
                        self.gradients[name] = grad.detach().clone()
                        return grad
                    return hook
                
                # 註冊鉤子
                hook = param.register_hook(get_hook(name))
                self.hooks.append((name, hook))
                logger.info(f"註冊梯度鉤子: {name}")
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """獲取當前保存的梯度
        
        Returns:
            Dict[str, torch.Tensor]: 參數名到梯度的映射
        """
        return self.gradients
    
    def get_gradient_statistics(self) -> Dict[str, Dict[str, float]]:
        """獲取梯度的統計數據 (均值、標準差、最大值、最小值等)
        
        Returns:
            Dict[str, Dict[str, float]]: 參數名到統計數據的映射
        """
        stats = {}
        for name, grad in self.gradients.items():
            if grad is not None:
                try:
                    flat_grad = grad.flatten()
                    stats[name] = {
                        'mean': float(flat_grad.mean().item()),
                        'std': float(flat_grad.std().item()),
                        'min': float(flat_grad.min().item()),
                        'max': float(flat_grad.max().item()),
                        'norm': float(torch.norm(flat_grad).item()),
                        'is_zero': float(torch.all(flat_grad == 0).item())
                    }
                except Exception as e:
                    logger.warning(f"計算梯度統計量時出錯 ({name}): {str(e)}")
                    stats[name] = {'error': str(e)}
        return stats
    
    def clear_gradients(self):
        """清除保存的梯度"""
        self.gradients.clear()
    
    def remove_hooks(self):
        """移除所有已註冊的鉤子"""
        for _, hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("移除所有梯度鉤子")
    
    def __del__(self):
        """析構時移除鉤子"""
        self.remove_hooks() 