"""
模型鉤子模組：用於從訓練模型中獲取內部狀態

這個模組提供了鉤子（hooks）工具，用於在不修改原始模型架構的情況下，
獲取模型內部的激活值、梯度和參數等信息，用於分析和監控模型訓練過程。
"""

from .activation_hook import ActivationHook
from .gradient_hook import GradientHook
from .model_hook_manager import ModelHookManager

__all__ = ['ActivationHook', 'GradientHook', 'ModelHookManager'] 