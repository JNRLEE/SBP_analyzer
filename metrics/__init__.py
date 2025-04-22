"""
指標計算模組。

此模組提供用於計算各種模型分析指標的功能，包括分布統計、性能指標等。
"""

# 從子模組導入功能
from .performance_metrics import (
    analyze_loss_curve,
    analyze_metric_curve,
    analyze_learning_efficiency,
    analyze_lr_schedule_impact,
    detect_training_anomalies,
    detect_overfitting,
    detect_oscillations,
    detect_plateaus
)

# 導出公開API
__all__ = [
    'analyze_loss_curve',
    'analyze_metric_curve',
    'analyze_learning_efficiency',
    'analyze_lr_schedule_impact',
    'detect_training_anomalies',
    'detect_overfitting',
    'detect_oscillations',
    'detect_plateaus'
]
