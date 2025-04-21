"""
視覺化模組。

此模組提供用於視覺化模型結構、訓練過程和分布特性的功能。
"""

from .distribution_plots import (
    plot_tensor_histogram,
    plot_tensor_kde,
    plot_distribution_comparison,
    plot_layer_activations
)

from .performance_plots import (
    plot_loss_curve,
    plot_metric_curves,
    plot_learning_rate,
    plot_training_comparison,
    plot_training_radar,
    plot_learning_stages,
    plot_learning_efficiency_comparison,
    plot_lr_schedule_impact
)

__all__ = [
    # distribution_plots
    'plot_tensor_histogram',
    'plot_tensor_kde',
    'plot_distribution_comparison',
    'plot_layer_activations',
    
    # performance_plots
    'plot_loss_curve',
    'plot_metric_curves',
    'plot_learning_rate',
    'plot_training_comparison',
    'plot_training_radar',
    'plot_learning_stages',
    'plot_learning_efficiency_comparison',
    'plot_lr_schedule_impact'
]
