"""
# 分析指標計算模組
# 提供各種分析指標計算功能

此模組包含分佈指標、性能指標和層級活動指標的計算功能。
"""

from .distribution_metrics import calculate_mean, calculate_variance, calculate_skewness, calculate_kurtosis, calculate_percentile, calculate_sparsity, calculate_entropy
from .performance_metrics import calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score, calculate_convergence_rate, calculate_generalization_gap
from .layer_activity_metrics import calculate_activation_statistics, detect_dead_neurons, calculate_activation_similarity, calculate_effective_rank

__all__ = [
    'calculate_mean', 'calculate_variance', 'calculate_skewness', 'calculate_kurtosis', 'calculate_percentile', 'calculate_sparsity', 'calculate_entropy',
    'calculate_accuracy', 'calculate_precision', 'calculate_recall', 'calculate_f1_score', 'calculate_convergence_rate', 'calculate_generalization_gap',
    'calculate_activation_statistics', 'detect_dead_neurons', 'calculate_activation_similarity', 'calculate_effective_rank'
]
