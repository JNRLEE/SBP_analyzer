#!/usr/bin/env python
"""
# 分佈分析範例：展示如何使用高級分佈檢測功能進行張量分析
"""
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 確保可以導入SBP_analyzer模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stat_utils import (
    detect_distribution_type,
    detect_distribution_type_advanced,
    calculate_distribution_stats,
    compare_distributions
)

def plot_distribution_and_analysis(data, title, analysis_result, axis=None):
    """
    繪製分佈及其分析結果
    
    Args:
        data: 數據張量
        title: 圖表標題
        analysis_result: 分佈分析結果
        axis: matplotlib軸對象
    """
    if axis is None:
        fig, axis = plt.subplots(figsize=(10, 6))
    
    # 繪製直方圖
    axis.hist(data.flatten(), bins=50, alpha=0.7, density=True)
    
    # 添加分佈類型和置信度信息
    info_text = f"Primary type: {analysis_result['primary_type']}\n"
    if analysis_result['secondary_type']:
        info_text += f"Secondary type: {analysis_result['secondary_type']}\n"
    info_text += f"Confidence: {analysis_result['confidence']:.3f}\n\n"
    
    # 添加關鍵指標
    for key, value in analysis_result['metrics'].items():
        info_text += f"{key}: {value:.3f}\n"
    
    axis.text(0.95, 0.95, info_text, transform=axis.transAxes, 
              verticalalignment='top', horizontalalignment='right',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    axis.set_title(title)
    axis.set_xlabel('Value')
    axis.set_ylabel('Density')
    axis.grid(alpha=0.3)

def generate_example_distributions():
    """
    生成各種類型的數據分佈以進行測試
    
    Returns:
        dict: 包含不同分佈類型的張量字典
    """
    # 設置隨機種子以確保結果可重現
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_samples = 10000
    
    distributions = {
        # 正態分佈
        'normal': torch.tensor(np.random.normal(0, 1, n_samples)),
        
        # 偏態分佈（右偏）
        'right_skewed': torch.tensor(np.random.lognormal(0, 0.7, n_samples)),
        
        # 偏態分佈（左偏）
        'left_skewed': torch.tensor(-np.random.lognormal(0, 0.7, n_samples)),
        
        # 均勻分佈
        'uniform': torch.tensor(np.random.uniform(-1, 1, n_samples)),
        
        # 稀疏分佈
        'sparse': torch.tensor(np.random.choice([0, 1], n_samples, p=[0.9, 0.1])),
        
        # 重度稀疏分佈
        'very_sparse': torch.tensor(np.random.choice([0, 1], n_samples, p=[0.95, 0.05])),
        
        # 飽和分佈
        'saturated': torch.tensor(np.random.choice([0, 1], n_samples, p=[0.05, 0.95])),
        
        # 雙峰分佈
        'bimodal': torch.tensor(np.concatenate([
            np.random.normal(-2, 0.5, n_samples // 2),
            np.random.normal(2, 0.5, n_samples // 2)
        ])),
        
        # 多模態分佈
        'multimodal': torch.tensor(np.concatenate([
            np.random.normal(-3, 0.5, n_samples // 3),
            np.random.normal(0, 0.5, n_samples // 3),
            np.random.normal(3, 0.5, n_samples // 3)
        ])),
        
        # 重尾分佈
        'heavy_tailed': torch.tensor(np.random.standard_t(2, n_samples)),
        
        # 高斯混合模型
        'gaussian_mixture': torch.tensor(np.concatenate([
            np.random.normal(-2, 0.5, int(n_samples * 0.3)),
            np.random.normal(0, 1, int(n_samples * 0.4)),
            np.random.normal(3, 0.7, int(n_samples * 0.3))
        ]))
    }
    
    return distributions

def compare_detection_methods(distributions):
    """
    比較基本和高級分佈檢測方法的結果
    
    Args:
        distributions: 包含不同分佈類型的張量字典
    """
    fig, axes = plt.subplots(len(distributions), 2, figsize=(20, 5*len(distributions)))
    
    for i, (dist_name, data) in enumerate(distributions.items()):
        # 基本檢測方法
        basic_result = detect_distribution_type(data)
        # 高級檢測方法
        advanced_result = detect_distribution_type_advanced(data)
        
        # 繪製基本檢測結果
        plot_distribution_and_analysis(
            data, 
            f"{dist_name} - Basic Detection", 
            {'primary_type': basic_result, 'secondary_type': None, 
             'confidence': 1.0, 'metrics': calculate_distribution_stats(data)},
            axes[i, 0]
        )
        
        # 繪製高級檢測結果
        plot_distribution_and_analysis(
            data, 
            f"{dist_name} - Advanced Detection", 
            advanced_result,
            axes[i, 1]
        )
    
    plt.tight_layout()
    
    # 確保存在輸出目錄
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # 保存圖表
    plt.savefig(output_dir / "distribution_detection_comparison.png", dpi=300)
    print(f"Comparison plot saved to {output_dir / 'distribution_detection_comparison.png'}")

def main():
    """主程式：執行分佈分析範例"""
    print("Generating example distributions...")
    distributions = generate_example_distributions()
    
    print("Comparing detection methods...")
    compare_detection_methods(distributions)
    
    # 演示分佈比較功能
    normal_dist = distributions['normal']
    bimodal_dist = distributions['bimodal']
    
    print("\nComparing normal and bimodal distributions:")
    comparison_results = compare_distributions(normal_dist, bimodal_dist)
    
    for metric_name, value in comparison_results.items():
        print(f"  {metric_name}: {value}")
        
    print("\nAnalysis complete! Check the outputs directory for visual results.")

if __name__ == "__main__":
    main() 