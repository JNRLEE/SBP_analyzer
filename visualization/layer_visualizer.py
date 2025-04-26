"""
# 神經網絡層級活動度可視化模組
#
# 此模組負責將神經網絡各層激活值視覺化，包括：
# - 激活值分佈直方圖
# - 特徵圖熱力圖
# - 神經元活躍度視覺化
# - 死亡與飽和神經元視覺化
# - 多批次對比分析視覺化
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Optional, Union
import seaborn as sns
from pathlib import Path

from metrics.layer_activity_metrics import (
    compute_activation_distribution,
    detect_dead_neurons,
    compute_saturation_metrics
)


def visualize_activation_distribution(activations: torch.Tensor, 
                                      layer_name: str = "layer",
                                      save_path: Optional[str] = None) -> Figure:
    """
    視覺化層級激活值分佈。
    
    Args:
        activations: 層級激活值張量
        layer_name: 層名稱，用於圖表標題
        save_path: 保存圖表的路徑，若為None則不保存
        
    Returns:
        matplotlib圖表物件
    """
    # 確保輸入是張量
    if not isinstance(activations, torch.Tensor):
        activations = torch.tensor(activations)
    
    # 將張量轉換為一維向量用於繪製直方圖
    flat_activations = activations.reshape(-1).detach().cpu().numpy()
    
    # 計算基本統計量
    mean_val = np.mean(flat_activations)
    median_val = np.median(flat_activations)
    min_val = np.min(flat_activations)
    max_val = np.max(flat_activations)
    std_val = np.std(flat_activations)
    
    # 計算激活分佈
    dist_info = compute_activation_distribution(activations)
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 繪製直方圖
    sns.histplot(flat_activations, bins=50, kde=True, ax=ax)
    
    # 添加統計資訊
    stats_text = (
        f"均值: {mean_val:.4f}\n"
        f"中位數: {median_val:.4f}\n"
        f"最小值: {min_val:.4f}\n"
        f"最大值: {max_val:.4f}\n"
        f"標準差: {std_val:.4f}\n"
        f"零值比例: {dist_info['zero_fraction']:.2%}\n"
        f"正值比例: {dist_info['positive_fraction']:.2%}\n"
        f"負值比例: {dist_info['negative_fraction']:.2%}"
    )
    
    # 添加統計文字框
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    # 添加標題和標籤
    ax.set_title(f"{layer_name} 激活值分佈")
    ax.set_xlabel("激活值")
    ax.set_ylabel("頻率")
    
    # 如果指定了保存路徑，則保存圖表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_feature_maps(activations: torch.Tensor,
                           layer_name: str = "layer",
                           sample_idx: int = 0,
                           max_features: int = 16,
                           save_path: Optional[str] = None) -> Optional[Figure]:
    """
    視覺化卷積層的特徵圖。
    
    Args:
        activations: 卷積層級激活值張量 [batch, channels, height, width]
        layer_name: 層名稱，用於圖表標題
        sample_idx: 要顯示的樣本索引
        max_features: 最多顯示的特徵圖數量
        save_path: 保存圖表的路徑，若為None則不保存
        
    Returns:
        matplotlib圖表物件，若輸入不是卷積層則返回None
    """
    # 檢查是否為卷積層激活值
    if len(activations.shape) != 4:
        print(f"警告: {layer_name} 不是卷積層，無法視覺化特徵圖")
        return None
    
    # 獲取樣本的激活值
    sample_activations = activations[sample_idx].detach().cpu()
    
    # 獲取通道數
    num_channels = sample_activations.shape[0]
    
    # 限制顯示的特徵圖數量
    channels_to_show = min(num_channels, max_features)
    
    # 計算網格布局
    grid_size = int(np.ceil(np.sqrt(channels_to_show)))
    
    # 創建圖表
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    # 如果只有一個子圖，則將axes轉換為數組
    if channels_to_show == 1:
        axes = np.array([[axes]])
    elif grid_size == 1:
        axes = np.array([axes])
    
    # 遍歷特徵圖
    for i in range(grid_size):
        for j in range(grid_size):
            channel_idx = i * grid_size + j
            
            if channel_idx < channels_to_show:
                # 獲取特徵圖
                feature_map = sample_activations[channel_idx].numpy()
                
                # 繪製特徵圖熱力圖
                im = axes[i, j].imshow(feature_map, cmap='viridis')
                axes[i, j].set_title(f"通道 {channel_idx}")
                axes[i, j].axis('off')
            else:
                # 隱藏多餘的子圖
                axes[i, j].axis('off')
    
    # 添加顏色條和標題
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    plt.suptitle(f"{layer_name} 特徵圖 (樣本 {sample_idx})", fontsize=16)
    plt.tight_layout()
    
    # 如果指定了保存路徑，則保存圖表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_neuron_activity(activations: torch.Tensor,
                             layer_name: str = "layer",
                             top_k: int = 20,
                             threshold: float = 0.0,
                             save_path: Optional[str] = None) -> Figure:
    """
    視覺化神經元活躍度，顯示最活躍和最不活躍的神經元。
    
    Args:
        activations: 層級激活值張量
        layer_name: 層名稱，用於圖表標題
        top_k: 顯示的頂部和底部神經元數量
        threshold: 神經元被視為活躍的激活閾值
        save_path: 保存圖表的路徑，若為None則不保存
        
    Returns:
        matplotlib圖表物件
    """
    # 確定張量的形狀和維度
    tensor_shape = activations.shape
    
    # 計算神經元維度上的平均激活值
    if len(tensor_shape) == 4:  # 卷積層 [batch, channels, height, width]
        # 對每個通道(卷積核)計算平均激活值
        neuron_means = activations.mean(dim=[0, 2, 3]).detach().cpu()  # 結果形狀: [channels]
        neuron_label = "卷積核"
    elif len(tensor_shape) == 2:  # 全連接層 [batch, neurons]
        # 對每個神經元計算平均激活值
        neuron_means = activations.mean(dim=0).detach().cpu()  # 結果形狀: [neurons]
        neuron_label = "神經元"
    else:
        raise ValueError(f"不支援的激活值張量形狀: {tensor_shape}")
    
    # 計算活躍神經元的數量和比例
    active_mask = neuron_means > threshold
    active_count = torch.sum(active_mask).item()
    total_neurons = neuron_means.numel()
    active_ratio = active_count / total_neurons if total_neurons > 0 else 0.0
    
    # 獲取神經元索引和激活值
    neuron_indices = np.arange(total_neurons)
    neuron_activations = neuron_means.numpy()
    
    # 排序神經元
    sorted_indices = np.argsort(neuron_activations)
    
    # 獲取最活躍和最不活躍的神經元
    top_k = min(top_k, total_neurons // 2)  # 確保top_k不超過總神經元數的一半
    
    least_active_indices = sorted_indices[:top_k]
    most_active_indices = sorted_indices[-top_k:][::-1]  # 反轉以獲得降序
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 繪製最活躍神經元
    ax1.bar(range(top_k), neuron_activations[most_active_indices], color='green')
    ax1.set_xticks(range(top_k))
    ax1.set_xticklabels([f"{idx}" for idx in most_active_indices], rotation=45)
    ax1.set_title(f"最活躍的{top_k}個{neuron_label}")
    ax1.set_xlabel(f"{neuron_label}索引")
    ax1.set_ylabel("平均激活值")
    
    # 繪製最不活躍神經元
    ax2.bar(range(top_k), neuron_activations[least_active_indices], color='red')
    ax2.set_xticks(range(top_k))
    ax2.set_xticklabels([f"{idx}" for idx in least_active_indices], rotation=45)
    ax2.set_title(f"最不活躍的{top_k}個{neuron_label}")
    ax2.set_xlabel(f"{neuron_label}索引")
    ax2.set_ylabel("平均激活值")
    
    # 添加總體信息
    plt.suptitle(f"{layer_name} {neuron_label}活躍度分析 (活躍比例: {active_ratio:.2%})", fontsize=14)
    plt.tight_layout()
    
    # 如果指定了保存路徑，則保存圖表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_dead_and_saturated_neurons(activations: torch.Tensor,
                                         layer_name: str = "layer",
                                         dead_threshold: float = 1e-6,
                                         saturation_threshold: float = 0.95,
                                         save_path: Optional[str] = None) -> Figure:
    """
    視覺化死亡和飽和神經元。
    
    Args:
        activations: 層級激活值張量
        layer_name: 層名稱，用於圖表標題
        dead_threshold: 判定神經元死亡的最大激活閾值
        saturation_threshold: 判定神經元飽和的閾值
        save_path: 保存圖表的路徑，若為None則不保存
        
    Returns:
        matplotlib圖表物件
    """
    # 檢測死亡神經元
    dead_info = detect_dead_neurons(activations, threshold=dead_threshold, return_indices=True)
    
    # 計算飽和度指標
    saturation_info = compute_saturation_metrics(activations, saturation_threshold=saturation_threshold)
    
    # 確定張量的形狀和維度
    tensor_shape = activations.shape
    
    if len(tensor_shape) == 4:  # 卷積層 [batch, channels, height, width]
        neuron_means = activations.mean(dim=[0, 2, 3]).detach().cpu()  # 結果形狀: [channels]
        neuron_label = "卷積核"
    elif len(tensor_shape) == 2:  # 全連接層 [batch, neurons]
        neuron_means = activations.mean(dim=0).detach().cpu()  # 結果形狀: [neurons]
        neuron_label = "神經元"
    else:
        raise ValueError(f"不支援的激活值張量形狀: {tensor_shape}")
    
    # 標準化激活值到[0,1]區間，用於檢測飽和神經元
    normalized = neuron_means.clone()
    if neuron_means.min() < 0 or neuron_means.max() > 1:
        normalized = (neuron_means - neuron_means.min()) / (neuron_means.max() - neuron_means.min() + 1e-8)
    
    # 創建飽和神經元掩碼
    saturated_mask = normalized >= saturation_threshold
    saturated_indices = torch.nonzero(saturated_mask).squeeze().tolist()
    if isinstance(saturated_indices, int) and saturation_info['saturated_count'] == 1:
        saturated_indices = [saturated_indices]
    elif saturation_info['saturated_count'] == 0:
        saturated_indices = []
    
    # 創建所有神經元的狀態掩碼
    total_neurons = neuron_means.numel()
    neuron_status = np.zeros(total_neurons)
    
    # 標記死亡神經元為-1，飽和神經元為1，其餘為0
    if isinstance(dead_info['dead_indices'], list):
        for idx in dead_info['dead_indices']:
            neuron_status[idx] = -1
    
    if isinstance(saturated_indices, list):
        for idx in saturated_indices:
            neuron_status[idx] = 1
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 創建神經元索引數組
    neuron_indices = np.arange(total_neurons)
    
    # 按神經元狀態將神經元分組
    normal_indices = neuron_indices[neuron_status == 0]
    dead_indices = neuron_indices[neuron_status == -1]
    saturated_indices = neuron_indices[neuron_status == 1]
    
    # 繪製神經元激活值，根據狀態著色
    ax.scatter(normal_indices, neuron_means[normal_indices], 
              color='blue', label=f'正常 ({len(normal_indices)})')
    ax.scatter(dead_indices, neuron_means[dead_indices], 
              color='red', label=f'死亡 ({len(dead_indices)})')
    ax.scatter(saturated_indices, neuron_means[saturated_indices], 
              color='orange', label=f'飽和 ({len(saturated_indices)})')
    
    # 添加閾值線
    ax.axhline(y=dead_threshold, color='r', linestyle='--', alpha=0.5, 
              label=f'死亡閾值 ({dead_threshold})')
    
    if saturation_threshold < 1:
        # 計算飽和閾值在原始值範圍中的值
        original_saturation_threshold = (saturation_threshold * 
                                       (neuron_means.max() - neuron_means.min()) + 
                                       neuron_means.min())
        ax.axhline(y=original_saturation_threshold, color='orange', linestyle='--', alpha=0.5,
                  label=f'飽和閾值 ({saturation_threshold}，值={original_saturation_threshold:.4f})')
    
    # 添加標題和標籤
    ax.set_title(f"{layer_name} 死亡和飽和{neuron_label}分析")
    ax.set_xlabel(f"{neuron_label}索引")
    ax.set_ylabel("平均激活值")
    ax.legend()
    
    # 添加統計信息
    stats_text = (
        f"總{neuron_label}數: {total_neurons}\n"
        f"死亡{neuron_label}數: {dead_info['dead_count']} ({dead_info['dead_ratio']:.2%})\n"
        f"飽和{neuron_label}數: {saturation_info['saturated_count']} ({saturation_info['saturation_ratio']:.2%})"
    )
    
    # 添加統計文字框
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # 如果指定了保存路徑，則保存圖表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_activation_across_samples(activations: torch.Tensor,
                                       layer_name: str = "layer",
                                       max_samples: int = 10,
                                       save_path: Optional[str] = None) -> Figure:
    """
    視覺化不同樣本間的激活值變化。
    
    Args:
        activations: 層級激活值張量
        layer_name: 層名稱，用於圖表標題
        max_samples: 最多顯示的樣本數量
        save_path: 保存圖表的路徑，若為None則不保存
        
    Returns:
        matplotlib圖表物件
    """
    # 檢查張量形狀
    tensor_shape = activations.shape
    num_samples = tensor_shape[0]
    
    # 限制顯示的樣本數量
    samples_to_show = min(num_samples, max_samples)
    
    # 計算每個樣本的平均激活值
    if len(tensor_shape) == 4:  # 卷積層 [batch, channels, height, width]
        sample_means = activations.mean(dim=[1, 2, 3]).detach().cpu().numpy()
    elif len(tensor_shape) == 2:  # 全連接層 [batch, neurons]
        sample_means = activations.mean(dim=1).detach().cpu().numpy()
    else:
        raise ValueError(f"不支援的激活值張量形狀: {tensor_shape}")
    
    # 計算神經元間的方差
    if len(tensor_shape) == 4:  # 卷積層
        neuron_means = activations[:samples_to_show].mean(dim=[2, 3]).detach().cpu().numpy()
    elif len(tensor_shape) == 2:  # 全連接層
        neuron_means = activations[:samples_to_show].detach().cpu().numpy()
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 繪製樣本平均激活值
    sample_indices = np.arange(samples_to_show)
    ax1.bar(sample_indices, sample_means[:samples_to_show])
    ax1.set_title(f"{layer_name} 樣本平均激活值")
    ax1.set_xlabel("樣本索引")
    ax1.set_ylabel("平均激活值")
    
    # 繪製熱力圖，顯示每個樣本中神經元的激活度
    im = ax2.imshow(neuron_means, aspect='auto', cmap='viridis')
    ax2.set_title(f"{layer_name} 樣本間神經元激活模式")
    ax2.set_xlabel("神經元索引")
    ax2.set_ylabel("樣本索引")
    
    # 添加顏色條
    plt.colorbar(im, ax=ax2, label="激活值")
    
    plt.tight_layout()
    
    # 如果指定了保存路徑，則保存圖表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_layer_metrics(activations: torch.Tensor,
                           layer_name: str = "layer",
                           output_dir: str = "./layer_visualizations",
                           include_feature_maps: bool = True) -> Dict[str, str]:
    """
    為層級激活值生成一整套可視化圖表。
    
    Args:
        activations: 層級激活值張量
        layer_name: 層名稱，用於圖表標題和文件名
        output_dir: 保存圖表的目錄
        include_feature_maps: 是否包括特徵圖可視化（僅適用於卷積層）
        
    Returns:
        包含所有生成圖表路徑的字典
    """
    # 創建輸出目錄
    layer_output_dir = os.path.join(output_dir, layer_name.replace('/', '_').replace(':', '_'))
    os.makedirs(layer_output_dir, exist_ok=True)
    
    # 保存圖表路徑的字典
    visualization_paths = {}
    
    # 生成激活值分佈圖
    dist_path = os.path.join(layer_output_dir, "activation_distribution.png")
    visualize_activation_distribution(activations, layer_name, dist_path)
    visualization_paths["activation_distribution"] = dist_path
    
    # 生成神經元活躍度圖
    activity_path = os.path.join(layer_output_dir, "neuron_activity.png")
    visualize_neuron_activity(activations, layer_name, save_path=activity_path)
    visualization_paths["neuron_activity"] = activity_path
    
    # 生成死亡和飽和神經元圖
    dead_sat_path = os.path.join(layer_output_dir, "dead_and_saturated.png")
    visualize_dead_and_saturated_neurons(activations, layer_name, save_path=dead_sat_path)
    visualization_paths["dead_and_saturated"] = dead_sat_path
    
    # 生成樣本間激活值對比圖
    samples_path = os.path.join(layer_output_dir, "activation_across_samples.png")
    visualize_activation_across_samples(activations, layer_name, save_path=samples_path)
    visualization_paths["activation_across_samples"] = samples_path
    
    # 對於卷積層，生成特徵圖
    if include_feature_maps and len(activations.shape) == 4:
        feature_maps_path = os.path.join(layer_output_dir, "feature_maps.png")
        visualize_feature_maps(activations, layer_name, save_path=feature_maps_path)
        visualization_paths["feature_maps"] = feature_maps_path
    
    return visualization_paths 