#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# 示例腳本：使用HookDataLoader和張量處理功能分析模型中間層數據

這個腳本展示如何使用HookDataLoader載入模型鉤子數據，並使用各種張量預處理功能對其進行分析。
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

from data_loader.hook_data_loader import HookDataLoader
from utils.tensor_utils import (
    normalize_tensor, preprocess_activations, handle_nan_values,
    reduce_dimensions, handle_sparse_tensor, batch_process_activations,
    compute_batch_statistics
)


def analyze_layer_distributions(experiment_dir: str, 
                               layer_names: List[str], 
                               epochs: List[int],
                               output_dir: str = 'outputs'):
    """
    分析層激活值分布及其在訓練過程中的變化
    
    Args:
        experiment_dir: 實驗結果目錄
        layer_names: 要分析的層名稱列表
        epochs: 要分析的輪次列表
        output_dir: 輸出目錄
    """
    print(f"正在分析實驗：{os.path.basename(experiment_dir)}")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化HookDataLoader
    loader = HookDataLoader(experiment_dir)
    
    # 獲取可用的層和輪次
    available_info = loader.load()
    print(f"可用的輪次: {available_info['available_epochs']}")
    print(f"各輪次可用的層: {len(available_info['available_layers'])}")
    
    # 篩選有效的層和輪次
    valid_layer_names = []
    for layer_name in layer_names:
        found = False
        for epoch in epochs:
            if epoch in available_info['available_epochs'] and \
               layer_name in available_info['available_layers'].get(epoch, []):
                found = True
                break
        
        if found:
            valid_layer_names.append(layer_name)
        else:
            print(f"警告: 層 '{layer_name}' 在指定的輪次中不可用，將被跳過")
    
    valid_epochs = [epoch for epoch in epochs if epoch in available_info['available_epochs']]
    
    if not valid_layer_names or not valid_epochs:
        print("錯誤: 沒有有效的層或輪次可供分析")
        return
    
    print(f"將分析以下層: {valid_layer_names}")
    print(f"將分析以下輪次: {valid_epochs}")
    
    # 批量載入激活值數據
    activations = loader.load_activations_batch(
        layer_names=valid_layer_names,
        epochs=valid_epochs,
        preprocess_config={'remove_outliers': True},
        verbose=True
    )
    
    # 分析每個層
    for layer_name, epoch_data in activations.items():
        print(f"\n分析層: {layer_name}")
        
        # 創建存放該層分析結果的目錄
        layer_output_dir = os.path.join(output_dir, layer_name)
        os.makedirs(layer_output_dir, exist_ok=True)
        
        # 統計每個輪次的激活值分布特性，用於比較
        stats_by_epoch = {}
        for epoch, data in epoch_data.items():
            # 計算基本統計數據
            stats = compute_batch_statistics(data)
            stats_by_epoch[epoch] = stats
            
            # 將預處理後的激活值保存回實驗目錄
            processed = normalize_tensor(data, method='zscore')
            loader.save_processed_activation(
                processed, layer_name, epoch, suffix='normalized'
            )
            
            # 繪製該輪次的激活值分布直方圖
            plt.figure(figsize=(10, 6))
            plt.hist(data.flatten().detach().cpu().numpy(), bins=50, alpha=0.7)
            plt.title(f"{layer_name} Epoch {epoch} Activation Distribution")
            plt.xlabel("Activation Value")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(layer_output_dir, f"{layer_name}_epoch_{epoch}_hist.png"))
            plt.close()
        
        # 比較不同輪次之間的統計指標變化
        plt.figure(figsize=(12, 8))
        epochs = sorted(stats_by_epoch.keys())
        
        metrics = ['mean', 'std', 'max', 'min', 'median', 'sparsity']
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i+1)
            values = [stats_by_epoch[epoch][metric] for epoch in epochs]
            plt.plot(epochs, values, 'o-', linewidth=2)
            plt.title(f"{metric.capitalize()}")
            plt.xlabel("Epoch")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(layer_output_dir, f"{layer_name}_metrics_evolution.png"))
        plt.close()
        
        # 分析並比較不同輪次的激活值空間分布
        if len(epoch_data) > 1:
            print(f"  分析層 {layer_name} 在訓練過程中的演化...")
            
            # 對於全連接層，可視化特徵空間的變化
            if len(data.shape) == 2:  # [batch_size, features]
                # 主成分分析或t-SNE可視化
                try:
                    from sklearn.decomposition import PCA
                    
                    plt.figure(figsize=(12, 10))
                    for i, (epoch, data) in enumerate(sorted(epoch_data.items())):
                        # 降到2維進行可視化
                        flattened = data.detach().cpu().numpy()
                        pca = PCA(n_components=2)
                        reduced = pca.fit_transform(flattened)
                        
                        plt.scatter(reduced[:, 0], reduced[:, 1], 
                                   label=f"Epoch {epoch}", alpha=0.7)
                    
                    plt.title(f"{layer_name} Activation Space Evolution")
                    plt.xlabel("PC1")
                    plt.ylabel("PC2")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(layer_output_dir, f"{layer_name}_pca_evolution.png"))
                    plt.close()
                except Exception as e:
                    print(f"  無法執行PCA降維：{e}")
            
            # 對於卷積層，比較特徵圖的變化
            elif len(data.shape) == 4:  # [batch_size, channels, height, width]
                try:
                    # 計算每個通道的平均激活強度
                    plt.figure(figsize=(12, 8))
                    for epoch, data in sorted(epoch_data.items()):
                        # 計算每個通道的平均激活強度
                        channel_means = data.mean(dim=(0, 2, 3)).detach().cpu().numpy()
                        plt.plot(channel_means, label=f"Epoch {epoch}", alpha=0.7)
                    
                    plt.title(f"{layer_name} Channel Activation Evolution")
                    plt.xlabel("Channel Index")
                    plt.ylabel("Average Activation")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(layer_output_dir, f"{layer_name}_channel_evolution.png"))
                    plt.close()
                    
                    # 可視化一些選定通道的特徵圖
                    first_epoch_data = epoch_data[min(epoch_data.keys())]
                    last_epoch_data = epoch_data[max(epoch_data.keys())]
                    
                    # 選擇一個樣本和一些通道
                    sample_idx = 0
                    num_channels = min(4, first_epoch_data.shape[1])
                    
                    plt.figure(figsize=(15, 8))
                    for i in range(num_channels):
                        # 第一個輪次的特徵圖
                        plt.subplot(2, num_channels, i+1)
                        plt.imshow(first_epoch_data[sample_idx, i].detach().cpu().numpy())
                        plt.title(f"Epoch {min(epoch_data.keys())}, Channel {i}")
                        plt.colorbar()
                        
                        # 最後一個輪次的特徵圖
                        plt.subplot(2, num_channels, num_channels+i+1)
                        plt.imshow(last_epoch_data[sample_idx, i].detach().cpu().numpy())
                        plt.title(f"Epoch {max(epoch_data.keys())}, Channel {i}")
                        plt.colorbar()
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(layer_output_dir, f"{layer_name}_featuremap_comparison.png"))
                    plt.close()
                    
                except Exception as e:
                    print(f"  無法執行特徵圖分析：{e}")
    
    print(f"\n分析完成！結果已保存到: {output_dir}")


def analyze_layer_sparsity_and_saturation(experiment_dir: str,
                                         layer_names: List[str], 
                                         epochs: List[int],
                                         output_dir: str = 'outputs'):
    """
    分析層的稀疏度和飽和度
    
    Args:
        experiment_dir: 實驗結果目錄
        layer_names: 要分析的層名稱列表
        epochs: 要分析的輪次列表
        output_dir: 輸出目錄
    """
    print(f"正在分析層的稀疏度和飽和度...")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化HookDataLoader
    loader = HookDataLoader(experiment_dir)
    
    # 篩選有效的層和輪次
    available_info = loader.load()
    valid_layer_names = [layer for layer in layer_names 
                        if any(layer in available_info['available_layers'].get(epoch, []) 
                              for epoch in epochs)]
    valid_epochs = [epoch for epoch in epochs if epoch in available_info['available_epochs']]
    
    # 批量載入激活值數據
    activations = loader.load_activations_batch(
        layer_names=valid_layer_names,
        epochs=valid_epochs,
        verbose=True
    )
    
    # 分析稀疏度
    sparsity_results = {}
    for layer_name, epoch_data in activations.items():
        sparsity_results[layer_name] = {}
        
        for epoch, data in epoch_data.items():
            # 計算不同閾值下的稀疏度
            thresholds = [0.0, 0.01, 0.05, 0.1]
            sparsity_at_thresholds = {}
            
            for threshold in thresholds:
                if threshold == 0.0:
                    # 完全為零的元素比例
                    sparsity = (data == 0.0).float().mean().item()
                else:
                    # 接近零的元素比例
                    sparsity = (torch.abs(data) < threshold).float().mean().item()
                
                sparsity_at_thresholds[threshold] = sparsity
            
            sparsity_results[layer_name][epoch] = sparsity_at_thresholds
            
            # 計算ReLU飽和度（如果適用）
            if data.min() >= 0:  # 可能是ReLU激活後的層
                saturation = (data == 0.0).float().mean().item()
                sparsity_results[layer_name][epoch]['relu_saturation'] = saturation
    
    # 繪製稀疏度隨訓練變化的圖表
    plt.figure(figsize=(15, 10))
    
    for i, layer_name in enumerate(sparsity_results.keys()):
        plt.subplot(len(sparsity_results), 1, i+1)
        
        epochs = sorted(sparsity_results[layer_name].keys())
        
        # 繪製不同閾值的稀疏度
        for threshold in [0.0, 0.01, 0.05, 0.1]:
            values = [sparsity_results[layer_name][epoch][threshold] for epoch in epochs]
            plt.plot(epochs, values, 'o-', linewidth=2, 
                    label=f"Threshold {threshold}")
        
        # 如果有ReLU飽和度，也繪製出來
        if 'relu_saturation' in sparsity_results[layer_name][epochs[0]]:
            values = [sparsity_results[layer_name][epoch]['relu_saturation'] for epoch in epochs]
            plt.plot(epochs, values, 's--', linewidth=2, color='red',
                    label="ReLU Saturation")
        
        plt.title(f"{layer_name} Sparsity Evolution")
        plt.xlabel("Epoch")
        plt.ylabel("Sparsity Ratio")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_sparsity_evolution.png"))
    plt.close()
    
    print(f"稀疏度分析完成！結果已保存到: {output_dir}")


def analyze_nan_and_infinity(experiment_dir: str,
                            layer_names: List[str], 
                            epochs: List[int],
                            output_dir: str = 'outputs'):
    """
    檢測並分析層激活值中的NaN和無限值
    
    Args:
        experiment_dir: 實驗結果目錄
        layer_names: 要分析的層名稱列表
        epochs: 要分析的輪次列表
        output_dir: 輸出目錄
    """
    print(f"正在檢測NaN和無限值...")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化HookDataLoader
    loader = HookDataLoader(experiment_dir)
    
    # 獲取有效的層和輪次
    available_info = loader.load()
    valid_layer_names = [layer for layer in layer_names 
                        if any(layer in available_info['available_layers'].get(epoch, []) 
                             for epoch in epochs)]
    valid_epochs = [epoch for epoch in epochs if epoch in available_info['available_epochs']]
    
    # 分析結果
    results = {}
    
    for layer_name in valid_layer_names:
        layer_results = {}
        
        for epoch in valid_epochs:
            if layer_name not in available_info['available_layers'].get(epoch, []):
                continue
                
            # 載入激活值
            activation = loader.load_activation(layer_name, epoch)
            if activation is None:
                continue
                
            # 檢測NaN和無限值
            has_nan = torch.isnan(activation).any().item()
            has_inf = torch.isinf(activation).any().item()
            
            # 如果存在NaN或無限值，計算其比例
            nan_ratio = torch.isnan(activation).float().mean().item() if has_nan else 0
            inf_ratio = torch.isinf(activation).float().mean().item() if has_inf else 0
            
            layer_results[epoch] = {
                'has_nan': has_nan,
                'has_inf': has_inf,
                'nan_ratio': nan_ratio,
                'inf_ratio': inf_ratio
            }
            
            if has_nan or has_inf:
                print(f"警告: 層 {layer_name} 在輪次 {epoch} 中包含 {nan_ratio:.6f} 的NaN和 {inf_ratio:.6f} 的無限值")
                
                # 修復並保存
                fixed_activation = handle_nan_values(activation, strategy='zero')
                # 將無限值替換為該維度的最大值或0
                if has_inf:
                    inf_mask = torch.isinf(fixed_activation)
                    fixed_activation[inf_mask] = 0.0
                
                loader.save_processed_activation(
                    fixed_activation, layer_name, epoch, suffix='fixed_nan_inf'
                )
                print(f"  已修復並保存到 epoch_{epoch}/{layer_name}_fixed_nan_inf_batch_0.pt")
        
        if layer_results:
            results[layer_name] = layer_results
    
    # 如果有發現問題，生成報告
    if results:
        report_path = os.path.join(output_dir, "nan_inf_report.txt")
        with open(report_path, 'w') as f:
            f.write("NaN和無限值檢測報告\n")
            f.write("=" * 50 + "\n\n")
            
            for layer_name, layer_results in results.items():
                f.write(f"{layer_name}:\n")
                f.write("-" * 30 + "\n")
                
                for epoch, epoch_results in sorted(layer_results.items()):
                    f.write(f"  輪次 {epoch}:\n")
                    f.write(f"    存在NaN: {epoch_results['has_nan']}\n")
                    f.write(f"    存在無限值: {epoch_results['has_inf']}\n")
                    f.write(f"    NaN比例: {epoch_results['nan_ratio']:.6f}\n")
                    f.write(f"    無限值比例: {epoch_results['inf_ratio']:.6f}\n\n")
        
        print(f"NaN和無限值檢測報告已保存到: {report_path}")
    else:
        print("未發現任何NaN或無限值，網絡正常！")


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析模型中間層數據')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='實驗結果目錄路徑')
    parser.add_argument('--output_dir', type=str, default='hook_analysis_outputs',
                        help='輸出目錄')
    parser.add_argument('--layers', type=str, nargs='+', default=[],
                        help='要分析的層名稱列表，為空則分析所有層')
    parser.add_argument('--epochs', type=int, nargs='+', default=[],
                        help='要分析的輪次列表，為空則分析所有輪次')
    parser.add_argument('--analysis_type', type=str, default='all',
                        choices=['distribution', 'sparsity', 'nan_inf', 'all'],
                        help='要執行的分析類型')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化HookDataLoader
    loader = HookDataLoader(args.experiment_dir)
    
    # 如果未指定層或輪次，則使用所有可用的
    available_info = loader.load()
    
    if not args.layers:
        all_layers = set()
        for epoch, layers in available_info['available_layers'].items():
            all_layers.update(layers)
        args.layers = list(all_layers)
    
    if not args.epochs:
        args.epochs = available_info['available_epochs']
    
    # 執行選定的分析
    if args.analysis_type in ['distribution', 'all']:
        analyze_layer_distributions(
            args.experiment_dir, 
            args.layers, 
            args.epochs,
            os.path.join(args.output_dir, 'distributions')
        )
    
    if args.analysis_type in ['sparsity', 'all']:
        analyze_layer_sparsity_and_saturation(
            args.experiment_dir, 
            args.layers, 
            args.epochs,
            os.path.join(args.output_dir, 'sparsity')
        )
    
    if args.analysis_type in ['nan_inf', 'all']:
        analyze_nan_and_infinity(
            args.experiment_dir, 
            args.layers, 
            args.epochs,
            os.path.join(args.output_dir, 'nan_inf')
        )


if __name__ == '__main__':
    # 設置torch的seed以確保可重現性
    torch.manual_seed(42)
    np.random.seed(42)
    
    main() 