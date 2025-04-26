"""
# 示例代碼：使用SBP_analyzer分析單一層級指標
# 
# 此範例展示如何深入分析單一網絡層的行為特性，包括:
# - 激活值分佈
# - 神經元活躍度
# - 輸出特徵圖
# - 飽和度與稀疏度分析
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 將項目根目錄添加到路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.tensor_utils import (
    normalize_tensor, 
    extract_feature_maps,
    compute_batch_statistics
)
from metrics.layer_activity_metrics import (
    compute_neuron_activity, 
    compute_activation_distribution,
    detect_dead_neurons,
    compute_saturation_metrics
)


def create_mock_layer_activations():
    """創建模擬層激活值用於示例分析"""
    # 模擬卷積層激活值 - 形狀為 [batch_size, channels, height, width]
    conv_layer = torch.randn(16, 64, 28, 28)
    
    # 模擬一些死亡神經元
    dead_channels = np.random.choice(64, size=10, replace=False)
    conv_layer[:, dead_channels, :, :] = 0.0
    
    # 模擬一些飽和神經元
    saturated_channels = np.random.choice(64, size=8, replace=False)
    conv_layer[:, saturated_channels, :, :] = torch.sigmoid(torch.randn(16, 8, 28, 28) * 10)
    
    # 模擬全連接層激活值 - 形狀為 [batch_size, num_neurons]
    fc_layer = torch.randn(16, 128)
    
    # 模擬一些死亡神經元
    dead_neurons = np.random.choice(128, size=20, replace=False)
    fc_layer[:, dead_neurons] = 0.0
    
    # 模擬一些恒定輸出神經元
    constant_neurons = np.random.choice(128, size=15, replace=False)
    fc_layer[:, constant_neurons] = 0.5
    
    return {
        'conv_layer': conv_layer,
        'fc_layer': fc_layer
    }


def analyze_single_layer(layer_activations, layer_name, layer_type):
    """分析單一層的激活值"""
    print(f"\n===== 分析層: {layer_name} ({layer_type}) =====")
    
    # 1. 基本統計信息
    stats = compute_batch_statistics(layer_activations)
    print(f"基本統計信息:")
    print(f"  平均值: {stats['mean']:.4f}")
    print(f"  標準差: {stats['std']:.4f}")
    print(f"  最小值: {stats['min']:.4f}")
    print(f"  最大值: {stats['max']:.4f}")
    print(f"  數值範圍: {stats['range']:.4f}")
    
    # 2. 激活值分佈
    activation_dist = compute_activation_distribution(layer_activations)
    print(f"激活值分佈:")
    print(f"  零值比例: {activation_dist['zero_fraction']:.4f}")
    print(f"  負值比例: {activation_dist['negative_fraction']:.4f}")
    print(f"  正值比例: {activation_dist['positive_fraction']:.4f}")
    
    # 3. 神經元活躍度
    neuron_activity = compute_neuron_activity(layer_activations)
    print(f"神經元活躍度:")
    print(f"  活躍神經元數量: {neuron_activity['active_neurons']}")
    print(f"  平均活躍度: {neuron_activity['mean_activity']:.4f}")
    
    # 4. 死神經元檢測
    dead_neurons = detect_dead_neurons(layer_activations)
    print(f"死神經元檢測:")
    print(f"  死神經元比例: {dead_neurons['dead_ratio']:.4f}")
    print(f"  死神經元數量: {dead_neurons['dead_count']}")
    
    # 5. 飽和度分析
    saturation = compute_saturation_metrics(layer_activations)
    print(f"飽和度分析:")
    print(f"  飽和度: {saturation['saturation_ratio']:.4f}")
    print(f"  飽和神經元數量: {saturation['saturated_count']}")
    
    return {
        'stats': stats,
        'activation_dist': activation_dist,
        'neuron_activity': neuron_activity,
        'dead_neurons': dead_neurons,
        'saturation': saturation
    }


def visualize_layer_activations(layer_activations, layer_name, output_dir):
    """視覺化層級激活值"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 激活值分佈直方圖
    plt.figure(figsize=(10, 6))
    flattened = layer_activations.reshape(-1).numpy()
    plt.hist(flattened, bins=50, alpha=0.7)
    plt.title(f'{layer_name} Activation Distribution')
    plt.xlabel('Activation Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{layer_name}_activation_histogram.png'))
    
    # 2. 神經元平均激活值熱圖
    if len(layer_activations.shape) == 4:  # 卷積層
        # 計算每個卷積核(channel)的平均激活值
        channel_means = layer_activations.mean(dim=[0, 2, 3]).numpy()
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(channel_means)), channel_means)
        plt.title(f'{layer_name} Channel Mean Activations')
        plt.xlabel('Channel Index')
        plt.ylabel('Mean Activation')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{layer_name}_channel_means.png'))
        
        # 特徵圖視覺化
        if layer_activations.shape[1] > 16:
            num_feature_maps = 16
        else:
            num_feature_maps = layer_activations.shape[1]
            
        feature_maps = extract_feature_maps(layer_activations[0], num_feature_maps)
        
        # 繪製特徵圖網格
        grid_size = int(np.ceil(np.sqrt(num_feature_maps)))
        plt.figure(figsize=(12, 12))
        
        for i, feature_map in enumerate(feature_maps):
            if i >= num_feature_maps:
                break
                
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(feature_map, cmap='viridis')
            plt.title(f'Channel {i}')
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{layer_name}_feature_maps.png'))
        
    else:  # 全連接層
        neuron_means = layer_activations.mean(dim=0).numpy()
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(neuron_means)), neuron_means)
        plt.title(f'{layer_name} Neuron Mean Activations')
        plt.xlabel('Neuron Index')
        plt.ylabel('Mean Activation')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{layer_name}_neuron_means.png'))
    
    # 3. 樣本間激活值變化圖
    if len(layer_activations.shape) == 2:  # 全連接層
        # 選擇前5個樣本和前20個神經元進行視覺化
        sample_activations = layer_activations[:5, :20].numpy()
        
        plt.figure(figsize=(12, 8))
        for i in range(5):
            plt.plot(range(20), sample_activations[i], label=f'Sample {i}', marker='o')
            
        plt.title(f'{layer_name} Activation Patterns Across Samples')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{layer_name}_sample_patterns.png'))
    
    print(f"層級激活值視覺化已保存至 {output_dir}")


def main():
    """主函數"""
    # 創建輸出目錄
    output_dir = './examples/layer_metrics_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 獲取模擬激活值
    print("創建模擬層激活值...")
    activations = create_mock_layer_activations()
    
    # 分析卷積層
    conv_results = analyze_single_layer(
        activations['conv_layer'], 
        'conv_example', 
        'Convolutional'
    )
    
    # 分析全連接層
    fc_results = analyze_single_layer(
        activations['fc_layer'], 
        'fc_example', 
        'Fully Connected'
    )
    
    # 視覺化結果
    print("\n生成視覺化圖表...")
    visualize_layer_activations(
        activations['conv_layer'], 
        'conv_example', 
        output_dir
    )
    
    visualize_layer_activations(
        activations['fc_layer'], 
        'fc_example', 
        output_dir
    )
    
    print("\n分析完成！請查看 examples/layer_metrics_output 目錄下的視覺化結果。")


if __name__ == "__main__":
    main() 