"""
# 示例代碼：使用SBP_analyzer分析模型中間層數據
# 
# 此範例展示如何使用SBP_analyzer分析模型訓練過程中的中間層激活值
# 並找出潛在的問題層、分析層間關係和激活值動態變化
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 將項目根目錄添加到路徑
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from analyzer.intermediate_data_analyzer import IntermediateDataAnalyzer
from data_loader.hook_data_loader import HookDataLoader


def create_mock_experiment_data(experiment_dir):
    """創建模擬實驗數據用於示例"""
    # 創建目錄結構
    hooks_dir = os.path.join(experiment_dir, 'hooks')
    os.makedirs(hooks_dir, exist_ok=True)
    
    # 創建訓練摘要
    training_summary = {
        'total_epochs': 10,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
        'batch_size': 32
    }
    torch.save(training_summary, os.path.join(hooks_dir, 'training_summary.pt'))
    
    # 創建輪次目錄與數據
    for epoch in range(10):
        epoch_dir = os.path.join(hooks_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 模擬批次數據
        batch_data = {
            'inputs': torch.randn(32, 3, 224, 224),
            'targets': torch.randint(0, 10, (32,)),
            'loss': 2.0 - epoch * 0.2 + np.random.randn() * 0.1
        }
        torch.save(batch_data, os.path.join(epoch_dir, 'batch_0_data.pt'))
        
        # 模擬輪次摘要
        epoch_summary = {
            'epoch': epoch,
            'train_loss': 2.0 - epoch * 0.2,
            'val_loss': 2.2 - epoch * 0.18,
            'lr': 0.001 if epoch < 5 else 0.0001
        }
        torch.save(epoch_summary, os.path.join(epoch_dir, 'epoch_summary.pt'))
        
        # 模擬各層激活值
        # 1. 正常的卷積層
        conv1 = torch.randn(32, 64, 112, 112) * (1.0 + epoch * 0.05)
        torch.save(conv1, os.path.join(epoch_dir, 'conv1_activation_batch_0.pt'))
        
        # 2. 正常的卷積層，但逐漸變得稀疏
        conv2 = torch.randn(32, 128, 56, 56) * (1.0 - epoch * 0.08)
        torch.save(conv2, os.path.join(epoch_dir, 'conv2_activation_batch_0.pt'))
        
        # 3. 逐漸死亡的卷積層（神經元逐漸失活）
        conv3 = torch.randn(32, 256, 28, 28)
        dead_ratio = min(0.8, epoch * 0.1)
        dead_mask = torch.rand(32, 256, 28, 28) < dead_ratio
        conv3[dead_mask] = 0
        torch.save(conv3, os.path.join(epoch_dir, 'conv3_activation_batch_0.pt'))
        
        # 4. 數值逐漸爆炸的層
        fc1 = torch.randn(32, 1024) * (1.0 + epoch * 0.5)
        torch.save(fc1, os.path.join(epoch_dir, 'fc1_activation_batch_0.pt'))
        
        # 5. 逐漸飽和的層
        fc2 = torch.sigmoid(torch.randn(32, 512) * (1.0 + epoch * 0.3))
        torch.save(fc2, os.path.join(epoch_dir, 'fc2_activation_batch_0.pt'))
        
        # 6. 恒定不變的層
        fc3 = torch.ones(32, 10) * 0.5
        torch.save(fc3, os.path.join(epoch_dir, 'fc3_activation_batch_0.pt'))


def run_intermediate_analysis(experiment_dir):
    """執行中間層數據分析"""
    print(f"開始分析實驗目錄: {experiment_dir}")
    
    # 初始化分析器
    analyzer = IntermediateDataAnalyzer(experiment_dir)
    
    # 進行分析
    results = analyzer.analyze()
    
    # 顯示層統計摘要
    df = analyzer.get_summary_dataframe()
    print("\n層統計摘要:")
    pd.set_option('display.max_columns', None)
    print(df[['layer_name', 'epoch', 'mean', 'std', 'min', 'max', 'sparsity']].head(10))
    
    # 查找問題層
    problematic_layers = analyzer.get_problematic_layers()
    print(f"\n檢測到 {len(problematic_layers)} 個可能存在問題的層:")
    for i, layer_info in enumerate(problematic_layers):
        print(f"{i+1}. {layer_info['layer_name']} (輪次 {layer_info['epoch']}): {', '.join(layer_info['issues'])}")
    
    # 特定層的統計分析
    print("\n特定層的統計分析:")
    for layer_name in ['conv3', 'fc1', 'fc2']:
        for epoch in [0, 5, 9]:
            stats = analyzer.get_layer_statistics(layer_name, epoch)
            if stats:
                print(f"{layer_name} (輪次 {epoch}):")
                print(f"  平均值: {stats['basic_statistics']['mean']:.4f}")
                print(f"  稀疏度: {stats['sparsity']:.4f}")
                print(f"  飽和度: {stats['saturation']:.4f}")
                if 'dead_neurons' in stats:
                    print(f"  死神經元比例: {stats['dead_neurons'].get('dead_ratio', 'N/A')}")
    
    # 層間關係分析
    print("\n層間關係分析:")
    sim = analyzer.get_layer_relationship('conv1', 'conv2', 0)
    if sim:
        print(f"conv1 vs conv2 相似度:")
        print(f"  相關係數: {sim['mean_correlation']:.4f}")
        print(f"  餘弦相似度: {sim['cosine_similarity']:.4f}")
    
    # 激活值動態變化
    print("\n激活值動態變化:")
    for layer_name in ['conv1', 'conv3', 'fc1']:
        dynamics = analyzer.get_layer_dynamics(layer_name)
        if dynamics and 'error' not in dynamics:
            print(f"{layer_name} 激活值變化率: {dynamics.get('mean_change_rate', 'N/A')}")
    
    # 生成完整報告
    report = analyzer.generate_report()
    print(f"\n生成報告包含 {report['summary']['total_layers_analyzed']} 個層的分析結果")
    
    return analyzer


def visualize_layer_metrics(analyzer, output_dir):
    """視覺化層級指標"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 獲取數據框
    df = analyzer.get_summary_dataframe()
    
    # 按層名分組
    layers = df['layer_name'].unique()
    
    # 1. 繪製每個層的平均值隨輪次變化
    plt.figure(figsize=(10, 6))
    for layer in layers:
        layer_df = df[df['layer_name'] == layer]
        plt.plot(layer_df['epoch'], layer_df['mean'], label=layer, marker='o')
    
    plt.title('Layer Mean Activation Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Activation')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'layer_mean_over_epochs.png'))
    
    # 2. 繪製每個層的稀疏度隨輪次變化
    plt.figure(figsize=(10, 6))
    for layer in layers:
        layer_df = df[df['layer_name'] == layer]
        plt.plot(layer_df['epoch'], layer_df['sparsity'], label=layer, marker='o')
    
    plt.title('Layer Sparsity Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Sparsity')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'layer_sparsity_over_epochs.png'))
    
    # 3. 繪製所有層在最後一個輪次的統計信息比較
    last_epoch = df['epoch'].max()
    last_epoch_df = df[df['epoch'] == last_epoch]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(layers))
    width = 0.2
    
    plt.bar(x - width, last_epoch_df['mean'], width, label='Mean')
    plt.bar(x, last_epoch_df['std'], width, label='Std')
    plt.bar(x + width, last_epoch_df['sparsity'], width, label='Sparsity')
    
    plt.xlabel('Layer')
    plt.ylabel('Value')
    plt.title(f'Layer Statistics Comparison (Epoch {last_epoch})')
    plt.xticks(x, layers, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_statistics_comparison.png'))
    
    print(f"圖表已保存到 {output_dir}")


def main():
    """主函數"""
    # 創建臨時實驗目錄
    experiment_dir = './examples/temp_experiment'
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 創建模擬數據
    print("創建模擬實驗數據...")
    create_mock_experiment_data(experiment_dir)
    
    # 運行分析
    analyzer = run_intermediate_analysis(experiment_dir)
    
    # 視覺化結果
    visualize_layer_metrics(analyzer, './examples/temp_experiment/plots')
    
    print("\n分析完成！請查看 examples/temp_experiment/plots 目錄下的視覺化結果。")


if __name__ == "__main__":
    main() 