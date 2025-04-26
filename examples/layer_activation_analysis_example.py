"""
此示例展示如何使用SBP_analyzer對模型訓練中的層級激活值數據進行分析，
並生成分析報告。

使用此示例，您可以：
1. 載入實驗目錄中的激活值數據
2. 分析層級活動特性
3. 生成視覺化圖表
4. 創建綜合報告
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# 添加項目根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader.hook_data_loader import HookDataLoader
from data_loader.experiment_loader import ExperimentLoader
from metrics.layer_activity_metrics import (
    calculate_activation_statistics,
    calculate_activation_sparsity,
    calculate_activation_saturation,
    calculate_effective_rank,
    calculate_feature_coherence,
    detect_dead_neurons,
    calculate_activation_dynamics,
    calculate_layer_similarity
)
from utils.tensor_utils import normalize_tensor_zscore
from reporter.report_generator import ReportGenerator

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='層級激活值分析示例')
    parser.add_argument(
        '--experiment_dir',
        type=str,
        required=True,
        help='實驗目錄路徑，應包含hooks子目錄'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./analysis_output',
        help='分析結果輸出目錄'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        nargs='+',
        help='要分析的輪次，如果未指定則分析所有可用輪次'
    )
    parser.add_argument(
        '--layers',
        type=str,
        nargs='+',
        help='要分析的層名稱，如果未指定則分析所有可用層'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=0,
        help='要分析的批次編號，預設為0'
    )
    parser.add_argument(
        '--report_format',
        type=str,
        choices=['markdown', 'html', 'pdf'],
        default='html',
        help='報告格式，預設為HTML'
    )
    
    return parser.parse_args()

def analyze_layer_activation(activation_tensor, layer_name):
    """
    分析層級激活值的特性
    
    Args:
        activation_tensor: 層激活值張量
        layer_name: 層名稱
        
    Returns:
        包含分析結果的字典
    """
    logger.info(f"分析層 {layer_name} 的激活值...")
    
    # 基本統計指標
    stats = calculate_activation_statistics(activation_tensor)
    logger.info(f"  均值: {stats['mean']:.4f}, 標準差: {stats['std']:.4f}")
    logger.info(f"  最小值: {stats['min']:.4f}, 最大值: {stats['max']:.4f}")
    
    # 稀疏度分析
    sparsity = calculate_activation_sparsity(activation_tensor)
    logger.info(f"  稀疏度: {sparsity:.4f}")
    
    # 飽和度分析
    saturation = calculate_activation_saturation(activation_tensor)
    logger.info(f"  飽和度: {saturation:.4f}")
    
    # 有效秩分析
    if len(activation_tensor.shape) >= 2:
        effective_rank = calculate_effective_rank(activation_tensor)
        logger.info(f"  有效秩: {effective_rank:.2f}")
    else:
        effective_rank = None
        logger.info("  激活值維度不足，無法計算有效秩")
    
    # 特徵一致性分析
    if len(activation_tensor.shape) >= 3:
        feature_coherence = calculate_feature_coherence(activation_tensor)
        logger.info(f"  特徵一致性: {feature_coherence:.4f}")
    else:
        feature_coherence = None
        logger.info("  激活值維度不足，無法計算特徵一致性")
    
    # 死亡神經元檢測
    dead_neurons = detect_dead_neurons(activation_tensor)
    logger.info(f"  死亡神經元: {dead_neurons['dead_count']} ({dead_neurons['dead_ratio']:.2%})")
    
    return {
        "statistics": stats,
        "sparsity": sparsity,
        "saturation": saturation,
        "effective_rank": effective_rank,
        "feature_coherence": feature_coherence,
        "dead_neurons": dead_neurons
    }

def create_activation_visualizations(activation_tensor, layer_name, output_dir):
    """
    為激活值創建視覺化圖表
    
    Args:
        activation_tensor: 層激活值張量
        layer_name: 層名稱
        output_dir: 輸出目錄
        
    Returns:
        包含生成的圖表路徑的字典
    """
    logger.info(f"為層 {layer_name} 創建視覺化圖表...")
    
    os.makedirs(output_dir, exist_ok=True)
    plots = {}
    
    # 激活值分布直方圖
    plt.figure(figsize=(10, 6))
    plt.hist(activation_tensor.flatten().numpy(), bins=50, alpha=0.7)
    plt.xlabel("激活值")
    plt.ylabel("頻率")
    plt.title(f"層 {layer_name} 的激活值分布")
    
    dist_plot_path = os.path.join(output_dir, f"{layer_name}_distribution.png")
    plt.savefig(dist_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    plots["distribution"] = dist_plot_path
    
    # 如果激活值張量是2D或更高維度，創建熱力圖
    if len(activation_tensor.shape) >= 2:
        # 獲取第一個批次的數據
        first_item = activation_tensor[0]
        
        # 根據維度不同，準備熱力圖數據
        if len(first_item.shape) == 1:
            # 1D特徵向量
            heatmap_data = first_item.reshape(1, -1)
        elif len(first_item.shape) == 2:
            # 2D特徵圖
            heatmap_data = first_item
        else:
            # 多維特徵圖，取第一個通道
            heatmap_data = first_item[0]
        
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap_data.numpy(), cmap='viridis', aspect='auto')
        plt.colorbar(label="激活值")
        plt.title(f"層 {layer_name} 的激活值熱力圖 (第一個樣本)")
        
        heat_plot_path = os.path.join(output_dir, f"{layer_name}_heatmap.png")
        plt.savefig(heat_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        plots["heatmap"] = heat_plot_path
    
    return plots

def compare_layer_activations_across_epochs(activations_dict, layer_name, output_dir):
    """
    比較不同輪次的層激活值
    
    Args:
        activations_dict: 輪次到激活值的映射字典
        layer_name: 層名稱
        output_dir: 輸出目錄
        
    Returns:
        包含比較圖表路徑的字典
    """
    logger.info(f"比較層 {layer_name} 在不同輪次中的激活值...")
    
    if len(activations_dict) <= 1:
        logger.warning("不足兩個輪次的數據，無法進行比較")
        return {}
    
    os.makedirs(output_dir, exist_ok=True)
    plots = {}
    
    # 比較不同輪次的激活值分布
    plt.figure(figsize=(12, 8))
    
    for epoch, activation in sorted(activations_dict.items()):
        plt.hist(
            activation.flatten().numpy(), 
            bins=50, 
            alpha=0.5, 
            label=f"輪次 {epoch}",
            density=True
        )
    
    plt.xlabel("激活值")
    plt.ylabel("密度")
    plt.title(f"層 {layer_name} 在不同輪次的激活值分布比較")
    plt.legend()
    
    compare_plot_path = os.path.join(output_dir, f"{layer_name}_epoch_comparison.png")
    plt.savefig(compare_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    plots["epoch_comparison"] = compare_plot_path
    
    # 比較不同輪次的統計指標
    epochs = sorted(activations_dict.keys())
    metrics = {
        "均值": [],
        "標準差": [],
        "稀疏度": [],
        "死亡神經元比例": []
    }
    
    for epoch in epochs:
        activation = activations_dict[epoch]
        stats = calculate_activation_statistics(activation)
        dead_info = detect_dead_neurons(activation)
        
        metrics["均值"].append(stats["mean"])
        metrics["標準差"].append(stats["std"])
        metrics["稀疏度"].append(calculate_activation_sparsity(activation))
        metrics["死亡神經元比例"].append(dead_info["dead_ratio"])
    
    # 繪製指標變化曲線
    plt.figure(figsize=(12, 10))
    
    for i, (metric_name, values) in enumerate(metrics.items(), 1):
        plt.subplot(2, 2, i)
        plt.plot(epochs, values, 'o-', linewidth=2)
        plt.xlabel("輪次")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name}隨輪次的變化")
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    metrics_plot_path = os.path.join(output_dir, f"{layer_name}_metrics_evolution.png")
    plt.savefig(metrics_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    plots["metrics_evolution"] = metrics_plot_path
    
    return plots

def compare_layers_within_epoch(activations_dict, epoch, output_dir):
    """
    比較同一輪次中不同層的激活值特性
    
    Args:
        activations_dict: 層名稱到激活值的映射字典
        epoch: 輪次編號
        output_dir: 輸出目錄
        
    Returns:
        包含比較圖表路徑的字典
    """
    logger.info(f"比較輪次 {epoch} 中不同層的激活值...")
    
    if len(activations_dict) <= 1:
        logger.warning("不足兩個層的數據，無法進行比較")
        return {}
    
    os.makedirs(output_dir, exist_ok=True)
    plots = {}
    
    # 比較不同層的激活值統計指標
    layer_names = list(activations_dict.keys())
    metrics = {
        "均值": [],
        "標準差": [],
        "稀疏度": [],
        "飽和度": [],
        "死亡神經元比例": []
    }
    
    for layer in layer_names:
        activation = activations_dict[layer]
        stats = calculate_activation_statistics(activation)
        dead_info = detect_dead_neurons(activation)
        
        metrics["均值"].append(stats["mean"])
        metrics["標準差"].append(stats["std"])
        metrics["稀疏度"].append(calculate_activation_sparsity(activation))
        metrics["飽和度"].append(calculate_activation_saturation(activation))
        metrics["死亡神經元比例"].append(dead_info["dead_ratio"])
    
    # 繪製柱狀圖比較
    for metric_name, values in metrics.items():
        plt.figure(figsize=(12, 6))
        plt.bar(layer_names, values, alpha=0.7)
        plt.xlabel("層名稱")
        plt.ylabel(metric_name)
        plt.title(f"輪次 {epoch} 中不同層的 {metric_name} 比較")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        metric_plot_path = os.path.join(output_dir, f"epoch{epoch}_layers_comparison_{metric_name}.png")
        plt.savefig(metric_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        plots[f"layers_comparison_{metric_name}"] = metric_plot_path
    
    # 計算層間相似度
    if len(layer_names) > 1:
        similarity_matrix = np.zeros((len(layer_names), len(layer_names)))
        
        for i, layer1 in enumerate(layer_names):
            for j, layer2 in enumerate(layer_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # 先將兩層的激活值展平
                    act1 = activations_dict[layer1].flatten()
                    act2 = activations_dict[layer2].flatten()
                    
                    # 如果長度不同，截斷較長的一個
                    min_len = min(len(act1), len(act2))
                    act1 = act1[:min_len]
                    act2 = act2[:min_len]
                    
                    # 計算相似度
                    similarity = calculate_layer_similarity(act1, act2)
                    similarity_matrix[i, j] = similarity.get('cosine_similarity', 0)
        
        # 繪製層間相似度熱力圖
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label="餘弦相似度")
        plt.title(f"輪次 {epoch} 中層間激活值相似度")
        plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
        plt.yticks(range(len(layer_names)), layer_names)
        
        for i in range(len(layer_names)):
            for j in range(len(layer_names)):
                plt.text(j, i, f"{similarity_matrix[i, j]:.2f}", 
                         ha="center", va="center", 
                         color="white" if similarity_matrix[i, j] < 0.5 else "black")
        
        plt.tight_layout()
        similarity_plot_path = os.path.join(output_dir, f"epoch{epoch}_layer_similarity.png")
        plt.savefig(similarity_plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        plots["layer_similarity"] = similarity_plot_path
    
    return plots

def main():
    """主函數"""
    args = parse_args()
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化數據載入器
    hook_loader = HookDataLoader(args.experiment_dir)
    exp_loader = ExperimentLoader(args.experiment_dir)
    
    # 載入實驗基本信息
    try:
        config = exp_loader.load_config()
        model_structure = exp_loader.load_model_structure()
        training_history = exp_loader.load_training_history()
        
        experiment_name = config.get("experiment_name", "未知實驗")
        logger.info(f"載入實驗: {experiment_name}")
    except Exception as e:
        logger.error(f"載入實驗基本信息時出錯: {str(e)}")
        config = {}
        model_structure = {}
        training_history = {}
        experiment_name = "未知實驗"
    
    # 創建圖表輸出目錄
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 確定要分析的輪次
    available_epochs = hook_loader.list_available_epochs()
    logger.info(f"發現可用輪次: {available_epochs}")
    
    epochs_to_analyze = args.epochs if args.epochs else available_epochs
    epochs_to_analyze = [e for e in epochs_to_analyze if e in available_epochs]
    
    if not epochs_to_analyze:
        logger.error("沒有可用的輪次進行分析！")
        return
    
    logger.info(f"將分析輪次: {epochs_to_analyze}")
    
    # 確定要分析的層
    first_epoch = epochs_to_analyze[0]
    available_layers = hook_loader.list_available_layers(first_epoch, args.batch)
    logger.info(f"發現可用層: {available_layers}")
    
    layers_to_analyze = args.layers if args.layers else available_layers
    layers_to_analyze = [l for l in layers_to_analyze if l in available_layers]
    
    if not layers_to_analyze:
        logger.error("沒有可用的層進行分析！")
        return
    
    logger.info(f"將分析層: {layers_to_analyze}")
    
    # 進行層級激活值分析
    layer_analysis_results = {}
    
    # 1. 分析各層在各輪次的激活值
    for layer_name in layers_to_analyze:
        layer_output_dir = os.path.join(plots_dir, layer_name)
        os.makedirs(layer_output_dir, exist_ok=True)
        
        # 收集不同輪次的激活值
        epoch_activations = {}
        epoch_results = {}
        
        for epoch in epochs_to_analyze:
            activation = hook_loader.load_layer_activation(layer_name, epoch, args.batch)
            
            if activation is not None:
                # 分析激活值特性
                analysis_result = analyze_layer_activation(activation, f"{layer_name} (輪次 {epoch})")
                
                # 創建視覺化
                visualizations = create_activation_visualizations(
                    activation, 
                    f"{layer_name}_epoch{epoch}", 
                    layer_output_dir
                )
                
                # 保存結果
                epoch_activations[epoch] = activation
                epoch_results[epoch] = {
                    "analysis": analysis_result,
                    "visualizations": visualizations
                }
        
        # 比較不同輪次的激活值
        if len(epoch_activations) > 1:
            comparison_plots = compare_layer_activations_across_epochs(
                epoch_activations, 
                layer_name, 
                layer_output_dir
            )
        else:
            comparison_plots = {}
        
        # 保存層分析結果
        layer_analysis_results[layer_name] = {
            "epochs": epoch_results,
            "comparisons": comparison_plots
        }
    
    # 2. 對於每個輪次，比較不同層之間的激活值
    epoch_layer_comparisons = {}
    
    for epoch in epochs_to_analyze:
        # 收集該輪次所有層的激活值
        layer_activations = {}
        
        for layer_name in layers_to_analyze:
            activation = hook_loader.load_layer_activation(layer_name, epoch, args.batch)
            if activation is not None:
                layer_activations[layer_name] = activation
        
        # 比較該輪次的不同層
        if len(layer_activations) > 1:
            comparison_dir = os.path.join(plots_dir, f"epoch{epoch}_comparisons")
            comparison_plots = compare_layers_within_epoch(
                layer_activations, 
                epoch, 
                comparison_dir
            )
            epoch_layer_comparisons[epoch] = comparison_plots
    
    # 3. 生成報告所需的圖表列表
    plots_for_report = []
    
    # 添加每層的關鍵圖表
    for layer_name, results in layer_analysis_results.items():
        # 添加最後一個輪次的分布圖
        last_epoch = max(results["epochs"].keys())
        last_epoch_visuals = results["epochs"][last_epoch]["visualizations"]
        
        if "distribution" in last_epoch_visuals:
            plots_for_report.append({
                "title": f"層 {layer_name} 的激活值分布 (輪次 {last_epoch})",
                "path": last_epoch_visuals["distribution"],
                "caption": f"層 {layer_name} 在輪次 {last_epoch} 的激活值分布直方圖"
            })
        
        # 添加跨輪次比較圖
        if "epoch_comparison" in results["comparisons"]:
            plots_for_report.append({
                "title": f"層 {layer_name} 在不同輪次的激活值分布比較",
                "path": results["comparisons"]["epoch_comparison"],
                "caption": f"層 {layer_name} 在不同輪次的激活值分布變化"
            })
        
        if "metrics_evolution" in results["comparisons"]:
            plots_for_report.append({
                "title": f"層 {layer_name} 的指標隨輪次變化",
                "path": results["comparisons"]["metrics_evolution"],
                "caption": f"層 {layer_name} 的各項指標隨訓練輪次的變化"
            })
    
    # 添加層間比較圖表
    for epoch, plots in epoch_layer_comparisons.items():
        if "layer_similarity" in plots:
            plots_for_report.append({
                "title": f"輪次 {epoch} 中層間相似度",
                "path": plots["layer_similarity"],
                "caption": f"輪次 {epoch} 中不同層的激活值相似度矩陣"
            })
        
        if "layers_comparison_稀疏度" in plots:
            plots_for_report.append({
                "title": f"輪次 {epoch} 中不同層的稀疏度比較",
                "path": plots["layers_comparison_稀疏度"],
                "caption": f"輪次 {epoch} 中各層的激活值稀疏度比較"
            })
    
    # 4. 整合分析結果
    analysis_results = {
        "model_structure": model_structure,
        "training_history": training_history,
        "layer_activations": {},
        "plots": plots_for_report
    }
    
    # 為每層準備報告數據
    for layer_name, results in layer_analysis_results.items():
        # 使用最後一個輪次的數據
        last_epoch = max(results["epochs"].keys())
        last_epoch_data = results["epochs"][last_epoch]
        
        analysis_results["layer_activations"][layer_name] = {
            "statistics": last_epoch_data["analysis"]["statistics"],
            "dead_neurons": last_epoch_data["analysis"]["dead_neurons"]
        }
        
        if "distribution" in last_epoch_data["visualizations"]:
            analysis_results["layer_activations"][layer_name]["distribution_plot"] = \
                last_epoch_data["visualizations"]["distribution"]
    
    # 5. 生成報告
    report_output_dir = os.path.join(args.output_dir, "report")
    os.makedirs(report_output_dir, exist_ok=True)
    
    report_title = f"{experiment_name} - 層級激活值分析報告"
    report_generator = ReportGenerator(
        output_dir=report_output_dir,
        experiment_name=experiment_name,
        report_title=report_title
    )
    
    report_path = report_generator.generate_report(
        analysis_results,
        format=args.report_format,
        include_plots=True
    )
    
    logger.info(f"分析完成！報告已生成：{report_path}")
    
    # 在命令行顯示簡要摘要
    print("\n" + "="*50)
    print(f"層級激活值分析摘要 - {experiment_name}")
    print("="*50)
    
    for layer_name in layers_to_analyze:
        if layer_name in layer_analysis_results:
            last_epoch = max(layer_analysis_results[layer_name]["epochs"].keys())
            analysis = layer_analysis_results[layer_name]["epochs"][last_epoch]["analysis"]
            
            print(f"\n層: {layer_name} (輪次 {last_epoch})")
            print(f"  均值: {analysis['statistics']['mean']:.4f}, 標準差: {analysis['statistics']['std']:.4f}")
            print(f"  稀疏度: {analysis['sparsity']:.4f}, 飽和度: {analysis['saturation']:.4f}")
            print(f"  死亡神經元: {analysis['dead_neurons']['dead_ratio']:.2%}")
    
    print("\n" + "="*50)
    print(f"完整報告: {report_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main() 