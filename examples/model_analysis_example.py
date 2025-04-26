"""
模型結構分析示例腳本。

此腳本展示了如何使用SBP_analyzer中的模型結構分析和視覺化功能，分析深度學習模型並生成報告。
"""

import os
import json
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from analyzer.model_structure_analyzer import ModelStructureAnalyzer
from visualization.model_structure_plots import ModelStructurePlotter
from visualization.distribution_plots import DistributionPlotter
from visualization.performance_plots import PerformancePlotter

# 定義一個示例的CNN模型，用於分析
class ExampleCNN(nn.Module):
    """簡單的CNN模型用於示例分析"""
    def __init__(self):
        super(ExampleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def create_model_structure_json(model, input_shape=(3, 32, 32), save_path=None):
    """
    創建模型結構的JSON文件，用於分析。
    
    Args:
        model: PyTorch模型。
        input_shape: 輸入張量的形狀。
        save_path: 保存JSON文件的路徑。
        
    Returns:
        str: JSON文件路徑。
    """
    # 創建臨時目錄
    if save_path is None:
        temp_dir = tempfile.mkdtemp()
        save_path = os.path.join(temp_dir, "model_structure.json")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 創建一個示例輸入，以便追蹤模型的參數和結構
    dummy_input = torch.randn(1, *input_shape)
    
    # 獲取模型的基本信息
    layers = []
    
    # 構建層級結構
    layer_id = 0
    for name, module in model.named_modules():
        if name == '':  # Skip the top-level module
            continue
        
        # 獲取層的基本信息
        layer_info = {
            "id": f"layer_{layer_id}",
            "name": name,
            "type": module.__class__.__name__,
            "parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "trainable": all(p.requires_grad for p in module.parameters()) if list(module.parameters()) else True,
            "inputs": []  # 將在下面填充
        }
        
        # 將層添加到列表中
        layers.append(layer_info)
        layer_id += 1
    
    # 創建連接關係（這是簡化版，實際中應該使用hooks來獲取準確的連接）
    for i in range(1, len(layers)):
        layers[i]["inputs"] = [layers[i-1]["id"]]
    
    # 創建模型結構JSON
    model_structure = {
        "model_name": model.__class__.__name__,
        "layers": layers
    }
    
    # 添加一些示例權重分布
    for layer in model_structure["layers"]:
        if layer["parameters"] > 0:
            # 添加假的權重統計信息
            layer["weight_stats"] = {
                "mean": 0.0,
                "std": 0.1,
                "min": -0.3,
                "max": 0.3,
                "sparsity": 0.0,
                "histogram": list(np.random.normal(0, 0.1, 10))
            }
    
    # 保存到JSON文件
    with open(save_path, 'w') as f:
        json.dump(model_structure, f, indent=2)
    
    return save_path

def simulate_training_data(epochs=50):
    """
    模擬訓練過程中的數據，用於性能視覺化。
    
    Args:
        epochs: 模擬的訓練周期數。
        
    Returns:
        dict: 包含模擬訓練數據的字典。
    """
    # 模擬損失數據
    train_loss = []
    val_loss = []
    
    # 使用指數衰減加上隨機噪聲模擬訓練損失
    for i in range(epochs):
        base_loss = 2.0 * np.exp(-0.1 * i)
        train_loss.append(base_loss + np.random.normal(0, 0.1))
        val_loss.append(base_loss * 1.2 + np.random.normal(0, 0.15))
    
    # 模擬準確率和F1分數
    train_acc = []
    val_acc = []
    train_f1 = []
    val_f1 = []
    
    # 使用S形曲線模擬準確率和F1分數的增長
    for i in range(epochs):
        progress = 1 / (1 + np.exp(-0.2 * (i - epochs/2)))  # sigmoid函數產生S形曲線
        train_acc.append(min(0.95, 0.5 + 0.45 * progress) + np.random.normal(0, 0.02))
        val_acc.append(min(0.9, 0.45 + 0.4 * progress) + np.random.normal(0, 0.03))
        train_f1.append(min(0.93, 0.48 + 0.43 * progress) + np.random.normal(0, 0.02))
        val_f1.append(min(0.88, 0.43 + 0.38 * progress) + np.random.normal(0, 0.03))
    
    # 模擬學習率
    lr_history = []
    for i in range(epochs):
        # 階梯式學習率衰減
        if i < 20:
            lr = 0.01
        elif i < 35:
            lr = 0.001
        else:
            lr = 0.0001
        lr_history.append(lr)
    
    # 模擬梯度範數
    gradient_norms = []
    for i in range(epochs):
        # 通常訓練開始時梯度較大，然後逐漸減小
        norm = 10.0 * np.exp(-0.08 * i) + np.random.normal(0, 0.5)
        # 添加一些異常值模擬梯度爆炸
        if i in [5, 12]:
            norm = norm * 5
        gradient_norms.append(max(0, norm))
    
    # 返回所有模擬數據
    return {
        "loss_history": {"train": train_loss, "val": val_loss},
        "metric_history": {
            "train": {"accuracy": train_acc, "f1_score": train_f1},
            "val": {"accuracy": val_acc, "f1_score": val_f1}
        },
        "lr_history": lr_history,
        "gradient_norms": gradient_norms
    }

def main():
    """主函數，展示完整的分析流程。"""
    # 創建輸出目錄
    output_dir = os.path.join(os.getcwd(), "analysis_results", "example")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"輸出結果將保存在: {output_dir}")
    
    # 創建模型
    print("創建示例CNN模型...")
    model = ExampleCNN()
    
    # 創建模型結構JSON
    model_json_path = os.path.join(output_dir, "model_structure.json")
    print(f"生成模型結構JSON: {model_json_path}")
    create_model_structure_json(model, save_path=model_json_path)
    
    # 使用ModelStructureAnalyzer分析模型
    print("分析模型結構...")
    analyzer = ModelStructureAnalyzer(output_dir)
    analysis_results = analyzer.analyze(save_results=True)
    
    # 打印分析結果摘要
    print("\n模型分析摘要:")
    model_summary = analyzer.get_model_summary()
    for key, value in model_summary.items():
        if key != "largest_layers":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: [共{len(value)}個大層]")
            for i, layer in enumerate(value):
                print(f"    {i+1}. {layer.get('layer_id', 'unknown')}: {layer.get('parameters', 0)} 參數")
    
    # 創建視覺化目錄
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 使用ModelStructurePlotter視覺化模型結構
    print("\n生成模型結構視覺化...")
    plotter = ModelStructurePlotter()
    
    # 獲取層級結構並視覺化
    hierarchy = analyzer.get_layer_hierarchy()
    
    # 將層級數據轉換為繪圖所需的格式
    plot_hierarchy = {
        "layers": [],
        "connections": []
    }
    
    for node_id, node_data in hierarchy["nodes"].items():
        layer_info = {
            "id": node_id,
            "type": node_data.get("type", "unknown"),
            "parameters": node_data.get("parameters", 0)
        }
        plot_hierarchy["layers"].append(layer_info)
        
        for child in node_data.get("children", []):
            plot_hierarchy["connections"].append({
                "from": node_id,
                "to": child
            })
    
    # 繪製模型結構圖
    model_structure_fig = plotter.plot_model_structure(
        plot_hierarchy, 
        title=f"模型架構: {analysis_results['model_name']}",
        show_params=True,
        highlight_large_layers=True
    )
    model_structure_fig.savefig(os.path.join(vis_dir, "model_structure.png"), dpi=120, bbox_inches='tight')
    
    # 視覺化參數分布
    param_dist_fig = plotter.plot_parameter_distribution(
        analysis_results["parameter_distribution"],
        title="模型參數分布"
    )
    param_dist_fig.savefig(os.path.join(vis_dir, "parameter_distribution.png"), dpi=120, bbox_inches='tight')
    
    # 視覺化層級複雜度
    if "layer_complexity" in analysis_results:
        complexity_fig = plotter.plot_layer_complexity(
            analysis_results["layer_complexity"],
            title="層級複雜度分析"
        )
        complexity_fig.savefig(os.path.join(vis_dir, "layer_complexity.png"), dpi=120, bbox_inches='tight')
    
    # 生成模擬訓練數據
    print("\n生成模擬訓練數據...")
    training_data = simulate_training_data(epochs=50)
    
    # 使用PerformancePlotter視覺化性能指標
    print("生成性能視覺化...")
    perf_plotter = PerformancePlotter(output_dir=vis_dir)
    
    # 繪製損失曲線
    loss_fig = perf_plotter.plot_loss_curve(
        training_data["loss_history"],
        title="訓練及驗證損失",
        filename="loss_curve.png",
        show=False
    )
    
    # 繪製指標曲線
    metrics_fig = perf_plotter.plot_metric_curves(
        training_data["metric_history"],
        title="訓練指標",
        filename="metrics.png",
        show=False
    )
    
    # 繪製學習率曲線
    lr_fig = perf_plotter.plot_learning_rate(
        training_data["lr_history"],
        title="學習率變化",
        filename="learning_rate.png",
        show=False
    )
    
    # 繪製收斂性分析
    convergence_epoch = np.argmin(training_data["loss_history"]["val"])
    conv_fig = perf_plotter.plot_convergence_analysis(
        training_data["loss_history"]["val"],
        convergence_epoch,
        title="驗證損失收斂分析",
        filename="convergence.png",
        show=False
    )
    
    # 繪製訓練穩定性分析
    stability_fig = perf_plotter.plot_training_stability(
        training_data["loss_history"]["train"],
        moving_avg_window=5,
        title="訓練穩定性分析",
        filename="stability.png",
        show=False
    )
    
    # 繪製梯度範數
    grad_fig = perf_plotter.plot_gradient_norm(
        training_data["gradient_norms"],
        title="梯度範數歷史",
        filename="gradient_norms.png",
        show=False
    )
    
    # 使用DistributionPlotter視覺化分布
    print("生成分布視覺化...")
    dist_plotter = DistributionPlotter(output_dir=vis_dir)
    
    # 生成隨機權重分布數據
    weight_distributions = {
        "conv1": np.random.normal(0, 0.1, 1000),
        "conv2": np.random.normal(0, 0.05, 1000),
        "fc1": np.random.normal(0, 0.2, 1000),
        "fc2": np.random.normal(0, 0.15, 1000)
    }
    
    # 繪製直方圖
    hist_fig = dist_plotter.plot_histogram(
        weight_distributions["conv1"],
        title="Conv1層權重分布",
        filename="conv1_weights.png",
        show=False
    )
    
    # 繪製分布比較
    dist_comp_fig = dist_plotter.plot_distribution_comparison(
        weight_distributions,
        title="各層權重分布比較",
        filename="weight_distributions.png",
        show=False
    )
    
    # 繪製箱形圖
    box_fig = dist_plotter.plot_boxplot(
        weight_distributions,
        title="各層權重箱形圖",
        filename="weight_boxplot.png",
        show=False
    )
    
    # 繪製Q-Q圖
    qq_fig = dist_plotter.plot_qq_plot(
        weight_distributions["fc1"],
        title="FC1層權重的Q-Q圖",
        filename="fc1_qq_plot.png",
        show=False
    )
    
    print(f"\n分析完成！所有結果已保存到: {output_dir}")
    print(f"視覺化圖表已保存到: {vis_dir}")

if __name__ == "__main__":
    main() 