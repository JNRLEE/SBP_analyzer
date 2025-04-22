# SBP Analyzer (Ad-hoc Analysis Focus)

SBP Analyzer 是一個用於 **Ad-hoc (離線) 分析** 已完成的深度學習模型訓練過程的 Python 套件。
它專為分析 MicDysphagiaFramework 產生的實驗結果而設計，旨在幫助開發者深入理解模型行為、訓練動態和潛在問題。

## 專案結構

```
SBP_analyzer/
├── analyzer/                    # 核心分析邏輯
│   ├── __init__.py
│   ├── base_analyzer.py
│   ├── model_structure_analyzer.py    # 增強：現支持複雜度和效率分析
│   ├── training_dynamics_analyzer.py
│   └── intermediate_data_analyzer.py
├── data_loader/                 # 數據載入與解析
│   ├── __init__.py
│   ├── base_loader.py
│   ├── experiment_loader.py
│   └── hook_data_loader.py
├── metrics/                     # 分析指標計算
│   ├── __init__.py
│   ├── distribution_metrics.py
│   ├── performance_metrics.py
│   └── layer_activity_metrics.py  # 新增：層級活動指標計算
├── visualization/               # 視覺化功能 (所有模組已完成實現)
│   ├── __init__.py
│   ├── plotter.py               # 所有繪圖功能的基類
│   ├── distribution_plots.py    # 分布視覺化
│   ├── performance_plots.py     # 性能指標視覺化
│   └── model_structure_plots.py # 模型結構視覺化
├── reporter/                    # 報告生成
│   ├── __init__.py
│   └── report_generator.py
├── utils/                       # 通用工具
│   ├── __init__.py
│   ├── file_utils.py
│   ├── tensor_utils.py
│   └── stat_utils.py
├── interfaces/                  # 用戶接口
│   ├── __init__.py
│   └── analyzer_interface.py
├── examples/                    # 使用範例
│   └── model_analysis_example.py
└── tests/                       # 測試模組
    ├── test_data/
    ├── test_layer_activity_metrics.py  # 新增：層級活動指標測試
    ├── test_distribution_plots.py
    ├── test_performance_plots.py
    ├── test_model_structure_analyzer.py
    └── ...
```

## 主要功能

* **數據載入**: 從 MicDysphagiaFramework 實驗結果目錄載入配置、模型結構、訓練歷史和 hook 數據。
* **模型結構分析**: 解析並視覺化模型架構和參數分布。
  * 層級複雜度計算、參數效率分析、FLOPs估算、連接性分析
  * 支持多種模型架構類型 (Transformer, Swin, ViT, GNN等)
* **訓練動態分析**: 分析損失函數和評估指標的趨勢和穩定性。
* **中間層數據分析**: 分析模型內部激活值的分佈和統計特性。
  * **新增**: 層級活動指標計算，包括激活值統計、稀疏度、飽和度、有效秩和特徵一致性分析
  * **新增**: 死亡神經元檢測、層間相似度和激活值動態變化分析
* **視覺化**: 提供多種圖表來展示分析結果。
  * 分布視覺化 (直方圖、箱形圖、Q-Q圖等)
  * 性能指標視覺化 (損失曲線、收斂分析、訓練穩定性等)
  * 模型結構視覺化，支持各種複雜網絡
* **報告生成**: 自動生成結構化的分析報告 (HTML/Markdown)。

## 支持的數據結構

SBP Analyzer 專為分析以下 MicDysphagiaFramework 產生的實驗結果目錄結構設計：

```
results/
└── {實驗名稱}_{時間戳}/               # 例：audio_swin_regression_20250417_142912/
    ├── config.json                 # 實驗配置文件
    ├── model_structure.json        # 模型結構信息
    ├── training_history.json       # 訓練歷史記錄
    ├── models/                     # 模型權重保存目錄
    ├── hooks/                      # 模型鉤子數據
    │   ├── training_summary.pt     # 整體訓練摘要
    │   ├── evaluation_results_test.pt  # 測試集評估結果
    │   └── epoch_N/                # 各輪次數據
    └── results/                    # 實驗結果
        └── results.json            # 最終結果摘要
```

## 安裝

```bash
pip install -e .
```

## 使用範例

### 基本使用方式

```python
from sbp_analyzer.interfaces.analyzer_interface import SBPAnalyzer

# 指向 MicDysphagiaFramework 產生的實驗結果目錄
analyzer = SBPAnalyzer(experiment_dir='results/audio_swin_regression_20250417_142912')

# 運行分析
analysis_results = analyzer.analyze(
    analyze_model_structure=True,
    analyze_training_history=True,
    analyze_hooks=True,
    epochs=[0, 5, 10],  # 可選擇指定要分析的特定輪次
    layers=['patch_embed', 'layers.0']  # 可選擇指定要分析的特定層
)

# 生成報告
analyzer.generate_report(output_dir='./analysis_report', report_format='html')

# 或者，直接訪問特定分析結果
model_summary = analysis_results.get_model_summary()
loss_curve_plot = analysis_results.get_plot('loss_curve')
layer_activation_dist = analysis_results.get_activation_distribution('patch_embed', epoch=0)

print("分析完成，報告已生成於 ./analysis_report")
```

### 使用層級活動分析功能

```python
import torch
from metrics.layer_activity_metrics import (
    calculate_activation_statistics,
    calculate_activation_sparsity,
    detect_dead_neurons,
    calculate_layer_similarity
)

# 假設這是某層的激活值
activations = torch.randn(32, 64, 16, 16)  # [batch, channels, height, width]

# 計算激活值的統計指標
stats = calculate_activation_statistics(activations)
print(f"層激活值均值: {stats['mean']:.4f}, 標準差: {stats['std']:.4f}")
print(f"稀疏度: {stats['sparsity']:.2f}, 熵: {stats['entropy']:.2f}")

# 檢測死亡神經元
dead_info = detect_dead_neurons(activations)
print(f"死亡神經元數量: {dead_info['dead_count']}, 比例: {dead_info['dead_ratio']:.2%}")

# 比較兩層的相似度
layer1 = torch.randn(32, 100)
layer2 = torch.randn(32, 100)
similarity = calculate_layer_similarity(layer1, layer2)
print(f"層間相似度: {similarity['cosine_similarity']:.4f}")
```

### 使用視覺化模組

```python
import numpy as np
from visualization.distribution_plots import DistributionPlotter
from visualization.performance_plots import PerformancePlotter

# 創建分布視覺化器
dist_plotter = DistributionPlotter(output_dir='./plots')

# 繪製權重分布直方圖
weights = np.random.normal(0, 0.1, 1000)
dist_plotter.plot_histogram(weights, title='Weight Distribution', filename='weights.png')

# 比較多個分布
distributions = {
    'Conv1': np.random.normal(0, 0.1, 1000),
    'Conv2': np.random.normal(0, 0.05, 1000),
    'FC': np.random.normal(0, 0.2, 1000)
}
dist_plotter.plot_distribution_comparison(distributions, title='Layer Comparisons')

# 創建性能視覺化器
perf_plotter = PerformancePlotter(output_dir='./plots')

# 繪製損失曲線
loss_history = {
    'train': [2.0, 1.8, 1.5, 1.3, 1.1, 0.9, 0.8, 0.7, 0.6, 0.5],
    'val': [2.1, 1.9, 1.6, 1.4, 1.2, 1.0, 0.9, 0.85, 0.8, 0.75]
}
perf_plotter.plot_loss_curve(loss_history, title='Training Loss')
```

### 完整示例

查看 `examples/model_analysis_example.py` 了解如何進行完整的模型分析和視覺化流程。

## 開發進度

目前已完成:
- ✅ 模型結構分析功能
- ✅ 訓練動態分析功能
- ✅ 視覺化模組 (分布、性能和模型結構視覺化)
- ✅ 層級活動指標計算功能

正在進行中:
- 🔄 中間層數據加載與處理
- 🔄 層級行為分析算法
- 🔄 報告生成器實現

## 貢獻

歡迎提交Pull Request或建立Issues來改進本項目。請確保新代碼包含適當的測試和文檔。

## 更多資訊

詳細的開發路線圖和設計請參見 `Instructor.md`。
