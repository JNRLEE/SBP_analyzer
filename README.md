# SBP Analyzer (Ad-hoc Analysis Focus)

SBP Analyzer 是一個用於 **Ad-hoc (離線) 分析** 已完成的深度學習模型訓練過程的 Python 套件。
它專為分析 MicDysphagiaFramework 產生的實驗結果而設計，旨在幫助開發者深入理解模型行為、訓練動態和潛在問題。

## 新架構

```
SBP_analyzer/
├── analyzer/                    # 核心分析邏輯
│   ├── __init__.py
│   ├── base_analyzer.py
│   ├── model_structure_analyzer.py
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
│   └── performance_metrics.py
│   └── layer_activity_metrics.py
├── visualization/               # 視覺化功能
│   ├── __init__.py
│   ├── distribution_plots.py
│   ├── performance_plots.py
│   ├── model_structure_plots.py
│   └── layer_activity_plots.py
├── reporter/                    # 報告生成
│   ├── __init__.py
│   └── report_generator.py
├── utils/                       # 通用工具
│   ├── __init__.py
│   ├── file_utils.py
│   └── tensor_utils.py
│   └── stat_utils.py
├── interfaces/                  # 用戶接口
│   ├── __init__.py
│   └── analyzer_interface.py
└── tests/                       # 測試模組
    ├── test_data/
    └── ...
```

## 主要功能

*   **數據載入**: 從 MicDysphagiaFramework 實驗結果目錄載入配置、模型結構、訓練歷史和 hook 數據。
*   **模型結構分析**: 解析並視覺化 model_structure.json 中的模型架構和參數分布。
*   **訓練動態分析**: 分析 training_history.json 中的損失函數和評估指標的趨勢和穩定性。
*   **中間層數據分析**: 分析 hooks/ 目錄中儲存的模型內部激活值的分佈和統計特性。
*   **視覺化**: 提供多種圖表來展示分析結果。
*   **報告生成**: 自動生成結構化的分析報告 (HTML/Markdown)。

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

## 下一步

詳細的開發路線圖和設計請參見 `Instructor.md`。
