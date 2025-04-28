# SBP Analyzer 分析結果

本目錄包含 SBP Analyzer 生成的實驗分析結果。每個實驗的分析結果都存儲在對應的子目錄中，以原始實驗名稱命名。

## 目錄結構

```
analysis_results/
├── {實驗名稱}_{時間戳}/           # 與原始實驗資料夾對應
│   ├── model_analysis/          # 模型結構分析
│   │   └── model_structure_analysis.json  # 模型結構分析結果
│   ├── training_analysis/       # 訓練過程分析
│   │   └── training_dynamics_analysis.json  # 訓練動態分析結果
│   ├── activation_analysis/     # 中間層激活分析
│   │   └── activation_analysis.json  # 激活值分析結果
│   ├── plots/                   # 生成的圖表
│   │   ├── loss_curve.png       # 損失曲線圖
│   │   ├── metric_curves.png    # 指標曲線圖
│   │   └── activation_distributions.png  # 激活分布圖
│   ├── metadata.json            # 分析元數據（原始資料路徑、分析時間等）
│   └── report/                  # 生成的HTML/Markdown報告
│       └── {實驗名稱}_analysis_report_{時間戳}.html  # HTML格式報告
└── index.json                   # 索引文件，關聯實驗與分析結果
```

## 文件說明

### 1. index.json

索引文件，包含所有已分析實驗的映射關係。格式如下：

```json
{
  "實驗名稱1": {
    "original_path": "/path/to/original/experiment1",
    "analysis_path": "analysis_results/實驗名稱1",
    "analysis_time": "2025-04-21T19:45:21.201405"
  },
  "實驗名稱2": {
    "original_path": "/path/to/original/experiment2",
    "analysis_path": "analysis_results/實驗名稱2",
    "analysis_time": "2025-04-22T10:15:33.567891"
  }
}
```

### 2. metadata.json

每個實驗分析目錄下的元數據文件，包含分析的基本信息。格式如下：

```json
{
  "original_path": "/path/to/original/experiment",
  "analysis_time": "2025-04-21T19:45:21.201405",
  "analyzer_version": "0.1.0",
  "status": "completed",
  "elapsed_time": 0.009963750839233398
}
```

### 3. 分析結果文件

- **model_structure_analysis.json**：包含模型結構、參數量、層級信息等分析結果
- **training_dynamics_analysis.json**：包含訓練過程中的損失函數、評估指標、收斂性等分析
- **activation_analysis.json**：包含中間層激活值的統計特性和分布信息

### 4. 圖表文件

各種可視化圖表，如損失曲線、指標變化、參數分布等，以PNG格式存儲。

### 5. 報告文件

綜合分析報告，包含各方面分析結果的摘要和詳細信息，通常以HTML或Markdown格式提供。

## 使用說明

1. 查看特定實驗的分析報告：打開對應目錄下的 `report/{實驗名稱}_analysis_report_{時間戳}.html` 文件
2. 比較多個實驗：可以參考各個實驗目錄下的JSON分析結果或報告
3. 查找特定實驗的分析結果：查閱 `index.json` 獲取實驗映射關係 