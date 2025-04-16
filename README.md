# SBP Analyzer

SBP Analyzer 是一個用於在模型訓練過程中進行數據分析、監控和視覺化的 Python 套件。
它旨在與 `MicDysphagiaFramework` 等訓練框架整合，透過回調機制提供即時的洞察。

/Users/jnrle/Library/CloudStorage/GoogleDrive-jenner.lee.com@gmail.com/My Drive/MicforDysphagia/SBP_analyzer/
├── SBP_analyzer/
│   ├── __init__.py
│   ├── callbacks/            # 回調類別
│   │   ├── __init__.py
│   │   ├── base_callback.py  # 基礎回調類 (可選)
│   │   ├── tensorboard.py    # TensorBoard 相關回調
│   │   └── data_monitor.py   # 數據監控相關回調 (如分佈)
│   ├── inspection/           # 數據檢查工具
│   │   ├── __init__.py
│   │   ├── sampler.py        # 數據抽樣邏輯
│   │   └── validator.py      # 數據驗證邏輯
│   ├── visualization/        # 視覺化工具
│   │   ├── __init__.py
│   │   └── plotter.py        # 繪圖函數 (如繪製直方圖)
│   ├── metrics/              # 指標計算
│   │   ├── __init__.py
│   │   └── distribution.py   # 計算數據分佈統計量
│   └── utils/                # 共用工具函數
│       └── __init__.py
├── setup.py                  # 套件安裝設定檔
├── requirements.txt          # 依賴套件
└── README.md                 # 套件說明

## 功能

* TensorBoard 整合
* 訓練數據抽樣與檢查
* 數據分佈監控
* (未來更多功能...)

## 安裝

```bash
pip install -e .
```

## 使用

(未來補充詳細使用範例)
