# SBP_analyzer 開發指南 (Ad-hoc 分析版)

## 1. 專案概述

SBP_analyzer 是一個專注於 **Ad-hoc (離線) 模型訓練分析** 的工具包。它旨在協助開發者深入理解已完成的深度學習模型訓練過程。本工具包的核心功能是分析儲存在指定目錄結構中的訓練結果，包括模型架構、訓練指標、以及透過模型鉤子 (hooks) 擷取的中間層數據。

本專案的目標是提供一個系統化、可擴展的框架，用以：
- 解析和載入不同來源的訓練數據。
- 計算多樣化的分析指標。
- 提供豐富的視覺化工具以呈現分析結果。
- 自動生成結構化的分析報告。

## 2. 系統架構

### 2.1 核心組件

```
SBP_analyzer/
├── analyzer/                    # 核心分析邏輯
│   ├── __init__.py
│   ├── base_analyzer.py         # 基礎分析器類別
│   ├── model_structure_analyzer.py # 分析模型結構
│   ├── training_dynamics_analyzer.py # 分析訓練動態 (loss, metrics)
│   └── intermediate_data_analyzer.py # 分析中間層數據
│
├── data_loader/                 # 數據載入與解析
│   ├── __init__.py
│   ├── base_loader.py           # 基礎載入器
│   ├── experiment_loader.py     # 載入標準實驗結果文件 (如 config.json, training_history.json)
│   └── hook_data_loader.py      # 載入由模型鉤子儲存的數據 (.pt 文件)
│
├── metrics/                     # 分析指標計算
│   ├── __init__.py
│   ├── distribution_metrics.py  # 分布比較指標 (e.g., KL散度, Wasserstein距離)
│   └── performance_metrics.py   # 模型性能指標 (e.g., 穩定性, 收斂速度)
│   └── layer_activity_metrics.py # 層活躍度相關指標 (e.g., 稀疏度)
│
├── visualization/               # 視覺化功能
│   ├── __init__.py
│   ├── distribution_plots.py    # 分布視覺化 (直方圖, 密度圖)
│   ├── performance_plots.py     # 性能視覺化 (學習曲線, 指標趨勢)
│   ├── model_structure_plots.py # 模型結構視覺化 (計算圖)
│   └── layer_activity_plots.py  # 層活動視覺化 (熱力圖, 激活值分布)
│
├── reporter/                    # 報告生成
│   ├── __init__.py
│   └── report_generator.py      # 生成 HTML 或 Markdown 格式的分析報告
│
├── utils/                       # 通用工具
│   ├── __init__.py
│   ├── file_utils.py            # 文件和目錄操作工具
│   └── tensor_utils.py          # 張量處理和轉換工具
│   └── stat_utils.py            # 常用統計函數
│
├── interfaces/                  # 用戶接口
│   ├── __init__.py
│   └── analyzer_interface.py    # 分析器主接口，用於啟動分析流程
│
└── tests/                       # 測試模組
    ├── test_data/               # 存放測試用的模擬數據文件
    ├── test_analyzer.py         # 測試分析器模組
    ├── test_data_loader.py      # 測試數據載入器
    ├── test_metrics.py          # 測試指標計算
    └── test_visualization.py    # 測試視覺化功能
```

### 2.2 Ad-hoc 分析流程

典型的 Ad-hoc 分析流程如下：

```python
# 使用 Ad-hoc 分析模式
from sbp_analyzer.interfaces.analyzer_interface import SBPAnalyzer

# 初始化分析器，指定包含訓練結果的目錄
# 這裡假設指向 MicDysphagiaFramework 產生的實驗結果目錄
analyzer = SBPAnalyzer(experiment_dir='results/audio_swin_regression_20250417_142912')

# 執行分析 (可以選擇性指定分析模組)
analysis_results = analyzer.analyze(
    analyze_model_structure=True,  # 分析模型結構
    analyze_training_history=True, # 分析訓練歷史
    analyze_hooks=True,            # 分析 hook 數據
    epochs=[0, 5, 10],             # 可選擇指定要分析的特定輪次
    layers=['patch_embed', 'layers.0']  # 可選擇指定要分析的特定層
)

# 生成分析報告
analyzer.generate_report(
    output_dir='./analysis_report',
    report_format='html' # 或 'markdown'
)

# 或者，直接訪問特定分析結果
model_summary = analysis_results.get_model_summary()
loss_curve_plot = analysis_results.get_plot('loss_curve')
layer_activation_dist = analysis_results.get_activation_distribution('patch_embed', epoch=0)
```

## 3. 開發路線圖

### 階段一：基礎架構和核心功能修復（已完成）

1.  **建立核心架構** ✅
    *   實現 `analyzer`, `data_loader`, `metrics`, `visualization`, `reporter`, `utils`, `interfaces` 目錄結構。
    *   定義 `BaseAnalyzer` 和 `BaseLoader` 抽象基礎類別。
2.  **基礎數據載入** ✅
    *   實現 `ExperimentLoader` 以載入 MicDysphagiaFramework 格式的訓練結果 (config.json, training_history.json 等)。
    *   實現 `file_utils` 來處理路徑和文件查找。
3.  **核心功能修復** ✅
    *   修復 `stat_utils.py` 中的 `find_convergence_point` 和 `detect_outliers` 函數。
    *   修正 `distribution_metrics.py` 中的 `calculate_histogram_intersection` 函數。
    *   添加缺少的 `calculate_distribution_similarity` 和 `compare_tensor_distributions` 函數。
4.  **建立測試基礎** ✅
    *   設置 `pytest` 環境。
    *   創建 `run_tests.py` 腳本用於自動化測試。
    *   為 `utils` 和 `metrics` 模組編寫基礎單元測試。

### 階段二：分析功能增強（進行中）

1.  **模型結構分析完善**（部分完成）
    *   ✅ 實現模型結構視覺化（`visualization/model_structure_plots.py`），包括：
        * 模型架構圖繪製
        * 參數分布視覺化
        * 層複雜度圖和順序路徑圖
    *   ✅ 進一步擴展 `ModelStructureAnalyzer`，支持更多模型類型和架構解析，包括：
        * 添加更多模型類型識別模式 (swin, vision_transformer, mlp, graph_neural_network, generative)
        * 增強層複雜度分析
        * 實現權重分布分析和參數效率計算
    *   ✅ 完成更多統計分析功能，如層連接性分析、參數不均勻度計算（Gini係數）等。
2.  **訓練動態分析強化**（已完成）
    *   ✅ 實現 `performance_metrics.py`，提供多種評估指標計算：
        * 均方根誤差 (RMSE)
        * 平均絕對誤差 (MAE) 
        * 平均絕對百分比誤差 (MAPE)
        * R平方 (R²)
        * 解釋方差比 (EVS)
        * 平均偏差 (MBD)
    *   ✅ 實現 `training_dynamics_analyzer.py` 用於分析訓練過程中的效能曲線。
    *   ✅ 實現核心分析函數，如 `analyze_loss_curve` 和 `analyze_metric_curve`。
    *   ✅ 增加學習效率和收斂性分析工具，如 `analyze_learning_efficiency` 和 `analyze_lr_schedule_impact`。
    *   ✅ 實現自動檢測訓練中的異常模式和瓶頸。
3.  **可視化功能基礎建設**（已完成）
    *   ✅ 建立統一的繪圖接口（`visualization/plotter.py` 中的 `BasePlotter` 類）。
    *   ✅ 實現模型結構視覺化功能 (`visualization/model_structure_plots.py`)。
    *   ✅ 實現 `distribution_plots.py` 中的基礎繪圖功能，包括：
        * 直方圖繪製
        * 分布比較圖
        * 箱形圖和熱圖
        * QQ圖及其他統計圖表
    *   ✅ 實現 `performance_plots.py` 中的效能視覺化功能，包括：
        * 損失曲線繪製
        * 多指標曲線繪製
        * 學習率曲線展示
        * 訓練穩定性和收斂性分析圖
        * 訓練熱圖
    *   ✅ 增強 `inference_analyzer.py` 的推理性能分析，修復JSON序列化和數據類型轉換問題。
4.  **擴展單元測試**（部分完成）
    *   ✅ 為性能指標功能添加單元測試 (`tests/test_performance_metrics.py`)。
    *   ✅ 為模型結構分析和視覺化功能添加單元測試 (`tests/test_model_structure_analyzer.py`)。
    *   ✅ 為新實現的分布和性能視覺化功能添加單元測試 (`tests/test_distribution_plots.py`, `tests/test_performance_plots.py`).
    *   ✅ 為推理分析器添加單元測試 (`tests/test_inference_analyzer.py`)。

### 階段三：中間層數據分析與報告生成（待開始）

1.  **中間數據加載與處理**
    *   ❌ 完成 `HookDataLoader` 以載入 hooks 目錄中的激活值數據。
    *   ❌ 實現張量數據的預處理和正規化功能。
2.  **層級活動分析**
    *   ❌ 實現 `layer_activity_metrics` 計算激活值的統計量和特徵。
    *   ❌ 設計層級行為分析算法，檢測異常激活模式。
    *   ❌ 實現層與層之間的相互關係分析。
3.  **報告生成器實現**
    *   ❌ 設計 `ReportGenerator`，整合分析結果到結構化報告。
    *   ❌ 支持多種報告格式（Markdown、HTML、PDF）。
    *   ❌ 建立報告模板系統，支持定制化報告內容。
4.  **整合測試與示例**
    *   ❌ 編寫端到端的整合測試案例。
    *   ✅ 創建示例腳本：
        * `examples/cross_validation_example.py`：展示交叉驗證評估模型表現。
        * `examples/analyze_training_dynamics.py`：展示分析訓練過程的動態特性。
        * `examples/feature_importance_example.py`：展示特徵重要性分析。

### 階段四：高級分析與生產環境準備（待開始）

1.  **高級分析算法**
    *   ❌ 實現模型解釋性分析，如特徵重要性和敏感度分析。
    *   ❌ 設計模型比較系統，支持不同模型架構和超參數的對比。
    *   ❌ 開發訓練過程優化建議引擎。
2.  **交互式可視化**
    *   ❌ 集成 Plotly 或 Bokeh 等庫，提供交互式圖表。
    *   ❌ 實現自定義儀表板功能，支持用戶配置顯示內容。
    *   ❌ 開發基於網頁的結果瀏覽界面。
3.  **系統優化與性能提升**
    *   ❌ 優化大數據處理流程，支持分批處理和並行計算。
    *   ❌ 提升內存效率，處理超大規模模型和數據集。
    *   ❌ 實現結果緩存機制，提高重複分析效率。
4.  **文檔與部署**
    *   ❌ 完善 API 文檔和使用指南。
    *   ❌ 提供部署指南和生產環境配置最佳實踐。
    *   ❌ 創建示例和教程，覆蓋不同使用場景。

## 4. 數據載入器設計

### 4.1 MicDysphagiaFramework 實驗結果結構

SBP_analyzer 主要針對 MicDysphagiaFramework 產生的實驗結果進行分析。以下是預期的輸入目錄結構和文件格式：

```
results/
└── {實驗名稱}_{時間戳}/               # 例：audio_swin_regression_20250417_142912/
    ├── config.json                 # 實驗配置文件
    ├── model_structure.json        # 模型結構信息
    ├── training_history.json       # 訓練歷史記錄
    ├── models/                     # 模型權重保存目錄
    │   ├── best_model.pth          # 最佳模型權重
    │   ├── checkpoint_epoch_0.pth  # 第0輪模型權重檢查點
    │   └── checkpoint_epoch_1.pth  # 第1輪模型權重檢查點
    ├── hooks/                      # 模型鉤子數據
    │   ├── training_summary.pt     # 整體訓練摘要
    │   ├── evaluation_results_test.pt  # 測試集評估結果
    │   ├── epoch_0/                # 第0輪數據
    │   │   ├── epoch_summary.pt    # 輪次摘要
    │   │   ├── batch_0_data.pt     # 第0批次數據
    │   │   └── head_activation_batch_0.pt  # 頭部層激活值
    │   └── epoch_1/                # 第1輪數據
    │       └── ...                 # 同上
    ├── results/                    # 實驗結果
    │   └── results.json            # 最終結果摘要
    ├── tensorboard_logs/           # TensorBoard日誌
    └── logs/                       # 訓練日誌
```

### 4.2 核心載入器設計

#### `ExperimentLoader`

負責載入和解析實驗的基本信息和訓練結果：

```python
class ExperimentLoader:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
    
    def load_config(self):
        """載入 config.json 中的實驗配置"""
        with open(os.path.join(self.experiment_dir, 'config.json'), 'r') as f:
            return json.load(f)
    
    def load_model_structure(self):
        """載入 model_structure.json 中的模型結構信息"""
        with open(os.path.join(self.experiment_dir, 'model_structure.json'), 'r') as f:
            return json.load(f)
    
    def load_training_history(self):
        """載入 training_history.json 中的訓練歷史記錄"""
        with open(os.path.join(self.experiment_dir, 'training_history.json'), 'r') as f:
            return json.load(f)
    
    def load_results(self):
        """載入 results/results.json 中的最終結果摘要"""
        results_path = os.path.join(self.experiment_dir, 'results', 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                return json.load(f)
        return None
```

#### `HookDataLoader`

負責載入和解析 hooks/ 目錄中的模型鉤子數據：

```python
class HookDataLoader:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.hooks_dir = os.path.join(experiment_dir, 'hooks')
    
    def load_training_summary(self):
        """載入整體訓練摘要"""
        return torch.load(os.path.join(self.hooks_dir, 'training_summary.pt'))
    
    def load_evaluation_results(self, dataset='test'):
        """載入評估結果"""
        return torch.load(os.path.join(self.hooks_dir, f'evaluation_results_{dataset}.pt'))
    
    def load_epoch_summary(self, epoch):
        """載入特定輪次的摘要"""
        return torch.load(os.path.join(self.hooks_dir, f'epoch_{epoch}', 'epoch_summary.pt'))
    
    def load_batch_data(self, epoch, batch):
        """載入特定批次的數據"""
        return torch.load(os.path.join(self.hooks_dir, f'epoch_{epoch}', f'batch_{batch}_data.pt'))
    
    def load_layer_activation(self, layer_name, epoch, batch):
        """載入特定層在特定批次的激活值"""
        pattern = f"{layer_name}_activation_batch_{batch}.pt"
        path = os.path.join(self.hooks_dir, f'epoch_{epoch}', pattern)
        if os.path.exists(path):
            return torch.load(path)
        return None
    
    def list_available_epochs(self):
        """列出可用的輪次"""
        return [int(d.split('_')[1]) for d in os.listdir(self.hooks_dir) 
                if d.startswith('epoch_') and os.path.isdir(os.path.join(self.hooks_dir, d))]
    
    def list_available_layer_activations(self, epoch):
        """列出特定輪次中可用的層激活值文件"""
        epoch_dir = os.path.join(self.hooks_dir, f'epoch_{epoch}')
        return [f for f in os.listdir(epoch_dir) if f.endswith('_activation_batch_0.pt')]
```

## 5. 開發環境與測試

### 5.1 開發環境設置

```bash
# 創建並激活虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows (使用適合您 shell 的命令)

# 安裝依賴 (確保 requirements.txt 已更新)
pip install -r requirements.txt

# 安裝開發模式
pip install -e .
```

### 5.2 測試環境

-   **測試框架**: 使用 `pytest`。
-   **測試數據**: 在 `tests/test_data/` 目錄下創建模擬的實驗結果目錄結構和數據文件，參照 MicDysphagiaFramework 的輸出格式。
-   **測試覆蓋率**: 目標是保持主要模組的測試覆蓋率在 80% 以上。

```python
# tests/test_data_loader.py 範例
import pytest
from sbp_analyzer.data_loader import ExperimentLoader, HookDataLoader

def test_experiment_loader_loads_config(mock_experiment_dir):
    loader = ExperimentLoader(mock_experiment_dir)
    config = loader.load_config()
    assert config is not None
    assert 'experiment_name' in config
    assert 'model' in config

def test_hook_data_loader_loads_training_summary(mock_experiment_dir):
    loader = HookDataLoader(mock_experiment_dir)
    summary = loader.load_training_summary()
    assert summary is not None
    assert 'total_training_time' in summary
```

## 6. 最佳實踐

### 6.1 數據處理與記憶體

1.  **延遲載入 (Lazy Loading)**: 對於大型的 hook 數據，僅在實際需要分析時才載入記憶體。
2.  **抽樣 (Sampling)**: 對於大型激活值張量，提供抽樣選項以加速分析和視覺化。
3.  **數據格式**: 內部盡量使用高效的數據結構 (如 Pandas DataFrame, NumPy arrays, PyTorch Tensors)。

### 6.2 模組化與可配置性

1.  **解耦**: 分析器、載入器、指標計算、視覺化應盡量解耦，方便獨立測試和替換。
2.  **配置**: 允許用戶通過接口參數或配置文件來自訂分析流程，例如選擇要運行的分析器、要計算的指標、要生成的圖表等。

### 6.3 錯誤處理與日誌

1.  **健壯性**: `DataLoader` 應能處理文件缺失或格式錯誤的情況，並提供有意義的錯誤訊息。
2.  **日誌**: 在分析過程中記錄詳細的日誌，包括載入了哪些數據、執行了哪些分析、遇到的警告或錯誤等。

## 7. 開發指南

### 7.1 編碼規範

1.  遵循 PEP 8 風格指南。
2.  為所有公共方法、類和模組提供清晰的 Docstrings (遵循如 Google 或 NumPy 風格)。
3.  使用類型註解 (Type Hinting) 增強代碼可讀性和可靠性。

### 7.2 測試規範

1.  為每個新功能或錯誤修復編寫單元測試。
2.  確保測試涵蓋邊界條件和預期的錯誤情況。
3.  定期運行測試套件並檢查覆蓋率報告。

### 7.3 提交規範

1.  遵循 Conventional Commits 規範或其他清晰的提交訊息格式。
2.  每個提交應專注於一個邏輯單元 (如一個功能、一個修復)。
3.  保持 Git 歷史清晰。

## 8. 常見問題解答 (預期)

### Q: `SBP_analyzer` 是否支持非 MicDysphagiaFramework 產生的實驗結果？
A: 初期會專注於支持 MicDysphagiaFramework 的輸出格式。未來計劃添加適配器，以支持其他框架 (如 PyTorch Lightning, TensorFlow) 的輸出格式。

### Q: 如何處理非常大的激活值數據文件？
A: `HookDataLoader` 提供選項來僅載入數據的子集或進行抽樣。同時，分析器和視覺化模組也設計為能夠處理抽樣後的數據。

### Q: 我可以添加自己的分析指標或視覺化方法嗎？
A: 是的，設計目標之一就是提供可擴展的接口。用戶應該能夠繼承基礎類別 (如 `BaseAnalyzer`, `BaseMetricCalculator`, `BasePlotter`) 並註冊自己的實現。

### Q: 分析報告可以自訂嗎？
A: `ReportGenerator` 將提供選項來自訂報告中包含的內容、順序和可能的格式。
