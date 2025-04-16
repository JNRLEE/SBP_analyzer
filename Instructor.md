# SBP_analyzer 開發指南

## 1. 專案概述

SBP_analyzer 是一個模型分析工具包，旨在協助深度學習模型訓練過程中的監控和分析。它提供兩種主要功能模式：

1. **即時回調模式**：在訓練過程中即時監控和分析模型行為
2. **離線分析模式**：分析已儲存在 `/results` 目錄中的訓練結果

本專案與 MicDysphagiaFramework 有適配的橋接機制，但也可以作為獨立工具使用。

## 2. 系統架構

### 2.1 核心組件

```
SBP_analyzer/
├── callbacks/                   # 即時回調實現
│   ├── base_callback.py         # 基礎回調類
│   ├── data_drift_callback.py   # 數據漂移監控
│   ├── feature_monitor_callback.py  # 特徵監控
│   ├── outlier_detection_callback.py  # 異常值檢測
│   └── ...
│
├── model_hooks/                 # 模型鉤子功能
│   ├── hook_manager.py          # 鉤子管理器
│   └── hook_utils.py            # 鉤子工具函數
│
├── metrics/                     # 分析指標計算
│   ├── drift_metrics.py         # 數據漂移指標
│   ├── distribution_metrics.py  # 分布比較指標
│   └── ...
│
├── visualization/               # 視覺化功能
│   ├── distribution_plots.py    # 分布視覺化
│   ├── feature_plots.py         # 特徵視覺化
│   └── ...
│
├── results_parser/              # 結果解析功能
│   ├── model_parser.py          # 解析模型結構
│   ├── metric_parser.py         # 解析訓練指標
│   └── ...
│
├── utils/                       # 通用工具
│   ├── tensor_utils.py          # 張量處理工具
│   ├── stat_utils.py            # 統計分析工具
│   └── ...
│
├── interfaces/                  # 用戶接口
│   ├── callback_interface.py    # 回調接口
│   └── analyzer_interface.py    # 分析接口
│
└── tests/                       # 測試模組
    ├── mock_framework/          # 模擬訓練框架
    └── ...
```

### 2.2 雙軌功能模式

#### 即時回調模式
在此模式下，SBP_analyzer 透過回調機制與訓練框架整合，實時監控訓練過程。

```python
# 使用即時回調模式
from sbp_analyzer.callbacks import DataDriftCallback
from micdsyphagia_framework.trainers import PyTorchTrainer

# 創建訓練器
trainer = PyTorchTrainer(model, criterion, optimizer)

# 添加回調
callback = DataDriftCallback(
    reference_data=reference_dataset,
    drift_metrics=['ks', 'psi'],
    output_dir='./analysis_results'
)
trainer.add_callback(callback)

# 執行訓練
trainer.train(train_loader, val_loader, epochs=30)
```

#### 離線分析模式
在此模式下，SBP_analyzer 分析已儲存的訓練結果。

```python
# 使用離線分析模式
from sbp_analyzer.interfaces.analyzer_interface import SBPAnalyzer

# 初始化分析器
analyzer = SBPAnalyzer(results_dir='results/audio_fcnn_regression_20250416_213304')

# 分析訓練過程
training_analysis = analyzer.analyze_training()

# 生成報告
analyzer.generate_report('./analysis_report')
```

## 3. 開發路線圖

### 階段一：基礎架構和核心功能

1. **基礎回調系統**
   - 實現 `BaseCallback` 類
   - 實現模擬測試環境
   - 建立與 MicDysphagiaFramework 的橋接

2. **核心指標計算功能**
   - 實現數據漂移指標
   - 實現分布比較指標
   - 實現異常值檢測指標

3. **基本視覺化功能**
   - 實現分布視覺化
   - 實現訓練曲線視覺化

### 階段二：數據監控與分析功能

1. **數據監控回調**
   - 實現 `DataDriftCallback`
   - 實現 `OutlierDetectionCallback`

2. **結果解析功能**
   - 實現 `ModelParser`
   - 實現 `MetricParser`

3. **提升測試覆蓋率**
   - 增加單元測試
   - 增加集成測試

### 階段三：模型內部分析功能

1. **模型鉤子系統**
   - 實現 `HookManager`
   - 實現激活值和梯度收集功能

2. **模型內部分析回調**
   - 實現 `FeatureMonitorCallback`
   - 實現 `GradientFlowCallback`
   - 實現 `WeightBiasCallback`

3. **增強視覺化功能**
   - 實現特徵相關性視覺化
   - 實現激活值熱力圖
   - 實現梯度流視覺化

### 階段四：高級功能與整合

1. **預測分析功能**
   - 實現 `ErrorAnalysisCallback`
   - 實現 `UncertaintyCallback`

2. **條件觸發機制**
   - 實現基於閾值的觸發
   - 實現基於規則的觸發

3. **文檔與示例**
   - 完善使用文檔
   - 提供典型使用場景的示例

## 4. 回調系統詳細設計

### 4.1 回調接口

所有回調都實現以下基本接口：

```python
class BaseCallback:
    def on_train_begin(self, model, logs=None): ...
    def on_epoch_begin(self, epoch, model, logs=None): ...
    def on_batch_begin(self, batch, model, inputs, targets, logs=None): ...
    def on_batch_end(self, batch, model, inputs, targets, outputs, loss, logs=None): ...
    def on_epoch_end(self, epoch, model, train_logs, val_logs, logs=None): ...
    def on_train_end(self, model, history, logs=None): ...
```

### 4.2 數據流動與調用時序

在訓練流程中，回調的調用時序如下：

1. **初始化階段**：
   ```
   用戶代碼 -> 創建回調實例 -> 添加到訓練器
   ```

2. **訓練啟動階段**：
   ```
   trainer.train() -> on_train_begin(model, logs) -> 各回調的初始化邏輯
   ```

3. **每個 Epoch 開始**：
   ```
   epoch 循環開始 -> on_epoch_begin(epoch, model, logs) -> 各回調的 epoch 開始邏輯
   ```

4. **每個 Batch 開始**：
   ```
   batch 循環開始 -> on_batch_begin(batch, model, inputs, targets, logs) -> 各回調的 batch 開始邏輯
   ```

5. **每個 Batch 結束**：
   ```
   batch 計算完成 -> on_batch_end(batch, model, inputs, targets, outputs, loss, logs) -> 各回調的 batch 結束邏輯
   ```

6. **每個 Epoch 結束**：
   ```
   epoch 循環結束 -> on_epoch_end(epoch, model, train_logs, val_logs, logs) -> 各回調的 epoch 結束邏輯
   ```

7. **訓練結束階段**：
   ```
   訓練完成 -> on_train_end(model, history, logs) -> 各回調的終結邏輯
   ```

### 4.3 測試方法

為確保回調系統的穩定性和正確性，我們採用以下測試方法：

#### 單元測試
針對每個回調類的單獨功能進行測試

```python
def test_data_drift_callback_initialization():
    # 測試回調初始化
    callback = DataDriftCallback(reference_data=torch.randn(100, 10))
    assert callback.reference_data is not None
    assert callback.drift_metrics == ['ks', 'psi', 'wasserstein']
```

#### 模擬環境測試
使用模擬的訓練環境測試回調系統

```python
def test_callback_in_mock_environment():
    # 創建模擬訓練器
    trainer = MockTrainer()
    model = create_simple_model()
    trainer.set_model(model)
    
    # 添加回調
    callback = DataDriftCallback(reference_data=torch.randn(100, 10))
    trainer.add_callback(callback)
    
    # 模擬訓練過程
    logs = trainer.simulate_training(epochs=2, batches_per_epoch=5)
    
    # 驗證回調行為
    assert callback.was_called_train_begin
    assert callback.epoch_call_count == 2
```

#### 集成測試
測試多個回調協同工作的情況

## 5. 開發環境與測試

### 5.1 開發環境設置

```bash
# 創建並激活虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安裝依賴
pip install -r requirements.txt

# 安裝開發模式
pip install -e .
```

### 5.2 測試環境

為了便於測試即時回調功能，我們提供了模擬訓練環境：

```
tests/mock_framework/
├── __init__.py
├── mock_trainer.py    # 模擬訓練器
└── mock_model.py      # 模擬模型
```

這使得我們可以在不依賴實際訓練框架的情況下測試回調功能。

### 5.3 調試工具

提供專用的調試腳本用於快速測試回調功能：

```
scripts/
├── debug_callback.py  # 測試特定回調
└── run_mock_training.py  # 運行模擬訓練
```

## 6. 即時回調與離線分析比較

### 6.1 即時回調優勢

1. **實時監測和干預**
   - 可以在訓練過程中即時發現問題
   - 能夠動態調整訓練參數（如學習率）
   - 可以根據特定條件提前停止訓練（如過擬合）

2. **資源利用效率**
   - 可以直接使用已載入記憶體的資料，無需額外 I/O 操作
   - 避免儲存大量中間狀態，節省磁碟空間

3. **完整性**
   - 可以捕捉訓練中每個批次的詳細資訊
   - 能夠記錄一些可能不會被儲存的暫時狀態（如梯度）

### 6.2 離線分析優勢

1. **非侵入式**
   - 不會影響模型訓練邏輯和效能
   - 訓練與分析解耦，避免分析邏輯錯誤影響訓練

2. **可重複性**
   - 可以反覆分析同一訓練結果
   - 便於比較不同分析方法或參數

3. **延遲分析**
   - 可在訓練完成後再進行複雜分析
   - 適合運算資源有限的環境

### 6.3 雙軌策略建議

我們建議採用雙軌並行的策略：

1. **訓練中使用輕量級回調**：在訓練期間使用資源消耗較小的回調進行基本監控
2. **訓練後進行深入分析**：在訓練完成後使用離線分析功能進行深入分析

## 7. 最佳實踐

### 7.1 記憶體管理

1. 避免在回調中長時間存儲大量張量
2. 使用 `detach().cpu().numpy()` 將張量轉換為NumPy數組並移出GPU
3. 實現滑動窗口機制，只保留最近N個批次的數據

### 7.2 性能優化

1. 使用 `monitoring_frequency` 參數控制回調執行頻率
2. 對計算密集型操作使用批處理或抽樣
3. 在非關鍵路徑（如epoch結束時）執行複雜分析

### 7.3 錯誤處理

1. 回調中包裝操作以捕獲異常，防止單個回調失敗導致整個訓練中斷
2. 使用日誌記錄警告和錯誤，而不是直接拋出異常

## 8. 開發指南

### 8.1 編碼規範

1. 遵循 PEP 8 風格指南
2. 為所有公共方法和類提供文檔字符串
3. 使用類型註解增強代碼可讀性

### 8.2 測試規範

1. 為每個新功能編寫單元測試
2. 保持測試覆蓋率在80%以上
3. 使用模擬對象減少外部依賴

### 8.3 提交規範

1. 每個提交專注於單一功能或修復
2. 提交訊息應簡明扼要地描述變更
3. 大型功能應分為多個較小的提交

## 9. 常見問題解答

### Q: 如何在沒有 TensorBoard 的情況下查看分析結果？
A: 所有回調都支持將結果保存為CSV、JSON或圖像文件，可以使用任何數據可視化工具查看。

### Q: 回調會影響訓練性能嗎？
A: 為最小化性能影響，可以調整回調的 `monitoring_frequency` 參數，並確保計算密集型操作在非關鍵路徑上執行。

### Q: 如何同時使用多個回調？
A: 訓練器支持添加多個回調，它們將按添加順序依次執行。回調之間可以通過 `logs` 字典共享信息。
