# SBP_analyzer Examples

本目錄包含展示如何使用 SBP_analyzer 庫的範例程式碼。

## 分佈分析範例 (Distribution Analysis Example)

`distribution_analysis_example.py` 展示了如何使用 SBP_analyzer 中的分佈分析功能：

1. 使用 `detect_distribution_type` 和 `detect_distribution_type_advanced` 函數來分析張量分佈
2. 比較不同類型的分佈特徵
3. 視覺化分佈及其分析結果

### 如何運行

```bash
# 從專案根目錄運行
python examples/distribution_analysis_example.py
```

運行後，程式將生成多種不同類型的分佈資料，並使用基本和高級檢測方法進行比較分析，結果將保存在 `outputs` 目錄下的 `distribution_detection_comparison.png` 檔案中。

### 支援的分佈類型

此範例程式生成並分析以下類型的分佈：

- 正態分佈 (Normal)
- 偏態分佈 (左偏和右偏)
- 均勻分佈 (Uniform)
- 稀疏分佈 (Sparse)
- 飽和分佈 (Saturated)
- 雙峰分佈 (Bimodal)
- 多模態分佈 (Multimodal)
- 重尾分佈 (Heavy-tailed)
- 高斯混合模型 (Gaussian Mixture)

### 分析指標

分佈分析會計算以下關鍵指標：

- 熵值 (Entropy)
- 偏度 (Skewness)
- 峰度 (Kurtosis)
- 稀疏性 (Sparsity)
- 雙峰指數 (Bimodality Index)
- 飽和度 (Saturation)

## 其他範例

更多範例將陸續添加。 