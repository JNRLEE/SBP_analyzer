"""
utils 模組

此模組包含以下主要組件：

Functions:
    _deprecated_detect_overfitting_simple: 已棄用: 使用上面的新實現替代

Args:
    train_loss (List[float] or np (stat_utils.py)
    _new_tensor_init: 無文檔字符串 (stat_utils.py)
    _tensor_name: 無文檔字符串 (stat_utils.py)
    _tensor_set_name: 無文檔字符串 (stat_utils.py)
    average_activations: 計算多個激活值的平均值

Args:
    activations_dict: 層名稱到激活值張量的映射字典
    
Returns:
    Dict[str, torch (tensor_utils.py)
    batch_process_activations: 批次處理多層激活值

Args:
    activations: 層名稱到激活值張量的映射字典
    operations: 要執行的操作列表，每個操作是一個字典，包含'type'和其他參數
               支持的操作類型：'normalize', 'preprocess', 'handle_nan',
               'reduce_dims', 'handle_sparse'
    
Returns:
    Dict[str, torch (tensor_utils.py)
    batch_to_features: 從批次數據中提取特徵

Args:
    batch_data: 批次數據字典
    feature_key: 特徵的鍵名
    
Returns:
    np (tensor_utils.py)
    calculate_distribution_metrics: 計算數據分佈的統計指標，用於分佈類型檢測。

Args:
    data (np (stat_utils.py)
    calculate_distribution_stats: 計算張量的基本統計量和分佈特性。

Args:
    tensor (torch (stat_utils.py)
    calculate_effective_rank: 計算張量的有效秩

有效秩是一個衡量張量中有效信息維度的指標，通過奇異值的正規化熵來計算。

Args:
    tensor: 輸入張量，將被扁平化為2D張量進行分析
    
Returns:
    float: 有效秩值 (tensor_utils.py)
    calculate_moving_average: 計算移動平均值

Args:
    values: 輸入數值列表或numpy數組
    window_size: 窗口大小
    
Returns:
    np (stat_utils.py)
    calculate_sparsity: 計算張量的稀疏度（零值或接近零值的比例）

Args:
    tensor: 輸入張量
    threshold: 視為零的閾值
    
Returns:
    float: 稀疏度 (0 (tensor_utils.py)
    calculate_stability_metrics: 計算序列的穩定性指標。

Args:
    values: 數值序列（列表或numpy數組）
    window_size: 計算局部穩定性的窗口大小

Returns:
    dict: 包含穩定性指標的字典。
          {'overall_std', 'last_window_std', 'relative_std', 'last_window_mean'} (stat_utils.py)
    compare_distributions: 比較兩個分佈的相似度

Args:
    dist1: 第一個分佈的張量或數組
    dist2: 第二個分佈的張量或數組
    method: 比較方法，可選 'ks'（Kolmogorov-Smirnov檢驗）或 'wasserstein'（Wasserstein距離）
    
Returns:
    Dict[str, Any]: 包含比較結果的字典，包括檢驗統計量、p值或距離 (stat_utils.py)
    compare_distributions: 比較兩個分佈是否顯著不同。

Args:
    data1 (np (stat_utils.py)
    deep_merge: 深度合併兩個字典

Args:
    dict1: 第一個字典
    dict2: 第二個字典
    
Returns:
    Dict[str, Any]: 合併後的字典 (file_utils.py)
    detect_distribution_type: 根據統計量判斷張量的分佈類型。

Args:
    tensor (torch (stat_utils.py)
    detect_distribution_type_advanced: 使用高階統計量和機器學習啟發的方法來判斷張量的分佈類型。

此函數是對基本detect_distribution_type的增強版本，增加了對以下項目的檢測：
- 更準確的稀疏性檢測
- 雙峰分佈檢測
- 基於熵的分佈區分
- 重尾分佈檢測
- 多模態分佈檢測

Args:
    tensor (torch (stat_utils.py)
    detect_outliers: 使用指定方法檢測數值序列中的異常值。

Args:
    data (np (stat_utils.py)
    detect_overfitting_simple: 檢測訓練中是否出現過擬合現象，通過對比訓練損失和驗證損失的趨勢。

Args:
    train_loss (List[float] or np (stat_utils.py)
    detect_plateaus: 檢測序列中的平台期（值變化非常小）。

Args:
    values (list or np (stat_utils.py)
    ensure_dir: 確保目錄存在，若不存在則創建

Args:
    directory: 目錄路徑
    
Returns:
    str: 創建或確認的目錄路徑 (file_utils.py)
    extract_feature_maps: 從激活值中提取最活躍的特徵圖

Args:
    activation: 激活值張量
    top_k: 需要提取的特徵圖數量
    
Returns:
    Tuple[torch (tensor_utils.py)
    find_convergence_point: 查找序列的收斂點。
收斂定義為：在移動平均窗口內，值的相對變化率持續低於閾值。

Args:
    values: 數值序列（列表或numpy數組）
    window_size: 計算移動平均和變化率的窗口大小
    threshold: 判斷收斂的相對變化率閾值

Returns:
    int or None: 收斂點的索引（第一個滿足條件的點）。如果未收斂則返回 None。 (stat_utils.py)
    find_files_recursive: 遞歸查找符合模式的文件

Args:
    directory: 起始目錄
    pattern: 文件名模式，例如 '* (file_utils.py)
    flatten_tensor: 將張量展平為一維向量

Args:
    tensor: 輸入張量
    
Returns:
    一維展平後的張量 (tensor_utils.py)
    framework_agnostic_to_numpy: Attempts to convert a tensor (PyTorch, TensorFlow) to a NumPy array (helpers.py)
    get_file_size: 獲取文件大小

Args:
    file_path: 文件路徑
    unit: 單位，可選 'B', 'KB', 'MB', 'GB'
    
Returns:
    float: 文件大小

Raises:
    FileNotFoundError: 如果文件不存在 (file_utils.py)
    get_files_by_extension: 獲取指定目錄下特定副檔名的文件

Args:
    directory: 目錄路徑
    extension: 文件副檔名，例如 ' (file_utils.py)
    get_most_recent_file: 獲取指定目錄下最近修改的符合模式的文件

Args:
    directory: 目錄路徑
    pattern: 文件名模式，例如 '* (file_utils.py)
    get_tensor_statistics: 計算張量的基礎統計指標

Args:
    tensor: 輸入張量
    
Returns:
    Dict[str, float]: 包含平均值、標準差、最小值、最大值等統計信息的字典 (tensor_utils.py)
    handle_nan_values: 處理張量中的NaN值

Args:
    tensor: 輸入張量
    strategy: 處理策略，可選 'zero'(替換為0), 'mean'(替換為均值), 'median'(替換為中位數), 
             'fill'(替換為指定值), 'interpolate'(使用插值)
    fill_value: 當strategy為'fill'時使用的填充值
    
Returns:
    處理後的張量，與輸入類型相同 (tensor_utils.py)
    handle_sparse_tensor: 處理稀疏張量

Args:
    tensor: 輸入張量
    threshold: 視為零的閾值
    strategy: 處理策略，可選 'keep'(保持原樣), 'remove_zeros'(移除零), 
              'compress'(稀疏表示)
    
Returns:
    處理後的張量或稀疏表示 (tensor_utils.py)
    list_subdirectories: 列出指定目錄下的所有子目錄

Args:
    directory: 父目錄路徑
    
Returns:
    List[str]: 子目錄路徑列表 (file_utils.py)
    load_json_file: 載入 JSON 檔案

Args:
    file_path: JSON 檔案路徑
    
Returns:
    Dict[str, Any]: 解析後的 JSON 內容

Raises:
    FileNotFoundError: 如果檔案不存在
    json (file_utils.py)
    load_pickle_file: 載入 pickle 檔案

Args:
    file_path: pickle 檔案路徑
    
Returns:
    Any: 解析後的 pickle 內容

Raises:
    FileNotFoundError: 如果檔案不存在 (file_utils.py)
    merge_json_files: 合併多個 JSON 文件的內容

Args:
    file_paths: JSON 文件路徑列表
    
Returns:
    Dict[str, Any]: 合併後的 JSON 內容 (file_utils.py)
    normalize_tensor: 正規化張量數據

Args:
    tensor: 輸入張量
    method: 正規化方法，可選 'minmax'/'min_max', 'zscore'/'z_score', 'l1', 'l2'
    dim: 應用正規化的維度，None表示整個張量
    
Returns:
    與輸入相同類型的正規化後張量 (tensor_utils.py)
    normalize_tensor_zscore: 使用Z分數方法正規化張量

Args:
    tensor: 輸入張量
    dim: 應用正規化的維度，None表示整個張量
    
Returns:
    與輸入相同類型的Z分數正規化後張量 (tensor_utils.py)
    perform_statistical_test: 執行統計檢驗比較兩個樣本

Args:
    sample1: 第一個樣本的張量或數組
    sample2: 第二個樣本的張量或數組
    test_type: 檢驗類型，可選 't_test', 'mann_whitney', 'kruskal'
    
Returns:
    Dict[str, Any]: 包含檢驗結果的字典，包括檢驗統計量、p值和結論 (stat_utils.py)
    preprocess_activations: 預處理激活值數據

Args:
    activations: 輸入激活值張量
    flatten: 是否將激活值展平
    normalize: 正規化方法，None表示不進行正規化
    remove_outliers: 是否移除離群值
    outlier_threshold: 離群值閾值（標準差的倍數）
    
Returns:
    torch (tensor_utils.py)
    reduce_dimensions: 對激活值進行維度縮減

Args:
    activation: 輸入張量，可以是PyTorch張量或NumPy陣列
    method: 縮減方法，可選 'mean', 'max', 'sum', 'flatten'
    dims: 需要縮減的維度列表，如果為None則根據activation的維度自動決定
    
Returns:
    縮減後的張量，與輸入類型相同 (tensor_utils.py)
    sample_tensor: 從張量中隨機抽樣，用於處理大型張量。

Args:
    tensor: 輸入張量
    max_samples: 最大樣本數量
    
Returns:
    抽樣後的張量，保持原始類型 (tensor_utils.py)
    save_json_file: 將數據保存為 JSON 檔案

Args:
    data: 要保存的數據
    file_path: 目標檔案路徑
    indent: JSON 縮排空格數 (file_utils.py)
    save_pickle_file: 將數據保存為 pickle 檔案

Args:
    data: 要保存的數據
    file_path: 目標檔案路徑 (file_utils.py)
    tensor_correlation: 計算兩個張量之間的皮爾森相關係數

Args:
    tensor1: 第一個張量
    tensor2: 第二個張量
    
Returns:
    float: 相關係數 (tensor_utils.py)
    to_numpy: 將張量轉換為NumPy陣列

Args:
    tensor: 輸入張量，可以是PyTorch張量、NumPy陣列或列表
    
Returns:
    np (tensor_utils.py)


完整的模組索引請參見：FUNCTION_CLASS_INDEX.md
"""
