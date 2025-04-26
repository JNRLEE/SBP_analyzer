# SBP Analyzer 函數與類別索引

此文件自動生成，包含專案中所有函數和類別的索引。

## 目錄結構

### __init__.py


### fix_imports.py

**Functions:**

- `create_init_files()`: 在項目目錄和子目錄中創建 __init__...
- `fix_python_path()`: 將項目根目錄添加到 Python 路徑。

Args:
    無

Returns:
    無...
- `verify_imports()`: 驗證常用模塊的導入。

Args:
    無

Returns:
    bool: 所有模塊導入是否成功...
- `fix_import_statements_in_file()`: 修復文件中的導入語句。

Args:
    file_path: 要修復的文件路徑
    
Returns:
    bool: 是否進行了更改...
- `fix_import_statements()`: 遍歷項目中的所有 Python 文件並修復導入語句。

Args:
    無
    
Returns:
    無...
- `ensure_flat_structure()`: 確保項目使用扁平結構，沒有嵌套的 sbp_analyzer 目錄。
如果發現嵌套目錄，會自動將其內容移動到項目根目錄。

Args:
    無
    
Returns:
    無...
- `main()`: 執行所有修復步驟。

Args:
    無
    
Returns:
    無...


### function_class_index_generator.py

**Classes:**

- `FunctionClassVisitor`: 訪問 Python 檔案的抽象語法樹，收集所有函數和類別的定義...
  - `visit_FunctionDef()`: 訪問函數定義節點...
  - `visit_ClassDef()`: 訪問類別定義節點...

**Functions:**

- `analyze_file()`: 分析單個 Python 檔案，提取所有函數和類別

Args:
    file_path: Python 檔案的路徑
    
Returns:
    包含檔案中所有函數和類別的字典...
- `analyze_directory()`: 遞迴分析目錄中的所有 Python 檔案

Args:
    root_dir: 根目錄路徑
    
Returns:
    包含所有檔案分析結果的嵌套字典...
- `generate_markdown_index()`: 從分析結果生成 Markdown 格式的索引表

Args:
    analysis_result: 分析結果字典
    base_path: 當前在索引表中的基礎路徑
    
Returns:
    Markdown 格式的索引表...
- `create_index_file()`: 創建索引文件

Args:
    root_dir: 根目錄路徑
    output_file: 輸出文件的路徑...


### setup.py


## analysis_results/

## analysis_results/audio_swin_regression_20250417_142912/

## analysis_results/audio_swin_regression_20250417_142912/activation_analysis/


## analysis_results/audio_swin_regression_20250417_142912/model_analysis/


## analysis_results/audio_swin_regression_20250417_142912/plots/


## analysis_results/audio_swin_regression_20250417_142912/report/


## analysis_results/audio_swin_regression_20250417_142912/training_analysis/


## analysis_results/training_history_good_training/


## analysis_results/training_history_overfitting/


## analysis_results/training_history_plateau/


## analysis_results/training_history_slow_convergence/


## analysis_results/training_history_unstable/


## analyzer/

### analyzer/__init__.py


### analyzer/adaptive_threshold_analyzer.py

**Classes:**

- `AdaptiveThresholdAnalyzer` (extends IntermediateDataAnalyzer): 自適應閾值分析器類

此分析器擴展了中間數據分析器的功能，通過分析每一層激活的分佈特性，
自適應地調整用於檢測死亡神經元和飽和神經元的閾值。支持多種自適應
方法，包括基於百分位數、分佈特性和混合方法。

Args:
    experiment_name: 實驗名稱
    output_dir: 輸出目錄路徑
    adaptive_method: 自適應方法，可選 'percentile'、'distribution' 或 'hybrid'
    adaptation_strength: 自適應強度，範圍 [0, 1]，越高表示越強的自適應調整
    base_dead_threshold: 基礎死亡神經元閾值，預設為 0...
  - `analyze()`: 分析模型權重資料夾中的中間層激活數據，使用自適應閾值

Args:
    model_dir: 模型資料夾路徑，包含中間層激活數據
    
Returns:
    Dict: 包含分析結果和自適應閾值的字典...
  - `analyze_with_loader()`: 使用給定的數據載入器分析中間層激活，使用自適應閾值

Args:
    data_loader: 數據載入器，必須提供 get_layer_names 和 load_activation_data 方法
    
Returns:
    Dict: 包含分析結果和自適應閾值的字典...
  - `_collect_layer_statistics()`: 收集所有層的激活統計數據

Args:
    data_loader: 數據載入器...
  - `_calculate_adaptive_thresholds()`: 根據統計數據計算每一層的自適應閾值...
  - `_percentile_based_dead_threshold()`: 基於百分位數計算死亡神經元閾值

Args:
    stats: 層的統計數據
    
Returns:
    float: 計算得到的閾值...
  - `_percentile_based_saturated_threshold()`: 基於百分位數計算飽和神經元閾值

Args:
    stats: 層的統計數據
    
Returns:
    float: 計算得到的閾值...
  - `_distribution_based_dead_threshold()`: 基於分佈特性計算死亡神經元閾值

Args:
    stats: 層的統計數據
    
Returns:
    float: 計算得到的閾值...
  - `_distribution_based_saturated_threshold()`: 基於分佈特性計算飽和神經元閾值

Args:
    stats: 層的統計數據
    
Returns:
    float: 計算得到的閾值...
  - `_analyze_layers_with_adaptive_thresholds()`: 使用自適應閾值分析所有層的激活數據

Args:
    data_loader: 數據載入器...
  - `_analyze_layer_with_thresholds()`: 使用給定閾值分析單個層的激活數據

Args:
    activations: 激活數據張量
    layer_name: 層名稱
    dead_threshold: 死亡神經元閾值
    saturated_threshold: 飽和神經元閾值
    
Returns:
    Dict: 分析結果字典...
  - `_save_results()`: 保存分析結果到 JSON 文件...
  - `_generate_report()`: 生成分析報告...
  - `_generate_threshold_comparison_plot()`: 生成閾值比較圖

Args:
    report_dir: 報告目錄路徑...
  - `_generate_detection_results_plot()`: 生成檢測結果圖

Args:
    report_dir: 報告目錄路徑...
  - `_generate_html_report()`: 生成 HTML 報告

Args:
    report_dir: 報告目錄路徑...


### analyzer/base_analyzer.py

**Classes:**

- `NumpyEncoder`: 專門處理NumPy類型的JSON編碼器...
  - `default()`
- `BaseAnalyzer` (extends ABC): 分析器的抽象基礎類別，提供通用接口和方法。

這個類別定義了所有分析器共有的屬性和方法。子類別應重寫抽象方法來實現特定的分析功能。

Attributes:
    experiment_dir (str): 實驗結果的目錄路徑。
    config (Dict): 實驗配置。
    logger (logging...
  - `load_config()`: 載入實驗配置。

Returns:
    Dict: 實驗配置字典。...
  - `analyze()`: 執行分析並返回結果。

這是一個抽象方法，必須由子類實現。

Returns:
    Dict: 分析結果字典。...
  - `save_results()`: 將分析結果保存到輸出目錄。

Args:
    output_dir (str): 保存結果的目錄路徑。...


### analyzer/inference_analyzer.py

**Classes:**

- `InferenceAnalyzer` (extends BaseAnalyzer): 推理分析器，用於分析模型推理性能和行為。

此分析器主要關注模型推理的時間效率、資源使用、輸出特性和準確度等方面。

Attributes:
    experiment_dir (str): 實驗結果的目錄路徑。
    inference_logs (Dict): 儲存推理日誌資料。
    model_structure (Dict): 模型結構資訊。
    results (Dict): 分析結果。
    device (str): 執行推理的設備。...
  - `load_inference_logs()`: 從 inference_logs...
  - `load_model_structure()`: 從 model_structure...
  - `analyze()`: 分析推理性能並計算相關統計資訊。

Args:
    batch_sizes (List[int], optional): 要分析的批次大小列表。
    input_shapes (List[List[int]], optional): 要分析的輸入形狀列表。
    save_results (bool, optional): 是否保存分析結果。默認為True。
    output_dir (str, optional): 保存結果的目錄。如果為None則使用experiment_dir。

Returns:
    Dict: 包含分析結果的字典。...
  - `_convert_numpy_types()`: 將結果字典中所有numpy類型轉換為Python內建類型，避免JSON序列化問題。...
  - `convert_item()`
  - `_analyze_time()`: 分析模型推理時間相關的效能指標。...
  - `_calculate_time_stats()`: 計算時間數據的統計指標。

Args:
    time_values (List[float]): 時間值列表。
    
Returns:
    Dict[str, float]: 包含各種統計指標的字典。...
  - `_generate_latency_optimization_tips()`: 根據延遲分析生成優化建議。...
  - `_analyze_throughput()`: 分析推理吞吐量。...
  - `_analyze_memory()`: 分析模型記憶體使用情況。...
  - `_generate_memory_optimization_tips()`: 根據記憶體使用分析生成優化建議。...
  - `_analyze_output_statistics()`: 分析模型輸出統計資訊。...
  - `_analyze_batch_size_impact()`: 分析不同批次大小對推理性能的影響。

Args:
    batch_sizes (List[int]): 要分析的批次大小列表。...
  - `_analyze_input_shape_impact()`: 分析輸入形狀對推理性能的影響。

Args:
    input_shapes (List[List[int]]): 要分析的輸入形狀列表。...
  - `_analyze_accuracy()`: 分析模型推理的準確性。...
  - `_analyze_resource_efficiency()`: 分析模型的資源效率，包括記憶體和計算效率。...
  - `_has_multiple_hardware_data()`: 檢查是否有多種硬體的推理數據。

Returns:
    bool: 如果有多種硬體數據，則返回True，否則返回False。...
  - `_analyze_hardware_comparison()`: 比較不同硬體上的推理性能。...
  - `get_time_performance()`: 獲取時間性能分析結果。

Returns:
    Dict: 包含時間性能分析的字典。...
  - `get_memory_usage()`: 獲取記憶體使用分析結果。

Returns:
    Dict: 包含記憶體使用分析的字典。...
  - `get_accuracy_analysis()`: 獲取準確性分析結果。

Returns:
    Dict: 包含準確性分析的字典。...
  - `get_resource_efficiency()`: 獲取資源使用效率分析結果。

Returns:
    Dict: 包含資源使用效率分析的字典。...
  - `save_results()`: 將分析結果保存到輸出目錄。

Args:
    output_dir (Optional[str]): 保存結果的目錄路徑。如果為None則使用experiment_dir。...
  - `_calculate_scaling_efficiency()`: 計算擴展效率。

Args:
    batch_sizes (List[int]): 要分析的批次大小列表。
    throughput_by_batch (Dict[int, float]): 不同批次大小的吞吐量字典。

Returns:
    Dict[str, float]: 擴展效率字典，使用字符串鍵。...


### analyzer/intermediate_data_analyzer.py

**Classes:**

- `IntermediateDataAnalyzer` (extends BaseAnalyzer): 中間層數據分析器，用於分析模型中間層輸出的特性。

此分析器關注模型訓練過程中通過hooks獲取的中間層數據，分析其分布、統計特性和變化趨勢。

Attributes:
    experiment_dir (str): 實驗結果的目錄路徑。
    hooks_dir (str): 存放hooks數據的目錄。
    available_epochs (List[int]): 可用的輪次列表。
    available_layers (Dict[int, List[str]]): 每個輪次可用的層列表。
    experiment_name (str): 實驗名稱。
    output_dir (str): 輸出目錄路徑。
    dead_neuron_threshold (float): 死亡神經元的閾值。
    saturated_neuron_threshold (float): 飽和神經元的閾值。
    layer_results (Dict): 依照層名稱組織的分析結果。...
  - `_scan_available_data()`: 掃描並記錄可用的輪次和層。...
  - `load_activation()`: 載入特定層、特定輪次、特定批次的激活值。

Args:
    layer_name (str): 層的名稱。
    epoch (int): 輪次。
    batch (int, optional): 批次索引。默認為0。

Returns:
    Optional[torch...
  - `analyze_with_loader()`: 使用提供的 HookDataLoader 分析中間層數據

此方法允許使用外部提供的數據加載器進行分析，便於測試和模擬數據分析。

Args:
    hook_loader: 用於加載數據的HookDataLoader實例
    epochs: 要分析的輪次列表，如果為None則分析所有可用輪次
    layers: 要分析的層名稱列表，如果為None則分析所有可用層
    batch_idx: 要分析的批次索引，預設為0
    preprocess_args: 預處理參數，可包含normalize, flatten, remove_outliers等選項
    
Returns:
    Dict: 分析結果摘要...
  - `analyze()`: 分析中間層數據

Args:
    epochs: 要分析的輪次列表，如果為None則分析所有可用輪次
    layers: 要分析的層名稱列表，如果為None則分析所有可用層
    batch_idx: 要分析的批次索引，預設為0
    preprocess_args: 預處理參數，可包含normalize, flatten, remove_outliers等選項
    
Returns:
    Dict: 分析結果摘要...
  - `_analyze_layer_activation()`: 分析單個層的激活值

Args:
    activation: 層的激活值張量
    layer_name: 層名稱
    epoch: 輪次編號
    
Returns:
    Dict: 層激活值分析結果...
  - `_analyze_layer_relationships()`: 分析不同層之間的關係

Args:
    epochs: 要分析的輪次列表
    layers: 要分析的層列表
    batch_idx: 批次索引...
  - `_analyze_activation_dynamics()`: 分析層激活值隨時間變化的動態特性

Args:
    epochs: 要分析的輪次列表
    layers: 要分析的層列表
    batch_idx: 批次索引...
  - `get_layer_statistics()`: 獲取指定層和輪次的統計分析結果

Args:
    layer_name: 層名稱
    epoch: 輪次編號
    
Returns:
    Optional[Dict]: 層統計信息，如果不存在則返回None...
  - `get_layer_dynamics()`: 獲取指定層的激活動態分析結果

Args:
    layer_name: 層名稱
    
Returns:
    Optional[Dict]: 層動態信息，如果不存在則返回None...
  - `get_layer_relationship()`: 獲取兩個層之間的關係分析結果

Args:
    layer1: 第一個層的名稱
    layer2: 第二個層的名稱
    epoch: 輪次編號
    
Returns:
    Optional[Dict]: 層關係信息，如果不存在則返回None...
  - `get_problematic_layers()`: 檢測模型中存在問題的層。

分析所有層的統計數據，找出符合問題標準的層，如死亡神經元比例高、
飽和神經元比例高、過大或過小的激活值等。

Args:
    criteria (Dict[str, float], optional): 問題判斷標準，包含閾值。
        可包含 'dead_neuron_threshold', 'saturated_neuron_threshold',
        'low_variance_threshold' 等。如果為None，則使用預設標準。
    
Returns:
    List[Dict]: 每個問題層的信息，包含層名稱、問題類型、問題指標值等。...
  - `get_summary_dataframe()`: 生成包含所有層統計信息的摘要數據框。

將所有層的統計數據整合到一個pandas DataFrame中，方便查詢和可視化。
數據框包含層名稱、輪次、基本統計量（平均值、標準差等）、稀疏度、有效秩等信息。

Returns:
    pd...
  - `generate_report()`: 生成分析報告

Returns:
    Dict: 包含所有分析結果的報告字典...


### analyzer/layer_correlation_analysis.py

**Classes:**

- `LayerCorrelationAnalysis`: 神經網絡層內神經元相關性分析類

此類提供方法分析神經網絡層內神經元之間的相關性模式，包括相關性矩陣計算、
神經元群聚識別、主成分分析等。...
  - `compute_neuron_correlation_matrix()`: 計算層內神經元之間的相關性矩陣

Args:
    activations: 層的激活值，形狀為[batch_size, neurons]或[batch_size, channels, height, width]
    method: 相關性計算方法，'pearson'或'spearman'
    remove_outliers: 是否移除離群值
    z_threshold: 用於識別離群值的z分數閾值
    
Returns:
    Dict: 包含相關性矩陣和統計特徵的字典...
  - `identify_neuron_clusters()`: 基於相關性矩陣識別神經元群組

Args:
    correlation_matrix: 神經元間的相關性矩陣
    method: 聚類方法，'kmeans'或'dbscan'
    n_clusters: 聚類數量(kmeans)或鄰居距離閾值(dbscan)
    
Returns:
    Dict: 包含聚類結果的字典...
  - `compute_principal_components()`: 對層的激活值進行主成分分析

Args:
    activations: 層的激活值，形狀為[batch_size, neurons]或[batch_size, channels, height, width]
    n_components: 要計算的主成分數量
    
Returns:
    Dict: 包含PCA結果的字典...
  - `analyze_neuron_correlations()`: 綜合分析層內神經元的相關性

Args:
    activations: 層的激活值
    detailed: 是否進行詳細分析，包括聚類和PCA
    
Returns:
    Dict: 分析結果...
  - `compute_activation_coordination()`: 計算層內神經元的激活協調程度

Args:
    activations: 層的激活值
    
Returns:
    Dict: 協調性指標...
  - `detect_functional_groups()`: 在相關性矩陣中檢測功能群組（高度相關的神經元子集）

Args:
    correlation_matrix: 神經元相關性矩陣
    threshold: 判定為功能群組的相關性閾值
    
Returns:
    Dict: 映射群組ID到神經元索引的字典...


### analyzer/layer_relationship_analysis.py

**Classes:**

- `LayerRelationshipAnalysis`: 神經網絡層間關係分析類

此類提供方法分析神經網絡中不同層之間的關係，包括相關性、
信息流動、表示相似度等。...
  - `compute_activation_correlation()`: 計算兩個層的激活值之間的相關性

Args:
    layer1_activations: 第一個層的激活值，形狀為[batch_size, features1]
    layer2_activations: 第二個層的激活值，形狀為[batch_size, features2]
    sample_size: 用於計算的樣本數量，若小於實際樣本數則隨機抽樣
    
Returns:
    Dict: 包含不同相關性度量的字典...
  - `_compute_cka()`: 計算中心核心對齊相似度 (Centered Kernel Alignment)

Args:
    X: 第一個層的激活值，形狀為[batch_size, features1]
    Y: 第二個層的激活值，形狀為[batch_size, features2]
    
Returns:
    float: CKA相似度，範圍[0, 1]...
  - `_compute_canonical_correlation()`: 計算兩組變量之間的典型相關係數 (第一典型相關)

Args:
    X: 第一個層的激活值，形狀為[batch_size, features1]
    Y: 第二個層的激活值，形狀為[batch_size, features2]
    
Returns:
    float: 典型相關係數，範圍[-1, 1]...
  - `_compute_cosine_similarity()`: 計算兩個層激活平均向量的餘弦相似度

Args:
    X: 第一個層的激活值，形狀為[batch_size, features1]
    Y: 第二個層的激活值，形狀為[batch_size, features2]
    
Returns:
    float: 餘弦相似度，範圍[-1, 1]...
  - `trace_activation_flow()`: 追踪激活值如何在層間流動，使用相鄰層之間的相關程度

Args:
    layer_activations: 層名到激活值的映射
    layer_order: 層的順序列表，應該對應於網絡中的前向順序
    
Returns:
    Dict: 每層到相鄰層的信息流指標...
  - `find_information_bottlenecks()`: 基於相鄰層之間的信息流變化找出網絡中的信息瓶頸

Args:
    layer_activations: 層名到激活值的映射
    layer_order: 層的順序列表，應該對應於網絡中的前向順序
    
Returns:
    Dict: 每層的瓶頸分數，分數越高表示該層可能是瓶頸...
  - `_compute_layer_entropy()`: 計算層激活值的熵...
  - `detect_influence_patterns()`: 檢測層之間的影響模式，包括長距離依賴關係

Args:
    layer_activations: 層名到激活值的映射
    layer_order: 層的順序列表
    influence_window: 考慮影響的最大層距離
    
Returns:
    Dict: 層之間的影響強度...
  - `compute_representation_similarity()`: 計算兩組模型在相同層上的表示相似度

Args:
    activations1: 第一個模型的層激活值
    activations2: 第二個模型的層激活值
    layer_names: 要比較的層名列表
    
Returns:
    Dict: 每層的表示相似度...
  - `detect_coactivation_patterns()`: 檢測層之間的共同激活模式

Args:
    layer_activations: 層名到激活值的映射
    layer_names: 層名列表
    threshold: 判定為共同激活的相關閾值
    
Returns:
    Dict: 層對之間的共同激活強度...
  - `analyze_layer_relationships()`: 綜合分析層間關係

Args:
    layer_activations: 層名到激活值的映射
    layer_order: 層的順序列表
    
Returns:
    Dict: 各種層間關係分析結果...
  - `compute_gradual_distortion()`: 計算信息在網絡中逐漸變形的程度

Args:
    layer_activations: 層名到激活值的映射
    layer_order: 層的順序列表
    
Returns:
    Dict: 每層到輸入層的變形程度...
  - `_flatten_activations()`: 將激活值展平為2D張量 [batch_size, features]

Args:
    activations: 輸入激活值，可以是任意維度
    
Returns:
    torch...


### analyzer/model_structure_analyzer.py

**Classes:**

- `ModelStructureAnalyzer` (extends BaseAnalyzer): 模型結構分析器，用於分析模型結構和參數分布。

此分析器主要關注模型架構、層級結構、參數分布和其他結構特性。

Attributes:
    experiment_dir (str): 實驗結果的目錄路徑。
    model_structure (Dict): 儲存解析後的模型結構。
    model_stats (Dict): 儲存模型統計信息。...
  - `load_model_structure()`: 從model_structure...
  - `analyze()`: 分析模型結構並計算相關統計信息。

Args:
    save_results (bool, optional): 是否保存分析結果。默認為True。

Returns:
    Dict: 包含分析結果的字典。...
  - `_analyze_basic_info()`: 分析模型的基本信息。...
  - `_identify_architecture_type()`: 識別模型的架構類型 (CNN, RNN, Transformer等)。...
  - `_analyze_layer_hierarchy()`: 分析層級結構關係，計算模型深度。...
  - `dfs()`
  - `_analyze_parameter_distribution()`: 分析模型參數分布情況。...
  - `_analyze_weight_distributions()`: 分析模型權重分布情況。

注意：在實際使用時，這個函數可能需要訪問模型權重文件，
目前僅實現一個基於層類型的理論權重分布估計。...
  - `_analyze_layer_complexity()`: 分析層級複雜度，包括參數量、計算複雜度、和IO複雜度。...
  - `_estimate_flops()`: 估計層的計算複雜度（FLOPs）。

Args:
    layer (Dict): 層的信息字典。
    
Returns:
    Optional[int]: 估計的FLOPs，如果無法估計則為None。...
  - `_identify_sparse_layers()`: 識別模型中的稀疏層。

Args:
    threshold (float, optional): 稀疏度閾值，默認為0...
  - `_identify_largest_layers()`: 識別模型中參數最多的層。

Args:
    top_n (int, optional): 返回的層數量，默認為5。...
  - `_analyze_sequential_paths()`: 分析模型中的序列路徑，找出主要的數據流路徑。...
  - `dfs_paths()`
  - `_calculate_gini_coefficient()`: 計算Gini係數，用於衡量參數分布的不均勻程度。

Args:
    values (List[float]): 一組數值。
    
Returns:
    float: Gini係數，範圍從0（完全平均）到1（完全不平均）。...
  - `_calculate_parameter_efficiency()`: 計算模型的參數效率，即每個參數的效用。

這裡僅執行一個簡單的估計。在實際應用中，可能需要結合模型性能指標。...
  - `_analyze_layer_connectivity()`: 分析層間連接性，計算圖的相關特性。...
  - `_analyze_computational_complexity()`: 分析計算複雜度，估計模型的理論計算量。...
  - `get_layer_hierarchy()`: 獲取層級結構信息。

Returns:
    Dict: 包含層級結構的字典。...
  - `get_parameter_distribution()`: 獲取參數分布信息。

Returns:
    Dict: 包含參數分布的字典。...
  - `get_model_summary()`: 獲取模型摘要信息。

Returns:
    Dict: 包含模型摘要的字典。...


### analyzer/training_dynamics_analyzer.py

**Classes:**

- `TrainingDynamicsAnalyzer` (extends BaseAnalyzer): 訓練動態分析器，用於分析模型訓練過程中的性能指標變化。

此分析器關注訓練過程中損失函數、評估指標等數據的變化趨勢，可用於評估模型收斂性、穩定性等。

Attributes:
    experiment_dir (str): 實驗結果的目錄路徑。
    training_history (Dict): 訓練歷史記錄。
    metrics_data (pd...
  - `load_training_history()`: 從training_history...
  - `preprocess_metrics()`: 預處理訓練指標數據，轉換為易於分析的格式，並處理常見問題。

處理邏輯:
- 如果歷史記錄為空，返回空 DataFrame 並記錄。
- 檢查指標列表長度是否一致，如果不一致，截斷至最短長度並記錄警告。
- 檢查是否存在 'epoch' 鍵，如果不存在，自動生成並記錄警告。
- 檢查是否存在 NaN 值，並記錄警告。

Returns:
    pd...
  - `analyze()`: 執行所有訓練動態分析步驟。

分析內容:
- 收斂性 (Convergence)
- 穩定性 (Stability)
- 過擬合 (Overfitting)
- 指標相關性 (Metric Correlations)
- 學習效率 (Learning Efficiency)
- 異常檢測 (Anomaly Detection)

Args:
    save_results (bool): 是否將結果保存到文件。
    convergence_params (Dict[str, Any], optional): 傳遞給 analyze_convergence 的參數。
    stability_params (Dict[str, Any], optional): 傳遞給 analyze_stability 的參數。
    overfitting_params (Dict[str, Any], optional): 傳遞給 analyze_overfitting 的參數。

Returns:
    Dict: 包含所有分析結果的字典。...
  - `_analyze_loss_curves()`: Analyzes training and validation loss curves...
  - `_analyze_metric_curves()`: Analyzes training and validation metric curves...
  - `_analyze_other_metric()`: 分析其他指標曲線。

Args:
    metric (str): 指標名稱。
    val_metric (Optional[str], optional): 對應的驗證指標名稱。默認為None。...
  - `_analyze_learning_efficiency()`: 分析學習效率和學習率調度的影響。

Args:
    loss_metric (str): 損失指標名稱。
    lr_metric (str): 學習率指標名稱。...
  - `_detect_anomalies()`: 檢測訓練過程中的異常。

Args:
    loss_metric (str): 損失指標名稱。
    val_loss_metric (Optional[str], optional): 驗證損失指標名稱。默認為None。...
  - `_calculate_metric_correlations()`: 計算指標之間的相關性矩陣。...
  - `get_summary()`: 生成分析結果的摘要字典。

Returns:
    Dict[str, Any]: 包含分析摘要的字典。...
  - `get_metric_correlation()`: 獲取指標間的相關性矩陣。

Returns:
    pd...
  - `analyze_convergence()`: Analyzes the convergence of a specific metric...
  - `analyze_stability()`: Analyzes the stability of a specific metric...
  - `analyze_overfitting()`: Analyzes overfitting based on training and validation loss metrics...
  - `plot_loss_curves()`: 繪製訓練和驗證損失曲線。...
  - `plot_metric_curves()`: 繪製指定指標的訓練和驗證曲線。...
  - `find_best_epoch()`: Finds the best epoch based on a given metric and criteria...


## data_loader/

### data_loader/__init__.py


### data_loader/base_loader.py

**Classes:**

- `BaseLoader` (extends ABC): 數據載入器的抽象基礎類別，提供通用接口和方法。

這個類別定義了所有數據載入器共有的屬性和方法。子類別應重寫抽象方法來實現特定的數據載入功能。

Attributes:
    experiment_dir (str): 實驗結果的目錄路徑。
    logger (logging...
  - `load()`: 載入數據並返回結果。

這是一個抽象方法，必須由子類實現。

Returns:
    Any: 載入的數據。...
  - `_check_file_exists()`: 檢查文件是否存在。

Args:
    file_path (str): 文件路徑。

Returns:
    bool: 文件是否存在。...
  - `_load_json()`: 載入JSON文件。

Args:
    file_path (str): JSON文件路徑。

Returns:
    Optional[Dict]: 載入的JSON數據，如果文件不存在或無法解析則返回None。...


### data_loader/experiment_loader.py

**Classes:**

- `ExperimentLoader` (extends BaseLoader): 實驗數據載入器，用於載入實驗的基本信息和訓練結果。

此載入器可以載入實驗目錄中的config...
  - `load()`: 載入實驗數據。

Args:
    load_config (bool, optional): 是否載入配置。默認為True。
    load_model_structure (bool, optional): 是否載入模型結構。默認為True。
    load_training_history (bool, optional): 是否載入訓練歷史。默認為True。
    load_results (bool, optional): 是否載入結果摘要。默認為True。

Returns:
    Dict: 包含載入數據的字典。...
  - `load_config()`: 載入config...
  - `load_model_structure()`: 載入model_structure...
  - `load_training_history()`: 載入training_history...
  - `load_results()`: 載入results/results...
  - `get_experiment_info()`: 獲取實驗的基本信息。

Returns:
    Dict: 包含實驗信息的字典。...


### data_loader/hook_data_loader.py

**Classes:**

- `HookDataLoader` (extends BaseLoader): Hook數據載入器，用於載入模型中間層的激活值等數據。

此載入器可以載入hooks/目錄下的...
  - `_scan_available_data()`: 掃描並記錄可用的輪次、批次和層。...
  - `load()`: 載入hook數據。

Args:
    layer_name (str, optional): 要載入的層名稱。如果為None，則返回所有可用層的信息。
    epoch (int, optional): 要載入的輪次。如果為None，則返回所有可用輪次的信息。
    batch (int, optional): 要載入的批次索引。默認為0。

Returns:
    Dict[str, Any]: 包含載入數據的字典。...
  - `load_training_summary()`: 載入整體訓練摘要

Returns:
    包含訓練摘要資訊的字典，若檔案不存在則返回None...
  - `load_evaluation_results()`: 載入特定資料集的評估結果

Args:
    dataset: 資料集名稱，預設為'test'
    
Returns:
    包含評估結果的字典，若檔案不存在則返回None...
  - `load_epoch_summary()`: 載入特定輪次的摘要

Args:
    epoch: 輪次編號
    
Returns:
    包含輪次摘要的字典，若檔案不存在則返回None
    
Raises:
    FileNotFoundError: 當指定的輪次目錄不存在時。...
  - `load_batch_data()`: 載入特定批次的資料

Args:
    epoch: 輪次編號
    batch: 批次編號
    
Returns:
    包含批次資料的字典，若檔案不存在則返回None...
  - `load_layer_activation()`: 載入指定層在特定輪次和批次的激活值。

Args:
    layer_name (str): 層名稱。
    epoch (int): 輪次編號。
    batch (int): 批次索引，默認為0。

Returns:
    Optional[torch...
  - `list_available_epochs()`: 返回可用的輪次列表。...
  - `list_available_batches()`: 返回指定輪次可用的批次索引列表。

Args:
    epoch (int): 輪次編號。

Returns:
    List[int]: 可用的批次索引列表。...
  - `list_available_layers()`: 返回指定輪次和批次可用的層名稱列表。

Args:
    epoch (int): 輪次編號。
    batch (int): 批次索引。

Returns:
    List[str]: 可用的層名稱列表。...
  - `list_available_layer_activations()`: [兼容性] 返回指定輪次第一個可用批次的層列表。建議使用 list_available_layers。...
  - `get_activation_shape()`: 獲取特定層激活值的形狀

Args:
    layer_name: 層的名稱
    epoch: 輪次編號
    batch: 批次編號，預設為0
    
Returns:
    激活值張量的形狀，若無法載入則返回None...
  - `load_all_epoch_activations()`: 載入特定層在所有可用輪次中的激活值

Args:
    layer_name: 層的名稱
    batch: 批次編號，預設為0
    
Returns:
    以輪次為鍵，激活值張量為值的字典...
  - `load_all_layer_activations_for_epoch()`: 載入特定輪次中所有層的激活值

Args:
    epoch: 輪次編號
    batch: 批次編號，預設為0
    
Returns:
    以層名稱為鍵，激活值張量為值的字典...
  - `get_experiment_summary()`: 獲取實驗摘要，包含hooks資料的概覽

Returns:
    包含實驗摘要的字典...
  - `load_activation()`: 載入特定層在特定輪次和批次的激活值。

Args:
    layer_name (str): 層的名稱。
    epoch (int): 輪次。
    batch (int, optional): 批次索引。默認為0。

Returns:
    Optional[torch...
  - `load_activation_with_preprocessing()`: 載入並預處理激活值數據。

Args:
    layer_name (str): 層的名稱。
    epoch (int): 輪次。
    batch (int, optional): 批次索引。默認為0。
    preprocess_config (Optional[Dict[str, Any]], optional): 預處理配置。
        可能的選項包括：
        - normalize: 正規化方法 ('minmax', 'zscore', 'robust')
        - handle_outliers: 是否處理離群值
        - reduce_dims: 降維方法
        - flatten: 是否將多維張量展平為2D
        - remove_outliers: 是否移除離群值
        - outlier_threshold: 離群值閾值
        
Returns:
    Optional[torch...
  - `load_activations_batch()`: 批量載入多個層和多個輪次的激活值。

Args:
    layer_names (List[str]): 層名稱列表。
    epochs (List[int]): 輪次列表。
    batch (int, optional): 批次索引。默認為0。
    preprocess_config (Optional[Dict[str, Any]], optional): 預處理配置。
        
Returns:
    Dict[str, Dict[int, torch...
  - `load_layer_statistics()`: 載入層統計數據，如果不存在則計算基本統計信息。

Args:
    layer_name (str): 層的名稱。
    epoch (int): 輪次。
    batch (int, optional): 批次索引。默認為0。
    
Returns:
    Optional[Dict[str, Any]]: 層統計數據字典。
    
Raises:
    FileNotFoundError: 當指定的層或輪次不存在時。...
  - `get_activation_metadata()`: 獲取所有可用激活值數據的元數據。

Returns:
    Dict[str, Any]: 包含元數據的字典。...
  - `save_processed_activation()`: 保存處理後的激活值數據。

Args:
    activation (torch...
  - `load_hook_data()`: 載入指定層和輪次的所有hook數據。

這個方法會載入多個層和輪次的激活值數據，作為analyzer_interface中analyze方法的輸入。

Args:
    epochs (Optional[List[int]], optional): 要載入的輪次列表。如果為None，載入所有可用輪次。
    layers (Optional[List[str]], optional): 要載入的層列表。如果為None，載入所有可用層。
    batch_idx (int, optional): 要載入的批次索引。默認為0。
    
Returns:
    Dict[str, Any]: 包含所有載入的數據的字典，結構為:
        {
            'metadata': {
                'epochs': [載入的輪次列表],
                'layers': [載入的層列表],
                'batch_idx': 批次索引
            },
            'activations': {
                'layer_name1': {
                    epoch1: tensor1,
                    epoch2: tensor2,
                    ...
  - `load_all_layer_activations()`: 載入特定輪次和批次的所有層激活值。

Args:
    epoch (int): 輪次編號。
    batch (int, optional): 批次索引。默認為0。
    
Returns:
    Dict[str, torch...
  - `load_layer_activations_across_epochs()`: 載入特定層在多個輪次的激活值。

Args:
    layer_name (str): 層名稱。
    epochs (List[int]): 輪次列表。
    batch (int, optional): 批次索引。默認為0。
    
Returns:
    Dict[int, torch...
  - `load_all_activations_across_epochs()`: 載入所有層在多個輪次的激活值。

Args:
    epochs (List[int]): 輪次列表。
    batch (int, optional): 批次索引。默認為0。
    
Returns:
    Dict[str, Dict[int, torch...


## examples/

### examples/__init__.py


### examples/analyze_hook_data.py

**Functions:**

- `analyze_layer_distributions()`: 分析層激活值分布及其在訓練過程中的變化

Args:
    experiment_dir: 實驗結果目錄
    layer_names: 要分析的層名稱列表
    epochs: 要分析的輪次列表
    output_dir: 輸出目錄...
- `analyze_layer_sparsity_and_saturation()`: 分析層的稀疏度和飽和度

Args:
    experiment_dir: 實驗結果目錄
    layer_names: 要分析的層名稱列表
    epochs: 要分析的輪次列表
    output_dir: 輸出目錄...
- `analyze_nan_and_infinity()`: 檢測並分析層激活值中的NaN和無限值

Args:
    experiment_dir: 實驗結果目錄
    layer_names: 要分析的層名稱列表
    epochs: 要分析的輪次列表
    output_dir: 輸出目錄...
- `main()`: 主函數...


### examples/analyze_training_dynamics.py

**Functions:**

- `simulate_training_history()`: 模擬一個訓練歷史，生成訓練損失和驗證損失...
- `main()`


### examples/cross_validation_example.py

**Functions:**

- `main()`


### examples/distribution_analysis_example.py

**Functions:**

- `plot_distribution_and_analysis()`: 繪製分佈及其分析結果

Args:
    data: 數據張量
    title: 圖表標題
    analysis_result: 分佈分析結果
    axis: matplotlib軸對象...
- `generate_example_distributions()`: 生成各種類型的數據分佈以進行測試

Returns:
    dict: 包含不同分佈類型的張量字典...
- `compare_detection_methods()`: 比較基本和高級分佈檢測方法的結果

Args:
    distributions: 包含不同分佈類型的張量字典...
- `main()`: 主程式：執行分佈分析範例...


### examples/feature_importance_example.py

**Functions:**

- `analyze_random_forest_importance()`: 使用RandomForest的內建特徵重要性進行分析...
- `analyze_permutation_importance()`: 使用排列重要性方法分析特徵重要性...
- `analyze_shap_values()`: 使用SHAP值分析特徵重要性和解釋模型預測...
- `compare_importance_methods()`: 比較不同特徵重要性方法的結果...
- `main()`


### examples/intermediate_analysis_example.py

**Functions:**

- `create_mock_experiment_data()`: 創建模擬實驗數據用於示例...
- `run_intermediate_analysis()`: 執行中間層數據分析...
- `visualize_layer_metrics()`: 視覺化層級指標...
- `main()`: 主函數...


### examples/layer_activation_analysis_example.py

**Functions:**

- `parse_args()`: 解析命令行參數...
- `analyze_layer_activation()`: 分析層級激活值的特性

Args:
    activation_tensor: 層激活值張量
    layer_name: 層名稱
    
Returns:
    包含分析結果的字典...
- `create_activation_visualizations()`: 為激活值創建視覺化圖表

Args:
    activation_tensor: 層激活值張量
    layer_name: 層名稱
    output_dir: 輸出目錄
    
Returns:
    包含生成的圖表路徑的字典...
- `compare_layer_activations_across_epochs()`: 比較不同輪次的層激活值

Args:
    activations_dict: 輪次到激活值的映射字典
    layer_name: 層名稱
    output_dir: 輸出目錄
    
Returns:
    包含比較圖表路徑的字典...
- `compare_layers_within_epoch()`: 比較同一輪次中不同層的激活值特性

Args:
    activations_dict: 層名稱到激活值的映射字典
    epoch: 輪次編號
    output_dir: 輸出目錄
    
Returns:
    包含比較圖表路徑的字典...
- `main()`: 主函數...


### examples/layer_metrics_analyzer_example.py

**Functions:**

- `create_mock_layer_activations()`: 創建模擬層激活值用於示例分析...
- `analyze_single_layer()`: 分析單一層的激活值...
- `visualize_layer_activations()`: 視覺化層級激活值...
- `main()`: 主函數...


### examples/model_analysis_example.py

**Classes:**

- `ExampleCNN`: 簡單的CNN模型用於示例分析...
  - `forward()`

**Functions:**

- `create_model_structure_json()`: 創建模型結構的JSON文件，用於分析。

Args:
    model: PyTorch模型。
    input_shape: 輸入張量的形狀。
    save_path: 保存JSON文件的路徑。
    
Returns:
    str: JSON文件路徑。...
- `simulate_training_data()`: 模擬訓練過程中的數據，用於性能視覺化。

Args:
    epochs: 模擬的訓練周期數。
    
Returns:
    dict: 包含模擬訓練數據的字典。...
- `main()`: 主函數，展示完整的分析流程。...


## interfaces/

### interfaces/__init__.py


### interfaces/analyzer_interface.py

**Classes:**

- `AnalysisResults`: 封裝分析結果的類別。

此類別用於整合和存儲來自不同分析器的分析結果，並提供方便的訪問方法。

Attributes:
    model_structure_results (Dict): 模型結構分析結果。
    training_dynamics_results (Dict): 訓練動態分析結果。
    intermediate_data_results (Dict): 中間層數據分析結果。
    inference_results (Dict): 推理性能分析結果。
    plots (Dict[str, Any]): 生成的圖表。...
  - `add_model_structure_results()`: 添加模型結構分析結果。

Args:
    results (Dict): 模型結構分析結果。...
  - `add_training_dynamics_results()`: 添加訓練動態分析結果。

Args:
    results (Dict): 訓練動態分析結果。...
  - `add_intermediate_data_results()`: 添加中間層數據分析結果。

Args:
    results (Dict): 中間層數據分析結果。...
  - `add_inference_results()`: 添加推理性能分析結果。

Args:
    results (Dict): 推理性能分析結果。...
  - `add_plot()`: 添加圖表。

Args:
    plot_name (str): 圖表名稱。
    plot_data (Any): 圖表數據。...
  - `get_model_structure_results()`: 獲取模型結構分析結果。

Returns:
    Dict: 模型結構分析結果。...
  - `get_training_dynamics_results()`: 獲取訓練動態分析結果。

Returns:
    Dict: 訓練動態分析結果。...
  - `get_intermediate_data_results()`: 獲取中間層數據分析結果。

Returns:
    Dict: 中間層數據分析結果。...
  - `get_inference_results()`: 獲取推理性能分析結果。

Returns:
    Dict: 推理性能分析結果。...
  - `get_plot()`: 獲取指定名稱的圖表。

Args:
    plot_name (str): 圖表名稱。

Returns:
    Any: 圖表數據，如果不存在則返回None。...
  - `get_all_plots()`: 獲取所有圖表。

Returns:
    Dict[str, Any]: 包含所有圖表的字典。...
- `SBPAnalyzer`: SBP分析器的主接口類。

此類整合了各種不同類型的分析器，並提供了統一的接口。

Attributes:
    experiment_dir (str): 實驗結果的目錄路徑。
    experiment_loader (ExperimentLoader): 實驗數據加載器。
    model_structure_analyzer (ModelStructureAnalyzer): 模型結構分析器。
    training_dynamics_analyzer (TrainingDynamicsAnalyzer): 訓練動態分析器。
    intermediate_data_analyzer (IntermediateDataAnalyzer): 中間層數據分析器。
    inference_analyzer (InferenceAnalyzer): 推理性能分析器。
    results (AnalysisResults): 分析結果。...
  - `analyze()`: 執行分析。

Args:
    analyze_model_structure (bool, optional): 是否分析模型結構。默認為True。
    analyze_training_history (bool, optional): 是否分析訓練歷史。默認為True。
    analyze_hooks (bool, optional): 是否分析hook數據。默認為True。
    analyze_inference (bool, optional): 是否分析推理性能。默認為False。
    epochs (List[int], optional): 要分析的輪次列表。如果為None，則分析所有可用輪次。
    layers (List[str], optional): 要分析的層列表。如果為None，則分析所有可用層。
    batch (int, optional): 要分析的批次索引。默認為0。
    batch_sizes (List[int], optional): 要分析的批次大小列表。用於推理分析。
    input_shapes (List[List[int]], optional): 要分析的輸入形狀列表。用於推理分析。
    save_results (bool, optional): 是否保存分析結果。默認為True。

Returns:
    str: 生成的報告文件路徑...
  - `generate_report()`: 生成分析報告。

Args:
    output_dir (Optional[str], optional): 報告輸出目錄。如果為None，則使用默認路徑。
    report_format (str, optional): 報告格式，可選值為 'html'、'markdown' 或 'pdf'。默認為 'html'。
    template_dir (Optional[str], optional): 模板目錄路徑。若為None，則使用默認模板。

Returns:
    str: 報告路徑。...
  - `get_results()`: 獲取分析結果。

Returns:
    AnalysisResults: 分析結果對象。...


## legacy/

### legacy/distribution.py

**Functions:**

- `calculate_distribution_stats()`: Calculates basic distribution statistics for given data...


### legacy/run_complete_tests.py


## metrics/

### metrics/__init__.py


### metrics/distribution_metrics.py

**Functions:**

- `calculate_histogram_intersection()`: 計算兩個分布之間的直方圖交叉值，用來衡量分布的相似度。

Args:
    dist1: 第一個分布的數據
    dist2: 第二個分布的數據
    bins: 用於構建直方圖的bins數量
    
Returns:
    兩個直方圖的交叉值 (0到1之間，1表示完全相同)...
- `calculate_distribution_similarity()`: 使用指定的方法計算兩個分布之間的相似度或距離。

Args:
    dist1: 第一個分布的數據
    dist2: 第二個分布的數據
    method: 相似度計算方法，選項包括：
        - 'kl_divergence': KL散度 (值越小越相似)
        - 'js_divergence': JS散度 (值越小越相似)
        - 'wasserstein': Wasserstein距離 (值越小越相似)
        - 'correlation': Pearson相關係數 (值越接近1越相似)
    
Returns:
    兩個分布之間的相似度或距離度量...
- `compare_tensor_distributions()`: 比較兩個張量的分布，可以沿著指定維度進行比較。

Args:
    tensor1: 第一個張量
    tensor2: 第二個張量
    dim: 要比較的維度，若為None則比較整個張量
    methods: 要使用的比較方法列表
    
Returns:
    包含各種相似度度量的字典...
- `calculate_distribution_stats()`: 計算分布的各種統計量。

Args:
    dist: 輸入分布數據
    
Returns:
    包含統計量的字典:
        - mean: 平均值
        - std: 標準差
        - min: 最小值
        - max: 最大值
        - median: 中位數
        - skewness: 偏度
        - kurtosis: 峰度...
- `calculate_distribution_entropy()`: 計算分布的香農熵。

Args:
    dist: 輸入分布數據
    bins: 用於構建直方圖的bins數量
    
Returns:
    分布的香農熵值 (以位元為單位)...


### metrics/layer_activity_metrics.py

**Functions:**

- `compute_activation_distribution()`: 計算激活值分佈的基本屬性，包括零值、正值和負值的比例。

Args:
    activations: 層級激活值張量
    
Returns:
    包含分佈特性的字典...
- `compute_neuron_activity()`: 計算神經元活躍度指標，包括活躍神經元的數量和平均活躍度。

Args:
    activations: 層級激活值張量
    threshold: 神經元被視為活躍的激活閾值
    
Returns:
    包含神經元活躍度指標的字典...
- `detect_dead_neurons()`: 檢測層中的死亡神經元（始終輸出接近零的神經元）。

Args:
    activations: 層級激活值張量
    threshold: 判定神經元死亡的最大激活閾值
    return_indices: 是否返回死亡神經元的索引
    
Returns:
    包含死亡神經元信息的字典...
- `compute_saturation_metrics()`: 計算神經元飽和度指標，識別飽和神經元(輸出始終接近最大值)。

Args:
    activations: 層級激活值張量
    saturation_threshold: 判定神經元飽和的閾值
    
Returns:
    包含飽和度指標的字典...
- `compute_layer_similarity()`: 計算兩組層級激活值之間的相似度。

Args:
    activations1: 第一組層級激活值
    activations2: 第二組層級激活值
    method: 相似度計算方法，可選'cosine'或'correlation'
    
Returns:
    相似度值...
- `compute_stability_over_batches()`: 計算多個批次激活值的穩定性指標。

Args:
    batch_activations: 多個批次的激活值列表，每個元素是一個批次的激活張量
    
Returns:
    包含穩定性指標的字典，如方差、變異係數、批次間相似度等...
- `detect_activation_anomalies()`: 檢測神經網路層的異常激活模式。

此函數分析激活值張量，檢測多種潛在異常：
1...
- `analyze_layer_relationships()`: # 分析不同層之間的相互關係

此函數分析神經網絡中不同層之間的關係，包括相關性、互信息和結構相似性等指標。

Args:
    activations_dict: 層名稱到激活值的字典
    reference_layer: 參考層名稱，如果提供，則分析所有層與此層的關係
    metrics: 要計算的關係指標列表
    
Returns:
    包含層間關係分析結果的字典，包括:
        - layer_pairs: 分析的層對列表
        - metrics: 每種度量的值列表
        - layer_graph: 層之間關係的圖表示...
- `calculate_activation_statistics()`: 計算激活值的統計指標。這個函數是 compute_activation_distribution 的擴展版本，
增加了更多的統計指標。

Args:
    activations: 層級激活值張量
    
Returns:
    包含統計指標的字典...
- `calculate_activation_sparsity()`: 計算激活值的稀疏度，即接近零的值的比例。

Args:
    activations: 層級激活值張量
    threshold: 判定為零的閾值
    
Returns:
    稀疏度 (0-1之間的值，1表示全為零)...
- `calculate_activation_saturation()`: 計算激活值的飽和度，即接近最大值的比例。

Args:
    activations: 層級激活值張量
    saturation_threshold: 飽和閾值
    
Returns:
    飽和度 (0-1之間的值)...
- `calculate_effective_rank()`: 計算激活值矩陣的有效秩，用於衡量特徵的多樣性。

Args:
    activations: 層級激活值張量
    
Returns:
    有效秩...
- `calculate_feature_coherence()`: 計算特徵間的一致性，衡量不同通道/特徵之間的相關程度。

Args:
    activations: 層級激活值張量，形狀為 [batch, channels, height, width] 或 [batch, seq_len, features]
    
Returns:
    一致性指標...
- `_calculate_feature_vectors_coherence()`: 計算特徵向量之間的一致性

Args:
    flattened: 已展平的張量，形狀為 [batch, channels/features, flattened_dim]
    
Returns:
    一致性指標...
- `calculate_activation_dynamics()`: 計算不同輪次間的激活值變化。

Args:
    epoch_activations: 輪次到激活值的字典
    
Returns:
    包含動態變化指標的字典...
- `calculate_layer_similarity()`: 計算兩個層的激活值之間的相似度。

Args:
    activations1: 第一個層的激活值
    activations2: 第二個層的激活值
    
Returns:
    包含不同相似度指標的字典...


### metrics/performance_metrics.py

**Functions:**

- `analyze_loss_curve()`: 分析損失曲線的收斂性、穩定性等特性。

Args:
    loss_values (List[float]): 訓練損失值序列。
    val_loss_values (Optional[List[float]], optional): 驗證損失值序列。默認為None。
    window_size (int, optional): 移動平均窗口大小。默認為5。
    convergence_threshold (float, optional): 收斂閾值。默認為0...
- `analyze_metric_curve()`: 分析評估指標曲線的特性。

Args:
    metric_values (List[float]): 訓練指標值序列。
    val_metric_values (Optional[List[float]], optional): 驗證指標值序列。默認為None。
    is_higher_better (bool, optional): 是否指標值越高越好。默認為True。
    window_size (int, optional): 移動平均窗口大小。默認為5。
    convergence_threshold (float, optional): 收斂閾值。默認為0...
- `analyze_learning_efficiency()`: 分析模型的學習效率。

Args:
    loss_values (List[float]): 損失值序列。
    epochs (List[int], optional): 對應的輪次。默認為None，使用從0開始的序列。
    iterations (List[int], optional): 對應的迭代次數。默認為None。
    
Returns:
    Dict[str, Any]: 包含學習效率分析結果的字典。...
- `analyze_lr_schedule_impact()`: 分析學習率調度對訓練的影響。

Args:
    loss_values (List[float]): 損失值序列。
    learning_rates (List[float]): 對應的學習率序列。
    
Returns:
    Dict[str, Any]: 包含學習率影響分析結果的字典。...
- `detect_training_anomalies()`: 檢測訓練過程中的異常。

Args:
    loss_values (List[float]): 訓練損失值序列。
    val_loss_values (Optional[List[float]], optional): 驗證損失值序列。默認為None。
    metrics (Optional[Dict[str, List[float]]], optional): 其他指標序列的字典。默認為None。
    
Returns:
    Dict[str, Any]: 包含異常檢測結果的字典。...
- `detect_overfitting()`: 檢測過擬合現象。

Args:
    train_values (List[float]): 訓練值序列。
    val_values (List[float]): 驗證值序列。
    window_size (int, optional): 移動平均窗口大小。默認為5。
    threshold (float, optional): 過擬合判定閾值。默認為0...
- `detect_oscillations()`: 檢測訓練曲線中的震盪現象。

Args:
    values (List[float]): 值序列。
    window_size (int, optional): 檢測窗口大小。默認為3。
    threshold (float, optional): 震盪判定閾值。默認為0...
- `detect_plateaus()`: 檢測訓練曲線中的平台期。

Args:
    values (List[float]): 值序列。
    window_size (int, optional): 檢測窗口大小。默認為5。
    threshold (float, optional): 平台期判定閾值。默認為0...
- `is_plateau_region()`
- `_calculate_moving_average()`: 計算移動平均。

Args:
    values (List[float]): 值序列。
    window_size (int, optional): 窗口大小。默認為5。
    
Returns:
    List[float]: 移動平均序列。...
- `_find_convergence_point()`: 查找收斂點。

Args:
    values (List[float]): 值序列。
    window_size (int, optional): 窗口大小。默認為5。
    threshold (float, optional): 收斂閾值。默認為0...
- `_calculate_improvement_rate()`: 計算指標的改進速率。

Args:
    values (List[float]): 值序列。
    is_higher_better (bool, optional): 是否指標值越高越好。默認為True。
    
Returns:
    float: 改進速率。...
- `_detect_spikes()`: 檢測數據中的異常尖峰。

Args:
    values (List[float]): 值序列。
    window_size (int, optional): 檢測窗口大小。默認為3。
    threshold (float, optional): 尖峰判定Z分數閾值。默認為2...


## reporter/

### reporter/__init__.py


### reporter/report_generator.py

**Classes:**

- `ReportGenerator`: 報告生成器，用於生成分析報告。

此類別可以基於分析結果生成HTML、Markdown或PDF格式的報告。

Attributes:
    experiment_name (str): 實驗名稱。
    output_dir (str): 輸出目錄路徑。
    logger (logging...
  - `_load_templates()`: 從指定目錄載入模板。

Args:
    template_dir (str): 模板目錄路徑。...
  - `_init_default_templates()`: 初始化默認模板。...
  - `generate()`: 生成分析報告。

Args:
    results (Any): 分析結果對象。
    format (str, optional): 報告格式，可選 'html'、'markdown' 或 'pdf'。默認為'html'。

Returns:
    str: 報告文件路徑。...
  - `generate_report()`: 生成分析報告（兼容性方法，用於測試）。

Args:
    analysis_results (Any): 分析結果對象。
    experiment_name (Optional[str], optional): 實驗名稱。如果提供，則會覆蓋初始化時設置的實驗名稱。
    output_dir (Optional[str], optional): 輸出目錄。如果提供，則會覆蓋初始化時設置的輸出目錄。
    format (str, optional): 報告格式，可選 'html'、'markdown' 或 'pdf'。默認為'html'。

Returns:
    str: 報告文件路徑。...
  - `_generate_html_report()`: 生成HTML格式的報告。

Args:
    results (Any): 分析結果對象。

Returns:
    str: 報告文件路徑。...
  - `_generate_model_structure_section()`: 生成模型結構分析部分的HTML。

Args:
    model_structure_results (Dict): 模型結構分析結果。

Returns:
    str: HTML內容。...
  - `_generate_training_dynamics_section()`: 生成訓練動態分析部分的HTML。

Args:
    training_dynamics_results (Dict): 訓練動態分析結果。

Returns:
    str: HTML內容。...
  - `_generate_intermediate_data_section()`: 生成中間層數據分析部分的HTML。

Args:
    intermediate_data_results (Dict): 中間層數據分析結果。

Returns:
    str: HTML內容。...
  - `_generate_markdown_report()`: 生成Markdown格式的報告。

Args:
    results (Any): 分析結果對象。

Returns:
    str: 報告文件路徑。...
  - `_generate_model_structure_section_md()`: 生成模型結構分析部分的Markdown內容。

Args:
    model_structure_results (Dict): 模型結構分析結果。

Returns:
    str: Markdown內容。...
  - `_generate_training_dynamics_section_md()`: 生成訓練動態分析部分的Markdown內容。

Args:
    training_dynamics_results (Dict): 訓練動態分析結果。

Returns:
    str: Markdown內容。...
  - `_generate_intermediate_data_section_md()`: 生成中間層數據分析部分的Markdown內容。

Args:
    intermediate_data_results (Dict): 中間層數據分析結果。

Returns:
    str: Markdown內容。...
  - `_generate_pdf_report()`: 生成PDF格式的報告。

首先生成HTML報告，然後使用weasyprint或pdfkit將其轉換為PDF。

Args:
    results (Any): 分析結果對象。

Returns:
    str: 報告文件路徑。...
  - `_generate_pdf_with_weasyprint()`: 使用weasyprint生成PDF報告。

Args:
    html_path (str): HTML報告文件路徑。

Returns:
    str: PDF報告文件路徑。...
  - `_generate_pdf_with_pdfkit()`: 使用pdfkit生成PDF報告。

Args:
    html_path (str): HTML報告文件路徑。

Returns:
    str: PDF報告文件路徑。...
  - `set_template()`: 設置指定格式的報告模板。

Args:
    format (str): 模板格式，可選 'html' 或 'markdown'。
    template_content (str): 模板內容。...


## sample_data/


## tests/

### tests/__init__.py


### tests/conftest.py

**Functions:**

- `pytest_sessionstart()`: 在測試會話開始前設置 Python 路徑。

Args:
    session: pytest 會話物件

Returns:
    無...


### tests/run_integration_tests.py

**Functions:**

- `run_pytest()`: 運行pytest測試...
- `run_unittest()`: 運行unittest框架的測試...


### tests/run_lr_schedule_test.py


### tests/run_model_structure_test.py

**Functions:**

- `setup_test_data()`: 設置測試數據。...
- `test_model_structure_analyzer()`: 測試模型結構分析器的基本功能。...


### tests/run_simple_tests.py

**Functions:**

- `test_basic()`


### tests/run_tests.py


### tests/test_adaptive_threshold_analyzer.py

**Classes:**

- `MockHookDataLoader`: 模擬鉤子資料載入器，用於測試不同類型的激活分佈...
  - `get_layer_names()`: 獲取層名稱列表...
  - `list_available_epochs()`: 獲取可用的輪次列表...
  - `list_available_layer_activations()`: 獲取特定輪次可用的層列表...
  - `list_available_batches()`: 獲取特定輪次可用的批次列表...
  - `load_layer_activation()`: 載入指定層的激活值。

Args:
    layer_name (str): 層名稱
    epoch (int): 輪次編號
    batch (int, optional): 批次編號，預設為0

Returns:
    torch...
  - `load_activation_data()`: 根據層名稱載入模擬激活資料

Args:
    layer_name: 層名稱，格式為 'layerX_distribution_type'
    
Returns:
    模擬的激活資料張量...
- `TestAdaptiveThresholdAnalyzer`: 測試自適應閾值分析器...
  - `setUp()`: 測試前準備工作...
  - `tearDown()`: 測試後清理工作...
  - `test_analyze_with_mock_loader()`: 測試使用模擬載入器的分析功能...
  - `test_distribution_type_detection()`: 測試分佈類型檢測功能...
  - `test_compare_with_baseline()`: 測試與基準分析器比較...
  - `_compare_detection_results()`: 比較基準分析器和自適應分析器的檢測結果...
  - `_generate_comparison_plots()`: 生成基準和自適應方法的比較圖...


### tests/test_adaptive_thresholds.py

**Classes:**

- `MockHookDataLoader`: 用於測試的模擬資料載入器，生成具有不同分佈特性的激活數據。...
  - `load_activation_data()`: 載入指定層的模擬激活數據。

Args:
    layer_name: 層名稱
    batch_indices: 批次索引，如果為None則返回所有批次
    
Returns:
    激活數據張量...
  - `get_layer_names()`: 返回所有層名稱。...
  - `list_available_epochs()`: 獲取可用的輪次列表（模擬）。...
  - `list_available_batches()`: 獲取特定輪次可用的批次列表（模擬）。...
  - `list_available_layer_activations()`: 獲取特定輪次可用的層列表（模擬）。...
  - `load_layer_activation()`: 載入指定層、輪次和批次的激活數據。

Args:
    layer_name: 層名稱
    epoch: 輪次
    batch: 批次編號
    
Returns:
    激活數據張量...
  - `get_epoch_layers()`: 返回特定輪次的所有層名稱。...
  - `get_batch_count()`: 返回特定輪次的批次數量。...
  - `list_epochs()`: 獲取可用的輪次列表。...
  - `list_batches()`: 獲取特定輪次可用的批次列表。...
  - `list_layers()`: 獲取特定輪次可用的層列表。...
  - `get_layer_activations()`: 獲取指定輪次的所有層的激活數據（模擬）。...
- `TestAdaptiveThresholds`: 測試不同閾值設定對神經元活性檢測的影響。...
  - `setUp()`: 測試前準備工作。...
  - `test_threshold_sensitivity()`: 測試不同閾值對死亡/飽和神經元檢測的敏感度。...
  - `_generate_comparison_plots()`: 生成不同閾值設定下的結果對比圖。...
  - `test_distribution_specific_thresholds()`: 測試針對不同分佈特性的閾值敏感度。...
  - `_generate_threshold_sensitivity_matrix()`: 生成閾值敏感度矩陣圖，顯示不同分佈類型對閾值變化的敏感度。...

**Functions:**

- `_tensor_name()`: 獲取張量名稱...
- `_tensor_set_name()`: 設置張量名稱...


### tests/test_complete_integration.py

**Classes:**

- `MockAnalysisResults`
  - `get_model_structure_results()`
  - `get_training_dynamics_results()`
  - `get_intermediate_data_results()`
  - `get_plot()`
  - `get_all_plots()`
  - `to_dict()`

**Functions:**

- `setup_mock_experiment()`: 創建完整的模擬實驗目錄結構，包含所有必要的文件和數據。...
- `test_hook_data_loading_complete()`: 測試完整的Hook數據加載功能...
- `test_intermediate_data_analysis()`: 測試中間層數據分析功能...
- `test_report_generation_with_intermediate_data()`: 測試包含中間層數據的報告生成功能...
- `test_end_to_end_with_sbp_analyzer()`: 使用SBPAnalyzer進行端到端測試...


### tests/test_data_loader.py

**Classes:**

- `TestExperimentLoader`: 測試ExperimentLoader類。...
  - `setUp()`: 測試前的準備工作。...
  - `test_load_config()`: 測試載入實驗配置文件。...
  - `test_load_model_structure()`: 測試載入模型結構文件。...
  - `test_load_training_history()`: 測試載入訓練歷史文件。...
  - `test_load_all()`: 測試一次性載入所有文件。...
- `TestHookDataLoader`: 測試HookDataLoader類。...
  - `_create_temp_dir()`
  - `_create_dummy_files()`
  - `test_initialization()`: 測試初始化過程。...
  - `test_load_training_summary()`: 測試載入訓練摘要。...
  - `test_load_evaluation_results()`: 測試載入評估結果。...
  - `test_list_available_epochs()`: 測試列出可用的輪次。...
  - `test_list_available_layers()`: 測試列出可用的層。...
  - `test_load()`: 測試基本的載入功能。...
  - `test_list_available_layers()`: 測試列出可用的層激活文件...
  - `test_load_activation()`: 測試載入單一層激活值...
  - `test_init()`: 測試初始化HookDataLoader...

**Functions:**

- `temp_experiment_dir()`: 創建臨時實驗目錄，含結果摘要文件以供測試...
- `test_experiment_load_results()`: 測試載入結果摘要...


### tests/test_distribution_metrics.py

**Classes:**

- `TestDistributionMetrics`: 測試分布度量功能的類。...
  - `setUp()`: 設置測試環境，創建測試數據。...
  - `test_histogram_intersection()`: 測試直方圖交叉計算功能。...
  - `test_distribution_similarity()`: 測試分布相似度計算功能。...
  - `test_compare_tensor_distributions()`: 測試張量分布比較功能。...
  - `test_distribution_stats()`: 測試分布統計計算功能。...
  - `test_distribution_entropy()`: 測試分布熵計算功能。...


### tests/test_distribution_plots.py

**Classes:**

- `TestDistributionPlotter`: 測試DistributionPlotter類的各個方法...
  - `plotter()`: 創建一個DistributionPlotter實例用於測試...
  - `sample_data()`: 生成測試用的數據樣本...
  - `sample_data_dict()`: 生成測試用的多個分佈數據...
  - `sample_tensor()`: 生成測試用的PyTorch張量...
  - `test_plot_histogram_with_tensor()`: 測試plot_histogram方法能否處理PyTorch張量...
  - `test_plot_distribution_comparison()`: 測試plot_distribution_comparison方法能否比較多個分佈...
  - `test_plot_boxplot()`: 測試plot_boxplot方法能否繪製箱形圖...
  - `test_plot_heatmap()`: 測試plot_heatmap方法能否繪製熱力圖...
  - `test_plot_qq_plot()`: 測試plot_qq_plot方法能否繪製Q-Q圖...
  - `test_with_different_data_types()`: 測試各種方法是否能處理不同類型的數據...


### tests/test_end_to_end.py

**Classes:**

- `MockAnalysisResults`
  - `to_dict()`
  - `to_dict()`
- `MockAnalysisResults`
  - `to_dict()`
  - `to_dict()`

**Functions:**

- `setup_test_data()`: 設置測試數據環境

該fixture會創建一個模擬的實驗結果目錄結構，包含：
- config...
- `test_hook_data_loader()`: 測試HookDataLoader能否正確載入並處理hook數據...
- `test_experiment_loader()`: 測試ExperimentLoader能否正確載入實驗配置和訓練歷史...
- `test_layer_activity_analysis()`: 測試層級活動分析功能...
- `test_report_generator()`: 測試報告生成器功能...
- `test_end_to_end_workflow()`: 測試完整的端到端分析流程...


### tests/test_hook_data_loader.py

**Functions:**

- `_create_dummy_hook_files()`
- `hook_data_dir()`
- `hook_loader()`
- `temp_experiment_dir()`: 創建一個臨時實驗目錄，包含測試所需的所有文件...
- `test_init_with_valid_dir()`
- `test_init_with_invalid_dir()`
- `test_list_available_epochs()`
- `test_list_available_layers()`
- `test_list_available_batches()`
- `test_load_layer_activation()`
- `test_load_nonexistent_layer_activation()`: 測試載入不存在層的激活值...
- `test_load_layer_activation_invalid_epoch_or_batch()`
- `test_load_epoch_summary()`
- `test_load_epoch_summary_invalid_epoch()`
- `test_load_training_summary()`
- `test_load_training_summary_not_found()`
- `test_load_all_layer_activations()`: 測試載入單一輪次所有層的激活值...
- `test_load_error_handling()`
- `test_load_training_summary()`: 測試載入訓練摘要...
- `test_load_evaluation_results()`: 測試載入評估結果...
- `test_load_epoch_summary()`: 測試載入輪次摘要...
- `test_load_batch_data()`: 測試載入批次數據...
- `test_load_layer_activation()`: 測試載入層激活值...
- `test_nonexistent_layer_should_raise_error()`: 測試載入不存在層的激活值應該拋出異常...
- `test_load_nonexistent_epoch()`: 測試載入不存在輪次的激活值...
- `test_load_all_layer_activations()`: 測試載入單一輪次所有層的激活值...
- `test_load_layer_activations_across_epochs()`: 測試載入跨輪次的層激活值...
- `test_load_activation_with_preprocessing()`: 測試載入並預處理激活值數據...
- `test_load_activations_batch()`: 測試批量載入激活值數據...
- `test_load_layer_statistics()`: 測試載入層統計數據...
- `test_get_activation_metadata()`: 測試獲取激活值元數據...
- `test_save_processed_activation()`: 測試保存處理後的激活值...


### tests/test_inference_analyzer.py

**Classes:**

- `TestInferenceAnalyzer`: 測試 InferenceAnalyzer 類的功能。...
  - `setUp()`: 設置測試環境和樣本數據。...
  - `tearDown()`: 清理測試環境。...
  - `test_initialization()`: 測試 InferenceAnalyzer 的初始化。...
  - `test_load_inference_logs()`: 測試載入推理日誌功能。...
  - `test_load_model_structure()`: 測試載入模型結構功能。...
  - `test_load_nonexistent_logs()`: 測試載入不存在的日誌檔案時的錯誤處理。...
  - `test_analyze_time_performance()`: 測試時間性能分析。...
  - `test_analyze_throughput()`: 測試吞吐量分析。...
  - `test_analyze_memory_usage()`: 測試記憶體使用分析。...
  - `test_analyze_output_statistics()`: 測試輸出統計分析。...
  - `test_analyze_batch_size_impact()`: 測試批次大小影響分析。...
  - `test_analyze_input_shape_impact()`: 測試輸入形狀影響分析。...
  - `test_analyze_accuracy()`: 測試準確性分析。...
  - `test_analyze_resource_efficiency()`: 測試資源效率分析。...
  - `test_analyze_hardware_comparison()`: 測試硬體比較分析。...
  - `test_full_analyze()`: 測試完整的分析流程。...
  - `test_getter_methods()`: 測試獲取分析結果的方法。...


### tests/test_integration.py

**Classes:**

- `MockAnalysisResults`
  - `to_dict()`

**Functions:**

- `_create_dummy_hook_files()`
- `setup_test_data()`: 創建模擬實驗目錄結構和測試數據...
- `test_full_analysis_flow()`: 測試完整的分析流程，從數據加載到報告生成。...
- `test_individual_analyzers()`: 測試各分析器獨立運行...
- `test_direct_report_generation()`: 測試直接使用報告生成器...


### tests/test_intermediate_data_analyzer.py

**Classes:**

- `MockHookDataLoader`: 用於測試的模擬HookDataLoader...
  - `list_available_epochs()`
  - `list_available_batches()`
  - `list_available_layers()`
  - `load_layer_activation()`
  - `get_activation_metadata()`

**Functions:**

- `mock_loader()`
- `temp_experiment_dir()`: 創建臨時實驗目錄...
- `analyzer()`: Creates an IntermediateDataAnalyzer instance initialized with tmp_path
and patches its hook_loader with MockHookDataLoader...
- `mock_list_activations()`
- `test_initialization()`: Test initialization with a directory path...
- `test_initialization_without_loader()`
- `test_analyze_basic_flow()`
- `test_analyze_specific_analyses()`
- `test_analyze_with_mock_loader()`
- `test_analyze_layer_subset()`
- `test_analyze_epoch_subset()`
- `test_effective_rank_calculation()`
- `test_similarity_calculation()`
- `test_analyze_error_handling_layer_load()`
- `faulty_load()`
- `test_get_summary()`
- `test_get_summary_before_analyze()`


### tests/test_layer_activity_metrics.py

**Functions:**

- `test_calculate_activation_statistics()`: 測試計算激活值統計指標...
- `test_calculate_activation_sparsity()`: 測試計算激活值稀疏度...
- `test_calculate_activation_saturation()`: 測試計算激活值飽和度...
- `test_calculate_effective_rank()`: 測試計算激活值有效秩...
- `test_calculate_feature_coherence()`: 測試計算特徵一致性...
- `test_detect_dead_neurons()`: 測試檢測失活神經元...
- `test_calculate_activation_dynamics()`: 測試計算激活值動態變化...
- `test_calculate_layer_similarity()`: 測試計算層間相似度...
- `test_handle_different_input_formats()`: 測試處理不同輸入格式...
- `test_detect_activation_anomalies()`: 測試檢測異常激活模式功能...
- `test_analyze_layer_relationships()`: 測試分析層與層之間關係功能...
- `test_error_handling()`: 測試錯誤處理...


### tests/test_lr_schedule_impact.py

**Classes:**

- `TestLRScheduleImpact`: 測試學習率調度影響分析功能。...
  - `setUp()`: 測試前的準備工作。...
  - `test_analyze_lr_schedule_detection()`: 測試學習率調度檢測功能。...
  - `test_analyze_lr_correlation()`: 測試學習率與損失的相關性分析。...
  - `test_compare_performance()`: 測試不同學習率調度的性能比較。...
  - `test_plot_lr_schedule_impact()`: 測試學習率調度影響的可視化。...
  - `test_with_missing_data()`: 測試處理缺失數據的情況。...
  - `test_recommendations_generation()`: 測試學習率調度建議的生成。...
  - `test_plot_with_english_labels()`: 測試使用英文標籤的可視化。...


### tests/test_model_structure_analyzer.py

**Classes:**

- `TestModelStructureAnalyzer`: 測試ModelStructureAnalyzer類的各個方法...
  - `temp_experiment_dir()`: 創建一個臨時實驗目錄用於測試...
  - `test_initialization()`: 測試分析器初始化...
  - `test_load_model_structure()`: 測試載入模型結構文件...
  - `test_analyze_basic_info()`: 測試模型基本信息分析...
  - `test_identify_architecture_type()`: 測試架構類型識別...
  - `test_analyze_layer_hierarchy()`: 測試層級結構分析...
  - `test_analyze_parameter_distribution()`: 測試參數分佈分析...
  - `test_analyze_weight_distributions()`: 測試權重分佈分析...
  - `test_analyze_layer_complexity()`: 測試層級複雜度分析...
  - `test_identify_largest_layers()`: 測試識別最大層...
  - `test_analyze_sequential_paths()`: 測試序列路徑分析...
  - `test_calculate_parameter_efficiency()`: 測試參數效率計算...
  - `test_analyze_layer_connectivity()`: 測試層間連接性分析...
  - `test_analyze_computational_complexity()`: 測試計算複雜度分析...
  - `test_full_analyze()`: 測試完整的分析過程...
  - `test_get_model_summary()`: 測試獲取模型摘要...
  - `test_get_layer_hierarchy()`: 測試獲取層級結構...


### tests/test_performance_metrics.py

**Classes:**

- `TestPerformanceMetrics`: 測試性能指標計算函數。...
  - `setUp()`: 設置測試環境。...
  - `test_analyze_loss_curve()`: 測試損失曲線分析函數。...
  - `test_analyze_metric_curve()`: 測試指標曲線分析函數。...
  - `test_analyze_learning_efficiency()`: 測試學習效率分析函數。...
  - `test_analyze_lr_schedule_impact()`: 測試學習率調度影響分析函數。...
  - `test_detect_training_anomalies()`: 測試訓練異常檢測函數。...
  - `test_detect_overfitting()`: 測試過擬合檢測函數。...
  - `test_detect_oscillations()`: 測試震盪檢測函數。...
  - `test_detect_plateaus()`: 測試平台期檢測函數。...
  - `test_calculate_moving_average()`: 測試移動平均計算函數。...
  - `test_find_convergence_point()`: 測試收斂點查找函數。...


### tests/test_performance_plots.py

**Classes:**

- `TestPerformancePlotter`: 測試PerformancePlotter類的各個方法...
  - `plotter()`: 創建一個PerformancePlotter實例用於測試...
  - `sample_loss_history()`: 生成測試用的損失歷史數據...
  - `sample_metric_history()`: 生成測試用的指標歷史數據...
  - `sample_lr_history()`: 生成測試用的學習率歷史數據...
  - `test_plot_loss_curve()`: 測試plot_loss_curve方法能否正確繪製損失曲線...
  - `test_plot_metric_curves()`: 測試plot_metric_curves方法能否正確繪製多個指標曲線...
  - `test_plot_learning_rate()`: 測試plot_learning_rate方法能否正確繪製學習率曲線...
  - `test_plot_convergence_analysis()`: 測試plot_convergence_analysis方法能否正確分析收斂情況...
  - `test_plot_training_stability()`: 測試plot_training_stability方法能否正確分析訓練穩定性...
  - `test_plot_gradient_norm()`: 測試plot_gradient_norm方法能否正確繪製梯度範數曲線...
  - `test_plot_training_heatmap()`: 測試plot_training_heatmap方法能否正確繪製訓練熱力圖...


### tests/test_report_generator.py

**Classes:**

- `MockAnalysisResults`
  - `to_dict()`

**Functions:**

- `setup_test_dirs()`: 設置測試目錄...
- `setup_custom_templates()`: 創建自定義報告模板...
- `mock_analysis_results()`: 創建模擬分析結果...
- `test_report_generator_init()`: 測試報告生成器初始化...
- `test_generate_html_report()`: 測試HTML報告生成...
- `test_generate_markdown_report()`: 測試Markdown報告生成...
- `test_custom_templates()`: 測試使用自定義模板...
- `test_set_template()`: 測試動態設置模板...
- `test_generate_pdf_report()`: 測試PDF報告生成 (如果支持)...
- `test_handle_corrupted_data()`: 測試處理損毀的分析結果數據...


### tests/test_simple_analyzer.py

**Functions:**

- `main()`: 執行訓練動態分析器的簡單測試...


### tests/test_tensor_utils.py

**Functions:**

- `test_to_numpy()`: 測試張量轉換為NumPy陣列...
- `test_normalize_tensor_minmax()`: 測試最小最大值正規化...
- `test_normalize_tensor_zscore()`: 測試Z分數正規化...
- `test_normalize_tensor_l1_l2()`: 測試L1和L2正規化...
- `test_normalize_tensor_error_handling()`: 測試正規化錯誤處理...
- `test_preprocess_activations()`: 測試激活值預處理...
- `test_get_tensor_statistics()`: 測試計算張量統計信息...
- `test_tensor_correlation()`: 測試張量相關性計算...
- `test_average_activations()`: 測試計算平均激活值...
- `test_extract_feature_maps()`: 測試提取特徵圖...
- `test_handle_nan_values()`: 測試處理NaN值的功能...
- `test_reduce_dimensions()`: 測試降維功能...
- `test_handle_sparse_tensor()`: 測試處理稀疏張量功能...
- `test_batch_process_activations()`: 測試批處理激活值功能...


### tests/test_training_dynamics_analyzer.py

**Functions:**

- `sample_training_history_dict()`
- `sample_training_history_overfitting()`
- `_create_exp_dir_with_history()`: Helper to create a temp dir and write history...
- `analyzer_tmp()`: 使用 tmp_path 創建分析器實例。...
- `overfitting_analyzer_tmp()`: 使用 tmp_path 創建一個用於測試過擬合的分析器實例。...
- `test_initialization()`: 測試初始化時是否正確加載數據並設置屬性。...
- `test_initialization_missing_history_file()`: 測試當 training_history...
- `test_initialization_invalid_json()`: 測試當 training_history...
- `test_initialization_empty_or_no_list_history()`: 測試當 training_history...
- `test_initialization_inconsistent_lengths()`: 測試當 history...
- `test_analyze_convergence()`: 測試收斂性分析。...
- `test_analyze_stability()`: 測試穩定性分析。...
- `test_analyze_overfitting()`: 測試過擬合分析。...
- `test_full_analyze_method()`: 測試完整的 analyze 方法。...
- `test_get_metric_correlation()`: 測試指標相關性計算。...
- `test_get_metric_correlation_insufficient_data()`: 測試數據不足時相關性計算的行為。...
- `test_get_summary()`: 測試生成摘要字典的功能。...
- `test_get_summary_before_analyze()`: 測試在 analyze 之前調用 get_summary 的行為。...
- `test_preprocess_history_missing_epoch()`: 測試歷史記錄缺少 'epoch' 鍵時的行為。...
- `test_analyze_single_data_point()`: 測試只有單個數據點時的分析行為。...
- `test_handling_nan_values()`: 測試處理歷史數據中 NaN 值的行為。...
- `test_load_training_history_error()`: 測試無法載入訓練歷史文件時的行為。...
- `test_plot_loss_curves()`: 測試繪製損失曲線的功能。...


### tests/test_utils.py

**Functions:**

- `sample_tensor_normal()`
- `sample_tensor_sparse()`
- `sample_tensor_saturated()`
- `sample_tensor_dead()`
- `sample_data_for_outliers()`
- `sample_data_converging()`
- `sample_data_not_converging()`
- `sample_data_stable()`
- `sample_data_unstable()`
- `test_get_files_by_extension()`: 測試獲取指定擴展名的文件列表。...
- `test_ensure_dir_exists()`: 測試確保目錄存在。...
- `test_flatten_tensor()`: 測試平坦化張量。...
- `test_tensor_to_numpy()`: 測試張量轉換為NumPy數組。...
- `test_get_tensor_statistics()`: 測試計算張量統計量。...
- `test_normalize_tensor()`: 測試規範化張量。...
- `test_sample_tensor()`: 測試從張量中抽樣。...
- `test_calculate_moving_average()`: 測試計算移動平均。...
- `test_find_convergence_point()`: 測試查找收斂點。...
- `test_calculate_stability_metrics()`: 測試計算穩定性指標。...
- `test_detect_outliers()`: 測試檢測異常值。...
- `test_moved_calculate_distribution_stats()`
- `test_moved_detect_distribution_type()`
- `test_detect_overfitting_simple()`: 測試簡單的過擬合檢測功能 (現在從 utils...


## tests/test_data/

### tests/test_data/__init__.py


## tests/test_data/mock_complete_test/

## tests/test_data/mock_complete_test/analysis/


## tests/test_data/mock_complete_test/hooks/

## tests/test_data/mock_complete_test/hooks/epoch_0/


## tests/test_data/mock_complete_test/hooks/epoch_10/


## tests/test_data/mock_complete_test/hooks/epoch_5/


## tests/test_data/mock_complete_test/models/


## tests/test_data/mock_complete_test/results/


## tests/test_data/mock_experiment/

## tests/test_data/mock_experiment/analysis/


## tests/test_data/mock_experiment/analysis_report/


## tests/test_data/mock_experiment/epoch_0/

## tests/test_data/mock_experiment/epoch_0/batch_0/


## tests/test_data/mock_experiment/epoch_1/

## tests/test_data/mock_experiment/epoch_1/batch_0/


## tests/test_data/mock_experiment/hooks/

## tests/test_data/mock_experiment/hooks/epoch_0/


## tests/test_data/mock_experiment/hooks/epoch_1/


## tests/test_data/mock_experiment/models/


## tests/test_data/mock_experiment/results/


## tests/test_output/

## tests/test_output/adaptive_thresholds/

## tests/test_output/adaptive_thresholds/20250426_123102/


## tests/test_output/adaptive_thresholds/20250426_123251/


## tests/test_output/adaptive_thresholds/20250426_124842/


## tests/test_output/adaptive_thresholds/20250426_124847/


## tests/test_output/adaptive_thresholds/20250426_125520/


## tests/test_output/adaptive_thresholds/20250426_125525/


## tests/test_output/adaptive_thresholds/20250426_145812/


## tests/test_output/adaptive_thresholds/20250426_145817/


## tests/test_output/adaptive_thresholds/20250426_150452/


## tests/test_output/adaptive_thresholds/20250426_150456/


## tests/test_output/adaptive_thresholds/20250426_150527/


## tests/test_output/adaptive_thresholds/20250426_150532/


## tests/test_output/adaptive_thresholds/20250426_151043/


## tests/test_output/adaptive_thresholds/20250426_151048/


## tests/test_output/end_to_end_report/


## tests/test_output/integration/

## tests/test_output/integration/direct_reports/


## tests/test_output/reports/


## tests/test_output/templates/


## utils/

### utils/__init__.py


### utils/file_utils.py

**Functions:**

- `ensure_dir()`: 確保目錄存在，若不存在則創建

Args:
    directory: 目錄路徑
    
Returns:
    str: 創建或確認的目錄路徑...
- `get_files_by_extension()`: 獲取指定目錄下特定副檔名的文件

Args:
    directory: 目錄路徑
    extension: 文件副檔名，例如 '...
- `load_json_file()`: 載入 JSON 檔案

Args:
    file_path: JSON 檔案路徑
    
Returns:
    Dict[str, Any]: 解析後的 JSON 內容

Raises:
    FileNotFoundError: 如果檔案不存在
    json...
- `save_json_file()`: 將數據保存為 JSON 檔案

Args:
    data: 要保存的數據
    file_path: 目標檔案路徑
    indent: JSON 縮排空格數...
- `load_pickle_file()`: 載入 pickle 檔案

Args:
    file_path: pickle 檔案路徑
    
Returns:
    Any: 解析後的 pickle 內容

Raises:
    FileNotFoundError: 如果檔案不存在...
- `save_pickle_file()`: 將數據保存為 pickle 檔案

Args:
    data: 要保存的數據
    file_path: 目標檔案路徑...
- `list_subdirectories()`: 列出指定目錄下的所有子目錄

Args:
    directory: 父目錄路徑
    
Returns:
    List[str]: 子目錄路徑列表...
- `find_files_recursive()`: 遞歸查找符合模式的文件

Args:
    directory: 起始目錄
    pattern: 文件名模式，例如 '*...
- `get_most_recent_file()`: 獲取指定目錄下最近修改的符合模式的文件

Args:
    directory: 目錄路徑
    pattern: 文件名模式，例如 '*...
- `get_file_size()`: 獲取文件大小

Args:
    file_path: 文件路徑
    unit: 單位，可選 'B', 'KB', 'MB', 'GB'
    
Returns:
    float: 文件大小

Raises:
    FileNotFoundError: 如果文件不存在...
- `merge_json_files()`: 合併多個 JSON 文件的內容

Args:
    file_paths: JSON 文件路徑列表
    
Returns:
    Dict[str, Any]: 合併後的 JSON 內容...
- `deep_merge()`: 深度合併兩個字典

Args:
    dict1: 第一個字典
    dict2: 第二個字典
    
Returns:
    Dict[str, Any]: 合併後的字典...


### utils/helpers.py

**Functions:**

- `framework_agnostic_to_numpy()`: Attempts to convert a tensor (PyTorch, TensorFlow) to a NumPy array...


### utils/stat_utils.py

**Functions:**

- `_new_tensor_init()`
- `_tensor_name()`
- `_tensor_set_name()`
- `calculate_distribution_metrics()`: 計算數據分佈的統計指標，用於分佈類型檢測。

Args:
    data (np...
- `calculate_distribution_stats()`: 計算張量的基本統計量和分佈特性。

Args:
    tensor (torch...
- `detect_distribution_type()`: 根據統計量判斷張量的分佈類型。

Args:
    tensor (torch...
- `compare_distributions()`: 比較兩個分佈的相似度

Args:
    dist1: 第一個分佈的張量或數組
    dist2: 第二個分佈的張量或數組
    method: 比較方法，可選 'ks'（Kolmogorov-Smirnov檢驗）或 'wasserstein'（Wasserstein距離）
    
Returns:
    Dict[str, Any]: 包含比較結果的字典，包括檢驗統計量、p值或距離...
- `perform_statistical_test()`: 執行統計檢驗比較兩個樣本

Args:
    sample1: 第一個樣本的張量或數組
    sample2: 第二個樣本的張量或數組
    test_type: 檢驗類型，可選 't_test', 'mann_whitney', 'kruskal'
    
Returns:
    Dict[str, Any]: 包含檢驗結果的字典，包括檢驗統計量、p值和結論...
- `detect_outliers()`: 使用指定方法檢測數值序列中的異常值。

Args:
    data (np...
- `calculate_moving_average()`: 計算移動平均值

Args:
    values: 輸入數值列表或numpy數組
    window_size: 窗口大小
    
Returns:
    np...
- `find_convergence_point()`: 查找序列的收斂點。
收斂定義為：在移動平均窗口內，值的相對變化率持續低於閾值。

Args:
    values: 數值序列（列表或numpy數組）
    window_size: 計算移動平均和變化率的窗口大小
    threshold: 判斷收斂的相對變化率閾值

Returns:
    int or None: 收斂點的索引（第一個滿足條件的點）。如果未收斂則返回 None。...
- `calculate_stability_metrics()`: 計算序列的穩定性指標。

Args:
    values: 數值序列（列表或numpy數組）
    window_size: 計算局部穩定性的窗口大小

Returns:
    dict: 包含穩定性指標的字典。
          {'overall_std', 'last_window_std', 'relative_std', 'last_window_mean'}...
- `detect_plateaus()`: 檢測序列中的平台期（值變化非常小）。

Args:
    values (list or np...
- `compare_distributions()`: 比較兩個分佈是否顯著不同。

Args:
    data1 (np...
- `detect_overfitting_simple()`: 檢測訓練中是否出現過擬合現象，通過對比訓練損失和驗證損失的趨勢。

Args:
    train_loss (List[float] or np...
- `_deprecated_detect_overfitting_simple()`: 已棄用: 使用上面的新實現替代

Args:
    train_loss (List[float] or np...
- `detect_distribution_type_advanced()`: 使用高階統計量和機器學習啟發的方法來判斷張量的分佈類型。

此函數是對基本detect_distribution_type的增強版本，增加了對以下項目的檢測：
- 更準確的稀疏性檢測
- 雙峰分佈檢測
- 基於熵的分佈區分
- 重尾分佈檢測
- 多模態分佈檢測

Args:
    tensor (torch...


### utils/tensor_utils.py

**Functions:**

- `to_numpy()`: 將張量轉換為NumPy陣列

Args:
    tensor: 輸入張量，可以是PyTorch張量、NumPy陣列或列表
    
Returns:
    np...
- `normalize_tensor()`: 正規化張量數據

Args:
    tensor: 輸入張量
    method: 正規化方法，可選 'minmax'/'min_max', 'zscore'/'z_score', 'l1', 'l2'
    dim: 應用正規化的維度，None表示整個張量
    
Returns:
    與輸入相同類型的正規化後張量...
- `preprocess_activations()`: 預處理激活值數據

Args:
    activations: 輸入激活值張量
    flatten: 是否將激活值展平
    normalize: 正規化方法，None表示不進行正規化
    remove_outliers: 是否移除離群值
    outlier_threshold: 離群值閾值（標準差的倍數）
    
Returns:
    torch...
- `batch_to_features()`: 從批次數據中提取特徵

Args:
    batch_data: 批次數據字典
    feature_key: 特徵的鍵名
    
Returns:
    np...
- `get_tensor_statistics()`: 計算張量的基礎統計指標

Args:
    tensor: 輸入張量
    
Returns:
    Dict[str, float]: 包含平均值、標準差、最小值、最大值等統計信息的字典...
- `tensor_correlation()`: 計算兩個張量之間的皮爾森相關係數

Args:
    tensor1: 第一個張量
    tensor2: 第二個張量
    
Returns:
    float: 相關係數...
- `average_activations()`: 計算多個激活值的平均值

Args:
    activations_dict: 層名稱到激活值張量的映射字典
    
Returns:
    Dict[str, torch...
- `extract_feature_maps()`: 從激活值中提取最活躍的特徵圖

Args:
    activation: 激活值張量
    top_k: 需要提取的特徵圖數量
    
Returns:
    Tuple[torch...
- `handle_nan_values()`: 處理張量中的NaN值

Args:
    tensor: 輸入張量
    strategy: 處理策略，可選 'zero'(替換為0), 'mean'(替換為均值), 'median'(替換為中位數), 
             'fill'(替換為指定值), 'interpolate'(使用插值)
    fill_value: 當strategy為'fill'時使用的填充值
    
Returns:
    處理後的張量，與輸入類型相同...
- `reduce_dimensions()`: 對激活值進行維度縮減

Args:
    activation: 輸入張量，可以是PyTorch張量或NumPy陣列
    method: 縮減方法，可選 'mean', 'max', 'sum', 'flatten'
    dims: 需要縮減的維度列表，如果為None則根據activation的維度自動決定
    
Returns:
    縮減後的張量，與輸入類型相同...
- `handle_sparse_tensor()`: 處理稀疏張量

Args:
    tensor: 輸入張量
    threshold: 視為零的閾值
    strategy: 處理策略，可選 'keep'(保持原樣), 'remove_zeros'(移除零), 
              'compress'(稀疏表示)
    
Returns:
    處理後的張量或稀疏表示...
- `batch_process_activations()`: 批次處理多層激活值

Args:
    activations: 層名稱到激活值張量的映射字典
    operations: 要執行的操作列表，每個操作是一個字典，包含'type'和其他參數
               支持的操作類型：'normalize', 'preprocess', 'handle_nan',
               'reduce_dims', 'handle_sparse'
    
Returns:
    Dict[str, torch...
- `normalize_tensor_zscore()`: 使用Z分數方法正規化張量

Args:
    tensor: 輸入張量
    dim: 應用正規化的維度，None表示整個張量
    
Returns:
    與輸入相同類型的Z分數正規化後張量...
- `calculate_effective_rank()`: 計算張量的有效秩

有效秩是一個衡量張量中有效信息維度的指標，通過奇異值的正規化熵來計算。

Args:
    tensor: 輸入張量，將被扁平化為2D張量進行分析
    
Returns:
    float: 有效秩值...
- `calculate_sparsity()`: 計算張量的稀疏度（零值或接近零值的比例）

Args:
    tensor: 輸入張量
    threshold: 視為零的閾值
    
Returns:
    float: 稀疏度 (0...
- `flatten_tensor()`: 將張量展平為一維向量

Args:
    tensor: 輸入張量
    
Returns:
    一維展平後的張量...
- `sample_tensor()`: 從張量中隨機抽樣，用於處理大型張量。

Args:
    tensor: 輸入張量
    max_samples: 最大樣本數量
    
Returns:
    抽樣後的張量，保持原始類型...


## visualization/

### visualization/__init__.py


### visualization/distribution_plots.py

**Classes:**

- `DistributionPlotter` (extends BasePlotter): 分佈視覺化器，用於繪製各種分佈相關的圖表。

此類提供多種方法來視覺化權重、激活值和其他張量數據的分佈情況。

Attributes:
    output_dir (str): 圖表保存的輸出目錄。
    style (str): Matplotlib 繪圖風格。
    figsize (Tuple[int, int]): 圖表大小。
    dpi (int): 圖表 DPI。...
  - `plot_histogram()`: 繪製直方圖來視覺化數據分佈。

Args:
    data: 要繪製的數據，可以是 NumPy 數組、PyTorch 張量或列表。
    title (str, optional): 圖表標題。默認為 'Distribution Histogram'。
    xlabel (str, optional): X 軸標籤。默認為 'Value'。
    ylabel (str, optional): Y 軸標籤。默認為 'Frequency'。
    bins (int, optional): 直方圖的箱數。默認為 50。
    color (str, optional): 直方圖的顏色。默認為 '#2077B4'。
    kde (bool, optional): 是否顯示核密度估計曲線。默認為 True。
    filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
    show (bool, optional): 是否顯示圖表。默認為 True。
    
Returns:
    plt...
  - `plot_distribution_comparison()`: 繪製多個分佈的比較圖。

Args:
    data_dict: 字典，鍵為分佈名稱，值為數據。
    title (str, optional): 圖表標題。默認為 'Distribution Comparison'。
    xlabel (str, optional): X 軸標籤。默認為 'Value'。
    ylabel (str, optional): Y 軸標籤。默認為 'Density'。
    filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
    show (bool, optional): 是否顯示圖表。默認為 True。
    
Returns:
    plt...
  - `plot_boxplot()`: 繪製箱形圖比較不同類別的數據分佈。

Args:
    data: 字典（鍵為類別名稱，值為數據）或 DataFrame。
    title (str, optional): 圖表標題。默認為 'Boxplot Comparison'。
    xlabel (str, optional): X 軸標籤。默認為 'Category'。
    ylabel (str, optional): Y 軸標籤。默認為 'Value'。
    vert (bool, optional): 是否垂直繪製箱形圖。默認為 True。
    filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
    show (bool, optional): 是否顯示圖表。默認為 True。
    
Returns:
    plt...
  - `plot_heatmap()`: 繪製熱力圖來視覺化二維數據。

Args:
    data: 二維數據，可以是 NumPy 數組、PyTorch 張量或嵌套列表。
    title (str, optional): 圖表標題。默認為 'Distribution Heatmap'。
    xlabel (str, optional): X 軸標籤。默認為 'X'。
    ylabel (str, optional): Y 軸標籤。默認為 'Y'。
    cmap (str, optional): 顏色映射。默認為 'viridis'。
    annot (bool, optional): 是否在熱力圖中顯示數值。默認為 False。
    filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
    show (bool, optional): 是否顯示圖表。默認為 True。
    
Returns:
    plt...
  - `plot_qq_plot()`: 繪製 Q-Q 圖來檢查數據是否符合正態分佈。

Args:
    data: 一維數據，可以是 NumPy 數組、PyTorch 張量或列表。
    title (str, optional): 圖表標題。默認為 'Q-Q Plot'。
    filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
    show (bool, optional): 是否顯示圖表。默認為 True。
    
Returns:
    plt...


### visualization/layer_visualizer.py

**Functions:**

- `visualize_activation_distribution()`: 視覺化層級激活值分佈。

Args:
    activations: 層級激活值張量
    layer_name: 層名稱，用於圖表標題
    save_path: 保存圖表的路徑，若為None則不保存
    
Returns:
    matplotlib圖表物件...
- `visualize_feature_maps()`: 視覺化卷積層的特徵圖。

Args:
    activations: 卷積層級激活值張量 [batch, channels, height, width]
    layer_name: 層名稱，用於圖表標題
    sample_idx: 要顯示的樣本索引
    max_features: 最多顯示的特徵圖數量
    save_path: 保存圖表的路徑，若為None則不保存
    
Returns:
    matplotlib圖表物件，若輸入不是卷積層則返回None...
- `visualize_neuron_activity()`: 視覺化神經元活躍度，顯示最活躍和最不活躍的神經元。

Args:
    activations: 層級激活值張量
    layer_name: 層名稱，用於圖表標題
    top_k: 顯示的頂部和底部神經元數量
    threshold: 神經元被視為活躍的激活閾值
    save_path: 保存圖表的路徑，若為None則不保存
    
Returns:
    matplotlib圖表物件...
- `visualize_dead_and_saturated_neurons()`: 視覺化死亡和飽和神經元。

Args:
    activations: 層級激活值張量
    layer_name: 層名稱，用於圖表標題
    dead_threshold: 判定神經元死亡的最大激活閾值
    saturation_threshold: 判定神經元飽和的閾值
    save_path: 保存圖表的路徑，若為None則不保存
    
Returns:
    matplotlib圖表物件...
- `visualize_activation_across_samples()`: 視覺化不同樣本間的激活值變化。

Args:
    activations: 層級激活值張量
    layer_name: 層名稱，用於圖表標題
    max_samples: 最多顯示的樣本數量
    save_path: 保存圖表的路徑，若為None則不保存
    
Returns:
    matplotlib圖表物件...
- `visualize_layer_metrics()`: 為層級激活值生成一整套可視化圖表。

Args:
    activations: 層級激活值張量
    layer_name: 層名稱，用於圖表標題和文件名
    output_dir: 保存圖表的目錄
    include_feature_maps: 是否包括特徵圖可視化（僅適用於卷積層）
    
Returns:
    包含所有生成圖表路徑的字典...


### visualization/model_structure_plots.py

**Classes:**

- `ModelStructurePlotter` (extends BasePlotter): 模型結構視覺化工具，用於繪製模型架構圖、參數分布等圖表。

Attributes:
    default_figsize (Tuple[int, int]): 默認圖表大小。
    default_dpi (int): 默認解析度。
    layer_colors (Dict[str, str]): 不同層類型的顏色映射。...
  - `_get_layer_color()`: 根據層類型獲取對應的顏色。

Args:
    layer_type (str): 層類型。
    
Returns:
    str: 顏色代碼。...
  - `plot_model_structure()`: 繪製模型架構圖。

Args:
    hierarchy (Dict): 模型層次結構，包含layers和connections。
    figsize (Tuple[int, int], optional): 圖表大小。默認為None，使用默認大小。
    show_params (bool, optional): 是否顯示參數數量。默認為True。
    highlight_large_layers (bool, optional): 是否突出參數量大的層。默認為True。
    title (str, optional): 圖表標題。默認為None。
    
Returns:
    Figure: Matplotlib圖表對象。...
  - `plot_parameter_distribution()`: 繪製模型參數分布圖。

Args:
    param_distribution (Dict): 參數分布數據。
    figsize (Tuple[int, int], optional): 圖表大小。默認為None，使用默認大小。
    title (str, optional): 圖表標題。默認為None。
    
Returns:
    Figure: Matplotlib圖表對象。...
  - `plot_layer_complexity()`: 繪製層級複雜度圖。

Args:
    layer_complexity (Dict): 層級複雜度數據。
    figsize (Tuple[int, int], optional): 圖表大小。默認為None，使用默認大小。
    title (str, optional): 圖表標題。默認為None。
    
Returns:
    Figure: Matplotlib圖表對象。...
  - `plot_sequential_paths()`: 繪製模型中的主要路徑圖。

Args:
    sequential_paths (List[Dict]): 順序執行路徑數據。
    figsize (Tuple[int, int], optional): 圖表大小。默認為None，使用默認大小。
    max_paths (int, optional): 最多顯示多少條路徑。默認為3。
    title (str, optional): 圖表標題。默認為None。
    
Returns:
    Figure: Matplotlib圖表對象。...


### visualization/performance_plots.py

**Classes:**

- `PerformancePlotter` (extends BasePlotter): 性能視覺化器，用於繪製訓練過程中的性能指標和動態。

此類提供多種方法來視覺化訓練過程中的指標變化，如損失曲線、學習率變化等。

Attributes:
    output_dir (str): 圖表保存的輸出目錄。
    style (str): Matplotlib 繪圖風格。
    figsize (Tuple[int, int]): 圖表大小。
    dpi (int): 圖表 DPI。...
  - `plot_loss_curve()`: 繪製訓練和驗證損失曲線。

Args:
    loss_history (Dict[str, List[float]]): 損失歷史記錄，例如 {'train': [...
  - `plot_metric_curves()`: 繪製多個性能指標的曲線圖。

Args:
    metric_history: 指標歷史記錄，例如 {'train': {'acc': [...
  - `plot_learning_rate()`: 繪製學習率變化曲線。

Args:
    lr_history (List[float]): 學習率歷史記錄。
    title (str, optional): 圖表標題。默認為 'Learning Rate Schedule'。
    xlabel (str, optional): X 軸標籤。默認為 'Epoch'。
    ylabel (str, optional): Y 軸標籤。默認為 'Learning Rate'。
    log_scale (bool, optional): 是否使用對數尺度。默認為 True。
    filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
    show (bool, optional): 是否顯示圖表。默認為 True。
    
Returns:
    plt...
  - `plot_convergence_analysis()`: 繪製收斂分析圖，包括收斂點標記。

Args:
    loss_history (List[float]): 損失歷史記錄。
    convergence_epoch (int): 收斂點的輪次。
    title (str, optional): 圖表標題。默認為 'Convergence Analysis'。
    xlabel (str, optional): X 軸標籤。默認為 'Epoch'。
    ylabel (str, optional): Y 軸標籤。默認為 'Loss'。
    filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
    show (bool, optional): 是否顯示圖表。默認為 True。
    
Returns:
    plt...
  - `plot_training_stability()`: 繪製訓練穩定性分析圖，包括移動平均和波動性。

Args:
    loss_history (List[float]): 損失歷史記錄。
    moving_avg_window (int, optional): 移動平均的窗口大小。默認為 5。
    title (str, optional): 圖表標題。默認為 'Training Stability Analysis'。
    xlabel (str, optional): X 軸標籤。默認為 'Epoch'。
    ylabel (str, optional): Y 軸標籤。默認為 'Loss'。
    filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
    show (bool, optional): 是否顯示圖表。默認為 True。
    
Returns:
    plt...
  - `plot_gradient_norm()`: 繪製梯度範數歷史曲線。

Args:
    gradient_norms (List[float]): 梯度範數歷史記錄。
    title (str, optional): 圖表標題。默認為 'Gradient Norm History'。
    xlabel (str, optional): X 軸標籤。默認為 'Epoch'。
    ylabel (str, optional): Y 軸標籤。默認為 'Gradient Norm'。
    log_scale (bool, optional): 是否使用對數尺度。默認為 False。
    filename (str, optional): 保存文件名。若為 None，則使用標題作為文件名。
    show (bool, optional): 是否顯示圖表。默認為 True。
    
Returns:
    plt...
  - `plot_training_heatmap()`: 繪製訓練進度熱力圖，每個單元格表示特定輪次和批次的某個指標。

Args:
    epoch_batch_data (np...


### visualization/plotter.py

**Classes:**

- `BasePlotter`: 繪圖器基類，提供所有繪圖器共用的基本功能。

Attributes:
    output_dir (str): 圖表保存的輸出目錄。
    style (str): Matplotlib 繪圖風格。
    figsize (Tuple[int, int]): 圖表大小。
    dpi (int): 圖表 DPI。...
  - `_save_fig()`: 保存圖表到文件。

Args:
    fig (plt...
  - `set_output_dir()`: 設置輸出目錄。

Args:
    output_dir (str): 新的輸出目錄。...
  - `set_style()`: 設置繪圖風格。

Args:
    style (str): Matplotlib 繪圖風格。...
  - `set_figsize()`: 設置圖表大小。

Args:
    figsize (Tuple[int, int]): 圖表大小，如 (寬度, 高度)。...
  - `set_dpi()`: 設置圖表 DPI。

Args:
    dpi (int): 圖表 DPI。...
  - `plot_histogram()`: 繪製直方圖。

Args:
    data (array-like): 要繪製的數據。
    title (str, optional): 圖表標題。默認為 'Histogram'。
    bins (int, optional): 直方圖的箱數。默認為 50。
    xlabel (str, optional): X 軸標籤。默認為 'Value'。
    ylabel (str, optional): Y 軸標籤。默認為 'Frequency'。
    filename (str, optional): 保存文件名。若為 None，則不保存。
    show (bool, optional): 是否顯示圖表。默認為 True。
    
Returns:
    plt...
  - `plot_waveform()`: 繪製波形圖。

Args:
    waveform (array-like): 一維音頻波形數據。
    sample_rate (int): 音頻的採樣率。
    title (str, optional): 圖表標題。默認為 'Waveform'。
    xlabel (str, optional): X 軸標籤。默認為 'Time (s)'。
    ylabel (str, optional): Y 軸標籤。默認為 'Amplitude'。
    filename (str, optional): 保存文件名。若為 None，則不保存。
    show (bool, optional): 是否顯示圖表。默認為 True。
    
Returns:
    plt...

