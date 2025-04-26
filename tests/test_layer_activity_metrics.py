"""
# 測試層級活動指標計算功能
"""
import pytest
import torch
import numpy as np
from metrics.layer_activity_metrics import (
    calculate_activation_statistics,
    calculate_activation_sparsity,
    calculate_activation_saturation,
    calculate_effective_rank,
    calculate_feature_coherence,
    detect_dead_neurons,
    calculate_activation_dynamics,
    calculate_layer_similarity,
    detect_activation_anomalies,
    analyze_layer_relationships
)


def test_calculate_activation_statistics():
    """測試計算激活值統計指標"""
    # 創建隨機激活值
    activation = torch.randn(32, 64, 16, 16)
    
    # 計算統計量
    stats = calculate_activation_statistics(activation)
    
    # 檢查返回值包含預期的鍵
    expected_keys = ['mean', 'std', 'min', 'max', 'median', 'q1', 'q3', 
                     'iqr', 'skewness', 'kurtosis', 'range', 'variance', 
                     'sparsity', 'positive_fraction', 'entropy']
    for key in expected_keys:
        assert key in stats
    
    # 檢查值是否合理
    assert -1 <= stats['mean'] <= 1  # 標準正態分布的均值接近0
    assert 0 <= stats['std'] <= 2    # 標準正態分布的標準差接近1
    assert stats['min'] <= stats['q1'] <= stats['median'] <= stats['q3'] <= stats['max']
    
    # 檢查range是否接近max-min，允許1e-5的浮點誤差
    expected_range = stats['max'] - stats['min']
    assert abs(stats['range'] - expected_range) < 1e-5, f"Range {stats['range']} should equal max-min {expected_range}"
    
    assert 0 <= stats['sparsity'] <= 1
    assert 0 <= stats['positive_fraction'] <= 1


def test_calculate_activation_sparsity():
    """測試計算激活值稀疏度"""
    # 創建不同稀疏度的激活值
    zeros = torch.zeros(100, 100)
    sparse = torch.zeros(100, 100)
    sparse[torch.randperm(100)[:10], torch.randperm(100)[:10]] = 1  # 10% 非零
    dense = torch.ones(100, 100)
    
    # 計算稀疏度
    sparsity_zeros = calculate_activation_sparsity(zeros)
    sparsity_sparse = calculate_activation_sparsity(sparse)
    sparsity_dense = calculate_activation_sparsity(dense)
    
    # 檢查結果
    assert sparsity_zeros == 1.0  # 全零應該是100%稀疏
    assert 0.8 <= sparsity_sparse <= 1.0  # 大約90%稀疏
    assert sparsity_dense == 0.0  # 全一應該是0%稀疏


def test_calculate_activation_saturation():
    """測試計算激活值飽和度"""
    # 創建不同飽和度的激活值
    zeros = torch.zeros(100, 100)
    mid_values = torch.ones(100, 100) * 0.5
    ones = torch.ones(100, 100)
    
    # 計算飽和度
    saturation_zeros = calculate_activation_saturation(zeros)
    saturation_mid = calculate_activation_saturation(mid_values)
    saturation_ones = calculate_activation_saturation(ones)
    
    # 檢查結果
    assert saturation_zeros == 0.0  # 全零應該是0%飽和
    assert saturation_mid == 0.0    # 0.5小於預設閾值0.9
    assert saturation_ones == 1.0   # 全一應該是100%飽和


def test_calculate_effective_rank():
    """測試計算激活值有效秩"""
    # 創建不同秩的矩陣
    low_rank = torch.zeros(100, 100)
    low_rank[:, :5] = torch.randn(100, 5)  # 秩為5
    
    full_rank = torch.randn(50, 50)  # 可能是滿秩的
    
    # 計算有效秩
    rank_low = calculate_effective_rank(low_rank)
    rank_full = calculate_effective_rank(full_rank)
    
    # 檢查結果
    assert 1 <= rank_low <= 10  # 有效秩應該接近但不完全等於5
    assert rank_low < rank_full  # 滿秩矩陣的有效秩應該更高


def test_calculate_feature_coherence():
    """測試計算特徵一致性"""
    # 創建具有不同一致性的特徵圖
    batch_size = 16
    channels = 32
    height = width = 8
    
    # 創建高度相關的特徵圖（每個通道都有相似的模式）
    high_coherence = torch.randn(batch_size, 1, height, width)
    high_coherence = high_coherence.repeat(1, channels, 1, 1)
    high_coherence += torch.randn(batch_size, channels, height, width) * 0.1  # 添加少量噪聲
    
    # 創建低相關的特徵圖（每個通道都是獨立的）
    low_coherence = torch.randn(batch_size, channels, height, width)
    
    # 計算一致性
    coherence_high = calculate_feature_coherence(high_coherence)
    coherence_low = calculate_feature_coherence(low_coherence)
    
    # 檢查結果
    assert -1 <= coherence_high <= 1
    assert -1 <= coherence_low <= 1
    assert coherence_high > coherence_low  # 高相關特徵的一致性應該更高


def test_detect_dead_neurons():
    """測試檢測失活神經元"""
    # 創建具有不同比例失活神經元的激活值
    all_active = torch.ones(32, 100)
    half_dead = torch.ones(32, 100)
    half_dead[:, :50] = 0  # 前50個神經元失活
    all_dead = torch.zeros(32, 100)
    
    # 檢測失活神經元
    active_info = detect_dead_neurons(all_active)
    half_info = detect_dead_neurons(half_dead)
    dead_info = detect_dead_neurons(all_dead)
    
    # 檢查結果
    assert active_info['dead_ratio'] == 0.0  # 所有神經元都活躍
    assert half_info['dead_ratio'] == 0.5    # 50%神經元失活
    assert dead_info['dead_ratio'] == 1.0    # 所有神經元都失活


def test_calculate_activation_dynamics():
    """測試計算激活值動態變化"""
    # 創建不同輪次的激活值
    epochs = {
        0: torch.randn(16, 32),
        1: torch.randn(16, 32) * 1.2,
        2: torch.randn(16, 32) * 1.5
    }
    
    # 計算動態變化
    dynamics = calculate_activation_dynamics(epochs)
    
    # 檢查結果
    assert 'epochs' in dynamics
    assert 'mean_changes' in dynamics
    assert 'std_changes' in dynamics
    assert len(dynamics['mean_changes']) == 2  # 3個輪次，有2個變化值
    assert 'mean_change_rate' in dynamics


def test_calculate_layer_similarity():
    """測試計算層間相似度"""
    # 創建不同相似度的層
    layer1 = torch.randn(16, 100)
    layer2 = layer1 + torch.randn(16, 100) * 0.1  # 與layer1高度相似
    layer3 = torch.randn(16, 100)                 # 與layer1不相關
    
    # 計算相似度
    sim_high = calculate_layer_similarity(layer1, layer2)
    sim_low = calculate_layer_similarity(layer1, layer3)
    
    # 檢查結果
    assert 'mean_correlation' in sim_high and 'mean_correlation' in sim_low
    assert 'cosine_similarity' in sim_high and 'cosine_similarity' in sim_low
    assert sim_high['cosine_similarity'] > sim_low['cosine_similarity']


def test_handle_different_input_formats():
    """測試處理不同輸入格式"""
    # 測試PyTorch張量輸入
    torch_tensor = torch.randn(16, 32)
    stats_torch = calculate_activation_statistics(torch_tensor)
    
    # 測試NumPy陣列輸入
    numpy_array = np.random.randn(16, 32)
    stats_numpy = calculate_activation_statistics(numpy_array)
    
    # 檢查兩者都能正確處理
    assert 'mean' in stats_torch
    assert 'mean' in stats_numpy


def test_detect_activation_anomalies():
    """測試檢測異常激活模式功能"""
    # 創建正常激活值（近似正態分布）
    normal_activation = torch.randn(32, 64, 16, 16)
    
    # 創建異常激活值（引入極端值）
    abnormal_activation = torch.randn(32, 64, 16, 16)
    abnormal_activation[0, 0, 0, 0] = 1000.0  # 引入極端值
    
    # 創建偏斜分布激活值
    skewed_activation = torch.pow(torch.abs(torch.randn(32, 64, 16, 16)), 2)  # 平方會使分布右偏
    
    # 創建高峰度激活值（多極端值）
    high_kurtosis = torch.randn(32, 64, 16, 16)
    extreme_positions = torch.randint(0, 32, (100,)), torch.randint(0, 64, (100,)), torch.randint(0, 16, (100,)), torch.randint(0, 16, (100,))
    high_kurtosis[extreme_positions] = torch.randn(100) * 10  # 在隨機位置設置極端值
    
    # 創建梯度異常激活值
    gradient_abnormal = torch.ones(32, 64, 16, 16)
    gradient_abnormal[0, 0, 8, 8] = 100.0  # 創建一個尖峰，產生大梯度
    
    # 檢測異常
    normal_result = detect_activation_anomalies(normal_activation)
    abnormal_result = detect_activation_anomalies(abnormal_activation)
    skewed_result = detect_activation_anomalies(skewed_activation)
    kurtosis_result = detect_activation_anomalies(high_kurtosis)
    gradient_result = detect_activation_anomalies(gradient_abnormal)
    
    # 檢查結果
    # 1. 確認返回正確的字典鍵
    expected_keys = ['has_anomaly', 'anomaly_score', 'extreme_value_ratio', 
                     'skewness', 'kurtosis', 'skewness_abnormal', 'kurtosis_abnormal',
                     'gradient_abnormal_ratio', 'concentration_abnormal', 'extreme_value_examples']
    
    for key in expected_keys:
        assert key in normal_result
        assert key in abnormal_result
    
    # 2. 確認異常檢測結果合理
    # 正常激活值應該沒有或很少異常
    assert normal_result['anomaly_score'] < 0.5
    
    # 有極端值的激活應該檢測到異常
    assert abnormal_result['has_anomaly']
    assert abnormal_result['extreme_value_ratio'] > 0
    assert 1000.0 in abnormal_result['extreme_value_examples'] or abs(1000.0 - abnormal_result['extreme_value_examples'][0]) < 1.0
    
    # 偏斜分布應該有較高的偏斜度
    assert abs(skewed_result['skewness']) > abs(normal_result['skewness'])
    
    # 高峰度激活值應該有較高的峰度
    assert kurtosis_result['anomaly_score'] > normal_result['anomaly_score']
    
    # 梯度異常激活值應該檢測到梯度問題
    assert gradient_result['has_anomaly']
    assert gradient_result['gradient_abnormal_ratio'] > 0


def test_analyze_layer_relationships():
    """測試分析層與層之間關係功能"""
    # 創建多個層的激活值
    batch_size = 16
    feature_size = 32
    
    # 創建基準層
    base_layer = torch.randn(batch_size, feature_size)
    
    # 創建高度相關的層
    highly_correlated = base_layer + torch.randn(batch_size, feature_size) * 0.1
    
    # 創建低相關的層
    low_correlated = torch.randn(batch_size, feature_size)
    
    # 創建一個4D激活值（用於測試結構相似性）
    conv_layer1 = torch.randn(batch_size, 8, 8, 8)
    conv_layer2 = torch.randn(batch_size, 8, 8, 8)
    
    # 創建層激活值字典
    activations_dict = {
        'base': base_layer,
        'high_corr': highly_correlated,
        'low_corr': low_correlated,
        'conv1': conv_layer1,
        'conv2': conv_layer2
    }
    
    # 測試不同的指標組合
    # 1. 只使用相關性指標
    corr_result = analyze_layer_relationships(
        activations_dict,
        metrics=['correlation']
    )
    
    # 2. 使用所有指標
    full_result = analyze_layer_relationships(
        activations_dict,
        metrics=['correlation', 'mutual_information', 'structural_similarity']
    )
    
    # 3. 指定參考層
    ref_result = analyze_layer_relationships(
        activations_dict,
        reference_layer='base',
        metrics=['correlation']
    )
    
    # 檢查結果
    # 1. 確認結果包含預期的鍵
    assert 'layer_pairs' in corr_result
    assert 'metrics' in corr_result
    assert 'layer_graph' in corr_result
    
    # 2. 確認度量結果合理
    assert 'correlation' in corr_result['metrics']
    assert len(corr_result['layer_pairs']) == len(corr_result['metrics']['correlation'])
    
    # 檢查完整結果包含所有請求的度量
    assert 'correlation' in full_result['metrics']
    assert 'mutual_information' in full_result['metrics']
    assert 'structural_similarity' in full_result['metrics']
    
    # 3. 驗證參考層模式
    assert len(ref_result['layer_pairs']) == 4  # 4對(base與其他4層)
    for pair in ref_result['layer_pairs']:
        assert 'base' in pair
    
    # 4. 驗證高相關層被正確標識
    base_high_pair = ('base', 'high_corr')
    base_low_pair = ('base', 'low_corr')
    
    # 查找這些對應的索引
    base_high_idx = -1
    base_low_idx = -1
    
    for i, pair in enumerate(full_result['layer_pairs']):
        if set(pair) == set(base_high_pair):
            base_high_idx = i
        elif set(pair) == set(base_low_pair):
            base_low_idx = i
    
    # 確保找到了這些對
    assert base_high_idx != -1
    assert base_low_idx != -1
    
    # 驗證高相關層對的分數高於低相關層對
    assert full_result['metrics']['correlation'][base_high_idx] > full_result['metrics']['correlation'][base_low_idx]
    
    # 5. 驗證層級關係圖
    assert 'base' in full_result['layer_graph']
    assert 'high_corr' in full_result['layer_graph']
    assert 'high_corr' in full_result['layer_graph']['base']


def test_error_handling():
    """測試錯誤處理"""
    # 測試錯誤的輸入維度
    with pytest.raises(ValueError):
        calculate_feature_coherence(torch.randn(16, 32))  # 維度不夠
    
    # 測試異常形狀
    strange_shape = torch.randn(1, 1, 1)
    # 這應該不會引發錯誤，但可能會返回特殊值
    result = calculate_effective_rank(strange_shape)
    assert result == 1.0  # 單元素矩陣的有效秩應為1 