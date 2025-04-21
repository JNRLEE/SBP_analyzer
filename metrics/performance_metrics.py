"""
性能度量模組。

此模組提供了用於評估模型性能的各種度量方法。
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Union, Tuple, Any

# 將相對導入改為絕對導入
from utils.tensor_utils import tensor_to_numpy, flatten_tensor
from utils.stat_utils import calculate_moving_average, detect_outliers, find_convergence_point

logger = logging.getLogger(__name__)

def calculate_error_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    計算各種錯誤度量指標。
    
    Args:
        predictions (np.ndarray): 模型預測值。
        targets (np.ndarray): 目標真實值。
        
    Returns:
        Dict[str, float]: 包含各種錯誤度量的字典。
    """
    # 確保輸入是numpy數組
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    
    # 計算各種錯誤度量
    abs_error = np.abs(predictions - targets)
    squared_error = np.square(predictions - targets)
    
    metrics = {
        'mae': float(np.mean(abs_error)),  # 平均絕對誤差
        'mse': float(np.mean(squared_error)),  # 均方誤差
        'rmse': float(np.sqrt(np.mean(squared_error))),  # 均方根誤差
        'max_error': float(np.max(abs_error)),  # 最大絕對誤差
        'min_error': float(np.min(abs_error)),  # 最小絕對誤差
    }
    
    # 如果目標值不全為零，還可以計算相對錯誤
    if not np.all(targets == 0):
        rel_error = abs_error / (np.abs(targets) + 1e-10)
        metrics['mape'] = float(np.mean(rel_error) * 100)  # 平均絕對百分比誤差
    
    return metrics

def calculate_classification_metrics(predictions: np.ndarray, targets: np.ndarray, 
                                    threshold: float = 0.5) -> Dict[str, float]:
    """
    計算二元分類度量指標。
    
    Args:
        predictions (np.ndarray): 模型預測概率，範圍在[0, 1]。
        targets (np.ndarray): 目標真實類別，值為0或1。
        threshold (float, optional): 將預測概率轉換為類別的閾值。默認為0.5。
        
    Returns:
        Dict[str, float]: 包含各種分類度量的字典。
    """
    # 確保輸入是numpy數組
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    
    # 將預測概率轉換為類別
    pred_classes = (predictions >= threshold).astype(int)
    
    # 計算混淆矩陣相關指標
    tp = np.sum((pred_classes == 1) & (targets == 1))
    tn = np.sum((pred_classes == 0) & (targets == 0))
    fp = np.sum((pred_classes == 1) & (targets == 0))
    fn = np.sum((pred_classes == 0) & (targets == 1))
    
    # 防止除零
    epsilon = 1e-10
    
    # 計算各種分類度量
    metrics = {
        'accuracy': float((tp + tn) / (tp + tn + fp + fn + epsilon)),
        'precision': float(tp / (tp + fp + epsilon)),
        'recall': float(tp / (tp + fn + epsilon)),
        'f1_score': float(2 * tp / (2 * tp + fp + fn + epsilon)),
        'specificity': float(tn / (tn + fp + epsilon)),
    }
    
    return metrics

def calculate_regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    計算回歸模型的性能度量。
    
    Args:
        predictions (np.ndarray): 模型預測值。
        targets (np.ndarray): 目標真實值。
        
    Returns:
        Dict[str, float]: 包含各種回歸度量的字典。
    """
    # 首先計算基本錯誤度量
    metrics = calculate_error_metrics(predictions, targets)
    
    # 計算R方值（確定係數）
    y_mean = np.mean(targets)
    total_sum_squares = np.sum(np.square(targets - y_mean))
    residual_sum_squares = np.sum(np.square(targets - predictions))
    
    if total_sum_squares > 0:
        r_squared = 1.0 - (residual_sum_squares / total_sum_squares)
        metrics['r_squared'] = float(r_squared)
    else:
        metrics['r_squared'] = 1.0  # 如果總平方和為零，則R方為1
    
    # 計算調整後的R方
    n = len(targets)
    if n > 1:  # 需要至少兩個樣本
        p = 1  # 特徵數量（這裡假設為1）
        if n > p + 1:
            adj_r_squared = 1 - (1 - metrics['r_squared']) * ((n - 1) / (n - p - 1))
            metrics['adjusted_r_squared'] = float(adj_r_squared)
    
    return metrics

def evaluate_convergence(loss_values: List[float], window_size: int = 5, 
                         threshold: float = 0.01) -> Dict[str, Any]:
    """
    評估訓練過程中的收斂情況。
    
    Args:
        loss_values (List[float]): 訓練過程中的損失值列表。
        window_size (int, optional): 用於計算移動平均的窗口大小。默認為5。
        threshold (float, optional): 判斷收斂的閾值。默認為0.01。
        
    Returns:
        Dict[str, Any]: 包含收斂相關指標的字典。
    """
    if not loss_values:
        return {'has_converged': False, 'convergence_point': None, 'final_loss': None}
    
    # 計算移動平均來平滑損失曲線
    smoothed_losses = calculate_moving_average(loss_values, window_size)
    
    # 找出可能的收斂點
    convergence_info = {}
    
    # 判斷是否收斂
    has_converged = False
    convergence_point = None
    
    if len(smoothed_losses) >= 2 * window_size:
        # 檢查最後window_size個值的變化
        last_window = smoothed_losses[-window_size:]
        max_diff = max(last_window) - min(last_window)
        
        if max_diff < threshold:
            has_converged = True
            
            # 尋找收斂點（損失降低幅度首次小於閾值的點）
            for i in range(window_size, len(smoothed_losses) - 1):
                diff = abs(smoothed_losses[i] - smoothed_losses[i-1])
                if diff < threshold:
                    convergence_point = i
                    break
    
    convergence_info = {
        'has_converged': has_converged,
        'convergence_point': convergence_point,
        'final_loss': float(loss_values[-1]) if loss_values else None,
        'min_loss': float(min(loss_values)) if loss_values else None,
        'loss_trajectory': smoothed_losses
    }
    
    return convergence_info

def detect_performance_anomalies(metric_values: List[float], detection_method: str = 'iqr',
                                threshold: float = 1.5) -> Dict[str, Any]:
    """
    檢測性能度量中的異常值。
    
    Args:
        metric_values (List[float]): 要分析的性能度量值列表。
        detection_method (str, optional): 異常檢測方法，'iqr'或'zscore'。默認為'iqr'。
        threshold (float, optional): 異常檢測閾值。默認為1.5。
        
    Returns:
        Dict[str, Any]: 包含異常檢測結果的字典。
    """
    anomalies = detect_outliers(metric_values, method=detection_method, threshold=threshold)
    
    # 計算異常值的統計信息
    anomaly_indices = []
    anomaly_values = []
    
    for i, is_anomaly in enumerate(anomalies):
        if is_anomaly:
            anomaly_indices.append(i)
            anomaly_values.append(metric_values[i])
    
    result = {
        'has_anomalies': len(anomaly_indices) > 0,
        'anomaly_count': len(anomaly_indices),
        'anomaly_indices': anomaly_indices,
        'anomaly_values': anomaly_values,
        'anomaly_percentage': len(anomaly_indices) / len(metric_values) if metric_values else 0
    }
    
    return result

def compare_model_performances(model1_metrics: Dict[str, float], 
                              model2_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    比較兩個模型的性能指標。
    
    Args:
        model1_metrics (Dict[str, float]): 第一個模型的性能度量字典。
        model2_metrics (Dict[str, float]): 第二個模型的性能度量字典。
        
    Returns:
        Dict[str, Any]: 包含兩個模型性能比較的字典。
    """
    comparison = {}
    
    # 獲取兩個模型共有的度量指標
    common_metrics = set(model1_metrics.keys()).intersection(set(model2_metrics.keys()))
    
    for metric in common_metrics:
        val1 = model1_metrics[metric]
        val2 = model2_metrics[metric]
        
        # 計算差異
        abs_diff = abs(val1 - val2)
        rel_diff = abs_diff / (abs(val1) + 1e-10) * 100
        
        # 判斷哪個模型更好（假設值越小越好，例如誤差）
        better_model = 1 if val1 < val2 else 2 if val2 < val1 else 0
        
        comparison[metric] = {
            'model1_value': val1,
            'model2_value': val2,
            'absolute_difference': abs_diff,
            'relative_difference_percent': rel_diff,
            'better_model': better_model
        }
    
    return comparison 

def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    計算分類問題的準確率。
    
    Args:
        predictions (np.ndarray): 預測類別的數組
        targets (np.ndarray): 真實類別的數組
    
    Returns:
        float: 準確率 (0到1之間)
    """
    # 確保輸入是numpy數組
    preds = tensor_to_numpy(predictions).flatten()
    tgts = tensor_to_numpy(targets).flatten()
    
    if len(preds) != len(tgts):
        raise ValueError("預測和目標數組的長度必須相同")
    
    # 計算正確預測的比例
    correct = np.sum(preds == tgts)
    return float(correct / len(tgts))

def calculate_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    計算均方誤差 (MSE)。
    
    Args:
        predictions (np.ndarray): 預測值數組
        targets (np.ndarray): 目標值數組
    
    Returns:
        float: MSE值
    """
    # 確保輸入是numpy數組
    preds = tensor_to_numpy(predictions).flatten()
    tgts = tensor_to_numpy(targets).flatten()
    
    if len(preds) != len(tgts):
        raise ValueError("預測和目標數組的長度必須相同")
    
    # 計算MSE
    mse = np.mean((preds - tgts) ** 2)
    return float(mse)

def calculate_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    計算平均絕對誤差 (MAE)。
    
    Args:
        predictions (np.ndarray): 預測值數組
        targets (np.ndarray): 目標值數組
    
    Returns:
        float: MAE值
    """
    # 確保輸入是numpy數組
    preds = tensor_to_numpy(predictions).flatten()
    tgts = tensor_to_numpy(targets).flatten()
    
    if len(preds) != len(tgts):
        raise ValueError("預測和目標數組的長度必須相同")
    
    # 計算MAE
    mae = np.mean(np.abs(preds - tgts))
    return float(mae)

def calculate_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    計算均方根誤差 (RMSE)。
    
    Args:
        predictions (np.ndarray): 預測值數組
        targets (np.ndarray): 目標值數組
    
    Returns:
        float: RMSE值
    """
    # 利用MSE計算RMSE
    mse = calculate_mse(predictions, targets)
    rmse = np.sqrt(mse)
    return float(rmse)

def calculate_convergence_metrics(train_loss: List[float], 
                                val_loss: List[float] = None, 
                                threshold: float = 0.01,
                                window_size: int = 10) -> Dict[str, Union[int, float, None]]:
    """
    計算訓練和驗證收斂相關的指標。
    
    Args:
        train_loss (List[float]): 訓練損失值列表
        val_loss (List[float], optional): 驗證損失值列表。默認為None。
        threshold (float, optional): 判定收斂的變化閾值。默認為0.01。
        window_size (int, optional): 用於計算移動平均的窗口大小。默認為10。
    
    Returns:
        Dict[str, Union[int, float, None]]: 包含以下字段的字典：
            - train_converged (bool): 訓練損失是否已收斂
            - train_convergence_epoch (int): 訓練損失收斂點的索引，如果沒有收斂則為None
            - val_converged (bool): 驗證損失是否已收斂，如果沒有提供驗證損失則為None
            - val_convergence_epoch (int): 驗證損失收斂點的索引，如果沒有收斂或沒有提供驗證損失則為None
            - overfitting_detected (bool): 是否檢測到過擬合
            - generalization_gap (float): 泛化間隔（驗證損失和訓練損失之間的差距）
    """
    result = {}
    
    # 分析訓練損失收斂情況
    train_conv_point = find_convergence_point(train_loss, threshold=threshold, window_size=window_size)
    result['train_converged'] = train_conv_point is not None
    result['train_convergence_epoch'] = train_conv_point
    
    # 如果提供了驗證損失，也分析其收斂情況
    if val_loss is not None:
        val_conv_point = find_convergence_point(val_loss, threshold=threshold, window_size=window_size)
        result['val_converged'] = val_conv_point is not None
        result['val_convergence_epoch'] = val_conv_point
        
        # 檢查過擬合：驗證損失開始上升而訓練損失繼續下降
        overfitting = False
        if len(val_loss) > window_size and len(train_loss) == len(val_loss):
            # 檢查驗證損失是否在某點之後開始上升
            min_val_idx = val_loss.index(min(val_loss))
            if min_val_idx < len(val_loss) - window_size:
                recent_val = val_loss[min_val_idx:]
                recent_train = train_loss[min_val_idx:]
                
                # 驗證損失上升趨勢
                val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]
                # 訓練損失趨勢
                train_trend = np.polyfit(range(len(recent_train)), recent_train, 1)[0]
                
                # 如果驗證損失上升而訓練損失下降，可能是過擬合
                overfitting = val_trend > 0 and train_trend < 0
        
        result['overfitting_detected'] = overfitting
        
        # 計算泛化間隔
        if len(val_loss) == len(train_loss):
            gen_gap = sum(v - t for v, t in zip(val_loss, train_loss)) / len(val_loss)
            result['generalization_gap'] = gen_gap
        else:
            result['generalization_gap'] = None
    else:
        result['val_converged'] = None
        result['val_convergence_epoch'] = None
        result['overfitting_detected'] = None
        result['generalization_gap'] = None
    
    return result

def calculate_improvement_rate(values: List[float], window_size: int = 10) -> float:
    """
    計算一個序列的改進率，即最近窗口中的平均變化率。
    
    Args:
        values (List[float]): 要分析的值列表，通常是損失或精度隨時間的變化
        window_size (int, optional): 用於計算改進率的窗口大小。默認為10。
    
    Returns:
        float: 改進率，正值表示改進，負值表示惡化
    """
    if not values or len(values) < window_size + 1:
        return 0.0
    
    # 獲取最近窗口的值
    recent_values = values[-window_size:]
    
    # 計算窗口內的平均變化率
    changes = [recent_values[i] - recent_values[i-1] for i in range(1, len(recent_values))]
    
    # 對於損失函數，負變化是改進；對於精度等指標，正變化是改進
    # 這裡我們返回平均變化，由調用者來解釋其含義
    avg_change = np.mean(changes) if changes else 0.0
    
    return float(avg_change)

def calculate_performance_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    計算常見的性能指標集合。
    
    Args:
        predictions (np.ndarray): 預測值數組
        targets (np.ndarray): 目標值數組
    
    Returns:
        Dict[str, float]: 包含多個性能指標的字典
    """
    metrics = {
        'mse': calculate_mse(predictions, targets),
        'rmse': calculate_rmse(predictions, targets),
        'mae': calculate_mae(predictions, targets)
    }
    
    # 如果預測和目標只包含0和1，計算分類指標
    if set(np.unique(predictions)).issubset({0, 1}) and set(np.unique(targets)).issubset({0, 1}):
        metrics['accuracy'] = calculate_accuracy(predictions, targets)
    
    return metrics 

def analyze_loss_curve(loss_values: List[float]) -> Dict[str, Any]:
    """
    分析損失曲線，提取重要特徵和指標。
    
    Args:
        loss_values (List[float]): 訓練過程中的損失值列表。
        
    Returns:
        Dict[str, Any]: 包含損失曲線分析結果的字典。
    """
    if not loss_values:
        return {
            'min_loss': None,
            'min_loss_epoch': None,
            'final_loss': None,
            'improvement_ratio': None,
            'early_convergence': False
        }
    
    # 基本統計量
    min_loss = min(loss_values)
    min_loss_epoch = loss_values.index(min_loss)
    final_loss = loss_values[-1]
    
    # 計算改善比例
    initial_loss = loss_values[0]
    improvement_ratio = (initial_loss - min_loss) / initial_loss if initial_loss > 0 else 0.0
    
    # 判斷是否早期收斂
    early_convergence = False
    if len(loss_values) >= 10:  # 需要足夠的輪次來判斷
        convergence_point = find_convergence_point(loss_values, threshold=0.01)
        if convergence_point is not None and convergence_point < len(loss_values) // 2:
            early_convergence = True
    
    return {
        'min_loss': min_loss,
        'min_loss_epoch': min_loss_epoch,
        'final_loss': final_loss,
        'improvement_ratio': improvement_ratio,
        'early_convergence': early_convergence,
        'loss_trend': 'decreasing' if final_loss <= initial_loss else 'increasing'
    }

def analyze_metric_curve(metric_values: List[float], higher_is_better: bool = True) -> Dict[str, Any]:
    """
    分析評估指標曲線，提取重要特徵和趨勢。
    
    Args:
        metric_values (List[float]): 評估指標值列表。
        higher_is_better (bool, optional): 指標是否越高越好。默認為True。
        
    Returns:
        Dict[str, Any]: 包含指標曲線分析結果的字典。
    """
    if not metric_values:
        return {
            'best_value': None,
            'best_value_epoch': None,
            'final_value': None,
            'improvement_trend': None
        }
    
    # 根據higher_is_better選擇最佳值
    if higher_is_better:
        best_value = max(metric_values)
        best_value_epoch = metric_values.index(best_value)
    else:
        best_value = min(metric_values)
        best_value_epoch = metric_values.index(best_value)
    
    final_value = metric_values[-1]
    
    # 計算改善趨勢
    initial_value = metric_values[0]
    if higher_is_better:
        improvement = (final_value - initial_value) / (max(abs(initial_value), 1e-10))
        improvement_trend = 'improving' if improvement > 0 else 'degrading'
    else:
        improvement = (initial_value - final_value) / (max(abs(initial_value), 1e-10))
        improvement_trend = 'improving' if improvement > 0 else 'degrading'
    
    # 計算波動性
    if len(metric_values) > 1:
        variations = [abs(metric_values[i] - metric_values[i-1]) for i in range(1, len(metric_values))]
        volatility = sum(variations) / len(variations) / (max(abs(max(metric_values)), 1e-10))
    else:
        volatility = 0.0
    
    return {
        'best_value': best_value,
        'best_value_epoch': best_value_epoch,
        'final_value': final_value,
        'improvement_trend': improvement_trend,
        'volatility': volatility
    }

def analyze_training_efficiency(loss_values: List[float], 
                               training_time: float = None, 
                               epoch_times: List[float] = None) -> Dict[str, Any]:
    """
    分析訓練效率，評估模型訓練速度和收斂效率。
    
    Args:
        loss_values (List[float]): 訓練過程中的損失值列表。
        training_time (float, optional): 總訓練時間（秒）。默認為None。
        epoch_times (List[float], optional): 每個輪次的訓練時間列表。默認為None。
        
    Returns:
        Dict[str, Any]: 包含訓練效率分析結果的字典。
    """
    if not loss_values:
        return {
            'convergence_speed': None,
            'time_efficiency': None,
            'improvement_per_epoch': None,
            'total_training_time': None,
            'time_per_epoch': None,
            'mean_epoch_time': None
        }
    
    # 獲取輪次數
    epochs_count = len(loss_values)
    
    # 分析損失曲線
    loss_analysis = analyze_loss_curve(loss_values)
    min_loss = loss_analysis['min_loss']
    min_loss_epoch = loss_analysis['min_loss_epoch']
    initial_loss = loss_values[0]
    
    # 計算每輪改善
    improvement_per_epoch = (initial_loss - min_loss) / max(min_loss_epoch, 1) if min_loss_epoch > 0 else 0.0
    
    # 計算收斂速度（收斂點佔總輪次的比例）
    convergence_point = find_convergence_point(loss_values, threshold=0.01)
    if convergence_point is not None:
        convergence_speed = 1.0 - (convergence_point / epochs_count)
    else:
        convergence_speed = 0.0  # 未收斂
    
    result = {
        'convergence_speed': convergence_speed,
        'improvement_per_epoch': improvement_per_epoch,
        'epochs_to_best': min_loss_epoch,
        'total_epochs': epochs_count
    }
    
    # 時間效率分析
    if training_time is not None:
        result['total_training_time'] = training_time
        result['time_per_epoch'] = training_time / epochs_count
        
        if convergence_point is not None:
            time_to_converge = result['time_per_epoch'] * convergence_point
            result['time_efficiency'] = 1.0 / (time_to_converge + 1e-10)  # 防止除零
        else:
            result['time_efficiency'] = 0.0  # 未收斂
    else:
        result['total_training_time'] = None
        result['time_per_epoch'] = None
        result['time_efficiency'] = None
    
    # 分析每輪次時間統計
    if epoch_times is not None:
        result['mean_epoch_time'] = sum(epoch_times) / len(epoch_times)
        result['min_epoch_time'] = min(epoch_times)
        result['max_epoch_time'] = max(epoch_times)
        result['epoch_time_trend'] = 'decreasing' if epoch_times[0] > epoch_times[-1] else 'increasing' if epoch_times[0] < epoch_times[-1] else 'stable'
    else:
        result['mean_epoch_time'] = None
        result['min_epoch_time'] = None
        result['max_epoch_time'] = None
        result['epoch_time_trend'] = None
    
    return result

def compare_training_runs(run1_metrics: Dict[str, Any], run2_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    比較兩次訓練實驗的性能差異。
    
    Args:
        run1_metrics (Dict[str, Any]): 第一個訓練實驗的指標。
        run2_metrics (Dict[str, Any]): 第二個訓練實驗的指標。
        
    Returns:
        Dict[str, Any]: 包含比較結果的字典。
    """
    comparison = {}
    
    # 比較最終損失
    if 'final_loss' in run1_metrics and 'final_loss' in run2_metrics:
        run1_final_loss = run1_metrics['final_loss']
        run2_final_loss = run2_metrics['final_loss']
        if run1_final_loss is not None and run2_final_loss is not None:
            loss_diff = run1_final_loss - run2_final_loss
            loss_diff_percent = (loss_diff / max(abs(run1_final_loss), 1e-10)) * 100
            comparison['final_loss_diff'] = loss_diff
            comparison['final_loss_diff_percent'] = loss_diff_percent
            comparison['better_final_loss'] = 'run2' if loss_diff > 0 else 'run1' if loss_diff < 0 else 'equal'
    
    # 比較最低損失
    if 'min_loss' in run1_metrics and 'min_loss' in run2_metrics:
        run1_min_loss = run1_metrics['min_loss']
        run2_min_loss = run2_metrics['min_loss']
        if run1_min_loss is not None and run2_min_loss is not None:
            min_loss_diff = run1_min_loss - run2_min_loss
            min_loss_diff_percent = (min_loss_diff / max(abs(run1_min_loss), 1e-10)) * 100
            comparison['min_loss_diff'] = min_loss_diff
            comparison['min_loss_diff_percent'] = min_loss_diff_percent
            comparison['better_min_loss'] = 'run2' if min_loss_diff > 0 else 'run1' if min_loss_diff < 0 else 'equal'
    
    # 比較收斂速度
    if 'convergence_speed' in run1_metrics and 'convergence_speed' in run2_metrics:
        run1_conv_speed = run1_metrics['convergence_speed']
        run2_conv_speed = run2_metrics['convergence_speed']
        if run1_conv_speed is not None and run2_conv_speed is not None:
            conv_speed_diff = run1_conv_speed - run2_conv_speed
            comparison['convergence_speed_diff'] = conv_speed_diff
            comparison['better_convergence'] = 'run2' if conv_speed_diff < 0 else 'run1' if conv_speed_diff > 0 else 'equal'
    
    # 比較改善率
    if 'improvement_ratio' in run1_metrics and 'improvement_ratio' in run2_metrics:
        run1_improvement = run1_metrics['improvement_ratio']
        run2_improvement = run2_metrics['improvement_ratio']
        if run1_improvement is not None and run2_improvement is not None:
            improvement_diff = run1_improvement - run2_improvement
            comparison['improvement_ratio_diff'] = improvement_diff
            comparison['better_improvement'] = 'run2' if improvement_diff < 0 else 'run1' if improvement_diff > 0 else 'equal'
    
    # 總體比較
    better_count = {
        'run1': sum(1 for key in comparison if key.startswith('better_') and comparison[key] == 'run1'),
        'run2': sum(1 for key in comparison if key.startswith('better_') and comparison[key] == 'run2'),
        'equal': sum(1 for key in comparison if key.startswith('better_') and comparison[key] == 'equal')
    }
    
    if better_count['run1'] > better_count['run2']:
        comparison['overall_better'] = 'run1'
    elif better_count['run2'] > better_count['run1']:
        comparison['overall_better'] = 'run2'
    else:
        comparison['overall_better'] = 'equal'
    
    comparison['better_metrics_summary'] = better_count
    
    return comparison 