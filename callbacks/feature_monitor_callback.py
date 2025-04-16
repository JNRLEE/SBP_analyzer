"""
特徵監控回調模塊

該模塊實現了特徵值監控的回調功能，用於跟蹤和分析模型訓練過程中特徵的變化與分佈。
"""

from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from .base_callback import BaseCallback

class FeatureMonitorCallback(BaseCallback):
    """
    特徵監控回調
    
    監控模型訓練過程中特定層或特定特徵的活動情況，可用於分析特徵重要性、
    特徵冗餘性、特徵相關性等。
    
    屬性:
        target_layers: 要監控的層名稱列表
        feature_indices: 要監控的特徵索引列表
        monitor_frequency: 監控頻率（每隔多少個batch監控一次）
        log_to_tensorboard: 是否記錄到TensorBoard
        track_correlations: 是否跟蹤特徵間的相關性
        track_activations: 是否跟蹤激活值分佈
        track_gradients: 是否跟蹤梯度分佈
    """
    
    def __init__(
        self,
        target_layers: Optional[List[str]] = None,
        feature_indices: Optional[List[int]] = None,
        monitor_frequency: int = 10,
        log_to_tensorboard: bool = True,
        track_correlations: bool = True,
        track_activations: bool = True,
        track_gradients: bool = False,
        save_plots: bool = False,
        plot_dir: str = './feature_plots',
        trigger_condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        """初始化特徵監控回調
        
        Args:
            target_layers: 要監控的層名稱列表，如果為None則嘗試監控所有層
            feature_indices: 要監控的特徵索引列表，如果為None則監控所有特徵
            monitor_frequency: 監控頻率（每隔多少個batch監控一次）
            log_to_tensorboard: 是否記錄到TensorBoard
            track_correlations: 是否跟蹤特徵間的相關性
            track_activations: 是否跟蹤激活值分佈
            track_gradients: 是否跟蹤梯度分佈
            save_plots: 是否保存特徵分析圖表到本地
            plot_dir: 圖表保存目錄
            trigger_condition: 觸發條件，決定何時執行回調邏輯
        """
        super().__init__(trigger_condition)
        self.target_layers = target_layers
        self.feature_indices = feature_indices
        self.monitor_frequency = monitor_frequency
        self.log_to_tensorboard = log_to_tensorboard
        self.track_correlations = track_correlations
        self.track_activations = track_activations
        self.track_gradients = track_gradients
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        
        # 內部狀態
        self.hook_handles = []
        self.activation_stats = defaultdict(list)
        self.gradient_stats = defaultdict(list)
        self.feature_correlations = {}
        self.current_batch = 0
        self.current_epoch = 0
        self.tensorboard_writer = None
        self.feature_importance = {}
    
    def _on_train_begin(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """訓練開始時設置模型鉤子
        
        Args:
            model: 正在訓練的模型
            logs: 訓練相關信息
        """
        if logs is None:
            logs = {}
        
        # 獲取tensorboard寫入器（如果有）
        self.tensorboard_writer = logs.get('tensorboard_writer', None)
        
        # 註冊層鉤子
        self._register_hooks(model)
        
        # 如果需要保存圖表，確保目錄存在
        if self.save_plots:
            import os
            os.makedirs(self.plot_dir, exist_ok=True)
    
    def _on_train_end(self, model: nn.Module, logs: Dict[str, Any] = None) -> None:
        """訓練結束時清理鉤子並生成最終報告
        
        Args:
            model: 訓練完成的模型
            logs: 訓練相關信息
        """
        # 移除所有鉤子
        self._remove_hooks()
        
        # 計算特徵重要性
        self._compute_feature_importance()
        
        # 生成最終報告圖表
        if self.save_plots:
            self._save_final_plots()
    
    def _on_batch_end(self, batch: int, model: nn.Module, loss: torch.Tensor, 
                     logs: Dict[str, Any] = None) -> None:
        """每個batch結束時分析特徵
        
        Args:
            batch: 當前batch索引
            model: 正在訓練的模型
            loss: 當前batch的損失值
            logs: batch相關信息
        """
        self.current_batch = batch
        
        # 根據監控頻率決定是否執行分析
        if batch % self.monitor_frequency != 0:
            return
        
        # 處理收集的數據
        if self.track_correlations:
            self._analyze_feature_correlations()
        
        # 記錄到TensorBoard
        if self.log_to_tensorboard and self.tensorboard_writer is not None:
            self._log_to_tensorboard()
    
    def _on_epoch_end(self, epoch: int, model: nn.Module, train_logs: Dict[str, Any], 
                     val_logs: Dict[str, Any], logs: Dict[str, Any] = None) -> None:
        """每個epoch結束時更新特徵統計信息
        
        Args:
            epoch: 當前epoch索引
            model: 正在訓練的模型
            train_logs: 訓練集上的指標和結果
            val_logs: 驗證集上的指標和結果
            logs: 其他相關信息
        """
        self.current_epoch = epoch
        
        # 計算每個特徵的平均激活值和標準差
        epoch_stats = self._compute_epoch_statistics()
        
        # 記錄到TensorBoard
        if self.log_to_tensorboard and self.tensorboard_writer is not None:
            for layer_name, stats in epoch_stats.items():
                for stat_name, value in stats.items():
                    if isinstance(value, np.ndarray) and value.size > 1:
                        # 對於多維統計量，只記錄平均值
                        value = np.mean(value)
                    
                    self.tensorboard_writer.add_scalar(
                        f'feature_stats/{layer_name}/{stat_name}',
                        value,
                        epoch
                    )
        
        # 保存epoch特徵分析圖表
        if self.save_plots and epoch % 5 == 0:  # 每5個epoch保存一次
            self._save_epoch_plots(epoch)
    
    def _register_hooks(self, model: nn.Module) -> None:
        """註冊層鉤子以收集激活值和梯度
        
        Args:
            model: 要監控的模型
        """
        if self.target_layers is None:
            # 如果未指定目標層，自動選擇可能的目標層
            layers_dict = {}
            for name, module in model.named_modules():
                # 監控卷積層、線性層和批歸一化層
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
                    layers_dict[name] = module
            
            self.target_layers = list(layers_dict.keys())
        
        # 為每個目標層註冊前向和後向鉤子
        for name, module in model.named_modules():
            if name in self.target_layers:
                # 註冊前向鉤子（收集激活值）
                if self.track_activations:
                    handle = module.register_forward_hook(
                        lambda m, inp, outp, name=name: self._activation_hook(name, outp)
                    )
                    self.hook_handles.append(handle)
                
                # 註冊後向鉤子（收集梯度）
                if self.track_gradients:
                    handle = module.register_backward_hook(
                        lambda m, grad_inp, grad_outp, name=name: self._gradient_hook(name, grad_outp[0])
                    )
                    self.hook_handles.append(handle)
    
    def _remove_hooks(self) -> None:
        """移除所有註冊的鉤子"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    def _activation_hook(self, layer_name: str, output: torch.Tensor) -> None:
        """收集層激活值的鉤子函數
        
        Args:
            layer_name: 層名稱
            output: 層輸出張量
        """
        if isinstance(output, tuple):
            output = output[0]
        
        # 將輸出轉換為numpy數組並存儲
        output_np = output.detach().cpu().numpy()
        
        # 如果指定了特徵索引，則只保留這些特徵
        if self.feature_indices is not None:
            # 假設特徵維度在第1維（通道維度）
            if output_np.ndim > 1 and output_np.shape[1] > max(self.feature_indices):
                output_np = output_np[:, self.feature_indices]
        
        # 存儲激活統計信息
        self.activation_stats[layer_name].append({
            'mean': np.mean(output_np, axis=0),
            'std': np.std(output_np, axis=0),
            'min': np.min(output_np, axis=0),
            'max': np.max(output_np, axis=0),
            'sparsity': np.mean(output_np == 0),
            'batch': self.current_batch,
            'epoch': self.current_epoch,
            # 存儲少量樣本數據，用於後續分析
            'samples': output_np[:min(5, output_np.shape[0])] if output_np.ndim > 1 else output_np
        })
    
    def _gradient_hook(self, layer_name: str, grad_output: torch.Tensor) -> None:
        """收集層梯度的鉤子函數
        
        Args:
            layer_name: 層名稱
            grad_output: 梯度張量
        """
        if grad_output is None:
            return
        
        # 將梯度轉換為numpy數組並存儲
        grad_np = grad_output.detach().cpu().numpy()
        
        # 如果指定了特徵索引，則只保留這些特徵
        if self.feature_indices is not None:
            # 假設特徵維度在第1維（通道維度）
            if grad_np.ndim > 1 and grad_np.shape[1] > max(self.feature_indices):
                grad_np = grad_np[:, self.feature_indices]
        
        # 存儲梯度統計信息
        self.gradient_stats[layer_name].append({
            'mean': np.mean(grad_np, axis=0),
            'std': np.std(grad_np, axis=0),
            'min': np.min(grad_np, axis=0),
            'max': np.max(grad_np, axis=0),
            'norm': np.mean(np.linalg.norm(grad_np, axis=-1)),
            'batch': self.current_batch,
            'epoch': self.current_epoch
        })
    
    def _analyze_feature_correlations(self) -> None:
        """分析特徵間的相關性"""
        # 只在有足夠數據時進行相關性分析
        for layer_name, stats_list in self.activation_stats.items():
            if not stats_list:
                continue
            
            # 獲取最近的激活值樣本
            latest_stats = stats_list[-1]
            samples = latest_stats['samples']
            
            # 如果樣本是多維的，重塑為2D進行相關性分析
            if samples.ndim > 2:
                # 假設形狀為 [batch_size, channels, height, width]
                batch_size, channels = samples.shape[:2]
                samples_reshaped = samples.reshape(batch_size, channels, -1)
                # 對每個通道特徵取平均
                samples_flat = np.mean(samples_reshaped, axis=2)
            else:
                samples_flat = samples
            
            # 計算相關性矩陣（僅當樣本數量足夠時）
            if samples_flat.shape[0] > 1 and samples_flat.shape[1] > 1:
                try:
                    corr_matrix = np.corrcoef(samples_flat, rowvar=False)
                    self.feature_correlations[layer_name] = corr_matrix
                except (ValueError, np.linalg.LinAlgError):
                    # 如果相關性計算失敗（例如，常量特徵），跳過
                    pass
    
    def _compute_epoch_statistics(self) -> Dict[str, Dict[str, Any]]:
        """計算當前epoch的特徵統計信息
        
        Returns:
            每個層的統計信息摘要
        """
        epoch_stats = {}
        
        # 處理激活統計信息
        for layer_name, stats_list in self.activation_stats.items():
            # 篩選當前epoch的統計信息
            epoch_stats_list = [s for s in stats_list if s['epoch'] == self.current_epoch]
            if not epoch_stats_list:
                continue
            
            # 計算均值統計量
            mean_values = np.array([s['mean'] for s in epoch_stats_list])
            std_values = np.array([s['std'] for s in epoch_stats_list])
            sparsity_values = np.array([s['sparsity'] for s in epoch_stats_list])
            
            epoch_stats[layer_name] = {
                'activation_mean': np.mean(mean_values, axis=0),
                'activation_std': np.mean(std_values, axis=0),
                'activation_stability': np.std(mean_values, axis=0),  # 均值的標準差，表示穩定性
                'activation_sparsity': np.mean(sparsity_values)
            }
        
        # 處理梯度統計信息
        for layer_name, stats_list in self.gradient_stats.items():
            # 篩選當前epoch的統計信息
            epoch_stats_list = [s for s in stats_list if s['epoch'] == self.current_epoch]
            if not epoch_stats_list:
                continue
            
            if layer_name not in epoch_stats:
                epoch_stats[layer_name] = {}
            
            # 計算均值統計量
            grad_mean_values = np.array([s['mean'] for s in epoch_stats_list])
            grad_norm_values = np.array([s['norm'] for s in epoch_stats_list])
            
            epoch_stats[layer_name].update({
                'gradient_mean': np.mean(grad_mean_values, axis=0),
                'gradient_magnitude': np.mean(grad_norm_values),
                'gradient_stability': np.std(grad_mean_values, axis=0)  # 梯度均值的標準差，表示穩定性
            })
        
        return epoch_stats
    
    def _compute_feature_importance(self) -> None:
        """計算特徵重要性
        
        使用激活值和梯度的統計信息來估計特徵重要性
        """
        for layer_name in self.activation_stats.keys():
            # 獲取所有epoch的激活值統計信息
            act_means = []
            for stats in self.activation_stats[layer_name]:
                if isinstance(stats['mean'], np.ndarray):
                    act_means.append(stats['mean'])
            
            if not act_means:
                continue
            
            # 計算平均激活值
            avg_activations = np.mean(np.array(act_means), axis=0)
            
            # 如果有梯度信息，結合使用
            if layer_name in self.gradient_stats and self.gradient_stats[layer_name]:
                grad_means = []
                for stats in self.gradient_stats[layer_name]:
                    if isinstance(stats['mean'], np.ndarray) and stats['mean'].shape == avg_activations.shape:
                        grad_means.append(stats['mean'])
                
                if grad_means:
                    # 計算平均梯度
                    avg_gradients = np.mean(np.array(grad_means), axis=0)
                    # 基於激活×梯度計算特徵重要性（類似於Grad-CAM方法）
                    importance = np.abs(avg_activations * avg_gradients)
                else:
                    # 僅使用激活值的幅度作為重要性指標
                    importance = np.abs(avg_activations)
            else:
                # 僅使用激活值的幅度作為重要性指標
                importance = np.abs(avg_activations)
            
            # 標準化重要性分數
            if np.sum(importance) > 0:
                importance = importance / np.sum(importance)
            
            self.feature_importance[layer_name] = importance
    
    def _log_to_tensorboard(self) -> None:
        """將特徵統計信息記錄到TensorBoard"""
        if self.tensorboard_writer is None:
            return
        
        # 記錄激活值統計信息
        for layer_name, stats_list in self.activation_stats.items():
            if not stats_list:
                continue
            
            latest_stats = stats_list[-1]
            
            # 對於多維統計量，計算均值進行記錄
            for stat_name in ['mean', 'std', 'min', 'max', 'sparsity']:
                value = latest_stats[stat_name]
                if isinstance(value, np.ndarray) and value.size > 1:
                    value = np.mean(value)
                
                self.tensorboard_writer.add_scalar(
                    f'features/{layer_name}/activation_{stat_name}',
                    value,
                    self.current_batch
                )
        
        # 記錄梯度統計信息
        for layer_name, stats_list in self.gradient_stats.items():
            if not stats_list:
                continue
            
            latest_stats = stats_list[-1]
            
            # 對於多維統計量，計算均值進行記錄
            for stat_name in ['mean', 'std', 'min', 'max', 'norm']:
                value = latest_stats[stat_name]
                if isinstance(value, np.ndarray) and value.size > 1:
                    value = np.mean(value)
                
                self.tensorboard_writer.add_scalar(
                    f'features/{layer_name}/gradient_{stat_name}',
                    value,
                    self.current_batch
                )
        
        # 記錄特徵相關性
        for layer_name, corr_matrix in self.feature_correlations.items():
            if corr_matrix.shape[0] > 1:
                # 計算相關性矩陣的統計信息
                off_diag_mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                off_diag_corr = corr_matrix[off_diag_mask]
                
                # 記錄平均絕對相關性（排除對角線）
                mean_abs_corr = np.mean(np.abs(off_diag_corr))
                
                self.tensorboard_writer.add_scalar(
                    f'features/{layer_name}/correlation',
                    mean_abs_corr,
                    self.current_batch
                )
    
    def _save_epoch_plots(self, epoch: int) -> None:
        """保存當前epoch的特徵分析圖表
        
        Args:
            epoch: 當前epoch索引
        """
        import matplotlib.pyplot as plt
        import os
        
        # 為當前epoch創建目錄
        epoch_dir = os.path.join(self.plot_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 繪製每個層的激活值分佈
        for layer_name, stats_list in self.activation_stats.items():
            # 篩選當前epoch的統計信息
            epoch_stats = [s for s in stats_list if s['epoch'] == epoch]
            if not epoch_stats:
                continue
            
            # 計算平均激活值
            mean_values = np.array([s['mean'] for s in epoch_stats])
            if mean_values.size == 0 or not isinstance(mean_values[0], np.ndarray):
                continue
            
            # 如果有多個特徵，計算平均值
            if mean_values.ndim > 1 and mean_values.shape[1] > 1:
                # 取前20個特徵（如果有）進行可視化
                num_features = min(20, mean_values.shape[1])
                
                plt.figure(figsize=(12, 6))
                for i in range(num_features):
                    feature_means = mean_values[:, i]
                    plt.plot(feature_means, label=f'Feature {i}')
                
                plt.title(f'Layer: {layer_name} - Feature Activations')
                plt.xlabel('Batch')
                plt.ylabel('Activation')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(epoch_dir, f'{layer_name}_activations.png'))
                plt.close()
                
                # 繪製特徵相關性熱圖
                if layer_name in self.feature_correlations:
                    corr_matrix = self.feature_correlations[layer_name]
                    if corr_matrix.shape[0] > 1:
                        plt.figure(figsize=(10, 8))
                        plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                        plt.colorbar()
                        plt.title(f'Layer: {layer_name} - Feature Correlations')
                        plt.tight_layout()
                        plt.savefig(os.path.join(epoch_dir, f'{layer_name}_correlations.png'))
                        plt.close()
    
    def _save_final_plots(self) -> None:
        """保存最終的特徵分析報告圖表"""
        import matplotlib.pyplot as plt
        import os
        
        # 創建最終報告目錄
        final_dir = os.path.join(self.plot_dir, 'final_report')
        os.makedirs(final_dir, exist_ok=True)
        
        # 繪製特徵重要性圖表
        for layer_name, importance in self.feature_importance.items():
            if not isinstance(importance, np.ndarray) or importance.size <= 1:
                continue
            
            # 取前N個最重要的特徵（如果有）
            num_features = min(30, importance.size)
            indices = np.argsort(importance)[-num_features:][::-1]
            top_importance = importance[indices]
            feature_names = [f'Feature {i}' for i in indices]
            
            plt.figure(figsize=(12, 6))
            plt.bar(feature_names, top_importance)
            plt.title(f'Layer: {layer_name} - Feature Importance')
            plt.xticks(rotation=90)
            plt.xlabel('Feature')
            plt.ylabel('Importance Score')
            plt.tight_layout()
            plt.savefig(os.path.join(final_dir, f'{layer_name}_importance.png'))
            plt.close()
        
        # 繪製特徵穩定性圖表
        epoch_stats_all = self._compute_all_epochs_statistics()
        for layer_name, stats in epoch_stats_all.items():
            if 'stability' not in stats or not isinstance(stats['stability'], np.ndarray):
                continue
            
            stability = stats['stability']
            if stability.size <= 1:
                continue
            
            # 取前N個穩定性最差的特徵（如果有）
            num_features = min(30, stability.size)
            indices = np.argsort(stability)[-num_features:][::-1]
            top_unstable = stability[indices]
            feature_names = [f'Feature {i}' for i in indices]
            
            plt.figure(figsize=(12, 6))
            plt.bar(feature_names, top_unstable)
            plt.title(f'Layer: {layer_name} - Feature Instability')
            plt.xticks(rotation=90)
            plt.xlabel('Feature')
            plt.ylabel('Instability Score')
            plt.tight_layout()
            plt.savefig(os.path.join(final_dir, f'{layer_name}_instability.png'))
            plt.close()
    
    def _compute_all_epochs_statistics(self) -> Dict[str, Dict[str, Any]]:
        """計算所有epoch的累積統計信息
        
        Returns:
            每個層的累積統計信息摘要
        """
        all_stats = {}
        
        # 處理激活統計信息
        for layer_name, stats_list in self.activation_stats.items():
            if not stats_list:
                continue
            
            # 按epoch分組
            epoch_groups = {}
            for stat in stats_list:
                epoch = stat['epoch']
                if epoch not in epoch_groups:
                    epoch_groups[epoch] = []
                epoch_groups[epoch].append(stat)
            
            # 計算每個epoch的均值
            epoch_means = []
            for epoch, epoch_stats in epoch_groups.items():
                if not epoch_stats:
                    continue
                
                mean_values = np.array([s['mean'] for s in epoch_stats])
                if mean_values.ndim > 1:
                    epoch_mean = np.mean(mean_values, axis=0)
                    epoch_means.append(epoch_mean)
            
            if not epoch_means:
                continue
            
            # 將所有epoch的均值轉換為數組
            all_means = np.array(epoch_means)
            
            # 計算特徵穩定性（跨epoch的標準差）
            stability = np.std(all_means, axis=0)
            
            all_stats[layer_name] = {
                'mean': np.mean(all_means, axis=0),
                'stability': stability
            }
        
        return all_stats 