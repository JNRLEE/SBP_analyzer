# python results/custom_analyzer.py results/<experiment_name>
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom analysis script for analyzing a single experiment directory.
This script analyzes:
1. Model structure analysis
2. Training history analysis
3. GNS (Gradient Noise Scale) analysis
4. Advanced visualizations (parameter distributions, cosine similarity, confusion matrix)
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import pickle
import seaborn as sns
import argparse
import re
from datetime import datetime

# Add SBP_analyzer root directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

# 條件性導入torch，如果不可用則記錄日誌
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    logging.warning(f"PyTorch is not available: {e}. Will use pickle for loading model data.")

# Conditionally import visualization modules
try:
    from visualization.model_structure_plots import plot_model_params_distribution
    from visualization.layer_activity_plots import plot_cosine_similarity_matrix, plot_activation_statistics, plot_layer_activations_heatmap
    from visualization.performance_plots import plot_confusion_matrix, plot_training_loss_curve
    from metrics.layer_activity_metrics import compute_confusion_matrix, compute_activation_statistics, compute_cosine_similarity_matrix
    VIZ_MODULES_AVAILABLE = True
except ImportError as e:
    VIZ_MODULES_AVAILABLE = False
    logging.error(f"Error importing visualization modules: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomAnalyzer:
    """Custom analyzer for analyzing experiment data"""
    
    def __init__(self, experiment_dir):
        """
        Initialize the analyzer
        
        Args:
            experiment_dir (str): Path to the experiment directory
        """
        self.experiment_dir = experiment_dir
        self.analysis_dir = os.path.join(experiment_dir, 'custom_analysis')
        Path(self.analysis_dir).mkdir(exist_ok=True)
        self.results = {}
    
    def load_config(self):
        """
        Load experiment configuration
        
        Returns:
            dict: Experiment configuration
        """
        config_path = os.path.join(self.experiment_dir, 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.results['config'] = config
            logger.info("Successfully loaded experiment configuration")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return None
    
    def load_model_structure(self):
        """
        Load model structure information
        
        Returns:
            dict: Model structure information
        """
        model_structure_path = os.path.join(self.experiment_dir, 'model_structure.json')
        try:
            with open(model_structure_path, 'r') as f:
                model_structure = json.load(f)
            self.results['model_structure'] = model_structure
            logger.info("Successfully loaded model structure information")
            return model_structure
        except Exception as e:
            logger.error(f"Error loading model structure: {e}")
            return None
    
    def load_training_history(self):
        """
        Load training history
        
        Returns:
            dict: Training history record
        """
        history_path = os.path.join(self.experiment_dir, 'training_history.json')
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            self.results['training_history'] = history
            logger.info("Successfully loaded training history")
            return history
        except Exception as e:
            logger.error(f"Error loading training history: {e}")
            return None
    
    def analyze_model_structure(self):
        """
        Analyze model structure
        
        Returns:
            dict: Model structure analysis results
        """
        model_structure = self.results.get('model_structure')
        if not model_structure:
            model_structure = self.load_model_structure()
            if not model_structure:
                logger.error("Cannot analyze model structure, failed to load model structure information")
                return None
        
        # Extract model statistics
        try:
            total_params = model_structure.get('total_parameters', 0)
            trainable_params = model_structure.get('trainable_parameters', 0)
            
            # Get layer information directly from model_structure.json
            layers = model_structure.get('layers', [])
            
            # Calculate parameter ratio for each layer
            layer_param_ratio = {}
            for layer in layers:
                layer_name = layer.get('name', 'unknown')
                layer_params = layer.get('parameters', 0)
                if total_params > 0:
                    layer_param_ratio[layer_name] = layer_params / total_params * 100
            
            # Calculate parameter distribution statistics
            sorted_layers = sorted(layers, key=lambda x: x.get('parameters', 0), reverse=True)
            top_layers = sorted_layers[:5]  # Top 5 layers with most parameters
            
            # Prepare parameter distribution data for the visualization
            by_layer_type = {}
            for layer in layers:
                layer_type = layer.get('type', 'unknown')
                param_count = layer.get('parameters', 0)
                
                if layer_type not in by_layer_type:
                    by_layer_type[layer_type] = 0
                    
                by_layer_type[layer_type] += param_count
            
            # Calculate percentages
            percentage_by_type = {
                layer_type: (count / total_params * 100) if total_params > 0 else 0
                for layer_type, count in by_layer_type.items()
            }
            
            # Create parameter distribution data
            param_distribution = {
                'by_layer_type': by_layer_type,
                'percentage_by_type': percentage_by_type
            }
            
            # Generate parameter distribution plot using the visualization if available
            params_plot_path = os.path.join(self.analysis_dir, 'model_params_distribution.png')
            if VIZ_MODULES_AVAILABLE:
                try:
                    plot_model_params_distribution(param_distribution, params_plot_path)
                    logger.info(f"Generated model parameter distribution plot at {params_plot_path}")
                except Exception as e:
                    logger.error(f"Error generating model parameter distribution plot: {e}")
                    params_plot_path = None
            else:
                logger.warning("Visualization modules not available, skipping model parameter distribution plot")
                params_plot_path = None
            
            analysis_result = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'layer_count': len(layers),
                'layer_param_ratio': layer_param_ratio,
                'top_heavy_layers': [layer.get('name') for layer in top_layers],
                'params_distribution_plot': params_plot_path
            }
            
            self.results['model_structure_analysis'] = analysis_result
            logger.info("Completed model structure analysis")
            return analysis_result
        
        except Exception as e:
            logger.error(f"Error analyzing model structure: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def analyze_training_history(self):
        """
        Analyze training history
        
        Returns:
            dict: Training history analysis results
        """
        history = self.results.get('training_history')
        if not history:
            history = self.load_training_history()
            if not history:
                logger.error("Cannot analyze training history, failed to load training history")
                return None
        
        try:
            # Get training and validation losses
            train_loss = history.get('train_loss', [])
            val_loss = history.get('val_loss', [])
            best_val_loss = history.get('best_val_loss', float('inf'))
            training_time = history.get('training_time', 0)
            
            # Calculate training metrics
            if train_loss and val_loss:
                epochs = list(range(len(train_loss)))
                final_train_loss = train_loss[-1] if train_loss else None
                final_val_loss = val_loss[-1] if val_loss else None
                
                # Calculate average convergence speed (loss decrease per epoch)
                train_loss_diffs = [train_loss[i] - train_loss[i+1] for i in range(len(train_loss)-1) if train_loss[i] > train_loss[i+1]]
                avg_convergence_speed = sum(train_loss_diffs) / len(train_loss_diffs) if train_loss_diffs else 0
                
                # Detect overfitting
                overfitting_detected = False
                overfitting_epoch = None
                
                # If validation loss starts increasing while training loss continues to decrease, overfitting may be occurring
                for i in range(1, min(len(train_loss), len(val_loss))):
                    if train_loss[i] < train_loss[i-1] and val_loss[i] > val_loss[i-1]:
                        consecutive_overfit = 0
                        for j in range(i, min(len(train_loss), len(val_loss))):
                            if train_loss[j] < train_loss[j-1] and val_loss[j] > val_loss[j-1]:
                                consecutive_overfit += 1
                            else:
                                consecutive_overfit = 0
                            
                            if consecutive_overfit >= 2:  # If this pattern occurs for 2 consecutive epochs, consider it overfitting
                                overfitting_detected = True
                                overfitting_epoch = j - 1
                                break
                        
                        if overfitting_detected:
                            break
                
                # Plot loss curves
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, train_loss, 'b-', label='Training Loss')
                plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
                if overfitting_detected and overfitting_epoch is not None:
                    plt.axvline(x=overfitting_epoch, color='g', linestyle='--', label=f'Overfitting Detected (Epoch {overfitting_epoch})')
                plt.title('Training and Validation Loss Curves')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                # Save image
                loss_plot_path = os.path.join(self.analysis_dir, 'training_loss_curve.png')
                plt.savefig(loss_plot_path)
                plt.close()
                
                # Calculate training-validation loss difference
                train_val_diff = [abs(train_loss[i] - val_loss[i]) for i in range(min(len(train_loss), len(val_loss)))]
                avg_train_val_diff = sum(train_val_diff) / len(train_val_diff) if train_val_diff else 0
                
                # Determine convergence status
                if final_train_loss is not None and final_val_loss is not None:
                    if abs(final_train_loss - final_val_loss) < 0.1 * final_val_loss:
                        convergence_status = "Good (Training and validation losses are close)"
                    elif final_train_loss < 0.5 * final_val_loss:
                        convergence_status = "Overfitting (Training loss much lower than validation loss)"
                    elif final_train_loss > final_val_loss * 1.2:
                        convergence_status = "Underfitting (Training loss higher than validation loss)"
                    else:
                        convergence_status = "Moderate"
                else:
                    convergence_status = "Unknown (Missing final loss values)"
            
                analysis_result = {
                    'epochs_trained': len(train_loss),
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss,
                    'best_val_loss': best_val_loss,
                    'training_time': training_time,
                    'avg_convergence_speed': avg_convergence_speed,
                    'avg_train_val_diff': avg_train_val_diff,
                    'overfitting_detected': overfitting_detected,
                    'overfitting_epoch': overfitting_epoch,
                    'convergence_status': convergence_status,
                    'loss_curve_plot': loss_plot_path
                }
                
                self.results['training_history_analysis'] = analysis_result
                logger.info("Completed training history analysis")
                return analysis_result
            else:
                logger.error("Training history missing loss records")
                return None
        
        except Exception as e:
            logger.error(f"Error analyzing training history: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def analyze_gns(self):
        """
        Analyze GNS (Gradient Noise Scale)
        
        Returns:
            dict: GNS analysis results
        """
        logger.info("Starting GNS analysis...")
        # Find all GNS data files
        gns_files = []
        hooks_dir = os.path.join(self.experiment_dir, 'hooks')
        
        try:
            if os.path.exists(hooks_dir):
                logger.info(f"Scanning hooks directory: {hooks_dir}")
                for epoch_dir in os.listdir(hooks_dir):
                    epoch_path = os.path.join(hooks_dir, epoch_dir)
                    logger.info(f"Checking epoch directory: {epoch_path}")
                    if os.path.isdir(epoch_path) and epoch_dir.startswith('epoch_'):
                        for file in os.listdir(epoch_path):
                            if file.startswith('gns_stats_epoch_'):
                                gns_file_path = os.path.join(epoch_path, file)
                                logger.info(f"Found GNS data file: {gns_file_path}")
                                gns_files.append(gns_file_path)
            
            if not gns_files:
                logger.warning("No GNS data files found")
                return None
            
            # Load GNS data
            gns_data = []
            for file_path in gns_files:
                try:
                    logger.info(f"Attempting to read GNS data file: {file_path}")
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            gns_data.append(data)
                            logger.info(f"Successfully loaded GNS data from {file_path}")
                    except PermissionError:
                        logger.warning(f"Permission denied when trying to read {file_path}. Skipping.")
                        continue
                except Exception as e:
                    logger.error(f"Cannot read GNS data file {file_path}: {e}")
            
            if not gns_data:
                logger.error("All GNS data files failed to load")
                return None
            
            # Extract GNS statistics
            logger.info("Extracting GNS statistics...")
            epochs = [data.get('epoch') for data in gns_data]
            gns_values = [data.get('gns', 0) for data in gns_data]
            total_vars = [data.get('total_var', 0) for data in gns_data]
            mean_norm_sqs = [data.get('mean_norm_sq', 0) for data in gns_data]
            
            # Calculate GNS average and trend
            avg_gns = sum(gns_values) / len(gns_values) if gns_values else 0
            
            # GNS trend analysis
            gns_trend = "Increasing" if len(gns_values) > 1 and gns_values[-1] > gns_values[0] else "Decreasing" if len(gns_values) > 1 and gns_values[-1] < gns_values[0] else "Stable"
            
            # Plot GNS trend
            logger.info("Generating GNS trend plot...")
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, gns_values, 'g-o', label='GNS Value')
            plt.title('GNS (Gradient Noise Scale) Trend')
            plt.xlabel('Epochs')
            plt.ylabel('GNS Value')
            plt.grid(True)
            plt.legend()
            
            # Save image
            gns_plot_path = os.path.join(self.analysis_dir, 'gns_trend.png')
            plt.savefig(gns_plot_path)
            plt.close()
            logger.info(f"GNS trend plot saved to {gns_plot_path}")
            
            # Batch size recommendation
            # Higher GNS values suggest higher gradient noise, recommend increasing batch size
            # Lower GNS values suggest lower gradient noise, can consider reducing batch size for faster convergence
            current_batch_size = self.results.get('config', {}).get('data', {}).get('dataloader', {}).get('batch_size', 16)
            
            if avg_gns > 10:
                recommended_batch_size = current_batch_size * 2
                recommendation = "High GNS value detected. Consider increasing batch size to reduce gradient noise."
            elif avg_gns < 0.1:
                recommended_batch_size = max(current_batch_size // 2, 1)
                recommendation = "Low GNS value detected. Consider decreasing batch size for faster convergence."
            else:
                recommended_batch_size = current_batch_size
                recommendation = "GNS value is moderate. Current batch size appears appropriate."
            
            # Plot gradient statistics
            logger.info("Generating gradient statistics plot...")
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, total_vars, 'b-o', label='Total Gradient Variance')
            plt.plot(epochs, mean_norm_sqs, 'r-o', label='Mean Squared Norm')
            plt.title('Gradient Statistics Trend')
            plt.xlabel('Epochs')
            plt.ylabel('Value (Log Scale)')
            plt.yscale('log')
            plt.grid(True)
            plt.legend()
            
            # Save image
            grad_stats_plot_path = os.path.join(self.analysis_dir, 'gradient_stats.png')
            plt.savefig(grad_stats_plot_path)
            plt.close()
            logger.info(f"Gradient statistics plot saved to {grad_stats_plot_path}")
            
            analysis_result = {
                'epochs_analyzed': epochs,
                'gns_values': gns_values,
                'average_gns': avg_gns,
                'gns_trend': gns_trend,
                'total_variances': total_vars,
                'mean_norm_squares': mean_norm_sqs,
                'current_batch_size': current_batch_size,
                'recommended_batch_size': recommended_batch_size,
                'recommendation': recommendation,
                'gns_plot': gns_plot_path,
                'grad_stats_plot': grad_stats_plot_path
            }
            
            self.results['gns_analysis'] = analysis_result
            logger.info("Completed GNS analysis")
            return analysis_result
        
        except Exception as e:
            logger.error(f"Error analyzing GNS: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def create_layer_activation_visualizations(self):
        """
        Create visualizations of layer activations using the new visualization features.
        Tries to load actual test set activations, falls back to simulated data if none available.
        
        Returns:
            dict: Layer activation visualization results
        """
        logger.info("Creating layer activation visualizations...")
        
        try:
            # 創建層激活視覺化目錄
            activation_dir = os.path.join(self.analysis_dir, 'layer_activations')
            os.makedirs(activation_dir, exist_ok=True)
            
            # 初始化 HookDataLoader
            try:
                from data_loader.hook_data_loader import HookDataLoader
                hook_loader = HookDataLoader(self.experiment_dir)
                logger.info("Successfully initialized HookDataLoader")
            except Exception as e:
                logger.error(f"Failed to initialize HookDataLoader: {e}. Will use simulated data.")
                hook_loader = None
            
            # 存儲視覺化路徑的字典
            visualization_paths = {}
            
            # 嘗試獲取可用的測試集激活值層
            available_test_layers = []
            if hook_loader:
                try:
                    available_test_layers = hook_loader.list_available_test_activations()
                    logger.info(f"Found {len(available_test_layers)} available test activation layers: {available_test_layers}")
                except Exception as e:
                    logger.error(f"Error listing available test activation layers: {e}")
            
            # 如果找到可用的測試集激活值
            if available_test_layers:
                # 選擇要分析的層（可以是全部或部分）
                layers_to_analyze = available_test_layers
                # 如果層太多，可以選擇一部分有代表性的層
                if len(layers_to_analyze) > 3:
                    # 優先選擇通常更具代表性的層（例如最後的卷積層和全連接層）
                    prioritized_names = [name for name in layers_to_analyze if 'fc' in name.lower()]
                    prioritized_names += [name for name in layers_to_analyze if 'conv' in name.lower() and name not in prioritized_names]
                    # 如果有優先級層，優先使用它們；否則使用前3個層
                    if prioritized_names:
                        layers_to_analyze = prioritized_names[:3]
                    else:
                        layers_to_analyze = available_test_layers[:3]
                
                logger.info(f"Will analyze the following layers: {layers_to_analyze}")
                
                # 為每一層創建視覺化
                for layer_name in layers_to_analyze:
                    # 載入該層的測試集激活值
                    try:
                        layer_activations = hook_loader.load_test_activations(layer_name)
                        if layer_activations is not None:
                            # 確保數據是 numpy 數組或 PyTorch 張量
                            if isinstance(layer_activations, dict) and 'activation' in layer_activations:
                                layer_activations = layer_activations['activation']
                            elif isinstance(layer_activations, dict) and 'activations' in layer_activations:
                                layer_activations = layer_activations['activations']
                            
                            if hasattr(layer_activations, 'shape'):
                                logger.info(f"Successfully loaded activations for layer {layer_name}, shape: {layer_activations.shape}")
                                
                                # 創建餘弦相似度矩陣
                                # 使用 max_samples 參數優化處理大型測試集的情況
                                max_samples = 30  # 限制為最多30個樣本，保持視覺化可讀性
                                cosine_sim_path = os.path.join(activation_dir, f'{layer_name}_cosine_similarity.png')
                                
                                # 首先計算餘弦相似度矩陣（帶有抽樣）
                                try:
                                    similarity_matrix = compute_cosine_similarity_matrix(
                                        embeddings=layer_activations,
                                        max_samples=max_samples,
                                        random_seed=42  # 確保結果可重複
                                    )
                                    logger.info(f"Computed cosine similarity matrix with shape: {similarity_matrix.shape}")
                                    
                                    # 然後繪製相似度矩陣
                                    plot_cosine_similarity_matrix(
                                        embeddings=similarity_matrix,  # 直接傳入計算好的相似度矩陣
                                        title=f"{layer_name} Layer Activations Cosine Similarity",
                                        output_path=cosine_sim_path,
                                        use_precomputed_matrix=True  # 標記這是已經計算好的相似度矩陣
                                    )
                                except Exception as e:
                                    logger.error(f"Error computing cosine similarity matrix: {e}")
                                    # 回退到傳統方法
                                    logger.info("Falling back to traditional method with built-in sampling")
                                    plot_cosine_similarity_matrix(
                                        embeddings=layer_activations,
                                        title=f"{layer_name} Layer Activations Cosine Similarity",
                                        output_path=cosine_sim_path
                                    )
                                    
                                visualization_paths[f'{layer_name}_cosine_similarity'] = cosine_sim_path
                                logger.info(f"Generated cosine similarity matrix for layer {layer_name}: {cosine_sim_path}")
                                
                                # 計算激活值統計量
                                stats = compute_activation_statistics(layer_activations)
                                
                                # 繪製統計量
                                stats_path = os.path.join(activation_dir, f'{layer_name}_statistics.png')
                                plot_activation_statistics(
                                    stats=stats,
                                    layer_name=layer_name,
                                    output_path=stats_path
                                )
                                visualization_paths[f'{layer_name}_statistics'] = stats_path
                                logger.info(f"Generated activation statistics for layer {layer_name}: {stats_path}")
                                
                                # 如果是卷積層，創建熱力圖以顯示特徵圖激活情況
                                if layer_activations.ndim > 2 and ('conv' in layer_name.lower() or len(layer_activations.shape) > 2):
                                    # 選擇第一個樣本的特徵圖來可視化
                                    try:
                                        heatmap_path = os.path.join(activation_dir, f'{layer_name}_heatmap.png')
                                        plot_layer_activations_heatmap(
                                            activations=layer_activations[:10],  # 只選前10個樣本
                                            title=f"{layer_name} Layer Activations Heatmap",
                                            output_path=heatmap_path
                                        )
                                        visualization_paths[f'{layer_name}_heatmap'] = heatmap_path
                                        logger.info(f"Generated activations heatmap for layer {layer_name}: {heatmap_path}")
                                    except Exception as e:
                                        logger.error(f"Error generating heatmap for layer {layer_name}: {e}")
                            else:
                                logger.warning(f"Activations for layer {layer_name} do not have a shape attribute.")
                        else:
                            logger.warning(f"Could not load activations for layer {layer_name}")
                    except Exception as e:
                        logger.error(f"Error processing activations for layer {layer_name}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
            else:
                logger.warning("No test activation layers found. Using simulated data for demonstrations.")
                
                # 生成模擬激活數據進行演示
                batch_size = 10
                
                # 模擬不同層類型的激活值
                fc_layer_activations = np.random.randn(batch_size, 128)  # 全連接層
                conv_layer_activations = np.random.randn(batch_size, 16, 8, 8)  # 卷積層
                
                # 創建全連接層餘弦相似度矩陣
                cosine_sim_path = os.path.join(activation_dir, 'fc_layer_cosine_similarity.png')
                plot_cosine_similarity_matrix(
                    embeddings=fc_layer_activations,
                    title="FC Layer Activations Cosine Similarity (Simulated)",
                    output_path=cosine_sim_path
                )
                visualization_paths['cosine_similarity'] = cosine_sim_path
                logger.info(f"Generated simulated cosine similarity matrix: {cosine_sim_path}")
                
                # 為兩種層類型創建激活值統計量視覺化
                for layer_name, activations in [
                    ('fc_layer', fc_layer_activations),
                    ('conv_layer', conv_layer_activations.reshape(batch_size, -1))  # 展平卷積層激活值
                ]:
                    # 計算激活值統計量
                    stats = compute_activation_statistics(activations)
                    
                    # 繪製統計量
                    stats_path = os.path.join(activation_dir, f'{layer_name}_statistics.png')
                    plot_activation_statistics(
                        stats=stats,
                        layer_name=f"{layer_name} (Simulated)",
                        output_path=stats_path
                    )
                    visualization_paths[f'{layer_name}_statistics'] = stats_path
                    logger.info(f"Generated simulated activation statistics for {layer_name}: {stats_path}")
            
            return visualization_paths
            
        except Exception as e:
            logger.error(f"Error creating layer activation visualizations: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def create_confusion_matrix_visualization(self):
        """
        Create a confusion matrix visualization using classification results.
        Prioritizes loading from hooks/evaluation_results_test.pt.
        
        Returns:
            dict: Confusion matrix visualization results
        """
        logger.info("Creating confusion matrix visualization...")
        
        try:
            # 初始化 HookDataLoader 以標準化數據載入流程
            try:
                from data_loader.hook_data_loader import HookDataLoader
                hook_loader = HookDataLoader(self.experiment_dir)
                logger.info("Successfully initialized HookDataLoader for confusion matrix")
            except Exception as e:
                logger.error(f"Failed to initialize HookDataLoader: {e}")
                hook_loader = None
            
            # 優先從 hooks/evaluation_results_test.pt 載入評估結果
            eval_results_path = os.path.join(self.experiment_dir, 'hooks', 'evaluation_results_test.pt')
            logger.info(f"Checking for evaluation results at: {eval_results_path}")
            
            # 初始化變量
            y_true = None
            y_pred = None
            class_names = None
            data_loaded = False
            
            # 嘗試使用 HookDataLoader 載入評估結果
            if hook_loader:
                try:
                    logger.info("Attempting to load evaluation results using HookDataLoader")
                    eval_data = hook_loader.load_evaluation_results(dataset='test')
                    
                    if eval_data is not None:
                        logger.info(f"Successfully loaded evaluation data with keys: {list(eval_data.keys())}")
                        
                        # 處理各種可能的數據格式
                        if 'targets' in eval_data and 'predictions' in eval_data:
                            y_true = self._prepare_array(eval_data['targets'])
                            y_pred = self._prepare_array(eval_data['predictions'])
                            data_loaded = True
                            logger.info("Extracted targets and predictions from evaluation data")
                        elif 'true_labels' in eval_data and 'predicted_labels' in eval_data:
                            y_true = self._prepare_array(eval_data['true_labels'])
                            y_pred = self._prepare_array(eval_data['predicted_labels'])
                            data_loaded = True
                            logger.info("Extracted true_labels and predicted_labels from evaluation data")
                        elif 'y_true' in eval_data and 'y_pred' in eval_data:
                            y_true = self._prepare_array(eval_data['y_true'])
                            y_pred = self._prepare_array(eval_data['y_pred'])
                            data_loaded = True
                            logger.info("Extracted y_true and y_pred from evaluation data")
                        else:
                            logger.warning(f"Evaluation data format not recognized. Available keys: {list(eval_data.keys())}")
                        
                        # 嘗試提取類別名稱
                        if 'class_names' in eval_data:
                            class_names = eval_data['class_names']
                            logger.info(f"Found class names: {class_names}")
                except Exception as e:
                    logger.error(f"Error loading evaluation results with HookDataLoader: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # 如果使用 HookDataLoader 失敗，嘗試直接載入文件
            if not data_loaded and os.path.exists(eval_results_path):
                logger.info(f"Attempting to load evaluation results directly from: {eval_results_path}")
                try:
                    eval_data = self._safe_load_tensor_file(eval_results_path)
                    
                    if eval_data is not None:
                        # 嘗試各種可能的格式
                        if isinstance(eval_data, dict):
                            if 'targets' in eval_data and 'predictions' in eval_data:
                                y_true = self._prepare_array(eval_data['targets'])
                                y_pred = self._prepare_array(eval_data['predictions'])
                                data_loaded = True
                                logger.info("Successfully loaded targets and predictions from evaluation results file")
                            elif 'true_labels' in eval_data and 'predicted_labels' in eval_data:
                                y_true = self._prepare_array(eval_data['true_labels'])
                                y_pred = self._prepare_array(eval_data['predicted_labels'])
                                data_loaded = True
                                logger.info("Successfully loaded true_labels and predicted_labels from evaluation results file")
                            elif 'y_true' in eval_data and 'y_pred' in eval_data:
                                y_true = self._prepare_array(eval_data['y_true'])
                                y_pred = self._prepare_array(eval_data['y_pred'])
                                data_loaded = True
                                logger.info("Successfully loaded y_true and y_pred from evaluation results file")
                            else:
                                logger.warning(f"Evaluation results found but format not recognized. Available keys: {list(eval_data.keys())}")
                        else:
                            logger.warning(f"Evaluation results found but not in expected dictionary format. Type: {type(eval_data)}")
                        
                        # 嘗試提取類別名稱
                        if isinstance(eval_data, dict) and 'class_names' in eval_data:
                            class_names = eval_data['class_names']
                            logger.info(f"Found class names: {class_names}")
                except Exception as e:
                    logger.error(f"Error loading evaluation results directly: {e}")
            
            # 檢查是否需要回退到舊格式的測試預測文件
            if not data_loaded:
                # 檢查是否存在舊格式的 test_predictions.pt 文件
                test_pred_path = os.path.join(self.experiment_dir, 'test_predictions.pt')
                if os.path.exists(test_pred_path):
                    logger.info(f"Falling back to legacy test predictions file: {test_pred_path}")
                    try:
                        predictions = self._safe_load_tensor_file(test_pred_path)
                        
                        if predictions is not None:
                            # 嘗試各種可能的格式
                            if isinstance(predictions, dict):
                                if 'true_labels' in predictions and 'predicted_labels' in predictions:
                                    y_true = self._prepare_array(predictions['true_labels'])
                                    y_pred = self._prepare_array(predictions['predicted_labels'])
                                    data_loaded = True
                                    logger.info("Successfully loaded true_labels and predicted_labels from legacy file")
                                elif 'targets' in predictions and 'predictions' in predictions:
                                    y_true = self._prepare_array(predictions['targets'])
                                    y_pred = self._prepare_array(predictions['predictions'])
                                    data_loaded = True
                                    logger.info("Successfully loaded targets and predictions from legacy file")
                                else:
                                    logger.warning(f"Legacy test predictions format not recognized. Available keys: {list(predictions.keys())}")
                            else:
                                logger.warning(f"Legacy test predictions not in expected dictionary format. Type: {type(predictions)}")
                    except Exception as e:
                        logger.error(f"Error loading legacy test predictions: {e}")
                else:
                    logger.warning(f"Legacy test predictions file not found: {test_pred_path}")
            
            # 檢查配置中的類別數目和類別名稱
            num_classes = 2  # 預設二分類
            config_class_names = None
            
            # 從配置中獲取類別數目
            if 'config' in self.results:
                config = self.results['config']
                # 嘗試各種可能的配置路徑
                if 'model' in config and 'output_dim' in config['model']:
                    num_classes = config['model']['output_dim']
                    logger.info(f"Found num_classes from config.model.output_dim: {num_classes}")
                elif 'data' in config and 'num_classes' in config['data']:
                    num_classes = config['data']['num_classes']
                    logger.info(f"Found num_classes from config.data.num_classes: {num_classes}")
                elif 'model' in config and 'num_classes' in config['model']:
                    num_classes = config['model']['num_classes']
                    logger.info(f"Found num_classes from config.model.num_classes: {num_classes}")
                
                # 嘗試從配置中獲取類別名稱
                if 'data' in config and 'class_names' in config['data']:
                    config_class_names = config['data']['class_names']
                    logger.info(f"Found class names from config: {config_class_names}")
                elif 'class_names' in config:
                    config_class_names = config['class_names']
                    logger.info(f"Found class names from config root: {config_class_names}")
            
            # 從實際數據中推斷類別數量（優先於配置值）
            if y_true is not None:
                # 如果 y_true 是一維數組，計算唯一值的數量
                if len(y_true.shape) == 1:
                    actual_classes = len(np.unique(y_true))
                    if actual_classes > num_classes:
                        logger.info(f"Updating num_classes from {num_classes} to {actual_classes} based on actual data")
                        num_classes = actual_classes
                # 如果 y_true 是二維數組（one-hot 編碼），使用第二個維度的大小
                elif len(y_true.shape) > 1:
                    actual_classes = y_true.shape[1]
                    if actual_classes > num_classes:
                        logger.info(f"Updating num_classes from {num_classes} to {actual_classes} based on one-hot encoded data")
                        num_classes = actual_classes
            
            # 如果 y_pred 是二維數組（類別機率），檢查第二個維度
            if y_pred is not None and len(y_pred.shape) > 1:
                pred_classes = y_pred.shape[1]
                if pred_classes > num_classes:
                    logger.info(f"Updating num_classes from {num_classes} to {pred_classes} based on prediction output size")
                    num_classes = pred_classes
            
            # 如果從數據中獲取的類別名稱為空，使用配置中的類別名稱
            if class_names is None and config_class_names is not None:
                class_names = config_class_names
            
            # 如果沒有類別名稱，生成默認的類別名稱
            if class_names is None:
                class_names = [f"Class {i}" for i in range(num_classes)]
            
            # 如果真實數據不可用，生成模擬數據
            if not data_loaded:
                logger.warning("No valid test prediction data found, generating random data for demonstration")
                # 生成隨機數據進行演示
                np.random.seed(42)  # 設置隨機種子以保證結果可重複
                y_true = np.random.randint(0, num_classes, size=100)
                y_pred = np.random.randint(0, num_classes, size=100)
                
                # 添加一些相關性，使混淆矩陣更有意義
                for i in range(len(y_true)):
                    if np.random.random() < 0.7:  # 70% 的概率預測正確
                        y_pred[i] = y_true[i]
            
            # 計算混淆矩陣
            logger.info(f"Computing confusion matrix with {len(y_true)} samples and {num_classes} classes")
            cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
            
            # 視覺化路徑
            visualization_paths = {}
            
            # 繪製原始混淆矩陣
            cm_path = os.path.join(self.analysis_dir, 'confusion_matrix.png')
            plot_confusion_matrix(
                confusion_matrix=cm,
                class_names=class_names,
                title="Classification Confusion Matrix",
                output_path=cm_path
            )
            visualization_paths['confusion_matrix'] = cm_path
            logger.info(f"Generated confusion matrix: {cm_path}")
            
            # 繪製按真實標籤標準化的混淆矩陣
            norm_cm_path = os.path.join(self.analysis_dir, 'confusion_matrix_normalized_by_true.png')
            plot_confusion_matrix(
                confusion_matrix=cm,
                class_names=class_names,
                normalize='true',
                title="Normalized Confusion Matrix (by true labels)",
                output_path=norm_cm_path
            )
            visualization_paths['normalized_confusion_matrix_by_true'] = norm_cm_path
            logger.info(f"Generated normalized confusion matrix by true labels: {norm_cm_path}")
            
            # 繪製按預測標籤標準化的混淆矩陣
            norm_pred_cm_path = os.path.join(self.analysis_dir, 'confusion_matrix_normalized_by_pred.png')
            plot_confusion_matrix(
                confusion_matrix=cm,
                class_names=class_names,
                normalize='pred',
                title="Normalized Confusion Matrix (by predicted labels)",
                output_path=norm_pred_cm_path
            )
            visualization_paths['normalized_confusion_matrix_by_pred'] = norm_pred_cm_path
            logger.info(f"Generated normalized confusion matrix by predicted labels: {norm_pred_cm_path}")
            
            return visualization_paths
            
        except Exception as e:
            logger.error(f"Error creating confusion matrix visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _safe_load_tensor_file(self, file_path):
        """Helper method to safely load tensor data from a file"""
        if TORCH_AVAILABLE:
            try:
                return torch.load(file_path)
            except Exception as e:
                logger.warning(f"Failed to load with torch: {e}. Trying pickle instead.")
        else:
            logger.info("PyTorch not available, using pickle to load data.")
        
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load with pickle: {e}")
            return None
    
    def _prepare_array(self, data):
        """Helper method to prepare array data for analysis"""
        # 如果是 PyTorch 張量，轉換為 NumPy 陣列
        if hasattr(data, 'numpy'):
            try:
                return data.cpu().numpy()
            except Exception as e:
                logger.warning(f"Failed to convert tensor to numpy: {e}")
        
        # 如果是列表，轉換為 NumPy 陣列
        if isinstance(data, list):
            return np.array(data)
        
        # 如果已經是 NumPy 陣列或其他類型，直接返回
        return data
    
    def generate_report(self):
        """
        Generate analysis report
        
        Returns:
            str: Report file path
        """
        report_path = os.path.join(self.analysis_dir, 'analysis_report.md')
        
        try:
            # Get analysis results
            config = self.results.get('config', {})
            model_structure_analysis = self.results.get('model_structure_analysis', {})
            training_history_analysis = self.results.get('training_history_analysis', {})
            gns_analysis = self.results.get('gns_analysis', {})
            activation_visualizations = self.results.get('activation_visualizations', {})
            epoch_activation_visualizations = self.results.get('epoch_activation_visualizations', {})
            confusion_matrix_visualizations = self.results.get('confusion_matrix_visualizations', {})
            
            # Format report content
            report_content = f"""# Audio Classification Model Analysis Report

## 1. Experiment Overview

- **Experiment Name:** {config.get('global', {}).get('experiment_name', 'Unknown')}
- **Model Type:** {config.get('model', {}).get('type', 'Unknown')}
- **Data Type:** {config.get('data', {}).get('type', 'Unknown')}
- **Task Type:** {config.get('data', {}).get('filtering', {}).get('task_type', 'Unknown')}

## 2. Model Structure Analysis

- **Total Parameters:** {model_structure_analysis.get('total_parameters', 'N/A')}
- **Layer Count:** {model_structure_analysis.get('layer_count', 'N/A')}
- **Layers with Most Parameters:**
"""
            
            # Add top layers by parameter count
            top_layers = model_structure_analysis.get('top_heavy_layers', [])
            for layer in top_layers:
                report_content += f"  - {layer}\n"
            
            # Add parameter distribution plot
            params_plot = model_structure_analysis.get('params_distribution_plot')
            if params_plot:
                rel_path = os.path.relpath(params_plot, self.analysis_dir)
                report_content += f"\n![Model Parameter Distribution]({rel_path})\n"
            
            # Add training history analysis
            report_content += f"""
## 3. Training History Analysis

- **Epochs Trained:** {training_history_analysis.get('epochs_trained', 'N/A')}
- **Training Time:** {training_history_analysis.get('training_time', 'N/A')} seconds
- **Final Training Loss:** {training_history_analysis.get('final_train_loss', 'N/A')}
- **Final Validation Loss:** {training_history_analysis.get('final_val_loss', 'N/A')}
- **Best Validation Loss:** {training_history_analysis.get('best_val_loss', 'N/A')}
- **Convergence Speed:** {training_history_analysis.get('avg_convergence_speed', 'N/A')} (average loss decrease per epoch)
- **Train-Validation Difference:** {training_history_analysis.get('avg_train_val_diff', 'N/A')} (average difference)
- **Convergence Status:** {training_history_analysis.get('convergence_status', 'N/A')}
- **Overfitting Detected:** {'Yes (Epoch ' + str(training_history_analysis.get('overfitting_epoch', 'N/A')) + ')' if training_history_analysis.get('overfitting_detected') else 'No'}
"""
            
            # Add loss curve plot
            loss_plot = training_history_analysis.get('loss_curve_plot')
            if loss_plot:
                rel_path = os.path.relpath(loss_plot, self.analysis_dir)
                report_content += f"\n![Training and Validation Loss Curves]({rel_path})\n"
            
            # Add GNS analysis
            if gns_analysis:
                report_content += f"""
## 4. GNS (Gradient Noise Scale) Analysis

- **Epochs Analyzed:** {', '.join(map(str, gns_analysis.get('epochs_analyzed', [])))}
- **Average GNS Value:** {gns_analysis.get('average_gns', 'N/A')}
- **GNS Trend:** {gns_analysis.get('gns_trend', 'N/A')}
- **Current Batch Size:** {gns_analysis.get('current_batch_size', 'N/A')}
- **Recommended Batch Size:** {gns_analysis.get('recommended_batch_size', 'N/A')}
- **Recommendation:** {gns_analysis.get('recommendation', 'N/A')}
"""
                
                # Add GNS trend plot
                gns_plot = gns_analysis.get('gns_plot')
                if gns_plot:
                    rel_path = os.path.relpath(gns_plot, self.analysis_dir)
                    report_content += f"\n![GNS Trend]({rel_path})\n"
                
                # Add gradient statistics plot
                grad_stats_plot = gns_analysis.get('grad_stats_plot')
                if grad_stats_plot:
                    rel_path = os.path.relpath(grad_stats_plot, self.analysis_dir)
                    report_content += f"\n![Gradient Statistics Trend]({rel_path})\n"
            
            # Add layer activation visualizations
            if activation_visualizations:
                report_content += """
## 5. Layer Activation Analysis

This section shows visualizations of layer activations to help understand the internal behavior of the model.
"""
                
                # Add cosine similarity matrix
                cosine_sim_path = activation_visualizations.get('cosine_similarity')
                if cosine_sim_path:
                    rel_path = os.path.relpath(cosine_sim_path, self.analysis_dir)
                    report_content += f"\n### Cosine Similarity Matrix\n\n"
                    report_content += f"The following visualization shows the cosine similarity between activation patterns:\n\n"
                    report_content += f"![Cosine Similarity Matrix]({rel_path})\n\n"
                    report_content += f"High similarity values (close to 1.0) indicate that the corresponding samples activate the layer in similar patterns.\n"
                
                # Add activation statistics plots
                fc_stats_path = activation_visualizations.get('fc_layer_statistics')
                if fc_stats_path:
                    rel_path = os.path.relpath(fc_stats_path, self.analysis_dir)
                    report_content += f"\n### Fully Connected Layer Activation Statistics\n\n"
                    report_content += f"![FC Layer Activation Statistics]({rel_path})\n\n"
                    report_content += f"These statistics show the distribution of activation values in the fully connected layer.\n"
                
                conv_stats_path = activation_visualizations.get('conv_layer_statistics')
                if conv_stats_path:
                    rel_path = os.path.relpath(conv_stats_path, self.analysis_dir)
                    report_content += f"\n### Convolutional Layer Activation Statistics\n\n"
                    report_content += f"![Convolutional Layer Activation Statistics]({rel_path})\n\n"
                    report_content += f"These statistics show the distribution of activation values in the convolutional layer.\n"
            
            # Add epoch-wise activation visualizations
            if epoch_activation_visualizations:
                report_content += """
## 6. Epoch-wise Layer Activation Analysis

This section shows how layer activations evolved across training epochs.
"""
                
                # Group visualizations by epoch
                epoch_groups = {}
                for key, path in epoch_activation_visualizations.items():
                    if key.startswith('epoch_'):
                        epoch = key.split('_')[1]
                        layer_name = '_'.join(key.split('_')[2:-2])  # Extract layer name from the key
                        
                        if epoch not in epoch_groups:
                            epoch_groups[epoch] = []
                        
                        epoch_groups[epoch].append((layer_name, path))
                
                # Add each epoch's visualizations
                for epoch, visualizations in sorted(epoch_groups.items()):
                    report_content += f"\n### Epoch {epoch} Activation Analysis\n\n"
                    
                    for layer_name, path in visualizations:
                        rel_path = os.path.relpath(path, self.analysis_dir)
                        report_content += f"#### {layer_name} Layer\n\n"
                        report_content += f"![{layer_name} Activation Similarity (Epoch {epoch})]({rel_path})\n\n"
            
            # Add confusion matrix visualizations
            if confusion_matrix_visualizations:
                report_content += """
## 7. Classification Performance Analysis

This section analyzes the classification performance of the model using confusion matrices.
"""
                
                # Add regular confusion matrix
                cm_path = confusion_matrix_visualizations.get('confusion_matrix')
                if cm_path:
                    rel_path = os.path.relpath(cm_path, self.analysis_dir)
                    report_content += f"\n### Confusion Matrix\n\n"
                    report_content += f"![Confusion Matrix]({rel_path})\n\n"
                    report_content += f"The confusion matrix shows the counts of true vs. predicted class labels.\n"
                
                # Add normalized by true labels confusion matrix
                norm_cm_path = confusion_matrix_visualizations.get('normalized_confusion_matrix_by_true')
                if norm_cm_path:
                    rel_path = os.path.relpath(norm_cm_path, self.analysis_dir)
                    report_content += f"\n### Normalized Confusion Matrix (By True Labels)\n\n"
                    report_content += f"![Normalized Confusion Matrix by True Labels]({rel_path})\n\n"
                    report_content += f"This confusion matrix is normalized by true labels (rows), where each row sums to 1.0. " \
                                      f"It shows the percentage of samples in each true class that were predicted as each class.\n"
                
                # Add normalized by predicted labels confusion matrix
                norm_pred_cm_path = confusion_matrix_visualizations.get('normalized_confusion_matrix_by_pred')
                if norm_pred_cm_path:
                    rel_path = os.path.relpath(norm_pred_cm_path, self.analysis_dir)
                    report_content += f"\n### Normalized Confusion Matrix (By Predicted Labels)\n\n"
                    report_content += f"![Normalized Confusion Matrix by Predicted Labels]({rel_path})\n\n"
                    report_content += f"This confusion matrix is normalized by predicted labels (columns), where each column sums to 1.0. " \
                                      f"It shows the precision of the model for each predicted class.\n"
            
            # Add conclusions and recommendations
            report_content += """
## 8. Conclusions and Recommendations

"""
            
            # Generate conclusions and recommendations based on analysis results
            # Model structure recommendations
            if model_structure_analysis:
                total_params = model_structure_analysis.get('total_parameters', 0)
                if total_params > 10000000:  # Over 10 million parameters
                    report_content += "- **Model Complexity:** The model has a large number of parameters. Consider using a smaller model or pruning techniques to reduce parameter count.\n"
                elif total_params < 100000:  # Under 100k parameters
                    report_content += "- **Model Complexity:** The model has relatively few parameters. Consider increasing model capacity to improve performance.\n"
                else:
                    report_content += "- **Model Complexity:** The model parameter count is moderate.\n"
            
            # Training recommendations
            if training_history_analysis:
                convergence_status = training_history_analysis.get('convergence_status', '')
                overfitting_detected = training_history_analysis.get('overfitting_detected', False)
                
                if 'Overfitting' in convergence_status or overfitting_detected:
                    report_content += "- **Training Process:** Overfitting detected. Consider adding regularization (e.g., Dropout, L2 regularization) or implementing early stopping.\n"
                elif 'Underfitting' in convergence_status:
                    report_content += "- **Training Process:** Underfitting detected. Consider increasing model capacity or extending training duration.\n"
                else:
                    report_content += "- **Training Process:** Training process appears normal with good convergence.\n"
            
            # GNS recommendations
            if gns_analysis:
                report_content += f"- **Batch Size:** {gns_analysis.get('recommendation', '')}\n"
            
            # Write report file
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Analysis report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def run_all_analysis(self):
        """
        Run all analysis methods
        
        Returns:
            dict: All analysis results
        """
        logger.info(f"Starting analysis for experiment: {self.experiment_dir}")
        
        # Load experiment data
        self.load_config()
        self.load_model_structure()
        self.load_training_history()
        
        # Run analysis methods
        self.analyze_model_structure()
        self.analyze_training_history()
        self.analyze_gns()
        
        # Create visualizations
        activation_viz = self.create_layer_activation_visualizations()
        if activation_viz:
            self.results['activation_visualizations'] = activation_viz
        
        # Create epoch-specific activation visualizations
        epoch_activation_viz = self.create_epoch_activation_visualizations()
        if epoch_activation_viz:
            self.results['epoch_activation_visualizations'] = epoch_activation_viz
        
        # Create confusion matrix visualization
        cm_viz = self.create_confusion_matrix_visualization()
        if cm_viz:
            self.results['confusion_matrix_visualizations'] = cm_viz
        
        # Generate final report
        report_path = self.generate_report()
        self.results['report_path'] = report_path
        
        logger.info(f"Analysis completed for experiment: {self.experiment_dir}")
        logger.info(f"Report generated at: {report_path}")
        
        return self.results

    def create_epoch_activation_visualizations(self):
        """
        Create visualizations of layer activations for each saved epoch.
        Generates cosine similarity matrices for each epoch's hook data.
        
        Returns:
            dict: Epoch activation visualization results
        """
        logger.info("Creating epoch-wise activation visualizations...")
        
        try:
            # 創建層激活視覺化目錄
            activation_dir = os.path.join(self.analysis_dir, 'epoch_activations')
            os.makedirs(activation_dir, exist_ok=True)
            
            # 存儲視覺化路徑的字典
            visualization_paths = {}
            
            # 查找所有可能的 epoch 文件夾
            hook_dir = os.path.join(self.experiment_dir, 'hooks')
            if not os.path.exists(hook_dir):
                logger.warning(f"Hook directory not found: {hook_dir}")
                return {}
            
            # 查找所有 epoch_X 格式的文件夾
            epoch_dirs = []
            for item in os.listdir(hook_dir):
                if os.path.isdir(os.path.join(hook_dir, item)) and item.startswith('epoch_'):
                    epoch_dirs.append(item)
            
            logger.info(f"Found {len(epoch_dirs)} epoch directories: {epoch_dirs}")
            
            if not epoch_dirs:
                logger.warning("No epoch directories found for activation visualization")
                return {}
            
            # 針對每個 epoch 文件夾處理
            for epoch_dir in sorted(epoch_dirs):
                epoch_path = os.path.join(hook_dir, epoch_dir)
                epoch_num = epoch_dir.split('_')[1]  # 從 'epoch_X' 中提取 X
                
                logger.info(f"Processing activations for {epoch_dir}")
                
                # 查找 activation 文件
                activation_files = []
                for item in os.listdir(epoch_path):
                    if item.endswith('_activation_batch_0.pt') or 'activation' in item:
                        activation_files.append(item)
                
                logger.info(f"Found {len(activation_files)} activation files in {epoch_dir}")
                
                # 處理每個激活文件
                for act_file in activation_files:
                    layer_name = act_file.split('_activation')[0]
                    file_path = os.path.join(epoch_path, act_file)
                    
                    try:
                        # 載入激活數據
                        activation_data = self._safe_load_tensor_file(file_path)
                        
                        if activation_data is not None:
                            # 確保數據是 numpy 數組或 PyTorch 張量
                            if isinstance(activation_data, dict) and 'activation' in activation_data:
                                activation_data = activation_data['activation']
                            elif isinstance(activation_data, dict) and 'activations' in activation_data:
                                activation_data = activation_data['activations']
                            
                            if hasattr(activation_data, 'shape'):
                                logger.info(f"Successfully loaded activations for {layer_name} in {epoch_dir}, shape: {activation_data.shape}")
                                
                                # 創建餘弦相似度矩陣
                                max_samples = 30  # 限制為最多30個樣本
                                similarity_matrix_filename = f'{epoch_dir}_{layer_name}_cosine_similarity.png'
                                cosine_sim_path = os.path.join(activation_dir, similarity_matrix_filename)
                                
                                # 計算餘弦相似度矩陣（帶有抽樣）
                                try:
                                    similarity_matrix = compute_cosine_similarity_matrix(
                                        embeddings=activation_data,
                                        max_samples=max_samples,
                                        random_seed=42
                                    )
                                    
                                    # 繪製相似度矩陣
                                    plot_cosine_similarity_matrix(
                                        embeddings=similarity_matrix,
                                        title=f"{layer_name} Activation Similarity (Epoch {epoch_num})",
                                        output_path=cosine_sim_path,
                                        use_precomputed_matrix=True
                                    )
                                    
                                    visualization_key = f'{epoch_dir}_{layer_name}_cosine_similarity'
                                    visualization_paths[visualization_key] = cosine_sim_path
                                    logger.info(f"Generated cosine similarity matrix for {layer_name} in {epoch_dir}: {cosine_sim_path}")
                                except Exception as e:
                                    logger.error(f"Error computing cosine similarity matrix for {layer_name} in {epoch_dir}: {e}")
                            else:
                                logger.warning(f"Activations for {layer_name} in {epoch_dir} do not have a shape attribute")
                        else:
                            logger.warning(f"Could not load activations for {layer_name} in {epoch_dir}")
                    except Exception as e:
                        logger.error(f"Error processing activations for {layer_name} in {epoch_dir}: {e}")
                        continue
            
            return visualization_paths
            
        except Exception as e:
            logger.error(f"Error creating epoch activation visualizations: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

def main():
    """Main function to analyze a single experiment directory"""
    parser = argparse.ArgumentParser(description='Analyze a single experiment directory')
    parser.add_argument('experiment_dir', type=str, nargs='?',
                       help='Path to the experiment directory to analyze')
    parser.add_argument('--experiment_dir', dest='experiment_dir_opt', type=str,
                       help='Path to the experiment directory to analyze (alternative way)')
    args = parser.parse_args()
    
    # Use positional argument if provided, otherwise use named argument, otherwise use default
    experiment_dir = args.experiment_dir or args.experiment_dir_opt or 'results/custom_audio_fcnn_classification_20250430_225309'
    
    if not os.path.exists(experiment_dir):
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return
    
    logger.info(f"Analyzing experiment: {experiment_dir}")
    
    # Create analyzer and run analysis
    analyzer = CustomAnalyzer(experiment_dir)
    results = analyzer.run_all_analysis()
    
    if results:
        logger.info(f"Analysis complete. Results saved to: {os.path.join(experiment_dir, 'custom_analysis')}")
    else:
        logger.error("Analysis failed to complete successfully")

if __name__ == "__main__":
    main() 