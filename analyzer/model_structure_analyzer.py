"""
模型結構分析模組。

此模組提供用於分析深度學習模型結構的功能。它可以解析模型架構、參數分布和結構特性。

Classes:
    ModelStructureAnalyzer: 分析模型結構的類別。
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
import re

import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from .base_analyzer import BaseAnalyzer

class ModelStructureAnalyzer(BaseAnalyzer):
    """
    模型結構分析器，用於分析模型結構和參數分布。
    
    此分析器主要關注模型架構、層級結構、參數分布和其他結構特性。
    
    Attributes:
        experiment_dir (str): 實驗結果的目錄路徑。
        model_structure (Dict): 儲存解析後的模型結構。
        model_stats (Dict): 儲存模型統計信息。
    """
    
    # 模型類型識別的正則表達式
    MODEL_TYPE_PATTERNS = {
        'cnn': [r'conv', r'resnet', r'vgg', r'densenet', r'mobilenet', r'efficientnet'],
        'rnn': [r'lstm', r'gru', r'rnn'],
        'transformer': [r'transformer', r'attention', r'mha', r'encoder', r'decoder', r'bert', r'gpt'],
        'swin': [r'swin', r'window', r'patch_embed', r'window_attention'],
        'vision_transformer': [r'vit', r'vision_transformer', r'patch_embedding'],
        'mlp': [r'mlp', r'linear', r'feed_forward', r'fc_layer'],
        'graph_neural_network': [r'gcn', r'gat', r'graphconv', r'graphsage'],
        'generative': [r'gan', r'vae', r'flow', r'diffusion', r'autoencoder', r'generator', r'discriminator']
    }
    
    def __init__(self, experiment_dir: str):
        """
        初始化模型結構分析器。
        
        Args:
            experiment_dir (str): 實驗結果的目錄路徑。
        """
        super().__init__(experiment_dir)
        self.model_structure = {}
        self.model_stats = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_model_structure(self) -> Dict:
        """
        從model_structure.json載入模型結構。
        
        Returns:
            Dict: 包含模型結構的字典。
        """
        structure_path = os.path.join(self.experiment_dir, 'model_structure.json')
        try:
            with open(structure_path, 'r') as f:
                self.model_structure = json.load(f)
                return self.model_structure
        except Exception as e:
            self.logger.error(f"載入模型結構文件失敗: {e}")
            raise
    
    def analyze(self, save_results: bool = True) -> Dict:
        """
        分析模型結構並計算相關統計信息。
        
        Args:
            save_results (bool, optional): 是否保存分析結果。默認為True。
        
        Returns:
            Dict: 包含分析結果的字典。
        """
        # 載入模型結構
        self.load_model_structure()
        
        # 初始化結果字典
        self.results = {
            'model_name': None,
            'model_type': None,
            'architecture_type': None,
            'total_params': 0,
            'trainable_params': 0,
            'non_trainable_params': 0,
            'layer_count': 0,
            'layer_types': {},
            'parameter_distribution': {},
            'model_depth': 0,
            'sparse_layers': [],
            'largest_layers': [],
            'layer_complexity': {},
            'weight_distributions': {},
            'sequential_paths': [],
            'parameter_efficiency': None
        }
        
        try:
            # 執行具體的分析邏輯
            if self.model_structure:
                self.results['model_name'] = self.model_structure.get('model_name', 'Unknown')
                
                # 計算基本信息
                self._analyze_basic_info()
                
                # 分析架構類型
                self._identify_architecture_type()
                
                # 提取層級結構
                self._analyze_layer_hierarchy()
                
                # 分析參數分布
                self._analyze_parameter_distribution()
                
                # 分析層級複雜度
                self._analyze_layer_complexity()
                
                # 分析權重分布
                self._analyze_weight_distributions()
                
                # 尋找稀疏層
                self._identify_sparse_layers()
                
                # 尋找參數最多的層
                self._identify_largest_layers()
                
                # 分析序列路徑
                self._analyze_sequential_paths()
                
                # 計算參數效率
                self._calculate_parameter_efficiency()
                
                # 分析層間連接性
                self._analyze_layer_connectivity()
                
                # 分析計算複雜度
                self._analyze_computational_complexity()
            
            if save_results:
                self.save_results(os.path.join(self.experiment_dir, 'analysis'))
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"分析模型結構時出錯: {e}")
            raise
    
    def _analyze_basic_info(self):
        """
        分析模型的基本信息。
        """
        if 'layers' in self.model_structure:
            layers = self.model_structure['layers']
            self.results['layer_count'] = len(layers)
            
            # 計算參數量
            total_params = 0
            trainable_params = 0
            
            for layer in layers:
                if 'parameters' in layer:
                    total_params += layer.get('parameters', 0)
                    if layer.get('trainable', True):
                        trainable_params += layer.get('parameters', 0)
            
            self.results['total_params'] = total_params
            self.results['trainable_params'] = trainable_params
            self.results['non_trainable_params'] = total_params - trainable_params
            
            # 計算層類型分布
            layer_types = Counter([layer.get('type', 'unknown') for layer in layers])
            self.results['layer_types'] = dict(layer_types)
    
    def _identify_architecture_type(self):
        """
        識別模型的架構類型 (CNN, RNN, Transformer等)。
        """
        if 'model_name' in self.model_structure:
            model_name = self.model_structure['model_name'].lower()
            
            # 先通過模型名稱判斷
            for model_type, patterns in self.MODEL_TYPE_PATTERNS.items():
                if any(re.search(pattern, model_name) for pattern in patterns):
                    self.results['architecture_type'] = model_type
                    return
            
            # 如果模型名稱無法判斷，通過層類型判斷
            if 'layers' in self.model_structure:
                layer_types = [layer.get('type', '').lower() for layer in self.model_structure['layers']]
                layer_names = [layer.get('name', '').lower() for layer in self.model_structure['layers']]
                
                all_strings = ' '.join(layer_types + layer_names)
                
                for model_type, patterns in self.MODEL_TYPE_PATTERNS.items():
                    if any(re.search(pattern, all_strings) for pattern in patterns):
                        self.results['architecture_type'] = model_type
                        return
        
        # 默認為未知架構
        self.results['architecture_type'] = 'unknown'
    
    def _analyze_layer_hierarchy(self):
        """
        分析層級結構關係，計算模型深度。
        """
        if 'layers' in self.model_structure:
            # 計算模型深度 (最長路徑)
            graph = defaultdict(list)
            indegree = defaultdict(int)
            
            # 建立依賴圖
            for layer in self.model_structure['layers']:
                layer_id = layer.get('id', layer.get('name', ''))
                inputs = layer.get('inputs', [])
                
                for input_id in inputs:
                    graph[input_id].append(layer_id)
                    indegree[layer_id] += 1
            
            # 拓撲排序找最長路徑
            visited = set()
            max_depth = 0
            depths = {}
            
            def dfs(node, depth=0):
                nonlocal max_depth
                if node in visited:
                    return depths.get(node, 0)
                visited.add(node)
                max_depth = max(max_depth, depth)
                depths[node] = depth
                
                for next_node in graph.get(node, []):
                    dfs(next_node, depth + 1)
                    depths[node] = max(depths[node], depths.get(next_node, 0) + 1)
                
                return depths[node]
            
            # 尋找入度為0的節點作為起點
            for layer in self.model_structure['layers']:
                layer_id = layer.get('id', layer.get('name', ''))
                if indegree[layer_id] == 0:
                    dfs(layer_id)
            
            self.results['model_depth'] = max_depth
    
    def _analyze_parameter_distribution(self):
        """
        分析模型參數分布情況。
        """
        if 'layers' in self.model_structure:
            # 按層類型統計參數
            params_by_type = defaultdict(int)
            params_by_layer = {}
            
            for layer in self.model_structure['layers']:
                layer_type = layer.get('type', 'unknown')
                params = layer.get('parameters', 0)
                layer_id = layer.get('id', layer.get('name', ''))
                
                params_by_type[layer_type] += params
                params_by_layer[layer_id] = params
            
            # 計算參數分布百分比
            total_params = sum(params_by_type.values())
            if total_params > 0:
                params_by_type_pct = {k: (v / total_params) * 100 for k, v in params_by_type.items()}
                self.results['parameter_distribution'] = {
                    'by_type': dict(params_by_type),
                    'by_type_pct': dict(params_by_type_pct),
                    'by_layer': dict(params_by_layer)
                }
                
                # 計算參數集中度 (Gini係數)
                gini = self._calculate_gini_coefficient(list(params_by_layer.values()))
                self.results['parameter_distribution']['concentration'] = {
                    'gini_coefficient': gini
                }
                
                # 計算十分位數分析
                layer_params = np.array(list(params_by_layer.values()))
                percentiles = np.percentile(layer_params, [10, 25, 50, 75, 90])
                self.results['parameter_distribution']['percentiles'] = {
                    'p10': float(percentiles[0]),
                    'p25': float(percentiles[1]),
                    'p50': float(percentiles[2]),
                    'p75': float(percentiles[3]),
                    'p90': float(percentiles[4])
                }
    
    def _analyze_weight_distributions(self):
        """
        分析模型權重分布情況。
        
        注意：在實際使用時，這個函數可能需要訪問模型權重文件，
        目前僅實現一個基於層類型的理論權重分布估計。
        """
        if 'layers' in self.model_structure:
            # 獲取各層信息
            weight_distribution_estimates = {}
        
            for layer in self.model_structure['layers']:
                layer_id = layer.get('id', layer.get('name', ''))
                layer_type = layer.get('type', 'unknown')
                params = layer.get('parameters', 0)
                
                # 根據層類型估計權重分布特性
                if params > 0:
                    # 默認估計值
                    distribution = {
                        'estimated_mean': 0.0,
                        'estimated_std': None,
                        'distribution_type': 'unknown'
                    }
                    
                    # 根據層類型設置不同的估計值
                    if 'conv' in layer_type.lower():
                        # 卷積層通常使用何凱明初始化
                        distribution['distribution_type'] = 'he_normal'
                        distribution['estimated_std'] = np.sqrt(2.0 / params)
                    elif 'batch_norm' in layer_type.lower():
                        # Batch Norm層通常有均值接近0、方差為1的分布
                        distribution['distribution_type'] = 'normal'
                        distribution['estimated_mean'] = 0.0
                        distribution['estimated_std'] = 1.0
                    elif 'linear' in layer_type.lower() or 'dense' in layer_type.lower():
                        # 線性層通常使用xavier初始化
                        input_size = layer.get('input_size', [0])
                        if isinstance(input_size, list) and len(input_size) > 0:
                            input_dim = input_size[0]
                            distribution['distribution_type'] = 'xavier'
                            # 避免除以零的錯誤
                            if input_dim > 0:
                                distribution['estimated_std'] = np.sqrt(2.0 / (input_dim + params / input_dim))
                            else:
                                # 對於無效的input_dim，使用備用公式
                                distribution['estimated_std'] = np.sqrt(2.0 / (1 + params))
                    
                    weight_distribution_estimates[layer_id] = distribution
            
            self.results['weight_distributions'] = weight_distribution_estimates
    
    def _analyze_layer_complexity(self):
        """
        分析層級複雜度，包括參數量、計算複雜度、和IO複雜度。
        """
        if 'layers' in self.model_structure:
            complexity_by_layer = {}
        
            for layer in self.model_structure['layers']:
                layer_id = layer.get('id', layer.get('name', ''))
                layer_type = layer.get('type', 'unknown')
                params = layer.get('parameters', 0)
            
                # 獲取輸入和輸出形狀
                input_shape = layer.get('input_size', [])
                output_shape = layer.get('output_size', [])
                
                # 初始化複雜度字典
                complexity = {
                    'parameter_complexity': params,
                    'computational_complexity': None,
                    'io_complexity': None,
                    'memory_footprint': None,
                    'complexity_score': None
                }
                
                # 估計計算複雜度（FLOPs）
                flops = self._estimate_flops(layer)
                complexity['computational_complexity'] = flops
                
                # 估計IO複雜度（輸入+輸出元素數量）
                io_elements = 0
                if isinstance(input_shape, list):
                    io_elements += np.prod(input_shape) if input_shape else 0
                if isinstance(output_shape, list):
                    io_elements += np.prod(output_shape) if output_shape else 0
                complexity['io_complexity'] = int(io_elements)
                
                # 估計記憶體佔用（參數+激活值）
                memory = params * 4  # 假設每個參數為4字節（float32）
                if isinstance(output_shape, list) and output_shape:
                    memory += np.prod(output_shape) * 4  # 激活值的記憶體
                complexity['memory_footprint'] = int(memory)
            
                # 計算綜合複雜度分數
                if flops is not None and params > 0:
                    # 使用log來處理大範圍的數值差異
                    log_params = np.log(max(1, params))
                    log_flops = np.log(max(1, flops))
                    log_io = np.log(max(1, io_elements)) if io_elements > 0 else 0
                    
                    # 綜合分數（可自定義權重）
                    complexity['complexity_score'] = 0.4 * log_params + 0.4 * log_flops + 0.2 * log_io
                
                complexity_by_layer[layer_id] = complexity
            
            self.results['layer_complexity'] = complexity_by_layer
            
            # 計算模型整體複雜度
            total_params = self.results.get('total_params', 0)
            total_flops = sum(c.get('computational_complexity', 0) or 0 for c in complexity_by_layer.values())
            total_memory = sum(c.get('memory_footprint', 0) or 0 for c in complexity_by_layer.values())
            
            self.results['model_complexity'] = {
                'total_params': total_params,
                'total_flops': total_flops,
                'total_memory': total_memory,
                'flops_per_param': total_flops / total_params if total_params > 0 else 0
            }
    
    def _estimate_flops(self, layer: Dict) -> Optional[int]:
        """
        估計層的計算複雜度（FLOPs）。
        
        Args:
            layer (Dict): 層的信息字典。
            
        Returns:
            Optional[int]: 估計的FLOPs，如果無法估計則為None。
        """
        layer_type = layer.get('type', '').lower()
        input_shape = layer.get('input_size', [])
        output_shape = layer.get('output_size', [])
        params = layer.get('parameters', 0)
        
        # 確保輸入和輸出形狀是有效的
        if not isinstance(input_shape, list) or not isinstance(output_shape, list):
            return None
        
        # 對於不同類型的層，估計方式不同
        if 'conv' in layer_type:
            # 卷積層
            if len(input_shape) >= 3 and len(output_shape) >= 3:
                kernel_size = layer.get('kernel_size', [3, 3])
                if not isinstance(kernel_size, list):
                    kernel_size = [kernel_size, kernel_size]
                    
                in_channels = input_shape[-3] if len(input_shape) >= 3 else 1
                out_channels = output_shape[-3] if len(output_shape) >= 3 else 1
                    
                # 輸出特徵圖的大小
                output_h = output_shape[-2] if len(output_shape) >= 2 else 1
                output_w = output_shape[-1] if len(output_shape) >= 1 else 1
                
                # 每個輸出位置的計算量：kernel_size[0] * kernel_size[1] * in_channels
                # 總計算量：輸出位置數 * 每個位置的計算量 * 輸出通道數
                return output_h * output_w * kernel_size[0] * kernel_size[1] * in_channels * out_channels
            
        elif 'linear' in layer_type or 'dense' in layer_type or 'fc' in layer_type:
            # 全連接層
            if params > 0:
                # FLOPs = 2 * 參數數量（乘法+加法）
                return 2 * params
                
        elif 'batch_norm' in layer_type:
            # Batch Normalization
            if len(input_shape) > 0:
                # 每個元素需要4個操作：減均值、除標準差、乘gamma、加beta
                return 4 * np.prod(input_shape)
                
        elif 'pool' in layer_type:
            # 池化層
            if len(input_shape) >= 3 and len(output_shape) >= 3:
                # 假設是常見的池化窗口（2x2或3x3）
                pool_size = layer.get('pool_size', [2, 2])
                if not isinstance(pool_size, list):
                    pool_size = [pool_size, pool_size]
                
                # 每個輸出位置需要在池化窗口中進行比較操作
                channels = input_shape[-3] if len(input_shape) >= 3 else 1
                output_h = output_shape[-2] if len(output_shape) >= 2 else 1
                output_w = output_shape[-1] if len(output_shape) >= 1 else 1
                
                return output_h * output_w * channels * np.prod(pool_size)
                
        # 對於其他類型的層，可能需要特定的估計方法
        return None
    
    def _identify_sparse_layers(self, threshold=0.5):
        """
        識別模型中的稀疏層。
        
        Args:
            threshold (float, optional): 稀疏度閾值，默認為0.5。
        """
        if 'layers' in self.model_structure:
            sparse_layers = []
        
            for layer in self.model_structure['layers']:
                layer_id = layer.get('id', layer.get('name', ''))
                sparsity = layer.get('sparsity', 0)
                
                # 如果層有稀疏度信息且超過閾值，則視為稀疏層
                if sparsity > threshold:
                    sparse_layers.append({
                        'layer_id': layer_id,
                        'sparsity': sparsity,
                        'parameters': layer.get('parameters', 0)
                    })
        
            # 按稀疏度降序排序
            self.results['sparse_layers'] = sorted(sparse_layers, key=lambda x: x['sparsity'], reverse=True)
    
    def _identify_largest_layers(self, top_n=5):
        """
        識別模型中參數最多的層。
        
        Args:
            top_n (int, optional): 返回的層數量，默認為5。
        """
        if 'layers' in self.model_structure:
            # 按參數量排序
            layers_by_params = sorted(
                self.model_structure['layers'],
                key=lambda x: x.get('parameters', 0),
                reverse=True
            )
            
            # 取前N個
            largest_layers = []
            for layer in layers_by_params[:top_n]:
                layer_id = layer.get('id', layer.get('name', ''))
                largest_layers.append({
                    'layer_id': layer_id,
                    'type': layer.get('type', 'unknown'),
                    'parameters': layer.get('parameters', 0)
                })
            
            self.results['largest_layers'] = largest_layers
    
    def _analyze_sequential_paths(self):
        """
        分析模型中的序列路徑，找出主要的數據流路徑。
        """
        if 'layers' in self.model_structure:
            # 構建圖
            graph = defaultdict(list)
            
            # 建立層之間的連接關係
            for layer in self.model_structure['layers']:
                layer_id = layer.get('id', layer.get('name', ''))
                inputs = layer.get('inputs', [])
                
                for input_id in inputs:
                    graph[input_id].append(layer_id)
            
            # 尋找入度為0的節點作為起點
            starting_nodes = []
            for layer in self.model_structure['layers']:
                layer_id = layer.get('id', layer.get('name', ''))
                inputs = layer.get('inputs', [])
                if not inputs:
                    starting_nodes.append(layer_id)
            
            # 找出所有從起點到終點的路徑
            paths = []
            visited = set()
            
            def dfs_paths(node, path, visited):
                if node in visited:
                    return
                
                visited.add(node)
                curr_path = path + [node]
                
                # 如果是終點（出度為0）
                if not graph[node]:
                    paths.append(curr_path)
                else:
                    for next_node in graph[node]:
                        dfs_paths(next_node, curr_path, visited.copy())
            
            # 從每個起點開始搜索
            for start_node in starting_nodes:
                dfs_paths(start_node, [], set())
            
            # 按路徑長度排序
            paths.sort(key=len, reverse=True)
            
            # 將層ID轉換為層類型，以便更好地理解路徑
            typed_paths = []
            id_to_type = {layer.get('id', layer.get('name', '')): layer.get('type', 'unknown') 
                          for layer in self.model_structure['layers']}
            
            for path in paths:
                typed_path = []
                for node in path:
                    typed_path.append({
                        'id': node,
                        'type': id_to_type.get(node, 'unknown')
                    })
                typed_paths.append(typed_path)
            
            self.results['sequential_paths'] = typed_paths
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """
        計算Gini係數，用於衡量參數分布的不均勻程度。
        
        Args:
            values (List[float]): 一組數值。
            
        Returns:
            float: Gini係數，範圍從0（完全平均）到1（完全不平均）。
        """
        if not values or len(values) <= 1 or sum(values) == 0:
            return 0.0
        
        # 排序值
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        # 計算Gini係數
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _calculate_parameter_efficiency(self):
        """
        計算模型的參數效率，即每個參數的效用。
        
        這裡僅執行一個簡單的估計。在實際應用中，可能需要結合模型性能指標。
        """
        # 獲取模型參數量和複雜度
        total_params = self.results.get('total_params', 0)
        model_complexity = self.results.get('model_complexity', {})
        total_flops = model_complexity.get('total_flops', 0)
        
        # 初始化效率字典
        efficiency = {
            'flops_per_param': 0.0,
            'params_utilization': 1.0,  # 默認值，實際應用中可能需要結合模型性能
            'relative_efficiency': 1.0
        }
        
        if total_params > 0 and total_flops > 0:
            # 計算每個參數的計算效率
            efficiency['flops_per_param'] = total_flops / total_params
            
            # 與典型模型架構進行比較
            architecture_type = self.results.get('architecture_type', 'unknown')
            if architecture_type == 'cnn':
                # CNN的參考值
                reference_flops_per_param = 100
                efficiency['relative_efficiency'] = efficiency['flops_per_param'] / reference_flops_per_param
            elif architecture_type == 'transformer':
                # Transformer的參考值
                reference_flops_per_param = 50
                efficiency['relative_efficiency'] = efficiency['flops_per_param'] / reference_flops_per_param
            
        # 無論如何都設置parameter_efficiency鍵
        self.results['parameter_efficiency'] = efficiency
    
    def _analyze_layer_connectivity(self):
        """
        分析層間連接性，計算圖的相關特性。
        """
        if 'layers' in self.model_structure:
            # 構建圖
            graph = defaultdict(list)
            
            # 計算每一層的入度和出度
            indegree = defaultdict(int)
            outdegree = defaultdict(int)
            
            # 建立層之間的連接關係
            for layer in self.model_structure['layers']:
                layer_id = layer.get('id', layer.get('name', ''))
                inputs = layer.get('inputs', [])
                
                for input_id in inputs:
                    graph[input_id].append(layer_id)
                    indegree[layer_id] += 1
                    outdegree[input_id] += 1
            
            # 計算連接度統計
            connectivity = {
                'avg_indegree': sum(indegree.values()) / len(indegree) if indegree else 0,
                'avg_outdegree': sum(outdegree.values()) / len(outdegree) if outdegree else 0,
                'max_indegree': max(indegree.values()) if indegree else 0,
                'max_outdegree': max(outdegree.values()) if outdegree else 0,
                'bottleneck_layers': [],
                'hub_layers': []
            }
            
            # 識別瓶頸層（高入度，低出度）
            for layer_id in indegree:
                if indegree[layer_id] > connectivity['avg_indegree'] * 1.5 and outdegree.get(layer_id, 0) <= 1:
                    connectivity['bottleneck_layers'].append({
                        'layer_id': layer_id,
                        'indegree': indegree[layer_id],
                        'outdegree': outdegree.get(layer_id, 0)
                    })
            
            # 識別樞紐層（高入度，高出度）
            for layer_id in set(indegree.keys()) | set(outdegree.keys()):
                if indegree.get(layer_id, 0) > 1 and outdegree.get(layer_id, 0) > 1:
                    connectivity['hub_layers'].append({
                        'layer_id': layer_id,
                        'indegree': indegree.get(layer_id, 0),
                        'outdegree': outdegree.get(layer_id, 0)
                    })
            
            # 按連接度排序
            connectivity['bottleneck_layers'].sort(key=lambda x: x['indegree'], reverse=True)
            connectivity['hub_layers'].sort(key=lambda x: x['indegree'] + x['outdegree'], reverse=True)
            
            self.results['layer_connectivity'] = connectivity
    
    def _analyze_computational_complexity(self):
        """
        分析計算複雜度，估計模型的理論計算量。
        """
        # 檢查模型複雜度是否已經計算
        if 'model_complexity' not in self.results:
            return
        
        # 獲取總FLOPs
        total_flops = self.results['model_complexity'].get('total_flops', 0)
        
        if total_flops > 0:
            # 將FLOPs轉換為GFLOPs
            gflops = total_flops / 1e9
            
            # 估計在不同硬件上的推理時間
            # 注意：這些是粗略估計，實際時間取決於很多因素
            hardware_estimates = {
                'cpu_inference_ms': gflops * 10,  # 假設1 GFLOPs在CPU上需要10ms
                'gpu_inference_ms': gflops * 0.5,  # 假設1 GFLOPs在GPU上需要0.5ms
                'mobile_inference_ms': gflops * 50  # 假設1 GFLOPs在移動設備上需要50ms
            }
            
            # 對模型計算複雜度進行分類
            complexity_class = 'unknown'
            if gflops < 1:
                complexity_class = 'very light'
            elif gflops < 5:
                complexity_class = 'light'
            elif gflops < 20:
                complexity_class = 'moderate'
            elif gflops < 100:
                complexity_class = 'heavy'
            else:
                complexity_class = 'very heavy'
            
            # 更新模型複雜度信息
            self.results['model_complexity'].update({
                'gflops': gflops,
                'hardware_estimates': hardware_estimates,
                'complexity_class': complexity_class
            })
    
    def get_layer_hierarchy(self) -> Dict:
        """
        獲取層級結構信息。
        
        Returns:
            Dict: 包含層級結構的字典。
        """
        if not self.results or 'layer_count' not in self.results:
            self.analyze(save_results=False)
        
        if 'layers' in self.model_structure:
            # 構建層級結構樹
            nodes = {}
            for layer in self.model_structure['layers']:
                layer_id = layer.get('id', layer.get('name', ''))
                nodes[layer_id] = {
                    'id': layer_id,
                    'type': layer.get('type', 'unknown'),
                    'name': layer.get('name', ''),
                    'parameters': layer.get('parameters', 0),
                    'input_size': layer.get('input_size', []),
                    'output_size': layer.get('output_size', []),
                    'children': []
                }
            
            # 添加子節點
            for layer in self.model_structure['layers']:
                layer_id = layer.get('id', layer.get('name', ''))
                inputs = layer.get('inputs', [])
                
                for input_id in inputs:
                    if input_id in nodes:
                        nodes[input_id]['children'].append(layer_id)
            
            # 找到根節點
            roots = []
            for layer in self.model_structure['layers']:
                layer_id = layer.get('id', layer.get('name', ''))
                if not layer.get('inputs', []):
                    roots.append(layer_id)
        
            return {
                'nodes': nodes,
                'roots': roots
            }
        
        return {'nodes': {}, 'roots': []}
    
    def get_parameter_distribution(self) -> Dict:
        """
        獲取參數分布信息。
        
        Returns:
            Dict: 包含參數分布的字典。
        """
        if not self.results or 'parameter_distribution' not in self.results:
            self.analyze(save_results=False)
        
        return self.results.get('parameter_distribution', {})
    
    def get_model_summary(self) -> Dict:
        """
        獲取模型摘要信息。
        
        Returns:
            Dict: 包含模型摘要的字典。
        """
        if not self.results or 'model_name' not in self.results:
            self.analyze(save_results=False)
        
        # 提取關鍵信息
        summary = {
            'model_name': self.results.get('model_name', 'Unknown'),
            'architecture_type': self.results.get('architecture_type', 'unknown'),
            'total_params': self.results.get('total_params', 0),
            'trainable_params': self.results.get('trainable_params', 0),
            'non_trainable_params': self.results.get('non_trainable_params', 0),
            'layer_count': self.results.get('layer_count', 0),
            'model_depth': self.results.get('model_depth', 0),
            'largest_layers': self.results.get('largest_layers', [])[:3],  # 僅包含前3個最大的層
            'sparse_layers_count': len(self.results.get('sparse_layers', [])),
            'computational_complexity': self.results.get('model_complexity', {}).get('gflops', 0)
        }
        
        return summary 