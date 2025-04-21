"""
模型結構分析模組。

此模組提供用於分析深度學習模型結構的功能。它可以解析模型架構、參數分布和模型特性。

Classes:
    ModelStructureAnalyzer: 分析模型結構的類別。
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base_analyzer import BaseAnalyzer

class ModelStructureAnalyzer(BaseAnalyzer):
    """
    模型結構分析器，用於分析模型結構和參數分布。
    
    此分析器主要關注模型架構、層級結構、參數分布和其他模型特性。
    
    Attributes:
        experiment_dir (str): 實驗結果的目錄路徑。
        model_structure (Dict): 存儲解析後的模型結構。
        model_stats (Dict): 存儲模型統計信息。
    """
    
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
            'total_params': 0,
            'trainable_params': 0,
            'non_trainable_params': 0,
            'layer_count': 0,
            'layer_types': {},
            'parameter_distribution': {},
            'model_depth': 0,
            'sparsity': 0.0,
            'parameter_statistics': {},
            'layerwise_parameters': [],
            'computational_complexity': {},
            'memory_footprint': {},
            'receptive_field': {}
        }
        
        try:
            if self.model_structure:
                # 提取模型名稱
                self.results['model_name'] = self.model_structure.get('model_name', 'Unknown')
                
                # 提取總參數量
                self.results['total_params'] = self.model_structure.get('total_parameters', 0)
                
                # 分析層級結構
                if 'layer_info' in self.model_structure:
                    layer_info = self.model_structure['layer_info']
                    self.results['layer_count'] = len(layer_info)
                    
                    # 統計層類型分布
                    layer_types = {}
                    parameter_by_layer = []
                    trainable_params = 0
                    
                    for layer in layer_info:
                        layer_type = layer.get('type', 'Unknown')
                        parameters = layer.get('parameters', 0)
                        trainable = layer.get('trainable', True)
                        
                        # 累計層類型
                        if layer_type in layer_types:
                            layer_types[layer_type]['count'] += 1
                            layer_types[layer_type]['parameters'] += parameters
                        else:
                            layer_types[layer_type] = {
                                'count': 1,
                                'parameters': parameters
                            }
                        
                        # 統計可訓練參數
                        if trainable:
                            trainable_params += parameters
                        
                        # 收集每層的參數信息
                        parameter_by_layer.append({
                            'name': layer.get('name', 'Unknown'),
                            'type': layer_type,
                            'parameters': parameters,
                            'trainable': trainable,
                            'shape': layer.get('shape', [])
                        })
                    
                    self.results['layer_types'] = layer_types
                    self.results['trainable_params'] = trainable_params
                    self.results['non_trainable_params'] = self.results['total_params'] - trainable_params
                    self.results['layerwise_parameters'] = parameter_by_layer
                    
                    # 計算模型深度 (考慮主幹層)
                    model_depth = self._calculate_model_depth(layer_info)
                    self.results['model_depth'] = model_depth
                    
                    # 計算模型稀疏度 (如果有權重稀疏信息)
                    if 'weight_sparsity' in self.model_structure:
                        self.results['sparsity'] = self.model_structure['weight_sparsity']
                    else:
                        # 使用增強的稀疏度估計
                        self.results['sparsity'] = self._estimate_sparsity(layer_info)
                    
                    # 計算參數分布統計量
                    param_stats = self._calculate_parameter_statistics(parameter_by_layer)
                    self.results['parameter_statistics'] = param_stats
                    
                    # 計算參數分布
                    param_distribution = self._calculate_parameter_distribution(parameter_by_layer)
                    self.results['parameter_distribution'] = param_distribution
                    
                    # 計算計算複雜度 (FLOPs)
                    computational_complexity = self.calculate_flops(layer_info)
                    self.results['computational_complexity'] = computational_complexity
                    
                    # 計算記憶體佔用
                    memory_footprint = self.calculate_memory_footprint(layer_info)
                    self.results['memory_footprint'] = memory_footprint
                    
                    # 計算理論感受野 (對CNN模型有意義)
                    has_conv_layers = any('conv' in layer_type.lower() for layer_type in layer_types.keys())
                    if has_conv_layers:
                        receptive_field = self.calculate_receptive_field(layer_info)
                        self.results['receptive_field'] = receptive_field
            
            if save_results:
                self.save_results(os.path.join(self.experiment_dir, 'analysis'))
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"分析模型結構時出錯: {e}")
            raise
    
    def get_layer_hierarchy(self) -> Dict:
        """
        提取模型的層級結構。
        
        Returns:
            Dict: 描述模型層級結構的字典。
        """
        if not self.model_structure:
            self.load_model_structure()
        
        # 初始化層級結構
        hierarchy = {
            'root': {
                'name': self.results.get('model_name', 'Model'),
                'children': []
            }
        }
        
        # 如果有層信息，則建立層級結構
        if 'layer_info' in self.model_structure:
            layer_info = self.model_structure['layer_info']
            
            # 依照層名稱建立層級關係
            for layer in layer_info:
                layer_name = layer.get('name', 'Unknown')
                layer_parts = layer_name.split('.')
                
                # 建立層路徑
                current_path = 'root'
                current_node = hierarchy['root']
                
                for i, part in enumerate(layer_parts):
                    path = f"{current_path}.{part}" if current_path != 'root' else part
                    
                    # 查找現有節點或創建新節點
                    child_found = False
                    for child in current_node.get('children', []):
                        if child['name'] == part:
                            child_found = True
                            current_node = child
                            current_path = path
                            break
                    
                    if not child_found:
                        # 創建新節點
                        new_node = {
                            'name': part,
                            'full_name': layer_name,
                            'type': layer.get('type', 'Unknown') if i == len(layer_parts) - 1 else 'Container',
                            'children': [],
                            'parameters': layer.get('parameters', 0) if i == len(layer_parts) - 1 else 0,
                            'shape': layer.get('shape', []) if i == len(layer_parts) - 1 else []
                        }
                        
                        if 'children' not in current_node:
                            current_node['children'] = []
                            
                        current_node['children'].append(new_node)
                        current_node = new_node
                        current_path = path
        
        return hierarchy
    
    def get_parameter_distribution(self) -> Dict:
        """
        分析模型參數的分布特性。
        
        Returns:
            Dict: 包含參數分布統計信息的字典。
        """
        if not self.results or 'layerwise_parameters' not in self.results:
            self.analyze()
        
        return self.results.get('parameter_distribution', {})
    
    def _calculate_model_depth(self, layer_info: List[Dict]) -> int:
        """
        計算模型的深度 (主幹層數)。
        
        Args:
            layer_info (List[Dict]): 層級信息列表。
            
        Returns:
            int: 模型深度估計值。
        """
        # 簡單的方法：找出最深的嵌套層
        max_depth = 0
        
        for layer in layer_info:
            name = layer.get('name', '')
            depth = name.count('.') + 1  # 計算層級深度
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _estimate_sparsity(self, layer_info: List[Dict]) -> float:
        """
        估計模型的稀疏度 (如果沒有直接提供)。
        
        稀疏度定義為模型中為零的參數比例。對於不同的層類型，會使用不同的估計方法。
        
        Args:
            layer_info (List[Dict]): 層級信息列表。
            
        Returns:
            float: 稀疏度估計值 (0-1範圍)。
        """
        total_params = 0
        estimated_zeros = 0
        
        for layer in layer_info:
            layer_type = layer.get('type', 'Unknown')
            parameters = layer.get('parameters', 0)
            
            # 跳過沒有參數的層
            if parameters <= 0:
                continue
                
            total_params += parameters
            
            # 根據層類型估計稀疏度
            if 'batch_norm' in layer_type.lower() or 'batchnorm' in layer_type.lower():
                # 批次正規化層通常有較低的稀疏度
                estimated_zeros += parameters * 0.05
            elif 'conv' in layer_type.lower():
                # 卷積層通常在訓練後會有中等稀疏度
                estimated_zeros += parameters * 0.2
            elif 'linear' in layer_type.lower() or 'dense' in layer_type.lower():
                # 全連接層通常具有較高的稀疏度
                estimated_zeros += parameters * 0.3
            elif 'lstm' in layer_type.lower() or 'gru' in layer_type.lower() or 'rnn' in layer_type.lower():
                # 循環層通常具有較低的稀疏度
                estimated_zeros += parameters * 0.1
            elif 'embedding' in layer_type.lower():
                # 嵌入層通常有一定的稀疏度
                estimated_zeros += parameters * 0.15
            else:
                # 默認估計值
                estimated_zeros += parameters * 0.1
        
        # 計算整體稀疏度
        if total_params > 0:
            return estimated_zeros / total_params
        else:
            return 0.0
    
    def _calculate_parameter_statistics(self, layer_params: List[Dict]) -> Dict:
        """
        計算參數的統計特性。
        
        Args:
            layer_params (List[Dict]): 每層的參數信息列表。
            
        Returns:
            Dict: 包含參數統計信息的字典。
        """
        if not layer_params:
            return {}
        
        # 提取參數數量
        param_counts = [layer['parameters'] for layer in layer_params if layer['parameters'] > 0]
        
        if not param_counts:
            return {
                'min': 0,
                'max': 0,
                'mean': 0,
                'median': 0,
                'std': 0
            }
        
        # 計算統計量
        return {
            'min': float(np.min(param_counts)),
            'max': float(np.max(param_counts)),
            'mean': float(np.mean(param_counts)),
            'median': float(np.median(param_counts)),
            'std': float(np.std(param_counts))
        }
    
    def _calculate_parameter_distribution(self, layer_params: List[Dict]) -> Dict:
        """
        計算參數分布。
        
        Args:
            layer_params (List[Dict]): 每層的參數信息列表。
            
        Returns:
            Dict: 包含參數分布信息的字典。
        """
        if not layer_params:
            return {}
        
        # 按層類型分布
        by_layer_type = {}
        
        for layer in layer_params:
            layer_type = layer['type']
            param_count = layer['parameters']
            
            if layer_type not in by_layer_type:
                by_layer_type[layer_type] = 0
            
            by_layer_type[layer_type] += param_count
        
        # 計算百分比
        total_params = sum(by_layer_type.values())
        percentage_by_type = {
            layer_type: count / total_params * 100 if total_params > 0 else 0
            for layer_type, count in by_layer_type.items()
        }
        
        # 按參數規模分布 (對數分bin)
        param_counts = np.array([layer['parameters'] for layer in layer_params if layer['parameters'] > 0])
        
        if len(param_counts) == 0:
            return {
                'by_layer_type': by_layer_type,
                'percentage_by_type': percentage_by_type,
                'by_parameter_scale': {}
            }
        
        # 使用對數刻度計算直方圖
        if np.min(param_counts) > 0:
            log_counts = np.log10(param_counts)
            hist, bin_edges = np.histogram(log_counts, bins=10)
            
            # 轉換回原始刻度
            bin_edges = 10 ** bin_edges
            
            by_parameter_scale = {
                f"{int(bin_edges[i])}-{int(bin_edges[i+1])}": int(hist[i])
                for i in range(len(hist))
            }
        else:
            by_parameter_scale = {}
        
        return {
            'by_layer_type': by_layer_type,
            'percentage_by_type': percentage_by_type,
            'by_parameter_scale': by_parameter_scale
        }
    
    def visualize_model_structure(self, output_path: Optional[str] = None) -> plt.Figure:
        """
        將模型結構視覺化為層次結構圖。
        
        使用networkx庫將模型結構可視化為有向圖，顯示層之間的連接關係。
        
        Args:
            output_path (str, optional): 輸出圖像的路徑。如果為None，則僅返回圖像對象。
            
        Returns:
            plt.Figure: Matplotlib圖像對象。
        """
        # 確保已解析模型結構
        if not self.results:
            self.analyze()
        
        # 獲取層級結構
        hierarchy = self.get_layer_hierarchy()
        
        try:
            # 檢查是否安裝了networkx
            import networkx as nx
            has_networkx = True
        except ImportError:
            has_networkx = False
            self.logger.warning("未安裝networkx庫，將使用簡單的圖表代替。若要使用完整的視覺化功能，請安裝networkx: pip install networkx")
            
        if has_networkx:
            # 創建有向圖
            G = nx.DiGraph()
            
            # 建立節點和邊的函數
            def add_nodes_edges(node, parent=None):
                node_id = node.get('full_name', node.get('name', 'Unknown'))
                
                # 添加節點，帶有屬性
                G.add_node(
                    node_id, 
                    name=node.get('name', 'Unknown'),
                    type=node.get('type', 'Unknown'),
                    params=node.get('parameters', 0),
                    shape=str(node.get('shape', []))
                )
                
                # 如果有父節點，添加一條邊
                if parent:
                    G.add_edge(parent, node_id)
                
                # 遞歸處理子節點
                for child in node.get('children', []):
                    add_nodes_edges(child, node_id)
            
            # 從根節點開始處理
            root = hierarchy.get('root', {})
            root_id = root.get('name', 'Model')
            G.add_node(root_id, name=root_id, type='Model', params=self.results.get('total_params', 0))
            
            for child in root.get('children', []):
                add_nodes_edges(child, root_id)
            
            # 創建圖形
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # 如果節點過多，調整圖形大小
            if len(G.nodes) > 50:
                fig, ax = plt.subplots(figsize=(20, 15))
            
            # 設置布局
            try:
                # 嘗試使用graphviz布局
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            except:
                # 如果graphviz不可用，使用spring布局
                pos = nx.spring_layout(G, seed=42)
            
            # 設置節點大小基於參數數量（使用對數刻度防止極端差異）
            node_sizes = []
            for n in G.nodes:
                params = G.nodes[n].get('params', 1)
                size = np.log1p(params) * 100 if params > 0 else 50
                node_sizes.append(size)
            
            # 設置節點顏色基於層類型
            layer_types = list(set([G.nodes[n].get('type', 'Unknown') for n in G.nodes]))
            color_map = plt.cm.get_cmap('tab20', len(layer_types))
            type_to_color = {t: color_map(i) for i, t in enumerate(layer_types)}
            node_colors = [type_to_color.get(G.nodes[n].get('type', 'Unknown'), 'gray') for n in G.nodes]
            
            # 繪製節點和邊
            nx.draw(
                G, pos, 
                with_labels=True, 
                node_size=node_sizes,
                node_color=node_colors,
                font_size=8,
                alpha=0.8,
                ax=ax
            )
            
            # 添加圖例
            patches = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10) 
                      for color in type_to_color.values()]
            ax.legend(patches, type_to_color.keys(), loc='upper right')
            
            # 添加標題和基本信息
            ax.set_title(f"Model Structure: {self.results.get('model_name', 'Unknown')}")
            
            # 添加信息文本框
            info_text = [
                f"Total Parameters: {self.results.get('total_params', 0):,}",
                f"Trainable Parameters: {self.results.get('trainable_params', 0):,}",
                f"Layer Count: {self.results.get('layer_count', 0)}",
                f"Model Depth: {self.results.get('model_depth', 0)}",
                f"Estimated FLOPs: {self.results.get('computational_complexity', {}).get('total_flops', 0):,}",
                f"Memory Footprint: {self.results.get('memory_footprint', {}).get('total_memory', 0) / (1024**2):.2f} MB"
            ]
            
            # 如果有感受野信息，添加到信息文本中
            if self.results.get('receptive_field') and self.results['receptive_field'].get('max_receptive_field'):
                info_text.append(f"Max Receptive Field: {self.results['receptive_field']['max_receptive_field']}")
                
            plt.figtext(0.02, 0.02, "\n".join(info_text), ha='left', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
            # 如果有感受野信息，創建第二個圖顯示感受野分布
            if self.results.get('receptive_field') and self.results['receptive_field'].get('sorted_by_size'):
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                
                # 提取層名稱和對應的感受野大小
                sorted_rf = self.results['receptive_field']['sorted_by_size']
                layer_names = list(sorted_rf.keys())[-15:]  # 只顯示最大的15個
                rf_sizes = [sorted_rf[name]['receptive_field_size'] for name in layer_names]
                
                # 繪製條形圖
                bars = ax2.barh(range(len(layer_names)), rf_sizes, align='center')
                ax2.set_yticks(range(len(layer_names)))
                ax2.set_yticklabels([name.split('.')[-1] if '.' in name else name for name in layer_names])
                
                # 在條形上添加數值標籤
                for i, bar in enumerate(bars):
                    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                             f"{rf_sizes[i]}", ha='left', va='center')
                
                ax2.set_title("Receptive Field Sizes (Top 15 Layers)")
                ax2.set_xlabel("Receptive Field Size")
                plt.tight_layout()
                
                # 保存第二個圖
                if output_path:
                    rf_path = os.path.splitext(output_path)[0] + "_receptive_field.png"
                    fig2.savefig(rf_path, dpi=300, bbox_inches='tight')
            
        else:
            # 如果沒有networkx，則使用簡單的文本顯示
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, "Model Structure Visualization\n(requires networkx package)", 
                    ha='center', va='center', fontsize=12)
            
            info_text = [
                f"Total Parameters: {self.results.get('total_params', 0):,}",
                f"Trainable Parameters: {self.results.get('trainable_params', 0):,}",
                f"Layer Count: {self.results.get('layer_count', 0)}",
                f"Model Depth: {self.results.get('model_depth', 0)}",
                f"Estimated FLOPs: {self.results.get('computational_complexity', {}).get('total_flops', 0):,}",
                f"Memory Footprint: {self.results.get('memory_footprint', {}).get('total_memory', 0) / (1024**2):.2f} MB"
            ]
            
            # 如果有感受野信息，添加到信息文本中
            if self.results.get('receptive_field') and self.results['receptive_field'].get('max_receptive_field'):
                info_text.append(f"Max Receptive Field: {self.results['receptive_field']['max_receptive_field']}")
            
            ax.text(0.5, 0.4, "\n".join(info_text), 
                    ha='center', va='center', fontsize=10)
        
        # 保存圖像
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig

    def calculate_flops(self, layer_info: Optional[List[Dict]] = None) -> Dict:
        """
        計算模型每一層的浮點運算數 (FLOPs)。
        
        對於卷積層：
        FLOPs = 2 * H * W * Cin * Cout * Kh * Kw / (stride_h * stride_w)
        
        對於全連接層：
        FLOPs = 2 * Nin * Nout
        
        Args:
            layer_info (List[Dict], optional): 層級信息列表。如果為None，則使用已載入的模型結構中的層級信息。
            
        Returns:
            Dict: 各層的FLOPs及總FLOPs。
        """
        if layer_info is None:
            if not self.model_structure or 'layer_info' not in self.model_structure:
                self.load_model_structure()
            layer_info = self.model_structure.get('layer_info', [])
            
        flops_by_layer = {}
        total_flops = 0
        
        for layer in layer_info:
            layer_type = layer.get('type', 'Unknown')
            layer_name = layer.get('name', 'Unknown')
            shape = layer.get('shape', [])
            
            flops = 0
            
            if layer_type == 'Conv2d' and len(shape) >= 4:
                # 對於卷積層，我們需要輸入尺寸、輸出通道數、kernel尺寸和stride
                out_channels = shape[0]
                kernel_h = shape[2] if len(shape) > 2 else 1
                kernel_w = shape[3] if len(shape) > 3 else 1
                
                # 這些信息可能存在於額外屬性中或需要從配置中估計
                in_h = layer.get('input_height', 32)  # 默認值，實際應從數據中獲取
                in_w = layer.get('input_width', 32)   # 默認值，實際應從數據中獲取
                in_channels = layer.get('input_channels', shape[1] if len(shape) > 1 else 3)
                stride_h = layer.get('stride_h', 1)
                stride_w = layer.get('stride_w', 1)
                
                # 計算輸出維度
                out_h = in_h // stride_h
                out_w = in_w // stride_w
                
                # 計算FLOPs: 2 * H * W * Cin * Cout * Kh * Kw / (stride_h * stride_w)
                # 2倍是因為乘法和加法操作
                flops = 2 * out_h * out_w * in_channels * out_channels * kernel_h * kernel_w
                
            elif ('Linear' in layer_type or 'Dense' in layer_type) and len(shape) >= 2:
                # 對於全連接層，FLOPs = 2 * Nin * Nout
                in_features = shape[1]
                out_features = shape[0]
                flops = 2 * in_features * out_features
                
            elif 'BatchNorm' in layer_type and len(shape) >= 1:
                # 對於批次正規化，每個元素大約需要4個操作
                num_features = shape[0]
                batch_size = 32  # 估計值
                feature_map_size = layer.get('feature_map_size', 1)
                flops = 4 * batch_size * num_features * feature_map_size
                
            elif 'LSTM' in layer_type or 'GRU' in layer_type:
                # 循環層的FLOPs估計比較複雜，這裡提供簡化估計
                input_size = shape[1] if len(shape) > 1 else 128
                hidden_size = shape[0] if len(shape) > 0 else 128
                sequence_length = layer.get('sequence_length', 10)  # 默認值
                
                if 'LSTM' in layer_type:
                    # LSTM有4個門，每個門的計算約為2 * (input_size + hidden_size) * hidden_size
                    gate_flops = 2 * (input_size + hidden_size) * hidden_size
                    flops = 4 * gate_flops * sequence_length
                else:  # GRU
                    # GRU有3個門
                    gate_flops = 2 * (input_size + hidden_size) * hidden_size
                    flops = 3 * gate_flops * sequence_length
            
            flops_by_layer[layer_name] = flops
            total_flops += flops
        
        return {
            'total_flops': total_flops,
            'flops_by_layer': flops_by_layer
        }
    
    def calculate_memory_footprint(self, layer_info: Optional[List[Dict]] = None) -> Dict:
        """
        估計模型的記憶體佔用情況。
        
        計算模型前向傳播時的啟用值(activation)記憶體，以及參數和梯度的記憶體。
        
        Args:
            layer_info (List[Dict], optional): 層級信息列表。如果為None，則使用已載入的模型結構中的層級信息。
            
        Returns:
            Dict: 記憶體佔用統計。
        """
        if layer_info is None:
            if not self.model_structure or 'layer_info' not in self.model_structure:
                self.load_model_structure()
            layer_info = self.model_structure.get('layer_info', [])
            
        total_parameters_memory = 0
        total_activations_memory = 0
        memory_by_layer = {}
        
        # 假設使用float32，每個參數佔4字節
        bytes_per_param = 4
        batch_size = 32  # 預設批次大小
        
        for layer in layer_info:
            layer_name = layer.get('name', 'Unknown')
            parameters = layer.get('parameters', 0)
            layer_type = layer.get('type', 'Unknown')
            shape = layer.get('shape', [])
            
            # 參數記憶體
            param_memory = parameters * bytes_per_param
            total_parameters_memory += param_memory
            
            # 估計啟用值記憶體
            activation_memory = 0
            
            # 根據層類型估計啟用值大小
            if 'Conv2d' in layer_type and len(shape) >= 4:
                out_channels = shape[0]
                # 輸出特徵圖大小，這需要更多信息才能精確計算
                out_h = layer.get('output_height', 32)  # 默認值
                out_w = layer.get('output_width', 32)   # 默認值
                activation_memory = batch_size * out_channels * out_h * out_w * bytes_per_param
                
            elif ('Linear' in layer_type or 'Dense' in layer_type) and len(shape) >= 2:
                out_features = shape[0]
                activation_memory = batch_size * out_features * bytes_per_param
                
            elif 'BatchNorm' in layer_type and len(shape) >= 1:
                num_features = shape[0]
                feature_map_size = layer.get('feature_map_size', 1024)  # 估計值
                activation_memory = batch_size * num_features * feature_map_size * bytes_per_param
                
            elif 'LSTM' in layer_type or 'GRU' in layer_type:
                hidden_size = shape[0] if len(shape) > 0 else 128
                sequence_length = layer.get('sequence_length', 10)  # 默認值
                activation_memory = batch_size * hidden_size * sequence_length * bytes_per_param
            
            total_activations_memory += activation_memory
            
            memory_by_layer[layer_name] = {
                'parameters_memory': param_memory,
                'activations_memory': activation_memory,
                'total_memory': param_memory + activation_memory
            }
        
        # 總梯度記憶體 (訓練時會是參數記憶體的兩倍)
        total_gradients_memory = total_parameters_memory
        
        # 總記憶體佔用 (包括參數、梯度和啟用值)
        total_memory = total_parameters_memory + total_gradients_memory + total_activations_memory
        
        return {
            'total_memory': total_memory,
            'parameters_memory': total_parameters_memory,
            'gradients_memory': total_gradients_memory,
            'activations_memory': total_activations_memory,
            'memory_by_layer': memory_by_layer,
            'memory_unit': 'bytes'
        }

    def calculate_receptive_field(self, layer_info: Optional[List[Dict]] = None) -> Dict:
        """
        計算CNN模型每一層的理論感受野。
        
        感受野是輸出特徵圖的一個點對應輸入圖像的區域大小。
        對於深度神經網絡，特別是CNN，感受野的大小反映了模型捕捉上下文信息的能力。
        
        Args:
            layer_info (List[Dict], optional): 層級信息列表。如果為None，則使用已載入的模型結構中的層級信息。
            
        Returns:
            Dict: 各層的理論感受野。
        """
        if layer_info is None:
            if not self.model_structure or 'layer_info' not in self.model_structure:
                self.load_model_structure()
            layer_info = self.model_structure.get('layer_info', [])
            
        # 創建層級結構的拓撲排序
        layer_names = [layer.get('name', f'layer_{i}') for i, layer in enumerate(layer_info)]
        parent_map = {}  # 用於存儲層級關係
        
        # 建立層級關係
        for name in layer_names:
            parts = name.split('.')
            if len(parts) > 1:
                parent = '.'.join(parts[:-1])
                if parent in layer_names:
                    parent_map[name] = parent
        
        # 初始化每一層的感受野信息
        receptive_fields = {}
        for layer in layer_info:
            layer_name = layer.get('name', 'Unknown')
            layer_type = layer.get('type', 'Unknown')
            
            # 默認初始值
            receptive_fields[layer_name] = {
                'receptive_field_size': 1,  # 初始感受野為1x1
                'stride': 1,                # 初始stride為1
                'padding': 0,               # 初始padding為0
                'kernel_size': 1,           # 初始kernel_size為1
                'dilation': 1               # 初始dilation為1
            }
            
            # 根據層類型設置參數
            if 'Conv2d' in layer_type:
                shape = layer.get('shape', [])
                if len(shape) >= 4:
                    kernel_size = max(shape[2], shape[3]) if len(shape) > 3 else 1
                else:
                    kernel_size = layer.get('kernel_size', 1)
                    
                stride = layer.get('stride', 1)
                padding = layer.get('padding', 0)
                dilation = layer.get('dilation', 1)
                
                receptive_fields[layer_name].update({
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding,
                    'dilation': dilation
                })
                
            elif 'Pool' in layer_type:
                kernel_size = layer.get('kernel_size', 2)
                stride = layer.get('stride', layer.get('kernel_size', 2))
                padding = layer.get('padding', 0)
                
                receptive_fields[layer_name].update({
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding
                })
        
        # 計算每一層的感受野
        for layer_name in layer_names:
            if layer_name in parent_map:
                parent_name = parent_map[layer_name]
                parent_rf = receptive_fields[parent_name]
                current_rf = receptive_fields[layer_name]
                
                # 計算當前層的感受野
                # 公式: rf_size = prev_rf_size + ((kernel_size - 1) * dilation)
                rf_size = parent_rf['receptive_field_size'] + \
                          ((current_rf['kernel_size'] - 1) * current_rf['dilation'])
                          
                # 更新stride (累積效應)
                stride = parent_rf['stride'] * current_rf['stride']
                
                # 更新當前層的感受野信息
                receptive_fields[layer_name]['receptive_field_size'] = rf_size
                receptive_fields[layer_name]['stride'] = stride
        
        # 按感受野大小排序
        sorted_rf = {k: v for k, v in sorted(
            receptive_fields.items(), 
            key=lambda item: item[1]['receptive_field_size']
        )}
        
        return {
            'receptive_fields': receptive_fields,
            'sorted_by_size': sorted_rf,
            'max_receptive_field': max([rf['receptive_field_size'] for rf in receptive_fields.values()])
        }