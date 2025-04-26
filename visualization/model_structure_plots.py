"""
模型結構視覺化模組。

此模組提供用於可視化深度學習模型結構的功能，包括模型架構圖、參數分佈圖等。

Classes:
    ModelStructurePlotter: 繪製模型結構圖的類別。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

from .plotter import BasePlotter

class ModelStructurePlotter(BasePlotter):
    """
    模型結構視覺化工具，用於繪製模型架構圖、參數分布等圖表。
    
    Attributes:
        default_figsize (Tuple[int, int]): 默認圖表大小。
        default_dpi (int): 默認解析度。
        layer_colors (Dict[str, str]): 不同層類型的顏色映射。
    """
    
    def __init__(self, default_figsize: Tuple[int, int] = (12, 8), default_dpi: int = 100):
        """
        初始化模型結構繪圖工具。
        
        Args:
            default_figsize (Tuple[int, int], optional): 默認圖表大小。默認為(12, 8)。
            default_dpi (int, optional): 默認解析度。默認為100。
        """
        super().__init__(default_figsize, default_dpi)
        
        # 設定不同層類型的顏色
        self.layer_colors = {
            'conv': '#3498db',  # 藍色
            'linear': '#e74c3c',  # 紅色
            'pool': '#2ecc71',  # 綠色
            'norm': '#9b59b6',  # 紫色
            'dropout': '#95a5a6',  # 灰色
            'activation': '#f39c12',  # 橙色
            'recurrent': '#1abc9c',  # 青色
            'attention': '#d35400',  # 深橙色
            'embedding': '#27ae60',  # 深綠色
            'unknown': '#7f8c8d'  # 深灰色
        }
    
    def _get_layer_color(self, layer_type: str) -> str:
        """
        根據層類型獲取對應的顏色。
        
        Args:
            layer_type (str): 層類型。
            
        Returns:
            str: 顏色代碼。
        """
        layer_type = layer_type.lower()
        
        # 根據包含的關鍵字判斷層類型
        if 'conv' in layer_type:
            return self.layer_colors['conv']
        elif any(t in layer_type for t in ['linear', 'fc', 'dense']):
            return self.layer_colors['linear']
        elif any(t in layer_type for t in ['pool', 'maxpool', 'avgpool']):
            return self.layer_colors['pool']
        elif any(t in layer_type for t in ['norm', 'bn', 'batchnorm', 'layernorm']):
            return self.layer_colors['norm']
        elif 'dropout' in layer_type:
            return self.layer_colors['dropout']
        elif any(t in layer_type for t in ['relu', 'sigmoid', 'tanh', 'gelu', 'activation']):
            return self.layer_colors['activation']
        elif any(t in layer_type for t in ['lstm', 'gru', 'rnn']):
            return self.layer_colors['recurrent']
        elif any(t in layer_type for t in ['attention', 'mha']):
            return self.layer_colors['attention']
        elif 'embed' in layer_type:
            return self.layer_colors['embedding']
        else:
            return self.layer_colors['unknown']
    
    def plot_model_structure(self, hierarchy: Dict, figsize: Optional[Tuple[int, int]] = None, 
                            show_params: bool = True, highlight_large_layers: bool = True,
                            title: Optional[str] = None) -> Figure:
        """
        繪製模型架構圖。
        
        Args:
            hierarchy (Dict): 模型層次結構，包含layers和connections。
            figsize (Tuple[int, int], optional): 圖表大小。默認為None，使用默認大小。
            show_params (bool, optional): 是否顯示參數數量。默認為True。
            highlight_large_layers (bool, optional): 是否突出參數量大的層。默認為True。
            title (str, optional): 圖表標題。默認為None。
            
        Returns:
            Figure: Matplotlib圖表對象。
        """
        if not hierarchy or 'layers' not in hierarchy or 'connections' not in hierarchy:
            raise ValueError("無效的層次結構數據")
        
        # 創建有向圖
        G = nx.DiGraph()
        
        # 添加節點
        for layer in hierarchy['layers']:
            layer_id = layer['id']
            layer_type = layer['type']
            params = layer.get('parameters', 0)
            
            # 節點大小與參數數量相關
            if highlight_large_layers and params > 0:
                size = 300 + 700 * (np.log1p(params) / np.log1p(np.max([l.get('parameters', 1) for l in hierarchy['layers']])))
            else:
                size = 500
            
            # 添加節點，包含層信息
            G.add_node(layer_id, 
                      type=layer_type, 
                      params=params, 
                      color=self._get_layer_color(layer_type),
                      size=size)
        
        # 添加連接
        for conn in hierarchy['connections']:
            G.add_edge(conn['from'], conn['to'])
        
        # 創建圖表
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
        
        # 嘗試使用階層佈局，如果不成功則回退到spring佈局
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            try:
                pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
            except:
                pos = nx.spring_layout(G)
        
        # 提取節點屬性
        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
        
        # 繪製連接
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5, arrowsize=15)
        
        # 繪製節點
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors, 
                              node_size=node_sizes, 
                              alpha=0.8)
        
        # 添加標籤
        labels = {}
        for node in G.nodes():
            node_type = G.nodes[node]['type']
            if show_params:
                params = G.nodes[node]['params']
                if params >= 1e6:
                    param_str = f"{params/1e6:.1f}M"
                elif params >= 1e3:
                    param_str = f"{params/1e3:.1f}K"
                else:
                    param_str = f"{params}"
                
                labels[node] = f"{node}\n({node_type}, {param_str})"
            else:
                labels[node] = f"{node}\n({node_type})"
                
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')
        
        # 添加圖例
        legend_elements = []
        unique_types = set()
        for layer in hierarchy['layers']:
            unique_types.add(layer['type'])
            
        for layer_type in sorted(unique_types):
            color = self._get_layer_color(layer_type)
            legend_elements.append(mpatches.Patch(color=color, label=layer_type))
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 添加標題
        if title:
            plt.title(title)
        else:
            plt.title('Model Architecture')
        
        plt.tight_layout()
        plt.axis('off')
        
        return fig
    
    def plot_parameter_distribution(self, param_distribution: Dict, figsize: Optional[Tuple[int, int]] = None,
                                   title: Optional[str] = None) -> Figure:
        """
        繪製模型參數分布圖。
        
        Args:
            param_distribution (Dict): 參數分布數據。
            figsize (Tuple[int, int], optional): 圖表大小。默認為None，使用默認大小。
            title (str, optional): 圖表標題。默認為None。
            
        Returns:
            Figure: Matplotlib圖表對象。
        """
        figsize = figsize or (12, 10)
        fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=self.default_dpi)
        
        # 提取數據
        params_by_type = param_distribution.get('by_type', {})
        params_by_layer = param_distribution.get('by_layer', {})
        
        # 1. 按層類型繪製參數分布餅圖
        if params_by_type:
            ax = axes[0]
            
            # 按參數量排序
            sorted_types = sorted(params_by_type.items(), key=lambda x: x[1], reverse=True)
            types = [t[0] for t in sorted_types]
            params = [t[1] for t in sorted_types]
            
            # 計算百分比
            total = sum(params)
            percentages = [100 * p / total for p in params]
            
            # 繪製餅圖
            wedges, texts, autotexts = ax.pie(
                params, 
                labels=None,
                autopct='',
                startangle=90,
                colors=[self._get_layer_color(t) for t in types]
            )
            
            # 添加圖例
            legend_labels = [f"{t} ({p/1e6:.1f}M, {perc:.1f}%)" if p >= 1e6 else 
                            f"{t} ({p/1e3:.1f}K, {perc:.1f}%)" if p >= 1e3 else
                            f"{t} ({p}, {perc:.1f}%)" 
                            for t, p, perc in zip(types, params, percentages)]
            
            ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_title("Parameter Distribution by Layer Type")
        
        # 2. 按層繪製參數條形圖
        if params_by_layer:
            ax = axes[1]
            
            # 選取參數最多的前10個層
            sorted_layers = sorted(params_by_layer.items(), key=lambda x: x[1], reverse=True)[:10]
            layer_names = [l[0] for l in sorted_layers]
            layer_params = [l[1] for l in sorted_layers]
            
            # 截斷較長的層名稱
            layer_names = [name[:20] + '...' if len(name) > 20 else name for name in layer_names]
            
            # 繪製條形圖
            bars = ax.barh(layer_names, layer_params, color=sns.color_palette("viridis", len(layer_names)))
            
            # 添加參數數量標籤
            for i, bar in enumerate(bars):
                params = layer_params[i]
                if params >= 1e6:
                    param_str = f"{params/1e6:.1f}M"
                elif params >= 1e3:
                    param_str = f"{params/1e3:.1f}K"
                else:
                    param_str = f"{params}"
                
                ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height()/2, 
                       param_str, va='center')
                
            ax.set_title("Top 10 Layers by Parameter Count")
            ax.set_xlabel("Number of Parameters")
        
        # 添加全局標題
        if title:
            fig.suptitle(title, fontsize=16, y=0.98)
        else:
            fig.suptitle("Model Parameter Distribution", fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def plot_layer_complexity(self, layer_complexity: Dict, figsize: Optional[Tuple[int, int]] = None,
                             title: Optional[str] = None) -> Figure:
        """
        繪製層級複雜度圖。
        
        Args:
            layer_complexity (Dict): 層級複雜度數據。
            figsize (Tuple[int, int], optional): 圖表大小。默認為None，使用默認大小。
            title (str, optional): 圖表標題。默認為None。
            
        Returns:
            Figure: Matplotlib圖表對象。
        """
        figsize = figsize or (10, 8)
        fig, ax = plt.subplots(figsize=figsize, dpi=self.default_dpi)
        
        # 提取數據
        layer_ids = []
        complexity_scores = []
        normalized_scores = []
        
        for layer_id, data in layer_complexity.items():
            layer_ids.append(layer_id)
            complexity_scores.append(data['complexity_score'])
            normalized_scores.append(data['normalized_complexity'])
        
        # 按複雜度排序
        sorted_indices = np.argsort(normalized_scores)[::-1]
        layer_ids = [layer_ids[i] for i in sorted_indices[:15]]  # 只顯示最複雜的15個層
        normalized_scores = [normalized_scores[i] for i in sorted_indices[:15]]
        
        # 截斷較長的層名稱
        layer_ids = [name[:20] + '...' if len(name) > 20 else name for name in layer_ids]
        
        # 繪製橫向條形圖
        bars = ax.barh(layer_ids, normalized_scores, color=sns.color_palette("rocket", len(layer_ids)))
        
        # 添加標籤
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height()/2, 
                   f"{normalized_scores[i]:.2f}", va='center')
        
        ax.set_title("Layer Complexity (Normalized)" if not title else title)
        ax.set_xlabel("Normalized Complexity Score")
        ax.set_xlim(0, 1.2)
        
        plt.tight_layout()
        
        return fig
    
    def plot_sequential_paths(self, sequential_paths: List[Dict], figsize: Optional[Tuple[int, int]] = None,
                             max_paths: int = 3, title: Optional[str] = None) -> Figure:
        """
        繪製模型中的主要路徑圖。
        
        Args:
            sequential_paths (List[Dict]): 順序執行路徑數據。
            figsize (Tuple[int, int], optional): 圖表大小。默認為None，使用默認大小。
            max_paths (int, optional): 最多顯示多少條路徑。默認為3。
            title (str, optional): 圖表標題。默認為None。
            
        Returns:
            Figure: Matplotlib圖表對象。
        """
        if not sequential_paths:
            raise ValueError("順序路徑數據為空")
        
        # 按路徑長度排序，選取最長的幾條路徑
        sorted_paths = sorted(sequential_paths, key=lambda x: x.get('length', 0), reverse=True)
        selected_paths = sorted_paths[:max_paths]
        
        figsize = figsize or (12, max_paths * 3)
        fig, axes = plt.subplots(max_paths, 1, figsize=figsize, dpi=self.default_dpi)
        
        # 處理只有一條路徑的情況
        if max_paths == 1:
            axes = [axes]
        
        for i, path_data in enumerate(selected_paths):
            if i >= max_paths:
                break
                
            ax = axes[i]
            path = path_data.get('path', [])
            layer_types = path_data.get('layer_types', [])
            
            # 為每個層創建一個節點
            x = np.arange(len(path))
            y = np.zeros(len(path))
            
            # 繪製連接線
            ax.plot(x, y, '-', color='gray', alpha=0.7, linewidth=2)
            
            # 繪製節點
            for j, (node, node_type) in enumerate(zip(path, layer_types)):
                color = self._get_layer_color(node_type)
                ax.scatter(j, 0, s=300, color=color, edgecolor='black', zorder=2)
                
                # 添加節點標籤
                node_name = node
                if len(node_name) > 15:
                    node_name = node_name[:12] + '...'
                    
                ax.annotate(f"{node_name}\n({node_type})", 
                           (j, 0), 
                           xytext=(0, -20), 
                           textcoords='offset points',
                           ha='center', 
                           fontsize=8, 
                           fontweight='bold')
            
            # 設置軸範圍和標籤
            ax.set_xlim(-0.5, len(path) - 0.5)
            ax.set_ylim(-1, 1)
            ax.set_title(f"Path {i+1} (Length: {len(path)})")
            ax.axis('off')
        
        # 添加全局標題
        if title:
            fig.suptitle(title, fontsize=16, y=0.98)
        else:
            fig.suptitle("Model Sequential Paths", fontsize=16, y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.4)
        
        return fig 