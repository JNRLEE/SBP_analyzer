"""
# 神經網絡層內神經元相關性分析模組
#
# 此模組負責分析神經網絡層內神經元之間的相關性和群聚模式，包括：
# - 神經元相關性矩陣計算：評估層內神經元活動的相互關係
# - 神經元群聚識別：基於相關性檢測共同活動的神經元群組
# - 主成分分析：找出層內對激活變異性貢獻最大的方向
# - 相關性統計特徵：提取反映神經元協作程度的統計指標
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import scipy.stats as stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import warnings

from utils.tensor_utils import to_numpy, normalize_tensor


class LayerCorrelationAnalysis:
    """
    神經網絡層內神經元相關性分析類
    
    此類提供方法分析神經網絡層內神經元之間的相關性模式，包括相關性矩陣計算、
    神經元群聚識別、主成分分析等。
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        初始化層相關性分析器
        
        Args:
            device: 計算設備，如果為None則使用CPU
        """
        self.device = device if device is not None else torch.device('cpu')
    
    def compute_neuron_correlation_matrix(self, 
                                        activations: torch.Tensor, 
                                        method: str = 'pearson', 
                                        remove_outliers: bool = False, 
                                        z_threshold: float = 3.0) -> Dict[str, Any]:
        """
        計算層內神經元之間的相關性矩陣
        
        Args:
            activations: 層的激活值，形狀為[batch_size, neurons]或[batch_size, channels, height, width]
            method: 相關性計算方法，'pearson'或'spearman'
            remove_outliers: 是否移除離群值
            z_threshold: 用於識別離群值的z分數閾值
            
        Returns:
            Dict: 包含相關性矩陣和統計特徵的字典
        """
        if not isinstance(activations, (torch.Tensor, np.ndarray)):
            raise TypeError("激活值必須是torch.Tensor或numpy.ndarray類型")
        
        # 轉換為numpy數組
        activations_np = to_numpy(activations)
        
        # 處理卷積層的情況（4D張量）
        if len(activations_np.shape) == 4:  # [batch_size, channels, height, width]
            # 將卷積特徵圖展平為 [batch_size, channels]
            batch_size, channels = activations_np.shape[0], activations_np.shape[1]
            activations_np = activations_np.reshape(batch_size, channels, -1)
            activations_np = np.mean(activations_np, axis=2)  # 對每個特徵圖取平均值
        
        # 處理RNN層的情況（3D張量）
        elif len(activations_np.shape) == 3:  # [batch_size, seq_len, hidden_size]
            # 對序列取平均，得到 [batch_size, hidden_size]
            activations_np = np.mean(activations_np, axis=1)
            
        # 確保張量為2D: [batch_size, num_neurons]
        if len(activations_np.shape) != 2:
            raise ValueError(f"無法處理形狀為 {activations_np.shape} 的激活值張量")
            
        # 移除離群值
        if remove_outliers:
            z_scores = stats.zscore(activations_np, axis=0)
            mask = np.abs(z_scores) < z_threshold
            # 將離群值替換為該行的平均值
            for i in range(activations_np.shape[1]):
                col_mean = np.mean(activations_np[mask[:, i], i])
                outliers = ~mask[:, i]
                activations_np[outliers, i] = col_mean
        
        # 計算相關性矩陣
        num_neurons = activations_np.shape[1]
        corr_matrix = np.zeros((num_neurons, num_neurons))
        
        # 計算相關性
        if method == 'pearson':
            # 使用numpy的corrcoef更高效
            corr_matrix = np.corrcoef(activations_np.T)
        elif method == 'spearman':
            for i in range(num_neurons):
                for j in range(i, num_neurons):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        corr, _ = stats.spearmanr(activations_np[:, i], activations_np[:, j])
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
        else:
            raise ValueError(f"不支持的相關性方法: {method}")
        
        # 處理可能的NaN值
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # 計算相關性統計量
        # 過濾掉對角線上的自相關
        mask = ~np.eye(num_neurons, dtype=bool)
        corr_values = corr_matrix[mask]
        
        avg_corr = np.mean(corr_values)
        pos_corr_ratio = np.sum(corr_values > 0) / len(corr_values)
        neg_corr_ratio = np.sum(corr_values < 0) / len(corr_values)
        strong_corr_ratio = np.sum(np.abs(corr_values) > 0.7) / len(corr_values)
        max_corr = np.max(corr_values) if len(corr_values) > 0 else 0
        min_corr = np.min(corr_values) if len(corr_values) > 0 else 0
        
        return {
            'correlation_matrix': corr_matrix,
            'avg_correlation': avg_corr,
            'pos_corr_ratio': pos_corr_ratio,
            'neg_corr_ratio': neg_corr_ratio,
            'strong_corr_ratio': strong_corr_ratio,
            'max_correlation': max_corr,
            'min_correlation': min_corr
        }
    
    def identify_neuron_clusters(self, 
                               correlation_matrix: np.ndarray, 
                               method: str = 'kmeans', 
                               n_clusters: int = 5) -> Dict[str, Any]:
        """
        基於相關性矩陣識別神經元群組
        
        Args:
            correlation_matrix: 神經元間的相關性矩陣
            method: 聚類方法，'kmeans'或'dbscan'
            n_clusters: 聚類數量(kmeans)或鄰居距離閾值(dbscan)
            
        Returns:
            Dict: 包含聚類結果的字典
        """
        # 將相關性矩陣轉換為距離矩陣：1 - |correlation|
        # 高相關（正或負）的神經元應該有低距離
        distance_matrix = 1.0 - np.abs(correlation_matrix)
        np.fill_diagonal(distance_matrix, 0.0)  # 確保對角線為0
        
        # 根據方法進行聚類
        if method.lower() == 'kmeans':
            # 使用主成分分析將距離矩陣轉換為特徵向量
            pca = PCA(n_components=min(n_clusters*2, distance_matrix.shape[0]))
            try:
                # 使用PCA降維
                features = pca.fit_transform(distance_matrix)
                
                # 應用KMeans聚類
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(features)
                
                # 計算每個群組的內部相關性和大小
                clusters = {}
                for i in range(n_clusters):
                    cluster_indices = np.where(labels == i)[0]
                    if len(cluster_indices) > 1:
                        # 獲取該群組內神經元的相關性子矩陣
                        sub_matrix = correlation_matrix[np.ix_(cluster_indices, cluster_indices)]
                        # 計算平均內部相關性（排除對角線的1）
                        mask = np.ones(sub_matrix.shape, dtype=bool)
                        np.fill_diagonal(mask, 0)
                        internal_correlation = np.mean(sub_matrix[mask])
                    else:
                        internal_correlation = 0.0
                    
                    clusters[f"cluster_{i}"] = {
                        "indices": cluster_indices.tolist(),
                        "size": len(cluster_indices),
                        "internal_correlation": float(internal_correlation)
                    }
                
                return {
                    "clustering_method": "kmeans",
                    "n_clusters": n_clusters,
                    "cluster_labels": labels.tolist(),
                    "clusters": clusters,
                    "explained_variance_ratio": pca.explained_variance_ratio_.tolist()
                }
            except Exception as e:
                warnings.warn(f"KMeans聚類失敗: {str(e)}")
                return {"error": str(e)}
                
        elif method.lower() == 'dbscan':
            try:
                # 直接從距離矩陣使用DBSCAN
                dbscan = DBSCAN(eps=n_clusters/10.0, metric='precomputed')
                labels = dbscan.fit_predict(distance_matrix)
                
                # 獲取唯一群組標籤
                unique_labels = np.unique(labels)
                n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
                
                # 計算每個群組的內部相關性和大小
                clusters = {}
                for i in unique_labels:
                    if i == -1:  # 噪聲樣本
                        continue
                    
                    cluster_indices = np.where(labels == i)[0]
                    if len(cluster_indices) > 1:
                        # 獲取該群組內神經元的相關性子矩陣
                        sub_matrix = correlation_matrix[np.ix_(cluster_indices, cluster_indices)]
                        # 計算平均內部相關性（排除對角線的1）
                        mask = np.ones(sub_matrix.shape, dtype=bool)
                        np.fill_diagonal(mask, 0)
                        internal_correlation = np.mean(sub_matrix[mask])
                    else:
                        internal_correlation = 0.0
                    
                    clusters[f"cluster_{i}"] = {
                        "indices": cluster_indices.tolist(),
                        "size": len(cluster_indices),
                        "internal_correlation": float(internal_correlation)
                    }
                
                # 標記為噪聲的神經元
                noise_indices = np.where(labels == -1)[0]
                
                return {
                    "clustering_method": "dbscan",
                    "eps_parameter": n_clusters/10.0,
                    "n_clusters_found": n_clusters_found,
                    "cluster_labels": labels.tolist(),
                    "clusters": clusters,
                    "noise_indices": noise_indices.tolist(),
                    "noise_count": len(noise_indices)
                }
            except Exception as e:
                warnings.warn(f"DBSCAN聚類失敗: {str(e)}")
                return {"error": str(e)}
        else:
            return {"error": f"不支持的聚類方法: {method}"}
    
    def compute_principal_components(self, 
                                   activations: torch.Tensor, 
                                   n_components: int = 10) -> Dict[str, Any]:
        """
        對層的激活值進行主成分分析
        
        Args:
            activations: 層的激活值，形狀為[batch_size, neurons]或[batch_size, channels, height, width]
            n_components: 要計算的主成分數量
            
        Returns:
            Dict: 包含PCA結果的字典
        """
        # 確保輸入是張量
        if not isinstance(activations, torch.Tensor):
            activations = torch.tensor(activations, device=self.device)
        
        activations = activations.to(self.device)
        
        # 將輸入轉換為二維形狀[batch_size, features]
        if activations.dim() == 2:
            # 已經是二維形狀
            act_2d = activations
        elif activations.dim() == 4:
            # 卷積層形狀[batch_size, channels, height, width]
            batch_size, channels, height, width = activations.shape
            act_2d = activations.reshape(batch_size, -1)
        else:
            raise ValueError(f"不支持的激活形狀: {activations.shape}，需要2維或4維張量")
        
        # 將數據轉移到CPU以進行PCA
        act_np = act_2d.detach().cpu().numpy()
        
        # 進行主成分分析
        try:
            pca = PCA(n_components=min(n_components, act_np.shape[1], act_np.shape[0]))
            transformed = pca.fit_transform(act_np)
            
            # 計算累積解釋方差
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            # 找出解釋90%方差所需的成分數
            components_for_90 = np.searchsorted(cumulative_variance, 0.9) + 1
            components_for_90 = min(components_for_90, len(cumulative_variance))
            
            return {
                "transformed_data": transformed,
                "components": pca.components_,
                "explained_variance": pca.explained_variance_.tolist(),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": cumulative_variance.tolist(),
                "components_for_90_percent": int(components_for_90),
                "shape": list(activations.shape)
            }
        except Exception as e:
            warnings.warn(f"PCA分析失敗: {str(e)}")
            return {
                "error": str(e),
                "shape": list(activations.shape)
            }
    
    def analyze_neuron_correlations(self, 
                                  activations: torch.Tensor, 
                                  detailed: bool = False) -> Dict[str, Any]:
        """
        綜合分析層內神經元的相關性
        
        Args:
            activations: 層的激活值
            detailed: 是否進行詳細分析，包括聚類和PCA
            
        Returns:
            Dict: 分析結果
        """
        results = {}
        
        # 1. 計算相關性矩陣
        correlation_data = self.compute_neuron_correlation_matrix(activations)
        
        if "error" in correlation_data:
            return correlation_data
        
        results["correlation_statistics"] = {
            "average_correlation": correlation_data["avg_correlation"],
            "positive_correlation_ratio": correlation_data["pos_corr_ratio"],
            "negative_correlation_ratio": correlation_data["neg_corr_ratio"],
            "strong_correlation_ratio": correlation_data["strong_corr_ratio"]
        }
        
        if detailed:
            # 2. 神經元聚類
            correlation_matrix = correlation_data["correlation_matrix"]
            kmeans_clusters = self.identify_neuron_clusters(correlation_matrix, method='kmeans')
            
            if "error" not in kmeans_clusters:
                results["neuron_clusters"] = kmeans_clusters
            
            # 3. 主成分分析
            pca_results = self.compute_principal_components(activations)
            
            if "error" not in pca_results:
                results["pca_analysis"] = {
                    "explained_variance_ratio": pca_results["explained_variance_ratio"],
                    "cumulative_variance": pca_results["cumulative_variance"],
                    "components_for_90_percent": pca_results["components_for_90_percent"]
                }
            
            # 4. 識別功能群組
            functional_groups = self.detect_functional_groups(correlation_matrix)
            results["functional_groups"] = functional_groups
        
        return results
    
    def compute_activation_coordination(self, 
                                      activations: torch.Tensor) -> Dict[str, float]:
        """
        計算層內神經元的激活協調程度
        
        Args:
            activations: 層的激活值
            
        Returns:
            Dict: 協調性指標
        """
        # 確保輸入是張量
        if not isinstance(activations, torch.Tensor):
            activations = torch.tensor(activations, device=self.device)
        
        activations = activations.to(self.device)
        
        # 將輸入轉換為二維形狀[batch_size, neurons]
        if activations.dim() == 2:
            # 已經是二維形狀
            act_2d = activations
        elif activations.dim() == 4:
            # 卷積層形狀[batch_size, channels, height, width]
            batch_size, channels, height, width = activations.shape
            # 對於卷積層，我們重點關注通道間的協調性
            act_2d = activations.reshape(batch_size, channels, -1).mean(dim=2)
        else:
            raise ValueError(f"不支持的激活形狀: {activations.shape}，需要2維或4維張量")
        
        # 將數據轉移到CPU以進行計算
        act_np = act_2d.detach().cpu().numpy()
        
        try:
            # 計算相關性矩陣
            correlation_matrix = np.corrcoef(act_np.T)
            
            # 處理可能的無效值
            correlation_matrix = np.nan_to_num(correlation_matrix)
            
            # 計算平均絕對相關性
            n_neurons = correlation_matrix.shape[0]
            mask = np.ones((n_neurons, n_neurons), dtype=bool)
            np.fill_diagonal(mask, 0)  # 排除對角線元素
            avg_abs_correlation = np.mean(np.abs(correlation_matrix[mask]))
            
            # 使用圖論方法計算協調度
            # 將相關性矩陣轉換為圖，只保留強相關
            threshold = 0.5
            adjacency = (np.abs(correlation_matrix) > threshold) & mask
            
            # 構建圖
            G = nx.from_numpy_array(adjacency)
            
            # 計算圖的連通性指標
            try:
                # 連通組件數量
                n_components = nx.number_connected_components(G)
                
                # 最大連通組件的大小
                largest_cc_size = len(max(nx.connected_components(G), key=len))
                
                # 集群係數(反映神經元間的緊密組織程度)
                avg_clustering = nx.average_clustering(G)
                
                # 計算神經元協調分數：結合相關性和圖結構
                coordination_score = avg_abs_correlation * (largest_cc_size / n_neurons)
                
                return {
                    "average_absolute_correlation": float(avg_abs_correlation),
                    "connected_components": int(n_components),
                    "largest_component_size": int(largest_cc_size),
                    "largest_component_ratio": float(largest_cc_size / n_neurons),
                    "average_clustering": float(avg_clustering),
                    "coordination_score": float(coordination_score)
                }
            except Exception as e:
                # 如果圖論分析失敗，僅返回基本相關性指標
                return {
                    "average_absolute_correlation": float(avg_abs_correlation),
                    "coordination_score": float(avg_abs_correlation),
                    "graph_analysis_error": str(e)
                }
                
        except Exception as e:
            return {
                "error": str(e)
            }
    
    def detect_functional_groups(self, 
                               correlation_matrix: np.ndarray, 
                               threshold: float = 0.6) -> Dict[str, List[int]]:
        """
        在相關性矩陣中檢測功能群組（高度相關的神經元子集）
        
        Args:
            correlation_matrix: 神經元相關性矩陣
            threshold: 判定為功能群組的相關性閾值
            
        Returns:
            Dict: 映射群組ID到神經元索引的字典
        """
        # 確保相關性矩陣有效
        correlation_matrix = np.nan_to_num(correlation_matrix)
        n_neurons = correlation_matrix.shape[0]
        
        # 創建二值化的相關性網絡（只保留強相關）
        mask = np.ones((n_neurons, n_neurons), dtype=bool)
        np.fill_diagonal(mask, 0)  # 排除對角線元素
        
        # 正相關群組
        positive_adjacency = (correlation_matrix > threshold) & mask
        positive_G = nx.from_numpy_array(positive_adjacency)
        
        # 負相關群組（相互抑制的神經元）
        negative_adjacency = (correlation_matrix < -threshold) & mask
        negative_G = nx.from_numpy_array(negative_adjacency)
        
        # 查找連通組件（功能群組）
        positive_groups = {}
        for i, component in enumerate(nx.connected_components(positive_G)):
            if len(component) >= 3:  # 只考慮至少有3個神經元的群組
                positive_groups[f"positive_group_{i}"] = sorted(list(component))
        
        negative_groups = {}
        for i, component in enumerate(nx.connected_components(negative_G)):
            if len(component) >= 3:  # 只考慮至少有3個神經元的群組
                negative_groups[f"negative_group_{i}"] = sorted(list(component))
        
        # 查找中心神經元（與多個其他神經元高度相關的神經元）
        centrality = np.sum(positive_adjacency, axis=1)
        central_neurons = np.where(centrality > n_neurons * 0.2)[0].tolist()
        
        return {
            "positive_groups": positive_groups,
            "negative_groups": negative_groups,
            "central_neurons": central_neurons,
            "threshold_used": threshold
        } 