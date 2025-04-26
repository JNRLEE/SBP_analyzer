"""
# 神經網絡層間關係分析模組
#
# 此模組負責分析神經網絡中不同層之間的關係，包括：
# - 層間相關性：評估不同層之間的激活值相關程度
# - 信息流分析：追踪信息如何在網絡層間流動
# - 瓶頸檢測：識別網絡中的信息瓶頸
# - 表示相似度：計算不同層表示的相似程度
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd
import warnings

class LayerRelationshipAnalysis:
    """
    神經網絡層間關係分析類
    
    此類提供方法分析神經網絡中不同層之間的關係，包括相關性、
    信息流動、表示相似度等。
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        初始化層關係分析器
        
        Args:
            device: 計算設備，如果為None則使用CPU
        """
        self.device = device if device is not None else torch.device('cpu')
    
    def compute_activation_correlation(self, 
                                     layer1_activations: torch.Tensor, 
                                     layer2_activations: torch.Tensor, 
                                     sample_size: int = 1000) -> Dict[str, Any]:
        """
        計算兩個層的激活值之間的相關性
        
        Args:
            layer1_activations: 第一個層的激活值，形狀為[batch_size, features1]
            layer2_activations: 第二個層的激活值，形狀為[batch_size, features2]
            sample_size: 用於計算的樣本數量，若小於實際樣本數則隨機抽樣
            
        Returns:
            Dict: 包含不同相關性度量的字典
        """
        # 將輸入轉換為2D張量
        layer1_flat = self._flatten_activations(layer1_activations)
        layer2_flat = self._flatten_activations(layer2_activations)
        
        # 確保批次大小一致
        assert layer1_flat.shape[0] == layer2_flat.shape[0], "兩層的批次大小必須相同"
        
        # 如果需要，進行隨機抽樣
        batch_size = layer1_flat.shape[0]
        if sample_size < batch_size:
            indices = torch.randperm(batch_size)[:sample_size]
            layer1_flat = layer1_flat[indices]
            layer2_flat = layer2_flat[indices]
        
        # 轉換為numpy進行計算
        layer1_np = layer1_flat.cpu().numpy()
        layer2_np = layer2_flat.cpu().numpy()
        
        # 計算CKA（中心核心對齊）相似度
        cka = self._compute_cka(layer1_flat, layer2_flat)
        
        # 計算典型相關分析（如果可行）
        try:
            cca_corr = self._compute_canonical_correlation(layer1_np, layer2_np)
        except Exception as e:
            warnings.warn(f"計算典型相關分析時出錯: {str(e)}")
            cca_corr = None
        
        # 計算餘弦相似度
        cosine_sim = self._compute_cosine_similarity(layer1_flat, layer2_flat)
        
        return {
            "cka_similarity": float(cka),
            "canonical_correlation": float(cca_corr) if cca_corr is not None else None,
            "cosine_similarity": float(cosine_sim),
            "layer1_shape": list(layer1_activations.shape),
            "layer2_shape": list(layer2_activations.shape)
        }
    
    def _compute_cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        計算中心核心對齊相似度 (Centered Kernel Alignment)
        
        Args:
            X: 第一個層的激活值，形狀為[batch_size, features1]
            Y: 第二個層的激活值，形狀為[batch_size, features2]
            
        Returns:
            float: CKA相似度，範圍[0, 1]
        """
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
        
        XTX = torch.matmul(X.T, X)
        YTY = torch.matmul(Y.T, Y)
        
        XTY = torch.matmul(X.T, Y)
        
        hsic = torch.sum(XTY * XTY)
        var1 = torch.sum(XTX * XTX)
        var2 = torch.sum(YTY * YTY)
        
        return hsic / (torch.sqrt(var1 * var2) + 1e-8)
    
    def _compute_canonical_correlation(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        計算兩組變量之間的典型相關係數 (第一典型相關)
        
        Args:
            X: 第一個層的激活值，形狀為[batch_size, features1]
            Y: 第二個層的激活值，形狀為[batch_size, features2]
            
        Returns:
            float: 典型相關係數，範圍[-1, 1]
        """
        # 標準化數據
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)
        
        # 處理低秩數據
        n, p = X.shape
        q = Y.shape[1]
        
        if p > n or q > n:
            # 使用SVD降維
            if p > n:
                u, s, v = svd(X, full_matrices=False)
                X = u * s
                p = n
            
            if q > n:
                u, s, v = svd(Y, full_matrices=False)
                Y = u * s
                q = n
        
        # 計算相關矩陣
        Cxx = np.corrcoef(X, rowvar=False)
        Cyy = np.corrcoef(Y, rowvar=False)
        Cxy = np.corrcoef(X, Y, rowvar=False)[:p, p:]
        
        # 處理可能的數值問題
        Cxx = Cxx + np.eye(p) * 1e-8
        Cyy = Cyy + np.eye(q) * 1e-8
        
        # 計算典型相關
        Cxx_sqrt_inv = np.linalg.inv(np.linalg.cholesky(Cxx))
        Cyy_sqrt_inv = np.linalg.inv(np.linalg.cholesky(Cyy))
        
        T = Cxx_sqrt_inv @ Cxy @ Cyy_sqrt_inv
        
        try:
            u, s, v = np.linalg.svd(T, full_matrices=False)
            return s[0]  # 返回第一典型相關係數
        except np.linalg.LinAlgError:
            # 如果SVD失敗，計算替代的相關係數
            corr_matrix = np.corrcoef(X.mean(axis=1), Y.mean(axis=1))
            return corr_matrix[0, 1]
    
    def _compute_cosine_similarity(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        計算兩個層激活平均向量的餘弦相似度
        
        Args:
            X: 第一個層的激活值，形狀為[batch_size, features1]
            Y: 第二個層的激活值，形狀為[batch_size, features2]
            
        Returns:
            float: 餘弦相似度，範圍[-1, 1]
        """
        X_mean = X.mean(dim=0)
        Y_mean = Y.mean(dim=0)
        
        cos_sim = torch.nn.functional.cosine_similarity(X_mean.unsqueeze(0), Y_mean.unsqueeze(0))
        return cos_sim.item()
    
    def trace_activation_flow(self, 
                            layer_activations: Dict[str, torch.Tensor], 
                            layer_order: List[str]) -> Dict[str, Dict[str, float]]:
        """
        追踪激活值如何在層間流動，使用相鄰層之間的相關程度
        
        Args:
            layer_activations: 層名到激活值的映射
            layer_order: 層的順序列表，應該對應於網絡中的前向順序
            
        Returns:
            Dict: 每層到相鄰層的信息流指標
        """
        flow_metrics = {}
        
        # 檢查輸入是否有效
        valid_layer_order = [layer for layer in layer_order if layer in layer_activations]
        if len(valid_layer_order) < 2:
            return {"error": "需要至少兩層有效的激活值"}
        
        # 計算相鄰層之間的信息流
        for i in range(len(valid_layer_order) - 1):
            src_layer = valid_layer_order[i]
            dst_layer = valid_layer_order[i + 1]
            
            src_activations = layer_activations[src_layer]
            dst_activations = layer_activations[dst_layer]
            
            # 計算相關性測量
            correlation = self.compute_activation_correlation(
                src_activations, dst_activations
            )
            
            # 計算信息流增益/損失 (基於相對熵的簡化估計)
            src_entropy = self._compute_layer_entropy(src_activations)
            dst_entropy = self._compute_layer_entropy(dst_activations)
            information_change = dst_entropy - src_entropy
            
            # 記錄結果
            if src_layer not in flow_metrics:
                flow_metrics[src_layer] = {}
            
            flow_metrics[src_layer][dst_layer] = {
                "correlation": correlation,
                "information_change": float(information_change),
                "normalized_flow": float(correlation["cka_similarity"] * (1 + np.tanh(information_change)))
            }
        
        return flow_metrics
    
    def find_information_bottlenecks(self, 
                                   layer_activations: Dict[str, torch.Tensor], 
                                   layer_order: List[str]) -> Dict[str, float]:
        """
        基於相鄰層之間的信息流變化找出網絡中的信息瓶頸
        
        Args:
            layer_activations: 層名到激活值的映射
            layer_order: 層的順序列表，應該對應於網絡中的前向順序
            
        Returns:
            Dict: 每層的瓶頸分數，分數越高表示該層可能是瓶頸
        """
        # 獲取激活流
        flow_data = self.trace_activation_flow(layer_activations, layer_order)
        
        # 檢查流數據是否有效
        if "error" in flow_data:
            return {"error": flow_data["error"]}
        
        bottleneck_scores = {}
        valid_layers = set(layer_order).intersection(set(layer_activations.keys()))
        valid_layer_order = [layer for layer in layer_order if layer in valid_layers]
        
        # 計算每個內部層的瓶頸分數
        for i in range(1, len(valid_layer_order) - 1):
            layer = valid_layer_order[i]
            prev_layer = valid_layer_order[i-1]
            next_layer = valid_layer_order[i+1]
            
            # 檢查必要的數據是否存在
            if prev_layer not in flow_data or layer not in flow_data:
                continue
                
            if next_layer not in flow_data[layer] or layer not in flow_data[prev_layer]:
                continue
            
            # 獲取流入和流出的指標
            flow_in = flow_data[prev_layer][layer]["normalized_flow"]
            flow_out = flow_data[layer][next_layer]["normalized_flow"]
            
            # 計算瓶頸分數：流入與流出的比率差異
            # 高流入低流出表示信息在該層被壓縮，可能是瓶頸
            bottleneck_score = (flow_in - flow_out) / (flow_in + 1e-8)
            
            bottleneck_scores[layer] = float(bottleneck_score)
        
        # 處理首尾層
        if len(valid_layer_order) > 0:
            bottleneck_scores[valid_layer_order[0]] = 0.0  # 首層不是瓶頸
        
        if len(valid_layer_order) > 1:
            bottleneck_scores[valid_layer_order[-1]] = 0.0  # 尾層不計算瓶頸分數
        
        return bottleneck_scores
    
    def _compute_layer_entropy(self, activations: torch.Tensor) -> float:
        """計算層激活值的熵"""
        # 將激活值展平並轉換為概率分布
        flat_act = activations.reshape(-1).cpu().numpy()
        hist, _ = np.histogram(flat_act, bins=100, density=True)
        hist = hist + 1e-10  # 避免log(0)
        hist = hist / np.sum(hist)  # 正規化為概率
        return -np.sum(hist * np.log(hist))
    
    def detect_influence_patterns(self, 
                                 layer_activations: Dict[str, torch.Tensor], 
                                 layer_order: List[str],
                                 influence_window: int = 3) -> Dict[str, Dict[str, float]]:
        """
        檢測層之間的影響模式，包括長距離依賴關係
        
        Args:
            layer_activations: 層名到激活值的映射
            layer_order: 層的順序列表
            influence_window: 考慮影響的最大層距離
            
        Returns:
            Dict: 層之間的影響強度
        """
        influence_patterns = {}
        valid_layers = set(layer_order).intersection(set(layer_activations.keys()))
        valid_layer_order = [layer for layer in layer_order if layer in valid_layers]
        
        if len(valid_layer_order) < 2:
            return {"error": "需要至少兩層有效的激活值"}
        
        # 計算每對層之間的影響模式
        for i in range(len(valid_layer_order)):
            src_layer = valid_layer_order[i]
            influence_patterns[src_layer] = {}
            
            # 考慮window範圍內的影響
            for j in range(i+1, min(i+influence_window+1, len(valid_layer_order))):
                dst_layer = valid_layer_order[j]
                
                # 計算影響強度
                correlation = self.compute_activation_correlation(
                    layer_activations[src_layer], 
                    layer_activations[dst_layer]
                )
                
                # 根據距離調整影響強度
                distance = j - i
                decay_factor = 1.0 / distance  # 影響隨距離衰減
                
                adjusted_influence = correlation["cka_similarity"] * decay_factor
                
                # 記錄結果
                influence_patterns[src_layer][dst_layer] = {
                    "raw_correlation": correlation["cka_similarity"],
                    "distance": distance,
                    "influence_strength": float(adjusted_influence)
                }
        
        return influence_patterns
    
    def compute_representation_similarity(self, 
                                        activations1: Dict[str, torch.Tensor],
                                        activations2: Dict[str, torch.Tensor],
                                        layer_names: List[str]) -> Dict[str, float]:
        """
        計算兩組模型在相同層上的表示相似度
        
        Args:
            activations1: 第一個模型的層激活值
            activations2: 第二個模型的層激活值
            layer_names: 要比較的層名列表
            
        Returns:
            Dict: 每層的表示相似度
        """
        similarity_scores = {}
        
        for layer_name in layer_names:
            if layer_name not in activations1 or layer_name not in activations2:
                continue
                
            # 獲取兩個模型的層激活值
            act1 = activations1[layer_name]
            act2 = activations2[layer_name]
            
            # 計算表示相似度
            correlation = self.compute_activation_correlation(act1, act2)
            
            # 記錄CKA相似度作為主要指標
            similarity_scores[layer_name] = float(correlation["cka_similarity"])
        
        return similarity_scores
    
    def detect_coactivation_patterns(self, 
                                   layer_activations: Dict[str, torch.Tensor],
                                   layer_names: List[str],
                                   threshold: float = 0.7) -> Dict[str, Dict[str, float]]:
        """
        檢測層之間的共同激活模式
        
        Args:
            layer_activations: 層名到激活值的映射
            layer_names: 層名列表
            threshold: 判定為共同激活的相關閾值
            
        Returns:
            Dict: 層對之間的共同激活強度
        """
        coactivation_patterns = {}
        valid_layers = set(layer_names).intersection(set(layer_activations.keys()))
        
        if len(valid_layers) < 2:
            return {"error": "需要至少兩層有效的激活值"}
        
        valid_layer_names = list(valid_layers)
        
        # 計算每對層之間的共同激活模式
        for i in range(len(valid_layer_names)):
            layer1 = valid_layer_names[i]
            if layer1 not in coactivation_patterns:
                coactivation_patterns[layer1] = {}
                
            for j in range(i+1, len(valid_layer_names)):
                layer2 = valid_layer_names[j]
                
                # 計算相關性
                correlation = self.compute_activation_correlation(
                    layer_activations[layer1], 
                    layer_activations[layer2]
                )
                
                # 判斷是否為共同激活
                is_coactivated = correlation["cka_similarity"] >= threshold
                
                # 記錄結果
                coactivation_patterns[layer1][layer2] = {
                    "correlation": correlation["cka_similarity"],
                    "is_coactivated": is_coactivated,
                    "strength": correlation["cka_similarity"] if is_coactivated else 0.0
                }
        
        return coactivation_patterns
    
    def analyze_layer_relationships(self, 
                                  layer_activations: Dict[str, torch.Tensor], 
                                  layer_order: List[str]) -> Dict[str, Any]:
        """
        綜合分析層間關係
        
        Args:
            layer_activations: 層名到激活值的映射
            layer_order: 層的順序列表
            
        Returns:
            Dict: 各種層間關係分析結果
        """
        results = {}
        
        # 1. 計算激活流
        flow_data = self.trace_activation_flow(layer_activations, layer_order)
        if "error" not in flow_data:
            results["activation_flow"] = flow_data
        
        # 2. 檢測信息瓶頸
        bottlenecks = self.find_information_bottlenecks(layer_activations, layer_order)
        if "error" not in bottlenecks:
            results["bottlenecks"] = bottlenecks
        
        # 3. 檢測影響模式
        influence = self.detect_influence_patterns(layer_activations, layer_order)
        if "error" not in influence:
            results["influence_patterns"] = influence
        
        # 4. 檢測共同激活模式
        coactivation = self.detect_coactivation_patterns(layer_activations, layer_order)
        if "error" not in coactivation:
            results["coactivation_patterns"] = coactivation
        
        return results
    
    def compute_gradual_distortion(self, 
                                 layer_activations: Dict[str, torch.Tensor], 
                                 layer_order: List[str]) -> Dict[str, float]:
        """
        計算信息在網絡中逐漸變形的程度
        
        Args:
            layer_activations: 層名到激活值的映射
            layer_order: 層的順序列表
            
        Returns:
            Dict: 每層到輸入層的變形程度
        """
        distortion = {}
        valid_layers = set(layer_order).intersection(set(layer_activations.keys()))
        
        if len(valid_layers) < 2:
            return {"error": "需要至少兩層有效的激活值"}
        
        valid_layer_order = [layer for layer in layer_order if layer in valid_layers]
        input_layer = valid_layer_order[0]
        
        # 對每層計算與輸入層的相似度
        for layer in valid_layer_order[1:]:
            correlation = self.compute_activation_correlation(
                layer_activations[input_layer], 
                layer_activations[layer]
            )
            
            # 變形程度 = 1 - 相似度
            distortion[layer] = 1.0 - correlation["cka_similarity"]
        
        # 輸入層到自身的變形為0
        distortion[input_layer] = 0.0
        
        return distortion
    
    def _flatten_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """
        將激活值展平為2D張量 [batch_size, features]
        
        Args:
            activations: 輸入激活值，可以是任意維度
            
        Returns:
            torch.Tensor: 展平的激活值
        """
        if activations.dim() <= 2:
            return activations
            
        batch_size = activations.shape[0]
        return activations.reshape(batch_size, -1) 