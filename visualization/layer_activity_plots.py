import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from typing import Dict, List, Optional, Union, Tuple, Any

# Plot cosine similarity matrix between layer activations
def plot_cosine_similarity_matrix(
    embeddings: Union[np.ndarray, torch.Tensor], 
    labels: Optional[List[str]] = None,
    title: str = "Cosine Similarity Matrix",
    output_path: Optional[str] = None,
    cmap: str = "viridis",
    annotate: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    use_precomputed_matrix: bool = False
) -> plt.Figure:
    """
    繪製餘弦相似度矩陣以分析嵌入向量之間的相似性。
    
    Parameters:
        embeddings (Union[np.ndarray, torch.Tensor]): 嵌入矩陣，形狀為 [n_samples, n_features]，
            或者如果 use_precomputed_matrix=True，則為已預計算的相似度矩陣，形狀為 [n_samples, n_samples]
        labels (List[str], optional): 每個樣本的標籤。如果為 None，則使用數字索引。
        title (str): 圖表標題
        output_path (str, optional): 輸出圖像的保存路徑。如果為 None，則只返回圖形對象。
        cmap (str): 熱力圖的顏色映射
        annotate (bool): 是否在熱力圖上註解相似度值
        figsize (Tuple[int, int]): 圖形大小
        use_precomputed_matrix (bool): 如果為 True，則 embeddings 參數被視為已預計算的相似度矩陣，
            而不是嵌入向量。這在處理大型數據集時非常有用。
            
    Returns:
        plt.Figure: Matplotlib 圖形對象。
    """
    # 確保為 numpy 陣列
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    # 計算餘弦相似度矩陣（如果尚未計算）
    if not use_precomputed_matrix:
        # 首先將每個向量歸一化為長度為 1
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # 計算向量之間的點積，即餘弦相似度
        cosine_sim = np.dot(norm_embeddings, norm_embeddings.T)
    else:
        # 直接使用已預計算的相似度矩陣
        cosine_sim = embeddings
    
    # 創建圖形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 繪製熱力圖
    im = ax.imshow(cosine_sim, cmap=cmap, vmin=-1, vmax=1)
    
    # 添加標題和標籤
    ax.set_title(title)
    
    # 添加坐標標籤
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
    
    # 添加顏色條
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity')
    
    # 在每個單元格上註解相似度值
    if annotate and cosine_sim.shape[0] <= 20:  # 如果單元格太多，註解會很擁擠
        for i in range(cosine_sim.shape[0]):
            for j in range(cosine_sim.shape[1]):
                text_color = 'white' if abs(cosine_sim[i, j]) > 0.5 else 'black'
                ax.text(j, i, f"{cosine_sim[i, j]:.2f}", 
                        ha="center", va="center", color=text_color)
    
    plt.tight_layout()
    
    # 保存圖像
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_layer_activations_heatmap(
    activations: Union[np.ndarray, torch.Tensor],
    title: str = "Layer Activations Heatmap",
    output_path: Optional[str] = None,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (12, 8),
    max_items: int = 100
) -> plt.Figure:
    """
    Plot heatmap of layer activations.
    
    Parameters:
        activations (Union[np.ndarray, torch.Tensor]): Activation tensor or array
        title (str): Chart title
        output_path (str, optional): Path to save the output image
        cmap (str): Colormap for the heatmap
        figsize (Tuple[int, int]): Figure size
        max_items (int): Maximum number of samples to display, to avoid oversized charts
            
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Ensure numpy array
    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().numpy()
    
    # Process based on activation dimensions
    if activations.ndim == 2:  # [batch_size, features]
        # Use original activations
        data = activations
    elif activations.ndim == 3:  # [batch_size, seq_len, features]
        # Take sequence average for each sample
        data = np.mean(activations, axis=1)
    elif activations.ndim == 4:  # [batch_size, channels, height, width]
        # For convolutional layers, compress feature maps into a single vector
        batch_size, channels, height, width = activations.shape
        data = activations.reshape(batch_size, channels * height * width)
    else:
        # For higher dimensions, perform appropriate dimensionality reduction
        data = activations.reshape(activations.shape[0], -1)
    
    # Limit sample count to avoid oversized charts
    if data.shape[0] > max_items:
        data = data[:max_items]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(data, aspect='auto', cmap=cmap)
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('Feature Dimension')
    ax.set_ylabel('Sample Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Activation Value')
    
    # If feature dimension is large, only show some ticks
    if data.shape[1] > 10:
        feature_ticks = np.linspace(0, data.shape[1]-1, 10, dtype=int)
        ax.set_xticks(feature_ticks)
    
    plt.tight_layout()
    
    # Save image
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_activation_statistics(
    stats: Dict, 
    layer_name: Optional[str] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot statistics of activation values.
    
    Parameters:
        stats (Dict): Dictionary containing statistical properties, should include 'mean', 'std', 'median', 'min', 'max', 'sparsity', etc.
        layer_name (str, optional): Name of the layer. If provided, will be added to the title.
        title (str, optional): Chart title. If None, default title will be used.
        output_path (str, optional): Path to save the output image. If None, only returns the figure object.
        figsize (Tuple[int, int], optional): Figure size.
            
    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Set title
    if title is None:
        if layer_name:
            title = f"{layer_name} Activation Statistics"
        else:
            title = "Activation Statistics"
    
    fig.suptitle(title, fontsize=16)
    
    # 1. Basic metrics bar chart
    basic_metrics = ['mean', 'std', 'median', 'min', 'max']
    values = [stats[m] for m in basic_metrics]
    
    axs[0, 0].bar(basic_metrics, values)
    axs[0, 0].set_title("Basic Statistics")
    axs[0, 0].set_ylabel("Value")
    axs[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Percentiles line plot
    if 'percentiles' in stats:
        percentiles = stats['percentiles']
        p_keys = sorted(percentiles.keys())
        p_values = [percentiles[k] for k in p_keys]
        
        axs[0, 1].plot(p_keys, p_values, 'o-')
        axs[0, 1].set_title("Percentile Distribution")
        axs[0, 1].set_xlabel("Percentile")
        axs[0, 1].set_ylabel("Value")
        axs[0, 1].grid(True)
    
    # 3. Sparsity pie chart
    sparsity = stats.get('sparsity', 0)
    labels = ['Zero Elements', 'Non-zero Elements']
    sizes = [sparsity, 1 - sparsity]
    
    axs[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axs[1, 0].set_title(f"Sparsity: {sparsity:.2%}")
    
    # 4. Statistics summary table
    axs[1, 1].axis('off')
    
    # Create statistics summary text
    txt = "Statistics Summary:\n\n"
    for k, v in stats.items():
        if k != 'percentiles':  # Skip percentiles dictionary
            txt += f"{k}: {v:.6f}\n"
    
    axs[1, 1].text(0.05, 0.95, txt, transform=axs[1, 1].transAxes, 
                  va='top', fontsize=10, family='monospace')
    
    plt.tight_layout()
    
    # Save image
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig 