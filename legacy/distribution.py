import numpy as np
# from scipy import stats # For skewness, kurtosis

def calculate_distribution_stats(data):
    """
    Calculates basic distribution statistics for given data.

    Args:
        data (array-like): Input data (flattened).

    Returns:
        dict: Dictionary containing statistics like mean, std, min, max, median, etc.
    """
    data_np = np.asarray(data).flatten()

    stats_dict = {
        'mean': np.mean(data_np),
        'std': np.std(data_np),
        'min': np.min(data_np),
        'max': np.max(data_np),
        'median': np.median(data_np),
        'q25': np.percentile(data_np, 25),
        'q75': np.percentile(data_np, 75),
    }

    # Optional: Add more complex stats if scipy is available
    try:
        from scipy import stats
        stats_dict['skewness'] = stats.skew(data_np)
        stats_dict['kurtosis'] = stats.kurtosis(data_np) # Fisher's definition (normal=0)
    except ImportError:
        pass # scipy not installed

    return stats_dict
