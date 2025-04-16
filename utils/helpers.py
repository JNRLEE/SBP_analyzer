# import numpy as np
# import torch
# import tensorflow as tf

def framework_agnostic_to_numpy(tensor):
    """
    Attempts to convert a tensor (PyTorch, TensorFlow) to a NumPy array.
    Handles CPU/GPU transfer for PyTorch.
    """
    if hasattr(tensor, 'cpu') and hasattr(tensor, 'numpy'): # PyTorch tensor
        return tensor.detach().cpu().numpy()
    elif hasattr(tensor, 'numpy'): # TensorFlow tensor or already numpy
        return tensor.numpy()
    elif isinstance(tensor, (np.ndarray, list, tuple, int, float)): # Already numpy or basic types
         return np.asarray(tensor)
    else:
        raise TypeError(f"Unsupported type for conversion to NumPy: {type(tensor)}")

# Add other common utilities here, e.g.,
# - Function to flatten nested data structures
# - Figure to image conversion for TensorBoard
