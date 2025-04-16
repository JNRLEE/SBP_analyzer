# import numpy as np # Or torch/tensorflow

class DataValidator:
    """
    Provides methods to validate data properties.
    """
    @staticmethod
    def check_shape(data, expected_shape, name="Data"):
        """ Checks if data shape matches the expected shape (allows None for dimensions). """
        actual_shape = getattr(data, 'shape', None)
        if actual_shape is None:
            print(f"Warning: Cannot get shape for {name}.")
            return True # Or False? Decide behavior

        if len(actual_shape) != len(expected_shape):
            print(f"Error: {name} has wrong number of dimensions. Expected {len(expected_shape)}, got {len(actual_shape)}.")
            return False

        for i, (expected_dim, actual_dim) in enumerate(zip(expected_shape, actual_shape)):
            if expected_dim is not None and expected_dim != actual_dim:
                print(f"Error: {name} has wrong dimension {i}. Expected {expected_dim}, got {actual_dim}. Full shape: {actual_shape}")
                return False
        print(f"{name} shape {actual_shape} matches expected {expected_shape}.")
        return True

    @staticmethod
    def check_dtype(data, expected_dtype, name="Data"):
        """ Checks if data type matches the expected dtype. """
        actual_dtype = getattr(data, 'dtype', None)
        if actual_dtype is None:
            print(f"Warning: Cannot get dtype for {name}.")
            return True # Or False?

        # Dtype comparison can be tricky (e.g., torch.float32 vs np.float32)
        # Add more robust comparison if needed
        if str(actual_dtype) != str(expected_dtype): # Simple string comparison
             print(f"Error: {name} has wrong dtype. Expected {expected_dtype}, got {actual_dtype}.")
             return False
        print(f"{name} dtype {actual_dtype} matches expected {expected_dtype}.")
        return True

    @staticmethod
    def check_range(data, min_val=None, max_val=None, name="Data"):
        """ Checks if data values fall within the expected range. """
        # Needs framework-specific min/max operations (np.min, torch.min, tf.reduce_min)
        try:
            # Placeholder - replace with actual min/max calculation
            data_min = data.min() # Example: Assumes numpy-like .min()
            data_max = data.max() # Example: Assumes numpy-like .max()

            valid = True
            if min_val is not None and data_min < min_val:
                print(f"Error: {name} minimum value {data_min} is less than expected {min_val}.")
                valid = False
            if max_val is not None and data_max > max_val:
                print(f"Error: {name} maximum value {data_max} is greater than expected {max_val}.")
                valid = False
            if valid:
                 print(f"{name} values are within range [{min_val}, {max_val}] (Actual: [{data_min:.4f}, {data_max:.4f}]).")
            return valid
        except Exception as e:
            print(f"Warning: Could not check range for {name}. Error: {e}")
            return True # Or False?
