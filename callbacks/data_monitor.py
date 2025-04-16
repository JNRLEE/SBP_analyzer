from .base_callback import Callback
# from ..metrics import calculate_distribution_stats # Example import
# from ..visualization import plot_histogram # Example import
import numpy as np # Or torch/tensorflow depending on data format

class DataMonitorCallback(Callback):
    """
    Callback to monitor input data distribution during training.
    """
    def __init__(self, frequency='epoch', log_histograms=True, log_stats=True):
        """
        Args:
            frequency (str): 'epoch' or 'batch' - how often to monitor.
                             'batch' can be very slow. Recommend using with a step interval.
            log_histograms (bool): Whether to log histograms to TensorBoard.
            log_stats (bool): Whether to log distribution statistics (mean, std, etc.) to TensorBoard.
        """
        super().__init__()
        if frequency not in ['epoch', 'batch']:
            raise ValueError("frequency must be 'epoch' or 'batch'")
        self.frequency = frequency
        self.log_histograms = log_histograms
        self.log_stats = log_stats
        self._tb_callback = None # To store reference to TensorBoardCallback

    def on_train_begin(self, logs=None):
        # Find the TensorBoard callback instance if it exists in the trainer
        if self.trainer and hasattr(self.trainer, 'callbacks'):
            for cb in self.trainer.callbacks:
                # A bit fragile, maybe improve discovery later
                if cb.__class__.__name__ == 'TensorBoardCallback':
                    self._tb_callback = cb
                    break
        if (self.log_histograms or self.log_stats) and not self._tb_callback:
            print("Warning: DataMonitorCallback requires TensorBoardCallback to log results, but it wasn't found.")

    def _monitor_data(self, epoch, batch_idx, batch_data):
        if not self._tb_callback:
            return # Cannot log without TensorBoard

        # --- This part needs customization based on your data structure ---
        # Assuming batch_data is a tensor or numpy array, or a dict/list containing them
        # Example: Assuming batch_data is the input tensor
        data_to_analyze = batch_data
        # If batch_data is a dict: data_to_analyze = batch_data['input_features']
        # -----------------------------------------------------------------

        if data_to_analyze is None:
            return

        # Convert to numpy for analysis (if needed)
        if hasattr(data_to_analyze, 'cpu') and hasattr(data_to_analyze, 'numpy'): # PyTorch tensor
             data_np = data_to_analyze.detach().cpu().numpy().flatten()
        elif hasattr(data_to_analyze, 'numpy'): # TensorFlow tensor or already numpy
             data_np = data_to_analyze.numpy().flatten()
        elif isinstance(data_to_analyze, np.ndarray):
             data_np = data_to_analyze.flatten()
        else:
             print(f"Warning: DataMonitorCallback doesn't know how to handle data type: {type(data_to_analyze)}")
             return

        global_step = epoch # Default to epoch step, refine if batch frequency is used

        # Log statistics
        if self.log_stats:
            mean = np.mean(data_np)
            std = np.std(data_np)
            min_val = np.min(data_np)
            max_val = np.max(data_np)
            self._tb_callback.log_scalar('Data/Input_Mean', mean, global_step)
            self._tb_callback.log_scalar('Data/Input_Std', std, global_step)
            self._tb_callback.log_scalar('Data/Input_Min', min_val, global_step)
            self._tb_callback.log_scalar('Data/Input_Max', max_val, global_step)
            # Add more stats from metrics module if needed

        # Log histogram
        if self.log_histograms:
            self._tb_callback.log_histogram('Data/Input_Distribution', data_np, global_step)
            # Could also use visualization module to plot and log as image

    def on_epoch_end(self, epoch, logs=None):
        if self.frequency == 'epoch' and logs and 'last_batch_data' in logs:
            # Assuming trainer passes the last batch data in logs
            self._monitor_data(epoch, -1, logs['last_batch_data'])

    def on_batch_end(self, epoch, batch_idx, logs=None):
         # Be careful with batch frequency - can slow down training significantly
         # Consider adding a step condition, e.g., if batch_idx % log_every_n_batches == 0:
        if self.frequency == 'batch' and logs and 'batch_data' in logs:
             self._monitor_data(epoch, batch_idx, logs['batch_data'])

