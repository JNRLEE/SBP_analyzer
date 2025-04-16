from .base_callback import Callback
# from torch.utils.tensorboard import SummaryWriter # For PyTorch
# import tensorflow as tf # For TensorFlow
import os

class TensorBoardCallback(Callback):
    """
    Callback that streams epoch results to TensorBoard.
    """
    def __init__(self, log_dir='./logs', comment=''):
        super().__init__()
        self.log_dir = log_dir
        self.comment = comment
        self.writer = None
        # TODO: Initialize SummaryWriter (PyTorch) or tf.summary.create_file_writer (TF)

    def on_train_begin(self, logs=None):
        # Example for PyTorch:
        # from torch.utils.tensorboard import SummaryWriter
        # self.writer = SummaryWriter(log_dir=self.log_dir, comment=self.comment)
        print(f"TensorBoard logging to: {self.log_dir}")
        # Example for TensorFlow:
        # self.writer = tf.summary.create_file_writer(self.log_dir)
        pass # Initialize the writer here

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.writer:
            for name, value in logs.items():
                if isinstance(value, (int, float)):
                    # self.writer.add_scalar(f'Epoch/{name}', value, epoch) # PyTorch
                    # with self.writer.as_default(): # TF
                    #    tf.summary.scalar(f'Epoch/{name}', value, step=epoch) # TF
                    pass # Add scalar logging
            # self.writer.flush() # PyTorch
            pass

    def on_batch_end(self, epoch, batch_idx, logs=None):
        logs = logs or {}
        if self.writer:
            # Optionally log batch-level metrics (can be very verbose)
            # global_step = epoch * (self.trainer.num_batches_per_epoch or 1) + batch_idx
            # for name, value in logs.items():
            #     if name == 'loss' and isinstance(value, (int, float)):
            #         # self.writer.add_scalar('Batch/loss', value, global_step) # PyTorch
            #         # with self.writer.as_default(): # TF
            #         #     tf.summary.scalar('Batch/loss', value, step=global_step) # TF
            #         pass # Add batch loss logging
            pass

    def on_train_end(self, logs=None):
        if self.writer:
            # self.writer.close() # PyTorch
            pass # Close the writer

    # --- Methods to be called by other callbacks or modules ---
    def log_scalar(self, tag, scalar_value, global_step):
        if self.writer:
            # self.writer.add_scalar(tag, scalar_value, global_step) # PyTorch
            # with self.writer.as_default(): # TF
            #    tf.summary.scalar(tag, scalar_value, step=global_step) # TF
            pass

    def log_histogram(self, tag, values, global_step):
         if self.writer:
            # self.writer.add_histogram(tag, values, global_step) # PyTorch
            # with self.writer.as_default(): # TF
            #    tf.summary.histogram(tag, values, step=global_step) # TF
            pass

    def log_image(self, tag, img_tensor, global_step, dataformats='CHW'):
         if self.writer:
            # self.writer.add_image(tag, img_tensor, global_step, dataformats=dataformats) # PyTorch
            # with self.writer.as_default(): # TF
            #    tf.summary.image(tag, img_tensor, step=global_step) # TF (ensure correct format)
            pass

    def log_audio(self, tag, snd_data, global_step, sample_rate):
         if self.writer:
            # self.writer.add_audio(tag, snd_data, global_step, sample_rate=sample_rate) # PyTorch
            # with self.writer.as_default(): # TF
            #    tf.summary.audio(tag, snd_data, sample_rate, step=global_step) # TF
            pass
