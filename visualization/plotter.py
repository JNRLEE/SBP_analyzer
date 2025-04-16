# import matplotlib.pyplot as plt
# import numpy as np

def plot_histogram(data, title="Histogram", bins=50):
    """
    Generates a histogram plot using matplotlib.

    Args:
        data (array-like): Data to plot.
        title (str): Plot title.
        bins (int): Number of bins for the histogram.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
                                  Can be converted to image for TensorBoard.
    """
    # Ensure matplotlib is installed and imported
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib not found. Please install it to use plotting functions.")
        return None

    fig, ax = plt.subplots()
    ax.hist(np.asarray(data).flatten(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    # Don't call plt.show() here if used in callbacks
    return fig

def plot_waveform(waveform, sample_rate, title="Waveform"):
    """
    Generates a waveform plot.

    Args:
        waveform (array-like): 1D audio waveform.
        sample_rate (int): Sample rate of the audio.
        title (str): Plot title.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib not found. Please install it to use plotting functions.")
        return None

    waveform = np.asarray(waveform).flatten()
    time_axis = np.arange(len(waveform)) / sample_rate

    fig, ax = plt.subplots()
    ax.plot(time_axis, waveform)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig

# Add functions to convert figures to images (e.g., PIL Image or numpy array)
# for TensorBoard logging if needed.
