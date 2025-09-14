# utils.py
import matplotlib.pyplot as plt
import os

def save_plot(path):
    """Save current matplotlib figure to path, creating directories if needed."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(path)
