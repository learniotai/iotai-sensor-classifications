import matplotlib.pylab as plt
import numpy as np
import pandas as pd


def column_histograms(data: pd.core.frame.DataFrame, name: str, bins=50, filepath=None):
    """Make histograms of data columns"""
    columns = data.columns
    n_col = len(columns)
    side_a = side_b = int(np.ceil(np.sqrt(n_col)))
    if n_col <= side_a * (side_b - 1):
        side_b = side_b - 1
    f, a = plt.subplots(side_a, side_b)
    f.suptitle(f"{name} histograms")
    x = 0
    y = 0
    for col in columns:
        col_data = data[col]
        a[x][y].hist(col_data, bins=bins)
        a[x][y].set_title("{col} {mean:.2f}+-{std:.2}".format(col=col, mean=col_data.mean(), std=col_data.std()))
        if x < side_a-1:
            x += 1
        else:
            y += 1
            x = 0
    plt.tight_layout()
    if filepath is not None:
        # plt.switch_backend("Agg")
        plt.savefig(filepath)
        plt.close()
    else:
        # plt.switch_backend("qt5agg")
        plt.show()


def plot_columns(data: pd.core.frame.DataFrame, name: str, filepath=None):
    """Make plots of data columns"""
    columns = data.columns
    n_col = len(columns)
    side_a = side_b = int(np.ceil(np.sqrt(n_col)))
    if n_col <= side_a * (side_b-1):
        side_b = side_b - 1
    f, a = plt.subplots(side_a, side_b)
    f.suptitle(f"{name} plots")
    x = 0
    y = 0
    for col in columns:
        col_data = data[col]
        a[x][y].plot(col_data)
        a[x][y].set_title("{col} {mean:.2f}+-{std:.2}".format(col=col, mean=col_data.mean(), std=col_data.std()))
        if x < side_a-1:
            x += 1
        else:
            y += 1
            x = 0
    plt.tight_layout()
    if filepath is not None:
        # plt.switch_backend("Agg")
        plt.savefig(filepath)
        plt.close()
    else:
        # plt.switch_backend("qt5agg")
        plt.show()


