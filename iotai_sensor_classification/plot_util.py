import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from .recording import filter_columns
from typing import Dict
import seaborn
import itertools


def column_histograms(data: pd.core.frame.DataFrame, name: str, bins=50, filepath=None):
    """Make histograms of numeric data columns"""
    data = filter_columns(data=data, keep_dtypes=[np.float])
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
        if side_b > 1:
            axe = a[x][y]
        else:
            axe = a[x]
        axe.hist(col_data, bins=bins)
        axe.set_title("{col} {mean:.2f}+-{std:.2}".format(col=col, mean=col_data.mean(), std=col_data.std()))
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


def plot_columns(data: pd.core.frame.DataFrame, name: str, filepath=None, vertical_ticks=None, title_mean=True):
    """Make plots of data columns each in own subplot."""
    data = filter_columns(data, keep_dtypes=[np.float])
    columns = data.columns
    n_col = len(columns)
    side_a = side_b = int(np.ceil(np.sqrt(n_col)))
    if n_col <= side_a * (side_b-1):
        side_b = side_b - 1
    f, a = plt.subplots(side_a, side_b)
    f.suptitle(f"{name}")
    x = 0
    y = 0
    for col in columns:
        col_data = data[col]
        if side_a > 1 and side_b > 1:
            axe = a[x][y]
        elif side_a > 1:
            axe = a[x]
        else:
            axe = a
        axe.plot(col_data)
        title_add_mean = ""
        if title_mean:
            title_add_mean = " {mean:.2f}+-{std:.2}".format(col=col, mean=col_data.mean(), std=col_data.std())
        axe.set_title(f"{col}{title_add_mean}")
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


def plot_lines(data: pd.core.frame.DataFrame, name: str, filepath=None, vertical_tick_spacing=None):
    """Make multiline plot of data columns."""
    data = filter_columns(data, keep_dtypes=[np.float])
    if vertical_tick_spacing:
        index_min = data.index.min()
        index_max = data.index.max()
    columns = data.columns
    fig = plt.figure(figsize=(12, 6))
    for col in columns:
        col_data = data[col]
        plt.plot(col_data, label="{col} {mean:.2f}+-{std:.2}".format(col=col, mean=col_data.mean(), std=col_data.std()))
        if vertical_tick_spacing:
            for tic in range(index_min, index_max+1, vertical_tick_spacing):
                plt.axvline(x=tic, color='b', linestyle='dashed', alpha=0.1)
    plt.legend()
    plt.title(name)
    if filepath is not None:
        # plt.switch_backend("Agg")
        plt.savefig(filepath)
        plt.close()
    else:
        # plt.switch_backend("qt5agg")
        plt.show()


def histogram_overlay(data: pd.core.frame.DataFrame, name: str, filepath=None, bins=50, alpha=0.5):
    """Make multihistogram of data columns."""
    data = filter_columns(data, keep_dtypes=[np.float])
    columns = data.columns
    plt.figure(figsize=(12, 6))
    for col in columns:
        col_data = data[col]
        plt.hist(col_data, label="{col} {mean:.2f}+-{std:.2}".format(col=col, mean=col_data.mean(), std=col_data.std()),
                 bins=bins, alpha=alpha)
    plt.legend()
    plt.title(name)
    if filepath is not None:
        # plt.switch_backend("Agg")
        plt.savefig(filepath)
        plt.close()
    else:
        # plt.switch_backend("qt5agg")
        plt.show()


def group_label_bars(group_labels: Dict, title: str, filepath: str=None):
    """Bar plot of group raw labels.
    :param groups_labels: dictionary of group name: [array of label names, ...]
    :param title: plot title
    :param filepath: filepath.png to save barplot to."""
    label_counts = {}
    group_label_names = group_labels.keys()
    col_names = []
    for group_name in group_label_names:
        series_counts = pd.Series(group_labels[group_name]).value_counts()
        label_counts[group_name] = series_counts
        col_names += list(series_counts.keys())
    unique_col_names = np.unique(col_names)
    plot_counts = {}
    for col_label in unique_col_names:
        counts = []
        for lc in label_counts.keys():
            counts.append(label_counts[lc][col_label])
        plot_counts[col_label] = counts
    label_count_frame = pd.DataFrame.from_dict(plot_counts)
    label_count_frame.index = group_label_names
    label_count_frame['dataset'] = group_label_names
    melted = pd.melt(label_count_frame, id_vars="dataset", var_name="gesture", value_name="count")
    sns_plot = seaborn.catplot(data=melted, x='dataset', hue='gesture', y='count', kind='bar')
    sns_plot.set(title=title)
    if filepath is not None:
        sns_plot.savefig(filepath)
    else:
        plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, output_path=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    # Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

    """
    plt.figure(figsize=(8, 8))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if output_path is not None:
        plt.savefig(output_path)
