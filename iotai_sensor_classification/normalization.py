"""Data normalization function.

Data for training machine learning models should be in similar ranges. Normalization converts data into
similar ranges.

References for further reading:
https://towardsdatascience.com/data-normalization-in-machine-learning-395fdec69d02
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py

One hot encode
ONE_HOT_ENCODED_GESTURES = np.eye(NUM_GESTURES)
"""

import numpy as np
from .recording import filter_columns
import pandas as pd


def normalize_mean_std_dict(recordings: dict):
    """Normalize numeric data from frames in dictionary.

    Keep non numeric data as is."""
    names = recordings.keys()
    normals = {}
    for name_ in names:
        raw = recordings[name_]
        raw_numbers = filter_columns(raw, keep_dtypes=[np.float])
        normalized = normalize_mean_std(raw_numbers)
        lost_columns = list(set(raw.columns) - set(raw_numbers.columns))
        lost_raw = raw[lost_columns]
        normalized_restored = pd.concat([normalized, lost_raw], axis=1)
        normals[name_] = normalized_restored
    return normals


def normalize_mean_std(x: np.array):
    """Normalize to mean 0 and standard deviation 1.
    :param x: 2D matrix
    :return: Normalized 2D matrix"""
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    return x
