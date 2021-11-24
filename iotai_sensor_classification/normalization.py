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


def normalize_mean_std_dict(recordings: dict):
    """Normalize data frames in dictionary"""
    names = recordings.keys()
    normals = {}
    for name_ in names:
        raw = recordings[name_]
        normalized = normalize_mean_std(raw)
        normals[name_] = normalized
    return normals


def normalize_mean_std(x: np.array):
    """Normalize to mean 0 and standard deviation 1.
    :param x: 2D matrix
    :return: Normalized 2D matrix"""
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    return x
