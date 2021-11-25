"""Encode raw data into forms for AI model training."""


import numpy as np
from typing import List


def create_one_hot_encoding_lookup(categories: List[str]):
    """One hot encode list of category names.
    :return: dictionary lookup of category encodings."""
    categories.sort()
    n_categories = len(categories)
    encodings = np.eye(n_categories)
    lookup = {}
    for category, code in zip(categories, encodings):
        lookup[category] = code
    return lookup
