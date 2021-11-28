"""Encode raw data into forms for AI model training."""


import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class LabelCoder:
    """Label encoder and decoder."""
    def __init__(self):
        """Initialize encoder/decoder of labels."""
        self._label_encoder = LabelEncoder()

    def encode_labels(self, labels):
        """One hot encode labels creating encoder/decoder.

        :return:
        """
        int_labels = self._label_encoder.fit_transform(labels)
        int_labels_transposed = int_labels.reshape(len(int_labels), 1)
        one_hot_encoder = OneHotEncoder(sparse=False)
        one_hot_encoded = one_hot_encoder.fit_transform(int_labels_transposed)
        return one_hot_encoded

    def decode_one_hots(self, label_codes):
        """Decode one hot encode into labels."""
        labels = self._label_encoder.inverse_transform([np.argmax(one) for one in label_codes])
        return labels
