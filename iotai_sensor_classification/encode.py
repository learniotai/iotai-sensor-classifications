"""Encode raw data into forms for AI model training."""


import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class LabelCoder:
    """Label encoder and decoder."""
    def __init__(self, is_one_hot=True):
        """Initialize encoder/decoder of labels."""
        self._label_encoder = LabelEncoder()
        self._is_one_hot=is_one_hot

    def encode_labels(self, labels):
        """Encode labels creating encoder/decoder.

        :return:
        """
        int_labels = self._label_encoder.fit_transform(labels)
        if self._is_one_hot:
            int_labels_transposed = int_labels.reshape(len(int_labels), 1)
            one_hot_encoder = OneHotEncoder(sparse=False)
            one_hot_encoded = one_hot_encoder.fit_transform(int_labels_transposed)
            return one_hot_encoded
        else:
            return int_labels

    def decode(self, label_codes):
        """Decode encoded labels into text labels."""
        if self._is_one_hot:
            return self._label_encoder.inverse_transform([np.argmax(one) for one in label_codes])
        else:
            return self._label_encoder.inverse_transform(label_codes)
