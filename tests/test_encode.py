"""Test encoding data for AI model training."""


from iotai_sensor_classification import encode
from data.gestures import linear_accelerometer
import os
from iotai_sensor_classification.recording import get_recording_names
import numpy as np
import pytest


@pytest.fixture
def gesture_names():
    recordings_dir = os.path.dirname(linear_accelerometer.__file__)
    return get_recording_names(recordings_dir, ext=".csv.gz")


def test_one_hot_encode_lookup(gesture_names):
    """Test one hot encode list of categories for classification."""
    encoding_lookup = encode.create_one_hot_encoding_lookup(gesture_names)
    assert "circle" in encoding_lookup
    assert np.allclose(encoding_lookup["circle"], np.array([1., 0., 0., 0., 0.]))


def test_label_encoder(gesture_names):
    """Test label encoder."""
    label_coder = encode.LabelCoder()
    one_hot_gestures = label_coder.encode_labels(gesture_names)
    decoded_gestures = label_coder.decode_one_hots(one_hot_gestures)
    assert all(decoded_gestures == gesture_names)
