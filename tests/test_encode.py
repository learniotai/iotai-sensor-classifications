"""Test encoding data for AI model training."""


import os
import pytest
from iotai_sensor_classification import encode
from data.gestures import linear_accelerometer
from iotai_sensor_classification.recording import get_recording_names


@pytest.fixture
def gesture_names():
    recordings_dir = os.path.dirname(linear_accelerometer.__file__)
    return get_recording_names(recordings_dir, ext=".csv.gz")


def test_label_encoder(gesture_names):
    """Test label encoder."""
    label_coder = encode.LabelCoder()
    one_hot_gestures = label_coder.encode_labels(gesture_names)
    decoded_gestures = label_coder.decode_one_hots(one_hot_gestures)
    assert all(decoded_gestures == gesture_names)
