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
    label_coder_one_hot = encode.LabelCoder(is_one_hot=True)
    one_hot_gestures = label_coder_one_hot.encode_labels(gesture_names)
    decoded_gestures = label_coder_one_hot.decode(one_hot_gestures)
    assert all(decoded_gestures == gesture_names)

    label_coder = encode.LabelCoder(is_one_hot=False)
    gesture_codes = label_coder.encode_labels(gesture_names)
    decoded = label_coder.decode(gesture_codes)
    assert all(decoded == gesture_names)
