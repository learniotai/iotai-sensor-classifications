"""Test data preprocessing."""

import os
import pytest
from data.gestures import linear_accelerometer
from iotai_sensor_classification.recording import  read_recordings
from iotai_sensor_classification.preprocess import parse_recording


@pytest.fixture
def gesture_recordings():
    """Read gesture recordings for all tests in file."""
    recordings_dir = os.path.dirname(linear_accelerometer.__file__)
    recordings = read_recordings(recordings_dir=recordings_dir)
    return recordings


def test_parse_gestures(gesture_recordings):
    """Test parsing gesture data creating labels into a dataset for training and or testing models."""
    parsed_gestures, label_coder = parse_recording(gesture_recordings)
    assert all(label_coder.decode_one_hots(parsed_gestures["label_code"]) == parsed_gestures["label"])
