"""Test encoding data for AI model training."""


from iotai_sensor_classification import encode
from data.gestures import linear_accelerometer
import os
import fnmatch
from iotai_sensor_classification.recording import strip_extensions
import numpy as np


def test_one_hot_encode_lookup():
    """Test one hot encode list of categories for classification."""
    recordings_dir = os.path.dirname(linear_accelerometer.__file__)
    csv_recordings = fnmatch.filter(os.listdir(recordings_dir), "*.csv.gz")
    gesture_names = [strip_extensions(recording) for recording in csv_recordings]
    encoding_lookup = encode.create_one_hot_encoding_lookup(gesture_names)
    assert "circle" in encoding_lookup
    assert np.allclose(encoding_lookup["circle"], np.array([1., 0., 0., 0., 0.]))

