"""Test normalizing gesture recording data."""

import os
import numpy as np
from iotai_sensor_classification.recording import read_recordings
from iotai_sensor_classification.normalization import normalize_mean_std_dict
from data.gestures import linear_accelerometer
from iotai_sensor_classification.plot_util import column_histograms, plot_columns


def test_normalize_gesture_data():
    recordings_dir = os.path.dirname(linear_accelerometer.__file__)
    raw_gestures = read_recordings(recordings_dir=recordings_dir)
    normalized_gestures = normalize_mean_std_dict(raw_gestures)
    test_output = os.path.join("test_output", "gestures", "normalized")
    os.makedirs(test_output, exist_ok=True)
    for gesture in normalized_gestures.keys():
        normalized = normalized_gestures[gesture]
        column_histograms(normalized, name=f"{gesture} gesture normalized",
                          filepath=os.path.join(test_output, f"{gesture}-norm-histograms.png"))
        plot_columns(normalized, name=f"{gesture} gesture normalized",
                     filepath=os.path.join(test_output, f"{gesture}-norm-plots.png"))
        # https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
        assert np.allclose(normalized.mean(), 0.0)
        assert np.allclose(normalized.std(), 1.0)
