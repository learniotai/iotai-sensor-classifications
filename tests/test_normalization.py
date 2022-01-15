"""Test normalizing gesture recording data."""

import os
import numpy as np
from iotai_sensor_classification.recording import read_recordings
from iotai_sensor_classification.normalization import normalize_mean_std_dict
from data.gestures import linear_accelerometer
from iotai_sensor_classification.plot_util import column_histograms, plot_columns, \
    plot_lines, histogram_overlay


SAMPLES_PER_RECORDING = 160


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
        motion_measures = normalized.drop(columns=['time', 'label'])
        plot_lines(motion_measures, name=f"{gesture} normalized measurements",
                   filepath=os.path.join(test_output, f"{gesture}-norm-lines.png"))
        plot_lines(motion_measures, name=f"{gesture} normalized window={SAMPLES_PER_RECORDING}",
                   filepath=os.path.join(test_output, f"{gesture}-norm-lines-window{SAMPLES_PER_RECORDING}.png"),
                   vertical_tick_spacing=SAMPLES_PER_RECORDING)
        histogram_overlay(motion_measures, name=f"{gesture} normalized measurements",
                          filepath=os.path.join(test_output, f"{gesture}-norm-over-hist.png"))
        # https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
        assert np.allclose(normalized.mean(), 0.0)
        assert np.allclose(normalized.std(), 1.0)
