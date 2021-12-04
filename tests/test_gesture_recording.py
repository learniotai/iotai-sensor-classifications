"""Test loading gesture recordings and visualizing data."""
from iotai_sensor_classification.recording import read_recordings

from data.gestures import linear_accelerometer
import os
from iotai_sensor_classification.plot_util import column_histograms, plot_columns, plot_lines

import pytest


@pytest.fixture
def gesture_recordings():
    """Read gesture recordings for all tests in file."""
    recordings_dir = os.path.dirname(linear_accelerometer.__file__)
    recordings = read_recordings(recordings_dir=recordings_dir)
    return recordings


def test_read_gestures(gesture_recordings):
    """Test reading gesture recordings."""
    gestures = gesture_recordings.keys()
    assert "shake" in gestures
    assert "rock" in gestures
    test_output = os.path.join("test_output", "gestures", "raw")
    os.makedirs(test_output, exist_ok=True)
    for gesture in gestures:
        gesture_data = gesture_recordings[gesture]
        column_histograms(gesture_data, name=f"{gesture} gesture",
                          filepath=os.path.join(test_output, f"{gesture}-histograms.png"))
        plot_columns(gesture_data, name=f"{gesture} gesture",
                     filepath=os.path.join(test_output, f"{gesture}-plots.png"))
        motion_measures = gesture_data.drop(columns=['time', 'label'])
        plot_lines(motion_measures, name=f"{gesture} gesture measurements",
                     filepath=os.path.join(test_output, f"{gesture}-lines.png"))
