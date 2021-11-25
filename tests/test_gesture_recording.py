"""Test loading gesture recordings and visualizing data."""
from iotai_sensor_classification.recording import read_recordings

from data.gestures import linear_accelerometer
import os
from iotai_sensor_classification.plot_util import column_histograms, plot_columns


def test_parse_motions():
    recordings_dir = os.path.dirname(linear_accelerometer.__file__)
    gesture_recording = read_recordings(recordings_dir=recordings_dir)
    gestures = gesture_recording.keys()
    assert "shake" in gestures
    assert "rock" in gestures
    test_output = os.path.join("test_output", "gestures", "raw")
    os.makedirs(test_output, exist_ok=True)
    for gesture in gestures:
        gesture_data = gesture_recording[gesture]
        column_histograms(gesture_data, name=f"{gesture} gesture",
                          filepath=os.path.join(test_output, f"{gesture}-histograms.png"))
        plot_columns(gesture_data, name=f"{gesture} gesture",
                     filepath=os.path.join(test_output, f"{gesture}-plots.png"))
