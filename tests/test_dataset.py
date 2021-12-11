"""Test datasets."""

import os
from iotai_sensor_classification import dataset
from iotai_sensor_classification.preprocess import check_windows, parse_recording, SAMPLES_PER_RECORDING
from iotai_sensor_classification.recording import read_recordings
from iotai_sensor_classification.plot_util import bar_plot
from data.gestures import linear_accelerometer
import pandas as pd
import numpy as np


def get_accelerometer_dataset():
    """Read gesture recordings for all tests in file."""
    recordings_dir = os.path.dirname(linear_accelerometer.__file__)
    recordings = read_recordings(recordings_dir=recordings_dir)
    window_checked = check_windows(recordings)
    normed_gesture_measures, encoded_labels, label_coder = \
        parse_recording(window_checked, samples_per_recording=SAMPLES_PER_RECORDING)
    return normed_gesture_measures, encoded_labels, label_coder


def test_split_data():
    """Test splitting data"""
    normed_gesture_measures, encoded_labels, label_coder = get_accelerometer_dataset()
    train_X, val_X, test_X, train_y, val_y, test_y = \
        dataset.split_dataset(normed_gesture_measures, encoded_labels, val_size=dataset.VALIDATION_SPLIT,
                              test_size=dataset.TEST_SPLIT)
    assert len(train_X) == len(train_y)
    assert len(val_X) == len(val_y)
    assert len(test_X) == len(test_y)
    train_y_labels = label_coder.decode_one_hots(train_y)
    val_y_labels = label_coder.decode_one_hots(val_y)
    test_y_labels = label_coder.decode_one_hots(test_y)
    assert len(train_y_labels) == len(train_y)
    assert len(val_y_labels) == len(val_y)
    assert len(test_y_labels) == len(test_y)
    unique_labels = list(np.unique((list(train_y_labels) + list(val_y_labels) + list(test_y_labels))))

    label_data = {}
    train_counts = pd.Series(train_y_labels).value_counts()
    validation_counts = pd.Series(val_y_labels).value_counts()
    test_counts = pd.Series(test_y_labels).value_counts()
    for label in unique_labels:
        label_data[label] = [train_counts[label], validation_counts[label], test_counts[label]]
    label_count_frame = pd.DataFrame.from_dict(label_data)
    label_count_frame.index = ['train', 'validation', 'test']
    label_count_frame['class'] = ['train', 'validation', 'test']
    melted = pd.melt(label_count_frame, id_vars="class", var_name="gesture")
    import seaborn
    seaborn.factorplot(data=melted, x='class', hue='gesture', y='value', kind='bar')
    test_output = os.path.join("test_output", "gestures", "dataset")
    os.makedirs(test_output, exist_ok=True)
    # bar_plot(label_data, name="Label split data", filepath=os.path.join(test_output, "label_splits.png"))
    bar_plot(label_data, name="Label split data")
