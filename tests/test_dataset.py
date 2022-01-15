"""Test datasets."""

import os
from iotai_sensor_classification import dataset
from iotai_sensor_classification.preprocess import check_windows, parse_recording
from iotai_sensor_classification.recording import read_recordings
from iotai_sensor_classification.plot_util import group_label_bars
from data.gestures import linear_accelerometer


def get_accelerometer_dataset(samples_per_recording=160):
    """Read gesture recordings for all tests in file."""
    recordings_dir = os.path.dirname(linear_accelerometer.__file__)
    recordings = read_recordings(recordings_dir=recordings_dir)
    window_checked = check_windows(recordings)
    normed_gesture_measures, encoded_labels, label_coder = \
        parse_recording(window_checked, samples_per_recording=samples_per_recording)
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
    train_y_labels = label_coder.decode(train_y)
    val_y_labels = label_coder.decode(val_y)
    test_y_labels = label_coder.decode(test_y)
    assert len(train_y_labels) == len(train_y)
    assert len(val_y_labels) == len(val_y)
    assert len(test_y_labels) == len(test_y)

    raw_labels = {}
    raw_labels['train'] = train_y_labels
    raw_labels['validation'] = val_y_labels
    raw_labels['test'] = test_y_labels

    test_output = os.path.join("test_output", "gestures", "dataset")
    os.makedirs(test_output, exist_ok=True)
    group_label_bars(raw_labels, title="Datasets label count", filepath=os.path.join(test_output, "dataset_splits.png"))
