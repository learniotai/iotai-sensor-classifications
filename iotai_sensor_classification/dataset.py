"""Dataset functionality for providing training, validation and testing data."""
import os

from sklearn.model_selection import train_test_split

from iotai_sensor_classification.plot_util import group_label_bars
from iotai_sensor_classification.preprocess import check_windows, parse_recording
from iotai_sensor_classification.recording import read_recordings

TEST_SPLIT = 0.15
VALIDATION_SPLIT = 0.15


def split_dataset(X, y, val_size, test_size, shuffle=True):
    """Split data into 3 parts: train, validate, test.

    Test size is determined by the remainder of the other two splits.
    :param X: input data
    :param y: output data
    :param val_size: validation fraction
    :param test_size: test fraction
    :param shuffle: randomize rows of data, default True
    :return: train_X, val_X, test_X, train_y, val_y, test_y
    """
    train_val_X, test_X, train_val_y, test_y = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    train_val_size = 1.0 - test_size
    # adjust validation proportion of train_val data up to match test_size for second split
    val_of_tv_size = val_size/train_val_size
    train_X, val_X, train_y, val_y = \
        train_test_split(train_val_X, train_val_y, test_size=val_of_tv_size, shuffle=shuffle)

    return train_X, val_X, test_X, train_y, val_y, test_y


def prepare_datasets(plot_path, title, get_dataset, output_dir):
    """Get datasets and barplot datasets."""
    normed_gesture_measures, encoded_labels, label_coder = get_dataset()
    train_X, val_X, test_X, train_y, val_y, test_y = \
        split_dataset(normed_gesture_measures, encoded_labels, val_size=VALIDATION_SPLIT,
                              test_size=TEST_SPLIT)
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

    os.makedirs(output_dir, exist_ok=True)
    group_label_bars(raw_labels, title=title,
                     filepath=plot_path)
    return train_X, val_X, test_X, train_y, val_y, test_y, label_coder


def read_dataset(recordings_dir, samples_per_recording=160):
    """Read gesture recordings for all tests in file."""

    recordings = read_recordings(recordings_dir=recordings_dir)
    window_checked = check_windows(recordings)
    normed_gesture_measures, encoded_labels, label_coder = \
        parse_recording(window_checked, samples_per_recording=samples_per_recording, is_one_hot=False)
    return normed_gesture_measures, encoded_labels, label_coder
