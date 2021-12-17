"""Test training of sensor classification."""

import os

import torch

from data.gestures import linear_accelerometer
from iotai_sensor_classification import dataset
from iotai_sensor_classification.preprocess import check_windows, parse_recording, SAMPLES_PER_RECORDING
from iotai_sensor_classification.plot_util import group_label_bars
from iotai_sensor_classification.recording import read_recordings
from iotai_sensor_classification.trainer import sensor_classification
from iotai_sensor_classification.plot_util import plot_columns, plot_confusion_matrix
import numpy as np


TEST_OUTPUT = os.path.join("test_output", "gestures", "trainer")


def get_accelerometer_dataset():
    """Read gesture recordings for all tests in file."""
    recordings_dir = os.path.dirname(linear_accelerometer.__file__)
    recordings = read_recordings(recordings_dir=recordings_dir)
    window_checked = check_windows(recordings)
    normed_gesture_measures, encoded_labels, label_coder = \
        parse_recording(window_checked, samples_per_recording=SAMPLES_PER_RECORDING, is_one_hot=False)
    return normed_gesture_measures, encoded_labels, label_coder


def get_datasets():
    """Get datasets and barplot datasets."""
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

    os.makedirs(TEST_OUTPUT, exist_ok=True)
    group_label_bars(raw_labels, title="Gesture classification datasets label count",
                     filepath=os.path.join(TEST_OUTPUT, "gesture_classification_dataset.png"))
    return train_X, val_X, test_X, train_y, val_y, test_y, label_coder


def test_train_gesture_classification_linear():
    """Test trainer gesture classification model from sensor data.
    :return:
    """
    train_X, val_X, test_X, train_y, val_y, test_y, label_coder = get_datasets()

    model = sensor_classification.LinearModel(input_dim=train_X.shape[1] * train_X.shape[2],
                                              output_dim=len(np.unique(train_y)))
    val_df = sensor_classification.train_gesture_classification(model, train_X, val_X, train_y, val_y)
    state_path = os.path.join(TEST_OUTPUT, "gesture_classification_linear_state_dict.zip")
    torch.save(model.state_dict(), state_path)
    max_val_acc = val_df['val_acc'].max()
    plot_columns(val_df, name=f"Gesture classification validation linear model, max acc={max_val_acc:.2}",
                 filepath=os.path.join(TEST_OUTPUT, "gesture_classification_val_linear.png"),
                 title_mean=False)
    load_model = sensor_classification.LinearModel(input_dim=train_X.shape[1] * train_X.shape[2],
                                                   output_dim=len(np.unique(train_y)))
    load_model.load_state_dict(torch.load(state_path))
    assert all(load_model.state_dict()['layer3.bias'] == model.state_dict()['layer3.bias'])
    test_accuracy, test_matrix = sensor_classification.test_accuracy(load_model, label_coder, test_X, test_y)
    unique_y = np.unique(train_y)
    unique_y.sort()
    unique_y_labels = label_coder.decode(unique_y)
    plot_confusion_matrix(test_matrix, classes=unique_y_labels,
                          title=f"Gesture classification linear acc={test_accuracy:.2}",
                          output_path=os.path.join(TEST_OUTPUT, "gesture_classification_linear_confusion.png"))


def test_train_gesture_classification_conv():
    """Test trainer gesture classification model from sensor data.
    :return:
    """
    train_X, val_X, test_X, train_y, val_y, test_y, label_coder = get_datasets()

    model = sensor_classification.ConvModel(input_dim=(train_X.shape[1], train_X.shape[2]),
                                            output_dim=len(np.unique(train_y)))
    val_df = sensor_classification.train_gesture_classification(model, train_X, val_X, train_y, val_y)
    state_path = os.path.join(TEST_OUTPUT, "gesture_classification_conv_state_dict.zip")
    torch.save(model.state_dict(), state_path)
    max_val_acc = val_df['val_acc'].max()
    plot_columns(val_df, name=f"Gesture classification validation conv 2D model, max acc={max_val_acc:.2}",
                 filepath=os.path.join(TEST_OUTPUT, "gesture_classification_val_conv.png"),
                 title_mean=False)
    load_model = sensor_classification.ConvModel(input_dim=(train_X.shape[1], train_X.shape[2]),
                                            output_dim=len(np.unique(train_y)))
    load_model.load_state_dict(torch.load(state_path))
    assert all(load_model.state_dict()['fc3.bias'] == model.state_dict()['fc3.bias'])
    test_accuracy, test_matrix = sensor_classification.test_accuracy(load_model, label_coder, test_X, test_y)
    unique_y = np.unique(train_y)
    unique_y.sort()
    unique_y_labels = label_coder.decode(unique_y)
    plot_confusion_matrix(test_matrix, classes=unique_y_labels,
                          title=f"Gesture classification conv acc={test_accuracy:.2}",
                          output_path=os.path.join(TEST_OUTPUT, "gesture_classification_conv_confusion.png"))
