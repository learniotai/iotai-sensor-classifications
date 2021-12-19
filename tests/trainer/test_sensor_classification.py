"""Test training of sensor classification."""

import os

import torch

from iotai_sensor_classification.model_handler import ModelCall
from iotai_sensor_classification.evaluation import evaluate_prediction, evaluate_model
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


def get_datasets(plot_path, title):
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
    group_label_bars(raw_labels, title=title,
                     filepath=plot_path)
    return train_X, val_X, test_X, train_y, val_y, test_y, label_coder


def test_train_gesture_classification_linear():
    """Test trainer gesture classification model from sensor data.
    :return:
    """
    X_train, X_val, X_test, y_train, y_val, y_test, label_coder = \
        get_datasets(os.path.join(TEST_OUTPUT, "gesture_classification_dataset_linear.png"),
                     "Gesture classification datasets label count linear")

    model = sensor_classification.LinearModel(input_dim=X_train.shape[1] * X_train.shape[2],
                                              output_dim=len(np.unique(y_train)))
    val_df = sensor_classification.train_gesture_classification(model, X_train, X_val, y_train, y_val)
    state_path = os.path.join(TEST_OUTPUT, "gesture_classification_linear_state_dict.zip")
    torch.save(model.state_dict(), state_path)
    max_val_acc = val_df['val_acc'].max()
    plot_columns(val_df, name=f"Gesture classification validation linear model, max acc={max_val_acc:.2}",
                 filepath=os.path.join(TEST_OUTPUT, "gesture_classification_val_linear.png"),
                 title_mean=False)
    load_model = sensor_classification.LinearModel(input_dim=X_train.shape[1] * X_train.shape[2],
                                                   output_dim=len(np.unique(y_train)))
    load_model.load_state_dict(torch.load(state_path))
    assert all(load_model.state_dict()['layer3.bias'] == model.state_dict()['layer3.bias'])
    model_call = ModelCall(model=load_model, decode=label_coder.decode)
    test_accuracy_, test_matrix = evaluate_prediction(model_call, label_coder.decode, X_test, y_test)
    unique_y = np.unique(y_train)
    unique_y.sort()
    unique_y_labels = label_coder.decode(unique_y)
    plot_confusion_matrix(test_matrix, classes=unique_y_labels,
                          title=f"Gesture classification linear acc={test_accuracy_:.2}",
                          output_path=os.path.join(TEST_OUTPUT, "gesture_classification_linear_confusion.png"))


def test_train_gesture_classification_conv():
    """Test trainer gesture classification model from sensor data.
    :return:
    """
    X_train, X_val, X_test, y_train, y_val, y_test, label_coder = \
        get_datasets(os.path.join(TEST_OUTPUT, "gesture_classification_dataset_conv.png"),
                     "Gesture classification datasets label count conv")

    model = sensor_classification.ConvModel(input_dim=(X_train.shape[1], X_train.shape[2]),
                                            output_dim=len(np.unique(y_train)))
    val_df = sensor_classification.train_gesture_classification(model, X_train, X_val, y_train, y_val)
    state_path = os.path.join(TEST_OUTPUT, "gesture_classification_conv_state_dict.zip")
    torch.save(model.state_dict(), state_path)
    max_val_acc = val_df['val_acc'].max()
    plot_columns(val_df, name=f"Gesture classification validation conv 2D model, max acc={max_val_acc:.2}",
                 filepath=os.path.join(TEST_OUTPUT, "gesture_classification_val_conv.png"),
                 title_mean=False)
    load_model = sensor_classification.ConvModel(input_dim=(X_train.shape[1], X_train.shape[2]),
                                            output_dim=len(np.unique(y_train)))
    load_model.load_state_dict(torch.load(state_path))
    assert all(load_model.state_dict()['fc3.bias'] == model.state_dict()['fc3.bias'])
    model_call = ModelCall(model=load_model, decode=label_coder.decode)
    test_accuracy_, test_matrix = evaluate_prediction(model_call, label_coder.decode, X_test, y_test)
    unique_y = np.unique(y_train)
    unique_y.sort()
    unique_y_labels = label_coder.decode(unique_y)
    plot_confusion_matrix(test_matrix, classes=unique_y_labels,
                          title=f"Gesture classification conv acc={test_accuracy_:.2}",
                          output_path=os.path.join(TEST_OUTPUT, "gesture_classification_conv_confusion.png"))
