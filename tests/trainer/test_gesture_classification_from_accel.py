"""Test training of sensor classification."""

import os

import torch

from iotai_sensor_classification.dataset import prepare_datasets, read_dataset
from iotai_sensor_classification.model_handler import ModelCall
from iotai_sensor_classification.evaluation import evaluate_prediction
from data.gestures import linear_accelerometer
from iotai_sensor_classification.trainer import sensor_classification
from iotai_sensor_classification.plot_util import plot_columns, plot_confusion_matrix
import numpy as np


TEST_OUTPUT = os.path.join("test_output", "gestures", "trainer", "accel")
SAMPLES_PER_RECORDING = 160


def test_train_gesture_classification_linear():
    """Test trainer gesture classification model from sensor data.
    :return:
    """
    def get_dataset():
        return read_dataset(os.path.dirname(linear_accelerometer.__file__), SAMPLES_PER_RECORDING)
    X_train, X_val, X_test, y_train, y_val, y_test, label_coder = \
        prepare_datasets(os.path.join(TEST_OUTPUT, "gesture_classification_dataset_linear.png"),
                     "Gesture classification datasets label count linear", get_dataset, TEST_OUTPUT)

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
    def get_dataset():
        return read_dataset(os.path.dirname(linear_accelerometer.__file__), SAMPLES_PER_RECORDING)
    X_train, X_val, X_test, y_train, y_val, y_test, label_coder = \
        prepare_datasets(os.path.join(TEST_OUTPUT, "gesture_classification_dataset_conv.png"),
                     "Gesture classification datasets label count conv", get_dataset, TEST_OUTPUT)

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
