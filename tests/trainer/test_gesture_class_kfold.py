"""Test training of sensor classification k-fold validation."""

import os

import torch

from iotai_sensor_classification.dataset import read_dataset, read_recordings
from iotai_sensor_classification.model_handler import ModelCall
from iotai_sensor_classification.evaluation import evaluate_prediction
from data.gestures import accelerometer_gyroscope
from iotai_sensor_classification.trainer import sensor_classification
from iotai_sensor_classification.plot_util import plot_columns, plot_confusion_matrix, plot_lines
from iotai_sensor_classification.preprocess import check_windows
import numpy as np
from sklearn import model_selection


TEST_OUTPUT = os.path.join("test_output", "gestures", "trainer", "kfold")
SAMPLES_PER_RECORDING = 120
K_FOLDS = 6


def test_window_size():
    """Determine window size for linear accelerometer, gyroscope measurements of motion gestures."""
    recordings = read_recordings(os.path.dirname(accelerometer_gyroscope.__file__))
    window_checked = check_windows(recordings, window_size=SAMPLES_PER_RECORDING)
    os.makedirs(TEST_OUTPUT, exist_ok=True)
    for label_name in window_checked.keys():
        label_data = window_checked[label_name]
        plot_lines(label_data, name=f"{label_name} gesture filtered {SAMPLES_PER_RECORDING} windows",
                   filepath=os.path.join(TEST_OUTPUT, f"{label_name}-filtered-windows.png"),
                   vertical_tick_spacing=SAMPLES_PER_RECORDING)


def test_train_gesture_class_kfold_linear():
    """Test trainer gesture classification model from sensor data.
    :return:
    """
    X, y, label_coder = read_dataset(os.path.dirname(accelerometer_gyroscope.__file__), SAMPLES_PER_RECORDING)
    X_train_val, X_test, y_train_val, y_test = model_selection.train_test_split(X, y, test_size=0.15, shuffle=True)
    kf = model_selection.KFold(n_splits=K_FOLDS)
    k_validations = []
    models = []
    max_val_accs = []
    for train, val in kf.split(X_train_val):
        X_train = X_train_val[train]
        X_val = X_train_val[val]
        y_train = y_train_val[train]
        y_val = y_train_val[val]
        model = sensor_classification.LinearModel(input_dim=X_train.shape[1] * X_train.shape[2],
                                                  output_dim=len(np.unique(y_train)))
        val_df = sensor_classification.train_gesture_classification(model, X_train, y_train, X_val, y_val)
        k_validations.append(val_df)
        models.append(model)
        max_val_acc = val_df['val_acc'].max()
        max_val_accs.append(max_val_acc)

    # train with all non test data
    model = sensor_classification.LinearModel(input_dim=X_train_val.shape[1] * X_train_val.shape[2],
                                              output_dim=len(np.unique(y_train_val)))
    train_val_loss_df = sensor_classification.train_gesture_classification(model, X_train_val, y_train_val)
    state_path = os.path.join(TEST_OUTPUT, "gesture_class_gyro_kval_linear_state_dict.zip")
    torch.save(model.state_dict(), state_path)

    plot_columns(train_val_loss_df,
                 name=f"Gesture classification K-validation linear model, max acc={max_val_acc:.2} [accel, gyro]",
                 filepath=os.path.join(TEST_OUTPUT, "gesture_class_gyro_kval_linear.png"),
                 title_mean=False)
    load_model = sensor_classification.LinearModel(input_dim=X_train.shape[1] * X_train.shape[2],
                                                   output_dim=len(np.unique(y_train)))
    load_model.load_state_dict(torch.load(state_path))
    assert all(load_model.state_dict()['layer3.bias'] == model.state_dict()['layer3.bias'])
    model_call = ModelCall(model=load_model, decode=label_coder.decode)
    test_accuracy_, test_matrix = evaluate_prediction(model_call, label_coder.decode, X_test, y_test)
    unique_y = np.unique(y)
    unique_y.sort()
    unique_y_labels = label_coder.decode(unique_y)
    unique_y_labels
    plot_confusion_matrix(test_matrix, classes=unique_y_labels,
                          title=f"Gesture classification k-validation linear acc={test_accuracy_:.2} [accel, gyro]",
                          output_path=os.path.join(TEST_OUTPUT, "gesture_class_gyro_kval_linear_confusion.png"))
