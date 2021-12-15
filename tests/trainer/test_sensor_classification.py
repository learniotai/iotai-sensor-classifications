"""Test training of sensor classification."""

import os
from data.gestures import linear_accelerometer
from iotai_sensor_classification import dataset
from iotai_sensor_classification.preprocess import check_windows, parse_recording, SAMPLES_PER_RECORDING
from iotai_sensor_classification.plot_util import group_label_bars
from iotai_sensor_classification.recording import read_recordings
from iotai_sensor_classification.trainer.sensor_classification import train_gesture_classification
from iotai_sensor_classification.plot_util import plot_columns


def get_accelerometer_dataset():
    """Read gesture recordings for all tests in file."""
    recordings_dir = os.path.dirname(linear_accelerometer.__file__)
    recordings = read_recordings(recordings_dir=recordings_dir)
    window_checked = check_windows(recordings)
    normed_gesture_measures, encoded_labels, label_coder = \
        parse_recording(window_checked, samples_per_recording=SAMPLES_PER_RECORDING)
    return normed_gesture_measures, encoded_labels, label_coder


def test_train_gesture_classification():
    """
    Test trainer gesture classification model from sensor data.
    :return:
    """
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

    raw_labels = {}
    raw_labels['train'] = train_y_labels
    raw_labels['validation'] = val_y_labels
    raw_labels['test'] = test_y_labels

    test_output = os.path.join("test_output", "gestures", "trainer")
    os.makedirs(test_output, exist_ok=True)
    group_label_bars(raw_labels, title="Gesture classification datasets label count",
                     filepath=os.path.join(test_output, "gesture_classification_dataset.png"))

    trained_model, val_df = train_gesture_classification(train_X, val_X, train_y, val_y)
    plot_columns(val_df, name="Gesture classification training validation",
                 filepath=os.path.join(test_output, "gesture_classification_training_validation.png"),
                 title_mean=False)
